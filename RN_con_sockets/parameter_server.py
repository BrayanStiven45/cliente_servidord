import socket
import threading
import numpy as np
import argparse
from sklearn.datasets import fetch_openml
import pickle
import struct
import json
import time
import os


def send_json(sock, obj):
    data = json.dumps(obj).encode('utf-8')
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_json(sock, timeout=None):
    if timeout:
        sock.settimeout(timeout)
    try:
        raw_len = recv_all(sock, 4)
        if raw_len is None:
            return None
        msg_len = struct.unpack(">I", raw_len)[0]
        data = recv_all(sock, msg_len)
        if data is None:
            return None
        return json.loads(data.decode('utf-8'))
    finally:
        if timeout:
            sock.settimeout(None)


def send_arrays(sock, arrays):
    data = pickle.dumps(arrays, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_arrays(sock, timeout=None):
    if timeout:
        sock.settimeout(timeout)
    try:
        raw_len = recv_all(sock, 4)
        if raw_len is None:
            return None
        msg_len = struct.unpack(">I", raw_len)[0]
        data = recv_all(sock, msg_len)
        if data is None:
            return None
        return pickle.loads(data)
    finally:
        if timeout:
            sock.settimeout(None)


def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


# ==============================
# MNIST CACHE
# ==============================

def load_mnist(train_size, cache_file="mnist.pkl"):

    if os.path.exists(cache_file):

        print("Loading MNIST from cache file...")

        with open(cache_file, "rb") as f:
            X, y, one_hot = pickle.load(f)

    else:

        print("Downloading MNIST dataset...")

        mnist = fetch_openml("mnist_784", version=1, as_frame=False)

        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int32)

        one_hot = np.zeros((y.size, 10))
        one_hot[np.arange(y.size), y] = 1

        with open(cache_file, "wb") as f:
            pickle.dump((X, y, one_hot), f)

        print("MNIST saved to cache.")

    return X[:train_size], one_hot[:train_size], y[:train_size]


class ParameterServer:

    def __init__(self, host, port, epochs,
                 n_batches, lr, hidden, train_size,
                 eval_interval=25, n_workers=3):

        self.host = host
        self.port = port
        self.epochs = epochs
        self.n_batches = n_batches
        self.lr = lr
        self.hidden = hidden
        self.train_size = train_size
        self.eval_interval = eval_interval
        self.n_workers = n_workers

        self.workers = []
        self.lock = threading.Lock()
        self.ready_workers = 0
        self.ready_condition = threading.Condition(self.lock)

        self.training_started = False
        self.training_done = False

        self.server_socket = None

        # MNIST
        self.X_train, self.Y_train, self.y_train_raw = load_mnist(self.train_size)

        self.create_batches(self.n_batches)

        input_size = self.X_train.shape[1]
        output_size = 10

        self.ws1 = np.random.randn(hidden, input_size) * np.sqrt(2. / input_size)
        self.bs1 = np.zeros((1, hidden))
        self.ws2 = np.random.randn(output_size, hidden) * np.sqrt(2. / hidden)
        self.bs2 = np.zeros((1, output_size))

    def create_batches(self, n_batches):
        self.batch_data = np.array_split(self.X_train, n_batches)
        self.batch_targets = np.array_split(self.Y_train, n_batches)

    def create_assignments(self):

        assignments = [[] for _ in range(self.n_workers)]

        for i in range(self.n_batches):
            assignments[i % self.n_workers].append(i)

        return assignments

    def send_dataset(self, conn, worker_id, assignments):

        arrays = {
            'batch_data': self.batch_data,
            'batch_targets': self.batch_targets,
        }

        send_arrays(conn, arrays)

        send_json(conn, {
            "type": "dataset",
            "n_batches": self.n_batches,
            "lr": self.lr,
            "worker_id": worker_id,
            "assignments": assignments[worker_id]
        })

    def compute_accuracy(self):

        z1 = self.X_train.dot(self.ws1.T) + self.bs1
        a1 = np.maximum(0, z1)
        z2 = a1.dot(self.ws2.T) + self.bs2

        predictions = np.argmax(z2, axis=1)

        accuracy = np.mean(predictions == self.y_train_raw) * 100

        return accuracy

    def training_loop(self):

        with self.ready_condition:
            while self.ready_workers < self.n_workers:
                self.ready_condition.wait()

        self.training_started = True

        start_time = time.time()

        for epoch in range(self.epochs):

            with self.lock:
                workers = list(self.workers)

            for worker in workers:
                conn, addr, worker_id, _ = worker

                send_json(conn, {
                    "type": "train",
                    "batches": [worker_id],
                    "lr": self.lr,
                    "epoch": epoch + 1
                })

            for worker in workers:
                conn, addr, worker_id, _ = worker
                send_arrays(conn, (self.ws1, self.bs1, self.ws2, self.bs2))

            grads = []

            for worker in workers:

                conn, addr, worker_id, _ = worker

                worker_grads = recv_arrays(conn, timeout=60.0)

                if worker_grads is not None:
                    grads.extend(worker_grads)

            if not grads:
                continue

            avg_dw1 = sum(g[0] for g in grads) / len(grads)
            avg_db1 = sum(g[1] for g in grads) / len(grads)
            avg_dw2 = sum(g[2] for g in grads) / len(grads)
            avg_db2 = sum(g[3] for g in grads) / len(grads)

            self.ws1 -= self.lr * avg_dw1
            self.bs1 -= self.lr * avg_db1
            self.ws2 -= self.lr * avg_dw2
            self.bs2 -= self.lr * avg_db2

        wall_time = time.time() - start_time
        final_acc = self.compute_accuracy()

        print(f"RESULT {self.n_workers} {wall_time} {final_acc}")

        self.training_done = True

        with self.lock:
            for worker in self.workers:
                try:
                    worker[0].close()
                except:
                    pass

        try:
            if self.server_socket:
                self.server_socket.close()
        except:
            pass

    def handle_worker(self, conn, addr):

        accept_worker = False

        with self.lock:

            if not self.training_started and len(self.workers) < self.n_workers:

                worker_id = len(self.workers)

                self.workers.append([conn, addr, worker_id, []])

                accept_worker = True

        if not accept_worker:

            try:
                send_json(conn, {"type": "reject"})
            except:
                pass

            conn.close()
            return

        assignments = self.create_assignments()

        self.send_dataset(conn, worker_id, assignments)

        response = recv_json(conn, timeout=30.0)

        if response is None or response.get("status") != "ready":
            conn.close()
            return

        with self.ready_condition:

            self.ready_workers += 1

            if self.ready_workers >= self.n_workers:
                self.ready_condition.notify_all()

        while not self.training_done:
            time.sleep(0.1)

    def start(self):

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server.bind((self.host, self.port))

        server.listen()

        self.server_socket = server

        print(f"Parameter server listening on {self.host}:{self.port}")
        print(f"Waiting for {self.n_workers} workers...")

        threading.Thread(target=self.training_loop, daemon=True).start()

        while not self.training_done:

            try:
                server.settimeout(1.0)
                conn, addr = server.accept()

            except socket.timeout:
                continue

            except OSError:
                break

            with self.lock:

                if len(self.workers) >= self.n_workers:

                    try:
                        send_json(conn, {"type": "reject"})
                    except:
                        pass

                    conn.close()
                    continue

            threading.Thread(
                target=self.handle_worker,
                args=(conn, addr),
                daemon=True
            ).start()

        print("Training finished. Server shutting down.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distributed Parameter Server")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--n_batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--train_size", type=int, default=60000)
    parser.add_argument("--n_workers", type=int, default=3)

    args = parser.parse_args()

    if args.n_batches is None:
        args.n_batches = args.n_workers

    ps = ParameterServer(
        host=args.host,
        port=args.port,
        epochs=args.epochs,
        n_batches=args.n_batches,
        lr=args.lr,
        hidden=args.hidden,
        train_size=args.train_size,
        n_workers=args.n_workers
    )

    ps.start()