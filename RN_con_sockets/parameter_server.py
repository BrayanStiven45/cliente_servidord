import socket
import threading
import numpy as np
import argparse
from sklearn.datasets import fetch_openml
import pickle
import struct
# import numpy as np
# import multiprocessing as mp
# from common import send_msg, recv_msg


def send_msg(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_msg(sock):
    raw_len = recv_all(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    return pickle.loads(recv_all(sock, msg_len))

def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# ==============================
# DATASET
# ==============================

def load_mnist(train_size):

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)

    one_hot = np.zeros((y.size, 10))
    one_hot[np.arange(y.size), y] = 1

    return X[:train_size], one_hot[:train_size]


# ==============================
# PARAMETER SERVER CLASS
# ==============================

class ParameterServer:

    def __init__(self, host, port, epochs,
                 initial_batches, batch_increment,
                 lr, hidden, train_size):

        self.host = host
        self.port = port
        self.epochs = epochs
        self.initial_batches = initial_batches
        self.batch_increment = batch_increment
        self.lr = lr
        self.hidden = hidden
        self.train_size = train_size

        self.workers = {}
        self.worker_id_counter = 0
        self.lock = threading.Lock()

        # Dataset
        self.X_train, self.Y_train = load_mnist(self.train_size)

        self.n_batches = self.initial_batches
        self.batch_data = []
        self.batch_targets = []
        self.create_batches(self.n_batches)

        # Model init
        input_size = self.X_train.shape[1]
        output_size = 10

        self.ws1 = np.random.randn(hidden, input_size) * np.sqrt(2. / input_size)
        self.bs1 = np.zeros((1, hidden))
        self.ws2 = np.random.randn(output_size, hidden) * np.sqrt(2. / hidden)
        self.bs2 = np.zeros((1, output_size))

        self.worker_condition = threading.Condition(self.lock)

    # ==============================
    # BATCH CREATION
    # ==============================

    def create_batches(self, n_batches):
        self.batch_data = np.array_split(self.X_train, n_batches)
        self.batch_targets = np.array_split(self.Y_train, n_batches)

    # ==============================
    # WORKER MANAGEMENT
    # ==============================

    def handle_worker(self, conn, wid):

        with self.worker_condition:
            self.workers[wid] = conn
            print(f"Worker {wid} connected")

            if len(self.workers) > self.n_batches:
                self.rebalance_batches()

            # Notificar al training loop que ya hay workers
            self.worker_condition.notify_all()

        try:
            while True:
                data = recv_msg(conn)
                if data is None:
                    break
        except:
            pass

        with self.worker_condition:
            print(f"Worker {wid} disconnected")
            if wid in self.workers:
                del self.workers[wid]

            self.rebalance_batches()
            self.worker_condition.notify_all()

    def rebalance_batches(self):

        if len(self.workers) > self.n_batches:
            self.n_batches += self.batch_increment
            print(f"Increasing batches to {self.n_batches}")
            self.create_batches(self.n_batches)

        self.broadcast_dataset()

    def broadcast_dataset(self):

        for wid, conn in self.workers.items():
            send_msg(conn, {
                "type": "dataset",
                "batch_data": self.batch_data,
                "batch_targets": self.batch_targets
            })

    # ==============================
    # TRAINING LOOP
    # ==============================

    def training_loop(self):

        print("Training thread started. Waiting for workers...")

        for epoch in range(self.epochs):
            
            with self.worker_condition:
                while len(self.workers) == 0:
                    print("No workers connected. Waiting...")
                    self.worker_condition.wait()

            print(f"\nEpoch {epoch}")

            worker_list = list(self.workers.items())

            if not worker_list:
                continue

            assignments = {wid: [] for wid, _ in worker_list}

            for i in range(self.n_batches):
                wid = worker_list[i % len(worker_list)][0]
                assignments[wid].append(i)

            # Send train commands
            for wid, conn in worker_list:
                send_msg(conn, {
                    "type": "train",
                    "batches": assignments[wid],
                    "weights": (self.ws1, self.bs1, self.ws2, self.bs2),
                    "lr": self.lr
                })

    #### Revisar esta parte de codigo, para el revalanceo de batches
    #### En el caso de que que se caiga un worker y no se tenga su gradiente
    #### ---------------------------------------------------------------------
            # Collect gradients
            all_grads = []

            for wid, conn in worker_list:
                try:
                    grads = recv_msg(conn)
                    all_grads.extend(grads)
                except:
                    print(f"Worker {wid} failed during epoch")
                    if wid in self.workers:
                        del self.workers[wid]
                    self.rebalance_batches()
                    return

            if not all_grads:
                continue
    #### -----------------------------------------------------------------------------

            avg_dw1 = sum(g[0] for g in all_grads) / len(all_grads)
            avg_db1 = sum(g[1] for g in all_grads) / len(all_grads)
            avg_dw2 = sum(g[2] for g in all_grads) / len(all_grads)
            avg_db2 = sum(g[3] for g in all_grads) / len(all_grads)

            self.ws1 -= self.lr * avg_dw1
            self.bs1 -= self.lr * avg_db1
            self.ws2 -= self.lr * avg_dw2
            self.bs2 -= self.lr * avg_db2

            print("Weights updated")

    # ==============================
    # START SERVER
    # ==============================

    def start(self):

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen()

        print(f"Parameter Server running on {self.host}:{self.port}")
        print(f"EPOCHS={self.epochs}")
        print(f"INITIAL_BATCHES={self.initial_batches}")
        print(f"BATCH_INCREMENT={self.batch_increment}")
        print(f"LR={self.lr}")
        print(f"HIDDEN={self.hidden}")
        print(f"TRAIN_SIZE={self.train_size}")

        threading.Thread(target=self.training_loop, daemon=True).start()

        while True:
            conn, addr = server.accept()
            self.worker_id_counter += 1
            threading.Thread(
                target=self.handle_worker,
                args=(conn, self.worker_id_counter),
                daemon=True
            ).start()


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distributed Parameter Server")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--initial_batches", type=int, default=4)
    parser.add_argument("--batch_increment", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--train_size", type=int, default=60000)

    args = parser.parse_args()

    ps = ParameterServer(
        host=args.host,
        port=args.port,
        epochs=args.epochs,
        initial_batches=args.initial_batches,
        batch_increment=args.batch_increment,
        lr=args.lr,
        hidden=args.hidden,
        train_size=args.train_size
    )

    ps.start()