import socket
import threading
import numpy as np
import argparse
from sklearn.datasets import fetch_openml
import pickle
import struct
import json
import time


def send_json(sock, obj):
    """Send JSON message with length prefix."""
    data = json.dumps(obj).encode('utf-8')
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_json(sock):
    """Receive JSON message with length prefix."""
    raw_len = recv_all(sock, 4)
    if raw_len is None:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    data = recv_all(sock, msg_len)
    if data is None:
        return None
    return json.loads(data.decode('utf-8'))


def send_arrays(sock, arrays):
    """Send numpy arrays using pickle."""
    data = pickle.dumps(arrays, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_arrays(sock):
    """Receive numpy arrays using pickle."""
    raw_len = recv_all(sock, 4)
    if raw_len is None:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    data = recv_all(sock, msg_len)
    if data is None:
        return None
    return pickle.loads(data)


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
    return X[:train_size], one_hot[:train_size], y[:train_size]


# ==============================
# PARAMETER SERVER
# ==============================

class ParameterServer:
    def __init__(self, host, port, epochs,
                 initial_batches, batch_increment,
                 lr, hidden, train_size, eval_interval=25,
                 min_workers=1):  # Added min_workers parameter
        self.host = host
        self.port = port
        self.epochs = epochs
        self.initial_batches = initial_batches
        self.batch_increment = batch_increment
        self.lr = lr
        self.hidden = hidden
        self.train_size = train_size
        self.eval_interval = eval_interval
        self.min_workers = min_workers  # Minimum workers to start training

        self.workers = []
        self.lock = threading.Lock()
        self.ready_workers = 0
        self.ready_condition = threading.Condition(self.lock)
        self.training_started = False
        self.training_done = False  # Flag to signal training completion

        # Dataset
        self.X_train, self.Y_train, self.y_train_raw = load_mnist(self.train_size)
        self.n_batches = initial_batches
        self.create_batches(self.n_batches)

        # Model
        input_size = self.X_train.shape[1]
        output_size = 10

        self.ws1 = np.random.randn(hidden, input_size) * np.sqrt(2. / input_size)
        self.bs1 = np.zeros((1, hidden))
        self.ws2 = np.random.randn(output_size, hidden) * np.sqrt(2. / hidden)
        self.bs2 = np.zeros((1, output_size))
        
        # Track accuracy history
        self.accuracy_history = []

    def create_batches(self, n_batches):
        self.batch_data = np.array_split(self.X_train, n_batches)
        self.batch_targets = np.array_split(self.Y_train, n_batches)

    def rebalance(self):
        n_workers = len(self.workers)
        if n_workers > self.n_batches:
            self.n_batches += self.batch_increment
            print("Increasing batches:", self.n_batches)
            self.create_batches(self.n_batches)

        assignments = [[] for _ in range(n_workers)]
        for i in range(self.n_batches):
            assignments[i % n_workers].append(i)
        return assignments

    def send_dataset(self, conn):
        """Send dataset as binary arrays first, then metadata as JSON."""
        arrays = {
            'batch_data': self.batch_data,
            'batch_targets': self.batch_targets,
        }
        send_arrays(conn, arrays)
        
        send_json(conn, {
            "type": "dataset",
            "n_batches": self.n_batches,
            "lr": self.lr
        })

    def compute_accuracy(self, X=None, y=None):
        """Compute accuracy on given data (default: training set)."""
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train_raw
            
        z1 = X.dot(self.ws1.T) + self.bs1
        a1 = np.maximum(0, z1)
        z2 = a1.dot(self.ws2.T) + self.bs2
        predictions = np.argmax(z2, axis=1)
        accuracy = np.mean(predictions == y) * 100
        return accuracy

    def training_loop(self):
        print(f"Training thread waiting for {self.min_workers} worker(s)...")
        
        # Wait for minimum number of workers to be ready
        with self.ready_condition:
            while self.ready_workers < self.min_workers:
                self.ready_condition.wait()
        
        self.training_started = True
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING with {self.ready_workers} worker(s)")
        print(f"{'='*60}")
        
        # Initial accuracy
        initial_acc = self.compute_accuracy()
        print(f"\nInitial Accuracy (Epoch 0): {initial_acc:.2f}%")
        self.accuracy_history.append((0, initial_acc))

        for epoch in range(self.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'='*50}")

            with self.lock:
                workers = list(self.workers)

            assignments = self.rebalance()

            # Send training commands as JSON
            for i, conn in enumerate(workers):
                try:
                    send_json(conn, {
                        "type": "train",
                        "batches": assignments[i],
                        "lr": self.lr
                    })
                except Exception as e:
                    print(f"Error sending train command to worker {i}: {e}")

            # Send current weights as binary arrays
            for i, conn in enumerate(workers):
                try:
                    send_arrays(conn, (self.ws1, self.bs1, self.ws2, self.bs2))
                except Exception as e:
                    print(f"Error sending weights to worker {i}: {e}")

            # Receive gradients as binary arrays
            grads = []
            for i, conn in enumerate(workers):
                try:
                    worker_grads = recv_arrays(conn)
                    if worker_grads is not None:
                        grads.extend(worker_grads)
                        print(f"Received {len(worker_grads)} gradients from worker {i}")
                except Exception as e:
                    print(f"Error receiving gradients from worker {i}: {e}")

            if not grads:
                print("No gradients received, skipping update")
                continue

            # Aggregate gradients
            avg_dw1 = sum(g[0] for g in grads) / len(grads)
            avg_db1 = sum(g[1] for g in grads) / len(grads)
            avg_dw2 = sum(g[2] for g in grads) / len(grads)
            avg_db2 = sum(g[3] for g in grads) / len(grads)

            # Update weights
            self.ws1 -= self.lr * avg_dw1
            self.bs1 -= self.lr * avg_db1
            self.ws2 -= self.lr * avg_dw2
            self.bs2 -= self.lr * avg_db2

            print(f"Weights updated using {len(grads)} gradient sets")
            
            # Evaluate every eval_interval epochs
            if (epoch + 1) % self.eval_interval == 0:
                acc = self.compute_accuracy()
                self.accuracy_history.append((epoch + 1, acc))
                print(f"\n*** ACCURACY CHECKPOINT (Epoch {epoch + 1}) ***")
                print(f"Training Accuracy: {acc:.2f}%")
                print(f"*** END CHECKPOINT ***\n")

        # Final evaluation
        final_acc = self.compute_accuracy()
        self.accuracy_history.append((self.epochs, final_acc))
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Final Training Accuracy: {final_acc:.2f}%")
        print(f"\nAccuracy History:")
        for epoch, acc in self.accuracy_history:
            print(f"  Epoch {epoch:3d}: {acc:6.2f}%")
        print(f"{'='*60}\n")
        
        self.training_done = True
        
        # Notify workers training is done
        with self.lock:
            workers = list(self.workers)
        for conn in workers:
            try:
                send_json(conn, {"type": "done"})
            except:
                pass

    def handle_worker(self, conn, addr):
        """Handle individual worker connection."""
        print(f"Worker connected: {addr}")
        
        # Check if training already started - reject new workers
        with self.lock:
            if self.training_started:
                print(f"Training already started, rejecting worker {addr}")
                try:
                    send_json(conn, {"type": "rejected", "reason": "training_already_started"})
                    conn.close()
                except:
                    pass
                return
            
            self.workers.append(conn)
        
        try:
            # Send dataset immediately upon connection
            self.send_dataset(conn)
            print(f"Dataset sent to worker {addr}")
            
            # Wait for worker to confirm dataset received
            response = recv_json(conn)
            if response is None or response.get("status") != "ready":
                print(f"Worker {addr} failed to acknowledge dataset")
                with self.lock:
                    if conn in self.workers:
                        self.workers.remove(conn)
                return
            
            print(f"Worker {addr} is ready")
            
            # Signal that a worker is ready
            with self.ready_condition:
                self.ready_workers += 1
                current_ready = self.ready_workers
                self.ready_condition.notify()
            
            print(f"Total ready workers: {current_ready}/{self.min_workers}")
            
            # Keep connection alive and handle communication during training
            # Just wait here - training_loop handles the actual communication
            while not self.training_done:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error handling worker {addr}: {e}")
        finally:
            with self.lock:
                if conn in self.workers:
                    self.workers.remove(conn)
                if self.ready_workers > 0:
                    self.ready_workers -= 1
            try:
                conn.close()
            except:
                pass

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen()

        print(f"Parameter Server running on {self.host}:{self.port}")
        print(f"Waiting for {self.min_workers} worker(s) to start training...")

        # Start training thread
        threading.Thread(target=self.training_loop, daemon=True).start()

        while True:
            conn, addr = server.accept()
            threading.Thread(
                target=self.handle_worker, 
                args=(conn, addr), 
                daemon=True
            ).start()


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Parameter Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--initial_batches", type=int, default=4)
    parser.add_argument("--batch_increment", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--train_size", type=int, default=60000)
    parser.add_argument("--eval_interval", type=int, default=25, 
                       help="Evaluate accuracy every N epochs")
    parser.add_argument("--min_workers", type=int, default=2,  # Default to 2 workers
                       help="Minimum number of workers to start training")

    args = parser.parse_args()

    ps = ParameterServer(
        host=args.host, port=args.port, epochs=args.epochs,
        initial_batches=args.initial_batches, 
        batch_increment=args.batch_increment,
        lr=args.lr, hidden=args.hidden, train_size=args.train_size,
        eval_interval=args.eval_interval,
        min_workers=args.min_workers
    )
    ps.start()