import socket
import threading
import numpy as np
import argparse
from sklearn.datasets import fetch_openml
import pickle
import struct
import json
import time
import sys
import select


def send_json(sock, obj):
    """Send JSON message with length prefix."""
    data = json.dumps(obj).encode('utf-8')
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_json(sock, timeout=None):
    """Receive JSON message with length prefix."""
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
    """Send numpy arrays using pickle."""
    data = pickle.dumps(arrays, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_arrays(sock, timeout=None):
    """Receive numpy arrays using pickle."""
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
        
        # Fixed number of workers for entire training
        self.n_workers = n_workers

        self.workers = []  # List of (conn, addr, worker_id, assignments)
        self.lock = threading.Lock()
        self.ready_workers = 0
        self.ready_condition = threading.Condition(self.lock)
        self.training_started = False
        self.training_done = False

        # Dataset
        self.X_train, self.Y_train, self.y_train_raw = load_mnist(self.train_size)
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

    def create_assignments(self):
        """Create fixed batch assignments for each worker."""
        assignments = [[] for _ in range(self.n_workers)]
        for i in range(self.n_batches):
            assignments[i % self.n_workers].append(i)
        return assignments

    def send_dataset(self, conn, worker_id, assignments):
        """Send dataset and worker-specific assignment."""
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

    def console_control(self):
        """Console thread for status only."""
        print(f"\n{'='*60}")
        print(f"CONSOLE MONITOR")
        print(f"Fixed workers: {self.n_workers}")
        print(f"Batches: {self.n_batches} (fixed, no rebalancing)")
        print(f"Commands:")
        print(f"  status   - Show current worker count and status")
        print(f"  help     - Show this help message")
        print(f"{'='*60}\n")
        
        while not self.training_done:
            try:
                # Non-blocking input check
                if sys.platform != 'win32':
                    ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if ready:
                        line = sys.stdin.readline().strip()
                    else:
                        continue
                else:
                    time.sleep(0.5)
                    continue
                
                if not line:
                    continue
                
                parts = line.split()
                cmd = parts[0].lower()
                
                if cmd == 'status':
                    with self.lock:
                        ready = self.ready_workers
                        total = len(self.workers)
                        started = self.training_started
                        done = self.training_done
                    
                    print(f"\n--- STATUS ---")
                    print(f"Target workers: {self.n_workers}")
                    print(f"Ready workers:  {ready}")
                    print(f"Total connected: {total}")
                    print(f"Training started: {started}")
                    print(f"Training done: {done}")
                    if not started:
                        print(f"Waiting for {max(0, self.n_workers - ready)} more workers")
                    print(f"--------------\n")
                
                elif cmd == 'help':
                    print(f"\nCommands:")
                    print(f"  status   - Show current status")
                    print(f"  help     - Show this help")
                    print(f"")
                
                else:
                    print(f"Unknown command: {line}")
                    print(f"Type 'help' for available commands")
                    
            except EOFError:
                break
            except Exception as e:
                print(f"Console error: {e}")

    def training_loop(self):
        """Training coordinator - fixed workers, no rebalancing."""
        target = self.n_workers
        
        print(f"\n{'='*60}")
        print(f"WAITING FOR {target} WORKERS")
        print(f"Configuration: {self.n_batches} batches ÷ {target} workers = ~{self.n_batches//target} batches each")
        print(f"{'='*60}\n")
        
        # Wait for exact number of workers
        with self.ready_condition:
            while self.ready_workers < target:
                remaining = target - self.ready_workers
                print(f"Waiting... {self.ready_workers}/{target} workers ready ({remaining} more needed)")
                self.ready_condition.wait(timeout=2.0)
        
        self.training_started = True
        
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING with {self.n_workers} worker(s)")
        print(f"Fixed assignment for all {self.epochs} epochs")
        print(f"{'='*60}")
        
        # Create fixed assignments once
        assignments = self.create_assignments()
        for i, a in enumerate(assignments):
            print(f"  Worker {i}: batches {a}")
        
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

            if len(workers) == 0:
                print("CRITICAL: All workers disconnected!")
                break

            # Send training commands with fixed assignments
            for worker in workers:
                conn, addr, worker_id, _ = worker
                try:
                    send_json(conn, {
                        "type": "train",
                        "batches": assignments[worker_id],
                        "lr": self.lr,
                        "epoch": epoch + 1
                    })
                except Exception as e:
                    print(f"Error sending train command to worker {worker_id}: {e}")

            # Send current weights
            for worker in workers:
                conn, addr, worker_id, _ = worker
                try:
                    send_arrays(conn, (self.ws1, self.bs1, self.ws2, self.bs2))
                except Exception as e:
                    print(f"Error sending weights to worker {worker_id}: {e}")

            # Receive gradients
            grads = []
            for worker in workers:
                conn, addr, worker_id, _ = worker
                try:
                    worker_grads = recv_arrays(conn, timeout=60.0)
                    if worker_grads is not None:
                        grads.extend(worker_grads)
                        print(f"Received {len(worker_grads)} gradients from worker {worker_id}")
                except Exception as e:
                    print(f"Error receiving gradients from worker {worker_id}: {e}")

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
        for worker in workers:
            conn = worker[0]
            try:
                send_json(conn, {"type": "done"})
                conn.close()
            except:
                pass

    def handle_worker(self, conn, addr):
        """Handle individual worker connection."""
        print(f"\n>>> Worker connected: {addr}")
        
        # Reject if training already started or too many workers
        with self.lock:
            if self.training_started:
                print(f"Training already started, rejecting worker {addr}")
                try:
                    send_json(conn, {"type": "rejected", "reason": "training_already_started"})
                    conn.close()
                except:
                    pass
                return
            
            if len(self.workers) >= self.n_workers:
                print(f"Already have {self.n_workers} workers, rejecting {addr}")
                try:
                    send_json(conn, {"type": "rejected", "reason": "worker_limit_reached"})
                    conn.close()
                except:
                    pass
                return
            
            # Assign worker ID
            worker_id = len(self.workers)
            self.workers.append([conn, addr, worker_id, []])
        
        try:
            # Create assignments and send to this worker
            assignments = self.create_assignments()
            self.send_dataset(conn, worker_id, assignments)
            print(f"Dataset sent to worker {worker_id} ({addr}) - assigned batches: {assignments[worker_id]}")
            
            # Wait for ready acknowledgment
            response = recv_json(conn, timeout=30.0)
            if response is None or response.get("status") != "ready":
                print(f"Worker {worker_id} failed to acknowledge")
                with self.lock:
                    self.workers = [w for w in self.workers if w[0] != conn]
                return
            
            print(f"Worker {worker_id} ({addr}) is ready")
            
            # Signal that a worker is ready
            with self.ready_condition:
                self.ready_workers += 1
                current_ready = self.ready_workers
                
                if current_ready >= self.n_workers:
                    self.ready_condition.notify_all()
                    print(f"\n*** ALL {self.n_workers} WORKERS READY! ***")
                    print(f"*** STARTING TRAINING NOW! ***\n")
                else:
                    print(f"Progress: {current_ready}/{self.n_workers} workers ready")
                
            # Keep connection alive
            while not self.training_done:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error handling worker {addr}: {e}")
        finally:
            with self.lock:
                self.workers = [w for w in self.workers if w[0] != conn]
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

        print(f"\n{'='*60}")
        print(f"Parameter Server running on {self.host}:{self.port}")
        print(f"Configuration:")
        print(f"  Workers: {self.n_workers} (fixed)")
        print(f"  Batches: {self.n_batches} (fixed, no rebalancing)")
        print(f"  Epochs: {self.epochs}")
        print(f"{'='*60}")

        # Start console monitor thread
        threading.Thread(target=self.console_control, daemon=True).start()
        
        # Start training coordinator thread
        threading.Thread(target=self.training_loop, daemon=True).start()

        # Accept connections until training starts
        while not self.training_started:
            conn, addr = server.accept()
            threading.Thread(
                target=self.handle_worker, 
                args=(conn, addr), 
                daemon=True
            ).start()
        
        # After training starts, stop accepting new connections
        print("Training started, no longer accepting new workers")
        server.close()


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Worker Distributed Parameter Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--n_batches", type=int, default=6,
                       help="Total number of batches (fixed)")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--train_size", type=int, default=60000)
    parser.add_argument("--eval_interval", type=int, default=25, 
                       help="Evaluate accuracy every N epochs")
    parser.add_argument("--n_workers", type=int, default=3,
                       help="Fixed number of workers for entire training")

    args = parser.parse_args()

    ps = ParameterServer(
        host=args.host, 
        port=args.port, 
        epochs=args.epochs,
        n_batches=args.n_workers, # La cantidad de batches sera igual a la cantidad de workers
        lr=args.lr, 
        hidden=args.hidden, 
        train_size=args.train_size,
        eval_interval=args.eval_interval,
        n_workers=args.n_workers
    )
    ps.start()