import socket
import argparse
import pickle
import struct
import numpy as np
import json


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
# TRAIN FUNCTION
# ==============================

def train_multiple_batches(batch_data, batch_targets, batch_indices, weights, lr):
    grads = []

    for idx in batch_indices:
        print("processing batch", idx)
        x_batch = batch_data[idx]
        y_batch = batch_targets[idx]
        ws1, bs1, ws2, bs2 = weights
        m = x_batch.shape[0]

        # Forward
        z1 = x_batch.dot(ws1.T) + bs1
        a1 = np.maximum(0, z1)
        z2 = a1.dot(ws2.T) + bs2
        z2 -= np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z2)
        a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Backward
        dz2 = a2 - y_batch
        # FIXED: Transpose to match ws2 shape (output_size, hidden)
        dw2 = dz2.T.dot(a1) / m  
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2.dot(ws2) * (z1 > 0)
        dw1 = dz1.T.dot(x_batch) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        grads.append((dw1, db1, dw2, db2))

    return grads


# ==============================
# WORKER
# ==============================

class Worker:
    def __init__(self, server_host, server_port):
        self.server_host = server_host
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_host, self.server_port))
        
        self.batch_data = None
        self.batch_targets = None
        self.lr = None

    def receive_dataset(self):
        """Receive dataset: binary arrays first, then JSON metadata."""
        print("Receiving dataset arrays...")
        arrays = recv_arrays(self.sock)
        if arrays is None:
            print("Failed to receive arrays")
            return False
            
        self.batch_data = arrays['batch_data']
        self.batch_targets = arrays['batch_targets']
        print(f"Received {len(self.batch_data)} batches")
        
        print("Receiving dataset metadata...")
        msg = recv_json(self.sock)
        if msg is None:
            print("Failed to receive metadata")
            return False
            
        if msg.get("type") != "dataset":
            print(f"Expected dataset message, got {msg.get('type')}")
            return False
            
        self.lr = msg['lr']
        print("Dataset received successfully")
        
        # Send acknowledgment
        send_json(self.sock, {"status": "ready"})
        return True

    def handle_train_command(self, msg):
        """Handle training command: JSON metadata already received, get weights."""
        batches = msg["batches"]
        lr = msg["lr"]
        
        print(f"Receiving weights for training...")
        weights = recv_arrays(self.sock)
        if weights is None:
            print("Failed to receive weights")
            return
            
        ws1, bs1, ws2, bs2 = weights
        print(f"Training on batches: {batches}")

        grads = train_multiple_batches(
            self.batch_data, self.batch_targets,
            batches, weights, lr
        )
        
        print(f"Sending gradients back...")
        send_arrays(self.sock, grads)
        print("Gradients sent")

    def start(self):
        print(f"Connected to {self.server_host}:{self.server_port}")
        
        # First message must be dataset
        if not self.receive_dataset():
            print("Failed to receive dataset. Disconnecting.")
            return

        # Main loop: receive JSON commands
        while True:
            print("Waiting for command...")
            msg = recv_json(self.sock)
            
            if msg is None:
                print("Server disconnected.")
                break

            msg_type = msg.get("type")
            print(f"Received command: {msg_type}")
            
            if msg_type == "train":
                if self.batch_data is None:
                    print("ERROR: train received before dataset")
                    continue
                self.handle_train_command(msg)
            
            elif msg_type == "done":
                print("Training complete.")
                break
            
            else:
                print(f"Unknown message type: {msg_type}")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Worker")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Parameter Server host")
    parser.add_argument("--port", type=int, default=5000,
                       help="Parameter Server port")

    args = parser.parse_args()

    worker = Worker(
        server_host=args.host,
        server_port=args.port
    )
    worker.start()