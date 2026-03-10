import socket
import argparse
import pickle
import struct
import numpy as np
import json
import time


def send_json(sock, obj):
    data = json.dumps(obj).encode('utf-8')
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_json(sock):
    raw_len = recv_all(sock, 4)
    if raw_len is None:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    data = recv_all(sock, msg_len)
    if data is None:
        return None
    return json.loads(data.decode('utf-8'))


def send_arrays(sock, arrays):
    data = pickle.dumps(arrays, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_arrays(sock):
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


def train_multiple_batches(batch_data, batch_targets, batch_indices, weights, lr):

    grads = []

    for idx in batch_indices:
        x_batch = batch_data[idx]
        y_batch = batch_targets[idx]

        ws1, bs1, ws2, bs2 = weights
        m = x_batch.shape[0]

        z1 = x_batch.dot(ws1.T) + bs1
        a1 = np.maximum(0, z1)

        z2 = a1.dot(ws2.T) + bs2
        z2 -= np.max(z2, axis=1, keepdims=True)

        exp_z = np.exp(z2)
        a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        dz2 = a2 - y_batch
        dw2 = dz2.T.dot(a1) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2.dot(ws2) * (z1 > 0)
        dw1 = dz1.T.dot(x_batch) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        grads.append((dw1, db1, dw2, db2))

    return grads


class Worker:

    def __init__(self, host, port):

        self.host = host
        self.port = port

    def run(self):

        while True:

            try:

                print("Trying to connect to server...")

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))

                print("Connected!")

                arrays = recv_arrays(sock)

                if arrays is None:
                    raise Exception("No dataset received")

                batch_data = arrays['batch_data']
                batch_targets = arrays['batch_targets']

                msg = recv_json(sock)

                if msg is None:
                    raise Exception("No dataset metadata")

                if msg.get("type") == "rejected":
                    print("Server rejected connection:", msg.get("reason"))
                    sock.close()
                    time.sleep(3)
                    continue

                lr = msg['lr']

                send_json(sock, {"status": "ready"})

                while True:

                    msg = recv_json(sock)

                    if msg is None:
                        raise Exception("Server disconnected")

                    msg_type = msg.get("type")

                    if msg_type == "train":

                        batches = msg["batches"]
                        lr = msg["lr"]

                        weights = recv_arrays(sock)

                        grads = train_multiple_batches(
                            batch_data,
                            batch_targets,
                            batches,
                            weights,
                            lr
                        )

                        send_arrays(sock, grads)

                    elif msg_type == "done":

                        print("Training finished. Reconnecting soon...")
                        sock.close()
                        time.sleep(5)
                        break

            except Exception as e:

                print("Connection error:", e)
                time.sleep(3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    worker = Worker(args.host, args.port)

    worker.run()