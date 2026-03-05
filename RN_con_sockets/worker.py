# import socket
# import multiprocessing as mp
# import argparse
# # from common import send_msg, recv_msg, init_worker, train_multiple_batches

# import pickle
# import struct
# import numpy as np
# # import multiprocessing as mp

# def send_msg(sock, obj):
#     data = pickle.dumps(obj)
#     sock.sendall(struct.pack(">I", len(data)) + data)

# def recv_msg(sock):
#     raw_len = recv_all(sock, 4)
#     if not raw_len:
#         return None
#     msg_len = struct.unpack(">I", raw_len)[0]
#     return pickle.loads(recv_all(sock, msg_len))

# def recv_all(sock, n):
#     data = b''
#     while len(data) < n:
#         packet = sock.recv(n - len(data))
#         if not packet:
#             return None
#         data += packet
#     return data

# # ==============================
# # GLOBAL DATA PER PROCESS
# # ==============================

# GLOBAL_BATCH_DATA = None
# GLOBAL_BATCH_TARGETS = None

# def init_worker(batch_data, batch_targets):
#     global GLOBAL_BATCH_DATA, GLOBAL_BATCH_TARGETS
#     GLOBAL_BATCH_DATA = batch_data
#     GLOBAL_BATCH_TARGETS = batch_targets

# # ==============================
# # TRAIN FUNCTION
# # ==============================

# def train_multiple_batches(args):
#     global GLOBAL_BATCH_DATA, GLOBAL_BATCH_TARGETS

#     batch_indices, weights, lr = args
#     grads = []

#     for idx in batch_indices:

#         print("hola")

#         x_batch = GLOBAL_BATCH_DATA[idx]
#         y_batch = GLOBAL_BATCH_TARGETS[idx]

#         ws1, bs1, ws2, bs2 = weights
#         m = x_batch.shape[0]

#         # Forward
#         z1 = x_batch.dot(ws1.T) + bs1
#         a1 = np.maximum(0, z1)

#         z2 = a1.dot(ws2.T) + bs2
#         z2 -= np.max(z2, axis=1, keepdims=True)
#         exp_z = np.exp(z2)
#         a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)

#         # Backward
#         dz2 = a2 - y_batch
#         dw2 = a1.T.dot(dz2) / m
#         db2 = np.sum(dz2, axis=0, keepdims=True) / m

#         dz1 = dz2.dot(ws2) * (z1 > 0)
#         dw1 = dz1.T.dot(x_batch) / m
#         db1 = np.sum(dz1, axis=0, keepdims=True) / m

#         grads.append((dw1, db1, dw2, db2))

#     return grads

# class Worker:

#     def __init__(self, server_host, server_port, n_processes):
#         self.server_host = server_host
#         self.server_port = server_port
#         self.n_processes = n_processes

#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.sock.connect((self.server_host, self.server_port))

#         self.batch_data = None
#         self.batch_targets = None

#     def start(self):

#         print(f"Connected to {self.server_host}:{self.server_port}")
#         print(f"Using {self.n_processes} local processes")

#         while True:

#             msg = recv_msg(self.sock)
#             if msg is None:
#                 print("Server disconnected.")
#                 break

#             if msg["type"] == "dataset":
#                 self.batch_data = msg["batch_data"]
#                 self.batch_targets = msg["batch_targets"]
#                 print("Dataset received.")

#             elif msg["type"] == "train":

#                 batches = msg["batches"]
#                 weights = msg["weights"]
#                 lr = msg["lr"]

#                 # Crear pool multiproceso
#                 pool = mp.Pool(
#                     processes=self.n_processes,
#                     initializer=init_worker,
#                     initargs=(self.batch_data, self.batch_targets)
#                 )

#                 args = [(batches, weights, lr)]
#                 results = pool.map(train_multiple_batches, args)

#                 pool.close()
#                 pool.join()

#                 grads = []
#                 for r in results:
#                     grads.extend(r)

#                 send_msg(self.sock, grads)


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="Distributed Worker")

#     parser.add_argument(
#         "--host",
#         type=str,
#         default="127.0.0.1",
#         help="Parameter Server host"
#     )

#     parser.add_argument(
#         "--port",
#         type=int,
#         default=5000,
#         help="Parameter Server port"
#     )

#     parser.add_argument(
#         "--processes",
#         type=int,
#         default=1,
#         help="Number of local CPU processes"
#     )

#     args = parser.parse_args()

#     worker = Worker(
#         server_host=args.host,
#         server_port=args.port,
#         n_processes=args.processes
#     )

#     worker.start()


import socket
import argparse
import pickle
import struct
import numpy as np


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
# TRAIN FUNCTION (SIN GLOBALS)
# ==============================

def train_multiple_batches(batch_data, batch_targets, batch_indices, weights, lr):

    grads = []

    for idx in batch_indices:

        print("hola")

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
        dw2 = a1.T.dot(dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2.dot(ws2) * (z1 > 0)
        dw1 = dz1.T.dot(x_batch) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        grads.append((dw1, db1, dw2, db2))

    return grads


class Worker:

    def __init__(self, server_host, server_port, n_processes):
        self.server_host = server_host
        self.server_port = server_port
        self.n_processes = n_processes

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_host, self.server_port))

        self.batch_data = None
        self.batch_targets = None

    def start(self):

        print(f"Connected to {self.server_host}:{self.server_port}")
        print(f"Using {self.n_processes} local processes")

        while True:

            msg = recv_msg(self.sock)
            if msg is None:
                print("Server disconnected.")
                break

            if msg["type"] == "dataset":
                self.batch_data = msg["batch_data"]
                self.batch_targets = msg["batch_targets"]
                print("Dataset received.")

            elif msg["type"] == "train":

                batches = msg["batches"]
                weights = msg["weights"]
                lr = msg["lr"]

                # 🔥 EJECUCIÓN SECUENCIAL
                grads = train_multiple_batches(
                    self.batch_data,
                    self.batch_targets,
                    batches,
                    weights,
                    lr
                )

                send_msg(self.sock, grads)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distributed Worker")

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Parameter Server host"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Parameter Server port"
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of local CPU processes"
    )

    args = parser.parse_args()

    worker = Worker(
        server_host=args.host,
        server_port=args.port,
        n_processes=args.processes
    )

    worker.start()