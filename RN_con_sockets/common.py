import pickle
import struct
import numpy as np
import multiprocessing as mp

# ==============================
# SOCKET UTILITIES
# ==============================

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
# GLOBAL DATA PER PROCESS
# ==============================

GLOBAL_BATCH_DATA = None
GLOBAL_BATCH_TARGETS = None

def init_worker(batch_data, batch_targets):
    global GLOBAL_BATCH_DATA, GLOBAL_BATCH_TARGETS
    GLOBAL_BATCH_DATA = batch_data
    GLOBAL_BATCH_TARGETS = batch_targets

# ==============================
# TRAIN FUNCTION
# ==============================

def train_multiple_batches(args):
    global GLOBAL_BATCH_DATA, GLOBAL_BATCH_TARGETS

    batch_indices, weights, lr = args
    grads = []

    for idx in batch_indices:

        x_batch = GLOBAL_BATCH_DATA[idx]
        y_batch = GLOBAL_BATCH_TARGETS[idx]

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