import numpy as np
import multiprocessing as mp
import time
import os
from sklearn.datasets import fetch_openml


# =========================================================
# VARIABLES GLOBALES EN CADA PROCESO
# =========================================================

GLOBAL_BATCH_DATA = None
GLOBAL_BATCH_TARGETS = None


def init_worker(batch_data, batch_targets):
    global GLOBAL_BATCH_DATA, GLOBAL_BATCH_TARGETS
    GLOBAL_BATCH_DATA = batch_data
    GLOBAL_BATCH_TARGETS = batch_targets


# =========================================================
# FUNCIÓN PARALELA
# =========================================================

def train_multiple_batches(args):

    global GLOBAL_BATCH_DATA, GLOBAL_BATCH_TARGETS

    batch_indices, batch_params, lr = args

    updated_params = []
    losses = []

    for idx in batch_indices:

        params = batch_params[idx]

        x_batch = GLOBAL_BATCH_DATA[idx]
        y_batch = GLOBAL_BATCH_TARGETS[idx]

        ws1 = params['ws1'].copy()
        bs1 = params['bs1'].copy()
        ws2 = params['ws2'].copy()
        bs2 = params['bs2'].copy()

        m = x_batch.shape[0]

        # Forward
        z1 = x_batch.dot(ws1.T) + bs1
        a1 = np.maximum(0, z1)

        z2 = a1.dot(ws2.T) + bs2
        z2 -= np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z2)
        a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        loss = -np.sum(y_batch * np.log(np.clip(a2, 1e-15, 1))) / m
        losses.append(loss)

        # Backward
        dz2 = a2 - y_batch
        dw2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        ws2 -= lr * dw2.T / m
        bs2 -= lr * db2 / m

        dz1 = dz2.dot(ws2) * (z1 > 0)
        dw1 = dz1.T.dot(x_batch)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        ws1 -= lr * dw1 / m
        bs1 -= lr * db1 / m

        updated_params.append({
            'ws1': ws1,
            'bs1': bs1,
            'ws2': ws2,
            'bs2': bs2
        })

    return updated_params, np.mean(losses)


# =========================================================
# CLASE
# =========================================================

class CBNN:

    def __init__(self, x_train, targets, n_iter, n_hidden,
                 lr=0.1, n_batches=2):

        self.x_train = np.array(x_train)
        self.targets = np.array(targets)
        self.n_iter = n_iter
        self.hidden_layers = n_hidden
        self.lr = lr
        self.n_batches = n_batches

        self.loss_history = []
        self.y_labels = np.argmax(self.targets, axis=1)

        self.initializeWeightsBias(n_hidden)
        self.initializeBatchParameters()
        self.batch_data, self.batch_targets = self.createClassBalancedBatches()

    def initializeWeightsBias(self, n_hidden):

        input_size = self.x_train.shape[1]
        output_size = self.targets.shape[1]

        self.ws1 = np.random.randn(n_hidden, input_size) * np.sqrt(2. / input_size)
        self.bs1 = np.zeros((1, n_hidden))
        self.ws2 = np.random.randn(output_size, n_hidden) * np.sqrt(2. / n_hidden)
        self.bs2 = np.zeros((1, output_size))

    def initializeBatchParameters(self):

        self.batch_params = []
        for _ in range(self.n_batches):
            self.batch_params.append({
                'ws1': self.ws1.copy(),
                'bs1': self.bs1.copy(),
                'ws2': self.ws2.copy(),
                'bs2': self.bs2.copy()
            })

    def createClassBalancedBatches(self):

        n_classes = self.targets.shape[1]
        samples_per_class = self.x_train.shape[0] // (self.n_batches * n_classes)

        batch_data = []
        batch_targets = []

        class_indices = [
            np.where(self.y_labels == c)[0]
            for c in range(n_classes)
        ]

        for indices in class_indices:
            np.random.shuffle(indices)

        for b in range(self.n_batches):

            batch_indices = []

            for c in range(n_classes):
                start = b * samples_per_class
                end = (b + 1) * samples_per_class
                batch_indices.extend(class_indices[c][start:end])

            batch_indices = np.array(batch_indices)
            np.random.shuffle(batch_indices)

            batch_data.append(self.x_train[batch_indices])
            batch_targets.append(self.targets[batch_indices])

        return batch_data, batch_targets

    def averageParameters(self):

        avg_ws1 = np.zeros_like(self.ws1)
        avg_bs1 = np.zeros_like(self.bs1)
        avg_ws2 = np.zeros_like(self.ws2)
        avg_bs2 = np.zeros_like(self.bs2)

        for params in self.batch_params:
            avg_ws1 += params['ws1']
            avg_bs1 += params['bs1']
            avg_ws2 += params['ws2']
            avg_bs2 += params['bs2']

        self.ws1 = avg_ws1 / self.n_batches
        self.bs1 = avg_bs1 / self.n_batches
        self.ws2 = avg_ws2 / self.n_batches
        self.bs2 = avg_bs2 / self.n_batches

        for params in self.batch_params:
            params['ws1'] = self.ws1.copy()
            params['bs1'] = self.bs1.copy()
            params['ws2'] = self.ws2.copy()
            params['bs2'] = self.bs2.copy()

    def training_parallel(self, n_processes):

        if n_processes != self.n_batches:
            raise ValueError("n_processes debe ser igual a n_batches.")

        start_time = time.perf_counter()

        pool = mp.Pool(
            processes=n_processes,
            initializer=init_worker,
            initargs=(self.batch_data, self.batch_targets)
        )

        for epoch in range(self.n_iter):

            args = [
                ([idx], self.batch_params, self.lr)
                for idx in range(self.n_batches)
            ]

            results = pool.map(train_multiple_batches, args)

            self.batch_params = []
            epoch_losses = []

            for updated_list, loss in results:
                self.batch_params.extend(updated_list)
                epoch_losses.append(loss)

            self.averageParameters()
            self.loss_history.append(np.mean(epoch_losses))

        pool.close()
        pool.join()

        wall_time = time.perf_counter() - start_time
        return wall_time

    def predict(self, x):

        z1 = x.dot(self.ws1.T) + self.bs1
        a1 = np.maximum(0, z1)

        z2 = a1.dot(self.ws2.T) + self.bs2
        z2 -= np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z2)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# =========================================================
# DATASET OPTIMIZADO
# =========================================================

def load_mnist_cached():

    if os.path.exists("mnist_X.npy") and os.path.exists("mnist_Y.npy"):
        X = np.load("mnist_X.npy")
        Y = np.load("mnist_Y.npy")
        return X, Y

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)

    one_hot = np.zeros((y.size, 10))
    one_hot[np.arange(y.size), y] = 1

    np.save("mnist_X.npy", X)
    np.save("mnist_Y.npy", one_hot)

    return X, one_hot


def evaluate_accuracy(model, x_test, y_test):

    predictions = model.predict(x_test)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    return np.mean(pred_labels == true_labels)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)

    args = parser.parse_args()

    X, Y = load_mnist_cached()

    x_train = X[:60000]
    y_train = Y[:60000]
    x_test = X[60000:70000]
    y_test = Y[60000:70000]

    model = CBNN(
        x_train,
        y_train,
        n_iter=args.epochs,
        n_hidden=args.hidden,
        lr=args.lr,
        n_batches=args.processes
    )

    wall_time = model.training_parallel(n_processes=args.processes)

    acc = evaluate_accuracy(model, x_test, y_test)

    print(f"time={wall_time:.6f}")
    print(f"accuracy={acc:.6f}")
