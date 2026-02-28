# Revisar que puede estar pasando que se ejecuta muy lento
# a tal punto que no ejecuta, version de memoria compartida

import numpy as np
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from sklearn.datasets import fetch_openml


# =========================================================
# VARIABLES GLOBALES PARA WORKERS
# =========================================================

GLOBAL_BATCH_DATA = None
GLOBAL_BATCH_TARGETS = None
GLOBAL_BATCH_PARAMS = None
GLOBAL_LR = None


# =========================================================
# INIT WORKER
# =========================================================

def init_worker(shm_data_names, shm_target_names,
                data_shapes, target_shapes,
                batch_params, lr):

    global GLOBAL_BATCH_DATA
    global GLOBAL_BATCH_TARGETS
    global GLOBAL_BATCH_PARAMS
    global GLOBAL_LR

    GLOBAL_BATCH_DATA = []
    GLOBAL_BATCH_TARGETS = []

    # Reconstruir arrays desde shared memory
    for name, shape in zip(shm_data_names, data_shapes):
        shm = shared_memory.SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        GLOBAL_BATCH_DATA.append(arr)

    for name, shape in zip(shm_target_names, target_shapes):
        shm = shared_memory.SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        GLOBAL_BATCH_TARGETS.append(arr)

    GLOBAL_BATCH_PARAMS = batch_params
    GLOBAL_LR = lr


# =========================================================
# FUNCIÓN PARALELA
# =========================================================

def train_multiple_batches(batch_indices):

    global GLOBAL_BATCH_DATA
    global GLOBAL_BATCH_TARGETS
    global GLOBAL_BATCH_PARAMS
    global GLOBAL_LR

    updated_params = []
    losses = []

    for idx in batch_indices:

        params = GLOBAL_BATCH_PARAMS[idx]

        x_batch = GLOBAL_BATCH_DATA[idx]
        y_batch = GLOBAL_BATCH_TARGETS[idx]

        ws1 = params['ws1'].copy()
        bs1 = params['bs1'].copy()
        ws2 = params['ws2'].copy()
        bs2 = params['bs2'].copy()

        m = x_batch.shape[0]

        # ------------------ Forward ------------------

        z1 = x_batch.dot(ws1.T) + bs1
        a1 = np.maximum(0, z1)

        z2 = a1.dot(ws2.T) + bs2
        z2 -= np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z2)
        a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        loss = -np.sum(y_batch * np.log(np.clip(a2, 1e-15, 1))) / m
        losses.append(loss)

        # ------------------ Backward ------------------

        dz2 = a2 - y_batch
        dw2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        ws2 -= GLOBAL_LR * dw2.T / m
        bs2 -= GLOBAL_LR * db2 / m

        dz1 = dz2.dot(ws2) * (z1 > 0)
        dw1 = dz1.T.dot(x_batch)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        ws1 -= GLOBAL_LR * dw1 / m
        bs1 -= GLOBAL_LR * db1 / m

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
                 lr=0.1, n_batches=5):

        self.x_train = np.array(x_train, dtype=np.float32)
        self.targets = np.array(targets, dtype=np.float32)
        self.n_iter = n_iter
        self.hidden_layers = n_hidden
        self.lr = lr
        self.n_batches = n_batches
        self.loss_history = []

        self.y_labels = np.argmax(self.targets, axis=1)

        self.initializeWeightsBias(n_hidden)
        self.initializeBatchParameters(n_hidden)

        self.batch_data, self.batch_targets = \
            self.createClassBalancedBatches()

    # ----------------------------------------------------

    def initializeWeightsBias(self, n_hidden):

        input_size = self.x_train.shape[1]
        output_size = self.targets.shape[1]

        self.ws1 = np.random.randn(n_hidden, input_size).astype(np.float32) * np.sqrt(2. / input_size)
        self.bs1 = np.zeros((1, n_hidden), dtype=np.float32)
        self.ws2 = np.random.randn(output_size, n_hidden).astype(np.float32) * np.sqrt(2. / n_hidden)
        self.bs2 = np.zeros((1, output_size), dtype=np.float32)

    # ----------------------------------------------------

    def initializeBatchParameters(self, n_hidden):

        input_size = self.x_train.shape[1]
        output_size = self.targets.shape[1]

        self.batch_params = []

        for _ in range(self.n_batches):
            params = {
                'ws1': np.random.randn(n_hidden, input_size).astype(np.float32) * np.sqrt(2. / input_size),
                'bs1': np.zeros((1, n_hidden), dtype=np.float32),
                'ws2': np.random.randn(output_size, n_hidden).astype(np.float32) * np.sqrt(2. / n_hidden),
                'bs2': np.zeros((1, output_size), dtype=np.float32)
            }
            self.batch_params.append(params)

    # ----------------------------------------------------

    def createClassBalancedBatches(self):

        n_classes = self.targets.shape[1]
        samples_per_class = self.x_train.shape[0] // (self.n_batches * n_classes)

        batch_data = []
        batch_targets = []

        class_indices = [np.where(self.y_labels == c)[0]
                         for c in range(n_classes)]

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

    # ----------------------------------------------------

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

    # ----------------------------------------------------

    def training_parallel(self, n_processes):

        start_time = time.perf_counter()

        # ===== CREAR MEMORIA COMPARTIDA =====

        shm_data = []
        shm_targets = []
        shm_data_names = []
        shm_target_names = []
        data_shapes = []
        target_shapes = []

        for data in self.batch_data:
            shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            shm_array = np.ndarray(data.shape, dtype=np.float32, buffer=shm.buf)
            shm_array[:] = data[:]
            shm_data.append(shm)
            shm_data_names.append(shm.name)
            data_shapes.append(data.shape)

        for target in self.batch_targets:
            shm = shared_memory.SharedMemory(create=True, size=target.nbytes)
            shm_array = np.ndarray(target.shape, dtype=np.float32, buffer=shm.buf)
            shm_array[:] = target[:]
            shm_targets.append(shm)
            shm_target_names.append(shm.name)
            target_shapes.append(target.shape)

        # ===== TRAINING =====

        for epoch in range(self.n_iter):

            indices = list(range(self.n_batches))
            split_indices = np.array_split(indices, n_processes)

            with mp.Pool(
                processes=n_processes,
                initializer=init_worker,
                initargs=(
                    shm_data_names,
                    shm_target_names,
                    data_shapes,
                    target_shapes,
                    self.batch_params,
                    self.lr
                )
            ) as pool:

                results = pool.map(train_multiple_batches, split_indices)

            new_params = []
            epoch_losses = []

            for updated_list, loss in results:
                new_params.extend(updated_list)
                epoch_losses.append(loss)

            for i in range(self.n_batches):
                self.batch_params[i] = new_params[i]

            self.averageParameters()
            self.loss_history.append(np.mean(epoch_losses))

            print(f"Epoch {epoch+1}/{self.n_iter} - Loss: {self.loss_history[-1]:.4f}")

        # ===== LIBERAR SHM =====
        for shm in shm_data + shm_targets:
            shm.close()
            shm.unlink()

        wall_time = time.perf_counter() - start_time
        print(f"Wall time total: {wall_time:.4f} segundos")

    # ----------------------------------------------------

    def predict(self, x):

        z1 = x.dot(self.ws1.T) + self.bs1
        a1 = np.maximum(0, z1)

        z2 = a1.dot(self.ws2.T) + self.bs2
        z2 -= np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z2)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# =========================================================
# UTILIDADES
# =========================================================

def load_mnist():

    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)

    one_hot = np.zeros((y.size, 10), dtype=np.float32)
    one_hot[np.arange(y.size), y] = 1

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

    X, Y = load_mnist()

    x_train = X[:60000]
    y_train = Y[:60000]
    x_test = X[60000:70000]
    y_test = Y[60000:70000]

    model = CBNN(x_train, y_train, 25, 50, lr=0.05, n_batches=5)

    print("Tiempo para 2 procesos")
    model.training_parallel(n_processes=2)

    acc = evaluate_accuracy(model, x_test, y_test)
    print(f"Accuracy final: {acc:.4f}")