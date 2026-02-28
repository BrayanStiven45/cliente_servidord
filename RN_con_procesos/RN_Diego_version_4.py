# =========================================================
# ENTRENAMIENTO PARALELO CON COLAS (PROCESOS PERSISTENTES)
# =========================================================

import numpy as np
import multiprocessing as mp
import time
from sklearn.datasets import fetch_openml


# =========================================================
# WORKER
# =========================================================

def worker(task_queue, result_queue):

    while True:

        task = task_queue.get()

        if task is None:  # señal de cierre
            break

        (batch_idx,
         x_batch,
         y_batch,
         params,
         lr) = task

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

        # ------------------ Backward ------------------

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

        updated = {
            'ws1': ws1,
            'bs1': bs1,
            'ws2': ws2,
            'bs2': bs2
        }

        result_queue.put((batch_idx, updated, loss))


# =========================================================
# CLASE
# =========================================================

class CBNN:

    def __init__(self, x_train, targets, n_iter, n_hidden,
                 lr=0.1, n_batches=5):

        self.x_train = np.array(x_train)
        self.targets = np.array(targets)
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

        self.ws1 = np.random.randn(n_hidden, input_size) * np.sqrt(2. / input_size)
        self.bs1 = np.zeros((1, n_hidden))
        self.ws2 = np.random.randn(output_size, n_hidden) * np.sqrt(2. / n_hidden)
        self.bs2 = np.zeros((1, output_size))

    # ----------------------------------------------------

    def initializeBatchParameters(self, n_hidden):

        input_size = self.x_train.shape[1]
        output_size = self.targets.shape[1]

        self.batch_params = []

        for _ in range(self.n_batches):
            params = {
                'ws1': np.random.randn(n_hidden, input_size) * np.sqrt(2. / input_size),
                'bs1': np.zeros((1, n_hidden)),
                'ws2': np.random.randn(output_size, n_hidden) * np.sqrt(2. / n_hidden),
                'bs2': np.zeros((1, output_size))
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

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

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
    # ENTRENAMIENTO PARALELO CON COLA
    # ----------------------------------------------------

    def training_parallel(self, n_processes):

        start_time = time.perf_counter()

        task_queue = mp.Queue()
        result_queue = mp.Queue()

        processes = []

        # Crear procesos persistentes
        for _ in range(n_processes):
            p = mp.Process(target=worker,
                           args=(task_queue, result_queue))
            p.start()
            processes.append(p)

        for epoch in range(self.n_iter):

            # Enviar tareas
            for batch_idx in range(self.n_batches):

                task_queue.put((
                    batch_idx,
                    self.batch_data[batch_idx],
                    self.batch_targets[batch_idx],
                    self.batch_params[batch_idx],
                    self.lr
                ))

            new_params = [None] * self.n_batches
            losses = []

            # Recibir resultados
            for _ in range(self.n_batches):
                batch_idx, updated, loss = result_queue.get()
                new_params[batch_idx] = updated
                losses.append(loss)

            self.batch_params = new_params

            self.averageParameters()
            self.loss_history.append(np.mean(losses))

            p#rint(f"Epoch {epoch+1}/{self.n_iter} - "
              #    f"Loss: {self.loss_history[-1]:.4f}")

        # Señal de cierre
        for _ in processes:
            task_queue.put(None)

        for p in processes:
            p.join()

        wall_time = time.perf_counter() - start_time
        print(f"Wall time total: {wall_time:.4f} segundos")

    # ----------------------------------------------------

    def predict(self, x):

        z1 = x.dot(self.ws1.T) + self.bs1
        a1 = np.maximum(0, z1)

        z2 = a1.dot(self.ws2.T) + self.bs2
        return self.softmax(z2)


# =========================================================
# UTILIDADES
# =========================================================

def load_mnist():

    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)

    one_hot = np.zeros((y.size, 10))
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

    print("Entrenando con 2 procesos")

    model = CBNN(x_train, y_train, 25, 50, lr=0.05, n_batches=5)
    model.training_parallel(n_processes=2)

    acc = evaluate_accuracy(model, x_test, y_test)
    print(f"Accuracy final: {acc:.4f}")