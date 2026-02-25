import numpy as np
import matplotlib.pyplot as plt

class CBNN:

    def __init__(self, x_train: list, targets: list, n_iter: int, n_hidden: int, lr: int = 0.1):
        """
        Initializes the neural network with training data and basic parameters.

        Parameters:
        - x_train (list): Input data for training.
        - targets (list): Target values that the neural network should learn to predict.
        - n_iter (int): Number of training iterations.
        - n_hidden (int): Number of neurons in the hidden layer.
        - lr (int): learning rate 

        """
        self.x_train = np.array(x_train)
        self.targets = np.array(targets)
        self.n_iter = n_iter 
        self.hidden_layers = n_hidden
        self.initializeWeightsBias(n_hidden)
        self.lr = lr

    def initializeWeightsBias(self, n_hidden: int) -> None:
        """
        Initializes the weight with method 'He initialization' and bias values for the hidden layer and the output layer

        Parameters:
        - n_hidden (int): Number of neurons in the hidden layer.
        
        """
        input_size = self.x_train.shape[1] # number of features in the input data
        output_size = self.targets.shape[1] # number of output neurons

        # 
        self.ws1 = np.random.randn(n_hidden, input_size) * np.sqrt(2. / input_size)
        
        self.bs1 = np.zeros([1, n_hidden])

        self.ws2 = np.random.randn(output_size, n_hidden) * np.sqrt(2. / n_hidden)
        
        self.bs2 = np.zeros([1, output_size])
        


    def f(self, x: np.ndarray, ws: np.ndarray, bs: np.ndarray) -> np.ndarray:
        """
        Performs the weighted sum of a perceptron(neuron)

        Parameters:
        - x (np.ndarray): Input data
        - ws (np.ndarray): weights values for the neural layer
        - bs (np.ndarray): bias values for the neural layer

        Return:
        - z(np.ndarray): The weighted sum resultating from dot product between of the input data and weights, plus bias
        """
        z = x.dot(ws.T) + bs
        return z

    def relu(self, x):
        act = np.where(x >= 0, x, 0)
        return act

    def gradiente_relu(self, x):
        x_d = np.where(x >= 0, 1, 0)
        return x_d

    def softmax(self, z):
        s=np.sum(np.exp(z), axis=1)
        return (np.exp(z)/s[:,np.newaxis]) #suma de cada fila, se agrega una nueva dimension para que sea un vector columna


    def log_loss(self, y,p):
        return -y * np.log(p) - (1-y) * np.log(1-p)
    
    # def initalizer(self):
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output for a given input using the trained weights and biases.

        Parameters:
        - x (np.ndarray): Input data for prediction.

        Return:
        - np.ndarray: Predicted output.
        """
        z1 = self.f(x, self.ws1, self.bs1)
        act1 = self.relu(z1)

        z2 = self.f(act1, self.ws2, self.bs2)
        act2 = self.softmax(z2)

        return act2

    def training(self, batch_size = 1):
        n_samples = self.x_train.shape[0]
        
        for i in range(self.n_iter):
            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size

                x_batch = self.x_train[start_idx : end_idx + 1]
                y_batch = self.targets[start_idx : end_idx + 1]

                z1 = self.f(x_batch, self.ws1, self.bs1)
                act1 = self.relu(z1)
                z2 = self.f(act1, self.ws2, self.bs2)
                act2 = self.softmax(z2)

                cost = self.log_loss(y_batch, act2)
                mse = np.mean(cost)

                z_d_2 = act2 - y_batch
                w_d_2 = act1.T.dot(z_d_2)
                b_d_2 = np.sum(z_d_2, axis=0, keepdims=True)

                self.ws2 -= self.lr * w_d_2.T / batch_size
                self.bs2 -= self.lr * b_d_2 / batch_size

                cost_l1 = z_d_2.dot(self.ws2) * self.gradiente_relu(z1)
                w_d_1 = cost_l1.T.dot(x_batch)
                b_d_1 = np.sum(cost_l1, axis=0, keepdims=True)

                self.ws1 -= self.lr * w_d_1 / batch_size
                self.bs1 -= self.lr * b_d_1 / batch_size
            
            print(f'Epoch {i} - Loss: {mse:.4f}')

            