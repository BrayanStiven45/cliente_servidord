import numpy as np
import matplotlib.pyplot as plt

class CBNN:
    def __init__(self, x_train: list, targets: list, n_iter: int, n_hidden: int, lr: float = 0.1, n_batches: int = 5):
        """
        Initializes the neural network with training data and basic parameters.
        
        Parameters:
        - x_train (list): Input data for training.
        - targets (list): Target values (one-hot encoded).
        - n_iter (int): Number of training iterations (epochs).
        - n_hidden (int): Number of neurons in the hidden layer.
        - lr (float): Learning rate.
        - n_batches (int): Number of class-balanced batches to create.
        """
        self.x_train = np.array(x_train)
        self.targets = np.array(targets)
        self.n_iter = n_iter 
        self.hidden_layers = n_hidden
        self.lr = lr
        self.n_batches = n_batches
        
        # Store original labels for class-balanced batching (convert one-hot back to indices)
        self.y_labels = np.argmax(self.targets, axis=1)
        
        self.initializeWeightsBias(n_hidden)
        
        # Initialize 5 separate sets of weights and biases for each batch
        self.initializeBatchParameters(n_hidden)
        
        # NEW: Store parameters at the end of each epoch for final averaging
        self.epoch_parameters = []

    def initializeWeightsBias(self, n_hidden: int) -> None:
        """Initialize main weights and biases (used for averaging)."""
        input_size = self.x_train.shape[1]
        output_size = self.targets.shape[1]
        
        # Main parameters (will store the averaged values)
        self.ws1 = np.random.randn(n_hidden, input_size) * np.sqrt(2. / input_size)
        self.bs1 = np.zeros([1, n_hidden])
        self.ws2 = np.random.randn(output_size, n_hidden) * np.sqrt(2. / n_hidden)
        self.bs2 = np.zeros([1, output_size])

    def initializeBatchParameters(self, n_hidden: int) -> None:
        """Initialize separate parameters for each of the 5 batches."""
        input_size = self.x_train.shape[1]
        output_size = self.targets.shape[1]
        
        # Create 5 sets of parameters: ws1_b1, ws1_b2, ..., ws1_b5, etc.
        self.batch_params = []
        
        for i in range(self.n_batches):
            params = {
                'ws1': np.random.randn(n_hidden, input_size) * np.sqrt(2. / input_size),
                'bs1': np.zeros([1, n_hidden]),
                'ws2': np.random.randn(output_size, n_hidden) * np.sqrt(2. / n_hidden),
                'bs2': np.zeros([1, output_size])
            }
            self.batch_params.append(params)

    def createClassBalancedBatches(self):
        """
        Creates n_batches class-balanced batches.
        Each batch contains equal number of samples from each class.
        """
        n_classes = self.targets.shape[1]
        samples_per_class_per_batch = self.x_train.shape[0] // (self.n_batches * n_classes)
        
        batch_data = []
        batch_targets = []
        
        # Group indices by class
        class_indices = [np.where(self.y_labels == c)[0] for c in range(n_classes)]
        
        # Shuffle each class's indices
        for indices in class_indices:
            np.random.shuffle(indices)
        
        # Create balanced batches
        for b in range(self.n_batches):
            batch_indices = []
            for c in range(n_classes):
                start_idx = b * samples_per_class_per_batch
                end_idx = (b + 1) * samples_per_class_per_batch
                # Handle case where we might run out of samples
                if end_idx > len(class_indices[c]):
                    end_idx = len(class_indices[c])
                batch_indices.extend(class_indices[c][start_idx:end_idx])
            
            # Shuffle the batch indices
            batch_indices = np.array(batch_indices)
            np.random.shuffle(batch_indices)
            
            batch_data.append(self.x_train[batch_indices])
            batch_targets.append(self.targets[batch_indices])
        
        return batch_data, batch_targets

    def f(self, x: np.ndarray, ws: np.ndarray, bs: np.ndarray) -> np.ndarray:
        """Performs the weighted sum of a perceptron."""
        z = x.dot(ws.T) + bs
        return z

    def relu(self, x):
        return np.where(x >= 0, x, 0)

    def gradiente_relu(self, x):
        return np.where(x >= 0, 1, 0)

    def softmax(self, z):
        s = np.sum(np.exp(z), axis=1)
        return np.exp(z) / s[:, np.newaxis]

    def log_loss(self, y, p):
        # Clip to avoid log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -y * np.log(p) - (1 - y) * np.log(1 - p)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts using the averaged (main) parameters."""
        z1 = self.f(x, self.ws1, self.bs1)
        act1 = self.relu(z1)
        z2 = self.f(act1, self.ws2, self.bs2)
        act2 = self.softmax(z2)
        return act2

    def trainBatch(self, x_batch, y_batch, params, batch_size):
        """Trains on a single batch using specific parameters."""
        n_samples = x_batch.shape[0]
        total_loss = 0
        
        # Create copies to update
        ws1 = params['ws1'].copy()
        bs1 = params['bs1'].copy()
        ws2 = params['ws2'].copy()
        bs2 = params['bs2'].copy()
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            x_mini = x_batch[start_idx:end_idx]
            y_mini = y_batch[start_idx:end_idx]
            current_batch_size = end_idx - start_idx
            
            # Forward pass
            z1 = self.f(x_mini, ws1, bs1)
            act1 = self.relu(z1)
            z2 = self.f(act1, ws2, bs2)
            act2 = self.softmax(z2)
            
            # Loss
            cost = self.log_loss(y_mini, act2)
            total_loss += np.sum(cost)
            
            # Backward pass
            z_d_2 = act2 - y_mini
            w_d_2 = act1.T.dot(z_d_2)
            b_d_2 = np.sum(z_d_2, axis=0, keepdims=True)
            
            ws2 -= self.lr * w_d_2.T / current_batch_size
            bs2 -= self.lr * b_d_2 / current_batch_size
            
            cost_l1 = z_d_2.dot(ws2) * self.gradiente_relu(z1)
            w_d_1 = cost_l1.T.dot(x_mini)
            b_d_1 = np.sum(cost_l1, axis=0, keepdims=True)
            
            ws1 -= self.lr * w_d_1 / current_batch_size
            bs1 -= self.lr * b_d_1 / current_batch_size
        
        # Update params dictionary
        params['ws1'] = ws1
        params['bs1'] = bs1
        params['ws2'] = ws2
        params['bs2'] = bs2
        
        return total_loss / n_samples

    def averageParameters(self):
        """Averages the parameters from all 5 batches into the main parameters."""
        # Initialize accumulators
        avg_ws1 = np.zeros_like(self.ws1)
        avg_bs1 = np.zeros_like(self.bs1)
        avg_ws2 = np.zeros_like(self.ws2)
        avg_bs2 = np.zeros_like(self.bs2)
        
        # Sum all batch parameters
        for params in self.batch_params:
            avg_ws1 += params['ws1']
            avg_bs1 += params['bs1']
            avg_ws2 += params['ws2']
            avg_bs2 += params['bs2']
        
        # Average
        self.ws1 = avg_ws1 / self.n_batches
        self.bs1 = avg_bs1 / self.n_batches
        self.ws2 = avg_ws2 / self.n_batches
        self.bs2 = avg_bs2 / self.n_batches
        
        # Copy averaged parameters back to each batch for next iteration
        for params in self.batch_params:
            params['ws1'] = self.ws1.copy()
            params['bs1'] = self.bs1.copy()
            params['ws2'] = self.ws2.copy()
            params['bs2'] = self.bs2.copy()

    def training(self, batch_size=32):
        """Training with class-balanced batches and parameter averaging."""
        # Create class-balanced batches once
        batch_data, batch_targets = self.createClassBalancedBatches()
        
        print(f"Created {self.n_batches} class-balanced batches")
        for i, (bd, bt) in enumerate(zip(batch_data, batch_targets)):
            print(f"  Batch {i+1}: {bd.shape[0]} samples, classes: {np.unique(np.argmax(bt, axis=1))}")
        
        for epoch in range(self.n_iter):
            epoch_losses = []
            
            # Train each of the 5 batches with their own parameters
            for b in range(self.n_batches):
                loss = self.trainBatch(
                    batch_data[b], 
                    batch_targets[b], 
                    self.batch_params[b], 
                    batch_size
                )
                epoch_losses.append(loss)
            
            # Average all parameters across the 5 batches
            # self.averageParameters()
            
            # NEW: Store parameters at the end of this epoch
            # self.storeCurrentParameters()
            
            avg_loss = np.mean(epoch_losses)
            print(f'Epoch {epoch} - Avg Loss: {avg_loss:.4f} (Batch losses: {[f"{l:.4f}" for l in epoch_losses]})')
        
        # NEW: After all epochs, calculate final average of all epoch parameters
        # Average all parameters across the 5 batches
        self.averageParameters()
        # self.calculateFinalAverageParameters()