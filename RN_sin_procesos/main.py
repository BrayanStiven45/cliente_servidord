# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
# from RN_Diego import CBNN
# import numpy as np
# import matplotlib.pyplot as plt

# # Load data
# (x_train, targets), (test_inputs, test_targets) = mnist.load_data()

# # Preprocess
# x_train = x_train / 255.0
# x_train = x_train.reshape(-1, 784)

# test_show = test_targets
# targets = to_categorical(targets, num_classes=10)

# test_inputs = test_inputs.reshape(-1, 784)
# test_inputs = test_inputs / 255.0
# test_targets = to_categorical(test_targets, num_classes=10)

# # Initialize with 5 batches
# rn = CBNN(
#     x_train=x_train, 
#     targets=targets, 
#     n_iter=100, 
#     n_hidden=20, 
#     lr=0.01,
#     n_batches=60  # 5 class-balanced batches
# )

# # Train with mini-batch size
# rn.training(batch_size=32)

# # Predict using averaged parameters
# print('\nPredictions:')
# prediction = rn.predict(np.array(test_inputs))

# print(f'Target\t\tPrediction')
# for i in range(20):
#     true_label = test_show[i]
#     pred_label = np.argmax(prediction[i])
#     print(f'{true_label}\t\t{pred_label}')

from sklearn.datasets import fetch_openml
from RN_Diego import CBNN
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. LOAD MNIST (NO TENSORFLOW)
# ==============================
print("Downloading MNIST dataset (first run only)...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int32)

print("Dataset loaded.")
print("Total samples:", X.shape[0])

# ==============================
# 2. SPLIT TRAIN / TEST
# ==============================
x_train = X[:60000]
test_inputs = X[60000:]

targets = y[:60000]
test_targets = y[60000:]

print("Train shape:", x_train.shape)
print("Test shape:", test_inputs.shape)

# ==============================
# 3. NORMALIZE PIXELS
# ==============================
x_train = x_train / 255.0
test_inputs = test_inputs / 255.0

# ==============================
# 4. ONE-HOT ENCODING (NO TF)
# ==============================
def to_categorical(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# Save original labels for printing later
test_show = test_targets.copy()

targets = to_categorical(targets, 10)
test_targets = to_categorical(test_targets, 10)

# ==============================
# 5. OPTIONAL: SHOW SAMPLE IMAGE
# ==============================
# plt.imshow(x_train[0].reshape(28, 28), cmap="gray")
# plt.title(f"Label: {np.argmax(targets[0])}")
# plt.show()

# ==============================
# 6. INITIALIZE NEURAL NETWORK
# ==============================
rn = CBNN(
    x_train=x_train,
    targets=targets,
    n_iter=10,
    n_hidden=20,
    lr=0.01,
    n_batches=10
)

# ==============================
# 7. TRAIN NETWORK
# ==============================
print("\nTraining network...")
rn.training(batch_size=32)

# ==============================
# 8. PREDICT
# ==============================
print("\nPredictions:")
prediction = rn.predict(test_inputs)

print("\nTarget\tPrediction")
for i in range(20):
    true_label = test_show[i]
    pred_label = np.argmax(prediction[i])
    print(f"{true_label}\t{pred_label}")
