from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from RN import CBNN
import numpy as np
import matplotlib.pyplot as plt


# Carga datos: ya vienen divididos en entrenamiento y prueba
(x_train, targets), (test_inputs, test_targets) = mnist.load_data()

# Datos para el entrenamiento de la red Neuronal
x_train = x_train/255
x_train = x_train.reshape(-1, 784)

test_show = test_targets
targets = to_categorical(targets, num_classes=10)

#Datos para prueba
test_inputs = test_inputs.reshape(-1, 784)
test_inputs = test_inputs/255

test_targets = to_categorical(test_targets, num_classes=10)

# Inicializacion de la red Neuronal
rn = CBNN(x_train = x_train, targets= targets, n_iter = 100, n_hidden = 20, lr = 0.01)

# Entrenamiento de la red Neuronal
rn.training(batch_size= 32)

# Mostrar las primeras 10 imagenes del set de imagenes de mnist
# for i in range(10):
#     plt.imshow(test_inputs[i], cmap='gray')
#     plt.title(f"Etiqueta: {test_targets[i]}")
#     plt.axis('off')
#     plt.show()

# Realiza la predicion de los datos de testeo
print('Predictions:')
prediction = rn.predict(np.array(test_inputs))

print(f'targets        prediction')
for i in range(20):
    indice = np.argmax(prediction[i])
    print(f'{test_show[i]}:    {indice}') 