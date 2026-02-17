## Informe de Reconocimiento de Dígitos Manuscritos (MNIST)

### Descripción General

En este segundo punto se implementó una red neuronal completamente conectada, diseñada desde cero en Python, para reconocer dígitos escritos a mano utilizando el conjunto de datos **MNIST**. La red cuenta con una arquitectura de tres capas: entrada, oculta y salida.

### Estructura de la Red

- **Capa de entrada**: 784 neuronas (28x28 píxeles).
- **Capa oculta**: 20 neuronas, activación ReLU.
- **Capa de salida**: 10 neuronas (una por cada dígito del 0 al 9), activación softmax.

### Inicialización de Pesos

Los pesos de las capas se inicializan según la distribución:

$$
w \sim \mathcal{U}(-0.5, 0.5)
$$

No obstante, se implementó internamente la **inicialización de He** para mejorar la estabilidad del entrenamiento:

$$
w \sim \mathcal{N}(0, \sqrt{\frac{2}{n}})
$$

### Funciones de Activación

- **ReLU** en la capa oculta: favorece la propagación de gradientes.
- **Softmax** en la capa de salida: convierte los valores en probabilidades para clasificación multiclase.

### Función de Pérdida

Se emplea la pérdida de **log loss** (entropía cruzada multiclase):

$$
\mathcal{L}(y, p) = - y \log(p) - (1 - y) \log(1 - p)
$$

### Propagación hacia Adelante

1. $$ Z1 = X \cdot W1 + b1 $$
2. $$ A1 = \text{ReLU}(Z1) $$ 
3. $$ Z2 = A1 \cdot W2 + b2 $$
4. $$ A2 = \text{Softmax}(Z2) $$

### Retropropagación y Actualización de Pesos

- Se calcula el error de la salida y se propaga hacia atrás.
- Se actualizan los pesos usando el gradiente descendente con batches.

### Entrenamiento

- Se entrenó por 100 épocas con un **batch size de 32** y **tasa de aprendizaje 0.01**.
- Se imprimió la pérdida promedio por época.

### Resultados

El modelo fue capaz de aprender a reconocer dígitos manuscritos, realizando predicciones correctas en la mayoría de los casos sobre el conjunto de prueba de MNIST.

Se imprimieron las predicciones para una muestra de imágenes de test, comparando el valor real con la predicción del modelo.

### Prediciones

A continuación se muestra las predicciones de la red neuronal despues del entrenamiento con 20 datos del set de testeo, en el archivo punto2.py toma las prediciones que son una una matriz con 10 columnas donde cada columna determina un numero segun la posicion, y se toma el numero de la columna la cual tiene la mayor probabilidad entre todas las columnas.

| Target | Predicción |
|--------|------------|
| 7      | 7          |
| 2      | 2          |
| 1      | 1          |
| 0      | 0          |
| 4      | 4          |
| 1      | 1          |
| 4      | 4          |
| 9      | 9          |
| 5      | 5          |
| 9      | 9          |
| 0      | 0          |
| 6      | 6          |
| 9      | 9          |
| 0      | 0          |
| 1      | 1          |
| 5      | 5          |
| 9      | 9          |
| 7      | 7          |
| 3      | 3          |
| 4      | 4          |

### Conclusión

La red neuronal cumple con los requisitos del enunciado:
- Usa 784 neuronas de entrada, 20 ocultas y 10 de salida.
- Utiliza funciones de activación ReLU y Softmax.
- Entrenada con 32 batch training y log loss.
- Reconoce correctamente dígitos manuscritos del dataset MNIST.

---

### Archivos Implementados

#### `RN.py`

Contiene la clase `CBNN`, donde se define la estructura de la red, funciones de activación, propagación hacia adelante, retropropagación y actualización de pesos.

#### `punto2.py`

Carga el conjunto de datos MNIST, normaliza las imágenes, define el modelo, entrena y realiza predicciones.

---