# TensorFlow/Keras
import tensorflow as tf
tf.random.set_seed(200)
from tensorflow.keras import datasets, Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

# Librerías auxiliares
import matplotlib.pyplot as plt
import numpy as np

# Funciones auxiliares
def graficar_imagen(i, predicciones, y_test, x_test):
    cat_real = y_test[i][0]
    img = x_test[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    cat_predicha = np.argmax(predicciones[i])
    if cat_predicha == cat_real:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("Pred.: {} (probabilidad: {:2.0f}%) (Cat. real: {})".format(categorias[cat_predicha],
                                100*np.max(predicciones[i]),
                                categorias[cat_real]),
                                color=color)

#
# 0. PRE-PROCESAMIENTO
#

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

categorias = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
               'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# Normalización de los niveles de intensidad de cada pixel (pasando de 0-255 a 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Mostrar 16 imágenes del set con sus correspondientes categorías
plt.figure(figsize=(7,7))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(categorias[y_train[i][0]])
plt.show()

#
# 1. MODELO: red convolucional Conv2D (32, 3x3) - MaxPooling (2x2) -
# Conv2D (64, 3x3) - MaxPooling (2x2) - Conv2D (64, 3x3) - MaxPooling (2x2) -
# Fully Connected (64 neuronas) - Softmax (10 categorías)
# 

modelo = Sequential()
modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
modelo.add(MaxPooling2D((2, 2)))
modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(MaxPooling2D((2, 2)))
modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(Flatten())
modelo.add(Dense(64, activation='relu'))
modelo.add(Dense(10, activation='softmax'))

#
# 2/3. COMPILACIÓN Y ENTRENAMIENTO DEL MODELO
#

# Nota: como métrica de error se usará "sparse_categorical_crossentropy" (en 
# lugar de la usual "categorical_crossentropy"). Esta métrica permite evitar
# la conversión de y_train y y_test al formato one-hot
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
	metrics=['accuracy'])
modelo.fit(x_train, y_train, epochs=10, verbose=1)

#
# 4/5. EVALUACIÓN Y PREDICCIÓN
#
error, precision = modelo.evaluate(x_test, y_test, verbose=2)
print('\nPrecisión con el set de validación:', precision)

predicciones = modelo.predict(x_test)

imagenes = [0, 2, 4]

plt.figure(figsize=(7,7))
for i,j in enumerate(imagenes):
    plt.subplot(3,1,i+1)
    graficar_imagen(j,predicciones,y_test,x_test)
plt.tight_layout()
plt.show()
