# TensorFlow/Keras
import tensorflow as tf
tf.random.set_seed(4)
from tensorflow.keras import datasets, Sequential
from tensorflow.keras.layers import Flatten, Dense

# Librerías auxiliares
import numpy as np
import matplotlib.pyplot as plt

# Funciones auxiliares
def graficar_imagen(i, predicciones, y_test, x_test):
    cat_real = np.argmax(y_test[i])
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
dataset = datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

categorias = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Tenis', 'Bolso', 'Bota']

x_train.shape           # 60,000 x 28 x 28 (60,000 imágenes, cada una de 28x28)
len(y_train)            # Arreglo con categorías de 0 a 9 (10 categorías en total)

x_test.shape            # 10,000 x 28 x 28 (10,000 imágenes, cada una de 28x28)
len(y_test)             # Arreglo con categorías de 0 a 9 (10 categorías en total)

# Normalización de los niveles de intensidad de cada pixel (pasando de 0-255 a 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Representación one-hot de y_train y y_test
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

# Mostrar 16 imágenes del set con sus correspondientes categorías
plt.figure(figsize=(7,7))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(categorias[np.argmax(y_train[i])])
plt.show()

#
# 1. MODELO: red neuronal con una capa oculta de 18 neuronas y salida tipo softmax con 10 neuronas
#

modelo = Sequential()
modelo.add( Flatten(input_shape=(28,28)) )
modelo.add( Dense(128, activation = 'relu') )
modelo.add( Dense(10, activation = 'softmax') )

#
# 2/3. COMPILACIÓN Y ENTRENAMIENTO DEL MODELO
#
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.fit(x_train, y_train, epochs=10, verbose=1)

#
# 4/5. EVALUACIÓN Y PREDICCIÓN
#
error, precision = modelo.evaluate(x_test, y_test, verbose=2)
print('\nPrecisión con el set de validación:', precision)

predicciones = modelo.predict(x_test)

imagenes = [0, 11, 12]

plt.figure(figsize=(7,7))
for i,j in enumerate(imagenes):
    plt.subplot(3,1,i+1)
    graficar_imagen(j,predicciones,y_test,x_test)
plt.tight_layout()
plt.show()

 





