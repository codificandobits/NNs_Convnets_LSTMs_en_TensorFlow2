# TensorFlow/Keras
import tensorflow as tf
tf.random.set_seed(9)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Activation

# Librerías auxiliares
import numpy as np

# Funciones auxiliares

def generar_texto(modelo, longitud, tam_vocab, ix_to_char):
    # Escoger un caracter aleatorio
    ix = [np.random.randint(tam_vocab)]
    y_char = [ix_to_char[ix[-1]]]

    # Crear texto
    X = np.zeros((1, longitud, tam_vocab))
    for i in range(longitud):
        # Convertir el caracter a formato "one-hot" y
        # predecir el siguiente caracter
        X[0, i, :][ix[-1]] = 1
        prediccion = modelo.predict(X[:, :i+1, :])
        ix = np.argmax(prediccion[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)

def cargar_datos(data_path, long_sec):

    # Leer datos, determinar número total de caracteres y tamaño del diccionario
    datos = open(data_path,'r',encoding='utf8').read()
    chars = list(set(datos))
    tam_vocab = len(chars)

    print('Cantidad total de caracteres: {}'.format(len(datos)))
    print('Tamaño del diccionario: {} caracteres'.format(tam_vocab))

    # Equivalencia índice-caracteres y caracteres-índices
    ix_to_char = {ix:char for ix, char in enumerate(chars)}
    char_to_ix = {char:ix for ix, char in enumerate(chars)}

    # Set de entrenamiento (X,y)
    # X: len(datos)/long_sec, long_sec, tam_vocab
    # y: el mismo tamaño de X (pues la LSTM convierte de secuencia a secuencia)
    X = np.zeros((int(len(datos)/long_sec), long_sec, tam_vocab))
    y = np.zeros(X.shape)

    for i in range(0,X.shape[0]):
        # Para cada ejemplo de entrenamiento:
        
        # 1. Extraer bloque de datos de tamaño long_sec
        X_sequence = datos[i*long_sec:(i+1)*long_sec]

        # 2. Convertir cada caracter a representación numérica
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]

        # 3. Convertir cada caracter (representado numéricamente) al formato
        # one-hot
        sec_in = np.zeros((long_sec, tam_vocab))
        for j in range(long_sec):
            sec_in[j][X_sequence_ix[j]] = 1.
            X[i] = sec_in

        # 4. Hacer algo similar para "y", pero desplazando la secuencia una posición
        y_sequence = datos[i*long_sec+1:(i+1)*long_sec+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((long_sec, tam_vocab))
        for j in range(long_sec):
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[i] = target_sequence
    return X, y, tam_vocab, ix_to_char, char_to_ix

#
# 0. LECTURA DE DATOS Y PRE-PROCESAMIENTO
#

DATA_PATH = './donquijote.txt'
LONG_SEC = 100           # Número de caracteres en cada ejemplo

# Lectura de datos y creación del set de entrenamiento
x_train, y_train, TAM_VOCAB, ix_to_char, char_to_ix = cargar_datos(DATA_PATH, LONG_SEC)

#
# 1. MODELO: red LSTM con 1024 neuronas
#

modelo = Sequential()
modelo.add(LSTM(1024, input_shape=(None, 92), return_sequences=True))
modelo.add(TimeDistributed(Dense(92)))
modelo.add(Activation('softmax'))

#
# 2/3. COMPILACIÓN Y ENTRENAMIENTO DEL MODELO
#
modelo.compile(loss="categorical_crossentropy", optimizer="adam")
modelo.fit(x_train,y_train, batch_size=64, epochs=20, verbose=1)

#
# 4. PREDICCIÓN
#
print(generar_texto(modelo, 500, 92, ix_to_char))


# Secuencia de 500 caracteres generada
# ver lo que de mi padre le había de ser con la muerte de la
# muerte.

# -Así es la verdad -respondió Sancho-, pero no me acuerdo delante de la
# muerte de mi alma en la mano, y el más desatino que te ha de ser con la muerte de
# la muerte, y que la primera se me ha de comer y de mi padre y de la mano
# de mi alma de su casa, y el mismo del mundo es manera que el cura le había
# de ser con la muerte de la mano, y la primera parte del castillo, y el
# cura de la mano estaba en la cabeza y dijo:

# -Por cierto, señ
 


