# -*- coding: utf-8 -*-
"""
Clasificador de Ropa (Fashion-MNIST) con una Red Neuronal Convolucional (CNN)

Este script entrena un modelo de Deep Learning para clasificar imágenes de ropa.
Es un problema significativamente más complejo que el de Iris y es ideal para
probar el rendimiento de CPU vs. GPU.
"""

# --- 1. Importar las librerías necesarias ---
# TensorFlow es la librería principal para Deep Learning.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

# Numpy para operaciones numéricas.
import numpy as np

# Time para medir el rendimiento.
import time

# Matplotlib para visualizar las imágenes.
import matplotlib.pyplot as plt

# --- 2. Cargar y explorar el conjunto de datos ---
def cargar_y_preparar_datos():
    """Carga el dataset Fashion-MNIST y lo preprocesa para la red neuronal."""
    print("Cargando el conjunto de datos Fashion-MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    print(f"Forma de los datos de entrenamiento: {x_train.shape}")
    print(f"Cantidad de etiquetas de entrenamiento: {y_train.shape[0]}")

    # Nombres de las clases de ropa.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # --- Preprocesamiento ---
    # Normalizar los píxeles: Los valores de los píxeles van de 0 a 255.
    # Los normalizamos a un rango de 0 a 1 para ayudar al entrenamiento.
    print("\nNormalizando los datos...")
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Añadir una dimensión de "canal". Las CNNs esperan esta dimensión.
    # Como son imágenes en blanco y negro, el canal es 1.
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"Nueva forma de los datos de entrenamiento: {x_train.shape}")

    return (x_train, y_train), (x_test, y_test), class_names

# --- 3. Construir el modelo de Red Neuronal Convolucional (CNN) ---
def construir_modelo(input_shape):
    """Define la arquitectura de la CNN."""
    print("\nConstruyendo el modelo de Red Neuronal Convolucional...")
    model = Sequential([
        # Primera capa convolucional: Extrae 32 patrones de las imágenes.
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # Capa de Pooling: Reduce el tamaño de la imagen para enfocarse en lo importante.
        MaxPooling2D((2, 2)),
        
        # Segunda capa convolucional: Aprende patrones más complejos.
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Aplanar los datos: Convierte la matriz 2D de la imagen en un vector 1D.
        Flatten(),
        
        # Capa Densa: Una capa neuronal clásica.
        Dense(128, activation='relu'),
        # Capa de Dropout: Ayuda a prevenir el sobreajuste (overfitting).
        Dropout(0.5),
        # Capa de Salida: 10 neuronas (una para cada clase de ropa) con activación softmax.
        Dense(10, activation='softmax')
    ])
    
    # Compilamos el modelo, definiendo cómo aprenderá.
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary() # Imprime un resumen de la arquitectura.
    return model

# --- 4. Entrenar el modelo ---
def entrenar_modelo(model, x_train, y_train, x_test, y_test):
    """Entrena el modelo y mide el tiempo que toma."""
    print("\n--- Iniciando el entrenamiento del modelo (esto puede tardar varios minutos) ---")
    
    start_time = time.time()
    
    # El entrenamiento se realiza en "épocas" (epochs).
    # Una época es una pasada completa por todo el conjunto de datos de entrenamiento.
    # validation_data se usa para evaluar el modelo al final de cada época.
    history = model.fit(x_train, y_train, epochs=10, 
                        validation_data=(x_test, y_test),
                        batch_size=64) # Procesamos los datos en lotes de 64 imágenes.

    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n--- Entrenamiento finalizado en {training_time:.2f} segundos ---")
    
    return history, training_time

# --- 5. Evaluar el modelo ---
def evaluar_y_visualizar(model, history, x_test, y_test):
    """Evalúa la precisión final y grafica los resultados del entrenamiento."""
    print("\nEvaluando la precisión final del modelo...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nPrecisión en el conjunto de prueba: {test_acc * 100:.2f}%')

    # Graficar la precisión y la pérdida durante el entrenamiento
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión (entrenamiento)')
    plt.plot(history.history['val_accuracy'], label='Precisión (validación)')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.title('Precisión del Modelo')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida (entrenamiento)')
    plt.plot(history.history['val_loss'], label='Pérdida (validación)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida del Modelo')
    
    plt.savefig("training_results.png")
    print("\nGráficos del entrenamiento guardados en 'training_results.png'")
    plt.close()


# --- Función principal ---
def main():
    print("==========================================================")
    print("   Experimento Avanzado: Clasificación de Ropa con CNN    ")
    print("==========================================================")
    
    # Verificamos si TensorFlow puede acceder a la GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n¡GPU detectada! TensorFlow utilizará: {gpus[0].name}")
    else:
        print("\nNo se detectó GPU. TensorFlow se ejecutará en la CPU.")

    (x_train, y_train), (x_test, y_test), class_names = cargar_y_preparar_datos()
    model = construir_modelo(x_train.shape[1:])
    history, tiempo = entrenar_modelo(model, x_train, y_train, x_test, y_test)
    evaluar_y_visualizar(model, history, x_test, y_test)

    print("\nExperimento completado.")
    print("Compara el tiempo de entrenamiento entre tu Mac y tu PC con RTX 3060.")
    print("¡Deberías ver una diferencia significativa!")

if __name__ == "__main__":
    main()
