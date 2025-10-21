Experimento Avanzado: Redes Neuronales y Rendimiento de GPU

¡Has subido de nivel! Bienvenido al mundo del Deep Learning.

Este segundo experimento está diseñado para una prueba de rendimiento real. Dejamos atrás el Machine Learning clásico y nos adentramos en las Redes Neuronales Convolucionales (CNNs), un tipo de modelo que requiere una capacidad de cómputo mucho mayor.

¿Qué hace esta nueva aplicación?

El script fashion_mnist_cnn.py entrena un modelo de Deep Learning para identificar qué tipo de prenda de vestir aparece en una imagen. Utiliza el dataset Fashion-MNIST, que es mucho más grande y complejo que el de Iris.

Aquí es donde la diferencia entre una CPU y una GPU se vuelve drástica.

¿Cómo empezar?

Los pasos son similares, pero la instalación de las librerías es CRUCIALMENTE DIFERENTE para cada sistema si quieres aprovechar el hardware al máximo.

1. Prepara tu entorno

Si ya tienes un entorno virtual del experimento anterior, puedes usarlo. Si no, crea uno nuevo.

# Navega a la carpeta del proyecto
# cd ruta/a/tu/proyecto

# Crea el entorno virtual (si no lo tienes)
python -m venv mi_entorno_ml

# Actívalo
# En macOS/Linux:
source mi_entorno_ml/bin/activate
# En Windows:
mi_entorno_ml\\Scripts\\activate


2. Instala las librerías (¡Paso Clave!)

Necesitamos instalar TensorFlow, la librería de Google para Deep Learning. La forma de instalarla depende de tu sistema operativo y hardware.

En tu PC con RTX 3060 (Windows/Linux):

Para que TensorFlow use tu GPU NVIDIA, necesitas tener los drivers de NVIDIA y el toolkit de CUDA instalados. La forma más sencilla de instalar TensorFlow y sus dependencias de CUDA es usando el siguiente comando. Esto asegura que obtengas la versión compatible con GPU.

# Este comando instala TensorFlow y los componentes necesarios para la GPU
pip install tensorflow[and-cuda]


Si tienes problemas, consulta la guía oficial de instalación de TensorFlow para GPU.

En tu Mac (con procesador Apple Silicon M1/M2/M3):

Apple mantiene su propia versión de TensorFlow que está optimizada para usar los aceleradores de Metal (el equivalente a CUDA para Apple).

# Instala la versión de TensorFlow optimizada por Apple
pip install tensorflow-macos


Actualiza tu requirements.txt

Tu nuevo archivo requirements.txt debería tener este contenido (aunque lo instalarás manualmente con los comandos de arriba, es bueno tenerlo de referencia).

# Usa 'pip install tensorflow[and-cuda]' en PC con NVIDIA
# Usa 'pip install tensorflow-macos' en Mac M1/M2/M3
tensorflow
numpy
matplotlib


No es necesario que pongas seaborn esta vez.

3. Ejecuta el experimento

Una vez instalado todo, corre el nuevo script:

python fashion_mnist_cnn.py


El script primero te dirá si ha detectado una GPU. Luego, comenzará el entrenamiento. Este proceso tardará varios minutos.

4. Compara los resultados

Al final, obtendrás el tiempo total de entrenamiento:

--- Entrenamiento finalizado en XXX.XX segundos ---


Anota el tiempo de ambos ordenadores.

¿Qué esperas encontrar? Aquí la diferencia será enorme. El entrenamiento en tu PC con la RTX 3060 debería ser varias veces más rápido que en la CPU de tu Mac (incluso si tu Mac usa su acelerador Metal, la RTX 3060 es una GPU dedicada y muy potente para estas tareas).

¿Por qué? Las Redes Neuronales realizan miles de operaciones matemáticas en paralelo (cálculos de matrices). Las GPUs están diseñadas específicamente para este tipo de paralelismo masivo, mientras que las CPUs están diseñadas para tareas secuenciales.

¡Felicidades! Ahora estás realizando un benchmark de hardware real y relevante para el campo de la inteligencia artificial.