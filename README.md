# proyectoParaElReconociminetoDeSe-asUtilizandoPytorch
Proyecto desarrollando en python utilizando pytorch, MediaPipe y OpenCV para la deteccion y clasificación de señas A y B del lenguaje de señas mexicano con una red neuronal convolucional.

El correr el proyecto es necesario tener una camara web, asi como pytorch, MediaPipe y OpenCV.

En la carpeta Validacion se encuentran las imagenes por seña para el entrenamiento de la red neuronal
El archivo Entrenamiento.py contiene el codigo para ejecutar el entrenamiento de la red neuronal
El archivo modulo1.py realiza la clasificacion en vivo de la seña realizada mediante la camara web e indica la letra de la seña correspondiente.
El archivo modelo.pt es el modelo de la red neuronal ya entrenada con las imagenes de la carpeta
