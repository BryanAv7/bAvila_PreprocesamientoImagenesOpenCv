# Unidad 4 – Parte C: Preprocesamiento de Imágenes con OpenCV usando CPU y GPU

## 📌 Descripción general

Este parte del trabajo se implementa un sistema de preprocesamiento de imágenes utilizando la biblioteca **OpenCV**, comparando el rendimiento entre procesamiento en **CPU** y **GPU (CUDA)**. Se aplica una serie de operaciones sobre un conjunto de imágenes para evaluar el tiempo total de ejecución en ambas arquitecturas.

---

## ⚙️ ¿Qué operaciones realiza?

Sobre cada imagen se ejecuta el siguiente pipeline:

- Conversión a escala de grises
- Suavizado con filtro Gaussiano
- Erosión y dilatación (morfología)
- Detección de bordes con Canny
- Ecualización del histograma

Este flujo se implementa en dos versiones:
- `procesoCPU`: versión tradicional usando la CPU
- `procesoGPU`: versión acelerada usando la GPU con CUDA

---

## 🧪 Objetivo

Comparar el rendimiento del procesamiento de imágenes entre CPU y GPU mediante la medición de tiempo total y calcular el **"speedup"** (aceleración lograda con GPU).

---

## 📁 Estructura del proyecto

├── Principal.cpp # Código fuente principal (CPU y GPU)
├── imagenes/ # Carpeta con las imágenes a procesar (.jpg)
├── resultados/
│ ├── cpu/ # Resultados procesados con CPU
│ └── gpu/ # Resultados procesados con GPU
├── Makefile # Para compilar automáticamente
└── vision.bin # Binario generado (si se compiló)

---

## 🛠 Requisitos

- Linux (Ubuntu recomendado)
- OpenCV con soporte para CUDA (`libopencv-dev`, `opencv-contrib`, etc.)
- Compilador C++17 o superior
- CMake (opcional, si deseas usarlo en lugar de Makefile)

---

## 🧰 Compilación y ejecución

### Usando Makefile:

```bash
make
./vision.bin
