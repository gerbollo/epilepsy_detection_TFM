# TFM: Detección de crisis epilépticas en EEG con deep learning

Repositorio del Trabajo Fin de Máster para la detección automática de crisis epilépticas en señales EEG mediante redes neuronales profundas (CNN-LSTM).

## Estructura

- `src/`: módulos Python para carga de datos, preprocesado, modelos, entrenamiento y visualización.
- `main.py`: pipeline principal para ejecutar el flujo completo.
- `keras_tuner`: realiza la búsqueda de hiperparámetros. Con los resultados obtenidos se creó el modelo en model.py.
- `data/`: (no incluido) se trata de dos archivos .npy obtenidos de https://www.kaggle.com/datasets/masahirogotoh/mit-chb-processed/data. signal_samples.npy contiene los segmentos de señal y is_sz.npy contiene las etiquetas de los segmentos (si es crisis o no).
- `requirements.txt`: dependencias del proyecto.

## Uso

1. Clona este repositorio:
2. Instala las dependencias:
3. Descarga y coloca los archivos `.npy` en la carpeta `data/`.
4. Ejecuta `main.py` para reproducir el flujo completo de análisis y entrenamiento.


