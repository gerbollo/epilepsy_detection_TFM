import numpy as np

def load_data(data_path, labels_path):
    """
    Carga las señales EEG y sus etiquetas desde archivos .npy.
    Args:
        data_path (str): Ruta al archivo de señales.
        labels_path (str): Ruta al archivo de etiquetas.
    Returns:
        X (np.ndarray): Array de señales EEG.
        y (np.ndarray): Array de etiquetas.
    """
    X = np.load(data_path)
    y = np.load(labels_path).astype(int)
    return X, y

