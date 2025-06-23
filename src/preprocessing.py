import numpy as np

def zscore_normalization(X):
    """
    Normalización Z-score por canal y segmento.
    """
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    X_norm = (X - mean) / (std + 1e-8)
    return X_norm

def reshape_for_model(X):
    """
    Cambia de (n, canales, muestras) a (n, muestras, canales).
    """
    return np.transpose(X, (0, 2, 1))
