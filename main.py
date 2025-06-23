from src.data_loading import load_data
from src.visualization import (
    plot_label_distribution, plot_eeg_segment, plot_training_curves, plot_confusion_matrix_and_report, plot_roc_curve
)
from src.preprocessing import zscore_normalization, reshape_for_model
from src.model import build_cnn_lstm
from src.train import split_data, train_model, predict_and_evaluate
from tensorflow.keras.utils import plot_model

# Rutas de tus datos
data_path = '/Users/germanencinas/Documents/Master UAX/TFM/codigo_modular/data/signal_samples.npy'
labels_path = '/Users/germanencinas/Documents/Master UAX/TFM/codigo_modular/data/is_sz.npy'

# Carga de datos
X, y = load_data(data_path, labels_path)
print("Forma de señales:", X.shape, "Forma de etiquetas:", y.shape)

# Visualización de la distribución
plot_label_distribution(y)

# Visualización de un segmento concreto
ch_labels = [ 'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3','P3-O1',
             'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
             'FZ-CZ', 'CZ-PZ'] # lista de tus etiquetas de canal

plot_eeg_segment(X[476], y[476], ch_labels, idx=476)

# Preprocesado
X_norm = zscore_normalization(X)
X_ready = reshape_for_model(X_norm)
print("Shape listo para modelo:", X_ready.shape)

# Split train/test
X_train, X_test, y_train, y_test = split_data(X_ready, y)

# Definir modelo
input_shape = X_train.shape[1:]
model = build_cnn_lstm(input_shape)
model.summary()
# Visualizar modelo
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# 6. Entrenamiento

history = train_model(model, X_train, y_train, X_test, y_test)

# 7. Curvas de entrenamiento
plot_training_curves(history)

# 8. Evaluación
y_pred, y_pred_proba = predict_and_evaluate(model, X_test, y_test)
plot_confusion_matrix_and_report(y_test, y_pred, y_pred_proba)
plot_roc_curve(y_test, y_pred_proba)

