import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, MaxPooling1D
from tensorflow.keras.utils import plot_model  
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score

"""
Búsqueda de hiperparámetros para un modelo CNN-LSTM en EEG,
mediante Keras Tuner.
"""
# IMPORTACIÓN DE DATOS
data_path = '/data_CHB_mit/signal_samples.npy'
data_path_labels = '/data_CHB_mit/is_sz.npy'

# Cargar señales y etiquetas
X = np.load(data_path)  # (n, 18, 2048)
y = np.load(data_path_labels).astype(int)   # (n,), booleano -> entero


#VISUALIZACIÓN DE DATOS
print("Forma del array de señales:", X.shape)
print("Forma del array de etiquetas:", y.shape)
print(y.sum(), "segmentos con crisis y", len(y) - y.sum(), "sin crisis")

# Visualización de la distribución
plt.figure(figsize=(5, 4))
plt.hist(y, bins=[-0.5, 0.5, 1.5], rwidth=0.6)
plt.xticks([0, 1], ['Sin crisis', 'Con crisis'])
plt.title('Distribución de segmentos con y sin crisis')
plt.ylabel('Número de segmentos')
plt.show()


ch_labels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3','P3-O1',
             'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
             'FZ-CZ', 'CZ-PZ']
# Visualización de un segmento, indicando si es crisis
i = 476  # Índice del segmento a visualizar
segmento = X[i]
es_crisis = y[i]
desfase = 400  # Desfase para separar los canales en el gráfico
plt.figure(figsize=(14, 8))
for canal in range(18):
    plt.plot(segmento[canal] + canal*desfase, label=ch_labels[canal])

# Mostrar etiquetas de canal en el eje y:
plt.yticks(
    ticks=[canal*desfase for canal in range(18)],
    labels=ch_labels
)
plt.title(f'Segmento {i} - {"Con crisis" if es_crisis else "Sin crisis"}')
plt.xlabel('Tiempo (muestras)')
plt.ylabel('Canales EEG')
plt.tight_layout()
plt.show()

# PREPROCESAMIENTO DE DATOS
# Normalización Z-score por canal y por segmento
mean = X.mean(axis=2, keepdims=True)   # (n, 18, 1)
std = X.std(axis=2, keepdims=True)     # (n, 18, 1)
X_norm = (X - mean) / (std + 1e-8)

# Reorganiza: (n_muestras, longitud_segmento, canales)
X_norm = np.transpose(X_norm, (0, 2, 1))
print(f"Shape reorganizado para CNN-LSTM: {X.shape}")  # (n, 2048, 18)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# CONSTRUCCIÓN DEL MODELO CNN-LSTM
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    # Capas convolucionales 1D
    model.add(Conv1D(
        filters=hp.Choice('conv1_filters', [16, 32, 64]),
        kernel_size=hp.Choice('conv1_kernel', [3, 5, 7]),
        activation='relu',
        input_shape=(2048, 18)
    ))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))

    model.add(Conv1D(
        filters=hp.Choice('conv2_filters', [32, 64, 128]),
        kernel_size=hp.Choice('conv2_kernel', [3, 5]),
        activation='relu'
    ))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))

    # Capa LSTM
    model.add(LSTM(
        hp.Choice('lstm_units', [32, 64, 128]),
        return_sequences=False
    ))
    model.add(Dropout(hp.Float('dropout3', 0.1, 0.5, step=0.1)))

    # Capas densas
    model.add(Dense(
        hp.Choice('dense_units', [32, 64, 128]),
        activation='relu'
    ))
    model.add(Dropout(hp.Float('dropout4', 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    # Compilación
    model.compile(
        optimizer=Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='ktuner_dir',
    project_name='cnn_lstm_eeg'
)

early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

tuner.search(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early, reduce_lr]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Mejores hiperparámetros encontrados:")
for key in best_hps.values.keys():
    print(f"{key}: {best_hps.get(key)}")

model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early, reduce_lr]
)

