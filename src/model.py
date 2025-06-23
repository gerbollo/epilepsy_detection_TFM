from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, LSTM, Dense, BatchNormalization, MaxPooling1D
from tensorflow.keras.optimizers import Adam


def build_cnn_lstm(input_shape):
    """
    Construye y compila el modelo CNN-LSTM.
    """
    model = Sequential([
        Conv1D(64, 5, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(2),
        BatchNormalization(),
        Dropout(0.2),
        
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(128, return_sequences=False),
        Dropout(0.4),

        Dense(32, activation='relu'),
        Dropout(0.1),

        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0013), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )   
    return model
