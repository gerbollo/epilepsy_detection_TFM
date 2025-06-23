from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos en train/test.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Entrena el modelo con EarlyStopping y ReduceLROnPlateau.
    """
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=3, factor=0.3, min_lr=1e-6, monitor='val_loss')
    ]
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    return history

def predict_and_evaluate(model, X_test, y_test):
    """
    Obtiene predicciones y probabilidades.
    """
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    return y_pred, y_pred_proba
