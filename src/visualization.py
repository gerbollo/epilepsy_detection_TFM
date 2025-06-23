import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve, auc

def plot_label_distribution(y):
    """
    Visualiza la distribución de clases (sin crisis / con crisis).
    """
    plt.figure(figsize=(5, 4))
    plt.hist(y, bins=[-0.5, 0.5, 1.5], rwidth=0.6)
    plt.xticks([0, 1], ['Sin crisis', 'Con crisis'])
    plt.title('Distribución de segmentos con y sin crisis')
    plt.ylabel('Número de segmentos')
    plt.show()

def plot_eeg_segment(segment, label, ch_labels, idx, desfase=400):
    """
    Visualiza un segmento multicanal EEG.
    """
    plt.figure(figsize=(14, 8))
    for canal in range(segment.shape[0]):
        plt.plot(segment[canal] + canal * desfase, label=ch_labels[canal])
    plt.yticks([canal*desfase for canal in range(segment.shape[0])], ch_labels)
    plt.title(f'Segmento {idx} - {"Con crisis" if label else "Sin crisis"}')
    plt.xlabel('Tiempo (muestras)')
    plt.ylabel('Canales EEG')
    plt.tight_layout()
    plt.show()

def plot_training_curves(history):
    """
    Grafica precisión y pérdida de entrenamiento/validación.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.title('Precisión durante el entrenamiento')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Pérdida durante el entrenamiento')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_and_report(y_true, y_pred, y_pred_proba, target_names=['Sin crisis', 'Con crisis']):
    """
    Dibuja la matriz de confusión y muestra el reporte de métricas.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión")
    plt.show()
    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("AUC ROC:", roc_auc_score(y_true, y_pred_proba))

def plot_roc_curve(y_true, y_score):
    """
    Dibuja la curva ROC y calcula el AUC.
    Args:
        y_true: etiquetas reales (0 o 1)
        y_score: probabilidades predichas (salida del modelo)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()   
