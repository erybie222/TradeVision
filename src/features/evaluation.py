import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Rysuje macierz pomyłek (confusion matrix).
    
    :param y_true: lista lub array z prawdziwymi etykietami
    :param y_pred: lista lub array z przewidywanymi etykietami
    :param labels: opcjonalnie – lista etykiet klas (np. [0, 1])
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywista')
    plt.title('Macierz pomyłek')
    plt.tight_layout()
    plt.show()