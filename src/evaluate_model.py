import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    df_y = pd.read_csv("data/eval/y.csv")
    y_test = df_y["y_test"]
    y_pred = df_y["y_pred"]

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("📉 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    #plt.show()
    plt.savefig("data/eval/confusion_matrix.png")

def plot_feature_importance(model, feature_names, top_n=20):
    importance = model.get_booster().get_score(importance_type="gain")
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values(by="Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1])
    plt.title("Top Feature Importances")
    plt.xlabel("Gain")
    plt.tight_layout()
    #plt.show()
    plt.savefig("data/eval/feature_importance.png")