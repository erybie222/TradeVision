import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def train_model(processed_path: str, model_path: str):
    dfs = []

    for filename in os.listdir(processed_path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(processed_path, filename))
            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    X = df_all.drop(columns=["Date", "Target"])
    y = df_all["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, model_path)
    print("✅ Model zapisany do:", model_path)