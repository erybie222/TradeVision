import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from src.optimize_model import optimize_model

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

    model = optimize_model(X_train, y_train)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print("✅ Model zapisany do:", model_path)

    X_test.to_csv("data/eval/X_test.csv", index=False)
    pd.DataFrame({"y_test": y_test, "y_pred": model.predict(X_test)}).to_csv("data/eval/y.csv", index=False)

    return model, X