from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import pandas as pd
import os
import numpy as np

def optimize_model(X_train, y_train):
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)

    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3, 0.5],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0]
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="accuracy",
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print("✅ Najlepsze parametry:")
    print(random_search.best_params_)
    print("📈 Najlepszy wynik:", random_search.best_score_)

    return random_search.best_estimator_