from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def optimize_model(X_train, y_train):
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }

    grid = GridSearchCV(model, param_grid, scoring="accuracy", cv=3, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("✅ Najlepsze parametry:")
    print(grid.best_params_)
    print("📈 Najlepszy wynik:", grid.best_score_)

    return grid.best_estimator_