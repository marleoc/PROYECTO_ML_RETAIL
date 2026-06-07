import numpy as np
import optuna

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier


# =====================================================
# 📊 REGRESIÓN (DEMANDA)
# =====================================================

def train_regression(X, y, n_trials=20):
    """
    Entrena modelo de regresión con Optuna + validación temporal.
    Nivel enterprise: estable, reproducible, sin leakage.
    """

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):

        params = {
            "max_iter": trial.suggest_int("max_iter", 150, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0)
        }

        fold_maes = []

        for train_idx, val_idx in tscv.split(X):

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = HistGradientBoostingRegressor(
                **params,
                random_state=42
            )

            model.fit(X_train, y_train)
            pred = model.predict(X_val)

            mae = mean_absolute_error(y_val, pred)
            fold_maes.append(mae)

        return np.mean(fold_maes)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("\n📊 Mejor MAE CV:", study.best_value)
    print("📌 Mejores parámetros:", study.best_params)

    # =====================================================
    # 🏁 MODELO FINAL
    # =====================================================
    final_model = HistGradientBoostingRegressor(
        **study.best_params,
        random_state=42
    )

    final_model.fit(X, y)

    return final_model


# =====================================================
# 🚨 CLASIFICACIÓN (RIESGO)
# =====================================================

def train_classifier_advanced(X, y):
    """
    Entrena modelo de riesgo (quiebre/satisfacción).
    Versión enterprise:
    - validación temporal correcta
    - métricas consistentes
    - modelo final estable
    """

    tscv = TimeSeriesSplit(n_splits=3)

    cv_scores = []

    # =====================================================
    # 📊 VALIDACIÓN CROSS-TEMPORAL
    # =====================================================
    for train_idx, val_idx in tscv.split(X):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        acc = accuracy_score(y_val, pred)
        cv_scores.append(acc)

    print("\n📊 Clasificación CV Accuracy medio:", np.mean(cv_scores))

    # =====================================================
    # 🏁 MODELO FINAL (FULL TRAIN)
    # =====================================================
    final_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X, y)

    return final_model