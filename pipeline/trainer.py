import numpy as np
import optuna

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier

# opcional (si lo tienes instalado)
from lightgbm import LGBMRegressor


# =====================================================
# 🧠 UTILIDAD: MÉTRICAS
# =====================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# =====================================================
# 🧠 SEGMENTACIÓN DE DEMANDA (ROUTING)
# =====================================================
def get_demand_segment(y):

    median = np.median(y)

    if median < 50:
        return "LOW"
    elif median < 500:
        return "MID"
    else:
        return "HIGH"


# =====================================================
# 🚀 MODELO LOW DEMAND (INTERMITENTE)
# =====================================================
def train_low_model(X, y):

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 60)
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for tr, val in tscv.split(X):

            model = RandomForestClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )

            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[val])

            scores.append(mean_absolute_error(y.iloc[val], pred))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)

    best = study.best_params

    model = RandomForestClassifier(
        **best,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X, y)

    return model


# =====================================================
# 🚀 MODELO MID / HIGH DEMAND (REGRESIÓN)
# =====================================================
def train_boosting_model(X, y):

    def objective(trial):

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "max_iter": trial.suggest_int("max_iter", 200, 600),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 2.0)
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for tr, val in tscv.split(X):

            model = HistGradientBoostingRegressor(
                **params,
                random_state=42
            )

            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[val])

            scores.append(rmse(y.iloc[val], pred))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best = study.best_params

    model = HistGradientBoostingRegressor(
        **best,
        random_state=42
    )

    model.fit(X, y)

    return model


# =====================================================
# 🧠 ROUTER PRINCIPAL
# =====================================================
def train_regression(X, y):

    print("\n🚀 Training LightGBM Retail Model")

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


# =====================================================
# 🚨 CLASIFICACIÓN (RIESGO)
# =====================================================
def train_classifier_advanced(X, y):

    tscv = TimeSeriesSplit(n_splits=3)

    best_model = None
    best_score = 0

    for tr, val in tscv.split(X):

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X.iloc[tr], y.iloc[tr])

        score = model.score(X.iloc[val], y.iloc[val])

        if score > best_score:
            best_score = score
            best_model = model

    print(f"📊 Best classification CV accuracy: {best_score:.4f}")

    best_model.fit(X, y)

    return best_model