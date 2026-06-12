from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib

from pipeline.features import build_all_features, get_feature_list

app = FastAPI(title="Retail Demand Forecast API")


# =========================
# LOAD MODELS ON STARTUP
# =========================
reg_model = joblib.load("models/modelo_demanda.pkl")
clf_model = joblib.load("models/modelo_riesgo.pkl")


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "ok", "service": "retail-forecast"}


# =========================
# SINGLE PREDICTION
# =========================
@app.post("/predict")
def predict(payload: dict):

    # Convert input → DataFrame
    df = pd.DataFrame([payload])

    # Feature engineering (MISMO PIPELINE)
    df = build_all_features(df)

    features = get_feature_list()
    X = df[features]

    # Predictions
    pred = reg_model.predict(X)
    pred = np.clip(pred, 0, 1e6)

    risk = clf_model.predict(X)[0]
    prob = clf_model.predict_proba(X)[0, 1]

    return {
        "prediccion_demanda": float(pred[0]),
        "riesgo_quiebre": int(risk),
        "probabilidad_riesgo": float(prob)
    }