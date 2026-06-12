from prefect import flow, task
import pandas as pd
import joblib

from pipeline.data_loader import load_data_pipeline
from pipeline.features import build_all_features, get_feature_list


# =========================
# LOAD MODELS
# =========================
reg_model = joblib.load("models/modelo_demanda.pkl")
clf_model = joblib.load("models/modelo_riesgo.pkl")


# =========================
# TASK: LOAD DATA
# =========================
@task
def load_data():
    df = load_data_pipeline(start_date="2023-01-01", use_chunks=True)
    return df


# =========================
# TASK: FEATURES
# =========================
@task
def features(df):
    df = build_all_features(df)
    return df


# =========================
# TASK: PREDICT
# =========================
@task
def predict(df):

    features = get_feature_list()
    X = df[features]

    df["Prediccion_Demanda"] = reg_model.predict(X)
    df["Riesgo_Quiebre"] = clf_model.predict(X)
    df["Prob_Riesgo"] = clf_model.predict_proba(X)[:, 1]

    return df


# =========================
# TASK: SAVE
# =========================
@task
def save(df):
    df.to_parquet("output/predicciones.parquet", index=False)
    print("Guardado en output/")


# =========================
# FLOW
# =========================
@flow
def retail_batch():

    df = load_data()
    df = features(df)
    df = predict(df)
    save(df)


if __name__ == "__main__":
    retail_batch()