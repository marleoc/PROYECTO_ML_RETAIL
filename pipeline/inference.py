import pandas as pd
import joblib

from pipeline.features import build_all_features, validate_features
from pipeline.config import FEATURES, RISK_THRESHOLD
from sqlalchemy import create_engine


def load_models(
    reg_path="models/modelo_demanda.pkl",
    clf_path="models/modelo_riesgo.pkl"
):

    reg_model = joblib.load(reg_path)
    clf_model = joblib.load(clf_path)

    return reg_model, clf_model

def predict_demand(model, df):

    df = build_all_features(df)
    validate_features(df, FEATURES)

    df = df.sort_values("Fecha").reset_index(drop=True)

    X = df[FEATURES]

    df["Prediccion_Demanda"] = model.predict(X)

    return df

def predict_risk(model, df):

    df = build_all_features(df)
    validate_features(df, FEATURES)

    df = df.sort_values("Fecha").reset_index(drop=True)

    X = df[FEATURES]

    df["Prob_Riesgo_Quiebre"] = model.predict_proba(X)[:, 1]

    df["Riesgo_Quiebre"] = (df["Prob_Riesgo_Quiebre"] >= RISK_THRESHOLD).astype(int)

    return df

def run_inference(df, reg_model, clf_model):

    df = build_all_features(df)
    validate_features(df, FEATURES)

    df = df.sort_values("Fecha").reset_index(drop=True)

    X = df[FEATURES]

    # 📈 demanda
    df["Prediccion_Demanda"] = reg_model.predict(X)

    # 🚨 riesgo
    df["Prob_Riesgo_Quiebre"] = clf_model.predict_proba(X)[:, 1]


    df["Riesgo_Quiebre"] = (df["Prob_Riesgo_Quiebre"] >= RISK_THRESHOLD).astype(int)

    return df

def export_predictions(df, path="outputs/predicciones.csv"):

    df.to_csv(path, index=False)

    print(f"📦 Predicciones guardadas en: {path}")

def save_to_sql(df):

    engine = create_engine(
        "mssql+pyodbc://@localhost/DB_RETAIL_ML"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&trusted_connection=yes"
        "&TrustServerCertificate=yes"
    )

    df_to_save = df[[
        "Fecha",
        "ProductoID",
        "SedeID",
        "Prediccion_Demanda",
        "Prob_Riesgo_Quiebre",
        "Riesgo_Quiebre"
    ]].copy()

    df_to_save["Fecha_Ejecucion"] = pd.Timestamp.now()

    df_to_save.to_sql(
        "predicciones_diarias",
        engine,
        schema="ml",
        if_exists="append",
        index=False
    )