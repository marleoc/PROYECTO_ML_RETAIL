import pandas as pd
import numpy as np
import joblib
import time

from pipeline.features import build_all_features, get_feature_list
from pipeline.data_loader import load_data_pipeline


# =====================================================
# 🚀 LOAD MODELS
# =====================================================
def load_models():

    reg_model = joblib.load("models/modelo_demanda.pkl")
    clf_model = joblib.load("models/modelo_riesgo.pkl")

    return reg_model, clf_model


# =====================================================
# 📥 LOAD DATA (T-1 ONLY)
# =====================================================
def load_inference_data(start_date="2024-01-01"):

    df = load_data_pipeline(start_date=start_date, use_chunks=True)

    print(f"📊 Data cargada: {df.shape}")

    return df


# =====================================================
# ⚙️ FEATURE PIPELINE (SAFE)
# =====================================================
def prepare_features(df):

    df = build_all_features(df)

    features = get_feature_list()

    df = df.sort_values("Fecha").reset_index(drop=True)

    X = df[features]

    return df, X


# =====================================================
# 🔮 BATCH SCORING
# =====================================================
def generate_predictions(X, reg_model, clf_model, use_log=False):

    print("🔮 Generando predicciones...")

    # -------------------------
    # REGRESIÓN
    # -------------------------
    pred_reg = reg_model.predict(X)

    if use_log:
        pred_reg = np.expm1(pred_reg)

    # -------------------------
    # CLASIFICACIÓN
    # -------------------------
    pred_risk = clf_model.predict(X)
    proba_risk = clf_model.predict_proba(X)[:, 1]

    return pred_reg, pred_risk, proba_risk


# =====================================================
# 💾 OUTPUT PARA POWER BI / SQL SERVER
# =====================================================
def build_output(df, pred_reg, pred_risk, proba_risk):

    results = df.copy()

    results["Prediccion_Demanda"] = pred_reg
    results["Riesgo_Quiebre"] = pred_risk
    results["Prob_Riesgo"] = proba_risk

    # métricas de negocio
    results["Stock_Recomendado"] = results["Prediccion_Demanda"] * 1.1

    return results


# =====================================================
# 📤 SAVE TO SQL SERVER (OPTIONAL)
# =====================================================
def save_to_sql(df, engine, table_name="ml.predicciones_demanda"):

    print(f"💾 Guardando en SQL Server: {table_name}")

    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False
    )


# =====================================================
# 🚀 MAIN INFERENCE PIPELINE
# =====================================================
def run_inference(engine=None, save_sql=False, use_log=False):

    start = time.time()

    print("\n" + "="*60)
    print("🚀 RETAIL INFERENCE SYSTEM")
    print("="*60)

    # 1. Load models
    reg_model, clf_model = load_models()

    # 2. Load data
    df = load_inference_data()

    # 3. Feature engineering
    df, X = prepare_features(df)

    # 4. Predictions
    pred_reg, pred_risk, proba_risk = generate_predictions(
        X, reg_model, clf_model, use_log=use_log
    )

    # 5. Build output
    results = build_output(df, pred_reg, pred_risk, proba_risk)

    print("\n📊 Preview resultados:")
    print(results[[
        "Fecha",
        "ProductoID",
        "Prediccion_Demanda",
        "Riesgo_Quiebre",
        "Prob_Riesgo"
    ]].head())

    # 6. Save to SQL (optional)
    if save_sql and engine is not None:
        save_to_sql(results, engine)

    print("\n⏱ Tiempo total:", round(time.time() - start, 2), "segundos")

    return results