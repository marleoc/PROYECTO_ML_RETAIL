import pandas as pd
import numpy as np
import joblib
import time

from pipeline.features import (
    build_all_features,
    get_feature_list,
    validate_features
)
from pipeline.data_loader import load_data_pipeline, get_engine


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
def load_inference_data(start_date="2026-04-10"):
    df = load_data_pipeline(start_date=start_date, use_chunks=True)

    print(f"📊 Data cargada: {df.shape}")

    return df


# =====================================================
# ⚙️ FEATURE PIPELINE (SAFE + PRODUCTION)
# =====================================================
def prepare_features(df):

    print("\n⚙️ Generando features...")

    df = build_all_features(df)

    features = get_feature_list()

    # 🔥 VALIDACIÓN CRÍTICA (EVITA DESALINEACIÓN)
    validate_features(df, features)

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

    # revertir log si aplica
    # if use_log:
    #    pred_reg = np.expm1(pred_reg)

    if use_log:
        pred_reg = np.clip(
            pred_reg,
            0,
            20
        )

    pred_reg = np.expm1(pred_reg)

    # protección contra valores extremos
    pred_reg = np.clip(pred_reg, 0, 1e6)

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

    # =================================================
    # 📦 STOCK RECOMENDADO (CON RIESGO)
    # =================================================
    results["Stock_Recomendado"] = (
        results["Prediccion_Demanda"] *
        (1.1 + results["Prob_Riesgo"] * 0.3)
    )

    return results


# =====================================================
# 📤 SAVE TO SQL SERVER (OPTIONAL)
# =====================================================
def save_to_sql(df, engine, table_name="PrediccionDemanda", schema="ml"):

    print(f"💾 Guardando en SQL Server: {table_name}")

    sql_df = pd.DataFrame({
        "FechaPrediccion": df["FechaPrediccion"],
        "FechaModelo": df["Fecha"],
        "ProductoID": df["ProductoID"],
        "SedeID": df["SedeID"],
        "Prediccion_Demanda": df["Prediccion_Demanda"],
        "Riesgo_Quiebre": df["Riesgo_Quiebre"],
        "Prob_Riesgo": df["Prob_Riesgo"],
        "Stock_Recomendado": df["Stock_Recomendado"],
        "FechaCarga": pd.Timestamp.now().normalize()
    })

    sql_df.to_sql(
        table_name,
        engine,
        schema=schema,
        if_exists="append",
        index=False
    )


# =====================================================
# 🚀 MAIN INFERENCE PIPELINE
# =====================================================
def run_inference(engine=None, save_sql=False, use_log=False):

    start = time.time()

    print("\n" + "="*60)
    print("🚀 RETAIL INFERENCE SYSTEM (ENTERPRISE)")
    print("="*60)

    # 1. Load models
    reg_model, clf_model = load_models()

    # 2. Load data
    df = load_inference_data("2026-04-11")

    # 3. Feature engineering
    df, X = prepare_features(df)
    ultima_fecha = df["Fecha"].max()

    print(f"Última fecha histórica: {ultima_fecha}")

    df_score = df[df["Fecha"] == ultima_fecha].copy()

    X_score = df_score[get_feature_list()]
    
    # 4. Predictions
    pred_reg, pred_risk, proba_risk = generate_predictions(
        X_score,
        reg_model,
        clf_model,
        use_log=use_log
    )

    # DEBUG LIGHT (solo producción)
    print("\n📊 DEBUG PREDICCIONES")
    print("Min:", pred_reg.min())
    print("Max:", pred_reg.max())
    print("NaN:", np.isnan(pred_reg).sum())
    print("Inf:", np.isinf(pred_reg).sum())

    # 5. Build output
    results = build_output(
        df_score,
        pred_reg,
        pred_risk,
        proba_risk
    )
    results["FechaPrediccion"] = (
        results["Fecha"] +
        pd.Timedelta(days=1)
    )

    ultima_fecha = results["Fecha"].max()

    predicciones_finales = (
        results
        .loc[results["Fecha"] == ultima_fecha]
        .copy()
    )
    
    print(
        predicciones_finales[
            [
                "FechaPrediccion",
                "ProductoID",
                "SedeID",
                "Prediccion_Demanda",
                "Riesgo_Quiebre",
                "Prob_Riesgo",
                "Stock_Recomendado"
            ]
        ].head()
    )

    print("\n📊 Preview resultados:")
    print(results[
        [
            "Fecha",
            "FechaPrediccion",
            "ProductoID",
            "Prediccion_Demanda",
            "Riesgo_Quiebre",
            "Prob_Riesgo"
        ]
        ].head())

    # 6. Save to SQL (optional)
    # if save_sql and engine is not None: 
        # save_to_sql(predicciones_finales, engine)

    print("\n⏱ Tiempo total:", round(time.time() - start, 2), "segundos")

    return results


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    run_inference(
        engine=get_engine(),  # Reemplazar con engine real si se quiere guardar
        save_sql=True,
        use_log=True
    )