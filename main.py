import time
import pandas as pd
import numpy as np
import joblib

from pipeline.data_loader import load_data_pipeline
from pipeline.features import build_all_features, validate_features
from pipeline.trainer import train_regression, train_classifier_advanced
from pipeline.evaluator import (
    evaluate_regression,
    evaluate_classification,
    evaluate_bias,
    evaluate_by_segment,
    top_errors
)
from pipeline.config import FEATURES, RISK_QUANTILE, USE_LOG_TARGET


def main():

    start_time = time.time()

    print("\n" + "="*60)
    print("🚀 RETAIL ML PIPELINE - PRODUCCIÓN")
    print("="*60)

    # =========================
    # 📥 DATA
    # =========================
    print("\n📥 Cargando datos...")
    df = load_data_pipeline(start_date="2024-01-01", use_chunks=True)

    print(f"📊 Dataset cargado: {df.shape}")

    # =========================
    # 🔍 ANÁLISIS EXPLORATORIO
    # =========================
    print("\n🔍 Análisis exploratorio inicial...")
    print(df["CantidadVendida"].quantile([
        0.90,
        0.95,
        0.99,
        0.995,
        0.999,
        0.9999
    ]))
    # =========================
    # ⚙️ FEATURES
    # =========================
    print("\n⚙️ Generando features...")
    df = build_all_features(df)
    validate_features(df, FEATURES)

    df = df.sort_values("Fecha").reset_index(drop=True)

    X = df[FEATURES]

    # =========================
    # 🎯 TARGET REGRESIÓN
    # =========================
    if USE_LOG_TARGET:
        y_reg = np.log1p(df["CantidadVendida"])
    else:
        y_reg = df["CantidadVendida"]

    # =========================
    # 🎯 TARGET CLASIFICACIÓN
    # =========================
    threshold = df["CantidadVendida"].quantile(RISK_QUANTILE)
    y_clf = (df["CantidadVendida"] > threshold).astype(int)

    # =========================
    # ✂️ SPLIT TEMPORAL
    # =========================
    split = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train_reg, y_test_reg = y_reg.iloc[:split], y_reg.iloc[split:]
    y_train_clf, y_test_clf = y_clf.iloc[:split], y_clf.iloc[split:]

    # =========================
    # 🏋️ TRAINING
    # =========================
    print("\n📈 Entrenando modelo de demanda (regresión)...")
    reg_model = train_regression(
        X_train,
        y_train_reg,
        n_trials=20
    )

    print("\n🚨 Entrenando modelo de riesgo (clasificación)...")
    clf_model = train_classifier_advanced(X_train, y_train_clf)

    # =========================
    # 🔮 PREDICCIONES
    # =========================
    print("\n🔮 Generando predicciones...")

    # REGRESIÓN
    pred_reg = reg_model.predict(X_test)

    if USE_LOG_TARGET:
        pred_reg = np.expm1(pred_reg)
        y_test_eval = np.expm1(y_test_reg)
    else:
        y_test_eval = y_test_reg.values

    # CLASIFICACIÓN
    pred_clf = clf_model.predict(X_test)
    proba_clf = clf_model.predict_proba(X_test)[:, 1]

    # =========================
    # 📊 EVALUACIÓN REGRESIÓN
    # =========================
    print("Predicción máxima:", np.max(pred_reg))
    print("Real máximo:", np.max(y_test_eval))
    evaluate_regression(
        y_test_eval,
        pred_reg,
        dataset_name="TEST"
    )

    # =========================
    # 🚨 EVALUACIÓN CLASIFICACIÓN
    # =========================
    evaluate_classification(
        y_test_clf,
        pred_clf,
        proba_clf,
        dataset_name="TEST"
    )

    # =========================
    # 📦 DATAFRAME RESULTADOS
    # =========================
    results = X_test.copy()

    results["Real"] = y_test_eval.values
    results["Predicho"] = pred_reg

    # =========================
    # 📉 BIAS
    # =========================
    print("\n📉 Evaluando sesgo...")
    evaluate_bias(y_test_eval, pred_reg)

    # =========================
    # 📦 SEGMENTACIÓN ERROR
    # =========================
    print("\n📦 Segmentación de error...")
    evaluate_by_segment(results)

    # =========================
    # 🔥 TOP ERRORES
    # =========================
    print("\n🔥 Top errores...")
    top_errors(results)

    # =========================
    # 💾 GUARDADO MODELOS
    # =========================
    print("\n💾 Guardando modelos...")

    joblib.dump(reg_model, "models/modelo_demanda.pkl")
    joblib.dump(clf_model, "models/modelo_riesgo.pkl")

    # =========================
    # ⏱ TIEMPO TOTAL
    # =========================
    end_time = time.time()

    print("\n" + "="*60)
    print("⏱ TIEMPO TOTAL DEL PIPELINE")
    print(f"{end_time - start_time:.2f} segundos")
    print("="*60)


if __name__ == "__main__":
    main()