import time
import numpy as np
import joblib

from pipeline.data_loader import load_data_pipeline
from pipeline.features import build_all_features, validate_features, get_feature_list
from pipeline.trainer import train_regression, train_classifier_advanced
from pipeline.evaluator import (
    evaluate_regression,
    evaluate_classification,
    evaluate_bias,
    evaluate_by_segment,
    top_errors
)
from pipeline.config import RISK_QUANTILE, USE_LOG_TARGET


# =====================================================
# 🚀 MAIN PIPELINE
# =====================================================
def main():

    start_time = time.time()

    print("\n" + "="*70)
    print("🚀 RETAIL ML PIPELINE - ENTERPRISE VERSION")
    print("="*70)

    # =================================================
    # 📥 LOAD DATA
    # =================================================
    print("\n📥 Cargando datos...")
    df = load_data_pipeline(start_date="2023-01-01", use_chunks=False)

    print(f"📊 Dataset cargado: {df.shape}")

    # =================================================
    # 🔍 EXPLORACIÓN TARGET
    # =================================================
    print("\n📊 DESCRIBE CantidadVendida")
    print(df["CantidadVendida"].describe())

    print("\n📊 QUANTILES CantidadVendida")
    print(df["CantidadVendida"].quantile([
        0.50, 0.75, 0.90, 0.95, 0.99, 0.999
    ]))

    # =================================================
    # 🔍 CANTIDAD DE PRODUCTOS
    # =================================================
    print("\n📊 CANTIDAD DE PRODUCTOS")
    print(df["ProductoID"].nunique())

    # =================================================
    # ⚙️ FEATURES
    # =================================================
    print("\n⚙️ Generando features...")
    df = build_all_features(df)

    FEATURES = get_feature_list()
    validate_features(df, FEATURES)

    df = df.sort_values("Fecha").reset_index(drop=True)

    X = df[FEATURES]

    # =================================================
    # 🎯 TARGET REGRESIÓN
    # =================================================
    if USE_LOG_TARGET:
        y_reg = np.log1p(df["CantidadVendida"])
    else:
        y_reg = df["CantidadVendida"]

    # =================================================
    # 🚨 TARGET CLASIFICACIÓN
    # =================================================
    threshold = df["CantidadVendida"].quantile(RISK_QUANTILE)
    y_clf = (df["CantidadVendida"] > threshold).astype(int)

    # =================================================
    # ✂️ SPLIT TEMPORAL
    # =================================================
    split = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train_reg, y_test_reg = y_reg.iloc[:split], y_reg.iloc[split:]
    y_train_clf, y_test_clf = y_clf.iloc[:split], y_clf.iloc[split:]

    # =================================================
    # 🏋️ TRAINING
    # =================================================
    print("\n📈 Entrenando modelo de demanda...")
    reg_model = train_regression(X_train, y_train_reg)

    print("\n🚨 Entrenando modelo de riesgo...")
    clf_model = train_classifier_advanced(X_train, y_train_clf)

    # =================================================
    # 🔮 PREDICCIONES
    # =================================================
    print("\n🔮 Generando predicciones...")

    # 🔹 predicción en escala modelo (log o normal)
    pred_reg_log = reg_model.predict(X_test)

    print("\n===== DEBUG LOG SCALE =====")
    print("Min:", np.min(pred_reg_log))
    print("Max:", np.max(pred_reg_log))
    print("Percentiles:", np.percentile(pred_reg_log, [50, 90, 95, 99, 99.9]))

    # =================================================
    # 🔄 SOLO UNA TRANSFORMACIÓN (CRÍTICO)
    # =================================================
    if USE_LOG_TARGET:
        safe_log_pred = np.clip(pred_reg_log, -20.0, 20.0)
        safe_log_true = np.clip(y_test_reg.to_numpy(), -20.0, 20.0)

        pred_reg = np.nan_to_num(np.expm1(safe_log_pred), nan=0.0, posinf=1e9, neginf=0.0)
        y_test_eval = np.nan_to_num(np.expm1(safe_log_true), nan=0.0, posinf=1e9, neginf=0.0)
    else:
        pred_reg = np.nan_to_num(pred_reg_log, nan=0.0, posinf=1e9, neginf=0.0)
        y_test_eval = np.nan_to_num(y_test_reg.values, nan=0.0, posinf=1e9, neginf=0.0)

    # =================================================
    # 🔍 DEBUG FINAL (POST TRANSFORMACIÓN)
    # =================================================
    print("\n===== DEBUG REAL SCALE =====")
    print("Min pred:", np.min(pred_reg))
    print("Max pred:", np.max(pred_reg))
    print("NaN:", np.isnan(pred_reg).sum())
    print("Inf:", np.isinf(pred_reg).sum())

    # =================================================
    # 🚨 CLASIFICACIÓN
    # =================================================
    pred_clf = clf_model.predict(X_test)
    proba_clf = clf_model.predict_proba(X_test)[:, 1]

    # =================================================
    # 📊 EVALUACIÓN REGRESIÓN
    # =================================================
    print("\n📊 Evaluación regresión...")
    evaluate_regression(
        y_test_eval,
        pred_reg,
        dataset_name="TEST",
        apply_inverse_transform=False
    )

    # =================================================
    # 🚨 EVALUACIÓN CLASIFICACIÓN
    # =================================================
    print("\n🚨 Evaluación clasificación...")
    evaluate_classification(
        y_test_clf,
        pred_clf,
        proba_clf,
        dataset_name="TEST"
    )

    # =================================================
    # 📦 RESULTADOS
    # =================================================
    results = X_test.copy()
    results["Real"] = y_test_eval
    results["Predicho"] = pred_reg

    # =================================================
    # 📉 BIAS
    # =================================================
    print("\n📉 Evaluando bias...")
    evaluate_bias(y_test_eval, pred_reg, apply_inverse_transform=False)

    # =================================================
    # 📦 SEGMENTACIÓN
    # =================================================
    print("\n📦 Segmentación de error...")
    evaluate_by_segment(results)

    # =================================================
    # 🔥 TOP ERRORES
    # =================================================
    print("\n🔥 Top errores...")
    top_errors(results)

    # =================================================
    # 🎯 PRODUCTOS QUE MÁS ERROR GENERAN
    # =================================================

    results["Error_Abs"] = np.abs(
        results["Real"] - results["Predicho"]
    )

    top_products = (
        results
        .groupby("ProductoID")
        .agg(
            Error_Total=("Error_Abs", "sum"),
            Error_Medio=("Error_Abs", "mean"),
            Casos=("Error_Abs", "count")
        )
        .sort_values(
            "Error_Total",
            ascending=False
        )
    )

    print("\n" + "="*60)
    print("🎯 TOP PRODUCTOS POR ERROR ACUMULADO")
    print("="*60)

    print(top_products.head(30))

    total_error = top_products["Error_Total"].sum()

    top10_error = (
        top_products.head(10)["Error_Total"].sum()
    )

    top20_error = (
        top_products.head(20)["Error_Total"].sum()
    )

    top50_error = (
        top_products.head(50)["Error_Total"].sum()
    )

    print("\n📊 Concentración del error")

    print(
        f"Top 10 productos explican "
        f"{100*top10_error/total_error:.2f}% "
        f"del error total"
    )

    print(
        f"Top 20 productos explican "
        f"{100*top20_error/total_error:.2f}% "
        f"del error total"
    )

    print(
        f"Top 50 productos explican "
        f"{100*top50_error/total_error:.2f}% "
        f"del error total"
    )

    # =================================================
    # 🔍 PRODUCTOS QUE MÁS GENERAN ERROR
    # =================================================

    results["Error_Abs"] = np.abs(
        results["Real"] - results["Predicho"]
    )

    top_error_ids = (
        results
        .sort_values("Error_Abs", ascending=False)
        .head(100)
        ["ProductoID"]
        .value_counts()
    )

    print("\n" + "="*60)
    print("🔍 PRODUCTOS MÁS FRECUENTES EN TOP 100 ERRORES")
    print("="*60)
    print(top_error_ids)

    # =================================================
    # 💾 MODELOS
    # =================================================
    print("\n💾 Guardando modelos...")

    joblib.dump(reg_model, "models/modelo_demanda.pkl")
    joblib.dump(clf_model, "models/modelo_riesgo.pkl")

    # =================================================
    # ⏱ TIEMPO TOTAL
    # =================================================
    end_time = time.time()

    print("\n" + "="*70)
    print("⏱ TIEMPO TOTAL PIPELINE")
    print(f"{end_time - start_time:.2f} segundos")
    print("="*70)


if __name__ == "__main__":
    main()