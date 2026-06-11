import numpy as np
import pandas as pd

from pipeline.config import USE_LOG_TARGET

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# =====================================================
# 📊 REGRESIÓN (CORE METRICS)
# =====================================================
def evaluate_regression(y_true, y_pred, dataset_name="TEST"):

    # =========================
    # 🔄 inverse transform seguro
    # =========================
    if USE_LOG_TARGET:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    # =========================
    # 🧮 métricas base
    # =========================
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # =========================
    # 📊 output
    # =========================
    print("\n" + "="*60)
    print(f"📊 REGRESIÓN - {dataset_name}")
    print("="*60)

    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    # =========================
    # ⚠️ diagnóstico automático
    # =========================
    bias = np.mean(y_true - y_pred)

    print(f"\n📉 Bias: {bias:.4f}")

    if bias > 0:
        print("⚠️ SUBESTIMA demanda")
    else:
        print("⚠️ SOBREESTIMA demanda")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Bias": bias
    }


# =====================================================
# 🚨 CLASIFICACIÓN
# =====================================================
def evaluate_classification(y_true, y_pred, y_proba=None, dataset_name="TEST"):

    print("\n" + "="*60)
    print(f"🚨 CLASIFICACIÓN - {dataset_name}")
    print("="*60)

    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 Classification Report:")
    print(report)

    print("\n📉 Confusion Matrix:")
    print(cm)

    results = {
        "classification_report": report,
        "confusion_matrix": cm
    }

    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"\n🎯 ROC-AUC: {auc:.4f}")
        results["ROC-AUC"] = auc

    return results


# =====================================================
# 📦 ANÁLISIS POR SEGMENTO (CRÍTICO RETAIL)
# =====================================================
def evaluate_by_segment(df_results,
                        target_col="Real",
                        pred_col="Predicho"):

    print("\n" + "="*60)
    print("📦 ANÁLISIS POR SEGMENTO")
    print("="*60)

    segments = [100, 500, 1000, 5000, 10000]

    summary = []

    for limit in segments:

        subset = df_results[df_results[target_col] <= limit]

        if len(subset) == 0:
            continue

        mae = mean_absolute_error(subset[target_col], subset[pred_col])
        rmse = np.sqrt(mean_squared_error(subset[target_col], subset[pred_col]))

        print(f"\n🔹 <= {limit}")
        print(f"Registros: {len(subset):,}")
        print(f"MAE : {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        summary.append({
            "segment": limit,
            "mae": mae,
            "rmse": rmse,
            "n": len(subset)
        })

    return pd.DataFrame(summary)


# =====================================================
# 📉 BIAS GLOBAL
# =====================================================
def evaluate_bias(y_true, y_pred):

    if USE_LOG_TARGET:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    error = np.array(y_true) - np.array(y_pred)
    bias = np.mean(error)

    print("\n" + "="*60)
    print("📉 BIAS GLOBAL")
    print("="*60)

    print(f"Bias: {bias:.4f}")

    if bias > 0:
        print("⚠️ SUBESTIMA demanda")
    elif bias < 0:
        print("⚠️ SOBREESTIMA demanda")
    else:
        print("✔ Sin sesgo")

    return bias


# =====================================================
# 🔥 TOP ERRORES (OUTLIERS OPERACIONALES)
# =====================================================
def top_errors(df_results, n=10):

    df = df_results.copy()

    df["Error_Abs"] = np.abs(df["Real"] - df["Predicho"])

    top = df.sort_values("Error_Abs", ascending=False).head(n)

    print("\n" + "="*60)
    print(f"🔥 TOP {n} ERRORES")
    print("="*60)

    print(top)

    return top