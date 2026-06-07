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

def evaluate_regression(y_true, y_pred, dataset_name="TEST"):
    """
    Evalúa modelo de regresión para demanda.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n" + "="*50)
    print(f"📊 EVALUACIÓN REGRESIÓN - {dataset_name}")
    print("="*50)

    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def evaluate_classification(y_true, y_pred, y_proba=None, dataset_name="TEST"):
    """
    Evalúa modelo de clasificación para riesgo de quiebre.
    """

    print("\n" + "="*50)
    print(f"🚨 EVALUACIÓN CLASIFICACIÓN - {dataset_name}")
    print("="*50)

    # 📊 Reporte principal
    report = classification_report(y_true, y_pred)
    print("\n📊 Classification Report:")
    print(report)

    # 📉 Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print("\n📉 Confusion Matrix:")
    print(cm)

    results = {
        "classification_report": report,
        "confusion_matrix": cm
    }

    # 📈 ROC-AUC (si hay probabilidades)
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"\n🎯 ROC-AUC: {auc:.4f}")
        results["ROC-AUC"] = auc

    return results

def evaluate_by_segment(df_results, target_col="Real", pred_col="Predicho"):
    """
    Evalúa error por segmentos de demanda.
    """

    print("\n" + "="*50)
    print("📦 ANÁLISIS POR SEGMENTO")
    print("="*50)

    segments = [500, 1000, 5000, 10000]

    segment_metrics = []

    for limit in segments:

        subset = df_results[df_results[target_col] <= limit]

        if len(subset) == 0:
            continue

        mae = mean_absolute_error(subset[target_col], subset[pred_col])

        print(f"\n🔹 Ventas <= {limit}")
        print(f"Registros: {len(subset):,}")
        print(f"MAE: {mae:.2f}")

        segment_metrics.append({
            "segment": limit,
            "mae": mae,
            "n": len(subset)
        })

    return pd.DataFrame(segment_metrics)

def evaluate_bias(y_true, y_pred):
    """
    Detecta si el modelo sobreestima o subestima.
    """

    error = np.array(y_true) - np.array(y_pred)
    bias = np.mean(error)

    print("\n" + "="*50)
    print("📉 BIAS DEL MODELO")
    print("="*50)

    print(f"Bias: {bias:.4f}")

    if bias > 0:
        print("⚠️ El modelo SUBESTIMA la demanda")
    elif bias < 0:
        print("⚠️ El modelo SOBREESTIMA la demanda")
    else:
        print("✔ Sin sesgo aparente")

    return bias

def top_errors(df_results, n=10):
    """
    Retorna los mayores errores absolutos.
    """

    df_results = df_results.copy()
    df_results["Error_Abs"] = np.abs(df_results["Real"] - df_results["Predicho"])

    top = df_results.sort_values("Error_Abs", ascending=False).head(n)

    print("\n" + "="*50)
    print(f"🔥 TOP {n} ERRORES")
    print("="*50)

    print(top)

    return top