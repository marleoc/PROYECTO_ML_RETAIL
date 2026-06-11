import numpy as np
import pandas as pd
from pipeline.config import FEATURES

def get_feature_list():
    """
    Retorna la lista oficial de features del modelo.
    Fuente única de verdad: conf.py
    """
    return FEATURES

# =====================================================
# 📅 FEATURES TEMPORALES
# =====================================================
def add_time_features(df):

    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    df["Semana"] = df["Fecha"].dt.isocalendar().week.astype(int)
    df["Mes"] = df["Fecha"].dt.month
    df["DiaSemana"] = df["Fecha"].dt.dayofweek
    df["DiaMes"] = df["Fecha"].dt.day
    df["Trimestre"] = df["Fecha"].dt.quarter

    df["FinDeSemana"] = (df["DiaSemana"] >= 5).astype(int)
    df["EsFinMes"] = df["Fecha"].dt.is_month_end.astype(int)
    df["EsQuincena"] = df["DiaMes"].isin([15, 30, 31]).astype(int)

    return df


# =====================================================
# 📉 LAGS (ANTI-LEAKAGE)
# =====================================================
def add_lag_features(df):

    df = df.sort_values(["ProductoID", "SedeID", "Fecha"])

    group = df.groupby(["ProductoID", "SedeID"])["CantidadVendida"]

    df["Venta_Ayer"] = group.transform(lambda x: x.shift(1))
    df["Venta_7_Dias"] = group.transform(lambda x: x.shift(7))
    df["Venta_30_Dias"] = group.transform(lambda x: x.shift(30))

    return df


# =====================================================
# 📊 ROLLING FEATURES (SIN RESET_INDEX)
# =====================================================
def add_rolling_features(df):

    df = df.sort_values(["ProductoID", "SedeID", "Fecha"])

    group = df.groupby(["ProductoID", "SedeID"])["CantidadVendida"]

    df["Promedio_7_Dias"] = group.transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df["Promedio_30_Dias"] = group.transform(
        lambda x: x.shift(1).rolling(30).mean()
    )

    df["Std_7_Dias"] = group.transform(
        lambda x: x.shift(1).rolling(7).std()
    )

    df["Std_30_Dias"] = group.transform(
        lambda x: x.shift(1).rolling(30).std()
    )

    return df


# =====================================================
# ⚡ FEATURES DERIVADAS (MEJORA R2)
# =====================================================
def add_advanced_features(df):

    # Momentum (tendencia relativa)
    df["Momentum_7_30"] = (
        df["Promedio_7_Dias"] /
        (df["Promedio_30_Dias"] + 1e-6)
    )

    # Growth rate
    df["Growth_7_30"] = (
        df["Promedio_7_Dias"] - df["Promedio_30_Dias"]
    ) / (df["Promedio_30_Dias"] + 1e-6)

    # Volatilidad basada en ventas recientes
    df["Volatilidad_7"] = df["Venta_Ayer"].rolling(7).std()
    df["Volatilidad_30"] = df["Venta_Ayer"].rolling(30).std()

    return df


# =====================================================
# 💰 FEATURES DE PRECIO
# =====================================================
def add_price_features(df):

    df["Precio_relativo_producto"] = (
        df["PrecioPromedio"] /
        (df.groupby("ProductoID")["PrecioPromedio"].transform("mean") + 1e-6)
    )

    return df


# =====================================================
# 📢 FEATURES PROMO (FIX CRÍTICO)
# =====================================================
def add_promo_features(df):

    df["TienePromocion"] = df["TienePromocion"].fillna(0)

    df = df.sort_values(["ProductoID", "SedeID", "Fecha"])

    group = df.groupby(["ProductoID", "SedeID"])["TienePromocion"]

    df["Promo_Acumulada_7d"] = group.transform(
        lambda x: x.shift(1).rolling(7).sum()
    )

    return df


# =====================================================
# 🚀 PIPELINE PRINCIPAL
# =====================================================
def build_all_features(df):

    df = df.copy()

    print("⚙️ Time features...")
    df = add_time_features(df)

    print("⚙️ Lag features...")
    df = add_lag_features(df)

    print("⚙️ Rolling features...")
    df = add_rolling_features(df)

    print("⚙️ Advanced features...")
    df = add_advanced_features(df)

    print("⚙️ Price features...")
    df = add_price_features(df)

    print("⚙️ Promo features...")
    df = add_promo_features(df)

    # limpieza final (CRÍTICO PARA TRAINING)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    print("✅ Features generadas correctamente (sin leakage + sin errores de índice)")

    return df

def validate_features(df, expected_features):

    print("\n🔍 Validando features del pipeline...")

    missing = [col for col in expected_features if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_features]

    # ❌ Features faltantes (CRÍTICO)
    if len(missing) > 0:
        raise ValueError(
            f"❌ Faltan features requeridas en el dataset: {missing}"
        )

    # ⚠️ Features extra (warning, no rompe pipeline)
    if len(extra) > 0:
        print(f"⚠️ Features adicionales no usadas por el modelo: {extra}")

    # 🔍 check NaNs críticos
    nan_cols = df[expected_features].isna().sum()
    nan_cols = nan_cols[nan_cols > 0]

    if len(nan_cols) > 0:
        print("\n⚠️ Features con valores nulos:")
        print(nan_cols)

    print("✅ Validación de features completada correctamente")

    return df