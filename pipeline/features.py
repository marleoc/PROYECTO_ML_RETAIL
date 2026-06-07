import pandas as pd
import numpy as np

def build_features(df):
    """
    Genera features temporales y de comportamiento para retail ML.
    """

    df = df.copy()

    # =========================
    # 📅 FEATURES TEMPORALES
    # =========================

    df["Fecha"] = pd.to_datetime(df["Fecha"])

    df["Mes"] = df["Fecha"].dt.month
    df["DiaSemana"] = df["Fecha"].dt.dayofweek
    df["Trimestre"] = df["Fecha"].dt.quarter
    df["DiaMes"] = df["Fecha"].dt.day

    df["FinDeSemana"] = (df["DiaSemana"] >= 5).astype(int)

    return df

def add_interaction_features(df):

    df = df.copy()

    # 💰 Precio relativo (proxy de elasticidad)
    df["Precio_x_Promo"] = df["PrecioPromedio"] * df["TienePromocion"]

    # 📈 Intensidad de ventas recientes
    df["Ratio_7_30"] = df["Venta_7_Dias"] / (df["Venta_30_Dias"] + 1)

    # 📊 Cambio de demanda
    df["Cambio_7_dias"] = df["Venta_7_Dias"] - df["Promedio_7_Dias"]

    df["Cambio_30_dias"] = df["Venta_30_Dias"] - df["Promedio_30_Dias"]

    return df

def add_volatility_features(df):

    df = df.copy()

    # ⚡ volatilidad simple
    df["Volatilidad_7"] = np.abs(df["Venta_7_Dias"] - df["Promedio_7_Dias"])

    df["Volatilidad_30"] = np.abs(df["Venta_30_Dias"] - df["Promedio_30_Dias"])

    return df

def add_risk_features(df):

    df = df.copy()

    # 🚨 presión de demanda (proxy de quiebre)
    df["Pressure_Index"] = (
        df["Venta_7_Dias"] /
        (df["Promedio_7_Dias"] + 1)
    )

    return df

def build_all_features(df):

    df = build_features(df)

    df = add_interaction_features(df)

    df = add_volatility_features(df)

    df = add_risk_features(df)

    return df

def validate_features(df, expected_features):

    missing = [col for col in expected_features if col not in df.columns]

    if len(missing) > 0:
        raise ValueError(f"❌ Faltan features: {missing}")

    print("✅ Features validadas correctamente")

    return df