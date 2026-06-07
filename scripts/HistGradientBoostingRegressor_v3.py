# =====================================================
# 📦 LIBRERÍAS
# =====================================================
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import gc
import time
import optuna

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.inspection import permutation_importance
# =====================================================
# ⏱ INICIO
# =====================================================
start_time = time.time()

# =====================================================
# 🔌 CONEXIÓN
# =====================================================
engine = create_engine(
    "mssql+pyodbc://@localhost/DB_RETAIL_ML"
    "?driver=ODBC+Driver+18+for+SQL+Server"
    "&trusted_connection=yes"
    "&TrustServerCertificate=yes"
)

# =====================================================
# 📥 DATA
# =====================================================
query = """
SELECT
    Fecha,
    CantidadVendida,
    ProductoID,
    SedeID,
    PrecioPromedio,
    Semana,
    EsFinMes,
    EsQuincena,
    Venta_Ayer,
    Venta_7_Dias,
    Venta_30_Dias,
    Promedio_7_Dias,
    Promedio_30_Dias,
    TienePromocion
FROM ml.dm_prediccion_demanda
WHERE Fecha >= '2024-01-01'
ORDER BY Fecha
"""

chunks = pd.read_sql(query, engine, chunksize=200_000)

df_list = []
for chunk in chunks:
    chunk = chunk.dropna()
    df_list.append(chunk)
    del chunk
    gc.collect()

df = pd.concat(df_list, ignore_index=True)
# =====================================================
# 📅 FEATURES TEMPORALES
# =====================================================

df["Fecha"] = pd.to_datetime(df["Fecha"])

df["Mes"] = df["Fecha"].dt.month

df["DiaSemana"] = df["Fecha"].dt.dayofweek

df["Trimestre"] = df["Fecha"].dt.quarter

df["DiaMes"] = df["Fecha"].dt.day

df["FinDeSemana"] = (df["DiaSemana"] >= 5).astype(int)
# =====================================================
# 🔍 DIAGNÓSTICO: TOP PRODUCTOS POR VOLUMEN
# =====================================================

print("\nTOP PRODUCTOS POR VOLUMEN (TOTAL VENTAS)")

top_productos = (
    df.groupby("ProductoID")
    .agg(
        ventas_totales=("CantidadVendida", "sum"),
        promedio=("CantidadVendida", "mean"),
        frecuencia=("CantidadVendida", "count")
    )
    .sort_values("ventas_totales", ascending=False)
    .head(20)
)

print(top_productos)

# =====================================================
# 🧠 FEATURES BASE
# =====================================================
X_columns = [
    "ProductoID",
    "SedeID",
    "PrecioPromedio",
    "Semana",
    "Mes",
    "DiaSemana",
    "DiaMes",
    "FinDeSemana",
    "Trimestre",
    "EsFinMes",
    "EsQuincena",
    "Venta_Ayer",
    "Venta_7_Dias",
    "Venta_30_Dias",
    "Promedio_7_Dias",
    "Promedio_30_Dias",
    "TienePromocion"
]

# =====================================================
# 🔥 MEJORAS POTENCIALES DE FEATURES (IMPORTANTE)
# =====================================================
# 🔴 Estas variables podrían MEJORAR el modelo significativamente:

# 📊 VARIABLES TEMPORALES (mejoran estacionalidad)
# df["DiaSemana"]        -> captura comportamiento semanal
# df["Mes"]              -> estacionalidad mensual
# df["Trimestre"]        -> ciclos largos
# df["DiaDelMes"]        -> comportamiento de quincena real

# 📈 VARIABLES DE TENDENCIA
# df["Cambio_7_dias"]    -> diferencia vs semana anterior
# df["Cambio_30_dias"]   -> tendencia mensual
# df["Rolling_Std_7"]    -> volatilidad de demanda

# 🏬 VARIABLES DE NEGOCIO
# df["StockDisponible"]  -> evita sobreestimación
# df["QuiebreStock"]     -> explica ventas artificialmente bajas

# 💰 VARIABLES DE PRECIO AVANZADO
# df["DescuentoReal"]    -> mejor que booleano
# df["Elasticidad"]      -> impacto del precio en demanda

# 📢 VARIABLES DE MARKETING
# df["CampañaActiva"]    -> campañas reales
# df["TipoPromocion"]    -> no todas las promos son iguales

# ⚠️ IMPORTANTE:
# agregar variables futuras (post-fecha) puede causar LEAKAGE
# ejemplo: ventas futuras o promedios calculados incorrectamente

# =====================================================
# 🎯 TARGET
# =====================================================
# y = df["CantidadVendida"]

# df = df.sort_values("Fecha")
# X = df[X_columns]
# Transformación logarítmica
df = df.sort_values("Fecha").reset_index(drop=True)

X = df[X_columns] 
y = np.log1p(df["CantidadVendida"])

# =====================================================
# 🔍 DIAGNÓSTICO INICIAL DEL DATASET
# =====================================================

print("\n" + "="*60)
print("DIAGNÓSTICO DEL DATASET")
print("="*60)

print("\nFORMA DEL DATASET:")
print(df.shape)

print("\nRANGO DE FECHAS:")
print("Fecha mínima:", df["Fecha"].min())
print("Fecha máxima:", df["Fecha"].max())

print("\nTIPOS DE DATOS:")
print(df[X_columns].dtypes)

print("\nNULOS POR COLUMNA:")
print(df[X_columns].isnull().sum())

print("\nESTADÍSTICAS DEL TARGET (CantidadVendida):")
print(df["CantidadVendida"].describe())

print("\nDISTRIBUCIÓN DE VENTAS (QUANTILES)")
print(
    df["CantidadVendida"].quantile([
        0.90,
        0.95,
        0.99,
        0.995,
        0.999
    ])
)

print("\nPRODUCTOS ÚNICOS:")
print(df["ProductoID"].nunique())

print("\nSEDES ÚNICAS:")
print(df["SedeID"].nunique())

print("\nMUESTRA DE DATOS:")
print(df.head())

# =====================================================
# 🔥 VALIDACIÓN TEMPORAL 
# =====================================================
# Debemos cambiarlo entre 5 o 10 pliegues para que tenga una validacion robusta
tscv = TimeSeriesSplit(n_splits=3)

# =====================================================
# 🚀 OPTUNA OBJECTIVE
# =====================================================
def objective(trial):

    params = {
        "max_iter": trial.suggest_int("max_iter", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 80)
    }

    maes = []

    for train_idx, val_idx in tscv.split(X):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = HistGradientBoostingRegressor(**params, random_state=42)
        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        maes.append(mean_absolute_error(y_val, pred))

    return np.mean(maes)

# =====================================================
# 🔥 OPTIMIZACIÓN
# =====================================================
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)

print("\nMEJORES PARÁMETROS:")
print(study.best_params)

# =====================================================
# 🎯 MODELO FINAL
# =====================================================
model = HistGradientBoostingRegressor(
    **study.best_params,
    random_state=42
)

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =====================================================
# 🔍 DIAGNÓSTICO TRAIN / TEST
# =====================================================

print("\n" + "="*60)
print("DIAGNÓSTICO TRAIN / TEST")
print("="*60)

print("\nSHAPE TRAIN:")
print(X_train.shape)

print("\nSHAPE TEST:")
print(X_test.shape)

print("\nTARGET TRAIN:")
print(y_train.describe())

print("\nTARGET TEST:")
print(y_test.describe())

print("\nFECHAS TRAIN:")
print(
    df.iloc[:split]["Fecha"].min(),
    "->",
    df.iloc[:split]["Fecha"].max()
)

print("\nFECHAS TEST:")
print(
    df.iloc[split:]["Fecha"].min(),
    "->",
    df.iloc[split:]["Fecha"].max()
)

print("\nMEDIA TRAIN:")
print(y_train.mean())

print("\nMEDIA TEST:")
print(y_test.mean())

print("\nDESVIACIÓN TRAIN:")
print(y_train.std())

print("\nDESVIACIÓN TEST:")
print(y_test.std())

model.fit(X_train, y_train)

# Cuando se usa log-transform, inhabilitar esta línea y usar la siguiente para predecir
# pred = model.predict(X_test)
# Cuando se usa log-transform, hay que hacer expm1 para volver a la escala original
pred_log = model.predict(X_test)
pred = np.expm1(pred_log)
y_test_real = np.expm1(y_test)

# =====================================================
# 📊 EVALUACIÓN
# =====================================================
print("\nRESULTADOS")
print("-"*40)

# print("MAE :", mean_absolute_error(y_test, pred))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
# print("R2  :", r2_score(y_test, pred))

# Si usas log-transform, las métricas deben calcularse con y_test_real
print("MAE :", mean_absolute_error(y_test_real, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_real, pred)))
print("R2  :", r2_score(y_test_real, pred))

# =====================================================
# 🔥 IMPORTANCIA DE VARIABLES
# =====================================================
print("\nCalculando importancia de variables...")

perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

importancias = pd.DataFrame({
    "Variable": X_columns,
    "Importancia": perm.importances_mean
}).sort_values("Importancia", ascending=False)

print(importancias)

print("\n🔍 VARIABLES QUE EXPLICAN LAS VENTAS:")
print(importancias)

# =====================================================
# ❗ ANÁLISIS DE ERROR (DIAGNÓSTICO)
# =====================================================
results = X_test.copy()

# results["Real"] = y_test.values
# results["Predicho"] = pred
# Cuando se usa log-transform, usar y_test_real para el análisis de error
results["Real"] = y_test_real
results["Predicho"] = pred

results["Error"] = results["Real"] - results["Predicho"]
results["Error_Abs"] = np.abs(results["Error"])

print("\nANÁLISIS POR SEGMENTO DE DEMANDA")

for limite in [500, 1000, 5000, 10000]:
    
    subset = results[results["Real"] <= limite]

    mae = mean_absolute_error(
        subset["Real"],
        subset["Predicho"]
    )

    print(f"Ventas <= {limite:,.0f}")
    print(f"Registros: {len(subset):,}")
    print(f"MAE: {mae:.2f}")
    print("-"*40)

results["Error_Porcentual"] = np.where(
    results["Real"] == 0,
    0,
    (results["Error_Abs"] / results["Real"]) * 100
)

print("\nERROR PORCENTUAL PROMEDIO:")
print(results["Error_Porcentual"].mean())

print("\nERROR PORCENTUAL MEDIANO:")
print(results["Error_Porcentual"].median())

# =====================================================
# 🔥 MEJORAS EN DIAGNÓSTICO (COMENTARIOS CLAVE)
# =====================================================
# 🔍 ESTE BLOQUE TE AYUDA A RESPONDER:
# - ¿En qué productos falla el modelo?
# - ¿En qué rangos de ventas falla?
# - ¿Hay sesgo sistemático?
# - ¿Falla en promociones?

# 📊 VARIABLES QUE PODRÍAS AÑADIR PARA MEJOR DIAGNÓSTICO:
# results["Error_Porcentual"] = Error / Real
# results["CategoriaError"] = bajo/medio/alto error
# results["SegmentoDemanda"] = baja/media/alta demanda

# =====================================================
# 📉 BIAS DEL MODELO
# =====================================================
bias = results["Error"].mean()

print("\nBIAS DEL MODELO:")
print(bias)

if bias > 0:
    print("\nEl modelo tiende a SUBESTIMAR las ventas.")
elif bias < 0:
    print("\nEl modelo tiende a SOBREESTIMAR las ventas.")
else:
    print("\nEl modelo no presenta sesgo aparente.")

# =====================================================
# 🔥 ERRORES MÁS GRANDES
# =====================================================
print("\nTOP ERRORES:")
print(results.sort_values("Error_Abs", ascending=False).head(10))

# =====================================================
# 📦 ERROR POR PRODUCTO
# =====================================================
print("\nERROR POR PRODUCTO:")
print(
    results.groupby("ProductoID")["Error_Abs"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# =====================================================
# 📊 MEJORAS PARA VALIDACIÓN MÁS ROBUSTA
# =====================================================
# 🔴 Para mejorar validación podrías agregar:
# - Validación por producto (modelos separados)
# - Validación por sede (segmentación geográfica)
# - Backtesting por ventanas móviles
# - Validación por promociones (split por campañas)

# =====================================================
# 💾 GUARDAR MODELO
# =====================================================
joblib.dump(model, "modelo_produccion_comentado.pkl")

# =====================================================
# ⏱ TIEMPO
# =====================================================
print("\nTIEMPO TOTAL:")
print(time.time() - start_time)