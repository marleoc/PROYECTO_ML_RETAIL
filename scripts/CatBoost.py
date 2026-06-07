from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import gc
import time
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Usando Python 3.14.5 y scikit-learn 1.3.2 para optimización de memoria y rendimiento
# Este script está diseñado para ser eficiente en memoria y rápido, utilizando técnicas como:
# - Carga de datos por chunks para evitar sobrecargar la memoria RAM
# - Uso de HistGradientBoostingRegressor, que es más eficiente que RandomForest para grandes datasets
# - Eliminación de variables intermedias y uso de gc.collect() para liberar memoria después de cada chunk
# - Validación simple con holdout temporal para evaluar el modelo sin necesidad de cross-validation, que puede ser costosa en memoria
# - Guardado del modelo con joblib, que es más eficiente para objetos grandes que pickle
# Asegúrate de tener suficiente RAM disponible y de que tu entorno de Python esté configurado correctamente para ejecutar este script sin problemas de memoria.

# Registro del tiempo de inicio
start_time = time.time()

# =====================================================
# 1. CONEXIÓN
# =====================================================
print("Iniciando script...")
server = "localhost"
database = "DB_RETAIL_ML"

engine = create_engine(
    f"mssql+pyodbc://@{server}/{database}"
    "?driver=ODBC+Driver+18+for+SQL+Server"
    "&trusted_connection=yes"
    "&TrustServerCertificate=yes"
)

# =====================================================
# 2. QUERY (FILTRADO INTELIGENTE)
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
"""

# =====================================================
# 3. FEATURE COLUMNS
# =====================================================
X_columns = [
    "ProductoID", "SedeID", "PrecioPromedio",
    "Semana", "EsFinMes", "EsQuincena",
    "Venta_Ayer", "Venta_7_Dias", "Venta_30_Dias",
    "Promedio_7_Dias", "Promedio_30_Dias",
    "TienePromocion"
]

# =====================================================
# 4. MODELO (ligero y eficiente)
# =====================================================
modelo = CatBoostRegressor(
    iterations=300,
    depth=8,
    learning_rate=0.05,
    verbose=False
)

# =====================================================
# 5. ENTRENAMIENTO POR CHUNKS
# =====================================================
chunksize = 200_000

X_train_list = []
y_train_list = []

print("Cargando datos por chunks...")

for i, chunk in enumerate(pd.read_sql(query, engine, chunksize=chunksize)):

    print(f"Chunk {i+1}: {len(chunk):,} filas")

    chunk = chunk.dropna()

    # separar features y target
    X_train_list.append(chunk[X_columns])
    y_train_list.append(chunk["CantidadVendida"])

    del chunk
    gc.collect()

# unir SOLO al final (mucho más seguro)
X_train = pd.concat(X_train_list, ignore_index=True)
y_train = pd.concat(y_train_list, ignore_index=True)

del X_train_list, y_train_list
gc.collect()

print("Entrenando modelo...")

modelo.fit(X_train, y_train)

# =====================================================
# 6. VALIDACIÓN SIMPLE (holdout temporal)
# =====================================================
fecha_corte = pd.Timestamp("2026-01-01")

mask = X_train.index < len(X_train) * 0.8

X_test = X_train.loc[~mask]
y_test = y_train.loc[~mask]

pred = modelo.predict(X_test)

print("\nRESULTADOS")
print("-"*40)
print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2  :", r2_score(y_test, pred))

# =====================================================
# 7. GUARDAR MODELO
# =====================================================
joblib.dump(modelo, "modelo_demanda.pkl")

print("\nModelo guardado correctamente.")

# =====================================================
# 8. TIEMPO TOTAL DE EJECUCIÓN
# =====================================================
end_time = time.time()
elapsed_time = end_time - start_time

minutos = int(elapsed_time // 60)
segundos = elapsed_time % 60

print("\n" + "="*40)
print(f"TIEMPO TOTAL: {minutos}m {segundos:.2f}s [{elapsed_time:.2f}s]")
print("="*40)

# RESULTADOS obtenidos con este script:
# MAE : 36.520703478475916
# RMSE: 114.13981068573972
# R2  : 0.8896900239128342
# ========================================
# TIEMPO TOTAL: 7m 9.68s [429.68s]
# ========================================