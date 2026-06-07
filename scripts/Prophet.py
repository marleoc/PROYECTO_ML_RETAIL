from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import time
import joblib

from prophet import Prophet

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# TIEMPO INICIO
# =====================================================

start_time = time.time()

# =====================================================
# CONEXIÓN
# =====================================================

server = "localhost"
database = "DB_RETAIL_ML"

engine = create_engine(
    f"mssql+pyodbc://@{server}/{database}"
    "?driver=ODBC+Driver+18+for+SQL+Server"
    "&trusted_connection=yes"
    "&TrustServerCertificate=yes"
)

# =====================================================
# CONSULTA
# =====================================================

query = """
SELECT
    Fecha,
    SUM(CantidadVendida) AS CantidadVendida
FROM ml.dm_prediccion_demanda
WHERE ProductoID = 10
GROUP BY Fecha
ORDER BY Fecha
"""

print("Cargando datos...")

df = pd.read_sql(
    query,
    engine,
    parse_dates=["Fecha"]
)

# =====================================================
# AGRUPACIÓN DIARIA
# =====================================================

ventas_diarias = (
    df.groupby("Fecha")["CantidadVendida"]
      .sum()
      .reset_index()
)

# Formato requerido por Prophet

ventas_diarias.rename(
    columns={
        "Fecha": "ds",
        "CantidadVendida": "y"
    },
    inplace=True
)

print(f"Registros diarios: {len(ventas_diarias)}")

# =====================================================
# HOLDOUT TEMPORAL
# =====================================================

split = int(len(ventas_diarias) * 0.8)

train = ventas_diarias.iloc[:split]
test = ventas_diarias.iloc[split:]

print(f"Train: {len(train)}")
print(f"Test : {len(test)}")

# =====================================================
# MODELO PROPHET
# =====================================================

modelo = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative"
)

print("Entrenando Prophet...")

modelo.fit(train)

# =====================================================
# PREDICCIÓN
# =====================================================

future = test[["ds"]]

forecast = modelo.predict(future)

pred = forecast["yhat"].values

# =====================================================
# MÉTRICAS
# =====================================================

mae = mean_absolute_error(
    test["y"],
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        test["y"],
        pred
    )
)

r2 = r2_score(
    test["y"],
    pred
)

print("\nRESULTADOS")
print("-" * 40)
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# =====================================================
# GRÁFICO
# =====================================================

fig = modelo.plot(forecast)

# =====================================================
# GUARDAR MODELO
# =====================================================

joblib.dump(
    modelo,
    "PROPHET.pkl"
)

print("\nModelo guardado correctamente.")

# =====================================================
# TIEMPO TOTAL
# =====================================================

elapsed_time = time.time() - start_time

minutos = int(elapsed_time // 60)
segundos = elapsed_time % 60

print("\n" + "=" * 40)
print(
    f"TIEMPO TOTAL: "
    f"{minutos}m {segundos:.2f}s "
    f"[{elapsed_time:.2f}s]"
)
print("=" * 40)

# RESULTADOS
#----------------------------------------
# MAE : 1071.347857728728
# RMSE: 1298.4374504851867
# R2  : -0.8790653993853936
# Modelo guardado correctamente.
# TIEMPO TOTAL: 0m 0.85s [0.85s]
 