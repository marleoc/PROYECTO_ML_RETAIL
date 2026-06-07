from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pyodbc

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt

import joblib


# 1. CONEXIÓN PROFESIONAL A SQL SERVER
server = 'localhost'
database = 'DB_RETAIL_ML'

connection_string = (
    f"mssql+pyodbc://@{server}/{database}"
    "?driver=ODBC Driver 18 for SQL Server"
    "&trusted_connection=yes"
    "&TrustServerCertificate=yes"
)

engine = create_engine(connection_string)

# print(pyodbc.drivers())

# 2. CARGA DE DATOS DESDE LA TABLA PRINCIPAL
df = pd.read_sql(
    "SELECT * FROM ml.dm_prediccion_demanda",
    engine
)
# 3. LIMPIEZA DE DATOS PROFESIONAL [5, 6]
df.fillna(0, inplace=True) # Rellenar nulos con 0
df.drop_duplicates(inplace=True) # Eliminar duplicados
df['Fecha'] = pd.to_datetime(df['Fecha']) # Asegurar formato de fecha

# 4. FEATURE ENGINEERING ADICIONAL EN PYTHON [7, 8]
df['Año'] = df['Fecha'].dt.year
df['Mes'] = df['Fecha'].dt.month
df['DiaSemana'] = df['Fecha'].dt.dayofweek
df['Trimestre'] = df['Fecha'].dt.quarter
df['EsFinSemana'] = np.where(df['DiaSemana'].isin([1, 9]), 1, 0)

# 5. SELECCIÓN DE VARIABLES (FEATURES Y TARGET) [10, 11]
# Target: Lo que queremos predecir
y = df['CantidadVendida'] 

# Features: Variables predictoras detectadas en el Data Mart
X_columns = [
    'ProductoID', 'CategoriaID', 'SedeID', 'PrecioPromedio', 
    'DiaSemana', 'Mes', 'Semana', 'EsFinSemana', 'EsFinMes', 
    'EsQuincena', 'Venta_Ayer', 'Venta_7_Dias', 'Venta_30_Dias', 
    'Promedio_7_Dias', 'Promedio_30_Dias', 'TienePromocion', 'PorcentajeDescuento'
]
X = df[X_columns]

# 6. DIVISIÓN TEMPORAL (TRAIN / TEST) [12, 13]
# Importante en retail: No usar split aleatorio, respetar la línea de tiempo
train = df[df['Fecha'] < '2026-01-01']
test = df[df['Fecha'] >= '2026-01-01']

X_train, y_train = train[X_columns], train['CantidadVendida']
X_test, y_test = test[X_columns], test['CantidadVendida']

# 7. ENTRENAMIENTO DEL MODELO RANDOM FOREST [14, 15]
modelo = RandomForestRegressor(
    n_estimators=300,   # Cantidad de árboles
    max_depth=20,       # Profundidad para capturar patrones complejos
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,    # Reproducibilidad académica
    n_jobs=-1           # Usar todos los núcleos del procesador
)
modelo.fit(X_train, y_train)

# 8. PREDICCIONES Y EVALUACIÓN [16, 17]
predicciones = modelo.predict(X_test)

print(f"MAE (Error Absoluto Medio): {mean_absolute_error(y_test, predicciones):.2f}")
print(f"RMSE (Raíz del Error Cuadrático): {np.sqrt(mean_squared_error(y_test, predicciones)):.2f}")
print(f"R² Score (Precisión General): {r2_score(y_test, predicciones):.2f}")

# 9. FEATURE IMPORTANCE (ORO GERENCIAL) [18, 19]
importancias = pd.DataFrame({
    'Variable': X_columns,
    'Importancia': modelo.feature_importances_
}).sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importancia', y='Variable', data=importancias)
plt.title('Variables que más influyen en la Demanda')
plt.show()

# 10. EXPORTACIÓN PARA PRODUCCIÓN Y POWER BI [20, 21]
joblib.dump(modelo, 'random_forest_retail_final.pkl')
print("Modelo exportado exitosamente como .pkl")
