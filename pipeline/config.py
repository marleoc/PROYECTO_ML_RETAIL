FEATURES = [
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

CHUNK_SIZE = 500_000
USE_FLOAT32 = True
GROUP_KEYS = ["ProductoID", "SedeID"]
TARGET = "CantidadVendida"
RISK_QUANTILE = 0.90
RANDOM_STATE = 42
RISK_THRESHOLD = 0.70 # umbral de riesgo para clasificación para quiebres
USE_LOG_TARGET = True # cambiar a True si se quiere usar log-transform en el target