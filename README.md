# PROYECTO_ML_RETAIL

Proyecto de Machine Learning para predicción de demanda y riesgo de quiebre en retail.

## 1. Librerías utilizadas

El proyecto usa las siguientes librerías principales:

- pandas: manipulación y análisis de datos tabulares.
- numpy: operaciones numéricas y transformaciones matemáticas.
- scikit-learn: modelos de regresión/clasificación, métricas y validación temporal.
- optuna: optimización automática de hiperparámetros.
- lightgbm: modelo LGBMRegressor (opcional en el entrenamiento).
- sqlalchemy: conexión y consulta a bases de datos SQL Server.
- pyodbc: acceso a SQL Server desde Python.
- joblib: carga y guardado de modelos entrenados.

### Instalación rápida

Ejecuta estos comandos desde la raíz del proyecto:

```bash
pip install pandas numpy scikit-learn optuna lightgbm sqlalchemy pyodbc joblib
```

Si tu entorno usa conda:

```bash
conda install pandas numpy scikit-learn optuna lightgbm sqlalchemy pyodbc joblib
```

> Nota: además debes tener instalado el driver ODBC de SQL Server (por ejemplo, ODBC Driver 18 for SQL Server) para que funcione la conexión en `pipeline/data_loader.py`.

## 2. Descripción de la carpeta pipeline

### `pipeline/config.py`
Define los parámetros globales del proyecto:
- lista de features utilizadas por el modelo,
- tamaño de chunks,
- semilla aleatoria,
- target principal (`CantidadVendida`),
- configuración de riesgo y uso de log-transform.

### `pipeline/data_loader.py`
Responsable de leer los datos desde SQL Server:
- `get_engine()` crea la conexión.
- `load_data()` carga el dataset completo.
- `load_data_chunks()` carga datos por bloques.
- `validate_data()` verifica calidad del dataset.
- `load_data_pipeline()` integra la carga y validación.

### `pipeline/features.py`
Genera ingeniería de features para mejorar el rendimiento:
- variables temporales (semana, mes, día de la semana, fin de mes, etc.),
- lags y rolling windows de ventas,
- features de precio y promoción,
- validación final del conjunto de features.

### `pipeline/trainer.py`
Contiene la lógica de entrenamiento del modelo:
- segmentación de demanda (`LOW`, `MID`, `HIGH`),
- entrenamiento de modelo de clasificación para demanda baja,
- entrenamiento de modelo de boosting para demanda media/alta,
- entrenamiento de modelo de clasificación para riesgo.

### `pipeline/evaluator.py`
Evalúa el desempeño del modelo:
- métricas de regresión (MAE, RMSE, R², bias),
- métricas de clasificación (precision, recall, ROC-AUC),
- análisis por segmento,
- detección de errores más grandes.

### `pipeline/inference.py`
Implementa el flujo de inferencia para producción:
- carga de modelos entrenados,
- carga de nuevos datos,
- preparación de features,
- generación de predicciones,
- exportación opcional de resultados a SQL Server.

## 3. Ejecución del proyecto

Desde la raíz del proyecto, puedes ejecutar:

```bash
python main.py
```

Esto cargará los datos, generará features, entrenará los modelos, evaluará resultados y guardará los artefactos en la carpeta `models/`.

## 4. Estructura principal

- `main.py`: pipeline completo de entrenamiento y evaluación.
- `pipeline/`: módulos reutilizables del proyecto.
- `scripts/`: experimentos y pruebas con distintos modelos.
