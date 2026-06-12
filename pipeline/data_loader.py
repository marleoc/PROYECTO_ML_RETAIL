import pandas as pd
from sqlalchemy import create_engine
import gc

def get_engine():

    engine = create_engine(
        "mssql+pyodbc://@localhost/DB_RETAIL_ML"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&trusted_connection=yes"
        "&TrustServerCertificate=yes"
    )

    return engine

def load_data(start_date="2024-01-01"):
    """
    Carga datos desde SQL Server para entrenamiento ML.
    """

    engine = get_engine()

    query = f"""
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
    WHERE Fecha >= '{start_date}'
    ORDER BY Fecha
    """

    print("📥 Ejecutando query en SQL Server...")

    df = pd.read_sql(query, engine)

    print(f"📊 Datos crudos cargados: {df.shape}")

    return df

def load_data_chunks(start_date="2024-01-01", chunksize=200_000):
    """
    Carga datos en chunks para datasets grandes.
    """

    engine = get_engine()

    query = f"""
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
    WHERE Fecha >= '{start_date}'
    ORDER BY Fecha
    """

    print("📥 Cargando datos en chunks...")

    chunks = pd.read_sql(query, engine, chunksize=chunksize)

    df_list = []

    for chunk in chunks:
        chunk = chunk.dropna()
        df_list.append(chunk)
        del chunk
        gc.collect()

    df = pd.concat(df_list, ignore_index=True)

    print(f"📊 Dataset final: {df.shape}")

    return df

def validate_data(df):
    """
    Validación básica de calidad de datos.
    """

    print("\n🔍 VALIDANDO DATASET...")

    # nulos
    nulls = df.isnull().sum()
    print("\n📉 Nulos por columna:")
    print(nulls[nulls > 0])

    # duplicados
    duplicates = df.duplicated().sum()
    print(f"\n🔁 Filas duplicadas: {duplicates}")

    # fechas
    print("\n📅 Rango de fechas:")
    print(df["Fecha"].min(), "→", df["Fecha"].max())

    # sanity check
    print("\n📊 Shape final:")
    print(df.shape)

    return df

def load_data_pipeline(start_date="2024-01-01", use_chunks=False):

    if use_chunks:
        df = load_data_chunks(start_date=start_date)
    else:
        df = load_data(start_date=start_date)

    df = validate_data(df)

    return df