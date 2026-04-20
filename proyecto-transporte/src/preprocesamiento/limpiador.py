import pandas as pd
import numpy as np
from pathlib import Path

# Este módulo se encarga de limpiar y preparar los datos para el análisis.
def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    n_antes = len(df)

    # Un duplicado real sería la misma línea el mismo día registrada dos veces
    df = df.drop_duplicates(subset=["Fecha", "Linea"])

    # Informamos cuántos duplicados se eliminaron
    n_eliminados = n_antes - len(df)
    if n_eliminados > 0:
        print(f"Duplicados eliminados: {n_eliminados}")

    return df

# Verificar tipos de datos y corregirlos si es necesario
def verificar_tipos(df: pd.DataFrame) -> pd.DataFrame:

    # Fecha debe ser datetime, si no lo es significa que el cargador falló
    if not pd.api.types.is_datetime64_any_dtype(df["Fecha"]):
        raise TypeError("La columna Fecha no es datetime — revisa el cargador")

    # Usamos Int16 para Linea, suficiente para los números de línea de la EMT
    # y más eficiente en memoria que int64
    df["Linea"] = df["Linea"].astype("Int16")

    # Int32 es suficiente para TotalViajeros
    df["TotalViajeros"] = df["TotalViajeros"].astype("Int32")

    return df

# Funcion para marcar outliers 
def marcar_outliers(df: pd.DataFrame) -> pd.DataFrame:

    # Funcion para marcar outliers dentro de cada grupo de línea usando el método IQR
    def marcar_grupo(grupo: pd.DataFrame) -> pd.DataFrame:
        Q1 = grupo["TotalViajeros"].quantile(0.25)
        Q3 = grupo["TotalViajeros"].quantile(0.75)
        IQR = Q3 - Q1

        # Usamos 3x IQR en lugar del estándar 1.5x para ser más permisivos
        # y no marcar como outlier demanda legítimamente alta o baja
        limite_inf = Q1 - 3 * IQR
        limite_sup = Q3 + 3 * IQR

        grupo["es_outlier"] = (
            (grupo["TotalViajeros"] < limite_inf) |
            (grupo["TotalViajeros"] > limite_sup)
        )
        return grupo

    df = df.groupby("Linea", group_keys=False).apply(marcar_grupo)

    n_outliers = df["es_outlier"].sum()
    pct = n_outliers / len(df) * 100
    print(f"Outliers detectados: {n_outliers} ({pct:.2f}%)")

    return df

# Funcion para marcar el periodo COVID
def marcar_periodo_covid(df: pd.DataFrame) -> pd.DataFrame:

    df["es_covid"] = df["Fecha"].dt.year.isin([2020, 2021])

    n_covid = df["es_covid"].sum()
    print(f"Registros marcados como COVID: {n_covid}")

    return df

# Funcion para validar el resultado final antes de guardar
def validar_resultado(df: pd.DataFrame) -> bool:

    errores = []

    # Sin nulos en columnas clave
    nulos = df[["Fecha", "Linea", "TotalViajeros"]].isnull().sum()
    if nulos.any():
        errores.append(f"Nulos encontrados: {nulos[nulos > 0].to_dict()}")

    # Sin duplicados
    dups = df.duplicated(subset=["Fecha", "Linea"]).sum()
    if dups > 0:
        errores.append(f"Duplicados encontrados: {dups}")

    # Sin valores negativos en TotalViajeros
    negativos = (df["TotalViajeros"] < 0).sum()
    if negativos > 0:
        errores.append(f"Valores negativos encontrados: {negativos}")

    # El rango de fechas debe cubrir 2019-2025
    anio_min = df["Fecha"].dt.year.min()
    anio_max = df["Fecha"].dt.year.max()
    if anio_min > 2019 or anio_max < 2025:
        errores.append(f"Rango de años incorrecto: {anio_min} - {anio_max}")

    if errores:
        for error in errores:
            print(f"ERROR: {error}")
        return False

    print("Validación correcta — datos listos")
    return True

# Funcion principal de limpieza
def limpiar(df: pd.DataFrame) -> pd.DataFrame:
 
    df = eliminar_duplicados(df)
    df = verificar_tipos(df)
    df = marcar_outliers(df)
    df = marcar_periodo_covid(df)

    # Reordenamos por Linea y Fecha, imprescindible para los lags temporales
    df = df.sort_values(["Linea", "Fecha"]).reset_index(drop=True)

    validar_resultado(df)

    return df