import pandas as pd
from pathlib import Path

# Ruta a los CSVs crudos — nunca se modifican, solo se leen
# __file__ es el archivo actual, parents[2] sube dos niveles para llegar a la raíz del proyecto 
# .resolve() devuelve siempre la ruta absoluta
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Función para cargar un CSV anual y unificar el formato de fecha
def cargar_csv_anual(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta, sep=";")

    # Convertimos la columna "Fecha" a datetime, manejando formatos mixtos
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="mixed", dayfirst=True)

    return df

# Función principal para cargar los datos de entrenamiento
# -> pd.DataFrame indica que esta función devuelve un DataFrame de pandas
def cargar_datos_entrenamiento() -> pd.DataFrame:

    # Seleccionamos solo los CSVs anuales, ignoramos el combinado original
    archivos = sorted(RAW_DIR.glob("demandadialinea_2[0-9]*.csv"))

    # Excluimos explícitamente 2026 del entrenamiento
    archivos_entrenamiento = [
        a for a in archivos if "2026" not in a.name
    ]

    if not archivos_entrenamiento:
        raise FileNotFoundError(
            f"No se encontraron CSVs de entrenamiento en {RAW_DIR}"
        )

    dfs = []
    for archivo in archivos_entrenamiento:
        df = cargar_csv_anual(archivo)
        # Añadimos el año de origen para trazabilidad, util para análisis posteriores
        df["anio_origen"] = int(archivo.stem.split("_")[-1])
        dfs.append(df)

    # Concatenamos todos los años en un único DataFrame
    combinado = pd.concat(dfs, ignore_index=True)

    # Ordenamos por línea y fecha, imprescindible para los lags temporales
    combinado = combinado.sort_values(["Linea", "Fecha"]).reset_index(drop=True)

    return combinado

# Función para cargar los datos de validación (2026)
def cargar_datos_validacion() -> pd.DataFrame:
    ruta = RAW_DIR / "demandadialinea_2026.csv"

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de validación en {ruta}")

    return cargar_csv_anual(ruta)