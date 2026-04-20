import yaml
import pandas as pd
from pathlib import Path

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parents[2] # subimos dos niveles para llegar a la raíz del proyecto
PROCESSED_DIR = BASE_DIR / "data" / "processed" 
CONFIGS_DIR = BASE_DIR / "configs" # carpeta para archivos de configuración como el YAML de festivos
RUTA_ENTRADA = PROCESSED_DIR / "emt_limpio.csv"
RUTA_SALIDA = PROCESSED_DIR / "emt_features.csv"

# Cargamos la lista de festivos desde un archivo YAML para centralizar la configuración
def cargar_festivos() -> pd.DatetimeIndex:
    # El archivo settings.yaml tiene una sección "festivos" con una lista de fechas en formato "YYYY-MM-DD"
    with open(CONFIGS_DIR / "settings.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return pd.to_datetime(config["festivos"])

# Funciones para construir features a partir del CSV limpio
def añadir_variables_temporales(df: pd.DataFrame, festivos: pd.DatetimeIndex) -> pd.DataFrame:
    # 0=lunes, 6=domingo
    df["dia_semana"] = df["Fecha"].dt.dayofweek
    df["mes"] = df["Fecha"].dt.month

    # Los fines de semana tienen patrones de demanda muy distintos a los laborables
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6]).astype("int8")

    # Los festivos se comportan de forma similar a los domingos en términos de demanda
    df["es_festivo"] = df["Fecha"].isin(festivos).astype("int8")

    # Combinamos fin de semana y festivo en una sola variable
    df["es_dia_especial"] = (
        (df["es_fin_semana"] == 1) | (df["es_festivo"] == 1)
    ).astype("int8")

    return df

# Función para añadir lags temporales. Los lags capturan la dependencia temporal en la serie de viajeros.
def añadir_lags(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [7, 14, 28]:
        # shift(lag) desplaza los valores lag días hacia adelante dentro de cada línea
        # groupby asegura que el lag no cruza entre líneas distintas
        df[f"lag_{lag}d"] = df.groupby("Linea")["TotalViajeros"].shift(lag)

    return df

# Función para añadir medias móviles. Las medias móviles suavizan la serie temporal y capturan tendencias recientes.
def añadir_medias_moviles(df: pd.DataFrame) -> pd.DataFrame:
    for ventana in [7, 14]:
        df[f"media_movil_{ventana}d"] = (
            df.groupby("Linea")["TotalViajeros"]
            .transform(lambda s: s.shift(1).rolling(ventana, min_periods=ventana).mean())
        )

    return df

# Función para eliminar filas que no tienen suficiente histórico para calcular lags y medias móviles.
def eliminar_filas_sin_historico(df: pd.DataFrame) -> pd.DataFrame:
    cols_features = [
        "lag_7d", "lag_14d", "lag_28d",
        "media_movil_7d", "media_movil_14d"
    ]

    n_antes = len(df)
    df = df.dropna(subset=cols_features).reset_index(drop=True)
    n_eliminadas = n_antes - len(df)

    print(f"Filas eliminadas por falta de histórico: {n_eliminadas}")
    print(f"Filas restantes: {len(df)}")

    return df

# Función principal que orquesta la construcción de features. 
# Lee el CSV limpio y genera el dataset final listo para el modelo.
def construir_features() -> pd.DataFrame:
    print("Construyendo features...")

    festivos = cargar_festivos()
    df = pd.read_csv(RUTA_ENTRADA, sep=";", parse_dates=["Fecha"])

    # Excluimos la línea 868, existió solo de octubre a diciembre de 2025
    # Es una línea temporal sin histórico suficiente que distorsiona el modelo
    n_antes = len(df)
    df = df[df["Linea"] != 868]
    print(f"Registros eliminados (línea 868): {n_antes - len(df)}")

    print("\n[1/4] Añadiendo variables temporales...")
    df = añadir_variables_temporales(df, festivos)

    print("[2/4] Añadiendo lags temporales...")
    df = añadir_lags(df)

    print("[3/4] Añadiendo medias móviles...")
    df = añadir_medias_moviles(df)

    print("[4/4] Eliminando filas sin histórico suficiente...")
    df = eliminar_filas_sin_historico(df)

    cols_redondear = ["media_movil_7d", "media_movil_14d"]
    df[cols_redondear] = df[cols_redondear].round(4)

    df.to_csv(RUTA_SALIDA, index=False, sep=";")
    size_mb = RUTA_SALIDA.stat().st_size / 1_048_576
    print(f"\nGuardado: {RUTA_SALIDA.name} ({size_mb:.2f} MB, {len(df)} filas)")

    return df

# Solo se ejecuta si lanzamos este archivo directamente
if __name__ == "__main__":
    construir_features()