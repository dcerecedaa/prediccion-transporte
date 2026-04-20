import pandas as pd
from pathlib import Path

# Importamos las funciones de los módulos anteriores
from src.ingesta.cargador import cargar_datos_entrenamiento
from src.preprocesamiento.limpiador import limpiar

# Ruta donde guardaremos el resultado limpio
# Subimos dos niveles desde src/ hasta la raíz del proyecto
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Función para guardar un DataFrame limpio en formato CSV
def guardar_csv(df: pd.DataFrame, nombre: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Guardamos el DataFrame con un nombre descriptivo, sin índice y usando ';' como separador
    ruta = PROCESSED_DIR / f"{nombre}.csv"
    df.to_csv(ruta, index=False, sep=";")

    # Imprimimos un mensaje con el nombre del archivo guardado, su tamaño en MB y el número de filas
    size_mb = ruta.stat().st_size / 1_048_576
    print(f"Guardado: {ruta.name} ({size_mb:.2f} MB, {len(df)} filas)")

    return ruta

# Función principal que ejecuta todo el pipeline de ingesta y preprocesamiento
def ejecutar_pipeline() -> pd.DataFrame:
    print("Iniciando pipeline...")

    # Paso 1: carga y unión de todos los CSVs anuales 2019-2025
    print("\n[1/3] Cargando datos...")
    df = cargar_datos_entrenamiento()
    print(f"Filas cargadas: {len(df)}")

    # Paso 2: limpieza, validación y marcado de outliers y período COVID
    print("\n[2/3] Limpiando datos...")
    df_limpio = limpiar(df)

    # Paso 3: guardamos el resultado limpio en data/processed/
    print("\n[3/3] Guardando datos procesados...")
    guardar_csv(df_limpio, "emt_limpio")

    print("\nPipeline completado correctamente")
    return df_limpio


# Solo se ejecuta si lanzamos este archivo directamente
# nunca cuando se importa desde otro módulo
if __name__ == "__main__":
    ejecutar_pipeline()