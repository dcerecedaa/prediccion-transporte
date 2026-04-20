import joblib
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from pathlib import Path
from src.caracteristicas.constructor import cargar_festivos
from src.ingesta.cargador import cargar_datos_validacion
from src.modelos.evaluador import calcular_metricas

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "data" / "models"

RUTA_HISTORICO = PROCESSED_DIR / "emt_limpio.csv"
RUTA_MODELO = MODELS_DIR / "lightgbm_emt.joblib"

# El modelo necesita al menos 28 dias previos para construir todos los lags y medias moviles
MIN_HISTORICO = 28

# Si el modelo no expone los nombres de features, usamos el orden conocido del entrenamiento actual
# Este fallback debe mantenerse sincronizado con el pipeline de entrenamiento
FEATURES_MODELO_POR_DEFECTO = [
    "Linea",
    "es_outlier",
    "es_covid",
    "dia_semana",
    "mes",
    "es_festivo",
    "es_dia_especial",
    "lag_7d",
    "lag_14d",
    "lag_28d",
    "media_movil_7d",
    "media_movil_14d",
]

# Modos de evaluacion:
MODO_ROLLING_REAL = "rolling_real" 
MODO_RECURSIVO = "recursivo"
MODOS_VALIDOS = {MODO_ROLLING_REAL, MODO_RECURSIVO}

# dataclass para almacenar los resultados de la validacion de forma estructurada
@dataclass
class ResultadoValidacion2026:
    nombre_modelo: str
    modo_evaluacion: str
    mae: float
    rmse: float
    r2: float
    wape: float
    acierto_10pct: float
    acierto_20pct: float
    registros_evaluados: int
    lineas_evaluadas: int
    fecha_inicio: str
    fecha_fin: str
    lineas_sin_historial: list[int]
    detalle: pd.DataFrame
    resumen_por_linea: pd.DataFrame
    ruta_detalle: Path | None = None
    ruta_resumen_por_linea: Path | None = None

# Función para cargar el modelo entrenado desde disco
def cargar_modelo_entrenado():
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            "No se encontro el modelo entrenado. Ejecuta primero src/modelos/entrenador.py"
        )

    # Devolvemos el modelo ya serializado para reutilizarlo directamente en la validacion
    return joblib.load(RUTA_MODELO)

# Función para cargar el histórico de entrenamiento, 
# aplicando la misma limpieza que en el pipeline de construcción de features
def cargar_historico_entrenamiento() -> pd.DataFrame:
    if not RUTA_HISTORICO.exists():
        raise FileNotFoundError(
            "No se encontro emt_limpio.csv. Ejecuta primero src/ingesta/pipeline.py"
        )

    df = pd.read_csv(RUTA_HISTORICO, sep=";", parse_dates=["Fecha"])

    # Mantenemos la misma exclusion que en la construccion de features
    return df[df["Linea"] != 868].sort_values(["Linea", "Fecha"]).reset_index(drop=True)

# Función para cargar el dataset de validación de 2026, 
# asegurando los tipos de datos correctos
def cargar_dataset_2026() -> pd.DataFrame:
    df = cargar_datos_validacion().copy()
    df["Linea"] = df["Linea"].astype("Int16")
    df["TotalViajeros"] = df["TotalViajeros"].astype("Int32")

    # El orden temporal es clave porque la validacion recorre cada linea dia a dia
    return df.sort_values(["Linea", "Fecha"]).reset_index(drop=True)

# Función para obtener el conjunto de fechas festivas en 2026
def obtener_festivos_2026() -> set[pd.Timestamp]:
    festivos = pd.to_datetime(cargar_festivos())

    # normalize() elimina la hora y deja solo la fecha para compararla correctamente
    return {pd.Timestamp(fecha).normalize() for fecha in festivos}

# Función para obtener los nombres de features que el modelo espera,
# con un fallback al orden conocido si el modelo no los expone
def obtener_features_modelo(modelo) -> list[str]:
    feature_names = getattr(modelo, "feature_name_", None)
    if feature_names is None:
        warnings.warn(
            "El modelo no expone 'feature_name_'. "
            "Se usará FEATURES_MODELO_POR_DEFECTO; revisa que siga alineado con el entrenamiento.",
            stacklevel=2,
        )
        return FEATURES_MODELO_POR_DEFECTO.copy()

    # Si el modelo expone sus columnas, usamos exactamente ese orden para evitar desajustes
    return [str(columna) for columna in feature_names]

# Función para extraer el historial de viajeros de una línea dada, ordenado por fecha
def extraer_historial_linea(df_linea: pd.DataFrame) -> list[float]:
    historial = (
        df_linea.sort_values("Fecha")["TotalViajeros"]
        .astype(float)
        .tolist()
    )

    if len(historial) < MIN_HISTORICO:
        raise ValueError(
            f"Se necesitan al menos {MIN_HISTORICO} observaciones para construir las features"
        )

    # Devolvemos una lista porque iremos ampliando este historico dentro del bucle de validacion
    return historial

# Función para construir el DataFrame de features para una fecha objetivo dada,
# utilizando el historial de viajeros y la información de festivos
def construir_features_para_fecha(
    linea: int,
    fecha_objetivo: pd.Timestamp,
    historial_viajeros: list[float],
    festivos: set[pd.Timestamp],
    columnas_modelo: list[str],
) -> pd.DataFrame:
    if len(historial_viajeros) < MIN_HISTORICO:
        raise ValueError(
            f"Historial insuficiente para la linea {linea}: "
            f"se requieren {MIN_HISTORICO} observaciones previas"
        )

    fecha_objetivo = pd.Timestamp(fecha_objetivo)
    es_festivo = int(fecha_objetivo.normalize() in festivos)
    es_fin_semana = int(fecha_objetivo.dayofweek in [5, 6])

    # Reunimos aqui todas las variables que el modelo necesita para una fecha concreta
    base_features = {
        "Linea": int(linea),
        # En inferencia real no sabemos si el dia futuro sera outlier
        "es_outlier": 0,
        "es_covid": int(fecha_objetivo.year in [2020, 2021]),
        "dia_semana": int(fecha_objetivo.dayofweek),
        "mes": int(fecha_objetivo.month),
        "es_fin_semana": es_fin_semana,
        "es_festivo": es_festivo,
        "es_dia_especial": int(es_fin_semana == 1 or es_festivo == 1),
        "lag_7d": float(historial_viajeros[-7]),
        "lag_14d": float(historial_viajeros[-14]),
        "lag_28d": float(historial_viajeros[-28]),
        "media_movil_7d": round(float(np.mean(historial_viajeros[-7:])), 4),
        "media_movil_14d": round(float(np.mean(historial_viajeros[-14:])), 4),
    }

    # Construimos la fila final respetando el orden exacto que el modelo espera al predecir
    fila = {columna: base_features.get(columna, 0) for columna in columnas_modelo}
    return pd.DataFrame([fila], columns=columnas_modelo)

# Función principal para generar el detalle de validación, iterando por cada línea y fecha,
# construyendo las features, obteniendo las predicciones y calculando los errores
def generar_detalle_validacion_2026(
    modelo,
    df_historico: pd.DataFrame,
    df_2026: pd.DataFrame,
    modo: str = MODO_ROLLING_REAL,
) -> tuple[pd.DataFrame, list[int]]:
    if modo not in MODOS_VALIDOS:
        raise ValueError(f"Modo no valido: {modo}. Usa uno de {sorted(MODOS_VALIDOS)}")

    festivos = obtener_festivos_2026()
    columnas_modelo = obtener_features_modelo(modelo)
    detalle = []

    # Esta lista nos permite informar luego de las lineas que no se han podido evaluar
    lineas_sin_historial = []

    # Bucle para: cada línea en el dataset de 2026, extraer su historial, construir features para cada fecha,
    # predecir, calcular errores y actualizar el historial según el modo de evaluación
    for linea, df_2026_linea in df_2026.groupby("Linea"):
        df_historico_linea = df_historico[df_historico["Linea"] == linea]
        if len(df_historico_linea) < MIN_HISTORICO:
            lineas_sin_historial.append(int(linea))
            continue

        historial = extraer_historial_linea(df_historico_linea)
        df_2026_linea = df_2026_linea.sort_values("Fecha").reset_index(drop=True)

        # Bucle para cada fecha de la línea, construir features, predecir y calcular errores
        for fila in df_2026_linea.itertuples(index=False):
            features = construir_features_para_fecha(
                linea=int(linea),
                fecha_objetivo=fila.Fecha,
                historial_viajeros=historial,
                festivos=festivos,
                columnas_modelo=columnas_modelo,
            )

            prediccion = int(round(float(modelo.predict(features)[0])))

            # La demanda nunca puede ser negativa, por eso limitamos el minimo a cero
            prediccion = max(0, prediccion)

            real = int(fila.TotalViajeros)
            error_absoluto = abs(real - prediccion)

            # Este error relativo permite medir si el fallo es pequeño o grande respecto al valor real
            error_porcentual = error_absoluto / real if real > 0 else np.nan

            # Agregamos la fila al detalle de validación
            detalle.append(
                {
                    "Fecha": pd.Timestamp(fila.Fecha).strftime("%Y-%m-%d"),
                    "Linea": int(linea),
                    "TotalViajeros_real": real,
                    "TotalViajeros_predicho": prediccion,
                    "error_absoluto": error_absoluto,
                    "error_porcentual": round(float(error_porcentual), 4)
                    # Si no hay valor real, el error porcentual no es aplicable, 
                    # lo dejamos como NaN para no distorsionar métricas de acierto porcentual
                    if not np.isnan(error_porcentual)
                    else np.nan,
                }
            )

            # Si modo es rolling_real, actualizamos el historial con el valor real para la siguiente fecha
            if modo == MODO_ROLLING_REAL:
                historial.append(real)
            else:
                # En modo recursivo usamos la propia prediccion como entrada para la fecha siguiente
                historial.append(prediccion)

    # Convertimos el detalle a DataFrame y ordenamos por fecha y línea
    return pd.DataFrame(detalle), sorted(lineas_sin_historial)

# Función para resumir los resultados por línea, calculando métricas de error y demanda media para cada línea
def resumir_por_linea(detalle: pd.DataFrame) -> pd.DataFrame:
    filas = []

    # Bucle para cada línea en el detalle, calcular métricas de error y demanda media, y agregar al resumen
    for linea, grupo in detalle.groupby("Linea"):
        y_real = grupo["TotalViajeros_real"].to_numpy()
        y_pred = grupo["TotalViajeros_predicho"].to_numpy()

        # Reutilizamos las metricas globales para obtener el error de cada linea por separado
        metricas = calcular_metricas(y_real, y_pred)

        filas.append(
            {
                "Linea": int(linea),
                "registros": int(len(grupo)),
                "mae": metricas["mae"],
                "rmse": metricas["rmse"],
                "r2": metricas["r2"],
                "demanda_media_real": round(float(grupo["TotalViajeros_real"].mean()), 2),
                "demanda_media_predicha": round(
                    float(grupo["TotalViajeros_predicho"].mean()), 2
                ),
            }
        )

    # Dejamos arriba las lineas con mayor MAE para detectar mas rapido donde falla mas el modelo
    return pd.DataFrame(filas).sort_values("mae", ascending=False).reset_index(drop=True)

# Función para guardar los resultados de la validación en archivos CSV, creando el directorio si no existe
def guardar_resultados(
    detalle: pd.DataFrame,
    resumen_por_linea: pd.DataFrame,
    modo: str,
) -> tuple[Path, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # El modo se incorpora al nombre del archivo para distinguir las dos estrategias de validacion
    ruta_detalle = PROCESSED_DIR / f"validacion_2026_{modo}_detalle.csv"
    ruta_resumen = PROCESSED_DIR / f"validacion_2026_{modo}_por_linea.csv"

    detalle.to_csv(ruta_detalle, index=False, sep=";")
    resumen_por_linea.to_csv(ruta_resumen, index=False, sep=";")

    return ruta_detalle, ruta_resumen

# Función principal para evaluar el modelo con los datos reales de 2026, generando el detalle de validación
def evaluar_modelo_2026(
    modo: str = MODO_ROLLING_REAL,
    guardar_csv: bool = False,
) -> ResultadoValidacion2026:
    modelo = cargar_modelo_entrenado()
    df_historico = cargar_historico_entrenamiento()
    df_2026 = cargar_dataset_2026()

    detalle, lineas_sin_historial = generar_detalle_validacion_2026(
        modelo=modelo,
        df_historico=df_historico,
        df_2026=df_2026,
        modo=modo,
    )

    if detalle.empty:
        raise ValueError("No se pudieron generar predicciones para el dataset de 2026")

    # Convertimos las columnas principales a arrays para calcular metricas sobre todo el periodo
    y_real = detalle["TotalViajeros_real"].to_numpy()
    y_pred = detalle["TotalViajeros_predicho"].to_numpy()
    metricas = calcular_metricas(y_real, y_pred)

    # WAPE reparte el error absoluto total sobre la demanda total real,
    # por eso se interpreta como un porcentaje global de error del modelo.
    error_total = np.abs(y_real - y_pred).sum()
    demanda_total = y_real.sum()
    wape = round(float(error_total / demanda_total), 4) if demanda_total > 0 else 0.0

    # Estas dos variables miden el porcentaje de predicciones cuyo error cae dentro de margenes razonables
    errores_pct = detalle["error_porcentual"].to_numpy(dtype=float)
    acierto_10pct = round(float(np.nanmean(errores_pct <= 0.10)), 4)
    acierto_20pct = round(float(np.nanmean(errores_pct <= 0.20)), 4)

    resumen_por_linea = resumir_por_linea(detalle)

    ruta_detalle = None
    ruta_resumen = None
    if guardar_csv:
        # Guardamos solo si se solicita para poder usar la funcion tambien desde notebooks o pruebas
        ruta_detalle, ruta_resumen = guardar_resultados(detalle, resumen_por_linea, modo)

    # Devolvemos tanto las metricas agregadas como los DataFrames,
    # para poder inspeccionar despues el detalle sin recalcular toda la validacion
    return ResultadoValidacion2026(
        nombre_modelo=type(modelo).__name__,
        modo_evaluacion=modo,
        mae=float(metricas["mae"]),
        rmse=float(metricas["rmse"]),
        r2=float(metricas["r2"]),
        wape=float(wape),
        acierto_10pct=float(acierto_10pct),
        acierto_20pct=float(acierto_20pct),
        registros_evaluados=int(len(detalle)),
        lineas_evaluadas=int(detalle["Linea"].nunique()),
        fecha_inicio=str(detalle["Fecha"].min()),
        fecha_fin=str(detalle["Fecha"].max()),
        lineas_sin_historial=lineas_sin_historial,
        detalle=detalle,
        resumen_por_linea=resumen_por_linea,
        ruta_detalle=ruta_detalle,
        ruta_resumen_por_linea=ruta_resumen,
    )


# Función para mostrar un resumen compacto por consola,
# util cuando queremos revisar rapido el rendimiento del modelo
def imprimir_resumen_validacion(resultado: ResultadoValidacion2026) -> None:
    print("\n" + "=" * 50)
    print("VALIDACION CON DATOS REALES DE 2026")
    print("=" * 50)
    print(f"Modelo: {resultado.nombre_modelo}")
    print(f"Modo de evaluacion: {resultado.modo_evaluacion}")
    print(f"Periodo evaluado: {resultado.fecha_inicio} a {resultado.fecha_fin}")
    print(f"Lineas evaluadas: {resultado.lineas_evaluadas}")
    print(f"Registros evaluados: {resultado.registros_evaluados}")
    print(f"MAE: {resultado.mae:,.2f}")
    print(f"RMSE: {resultado.rmse:,.2f}")
    print(f"R2: {resultado.r2:.4f}")
    print(f"WAPE: {resultado.wape:.2%}")
    print(f"Acierto dentro de +/-10%: {resultado.acierto_10pct:.2%}")
    print(f"Acierto dentro de +/-20%: {resultado.acierto_20pct:.2%}")
    print(f"Lineas sin historico suficiente: {len(resultado.lineas_sin_historial)}")

    if resultado.ruta_detalle is not None and resultado.ruta_resumen_por_linea is not None:
        print(f"Detalle guardado en: {resultado.ruta_detalle}")
        print(f"Resumen por linea guardado en: {resultado.ruta_resumen_por_linea}")


if __name__ == "__main__":
    resultado = evaluar_modelo_2026(modo=MODO_ROLLING_REAL, guardar_csv=False)
    imprimir_resumen_validacion(resultado)
