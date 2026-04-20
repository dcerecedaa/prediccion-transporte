import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.caracteristicas.constructor import cargar_festivos

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "data" / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIGS_DIR = BASE_DIR / "configs"

# Número mínimo de días de histórico necesarios para construir los lags
MIN_HISTORICO = 28


# Cargamos la configuración de umbrales desde el YAML
def cargar_config() -> dict:
    with open(CONFIGS_DIR / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Cargamos el modelo serializado al arrancar la API
def cargar_modelo():
    ruta = MODELS_DIR / "lightgbm_emt.joblib"
    if not ruta.exists():
        raise FileNotFoundError(
            "No se encontró el modelo. Ejecuta primero src/modelos/entrenador.py"
        )
    return joblib.load(ruta)


# Cargamos el histórico completo combinando datos de entrenamiento y 2026
# de esta forma los lags siempre se calculan sobre datos reales
def cargar_historico_completo() -> pd.DataFrame:
    df = pd.read_csv(
        PROCESSED_DIR / "emt_limpio.csv",
        sep=";",
        parse_dates=["Fecha"]
    )

    # Excluimos la línea 868 — línea temporal que distorsiona el modelo
    df = df[df["Linea"] != 868]

    return df.sort_values(["Linea", "Fecha"]).reset_index(drop=True)


# Calculamos los máximos históricos por línea para estimar el riesgo de saturación
def cargar_maximos_historicos(df: pd.DataFrame) -> dict:
    return df.groupby("Linea")["TotalViajeros"].max().to_dict()

# Construye las features para un día concreto a partir del histórico de viajeros
# usa exactamente la misma lógica que validacion_2026.py para garantizar consistencia
def construir_features(
    linea: int,
    fecha: pd.Timestamp,
    historial: list[float],
    festivos: set,
    columnas_modelo: list[str]
) -> pd.DataFrame:
    es_festivo = int(fecha.normalize() in festivos)
    es_fin_semana = int(fecha.dayofweek in [5, 6])

    base = {
        "Linea": int(linea),
        "es_outlier": 0,
        "es_covid": 0,
        "dia_semana": int(fecha.dayofweek),
        "mes": int(fecha.month),
        "es_festivo": es_festivo,
        "es_dia_especial": int(es_festivo == 1 or es_fin_semana == 1),
        "lag_7d": float(historial[-7]),
        "lag_14d": float(historial[-14]),
        "lag_28d": float(historial[-28]),
        "media_movil_7d": round(float(np.mean(historial[-7:])), 4),
        "media_movil_14d": round(float(np.mean(historial[-14:])), 4),
    }

    fila = {col: base.get(col, 0) for col in columnas_modelo}
    return pd.DataFrame([fila], columns=columnas_modelo)


# Calcula el nivel de riesgo comparando la predicción con el máximo histórico
def calcular_riesgo(prediccion: float, maximo: float, config: dict) -> str:
    if maximo == 0:
        return "desconocido"
    pct = prediccion / maximo
    if pct >= config["umbral_riesgo_alto"]:
        return "alto"
    elif pct >= config["umbral_riesgo_medio"]:
        return "medio"
    return "bajo"


# Inicializamos la aplicación y cargamos todo al arrancar
app = FastAPI(
    title="API de Predicción de Demanda — EMT Madrid",
    description="Predice la demanda diaria por línea de autobús y estima el riesgo de saturación",
    version="2.0.0"
)

modelo = cargar_modelo()
config = cargar_config()
df_historico = cargar_historico_completo()
maximos_historicos = cargar_maximos_historicos(df_historico)

# Obtenemos las columnas que espera el modelo para construir las features en el orden correcto
columnas_modelo = [str(c) for c in getattr(modelo, "feature_name_", [
    "Linea", "es_outlier", "es_covid", "dia_semana", "mes",
    "es_festivo", "es_dia_especial", "lag_7d", "lag_14d", "lag_28d",
    "media_movil_7d", "media_movil_14d"
])]

# Cargamos los festivos como conjunto de fechas normalizadas para búsqueda rápida
festivos = set(pd.to_datetime(cargar_festivos()).normalize())
# Añadimos festivos de 2026 que pueden no estar en el YAML todavía
festivos.update(pd.to_datetime(["2026-01-01", "2026-01-06"]).normalize())


# Schemas de respuesta — Pydantic valida y documenta los tipos automáticamente
class PrediccionDia(BaseModel):
    fecha: str
    linea: int
    viajeros_predichos: int
    riesgo: str
    porcentaje_maximo_historico: float


class RespuestaPrediccion(BaseModel):
    linea: int
    predicciones: list[PrediccionDia]


# Endpoint de salud — verifica que la API está operativa
@app.get("/salud")
def salud():
    return {
        "estado": "operativa",
        "lineas_disponibles": len(maximos_historicos),
        "modelo": "LightGBM",
        "version": "2.0.0"
    }


# Endpoint principal — predice la demanda de los próximos días para una línea
@app.get("/predecir/{linea}", response_model=RespuestaPrediccion)
def predecir(linea: int, dias: int = 7):
    if linea not in maximos_historicos:
        raise HTTPException(
            status_code=404,
            detail=f"La línea {linea} no existe en los datos históricos"
        )

    if dias < 1 or dias > 14:
        raise HTTPException(
            status_code=400,
            detail="El número de días debe estar entre 1 y 14"
        )

    # Extraemos el histórico real de esa línea — incluye datos de 2026 si existen
    df_linea = df_historico[df_historico["Linea"] == linea].sort_values("Fecha")

    if len(df_linea) < MIN_HISTORICO:
        raise HTTPException(
            status_code=422,
            detail=f"La línea {linea} no tiene histórico suficiente para predecir"
        )

    # Construimos el historial como lista de floats para calcular lags y medias móviles
    historial = df_linea["TotalViajeros"].astype(float).tolist()
    ultima_fecha = df_linea["Fecha"].max()
    maximo = maximos_historicos[linea]

    predicciones = []

    for i in range(1, dias + 1):
        fecha_pred = ultima_fecha + pd.Timedelta(days=i)

        features = construir_features(
            linea=linea,
            fecha=fecha_pred,
            historial=historial,
            festivos=festivos,
            columnas_modelo=columnas_modelo
        )

        viajeros = int(max(0, round(float(modelo.predict(features)[0]))))
        porcentaje = round(viajeros / maximo, 4) if maximo > 0 else 0.0
        riesgo = calcular_riesgo(viajeros, maximo, config)

        predicciones.append(PrediccionDia(
            fecha=fecha_pred.strftime("%Y-%m-%d"),
            linea=linea,
            viajeros_predichos=viajeros,
            riesgo=riesgo,
            porcentaje_maximo_historico=porcentaje
        ))

        # Añadimos la predicción al historial para que los lags del siguiente día
        # se calculen sobre datos reales + predicciones anteriores
        historial.append(float(viajeros))

    return RespuestaPrediccion(linea=linea, predicciones=predicciones)