import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from src.modelos.evaluador import evaluar_con_timeseries_split, ResultadoEvaluacion

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RUTA_FEATURES = PROCESSED_DIR / "emt_features.csv"

# Baseline simple que predice el valor de viajeros del mismo día de la semana anterior (lag_7d)
class BaselineLag7(BaseEstimator, RegressorMixin):

    # función para cumplir con la interfaz de scikit-learn, 
    # aunque no hace nada en este caso porque no hay entrenamiento real
    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    # función de predicción que devuelve el valor del lag de 7 días como predicción
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_7d"].values


# Función principal que carga los datos, instancia el baseline y lo evalúa
def entrenar_baseline() -> ResultadoEvaluacion:
    print("Iniciando evaluación del baseline ...")

    df = pd.read_csv(RUTA_FEATURES, sep=";", parse_dates=["Fecha"])

    # Columnas que NO usamos como features, son la variable objetivo
    # o columnas de metadatos que no deben entrar al modelo
    cols_excluir = ["Fecha", "TotalViajeros", "anio_origen"]
    X = df.drop(columns=cols_excluir)
    y = df["TotalViajeros"]

    modelo = BaselineLag7()

    # Evaluamos el baseline usando TimeSeriesSplit para obtener, 
    # métricas robustas y comparables con modelos más complejos
    resultado = evaluar_con_timeseries_split(
        modelo=modelo,
        X=X,
        y=y,
        nombre_modelo="Baseline lag_7d"
    )

    return resultado

# Solo se ejecuta si lanzamos este archivo directamente
if __name__ == "__main__":
    from src.modelos.evaluador import imprimir_resumen
    resultado = entrenar_baseline()
    imprimir_resumen(resultado)