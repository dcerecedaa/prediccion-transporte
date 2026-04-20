import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from src.modelos.evaluador import evaluar_con_timeseries_split, ResultadoEvaluacion

# Definimos rutas y constantes para organizar el proyecto
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "data" / "models"
RUTA_FEATURES = PROCESSED_DIR / "emt_features.csv"

# Columnas que no entran al modelo como features
# Fecha y anio_origen son metadatos, TotalViajeros es la variable objetivo
# es_fin_semana se eliminó por tener importancia prácticamente cero
COLS_EXCLUIR = ["Fecha", "TotalViajeros", "anio_origen", "es_fin_semana"]

# Número de intentos que hará Optuna para encontrar los mejores hiperparámetros
# más intentos = mejor resultado pero más tiempo de ejecución
N_TRIALS = 50

# Función que carga el dataset de features y separa X e y
def cargar_datos() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(RUTA_FEATURES, sep=";", parse_dates=["Fecha"])
    X = df.drop(columns=COLS_EXCLUIR)
    y = df["TotalViajeros"]
    return X, y, df

# Función objetivo que Optuna llama en cada intento
# recibe un trial con los hiperparámetros a probar y devuelve el MAE
# Optuna minimiza este valor buscando la mejor combinación
def objetivo_optuna(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        # suggest_int y suggest_float definen el rango de búsqueda de cada parámetro
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "verbose": -1,
        "random_state": 42,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []

    for idx_train, idx_test in tscv.split(X):
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

        modelo = lgb.LGBMRegressor(**params)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        maes.append(mean_absolute_error(y_test, y_pred))

    return float(np.mean(maes))

# Función que lanza la optimización de hiperparámetros con Optuna
# devuelve los mejores parámetros encontrados tras N_TRIALS intentos
def optimizar_hiperparametros(X: pd.DataFrame, y: pd.Series) -> dict:
    print(f"\nOptimizando hiperparámetros con Optuna ({N_TRIALS} intentos)...")

    # optuna.logging.WARNING silencia los mensajes de cada intento
    # para no inundar la consola — solo veremos el resultado final
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # direction="minimize" indica que queremos minimizar el MAE
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objetivo_optuna(trial, X, y),
        n_trials=N_TRIALS
    )

    print(f"Mejor MAE encontrado: {study.best_value:,.0f}")
    print(f"Mejores parámetros: {study.best_params}")

    return study.best_params

# Función que muestra la importancia de cada feature según LightGBM
# útil para entender qué variables han sido más relevantes para predecir
def mostrar_importancia_features(modelo: lgb.LGBMRegressor, X: pd.DataFrame) -> None:
    importancias = pd.Series(
        modelo.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nImportancia de features:")
    for feature, importancia in importancias.items():
        print(f"  {feature:25} {importancia:,.0f}")

# Función que guarda el modelo entrenado en data/models/
# usamos joblib porque es el estándar para serializar modelos de sklearn y compatibles
def guardar_modelo(modelo: lgb.LGBMRegressor, nombre: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = MODELS_DIR / f"{nombre}.joblib"
    joblib.dump(modelo, ruta)
    print(f"\nModelo guardado en: {ruta}")
    return ruta

# Función principal que entrena LightGBM con hiperparámetros optimizados
# y devuelve el resultado de evaluación
def entrenar_modelo_principal() -> tuple[ResultadoEvaluacion, lgb.LGBMRegressor]:
    print("Iniciando entrenamiento del modelo principal...")

    X, y, df = cargar_datos()

    # Primero optimizamos los hiperparámetros con Optuna
    mejores_params = optimizar_hiperparametros(X, y)

    # Añadimos los parámetros fijos que no optimizamos
    mejores_params["verbose"] = -1
    mejores_params["random_state"] = 42

    modelo = lgb.LGBMRegressor(**mejores_params)

    resultado = evaluar_con_timeseries_split(
        modelo=modelo,
        X=X,
        y=y,
        nombre_modelo="LightGBM"
    )

    # Entrenamiento final con todos los datos disponibles
    # después de la evaluación, entrenamos con el dataset completo
    # para que el modelo guardado haya visto todos los patrones históricos
    print("\nEntrenando modelo final con todos los datos...")
    modelo.fit(X, y)

    mostrar_importancia_features(modelo, X)
    guardar_modelo(modelo, "lightgbm_emt")

    return resultado, modelo

# Solo se ejecuta si lanzamos este archivo directamente
if __name__ == "__main__":
    from src.modelos.evaluador import imprimir_resumen
    resultado, modelo = entrenar_modelo_principal()
    imprimir_resumen(resultado)