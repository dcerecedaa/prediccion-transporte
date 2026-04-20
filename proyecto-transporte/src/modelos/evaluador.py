import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataclasses import dataclass

# Número de folds para la validación temporal
N_FOLDS = 5

# dataclass para almacenar resultados de evaluación de modelos de forma estructurada y legible
@dataclass
class ResultadoEvaluacion:
    nombre_modelo: str
    mae: float
    rmse: float
    r2: float
    mae_por_fold: list[float]

# Función para calcular las métricas de evaluación: MAE, RMSE y R2.
def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_real, y_pred)

    # np.sqrt sobre mean_squared_error para obtener RMSE directamente
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))

    r2 = r2_score(y_real, y_pred)

    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}

# Función para evaluar un modelo usando TimeSeriesSplit
def evaluar_con_timeseries_split(
    modelo,
    X: pd.DataFrame,
    y: pd.Series,
    nombre_modelo: str
) -> ResultadoEvaluacion:
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    maes_por_fold = []

    print(f"\nEvaluando {nombre_modelo} con {N_FOLDS} folds...")

    # Bucle for para entrenar y evaluar el modelo en cada fold. 
    # El modelo se entrena solo con datos anteriores al fold de test, respetando la secuencia temporal.
    for fold, (idx_train, idx_test) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

        # Entrenamos con el pasado y predecimos el futuro — nunca al revés
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Calculamos métricas para este fold y las almacenamos para el resumen final
        metricas_fold = calcular_metricas(y_test.values, y_pred)
        maes_por_fold.append(metricas_fold["mae"])

        print(f" Fold {fold}: MAE={metricas_fold['mae']:,.0f} | RMSE={metricas_fold['rmse']:,.0f} | R2={metricas_fold['r2']:.4f}")

    # Métricas finales como media de todos los folds
    mae_medio = round(float(np.mean(maes_por_fold)), 2)
    rmse_medio = round(float(np.sqrt(np.mean([m**2 for m in maes_por_fold]))), 2)
    r2_medio = round(float(np.mean([
        r2_score(
            y.iloc[idx_test].values,
            modelo.predict(X.iloc[idx_test])
        )
        for _, idx_test in tscv.split(X)
    ])), 4)

    print(f"\n  Media {nombre_modelo}: MAE={mae_medio:,.0f} | RMSE={rmse_medio:,.0f} | R2={r2_medio:.4f}")

    return ResultadoEvaluacion(
        nombre_modelo=nombre_modelo,
        mae=mae_medio,
        rmse=rmse_medio,
        r2=r2_medio,
        mae_por_fold=maes_por_fold
    )

# Función para comparar varios modelos de forma estructurada y elegir el mejor según el MAE.
def comparar_modelos(resultados: list[ResultadoEvaluacion]) -> ResultadoEvaluacion:
    print("\n" + "="*50)
    print("COMPARATIVA DE MODELOS")
    print("="*50)

    for r in resultados:
        print(f"{r.nombre_modelo:25} MAE={r.mae:,.0f} | RMSE={r.rmse:,.0f} | R2={r.r2:.4f}")

    # MAE como criterio de selección, es la métrica más interpretable
    # si MAE=500 significa que de media nos equivocamos en 500 viajeros por día
    mejor = min(resultados, key=lambda r: r.mae)
    print(f"\nMejor modelo: {mejor.nombre_modelo} (MAE={mejor.mae:,.0f})")

    return mejor

# Imprime un resumen legible de los resultados — útil para la documentación y la defensa
def imprimir_resumen(resultado: ResultadoEvaluacion) -> None:
    print("\n" + "="*50)
    print(f"RESUMEN: {resultado.nombre_modelo}")
    print("="*50)
    print(f"MAE  (error medio en viajeros): {resultado.mae:,.0f}")
    print(f"RMSE (penaliza errores grandes): {resultado.rmse:,.0f}")
    print(f"R2   (bondad del ajuste 0-1):   {resultado.r2:.4f}")
    print(f"MAE por fold: {[f'{m:,.0f}' for m in resultado.mae_por_fold]}")

    # La desviación estándar del MAE mide la estabilidad del modelo
    # si varía mucho entre folds hay que investigar por qué
    print(f"Estabilidad (std MAE): {np.std(resultado.mae_por_fold):,.0f}")