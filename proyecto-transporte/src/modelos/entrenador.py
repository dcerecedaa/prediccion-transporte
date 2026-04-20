import mlflow
import mlflow.lightgbm
from pathlib import Path
from src.modelos.baseline import entrenar_baseline
from src.modelos.modelo_principal import entrenar_modelo_principal
from src.modelos.evaluador import comparar_modelos, imprimir_resumen

# Definimos la carpeta base del proyecto para organizar los experimentos de MLflow
BASE_DIR = Path(__file__).resolve().parents[2]

# Carpeta donde MLflow guardará los experimentos 
MLFLOW_DB = BASE_DIR / "mlflow.db"

# Función principal que orquesta el flujo completo de entrenamiento y evaluación
def ejecutar_entrenamiento() -> None:
    print("="*50)
    print("INICIANDO PIPELINE DE ENTRENAMIENTO")
    print("="*50)

    # Configuramos MLflow para usar una base de datos SQLite local 
    # y definimos el experimento donde se guardarán los resultados
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("emt-prediccion-demanda")

    # ── BASELINE ──
    # entrenamos el modelo baseline (lag7d) y registramos sus métricas en MLflow
    with mlflow.start_run(run_name="baseline_lag7d"):
        resultado_baseline = entrenar_baseline()

        # Registramos las métricas del baseline en MLflow
        # para poder compararlo visualmente con el modelo principal
        mlflow.log_metric("mae", resultado_baseline.mae)
        mlflow.log_metric("rmse", resultado_baseline.rmse)
        mlflow.log_metric("r2", resultado_baseline.r2)

    # ── MODELO PRINCIPAL ──
    # entrenamos el modelo principal (LightGBM) y registramos sus métricas e hiperparámetros en MLflow
    with mlflow.start_run(run_name="lightgbm_optuna"):
        resultado_lgbm, modelo_lgbm = entrenar_modelo_principal()

        mlflow.log_metric("mae", resultado_lgbm.mae)
        mlflow.log_metric("rmse", resultado_lgbm.rmse)
        mlflow.log_metric("r2", resultado_lgbm.r2)

        # Registramos el modelo en MLflow para poder cargarlo desde la API
        mlflow.lightgbm.log_model(modelo_lgbm, "modelo")

    # ── COMPARATIVA ──
    # comparamos ambos modelos y mostramos un resumen de cuál ha sido mejor según las métricas obtenidas
    mejor = comparar_modelos([resultado_baseline, resultado_lgbm])
    imprimir_resumen(mejor)

    print("\nPipeline de entrenamiento completado.")
    print(f"Experimentos guardados en: {MLFLOW_DB}")
    print("Para visualizarlos ejecuta: mlflow ui")


# Solo se ejecuta si lanzamos este archivo directamente
if __name__ == "__main__":
    ejecutar_entrenamiento()