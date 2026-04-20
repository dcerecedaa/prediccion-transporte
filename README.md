# 🚍 Predicción de Demanda — Transporte Público de Madrid EMT
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)
![Data](https://img.shields.io/badge/Data-EMT%20Madrid%202019--2026-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Sistema de análisis y predicción de la demanda diaria por línea de autobús de la EMT de Madrid, con estimación de riesgo de saturación, API REST y dashboard interactivo.

---

## 📌 Descripción

Este proyecto desarrolla un pipeline completo de Machine Learning para anticipar escenarios de alta ocupación en el transporte público de Madrid. A partir de datos históricos reales de la EMT (2019–2025), el sistema predice la demanda diaria por línea con un horizonte de 7 a 14 días y clasifica el riesgo de saturación como **Bajo**, **Medio** o **Alto**.

El sistema ha sido validado con datos reales de enero de 2026 obteniendo un **MAE de 390 viajeros** y un **R² de 0.985**, lo que representa menos de un 7% de error relativo medio.

---

## 🎯 Objetivos

- Predecir la demanda diaria por línea de autobús con 7–14 días de antelación
- Detectar patrones temporales — semanales, estacionales y atípicos
- Clasificar el riesgo de saturación por línea y día
- Exponer las predicciones mediante una API REST consumible por cualquier sistema
- Visualizar los resultados en un dashboard interactivo

---

## 🏗️ Arquitectura del sistema

```
Datos crudos (CSV)
      │
      ▼
Ingesta y limpieza          src/ingesta/        src/preprocesamiento/
      │
      ▼
Feature engineering         src/caracteristicas/
      │
      ▼
Entrenamiento y evaluación  src/modelos/
      │
      ▼
Modelo serializado          data/models/lightgbm_emt.joblib
      │
      ▼
API REST                    src/api/main.py
      │
      ▼
Dashboard interactivo       dashboard/app.py
```

---

## 📊 Resultados

| Versión | Cambio | MAE | R² |
|---------|--------|-----|----|
| Baseline lag 7 días | Referencia mínima | 578 | 0.880 |
| LightGBM base | Modelo inicial | 451 | 0.882 |
| LightGBM + Optuna | Optimización de hiperparámetros | 430 | 0.886 |
| LightGBM + Optuna + sin línea 868 | Versión final | 422 | 0.895 |
| **Validación enero 2026** | **Datos nunca vistos** | **390** | **0.985** |

La validación con datos reales de enero de 2026 confirma que el modelo generaliza bien — el 85% de las predicciones tienen un error inferior al 20%.

---

## 🗂️ Estructura del proyecto

```
proyecto-transporte/
├── data/
│   ├── raw/                         # CSV originales de la EMT — nunca se modifican
│   ├── processed/                   # Datasets procesados y con features
│   └── models/                      # Modelo serializado con joblib
├── src/
│   ├── ingesta/
│   │   ├── cargador.py              # Carga y unificación de CSV anuales
│   │   └── pipeline.py             # Orquestador ingesta → limpieza → guardado
│   ├── preprocesamiento/
│   │   └── limpiador.py            # Limpieza, validación y marcado de outliers
│   ├── caracteristicas/
│   │   └── constructor.py          # Feature engineering — lags, medias móviles, festivos
│   ├── modelos/
│   │   ├── evaluador.py            # Métricas y validación temporal compartida
│   │   ├── baseline.py             # Modelo de referencia basado en lag de 7 días
│   │   ├── modelo_principal.py     # LightGBM con optimización Optuna
│   │   ├── entrenador.py           # Orquestador de entrenamiento + MLflow
│   │   └── validacion_2026.py      # Validación con datos reales de 2026
│   └── api/
│       └── main.py                 # API REST con FastAPI
├── dashboard/
│   └── app.py                      # Dashboard interactivo con Streamlit
├── notebooks/
│   └── 01_eda.ipynb                # Análisis exploratorio de los datos
├── configs/
│   └── settings.yaml               # Configuración centralizada — festivos y umbrales
├── mlflow.db                        # Base de datos SQLite de experimentos
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

### Requisitos previos

- Python 3.10 o superior
- Git

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/decerecedaa/prediccion-transporte
cd proyecto-transporte

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Uso

### 1. Preparar los datos

Coloca los CSV anuales de la EMT en `data/raw/` con el formato `demandadialinea_YYYY.csv`.

```bash
# Ejecutar la pipeline de ingesta y limpieza
python -m src.ingesta.pipeline

# Construir las features
python -m src.caracteristicas.constructor
```

### 2. Entrenar el modelo

```bash
python -m src.modelos.entrenador
```

Esto entrena el baseline y el modelo principal LightGBM, registra los experimentos en MLflow y guarda el modelo en `data/models/lightgbm_emt.joblib`.

Para visualizar los experimentos en MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Abrir http://127.0.0.1:5000
```

### 3. Validar con datos de 2026

```bash
python -m src.modelos.validacion_2026
```

### 4. Lanzar la API

```bash
uvicorn src.api.main:app --reload
# Disponible en http://127.0.0.1:8000
# Documentación interactiva en http://127.0.0.1:8000/docs
```

### 5. Lanzar el dashboard

En otra terminal, con la API corriendo:

```bash
streamlit run dashboard/app.py
# Disponible en http://localhost:8501
```

---

## 🔌 API REST

### Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/salud` | Estado de la API y número de líneas disponibles |
| GET | `/predecir/{linea}?dias=7` | Predicción de demanda con nivel de riesgo |
| GET | `/docs` | Documentación interactiva |

### Ejemplo de respuesta

```bash
GET /predecir/1?dias=3
```

```json
{
  "linea": 1,
  "predicciones": [
    {
      "fecha": "2026-02-01",
      "linea": 1,
      "viajeros_predichos": 3582,
      "riesgo": "bajo",
      "porcentaje_maximo_historico": 0.3897
    },
    {
      "fecha": "2026-02-02",
      "linea": 1,
      "viajeros_predichos": 6432,
      "riesgo": "bajo",
      "porcentaje_maximo_historico": 0.6997
    },
    {
      "fecha": "2026-02-03",
      "linea": 1,
      "viajeros_predichos": 6826,
      "riesgo": "medio",
      "porcentaje_maximo_historico": 0.7426
    }
  ]
}
```

### Niveles de riesgo

| Nivel | Criterio |
|-------|----------|
| 🟢 Bajo | Predicción < 70% del máximo histórico de la línea |
| 🟡 Medio | Entre 70% y 85% del máximo histórico |
| 🔴 Alto | Por encima del 85% del máximo histórico |

---

## 🧠 Modelado

### Fuente de datos

- **EMT Madrid** — viajeros diarios por línea desde 2019 hasta 2026
- Más de 536.000 registros tras la limpieza
- Incluye el periodo COVID (2020–2021) como dato contextual

### Features construidas

| Feature | Descripción |
|---------|-------------|
| `lag_7d` | Demanda hace 7 días — mismo día de la semana |
| `lag_14d` | Demanda hace 14 días |
| `lag_28d` | Demanda hace 28 días — mismo día hace 4 semanas |
| `media_movil_7d` | Media de los últimos 7 días |
| `media_movil_14d` | Media de los últimos 14 días |
| `dia_semana` | Día de la semana (0=lunes, 6=domingo) |
| `mes` | Mes del año |
| `es_festivo` | Festivo nacional o de la Comunidad de Madrid |
| `es_dia_especial` | Fin de semana o festivo |
| `es_covid` | Periodo pandémico (2020–2021) |
| `es_outlier` | Valor atípico detectado con IQR × 3 |

### Decisiones técnicas relevantes

- **Baseline**: lag de 7 días — predice que la demanda de hoy será igual a la del mismo día de la semana anterior. Se eligió frente a Prophet o SARIMA porque ambos requieren entrenar un modelo por línea (más de 200), lo que es ineficiente e imposible de integrar con el evaluador compartido.
- **Modelo principal**: LightGBM con hiperparámetros optimizados por Optuna en 50 iteraciones.
- **Validación**: TimeSeriesSplit con 5 folds — siempre se entrena con datos pasados y se valida con datos futuros.
- **Outliers**: marcados pero no eliminados — son eventos reales (huelgas, eventos masivos) que el modelo debe conocer.
- **COVID**: marcado con columna booleana para que el modelo aprenda el comportamiento atípico de esos años.
- **Línea 868**: excluida del entrenamiento — línea temporal que solo existió de octubre a diciembre de 2025 con demanda 15–20 veces superior a cualquier otra línea.

---

## 📦 Dependencias principales

| Librería | Uso |
|----------|-----|
| pandas, numpy | Manipulación y procesamiento de datos |
| lightgbm | Modelo principal de predicción |
| optuna | Optimización automática de hiperparámetros |
| scikit-learn | Validación temporal y métricas |
| mlflow | Registro de experimentos |
| joblib | Serialización del modelo |
| fastapi, uvicorn | API REST |
| streamlit, plotly | Dashboard interactivo |
| pyyaml | Lectura de configuración |

Instalar todo con:

```bash
pip install -r requirements.txt
```

---

## 📅 Estado del proyecto

El proyecto está completo e incluye pipeline de datos, modelo entrenado, validación con datos reales, API REST y dashboard interactivo. Las siguientes funcionalidades están contempladas como evolución futura:

- Reentrenamiento automático periódico con datos nuevos
- Integración de señales externas — feed de Twitter/X de la EMT para detección de incidencias
- Predicción a nivel de parada (requiere datos horarios por parada)
- Despliegue en servidor con autenticación en los endpoints

---

## 🤝 Contribuciones

Este proyecto está abierto a sugerencias y feedback. Si tienes ideas de mejora o encuentras algún bug, no dudes en abrir un issue.

---

## 👨‍💻 Autor

David Cereceda Pérez  
[GitHub](https://github.com/dcerecedaa) | [LinkedIn](https://linkedin.com/in/david-cereceda-perez-3ba0962b6)

---

> ⚠️ **Nota final:** Este proyecto es educativo y de demostración.  
> No está pensado para uso comercial ni producción, y se incluyen limitaciones intencionadas para mantener la implementación clara y enfocada en la lógica técnica.
