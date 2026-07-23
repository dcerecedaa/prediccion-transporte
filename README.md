# 🚍 Public Transport Demand Forecasting — Madrid EMT

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)
![Data](https://img.shields.io/badge/Data-EMT%20Madrid%202019--2026-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end Machine Learning system for forecasting daily passenger demand across Madrid EMT bus lines, estimating congestion risk, and exposing predictions through a REST API and an interactive dashboard.

---

# 📌 Overview

This project implements a complete Machine Learning pipeline to anticipate high-demand scenarios in Madrid's public transportation network.

Using real historical data provided by EMT Madrid (2019–2025), the system forecasts daily passenger demand for each bus line up to **7–14 days ahead** and classifies congestion risk into three levels:

- **Low**
- **Medium**
- **High**

The model was validated using unseen data from **January 2026**, achieving:

- **MAE:** 390 passengers
- **R²:** 0.985

This corresponds to an average relative prediction error below **7%**.

---

# 🎯 Objectives

- Forecast daily passenger demand for each bus line up to 14 days in advance
- Capture weekly, seasonal, and anomalous demand patterns
- Estimate congestion risk for every route and day
- Provide predictions through a REST API
- Visualize forecasts using an interactive dashboard

---

# 🏗️ System Architecture

```text
Raw CSV files
      │
      ▼
Data ingestion & cleaning      src/ingesta/        src/preprocesamiento/
      │
      ▼
Feature Engineering            src/caracteristicas/
      │
      ▼
Model Training & Evaluation    src/modelos/
      │
      ▼
Serialized Model               data/models/lightgbm_emt.joblib
      │
      ▼
REST API                       src/api/main.py
      │
      ▼
Interactive Dashboard          dashboard/app.py
```

---

# 📊 Results

| Version | Description | MAE | R² |
|----------|-------------|-----|----|
| 7-Day Lag Baseline | Simple reference model | 578 | 0.880 |
| LightGBM | Initial model | 451 | 0.882 |
| LightGBM + Optuna | Hyperparameter optimization | 430 | 0.886 |
| LightGBM + Optuna (without Line 868) | Final model | 422 | 0.895 |
| **January 2026 Validation** | **Unseen real-world data** | **390** | **0.985** |

Evaluation on unseen data confirms that the model generalizes well, with **85% of predictions presenting less than 20% error**.

---

# 📂 Project Structure

```text
project/
├── data/
│   ├── raw/                        # Original EMT CSV files
│   ├── processed/                  # Processed datasets
│   └── models/                     # Serialized models
│
├── src/
│   ├── ingesta/
│   │   ├── cargador.py             # CSV loader
│   │   └── pipeline.py             # Data ingestion pipeline
│   │
│   ├── preprocesamiento/
│   │   └── limpiador.py            # Cleaning & validation
│   │
│   ├── caracteristicas/
│   │   └── constructor.py          # Feature engineering
│   │
│   ├── modelos/
│   │   ├── evaluador.py            # Shared evaluation utilities
│   │   ├── baseline.py             # 7-day lag baseline
│   │   ├── modelo_principal.py     # LightGBM + Optuna
│   │   ├── entrenador.py           # Training orchestrator
│   │   └── validacion_2026.py      # Final validation
│   │
│   └── api/
│       └── main.py                 # FastAPI application
│
├── dashboard/
│   └── app.py                      # Streamlit dashboard
│
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory Data Analysis
│
├── configs/
│   └── settings.yaml               # Configuration
│
├── mlflow.db
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Requirements

- Python 3.10+
- Git

## Clone the repository

```bash
git clone https://github.com/decerecedaa/prediccion-transporte
cd proyecto-transporte
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## 1. Prepare the data

Place the yearly EMT CSV files inside:

```
data/raw/
```

using the following naming convention:

```
demandadialinea_YYYY.csv
```

Run the ingestion pipeline:

```bash
python -m src.ingesta.pipeline
```

Generate features:

```bash
python -m src.caracteristicas.constructor
```

---

## 2. Train the models

```bash
python -m src.modelos.entrenador
```

This process:

- Trains the baseline model
- Trains the optimized LightGBM model
- Logs experiments using MLflow
- Saves the trained model into

```
data/models/lightgbm_emt.joblib
```

Launch MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```
http://127.0.0.1:5000
```

---

## 3. Validate using 2026 data

```bash
python -m src.modelos.validacion_2026
```

---

## 4. Run the REST API

```bash
uvicorn src.api.main:app --reload
```

Available at:

```
http://127.0.0.1:8000
```

Interactive documentation:

```
http://127.0.0.1:8000/docs
```

---

## 5. Launch the dashboard

With the API running:

```bash
streamlit run dashboard/app.py
```

Dashboard:

```
http://localhost:8501
```

---

# 🔌 REST API

## Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/salud` | API health status |
| GET | `/predecir/{linea}?dias=7` | Passenger demand forecast |
| GET | `/docs` | Swagger documentation |

Example request:

```http
GET /predecir/1?dias=3
```

Example response:

```json
{
  "linea": 1,
  "predicciones": [
    {
      "fecha": "2026-02-01",
      "linea": 1,
      "viajeros_predichos": 3582,
      "riesgo": "low",
      "porcentaje_maximo_historico": 0.3897
    }
  ]
}
```

---

# 🚦 Congestion Risk Levels

| Level | Criteria |
|--------|----------|
| 🟢 Low | Below 70% of the historical maximum |
| 🟡 Medium | Between 70% and 85% |
| 🔴 High | Above 85% of the historical maximum |

---

# 🧠 Machine Learning

## Dataset

- EMT Madrid daily passenger demand (2019–2026)
- More than **536,000 cleaned observations**
- Includes the COVID-19 period (2020–2021)

## Engineered Features

| Feature | Description |
|----------|-------------|
| `lag_7d` | Demand one week earlier |
| `lag_14d` | Demand two weeks earlier |
| `lag_28d` | Demand four weeks earlier |
| `rolling_mean_7d` | 7-day moving average |
| `rolling_mean_14d` | 14-day moving average |
| `day_of_week` | Weekday indicator |
| `month` | Month of the year |
| `is_holiday` | Public holiday |
| `is_special_day` | Weekend or holiday |
| `is_covid` | COVID period |
| `is_outlier` | IQR × 3 outlier flag |

---

# ⚙️ Technical Decisions

### Baseline

A simple **7-day lag model** was selected as the benchmark.

Unlike Prophet or SARIMA, this approach scales efficiently across more than **200 bus lines**, allowing a unified evaluation framework without training one model per route.

### Main Model

- LightGBM
- Hyperparameter optimization with Optuna (50 trials)

### Validation Strategy

- TimeSeriesSplit
- 5 folds
- Strict chronological evaluation

### Outliers

Outliers are **flagged but not removed**, since they represent real operational events such as strikes or city-wide events.

### COVID Feature

A dedicated boolean feature allows the model to learn abnormal passenger behavior during the pandemic.

### Bus Line 868

Route 868 was excluded from training because it only operated between October and December 2025 and presented demand values **15–20× larger** than every other route, severely biasing the model.

---

# 📦 Main Dependencies

| Library | Purpose |
|----------|---------|
| pandas, numpy | Data manipulation |
| lightgbm | Gradient boosting model |
| optuna | Hyperparameter optimization |
| scikit-learn | Metrics & validation |
| mlflow | Experiment tracking |
| joblib | Model serialization |
| fastapi, uvicorn | REST API |
| streamlit, plotly | Interactive dashboard |
| pyyaml | Configuration management |

Install everything with:

```bash
pip install -r requirements.txt
```

---

# 📅 Project Status

The project is fully functional and includes:

- Data ingestion pipeline
- Feature engineering
- Model training
- Hyperparameter optimization
- Experiment tracking
- Validation on unseen data
- REST API
- Interactive dashboard

Future improvements may include:

- Automatic model retraining
- Integration of external incident signals (Twitter/X)
- Bus stop-level forecasting
- Cloud deployment with authentication

---

# 🤝 Contributing

Suggestions, improvements, and bug reports are welcome.

Feel free to open an issue or submit a pull request.

---

# 👨‍💻 Author

**David Cereceda Pérez**

- GitHub: https://github.com/decerecedaa
- LinkedIn: https://linkedin.com/in/david-cereceda-perez-3ba0962b6

---

> **Disclaimer**
>
> This project was developed for educational and portfolio purposes.
> It is not intended for production use, and some implementation decisions were deliberately simplified to keep the codebase focused on demonstrating Machine Learning and software engineering concepts.
