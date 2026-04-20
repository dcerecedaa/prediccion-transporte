import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# URL base de la API — debe estar corriendo antes de lanzar el dashboard
API_URL = "http://127.0.0.1:8000"

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# Configuración general de la página
st.set_page_config(
    page_title="Predicción de Demanda EMT Madrid",
    page_icon="🚍",
    layout="wide"
)

st.title("🚍 Predicción de Demanda — EMT Madrid")
st.markdown("Sistema de predicción de demanda diaria por línea de autobús con estimación de riesgo de saturación.")


# Comprobamos que la API está disponible antes de continuar
@st.cache_data(ttl=60)
def verificar_api() -> dict | None:
    try:
        respuesta = requests.get(f"{API_URL}/salud", timeout=3)
        if respuesta.status_code == 200:
            return respuesta.json()
        return None
    except requests.exceptions.ConnectionError:
        return None


# Cargamos las líneas disponibles desde el dataset de features
@st.cache_data
def cargar_lineas_disponibles() -> list[int]:
    ruta = PROCESSED_DIR / "emt_features.csv"
    if not ruta.exists():
        return []
    df = pd.read_csv(ruta, sep=";", usecols=["Linea"])
    return sorted(df["Linea"].unique().tolist())


# Cargamos el histórico reciente de una línea para mostrarlo como contexto
@st.cache_data
def cargar_historico_linea(linea: int, ultimos_dias: int = 30) -> pd.DataFrame:
    ruta = PROCESSED_DIR / "emt_features.csv"
    df = pd.read_csv(ruta, sep=";", parse_dates=["Fecha"])
    df_linea = df[df["Linea"] == linea].sort_values("Fecha").tail(ultimos_dias)
    return df_linea[["Fecha", "TotalViajeros"]]


# Llamamos a la API para obtener las predicciones de una línea
def obtener_predicciones(linea: int, dias: int) -> list[dict] | None:
    try:
        respuesta = requests.get(
            f"{API_URL}/predecir/{linea}",
            params={"dias": dias},
            timeout=10
        )
        if respuesta.status_code == 200:
            return respuesta.json()["predicciones"]
        return None
    except requests.exceptions.ConnectionError:
        return None


# Devuelve el color asociado al nivel de riesgo para los indicadores visuales
def color_riesgo(riesgo: str) -> str:
    return {"bajo": "#2ECC71", "medio": "#F39C12", "alto": "#E74C3C"}.get(riesgo, "#95A5A6")


def emoji_riesgo(riesgo: str) -> str:
    return {"bajo": "🟢", "medio": "🟡", "alto": "🔴"}.get(riesgo, "⚪")


# ── ESTADO DE LA API ──
estado_api = verificar_api()

if estado_api is None:
    st.error("La API no está disponible. Asegúrate de que está corriendo con: uvicorn src.api.main:app --reload")
    st.stop()

st.success(f"API operativa — {estado_api['lineas_disponibles']} líneas disponibles")

# ── CONTROLES ──
col1, col2 = st.columns([2, 1])

with col1:
    lineas = cargar_lineas_disponibles()
    linea_seleccionada = st.selectbox(
        "Selecciona una línea de autobús",
        options=lineas,
        index=0
    )

with col2:
    dias_prediccion = st.slider(
        "Días a predecir",
        min_value=1,
        max_value=14,
        value=7
    )

# ── PREDICCIONES ──
if st.button("Generar predicción", type="primary"):
    with st.spinner("Consultando la API..."):
        predicciones = obtener_predicciones(linea_seleccionada, dias_prediccion)

    if predicciones is None:
        st.error("Error al obtener las predicciones. Comprueba que la API está funcionando.")
    else:
        df_pred = pd.DataFrame(predicciones)
        df_pred["Fecha"] = pd.to_datetime(df_pred["fecha"])
        df_historico = cargar_historico_linea(linea_seleccionada)

        st.markdown("---")

        # ── INDICADORES DE RIESGO ──
        st.subheader(f"Riesgo de saturación — línea {linea_seleccionada}")
        cols = st.columns(len(predicciones))

        for i, pred in enumerate(predicciones):
            with cols[i]:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='font-size:11px;color:#666'>{pred['fecha']}</div>"
                    f"<div style='font-size:24px'>{emoji_riesgo(pred['riesgo'])}</div>"
                    f"<div style='font-size:13px;font-weight:bold;color:{color_riesgo(pred['riesgo'])}'>"
                    f"{pred['riesgo'].upper()}</div>"
                    f"<div style='font-size:11px;color:#888'>{pred['viajeros_predichos']:,} viaj.</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # ── GRÁFICA ──
        st.subheader("Histórico y predicción de demanda")

        fig = go.Figure()

        # Histórico real de los últimos 30 días
        fig.add_trace(go.Scatter(
            x=df_historico["Fecha"],
            y=df_historico["TotalViajeros"],
            name="Histórico real",
            line=dict(color="#2E75B6", width=2),
            mode="lines+markers",
            marker=dict(size=4)
        ))

        # Predicciones con color según riesgo
        colores_pred = [color_riesgo(p["riesgo"]) for p in predicciones]

        fig.add_trace(go.Scatter(
            x=df_pred["Fecha"],
            y=df_pred["viajeros_predichos"],
            name="Predicción",
            line=dict(color="#E67E22", width=2, dash="dash"),
            mode="lines+markers",
            marker=dict(size=8, color=colores_pred, line=dict(width=1, color="white"))
        ))

        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Viajeros",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── TABLA DE PREDICCIONES ──
        st.subheader("Detalle de predicciones")

        df_tabla = df_pred[["fecha", "viajeros_predichos", "riesgo", "porcentaje_maximo_historico"]].copy()
        df_tabla.columns = ["Fecha", "Viajeros predichos", "Riesgo", "% del máximo histórico"]
        df_tabla["% del máximo histórico"] = (df_tabla["% del máximo histórico"] * 100).round(1).astype(str) + "%"
        df_tabla["Viajeros predichos"] = df_tabla["Viajeros predichos"].apply(lambda x: f"{x:,}")

        st.dataframe(df_tabla, use_container_width=True, hide_index=True)