"""Streamlit dashboard for solar power prediction.

Run with:  streamlit run app.py
"""

import datetime
import os
import sys

import pandas as pd
import streamlit as st

# Make the package importable when the app is launched from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.solar_prediction.data_processing import load_dataset
from src.solar_prediction.model_training import load_saved_model, train_and_evaluate
from src.solar_prediction.prediction import predict_solar_output
from src.solar_prediction.visualization import (
    plot_historical_solar,
    plot_historical_vs_forecast,
)
from src.solar_prediction.weather_service import get_location_info

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

location = get_location_info()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60)
def get_dataset() -> pd.DataFrame:
    """Load the generated dataset (cached for one hour)."""
    return load_dataset()


def dataset_exists() -> bool:
    from src.solar_prediction.config import DATA_PATH

    return os.path.exists(DATA_PATH)


def model_exists() -> bool:
    from src.solar_prediction.config import ARTIFACTS_DIR

    return os.path.exists(os.path.join(ARTIFACTS_DIR, "solar_model_metadata.joblib"))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Solar Power Prediction")
st.caption(
    f"Pune, Maharashtra, India · {location['latitude']}°N, {location['longitude']}°E "
    f"· {location['timezone']}"
)
st.markdown(
    "Predict hourly solar output from weather data. The model is trained on a "
    "synthetic 60-day history and uses the free Open-Meteo forecast API for live "
    "predictions."
)

# ---------------------------------------------------------------------------
# Step 1 - dataset
# ---------------------------------------------------------------------------
st.divider()
st.header("1. Dataset")

left, right = st.columns([1, 2], vertical_alignment="center")
with left:
    if st.button("Generate dataset", use_container_width=True):
        from scripts.generate_dataset import generate_dataset

        generate_dataset()
        get_dataset.clear()
        st.rerun()
with right:
    if dataset_exists():
        df = get_dataset()
        st.success(
            f"{len(df):,} hourly rows · "
            f"{df['timestamp'].min():%d %b %Y} to {df['timestamp'].max():%d %b %Y} · "
            f"{int(df.isnull().sum().sum())} missing values"
        )
    else:
        st.info("No dataset yet. Click **Generate dataset** to create it.")

if dataset_exists():
    df = get_dataset()
    st.plotly_chart(
        plot_historical_solar(df, title="Solar output over the last 60 days", height=320),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Step 2 - model
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Model")

left, right = st.columns([1, 2], vertical_alignment="center")
with left:
    if st.button("Train model", use_container_width=True):
        with st.spinner("Training and tuning models..."):
            train_and_evaluate()
        st.rerun()
with right:
    if model_exists():
        st.caption(
            "Trains RandomForest and tuned HistGradientBoosting, then keeps the "
            "better one on the last 10 days of data."
        )
    else:
        st.info("No trained model yet. Click **Train model** to create one.")

if model_exists():
    model_data = load_saved_model()
    metrics = model_data["train_metrics"]
    tuning = model_data.get("tuning_results", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected model", model_data["model_name"])
    col2.metric("MAE", f"{metrics['mae']:.2f}")
    col3.metric("RMSE", f"{metrics['rmse']:.2f}")
    col4.metric("R²", f"{metrics['r2']:.4f}")

    with st.expander("Details"):
        st.write(
            f"**Training period:** {model_data['train_date_range'][0]:%Y-%m-%d %H:%M} "
            f"to {model_data['train_date_range'][1]:%Y-%m-%d %H:%M}"
        )
        st.write(
            f"**Validation period:** {model_data['val_date_range'][0]:%Y-%m-%d %H:%M} "
            f"to {model_data['val_date_range'][1]:%Y-%m-%d %H:%M}"
        )
        if tuning.get("best_params"):
            st.write(
                "**Hyperparameters:** HistGradientBoosting tuned with "
                f"{tuning['cv']} — best parameters {tuning['best_params']} "
                f"(CV MAE {tuning['best_cv_mae']:.2f})."
            )
        st.write("**Features used by the model**")
        st.code(", ".join(model_data["feature_names"]), language=None)

# ---------------------------------------------------------------------------
# Step 3 - forecast
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Forecast")

horizon = st.radio(
    "Forecast horizon (hours)",
    options=[24, 48],
    index=1,
    horizontal=True,
)

if st.button("Predict solar output", type="primary"):
    if not model_exists():
        st.error("Train a model first (step 2).")
    else:
        with st.spinner("Fetching the weather forecast and predicting..."):
            try:
                st.session_state["predictions"] = predict_solar_output(
                    forecast_horizon_hours=horizon
                )
                st.session_state["prediction_horizon"] = horizon
            except Exception as exc:
                st.session_state.pop("predictions", None)
                st.error(f"Prediction failed: {exc}")

if "predictions" in st.session_state:
    pred_df = st.session_state["predictions"]
    horizon = st.session_state.get("prediction_horizon", horizon)

    if pred_df.attrs.get("forecast_source") == "synthetic-fallback":
        st.warning(
            "The live weather API is unavailable, so predictions use synthetic "
            "weather history as a fallback."
        )

    values = pred_df["predicted_solar_output"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Average output", f"{values.mean():.1f} W/m²")
    col2.metric("Peak output", f"{values.max():.1f} W/m²")
    col3.metric("Total output", f"{values.sum():.0f} W/m²")

    st.plotly_chart(
        plot_historical_vs_forecast(
            get_dataset().tail(24 * 7),
            pred_df,
            historical_name="Historical (last 7 days)",
            forecast_name=f"Predicted (next {horizon} hours)",
            title=f"Historical vs. {horizon}-hour forecast",
        ),
        use_container_width=True,
    )

    st.subheader("Prediction details")
    st.dataframe(
        pred_df.assign(
            timestamp=pred_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
            predicted_solar_output=values.round(2),
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download predictions as CSV",
        data=pred_df.to_csv(index=False),
        file_name=f"solar_predictions_{datetime.datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    st.caption(
        "The training dataset is synthetic, so predictions demonstrate the "
        "pipeline rather than real-world accuracy."
    )
