"""Solar power prediction module.

Loads persisted model and metadata, applies feature engineering,
and generates solar output predictions from weather forecasts.

Documents that the historical dataset is synthetic and predictions
are for demonstration purposes.
"""

import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR
from .data_processing import (
    FeatureSchema,
    clean_and_engineer_features,
    load_dataset,
)
from .weather_service import fetch_hourly_forecasts, forecast_to_model_features

logger = logging.getLogger(__name__)

MODEL_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "solar_model.joblib")
METADATA_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "solar_model_metadata.joblib")


def _ensure_artifacts_exist() -> None:
    """Check that model artifacts exist, raise informative error if not."""
    if not os.path.exists(MODEL_ARTIFACT_PATH):
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_ARTIFACT_PATH}. "
            "Run the training pipeline first, or ensure the model has been "
            "trained and saved."
        )
    if not os.path.exists(METADATA_ARTIFACT_PATH):
        raise FileNotFoundError(
            f"Model metadata not found at {METADATA_ARTIFACT_PATH}. "
            "Run the training pipeline first."
        )


def load_model_and_metadata() -> Tuple:
    """Load the persisted model and its metadata.

    Returns
    -------
    tuple
        (model, feature_names, model_config, train_metrics, train_date_range, val_date_range)

    Raises
    ------
    FileNotFoundError
        If model artifacts don't exist.
    """
    import joblib

    _ensure_artifacts_exist()

    model = joblib.load(MODEL_ARTIFACT_PATH)
    metadata = joblib.load(METADATA_ARTIFACT_PATH)

    feature_names = metadata.get("feature_names", FeatureSchema.get_feature_names())
    model_config = metadata.get("model_config", {})
    train_metrics = metadata.get("train_metrics", {})
    train_date_range = metadata.get("train_date_range")
    val_date_range = metadata.get("val_date_range")
    model_name = metadata.get("model_name", "Unknown")

    print(f"Loaded model: {model_name}")
    print(f"Feature names: {feature_names}")
    print(f"Train metrics: MAE={train_metrics.get('mae', 'N/A'):.2f}, "
          f"RMSE={train_metrics.get('rmse', 'N/A'):.2f}, "
          f"R²={train_metrics.get('r2', 'N/A'):.4f}")

    return model, feature_names, model_config, train_metrics, train_date_range, val_date_range


def predict_solar_output(
    forecast_horizon_hours: int = 48,
    use_cached_forecast: bool = True,
) -> pd.DataFrame:
    """Predict future solar output from weather forecasts.

    Full pipeline:
    1. Fetch hourly weather forecasts from Open-Meteo API
    2. Transform forecasts into the model's expected feature format
    3. Generate predictions using the persisted model
    4. Clip predictions to non-negative values
    5. Return structured DataFrame with timestamps and predictions

    If the Open-Meteo API is unavailable, synthetic weather data
    generated from the training dataset is used as a fallback.

    Parameters
    ----------
    forecast_horizon_hours : int
        Number of future hours to predict (24 or 48). Default 48.
    use_cached_forecast : bool
        Whether to use cached forecast data if available.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - timestamp: original forecast timestamps
        - temperature: temperature measurements
        - humidity: humidity measurements
        - cloud_cover: cloud cover measurements
        - wind_speed: wind speed measurements
        - precipitation: precipitation measurements
        - predicted_solar_output: clipped non-negative predictions

    Raises
    ------
    FileNotFoundError
        If model artifacts don't exist.
    ValueError
        If forecast data is incomplete or invalid.
    RuntimeError
        If prediction generation fails.
    """
    import os
    import joblib

    # Load model and metadata
    try:
        model, feature_names, model_config, train_metrics, train_date_range, val_date_range = load_model_and_metadata()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Cannot generate predictions: {e}. "
            "Train the model first or ensure model artifacts exist."
        )

    # Fetch weather forecasts - try API first, fall back to synthetic data
    forecast_records = None
    forecast_source = "open-meteo"
    fallback_reason = None
    try:
        weather_result = fetch_hourly_forecasts(
            hours=forecast_horizon_hours,
            use_cache=use_cached_forecast,
        )
        forecast_records = weather_result["forecast_data"]
    except Exception as exc:
        # Fall back to the synthetic weather history. Timestamps continue
        # hourly from the end of the dataset so the result is still a future
        # forecast window rather than a re-labelled slice of the past.
        fallback_reason = f"{type(exc).__name__}: {exc}"
        forecast_source = "synthetic-fallback"
        logger.warning(
            "Weather API unavailable (%s); using synthetic fallback data.",
            fallback_reason,
        )

        df = clean_and_engineer_features(load_dataset(), reindex=False)
        history = df.tail(forecast_horizon_hours).reset_index(drop=True)
        if history.empty:
            raise RuntimeError(
                "No forecast data available. "
                "Cannot generate predictions without weather data."
            )

        start_time = df["timestamp"].max() + pd.Timedelta(hours=1)
        forecast_records = []
        for i in range(forecast_horizon_hours):
            row = history.iloc[i % len(history)]
            forecast_records.append({
                "timestamp": start_time + pd.Timedelta(hours=i),
                "temperature": float(row["temperature"]),
                "humidity": float(row["humidity"]),
                "cloud_cover": float(row["cloud_cover"]),
                "wind_speed": float(row["wind_speed"]),
                "precipitation": float(row["precipitation"]),
            })

    if not forecast_records:
        raise ValueError("No forecast records available for prediction.")

    # Transform forecasts to model features
    try:
        X = forecast_to_model_features(forecast_records)
    except ValueError as e:
        raise ValueError(
            f"Failed to transform forecast data to model features: {e}. "
            "Forecast data may be missing required fields."
        )

    # Ensure feature column order matches what the model expects
    X = X[feature_names]

    # Generate predictions
    try:
        raw_predictions = model.predict(X)
    except Exception as e:
        raise RuntimeError(
            f"Model prediction failed: {e}. "
            "The model may have incompatible features or be corrupted."
        )

    # Solar output cannot be negative
    predictions = np.clip(raw_predictions, 0, None)

    # Build structured result DataFrame
    df = pd.DataFrame({
        "timestamp": [r["timestamp"] for r in forecast_records],
        "temperature": [r["temperature"] for r in forecast_records],
        "humidity": [r["humidity"] for r in forecast_records],
        "cloud_cover": [r["cloud_cover"] for r in forecast_records],
        "wind_speed": [r["wind_speed"] for r in forecast_records],
        "precipitation": [r["precipitation"] for r in forecast_records],
        "predicted_solar_output": predictions,
    })

    # Preserve original forecast timestamps (already in df)
    # Add a note that predictions are clipped to non-negative
    df.attrs["model_name"] = train_metrics.get("model_name", "HistGradientBoosting") if train_metrics else "Unknown"
    df.attrs["model_config"] = model_config
    df.attrs["forecast_horizon_hours"] = forecast_horizon_hours
    df.attrs["forecast_source"] = forecast_source
    df.attrs["fallback_reason"] = fallback_reason
    df.attrs["clipping_applied"] = True
    df.attrs["synthetic_data_disclaimer"] = (
        "WARNING: The historical dataset used to train this model is synthetic. "
        "Predictions demonstrate the pipeline rather than establish real-world "
        "solar-generation accuracy. This model has not been validated against "
        "real measured solar production."
    )

    return df


def get_prediction_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a summary of prediction results.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction result DataFrame from predict_solar_output().

    Returns
    -------
    dict
        Summary statistics including total predicted output, average, min, max.
    """
    if "predicted_solar_output" not in df.columns:
        raise ValueError("DataFrame does not contain predicted_solar_output column")

    predictions = df["predicted_solar_output"]
    summary = {
        "total_predicted_solar_output": float(np.sum(predictions)),
        "average_predicted_solar_output": float(np.mean(predictions)),
        "min_predicted_solar_output": float(np.min(predictions)),
        "max_predicted_solar_output": float(np.max(predictions)),
        "num_predictions": len(df),
        "non_negative_count": int((predictions >= 0).sum()),
        "negative_count": int((predictions < 0).sum()),
    }
    return summary