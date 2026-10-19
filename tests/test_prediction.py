"""Tests for solar prediction module.

Tests model loading, prediction generation, output validation,
and the synthetic data disclaimer.
"""

import os

import pandas as pd
import pytest

from src.solar_prediction.model_training import train_and_evaluate
from src.solar_prediction.prediction import (
    get_prediction_summary,
    load_model_and_metadata,
    predict_solar_output,
)


# ============================================================================
# Model loading tests
# ============================================================================


def test_load_model_and_metadata_returns_structure():
    """Test that loaded model metadata has expected structure."""
    # First ensure model is trained
    train_and_evaluate(df=None)

    # The model and metadata were saved by train_and_evaluate.
    # Load them using joblib directly to test persistence.
    import joblib
    import os
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    model_path = os.path.join(artifacts_dir, "solar_model.joblib")
    metadata_path = os.path.join(artifacts_dir, "solar_model_metadata.joblib")

    assert os.path.exists(model_path), f"Model artifact not found at {model_path}"
    assert os.path.exists(metadata_path), f"Model metadata not found at {metadata_path}"

    # Load model and metadata
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)

    # Verify model type and metadata structure
    assert hasattr(model, "predict")  # model has predict method
    assert "model_name" in metadata
    assert "feature_names" in metadata
    assert "train_metrics" in metadata
    assert "train_date_range" in metadata
    assert "val_date_range" in metadata

    # Check feature names match schema
    expected_features = [
        "temperature", "humidity", "cloud_cover", "wind_speed", "precipitation",
        "hour_sine", "hour_cosine", "day_of_week", "weekend",
    ]
    assert metadata["feature_names"] == expected_features
    assert metadata["model_name"] in ["RandomForest", "HistGradientBoosting"]


def test_load_model_and_metadata_without_training():
    """Test that loading model raises error when not trained."""
    # Clean artifacts to simulate untrained state
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    # Remove only the model file, keep metadata or vice versa
    model_path = os.path.join(artifacts_dir, "solar_model.joblib")
    if os.path.exists(model_path):
        os.remove(model_path)

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model_and_metadata()


# ============================================================================
# Prediction generation tests
# ============================================================================


def test_predict_solar_output_returns_dataframe():
    """Test that predict_solar_output returns a DataFrame with expected columns."""
    # Ensure model is trained
    train_and_evaluate(df=None)

    pred_df = predict_solar_output(forecast_horizon_hours=24)

    assert isinstance(pred_df, pd.DataFrame)
    expected_columns = ["timestamp", "temperature", "humidity", "cloud_cover",
                        "wind_speed", "precipitation", "predicted_solar_output"]
    for col in expected_columns:
        assert col in pred_df.columns, f"Missing column: {col}"

    # Should have 24 rows for 24-hour forecast
    assert len(pred_df) == 24


def test_predict_solar_output_non_negative():
    """Test that all predictions are non-negative (clipped)."""
    train_and_evaluate(df=None)
    pred_df = predict_solar_output(forecast_horizon_hours=24)

    predictions = pred_df["predicted_solar_output"].values
    assert (predictions >= 0).all(), "Some predictions are negative despite clipping"


def test_predict_solar_output_with_48_hours():
    """Test prediction with 48-hour horizon."""
    train_and_evaluate(df=None)
    pred_df = predict_solar_output(forecast_horizon_hours=48)

    assert isinstance(pred_df, pd.DataFrame)
    assert len(pred_df) == 48
    assert (pred_df["predicted_solar_output"].values >= 0).all()


def test_predict_solar_output_has_disclaimer():
    """Test that synthetic data disclaimer is in prediction attrs."""
    train_and_evaluate(df=None)
    pred_df = predict_solar_output(forecast_horizon_hours=24)

    disclaimer = pred_df.attrs.get("synthetic_data_disclaimer", "")
    assert bool(disclaimer), "Synthetic data disclaimer should be present in prediction attrs"


# ============================================================================
# Prediction summary tests
# ============================================================================


def test_get_prediction_summary_basic():
    """Test that prediction summary returns expected statistics."""
    train_and_evaluate(df=None)
    pred_df = predict_solar_output(forecast_horizon_hours=24)

    summary = get_prediction_summary(pred_df)

    assert "total_predicted_solar_output" in summary
    assert "average_predicted_solar_output" in summary
    assert "min_predicted_solar_output" in summary
    assert "max_predicted_solar_output" in summary
    assert "num_predictions" in summary
    assert "non_negative_count" in summary
    assert "negative_count" in summary

    # Basic validation of summary values
    assert summary["num_predictions"] == 24
    assert summary["non_negative_count"] == 24  # All should be non-negative
    assert summary["negative_count"] == 0


# ============================================================================
# Edge case tests
# ============================================================================


def test_predict_solar_output_error_when_no_model():
    """Test that prediction fails gracefully when model not trained."""
    # Remove model artifact temporarily
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "artifacts", "solar_model.joblib"
    )
    backup_path = model_path + ".backup"

    if os.path.exists(model_path):
        os.rename(model_path, backup_path)

    try:
        with pytest.raises((FileNotFoundError, RuntimeError)):
            predict_solar_output(forecast_horizon_hours=24)
    finally:
        if os.path.exists(backup_path):
            os.rename(backup_path, model_path)


def test_get_prediction_summary_no_predicted_column():
    """Test that summary raises error without predicted column."""
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="h")})

    with pytest.raises(ValueError, match="does not contain predicted_solar_output"):
        get_prediction_summary(df)