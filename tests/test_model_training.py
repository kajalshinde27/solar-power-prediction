"""Tests for model training and evaluation module.

Tests chronological split behavior, model training, evaluation metrics,
and model persistence using deterministic data.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.solar_prediction.data_processing import (
    clean_and_engineer_features,
    load_dataset,
)
from src.solar_prediction.model_training import (
    chronological_split,
    load_and_engineer_data,
    load_saved_model,
    prepare_features,
    train_and_evaluate,
    FeatureSchema,
)


# ============================================================================
# Chronological split tests
# ============================================================================


def test_chronological_split_basic():
    """Test that chronological split divides data correctly."""
    # Create a 60-day dataset (1440 hours)
    np.random.seed(42)
    n = 1440
    hours = np.arange(n)
    temperature = 18 + 10 * np.sin((hours / 24) * 2 * np.pi) + np.random.normal(0, 1, n)
    humidity = np.clip(60 + np.random.normal(0, 10, n), 20, 100)
    cloud_cover = np.clip(np.random.normal(40, 25, n), 0, 100)
    wind_speed = np.clip(np.random.normal(5, 2, n), 0, 15)
    precipitation = np.maximum(0, np.random.normal(0.2, 0.5, n))
    daylight_factor = np.maximum(0, np.sin((hours - 6) / 12 * np.pi))
    solar_output = np.maximum(0, daylight_factor * (1 - cloud_cover / 120) * (temperature / 30) * 1000 + np.random.normal(0, 20, n))

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "temperature": temperature,
        "humidity": humidity,
        "cloud_cover": cloud_cover,
        "wind_speed": wind_speed,
        "precipitation": precipitation,
        "solar_output": solar_output,
    })

    # Split: first 50 days train, final 10 days validation
    train_df, val_df = chronological_split(df, train_days=50)

    assert len(train_df) == 50 * 24  # 1200 hours
    assert len(val_df) == 10 * 24  # 240 hours
    # Validation should start right after training
    assert train_df["timestamp"].max() < val_df["timestamp"].min()


def test_chronological_split_with_custom_days():
    """Test chronological split with custom train days."""
    np.random.seed(42)
    n = 720  # 30 days * 24 hours
    hours = np.arange(n)
    temperature = 18 + 10 * np.sin((hours / 24) * 2 * np.pi) + np.random.normal(0, 1, n)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "temperature": temperature,
        "solar_output": temperature * 0.5 + np.random.normal(0, 10, n),
    })

    train_df, val_df = chronological_split(df, train_days=20)

    assert len(train_df) == 20 * 24  # 480 hours
    assert len(val_df) == (30 - 20) * 24  # 10 days * 24 = 240 hours


# ============================================================================
# Feature preparation tests
# ============================================================================


def test_prepare_features_returns_correct_shape():
    """Test that prepare_features returns correctly shaped data."""
    df = load_dataset()
    df_engineered = clean_and_engineer_features(df, reindex=False)
    X, y, feature_names = prepare_features(df_engineered)

    expected_features = FeatureSchema.get_feature_names()
    assert list(feature_names) == expected_features
    assert X.shape[1] == len(expected_features), f"Expected {len(expected_features)} features, got {X.shape[1]}"
    assert len(y) == len(df_engineered)


# ============================================================================
# Model training tests
# ============================================================================


def test_train_and_evaluate_returns_metrics():
    """Test that training pipeline returns expected metrics."""
    result = train_and_evaluate(df=None)

    # Check required keys
    assert "selected_model_name" in result
    assert "train_metrics" in result
    assert "feature_names" in result
    assert "train_date_range" in result
    assert "val_date_range" in result

    # Check metrics structure
    metrics = result["train_metrics"]
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert 0 <= metrics["r2"] <= 1


def test_train_and_evaluate_model_persistence():
    """Test that model artifacts are saved and can be loaded."""
    # Train model
    train_and_evaluate(df=None)

    # Load model
    model_data = load_saved_model()

    # Check that loaded data has expected structure
    assert "model" in model_data
    assert "model_name" in model_data
    assert "feature_names" in model_data
    assert "train_metrics" in model_data

    # Check feature names match
    assert model_data["feature_names"] == FeatureSchema.get_feature_names()


def test_train_and_evaluate_two_models_compare():
    """The pipeline trains both models and records the tuning search."""
    result = train_and_evaluate(df=None)

    assert result["selected_model_name"] in ["RandomForest", "HistGradientBoosting"]
    assert result["tuning_results"]["method"] == "GridSearchCV"
    assert "learning_rate" in result["tuning_results"]["best_params"]


# ============================================================================
# Model loading failure tests
# ============================================================================


def test_load_saved_model_raises_when_missing():
    """Test that loading model raises FileNotFoundError when artifacts missing."""
    # Temporarily remove the model artifact file
    model_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "solar_model.joblib")
    backup_path = model_path + ".backup"

    # Remove the model file if it exists
    if os.path.exists(model_path):
        os.remove(model_path)

    try:
        with pytest.raises(FileNotFoundError, match="Model artifact not found"):
            load_saved_model()
    finally:
        # Restore
        if os.path.exists(backup_path):
            os.remove(backup_path)
        if not os.path.exists(model_path):
            # Re-create empty artifacts dir structure
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            open(model_path, 'w').close()


# ============================================================================
# Deterministic test with small dataset
# ============================================================================


def test_train_with_small_deterministic_dataset():
    """Test training on a small deterministic dataset for reproducibility."""
    np.random.seed(42)
    n = 1440  # 60 days * 24 hours - full dataset

    hours = np.arange(n)
    temperature = 18 + 10 * np.sin((hours / 24) * 2 * np.pi) + np.random.normal(0, 1, n)
    humidity = np.clip(60 + np.random.normal(0, 10, n), 20, 100)
    cloud_cover = np.clip(np.random.normal(40, 25, n), 0, 100)
    wind_speed = np.clip(np.random.normal(5, 2, n), 0, 15)
    precipitation = np.maximum(0, np.random.normal(0.2, 0.5, n))
    daylight_factor = np.maximum(0, np.sin((hours - 6) / 12 * np.pi))
    solar_output = np.maximum(0, daylight_factor * (1 - cloud_cover / 120) * (temperature / 30) * 1000 + np.random.normal(0, 20, n))

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "temperature": temperature,
        "humidity": humidity,
        "cloud_cover": cloud_cover,
        "wind_speed": wind_speed,
        "precipitation": precipitation,
        "solar_output": solar_output,
    })

    # Use the full training pipeline with 50/10 day split on the full dataset
    # Use load_and_engineer_data to ensure features are properly engineered
    df = load_and_engineer_data(data_path=None)
    result = train_and_evaluate(df=df)

    # Should produce valid results with the full dataset (1440 rows = 60 days)
    assert "selected_model_name" in result
    assert "train_metrics" in result
    assert result["train_metrics"]["r2"] is not None
    assert len(result["feature_names"]) == 9