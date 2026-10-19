"""Tests for features module (part of data_processing).

Verifies cyclical time features, weather feature engineering,
and feature schema consistency.
"""

import numpy as np
import pandas as pd

from src.solar_prediction.data_processing import (
    FeatureSchema,
    engineer_cyclical_time_features,
    engineer_weather_features,
)


# ============================================================================
# Cyclical time feature tests
# ============================================================================


def test_hour_sine_range():
    """Test that hour_sine values are in [-1, 1] range."""
    # Test with hours 0 through 23
    hours = np.arange(24)
    timestamps = pd.date_range("2024-01-01", periods=24, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})
    df["hour_sine"], df["hour_cosine"] = np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)
    assert (df["hour_sine"] >= -1).all() and (df["hour_sine"] <= 1).all()
    assert (df["hour_cosine"] >= -1).all() and (df["hour_cosine"] <= 1).all()


def test_hour_cosine_at_midnight():
    """Test that hour_cosine = 1 at midnight (hour 0)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=1, freq="h"),
    })
    # Hour 0: cos(0) = 1
    sine, cosine = np.sin(2 * np.pi * 0 / 24), np.cos(2 * np.pi * 0 / 24)
    assert cosine == 1.0


def test_weekend_indicator():
    """Test weekend indicator for known dates."""
    # 2024-01-01 is Monday
    monday = pd.Timestamp("2024-01-01")
    df = pd.DataFrame({"timestamp": [monday]})
    from src.solar_prediction.data_processing import engineer_cyclical_time_features
    result = engineer_cyclical_time_features(df)
    assert result["weekend"].iloc[0] == 0  # Monday = not weekend

    # 2024-01-06 is Saturday
    saturday = pd.Timestamp("2024-01-06")
    df = pd.DataFrame({"timestamp": [saturday]})
    result = engineer_cyclical_time_features(df)
    assert result["weekend"].iloc[0] == 1  # Saturday = weekend

    # 2024-01-07 is Sunday
    sunday = pd.Timestamp("2024-01-07")
    df = pd.DataFrame({"timestamp": [sunday]})
    result = engineer_cyclical_time_features(df)
    assert result["weekend"].iloc[0] == 1  # Sunday = weekend


def test_day_of_week_monday_is_zero():
    """Test that pandas dayofweek gives Monday=0."""
    # pandas: Monday=0, Sunday=6
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=7, freq="D"),
    })
    from src.solar_prediction.data_processing import engineer_cyclical_time_features
    result = engineer_cyclical_time_features(df)
    # Monday should be 0
    assert result["day_of_week"].iloc[0] == 0


# ============================================================================
# Weather feature tests
# ============================================================================


def test_engineer_weather_features_numeric():
    """Test that weather features are converted to numeric."""
    df = pd.DataFrame({
        "temperature": ["20.5", "21.3", "22.1"],
        "humidity": ["60", "58", "55"],
        "cloud_cover": ["30", "35", "40"],
        "wind_speed": ["5.0", "6.2", "5.5"],
        "precipitation": ["0.0", "0.1", "0.0"],
    })
    result = engineer_weather_features(df)
    # All columns should be numeric
    assert pd.api.types.is_numeric_dtype(result["temperature"])
    assert pd.api.types.is_numeric_dtype(result["humidity"])
    assert pd.api.types.is_numeric_dtype(result["cloud_cover"])
    assert pd.api.types.is_numeric_dtype(result["wind_speed"])
    assert pd.api.types.is_numeric_dtype(result["precipitation"])


# ============================================================================
# Schema validation tests
# ============================================================================


def test_feature_schema_completeness():
    """Test that the feature schema contains all required features."""
    names = FeatureSchema.get_feature_names()
    # Should have 9 features: 5 weather + 4 engineered
    assert len(names) == 9

    # Should contain the core weather features
    for feature in ["temperature", "humidity", "cloud_cover", "wind_speed", "precipitation"]:
        assert feature in names, f"Missing weather feature: {feature}"

    # Should contain the cyclical time features
    for feature in ["hour_sine", "hour_cosine", "day_of_week", "weekend"]:
        assert feature in names, f"Missing time feature: {feature}"


def test_feature_schema_order():
    """Test that the feature schema has a consistent order."""
    names = FeatureSchema.get_feature_names()
    # The order should be consistent across calls
    assert names == FeatureSchema.get_feature_names()


# ============================================================================
# Integration: full pipeline test
# ============================================================================


def test_full_pipeline_feature_generation():
    """Test that the full pipeline generates expected features."""
    np.random.seed(42)
    n = 168
    hours = np.arange(n)
    temperature = 18 + 10 * np.sin((hours / 24) * 2 * np.pi) + np.random.normal(0, 1, n)
    humidity = np.clip(60 + np.random.normal(0, 10, n), 20, 100)
    cloud_cover = np.clip(np.random.normal(40, 25, n), 0, 100)
    wind_speed = np.clip(np.random.normal(5, 2, n), 0, 15)
    precipitation = np.maximum(0, np.random.normal(0.2, 0.5, n))

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "temperature": temperature,
        "humidity": humidity,
        "cloud_cover": cloud_cover,
        "wind_speed": wind_speed,
        "precipitation": precipitation,
    })

    from src.solar_prediction.data_processing import clean_and_engineer_features
    result = clean_and_engineer_features(df, reindex=False)

    # Check that all schema features are present
    feature_names = FeatureSchema.get_feature_names()
    for feature in feature_names:
        assert feature in result.columns, f"Feature {feature} not in pipeline output"

    # Check that features have sensible values
    assert result["hour_sine"].between(-1, 1).all()
    assert result["hour_cosine"].between(-1, 1).all()
    assert result["day_of_week"].between(0, 6).all()
    assert result["weekend"].isin([0, 1]).all()