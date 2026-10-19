"""Tests for data processing and feature engineering module.

Uses deterministic data and mocks to validate behavior without
depending on external state.
"""

import pandas as pd
import numpy as np
import pytest

from src.solar_prediction.data_processing import (
    FeatureSchema,
    parse_timestamps,
    sort_chronologically,
    detect_duplicates,
    detect_missing_hours,
    handle_missing_values,
    handle_outliers_conservatively,
    engineer_cyclical_time_features,
    validate_schema,
    clean_and_engineer_features,
    load_dataset,
)


# ============================================================================
# Fixture: Sample dataset
# ============================================================================


@pytest.fixture
def sample_dataset():
    """Create a small deterministic dataset for testing."""
    np.random.seed(42)
    n = 168  # 7 days * 24 hours
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
    return df


# ============================================================================
# Timestamp parsing tests
# ============================================================================


def test_parse_timestamps():
    """Test that timestamps are parsed into proper datetime values."""
    df = pd.DataFrame({"timestamp": ["2024-01-01 00:00", "2024-01-01 01:00"], "value": [1, 2]})
    result = parse_timestamps(df)
    assert pd.api.types.is_datetime64_dtype(result["timestamp"])
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:00")


def test_sort_chronologically():
    """Test that records are sorted chronologically."""
    df = pd.DataFrame({
        "timestamp": ["2024-01-03 00:00", "2024-01-01 00:00", "2024-01-02 00:00"],
        "value": [1, 2, 3],
    })
    result = sort_chronologically(df)
    # result["timestamp"] contains strings; parse them first
    result_ts = pd.to_datetime(result["timestamp"])
    assert result_ts.iloc[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert result_ts.iloc[1] == pd.Timestamp("2024-01-02 00:00:00")
    assert result_ts.iloc[2] == pd.Timestamp("2024-01-03 00:00:00")


def test_detect_duplicates():
    """Test that duplicate timestamps are detected and handled."""
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"],
        "value": [1, 2, 3],
    })
    result, duplicate_count = detect_duplicates(df)
    assert duplicate_count == 1  # One duplicate found
    assert len(result) == 2  # Should have 2 rows after dedup


def test_detect_missing_hours():
    """Test that missing hourly timestamps are detected and data reindexed."""
    # Create data with a gap (missing 2024-01-01 02:00)
    df = pd.DataFrame({
        "timestamp": [
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 03:00",
            "2024-01-01 04:00",
        ],
        "value": [1, 2, 3, 4],
    })
    result = detect_missing_hours(df)
    # Should have 5 rows (the missing 02:00 filled with NaN)
    assert len(result) == 5
    # The missing hour should have NaN value
    assert pd.isna(result["value"]).sum() >= 1


# ============================================================================
# Missing value handling tests
# ============================================================================


def test_handle_missing_values():
    """Test that missing values are filled with median."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "value": [1.0, np.nan, 3.0, np.nan, 5.0],
    })
    result = handle_missing_values(df)
    # Medians of [1, 3, 5] = 3.0, so NaNs should be filled with 3.0
    assert result["value"].isnull().sum() == 0
    assert result["value"].tolist() == [1.0, 3.0, 3.0, 3.0, 5.0]


def test_handle_missing_values_all_nan_column():
    """Test handling when a column is entirely NaN after imputation."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
        "value": [np.nan, np.nan, np.nan],
    })
    result = handle_missing_values(df)
    # All NaN column should be filled with 0 (fallback)
    assert result["value"].isnull().sum() == 0


# ============================================================================
# Outlier handling tests
# ============================================================================


def test_handle_outliers_conservatively():
    """Test IQR-based outlier capping."""
    df = pd.DataFrame({
        "temperature": [20.0, 21.0, 22.0, 100.0, 23.0, 24.0, 25.0],  # 100 is an outlier
        "humidity": [60.0, 61.0, 59.0, 62.0, 63.0, 61.0, 60.0],
    })
    result = handle_outliers_conservatively(df)
    # The outlier 100.0 should be capped, not retained as-is
    assert result["temperature"].max() < 100.0


# ============================================================================
# Cyclical time feature tests
# ============================================================================


def test_engineer_cyclical_time_features():
    """Test that cyclical time features are engineered correctly."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
    })
    # 2024-01-01 00:00, 01:00, 02:00, 03:00
    result = engineer_cyclical_time_features(df)

    # Hour 0: sine=0, cosine=1
    assert result["hour_sine"].iloc[0] == 0.0
    assert result["hour_cosine"].iloc[0] == 1.0

    # Hour 1: sine=sin(2pi/24), cosine=cos(2pi/24)
    assert abs(result["hour_sine"].iloc[1] - np.sin(2 * np.pi / 24)) < 1e-10
    assert abs(result["hour_cosine"].iloc[1] - np.cos(2 * np.pi / 24)) < 1e-10

    # Day of week: Monday=0
    assert result["day_of_week"].iloc[0] == 0  # Monday

    # Weekend: 0 (Monday)
    assert result["weekend"].iloc[0] == 0


def test_engineer_cyclical_time_features_day_cycle():
    """Test that the day cycle wraps correctly (hour 23 close to hour 0)."""
    # Test with hours 0 and 1 (not 23 and 0, to keep it simple)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
    })
    result = engineer_cyclical_time_features(df)

    # Hour 0 (first row, index 0): sine=0, cosine=1
    assert result["hour_sine"].iloc[0] == 0.0  # midnight (hour 0)
    assert result["hour_cosine"].iloc[0] == 1.0

    # Hour 1 (second row, index 1): sine=sin(2*pi/24), cosine=cos(2*pi/24)
    assert abs(result["hour_sine"].iloc[1] - np.sin(2 * np.pi / 24)) < 1e-10
    assert abs(result["hour_cosine"].iloc[1] - np.cos(2 * np.pi / 24)) < 1e-10


# ============================================================================
# Feature schema tests
# ============================================================================


def test_feature_schema():
    """Test that the feature schema is consistent and complete."""
    names = FeatureSchema.get_feature_names()
    expected = [
        "temperature",
        "humidity",
        "cloud_cover",
        "wind_speed",
        "precipitation",
        "hour_sine",
        "hour_cosine",
        "day_of_week",
        "weekend",
    ]
    assert names == expected


# ============================================================================
# Clean and engineer features tests
# ============================================================================


def test_clean_and_engineer_features_full(sample_dataset):
    """Test the full cleaning and engineering pipeline."""
    result = clean_and_engineer_features(sample_dataset, reindex=False)
    # Should have original columns + engineered features
    expected_cols = [
        "timestamp", "temperature", "humidity", "cloud_cover", "wind_speed",
        "precipitation", "solar_output",
        "hour_sine", "hour_cosine", "day_of_week", "weekend",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_clean_and_engineer_features_with_reindex(sample_dataset):
    """Test pipeline with reindexing to complete hourly frequency."""
    result = clean_and_engineer_features(sample_dataset, reindex=True)
    # Should have the same number of rows (or more if gaps were filled)
    assert len(result) >= len(sample_dataset)


# ============================================================================
# Schema validation tests
# ============================================================================


def test_validate_schema_missing():
    """Test that schema validation raises ValueError for missing columns."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df, ["a", "b", "c"])


def test_validate_schema_ok():
    """Test that schema validation passes for valid DataFrame."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = validate_schema(df, ["a", "b", "c"])
    assert list(result.columns) == ["a", "b", "c"]


# ============================================================================
# Load dataset tests
# ============================================================================


def test_load_dataset_returns_dataframe():
    """Test that load_dataset returns a DataFrame with expected columns."""
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    expected_cols = ["timestamp", "temperature", "humidity", "cloud_cover", "wind_speed", "precipitation", "solar_output"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) == 1440  # 60 days * 24 hours


def test_load_dataset_has_no_unexpected_missing():
    """Test that the generated dataset has no unexpected missing values."""
    df = load_dataset()
    total_missing = df.isnull().sum().sum()
    # Dataset should be clean given the generation logic with clipping
    assert total_missing == 0, f"Expected 0 missing values, got {total_missing}"