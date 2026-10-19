"""Tests for weather service module.

Uses mocked HTTP requests to validate behavior without depending
on external internet connectivity. Tests successful responses,
error handling, and forecast transformation.
"""

import pytest
from unittest.mock import patch, MagicMock

import pandas as pd

from src.solar_prediction.weather_service import (
    fetch_hourly_forecasts,
    forecast_to_model_features,
    _normalize_forecast_data,
    _validate_api_response,
    get_location_info,
)


# ============================================================================
# Mocked API response fixtures
# ============================================================================


@pytest.fixture
def mock_successful_response():
    """Mock successful Open-Meteo API response for 24-hour forecast."""
    times = []
    temps = []
    hums = []
    clouds = []
    winds = []
    precs = []

    for i in range(24):
        hour_str = f"2026-10-19T{(11 + i):02d}:00:00+05:30"
        times.append(hour_str)
        temps.append(31.8 + i * 0.1)
        hums.append(40.0 + i * 0.5)
        clouds.append(34.0 + i * 0.2)
        winds.append(9.2 + i * 0.1)
        precs.append(0.0 if i < 24 else (0.1 if i % 10 == 0 else 0.0))

    return {
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "relative_humidity_2m": hums,
            "cloud_cover": clouds,
            "wind_speed_10m": winds,
            "precipitation": precs,
        }
    }


@pytest.fixture
def mock_48_hour_response():
    """Mock successful Open-Meteo API response for 48-hour forecast."""
    times = []
    temps = []
    hums = []
    clouds = []
    winds = []
    precs = []

    for i in range(48):
        hour = 11 + i
        hour_str = (
            f"2026-10-19T{hour:02d}:00:00+05:30" if hour <= 23
            else f"2026-10-20T{hour-24:02d}:00:00+05:30"
        )
        times.append(hour_str)
        temps.append(30.0 + i * 0.1)
        hums.append(40.0 + i * 0.5)
        clouds.append(30.0 + i * 0.2)
        winds.append(5.0 + i * 0.1)
        precs.append(0.0 if i < 24 else (0.1 if i % 10 == 0 else 0.0))

    return {
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "relative_humidity_2m": hums,
            "cloud_cover": clouds,
            "wind_speed_10m": winds,
            "precipitation": precs,
        }
    }


# ============================================================================
# Fetch hourly forecasts tests
# ============================================================================


@patch("src.solar_prediction.weather_service.requests.get")
def test_fetch_hourly_forecasts_successful(mock_get, mock_successful_response):
    """Test that successful API response is parsed correctly."""
    # Configure the mock
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = mock_successful_response
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    result = fetch_hourly_forecasts(hours=24)

    assert "forecast_data" in result
    assert len(result["forecast_data"]) == 24
    assert result["source"] == "open-meteo"
    assert "retrieval_time" in result


@patch("src.solar_prediction.weather_service.requests.get")
def test_fetch_hourly_forecasts_uses_cache(mock_get, mock_successful_response):
    """Test that cache is used on second call."""
    # Configure the mock for first call
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = mock_successful_response
    mock_response.status_code = 200

    # Configure the mock for second call (cache hit)
    mock_response2 = MagicMock()
    mock_response2.raise_for_status.return_value = None
    mock_response2.json.return_value = mock_successful_response
    mock_response2.status_code = 200

    # First call - should hit the API and populate cache
    result1 = fetch_hourly_forecasts(hours=24, use_cache=True)
    # Second call should use cache
    result2 = fetch_hourly_forecasts(hours=24, use_cache=True)

    assert len(result1["forecast_data"]) == 24
    assert len(result2["forecast_data"]) == 24


@patch("src.solar_prediction.weather_service.requests.get")
def test_fetch_hourly_forecasts_48_hours(mock_get, mock_48_hour_response):
    """Test 48-hour forecast retrieval."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = mock_48_hour_response
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    result = fetch_hourly_forecasts(hours=48)

    assert len(result["forecast_data"]) == 48


@patch("src.solar_prediction.weather_service.requests.get")
def test_fetch_hourly_forecasts_timeout(mock_get):
    """Test that timeout errors are propagated."""
    import requests

    # Set up the mock to raise a Timeout exception
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.Timeout("Request timed out")
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    # Call the function and verify Timeout propagates
    with pytest.raises(requests.exceptions.Timeout):
        fetch_hourly_forecasts(hours=24, use_cache=False)


@patch("src.solar_prediction.weather_service.requests.get")
def test_fetch_hourly_forecasts_connection_error(mock_get):
    """Test that connection errors are propagated."""
    import requests

    # Set up the mock to raise a ConnectionError exception
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.ConnectionError("Connection refused")
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    # Call the function and verify ConnectionError propagates
    with pytest.raises(requests.exceptions.ConnectionError):
        fetch_hourly_forecasts(hours=24, use_cache=False)


@patch("src.solar_prediction.weather_service.requests.get")
def test_fetch_hourly_forecasts_http_error(mock_get):
    """Test that HTTP errors are propagated."""
    import requests

    # Set up the mock to raise an HTTPError exception
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    # Call the function and verify HTTPError propagates
    with pytest.raises(requests.exceptions.HTTPError):
        fetch_hourly_forecasts(hours=24, use_cache=False)


# ============================================================================
# API response validation tests
# ============================================================================


def test_validate_api_response_valid():
    """Test that valid API response passes validation."""
    valid_data = {
        "hourly": {
            "time": ["2026-10-19T11:00:00+05:30"],
            "temperature_2m": [31.8],
            "relative_humidity_2m": [40.0],
            "cloud_cover": [34.0],
            "wind_speed_10m": [9.2],
            "precipitation": [0.0],
        }
    }
    # Should not raise
    _validate_api_response(valid_data)


def test_validate_api_response_missing_hourly():
    """Test that missing 'hourly' field raises ValueError."""
    invalid_data = {"other": "data"}
    with pytest.raises(ValueError, match="missing 'hourly' field"):
        _validate_api_response(invalid_data)


def test_validate_api_response_missing_fields():
    """Test that missing required hourly fields raise ValueError."""
    invalid_data = {
        "hourly": {
            "time": ["2026-10-19T11:00:00+05:30"],
            # Missing many required fields
        }
    }
    with pytest.raises(ValueError, match="missing required hourly fields"):
        _validate_api_response(invalid_data)


# ============================================================================
# Forecast data normalization tests
# ============================================================================


def test_normalize_forecast_data_basic():
    """Test that forecast data is normalized correctly."""
    data = {
        "hourly": {
            "time": [
                "2026-10-19T11:00:00+05:30",
                "2026-10-19T12:00:00+05:30",
            ],
            "temperature_2m": [31.8, 32.5],
            "relative_humidity_2m": [40.0, 42.0],
            "cloud_cover": [34.0, 35.0],
            "wind_speed_10m": [9.2, 8.5],
            "precipitation": [0.0, 0.0],
        }
    }

    records = _normalize_forecast_data(data, hours=2)
    assert len(records) == 2
    assert records[0]["temperature"] == 31.8
    assert records[0]["humidity"] == 40.0
    assert records[0]["cloud_cover"] == 34.0
    assert records[0]["wind_speed"] == 9.2
    assert records[0]["precipitation"] == 0.0
    assert records[1]["timestamp"] is not None  # Should be a datetime


def test_normalize_forecast_data_validates_ranges():
    """Test that invalid values raise ValueError."""
    data = {
        "hourly": {
            "time": ["2026-10-19T11:00:00+05:30"],
            "temperature_2m": [100.0],  # Invalid: > 60
            "relative_humidity_2m": [40.0],
            "cloud_cover": [34.0],
            "wind_speed_10m": [9.2],
            "precipitation": [0.0],
        }
    }

    with pytest.raises(ValueError, match="Invalid temperature value"):
        _normalize_forecast_data(data, hours=1)


# ============================================================================
# Forecast to model features tests
# ============================================================================


def test_forecast_to_model_features_basic():
    """Test that forecast records are transformed to model features."""
    forecast_records = [
        {
            "timestamp": pd.Timestamp("2026-10-19 11:00:00"),
            "temperature": 31.8,
            "humidity": 40.0,
            "cloud_cover": 34.0,
            "wind_speed": 9.2,
            "precipitation": 0.0,
        },
        {
            "timestamp": pd.Timestamp("2026-10-19 12:00:00"),
            "temperature": 32.5,
            "humidity": 42.0,
            "cloud_cover": 35.0,
            "wind_speed": 8.5,
            "precipitation": 0.0,
        },
    ]

    X = forecast_to_model_features(forecast_records)
    assert isinstance(X, pd.DataFrame)
    assert X.shape[0] == 2
    assert X.shape[1] == 9  # 5 weather + 4 engineered features
    # Check feature names
    expected_names = [
        "temperature", "humidity", "cloud_cover", "wind_speed", "precipitation",
        "hour_sine", "hour_cosine", "day_of_week", "weekend",
    ]
    assert list(X.columns) == expected_names


def test_forecast_to_model_features_missing_fields():
    """Test that missing required fields raise ValueError."""
    incomplete_records = [
        {
            "timestamp": pd.Timestamp("2026-10-19 11:00:00"),
            "temperature": 31.8,
            # Missing other fields
        }
    ]

    with pytest.raises(ValueError, match="Missing required weather field"):
        forecast_to_model_features(incomplete_records)


# ============================================================================
# Location info tests
# ============================================================================


def test_get_location_info():
    """Test that location info returns correct Pune coordinates."""
    info = get_location_info()
    assert info["latitude"] == 18.52
    assert info["longitude"] == 73.86
    assert info["timezone"] == "Asia/Kolkata"