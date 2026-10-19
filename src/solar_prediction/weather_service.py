"""Weather service module using Open-Meteo Forecast API.

Provides hourly weather forecasts for Pune, Maharashtra, India.
No API key required - uses the free Open-Meteo endpoint.
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# Open-Meteo API endpoint
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Location: Pune, Maharashtra, India
LATITUDE = 18.52
LONGITUDE = 73.86
TIMEZONE = "Asia/Kolkata"

# Default forecast horizon
DEFAULT_FORECAST_HOURS = 48

# Cache configuration
CACHE_TTL_SECONDS = 3600  # 1 hour cache

# Defaults used when the API returns null for an individual field.
DEFAULT_WEATHER_VALUES = {
    "temperature": 25.0,
    "humidity": 60.0,
    "cloud_cover": 50.0,
    "wind_speed": 5.0,
    "precipitation": 0.0,
}


class WeatherCache:
    """Simple in-memory cache for weather forecast responses."""

    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds

    def _make_key(self, latitude: float, longitude: float, hours: int) -> str:
        """Create a cache key from parameters."""
        data = f"{latitude}:{longitude}:{hours}"
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if not expired."""
        if key in self._cache:
            entry = self._cache[key]
            # Parse ISO format retrieval_time for comparison
            cached_time = datetime.fromisoformat(entry["retrieval_time"]).timestamp()
            if time.time() - cached_time < self._ttl:
                entry["access_count"] = entry.get("access_count", 0) + 1
                # Return the full cached response structure
                return entry
            else:
                # Expired, remove
                del self._cache[key]
        return None

    def set(self, key: str, data: Dict[str, Any]) -> None:
        """Set cached response with current timestamp."""
        # data is already the full response dict from fetch_hourly_forecasts
        # e.g., {"forecast_data": [...], "retrieval_time": "...", "source": "open-meteo"}
        self._cache[key] = data.copy()


# Global cache instance
_weather_cache = WeatherCache()


def _validate_positive_number(value: Any, name: str) -> float:
    """Validate that a value is a positive number."""
    try:
        v = float(value)
        if v < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return v
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a valid number, got {value}")


def _validate_timezone(tz: str) -> str:
    """Validate timezone string."""
    if not tz or not isinstance(tz, str):
        raise ValueError(f"Timezone must be a non-empty string, got {tz}")
    return tz


def fetch_hourly_forecasts(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    hours: int = DEFAULT_FORECAST_HOURS,
    timezone: str = TIMEZONE,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Fetch hourly weather forecasts from Open-Meteo API.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees.
    longitude : float
        Longitude in decimal degrees.
    hours : int
        Number of hourly forecast hours (24 or 48).
    timezone : str
        IANA timezone string (e.g., "Asia/Kolkata").
    use_cache : bool
        Whether to use response caching to avoid unnecessary API requests.

    Returns
    -------
    dict
        Parsed and validated forecast data with normalized field names.

    Raises
    ------
    requests.exceptions.Timeout
        If the API request times out.
    requests.exceptions.ConnectionError
        If there's a network connection error.
    requests.exceptions.HTTPError
        If the API returns an error status code.
    ValueError
        If the response is missing required fields or has invalid data.
    """
    # Input validation
    latitude = _validate_positive_number(latitude, "latitude")
    longitude = _validate_positive_number(longitude, "longitude")
    hours = int(_validate_positive_number(hours, "hours"))
    if hours not in (24, 48):
        raise ValueError(f"Hours must be 24 or 48, got {hours}")
    timezone = _validate_timezone(timezone)

    # Check cache first
    if use_cache:
        cache_key = _weather_cache._make_key(latitude, longitude, hours)
        cached = _weather_cache.get(cache_key)
        if cached is not None:
            print(f"Using cached forecast data (retrieved at {cached.get('retrieval_time', 'N/A')})")
            return cached

    # Build API request parameters
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,relative_humidity_2m,cloud_cover,"
            "wind_speed_10m,precipitation"
        ),
        "timezone": timezone,
        "forecast_hours": hours,
    }

    # Set timeouts for connection and read
    timeout = (5.0, 15.0)  # (connect timeout, read timeout)

    try:
        response = requests.get(
            OPEN_METEO_BASE_URL,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout(
            f"Open-Meteo API request timed out after {timeout[0]}s connect / {timeout[1]}s read"
        )
    except requests.exceptions.ConnectionError:
        raise requests.exceptions.ConnectionError(
            "Failed to connect to Open-Meteo API. "
            "Check network connectivity."
        )
    except requests.exceptions.HTTPError as e:
        # Handle specific HTTP error codes
        if response.status_code == 429:
            raise requests.exceptions.HTTPError(
                f"Open-Meteo API rate limit exceeded. "
                "Please wait before making another request."
            ) from e
        raise requests.exceptions.HTTPError(
            f"Open-Meteo API returned error {response.status_code}: "
            f"{response.text}"
        ) from e

    # Parse and validate JSON response
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        raise ValueError(
            "Failed to parse Open-Meteo API response as JSON"
        ) from e

    # Validate required fields in response
    _validate_api_response(data)

    # Normalize field names and extract hourly data
    forecast_data = _normalize_forecast_data(data, hours)

    # Cache the successful response
    if use_cache:
        cache_key = _weather_cache._make_key(latitude, longitude, hours)
        _weather_cache.set(cache_key, {"source": "open-meteo", 
            "forecast_data": forecast_data,
            "retrieval_time": datetime.now().isoformat(),
        })

    return {
        "forecast_data": forecast_data,
        "retrieval_time": datetime.now().isoformat(),
        "source": "open-meteo",
    }


def _validate_api_response(data: Dict[str, Any]) -> None:
    """Validate that the API response has the required structure.

    Raises ValueError if required fields are missing.
    """
    if "hourly" not in data:
        raise ValueError(
            "API response missing 'hourly' field. "
            "Unexpected response structure from Open-Meteo."
        )

    hourly = data["hourly"]
    required_hourly_fields = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "cloud_cover",
        "wind_speed_10m",
        "precipitation",
    ]

    missing = [f for f in required_hourly_fields if f not in hourly]
    if missing:
        raise ValueError(
            f"API response missing required hourly fields: {missing}"
        )

    # Validate that arrays have the expected length
    expected_len = len(hourly["time"])
    for field in required_hourly_fields:
        if len(hourly[field]) != expected_len:
            raise ValueError(
                f"Inconsistent array length for '{field}': "
                f"expected {expected_len}, got {len(hourly[field])}"
            )


def _value_or_default(value: Any, default: float) -> float:
    """Convert an API value to float, substituting a default for nulls/NaN."""
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def _normalize_forecast_data(data: Dict[str, Any], hours: int) -> List[Dict[str, Any]]:
    """Normalize API response into internal feature schema format.

    Maps Open-Meteo field names to the internal feature names used by the model.
    Validates and clips weather values to sensible ranges.
    """
    hourly = data["hourly"]

    # Extract time arrays
    times = hourly["time"]
    temperatures = hourly["temperature_2m"]
    humidities = hourly["relative_humidity_2m"]
    cloud_covers = hourly["cloud_cover"]
    wind_speeds = hourly["wind_speed_10m"]
    precipitations = hourly["precipitation"]

    # Validate array lengths
    if len(times) != hours:
        raise ValueError(
            f"Expected {hours} forecast hours, got {len(times)} "
            f"from API response"
        )

    # Convert timestamps and normalize values
    forecast_records = []
    for i in range(hours):
        # Parse ISO timestamp
        ts = times[i]
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            dt = datetime.now()  # Fallback, though should not happen

        # Normalize weather values. Fields the API returns as null are filled
        # with documented defaults instead of being passed to the model.
        temp = _value_or_default(temperatures[i], DEFAULT_WEATHER_VALUES["temperature"])
        humidity = _value_or_default(humidities[i], DEFAULT_WEATHER_VALUES["humidity"])
        cloud_cover = _value_or_default(cloud_covers[i], DEFAULT_WEATHER_VALUES["cloud_cover"])
        wind_speed = _value_or_default(wind_speeds[i], DEFAULT_WEATHER_VALUES["wind_speed"])
        precipitation = _value_or_default(precipitations[i], DEFAULT_WEATHER_VALUES["precipitation"])

        # Validate ranges
        if temp is not None and (temp < -50 or temp > 60):
            raise ValueError(
                f"Invalid temperature value: {temp}. "
                "Expected reasonable range -50 to 60°C"
            )
        if humidity is not None and (humidity < 0 or humidity > 100):
            raise ValueError(
                f"Invalid humidity value: {humidity}. "
                "Expected range 0 to 100%"
            )
        if cloud_cover is not None and (cloud_cover < 0 or cloud_cover > 100):
            raise ValueError(
                f"Invalid cloud_cover value: {cloud_cover}. "
                "Expected range 0 to 100%"
            )
        if wind_speed is not None and wind_speed < 0:
            raise ValueError(
                f"Invalid wind_speed value: {wind_speed}. "
                "Expected non-negative value"
            )
        if precipitation is not None and precipitation < 0:
            raise ValueError(
                f"Invalid precipitation value: {precipitation}. "
                "Expected non-negative value"
            )

        record = {
            "timestamp": dt,
            "temperature": temp,
            "humidity": humidity,
            "cloud_cover": cloud_cover,
            "wind_speed": wind_speed,
            "precipitation": precipitation,
        }
        forecast_records.append(record)

    return forecast_records


def forecast_to_model_features(
    forecast_records: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Transform forecast records into the model's expected feature format.

    Applies the same timestamp transformations and feature ordering
    used during training. Validates that all required features are present.

    Parameters
    ----------
    forecast_records : list of dict
        Forecast records from the weather service, each containing
        timestamp and weather measurements.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching the model's feature schema,
        ready for prediction.
    """
    from .data_processing import (
        engineer_cyclical_time_features,
        engineer_weather_features,
        FeatureSchema,
    )

    if not forecast_records:
        raise ValueError("No forecast records provided")

    df = pd.DataFrame(forecast_records)

    # Validate required weather fields exist
    required_weather = ["temperature", "humidity", "cloud_cover", "wind_speed", "precipitation"]
    for col in required_weather:
        if col not in df.columns:
            raise ValueError(
                f"Missing required weather field in forecast: {col}. "
                f"Available columns: {list(df.columns)}"
            )

    # Engineer cyclical time features (hour_sine, hour_cosine, day_of_week, weekend)
    df = engineer_cyclical_time_features(df)

    # Engineer weather features (ensure numeric types)
    df = engineer_weather_features(df)

    # Select and order features per the authoritative schema
    required_feature_names = FeatureSchema.get_feature_names()
    missing = [f for f in required_feature_names if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required features after engineering: {missing}. "
            f"Cannot produce predictions without complete feature set."
        )

    # Return only the features the model expects, in the correct order
    X = df[required_feature_names].copy()

    return X


def get_location_info() -> Dict[str, float]:
    """Return the location coordinates and timezone used by this service."""
    return {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
    }