"""Configuration for solar power prediction application."""

import os

# Project root: two levels above src/solar_prediction
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROJECT_ROOT = BASE_DIR

# Data paths
DATA_PATH = os.path.join(BASE_DIR, "data", "solar_weather_dataset.csv")

# Artifacts directory for model persistence
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Location constants (Pune, Maharashtra, India)
LATITUDE = 18.52
LONGITUDE = 73.86
TIMEZONE = "Asia/Kolkata"

# Chronological split settings
TRAIN_DAYS = 50
VAL_DAYS = 10
TOTAL_DAYS = 60  # TRAIN_DAYS + VAL_DAYS

# Forecast horizon hours
FORECAST_HOURS_DEFAULT = 24
FORECAST_HOURS_MAX = 48

# Model parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

HIST_GRADIENT_PARAMS = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": 42,
}

# Feature schema
FEATURE_SCHEMA = [
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

TARGET = "solar_output"