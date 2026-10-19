"""Data processing and feature engineering for solar power prediction."""

import pandas as pd
import numpy as np

RNG_SEED = 42


class FeatureSchema:
    """Authoritative feature schema shared by training and prediction."""

    # Core weather features (in this order)
    WEATHER_FEATURES = ["temperature", "humidity", "cloud_cover", "wind_speed", "precipitation"]

    # Cyclical time features
    HOUR_SINE = "hour_sine"
    HOUR_COSINE = "hour_cosine"
    DAY_OF_WEEK = "day_of_week"
    WEEKEND = "weekend"

    # Target variable
    TARGET = "solar_output"

    # Full feature list in order used for modeling
    ALL_FEATURES = (
        WEATHER_FEATURES
        + [HOUR_SINE, HOUR_COSINE, DAY_OF_WEEK, WEEKEND]
    )

    @classmethod
    def get_feature_names(cls):
        return list(cls.ALL_FEATURES)


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps into proper datetime values.

    Expects a 'timestamp' column that can be parsed by pd.to_datetime.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Sort records chronologically by timestamp."""
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect duplicate timestamps and handle them consistently.

    Duplicates are kept as the first occurrence; a count is added to help
    with debugging. Returns a DataFrame with a duplicate_info column.
    """
    df = df.copy()
    dup_mask = df["timestamp"].duplicated(keep="first")
    duplicate_count = int(dup_mask.sum())
    if duplicate_count > 0:
        # Keep first occurrence, drop subsequent duplicates
        df = df[~dup_mask].reset_index(drop=True)
    return df, duplicate_count


def detect_missing_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Detect missing hourly timestamps and reindex the data where appropriate.

    Returns the data reindexed to a complete hourly frequency, with NaNs
    introduced for any missing timestamps.
    """
    df = df.copy()
    # Set timestamp as index for reindexing
    df_indexed = df.set_index("timestamp")
    # Create complete hourly range
    full_range = pd.date_range(
        start=df_indexed.index.min(),
        end=df_indexed.index.max(),
        freq="h",
    )
    # Reindex to full range, introducing NaNs for missing hours
    df_reindexed = df_indexed.reindex(full_range).reset_index()
    df_reindexed = df_reindexed.rename(columns={"index": "timestamp"})
    return df_reindexed


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using sensible, documented strategies.

    Strategy:
    - Numeric columns: fill with median of that column (robust to outliers)
    - If any NaNs remain after filling (e.g. from reindexing), fill with 0
      and document the imputation.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Fill any remaining NaNs (shouldn't happen after the above, but be safe)
    df = df.fillna(0)
    return df


def compute_outlier_bounds(df: pd.DataFrame) -> dict:
    """Compute IQR-based capping bounds for each weather feature.

    The bounds are computed once on the training block so validation and
    inference data are capped with exactly the same thresholds.
    """
    numeric_cols = ["temperature", "humidity", "cloud_cover", "wind_speed", "precipitation"]
    bounds = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        if pd.notna(lower_bound) and pd.notna(upper_bound):
            bounds[col] = (lower_bound, upper_bound)
    return bounds


def handle_outliers_conservatively(
    df: pd.DataFrame,
    bounds: dict = None,
) -> pd.DataFrame:
    """Handle outliers conservatively without arbitrarily deleting valid observations.

    Uses IQR-based capping: values beyond 1.5*IQR from quartiles are capped
    at the boundary, preserving the observation but reducing the influence of
    extreme values. This is applied separately to each weather feature.

    Parameters
    ----------
    df : pd.DataFrame
        Data to cap.
    bounds : dict, optional
        Pre-computed bounds from ``compute_outlier_bounds``. When omitted,
        bounds are fitted on ``df``. Training passes bounds fitted on the
        training block so validation data never influences the thresholds.
    """
    df = df.copy()

    if bounds is None:
        bounds = compute_outlier_bounds(df)

    for col, (lower_bound, upper_bound) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower_bound, upper_bound)

    return df


def engineer_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer cyclical time features from timestamp.

    Hour and day-of-week are converted to sine/cosine pairs to preserve
    the circular nature of time (e.g., hour 23 is close to hour 0).

    Features added:
    - hour_sine: sin(2 * pi * hour / 24)
    - hour_cosine: cos(2 * pi * hour / 24)
    - day_of_week: integer 0=Monday..6=Sunday
    - weekend: 1 if Saturday or Sunday, 0 otherwise
    """
    df = df.copy()
    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Hour of day (0-23) as a plain integer feature for inspection/analysis
    hour = df["timestamp"].dt.hour
    df["hour_of_day"] = hour.astype(int)

    # Cyclical hour features
    df[FeatureSchema.HOUR_SINE] = np.sin(2 * np.pi * hour / 24)
    df[FeatureSchema.HOUR_COSINE] = np.cos(2 * np.pi * hour / 24)

    # Day of week (Monday=0 to Sunday=6, matching pandas)
    df[FeatureSchema.DAY_OF_WEEK] = df["timestamp"].dt.dayofweek

    # Weekend indicator (Saturday=5, Sunday=6)
    df[FeatureSchema.WEEKEND] = df[FeatureSchema.DAY_OF_WEEK].isin([5, 6]).astype(int)

    return df


def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer derived weather features.

    Currently, the raw weather features (temperature, humidity, cloud_cover,
    wind_speed, precipitation) are retained as-is. Derived features could
    include heat index, wind chill, etc., but the raw features are the core
    inputs and must remain consistent between training and inference.
    """
    df = df.copy()

    # Ensure numeric types
    for col in ["temperature", "humidity", "cloud_cover", "wind_speed", "precipitation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_schema(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    """Validate that DataFrame has required columns.

    Raises ValueError if required columns are missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def clean_and_engineer_features(
    df: pd.DataFrame,
    reindex: bool = True,
    apply_outliers: bool = True,
    outlier_bounds: dict = None,
) -> pd.DataFrame:
    """Full cleaning and feature engineering pipeline.

    Applies in order:
    1. Parse timestamps
    2. Sort chronologically
    3. Detect and handle duplicate timestamps
    4. Optionally reindex to complete hourly frequency
    5. Handle missing values (median imputation)
    6. Handle outliers conservatively (IQR capping, optional)
    7. Engineer cyclical time features
    8. Engineer weather features

    When training a model, pass ``apply_outliers=False`` here, split the
    cleaned data chronologically, then cap both blocks with bounds fitted on
    the training block only.

    Returns a cleaned, feature-engineered DataFrame ready for modeling.
    """
    df = parse_timestamps(df)
    df = sort_chronologically(df)
    df, dup_count = detect_duplicates(df)
    if dup_count > 0:
        print(f"Removed {dup_count} duplicate timestamp(s)")

    if reindex:
        df = detect_missing_hours(df)
        missing_after = int(df.isnull().sum().sum())
        if missing_after > 0:
            print(f"Introduced NaN for {missing_after} missing hourly entries")

    df = handle_missing_values(df)
    if apply_outliers:
        df = handle_outliers_conservatively(df, bounds=outlier_bounds)
    df = engineer_cyclical_time_features(df)
    df = engineer_weather_features(df)

    return df


# Convenience function for loading the generated dataset
def load_dataset(data_path: str = None) -> pd.DataFrame:
    """Load the generated solar-weather dataset.

    If data_path is None, loads from the project default location.
    Parses the timestamp column as datetime.
    """
    from .config import DATA_PATH
    if data_path is None:
        data_path = DATA_PATH
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    return df


# For standalone use
if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded dataset: {len(df)} rows")
    df = clean_and_engineer_features(df)
    print(f"After engineering: {len(df)} rows, {len(df.columns)} columns")
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nFeature summary:")
    for col in FeatureSchema.ALL_FEATURES:
        if col in df.columns:
            print(f"  {col}: median={df[col].median():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")