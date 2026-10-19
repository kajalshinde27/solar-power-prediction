"""Model training and evaluation pipeline for solar power prediction.

Trains regression models using chronological validation (first 50 days train,
final 10 days validation). Persists the selected model and metadata.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .data_processing import (
    clean_and_engineer_features,
    compute_outlier_bounds,
    handle_outliers_conservatively,
    FeatureSchema,
    load_dataset,
)
from .config import ARTIFACTS_DIR

# Chronological split: first 50 days for training, final 10 days for validation
TRAIN_DAYS = 50
VAL_DAYS = 10

# Model configuration - modest, computationally reasonable hyperparameters
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


def load_and_engineer_data(data_path: str = None) -> pd.DataFrame:
    """Load dataset and apply full feature engineering pipeline."""
    df = load_dataset(data_path=data_path)
    df_engineered = clean_and_engineer_features(df, reindex=True)
    return df_engineered


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare feature matrix X and target vector y.

    Uses the authoritative feature schema to ensure consistency.
    Returns X (DataFrame), y (Series), feature_names (list).
    """
    required_features = FeatureSchema.get_feature_names()
    X = df[required_features]
    y = df[FeatureSchema.TARGET]

    # Ensure feature order matches schema
    X = X[required_features]

    return X, y, required_features


def chronological_split(df: pd.DataFrame, train_days: int = TRAIN_DAYS) -> tuple:
    """Split data chronologically.

    First `train_days` days for training, remaining for validation.
    No shuffling - preserves temporal order.
    """
    # Sort just in case
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calculate split point based on number of rows per day (24 hours)
    train_rows = train_days * 24
    val_rows = (len(df) - train_rows) if len(df) > train_rows else 0

    if train_rows >= len(df):
        raise ValueError(
            f"Training days ({train_days}) exceeds total data duration."
        )

    train_df = df.iloc[:train_rows].reset_index(drop=True)
    val_df = df.iloc[train_rows : train_rows + VAL_DAYS * 24].reset_index(drop=True)

    return train_df, val_df


def train_random_forest(train_X, train_y) -> RandomForestRegressor:
    """Train RandomForestRegressor on training data."""
    model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    model.fit(train_X, train_y)
    return model


def train_hist_gradient_boosting(train_X, train_y) -> HistGradientBoostingRegressor:
    """Train HistGradientBoostingRegressor with a fixed baseline configuration."""
    model = HistGradientBoostingRegressor(**HIST_GRADIENT_PARAMS)
    model.fit(train_X, train_y)
    return model


def tune_hist_gradient(train_X, train_y) -> tuple:
    """Optimize HistGradientBoosting hyperparameters with time-series CV.

    A small grid is searched using ``TimeSeriesSplit`` on the training block
    only, so no validation-period data is used for tuning. Returns the best
    estimator, its parameters, and the best cross-validated MAE.
    """
    param_grid = {
        "max_iter": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
    }
    search = GridSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    search.fit(train_X, train_y)
    return search.best_estimator_, search.best_params_, float(-search.best_score_)


def evaluate_model(model, test_X, test_y) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(test_X)
    mae = mean_absolute_error(test_y, y_pred)
    rmse = np.sqrt(mean_squared_error(test_y, y_pred))
    r2 = r2_score(test_y, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "predictions": y_pred,
    }


def train_and_evaluate(df: pd.DataFrame = None) -> dict:
    """Full training pipeline: train models, evaluate, select best, persist.

    Returns dict with:
    - selected_model_name: str
    - selected_model: fitted model object
    - feature_names: list of exact ordered feature names
    - train_metrics: dict with MAE, RMSE, R2 on validation set
    - train_date_range: tuple of (start, end)
    - val_date_range: tuple of (start, end)
    - model_config: dict of hyperparameters used
    """
    # Load and engineer data. Outlier capping is deliberately deferred until
    # after the chronological split so validation rows cannot influence the
    # capping thresholds (no leakage).
    if df is None:
        raw_df = load_dataset()
        df = clean_and_engineer_features(raw_df, reindex=True, apply_outliers=False)

    # Chronological split
    train_df, val_df = chronological_split(df)

    # Fit IQR bounds on the training block only, then apply to both blocks
    outlier_bounds = compute_outlier_bounds(train_df)
    train_df = handle_outliers_conservatively(train_df, bounds=outlier_bounds)
    val_df = handle_outliers_conservatively(val_df, bounds=outlier_bounds)

    # Prepare features
    train_X, train_y, feature_names = prepare_features(train_df)
    val_X, val_y, _ = prepare_features(val_df)

    # Train models: RandomForest is the fixed baseline, HistGradientBoosting
    # has its hyperparameters optimized on the training block.
    rf_model = train_random_forest(train_X, train_y)
    hgb_model, hgb_best_params, hgb_cv_mae = tune_hist_gradient(train_X, train_y)

    # Evaluate
    rf_metrics = evaluate_model(rf_model, val_X, val_y)
    hgb_metrics = evaluate_model(hgb_model, val_X, val_y)

    # Select best model using validation R² (primary criterion)
    if hgb_metrics["r2"] > rf_metrics["r2"]:
        selected_name = "HistGradientBoosting"
        selected_model = hgb_model
        selected_metrics = hgb_metrics
    else:
        selected_name = "RandomForest"
        selected_model = rf_model
        selected_metrics = rf_metrics

    # Persist model and metadata
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model_artifact_path = os.path.join(ARTIFACTS_DIR, "solar_model.joblib")
    metadata_artifact_path = os.path.join(ARTIFACTS_DIR, "solar_model_metadata.joblib")

    # Save the model
    joblib.dump(selected_model, model_artifact_path)

    # Save metadata
    metadata = {
        "feature_names": feature_names,
        "model_name": selected_name,
        "model_config": (
            RANDOM_FOREST_PARAMS
            if selected_name == "RandomForest"
            else hgb_best_params
        ),
        "train_metrics": selected_metrics,
        "tuning_results": {
            "method": "GridSearchCV",
            "cv": "TimeSeriesSplit(n_splits=3)",
            "best_params": hgb_best_params,
            "best_cv_mae": hgb_cv_mae,
        },
        "train_date_range": (
            train_df["timestamp"].min(),
            train_df["timestamp"].max(),
        ),
        "val_date_range": (
            val_df["timestamp"].min(),
            val_df["timestamp"].max(),
        ),
        "feature_schema": FeatureSchema.get_feature_names(),
    }
    joblib.dump(metadata, metadata_artifact_path)

    print(f"Model trained and saved: {selected_name}")
    print(f"Validation metrics - MAE: {selected_metrics['mae']:.2f}, "
          f"RMSE: {selected_metrics['rmse']:.2f}, R²: {selected_metrics['r2']:.4f}")
    print(f"Training period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Validation period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")

    return {
        "selected_model_name": selected_name,
        "selected_model": selected_model,
        "feature_names": feature_names,
        "train_metrics": selected_metrics,
        "train_date_range": (train_df["timestamp"].min(), train_df["timestamp"].max()),
        "val_date_range": (val_df["timestamp"].min(), val_df["timestamp"].max()),
        "model_config": (
            RANDOM_FOREST_PARAMS
            if selected_name == "RandomForest"
            else hgb_best_params
        ),
        "tuning_results": {
            "method": "GridSearchCV",
            "cv": "TimeSeriesSplit(n_splits=3)",
            "best_params": hgb_best_params,
            "best_cv_mae": hgb_cv_mae,
        },
    }


def load_saved_model() -> dict:
    """Load the persisted model and metadata.

    Returns the same dict structure as train_and_evaluate().
    Raises FileNotFoundError with helpful message if artifacts are missing.
    """
    model_path = os.path.join(ARTIFACTS_DIR, "solar_model.joblib")
    metadata_path = os.path.join(ARTIFACTS_DIR, "solar_model_metadata.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run training pipeline first, or check that artifacts directory"
            " contains the expected files."
        )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Model metadata not found at {metadata_path}. "
            "Run training pipeline first."
        )

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)

    # Verify feature names consistency
    expected_features = metadata.get("feature_names", [])
    # Reconstruct feature names from model if possible
    print(f"Loaded model: {metadata.get('model_name')}")
    print(f"Feature names: {expected_features}")

    return {
        "model": model,
        "model_name": metadata.get("model_name"),
        "feature_names": expected_features,
        "model_config": metadata.get("model_config"),
        "train_metrics": metadata.get("train_metrics"),
        "tuning_results": metadata.get("tuning_results", {}),
        "train_date_range": metadata.get("train_date_range"),
        "val_date_range": metadata.get("val_date_range"),
    }