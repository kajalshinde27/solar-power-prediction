"""Generate the synthetic solar-weather dataset.

Rows run hourly from 1 Jan 2024 for 60 days (1,440 rows) with a fixed seed,
so every run produces exactly the same CSV.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "solar_weather_dataset.csv")


def generate_dataset(seed: int = 42, duration_days: int = 60):
    """Generate the dataset and save it to ``data/solar_weather_dataset.csv``.

    Parameters
    ----------
    seed : int
        Random seed (42) so the data is reproducible.
    duration_days : int
        Number of days to generate; 60 days gives 1,440 hourly rows.
    """
    np.random.seed(seed)

    rows = []
    start = datetime(2024, 1, 1)
    for i in range(24 * duration_days):
        ts = start + timedelta(hours=i)
        hour = ts.hour

        temperature = 18 + 10 * np.sin((hour / 24) * 2 * np.pi) + np.random.normal(0, 1)
        humidity = np.clip(60 + np.random.normal(0, 10), 20, 100)
        cloud_cover = np.clip(np.random.normal(40, 25), 0, 100)
        wind_speed = np.clip(np.random.normal(5, 2), 0, 15)
        precipitation = max(0, np.random.normal(0.2, 0.5))

        daylight_factor = max(0, np.sin((hour - 6) / 12 * np.pi))
        solar_output = max(
            0,
            daylight_factor * (1 - cloud_cover / 120) * (temperature / 30) * 1000,
        )
        solar_output += np.random.normal(0, 20)
        solar_output = max(0, solar_output)

        rows.append([
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            temperature,
            humidity,
            cloud_cover,
            wind_speed,
            precipitation,
            solar_output,
        ])

    df = pd.DataFrame(rows, columns=[
        "timestamp", "temperature", "humidity", "cloud_cover",
        "wind_speed", "precipitation", "solar_output",
    ])

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset created: {DATA_PATH} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    generate_dataset()
