"""Train and evaluate the solar prediction models.

Runs the full pipeline: load the generated dataset, engineer features,
split chronologically (50 days train / 10 days validation), tune and compare
models, then save the selected model and metadata to artifacts/.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.solar_prediction.model_training import train_and_evaluate


def main():
    result = train_and_evaluate()
    metrics = result["train_metrics"]
    print("\nTraining complete")
    print(f"Selected model: {result['selected_model_name']}")
    print(f"Validation MAE:  {metrics['mae']:.2f}")
    print(f"Validation RMSE: {metrics['rmse']:.2f}")
    print(f"Validation R²:   {metrics['r2']:.4f}")
    print(
        f"Train period: {result['train_date_range'][0]} to "
        f"{result['train_date_range'][1]}"
    )
    print(
        f"Validation period: {result['val_date_range'][0]} to "
        f"{result['val_date_range'][1]}"
    )


if __name__ == "__main__":
    main()
