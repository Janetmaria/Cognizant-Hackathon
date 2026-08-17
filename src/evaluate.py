"""
Module 5 - Shared evaluation metrics for all forecasting models.

Scores predictions from src/models/*.py against data/processed/model_holdout.csv
(config.PROCESSED_HOLDOUT_CSV, config.TARGET_COL) so every model is compared
on the same metric. Do not duplicate metric logic inside individual model files.
"""

import sys
from pathlib import Path
from typing import Dict, Union, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


def calculate_wmae(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    is_holiday: Union[pd.Series, np.ndarray, None] = None,
) -> float:
    """
    Calculate Weighted Mean Absolute Error (WMAE), Kaggle's official Walmart metric.
    
    Holiday weeks receive a weight of 5, while non-holiday weeks receive a weight of 1.
    Formula: WMAE = sum(w_i * |y_true - y_pred|) / sum(w_i)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        return 0.0

    abs_errors = np.abs(y_true - y_pred)

    if is_holiday is None:
        weights = np.ones_like(y_true, dtype=np.float64)
    else:
        # Robust boolean conversion for Series/Arrays containing bools, ints (0/1), or strings
        if isinstance(is_holiday, pd.Series):
            is_holiday_bool = is_holiday.astype(bool).to_numpy()
        else:
            is_holiday_bool = np.asarray(is_holiday, dtype=bool)
        weights = np.where(is_holiday_bool, 5.0, 1.0)

    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0

    return float(np.sum(weights * abs_errors) / total_weight)


def calculate_mape(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    eps: float = 1.0,
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE) safely.
    Uses max(|y_true|, eps) in the denominator to avoid division by zero or extreme spikes on near-zero sales.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    is_holiday: Union[pd.Series, np.ndarray, None] = None,
) -> Dict[str, float]:
    """
    Compute full dictionary of forecasting evaluation metrics:
    - WMAE (Weighted MAE - official Kaggle metric)
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error %)
    - R2 (Coefficient of Determination)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        return {
            "WMAE": 0.0,
            "MAE": 0.0,
            "RMSE": 0.0,
            "MAPE (%)": 0.0,
            "R2 Score": 0.0,
        }

    wmae = calculate_wmae(y_true, y_pred, is_holiday)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = calculate_mape(y_true, y_pred)
    
    # Handle edge case where y_true variance is 0
    if len(np.unique(y_true)) <= 1:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))

    return {
        "WMAE": round(wmae, 2),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R2 Score": round(r2, 4),
    }


def evaluate_predictions(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    target_col: str = config.TARGET_COL,
    holiday_col: str = "is_holiday",
) -> Dict[str, float]:
    """
    Evaluate model predictions directly from a pandas DataFrame containing true target and prediction columns.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    if pred_col not in df.columns:
        raise KeyError(f"Prediction column '{pred_col}' not found in DataFrame.")

    # Remove missing values if any exist in target or prediction
    valid_df = df.dropna(subset=[target_col, pred_col])
    if len(valid_df) == 0:
        print(f"[WARNING] No valid rows found for evaluation (subset '{target_col}', '{pred_col}').")
        return {
            "WMAE": 0.0,
            "MAE": 0.0,
            "RMSE": 0.0,
            "MAPE (%)": 0.0,
            "R2 Score": 0.0,
        }

    is_holiday = valid_df[holiday_col] if holiday_col in valid_df.columns else None

    return calculate_metrics(valid_df[target_col], valid_df[pred_col], is_holiday)



def print_evaluation_report(model_name: str, metrics: Dict[str, float]) -> None:
    """
    Print a clean formatted evaluation report for a given model.
    """
    print("=" * 55)
    print(f" EVALUATION REPORT: {model_name.upper()}")
    print("=" * 55)
    for metric_name, value in metrics.items():
        print(f"  {metric_name:<12} : {value}")
    print("=" * 55)


def compare_models(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Combine metrics from multiple models into a single comparison DataFrame.
    
    Args:
        results_dict: Dictionary where key is model name and value is dictionary of metrics.
                      e.g., {'Baseline': {'WMAE': 1500.0, ...}, 'LightGBM': {'WMAE': 1200.0, ...}}
    Returns:
        pd.DataFrame sorted by WMAE (ascending).
    """
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    if "WMAE" in df.columns:
        df = df.sort_values(by="WMAE", ascending=True)
    return df


if __name__ == "__main__":
    print("Testing evaluate.py with sample forecasting data...")
    # Synthetic test to verify calculation logic
    np.random.seed(config.RANDOM_SEED)
    sample_y_true = np.array([1000.0, 1500.0, 2000.0, 5000.0, 1200.0])
    sample_y_pred = np.array([1050.0, 1400.0, 2100.0, 4800.0, 1250.0])
    sample_holiday = np.array([False, False, False, True, False])

    test_metrics = calculate_metrics(sample_y_true, sample_y_pred, sample_holiday)
    print_evaluation_report("Sample Test Run", test_metrics)

