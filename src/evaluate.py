"""
Module 5 - Shared evaluation metrics for all forecasting models.

Scores predictions from src/models/*.py against data/processed/model_holdout.csv
(config.PROCESSED_HOLDOUT_CSV, config.TARGET_COL) so every model is compared
on the same metric. Do not duplicate metric logic inside individual model files.

Metrics reported: MAE, RMSE, MAPE (%), R2 Score.
(WMAE has been removed — it was a Kaggle-Walmart-specific holiday-weighted metric
that isn't part of this project's evaluation criteria. The `holiday_col` parameter
is still accepted, but ignored, so existing calls from sarima_model.py / prophet_model.py
that pass holiday_col="is_holiday" keep working without modification.)
"""

import sys
from pathlib import Path
from typing import Dict, Union, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


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
) -> Dict[str, float]:
    """
    Compute full dictionary of forecasting evaluation metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error %)
    - R2 (Coefficient of Determination)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        return {
            "MAE": 0.0,
            "RMSE": 0.0,
            "MAPE (%)": 0.0,
            "R2 Score": 0.0,
        }

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = calculate_mape(y_true, y_pred)

    # Handle edge case where y_true variance is 0
    if len(np.unique(y_true)) <= 1:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R2 Score": round(r2, 4),
    }


def evaluate_predictions(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    target_col: str = config.TARGET_COL,
    holiday_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate model predictions directly from a pandas DataFrame containing true target and prediction columns.

    Note: `holiday_col` is accepted for backward compatibility with existing calls
    (e.g. sarima_model.py / prophet_model.py pass holiday_col="is_holiday") but is
    no longer used internally now that WMAE has been removed.
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
            "MAE": 0.0,
            "RMSE": 0.0,
            "MAPE (%)": 0.0,
            "R2 Score": 0.0,
        }

    return calculate_metrics(valid_df[target_col], valid_df[pred_col])


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
                      e.g., {'Baseline': {'MAPE (%)': 138.8, ...}, 'Hybrid Model': {'MAPE (%)': 95.2, ...}}
    Returns:
        pd.DataFrame sorted by MAPE (%) ascending (lower MAPE = better).
    """
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    if "MAPE (%)" in df.columns:
        df = df.sort_values(by="MAPE (%)", ascending=True)
    return df


def run_all_evaluations(max_groups: Optional[int] = None) -> pd.DataFrame:
    """
    Run evaluation across:
      - Seasonal-Naive Baseline
      - Standalone Statistical Base (whichever of SARIMA / Prophet was selected as better)
      - Stacked Hybrid Model (Best Statistical Model + LightGBM Residual Stacking)
    """
    results = {}

    # Lazy imports to avoid circular dependencies
    from src.models.baseline import run_baseline_pipeline
    from src.models.lightgbm_model import run_hybrid_pipeline

    print("\n" + "=" * 65)
    print(" EVALUATING FORECASTING MODELS LEADERBOARD")
    print("=" * 65)

    # 1. Seasonal Naive Baseline
    _, baseline_metrics = run_baseline_pipeline()
    if baseline_metrics:
        results["Seasonal-Naive Baseline"] = baseline_metrics

    # 2. Stacked Hybrid Model (Best of SARIMA/Prophet + LightGBM residuals)
    hybrid_pred_df, hybrid_metrics = run_hybrid_pipeline(max_groups=max_groups)
    if hybrid_metrics:
        results["Stacked Hybrid (Best Stat Model + LightGBM)"] = hybrid_metrics

        # Evaluate the standalone statistical base that was actually selected
        # (lightgbm_model.py stores which one won in pred_df.attrs["stat_model_used"])
        if "stat_prediction" in hybrid_pred_df.columns:
            stat_model_name = hybrid_pred_df.attrs.get("stat_model_used", "Statistical Model")
            stat_metrics = evaluate_predictions(hybrid_pred_df, pred_col="stat_prediction")
            results[f"Standalone {stat_model_name} (Selected Base)"] = stat_metrics

    if results:
        comparison_df = compare_models(results)
        print("\n" + "=" * 65)
        print(" MODEL PERFORMANCE COMPARISON SUMMARY")
        print("=" * 65)
        print(comparison_df.to_string())
        print("=" * 65 + "\n")
        return comparison_df
    else:
        print("[WARNING] No model evaluation results produced (data files may be missing).")
        return pd.DataFrame()


if __name__ == "__main__":
    print("Running shared evaluation module test...")
    if config.PROCESSED_TRAIN_CSV.exists() and config.PROCESSED_HOLDOUT_CSV.exists():
        run_all_evaluations()
    else:
        print("Processed CSVs not found. Testing evaluate.py metric functions with sample data...")
        np.random.seed(config.RANDOM_SEED)
        sample_y_true = np.array([1000.0, 1500.0, 2000.0, 5000.0, 1200.0])
        sample_y_pred = np.array([1050.0, 1400.0, 2100.0, 4800.0, 1250.0])

        test_metrics = calculate_metrics(sample_y_true, sample_y_pred)
        print_evaluation_report("Sample Test Run", test_metrics)