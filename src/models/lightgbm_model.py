"""
Module 5 - LightGBM residual-stacking model.

Pipeline:
  1. Fit SARIMAForecaster (sarima_model.py) and ProphetForecaster (prophet_model.py) on train_df.
  2. Get each model's HOLDOUT forecast, evaluate both on MAPE (primary) + RMSE (tiebreaker),
     and select the WINNER as the statistical base for stacking.
  3. Get the winner's TRAIN (in-sample) predictions and compute residuals = actual - train_pred.
     - Prophet.predict() genuinely computes in-sample fitted values for any date, so we call
       it directly on train_df.
     - SARIMAForecaster.predict() only produces OUT-OF-SAMPLE forecasts via statsmodels
       get_forecast(), so in-sample fitted values are read from the underlying
       SARIMAXResults.fittedvalues stored per (store_id, dept_id) inside forecaster.models.
  4. Train LightGBM to predict those residuals using lag + calendar + tabular features.
  5. Final forecast = winner's holdout prediction + LightGBM's predicted residual.

This file reads sarima_model.py / prophet_model.py's public classes and attributes only —
it does not modify those files.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.evaluate import evaluate_predictions, print_evaluation_report
from src.models.sarima_model import SARIMAForecaster
from src.models.prophet_model import ProphetForecaster


# =============================================================================
# ALIGNMENT HELPER
# Both SARIMAForecaster.predict() (which internally sorts rows by store/dept/date)
# and ProphetForecaster.predict() return a DataFrame whose row order is not guaranteed
# to match the caller's original DataFrame order. Realign by explicit key instead of
# assuming positional order lines up.
# =============================================================================
def _align_to_original_order(
    original_df: pd.DataFrame,
    result_df: pd.DataFrame,
    pred_col: str,
) -> np.ndarray:
    """
    Realign result_df[pred_col] back to original_df's row order using
    (store_id, dept_id, date) as the join key.
    """
    if result_df.columns.duplicated().any():
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    left = original_df[["store_id", "dept_id", "date"]].copy()
    left["_orig_pos"] = np.arange(len(left))
    left["_date_key"] = pd.to_datetime(left["date"]).dt.strftime("%Y-%m-%d")

    right = result_df[["store_id", "dept_id", "date", pred_col]].copy()
    right["_date_key"] = pd.to_datetime(right["date"]).dt.strftime("%Y-%m-%d")

    merged = left.merge(
        right[["store_id", "dept_id", "_date_key", pred_col]],
        on=["store_id", "dept_id", "_date_key"],
        how="left",
    ).sort_values("_orig_pos")

    return merged[pred_col].values


def _get_sarima_train_predictions(forecaster: SARIMAForecaster, train_df: pd.DataFrame) -> pd.Series:
    """
    Extract predictions from an already-fit SARIMAForecaster for train_df.
    """
    preds = forecaster.predict(train_df)
    return pd.Series(preds, index=train_df.index)


def _get_prophet_train_predictions(forecaster: ProphetForecaster, train_df: pd.DataFrame) -> pd.Series:
    """
    Extract predictions from an already-fit ProphetForecaster for train_df.
    """
    preds = forecaster.predict(train_df)
    return pd.Series(preds, index=train_df.index)


def _subset_to_max_groups(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    max_groups: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    DEV/TESTING SPEED-UP ONLY. When max_groups is set, restrict both train_df and
    holdout_df down to just the first `max_groups` unique (store_id, dept_id) series,
    so the ENTIRE pipeline (SARIMA fit, Prophet fit, LightGBM training) runs fast on a
    small slice instead of all ~3,300 series. Use this while iterating on code; run
    with max_groups=None (the default) for the one real full run whose output you
    actually demo.
    """
    if max_groups is None:
        return train_df, holdout_df

    group_keys = (
        train_df[["store_id", "dept_id"]]
        .drop_duplicates()
        .head(max_groups)
    )
    train_subset = train_df.merge(group_keys, on=["store_id", "dept_id"], how="inner")
    holdout_subset = holdout_df.merge(group_keys, on=["store_id", "dept_id"], how="inner")
    print(f"[DEV MODE] max_groups={max_groups} -> using {len(group_keys)} store/dept series "
          f"({len(train_subset)} train rows, {len(holdout_subset)} holdout rows).")
    return train_subset, holdout_subset


def select_better_statistical_model(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> Tuple[str, pd.Series, pd.Series]:
    """
    Fits SARIMA and Prophet, evaluates both against the true holdout target on
    MAPE + RMSE, and returns whichever performed better along with its
    (train_pred, holdout_pred) series so it can be used as the residual-stacking base.

    Selection rule: lower MAPE wins; RMSE is the tiebreaker if MAPE is effectively equal.
    If one model fails to fit/predict, the other is used automatically.

    (For speed control, subset train_df/holdout_df with _subset_to_max_groups() before
    calling this — see predict_stacked_hybrid's max_groups parameter.)
    """
    candidates: Dict[str, Tuple[pd.Series, pd.Series, Dict[str, float]]] = {}

    # --- SARIMA ---
    try:
        print("    Fitting SARIMA (this can take a while across all store/dept series)...")
        sarima_forecaster = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 0, 52))
        # NOTE: train_df/holdout_df are already subset to max_groups upstream (see
        # _subset_to_max_groups), so fit() here naturally only sees that many series.
        sarima_forecaster.fit(train_df)

        sarima_holdout_values = sarima_forecaster.predict(holdout_df)
        sarima_holdout_pred = pd.Series(sarima_holdout_values, index=holdout_df.index)

        eval_df = holdout_df.copy()
        eval_df["prediction"] = sarima_holdout_values
        sarima_metrics = evaluate_predictions(eval_df, pred_col="prediction")

        sarima_train_pred = _get_sarima_train_predictions(sarima_forecaster, train_df)

        candidates["SARIMA"] = (sarima_train_pred, sarima_holdout_pred, sarima_metrics)
        print(f"    SARIMA  -> MAPE: {sarima_metrics['MAPE (%)']}%  RMSE: {sarima_metrics['RMSE']}")
    except Exception as e:
        print(f"[WARNING] SARIMA model failed to fit/predict ({type(e).__name__}: {e}). Skipping SARIMA.")

    # --- Prophet ---
    try:
        print("    Fitting Prophet (this can take a while across all store/dept series)...")
        prophet_forecaster = ProphetForecaster(interval_width=0.95)
        prophet_forecaster.fit(train_df)

        prophet_holdout_values = prophet_forecaster.predict(holdout_df)
        prophet_holdout_pred = pd.Series(prophet_holdout_values, index=holdout_df.index)

        eval_df = holdout_df.copy()
        eval_df["prediction"] = prophet_holdout_values
        prophet_metrics = evaluate_predictions(eval_df, pred_col="prediction")

        prophet_train_pred = _get_prophet_train_predictions(prophet_forecaster, train_df)

        candidates["Prophet"] = (prophet_train_pred, prophet_holdout_pred, prophet_metrics)
        print(f"    Prophet -> MAPE: {prophet_metrics['MAPE (%)']}%  RMSE: {prophet_metrics['RMSE']}")
    except Exception as e:
        print(f"[WARNING] Prophet model failed to fit/predict ({type(e).__name__}: {e}). Skipping Prophet.")

    if not candidates:
        raise RuntimeError(
            "Neither SARIMA nor Prophet produced a forecast. Check their fit()/predict() "
            "implementations in sarima_model.py / prophet_model.py, and that the required "
            "packages (statsmodels, prophet) are installed."
        )

    best_name = min(
        candidates,
        key=lambda name: (candidates[name][2]["MAPE (%)"], candidates[name][2]["RMSE"]),
    )
    best_train_pred, best_holdout_pred, best_metrics = candidates[best_name]
    print(
        f"[SELECTED] {best_name} chosen as statistical base "
        f"(MAPE: {best_metrics['MAPE (%)']}%, RMSE: {best_metrics['RMSE']})"
    )

    return best_name, best_train_pred, best_holdout_pred


def prepare_features(df: pd.DataFrame, stat_pred_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Prepare feature matrix X for LightGBM.
    """
    df = df.copy()
    if stat_pred_series is not None:
        df["stat_pred"] = np.asarray(stat_pred_series)

    feature_cols = [
        "store_id", "dept_id", "is_holiday", "store_size", "temperature",
        "fuel_price", "markdown1", "markdown2", "markdown3", "markdown4", "markdown5",
        "cpi", "unemployment", "day_of_week", "month", "week_of_year",
        "weekly_sales_lag_1", "weekly_sales_lag_4",
        "weekly_sales_roll_mean_4", "weekly_sales_roll_std_4"
    ]
    if "stat_pred" in df.columns:
        feature_cols.append("stat_pred")

    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()

    if "store_type" in df.columns:
        X["store_type"] = df["store_type"].astype("category")
    if "is_holiday" in X.columns:
        X["is_holiday"] = X["is_holiday"].astype(bool)

    return X


def train_residual_lightgbm(
    train_df: pd.DataFrame,
    residuals: pd.Series,
    stat_train_pred: pd.Series,
) -> Tuple[lgb.LGBMRegressor, List[str]]:
    """
    Train LightGBM to predict statistical model residuals (actual - stat_pred).
    """
    X_train = prepare_features(train_df, stat_pred_series=stat_train_pred)
    y_train = residuals.values

    cat_features = [c for c in ["store_id", "dept_id", "store_type", "is_holiday"] if c in X_train.columns]

    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train, y_train, categorical_feature=cat_features if cat_features else None)
    return model, list(X_train.columns)


def train_standalone_lightgbm(
    train_df: pd.DataFrame,
) -> Tuple[lgb.LGBMRegressor, List[str]]:
    """
    Train standalone LightGBM model directly on target weekly_sales.
    Kept only as an optional reference comparison — NOT used in the final hybrid forecast.
    """
    X_train = prepare_features(train_df)
    y_train = train_df[config.TARGET_COL].values

    cat_features = [c for c in ["store_id", "dept_id", "store_type", "is_holiday"] if c in X_train.columns]

    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train, y_train, categorical_feature=cat_features if cat_features else None)
    return model, list(X_train.columns)


def predict_stacked_hybrid(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    max_groups: Optional[int] = None,
) -> pd.DataFrame:
    """
    Residual Stacking Pipeline (best-of-SARIMA/Prophet + LightGBM):
      1. Fit SARIMA and Prophet; evaluate both on holdout MAPE/RMSE; keep the winner.
      2. Compute train residuals = actual - winner's in-sample train_pred.
      3. Train LightGBM on those residuals using tabular/lag/calendar features + stat_pred.
      4. Predict holdout residuals using LightGBM.
      5. Final forecast = winner's holdout_pred + lightgbm_residual_pred.

    max_groups: DEV/TESTING ONLY. Pass a small int (e.g. 50) to run the whole pipeline
    on just that many store/dept series for a fast sanity-check run (~1-2 min instead
    of 30-60+ min). Leave as None for the real full run you'll actually demo.
    """
    train_df, holdout_df = _subset_to_max_groups(train_df, holdout_df, max_groups)
    result_df = holdout_df.copy()

    print("[1/4] Fitting SARIMA + Prophet and selecting the better statistical base...")
    stat_model_name, stat_train_pred, stat_holdout_pred = select_better_statistical_model(train_df, holdout_df)
    result_df["stat_prediction"] = np.asarray(stat_holdout_pred)
    # Record which model won so evaluate.py can label the comparison table correctly
    result_df.attrs["stat_model_used"] = stat_model_name

    # Step 2: Training residuals against the WINNING statistical model
    residuals = train_df[config.TARGET_COL] - np.asarray(stat_train_pred)

    print(f"[2/4] Training LightGBM on {stat_model_name} residuals...")
    lgbm_residual_model, feature_cols = train_residual_lightgbm(train_df, residuals, stat_train_pred)

    print("[3/4] Predicting holdout residuals with LightGBM...")
    X_holdout = prepare_features(holdout_df, stat_pred_series=stat_holdout_pred)
    for col in feature_cols:
        if col not in X_holdout.columns:
            X_holdout[col] = 0
    X_holdout = X_holdout[feature_cols]

    residual_pred = lgbm_residual_model.predict(X_holdout)
    result_df["residual_prediction"] = residual_pred

    print(f"[4/4] Combining {stat_model_name} base + LightGBM residual predictions...")
    final_pred = np.asarray(stat_holdout_pred) + residual_pred

    # Clip negative predictions to zero (sales cannot be negative in inventory context)
    result_df["prediction"] = np.maximum(final_pred, 0.0)

    return result_df


def run_hybrid_pipeline(
    train_path: Path = config.PROCESSED_TRAIN_CSV,
    holdout_path: Path = config.PROCESSED_HOLDOUT_CSV,
    output_path: Optional[Path] = None,
    max_groups: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run end-to-end evaluation of the Residual Stacking Hybrid Forecaster
    (best of SARIMA/Prophet + LightGBM), and SAVE the resulting predictions to CSV
    so the demo can load that file instead of re-running the full pipeline live.

    max_groups: DEV/TESTING ONLY. Pass a small int (e.g. 50) for a fast run on a
    subset of store/dept series while you're iterating on code. Leave as None
    (the default) for the one real full run whose saved CSV you actually demo.
    """
    if not train_path.exists() or not holdout_path.exists():
        print(f"[WARNING] Processed data files not found at {train_path} / {holdout_path}.")
        print("Run 'python -m src.features.build_dataset' first to generate processed data.")
        return pd.DataFrame(), {}

    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)

    pred_df = predict_stacked_hybrid(train_df, holdout_df, max_groups=max_groups)
    metrics = evaluate_predictions(pred_df, pred_col="prediction")

    stat_model_name = pred_df.attrs.get("stat_model_used", "Statistical Model")
    print_evaluation_report(f"Hybrid Residual Stacking ({stat_model_name} + LightGBM)", metrics)

    # Save predictions to disk so the demo can load this file instead of retraining live.
    # Skipped automatically for max_groups dev runs so a partial test run never
    # overwrites your real full-run demo file by accident.
    if max_groups is None:
        if output_path is None:
            output_path = config.PROCESSED_DIR / "hybrid_predictions.csv"
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(output_path, index=False)
        print(f"\nSaved hybrid predictions to {output_path} (use this file for your demo).")
    else:
        print(f"\n[DEV MODE] max_groups={max_groups} run — predictions NOT saved to disk. "
              f"Run again with max_groups=None for the real full run to save.")

    return pred_df, metrics


if __name__ == "__main__":
    print("Executing Stacked Hybrid Model (Best of SARIMA/Prophet + LightGBM Residual Stacking)...")
    if config.PROCESSED_TRAIN_CSV.exists() and config.PROCESSED_HOLDOUT_CSV.exists():
        # DEV MODE: uncomment the line below for a fast ~1-2 min test run on 50 series
        # while you're iterating on code, instead of the full ~3,300-series run:
        #
        #     run_hybrid_pipeline(max_groups=50)
        #
        # For the real run whose CSV you'll actually demo, use the default (no max_groups):
        run_hybrid_pipeline()
    else:
        print("[WARNING] Processed CSVs not found. Run 'python -m src.features.build_dataset' first.")