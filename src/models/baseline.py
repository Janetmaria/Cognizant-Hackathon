import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.evaluate import evaluate_predictions, print_evaluation_report


def predict_seasonal_naive(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    seasonal_weeks: int = 52,
) -> pd.DataFrame:
    """
    Generate Seasonal-Naive baseline forecast for weekly sales.

    Rule:
    Forecast for time t = Actual sales from time (t - 52 weeks) [same week last year].
    Fallback 1: Match by (store_id, dept_id, week_of_year) in train_df.
    Fallback 2: Last known sales value (weekly_sales_lag_1).
    Fallback 3: Overall mean sales for the store-dept series.
    """
    result_df = holdout_df.copy()

    # Ensure date is datetime
    train_copy = train_df.copy()
    train_copy["date_dt"] = pd.to_datetime(train_copy["date"])
    result_df["date_dt"] = pd.to_datetime(result_df["date"])

    # Method 1: Exact 52-week offset match (364 days)
    result_df["target_lookup_date"] = result_df["date_dt"] - pd.Timedelta(days=7 * seasonal_weeks)

    lookup = train_copy[["store_id", "dept_id", "date_dt", config.TARGET_COL]].rename(
        columns={"date_dt": "target_lookup_date", config.TARGET_COL: "naive_pred_52w"}
    )

    merged = result_df.merge(lookup, on=["store_id", "dept_id", "target_lookup_date"], how="left")
    predictions = merged["naive_pred_52w"].copy()

    # Method 2 Fallback: Match by week_of_year in train_df
    if "week_of_year" in train_copy.columns and "week_of_year" in result_df.columns:
        woy_lookup = train_copy.groupby(["store_id", "dept_id", "week_of_year"])[config.TARGET_COL].mean().reset_index()
        woy_lookup = woy_lookup.rename(columns={config.TARGET_COL: "naive_pred_woy"})
        merged_woy = result_df.merge(woy_lookup, on=["store_id", "dept_id", "week_of_year"], how="left")
        predictions = predictions.fillna(merged_woy["naive_pred_woy"])

    # Method 3 Fallback: Use lag_1 if available
    if "weekly_sales_lag_1" in result_df.columns:
        predictions = predictions.fillna(result_df["weekly_sales_lag_1"])

    # Method 4 Fallback: Store-Dept overall mean from train
    store_dept_means = train_copy.groupby(["store_id", "dept_id"])[config.TARGET_COL].mean()
    fallback_series = result_df.set_index(["store_id", "dept_id"]).index.map(store_dept_means)
    predictions = predictions.fillna(pd.Series(fallback_series, index=result_df.index))

    # Final global mean fallback if needed
    global_mean = train_copy[config.TARGET_COL].mean() if len(train_copy) > 0 else 0.0
    predictions = predictions.fillna(global_mean)

    result_df["prediction"] = predictions.values

    # Clean temporary helper columns
    drop_cols = ["date_dt", "target_lookup_date"]
    result_df = result_df.drop(columns=[c for c in drop_cols if c in result_df.columns])

    return result_df


class SeasonalNaiveBaseline:
    """
    Class wrapper around predict_seasonal_naive(), matching the .fit(train_df) /
    .predict(holdout_df) -> array interface that sarima_model.py and prophet_model.py
    expect for benchmarking against the seasonal-naive baseline.
    """

    def __init__(self, seasonal_weeks: int = 52):
        self.seasonal_weeks = seasonal_weeks
        self.train_df: Optional[pd.DataFrame] = None

    def fit(self, train_df: pd.DataFrame) -> "SeasonalNaiveBaseline":
        """Store the training data; seasonal-naive has no parameters to learn."""
        self.train_df = train_df.copy()
        return self

    def predict(self, holdout_df: pd.DataFrame) -> np.ndarray:
        """Return seasonal-naive point predictions as a numpy array, aligned to holdout_df's row order."""
        if self.train_df is None:
            raise RuntimeError("SeasonalNaiveBaseline.fit(train_df) must be called before predict().")
        pred_df = predict_seasonal_naive(self.train_df, holdout_df, seasonal_weeks=self.seasonal_weeks)
        return pred_df["prediction"].values


def run_baseline_pipeline(
    train_path: Path = config.PROCESSED_TRAIN_CSV,
    holdout_path: Path = config.PROCESSED_HOLDOUT_CSV,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load data, execute Seasonal-Naive baseline predictions, and return evaluation metrics.
    """
    if not train_path.exists() or not holdout_path.exists():
        print(f"[WARNING] Processed data files not found at {train_path} / {holdout_path}.")
        print("Run 'python -m src.features.build_dataset' first to generate processed data.")
        return pd.DataFrame(), {}

    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)

    pred_df = predict_seasonal_naive(train_df, holdout_df)
    metrics = evaluate_predictions(pred_df, pred_col="prediction")

    print_evaluation_report("Seasonal-Naive Baseline", metrics)
    return pred_df, metrics


if __name__ == "__main__":
    print("Executing Baseline Model (Seasonal-Naive)...")
    if config.PROCESSED_TRAIN_CSV.exists() and config.PROCESSED_HOLDOUT_CSV.exists():
        run_baseline_pipeline()
    else:
        print("[DEMO MODE] Generating synthetic test data for baseline demonstration...")
        dates_train = pd.date_range("2010-02-05", "2012-08-17", freq="W-FRI")
        dates_holdout = pd.date_range("2012-08-24", "2012-10-26", freq="W-FRI")

        train_list, holdout_list = [], []
        for s in [1, 2]:
            for d in [1, 2]:
                for dt in dates_train:
                    sales = 10000 + 2000 * np.sin(dt.dayofyear / 365.0 * 2 * np.pi) + np.random.normal(0, 500)
                    train_list.append({"store_id": s, "dept_id": d, "date": dt.strftime("%Y-%m-%d"), "weekly_sales": sales, "week_of_year": dt.isocalendar().week})
                for dt in dates_holdout:
                    sales = 10000 + 2000 * np.sin(dt.dayofyear / 365.0 * 2 * np.pi) + np.random.normal(0, 500)
                    holdout_list.append({"store_id": s, "dept_id": d, "date": dt.strftime("%Y-%m-%d"), "weekly_sales": sales, "week_of_year": dt.isocalendar().week, "is_holiday": False})

        demo_train = pd.DataFrame(train_list)
        demo_holdout = pd.DataFrame(holdout_list)

        demo_pred = predict_seasonal_naive(demo_train, demo_holdout)
        metrics = evaluate_predictions(demo_pred, pred_col="prediction")
        print_evaluation_report("Seasonal-Naive Baseline (Synthetic Demo)", metrics)