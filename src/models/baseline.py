"""
Module 5 - Baseline forecasting model (Seasonal Naïve / Last-observed Naïve).

Owner: Module 5.
Reads standardized data via config.PROCESSED_TRAIN_CSV / PROCESSED_HOLDOUT_CSV,
predicts config.TARGET_COL. Compared against other models via src/evaluate.py.
"""

import sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.evaluate import evaluate_predictions, print_evaluation_report


class SeasonalNaiveBaseline:
    """
    Seasonal Naïve Baseline Model.
    
    Prediction strategy:
    1. Primary: Historical mean for the same (store_id, dept_id, week_of_year) in training data.
    2. Fallback 1: Previous week's sales (weekly_sales_lag_1).
    3. Fallback 2: Store-Department average sales.
    4. Fallback 3: Global average weekly sales.
    """

    def __init__(self):
        self.seasonal_lookup: Dict = {}
        self.store_dept_lookup: Dict = {}
        self.global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "SeasonalNaiveBaseline":
        """
        Fit the baseline lookup tables on training data.
        """
        df = train_df.copy()
        
        # Ensure week_of_year column exists
        if "week_of_year" not in df.columns:
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
                df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
            else:
                raise ValueError("Training data must contain 'week_of_year' or 'date' column.")

        # 1. Seasonal lookup: (store_id, dept_id, week_of_year) -> mean sales
        seasonal_group = df.groupby(["store_id", "dept_id", "week_of_year"])[config.TARGET_COL].mean()
        self.seasonal_lookup = seasonal_group.to_dict()

        # 2. Store-Dept lookup: (store_id, dept_id) -> mean sales
        store_dept_group = df.groupby(["store_id", "dept_id"])[config.TARGET_COL].mean()
        self.store_dept_lookup = store_dept_group.to_dict()

        # 3. Global mean
        self.global_mean = float(df[config.TARGET_COL].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate baseline predictions for the input DataFrame.
        """
        predict_df = df.copy()
        if "week_of_year" not in predict_df.columns and "date" in predict_df.columns:
            dates = pd.to_datetime(predict_df["date"])
            predict_df["week_of_year"] = dates.dt.isocalendar().week.astype(int)

        preds = []
        for idx, row in predict_df.iterrows():
            store_id = row.get("store_id")
            dept_id = row.get("dept_id")
            week = row.get("week_of_year")

            # Try Seasonal lookup
            key_seasonal = (store_id, dept_id, week)
            if key_seasonal in self.seasonal_lookup and not np.isnan(self.seasonal_lookup[key_seasonal]):
                val = self.seasonal_lookup[key_seasonal]
            # Try Lag 1 feature if available
            elif "weekly_sales_lag_1" in row and not pd.isna(row["weekly_sales_lag_1"]):
                val = row["weekly_sales_lag_1"]
            # Try Store-Dept lookup
            elif (store_id, dept_id) in self.store_dept_lookup:
                val = self.store_dept_lookup[(store_id, dept_id)]
            # Global mean fallback
            else:
                val = self.global_mean

            preds.append(val)

        return np.array(preds, dtype=np.float64)


def run_baseline_pipeline(
    train_path: Path = config.PROCESSED_TRAIN_CSV,
    holdout_path: Path = config.PROCESSED_HOLDOUT_CSV,
    output_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Loads train and holdout CSVs, fits baseline model, evaluates on holdout, and saves predictions.
    """
    if not train_path.exists() or not holdout_path.exists():
        print(f"[WARNING] Processed data files not found at:\n  - {train_path}\n  - {holdout_path}")
        print("Please run Module 1 (`python -m src.features.build_dataset`) first when raw data is available.")
        return {}

    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading holdout data from {holdout_path}...")
    holdout_df = pd.read_csv(holdout_path)

    baseline = SeasonalNaiveBaseline()
    baseline.fit(train_df)

    holdout_df["baseline_pred"] = baseline.predict(holdout_df)

    metrics = evaluate_predictions(
        holdout_df,
        pred_col="baseline_pred",
        target_col=config.TARGET_COL,
        holiday_col="is_holiday",
    )

    print_evaluation_report("Seasonal Naïve Baseline", metrics)

    if output_path is None:
        output_path = config.PROCESSED_DIR / "baseline_predictions.csv"

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(output_path, index=False)
    print(f"Saved baseline predictions to {output_path}")

    return metrics


if __name__ == "__main__":
    if config.PROCESSED_TRAIN_CSV.exists() and config.PROCESSED_HOLDOUT_CSV.exists():
        run_baseline_pipeline()
    else:
        print("[INFO] Running baseline.py test mode on synthetic data...")
        # Create synthetic dataset to demonstrate functionality
        np.random.seed(config.RANDOM_SEED)
        dates_train = pd.date_range(start="2010-02-05", periods=50, freq="W-FRI")
        dates_holdout = pd.date_range(start="2011-02-04", periods=10, freq="W-FRI")

        train_rows, holdout_rows = [], []
        for s in range(1, 3):
            for d in range(1, 4):
                for dt in dates_train:
                    train_rows.append({
                        "store_id": s, "dept_id": d, "date": dt,
                        "weekly_sales": 10000 + s * 1000 + d * 500 + np.random.normal(0, 500),
                        "is_holiday": False
                    })
                for dt in dates_holdout:
                    holdout_rows.append({
                        "store_id": s, "dept_id": d, "date": dt,
                        "weekly_sales": 10000 + s * 1000 + d * 500 + np.random.normal(0, 500),
                        "is_holiday": False
                    })

        synth_train = pd.DataFrame(train_rows)
        synth_holdout = pd.DataFrame(holdout_rows)

        model = SeasonalNaiveBaseline()
        model.fit(synth_train)
        synth_holdout["baseline_pred"] = model.predict(synth_holdout)

        metrics = evaluate_predictions(synth_holdout, pred_col="baseline_pred")
        print_evaluation_report("Baseline Synthetic Test", metrics)

