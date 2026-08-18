"""
Module 5 - LightGBM forecasting model.

Owner: Module 5.
Reads standardized data via config.PROCESSED_TRAIN_CSV / PROCESSED_HOLDOUT_CSV,
predicts config.TARGET_COL. Compared against other models via src/evaluate.py.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.evaluate import evaluate_predictions, print_evaluation_report, compare_models
from src.models.baseline import SeasonalNaiveBaseline


# Feature specification as per docs/data_schema.md and config.py
CATEGORICAL_FEATURES = ["store_id", "dept_id", "store_type", "is_holiday"]
NUMERICAL_FEATURES = [
    "store_size",
    "temperature",
    "fuel_price",
    "markdown1",
    "markdown2",
    "markdown3",
    "markdown4",
    "markdown5",
    "cpi",
    "unemployment",
    "month",
    "week_of_year",
    "weekly_sales_lag_1",
    "weekly_sales_lag_4",
    "weekly_sales_roll_mean_4",
    "weekly_sales_roll_std_4",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


class LightGBMForecaster:
    """
    LightGBM Gradient Boosting Model for Retail Sales Forecasting.
    Uses calendar, promotional, economic, and historical lag/rolling features.
    """

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": config.RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = ALL_FEATURES

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess input DataFrame into feature matrix X and target y.
        Encodes categorical variables into Pandas 'category' dtype for native LightGBM handling.
        """
        data = df.copy()
        
        # Ensure calendar features exist if absent
        if "month" not in data.columns or "week_of_year" not in data.columns:
            if "date" in data.columns:
                dates = pd.to_datetime(data["date"])
                data["month"] = dates.dt.month
                data["week_of_year"] = dates.dt.isocalendar().week.astype(int)

        # Ensure all required features are present
        for col in ALL_FEATURES:
            if col not in data.columns:
                data[col] = np.nan

        # Format categorical features for LightGBM
        for col in CATEGORICAL_FEATURES:
            data[col] = data[col].astype("category")

        X = data[ALL_FEATURES]
        y = data[config.TARGET_COL] if config.TARGET_COL in data.columns else None

        return X, y

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        early_stopping_rounds: int = 50,
    ) -> "LightGBMForecaster":
        """
        Fit the LightGBM model on training data, using optional validation split for early stopping.
        """
        X_train, y_train = self.preprocess(train_df)
        
        callbacks = []
        eval_set = None
        
        if val_df is not None:
            X_val, y_val = self.preprocess(val_df)
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate sales predictions using the trained LightGBM model.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() before predict().")
        X, _ = self.preprocess(df)
        preds = self.model.predict(X)
        return preds

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importances measured by split frequency and gain.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        
        importance_split = self.model.booster_.feature_importance(importance_type="split")
        importance_gain = self.model.booster_.feature_importance(importance_type="gain")
        
        df_importance = pd.DataFrame({
            "feature": self.feature_names,
            "split_importance": importance_split,
            "gain_importance": np.round(importance_gain, 2),
        }).sort_values(by="gain_importance", ascending=False).reset_index(drop=True)

        return df_importance

    def print_feature_importance(self, top_n: int = 15) -> None:
        """
        Print top N important features.
        """
        df_imp = self.get_feature_importance()
        print("\n" + "=" * 55)
        print(f" TOP {top_n} LIGHTGBM FEATURE IMPORTANCES (BY GAIN)")
        print("=" * 55)
        for idx, row in df_imp.head(top_n).iterrows():
            print(f"  {idx+1:2d}. {row['feature']:<25} Gain: {row['gain_importance']:>12.2f} | Split: {row['split_importance']:>5d}")
        print("=" * 55)


def run_lightgbm_pipeline(
    train_path: Path = config.PROCESSED_TRAIN_CSV,
    holdout_path: Path = config.PROCESSED_HOLDOUT_CSV,
    output_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Full pipeline for Module 5:
    1. Loads processed train and holdout data.
    2. Performs chronological train/val split on train data (last 10 weeks of train set for validation).
    3. Fits LightGBM model with early stopping.
    4. Evaluates on holdout dataset.
    5. Benchmarks LightGBM against Seasonal Naïve Baseline.
    6. Saves predictions to CSV.
    """
    if not train_path.exists() or not holdout_path.exists():
        print(f"[WARNING] Processed data files not found at:\n  - {train_path}\n  - {holdout_path}")
        print("Please run Module 1 (`python -m src.features.build_dataset`) first when raw data is available.")
        return {}

    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading holdout data from {holdout_path}...")
    holdout_df = pd.read_csv(holdout_path)

    # Chronological train/validation split within train_df for hyperparameter early stopping
    train_df["date_dt"] = pd.to_datetime(train_df["date"])
    max_train_date = train_df["date_dt"].max()
    val_cutoff_date = max_train_date - pd.Timedelta(weeks=10)

    sub_train = train_df[train_df["date_dt"] <= val_cutoff_date].copy()
    sub_val = train_df[train_df["date_dt"] > val_cutoff_date].copy()

    print(f"Sub-train split: {len(sub_train)} rows | Validation split: {len(sub_val)} rows")

    # Fit LightGBM
    lgb_forecaster = LightGBMForecaster()
    lgb_forecaster.fit(sub_train, val_df=sub_val, early_stopping_rounds=50)

    # Predict on holdout set
    holdout_df["lgbm_pred"] = lgb_forecaster.predict(holdout_df)

    # Evaluate LightGBM
    lgb_metrics = evaluate_predictions(
        holdout_df,
        pred_col="lgbm_pred",
        target_col=config.TARGET_COL,
        holiday_col="is_holiday",
    )
    print_evaluation_report("LightGBM Model", lgb_metrics)

    # Print feature importance
    lgb_forecaster.print_feature_importance(top_n=15)

    # Run baseline model for side-by-side model comparison table
    baseline = SeasonalNaiveBaseline()
    baseline.fit(train_df)
    holdout_df["baseline_pred"] = baseline.predict(holdout_df)
    baseline_metrics = evaluate_predictions(
        holdout_df,
        pred_col="baseline_pred",
        target_col=config.TARGET_COL,
        holiday_col="is_holiday",
    )

    # Compare models
    comparison_dict = {
        "Seasonal Naïve Baseline": baseline_metrics,
        "LightGBM Model": lgb_metrics,
    }
    df_comparison = compare_models(comparison_dict)

    print("\n" + "=" * 55)
    print(" CONSOLIDATED MODEL COMPARISON TABLE (HOLDOUT SET)")
    print("=" * 55)
    print(df_comparison.to_string())
    print("=" * 55)

    if output_path is None:
        output_path = config.PROCESSED_DIR / "lightgbm_predictions.csv"

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(output_path, index=False)
    print(f"\nSaved LightGBM predictions to {output_path}")

    return comparison_dict


if __name__ == "__main__":
    if config.PROCESSED_TRAIN_CSV.exists() and config.PROCESSED_HOLDOUT_CSV.exists():
        run_lightgbm_pipeline()
    else:
        print("[INFO] Running lightgbm_model.py test mode on synthetic data...")
        np.random.seed(config.RANDOM_SEED)
        dates_train = pd.date_range(start="2010-02-05", periods=50, freq="W-FRI")
        dates_holdout = pd.date_range(start="2011-02-04", periods=10, freq="W-FRI")

        train_rows, holdout_rows = [], []
        for s in range(1, 3):
            for d in range(1, 4):
                for dt in dates_train:
                    sales = 10000 + s * 1000 + d * 500 + np.random.normal(0, 500)
                    train_rows.append({
                        "store_id": s, "dept_id": d, "date": dt, "weekly_sales": sales,
                        "is_holiday": False, "store_type": "A", "store_size": 150000,
                        "temperature": 60.0, "fuel_price": 3.5, "markdown1": 0.0,
                        "markdown2": 0.0, "markdown3": 0.0, "markdown4": 0.0, "markdown5": 0.0,
                        "cpi": 211.0, "unemployment": 8.1, "weekly_sales_lag_1": sales - 50,
                        "weekly_sales_lag_4": sales - 200, "weekly_sales_roll_mean_4": sales - 100,
                        "weekly_sales_roll_std_4": 300.0,
                    })
                for dt in dates_holdout:
                    sales = 10000 + s * 1000 + d * 500 + np.random.normal(0, 500)
                    holdout_rows.append({
                        "store_id": s, "dept_id": d, "date": dt, "weekly_sales": sales,
                        "is_holiday": False, "store_type": "A", "store_size": 150000,
                        "temperature": 62.0, "fuel_price": 3.6, "markdown1": 0.0,
                        "markdown2": 0.0, "markdown3": 0.0, "markdown4": 0.0, "markdown5": 0.0,
                        "cpi": 212.0, "unemployment": 8.0, "weekly_sales_lag_1": sales - 30,
                        "weekly_sales_lag_4": sales - 150, "weekly_sales_roll_mean_4": sales - 80,
                        "weekly_sales_roll_std_4": 280.0,
                    })

        synth_train = pd.DataFrame(train_rows)
        synth_holdout = pd.DataFrame(holdout_rows)

        forecaster = LightGBMForecaster()
        forecaster.fit(synth_train)
        synth_holdout["lgbm_pred"] = forecaster.predict(synth_holdout)

        metrics = evaluate_predictions(synth_holdout, pred_col="lgbm_pred")
        print_evaluation_report("LightGBM Synthetic Test", metrics)
        forecaster.print_feature_importance(top_n=10)

