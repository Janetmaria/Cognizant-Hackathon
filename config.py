"""
Shared constants for the Retail Sales Forecasting Platform.

Import from this module instead of hardcoding paths, column names, or
split dates. See docs/data_schema.md for the full processed-data schema
this file encodes.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw Kaggle files (Walmart Recruiting - Store Sales Forecasting)
RAW_TRAIN_CSV = RAW_DIR / "train.csv"
RAW_TEST_CSV = RAW_DIR / "test.csv"        # Kaggle holdout: no Weekly_Sales, not usable for validation
RAW_FEATURES_CSV = RAW_DIR / "features.csv"
RAW_STORES_CSV = RAW_DIR / "stores.csv"

# Processed, standardized files (see docs/data_schema.md)
PROCESSED_TRAIN_CSV = PROCESSED_DIR / "model_train.csv"      # rows with date <= TRAIN_END_DATE, has target
PROCESSED_HOLDOUT_CSV = PROCESSED_DIR / "model_holdout.csv"  # rows in [TEST_START_DATE, TEST_END_DATE], has target
PROCESSED_PREDICT_CSV = PROCESSED_DIR / "kaggle_predict.csv" # processed version of raw test.csv, no target

# ---------------------------------------------------------------------------
# Train / holdout split
# ---------------------------------------------------------------------------
# train.csv covers 2010-02-05 .. 2012-10-26 (143 weekly snapshots).
# Kaggle's test.csv has no ground-truth Weekly_Sales, so our internal
# validation set is carved out of train.csv's last 10 weeks instead.

TRAIN_END_DATE = "2012-08-17"    # last date included in the training set (inclusive)
TEST_START_DATE = "2012-08-24"   # first date in the holdout/validation set (inclusive)
TEST_END_DATE = "2012-10-26"     # last date in the holdout/validation set (inclusive) == max(train.csv date)

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

TARGET_COL = "weekly_sales"

# Standardized dtypes for the processed dataset (data/processed/*.csv).
# Keys are the column names every feature/model/API script should use.
PROCESSED_DTYPES = {
    "store_id": "int64",
    "dept_id": "int64",
    "date": "datetime64[ns]",
    "weekly_sales": "float64",   # target; absent from PROCESSED_PREDICT_CSV
    "is_holiday": "bool",
    "store_type": "category",    # A / B / C
    "store_size": "int64",
    "temperature": "float64",
    "fuel_price": "float64",
    "markdown1": "float64",      # NaN filled with 0.0 by src/features/build_dataset.py
    "markdown2": "float64",
    "markdown3": "float64",
    "markdown4": "float64",
    "markdown5": "float64",
    "cpi": "float64",            # ffilled per-store; 0 nulls in model_train/model_holdout
    "unemployment": "float64",   # ffilled per-store; 0 nulls in model_train/model_holdout
    "is_negative_sales": "bool",       # True if weekly_sales < 0 (kept unmodified, see docs/data_schema.md)
    "day_of_week": "int64",            # constant (4 = Friday) for this dataset; kept for completeness
    "month": "int64",
    "week_of_year": "int64",
    "weekly_sales_lag_1": "float64",   # nullable — NaN for a group's first observed week(s)
    "weekly_sales_lag_4": "float64",   # nullable
    "weekly_sales_roll_mean_4": "float64",  # nullable; computed on shift(1) series, see build_dataset.py
    "weekly_sales_roll_std_4": "float64",   # nullable
}

# Columns that are allowed to contain NaN in the processed data.
NULLABLE_COLS = [
    "weekly_sales_lag_1", "weekly_sales_lag_4",
    "weekly_sales_roll_mean_4", "weekly_sales_roll_std_4",
]

# Natural key that uniquely identifies a row (one store-dept-week).
ID_COLS = ["store_id", "dept_id", "date"]

# Feature engineering knobs used by src/features/build_dataset.py.
LAG_WEEKS = [1, 4]
ROLLING_WINDOW_WEEKS = 4

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
