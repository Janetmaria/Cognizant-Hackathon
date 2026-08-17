"""
Module 1 - builds data/processed/model_train.csv and model_holdout.csv
from the raw Kaggle files. Run as a script:

    python -m src.features.build_dataset

Does not touch data/processed/kaggle_predict.csv (raw test.csv has no
Weekly_Sales history to compute lag/rolling features from — building
that file needs to stitch train.csv's tail onto test.csv first, which
is out of scope here).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

LAG_WEEKS = config.LAG_WEEKS
ROLLING_WINDOW_WEEKS = config.ROLLING_WINDOW_WEEKS


def load_raw():
    train = pd.read_csv(config.RAW_TRAIN_CSV, parse_dates=["Date"])
    features = pd.read_csv(config.RAW_FEATURES_CSV, parse_dates=["Date"])
    stores = pd.read_csv(config.RAW_STORES_CSV)
    return train, features, stores


def fill_economic_indicators(features: pd.DataFrame) -> pd.DataFrame:
    features = features.sort_values(["Store", "Date"]).copy()
    # Forward-fill within each store's own time series, not a global mean —
    # CPI/Unemployment are regional and slow-moving, so "last known value for
    # this store" is a far better estimate than a cross-store average.
    features[["CPI", "Unemployment"]] = features.groupby("Store")[
        ["CPI", "Unemployment"]
    ].ffill()
    return features


def merge_raw(train: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    features = fill_economic_indicators(features)
    # IsHoliday is duplicated in train and features; they agree on every
    # overlapping (Store, Date) row (verified against the raw data), so keep
    # train's copy and drop features' to avoid a merge suffix.
    df = train.merge(
        features.drop(columns=["IsHoliday"]), on=["Store", "Date"], how="left"
    )
    df = df.merge(stores, on="Store", how="left")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Store": "store_id",
            "Dept": "dept_id",
            "Date": "date",
            "Weekly_Sales": "weekly_sales",
            "IsHoliday": "is_holiday",
            "Type": "store_type",
            "Size": "store_size",
            "Temperature": "temperature",
            "Fuel_Price": "fuel_price",
            "MarkDown1": "markdown1",
            "MarkDown2": "markdown2",
            "MarkDown3": "markdown3",
            "MarkDown4": "markdown4",
            "MarkDown5": "markdown5",
            "CPI": "cpi",
            "Unemployment": "unemployment",
        }
    )
    df["store_type"] = df["store_type"].astype("category")
    return df


def add_negative_sales_flag(df: pd.DataFrame) -> pd.DataFrame:
    # Negative Weekly_Sales (returns exceeding sales that week) is real
    # signal, not an error — kept unmodified per docs/data_schema.md. This
    # flag lets downstream models opt out of those rows without us having
    # to touch the target value itself.
    df["is_negative_sales"] = df["weekly_sales"] < 0
    return df


def fill_markdowns(df: pd.DataFrame) -> pd.DataFrame:
    markdown_cols = [f"markdown{i}" for i in range(1, 6)]
    df[markdown_cols] = df[markdown_cols].fillna(0.0)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int64")
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "dept_id", "date"]).reset_index(drop=True)
    grouped_sales = df.groupby(["store_id", "dept_id"])["weekly_sales"]

    for lag in LAG_WEEKS:
        df[f"weekly_sales_lag_{lag}"] = grouped_sales.shift(lag)

    # shift(1) BEFORE rolling: rolling(window) alone still includes the
    # current row, which is exactly the value we're trying to predict. Every
    # rolling stat here is computed over the window ending the week *before*
    # the current row, so no row ever sees its own or a future week's sales.
    shifted = grouped_sales.shift(1)
    roll = shifted.groupby([df["store_id"], df["dept_id"]]).rolling(
        ROLLING_WINDOW_WEEKS
    )
    df[f"weekly_sales_roll_mean_{ROLLING_WINDOW_WEEKS}"] = roll.mean().reset_index(
        level=[0, 1], drop=True
    )
    df[f"weekly_sales_roll_std_{ROLLING_WINDOW_WEEKS}"] = roll.std().reset_index(
        level=[0, 1], drop=True
    )
    return df


def split_and_save(df: pd.DataFrame) -> None:
    train_end = pd.Timestamp(config.TRAIN_END_DATE)
    test_start = pd.Timestamp(config.TEST_START_DATE)
    test_end = pd.Timestamp(config.TEST_END_DATE)

    model_train = df[df["date"] <= train_end].copy()
    model_holdout = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    for out in (model_train, model_holdout):
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model_train.to_csv(config.PROCESSED_TRAIN_CSV, index=False)
    model_holdout.to_csv(config.PROCESSED_HOLDOUT_CSV, index=False)
    print(f"wrote {len(model_train)} rows -> {config.PROCESSED_TRAIN_CSV}")
    print(f"wrote {len(model_holdout)} rows -> {config.PROCESSED_HOLDOUT_CSV}")


def build() -> pd.DataFrame:
    train, features, stores = load_raw()
    df = merge_raw(train, features, stores)
    df = standardize_columns(df)
    df = add_negative_sales_flag(df)
    df = fill_markdowns(df)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    return df


if __name__ == "__main__":
    final_df = build()
    split_and_save(final_df)
