"""
CSV upload -> forecast + reorder alerts pipeline.

Lets a user upload a CSV shaped like our raw Kaggle data (Store, Dept, Date,
Weekly_Sales at minimum) and get back forecasts + reorder alerts, reusing the
existing pipeline rather than retraining anything:

  1. Validate + coerce the uploaded rows (src/features/build_dataset.py's
     standardize_columns/add_negative_sales_flag/fill_markdowns/
     add_calendar_features/add_lag_and_rolling_features run on the upload
     itself, purely for schema QA and display stats -- they do not feed the
     forecast).
  2. Match the uploaded (store_id, dept_id) pairs against the hybrid model's
     already-computed data/processed/hybrid_predictions.csv.
  3. Score matched rows with src/reorder_logic.py's get_reorder_alerts_from_df().

Store/Dept pairs outside the trained catalog (the 45-store Kaggle universe)
come back in "unmatched_pairs" with a reason -- retraining for genuinely new
stores/departments is out of scope here.
"""

from typing import Optional

import numpy as np
import pandas as pd

import src.features.build_dataset as build_dataset
import src.reorder_logic as reorder_logic

REQUIRED_COLUMNS = ["Store", "Dept", "Date", "Weekly_Sales"]


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks required columns are present, coerces types, and drops rows that
    fail to coerce. Raises ValueError (-> 400 at the route layer) on missing
    columns or on zero valid rows remaining.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if df.empty:
        raise ValueError("Uploaded CSV has no data rows")

    df = df.copy()
    df["Store"] = pd.to_numeric(df["Store"], errors="coerce")
    df["Dept"] = pd.to_numeric(df["Dept"], errors="coerce")
    df["Weekly_Sales"] = pd.to_numeric(df["Weekly_Sales"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    valid_mask = (
        df["Store"].notna()
        & df["Dept"].notna()
        & df["Weekly_Sales"].notna()
        & df["Date"].notna()
    )
    df = df.loc[valid_mask].copy()
    if df.empty:
        raise ValueError(
            "Uploaded CSV has no valid rows -- Store, Dept, Date, and "
            "Weekly_Sales must all be present and correctly typed"
        )

    df["Store"] = df["Store"].astype("int64")
    df["Dept"] = df["Dept"].astype("int64")

    if "IsHoliday" not in df.columns:
        df["IsHoliday"] = False
    else:
        df["IsHoliday"] = df["IsHoliday"].fillna(False).astype(bool)

    # fill_markdowns() needs these columns to exist even if the user didn't
    # supply them; missing ones become NaN here, then 0.0 via fill_markdowns.
    for i in range(1, 6):
        col = f"MarkDown{i}"
        if col not in df.columns:
            df[col] = np.nan

    return df


def engineer_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the uploaded rows through the same standardize/fill/calendar/lag
    helpers build_dataset.py uses on the raw Kaggle files. Output isn't fed
    into the forecast (see module docstring) -- it's real feature engineering
    over the user's own sales history, used only for the upload_summary QA
    stats returned alongside the forecast.
    """
    df = build_dataset.standardize_columns(df)
    df = build_dataset.add_negative_sales_flag(df)
    df = build_dataset.fill_markdowns(df)
    df = build_dataset.add_calendar_features(df)
    df = build_dataset.add_lag_and_rolling_features(df)
    return df


def match_pairs(pairs: list, as_of_date: str) -> tuple:
    """
    For each (store_id, dept_id) pair, look up its forecast row in
    hybrid_predictions.csv for as_of_date (mapped back to the historical
    holdout week, same mechanism as reorder_logic.get_reorder_alerts). Falls
    back to that pair's earliest available prediction date if the exact week
    isn't present, mirroring get_reorder_alerts' own fallback.

    Returns (matched_df, unmatched_list) where unmatched_list holds
    {"store_id", "dept_id", "reason"} dicts for pairs never seen in training.
    """
    predictions_path = reorder_logic.PREDICTIONS_PATH
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions data file {predictions_path} not found. Ensure the hybrid model has been trained."
        )

    preds = pd.read_csv(predictions_path)
    historical_date = reorder_logic.map_date_backward(as_of_date)

    matched_rows = []
    unmatched = []
    for store_id, dept_id in pairs:
        subset = preds[(preds["store_id"] == store_id) & (preds["dept_id"] == dept_id)]
        if subset.empty:
            unmatched.append({
                "store_id": store_id,
                "dept_id": dept_id,
                "reason": "no trained forecast for this store/department",
            })
            continue

        row = subset[subset["date"] == historical_date]
        if row.empty:
            row = subset.sort_values("date").head(1)
        matched_rows.append(row)

    matched_df = (
        pd.concat(matched_rows, ignore_index=True)
        if matched_rows
        else pd.DataFrame(columns=preds.columns)
    )
    return matched_df, unmatched


def build_forecasts_list(matched_df: pd.DataFrame) -> list:
    forecasts = []
    for _, row in matched_df.iterrows():
        pred = float(
            row.get(
                "prediction",
                row.get("lgbm_pred", row.get("stat_prediction", row.get("baseline_pred", row.get("weekly_sales", 0.0)))),
            )
        )
        pred = max(0.0, pred)
        forecasts.append({
            "store_id": str(row["store_id"]),
            "dept_id": str(row["dept_id"]),
            "week_ending_date": reorder_logic.map_date_forward(str(row["date"])),
            "predicted_weekly_sales": round(pred, 2),
        })
    return forecasts


def build_upload_summary(engineered_df: pd.DataFrame, pair_count: int) -> dict:
    return {
        "unique_store_dept_pairs": pair_count,
        "date_range": {
            "start": str(engineered_df["date"].min().date()),
            "end": str(engineered_df["date"].max().date()),
        },
        "rows_with_negative_sales": int(engineered_df["is_negative_sales"].sum()),
    }


def process_upload(
    raw_df: pd.DataFrame,
    urgency_filter: Optional[str] = None,
    as_of_date: str = "2026-08-21",
) -> dict:
    """
    Full pipeline: validate -> engineer (display stats only) -> match against
    hybrid_predictions.csv -> score with reorder_logic. Raises ValueError for
    input problems (-> 400) and FileNotFoundError if predictions haven't been
    generated yet (-> 500), both handled by the route layer.
    """
    rows_received = len(raw_df)

    prepared = validate_and_prepare(raw_df)
    engineered = engineer_for_display(prepared)

    pairs = sorted(set(zip(engineered["store_id"].tolist(), engineered["dept_id"].tolist())))

    matched_df, unmatched = match_pairs(pairs, as_of_date)
    if matched_df.empty:
        raise ValueError(
            "None of the uploaded Store/Dept pairs have a trained forecast available "
            f"({len(unmatched)} pair(s) unmatched)"
        )

    forecasts = build_forecasts_list(matched_df)
    alerts = reorder_logic.get_reorder_alerts_from_df(matched_df, urgency_filter=urgency_filter)

    total_depts_at_risk = sum(1 for a in alerts if a["urgency"] in ("red", "amber"))
    total_capital_freed_estimate = round(sum(a["capital_freed_estimate"] for a in alerts), 2)

    return {
        "rows_received": rows_received,
        "matched_pairs": len(matched_df),
        "unmatched_pairs": unmatched,
        "forecasts": forecasts,
        "reorder_alerts": alerts,
        "summary": {
            "total_depts_at_risk": total_depts_at_risk,
            "total_capital_freed_estimate": total_capital_freed_estimate,
        },
        "upload_summary": build_upload_summary(engineered, len(pairs)),
    }
