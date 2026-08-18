"""
Module 8 - Admin-only endpoint: GET /admin/summary

Returns the same aggregated dataset stats as Module 2's /data/summary,
but is restricted to admin role only (403 for managers).

This endpoint reads directly from the processed CSVs in data/processed/.
If those files don't exist yet (i.e. Module 1 hasn't run), it falls back
to the raw train.csv + stores.csv so the endpoint works in isolation.
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from backend.auth import require_admin

router = APIRouter()

ROOT_DIR = Path(__file__).resolve().parent.parent

# Prefer processed data; fall back to raw if not yet generated
_PROCESSED_TRAIN = ROOT_DIR / "data" / "processed" / "model_train.csv"
_RAW_TRAIN = ROOT_DIR / "data" / "raw" / "train.csv"
_RAW_STORES = ROOT_DIR / "data" / "raw" / "stores.csv"
_STORES_PROCESSED = ROOT_DIR / "data" / "processed" / "model_train.csv"  # has store_type/size merged

_TOP_N = 5  # number of top depts/stores to return


# ---------------------------------------------------------------------------
# Aggregation helper (shared logic — mirrors what Module 2 does)
# ---------------------------------------------------------------------------

def _compute_summary() -> dict:
    """
    Compute the admin summary from available data files.
    Raises FileNotFoundError if neither processed nor raw files exist.
    """
    # ── Load sales data ──────────────────────────────────────────────────────
    if _PROCESSED_TRAIN.exists():
        df = pd.read_csv(_PROCESSED_TRAIN, low_memory=False)
        # Normalise column names from processed schema
        df = df.rename(columns={
            "weekly_sales": "Weekly_Sales",
            "is_holiday": "IsHoliday",
            "store_id": "Store",
            "dept_id": "Dept",
            "date": "Date",
            "store_type": "Type",
            "store_size": "Size",
        })
    elif _RAW_TRAIN.exists():
        df = pd.read_csv(_RAW_TRAIN, low_memory=False)
        # Merge store metadata
        if _RAW_STORES.exists():
            stores_df = pd.read_csv(_RAW_STORES)
            df = df.merge(stores_df, on="Store", how="left")
    else:
        raise FileNotFoundError(
            "No data files found. Run Module 1's feature pipeline first "
            "or place raw CSVs in data/raw/."
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df["Weekly_Sales"] = pd.to_numeric(df["Weekly_Sales"], errors="coerce")

    # ── Scalar stats ─────────────────────────────────────────────────────────
    total_weekly_sales = float(df["Weekly_Sales"].sum())
    date_range = {
        "start": df["Date"].min().strftime("%Y-%m-%d"),
        "end":   df["Date"].max().strftime("%Y-%m-%d"),
    }
    store_count = int(df["Store"].nunique())
    dept_count  = int(df["Dept"].nunique())

    # ── Top departments ───────────────────────────────────────────────────────
    top_depts = (
        df.groupby("Dept")["Weekly_Sales"]
        .sum()
        .nlargest(_TOP_N)
        .reset_index()
        .rename(columns={"Dept": "dept_id", "Weekly_Sales": "total_sales"})
    )
    top_depts["dept_id"] = top_depts["dept_id"].astype(str)
    top_depts_list = top_depts.to_dict(orient="records")

    # ── Top stores ────────────────────────────────────────────────────────────
    store_sales = (
        df.groupby("Store")["Weekly_Sales"].sum().nlargest(_TOP_N).reset_index()
    )
    top_stores_list = []
    for _, row in store_sales.iterrows():
        store_row = df[df["Store"] == row["Store"]].iloc[0]
        top_stores_list.append({
            "store_id":    str(int(row["Store"])),
            "type":        str(store_row.get("Type", "")) if "Type" in store_row else "",
            "size":        int(store_row.get("Size", 0)) if "Size" in store_row else 0,
            "total_sales": float(row["Weekly_Sales"]),
        })

    # ── Holiday lift ──────────────────────────────────────────────────────────
    holiday_avg = df[df["IsHoliday"] == True]["Weekly_Sales"].mean()
    non_holiday_avg = df[df["IsHoliday"] == False]["Weekly_Sales"].mean()
    holiday_lift_pct = (
        round(((holiday_avg - non_holiday_avg) / non_holiday_avg) * 100, 4)
        if non_holiday_avg and non_holiday_avg != 0
        else 0.0
    )

    # ── Markdown lift ─────────────────────────────────────────────────────────
    md_cols = [c for c in df.columns if c.lower().startswith("markdown")]
    if md_cols:
        has_markdown = df[md_cols].notna().any(axis=1) & (df[md_cols].sum(axis=1) > 0)
        md_avg = df[has_markdown]["Weekly_Sales"].mean()
        no_md_avg = df[~has_markdown]["Weekly_Sales"].mean()
        markdown_lift_pct = (
            round(((md_avg - no_md_avg) / no_md_avg) * 100, 4)
            if no_md_avg and no_md_avg != 0
            else 0.0
        )
    else:
        markdown_lift_pct = 0.0

    return {
        "total_weekly_sales": total_weekly_sales,
        "date_range": date_range,
        "store_count": store_count,
        "dept_count": dept_count,
        "top_depts": top_depts_list,
        "top_stores": top_stores_list,
        "holiday_lift_pct": holiday_lift_pct,
        "markdown_lift_pct": markdown_lift_pct,
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get("/admin/summary")
def admin_summary(
    _admin_user: Annotated[dict, Depends(require_admin)],
):
    """
    GET /admin/summary

    Admin-only aggregated dataset stats. Same shape as /data/summary.
    Returns 403 if called by a manager role (enforced server-side, not
    just UI-hidden — per api_contract.md design requirement).

    Auth: admin only.
    """
    try:
        return _compute_summary()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": True,
                "message": str(exc),
                "status_code": 503,
            },
        )
