"""
CSV Upload -> Forecast + Reorder Alerts.

Exposes POST /upload/forecast: accepts a multipart CSV file shaped like our
raw Kaggle data (Store, Dept, Date, Weekly_Sales at minimum), matches the
uploaded (store, dept) pairs against the already-trained hybrid model's
predictions, and returns forecasts + reorder alerts. See
src/upload_pipeline.py for the core logic; this file only wires up auth,
request parsing, and the standard HTTP error envelope.
"""

import io
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

import backend.auth as auth
import src.upload_pipeline as upload_pipeline

router = APIRouter()


@router.post("/upload/forecast")
async def upload_forecast(
    file: UploadFile = File(..., description="CSV with at least Store, Dept, Date, Weekly_Sales columns"),
    urgency: Optional[str] = Query(None, description="Filter reorder alerts by urgency ('red', 'amber', 'green')"),
    as_of_date: str = Query("2026-08-21", description="Reference run date for the forecast lookup (default: '2026-08-21')"),
    current_user: dict = Depends(auth.require_manager_or_admin),
):
    """
    Validates the uploaded CSV, engineers display stats from the user's own
    sales history, matches (store, dept) pairs against the hybrid model's
    precomputed forecasts, and returns forecasts + reorder alerts for the
    matched rows. Store managers only get results for their assigned store.
    """
    if urgency and urgency.lower() not in ["red", "amber", "green"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "message": "Urgency filter must be one of: 'red', 'amber', 'green'",
                "status_code": 400,
            },
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=400,
            detail={"error": True, "message": "Uploaded file is empty", "status_code": 400},
        )

    try:
        raw_df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "message": f"Could not parse uploaded file as CSV: {err}",
                "status_code": 400,
            },
        )

    # Store managers only see their own store's rows. Filtered rather than
    # rejected outright since one upload can legitimately span many stores.
    user_role = current_user.get("role", "").lower()
    assigned_store = current_user.get("assigned_store")
    if user_role != "admin" and assigned_store and str(assigned_store).strip() and "Store" in raw_df.columns:
        try:
            assigned_store_num = int(str(assigned_store).strip())
        except ValueError:
            assigned_store_num = None
        if assigned_store_num is not None:
            store_numeric = pd.to_numeric(raw_df["Store"], errors="coerce")
            raw_df = raw_df[store_numeric == assigned_store_num]
            if raw_df.empty:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": True,
                        "message": f"None of the uploaded rows belong to your assigned store ({assigned_store}).",
                        "status_code": 403,
                    },
                )

    try:
        result = upload_pipeline.process_upload(raw_df, urgency_filter=urgency, as_of_date=as_of_date)
    except ValueError as err:
        raise HTTPException(
            status_code=400,
            detail={"error": True, "message": str(err), "status_code": 400},
        )
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=500,
            detail={"error": True, "message": str(err), "status_code": 500},
        )
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "message": f"An unexpected error occurred: {str(err)}",
                "status_code": 500,
            },
        )

    return result
