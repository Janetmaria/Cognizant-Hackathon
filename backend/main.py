"""
Module 7 - FastAPI app entrypoint: forecast-serving + business logic.

Registers routes from backend/reorder_routes.py and defines the main /forecast/{store_id}
endpoint directly. Ensures complete alignment with docs/api-contract-v2.md.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Load .env file automatically (supports GROQ_API_KEY, OPENAI_API_KEY, JWT_SECRET_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import backend.auth as auth
import backend.database as database
import backend.llm_insights as llm_insights
import backend.reorder_routes as reorder_routes
import backend.data_routes as data_routes
import src.reorder_logic as reorder_logic


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield


app = FastAPI(
    title="Retail Sales Forecasting & Inventory Intelligence API",
    description="Hackathon backend serving department forecasts and inventory reorder alerts.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, tags=["Authentication & RBAC"])
app.include_router(reorder_routes.router, tags=["Inventory Intelligence"])

app.include_router(llm_insights.router, tags=["Executive Insights"])
app.include_router(data_routes.router, tags=["Data Summary"])
# Ensure tables/seeds exist for import-time clients (e.g. TestClient without lifespan context).
database.init_db()

app.include_router(data_routes.router, tags=["Data Summary"])


try:
    import config
    PREDICTIONS_PATH = config.PROCESSED_DIR / "lightgbm_predictions.csv"
except (ImportError, AttributeError):
    PREDICTIONS_PATH = Path("data/processed/lightgbm_predictions.csv")


@app.get("/")
def get_root():
    return {
        "status": "healthy",
        "service": "Retail Sales Forecasting API",
        "docs": "/docs",
    }


@app.get("/forecast/{store_id}")
def get_forecast(
    store_id: str,
    dept_id: str = Query(..., description="Department Identifier (e.g. '92')"),
    start_date: str = Query(..., description="Forecast start date in YYYY-MM-DD format (e.g. '2026-08-21')"),
    end_date: str = Query(..., description="Forecast end date in YYYY-MM-DD format (e.g. '2026-09-11')"),
    model: str = Query("lightgbm", description="Forecasting model to use ('lightgbm', 'baseline')"),
    current_user: dict = Depends(auth.require_manager_or_admin),
):
    """
    Retrieve weekly sales projections for a store department over a date window.
    Requires manager or admin role.
    """
    if model.lower() not in ["lightgbm", "baseline"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "message": "Model parameter must be one of: 'lightgbm', 'baseline'",
                "status_code": 400,
            },
        )

    if not PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "message": f"Predictions data file {PREDICTIONS_PATH} not found. Train the model first.",
                "status_code": 500,
            },
        )

    try:
        df = pd.read_csv(PREDICTIONS_PATH)

        try:
            store_int = int(store_id)
            dept_int = int(dept_id)
        except ValueError:
            store_int = store_id
            dept_int = dept_id

        hist_start = reorder_logic.map_date_backward(start_date)
        hist_end = reorder_logic.map_date_backward(end_date)

        matched_df = df[
            (df["store_id"] == store_int)
            & (df["dept_id"] == dept_int)
            & (df["date"] >= hist_start)
            & (df["date"] <= hist_end)
        ].copy()

        matched_df = matched_df.sort_values(by="date")

        predictions_list: List[dict] = []

        for _, row in matched_df.iterrows():
            raw_date = str(row["date"])
            display_date = reorder_logic.map_date_forward(raw_date)

            if model.lower() == "baseline":
                predicted_val = float(row.get("baseline_pred", 0.0))
            else:
                predicted_val = float(row.get("lgbm_pred", row.get("baseline_pred", 0.0)))

            predicted_val = max(0.0, predicted_val)
            actual_val = float(row.get("weekly_sales", 0.0))

            predictions_list.append({
                "week_ending_date": display_date,
                "predicted_weekly_sales": round(predicted_val, 2),
                "actual_weekly_sales": round(actual_val, 2),
                "is_holiday": bool(row.get("is_holiday", False)),
                "lower_bound": None,
                "upper_bound": None,
            })

        return {
            "store_id": str(store_id),
            "dept_id": str(dept_id),
            "model_used": model.lower(),
            "predictions": predictions_list,
        }

    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "message": f"Server failed parsing forecast request: {str(err)}",
                "status_code": 500,
            },
        )
