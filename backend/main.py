"""
Module 7 - Retail Sales Forecasting & Functional Bridge API
FastAPI Backend Server with CORS, /forecast/{store_id}, and /whatif simulation endpoints.
"""

from fastapi import FastAPI, HTTPException, Query, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Optional, List
import datetime as dt
from datetime import datetime, timedelta
import math
import random
from pathlib import Path

app = FastAPI(
    title="Retail Sales Forecasting Platform API",
    description="Module 7 Forecast Serving and What-If Simulation API for Retail Store Management",
    version="2.0.0"
)

# Enable CORS middleware to allow cross-origin requests from frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 1. ERROR HANDLING OVERRIDES
# ==========================================

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Ensure all HTTP errors match the unified API contract format:
    {"error": true, "message": "...", "status_code": ...}
    """
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": str(exc.detail),
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Format Pydantic schema validation errors into the standard contract shape.
    """
    error_msg = "; ".join([f"{err['loc'][-1]}: {err['msg']}" for err in exc.errors()])
    return JSONResponse(
        status_code=400,
        content={
            "error": True,
            "message": f"Validation Error: {error_msg}",
            "status_code": 400
        }
    )


# ==========================================
# 2. PYDANTIC SCHEMAS
# ==========================================

class PredictionItem(BaseModel):
    week_ending_date: str
    predicted_weekly_sales: float
    actual_weekly_sales: Optional[float] = None
    is_holiday: bool
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ForecastResponse(BaseModel):
    store_id: str
    dept_id: str
    model_used: str
    predictions: List[PredictionItem]


class WhatIfRequest(BaseModel):
    store_id: str = Field(..., description="Store identifier (e.g., '4')")
    dept_id: str = Field(..., description="Department identifier (e.g., '92')")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    markdown_amount: float = Field(..., description="Promotional markdown spending amount in dollars")
    is_holiday_override: Optional[bool] = Field(None, description="Optional boolean override for holiday status")


class WhatIfPredictionItem(BaseModel):
    week_ending_date: str
    predicted_weekly_sales: float


class WhatIfResponse(BaseModel):
    store_id: str
    dept_id: str
    baseline_predictions: List[WhatIfPredictionItem]
    adjusted_predictions: List[WhatIfPredictionItem]
    projected_lift_pct: float


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def parse_and_validate_dates(start_date_str: str, end_date_str: str):
    """
    Parse dates and enforce maximum 104-week date range limit.
    Returns (start_date, end_date) tuple or raises standard JSON 400 error.
    """
    try:
        s_date = datetime.strptime(start_date_str.strip(), "%Y-%m-%d")
        e_date = datetime.strptime(end_date_str.strip(), "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "message": "Invalid date format. Expected YYYY-MM-DD",
                "status_code": 400
            }
        )

    if e_date < s_date:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "message": "end_date must be on or after start_date",
                "status_code": 400
            }
        )

    # 104 weeks = 104 * 7 days = 728 days
    date_diff_days = (e_date - s_date).days
    if date_diff_days > (104 * 7):
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "message": "Date range exceeds maximum of 104 weeks",
                "status_code": 400
            }
        )

    return s_date, e_date


def generate_mock_predictions(store_id: str, dept_id: str, start_dt: datetime, end_dt: datetime, model: str):
    """
    Generate realistic weekly prediction points between start_dt and end_dt.
    Uses deterministic math seeded by store_id and dept_id for consistency.
    """
    predictions = []
    
    # Calculate base level from store & dept
    try:
        s_num = int(store_id)
        d_num = int(dept_id)
    except ValueError:
        s_num, d_num = 4, 92

    base_sales = 15000.0 + ((s_num * 1337 + d_num * 719) % 25000)
    
    current_dt = start_dt
    step_week = 0

    while current_dt <= end_dt:
        week_date_str = current_dt.strftime("%Y-%m-%d")
        
        # Determine holiday (simulate November/December holidays or cyclic)
        is_holiday = (current_dt.month in [2, 9, 11, 12] and current_dt.day in range(5, 16)) or (step_week % 8 == 0)
        holiday_multiplier = 1.18 if is_holiday else 1.0
        
        # Seasonal wave
        seasonal_factor = 1.0 + 0.15 * math.sin(step_week * 0.5) + 0.08 * math.cos(step_week * 0.2)
        noise = (((step_week * 37 + d_num * 17) % 100) - 50) * 15.0
        
        pred_val = (base_sales * seasonal_factor * holiday_multiplier) + noise
        pred_val = max(1200.0, pred_val)
        
        # Historical actual sales simulation (slight variance from prediction)
        actual_val = pred_val * (1.0 + (((step_week * 13 + s_num * 7) % 30) - 15) / 100.0)
        
        # Uncertainty bounds for prophet model
        if model.lower() == "prophet":
            lower_bound = round(pred_val * 0.91, 2)
            upper_bound = round(pred_val * 1.09, 2)
        else:
            lower_bound = None
            upper_bound = None

        predictions.append({
            "week_ending_date": week_date_str,
            "predicted_weekly_sales": round(pred_val, 2),
            "actual_weekly_sales": round(actual_val, 2),
            "is_holiday": is_holiday,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        })

        # Advance by 1 week (7 days)
        current_dt += timedelta(days=7)
        step_week += 1

    # Guarantee at least 1 prediction object if start_dt == end_dt
    if not predictions:
        predictions.append({
            "week_ending_date": start_dt.strftime("%Y-%m-%d"),
            "predicted_weekly_sales": round(base_sales, 2),
            "actual_weekly_sales": round(base_sales * 0.98, 2),
            "is_holiday": False,
            "lower_bound": None,
            "upper_bound": None
        })

    return predictions


# ==========================================
# 4. ROUTE DEFINITIONS
# ==========================================

@app.get("/")
def get_root():
    """
    Health check and platform status.
    """
    return {
        "status": "online",
        "module": "Module 7 - Visual and Functional Bridge",
        "description": "Retail Sales Forecasting Platform API",
        "endpoints": {
            "forecast": "/forecast/{store_id}",
            "whatif": "/whatif",
            "reorder": "/reorder/{store_id}",
            "docs": "/docs"
        }
    }


@app.get("/forecast/{store_id}", response_model=ForecastResponse)
def get_forecast(
    store_id: str,
    dept_id: str = Query(..., description="Department Identifier (e.g. '92')"),
    start_date: str = Query(..., description="Forecast start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="Forecast end date in YYYY-MM-DD format"),
    model: str = Query("lightgbm", description="Forecasting model (e.g. 'lightgbm', 'prophet', 'sarima', 'baseline')")
):
    """
    GET /forecast/{store_id}
    Retrieves weekly sales predictions for a specific store and department.
    Validates that the date range does not exceed 104 weeks (returns 400 if exceeded).
    """
    # 1. Validate date range and 104-week limit
    start_dt, end_dt = parse_and_validate_dates(start_date, end_date)

    # 2. Generate predictions array
    predictions = generate_mock_predictions(
        store_id=store_id,
        dept_id=dept_id,
        start_dt=start_dt,
        end_dt=end_dt,
        model=model
    )

    return {
        "store_id": str(store_id),
        "dept_id": str(dept_id),
        "model_used": model.lower(),
        "predictions": predictions
    }


@app.post("/whatif", response_model=WhatIfResponse)
def post_whatif(payload: WhatIfRequest):
    """
    POST /whatif
    Simulates promotional markdown impact and holiday overrides on department forecasts.
    Returns baseline predictions, adjusted predictions, and projected lift percentage.
    """
    # 1. Validation
    if payload.markdown_amount < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "message": "markdown_amount must be zero or positive",
                "status_code": 400
            }
        )

    try:
        start_dt = datetime.strptime(payload.start_date.strip(), "%Y-%m-%d")
        end_dt = datetime.strptime(payload.end_date.strip(), "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "message": "Invalid date format. Expected YYYY-MM-DD",
                "status_code": 400
            }
        )

    if end_dt < start_dt:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "message": "end_date must be on or after start_date",
                "status_code": 400
            }
        )

    # 2. Calculate promotion lift factor using Michaelis-Menten saturation curve
    # Lift % = Max Lift (30%) * Markdown / (Markdown + Half-Saturation Point ($2000))
    if payload.markdown_amount > 0:
        markdown_lift_rate = 0.30 * (payload.markdown_amount / (payload.markdown_amount + 2000.0))
    else:
        markdown_lift_rate = 0.0

    HOLIDAY_MULTIPLIER = 1.074  # 7.4% holiday lift

    # 3. Generate baseline and adjusted weekly trajectories
    baseline_predictions = []
    adjusted_predictions = []
    
    current_dt = start_dt
    step = 0
    
    try:
        s_num = int(payload.store_id)
        d_num = int(payload.dept_id)
    except ValueError:
        s_num, d_num = 4, 92

    base_val = 18000.0 + ((s_num * 883 + d_num * 457) % 16000)

    while current_dt <= end_dt:
        week_date_str = current_dt.strftime("%Y-%m-%d")
        
        # Baseline seasonality & natural holiday
        natural_holiday = (step % 5 == 0) or (current_dt.month in [11, 12])
        seasonal_wave = 1.0 + 0.12 * math.sin(step * 0.6)
        
        base_sales = base_val * seasonal_wave * (HOLIDAY_MULTIPLIER if natural_holiday else 1.0)
        
        # Adjusted sales calculation with markdown + holiday override
        active_holiday = payload.is_holiday_override if payload.is_holiday_override is not None else natural_holiday
        
        # Strip natural holiday if applied, then apply promo and active holiday
        core_sales = base_sales / HOLIDAY_MULTIPLIER if natural_holiday else base_sales
        promoted_sales = core_sales * (1.0 + markdown_lift_rate)
        adjusted_sales = promoted_sales * (HOLIDAY_MULTIPLIER if active_holiday else 1.0)

        baseline_predictions.append({
            "week_ending_date": week_date_str,
            "predicted_weekly_sales": round(base_sales, 2)
        })
        
        adjusted_predictions.append({
            "week_ending_date": week_date_str,
            "predicted_weekly_sales": round(adjusted_sales, 2)
        })

        current_dt += timedelta(days=7)
        step += 1

    if not baseline_predictions:
        baseline_predictions.append({"week_ending_date": payload.start_date, "predicted_weekly_sales": round(base_val, 2)})
        adjusted_predictions.append({"week_ending_date": payload.start_date, "predicted_weekly_sales": round(base_val * (1.0 + markdown_lift_rate), 2)})

    # Calculate overall lift percentage
    total_baseline = sum(item["predicted_weekly_sales"] for item in baseline_predictions)
    total_adjusted = sum(item["predicted_weekly_sales"] for item in adjusted_predictions)
    
    if total_baseline > 0:
        projected_lift_pct = ((total_adjusted - total_baseline) / total_baseline) * 100.0
    else:
        projected_lift_pct = 0.0

    return {
        "store_id": str(payload.store_id),
        "dept_id": str(payload.dept_id),
        "baseline_predictions": baseline_predictions,
        "adjusted_predictions": adjusted_predictions,
        "projected_lift_pct": round(projected_lift_pct, 2)
    }


@app.get("/reorder/{store_id}")
def get_reorder_recommendations(
    store_id: str,
    urgency: Optional[str] = Query(None, description="Optional urgency filter ('red', 'amber', 'green')")
):
    """
    GET /reorder/{store_id}
    Returns department stock health and reorder recommendations for the store.
    """
    mock_alerts = [
        {"dept_id": "92", "current_stock_units": 320, "predicted_weekly_demand_units": 410, "weeks_to_stockout": 0.78, "urgency": "red", "recommended_reorder_units": 500, "capital_freed_estimate": 0.0},
        {"dept_id": "95", "current_stock_units": 850, "predicted_weekly_demand_units": 620, "weeks_to_stockout": 1.37, "urgency": "amber", "recommended_reorder_units": 390, "capital_freed_estimate": 0.0},
        {"dept_id": "38", "current_stock_units": 120, "predicted_weekly_demand_units": 280, "weeks_to_stockout": 0.43, "urgency": "red", "recommended_reorder_units": 440, "capital_freed_estimate": 0.0},
        {"dept_id": "72", "current_stock_units": 1450, "predicted_weekly_demand_units": 800, "weeks_to_stockout": 1.81, "urgency": "amber", "recommended_reorder_units": 150, "capital_freed_estimate": 0.0},
        {"dept_id": "40", "current_stock_units": 3100, "predicted_weekly_demand_units": 950, "weeks_to_stockout": 3.26, "urgency": "green", "recommended_reorder_units": 0, "capital_freed_estimate": 1850.0},
        {"dept_id": "90", "current_stock_units": 2400, "predicted_weekly_demand_units": 710, "weeks_to_stockout": 3.38, "urgency": "green", "recommended_reorder_units": 0, "capital_freed_estimate": 2340.0},
        {"dept_id": "1",  "current_stock_units": 90,   "predicted_weekly_demand_units": 210, "weeks_to_stockout": 0.43, "urgency": "red", "recommended_reorder_units": 330, "capital_freed_estimate": 0.0}
    ]

    if urgency:
        mock_alerts = [a for a in mock_alerts if a["urgency"].lower() == urgency.lower()]

    depts_at_risk = sum(1 for a in mock_alerts if a["urgency"] in ["red", "amber"])
    total_capital = sum(a["capital_freed_estimate"] for a in mock_alerts)

    return {
        "store_id": str(store_id),
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "alerts": mock_alerts,
        "summary": {
            "total_depts_at_risk": depts_at_risk,
            "total_capital_freed_estimate": round(total_capital, 2)
        },
        "data_note": "current_stock_units is simulated for inventory optimization."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
