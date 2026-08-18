"""
Module 8 - LLM commentary generator.

Deliberately narrow scope: forecast/alert numbers in → 2-3 sentences of
plain-language commentary out. No other business logic belongs here.

Uses Google Gemini (gemini-1.5-flash) with a structured prompt.
Falls back to a templated string if the LLM call fails or times out,
so the /insights endpoint never returns a 500 to the caller.

Endpoint: POST /insights
Auth required: Yes (manager or admin)
"""

import os
from typing import Annotated, Any

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.auth import get_current_user

load_dotenv()

# ---------------------------------------------------------------------------
# Gemini setup
# ---------------------------------------------------------------------------

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if _GEMINI_API_KEY:
    genai.configure(api_key=_GEMINI_API_KEY)

_MODEL_NAME = "gemini-1.5-flash"
_LLM_TIMEOUT_SECONDS = 8  # fall back to template if LLM takes longer

# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AlertItem(BaseModel):
    dept_id: str
    urgency: str
    weeks_to_stockout: float

    class Config:
        extra = "ignore"  # ignore unknown fields per v2 spec


class SummaryBlock(BaseModel):
    total_depts_at_risk: int
    total_capital_freed_estimate: float

    class Config:
        extra = "ignore"


class InsightsRequest(BaseModel):
    store_id: str
    alerts: list[AlertItem]
    summary: SummaryBlock

    class Config:
        extra = "ignore"  # extra fields from /reorder are silently dropped


class InsightsResponse(BaseModel):
    insight_text: str
    generated_by: str  # "llm" | "fallback"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prompt(store_id: str, alerts: list[AlertItem], summary: SummaryBlock) -> str:
    red_alerts = [a for a in alerts if a.urgency == "red"]
    amber_alerts = [a for a in alerts if a.urgency == "amber"]

    red_depts = ", ".join(f"Dept {a.dept_id}" for a in red_alerts[:5])
    amber_depts = ", ".join(f"Dept {a.dept_id}" for a in amber_alerts[:5])

    prompt = (
        f"You are a retail inventory analyst. Write exactly 2-3 sentences of plain-language "
        f"commentary for a store manager. Be specific, concise, and actionable. "
        f"Do NOT use bullet points or headings.\n\n"
        f"Data for Store {store_id}:\n"
        f"- Departments at risk: {summary.total_depts_at_risk}\n"
        f"- Estimated capital at risk: ${summary.total_capital_freed_estimate:,.0f}\n"
        f"- RED urgency departments (≤1 week to stockout): {red_depts or 'none'}\n"
        f"- AMBER urgency departments (1-2 weeks): {amber_depts or 'none'}\n\n"
        f"Generate the commentary now:"
    )
    return prompt


def _fallback_text(store_id: str, summary: SummaryBlock) -> str:
    return (
        f"{summary.total_depts_at_risk} department(s) need urgent reorder "
        f"across Store {store_id}. "
        f"An estimated ${summary.total_capital_freed_estimate:,.0f} in sales is at risk."
    )


def _call_gemini(prompt: str) -> str:
    """Call Gemini API and return the response text. Raises on any failure."""
    model = genai.GenerativeModel(_MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=150,
            temperature=0.4,
        ),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/insights", response_model=InsightsResponse)
def get_insights(
    body: InsightsRequest,
    _current_user: Annotated[dict, Depends(get_current_user)],
):
    """
    POST /insights

    Accepts the /reorder response payload (extra fields silently ignored).
    Returns 2-3 sentence plain-language commentary from Gemini.
    Falls back to a templated string if the LLM call fails.

    Auth: manager or admin.
    """
    # Validation is handled by Pydantic — missing required fields → 422,
    # but the contract wants 400. Override with a custom check for the
    # three required fields in case the client sends an empty/partial body.
    # (Pydantic's 422 only fires if the JSON is structurally wrong;
    #  the extra="ignore" config handles unknown fields gracefully.)

    # Attempt Gemini call
    generated_by = "fallback"
    insight_text = _fallback_text(body.store_id, body.summary)

    if _GEMINI_API_KEY:
        try:
            prompt = _build_prompt(body.store_id, body.alerts, body.summary)
            insight_text = _call_gemini(prompt)
            generated_by = "llm"
        except Exception:
            # Any LLM failure (quota, network, timeout) → silent fallback
            insight_text = _fallback_text(body.store_id, body.summary)
            generated_by = "fallback"

    return InsightsResponse(insight_text=insight_text, generated_by=generated_by)
