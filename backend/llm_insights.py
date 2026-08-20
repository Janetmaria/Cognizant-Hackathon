"""
Module 8 - Executive LLM Commentary & Inventory Intelligence Generator.

Narrow executive commentary layer:
Receives structured summary metrics (forecast totals, departments at risk,
stockout urgency counts, capital freed estimates) and generates 2-3 sentences
of actionable business commentary.

Resilience Guarantee:
- If an LLM API key (GROQ_API_KEY, OPENAI_API_KEY, or LLM_API_KEY) is available,
  calls the model using a strict executive system prompt.
- If the key is missing, network fails, or timeout occurs, gracefully falls back
  to a deterministic, high-quality rule-based template generator.
- Strictly protected by JWT authentication (RBAC: manager or admin).
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

# Import authentication dependencies
import backend.auth as auth

# Configure logger
logger = logging.getLogger("retail_insights")
logger.setLevel(logging.INFO)

# =============================================================================
# 1. PYDANTIC REQUEST & RESPONSE SCHEMAS
# =============================================================================

class InsightRequest(BaseModel):
    """
    Structured summary metrics payload for executive commentary generation.
    Supports both direct metric inputs and payloads mirroring Module 6 /reorder responses.
    """
    store_id: str = Field(..., description="Store Identifier (e.g. '4')")
    total_depts_at_risk: int = Field(0, description="Total number of departments showing stockout risk")
    red_alerts: int = Field(0, description="Number of departments with urgent (red) stockout urgency")
    amber_alerts: int = Field(0, description="Number of departments with moderate (amber) stockout urgency")
    capital_freed_estimate: float = Field(0.0, description="Estimated working capital in dollars freed by inventory optimization")
    projected_weekly_sales: Optional[float] = Field(None, description="Total projected weekly sales for the store/window in dollars")

    # Extra fields (such as 'alerts' or 'summary' from /reorder) are ignored, not rejected
    model_config = {"extra": "ignore"}

    @model_validator(mode="before")
    @classmethod
    def parse_flexible_payload(cls, data: Any) -> Any:
        """
        Extracts metrics from nested /reorder or summary shapes if provided.
        """
        if isinstance(data, dict):
            alerts = data.get("alerts")
            summary = data.get("summary")

            if isinstance(alerts, list):
                if "red_alerts" not in data:
                    data["red_alerts"] = sum(
                        1 for a in alerts if isinstance(a, dict) and str(a.get("urgency", "")).lower() == "red"
                    )
                if "amber_alerts" not in data:
                    data["amber_alerts"] = sum(
                        1 for a in alerts if isinstance(a, dict) and str(a.get("urgency", "")).lower() == "amber"
                    )
                if "total_depts_at_risk" not in data:
                    data["total_depts_at_risk"] = len(alerts)

            if isinstance(summary, dict):
                if "total_depts_at_risk" not in data and "total_depts_at_risk" in summary:
                    try:
                        data["total_depts_at_risk"] = int(summary["total_depts_at_risk"])
                    except (ValueError, TypeError):
                        pass
                if "capital_freed_estimate" not in data:
                    val = summary.get("total_capital_freed_estimate", summary.get("capital_freed_estimate", 0.0))
                    try:
                        data["capital_freed_estimate"] = float(val)
                    except (ValueError, TypeError):
                        data["capital_freed_estimate"] = 0.0
                if "projected_weekly_sales" not in data and "projected_weekly_sales" in summary:
                    try:
                        data["projected_weekly_sales"] = float(summary["projected_weekly_sales"])
                    except (ValueError, TypeError):
                        pass
        return data


class InsightResponse(BaseModel):
    """
    Executive commentary response schema.
    """
    store_id: str = Field(..., description="Store Identifier")
    commentary: str = Field(..., description="2-3 sentence actionable executive business commentary")
    generated_at: str = Field(..., description="ISO 8601 UTC timestamp of insight generation")
    source: str = Field(..., description="Commentary source: 'llm' or 'template_fallback'")
    
    # Contract compatibility aliases
    insight_text: Optional[str] = Field(None, description="Alias for commentary for contract v2 compatibility")
    generated_by: Optional[str] = Field(None, description="Alias for source ('llm' or 'fallback')")


# =============================================================================
# 2. DETERMINISTIC RULE-BASED TEMPLATE GENERATOR
# =============================================================================

def generate_template_commentary(
    store_id: str,
    total_depts_at_risk: int,
    red_alerts: int,
    amber_alerts: int,
    capital_freed_estimate: float,
    projected_weekly_sales: Optional[float] = None,
) -> str:
    """
    High-quality deterministic template generator that produces realistic,
    executive-ready 2-3 sentence commentary based strictly on the input metrics.
    """
    store_str = str(store_id).strip()
    cap_formatted = f"${capital_freed_estimate:,.2f}" if capital_freed_estimate > 0 else "$0.00"
    
    red_unit = f"{red_alerts} department" if red_alerts == 1 else f"{red_alerts} departments"
    amber_unit = f"{amber_alerts} department" if amber_alerts == 1 else f"{amber_alerts} departments"
    total_unit = f"{total_depts_at_risk} department" if total_depts_at_risk == 1 else f"{total_depts_at_risk} departments"

    sales_phrase = ""
    if projected_weekly_sales and projected_weekly_sales > 0:
        sales_phrase = f"to protect projected weekly revenue of ${projected_weekly_sales:,.2f} and "

    # Case 1: Critical urgency (Red alerts > 0)
    if red_alerts > 0:
        sentence_1 = (
            f"Store {store_str} exhibits critical inventory vulnerability with {red_unit} in immediate "
            f"red-alert stockout risk and {amber_unit} approaching safety threshold."
        )
        sentence_2 = (
            f"Immediate purchase order escalation is required {sales_phrase}to prevent revenue loss "
            f"across high-velocity categories."
        )
        sentence_3 = (
            f"Executing targeted inventory reorders will restore shelf availability while optimizing "
            f"an estimated {cap_formatted} in freed working capital."
        )
        return f"{sentence_1} {sentence_2} {sentence_3}"

    # Case 2: Moderate urgency (Amber alerts > 0 or depts at risk > 0)
    elif amber_alerts > 0 or total_depts_at_risk > 0:
        sentence_1 = (
            f"Store {store_str} maintains stable operations with zero red stockouts, though {amber_unit} "
            f"remain on amber watch for inventory depletion."
        )
        sentence_2 = (
            f"Proactive supplier replenishment should be scheduled {sales_phrase}to maintain safety "
            f"stock buffers before weekly demand cycles peak."
        )
        sentence_3 = (
            f"Rebalancing inventory across these {total_unit} will protect on-shelf availability and "
            f"unlock {cap_formatted} in working capital efficiency."
        )
        return f"{sentence_1} {sentence_2} {sentence_3}"

    # Case 3: Optimal / Healthy state (0 alerts)
    else:
        sales_support = (
            f"comfortably supports projected weekly revenue of ${projected_weekly_sales:,.2f}"
            if (projected_weekly_sales and projected_weekly_sales > 0)
            else "comfortably supports forecasted customer demand"
        )
        sentence_1 = (
            f"Store {store_str} demonstrates strong inventory health with zero departments currently "
            f"flagged for stockout risk."
        )
        sentence_2 = (
            f"Current stock positioning {sales_support} without immediate replenishment intervention needed."
        )
        sentence_3 = (
            f"Standard procurement cadences should be maintained while monitoring working capital "
            f"efficiencies estimated at {cap_formatted}."
        )
        return f"{sentence_1} {sentence_2} {sentence_3}"


# =============================================================================
# 3. LLM INVOCATION LAYER (RESILIENT CLIENT)
# =============================================================================

STRICT_SYSTEM_PROMPT = """You are a senior executive retail supply chain and inventory analyst.
Analyze the provided store inventory metrics and generate EXACTLY 2 to 3 concise, highly actionable business commentary sentences.

Rules:
1. Write strictly 2 to 3 sentences of professional executive commentary.
2. Directly cite the provided numbers (store ID, departments at risk, alert counts, capital freed, and projected sales if given).
3. Recommend concrete operational steps (e.g. prioritizing purchase orders for urgent stockout departments, expediting supplier replenishment, and optimizing safety stock).
4. Do NOT output markdown headers, bullet points, introductory preamble (like 'Here is your commentary:'), conversational filler, or chatbot greetings.
5. Focus strictly on inventory risk, working capital efficiency, and revenue protection."""


def _call_http_llm(
    endpoint_url: str,
    api_key: str,
    model_name: str,
    user_prompt: str,
    timeout_seconds: float = 10.0,
) -> Optional[str]:
    """
    Sends chat completion request to an OpenAI-compatible API endpoint via httpx/requests.
    Returns cleaned commentary text on success, or None on failure/timeout.
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": STRICT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 200,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        # Prefer httpx if available
        try:
            import httpx
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(endpoint_url, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    # Clean any spurious markdown quotes
                    content = content.replace('"', '').strip()
                    if content:
                        return content
                else:
                    logger.warning("LLM API returned non-200 status: %s - %s", response.status_code, response.text[:200])
        except ImportError:
            import urllib.request
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint_url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"].strip()
                    content = content.replace('"', '').strip()
                    if content:
                        return content
    except Exception as exc:
        logger.warning("LLM call encountered error (%s). Falling back to template generator.", exc)

    return None


def call_llm_commentary(
    store_id: str,
    total_depts_at_risk: int,
    red_alerts: int,
    amber_alerts: int,
    capital_freed_estimate: float,
    projected_weekly_sales: Optional[float] = None,
) -> Optional[str]:
    """
    Attempts to call available LLM providers (Groq, OpenAI, or Custom LLM).
    Returns generated commentary if successful, else None.
    """
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    generic_api_key = os.getenv("LLM_API_KEY", "").strip()

    if not (groq_api_key or openai_api_key or generic_api_key):
        return None

    # Construct concise metric prompt
    sales_line = f"- Projected Weekly Sales: ${projected_weekly_sales:,.2f}" if projected_weekly_sales else "- Projected Weekly Sales: N/A"
    user_prompt = f"""Store Inventory Metrics:
- Store ID: {store_id}
- Total Departments at Risk: {total_depts_at_risk}
- Critical Red Alerts (< 1 week to stockout): {red_alerts}
- Moderate Amber Alerts (1-3 weeks to stockout): {amber_alerts}
- Estimated Working Capital Freed: ${capital_freed_estimate:,.2f}
{sales_line}

Generate 2-3 sentences of actionable executive commentary:"""

    # 1. Try Groq if configured
    if groq_api_key:
        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        result = _call_http_llm(
            endpoint_url="https://api.groq.com/openai/v1/chat/completions",
            api_key=groq_api_key,
            model_name=groq_model,
            user_prompt=user_prompt,
        )
        if result:
            return result

    # 2. Try OpenAI if configured
    if openai_api_key:
        openai_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        result = _call_http_llm(
            endpoint_url=openai_url,
            api_key=openai_api_key,
            model_name=openai_model,
            user_prompt=user_prompt,
        )
        if result:
            return result

    # 3. Try Generic LLM if configured
    if generic_api_key:
        llm_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        result = _call_http_llm(
            endpoint_url=llm_url,
            api_key=generic_api_key,
            model_name=llm_model,
            user_prompt=user_prompt,
        )
        if result:
            return result

    return None


def generate_executive_commentary(payload: InsightRequest) -> Tuple[str, str]:
    """
    Orchestrates commentary generation: tries LLM first, falls back to deterministic template.
    Returns (commentary_text, source).
    """
    llm_output = call_llm_commentary(
        store_id=payload.store_id,
        total_depts_at_risk=payload.total_depts_at_risk,
        red_alerts=payload.red_alerts,
        amber_alerts=payload.amber_alerts,
        capital_freed_estimate=payload.capital_freed_estimate,
        projected_weekly_sales=payload.projected_weekly_sales,
    )

    if llm_output:
        return llm_output, "llm"

    fallback_text = generate_template_commentary(
        store_id=payload.store_id,
        total_depts_at_risk=payload.total_depts_at_risk,
        red_alerts=payload.red_alerts,
        amber_alerts=payload.amber_alerts,
        capital_freed_estimate=payload.capital_freed_estimate,
        projected_weekly_sales=payload.projected_weekly_sales,
    )
    return fallback_text, "template_fallback"


# =============================================================================
# 4. FASTAPI ROUTER & ENDPOINTS
# =============================================================================

router = APIRouter(prefix="/insights", tags=["Executive Insights"])


@router.post(
    "/generate",
    response_model=InsightResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Executive Inventory Commentary",
    description=(
        "Produces 2-3 sentences of actionable business commentary from structured summary metrics. "
        "Utilizes LLM if configured or deterministic rule-based template fallback on failure. "
        "Requires valid JWT with manager or admin role."
    ),
)
def generate_insights_endpoint(
    payload: InsightRequest,
    current_user: Dict[str, Any] = Depends(auth.require_manager_or_admin),
) -> InsightResponse:
    """
    POST /insights/generate handler. Enforces store isolation for managers.
    """
    auth.enforce_store_access(payload.store_id, current_user)
    commentary_text, source = generate_executive_commentary(payload)
    now_iso = datetime.now(timezone.utc).isoformat()

    return InsightResponse(
        store_id=payload.store_id,
        commentary=commentary_text,
        generated_at=now_iso,
        source=source,
        insight_text=commentary_text,
        generated_by="llm" if source == "llm" else "fallback",
    )


@router.post(
    "",
    response_model=InsightResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Executive Inventory Commentary (Root Alias)",
    description="Alias endpoint matching POST /insights specifications.",
    include_in_schema=True,
)
@router.post(
    "/",
    response_model=InsightResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def generate_insights_root_alias(
    payload: InsightRequest,
    current_user: Dict[str, Any] = Depends(auth.require_manager_or_admin),
) -> InsightResponse:
    """
    POST /insights root alias handler for contract v2 compatibility.
    """
    return generate_insights_endpoint(payload=payload, current_user=current_user)
