"""
Module 7 - FastAPI app entrypoint: forecast-serving + business logic.

Wires up routers from all modules. See docs/api_contract.md for the
response envelope and endpoint contracts.

Module 8 routers (auth, insights, admin_summary) are pre-registered here
so Module 7 only needs to add the /forecast/{store_id} and /whatif routes.
"""

from fastapi import FastAPI

# ── Module 8 routers (already implemented) ─────────────────────────────────
from backend.auth import router as auth_router
from backend.llm_insights import router as insights_router
from backend.admin_routes import router as admin_router
from backend.database import init_db

# ── Module 2 router ─────────────────────────────────────────────────────────
from backend.data_routes import router as data_router        # TODO(Module 2): implement

# ── Module 6 router ─────────────────────────────────────────────────────────
from backend.reorder_routes import router as reorder_router  # TODO(Module 6): implement

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Retail Sales Forecasting Platform",
    description="Walmart store sales forecasting API — see docs/api_contract.md",
    version="2.0.0",
)

# Seed the user DB on startup (idempotent — safe to call every boot)
@app.on_event("startup")
def on_startup():
    init_db()

# ── Include routers ──────────────────────────────────────────────────────────
app.include_router(auth_router)          # POST /auth/login
app.include_router(insights_router)      # POST /insights
app.include_router(admin_router)         # GET  /admin/summary
app.include_router(data_router)          # GET  /data/summary         (Module 2)
app.include_router(reorder_router)       # GET  /reorder/{store_id}   (Module 6)

# TODO(Module 7): add /forecast/{store_id} route here
# TODO(Module 7): add /whatif route here (wires into Module 6 logic)
