# Retail Sales Forecasting Platform

> Hackathon project — Walmart Recruiting: Store Sales Forecasting (Kaggle)

A full-stack machine-learning platform that forecasts weekly retail sales at the store + department level, generates reorder alerts, and provides LLM-powered inventory insights.

---

## What It Does

- **Forecasts** weekly sales per store/department using four models (Naive baseline, SARIMA, Prophet, LightGBM)
- **Alerts** managers when departments are at risk of stockout (red/amber/green urgency tiers)
- **What-if analysis** — simulate the effect of markdown promotions on sales
- **LLM insights** — plain-language commentary on reorder risk, powered by Google Gemini
- **Role-based access** — managers see forecasts and alerts; admins get additional aggregation endpoints (JWT auth, server-side enforced)

---

## Project Structure

```
data/
  raw/                    # Original Kaggle CSVs — never edit in place
  processed/              # Cleaned/merged features (Module 1 output)
src/
  features/               # Module 1 — feature engineering
  models/
    baseline.py           # Module 5 — seasonal-naive baseline
    sarima_model.py       # Module 3 — SARIMA
    prophet_model.py      # Module 4 — Prophet with confidence intervals
    lightgbm_model.py     # Module 5 — LightGBM
  evaluate.py             # Module 5 — MAPE/RMSE evaluation
  reorder_logic.py        # Module 6 — reorder quantity + urgency logic
backend/
  main.py                 # Module 7 — FastAPI app, includes all routers
  auth.py                 # Module 8 — JWT auth + /auth/login endpoint
  database.py             # Module 8 — SQLite user store + seeding
  llm_insights.py         # Module 8 — /insights (Gemini + fallback)
  admin_routes.py         # Module 8 — /admin/summary (admin-only)
  data_routes.py          # Module 2 — /data/summary
  reorder_routes.py       # Module 6 — /reorder/{store_id}
frontend/
  index.html              # Login page
  manager.html            # Manager dashboard
  admin.html              # Admin dashboard
  style.css
notebooks/
  eda.ipynb               # Module 2 — EDA and seasonality analysis
docs/
  api_contract.md         # Module 8 — single source of truth for all endpoints
  data_schema.md          # Processed data column definitions
  architecture.md         # System diagram and design decisions
deployment/
  Dockerfile              # Container build
  render.yaml             # Render cloud deployment config
config.py                 # Shared constants (paths, column names, split dates)
requirements.txt
.env.example              # Environment variable template
```

---

## Setup (Local)

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd Cognizant-Hackathon
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

```bash
cp .env.example .env
# Edit .env and fill in SECRET_KEY and GEMINI_API_KEY
```

Generate a strong `SECRET_KEY`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Get a free Gemini API key at: https://aistudio.google.com/apikey

### 4. Add Kaggle data

Download the [Walmart Store Sales dataset](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) and place the CSVs in `data/raw/`:
- `train.csv`
- `test.csv`
- `features.csv`
- `stores.csv`

### 5. Run the feature pipeline (Module 1)

```bash
python src/features/build_dataset.py
```

### 6. Train models (Modules 3/4/5)

```bash
python src/models/lightgbm_model.py   # recommended — best MAPE
python src/models/sarima_model.py
python src/models/prophet_model.py
python src/models/baseline.py
```

### 7. Start the API server

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

---

## API Reference (Summary)

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/auth/login` | POST | None | Issue JWT token |
| `/data/summary` | GET | manager / admin | Dataset aggregation stats |
| `/forecast/{store_id}` | GET | manager / admin | Weekly sales forecast |
| `/models/comparison` | GET | manager / admin | MAPE/RMSE across all models |
| `/reorder/{store_id}` | GET | manager / admin | Reorder alerts by urgency |
| `/whatif` | POST | manager / admin | Markdown promotion scenario |
| `/insights` | POST | manager / admin | LLM inventory commentary |
| `/admin/summary` | GET | **admin only** | Privileged aggregation stats |

Full contract: [`docs/api_contract.md`](docs/api_contract.md)

**Default test accounts:**
- `admin` / `admin123` (role: admin)
- `manager1` / `manager123` (role: manager)

---

## Deployment (Render)

The app deploys automatically from the `main` branch via `deployment/render.yaml`.

Required environment variables (set in Render dashboard):
- `SECRET_KEY` — random hex string for JWT signing
- `GEMINI_API_KEY` — Google Gemini API key
- `ACCESS_TOKEN_EXPIRE_SECONDS` — default `28800` (8h)

```bash
# Build and run locally with Docker
docker build -f deployment/Dockerfile -t retail-forecast .
docker run -p 8000:8000 --env-file .env retail-forecast
```

---

## Module Ownership

| Module | Owns |
|---|---|
| 1 | `src/features/` — data cleaning, feature engineering |
| 2 | `notebooks/eda.ipynb`, `backend/data_routes.py` |
| 3 | `src/models/sarima_model.py` |
| 4 | `src/models/prophet_model.py` |
| 5 | `src/models/baseline.py`, `src/models/lightgbm_model.py`, `src/evaluate.py` |
| 6 | `src/reorder_logic.py`, `backend/reorder_routes.py` |
| 7 | `backend/main.py`, `frontend/`, `deployment/` |
| 8 | `backend/auth.py`, `backend/database.py`, `backend/llm_insights.py`, `backend/admin_routes.py`, `docs/api_contract.md`, `docs/architecture.md`, `deployment/` |

---

## Dataset

**Walmart Recruiting — Store Sales Forecasting** (Kaggle)

- Training range: `2010-02-05` → `2012-10-26`
- Granularity: Weekly, Friday-aligned
- Entity: Store + Department (no SKU/product level)
- 45 stores, 81 departments

> **Note:** Forecasts for dates beyond 2012-10-26 (including present-day) are deliberate extrapolations — the app simulates "today" against a historical model. MAPE/RMSE figures describe accuracy within the historical holdout only.
