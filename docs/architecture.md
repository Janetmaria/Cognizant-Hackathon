# Architecture — Retail Sales Forecasting Platform

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER (Module 1)                        │
│                                                                     │
│  data/raw/                     data/processed/                      │
│  ├── train.csv      ──────►    ├── model_train.csv                  │
│  ├── features.csv   (M1)       ├── model_holdout.csv                │
│  └── stores.csv                └── model_predict.csv                │
│                                                                     │
│  src/features/build_dataset.py                                      │
│  • Merge train + features + stores                                  │
│  • Fill MarkDown NaNs → 0, shift(1) lag features                   │
│  • Calendar features: week_of_year, month, day_of_week, holiday     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MODEL LAYER (Modules 3/4/5)                  │
│                                                                     │
│  src/models/                                                        │
│  ├── baseline.py          (M5) Seasonal-naive baseline              │
│  ├── sarima_model.py      (M3) SARIMA per store/dept                │
│  ├── prophet_model.py     (M4) Prophet with confidence intervals    │
│  └── lightgbm_model.py    (M5) LightGBM with lag/calendar features │
│                                                                     │
│  src/evaluate.py          (M5) Shared MAPE/RMSE evaluation         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       BACKEND API (FastAPI)                         │
│                   Base URL: http://localhost:8000                   │
│                                                                     │
│  backend/main.py          (M7) App entrypoint — includes all routers│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Module 8 Routes                                             │   │
│  │  POST  /auth/login        JWT issuance (no auth required)   │   │
│  │  POST  /insights          LLM commentary (Gemini 1.5 Flash) │   │
│  │  GET   /admin/summary     Admin-only aggregated stats       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Module 7 Routes                                             │   │
│  │  GET   /forecast/{store_id}   Serve trained model output    │   │
│  │  POST  /whatif                Markdown scenario simulation   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Module 6 Routes                                             │   │
│  │  GET   /reorder/{store_id}    Reorder alerts + urgency      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Module 2 Routes                                             │   │
│  │  GET   /data/summary          Dataset aggregation stats     │   │
│  │  GET   /models/comparison     MAPE/RMSE across all models   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Auth Layer (Module 8)                                              │
│  • All routes except /auth/login require Bearer JWT                 │
│  • /admin/summary requires role="admin" (403 otherwise)             │
│  • SQLite users DB seeded with admin + manager1 accounts            │
│                                                                     │
│  External service: Google Gemini 1.5 Flash (for /insights)         │
│  Falls back to templated string if LLM unavailable                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FRONTEND (Module 7)                            │
│                                                                     │
│  frontend/                                                          │
│  ├── index.html       Landing / login                               │
│  ├── manager.html     Manager dashboard (urgency list, forecasts)   │
│  ├── admin.html       Admin dashboard (admin/summary, full access)  │
│  └── style.css                                                      │
│                                                                     │
│  • Calls /auth/login → stores JWT in sessionStorage                 │
│  • Attaches Authorization: Bearer <token> on every API call         │
│  • Color-codes urgency: 🔴 red ≤1wk | 🟡 amber 1-2wk | 🟢 green >2wk│
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT (Modules 7/8)                        │
│                                                                     │
│  Platform: Render (render.yaml)                                     │
│  Container: Docker (deployment/Dockerfile)                          │
│  Runtime:   Python 3.11 + Uvicorn                                   │
│  Port:      8000                                                    │
│                                                                     │
│  Env vars (set in Render dashboard):                                │
│    SECRET_KEY, GEMINI_API_KEY, ACCESS_TOKEN_EXPIRE_SECONDS          │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow — Forecast Request

```
Client → POST /auth/login
       ← access_token (JWT, 8h)

Client → GET /forecast/{store_id}?dept_id=92&start_date=...&end_date=...
         Authorization: Bearer <token>
       ← { store_id, dept_id, model_used, predictions: [...] }

Client → POST /reorder/{store_id}
         Authorization: Bearer <token>
       ← { alerts: [...], summary: { total_depts_at_risk, total_capital_freed } }

Client → POST /insights          (sends /reorder response as body)
         Authorization: Bearer <token>
       ← { insight_text: "...", generated_by: "llm" | "fallback" }
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Auth enforced server-side (not UI-hidden) | Any direct API call bypassing the UI must still be blocked. JWT in every request header is the standard stateless REST pattern. |
| SQLite for user store | Zero-install, single-file DB — sufficient for a fixed set of seeded users at hackathon scale. No PostgreSQL dependency needed. |
| Gemini 1.5 Flash for /insights | Free tier, fast, 1M token context. Insights prompt is tiny (~100 tokens) — cost is negligible. |
| LLM fallback on failure | Availability > insight quality. The endpoint must never 500 because the rest of the dashboard depends on it loading. |
| No refresh token in v2 | 8h sessions are sufficient for the demo; eliminates a second auth surface area. |
| Render for deployment | Python-native, free tier, zero-config Docker deploy, auto-redeploy from Git. |
