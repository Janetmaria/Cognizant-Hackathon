# Architecture

This reflects the system as actually built on `main`, not the original
8-module plan (several modules diverged from their planned scope —
notably auth/LLM/deployment, which were rebuilt directly on `main`
rather than merged from the `module-8` branch; see `README.md`'s module
table for the original plan and the gap against it).

## System diagram

```mermaid
flowchart TD
    subgraph Data["Data Pipeline (src/features/)"]
        raw["data/raw/*.csv\ntrain, features, stores, test"]
        build["build_dataset.py\nmerge + clean + engineer features"]
        leak["leakage_check.py\nverify no target leakage"]
        train_csv["data/processed/model_train.csv\n(date <= 2012-08-17)"]
        holdout_csv["data/processed/model_holdout.csv\n(2012-08-24 .. 2012-10-26)"]
        raw --> build --> train_csv & holdout_csv
        train_csv -.verified by.-> leak
    end

    subgraph Models["Models (src/models/, src/evaluate.py)"]
        baseline["baseline.py\nSeasonal-Naive"]
        sarima["sarima_model.py\nSARIMA per (store,dept)"]
        prophet["prophet_model.py\nProphet per (store,dept)"]
        lgbm["lightgbm_model.py\nLightGBM regressor"]
        evaluate["evaluate.py\nshared MAE/RMSE/MAPE/R2"]
        preds["data/processed/lightgbm_predictions.csv\nholdout rows + lgbm_pred + baseline_pred"]
        train_csv & holdout_csv --> baseline & sarima & prophet & lgbm
        baseline & sarima & prophet & lgbm -.scored by.-> evaluate
        lgbm --> preds
    end

    subgraph Backend["Backend (FastAPI, backend/)"]
        authpy["auth.py\nJWT (HS256) + bcrypt\nSQLAlchemy users -> data/app.db"]
        mainpy["main.py\napp wiring, global error handler,\n/, /forecast, /models/comparison, /admin/summary"]
        reorderroutes["reorder_routes.py\n/reorder/{store_id}, /whatif"]
        reorderlogic["src/reorder_logic.py\nstockout math, promo simulation"]
        dataroutes["data_routes.py\n/data/summary"]
        llm["llm_insights.py\n/insights\nGroq -> OpenAI -> generic -> template fallback"]
        db[("data/app.db\nadmin / manager / manager1")]

        preds --> mainpy & reorderlogic
        mainpy --> reorderroutes & dataroutes & llm & authpy
        reorderroutes --> reorderlogic
        authpy --> db
    end

    subgraph Frontend["Frontend (static HTML/JS, frontend/)"]
        index["index.html\nlogin -> stores JWT in localStorage"]
        admin["admin.html\nadmin view: /admin/summary, /models/comparison"]
        manager["manager.html\nmanager view: /forecast, /reorder, /whatif, /insights"]
        index -->|role: admin| admin
        index -->|role: manager| manager
    end

    admin -- "Bearer JWT" --> mainpy
    manager -- "Bearer JWT" --> mainpy

    subgraph Deploy["Deployment (deployment/)"]
        docker["Dockerfile\npython:3.11-slim, uvicorn"]
        render["render.yaml\nRender.com web service"]
    end
    mainpy -.packaged by.-> docker --> render
```

## Notes on what diverged from the original plan

- **Auth/LLM/admin-summary were rebuilt on `main`, not merged from `module-8`.**
  `backend/auth.py` (real PyJWT + bcrypt + SQLAlchemy), `backend/database.py`,
  and `backend/llm_insights.py` (Groq/OpenAI/generic + template fallback) are
  all independent, more complete implementations than what exists on the
  unmerged `module-8` branch. Only `deployment/Dockerfile` and
  `deployment/render.yaml` were cherry-picked from that branch (env vars
  rewritten to match `main`'s actual auth/LLM config).
- **`/data/summary` (`backend/data_routes.py`) previously had its own stub
  auth check** instead of using `backend/auth.py`'s real JWT validation like
  every other endpoint — fixed to use `auth.require_manager_or_admin`.
- **`data/processed/lightgbm_predictions.csv`** is the single file every
  forecast-dependent endpoint (`/forecast`, `/reorder`, `/whatif`) reads
  from. It's produced by `src/models/lightgbm_model.py`'s standalone
  `run_lightgbm_pipeline()` — a self-contained LightGBM regressor, not
  chained through SARIMA/Prophet.
- **Error responses** are normalized by a single global exception handler
  in `backend/main.py` (`http_exception_handler` / `validation_exception_handler`)
  so every error — including FastAPI's default 422 validation errors —
  matches `docs/api_contract.md`'s `{error, message, status_code}` shape.
- **`/models/comparison`** should be read from real, freshly-computed
  evaluation results (baseline/SARIMA/Prophet/LightGBM run against the
  same holdout set) rather than a hardcoded leaderboard.
