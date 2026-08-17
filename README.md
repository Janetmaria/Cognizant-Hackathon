# Retail Sales Forecasting Platform

Hackathon project forecasting Walmart store sales using the Kaggle
Walmart Store Sales dataset.

## Project structure

```
data/
  raw/                    # Original Kaggle CSVs — never edit these in place
  processed/               # Cleaned/merged data that everyone builds on
src/
  features/                 # Module 1 — feature engineering
  models/
    baseline.py              # Module 5
    sarima_model.py           # Module 3
    prophet_model.py           # Module 4
    lightgbm_model.py           # Module 5
  evaluate.py                   # Module 5 — shared metrics for all models
  reorder_logic.py                # Module 6 — reorder/restock logic
backend/
  main.py                          # Module 7 — forecast-serving + business logic
  auth.py                           # Module 8 — JWT, password hashing
  database.py                        # Module 8
  reorder_routes.py                   # Module 6 — /reorder/{store_id}
  data_routes.py                       # Module 2 — /data/summary
  llm_insights.py                       # Module 8 — narrow: numbers in, commentary out
frontend/
  index.html, style.css, admin.html, manager.html
notebooks/
  eda.ipynb                              # Module 2
docs/
  api_contract.md                         # Module 8 — write FIRST, others build routes against it
  data_schema.md
  architecture.md
  README.md                                # docs index
deployment/                                 # Module 7/8 — deploy configs
config.py                                    # Shared, read-only after Day 1
requirements.txt
```

### Module ownership

| Module | Owns |
|---|---|
| 1 | `src/features/` |
| 2 | `notebooks/eda.ipynb`, `backend/data_routes.py` (`/data/summary`) |
| 3 | `src/models/sarima_model.py` |
| 4 | `src/models/prophet_model.py` |
| 5 | `src/models/baseline.py`, `src/models/lightgbm_model.py`, `src/evaluate.py` |
| 6 | `src/reorder_logic.py`, `backend/reorder_routes.py` (`/reorder/{store_id}`) |
| 7 | `backend/main.py` (`/forecast/{store_id}` + app wiring), `deployment/` |
| 8 | `docs/api_contract.md`, `backend/auth.py`, `backend/database.py`, `backend/llm_insights.py`, `deployment/` |

Everyone works inside their own file(s) above so we can merge without
stepping on each other. Shared files (`config.py`, `docs/data_schema.md`)
are edit-with-announcement only. **Module 8 writes `docs/api_contract.md`
first** — Modules 2, 6, 7 build their endpoints to match it, not the
other way around.

## Setup

1. **Create a virtual environment**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add the data**

   Download the Walmart Store Sales dataset from Kaggle and place the
   unzipped CSV file(s) in `data/raw/`. Do not commit raw data files —
   they're git-ignored.

4. **Check the schema**

   Read `docs/data_schema.md` before writing any feature or model code.
   It defines the processed column names, dtypes, and target column
   that features/models/API all agree on. If you need a new column,
   update that doc first and flag it to the team.

## Workflow

- Raw data goes in `data/raw/` and is never modified in place.
- Cleaning/merging scripts write standardized output to `data/processed/`
  using the schema in `docs/data_schema.md`.
- Feature code lives in `src/features/`, model code under `src/models/`.
- `config.py` is the single source of truth for file paths, column
  names, the target column, and the train/test split dates — import
  from it rather than hardcoding strings.
- API endpoints must match `docs/api_contract.md` (field naming, date
  format, response envelope) once Module 8 fills it in.

## Team

Team of 8. Coordinate schema/config changes in the team channel before
merging — features, models, and the API all depend on the same
processed data format.
