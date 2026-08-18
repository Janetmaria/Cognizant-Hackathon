# API Contract — Retail Sales Forecasting Platform (v2)

**Dataset:** Walmart Recruiting — Store Sales Forecasting (Kaggle)

**Base URL (local):** `http://localhost:8000`

**Base URL (deployed):** `<TBD — will be updated after deployment; use local URL for now>`

This document is the single source of truth for all module communication. If your code does not match this doc, integration will break. Do not diverge silently.

**Changelog from v1:** clarified forecast horizon vs. training data range, defined a single DRI for `/whatif`, defined `is_holiday_override` default behavior, added a soft bound on forecast date ranges, clarified the "unrounded floats" rule against examples, defined `/insights` payload handling, and documented the token-refresh story. See inline "v2" notes below.

---

### 0. Dataset Reality Check

The Kaggle dataset contains no product-level (SKU) or daily data.

| Source File | Key Columns |
| --- | --- |
| `train.csv` | `Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday` |
| `features.csv` | `Store`, `Date`, `Temperature`, `Fuel_Price`, `MarkDown1`–`MarkDown5`, `CPI`, `Unemployment`, `IsHoliday` |
| `stores.csv` | `Store`, `Type` (A/B/C), `Size` |

* **Entity:** `store_id` + `dept_id` (Department level). Never use `product_id` or `sku`.
* **Granularity:** Weekly, Friday-aligned. Key name is `week_ending_date`.
* **Promotions:** Measured in dollar amounts across the 5 `MarkDown` columns, not generic percentages.
* **Inventory:** Source data has no inventory. Module 1 simulates a `stock_levels` table. This is a deliberate, documented assumption.
* **Alerts:** Issued strictly at store + department level.

**Training data range:** `2010-02-05` to `2012-10-26`. **[v2]** Any `/forecast` or `/whatif` request for dates beyond `2012-10-26` (including present-day dates like `2026-08-21` used in examples below) is an **extrapolation far outside the training window**. This is intentional — the app simulates "today" against a historical model — but it means:
- Confidence intervals (`lower_bound`/`upper_bound` on `prophet`) and MAPE/RMSE figures in `/models/comparison` describe accuracy *within the historical holdout period only*, not accuracy at this extrapolation distance.
- Module 7/5 should not treat long-horizon predictions as validated; no additional guardrail is required in v2, but do not present these numbers as if they carry the holdout-period error bars.

---

### 1. Global Rules

* **Identifiers:** `store_id` (string, e.g., `"4"`), `dept_id` (string, e.g., `"92"`).
* **Dates:** `week_ending_date` as `"YYYY-MM-DD"` string (Friday-aligned).
* **Headers:**
  * `Content-Type: application/json` on all requests with a body.
  * `Authorization: Bearer <jwt_token>` on all protected endpoints.
* **Numeric Fields:** Return unrounded floats. **[v2]** Note: every example in this doc is shown rounded to 1 decimal place for readability only — this is a documentation convention, not a spec. Do not round in actual responses.
* **Empty Collections:** Return empty arrays `[]`, never `null` or 404 for valid queries with 0 results.
* **Error Response Format:** Every error must return this exact JSON shape matching the HTTP status code:

```json
{
  "error": true,
  "message": "Human readable reason",
  "status_code": 401
}
```

---

### 2. Endpoints & Schemas

**Module 8 — POST /auth/login**

Auth required: No

Default seeded test accounts: `admin` / `admin123` (role: `"admin"`), `manager1` / `manager123` (role: `"manager"`).

```json
// Request
{
  "username": "manager1",
  "password": "manager123"
}

// Response 200 OK
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "role": "manager",
  "expires_in": 28800
}

// Response 401 Unauthorized
{ "error": true, "message": "Invalid username or password", "status_code": 401 }
```

**[v2] Token lifecycle:** Tokens are valid for `expires_in` seconds (8h) with **no refresh endpoint in v1/v2 scope**. On expiry, the client must call `/auth/login` again. This is acceptable for the current project scope (no long-lived sessions expected); revisit if the deployed app needs multi-day sessions.

---

**Module 2 — GET /data/summary**

> **Scope note:** Returns aggregated dataset stats accessible to both `manager` and `admin` roles. This is the standard summary for day-to-day use. The admin-only `/admin/summary` endpoint (Module 8) returns the same shape but is restricted to `admin` role only — its purpose is to provide a privileged, audit-level aggregation entry point without exposing it to managers.

Auth required: Yes (`manager` or `admin`)

```json
// Response 200 OK
{
  "total_weekly_sales": 4582310.5,
  "date_range": { "start": "2010-02-05", "end": "2012-10-26" },
  "store_count": 45,
  "dept_count": 81,
  "top_depts": [
    { "dept_id": "92", "total_sales": 88210.0 },
    { "dept_id": "95", "total_sales": 71040.0 }
  ],
  "top_stores": [
    { "store_id": "4", "type": "A", "size": 205863, "total_sales": 312400.0 }
  ],
  "holiday_lift_pct": 34.2,
  "markdown_lift_pct": 18.6
}
```

---

**Module 7 (serving 3/4/5) — GET /forecast/{store_id}**

Auth required: Yes (`manager` or `admin`)

Query Parameters: `dept_id` (required), `start_date` (required), `end_date` (required), `model` (optional: `"lightgbm"` | `"sarima"` | `"prophet"` | `"naive"`, default: `"lightgbm"`).

**[v2] Range limit:** `end_date - start_date` must not exceed **104 weeks (2 years)**. Requests exceeding this return `400 Bad Request` with `"message": "Date range exceeds maximum of 104 weeks"`. This bounds response size and avoids unbounded prediction arrays / timeouts.

```json
// Response 200 OK
{
  "store_id": "4",
  "dept_id": "92",
  "model_used": "lightgbm",
  "predictions": [
    {
      "week_ending_date": "2026-08-21",
      "predicted_weekly_sales": 22140.5,
      "actual_weekly_sales": 21830.0,
      "is_holiday": false,
      "lower_bound": null,
      "upper_bound": null
    }
  ]
}
// Note: actual_weekly_sales is null for future dates (i.e., beyond 2012-10-26 in this dataset).
// lower_bound/upper_bound populated only when model="prophet".
// See Section 0 for caveats on long-horizon extrapolation accuracy.

// Response 404 Not Found
{ "error": true, "message": "No data found for store_id/dept_id combination", "status_code": 404 }

// Response 400 Bad Request
{ "error": true, "message": "Date range exceeds maximum of 104 weeks", "status_code": 400 }
```

---

**Module 5 — GET /models/comparison**

Auth required: Yes (`manager` or `admin`)

**Scope:** Global — aggregated across **all stores and departments** in the holdout period. There is no `store_id` or `dept_id` filtering. This endpoint returns a single platform-wide model comparison table, not a per-store/per-dept breakdown.

```json
// Response 200 OK
{
  "holdout_period": { "start": "2012-05-01", "end": "2012-10-26" },
  "models": [
    { "name": "naive_seasonal", "mape": 24.8, "rmse": 3210.4 },
    { "name": "sarima", "mape": 16.2, "rmse": 2440.1 },
    { "name": "prophet", "mape": 15.7, "rmse": 2380.6 },
    { "name": "lightgbm", "mape": 9.3, "rmse": 1510.8 }
  ],
  "production_model": "lightgbm",
  "improvement_vs_baseline_pct": 62.5
}
```

---

**Module 6 — GET /reorder/{store_id}**

Auth required: Yes (`manager` or `admin`)

Query Parameters: `urgency` (optional: `"red"` | `"amber"` | `"green"`)

**Dept filter:** There is **no `dept_id` query parameter**. This endpoint always returns alerts for **all departments** in the given store. Filtering to a specific department is intentionally out of scope — the frontend (Modules 3/4) should filter client-side if needed. This keeps the backend logic simple and the response self-contained.

```json
// Response 200 OK
{
  "store_id": "4",
  "generated_at": "2026-08-17",
  "alerts": [
    {
      "dept_id": "92",
      "current_stock_units": 320,
      "predicted_weekly_demand_units": 410,
      "weeks_to_stockout": 0.8,
      "urgency": "red",
      "recommended_reorder_units": 500,
      "capital_freed_estimate": 6200.0
    }
  ],
  "summary": {
    "total_depts_at_risk": 6,
    "total_capital_freed_estimate": 41200.0
  },
  "data_note": "current_stock_units is simulated; no inventory data exists in the source dataset."
}
```

Urgency thresholds:

* `red`: `weeks_to_stockout` ≤ 1
* `amber`: 1 < `weeks_to_stockout` ≤ 2
* `green`: `weeks_to_stockout` > 2

---

**Module 6 (logic) / Module 7 (route) — POST /whatif**

Auth required: Yes (`manager` or `admin`)

**[v2] Ownership:** Single DRI for this endpoint end-to-end is **Module 6**. Module 7 owns only the route registration/wiring (auth check, request validation, calling into Module 6's logic) and must not modify markdown-adjustment math independently. Any change to the adjusted-prediction calculation goes through Module 6.

**[v2] `is_holiday_override` default:** If omitted from the request, the endpoint uses the dataset's actual `IsHoliday` flag for each week in range (i.e., no override — behaves as if `is_holiday_override` were not sent at all, not as `false`). Explicitly passing `false` forces non-holiday treatment even on a real holiday week; explicitly passing `true` forces holiday treatment even on a non-holiday week.

**Date range cap:** This endpoint has **no 104-week range limit** (unlike `/forecast`). The request date range is unbounded. This is deliberate — what-if scenarios are exploratory and may span longer planning horizons. Be aware that very long ranges will produce proportionally larger response payloads; no server-side timeout protection is in scope for v2.

```json
// Request
{
  "store_id": "4",
  "dept_id": "92",
  "start_date": "2026-08-21",
  "end_date": "2026-09-11",
  "markdown_amount": 5000.0,
  "is_holiday_override": false
}

// Response 200 OK
{
  "store_id": "4",
  "dept_id": "92",
  "baseline_predictions": [
    { "week_ending_date": "2026-08-21", "predicted_weekly_sales": 22140.5 }
  ],
  "adjusted_predictions": [
    { "week_ending_date": "2026-08-21", "predicted_weekly_sales": 26980.2 }
  ],
  "projected_lift_pct": 21.9
}

// Response 400 Bad Request
{ "error": true, "message": "markdown_amount must be zero or positive", "status_code": 400 }
```

---

**Module 8 — POST /insights**

Auth required: Yes (`manager` or `admin`)

**[v2] Payload handling:** This endpoint accepts the exact shape shown below (mirroring Module 6's `/reorder` response). Any additional/unrecognized fields in the request body **are ignored, not rejected** — Module 8 must not error on extra fields, so Module 6 can add fields to `/reorder` later without breaking `/insights`. Required fields (`store_id`, `alerts`, `summary`) missing from the payload return `400 Bad Request` with the standard error shape.

```json
// Request (Accepts Module 6 /reorder payload format; extra fields ignored)
{
  "store_id": "4",
  "alerts": [
    { "dept_id": "92", "urgency": "red", "weeks_to_stockout": 0.8 }
  ],
  "summary": { "total_depts_at_risk": 6, "total_capital_freed_estimate": 41200.0 }
}

// Response 200 OK (LLM Success)
{
  "insight_text": "Store 4 has 6 departments at high risk of stockout this week, led by Dept 92 which is under a week from running out. Reordering now would help prevent an estimated $41,200 in lost sales.",
  "generated_by": "llm"
}

// Response 200 OK (Fallback on LLM timeout/failure)
{
  "insight_text": "6 departments need urgent reorder across store 4.",
  "generated_by": "fallback"
}

// Response 400 Bad Request
{ "error": true, "message": "Missing required field: store_id", "status_code": 400 }
```

---

**Module 8 — GET /admin/summary**

Auth required: Yes (`admin` role strictly enforced server-side)

```json
// Response 200 OK (Aggregated across all 45 stores)
{
  "total_weekly_sales": 4582310.5,
  "date_range": { "start": "2010-02-05", "end": "2012-10-26" },
  "store_count": 45,
  "dept_count": 81,
  "top_depts": [
    { "dept_id": "92", "total_sales": 88210.0 }
  ],
  "top_stores": [
    { "store_id": "4", "type": "A", "size": 205863, "total_sales": 312400.0 }
  ],
  "holiday_lift_pct": 34.2,
  "markdown_lift_pct": 18.6
}

// Response 403 Forbidden (If called by manager role)
{ "error": true, "message": "Admin access required", "status_code": 403 }
```

---

### 3. Shared Baseline Assumptions (Module 1 Config)

These fixed values must be stored in `src/config.py` by Module 1:

| Assumption | Owner | Value / Rule |
| --- | --- | --- |
| `current_stock_units` | Module 1 | Simulated baseline units per department |
| `avg_unit_price` | Module 1 | Estimated price per department (Sales ÷ Price = Demand Units) |
| `lead_time_weeks` | Module 1 / 6 | Fixed constant: 1.0 week across departments |

---

### 4. Ownership Matrix

| Endpoint | Method | Owner (DRI) | Dependencies |
| --- | --- | --- | --- |
| `/auth/login` | POST | Module 8 | None (seeded `database.py`) |
| `/data/summary` | GET | Module 2 | `train.csv`, `stores.csv` |
| `/forecast/{store_id}` | GET | Module 7 (Serving 3/4/5) | Processed features + trained models |
| `/models/comparison` | GET | Module 5 | Chronological holdout evaluation |
| `/reorder/{store_id}` | GET | Module 6 | Forecast output + simulated stock table |
| `/whatif` | POST | **Module 6** (logic + calculation; Module 7 owns route wiring only — see v2 note above) | Feature pipeline with markdown adjustments |
| `/insights` | POST | Module 8 | Output of `/reorder` |
| `/admin/summary` | GET | Module 8 | Aggregated dataset + Admin RBAC check |
