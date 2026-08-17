# Data Schema

This is the contract between data cleaning, feature engineering, models,
and the API. If you need a column that isn't here, add it to this doc
and to `config.py`'s `PROCESSED_DTYPES` first, then tell the team —
don't silently add columns downstream.

## Source data (`data/raw/`)

Kaggle "Walmart Recruiting - Store Sales Forecasting" dataset, 4 files:

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | 421,570 | `Store, Dept, Date, Weekly_Sales, IsHoliday` | 45 stores x 81 depts, weekly, 2010-02-05 to 2012-10-26. No nulls. **1,285 rows have negative `Weekly_Sales`** (returns exceeded sales that week) — kept as-is, see below. |
| `test.csv` | 115,064 | `Store, Dept, Date, IsHoliday` | Kaggle's competition holdout, **no `Weekly_Sales`**. Not usable for our internal validation — only useful if we want Kaggle-style predictions with no way to score them locally. Date range 2012-11-02 to 2013-07-26. |
| `features.csv` | 8,190 | `Store, Date, Temperature, Fuel_Price, MarkDown1..5, CPI, Unemployment, IsHoliday` | Covers the full date range of both train and test. `MarkDown1..5` are 50-64% null (promos didn't start until partway through the dataset). `CPI`/`Unemployment` are ~7% null. |
| `stores.csv` | 45 | `Store, Type, Size` | Static lookup, one row per store. `Type` is A/B/C (22/17/6 stores). Saved with `\r`-only line endings — `pandas.read_csv` handles this fine, but plain `wc -l`/text tools may misreport line count. |

`IsHoliday` appears in both `train`/`test` and `features` — they agree
for the same (Store, Date), so keep one copy after merging.

## Processed schema (`data/processed/`)

Built by joining `train.csv` (or `test.csv`) with `stores.csv` on
`Store`, and with `features.csv` on `(Store, Date)`. Column names are
standardized to `snake_case`. This is exactly what's encoded in
`config.py`'s `PROCESSED_DTYPES` — import from there rather than
retyping column names.

| Column | Dtype | Nullable | Description |
|---|---|---|---|
| `store_id` | `int64` | no | Store number, 1-45. From `Store`. |
| `dept_id` | `int64` | no | Department number within a store. From `Dept`. Absent from `stores.csv`-only contexts. |
| `date` | `datetime64[ns]` | no | Week-ending date (weekly grain, typically Fridays). From `Date`. |
| `weekly_sales` | `float64` | no (target) | **Target column.** Sales for that store/dept/week. Present in `model_train.csv` and `model_holdout.csv`; absent from `kaggle_predict.csv`. Negative values are kept as-is (real return activity, not an error). |
| `is_holiday` | `bool` | no | Whether the week contains a major holiday. |
| `store_type` | `category` (`A`/`B`/`C`) | no | From `stores.csv` `Type`. |
| `store_size` | `int64` | no | From `stores.csv` `Size`. |
| `temperature` | `float64` | no | Avg regional temperature (°F) that week. |
| `fuel_price` | `float64` | no | Regional fuel price that week. |
| `markdown1`..`markdown5` | `float64` | **yes** | Promotional markdown amounts. `NaN` means "no promo/no data that week" — do **not** fill with 0, that would conflate "no promo" with "promo worth $0". |
| `cpi` | `float64` | **yes** (~7%) | Consumer Price Index for the region. |
| `unemployment` | `float64` | **yes** (~7%) | Regional unemployment rate. |

Natural key: `(store_id, dept_id, date)` uniquely identifies a row
(`config.ID_COLS`).

### Negative `weekly_sales`

0.3% of rows have negative sales. We keep them unmodified in the
processed data. If a model needs non-negative targets (e.g. a
log-transform), handle that transform locally in that model's code —
don't clip in the shared processed data, since other models may want
the raw signal.

### Processed files

| File | Rows | Has `weekly_sales`? | Definition |
|---|---|---|---|
| `model_train.csv` | `date <= 2012-08-17` | yes | Training set. |
| `model_holdout.csv` | `2012-08-24 <= date <= 2012-10-26` | yes | Internal validation set — last 10 weeks of `train.csv`, held out. **This is our only labeled evaluation set**; Kaggle's `test.csv` has no ground truth. |
| `kaggle_predict.csv` | `date >= 2012-11-02` | no | Processed version of Kaggle's `test.csv`, for generating submission-style predictions only (can't be scored locally). |

Split dates are defined once in `config.py` (`TRAIN_END_DATE`,
`TEST_START_DATE`, `TEST_END_DATE`) — import them, don't hardcode.

## Open questions / not yet decided

- Feature engineering (lags, rolling windows, holiday-proximity flags,
  etc.) is out of scope for this scaffold — lives in `src/features/`
  once we start building.
- Whether `store_type`/`store_size` get one-hot encoded or left as
  category/int is a model-specific choice, not part of this shared
  schema.
