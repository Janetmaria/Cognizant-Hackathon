# Module 1 Summary: Data Cleaning + Feature Engineering

Short-form write-up of what Module 1 built. For the full column-by-column
contract see `data_schema.md`; for the "why" behind cleaning decisions and
what the data looks like see `preprocessing_summary.md`. This doc pulls
the pieces most relevant to onboarding into one place.

Built by `src/features/build_dataset.py`. Verified for leakage by
`src/features/leakage_check.py` (see "Leakage check" below).

## 1. Dataset overview

Source: Kaggle "Walmart Recruiting - Store Sales Forecasting", 4 raw
files in `data/raw/`.

| File | Rows | Columns | Join key(s) | What it contains |
|---|---|---|---|---|
| `train.csv` | 421,570 | `Store, Dept, Date, Weekly_Sales, IsHoliday` | `(Store, Date)` -> `features.csv`; `Store` -> `stores.csv` | Weekly sales per store/department, 45 stores x 81 depts, 2010-02-05 to 2012-10-26. No nulls. Labeled (has `Weekly_Sales`) - this is the only file we can train/validate against. |
| `test.csv` | 115,064 | `Store, Dept, Date, IsHoliday` | Same as `train.csv` | Kaggle's competition holdout, 2012-11-02 to 2013-07-26. **No `Weekly_Sales`** - can't be scored locally, only usable for Kaggle-style submission predictions. Not touched by this module's output. |
| `features.csv` | 8,190 | `Store, Date, Temperature, Fuel_Price, MarkDown1..5, CPI, Unemployment, IsHoliday` | `(Store, Date)` | Store-level weekly context covering the full date range of both `train.csv` and `test.csv`. `MarkDown1..5` are 50-64% null (promo program didn't exist for the first ~21 months). `CPI`/`Unemployment` are ~7% null. `IsHoliday` duplicates `train`/`test`'s column and agrees on every overlapping row, so we drop this copy on merge. |
| `stores.csv` | 45 | `Store, Type, Size` | `Store` | Static per-store lookup: `Type` (A/B/C, 22/17/6 stores) and `Size` (square footage). One row per store. |

Processed output joins `train.csv` with `stores.csv` on `Store`, and with
`features.csv` on `(Store, Date)` - the natural key of the resulting
table is `(store_id, dept_id, date)` (`config.ID_COLS`). Column names are
standardized to `snake_case` (e.g. `Store` -> `store_id`,
`Weekly_Sales` -> `weekly_sales`).

## 2. Key cleaning decisions

- **Negative `weekly_sales`: kept, not dropped or clipped.** 0.3% of rows
  (1,199 in `model_train.csv`) have negative sales - real weeks where
  returns exceeded purchases, not a data error. Dropping or clipping
  would throw away real signal and bias the target's distribution. Instead
  we add a boolean flag column, `is_negative_sales`, so downstream models
  can filter or weight these rows themselves without the shared dataset
  losing information for models that want the raw signal.

- **`MarkDown1`-`MarkDown5` NaN filled with `0.0`.** In the raw data
  `MarkDown1` has zero non-null values before 2011-11-11 - the
  promotional-markdown program simply didn't exist yet for roughly the
  first 21 months of the dataset. Filling with 0.0 treats "no markdown
  data yet" the same as "markdown program active, $0 discount," which
  does lose the ability to distinguish those two cases. This was an
  explicit reversal of `data_schema.md`'s original "do not fill with 0"
  guidance, done per this task's requirement - flagged in
  `data_schema.md` and `preprocessing_summary.md` as an open item for the
  team to confirm before models lock in on it. If a model needs to
  separate "no program" from "program, no discount," it can re-derive
  that from the known 2011-11-11 program start date.

- **`CPI`/`Unemployment` forward-filled per store**, not mean-imputed.
  These are regional, slow-moving economic indicators, so a given store's
  own most recent known value is a far better estimate than a cross-store
  average for the same week. In practice this is currently a no-op for
  `model_train.csv`/`model_holdout.csv`: all of `features.csv`'s ~7%
  missing values fall in May-July 2013, which is entirely inside
  `test.csv`'s date range (`train.csv` ends 2012-10-26). Both processed
  outputs have zero nulls in `cpi`/`unemployment` regardless of
  imputation method today - the ffill logic will start to matter once a
  processed version of `test.csv` (`kaggle_predict.csv`) is built.

## 3. Features engineered

- **Calendar**: `day_of_week`, `month`, `week_of_year`, derived directly
  from `date`. Note `day_of_week` is constant (always Friday, since every
  row is a Friday-dated weekly snapshot) and therefore has zero variance
  - kept only for schema completeness, safe for modelers to drop.

- **Lag features**: `weekly_sales_lag_1` and `weekly_sales_lag_4` - the
  same `(store_id, dept_id)`'s `weekly_sales` from 1 and 4 rows back,
  via `groupby(["store_id","dept_id"])["weekly_sales"].shift(lag)`.

- **Rolling features**: `weekly_sales_roll_mean_4` and
  `weekly_sales_roll_std_4` - mean/std of `weekly_sales` over the
  trailing 4 available weeks per `(store_id, dept_id)`.

- **Why `shift(1)` before rolling matters:** `.rolling(4).mean()` computed
  directly on `weekly_sales` would include the *current* row in its own
  window - i.e. the feature for week N would partly be made of the value
  we're trying to predict for week N. That's leakage: the model would
  learn to lean on a feature that won't exist at real prediction time.
  To avoid this, the rolling stats here are computed on
  `weekly_sales.groupby([...]).shift(1)` first, so the window for row N
  only ever spans weeks `N-4 .. N-1`. The single-value lag features avoid
  the same trap by construction - `shift(1)`/`shift(4)` never returns the
  current row's own value.

- **Leakage check (this update):** `src/features/leakage_check.py` was
  added to make this verifiable rather than just asserted. It (1) reports
  each engineered feature's correlation against the current week's
  `weekly_sales` and flags anything at or above 0.999 as suspiciously
  close to the target itself, and (2) directly recomputes
  `weekly_sales_lag_1` and the rolling mean/std from raw `weekly_sales`
  on a sample store/dept group and confirms they match the stored
  columns, while every row's `weekly_sales_lag_1`/rolling window never
  includes that same row's own `weekly_sales`. Run it with
  `python -m src.features.leakage_check`.

  Current results on `model_train.csv`: `weekly_sales_lag_1` r=0.947,
  `weekly_sales_lag_4` r=0.930, `weekly_sales_roll_mean_4` r=0.956,
  `weekly_sales_roll_std_4` r=0.418 - all well below the 0.999
  suspicious-leakage threshold, and consistent with real week-to-week
  sales autocorrelation rather than the feature echoing the target. The
  shift(1) verification (sampled on store 1 / dept 1) confirmed
  `weekly_sales_lag_1` exactly matches the prior row's `weekly_sales`
  and never its own row's value, and the rolling mean/std exactly match
  values recomputed from the raw `weekly_sales` series windowed over
  rows strictly before the current one. No issues flagged.

- **Known simplification**: lag/rolling use "N rows back" per
  `(store_id, dept_id)`, not strictly "N calendar weeks back." ~1.3% of
  store-dept week-to-week transitions have a gap (a missing week) rather
  than exactly 7 days, so a small fraction of lag values are technically
  "last observed week" rather than "exactly 7/28 days ago." Not corrected
  in this pass - would require reindexing every group to a full weekly
  calendar.

## 4. Final processed dataset

Two labeled outputs are written to `data/processed/`, split by date
(`config.TRAIN_END_DATE` / `TEST_START_DATE` / `TEST_END_DATE`); lag and
rolling features are computed on the full continuous series per
store+dept *before* the split, so `model_holdout.csv`'s first rows
correctly see `model_train.csv`'s tail history instead of starting cold:

| File | Rows | Date range |
|---|---|---|
| `model_train.csv` | 391,919 | <= 2012-08-17 |
| `model_holdout.csv` | 29,651 | 2012-08-24 to 2012-10-26 |

(`kaggle_predict.csv`, the processed version of `test.csv`, is out of
scope for this module - it needs `train.csv`'s tail stitched onto it
before lag/rolling features would make sense for it.)

Both files share the same 24 columns:

`store_id, dept_id, date, weekly_sales, is_holiday, temperature,
fuel_price, markdown1, markdown2, markdown3, markdown4, markdown5, cpi,
unemployment, store_type, store_size, is_negative_sales, day_of_week,
month, week_of_year, weekly_sales_lag_1, weekly_sales_lag_4,
weekly_sales_roll_mean_4, weekly_sales_roll_std_4`

(24 columns total - dtypes for each are the source of truth in
`config.PROCESSED_DTYPES`; nullable columns, all four lag/rolling
features, are listed in `config.NULLABLE_COLS`.)
