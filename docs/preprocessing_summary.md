# Preprocessing & Feature Engineering Summary (Module 1)

For the EDA/presentation team (Module 2) and anyone else who wants the
short version of what happened between `data/raw/` and
`data/processed/`. Full column-by-column contract is in
`data_schema.md`; this doc is the "why" and "what it looks like."

Built by `src/features/build_dataset.py`. Re-run it any time raw data
changes — outputs are fully deterministic given the raw CSVs.

## What was built

- `data/processed/model_train.csv` — 391,919 rows, dates <= 2012-08-17
- `data/processed/model_holdout.csv` — 29,651 rows, 2012-08-24 to 2012-10-26

`kaggle_predict.csv` (from the raw, unlabeled `test.csv`) was **not**
built in this pass — it needs train history stitched onto it before lag
features make sense, which is a separate task.

## Cleaning decisions

- **Negative `weekly_sales` kept, not dropped.** 0.31% of rows (1,199 in
  `model_train.csv`) have negative sales — real weeks where returns
  exceeded sales. A new `is_negative_sales` boolean flags them so models
  can exclude/weight them without the shared data losing information.
- **MarkDown1-5 NaN filled with 0.0.** In the raw data, MarkDown1 has no
  non-null values before 2011-11-11 — the promo program simply didn't
  exist yet, so "NaN" and "$0 markdown" were already almost
  indistinguishable in practice. Note: this reverses what
  `data_schema.md` originally said ("don't fill with 0") — flagged
  there and to the team; worth a second look if any model wants to
  distinguish "no program yet" from "program active, no markdown."
- **CPI/Unemployment forward-filled per store**, not mean-imputed —
  these are regional, slow-moving indicators, so a store's own last
  known value is a much better estimate than a cross-store average.
  Turns out to be moot for this output: all of the ~7% missing values
  in `features.csv` fall in May-July 2013, which is entirely inside
  `test.csv`'s date range. `model_train.csv`/`model_holdout.csv` have
  **zero** nulls in `cpi`/`unemployment` regardless of method. This will
  matter once `kaggle_predict.csv` gets built.

## Features added

- **Calendar**: `day_of_week`, `month`, `week_of_year`. Note:
  `day_of_week` is **constant** (always Friday) across the entire
  dataset — every row is a Friday-dated weekly snapshot. Zero variance,
  so it has no predictive value; kept only for schema completeness.
- **Lag features**: `weekly_sales_lag_1`, `weekly_sales_lag_4` — previous
  observed week's sales for the same store+dept, 1 and 4 weeks back.
- **Rolling features**: `weekly_sales_roll_mean_4`, `weekly_sales_roll_std_4`
  — mean/std of the trailing 4 available weeks. All lag/rolling stats are
  computed on `shift(1)` of the sales series first, so a row's own value
  is never included in its own rolling window — otherwise you'd be
  handing the model the answer.
- Lag/rolling features were computed on the **full continuous
  2010-02-05..2012-10-26 series per store+dept before splitting**, so
  `model_holdout.csv`'s first rows correctly see `train.csv`'s tail
  history instead of starting with NaNs. Verified: holdout's first row
  for store 1 / dept 1 (2012-08-24) has `weekly_sales_lag_1` = train's
  last row's `weekly_sales` (2012-08-17), exactly as expected.
- Caveat: lag/rolling use `groupby(store, dept).shift(...)`, i.e. "N rows
  back," not strictly "N calendar weeks back." ~1.3% of store-dept
  week-to-week transitions in the raw data have a gap (missing week)
  rather than exactly 7 days, so a small fraction of lag values are
  technically "last observed week" rather than "exactly 7 days ago."

## What the data shows

- 45 stores (Type A: 22, B: 17, C: 6), 81 departments, weekly cadence,
  2010-02-05 to 2012-10-26 in `train.csv`.
- `weekly_sales`: mean $16,018, median $7,629, std $22,778, range
  -$4,989 to $693,099 — heavily right-skewed (a log transform is
  probably worth trying for models sensitive to scale).
- Holiday weeks average higher sales than non-holiday weeks ($17,118 vs
  $15,937, +7.4%) — consistent with the dataset's known Black
  Friday/Christmas/Thanksgiving/Super Bowl holiday flags.
- Store type strongly predicts scale: Type A stores average $20,142/week,
  B $12,270, C $9,525 — `store_type`/`store_size` are likely
  high-value features.
- MarkDown promos only exist from 2011-11-11 onward — any model using
  markdown features implicitly has less signal for the first ~21 months
  of the dataset.

## Open item for the team

Flagging explicitly since it reverses a documented decision: **MarkDown
NaN -> 0 fill**. If any model wants "no promo program yet" to be
distinguishable from "program active, $0 markdown," it'll need to
either re-derive that from the raw `features.csv` promo start date
(2011-11-11) or we add a separate `markdown_program_active` flag to the
shared schema. Raise in team channel before modeling locks this in.
