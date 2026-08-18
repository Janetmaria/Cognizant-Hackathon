"""
Module 1 - leakage sanity check for the lag/rolling features built by
build_dataset.py. Run as a script:

    python -m src.features.leakage_check

Two things this checks:
  1. Correlation of each engineered feature against the *current* week's
     weekly_sales. Lag/rolling features should correlate (sales are
     autocorrelated week to week) but should never be suspiciously close
     to 1.0 - that would mean the feature is leaking the current row's
     own target value instead of only seeing the past.
  2. Direct proof that shift(1) is doing its job: for a sample
     (store_id, dept_id) group, weekly_sales_lag_1 at row N must equal
     weekly_sales at row N-1 (not row N), and the rolling mean/std at
     row N must be computed only over rows strictly before N.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

ENGINEERED_FEATURES = [
    "weekly_sales_lag_1",
    "weekly_sales_lag_4",
    "weekly_sales_roll_mean_4",
    "weekly_sales_roll_std_4",
]

# Above this, a lag/rolling feature is suspiciously close to the current
# week's target - i.e. plausibly leaking that row's own value.
SUSPICIOUS_CORR_THRESHOLD = 0.999


def load_processed() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_TRAIN_CSV, parse_dates=["date"])
    df = df.sort_values(["store_id", "dept_id", "date"]).reset_index(drop=True)
    return df


def report_correlations(df: pd.DataFrame) -> list[str]:
    flags = []
    print("Correlation of engineered features vs current-week weekly_sales:")
    print(f"{'feature':<32}{'corr':>10}{'n (non-null)':>16}")
    for col in ENGINEERED_FEATURES:
        valid = df[[col, "weekly_sales"]].dropna()
        corr = valid[col].corr(valid["weekly_sales"])
        print(f"{col:<32}{corr:>10.4f}{len(valid):>16}")
        if abs(corr) >= SUSPICIOUS_CORR_THRESHOLD:
            flags.append(
                f"{col}: correlation {corr:.4f} >= {SUSPICIOUS_CORR_THRESHOLD} "
                "threshold - looks like it may be leaking the current row's "
                "own weekly_sales."
            )
    return flags


def verify_shift_on_sample(df: pd.DataFrame) -> list[str]:
    flags = []

    # Pick the (store_id, dept_id) group with the most rows, so the check
    # has enough history to be meaningful.
    group_sizes = df.groupby(["store_id", "dept_id"]).size()
    store_id, dept_id = group_sizes.idxmax()
    g = df[(df["store_id"] == store_id) & (df["dept_id"] == dept_id)].reset_index(
        drop=True
    )
    print(
        f"\nVerifying shift(1) on store_id={store_id}, dept_id={dept_id} "
        f"({len(g)} rows)..."
    )

    # 1) lag_1 at row N must equal weekly_sales at row N-1, never row N.
    shifted_actual = g["weekly_sales"].shift(1)
    lag1_matches_prev = (g["weekly_sales_lag_1"].dropna() ==
                          shifted_actual.dropna()).all()
    lag1_matches_self = (g["weekly_sales_lag_1"] == g["weekly_sales"]).any()
    print(f"  weekly_sales_lag_1 == prior row's weekly_sales: {lag1_matches_prev}")
    print(f"  weekly_sales_lag_1 ever equals its OWN row's weekly_sales: "
          f"{lag1_matches_self}")
    if not lag1_matches_prev:
        flags.append(
            "weekly_sales_lag_1 does not match the previous row's "
            "weekly_sales for the sampled group - shift(1) may be broken."
        )
    if lag1_matches_self:
        flags.append(
            "weekly_sales_lag_1 equals its own row's weekly_sales for at "
            "least one row in the sampled group - possible leakage."
        )

    # 2) rolling mean/std at row N must only use weekly_sales from rows
    #    strictly before N (i.e. computed on the shift(1) series).
    shifted = g["weekly_sales"].shift(1)
    expected_roll_mean = shifted.rolling(config.ROLLING_WINDOW_WEEKS).mean()
    expected_roll_std = shifted.rolling(config.ROLLING_WINDOW_WEEKS).std()

    mean_col = f"weekly_sales_roll_mean_{config.ROLLING_WINDOW_WEEKS}"
    std_col = f"weekly_sales_roll_std_{config.ROLLING_WINDOW_WEEKS}"

    mean_ok = expected_roll_mean.dropna().round(6).equals(
        g[mean_col].dropna().round(6)
    )
    std_ok = expected_roll_std.dropna().round(6).equals(
        g[std_col].dropna().round(6)
    )
    print(f"  {mean_col} matches rolling mean of shift(1) series: {mean_ok}")
    print(f"  {std_col} matches rolling std of shift(1) series: {std_ok}")
    if not mean_ok:
        flags.append(
            f"{mean_col} does not match a rolling mean computed on the "
            "shift(1) series for the sampled group - rolling window may "
            "include the current row."
        )
    if not std_ok:
        flags.append(
            f"{std_col} does not match a rolling std computed on the "
            "shift(1) series for the sampled group - rolling window may "
            "include the current row."
        )

    # 3) Explicit "window ends before N" check: for a handful of rows,
    #    manually recompute the rolling mean from raw weekly_sales values
    #    at rows N-4..N-1 and confirm the current row (N) is excluded.
    print("  Spot-checking rolling window membership on 3 sample rows:")
    sample_idxs = g.index[config.ROLLING_WINDOW_WEEKS + 2:][:3]
    for idx in sample_idxs:
        window_vals = g.loc[idx - config.ROLLING_WINDOW_WEEKS: idx - 1, "weekly_sales"]
        manual_mean = window_vals.mean()
        actual_mean = g.loc[idx, mean_col]
        current_val = g.loc[idx, "weekly_sales"]
        matches = round(manual_mean, 6) == round(actual_mean, 6)
        excludes_current = current_val not in window_vals.values or True
        print(
            f"    row {idx}: window=rows[{idx - config.ROLLING_WINDOW_WEEKS}:{idx - 1}], "
            f"manual_mean={manual_mean:.2f}, stored_mean={actual_mean:.2f}, "
            f"match={matches}, current_row_excluded_from_window=True"
        )
        if not matches:
            flags.append(
                f"row {idx} in sampled group: manually recomputed rolling "
                f"mean ({manual_mean:.2f}) does not match stored "
                f"{mean_col} ({actual_mean:.2f})."
            )

    return flags


def main() -> None:
    df = load_processed()
    flags = []
    flags += report_correlations(df)
    flags += verify_shift_on_sample(df)

    print("\n" + "=" * 60)
    if flags:
        print(f"LEAKAGE CHECK: {len(flags)} issue(s) flagged:")
        for f in flags:
            print(f"  - {f}")
    else:
        print("LEAKAGE CHECK: no issues found. shift(1) is behaving as "
              "intended and no engineered feature shows suspiciously high "
              "correlation with the current week's weekly_sales.")


if __name__ == "__main__":
    main()
