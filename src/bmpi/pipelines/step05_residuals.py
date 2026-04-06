# -*- coding: utf-8 -*-
"""
pipelines/step05_residuals.py
================================
BMPI v2 event-level abnormal return analysis.

What changed vs BMPI v1:
- We no longer treat baseline residual sums as "media anomaly".
- We use abnormal BTC returns from step04 as the event-study foundation.
- We compute CAR (cumulative abnormal return) over multiple windows.
- We compute pre-event drift, post-event continuation, and reversal metrics.

Input:
  data/processed/baseline_predictions.csv       (from step04)
  data/processed/events_peaks_balanced.csv      (from step03)
  data/processed/events_peaks_strong.csv        (from step03)
  data/processed/events_peaks_sensitive.csv     (from step03)

Expected baseline columns:
  date
  btc_price
  btc_mcap
  btc_logret
  expected_btc_logret
  abnormal_btc_logret
  abnormal_btc_logret_zscore_60d
  abnormal_btc_return_pct
  abnormal_btc_mcap_usd

Output per preset:
  data/processed/residuals_by_event_balanced.csv
  data/processed/residuals_by_event_strong.csv
  data/processed/residuals_by_event_sensitive.csv
  data/processed/residuals_summary_balanced.csv
  data/processed/residuals_summary_strong.csv
  data/processed/residuals_summary_sensitive.csv

Combined:
  data/processed/residuals_by_event_all.csv
  data/processed/residuals_summary_all.csv

Notes:
- File name is kept unchanged to preserve pipeline compatibility.
- Semantically, this is now "event abnormal return analysis", not raw residual summation.

Next step:
  step06_merge_news_market.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

BASELINE_FILE = DATA_PROCESSED / "baseline_predictions.csv"
PRESETS = ["balanced", "strong", "sensitive"]

# ---------------------------------------------------------------------------
# Event windows
# ---------------------------------------------------------------------------

CAR_WINDOWS = [3, 7, 14, 30]

PRE_EVENT_WINDOW = 7
POST_EVENT_WINDOW = 7
REVERSAL_WINDOW = 7

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_baseline() -> pd.DataFrame:
    """
    Load baseline predictions from step04.
    Supports both old and new naming if needed.
    """
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {BASELINE_FILE}\n"
            "Run step04_baseline_model.py first."
        )

    df = pd.read_csv(BASELINE_FILE)

    date_col = "date" if "date" in df.columns else "data"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if date_col != "date":
        df = df.rename(columns={"data": "date"})

    required = {
        "btc_logret",
        "expected_btc_logret",
        "abnormal_btc_logret",
        "abnormal_btc_return_pct",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in baseline_predictions.csv: {missing}\n"
            "Expected BMPI v2 output from step04."
        )

    numeric_candidates = [
        "btc_price",
        "btc_mcap",
        "btc_logret",
        "expected_btc_logret",
        "abnormal_btc_logret",
        "abnormal_btc_logret_zscore_60d",
        "btc_return_pct",
        "expected_btc_return_pct",
        "abnormal_btc_return_pct",
        "baseline_btc_mcap_hat_usd",
        "abnormal_btc_mcap_usd",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_events(preset: str) -> pd.DataFrame:
    """
    Load event peaks file for one preset.
    Supports English and Polish column names.
    """
    events_file = DATA_PROCESSED / f"events_peaks_{preset}.csv"
    if not events_file.exists():
        raise FileNotFoundError(f"Events file not found: {events_file}")

    events = pd.read_csv(events_file)

    peak_col = "peak_date" if "peak_date" in events.columns else "data_piku"
    start_col = "window_start" if "window_start" in events.columns else "okno_start"
    end_col = "window_end" if "window_end" in events.columns else "okno_koniec"

    for col in [peak_col, start_col, end_col]:
        events[col] = pd.to_datetime(events[col], errors="coerce")

    rename_map = {}
    if peak_col != "peak_date":
        rename_map[peak_col] = "peak_date"
    if start_col != "window_start":
        rename_map[start_col] = "window_start"
    if end_col != "window_end":
        rename_map[end_col] = "window_end"

    events = events.rename(columns=rename_map)
    return events.sort_values("peak_date").reset_index(drop=True)


def nearest_index_by_date(df: pd.DataFrame, dt: pd.Timestamp) -> int | None:
    """
    Return exact matching index for date.
    If date not found, return None.
    """
    matches = df.index[df["date"] == dt].tolist()
    if not matches:
        return None
    return matches[0]


def slice_by_relative_window(df: pd.DataFrame, center_idx: int, left_days: int, right_days: int) -> pd.DataFrame:
    """
    Slice by row offsets around an event center.
    This uses aligned daily series rows, which is acceptable here because
    baseline_predictions is already a daily aligned frame.
    """
    start_idx = max(0, center_idx - left_days)
    end_idx = min(len(df) - 1, center_idx + right_days)
    return df.iloc[start_idx:end_idx + 1].copy()


def compute_car(window_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute cumulative abnormal return in log-return space and approximate pct.
    """
    if window_df.empty:
        return {
            "car_log": np.nan,
            "car_pct_approx": np.nan,
            "mean_abnormal_logret": np.nan,
            "mean_abnormal_return_pct": np.nan,
            "sum_abnormal_mcap_usd": np.nan,
            "sum_abs_abnormal_mcap_usd": np.nan,
            "window_len": 0,
        }

    car_log = float(window_df["abnormal_btc_logret"].sum())
    car_pct_approx = float((np.exp(car_log) - 1.0) * 100.0)

    out = {
        "car_log": car_log,
        "car_pct_approx": car_pct_approx,
        "mean_abnormal_logret": float(window_df["abnormal_btc_logret"].mean()),
        "mean_abnormal_return_pct": float(window_df["abnormal_btc_return_pct"].mean())
        if "abnormal_btc_return_pct" in window_df.columns
        else np.nan,
        "sum_abnormal_mcap_usd": float(window_df["abnormal_btc_mcap_usd"].sum())
        if "abnormal_btc_mcap_usd" in window_df.columns
        else np.nan,
        "sum_abs_abnormal_mcap_usd": float(window_df["abnormal_btc_mcap_usd"].abs().sum())
        if "abnormal_btc_mcap_usd" in window_df.columns
        else np.nan,
        "window_len": int(len(window_df)),
    }
    return out


def safe_sign(x: float) -> float:
    if pd.isna(x):
        return np.nan
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def process_preset(df_baseline: pd.DataFrame, preset: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute event-level abnormal return metrics for one preset.
    """
    events = load_events(preset)
    rows: List[Dict[str, float | int | str | pd.Timestamp]] = []

    for _, ev in events.iterrows():
        peak_date = ev["peak_date"]
        peak_idx = nearest_index_by_date(df_baseline, peak_date)
        if peak_idx is None:
            continue

        peak_row = df_baseline.iloc[peak_idx]

        row: Dict[str, float | int | str | pd.Timestamp] = {
            "event_id": ev["event_id"] if "event_id" in ev.index else None,
            "preset": preset,
            "peak_date": peak_date,
            "window_start": ev["window_start"] if "window_start" in ev.index else pd.NaT,
            "window_end": ev["window_end"] if "window_end" in ev.index else pd.NaT,
            "btc_price_at_peak": float(peak_row["btc_price"]) if "btc_price" in peak_row.index else np.nan,
            "btc_mcap_at_peak": float(peak_row["btc_mcap"]) if "btc_mcap" in peak_row.index else np.nan,
            "peak_abnormal_logret": float(peak_row["abnormal_btc_logret"]),
            "peak_abnormal_return_pct": float(peak_row["abnormal_btc_return_pct"])
            if "abnormal_btc_return_pct" in peak_row.index
            else np.nan,
            "peak_abnormal_zscore_60d": float(peak_row["abnormal_btc_logret_zscore_60d"])
            if "abnormal_btc_logret_zscore_60d" in peak_row.index
            else np.nan,
            "peak_abnormal_mcap_usd": float(peak_row["abnormal_btc_mcap_usd"])
            if "abnormal_btc_mcap_usd" in peak_row.index
            else np.nan,
        }

        # Pre-event drift: [-PRE_EVENT_WINDOW, -1]
        pre_df = slice_by_relative_window(df_baseline, peak_idx, PRE_EVENT_WINDOW, 0)
        if not pre_df.empty:
            pre_df = pre_df.iloc[:-1].copy()  # exclude event day itself
        pre_stats = compute_car(pre_df)
        row["pre_event_days"] = pre_stats["window_len"]
        row["pre_event_car_log"] = pre_stats["car_log"]
        row["pre_event_car_pct"] = pre_stats["car_pct_approx"]
        row["pre_event_mean_abnormal_logret"] = pre_stats["mean_abnormal_logret"]
        row["pre_event_mean_abnormal_return_pct"] = pre_stats["mean_abnormal_return_pct"]

        # Post-event continuation: [+1, +POST_EVENT_WINDOW]
        post_df = slice_by_relative_window(df_baseline, peak_idx, 0, POST_EVENT_WINDOW)
        if not post_df.empty:
            post_df = post_df.iloc[1:].copy()  # exclude event day
        post_stats = compute_car(post_df)
        row["post_event_days"] = post_stats["window_len"]
        row["post_event_car_log"] = post_stats["car_log"]
        row["post_event_car_pct"] = post_stats["car_pct_approx"]
        row["post_event_mean_abnormal_logret"] = post_stats["mean_abnormal_logret"]
        row["post_event_mean_abnormal_return_pct"] = post_stats["mean_abnormal_return_pct"]

        # Reversal proxy: opposite sign after event
        row["reversal_ratio"] = (
            float(-post_stats["car_log"] / row["peak_abnormal_logret"])
            if pd.notna(post_stats["car_log"])
            and pd.notna(row["peak_abnormal_logret"])
            and row["peak_abnormal_logret"] != 0
            else np.nan
        )
        row["peak_sign"] = safe_sign(float(row["peak_abnormal_logret"]))
        row["post_sign"] = safe_sign(float(row["post_event_car_log"])) if pd.notna(row["post_event_car_log"]) else np.nan
        row["is_reversal"] = (
            int(row["peak_sign"] * row["post_sign"] < 0)
            if pd.notna(row["peak_sign"]) and pd.notna(row["post_sign"])
            else np.nan
        )

        # CAR windows centered on event day: [0, +W-1]
        for w in CAR_WINDOWS:
            car_df = slice_by_relative_window(df_baseline, peak_idx, 0, w - 1)
            car_stats = compute_car(car_df)

            row[f"car_{w}d_log"] = car_stats["car_log"]
            row[f"car_{w}d_pct"] = car_stats["car_pct_approx"]
            row[f"car_{w}d_mean_abnormal_logret"] = car_stats["mean_abnormal_logret"]
            row[f"car_{w}d_mean_abnormal_return_pct"] = car_stats["mean_abnormal_return_pct"]
            row[f"car_{w}d_sum_abnormal_mcap_usd"] = car_stats["sum_abnormal_mcap_usd"]
            row[f"car_{w}d_sum_abs_abnormal_mcap_usd"] = car_stats["sum_abs_abnormal_mcap_usd"]
            row[f"car_{w}d_window_len"] = car_stats["window_len"]

        # Symmetric event window from step03 if available
        if "window_start" in ev.index and "window_end" in ev.index:
            event_window_df = df_baseline[
                (df_baseline["date"] >= ev["window_start"]) &
                (df_baseline["date"] <= ev["window_end"])
            ].copy()
            event_stats = compute_car(event_window_df)

            row["event_window_days"] = event_stats["window_len"]
            row["event_window_car_log"] = event_stats["car_log"]
            row["event_window_car_pct"] = event_stats["car_pct_approx"]
            row["event_window_mean_abnormal_logret"] = event_stats["mean_abnormal_logret"]
            row["event_window_mean_abnormal_return_pct"] = event_stats["mean_abnormal_return_pct"]
            row["event_window_sum_abnormal_mcap_usd"] = event_stats["sum_abnormal_mcap_usd"]
            row["event_window_sum_abs_abnormal_mcap_usd"] = event_stats["sum_abs_abnormal_mcap_usd"]

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("peak_date").reset_index(drop=True)

    if result.empty:
        return result, {"preset": preset, "n_events": 0}

    summary: Dict[str, float] = {
        "preset": preset,
        "n_events": int(len(result)),
        "mean_peak_abnormal_logret": float(result["peak_abnormal_logret"].mean()),
        "median_peak_abnormal_logret": float(result["peak_abnormal_logret"].median()),
        "mean_peak_abnormal_return_pct": float(result["peak_abnormal_return_pct"].mean()),
        "share_reversal_events": float(result["is_reversal"].dropna().mean()) if "is_reversal" in result.columns else np.nan,
        "mean_pre_event_car_pct": float(result["pre_event_car_pct"].mean()),
        "mean_post_event_car_pct": float(result["post_event_car_pct"].mean()),
        "mean_event_window_car_pct": float(result["event_window_car_pct"].mean())
        if "event_window_car_pct" in result.columns
        else np.nan,
        "mean_event_window_abs_abnormal_mcap_usd": float(result["event_window_sum_abs_abnormal_mcap_usd"].mean())
        if "event_window_sum_abs_abnormal_mcap_usd" in result.columns
        else np.nan,
    }

    for w in CAR_WINDOWS:
        summary[f"mean_car_{w}d_pct"] = float(result[f"car_{w}d_pct"].mean())
        summary[f"median_car_{w}d_pct"] = float(result[f"car_{w}d_pct"].median())
        summary[f"mean_car_{w}d_abs_abnormal_mcap_usd"] = float(result[f"car_{w}d_sum_abs_abnormal_mcap_usd"].mean())
        summary[f"sum_car_{w}d_abnormal_mcap_usd"] = float(result[f"car_{w}d_sum_abnormal_mcap_usd"].sum())

    return result, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 05 — EVENT ABNORMAL RETURNS / CAR (BMPI v2)")
    print("=" * 60 + "\n")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df_baseline = load_baseline()
    print(
        f"  Baseline loaded: {len(df_baseline)} rows  "
        f"({df_baseline['date'].min().date()} -> {df_baseline['date'].max().date()})\n"
    )

    all_results: List[pd.DataFrame] = []
    all_summaries: List[Dict[str, float]] = []

    for preset in PRESETS:
        try:
            result, summary = process_preset(df_baseline, preset)
        except FileNotFoundError as exc:
            print(f"  [{preset}] Skipping: {exc}")
            continue

        if result.empty:
            print(f"  [{preset}] No matched events found.")
            continue

        out_events = DATA_PROCESSED / f"residuals_by_event_{preset}.csv"
        out_summary = DATA_PROCESSED / f"residuals_summary_{preset}.csv"

        result.to_csv(out_events, index=False)
        pd.DataFrame([summary]).to_csv(out_summary, index=False)

        all_results.append(result)
        all_summaries.append(summary)

        print(
            f"  [{preset}] events={summary['n_events']}  "
            f"mean CAR_7d={summary['mean_car_7d_pct']:.4f}%  "
            f"mean event_window CAR={summary['mean_event_window_car_pct']:.4f}%"
        )

    if all_results:
        combined_events = pd.concat(all_results, ignore_index=True)
        combined_events = combined_events.sort_values(["preset", "peak_date"]).reset_index(drop=True)
        combined_events.to_csv(DATA_PROCESSED / "residuals_by_event_all.csv", index=False)

    if all_summaries:
        combined_summary = pd.DataFrame(all_summaries)
        combined_summary.to_csv(DATA_PROCESSED / "residuals_summary_all.csv", index=False)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Saved per-preset event files and summaries to: {DATA_PROCESSED}")
    print("Next step: step06_merge_news_market.py")


if __name__ == "__main__":
    main()