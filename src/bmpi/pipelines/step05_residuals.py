# -*- coding: utf-8 -*-
"""
pipelines/step05_residuals.py
================================
Compute residual statistics per peak event window — all three presets.

What it does:
  For each detected BTC price peak (from step03), takes the event window
  and computes how much the actual BTC market cap DEVIATED from the baseline
  model prediction (from step04).

  Runs for all three presets: balanced, strong, sensitive.

Input:  data/processed/baseline_predictions.csv       (from step04)
        data/processed/events_peaks_balanced.csv      (from step03)
        data/processed/events_peaks_strong.csv        (from step03)
        data/processed/events_peaks_sensitive.csv     (from step03)

Output per preset:
  data/processed/residuals_by_event_balanced.csv
  data/processed/residuals_by_event_strong.csv
  data/processed/residuals_by_event_sensitive.csv
  data/processed/residuals_summary_balanced.csv
  data/processed/residuals_summary_strong.csv
  data/processed/residuals_summary_sensitive.csv
  data/processed/residuals_by_event_all.csv   — all presets combined
  data/processed/residuals_summary_all.csv

Key output columns:
  total_anomaly_usd       — sum of residuals in USD over the event window
  mean_daily_anomaly_usd  — average daily residual in USD
  mean_anomaly_pct        — average residual as % of baseline
  total_abs_anomaly_usd   — absolute sum (total capital affected)

Next step: step06_merge_news_market.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

BASELINE_FILE = DATA_PROCESSED / "baseline_predictions.csv"
PRESETS       = ["balanced", "strong", "sensitive"]


def load_baseline() -> pd.DataFrame:
    """Load baseline predictions from step04. Supports English and Polish column names."""
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
    required = {"baseline_btc_mcap_hat_usd", "resid_btc_mcap_usd", "resid_btc_mcap_pct"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in baseline_predictions.csv: {missing}")
    return df.sort_values("date").reset_index(drop=True)


def process_preset(df_baseline: pd.DataFrame, preset: str) -> tuple[pd.DataFrame, dict]:
    """Compute residual statistics for one preset."""
    events_file = DATA_PROCESSED / f"events_peaks_{preset}.csv"
    if not events_file.exists():
        print(f"  [{preset}] Events file not found — skipping: {events_file}")
        return pd.DataFrame(), {}

    events = pd.read_csv(events_file)

    # Support both English and Polish column names
    peak_col  = "peak_date"    if "peak_date"    in events.columns else "data_piku"
    start_col = "window_start" if "window_start" in events.columns else "okno_start"
    end_col   = "window_end"   if "window_end"   in events.columns else "okno_koniec"

    for col in [peak_col, start_col, end_col]:
        events[col] = pd.to_datetime(events[col], errors="coerce")

    rows = []
    for _, ev in events.iterrows():
        start = ev[start_col]
        end   = ev[end_col]

        window_df = df_baseline[
            (df_baseline["date"] >= start) &
            (df_baseline["date"] <= end)
        ].copy()

        if window_df.empty:
            continue

        resid_usd = window_df["resid_btc_mcap_usd"]
        resid_pct = window_df["resid_btc_mcap_pct"]

        rows.append({
            "event_id":               ev["event_id"],
            "preset":                 preset,
            "peak_date":              ev[peak_col],
            "window_start":           start,
            "window_end":             end,
            "window_days":            len(window_df),
            "total_anomaly_usd":      float(resid_usd.sum()),
            "mean_daily_anomaly_usd": float(resid_usd.mean()),
            "mean_anomaly_pct":       float(resid_pct.mean()),
            "total_abs_anomaly_usd":  float(resid_usd.abs().sum()),
            "max_daily_anomaly_usd":  float(resid_usd.max()),
            "min_daily_anomaly_usd":  float(resid_usd.min()),
            "positive_days":          int((resid_usd > 0).sum()),
            "negative_days":          int((resid_usd < 0).sum()),
        })

    result = pd.DataFrame(rows).sort_values("peak_date").reset_index(drop=True)

    if result.empty:
        return result, {"preset": preset, "n_events": 0}

    summary = {
        "preset":                     preset,
        "n_events":                   int(len(result)),
        "total_anomaly_usd":          float(result["total_anomaly_usd"].sum()),
        "total_abs_anomaly_usd":      float(result["total_abs_anomaly_usd"].sum()),
        "mean_anomaly_per_event_usd": float(result["total_anomaly_usd"].mean()),
        "median_anomaly_usd":         float(result["total_anomaly_usd"].median()),
        "max_anomaly_event_usd":      float(result["total_anomaly_usd"].max()),
        "min_anomaly_event_usd":      float(result["total_anomaly_usd"].min()),
        "mean_anomaly_pct":           float(result["mean_anomaly_pct"].mean()),
    }
    return result, summary


def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 05 — RESIDUALS BY EVENT (all presets)")
    print("=" * 60 + "\n")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_baseline = load_baseline()
    print(f"  Baseline loaded: {len(df_baseline)} rows  "
          f"({df_baseline['date'].min().date()} → {df_baseline['date'].max().date()})\n")

    all_results, all_summaries = [], []

    for preset in PRESETS:
        result, summary = process_preset(df_baseline, preset)
        if result.empty:
            continue

        result.to_csv(DATA_PROCESSED / f"residuals_by_event_{preset}.csv",  index=False)
        pd.DataFrame([summary]).to_csv(
            DATA_PROCESSED / f"residuals_summary_{preset}.csv", index=False
        )

        all_results.append(result)
        all_summaries.append(summary)

        print(f"  [{preset.upper()}]  {summary['n_events']} events")
        print(f"    total_anomaly_usd:      ${summary['total_anomaly_usd']:>20,.0f}")
        print(f"    total_abs_anomaly_usd:  ${summary['total_abs_anomaly_usd']:>20,.0f}")
        print(f"    mean_per_event_usd:     ${summary['mean_anomaly_per_event_usd']:>20,.0f}")
        print(f"    median_event_usd:       ${summary['median_anomaly_usd']:>20,.0f}")
        print(f"    mean_anomaly_pct:        {summary['mean_anomaly_pct']:>20.2f}%")
        print()

    if not all_results:
        print("  No results — check that step03 and step04 completed successfully.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(DATA_PROCESSED / "residuals_by_event_all.csv",   index=False)
    pd.DataFrame(all_summaries).to_csv(
        DATA_PROCESSED / "residuals_summary_all.csv", index=False
    )

    print("=" * 60)
    print(f"  ✓  residuals_by_event_all.csv : {len(combined)} rows")
    print(f"  ✓  residuals_summary_all.csv  : {len(all_summaries)} presets")
    print("=" * 60)
    print("Next step: step06_merge_news_market.py\n")


if __name__ == "__main__":
    main()