# -*- coding: utf-8 -*-
"""
pipelines/step10_robustness_analysis.py
=========================================
Robustness checks: verify results are stable across presets and window sizes.

What it does:
  For every combination of (preset × window_size), computes:
    - sum of absolute media effect in all event windows
    - sum of BMPI-weighted excess media effect
    - excess share of total media effect (%)
    - excess share of total residual (%)
    - average BMPI score within event windows

  If the main finding holds across all 9 combinations (3 presets × 3 windows),
  it demonstrates robustness — a key requirement for Q1 publication.

  Paper result: stable range 51.2–53.5 percentage points across all 9 combos.

Input:  data/processed/news_effect_daily.csv          (from step07)
        data/processed/excess_media_effect_daily.csv  (from step09)
        data/processed/events_peaks_*.csv             (from step03)

Output: data/processed/robustness_results.csv
        data/processed/robustness_results_summary.json

Next step: step11_advanced_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

NEWS_EFFECT_CSV  = DATA_PROCESSED / "news_effect_daily.csv"
EXCESS_MEDIA_CSV = DATA_PROCESSED / "excess_media_effect_daily.csv"

PEAKS_SENSITIVE = DATA_PROCESSED / "events_peaks_sensitive.csv"
PEAKS_BALANCED  = DATA_PROCESSED / "events_peaks_balanced.csv"
PEAKS_STRONG    = DATA_PROCESSED / "events_peaks_strong.csv"

OUT_CSV  = DATA_PROCESSED / "robustness_results.csv"
OUT_JSON = DATA_PROCESSED / "robustness_results_summary.json"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

WINDOWS = [
    ("w7",  7,  7),
    ("w14", 14, 14),
    ("w30", 30, 30),
]

PRESETS = [
    ("sensitive", PEAKS_SENSITIVE),
    ("balanced",  PEAKS_BALANCED),
    ("strong",    PEAKS_STRONG),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _detect_peak_date_col(df: pd.DataFrame) -> str:
    for c in ["peak_date", "data_piku", "date", "data", "event_date"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "data"]:
        if c in df.columns:
            return c
    raise ValueError("Date column not found")


def read_peaks(path: Path, preset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = _norm_cols(pd.read_csv(path))
    date_col = _detect_peak_date_col(df)
    df["peak_date"] = _parse_dates(df[date_col])
    return (df[["peak_date"]]
              .assign(preset=preset_name)
              .dropna(subset=["peak_date"])
              .drop_duplicates(subset=["preset", "peak_date"])
              .sort_values("peak_date")
              .reset_index(drop=True))


def get_window(df: pd.DataFrame, peak_date: pd.Timestamp,
               before: int, after: int) -> pd.DataFrame:
    start = peak_date - pd.Timedelta(days=before)
    end   = peak_date + pd.Timedelta(days=after)
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def safe_abs_sum(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").fillna(0.0).abs().sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 10 — ROBUSTNESS ANALYSIS (3 presets × 3 windows)")
    print("=" * 60 + "\n")

    for path in [NEWS_EFFECT_CSV, EXCESS_MEDIA_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                "Run step07 and step09 first."
            )

    # Load daily data
    news = _norm_cols(pd.read_csv(NEWS_EFFECT_CSV))
    news["date"] = _parse_dates(news[_detect_date_col(news)])

    excess = _norm_cols(pd.read_csv(EXCESS_MEDIA_CSV))
    excess["date"] = _parse_dates(excess[_detect_date_col(excess)])

    # Merge: news_effect + excess_media + bmpi_score
    df = news.merge(
        excess[["date", "excess_media_effect_usd", "bmpi_score"]],
        on="date", how="left"
    )
    print(f"  Merged daily data: {len(df)} rows")

    # Resolve column names (English or Polish)
    effect_col = next(
        (c for c in ["media_effect_usd", "news_effect_usd"] if c in df.columns),
        None
    )
    resid_col = "resid_btc_mcap_usd"

    for col in [effect_col, resid_col, "excess_media_effect_usd", "bmpi_score"]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        elif col:
            raise ValueError(f"Column missing: {col}")

    rows: List[Dict] = []

    for preset_name, preset_path in PRESETS:
        if not preset_path.exists():
            print(f"  [SKIP] {preset_name}: file not found")
            continue

        peaks = read_peaks(preset_path, preset_name)
        print(f"  [{preset_name}] {len(peaks)} peaks")

        for wtag, before, after in WINDOWS:
            sum_abs_media  = 0.0
            sum_abs_excess = 0.0
            sum_abs_resid  = 0.0
            bmpi_means: List[float] = []

            for _, pr in peaks.iterrows():
                w = get_window(df, pr["peak_date"], before, after)
                sum_abs_media  += safe_abs_sum(w[effect_col])
                sum_abs_excess += safe_abs_sum(w["excess_media_effect_usd"])
                sum_abs_resid  += safe_abs_sum(w[resid_col])
                if len(w) > 0:
                    bmpi_means.append(float(w["bmpi_score"].mean()))

            denom_media = sum_abs_media if sum_abs_media > 0 else 1e-9
            denom_resid = sum_abs_resid if sum_abs_resid > 0 else 1e-9

            rows.append({
                "preset":                    preset_name,
                "window":                    wtag,
                "before_days":               before,
                "after_days":                after,
                "n_peaks":                   int(len(peaks)),
                "sum_abs_media_effect_usd":  sum_abs_media,
                "sum_abs_excess_media_usd":  sum_abs_excess,
                "sum_abs_resid_usd":         sum_abs_resid,
                "excess_share_of_media_pct": float(100.0 * sum_abs_excess / denom_media),
                "excess_share_of_resid_pct": float(100.0 * sum_abs_excess / denom_resid),
                "avg_bmpi_in_windows":       float(np.mean(bmpi_means)) if bmpi_means else 0.0,
            })

    if not rows:
        raise RuntimeError("No results. Check events_peaks_*.csv in data/processed.")

    out = (pd.DataFrame(rows)
             .sort_values(["preset", "before_days"])
             .reset_index(drop=True))
    out.to_csv(OUT_CSV, index=False)

    summary: Dict = {
        "windows":        [{"tag": t, "before": b, "after": a} for t, b, a in WINDOWS],
        "presets_used":   [p for p, path in PRESETS if path.exists()],
        "n_combinations": int(len(out)),
        "excess_share_range": {
            "min_pct": float(out["excess_share_of_media_pct"].min()),
            "max_pct": float(out["excess_share_of_media_pct"].max()),
            "mean_pct": float(out["excess_share_of_media_pct"].mean()),
            "std_pct":  float(out["excess_share_of_media_pct"].std()),
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 78)
    print(f"  {'Preset':<12} {'Window':<8} {'N peaks':<9} "
          f"{'Excess/Media%':>14} {'Excess/Resid%':>14} {'Avg BMPI':>9}")
    print("  " + "─" * 74)
    for _, row in out.iterrows():
        print(f"  {row['preset']:<12} {row['window']:<8} {int(row['n_peaks']):<9} "
              f"{row['excess_share_of_media_pct']:>13.1f}% "
              f"{row['excess_share_of_resid_pct']:>13.1f}% "
              f"{row['avg_bmpi_in_windows']:>9.4f}")
    print("=" * 78)
    print()
    rng = summary["excess_share_range"]
    print(f"  Excess/Media range: {rng['min_pct']:.1f}% – {rng['max_pct']:.1f}%  "
          f"(std={rng['std_pct']:.2f}pp)")
    print(f"  Paper target: 51.2% – 53.5%  (range=2.3pp)")
    print()
    print(f"  ✓  robustness_results.csv       : {len(out)} rows")
    print(f"  ✓  robustness_results_summary.json")
    print("=" * 78)
    print("Next step: step11_advanced_metrics.py\n")


if __name__ == "__main__":
    main()