# -*- coding: utf-8 -*-
"""
pipelines/step10_robustness_analysis.py
=========================================
BMPI v2 robustness checks: verify results are stable across presets and window sizes.

What it does:
  For every combination of (preset × window_size), computes:
    - sum of absolute media effect in all UNIQUE event-window days
    - sum of BMPI-weighted excess media effect
    - excess share of total media effect (%)
    - excess share of total abnormal move (%)
    - average BMPI score within event windows

  IMPORTANT:
  - Uses UNIQUE DAYS per (preset × window) to avoid double counting overlapping events.
  - Uses OOF media effect as primary if available.

Input:
  data/processed/news_effect_daily.csv          (from step07)
  data/processed/excess_media_effect_daily.csv  (from step09)
  data/processed/events_peaks_*.csv             (from step03)

Output:
  data/processed/robustness_results.csv
  data/processed/robustness_results_summary.json

Next step:
  step11_advanced_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

NEWS_EFFECT_CSV = DATA_PROCESSED / "news_effect_daily.csv"
EXCESS_MEDIA_CSV = DATA_PROCESSED / "excess_media_effect_daily.csv"

PEAKS_SENSITIVE = DATA_PROCESSED / "events_peaks_sensitive.csv"
PEAKS_BALANCED = DATA_PROCESSED / "events_peaks_balanced.csv"
PEAKS_STRONG = DATA_PROCESSED / "events_peaks_strong.csv"

OUT_CSV = DATA_PROCESSED / "robustness_results.csv"
OUT_JSON = DATA_PROCESSED / "robustness_results_summary.json"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

WINDOWS: List[Tuple[str, int, int]] = [
    ("w7", 7, 7),
    ("w14", 14, 14),
    ("w30", 30, 30),
]

PRESETS: List[Tuple[str, Path]] = [
    ("sensitive", PEAKS_SENSITIVE),
    ("balanced", PEAKS_BALANCED),
    ("strong", PEAKS_STRONG),
]

PRIMARY_EFFECT_COL = "predicted_media_effect_usd_oof"
SECONDARY_EFFECT_COL = "predicted_media_effect_usd"
PRIMARY_ABNORMAL_COL = "abnormal_btc_mcap_usd"

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

    return (
        df[["peak_date"]]
        .assign(preset=preset_name)
        .dropna(subset=["peak_date"])
        .drop_duplicates(subset=["preset", "peak_date"])
        .sort_values("peak_date")
        .reset_index(drop=True)
    )


def safe_abs_sum(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").fillna(0.0).abs().sum())


def safe_mean(s: pd.Series) -> float:
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    if len(s_num) == 0:
        return 0.0
    return float(s_num.mean())


def get_effect_col(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Return (column_name, mode_label)
    """
    if PRIMARY_EFFECT_COL in df.columns:
        return PRIMARY_EFFECT_COL, "OOF"
    if SECONDARY_EFFECT_COL in df.columns:
        return SECONDARY_EFFECT_COL, "IN_SAMPLE_FALLBACK"
    raise ValueError(
        "No media effect column found.\n"
        f"Expected one of: {PRIMARY_EFFECT_COL}, {SECONDARY_EFFECT_COL}"
    )


def get_abnormal_col(df: pd.DataFrame) -> str:
    if PRIMARY_ABNORMAL_COL in df.columns:
        return PRIMARY_ABNORMAL_COL
    raise ValueError(
        f"Missing abnormal move column: {PRIMARY_ABNORMAL_COL}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 10 — ROBUSTNESS ANALYSIS (BMPI v2, FIXED)")
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

    # Merge daily tables
    merge_cols = ["date", "excess_media_effect_usd", "bmpi_score"]
    existing_merge_cols = [c for c in merge_cols if c in excess.columns]

    df = news.merge(
        excess[existing_merge_cols],
        on="date",
        how="left",
        suffixes=("", "_excess"),
    )

    print(f"  Merged daily data: {len(df)} rows")

    # Detect core columns
    effect_col, effect_mode = get_effect_col(df)
    abnormal_col = get_abnormal_col(df)

    # Numeric cleanup
    cols_to_numeric = [effect_col, abnormal_col]
    if "excess_media_effect_usd" in df.columns:
        cols_to_numeric.append("excess_media_effect_usd")
    if "bmpi_score" in df.columns:
        cols_to_numeric.append("bmpi_score")

    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    rows: List[Dict] = []

    for preset_name, preset_path in PRESETS:
        if not preset_path.exists():
            print(f"  [SKIP] {preset_name}: file not found")
            continue

        peaks = read_peaks(preset_path, preset_name)
        print(f"  [{preset_name}] {len(peaks)} peaks")

        for wtag, before, after in WINDOWS:
            used_dates: Set[pd.Timestamp] = set()
            bmpi_values: List[float] = []

            for _, pr in peaks.iterrows():
                start = pr["peak_date"] - pd.Timedelta(days=before)
                end = pr["peak_date"] + pd.Timedelta(days=after)

                window = df[
                    (df["date"] >= start) &
                    (df["date"] <= end)
                ].copy()

                if len(window) == 0:
                    continue

                used_dates.update(window["date"].dropna().tolist())

                if "bmpi_score" in window.columns:
                    bmpi_values.append(float(window["bmpi_score"].mean()))

            # IMPORTANT: use unique days only
            window_df = df[df["date"].isin(used_dates)].copy()

            sum_abs_media = safe_abs_sum(window_df[effect_col])
            sum_abs_excess = (
                safe_abs_sum(window_df["excess_media_effect_usd"])
                if "excess_media_effect_usd" in window_df.columns else 0.0
            )
            sum_abs_abnormal = safe_abs_sum(window_df[abnormal_col])

            denom_media = sum_abs_media if sum_abs_media > 0 else 1e-9
            denom_abnormal = sum_abs_abnormal if sum_abs_abnormal > 0 else 1e-9

            rows.append({
                "preset": preset_name,
                "window": wtag,
                "before_days": before,
                "after_days": after,
                "n_peaks": int(len(peaks)),
                "n_unique_days": int(len(used_dates)),
                "effect_mode": effect_mode,
                "effect_column_used": effect_col,
                "abnormal_column_used": abnormal_col,

                "sum_abs_media_effect_usd": float(sum_abs_media),
                "sum_abs_excess_media_usd": float(sum_abs_excess),
                "sum_abs_abnormal_usd": float(sum_abs_abnormal),

                "excess_share_of_media_pct": float(100.0 * sum_abs_excess / denom_media),
                "excess_share_of_abnormal_pct": float(100.0 * sum_abs_excess / denom_abnormal),
                "avg_bmpi_in_windows": float(np.mean(bmpi_values)) if bmpi_values else 0.0,
            })

    if not rows:
        raise RuntimeError("No robustness results produced. Check event files.")

    out = pd.DataFrame(rows).sort_values(["preset", "before_days"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    summary: Dict = {
        "effect_mode_used": effect_mode,
        "windows": [{"tag": t, "before": b, "after": a} for t, b, a in WINDOWS],
        "presets_used": [p for p, path in PRESETS if path.exists()],
        "n_combinations": int(len(out)),
        "excess_share_of_media_range": {
            "min_pct": float(out["excess_share_of_media_pct"].min()),
            "max_pct": float(out["excess_share_of_media_pct"].max()),
            "mean_pct": float(out["excess_share_of_media_pct"].mean()),
            "std_pct": float(out["excess_share_of_media_pct"].std(ddof=0)),
        },
        "excess_share_of_abnormal_range": {
            "min_pct": float(out["excess_share_of_abnormal_pct"].min()),
            "max_pct": float(out["excess_share_of_abnormal_pct"].max()),
            "mean_pct": float(out["excess_share_of_abnormal_pct"].mean()),
            "std_pct": float(out["excess_share_of_abnormal_pct"].std(ddof=0)),
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 84)
    print(f"  {'Preset':<12} {'Window':<8} {'N peaks':<9} {'Unique days':<12} "
          f"{'Excess/Media%':>14} {'Excess/Abn%':>14} {'Avg BMPI':>10}")
    print("  " + "─" * 80)

    for _, row in out.iterrows():
        print(
            f"  {row['preset']:<12} {row['window']:<8} {int(row['n_peaks']):<9} "
            f"{int(row['n_unique_days']):<12} "
            f"{row['excess_share_of_media_pct']:>13.2f}% "
            f"{row['excess_share_of_abnormal_pct']:>13.2f}% "
            f"{row['avg_bmpi_in_windows']:>10.4f}"
        )

    print("=" * 84)

    rng_media = summary["excess_share_of_media_range"]
    rng_abn = summary["excess_share_of_abnormal_range"]

    print(
        f"\n  Excess/Media range:    {rng_media['min_pct']:.2f}% – {rng_media['max_pct']:.2f}%  "
        f"(std={rng_media['std_pct']:.2f}pp)"
    )
    print(
        f"  Excess/Abnormal range: {rng_abn['min_pct']:.2f}% – {rng_abn['max_pct']:.2f}%  "
        f"(std={rng_abn['std_pct']:.2f}pp)"
    )
    print(f"\n  ✓  robustness_results.csv       : {len(out)} rows")
    print(f"  ✓  robustness_results_summary.json")
    print("=" * 84)
    print("Next step: step11_advanced_metrics.py\n")


if __name__ == "__main__":
    main()