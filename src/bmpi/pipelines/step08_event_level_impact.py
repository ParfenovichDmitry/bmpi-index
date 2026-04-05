# -*- coding: utf-8 -*-
"""
pipelines/step08_event_level_impact.py
========================================
Compute media effect statistics per peak event window.

What it does:
  For each detected BTC price peak (from step03), takes a ±30 day window
  and computes how much of the price anomaly is attributable to media pressure
  (from the Ridge model in step07).

  Also computes additional diagnostic windows: ±7, ±14, ±30 days.

  Key metric per event:
    news_share_of_anomaly_pct = |media_effect_usd| / |resid_btc_mcap_usd| × 100
    → "What % of the unexplained BTC movement was driven by media?"

Input:  data/processed/news_effect_daily.csv        (from step07)
        data/processed/events_peaks_balanced.csv    (from step03)
        data/processed/events_peaks_strong.csv      (from step03)
        data/processed/events_peaks_sensitive.csv   (from step03)

Output: data/processed/news_effect_by_event.csv     — per-event media stats
        data/processed/news_effect_top_events.csv   — top 50 events by media effect
        data/processed/news_effect_by_event_summary.json — summary per preset

Next step: step09_fake_classification.py
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

NEWS_EFFECT_CSV = DATA_PROCESSED / "news_effect_daily.csv"
PEAKS_SENSITIVE = DATA_PROCESSED / "events_peaks_sensitive.csv"
PEAKS_BALANCED  = DATA_PROCESSED / "events_peaks_balanced.csv"
PEAKS_STRONG    = DATA_PROCESSED / "events_peaks_strong.csv"

OUT_BY_EVENT  = DATA_PROCESSED / "news_effect_by_event.csv"
OUT_SUMMARY   = DATA_PROCESSED / "news_effect_by_event_summary.json"
OUT_TOP_EVENTS = DATA_PROCESSED / "news_effect_top_events.csv"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

WINDOW_BEFORE_DAYS = 30
WINDOW_AFTER_DAYS  = 30

EXTRA_WINDOWS = [
    ("w7",  7,  7),
    ("w14", 14, 14),
    ("w30", 30, 30),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _detect_peak_date_col(df: pd.DataFrame) -> str:
    for c in ["peak_date", "data_piku", "date", "data", "event_date"]:
        if c in df.columns:
            return c
    return df.columns[0]


def read_peaks(path: Path, preset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    df = _normalise_cols(df)
    date_col = _detect_peak_date_col(df)
    df["peak_date"] = _parse_dates(df[date_col])
    keep = ["peak_date"]
    for c in ["event_id", "preset", "zscore", "zscore_anomalii"]:
        if c in df.columns:
            keep.append(c)
    out = df[keep].copy()
    out["preset"] = preset_name
    return (out.dropna(subset=["peak_date"])
               .drop_duplicates(subset=["preset", "peak_date"])
               .sort_values("peak_date")
               .reset_index(drop=True))


def load_all_peaks() -> pd.DataFrame:
    parts = []
    for path, name in [
        (PEAKS_SENSITIVE, "sensitive"),
        (PEAKS_BALANCED,  "balanced"),
        (PEAKS_STRONG,    "strong"),
    ]:
        if path.exists():
            parts.append(read_peaks(path, name))
    if not parts:
        raise FileNotFoundError(
            "No peaks files found. Run step03_peak_detection.py first."
        )
    return (pd.concat(parts, ignore_index=True)
              .sort_values(["preset", "peak_date"])
              .reset_index(drop=True))


def get_window(df_daily: pd.DataFrame, peak_date: pd.Timestamp,
               before: int, after: int) -> pd.DataFrame:
    start = peak_date - pd.Timedelta(days=before)
    end   = peak_date + pd.Timedelta(days=after)
    return df_daily[
        (df_daily["date"] >= start) & (df_daily["date"] <= end)
    ].copy()


def safe_abs_sum(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").fillna(0.0).abs().sum())


def safe_sum(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())


def safe_max_abs(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").fillna(0.0).abs()
    return float(v.max()) if len(v) else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 08 — EVENT-LEVEL MEDIA IMPACT")
    print("=" * 60 + "\n")

    if not NEWS_EFFECT_CSV.exists():
        raise FileNotFoundError(
            f"File not found: {NEWS_EFFECT_CSV}\n"
            "Run step07_news_effect_model.py first."
        )

    df = pd.read_csv(NEWS_EFFECT_CSV)
    df = _normalise_cols(df)

    # Resolve date and required columns
    date_col = "date" if "date" in df.columns else "data"
    df["date"] = _parse_dates(df[date_col])

    # Support both English and Polish column names from step07
    effect_col = "media_effect_usd" if "media_effect_usd" in df.columns else "news_effect_usd"
    resid_col  = "resid_btc_mcap_usd"

    for col in [effect_col, resid_col]:
        if col not in df.columns:
            raise ValueError(f"news_effect_daily.csv: column '{col}' not found")

    # Normalise to standard names
    if effect_col != "media_effect_usd":
        df["media_effect_usd"] = df[effect_col]
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"  Daily data: {len(df)} rows  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")

    peaks = load_all_peaks()
    print(f"  Peaks loaded: {len(peaks)} events across "
          f"{peaks['preset'].nunique()} presets\n")

    rows: List[Dict] = []

    for preset_name in peaks["preset"].unique():
        pp = peaks[peaks["preset"] == preset_name].reset_index(drop=True)

        for i, r in pp.iterrows():
            peak_date = r["peak_date"]
            w = get_window(df, peak_date, WINDOW_BEFORE_DAYS, WINDOW_AFTER_DAYS)

            row: Dict = {
                "preset":                   preset_name,
                "event_number":             int(i + 1),
                "event_id":                 r.get("event_id", f"{preset_name.upper()}_EVT_{i+1:04d}"),
                "peak_date":                str(peak_date.date()),
                "window_start":             str((peak_date - pd.Timedelta(days=WINDOW_BEFORE_DAYS)).date()),
                "window_end":               str((peak_date + pd.Timedelta(days=WINDOW_AFTER_DAYS)).date()),
                "window_days":              int(len(w)),
                "sum_media_effect_usd":     safe_sum(w["media_effect_usd"]),
                "sum_abs_media_effect_usd": safe_abs_sum(w["media_effect_usd"]),
                "max_abs_media_effect_usd": safe_max_abs(w["media_effect_usd"]),
                "sum_resid_usd":            safe_sum(w[resid_col]),
                "sum_abs_resid_usd":        safe_abs_sum(w[resid_col]),
                "max_abs_resid_usd":        safe_max_abs(w[resid_col]),
            }

            denom = row["sum_abs_resid_usd"]
            row["news_share_of_anomaly_pct"] = (
                float(100.0 * row["sum_abs_media_effect_usd"] / denom)
                if denom > 0 else 0.0
            )

            # Diagnostic windows
            for tag, b, a in EXTRA_WINDOWS:
                ww = get_window(df, peak_date, b, a)
                abs_news  = safe_abs_sum(ww["media_effect_usd"])
                abs_resid = safe_abs_sum(ww[resid_col])
                row[f"{tag}_sum_abs_news_usd"]  = abs_news
                row[f"{tag}_sum_abs_resid_usd"] = abs_resid
                row[f"{tag}_news_share_pct"]    = (
                    float(100.0 * abs_news / abs_resid)
                    if abs_resid > 0 else 0.0
                )

            rows.append(row)

    by_event = (pd.DataFrame(rows)
                  .sort_values(["preset", "peak_date"])
                  .reset_index(drop=True))
    by_event.to_csv(OUT_BY_EVENT, index=False)

    top = (by_event.sort_values("sum_abs_media_effect_usd", ascending=False)
                   .head(50).copy())
    top.to_csv(OUT_TOP_EVENTS, index=False)

    # Summary
    summary: Dict = {
        "parameters": {
            "window_before_days": WINDOW_BEFORE_DAYS,
            "window_after_days":  WINDOW_AFTER_DAYS,
        },
        "by_preset": {},
    }

    for preset_name in by_event["preset"].unique():
        sub = by_event[by_event["preset"] == preset_name]
        summary["by_preset"][preset_name] = {
            "n_events":                 int(len(sub)),
            "sum_abs_media_effect_usd": float(sub["sum_abs_media_effect_usd"].sum()),
            "mean_abs_media_effect_usd":float(sub["sum_abs_media_effect_usd"].mean()),
            "max_abs_media_effect_usd": float(sub["sum_abs_media_effect_usd"].max()),
            "mean_news_share_pct":      float(sub["news_share_of_anomaly_pct"].mean()),
            "median_news_share_pct":    float(sub["news_share_of_anomaly_pct"].median()),
        }

    summary["global"] = {
        "total_event_rows":         int(len(by_event)),
        "sum_abs_media_effect_usd": float(by_event["sum_abs_media_effect_usd"].sum()),
        "max_abs_media_effect_usd": float(by_event["sum_abs_media_effect_usd"].max()),
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"  ✓  news_effect_by_event.csv      : {len(by_event)} rows")
    print(f"  ✓  news_effect_top_events.csv     : {len(top)} rows")
    print(f"  ✓  news_effect_by_event_summary.json")
    print()
    print("  Quick summary (per preset):")
    for pn, s in summary["by_preset"].items():
        print(f"    [{pn}]  n={s['n_events']}  "
              f"mean_news_share={s['mean_news_share_pct']:.1f}%  "
              f"sum_abs=${s['sum_abs_media_effect_usd']:,.0f}")
    print("=" * 60)
    print("Next step: step09_fake_classification.py\n")


if __name__ == "__main__":
    main()