# -*- coding: utf-8 -*-
"""
pipelines/step06_merge_news_market.py
============================================
Build the final modelling dataset by merging all data sources.

What it does:
  Combines market features, baseline model residuals, and GDELT media
  signals into one aligned daily DataFrame ready for Ridge regression
  in step07.

  Also engineers media features:
    gdelt_log_mentions_all          — log(1 + mentions)
    gdelt_tone_x_log_mentions_all   — tone x log_mentions interaction
    *_lag1, *_lag3, *_lag7          — lagged values (1, 3, 7 days back)
    *_ma3, *_ma7                    — 3-day and 7-day rolling means

Input:
  data/processed/features_daily.parquet                      (from step02)
  data/processed/baseline_predictions.csv                    (from step04)
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv (required)
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_strong.csv   (optional)
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_sensitive.csv(optional)

  Note: column names in GDELT CSVs are in Polish
  (data, liczba_wzmianek, sredni_tone, zrodlo) — handled automatically.

Output:
  data/processed/model_dataset_daily.csv
  data/processed/model_dataset_daily.parquet

Next step: step07_news_effect_model.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_GDELT     = BASE_DIR / "data" / "raw" / "gdelt"

FEATURES_PARQUET = DATA_PROCESSED / "features_daily.parquet"
BASELINE_CSV     = DATA_PROCESSED / "baseline_predictions.csv"

# GDELT daily signal files — produced by gdelt_btc_downloader.py
# Column names: data, liczba_wzmianek, sredni_tone, zrodlo
GDELT_BALANCED  = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_balanced.csv"
GDELT_STRONG    = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_strong.csv"
GDELT_SENSITIVE = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_sensitive.csv"

OUT_CSV     = DATA_PROCESSED / "model_dataset_daily.csv"
OUT_PARQUET = DATA_PROCESSED / "model_dataset_daily.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_cols(cols) -> list[str]:
    """Normalise column names to lowercase snake_case."""
    out = []
    for c in cols:
        c2 = str(c).strip().replace("\ufeff", "").replace(" ", "_")
        c2 = re.sub(r"[^\w_]+", "_", c2)
        c2 = re.sub(r"__+", "_", c2).strip("_")
        out.append(c2.lower())
    return out


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse date column regardless of format — normalise to date-only."""
    return pd.to_datetime(s.astype(str).str[:10], errors="coerce")


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return first candidate column that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _log1p(series: pd.Series) -> pd.Series:
    """Safe log(1 + x)."""
    x = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return pd.Series(np.log1p(x.to_numpy(dtype=float)), index=series.index)


def _add_lags(df: pd.DataFrame, col: str, lags=(1, 3, 7)) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def _add_rolling_means(df: pd.DataFrame, col: str, windows=(3, 7)) -> pd.DataFrame:
    for w in windows:
        df[f"{col}_ma{w}"] = df[col].rolling(window=w, min_periods=1).mean()
    return df


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path)
    df.columns = _normalise_cols(df.columns)
    date_col = _pick_col(df, ["date", "data"])
    if not date_col:
        raise ValueError("features_daily.parquet: date column not found")
    df["date"] = _parse_dates(df[date_col])
    if date_col != "date":
        df = df.drop(columns=[date_col])
    return (df.dropna(subset=["date"])
              .drop_duplicates("date")
              .sort_values("date")
              .reset_index(drop=True))


def load_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    df.columns = _normalise_cols(df.columns)
    date_col = _pick_col(df, ["date", "data"])
    if not date_col:
        raise ValueError("baseline_predictions.csv: date column not found")
    df["date"] = _parse_dates(df[date_col])
    if date_col != "date":
        df = df.drop(columns=[date_col])
    return (df.dropna(subset=["date"])
              .drop_duplicates("date")
              .sort_values("date")
              .reset_index(drop=True))


def load_gdelt(path: Path, suffix: str) -> pd.DataFrame:
    """
    Load a GDELT daily signal CSV produced by gdelt_btc_downloader.py.

    Expected columns (Polish names from downloader):
      data           — date
      liczba_wzmianek — mention count
      sredni_tone    — average tone
      zrodlo         — source label (ignored)

    Also accepts English aliases: date/mentions/tone.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    df.columns = _normalise_cols(df.columns)

    date_col = _pick_col(df, ["date", "data", "day"])
    if not date_col:
        raise ValueError(f"{path.name}: date column not found")
    df["date"] = _parse_dates(df[date_col])

    # Polish column names from gdelt_btc_downloader.py
    mentions_col = _pick_col(df, [
        "liczba_wzmianek", "mentions", "mention_count", "count_mentions"
    ])
    tone_col = _pick_col(df, [
        "sredni_tone", "tone", "avg_tone", "tone_avg", "tone_mean"
    ])

    out = pd.DataFrame({"date": df["date"]})
    out[f"gdelt_mentions{suffix}"] = (
        pd.to_numeric(df[mentions_col], errors="coerce").fillna(0.0)
        if mentions_col else 0.0
    )
    out[f"gdelt_tone{suffix}"] = (
        pd.to_numeric(df[tone_col], errors="coerce")
        if tone_col else np.nan
    )

    return (out.dropna(subset=["date"])
               .drop_duplicates("date")
               .sort_values("date")
               .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_model_dataset() -> pd.DataFrame:
    # 1. Market features (from step02)
    features = load_features(FEATURES_PARQUET)
    print(f"  features_daily:        {len(features)} rows")

    # 2. Baseline predictions + residuals (from step04)
    baseline = load_baseline(BASELINE_CSV)
    print(f"  baseline_predictions:  {len(baseline)} rows")

    # 3. GDELT signal files — balanced is required, others optional
    gdelt_parts: list[pd.DataFrame] = []
    has_all = False

    for path, suffix, label in [
        (GDELT_BALANCED,  "_balanced",  "balanced"),
        (GDELT_STRONG,    "_strong",    "strong"),
        (GDELT_SENSITIVE, "_sensitive", "sensitive"),
    ]:
        if not path.exists():
            print(f"  gdelt_{label}: not found — skipping")
            continue
        g = load_gdelt(path, suffix)
        print(f"  gdelt_{label}:{' ' * max(1, 13-len(label))}{len(g)} rows")
        gdelt_parts.append(g)

        # First available preset becomes the "_all" primary signal
        if not has_all:
            g_all = load_gdelt(path, "_all")
            gdelt_parts.insert(0, g_all)
            print(f"  gdelt_all (from {label}): {len(g_all)} rows")
            has_all = True

    if not gdelt_parts:
        raise FileNotFoundError(
            "No GDELT signal files found in data/raw/gdelt/\n"
            "Expected: gdelt_gkg_bitcoin_daily_signal_balanced.csv\n"
            "Run: python src/bmpi/utils/gdelt_btc_downloader.py "
            "--start 2015-10-01 --end 2026-01-31 --preset balanced"
        )

    # 4. Merge everything on date axis
    df = baseline.merge(features, on="date", how="left")
    for g in gdelt_parts:
        df = df.merge(g, on="date", how="left")

    # 5. Fill missing media values
    for col in df.columns:
        if col.startswith("gdelt_mentions"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if col.startswith("gdelt_tone"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 6. Feature engineering on primary (_all) signal
    if "gdelt_mentions_all" in df.columns:
        df["gdelt_log_mentions_all"] = _log1p(df["gdelt_mentions_all"])
        df["gdelt_tone_x_log_mentions_all"] = (
            df["gdelt_tone_all"] * df["gdelt_log_mentions_all"]
        )
        df = _add_lags(df, "gdelt_log_mentions_all",        lags=(1, 3, 7))
        df = _add_lags(df, "gdelt_tone_all",                lags=(1, 3, 7))
        df = _add_lags(df, "gdelt_tone_x_log_mentions_all", lags=(1, 3, 7))
        df = _add_rolling_means(df, "gdelt_log_mentions_all",        windows=(3, 7))
        df = _add_rolling_means(df, "gdelt_tone_all",                windows=(3, 7))
        df = _add_rolling_means(df, "gdelt_tone_x_log_mentions_all", windows=(3, 7))

    # 7. Sort and trim to rows with BTC market cap
    df = df.sort_values("date").reset_index(drop=True)
    btc_col = _pick_col(df, ["btc_mcap", "btc_kapitalizacja_usd"])
    if btc_col:
        df = df.dropna(subset=[btc_col]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 06 — MERGE NEWS WITH MARKET DATA")
    print("=" * 60 + "\n")

    for path in [FEATURES_PARQUET, BASELINE_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run step02 and step04 first."
            )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build_model_dataset()

    df.to_csv(OUT_CSV, index=False)
    print(f"\n  ✓  model_dataset_daily.csv     : {len(df)} rows, {len(df.columns)} columns")

    try:
        df.to_parquet(OUT_PARQUET, index=False)
        print(f"  ✓  model_dataset_daily.parquet : saved")
    except Exception as e:
        print(f"  ⚠  parquet not saved: {e}")

    print(f"\n  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    key_cols = [
        "date", "btc_mcap", "btc_kapitalizacja_usd",
        "baseline_btc_mcap_hat_usd", "resid_btc_mcap_usd",
        "gdelt_mentions_all", "gdelt_tone_all", "gdelt_log_mentions_all",
    ]
    present = [c for c in key_cols if c in df.columns]
    print(f"  Key columns present: {present}")

    # NaN report for media columns
    media_cols = [c for c in df.columns if c.startswith("gdelt_")]
    if media_cols:
        nan_pct = df[media_cols].isna().mean() * 100
        nan_nonzero = nan_pct[nan_pct > 0].round(1)
        if len(nan_nonzero):
            print(f"\n  NaN in media columns:")
            for col, pct in nan_nonzero.items():
                print(f"    {col}: {pct:.1f}%")

    print("\n" + "=" * 60)
    print("Next step: step07_news_effect_model.py\n")


if __name__ == "__main__":
    main()