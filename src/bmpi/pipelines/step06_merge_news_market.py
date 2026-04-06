# -*- coding: utf-8 -*-
"""
pipelines/step06_merge_news_market.py
============================================
BMPI v2: build the final modeling dataset by merging market data,
baseline abnormal-return outputs, and GDELT media signals.

What changed vs BMPI v1:
- Keeps ALL days in the merged dataset (no dropping zero-news days).
- Uses BMPI v2 abnormal-return outputs from step04.
- Builds media-pressure features suitable for causal/event modeling.
- Prepares both "all news" and preset-specific news signals when available.
- Optimized to avoid Pandas DataFrame fragmentation warnings by generating
  feature blocks separately and concatenating them in bulk.

Input:
  data/processed/features_daily.parquet                       (from step02)
  data/processed/baseline_predictions.csv                     (from step04)
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv  (required)
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_strong.csv    (optional)
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_sensitive.csv (optional)

Expected GDELT columns (Polish or English aliases supported):
  data / date
  liczba_wzmianek / mentions
  sredni_tone / tone
  zrodlo / source   (optional)

Output:
  data/processed/model_dataset_daily.csv
  data/processed/model_dataset_daily.parquet

Notes:
- Days with no news are preserved with mentions=0.
- Tone for no-news days is encoded as 0.0.
- This file only prepares the modeling panel. Attribution happens in step07.

Next step:
  step07_news_effect_model.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_GDELT = BASE_DIR / "data" / "raw" / "gdelt"

FEATURES_PARQUET = DATA_PROCESSED / "features_daily.parquet"
BASELINE_CSV = DATA_PROCESSED / "baseline_predictions.csv"

GDELT_BALANCED = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_balanced.csv"
GDELT_STRONG = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_strong.csv"
GDELT_SENSITIVE = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_sensitive.csv"

OUT_CSV = DATA_PROCESSED / "model_dataset_daily.csv"
OUT_PARQUET = DATA_PROCESSED / "model_dataset_daily.parquet"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LAG_WINDOWS = (1, 3, 7)
ROLL_WINDOWS = (3, 7, 14, 30)
STD_WINDOWS = (7, 14, 30)
ZSCORE_WINDOW = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_cols(cols) -> List[str]:
    """Normalise column names to lowercase snake_case."""
    out: List[str] = []
    for c in cols:
        c2 = str(c).strip().replace("\ufeff", "").replace(" ", "_")
        c2 = re.sub(r"[^\w_]+", "_", c2)
        c2 = re.sub(r"__+", "_", c2).strip("_")
        out.append(c2.lower())
    return out


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse date column and normalise to date-only."""
    return pd.to_datetime(s.astype(str).str[:10], errors="coerce").dt.normalize()


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_num(series: pd.Series, fill_value: Optional[float] = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if fill_value is not None:
        s = s.fillna(fill_value)
    return s


def _log1p(series: pd.Series) -> pd.Series:
    x = _safe_num(series, fill_value=0.0)
    return pd.Series(np.log1p(x.to_numpy(dtype=float)), index=series.index)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean_ = series.rolling(window, min_periods=window).mean()
    std_ = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean_) / std_


def _safe_negative_tone(series: pd.Series) -> pd.Series:
    """Keep only negative tone intensity as positive magnitude."""
    s = _safe_num(series, fill_value=0.0)
    return (-s).clip(lower=0.0)


def _safe_positive_tone(series: pd.Series) -> pd.Series:
    s = _safe_num(series, fill_value=0.0)
    return s.clip(lower=0.0)


def _concat_feature_blocks(base_df: pd.DataFrame, blocks: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate feature blocks in one shot to avoid DataFrame fragmentation."""
    valid_blocks = [b for b in blocks if b is not None and not b.empty]
    if not valid_blocks:
        return base_df
    out = pd.concat([base_df] + valid_blocks, axis=1)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


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

    df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


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

    df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def load_gdelt(path: Path, suffix: str) -> pd.DataFrame:
    """
    Load a GDELT daily signal file.

    Accepted column aliases:
      date/data/day
      liczba_wzmianek / mentions / mention_count
      sredni_tone / tone / avg_tone / tone_mean
      zrodlo / source (optional)
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df.columns = _normalise_cols(df.columns)

    date_col = _pick_col(df, ["date", "data", "day"])
    if not date_col:
        raise ValueError(f"{path.name}: date column not found")

    mentions_col = _pick_col(
        df,
        ["liczba_wzmianek", "mentions", "mention_count", "count_mentions"],
    )
    tone_col = _pick_col(
        df,
        ["sredni_tone", "tone", "avg_tone", "tone_avg", "tone_mean"],
    )
    source_col = _pick_col(
        df,
        ["zrodlo", "source", "sources", "n_sources", "source_count"],
    )

    out = pd.DataFrame({"date": _parse_dates(df[date_col])})

    out[f"gdelt_mentions{suffix}"] = (
        _safe_num(df[mentions_col], fill_value=0.0)
        if mentions_col else 0.0
    )
    out[f"gdelt_tone{suffix}"] = (
        _safe_num(df[tone_col], fill_value=np.nan)
        if tone_col else np.nan
    )

    if source_col:
        out[f"gdelt_source_count{suffix}"] = _safe_num(df[source_col], fill_value=np.nan)

    out = out.dropna(subset=["date"]).groupby("date", as_index=False).agg("mean")
    out = out.sort_values("date").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_news_features(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Engineer news-pressure features for one news stream suffix, e.g.:
      suffix = "_all", "_balanced", "_strong", "_sensitive"

    Returns a new DataFrame containing ONLY newly created columns.
    Existing columns like gdelt_mentions_* and gdelt_tone_* are NOT re-added.
    """
    mentions_col = f"gdelt_mentions{suffix}"
    tone_col = f"gdelt_tone{suffix}"

    if mentions_col not in df.columns:
        return pd.DataFrame(index=df.index)

    mentions = _safe_num(df[mentions_col], fill_value=0.0)
    if tone_col in df.columns:
        tone = _safe_num(df[tone_col], fill_value=0.0)
        tone = tone.where(mentions > 0, 0.0)
    else:
        tone = pd.Series(0.0, index=df.index)

    log_mentions_col = f"gdelt_log_mentions{suffix}"
    neg_tone_col = f"gdelt_negative_tone{suffix}"
    pos_tone_col = f"gdelt_positive_tone{suffix}"
    interaction_all_col = f"gdelt_tone_x_log_mentions{suffix}"
    bad_news_intensity_col = f"gdelt_bad_news_intensity{suffix}"
    good_news_intensity_col = f"gdelt_good_news_intensity{suffix}"

    mentions_z_col = f"gdelt_log_mentions_z{ZSCORE_WINDOW}{suffix}"
    tone_z_col = f"gdelt_tone_z{ZSCORE_WINDOW}{suffix}"
    bad_news_z_col = f"gdelt_bad_news_intensity_z{ZSCORE_WINDOW}{suffix}"
    extreme_dummy_col = f"gdelt_extreme_attention_dummy{suffix}"
    weekend_dummy_col = f"gdelt_weekend_news_dummy{suffix}"

    feats: Dict[str, pd.Series] = {}

    feats[log_mentions_col] = _log1p(mentions)
    feats[neg_tone_col] = _safe_negative_tone(tone)
    feats[pos_tone_col] = _safe_positive_tone(tone)
    feats[interaction_all_col] = tone * feats[log_mentions_col]
    feats[bad_news_intensity_col] = feats[neg_tone_col] * feats[log_mentions_col]
    feats[good_news_intensity_col] = feats[pos_tone_col] * feats[log_mentions_col]

    feats[mentions_z_col] = _rolling_zscore(feats[log_mentions_col], ZSCORE_WINDOW)
    feats[tone_z_col] = _rolling_zscore(tone, ZSCORE_WINDOW)
    feats[bad_news_z_col] = _rolling_zscore(feats[bad_news_intensity_col], ZSCORE_WINDOW)

    feats[extreme_dummy_col] = (feats[mentions_z_col] >= 2.0).astype(float)
    feats[weekend_dummy_col] = ((df["date"].dt.weekday >= 5) & (mentions > 0)).astype(float)

    base_for_windows = {
        log_mentions_col: feats[log_mentions_col],
        tone_col: tone,
        interaction_all_col: feats[interaction_all_col],
        bad_news_intensity_col: feats[bad_news_intensity_col],
        good_news_intensity_col: feats[good_news_intensity_col],
        mentions_z_col: feats[mentions_z_col],
        tone_z_col: feats[tone_z_col],
        bad_news_z_col: feats[bad_news_z_col],
        mentions_col: mentions,
    }

    for col, series in base_for_windows.items():
        for lag in LAG_WINDOWS:
            feats[f"{col}_lag{lag}"] = series.shift(lag)

    rolling_mean_cols = {
        mentions_col: mentions,
        log_mentions_col: feats[log_mentions_col],
        tone_col: tone,
        bad_news_intensity_col: feats[bad_news_intensity_col],
        good_news_intensity_col: feats[good_news_intensity_col],
    }
    for col, series in rolling_mean_cols.items():
        for w in ROLL_WINDOWS:
            feats[f"{col}_ma{w}"] = series.rolling(window=w, min_periods=1).mean()
            feats[f"{col}_sum{w}"] = series.rolling(window=w, min_periods=1).sum()

    rolling_std_cols = {
        log_mentions_col: feats[log_mentions_col],
        tone_col: tone,
        bad_news_intensity_col: feats[bad_news_intensity_col],
    }
    for col, series in rolling_std_cols.items():
        for w in STD_WINDOWS:
            feats[f"{col}_std{w}"] = series.rolling(
                window=w,
                min_periods=max(2, min(w, 2))
            ).std(ddof=0)

    return pd.DataFrame(feats, index=df.index)


def add_global_calendar_and_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with general helper features only."""
    feats: Dict[str, pd.Series] = {}

    feats["weekday"] = df["date"].dt.weekday
    feats["month"] = df["date"].dt.month
    feats["is_weekend"] = (feats["weekday"] >= 5).astype(float)

    if "abnormal_btc_logret" in df.columns and "btc_vol_30d" in df.columns:
        feats["abnormal_x_volatility"] = (
            _safe_num(df["abnormal_btc_logret"], fill_value=0.0) *
            _safe_num(df["btc_vol_30d"], fill_value=0.0)
        )

    if "abnormal_btc_logret" in df.columns and "btc_regime" in df.columns:
        feats["abnormal_x_regime"] = (
            _safe_num(df["abnormal_btc_logret"], fill_value=0.0) *
            _safe_num(df["btc_regime"], fill_value=0.0)
        )

    if "gdelt_mentions_all" in df.columns:
        mentions_all = df["gdelt_mentions_all"]
        if isinstance(mentions_all, pd.DataFrame):
            mentions_all = mentions_all.iloc[:, 0]
        feats["has_any_news_all"] = (_safe_num(mentions_all, fill_value=0.0) > 0).astype(float)

    if "gdelt_bad_news_intensity_all" in df.columns:
        bad_news_all = df["gdelt_bad_news_intensity_all"]
        if isinstance(bad_news_all, pd.DataFrame):
            bad_news_all = bad_news_all.iloc[:, 0]
        feats["has_bad_news_all"] = (_safe_num(bad_news_all, fill_value=0.0) > 0).astype(float)

    return pd.DataFrame(feats, index=df.index)


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_model_dataset() -> pd.DataFrame:
    # 1. Load market features
    features = load_features(FEATURES_PARQUET)
    print(f"  features_daily:            {len(features)} rows")

    # 2. Load baseline abnormal-return outputs
    baseline = load_baseline(BASELINE_CSV)
    print(f"  baseline_predictions:      {len(baseline)} rows")

    # 3. Load GDELT signals
    gdelt_parts: List[pd.DataFrame] = []
    first_available_label: Optional[str] = None
    first_available_df: Optional[pd.DataFrame] = None

    for path, suffix, label in [
        (GDELT_BALANCED, "_balanced", "balanced"),
        (GDELT_STRONG, "_strong", "strong"),
        (GDELT_SENSITIVE, "_sensitive", "sensitive"),
    ]:
        if not path.exists():
            print(f"  gdelt_{label}:              not found — skipping")
            continue

        g = load_gdelt(path, suffix)
        print(f"  gdelt_{label}:              {len(g)} rows")
        gdelt_parts.append(g)

        if first_available_df is None:
            first_available_df = load_gdelt(path, "_all")
            first_available_label = label

    if first_available_df is None:
        raise FileNotFoundError(
            "No GDELT signal files found in data/raw/gdelt/\n"
            "Expected at least:\n"
            "  gdelt_gkg_bitcoin_daily_signal_balanced.csv"
        )

    print(f"  gdelt_all (from {first_available_label}): {len(first_available_df)} rows")
    gdelt_parts.insert(0, first_available_df)

    # 4. Merge on date
    df = baseline.merge(features, on="date", how="left", suffixes=("", "_feat"))
    for g in gdelt_parts:
        df = df.merge(g, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True).copy()

    # 5. Normalize missing news inputs before feature engineering
    gdelt_mentions_cols = [c for c in df.columns if c.startswith("gdelt_mentions")]
    gdelt_tone_cols = [c for c in df.columns if c.startswith("gdelt_tone") and "_x_" not in c]

    for c in gdelt_mentions_cols:
        df[c] = _safe_num(df[c], fill_value=0.0)
    for c in gdelt_tone_cols:
        df[c] = _safe_num(df[c], fill_value=0.0)

    # 6. Engineer news features as blocks
    feature_blocks: List[pd.DataFrame] = []

    for suffix in ["_all", "_balanced", "_strong", "_sensitive"]:
        mentions_col = f"gdelt_mentions{suffix}"
        if mentions_col in df.columns:
            feature_blocks.append(engineer_news_features(df, suffix))

    df = _concat_feature_blocks(df, feature_blocks)

    # 7. General market/calendar features as one block
    context_block = add_global_calendar_and_market_context(df)
    df = _concat_feature_blocks(df, [context_block])

    # 8. Keep date sorted and de-duplicated
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True).copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 06 — MERGE NEWS + MARKET DATASET (BMPI v2)")
    print("=" * 60 + "\n")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = build_model_dataset()

    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQUET, index=False)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Final rows:                {len(df)}")
    print(f"  Date range:                {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Saved CSV:                 {OUT_CSV}")
    print(f"  Saved Parquet:             {OUT_PARQUET}")

    gdelt_cols = [c for c in df.columns if c.startswith("gdelt_")]
    print(f"  GDELT-derived columns:     {len(gdelt_cols)}")

    if "gdelt_mentions_all" in df.columns:
        print(f"  Days with any news (_all): {int((df['gdelt_mentions_all'] > 0).sum())}")

    if "abnormal_btc_logret" in df.columns:
        print(f"  Rows with abnormal return: {int(df['abnormal_btc_logret'].notna().sum())}")

    print("\nNext step: step07_news_effect_model.py")


if __name__ == "__main__":
    main()