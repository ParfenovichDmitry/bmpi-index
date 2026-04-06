# -*- coding: utf-8 -*-
"""
pipelines/step09_fake_classification.py
=========================================
BMPI v2: compute daily BMPI score and classify excess media effect.

What it does:
  Combines the BMPI v2 media effect output from step07 with the GDELT signal
  to compute a daily BMPI score for each day, then estimates how much of the
  media effect occurred on anomalous / high-pressure days.

Key design choices for BMPI v2:
- Uses predicted_media_effect_usd_oof as the PRIMARY effect column
  (honest out-of-fold estimate).
- Falls back to predicted_media_effect_usd if OOF is unavailable.
- Uses raw local GDELT file for BMPI calibration inputs (mentions + tone).
- Adds BMPI-weighted media effect and share-of-abnormal-move diagnostics.

BMPI score formula:
  z1 = clip[(mentions - mu_M) / sigma_M, -3, 3]
  z2 = clip[(tone     - mu_T) / sigma_T, -3, 3]
  BMPI = sigmoid(0.25·z1 + 0.20·z2)  ∈ (0, 1)

BMPI zones:
  CALM         < 0.470
  NORMAL       0.470–0.530
  ELEVATED     0.530–0.590
  ALERT        0.590–0.650
  MANIPULATION > 0.650

Core metric:
  excess_media_effect_usd = |media_effect_used| × bmpi_score

Input:
  data/processed/news_effect_daily.csv
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv  (preferred)
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_sensitive.csv
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_strong.csv

Output:
  data/processed/excess_media_effect_daily.csv
  data/processed/fake_news_by_source.csv
  data/processed/fake_news_summary.json

Next step:
  step10_robustness_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_GDELT = BASE_DIR / "data" / "raw" / "gdelt"

NEWS_EFFECT_CSV = DATA_PROCESSED / "news_effect_daily.csv"

# Prefer balanced; fallback to others if needed
GDELT_CANDIDATES = [
    DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_balanced.csv",
    DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_sensitive.csv",
    DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_strong.csv",
]

OUT_DAILY = DATA_PROCESSED / "excess_media_effect_daily.csv"
OUT_SOURCE = DATA_PROCESSED / "fake_news_by_source.csv"
OUT_SUMMARY = DATA_PROCESSED / "fake_news_summary.json"

# ---------------------------------------------------------------------------
# Default BMPI calibration parameters
# Used only as fallback if loaded data is insufficient
# ---------------------------------------------------------------------------

MU_M = 379.0
SIGMA_M = 305.8
MU_T = -0.912
SIGMA_T = 0.714
W1 = 0.25
W2 = 0.20

BMPI_ZONES = [
    (0.000, 0.470, "CALM"),
    (0.470, 0.530, "NORMAL"),
    (0.530, 0.590, "ELEVATED"),
    (0.590, 0.650, "ALERT"),
    (0.650, 1.001, "MANIPULATION"),
]

PRIMARY_EFFECT_COL = "predicted_media_effect_usd_oof"
SECONDARY_EFFECT_COL = "predicted_media_effect_usd"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str[:10], errors="coerce").dt.normalize()


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _bmpi_zone(score: float) -> str:
    for lo, hi, label in BMPI_ZONES:
        if lo <= score < hi:
            return label
    return "MANIPULATION"


def _safe_mean(series: pd.Series) -> float:
    s = _safe_num(series).dropna()
    if len(s) == 0:
        return float("nan")
    return float(s.mean())


def _safe_sum(series: pd.Series) -> float:
    s = _safe_num(series).fillna(0.0)
    return float(s.sum())


def _safe_abs_sum(series: pd.Series) -> float:
    s = _safe_num(series).fillna(0.0)
    return float(s.abs().sum())


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    if not NEWS_EFFECT_CSV.exists():
        raise FileNotFoundError(
            f"File not found: {NEWS_EFFECT_CSV}\n"
            "Run step07_news_effect_model.py first."
        )

    gdelt_path = next((p for p in GDELT_CANDIDATES if p.exists()), None)
    if gdelt_path is None:
        raise FileNotFoundError(
            "No GDELT signal file found in data/raw/gdelt/\n"
            "Expected one of:\n"
            "  gdelt_gkg_bitcoin_daily_signal_balanced.csv\n"
            "  gdelt_gkg_bitcoin_daily_signal_sensitive.csv\n"
            "  gdelt_gkg_bitcoin_daily_signal_strong.csv"
        )

    market = _norm_cols(pd.read_csv(NEWS_EFFECT_CSV))
    gdelt = _norm_cols(pd.read_csv(gdelt_path))

    for df_ref in [market, gdelt]:
        date_col = next((c for c in df_ref.columns if c in ("date", "data")), None)
        if not date_col:
            raise ValueError("Date column not found in one of the input files.")
        df_ref["date"] = _to_date(df_ref[date_col])

    # Rename GDELT cols to raw_* to avoid collisions with step07/step06 columns
    rename = {}
    for old, new in [
        ("liczba_wzmianek", "raw_mentions"),
        ("mentions", "raw_mentions"),
        ("mention_count", "raw_mentions"),
        ("count_mentions", "raw_mentions"),
        ("sredni_tone", "raw_tone"),
        ("tone", "raw_tone"),
        ("avg_tone", "raw_tone"),
        ("tone_mean", "raw_tone"),
        ("tone_avg", "raw_tone"),
        ("zrodlo", "raw_source"),
        ("source", "raw_source"),
    ]:
        if old in gdelt.columns:
            rename[old] = new

    if rename:
        gdelt = gdelt.rename(columns=rename)

    return market, gdelt, gdelt_path


# ---------------------------------------------------------------------------
# BMPI computation
# ---------------------------------------------------------------------------


def compute_bmpi(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute BMPI using mentions + tone from raw GDELT merge.

    Calibration is estimated from the loaded data itself for robustness.
    """
    mentions_col = "raw_mentions" if "raw_mentions" in df.columns else None
    tone_col = "raw_tone" if "raw_tone" in df.columns else None

    mentions = (
        _safe_num(df[mentions_col]) if mentions_col
        else pd.Series(np.nan, index=df.index)
    )
    tone = (
        _safe_num(df[tone_col]) if tone_col
        else pd.Series(np.nan, index=df.index)
    )

    valid_mentions = mentions.dropna()
    valid_tone = tone.dropna()

    m_mu = float(valid_mentions.mean()) if len(valid_mentions) > 0 else MU_M
    m_sigma = float(valid_mentions.std(ddof=0)) if len(valid_mentions) > 1 else SIGMA_M
    t_mu = float(valid_tone.mean()) if len(valid_tone) > 0 else MU_T
    t_sigma = float(valid_tone.std(ddof=0)) if len(valid_tone) > 1 else SIGMA_T

    m_sigma = max(m_sigma, 1e-6)
    t_sigma = max(t_sigma, 1e-6)

    # Fill missing as neutral -> z = 0 -> BMPI around 0.5
    mentions_filled = mentions.fillna(m_mu)
    tone_filled = tone.fillna(t_mu)

    z1 = ((mentions_filled - m_mu) / m_sigma).clip(-3, 3)
    z2 = ((tone_filled - t_mu) / t_sigma).clip(-3, 3)
    raw = W1 * z1 + W2 * z2

    out = df.copy()
    out["bmpi_mentions_used"] = mentions_filled
    out["bmpi_tone_used"] = tone_filled
    out["z1_volume"] = z1
    out["z2_tone"] = z2
    out["bmpi_raw"] = raw
    out["bmpi_score"] = _sigmoid(raw.to_numpy(dtype=float))
    out["bmpi_zone"] = out["bmpi_score"].apply(_bmpi_zone)

    calibration = {
        "mu_M": m_mu,
        "sigma_M": m_sigma,
        "mu_T": t_mu,
        "sigma_T": t_sigma,
        "w1": W1,
        "w2": W2,
    }
    return out, calibration


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 09 — BMPI SCORE + EXCESS MEDIA CLASSIFICATION (BMPI v2)")
    print("=" * 60 + "\n")

    market, gdelt, gdelt_path = load_data()
    print(f"  GDELT file: {gdelt_path.name}")

    # Merge: market is anchor, keep all modeled days
    df = market.merge(gdelt, on="date", how="left", suffixes=("", "_gdelt"))
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"  Rows: {len(df)}  ({df['date'].min().date()} -> {df['date'].max().date()})")

    # Choose correct effect column
    if PRIMARY_EFFECT_COL in df.columns:
        effect_col = PRIMARY_EFFECT_COL
        effect_mode = "OOF"
        print("  Using OOF media effect (primary)")
    elif SECONDARY_EFFECT_COL in df.columns:
        effect_col = SECONDARY_EFFECT_COL
        effect_mode = "IN_SAMPLE_FALLBACK"
        print("  Using in-sample media effect (fallback)")
    else:
        raise ValueError(
            "No media effect column found in news_effect_daily.csv.\n"
            f"Expected one of: {PRIMARY_EFFECT_COL}, {SECONDARY_EFFECT_COL}"
        )

    df["media_effect_used"] = _safe_num(df[effect_col]).fillna(0.0)

    # Compute BMPI
    df, calibration = compute_bmpi(df)
    print(
        f"  BMPI calibration: "
        f"mu_M={calibration['mu_M']:.1f}  sigma_M={calibration['sigma_M']:.1f}  "
        f"mu_T={calibration['mu_T']:.4f}  sigma_T={calibration['sigma_T']:.4f}"
    )

    # BMPI-weighted media effect
    df["raw_abs_media_effect_usd"] = df["media_effect_used"].abs()
    df["excess_media_effect_usd"] = df["raw_abs_media_effect_usd"] * df["bmpi_score"]

    # Additional diagnostics
    if "abnormal_btc_mcap_usd" in df.columns:
        abnormal_abs = _safe_num(df["abnormal_btc_mcap_usd"]).abs()
        df["media_share_of_abnormal_move_pct"] = (
            df["excess_media_effect_usd"] / (abnormal_abs + 1e-9) * 100.0
        )

    if "bmip_v2_daily" in df.columns:
        df["bmip_v2_daily_abs_step07"] = _safe_num(df["bmip_v2_daily"]).abs()

    # Save daily output
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DAILY, index=False)

    # Aggregate "source" table
    # Current pipeline uses aggregated GDELT source, so keep explicit wording
    n_manipulation_days = int((df["bmpi_zone"] == "MANIPULATION").sum())
    n_alert_days = int((df["bmpi_zone"] == "ALERT").sum())
    n_elevated_days = int((df["bmpi_zone"] == "ELEVATED").sum())

    source_df = pd.DataFrame({
        "source": ["GDELT_AGGREGATED"],
        "effect_mode": [effect_mode],
        "raw_abs_media_effect_usd": [_safe_abs_sum(df["media_effect_used"])],
        "excess_media_effect_usd": [_safe_sum(df["excess_media_effect_usd"])],
        "avg_bmpi_score": [_safe_mean(df["bmpi_score"])],
        "avg_media_share_of_abnormal_move_pct": [
            _safe_mean(df["media_share_of_abnormal_move_pct"])
            if "media_share_of_abnormal_move_pct" in df.columns else float("nan")
        ],
        "n_days": [int(len(df))],
        "n_manipulation_days": [n_manipulation_days],
        "n_alert_days": [n_alert_days],
        "n_elevated_days": [n_elevated_days],
    })
    source_df.to_csv(OUT_SOURCE, index=False)

    # Summary
    abs_total = _safe_abs_sum(df["media_effect_used"])
    excess_total = _safe_sum(df["excess_media_effect_usd"])
    zone_counts = df["bmpi_zone"].value_counts().to_dict()
    zone_pct = (df["bmpi_zone"].value_counts(normalize=True) * 100.0).round(2).to_dict()

    summary = {
        "effect_mode_used": effect_mode,
        "total_days": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "gdelt_file_used": gdelt_path.name,
        "raw_abs_media_effect_usd": abs_total,
        "excess_media_effect_usd": excess_total,
        "excess_share_pct_of_raw_media_effect": float(100.0 * excess_total / (abs_total + 1e-9)),
        "avg_bmpi_score": _safe_mean(df["bmpi_score"]),
        "avg_media_share_of_abnormal_move_pct": (
            _safe_mean(df["media_share_of_abnormal_move_pct"])
            if "media_share_of_abnormal_move_pct" in df.columns else float("nan")
        ),
        "bmpi_calibration": calibration,
        "bmpi_zone_counts": zone_counts,
        "bmpi_zone_pct": zone_pct,
        "n_manipulation_days": n_manipulation_days,
        "n_alert_days": n_alert_days,
        "n_elevated_days": n_elevated_days,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print results
    print("\n" + "=" * 60)
    print(f"  ✓  excess_media_effect_daily.csv : {len(df)} rows")
    print(f"  ✓  fake_news_by_source.csv")
    print(f"  ✓  fake_news_summary.json")
    print("\n  BMPI zone distribution:")
    for lo, hi, zone in BMPI_ZONES:
        cnt = int(zone_counts.get(zone, 0))
        pct = float(zone_pct.get(zone, 0.0))
        bar = "█" * int(pct / 2)
        print(f"    {zone:<14} {cnt:>5} days ({pct:>6.2f}%)  {bar}")

    print()
    print(f"  Effect mode used:         {effect_mode}")
    print(f"  Total |media_effect|:     ${abs_total:>20,.0f}")
    print(f"  Excess (BMPI-weighted):   ${excess_total:>20,.0f}")
    print(f"  Excess share:             {100.0 * excess_total / (abs_total + 1e-9):.2f}%")
    print(f"  Avg BMPI score:           {summary['avg_bmpi_score']:.4f}")

    if "media_share_of_abnormal_move_pct" in df.columns:
        print(f"  Avg share of abnormal move: {summary['avg_media_share_of_abnormal_move_pct']:.2f}%")

    print("=" * 60)
    print("Next step: step10_robustness_analysis.py\n")


if __name__ == "__main__":
    main()