# -*- coding: utf-8 -*-
"""
pipelines/step09_fake_classification.py
=========================================
Compute daily BMPI score and classify excess media effect.

What it does:
  Combines the Ridge media effect (step07) with the GDELT signal to compute
  a daily BMPI score for each day, then estimates how much of the media
  effect occurred on "anomalous" high-pressure days.

  BMPI score formula (paper Section 3.2):
    z1 = clip[(mentions - mu_M) / sigma_M, -3, 3]
    z2 = clip[(tone     - mu_T) / sigma_T, -3, 3]
    BMPI = sigmoid(0.25·z1 + 0.20·z2)  ∈ (0, 1)

  Calibration parameters (estimated on 2220-day full sample):
    mu_M = 379.0,  sigma_M = 305.8
    mu_T = -0.912, sigma_T = 0.714

  BMPI zones:
    CALM      < 0.470
    NORMAL    0.470–0.530
    ELEVATED  0.530–0.590
    ALERT     0.590–0.650
    MANIPULATION > 0.650

  excess_media_effect_usd = |media_effect_usd| × bmpi_score
  → proxy for USD value of anomalous media-driven BTC movements

Input:  data/processed/news_effect_daily.csv                        (from step07)
        data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv  (primary)

Output: data/processed/excess_media_effect_daily.csv
        data/processed/fake_news_by_source.csv
        data/processed/fake_news_summary.json

Next step: step10_robustness_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_GDELT     = BASE_DIR / "data" / "raw" / "gdelt"

NEWS_EFFECT_CSV = DATA_PROCESSED / "news_effect_daily.csv"

# GDELT signal files — correct filenames from gdelt_btc_downloader.py
GDELT_CANDIDATES = [
    DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_balanced.csv",
    DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_sensitive.csv",
    DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_strong.csv",
]

OUT_DAILY   = DATA_PROCESSED / "excess_media_effect_daily.csv"
OUT_SOURCE  = DATA_PROCESSED / "fake_news_by_source.csv"
OUT_SUMMARY = DATA_PROCESSED / "fake_news_summary.json"

# ---------------------------------------------------------------------------
# BMPI calibration parameters (paper Section 3.2, full 2220-day sample)
# ---------------------------------------------------------------------------

MU_M    = 379.0    # mean mentions
SIGMA_M = 305.8    # std mentions
MU_T    = -0.912   # mean tone
SIGMA_T = 0.714    # std tone
W1      = 0.25     # weight: volume
W2      = 0.20     # weight: tone

BMPI_ZONES = [
    (0.000, 0.470, "CALM"),
    (0.470, 0.530, "NORMAL"),
    (0.530, 0.590, "ELEVATED"),
    (0.590, 0.650, "ALERT"),
    (0.650, 1.001, "MANIPULATION"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str[:10], errors="coerce")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _bmpi_zone(score: float) -> str:
    for lo, hi, label in BMPI_ZONES:
        if lo <= score < hi:
            return label
    return "MANIPULATION"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not NEWS_EFFECT_CSV.exists():
        raise FileNotFoundError(
            f"File not found: {NEWS_EFFECT_CSV}\n"
            "Run step07_news_effect_model.py first."
        )

    gdelt_path = next((p for p in GDELT_CANDIDATES if p.exists()), None)
    if gdelt_path is None:
        raise FileNotFoundError(
            "No GDELT signal file found in data/raw/gdelt/\n"
            "Expected: gdelt_gkg_bitcoin_daily_signal_balanced.csv"
        )
    print(f"  GDELT file: {gdelt_path.name}")

    market = _norm_cols(pd.read_csv(NEWS_EFFECT_CSV))
    gdelt  = _norm_cols(pd.read_csv(gdelt_path))

    for df_ref in [market, gdelt]:
        date_col = next((c for c in df_ref.columns if c in ("date", "data")), None)
        if date_col:
            df_ref["date"] = _to_date(df_ref[date_col])

    # Rename GDELT columns explicitly to avoid collision with step07 columns
    rename = {}
    for old, new in [("liczba_wzmianek","raw_mentions"),
                     ("sredni_tone","raw_tone"),
                     ("mentions","raw_mentions"),
                     ("tone","raw_tone")]:
        if old in gdelt.columns:
            rename[old] = new
    if rename:
        gdelt = gdelt.rename(columns=rename)

    return market, gdelt


# ---------------------------------------------------------------------------
# BMPI computation — paper formula (Section 3.2)
# ---------------------------------------------------------------------------

def compute_bmpi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute BMPI using formula from paper Section 3.2.
    Parameters mu/sigma are estimated from the loaded data itself
    for robustness across different GDELT file versions.
    """
    # Prefer raw GDELT file values over pre-merged columns from step07
    mentions_col = next(
        (c for c in ["raw_mentions", "gdelt_mentions_all",
                     "gdelt_wzmianki_all"] if c in df.columns), None
    )
    tone_col = next(
        (c for c in ["raw_tone", "gdelt_tone_all",
                     "gdelt_ton_all"] if c in df.columns), None
    )

    mentions = (pd.to_numeric(df[mentions_col], errors="coerce")
               if mentions_col else pd.Series(np.nan, index=df.index))
    tone     = (pd.to_numeric(df[tone_col],     errors="coerce")
               if tone_col     else pd.Series(np.nan, index=df.index))

    # Compute mu/sigma from non-NaN values in the loaded data
    # This makes BMPI calibration consistent with the actual GDELT file used
    m_mu    = float(mentions.dropna().mean()) if mentions.notna().any() else MU_M
    m_sigma = float(mentions.dropna().std())  if mentions.notna().any() else SIGMA_M
    t_mu    = float(tone.dropna().mean())     if tone.notna().any()     else MU_T
    t_sigma = float(tone.dropna().std())      if tone.notna().any()     else SIGMA_T
    m_sigma = max(m_sigma, 1e-6)
    t_sigma = max(t_sigma, 1e-6)

    print(f"  BMPI calibration: mu_M={m_mu:.1f} sigma_M={m_sigma:.1f} "
          f"mu_T={t_mu:.4f} sigma_T={t_sigma:.4f}")

    # Fill NaN with mu (neutral → z=0 → BMPI≈0.5)
    mentions = mentions.fillna(m_mu)
    tone     = tone.fillna(t_mu)

    z1 = ((mentions - m_mu) / m_sigma).clip(-3, 3)
    z2 = ((tone     - t_mu) / t_sigma).clip(-3, 3)
    raw = W1 * z1 + W2 * z2

    df["z1_volume"]  = z1
    df["z2_tone"]    = z2
    df["bmpi_raw"]   = raw
    df["bmpi_score"] = _sigmoid(raw.to_numpy())
    df["bmpi_zone"]  = df["bmpi_score"].apply(_bmpi_zone)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 09 — BMPI SCORE + EXCESS MEDIA CLASSIFICATION")
    print("=" * 60 + "\n")

    market, gdelt = load_data()

    # Merge: use LEFT JOIN on date — market is the anchor
    gdelt = gdelt.rename(columns={
        "liczba_wzmianek": "mentions_gdelt",
        "sredni_tone":     "tone_gdelt",
    })
    df = market.merge(gdelt, on="date", how="left", suffixes=("", "_gdelt"))

    # Prefer step07 columns if they already contain GDELT data
    # Otherwise use raw GDELT columns
    # Note: we use raw_mentions/raw_tone from local GDELT file
    # Do NOT use gdelt_mentions_all (from step07, may have different scale)
    # Missing days (no local GDELT match) will be filled with mu_M/mu_T → BMPI≈0.5

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"  Rows: {len(df)}  ({df['date'].min().date()} → {df['date'].max().date()})")

    # Compute BMPI (paper formula)
    df = compute_bmpi(df)

    # excess_media_effect_usd = |media_effect_usd| × bmpi_score
    effect_col = "media_effect_usd" if "media_effect_usd" in df.columns else "news_effect_usd"
    df["excess_media_effect_usd"] = (
        pd.to_numeric(df[effect_col], errors="coerce").fillna(0.0).abs()
        * df["bmpi_score"]
    )

    # Save outputs
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DAILY, index=False)

    source_df = pd.DataFrame({
        "source":                  ["GDELT_AGGREGATED"],
        "excess_media_effect_usd": [float(df["excess_media_effect_usd"].sum())],
        "avg_bmpi_score":          [float(df["bmpi_score"].mean())],
        "n_days":                  [int(len(df))],
        "n_manipulation_days":     [int((df["bmpi_zone"] == "MANIPULATION").sum())],
        "n_alert_days":            [int((df["bmpi_zone"] == "ALERT").sum())],
    })
    source_df.to_csv(OUT_SOURCE, index=False)

    abs_total    = float(df[effect_col].abs().sum())
    excess_total = float(df["excess_media_effect_usd"].sum())
    zone_counts  = df["bmpi_zone"].value_counts().to_dict()
    zone_pct     = (df["bmpi_zone"].value_counts(normalize=True) * 100).round(1).to_dict()

    summary = {
        "total_days":                 int(len(df)),
        "date_min":                   str(df["date"].min()),
        "date_max":                   str(df["date"].max()),
        "excess_media_effect_usd":    excess_total,
        "total_abs_media_effect_usd": abs_total,
        "excess_share_pct":           float(100 * excess_total / (abs_total + 1e-9)),
        "avg_bmpi_score":             float(df["bmpi_score"].mean()),
        "bmpi_calibration": {
            "mu_M": MU_M, "sigma_M": SIGMA_M,
            "mu_T": MU_T, "sigma_T": SIGMA_T,
            "w1": W1, "w2": W2,
        },
        "bmpi_zone_counts": zone_counts,
        "bmpi_zone_pct":    zone_pct,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print results
    print()
    print("=" * 60)
    print(f"  ✓  excess_media_effect_daily.csv : {len(df)} rows")
    print(f"  ✓  fake_news_summary.json")
    print()
    print("  BMPI zone distribution:")
    for zone, _, label in [(z[2], z[0], z[2]) for z in BMPI_ZONES]:
        cnt = zone_counts.get(zone, 0)
        pct = zone_pct.get(zone, 0.0)
        bar = "█" * int(pct / 2)
        print(f"    {zone:<14} {cnt:>5} days ({pct:>5.1f}%)  {bar}")
    print()
    print(f"  Total |media_effect|:   ${abs_total:>20,.0f}")
    print(f"  Excess (BMPI-weighted): ${excess_total:>20,.0f}")
    print(f"  Excess share:            {100*excess_total/(abs_total+1e-9):.1f}%")
    print(f"  Avg BMPI score:          {df['bmpi_score'].mean():.4f}")
    print("=" * 60)
    print("Next step: step10_robustness_analysis.py\n")


if __name__ == "__main__":
    main()