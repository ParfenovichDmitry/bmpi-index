# -*- coding: utf-8 -*-
"""
pipelines/step02_align_and_features.py
========================================
Align the analysis window and compute base features.

Input:  data/interim/macro_merged_daily.csv  (from step01)
Output: data/processed/features_daily.parquet

Features computed:
  btc_logret   — daily log-return of BTC price
  eth_logret   — daily log-return of ETH price
  btc_vol_30d  — 30-day rolling volatility of BTC log-returns
  eth_vol_30d  — 30-day rolling volatility of ETH log-returns
  btc_regime   — market regime: +1 (up day) / -1 (down day)

Analysis window starts from ETH_START_DATE (2015-08-07) so that
ETH data is available for all rows — required by the baseline model.

Note: rolling-window columns (btc_vol_30d, eth_vol_30d) will contain NaN
for the first 29 rows. These are kept intentionally — downstream steps
handle NaN via their own logic. Use dropna(subset=[...]) if needed.

Next step: step03_peak_detection.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_INTERIM   = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

INPUT_FILE  = DATA_INTERIM   / "macro_merged_daily.csv"
OUTPUT_FILE = DATA_PROCESSED / "features_daily.parquet"

# Analysis window: start from ETH listing date so ETH data is present
ETH_START_DATE = "2015-08-07"


def log_return(series: pd.Series) -> pd.Series:
    """Daily log-return: ln(price_t / price_{t-1})."""
    return np.log(series / series.shift(1))


def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 02 — ALIGN AND COMPUTE FEATURES")
    print("=" * 60)

    # 1. Load merged market data (Polish column names from step01)
    df = pd.read_csv(INPUT_FILE, parse_dates=["data"])
    print(f"  Loaded: {len(df)} rows from {INPUT_FILE.name}")

    # 2. Align analysis window — drop rows before ETH was listed
    df = df[df["data"] >= ETH_START_DATE].copy()
    df = df.sort_values("data").reset_index(drop=True)
    print(f"  After ETH filter: {len(df)} rows  "
          f"({df['data'].min().date()} → {df['data'].max().date()})")

    # 3. Log-returns (standard in financial econometrics)
    df["btc_logret"] = log_return(df["btc_cena_usd"])
    df["eth_logret"] = log_return(df["eth_cena_usd"])

    # 4. 30-day rolling volatility (std of log-returns)
    df["btc_vol_30d"] = df["btc_logret"].rolling(30).std()
    df["eth_vol_30d"] = df["eth_logret"].rolling(30).std()

    # 5. Market regime: +1 = up day, -1 = down day
    df["btc_regime"] = np.where(df["btc_logret"] > 0, 1, -1)

    # 6. Also expose English column aliases for downstream compatibility
    df["btc_price"]  = df["btc_cena_usd"]
    df["btc_mcap"]   = df["btc_kapitalizacja_usd"]
    df["date"]       = df["data"]

    # 7. Save as parquet (faster read for downstream steps)
    # NOTE: we do NOT dropna() here — first 29 rows have NaN vol columns
    # which is expected and handled by downstream steps individually.
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n  ✓  features_daily.parquet : {len(df)} rows")
    print(f"  ✓  Columns : {list(df.columns)}")
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols):
        print("\n  NaN counts (expected for rolling columns):")
        for col, n in nan_cols.items():
            print(f"    {col}: {n}")
    print("\n  [OK] Saved:", OUTPUT_FILE)
    print("=" * 60)
    print("Next step: step03_peak_detection.py\n")


if __name__ == "__main__":
    main()