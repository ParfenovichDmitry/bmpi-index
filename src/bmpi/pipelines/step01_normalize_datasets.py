# -*- coding: utf-8 -*-
"""
pipelines/step01_normalize_datasets.py
========================================
Normalise and merge market data from 5 raw CSV sources.

What it does:
  1. Reads 5 raw CSV files from data/raw/market/
  2. Normalises all dates to YYYY-MM-DD format
  3. Parses numbers from any format ("4,745.10", "0.57K", "-11.39%")
  4. Merges all series on the BTC date axis (LEFT JOIN)
  5. Saves individual normalised CSVs + combined macro_merged_daily.csv

Polish column names (btc_cena_usd, btc_kapitalizacja_usd, etc.) are preserved
as aliases alongside the English names. This is required for backward
compatibility with step02–step16 which were originally written in Polish.

Input files:
  data/raw/market/btcusdmax.csv                    ← CoinGecko
  data/raw/market/ethusdmax.csv                    ← CoinGecko
  data/raw/market/NASDAQCOM.csv                    ← FRED
  data/raw/market/DTWEXBGS.csv                     ← FRED (Broad Dollar Index / DXY)
  data/raw/market/Gold_Futures_Historical_Data.csv ← Investing.com

Output files:
  data/interim/macro_merged_daily.csv   ← main output, required by step02
  data/interim/btc_normalized.csv
  data/interim/eth_normalized.csv
  data/interim/nasdaq_normalized.csv
  data/interim/dxy_normalized.csv
  data/interim/gold_normalized.csv

Next step: step02_align_and_features.py
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — all derived from __file__, no hardcoded paths
# ---------------------------------------------------------------------------

BASE_DIR     = Path(__file__).resolve().parents[3]
DATA_MARKET  = BASE_DIR / "data" / "raw" / "market"
DATA_INTERIM = BASE_DIR / "data" / "interim"

BTC_PATH    = DATA_MARKET / "btcusdmax.csv"
ETH_PATH    = DATA_MARKET / "ethusdmax.csv"
NASDAQ_PATH = DATA_MARKET / "NASDAQCOM.csv"
DXY_PATH    = DATA_MARKET / "DTWEXBGS.csv"
GOLD_PATH   = DATA_MARKET / "Gold_Futures_Historical_Data.csv"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# False = keep NaN on weekends/holidays (academic standard — do not impute)
# True  = forward-fill missing values (for trading strategies)
FFILL_CONTROLS = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Number parsers
# ---------------------------------------------------------------------------

def parse_float_mixed(x) -> float:
    """
    Universal number parser:
      "4,745.10"  → 4745.10
      1575032004  → 1575032004.0
      ""  / None  → NaN
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(" ", "").replace(",", "").replace('"', "")
    if not s or s.lower() in {"nan", "null", "none", "-"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_percent(x) -> float:
    """"-11.39%" → -11.39"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "").replace('"', "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_volume_abbrev(x) -> float:
    """
    Parse abbreviated volume strings from Investing.com / MarketWatch:
      "0.57K"  → 570.0
      "146.84K"→ 146840.0
      "1.23M"  → 1_230_000.0
      "2.1B"   → 2_100_000_000.0
      "-" / "" → NaN
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace('"', "").replace(" ", "").replace(",", "")
    if s in {"", "-", "null", "None", "nan"}:
        return np.nan
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)([KMBkmb])?", s)
    if not m:
        return np.nan
    value = float(m.group(1))
    suffix = (m.group(2) or "").upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(suffix, 1.0)
    return value * mult


def to_date(series: pd.Series) -> pd.Series:
    """Strip ' UTC' suffix and normalise to date (time part set to 00:00:00)."""
    s = series.astype(str).str.replace(" UTC", "", regex=False).str.strip()
    return pd.to_datetime(s, errors="coerce").dt.normalize()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_coingecko(path: Path, prefix: str) -> pd.DataFrame:
    """
    CoinGecko CSV: snapped_at, price, market_cap, total_volume
    Output: date, {prefix}_price, {prefix}_mcap, {prefix}_volume
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)

    missing = {"snapped_at", "price", "market_cap", "total_volume"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    df["date"]             = to_date(df["snapped_at"])
    df[f"{prefix}_price"]  = df["price"].apply(parse_float_mixed)
    df[f"{prefix}_mcap"]   = df["market_cap"].apply(parse_float_mixed)
    df[f"{prefix}_volume"] = df["total_volume"].apply(parse_float_mixed)

    out = (df[["date", f"{prefix}_price", f"{prefix}_mcap", f"{prefix}_volume"]]
           .dropna(subset=["date"])
           .drop_duplicates("date")
           .sort_values("date")
           .reset_index(drop=True))
    logger.info("%-10s %d rows  %s → %s",
                prefix.upper(), len(out),
                out["date"].min().date(), out["date"].max().date())
    return out


def load_fred(path: Path, fred_col: str, out_col: str) -> pd.DataFrame:
    """
    FRED CSV: observation_date, <SERIES>
    Output: date, out_col
    Missing values (".") expected on weekends and public holidays.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)

    missing = {"observation_date", fred_col} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    df["date"]  = pd.to_datetime(df["observation_date"], errors="coerce").dt.normalize()
    df[out_col] = pd.to_numeric(df[fred_col].apply(parse_float_mixed), errors="coerce")

    out = (df[["date", out_col]]
           .dropna(subset=["date"])
           .drop_duplicates("date")
           .sort_values("date")
           .reset_index(drop=True))
    logger.info("%-10s %d rows  %s → %s",
                out_col.upper(), len(out),
                out["date"].min().date(), out["date"].max().date())
    return out


def load_gold(path: Path) -> pd.DataFrame:
    """
    Gold Futures CSV (Investing.com): Date, Price, Open, High, Low, Vol., Change %
    Supported date formats: MM/DD/YYYY, DD.MM.YYYY, YYYY-MM-DD, DD/MM/YYYY
    Output: date, gold_close, gold_volume, gold_chg_pct
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"').strip() for c in df.columns]

    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(f"{path.name}: expected columns 'Date' and 'Price'")

    date_series = None
    for fmt in ("%m/%d/%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            date_series = pd.to_datetime(
                df["Date"].astype(str).str.strip(), format=fmt, errors="raise"
            ).dt.normalize()
            break
        except Exception:
            pass
    if date_series is None:
        date_series = pd.to_datetime(df["Date"].astype(str), errors="coerce").dt.normalize()

    df["date"]         = date_series
    df["gold_close"]   = df["Price"].apply(parse_float_mixed)
    df["gold_volume"]  = df["Vol."].apply(parse_volume_abbrev) if "Vol." in df.columns else np.nan
    df["gold_chg_pct"] = df["Change %"].apply(parse_percent) if "Change %" in df.columns else np.nan

    out = (df[["date", "gold_close", "gold_volume", "gold_chg_pct"]]
           .dropna(subset=["date"])
           .drop_duplicates("date")
           .sort_values("date")
           .reset_index(drop=True))
    logger.info("%-10s %d rows  %s → %s",
                "GOLD", len(out),
                out["date"].min().date(), out["date"].max().date())
    return out


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_on_btc_dates(btc, eth, nasdaq, dxy, gold) -> pd.DataFrame:
    """
    Time axis = BTC dates (BTC trades every day including weekends).
    All other series joined LEFT — missing values stay NaN for weekends/holidays.
    """
    m = btc.merge(eth,    on="date", how="left")
    m = m.merge(nasdaq,   on="date", how="left")
    m = m.merge(dxy,      on="date", how="left")
    m = m.merge(gold,     on="date", how="left")
    m = m.sort_values("date").reset_index(drop=True)

    if FFILL_CONTROLS:
        for col in ["nasdaq_close", "dxy_close", "gold_close", "gold_volume"]:
            if col in m.columns:
                m[col] = m[col].ffill()

    return m


# ---------------------------------------------------------------------------
# Polish column aliases — required for backward compatibility with step02–step16
# ---------------------------------------------------------------------------

POLISH_ALIASES = {
    "btc_price":    "btc_cena_usd",
    "btc_mcap":     "btc_kapitalizacja_usd",
    "btc_volume":   "btc_wolumen_usd",
    "eth_price":    "eth_cena_usd",
    "eth_mcap":     "eth_kapitalizacja_usd",
    "eth_volume":   "eth_wolumen_usd",
    "nasdaq_close": "nasdaq_zamkniecie",
    "dxy_close":    "usd_indeks_szeroki",
    "gold_close":   "zloto_zamkniecie_usd",
    "gold_volume":  "zloto_wolumen",
    "date":         "data",
}


def add_polish_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Add Polish duplicate columns for compatibility with downstream steps."""
    for eng, pol in POLISH_ALIASES.items():
        if eng in df.columns and pol not in df.columns:
            df[pol] = df[eng]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 01 — NORMALIZE MARKET DATASETS")
    print("=" * 60)
    print(f"  Source dir : {DATA_MARKET}")
    print(f"  Output dir : {DATA_INTERIM}\n")

    btc    = load_coingecko(BTC_PATH,  "btc")
    eth    = load_coingecko(ETH_PATH,  "eth")
    nasdaq = load_fred(NASDAQ_PATH, fred_col="NASDAQCOM", out_col="nasdaq_close")
    dxy    = load_fred(DXY_PATH,    fred_col="DTWEXBGS",  out_col="dxy_close")
    gold   = load_gold(GOLD_PATH)

    for name, df in [("btc", btc), ("eth", eth), ("nasdaq", nasdaq),
                     ("dxy", dxy), ("gold", gold)]:
        out = DATA_INTERIM / f"{name}_normalized.csv"
        df.to_csv(out, index=False)
        logger.info("Saved: %s", out.name)

    merged = merge_on_btc_dates(btc, eth, nasdaq, dxy, gold)
    merged = add_polish_aliases(merged)

    out_merged = DATA_INTERIM / "macro_merged_daily.csv"
    merged.to_csv(out_merged, index=False)
    logger.info("Saved: %s", out_merged.name)

    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    for label, df in [("BTC", btc), ("ETH", eth), ("NASDAQ", nasdaq),
                      ("DXY", dxy), ("Gold", gold)]:
        print(f"  {label:<8} {df['date'].min().date()} → {df['date'].max().date()}"
              f"  ({len(df)} days)")
    print()
    print(f"  ✓  macro_merged_daily.csv : {len(merged)} rows"
          f"  ({merged['date'].min().date()} → {merged['date'].max().date()})")

    nan_pct  = merged.isna().mean() * 100
    nan_cols = nan_pct[nan_pct > 0].round(1)
    if len(nan_cols):
        print("\n  NaN by column (expected — weekends/holidays):")
        for col, pct in nan_cols.items():
            print(f"    {col}: {pct:.1f}%")
    print("=" * 60 + "\n")
    print("Next step: step02_align_and_features.py")

    return merged


def main() -> None:
    run()


if __name__ == "__main__":
    main()