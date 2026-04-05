# -*- coding: utf-8 -*-
"""
config.py
=========
Central configuration for the BMPI project.
(Bitcoin Media Pressure Index — formerly BIMI)

NAMING CONVENTION
-----------------
Variable prefix   Meaning
──────────────────────────────────────────────────────
BMPI_*            Index-level constants and calibration
PRESSURE_*        Zone thresholds and labels
MEDIA_SIGNAL_*    GDELT input file paths
EXCESS_*          Media anomaly effect (formerly "fake_usd")
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

if os.environ.get("BMPI_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["BMPI_PROJECT_ROOT"]).resolve()

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

DATA_DIR:       Path = PROJECT_ROOT / "data"
DATA_RAW:       Path = DATA_DIR / "raw"
DATA_MARKET:    Path = DATA_RAW / "market"
DATA_GDELT:     Path = DATA_RAW / "gdelt"
DATA_NEWS_RAW:  Path = DATA_RAW / "news_raw"
DATA_INTERIM:   Path = DATA_DIR / "interim"
DATA_PROCESSED: Path = DATA_DIR / "processed"
REPORTS_DIR:    Path = PROJECT_ROOT / "reports"
LOGS_DIR:       Path = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR:  Path = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Raw input files — market data
# ---------------------------------------------------------------------------

BTC_RAW:    Path = DATA_MARKET / "btcusdmax.csv"
ETH_RAW:    Path = DATA_MARKET / "ethusdmax.csv"
NASDAQ_RAW: Path = DATA_MARKET / "NASDAQCOM.csv"
DXY_RAW:    Path = DATA_MARKET / "DTWEXBGS.csv"
GOLD_RAW:   Path = DATA_MARKET / "Gold_Futures_Historical_Data.csv"

# ---------------------------------------------------------------------------
# Raw input files — GDELT media signal (four filter quality presets)
#
# Naming: gdelt_btc_media_signal_<preset>.csv
# (renamed from gdelt_gkg_bitcoin_daily_signal_<preset>.csv)
#
# Columns: date, mentions, tone, [preset]
# ---------------------------------------------------------------------------

MEDIA_SIGNAL_ALL:       Path = DATA_GDELT / "gdelt_btc_media_signal_all.csv"
MEDIA_SIGNAL_SENSITIVE: Path = DATA_GDELT / "gdelt_btc_media_signal_sensitive.csv"
MEDIA_SIGNAL_BALANCED:  Path = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_balanced.csv"
MEDIA_SIGNAL_STRONG:    Path = DATA_GDELT / "gdelt_btc_media_signal_strong.csv"

# Legacy filenames (backward compatibility — read-only aliases)
GDELT_ALL:       Path = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_ALL.csv"
GDELT_SENSITIVE: Path = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_sensitive.csv"
GDELT_BALANCED:  Path = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_balanced.csv"
GDELT_STRONG:    Path = DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_strong.csv"

MEDIA_SIGNAL_PRESETS: dict[str, Path] = {
    "all":       MEDIA_SIGNAL_ALL,
    "sensitive": MEDIA_SIGNAL_SENSITIVE,
    "balanced":  MEDIA_SIGNAL_BALANCED,
    "strong":    MEDIA_SIGNAL_STRONG,
}

# Legacy mapping for backward compatibility
GDELT_PRESETS = MEDIA_SIGNAL_PRESETS

# ---------------------------------------------------------------------------
# Processed outputs
# ---------------------------------------------------------------------------

MACRO_MERGED_CSV:       Path = DATA_INTERIM  / "macro_merged_daily.csv"
FEATURES_PARQUET:       Path = DATA_PROCESSED / "features_daily.parquet"
BASELINE_PREDICTIONS:   Path = DATA_PROCESSED / "baseline_predictions.csv"
MODEL_DATASET_CSV:      Path = DATA_PROCESSED / "model_dataset_daily.csv"
MODEL_DATASET_PARQUET:  Path = DATA_PROCESSED / "model_dataset_daily.parquet"
GRANGER_RESULTS:        Path = DATA_PROCESSED / "granger_results.csv"
SOURCE_RANKING:         Path = DATA_PROCESSED / "source_ranking.csv"

# Media effect outputs
# (renamed from fake_news_daily / fake_usd_effect → media_effect_daily / excess_media_effect_usd)
MEDIA_EFFECT_DAILY:     Path = DATA_PROCESSED / "media_effect_daily.csv"
EXCESS_EFFECT_DAILY:    Path = DATA_PROCESSED / "excess_media_effect_daily.csv"

# Backward-compatible aliases
NEWS_EFFECT_DAILY = MEDIA_EFFECT_DAILY
FAKE_NEWS_DAILY   = EXCESS_EFFECT_DAILY

# Peak event files
PEAKS_SENSITIVE: Path = DATA_PROCESSED / "events_peaks_sensitive.csv"
PEAKS_BALANCED:  Path = DATA_PROCESSED / "events_peaks_balanced.csv"
PEAKS_STRONG:    Path = DATA_PROCESSED / "events_peaks_strong.csv"
PEAKS_ALL:       Path = DATA_PROCESSED / "events_peaks_all.csv"

PEAKS_BY_PRESET: dict[str, Path] = {
    "sensitive": PEAKS_SENSITIVE,
    "balanced":  PEAKS_BALANCED,
    "strong":    PEAKS_STRONG,
    "all":       PEAKS_ALL,
}

# ---------------------------------------------------------------------------
# Analysis window & OOS split
# ---------------------------------------------------------------------------

ETH_START_DATE:  str = "2015-08-07"
ANALYSIS_START:  str = "2015-10-31"
ANALYSIS_END:    str = "2026-01-31"

# Train/test split at the 2021/2022 regime boundary.
# Rationale: Terra/LUNA crash (May 2022), FTX collapse (Nov 2022),
#            BTC spot ETF approval (Jan 2024) — three structural breaks
#            that make 2022+ a fundamentally different market regime.
OOS_TRAIN_END:   str = "2021-12-31"
OOS_TEST_START:  str = "2022-01-01"

EVENT_PRE_DAYS:  int = 7
EVENT_POST_DAYS: int = 7
MAX_LAG_GRANGER: int = 14

# ---------------------------------------------------------------------------
# BMPI calibration — two parameter sets (see bmpi_core.py for full docs)
# ---------------------------------------------------------------------------

#: Use for OOS evaluation, production signals, any forward-looking test.
BMPI_CALIB_TRAIN: dict = {
    "mu_mentions":  312.4,
    "sd_mentions":  274.1,
    "mu_tone":     -0.8834,
    "sd_tone":      0.6912,
    "n_days":       1412,
    "period":       "2015-10-31 to 2021-12-31",
    "label":        "TRAIN",
}

#: Use ONLY for descriptive statistics and historical time-series plots.
BMPI_CALIB_FULL: dict = {
    "mu_mentions":  379.0,
    "sd_mentions":  305.8,
    "mu_tone":     -0.9121,
    "sd_tone":      0.7139,
    "n_days":       2220,
    "period":       "2015-10-31 to 2026-01-02",
    "label":        "FULL",
}

# Default — always TRAIN
BMPI_CALIB = BMPI_CALIB_TRAIN

# Signal weights
BMPI_WEIGHTS: dict[str, float] = {
    "w_volume": 0.25,   # S1 — media volume anomaly (mentions z-score)
    "w_tone":   0.20,   # S2 — sentiment anomaly (tone z-score)
}

# ---------------------------------------------------------------------------
# Pressure zone thresholds
# (renamed from BIMI_ZONES with old MANIPULATION label)
# ---------------------------------------------------------------------------

PRESSURE_THRESHOLD_EXTREME: float = 0.650   # top ~5% historical (was 0.65)

PRESSURE_ZONES: list[tuple[float, float, str, str]] = [
    (0.000, 0.470, "MINIMAL",  "Below-average coverage. Organic price discovery."),
    (0.470, 0.530, "BASELINE", "Standard media activity. No anomalous pressure."),
    (0.530, 0.590, "ELEVATED", "Above-average coverage. Price sensitivity increases."),
    (0.590, 0.650, "HIGH",     "Significantly elevated pressure. Reversal risk ~55%."),
    (0.650, 1.001, "EXTREME",  "Top 5% historical. Half-life of excess effect ~2 days."),
]

BMPI_PERCENTILE_BREAKS: list[tuple[float, int]] = [
    (0.425,  5), (0.426, 10), (0.434, 20), (0.438, 25), (0.445, 30),
    (0.461, 40), (0.478, 50), (0.495, 60), (0.516, 70), (0.528, 75),
    (0.548, 80), (0.571, 85), (0.598, 90), (0.648, 95), (0.805, 99),
]

# ---------------------------------------------------------------------------
# GDELT keyword filters
# ---------------------------------------------------------------------------

BTC_KEYWORDS: list[str] = [
    "bitcoin", "btc", "cryptocurrency", "crypto currency",
    "satoshi", "blockchain", "coinbase", "binance",
    "cryptomarket", "digital currency", "digitalcurrency",
]

GKG_THEMES: list[str] = [
    "WEB_BITCOIN", "ECON_CRYPTOCURRENCY", "ECON_DIGITALCURRENCY",
    "BITCOIN", "CRYPTO", "BTC",
]

# ---------------------------------------------------------------------------
# Column name mapping — old (BIMI-era) → new (BMPI-era)
# ---------------------------------------------------------------------------

COLUMN_RENAME_MAP: dict[str, str] = {
    # GDELT signal columns
    "liczba_wzmianek": "mentions",
    "sredni_tone":     "tone",
    "zrodlo":          "preset",

    # Effect columns (removing "fake" terminology)
    "fake_usd_effect":       "excess_media_effect_usd",
    "fake_score":            "bmpi",
    "news_effect_usd":       "media_effect_usd",
    "fake_news_daily":       "excess_media_effect_daily",

    # Index output columns
    "bimi":                  "bmpi",
    "z_mentions":            "z_volume",
    "z1_news":               "z_volume",
    "z2_tone":               "z_tone",
    "zone":                  "pressure_zone",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in (DATA_INTERIM, DATA_PROCESSED, REPORTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def data_file(relative_path: str) -> Path:
    return DATA_DIR / relative_path


def resolve_media_signal(preset: str) -> Path:
    """
    Return the media signal CSV path for a given preset name.
    Falls back to legacy filename if new-style file does not exist.
    """
    new_path    = MEDIA_SIGNAL_PRESETS.get(preset)
    legacy_path = {
        "all":       GDELT_ALL,
        "sensitive": GDELT_SENSITIVE,
        "balanced":  GDELT_BALANCED,
        "strong":    GDELT_STRONG,
    }.get(preset)

    if new_path and new_path.exists():
        return new_path
    if legacy_path and legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(
        f"Media signal file not found for preset '{preset}'.\n"
        f"Tried:\n  {new_path}\n  {legacy_path}\n"
        "Place GDELT signal CSVs in data/raw/gdelt/."
    )

# GDELT download endpoints (added for gdelt_btc_downloader.py)
GDELT_GKG_MASTERLIST: str = "http://data.gdeltproject.org/gkg/masterfilelist.txt"
GDELT_BASE_URL:       str = "http://data.gdeltproject.org/gkg/"
