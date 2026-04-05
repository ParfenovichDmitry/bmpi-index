# -*- coding: utf-8 -*-
"""
bmpi
====
Bitcoin Media Pressure Index — core package.

Quick start
-----------
    from bmpi import compute_bmpi

    result = compute_bmpi(mentions=850, tone=-0.5)
    print(result.bmpi)           # 0.5723
    print(result.zone.label)     # 'ELEVATED'
    print(result.zone.description)
    print(result)                # full formatted output
    print(result.is_extreme)     # False

Batch computation (pandas-friendly):
    from bmpi import compute_bmpi_series
    import pandas as pd

    df = pd.read_csv("data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv")
    df["bmpi"] = compute_bmpi_series(df["mentions"], df["tone"])
"""

from bmpi.bmpi_core import (
    compute_bmpi,
    compute_bmpi_oos,
    compute_bmpi_series,
    get_pressure_zone,
    get_percentile,
    zone_label_series,
    zone_short_label_series,
    BmpiResult,
    PressureZone,
    PRESSURE_ZONES,
    CALIB_TRAIN,
    CALIB_FULL,
    EXTREME_THRESHOLD,
)

__version__  = "2.0.0"
__name_long__ = "Bitcoin Media Pressure Index"
__acronym__   = "BMPI"

# Backward compatibility aliases (for code that imported the old BIMI names)
compute_bimi        = compute_bmpi
compute_bimi_series = compute_bmpi_series
compute_bimi_oos    = compute_bmpi_oos
ZONES               = PRESSURE_ZONES

__all__ = [
    "compute_bmpi",
    "compute_bmpi_oos",
    "compute_bmpi_series",
    "get_pressure_zone",
    "get_percentile",
    "zone_label_series",
    "zone_short_label_series",
    "BmpiResult",
    "PressureZone",
    "PRESSURE_ZONES",
    "CALIB_TRAIN",
    "CALIB_FULL",
    "EXTREME_THRESHOLD",
    # backward compat
    "compute_bimi",
    "compute_bimi_series",
]
