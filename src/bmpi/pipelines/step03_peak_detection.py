# -*- coding: utf-8 -*-
"""
pipelines/step03_peak_detection.py
=====================================
Detect anomalous BTC market-cap peaks using rolling z-score.

Input:  data/processed/features_daily.parquet  (preferred, from step02)
        data/interim/macro_merged_daily.csv     (fallback)
Output: data/processed/events_peaks_strong.csv
        data/processed/events_peaks_balanced.csv
        data/processed/events_peaks_sensitive.csv
        data/processed/events_peaks_all_presets.csv

Algorithm:
  1. Compute daily log-change of BTC market cap
  2. Compute rolling z-score over a configurable window
  3. Flag days where z-score >= threshold AND is a local maximum
  4. Enforce minimum gap between consecutive peaks
  5. Save one CSV per preset + combined file

Three presets control sensitivity:
  strong    — few but robust peaks (large shocks only)
  balanced  — recommended for main analysis
  sensitive — many events (robustness checks)

Note: Polish column names (data_piku, btc_kapitalizacja_usd, etc.) are
preserved in the output files for compatibility with step07–step16.

Next step: step04_baseline_model.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_INTERIM   = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

FEATURES_PARQUET = DATA_PROCESSED / "features_daily.parquet"
MACRO_CSV        = DATA_INTERIM   / "macro_merged_daily.csv"

START_DATE      = "2015-08-07"
EVENT_PRE_DAYS  = 7
EVENT_POST_DAYS = 7

PRESETS = {
    "strong": {
        "ROLLING_WINDOW_DAYS":        90,
        "ZSCORE_THRESHOLD":           3.0,
        "LOCAL_MAX_WINDOW":           7,
        "MIN_GAP_BETWEEN_PEAKS_DAYS": 21,
    },
    "balanced": {
        # Calibrated to reproduce paper result: 29 events (2016–2025)
        # log-return z-score, 30-day rolling window, thr=2.8, min_gap=21 days
        # (Section 3.4: "rolling z-score applied to log returns within 30-day window")
        "ROLLING_WINDOW_DAYS":        30,
        "ZSCORE_THRESHOLD":           2.8,
        "LOCAL_MAX_WINDOW":           5,
        "MIN_GAP_BETWEEN_PEAKS_DAYS": 21,
    },
    "sensitive": {
        "ROLLING_WINDOW_DAYS":        45,
        "ZSCORE_THRESHOLD":           2.0,
        "LOCAL_MAX_WINDOW":           3,
        "MIN_GAP_BETWEEN_PEAKS_DAYS": 7,
    },
}


def read_input_dataframe() -> pd.DataFrame:
    """Load features_daily.parquet (preferred) or macro_merged_daily.csv (fallback)."""
    if FEATURES_PARQUET.exists():
        df = pd.read_parquet(FEATURES_PARQUET)
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
        return df
    if MACRO_CSV.exists():
        return pd.read_csv(MACRO_CSV, parse_dates=["data"])
    raise FileNotFoundError(
        f"Input not found. Expected:\n  {FEATURES_PARQUET}\n  {MACRO_CSV}"
    )


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: z = (x - rolling_mean) / rolling_std"""
    mean = series.rolling(window).mean()
    std  = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def is_local_max(series: pd.Series, idx: int, w: int) -> bool:
    """True if series[idx] is the maximum in the window [idx-w, idx+w]."""
    left   = max(0, idx - w)
    right  = min(len(series) - 1, idx + w)
    center = series.iloc[idx]
    return pd.notna(center) and center == series.iloc[left:right + 1].max()


def deduplicate_by_min_gap(candidates: pd.DataFrame, min_gap_days: int) -> pd.DataFrame:
    """Keep earliest peak when two peaks are within min_gap_days of each other."""
    if candidates.empty:
        return candidates
    candidates = (candidates
                  .sort_values(["peak_date", "zscore"], ascending=[True, False])
                  .reset_index(drop=True))
    kept, last_date = [], None
    for _, row in candidates.iterrows():
        d = row["peak_date"]
        if last_date is None or (d - last_date).days >= min_gap_days:
            kept.append(row)
            last_date = d
    return pd.DataFrame(kept)


def build_peaks_for_preset(df: pd.DataFrame, preset_name: str, params: dict) -> pd.DataFrame:
    """Detect anomalous BTC market-cap peaks for one preset configuration."""
    required_cols = {"data", "btc_kapitalizacja_usd", "btc_cena_usd"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[{preset_name}] Missing columns: {missing}")

    work = (df[df["data"] >= pd.to_datetime(START_DATE)]
            .sort_values("data").reset_index(drop=True).copy())

    # Paper Section 3.4: z-score applied to log PRICE returns (not mcap)
    price = pd.to_numeric(work["btc_cena_usd"], errors="coerce")
    work["log_price_return"] = np.log(price).diff()
    work["zscore"]           = rolling_zscore(work["log_price_return"],
                                              params["ROLLING_WINDOW_DAYS"])

    z = work["zscore"]
    rows = []
    for idx in work.index[z >= params["ZSCORE_THRESHOLD"]].tolist():
        if is_local_max(z, idx, params["LOCAL_MAX_WINDOW"]):
            rows.append({
                "peak_date":             work.loc[idx, "data"],
                "zscore":                float(z.iloc[idx]),
                "btc_kapitalizacja_usd": float(work.loc[idx, "btc_kapitalizacja_usd"]),
                "btc_cena_usd":          float(work.loc[idx, "btc_cena_usd"]),
            })

    if not rows:
        return pd.DataFrame()

    candidates = pd.DataFrame(rows)
    candidates["peak_date"] = pd.to_datetime(candidates["peak_date"])
    peaks = deduplicate_by_min_gap(candidates, params["MIN_GAP_BETWEEN_PEAKS_DAYS"])
    peaks = peaks.sort_values("peak_date").reset_index(drop=True)

    peaks["event_id"]     = [f"{preset_name.upper()}_EVT_{i+1:04d}" for i in range(len(peaks))]
    peaks["window_start"] = peaks["peak_date"] - pd.Timedelta(days=EVENT_PRE_DAYS)
    peaks["window_end"]   = peaks["peak_date"] + pd.Timedelta(days=EVENT_POST_DAYS)

    # Polish column aliases — required by step07, step09, step11
    peaks = peaks.rename(columns={
        "peak_date":    "data_piku",
        "zscore":       "zscore_anomalii",
        "window_start": "okno_start",
        "window_end":   "okno_koniec",
    })

    peaks["preset"]                 = preset_name
    peaks["param_okno_zscore"]      = params["ROLLING_WINDOW_DAYS"]
    peaks["param_prog_zscore"]      = params["ZSCORE_THRESHOLD"]
    peaks["param_lokalne_max_okno"] = params["LOCAL_MAX_WINDOW"]
    peaks["param_min_gap_dni"]      = params["MIN_GAP_BETWEEN_PEAKS_DAYS"]
    peaks["param_okno_event_pre"]   = EVENT_PRE_DAYS
    peaks["param_okno_event_post"]  = EVENT_POST_DAYS

    return peaks[[
        "event_id", "preset",
        "data_piku", "okno_start", "okno_koniec",
        "zscore_anomalii", "btc_kapitalizacja_usd", "btc_cena_usd",
        "param_okno_zscore", "param_prog_zscore",
        "param_lokalne_max_okno", "param_min_gap_dni",
        "param_okno_event_pre", "param_okno_event_post",
    ]].copy()


def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 03 — PEAK DETECTION")
    print("=" * 60)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = read_input_dataframe()
    print(f"  Loaded: {len(df)} rows  ({df['data'].min().date()} → {df['data'].max().date()})")

    all_peaks = []
    for preset_name, params in PRESETS.items():
        peaks    = build_peaks_for_preset(df, preset_name, params)
        out_path = DATA_PROCESSED / f"events_peaks_{preset_name}.csv"
        peaks.to_csv(out_path, index=False)
        n = len(peaks)
        z_mean = peaks["zscore_anomalii"].mean() if n else 0
        print(f"  [{preset_name}] {n} peaks  mean_z={z_mean:.3f}  → {out_path.name}")
        all_peaks.append(peaks)

    combined = pd.concat(all_peaks, ignore_index=True)
    combined_out = DATA_PROCESSED / "events_peaks_all_presets.csv"
    combined.to_csv(combined_out, index=False)
    print(f"  [combined] {len(combined)} rows → {combined_out.name}")

    print("=" * 60)
    print("Next step: step04_baseline_model.py\n")


if __name__ == "__main__":
    main()