# -*- coding: utf-8 -*-
"""
pipelines/step04_baseline_model.py
=====================================
Baseline market model for BTC market cap (without news).

Input:  data/processed/features_daily.parquet  (preferred, from step02)
        data/interim/macro_merged_daily.csv     (fallback)

Output: data/processed/baseline_predictions.csv
        data/processed/granger_results.csv

Model: SARIMAX AR(1) with exogenous controls
  Target:   log(btc_mcap)
  Controls: log(eth_mcap), log(nasdaq), log(dxy), log(gold)

The residual (actual - predicted) represents the component of BTC market cap
not explained by macro/market fundamentals — this is the "media component"
used in all downstream steps (step05, step08, step09, step20).

Granger causality tests check whether control series predict changes in BTC.
Note: Granger causality != true causality — it is predictive direction only.

Next step: step05_residuals.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_INTERIM   = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

FEATURES_PARQUET = DATA_PROCESSED / "features_daily.parquet"
MACRO_CSV        = DATA_INTERIM   / "macro_merged_daily.csv"

OUT_BASELINE = DATA_PROCESSED / "baseline_predictions.csv"
OUT_GRANGER  = DATA_PROCESSED / "granger_results.csv"

START_DATE      = "2015-08-07"
MAX_LAG_GRANGER = 14
FFILL_CONTROLS  = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_input_dataframe() -> pd.DataFrame:
    """
    Load input data:
      1) data/processed/features_daily.parquet  — preferred (output of step02)
      2) data/interim/macro_merged_daily.csv    — fallback

    Expects columns (English names from step01, with Polish aliases):
      date / data,
      btc_mcap / btc_kapitalizacja_usd,
      eth_mcap / eth_kapitalizacja_usd,
      nasdaq_close / nasdaq_zamkniecie,
      dxy_close / usd_indeks_szeroki,
      gold_close / zloto_zamkniecie_usd
    """
    if FEATURES_PARQUET.exists():
        df = pd.read_parquet(FEATURES_PARQUET)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "data" in df.columns:
            df["date"] = pd.to_datetime(df["data"], errors="coerce")
        else:
            raise ValueError("Date column not found in features_daily.parquet")
        return df

    if MACRO_CSV.exists():
        df = pd.read_csv(MACRO_CSV)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "data" in df.columns:
            df["date"] = pd.to_datetime(df["data"], errors="coerce")
        return df

    raise FileNotFoundError(
        "Input data not found. Expected one of:\n"
        f"  {FEATURES_PARQUET}\n"
        f"  {MACRO_CSV}"
    )


def _resolve_col(df: pd.DataFrame, english: str, polish: str) -> str:
    """Return whichever column name (English or Polish) is present in df."""
    if english in df.columns:
        return english
    if polish in df.columns:
        return polish
    raise ValueError(f"Column not found: '{english}' or '{polish}'.")


def safe_log(series: pd.Series) -> pd.Series:
    """Safe logarithm: values <= 0 become NaN."""
    s = pd.to_numeric(series, errors="coerce")
    return np.log(s.where(s > 0))


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to analysis window, build log-levels, optionally forward-fill controls.
    Supports both English and Polish column names.
    """
    col_date   = "date"  if "date"   in df.columns else "data"
    col_btc    = _resolve_col(df, "btc_mcap",    "btc_kapitalizacja_usd")
    col_eth    = _resolve_col(df, "eth_mcap",    "eth_kapitalizacja_usd")
    col_nasdaq = _resolve_col(df, "nasdaq_close","nasdaq_zamkniecie")
    col_dxy    = _resolve_col(df, "dxy_close",   "usd_indeks_szeroki")
    col_gold   = _resolve_col(df, "gold_close",  "zloto_zamkniecie_usd")

    out = df.copy()
    out = out.sort_values(col_date).reset_index(drop=True)
    out = out[out[col_date] >= pd.to_datetime(START_DATE)].copy()

    out["log_btc_mcap"] = safe_log(out[col_btc])
    out["log_eth_mcap"] = safe_log(out[col_eth])
    out["log_nasdaq"]   = safe_log(out[col_nasdaq])
    out["log_usd_idx"]  = safe_log(out[col_dxy])
    out["log_gold"]     = safe_log(out[col_gold])

    if FFILL_CONTROLS:
        for col in ["log_nasdaq", "log_usd_idx", "log_gold"]:
            out[col] = out[col].ffill()

    out = out.dropna(subset=["log_btc_mcap", "log_eth_mcap",
                              "log_nasdaq",  "log_usd_idx", "log_gold"])

    if col_date == "data":
        out = out.rename(columns={"data": "date"})

    # Set DatetimeIndex — required by SARIMAX to suppress ValueWarning
    # about unsupported index. Does not affect residual calculation.
    out = out.set_index(pd.DatetimeIndex(out["date"]))

    return out


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------

def fit_baseline_sarimax(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit SARIMAX AR(1) baseline model.
      Target:   log_btc_mcap
      Exog:     log_eth_mcap, log_nasdaq, log_usd_idx, log_gold

    Returns DataFrame with fitted values, residuals, and model metadata.
    """
    import warnings
    # Suppress SARIMAX date-frequency warning — we use daily data with gaps
    # (weekends/holidays) so no fixed frequency can be assigned. This does not
    # affect model fitting or residual calculation — only forecasting is limited.
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=".*frequency information.*")

    y    = df["log_btc_mcap"]
    exog = df[["log_eth_mcap", "log_nasdaq", "log_usd_idx", "log_gold"]]

    model = SARIMAX(
        endog=y,
        exog=exog,
        order=(1, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    fitted   = res.fittedvalues
    residual = y - fitted

    mcap_col = "btc_mcap" if "btc_mcap" in df.columns else "btc_kapitalizacja_usd"

    out = df[["date", mcap_col, "log_btc_mcap", "log_eth_mcap",
              "log_nasdaq", "log_usd_idx", "log_gold"]].copy()
    out = out.rename(columns={mcap_col: "btc_mcap"})

    out["baseline_log_btc_mcap_hat"] = fitted
    out["resid_log_btc_mcap"]        = residual
    out["baseline_btc_mcap_hat_usd"] = np.exp(out["baseline_log_btc_mcap_hat"])
    out["resid_btc_mcap_usd"]        = out["btc_mcap"] - out["baseline_btc_mcap_hat_usd"]
    out["resid_btc_mcap_pct"]        = 100.0 * (
        out["resid_btc_mcap_usd"] / out["baseline_btc_mcap_hat_usd"]
    )
    out["resid_log_zscore_60d"] = (
        (out["resid_log_btc_mcap"] - out["resid_log_btc_mcap"].rolling(60).mean())
        / out["resid_log_btc_mcap"].rolling(60).std(ddof=0)
    )
    out["model_type"]        = "SARIMAX_AR1_exog"
    out["model_param_order"] = "(1,0,0)"
    out["model_param_trend"] = "c"
    out["model_param_ffill"] = FFILL_CONTROLS

    print(f"  SARIMAX fitted | AIC: {res.aic:.2f} | BIC: {res.bic:.2f}")
    return out


# ---------------------------------------------------------------------------
# Granger causality tests
# ---------------------------------------------------------------------------

def run_granger_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether each control series Granger-predicts changes in BTC market cap.
    IMPORTANT: Granger causality != structural causality.
    """
    tmp = df[["date", "log_btc_mcap", "log_eth_mcap",
              "log_nasdaq", "log_usd_idx", "log_gold"]].copy()

    tmp["dlog_btc_mcap"] = tmp["log_btc_mcap"].diff()
    tmp["dlog_eth_mcap"] = tmp["log_eth_mcap"].diff()
    tmp["dlog_nasdaq"]   = tmp["log_nasdaq"].diff()
    tmp["dlog_usd_idx"]  = tmp["log_usd_idx"].diff()
    tmp["dlog_gold"]     = tmp["log_gold"].diff()
    tmp = tmp.dropna()

    tests = [
        ("dlog_eth_mcap", "ETH"),
        ("dlog_nasdaq",   "NASDAQ"),
        ("dlog_usd_idx",  "USD_INDEX"),
        ("dlog_gold",     "GOLD"),
    ]

    rows = []
    for x_col, label in tests:
        result = grangercausalitytests(
            tmp[["dlog_btc_mcap", x_col]].astype(float),
            maxlag=MAX_LAG_GRANGER
        )
        for lag, r in result.items():
            f_stat, p_value, df_denom, df_num = r[0]["ssr_ftest"]
            rows.append({
                "variable_x":   label,
                "column_x":     x_col,
                "lag":          int(lag),
                "f_stat":       float(f_stat),
                "p_value":      float(p_value),
                "df_denom":     float(df_denom),
                "df_num":       float(df_num),
                "note": "Granger test: does X improve prediction of dlog(BTC mcap)?",
            })

    return (pd.DataFrame(rows)
            .sort_values(["variable_x", "lag"])
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 04 — BASELINE MODEL (SARIMAX AR1 + EXOG)")
    print("=" * 60)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df_raw = read_input_dataframe()
    df     = prepare_model_frame(df_raw)
    print(f"  Rows after filter: {len(df)}  "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")

    # 1. Baseline model
    baseline_df = fit_baseline_sarimax(df)
    baseline_df.to_csv(OUT_BASELINE, index=False)
    print(f"  Saved: {OUT_BASELINE.name}")

    # 2. Granger causality tests
    print("  Running Granger causality tests...")
    granger_df = run_granger_tests(df)
    granger_df.to_csv(OUT_GRANGER, index=False)
    print(f"  Saved: {OUT_GRANGER.name}")

    print("=" * 60)
    print("Next step: step05_residuals.py\n")


if __name__ == "__main__":
    main()