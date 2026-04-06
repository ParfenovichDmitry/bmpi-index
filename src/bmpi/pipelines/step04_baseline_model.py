# -*- coding: utf-8 -*-
"""
pipelines/step04_baseline_model.py
==================================
BMPI v2 baseline model for BTC daily returns (Q1-ready foundation).

What changed vs BMPI v1:
- We model BTC daily log-returns instead of log-level market cap.
- We compute expected return and abnormal return.
- ETH is excluded from the MAIN baseline to reduce endogeneity.
- We keep all days and forward-fill macro price levels before computing returns.
- Residuals are interpreted as abnormal returns, NOT as pure "media effect".

Input:
  1) data/processed/features_daily.parquet   (preferred, from step02)
  2) data/interim/macro_merged_daily.csv     (fallback)

Expected columns (English or Polish aliases supported):
  date / data
  btc_price / btc_cena_usd
  btc_mcap / btc_kapitalizacja_usd
  btc_logret (optional, from step02)
  btc_vol_30d (optional, from step02)
  btc_regime (optional, from step02)

  nasdaq_close / nasdaq_zamkniecie
  dxy_close / usd_indeks_szeroki
  gold_close / zloto_zamkniecie_usd

Output:
  data/processed/baseline_predictions.csv
  data/processed/granger_results.csv

Key output columns:
  expected_btc_logret
  abnormal_btc_logret
  abnormal_btc_logret_zscore_60d
  abnormal_btc_return_pct
  baseline_btc_mcap_hat_usd
  abnormal_btc_mcap_usd

Notes:
- This step creates the "abnormal movement" layer.
- News attribution is done later in step07.
- Abnormal return != media effect. It is only the unexplained component
  after removing core market controls.

Next step:
  step05_residuals.py  (should later be adapted to event abnormal returns / CAR)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import grangercausalitytests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_INTERIM = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

FEATURES_PARQUET = DATA_PROCESSED / "features_daily.parquet"
MACRO_CSV = DATA_INTERIM / "macro_merged_daily.csv"

OUT_BASELINE = DATA_PROCESSED / "baseline_predictions.csv"
OUT_GRANGER = DATA_PROCESSED / "granger_results.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

START_DATE = "2015-08-07"
MAX_LAG_GRANGER = 14

# Main BMPI v2 baseline:
# BTC return ~ BTC own dynamics + macro returns + volatility/regime
USE_WEEKDAY_DUMMIES = True
USE_MONTH_DUMMIES = False
USE_REGIME_IF_AVAILABLE = True

# Forward-fill macro price levels before computing returns.
# This is safer than dropping all non-trading days.
FFILL_PRICE_LEVELS = True

ROLLING_Z_WINDOW = 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_input_dataframe() -> pd.DataFrame:
    """
    Load input data:
      1) data/processed/features_daily.parquet  — preferred
      2) data/interim/macro_merged_daily.csv    — fallback
    """
    if FEATURES_PARQUET.exists():
        df = pd.read_parquet(FEATURES_PARQUET)
    elif MACRO_CSV.exists():
        df = pd.read_csv(MACRO_CSV)
    else:
        raise FileNotFoundError(
            "Input data not found. Expected one of:\n"
            f"  {FEATURES_PARQUET}\n"
            f"  {MACRO_CSV}"
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "data" in df.columns:
        df["date"] = pd.to_datetime(df["data"], errors="coerce")
    else:
        raise ValueError("Date column not found. Expected 'date' or 'data'.")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _resolve_col(df: pd.DataFrame, english: str, polish: str) -> str:
    """Return whichever column name exists in the DataFrame."""
    if english in df.columns:
        return english
    if polish in df.columns:
        return polish
    raise ValueError(f"Column not found: '{english}' or '{polish}'.")


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_logret_from_level(series: pd.Series) -> pd.Series:
    s = _safe_numeric(series)
    s = s.where(s > 0)
    return np.log(s / s.shift(1))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean_ = series.rolling(window, min_periods=window).mean()
    std_ = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean_) / std_


def _add_lag(df: pd.DataFrame, col: str, lag: int) -> str:
    out_col = f"{col}_lag{lag}"
    df[out_col] = df[col].shift(lag)
    return out_col


def _pick_existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _safe_pct_from_logret(logret: pd.Series) -> pd.Series:
    return (np.exp(logret) - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build BMPI v2 modeling frame.

    Target:
      btc_logret

    Controls:
      - lagged BTC returns
      - lagged BTC volatility (30d if available)
      - NASDAQ return
      - DXY return
      - Gold return
      - weekday dummies
      - month dummies (optional)
      - regime dummy if available

    ETH is intentionally excluded from the MAIN baseline to reduce endogeneity.
    """
    col_btc_price = _resolve_col(df, "btc_price", "btc_cena_usd")
    col_btc_mcap = _resolve_col(df, "btc_mcap", "btc_kapitalizacja_usd")
    col_nasdaq = _resolve_col(df, "nasdaq_close", "nasdaq_zamkniecie")
    col_dxy = _resolve_col(df, "dxy_close", "usd_indeks_szeroki")
    col_gold = _resolve_col(df, "gold_close", "zloto_zamkniecie_usd")

    out = df.copy()
    out = out[out["date"] >= pd.to_datetime(START_DATE)].copy()
    out = out.sort_values("date").reset_index(drop=True)

    # Raw numeric columns
    out["btc_price"] = _safe_numeric(out[col_btc_price])
    out["btc_mcap"] = _safe_numeric(out[col_btc_mcap])
    out["nasdaq_close"] = _safe_numeric(out[col_nasdaq])
    out["dxy_close"] = _safe_numeric(out[col_dxy])
    out["gold_close"] = _safe_numeric(out[col_gold])

    # Forward-fill macro levels across calendar days
    if FFILL_PRICE_LEVELS:
        for c in ["nasdaq_close", "dxy_close", "gold_close"]:
            out[c] = out[c].ffill()

    # BTC return
    if "btc_logret" in out.columns:
        out["btc_logret"] = _safe_numeric(out["btc_logret"])
    else:
        out["btc_logret"] = _safe_logret_from_level(out["btc_price"])

    # Macro returns
    out["nasdaq_logret"] = _safe_logret_from_level(out["nasdaq_close"])
    out["dxy_logret"] = _safe_logret_from_level(out["dxy_close"])
    out["gold_logret"] = _safe_logret_from_level(out["gold_close"])

    # BTC volatility
    if "btc_vol_30d" in out.columns:
        out["btc_vol_30d"] = _safe_numeric(out["btc_vol_30d"])
    else:
        out["btc_vol_30d"] = out["btc_logret"].rolling(30, min_periods=30).std(ddof=0)

    # BTC regime (optional)
    if "btc_regime" in out.columns:
        out["btc_regime"] = _safe_numeric(out["btc_regime"])
    else:
        out["btc_regime"] = np.nan

    # Lagged BTC dynamics
    lagged_cols = []
    lagged_cols.append(_add_lag(out, "btc_logret", 1))
    lagged_cols.append(_add_lag(out, "btc_logret", 2))
    lagged_cols.append(_add_lag(out, "btc_logret", 3))
    lagged_cols.append(_add_lag(out, "btc_vol_30d", 1))

    # Calendar effects
    out["weekday"] = out["date"].dt.weekday
    out["month"] = out["date"].dt.month

    feature_cols = [
        "btc_logret_lag1",
        "btc_logret_lag2",
        "btc_logret_lag3",
        "btc_vol_30d_lag1",
        "nasdaq_logret",
        "dxy_logret",
        "gold_logret",
    ]

    if USE_REGIME_IF_AVAILABLE and out["btc_regime"].notna().any():
        feature_cols.append("btc_regime")

    if USE_WEEKDAY_DUMMIES:
        weekday_dummies = pd.get_dummies(out["weekday"], prefix="wd", drop_first=True, dtype=float)
        out = pd.concat([out, weekday_dummies], axis=1)
        feature_cols.extend(list(weekday_dummies.columns))

    if USE_MONTH_DUMMIES:
        month_dummies = pd.get_dummies(out["month"], prefix="m", drop_first=True, dtype=float)
        out = pd.concat([out, month_dummies], axis=1)
        feature_cols.extend(list(month_dummies.columns))

    # Clean model frame
    required_cols = ["date", "btc_price", "btc_mcap", "btc_logret"] + feature_cols
    out = out[required_cols].copy()

    # Replace inf with NaN
    out = out.replace([np.inf, -np.inf], np.nan)

    # Keep rows only where all regression inputs are available
    out = out.dropna(subset=["btc_logret"] + feature_cols).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------


def fit_baseline_ols(df: pd.DataFrame) -> Tuple[pd.DataFrame, OLS]:
    """
    Fit BMPI v2 baseline:
      btc_logret ~ lagged BTC dynamics + macro returns + optional dummies

    Returns:
      output DataFrame
      fitted statsmodels result object
    """
    feature_cols = [
        c for c in df.columns
        if c not in {"date", "btc_price", "btc_mcap", "btc_logret"}
    ]

    X = add_constant(df[feature_cols], has_constant="add")
    y = df["btc_logret"]

    model = OLS(y, X, missing="drop")
    res = model.fit()

    df_out = df.copy()
    df_out["expected_btc_logret"] = res.predict(X)
    df_out["abnormal_btc_logret"] = df_out["btc_logret"] - df_out["expected_btc_logret"]
    df_out["abnormal_btc_logret_zscore_60d"] = _rolling_zscore(
        df_out["abnormal_btc_logret"], ROLLING_Z_WINDOW
    )

    # Percent interpretation
    df_out["btc_return_pct"] = _safe_pct_from_logret(df_out["btc_logret"])
    df_out["expected_btc_return_pct"] = _safe_pct_from_logret(df_out["expected_btc_logret"])
    df_out["abnormal_btc_return_pct"] = _safe_pct_from_logret(df_out["abnormal_btc_logret"])

    # First-order USD approximation on market cap
    # expected mcap_t ≈ mcap_{t-1} * exp(expected_return_t)
    df_out["btc_mcap_lag1"] = df_out["btc_mcap"].shift(1)
    df_out["baseline_btc_mcap_hat_usd"] = df_out["btc_mcap_lag1"] * np.exp(df_out["expected_btc_logret"])
    df_out["abnormal_btc_mcap_usd"] = df_out["btc_mcap"] - df_out["baseline_btc_mcap_hat_usd"]

    # Model metadata
    df_out["model_type"] = "OLS_returns_baseline_main"
    df_out["target_type"] = "btc_logret"
    df_out["baseline_spec"] = (
        "btc_logret ~ lagged_btc_returns + lagged_btc_volatility + "
        "nasdaq_logret + dxy_logret + gold_logret + calendar_effects"
    )
    df_out["eth_included"] = False
    df_out["ffill_price_levels"] = FFILL_PRICE_LEVELS
    df_out["use_weekday_dummies"] = USE_WEEKDAY_DUMMIES
    df_out["use_month_dummies"] = USE_MONTH_DUMMIES
    df_out["use_regime_if_available"] = USE_REGIME_IF_AVAILABLE
    df_out["r_squared"] = float(res.rsquared)
    df_out["adj_r_squared"] = float(res.rsquared_adj)
    df_out["aic"] = float(res.aic)
    df_out["bic"] = float(res.bic)

    return df_out, res


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------


def run_granger_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Granger causality tests on return series.

    Tests whether each macro return helps predict BTC return.
    """
    candidate_predictors = [
        "nasdaq_logret",
        "dxy_logret",
        "gold_logret",
    ]

    rows: List[Dict[str, float | int | str]] = []

    for predictor in candidate_predictors:
        pair = df[["btc_logret", predictor]].dropna().copy()
        if len(pair) < (MAX_LAG_GRANGER + 30):
            print(f"  [WARN] Not enough rows for Granger test: {predictor}")
            continue

        try:
            # grangercausalitytests expects [target, predictor]
            result = grangercausalitytests(
                pair[["btc_logret", predictor]],
                maxlag=MAX_LAG_GRANGER,
                verbose=False,
            )
        except Exception as exc:
            print(f"  [WARN] Granger failed for {predictor}: {exc}")
            continue

        for lag, lag_result in result.items():
            ssr_ftest = lag_result[0].get("ssr_ftest")
            ssr_chi2test = lag_result[0].get("ssr_chi2test")
            lrtest = lag_result[0].get("lrtest")
            params_ftest = lag_result[0].get("params_ftest")

            rows.append({
                "target": "btc_logret",
                "predictor": predictor,
                "lag": int(lag),
                "ssr_ftest_F": float(ssr_ftest[0]) if ssr_ftest else np.nan,
                "ssr_ftest_pvalue": float(ssr_ftest[1]) if ssr_ftest else np.nan,
                "ssr_chi2_stat": float(ssr_chi2test[0]) if ssr_chi2test else np.nan,
                "ssr_chi2_pvalue": float(ssr_chi2test[1]) if ssr_chi2test else np.nan,
                "lrtest_stat": float(lrtest[0]) if lrtest else np.nan,
                "lrtest_pvalue": float(lrtest[1]) if lrtest else np.nan,
                "params_ftest_F": float(params_ftest[0]) if params_ftest else np.nan,
                "params_ftest_pvalue": float(params_ftest[1]) if params_ftest else np.nan,
                "n_obs": int(len(pair)),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["is_significant_5pct"] = out["ssr_ftest_pvalue"] < 0.05
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 04 — BASELINE MODEL (BMPI v2, RETURNS-BASED)")
    print("=" * 60)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    raw_df = read_input_dataframe()
    print(f"  Loaded raw rows: {len(raw_df)}  "
          f"({raw_df['date'].min().date()} -> {raw_df['date'].max().date()})")

    model_df = prepare_model_frame(raw_df)
    print(f"  Rows after preparation: {len(model_df)}  "
          f"({model_df['date'].min().date()} -> {model_df['date'].max().date()})")

    baseline_df, model_res = fit_baseline_ols(model_df)

    # Save baseline predictions
    save_cols_first = [
        "date",
        "btc_price",
        "btc_mcap",
        "btc_logret",
        "expected_btc_logret",
        "abnormal_btc_logret",
        "abnormal_btc_logret_zscore_60d",
        "btc_return_pct",
        "expected_btc_return_pct",
        "abnormal_btc_return_pct",
        "baseline_btc_mcap_hat_usd",
        "abnormal_btc_mcap_usd",
        "btc_mcap_lag1",
        "model_type",
        "target_type",
        "baseline_spec",
        "eth_included",
        "ffill_price_levels",
        "use_weekday_dummies",
        "use_month_dummies",
        "use_regime_if_available",
        "r_squared",
        "adj_r_squared",
        "aic",
        "bic",
    ]

    feature_cols_rest = [c for c in baseline_df.columns if c not in save_cols_first]
    baseline_df = baseline_df[save_cols_first + feature_cols_rest]
    baseline_df.to_csv(OUT_BASELINE, index=False)

    # Run Granger
    granger_df = run_granger_tests(model_df)
    if not granger_df.empty:
        granger_df.to_csv(OUT_GRANGER, index=False)
    else:
        pd.DataFrame(columns=[
            "target",
            "predictor",
            "lag",
            "ssr_ftest_F",
            "ssr_ftest_pvalue",
            "ssr_chi2_stat",
            "ssr_chi2_pvalue",
            "lrtest_stat",
            "lrtest_pvalue",
            "params_ftest_F",
            "params_ftest_pvalue",
            "n_obs",
            "is_significant_5pct",
        ]).to_csv(OUT_GRANGER, index=False)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Saved baseline: {OUT_BASELINE}")
    print(f"  Saved granger : {OUT_GRANGER}")
    print(f"  Model R²      : {model_res.rsquared:.6f}")
    print(f"  Model Adj R²  : {model_res.rsquared_adj:.6f}")
    print(f"  Model AIC     : {model_res.aic:.2f}")
    print(f"  Model BIC     : {model_res.bic:.2f}")

    if "abnormal_btc_logret" in baseline_df.columns:
        print("  Abnormal return stats:")
        print(f"    mean  = {baseline_df['abnormal_btc_logret'].mean():.8f}")
        print(f"    std   = {baseline_df['abnormal_btc_logret'].std(ddof=0):.8f}")
        print(f"    min   = {baseline_df['abnormal_btc_logret'].min():.8f}")
        print(f"    max   = {baseline_df['abnormal_btc_logret'].max():.8f}")

    print("\nNext step: step05_residuals.py")


if __name__ == "__main__":
    main()