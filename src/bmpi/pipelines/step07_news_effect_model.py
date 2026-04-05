# -*- coding: utf-8 -*-
"""
pipelines/step07_news_effect_model.py
========================================
Ridge regression: quantify the media effect on BTC market cap residuals.

What it does:
  Uses resid_btc_mcap_usd from the SARIMAX baseline (step04) as the target.
  Fits Ridge regression with GDELT media features to estimate the media
  component of that residual on each day.

  The output 'media_effect_usd' (= news_effect_usd) is used in step08–step16
  to answer: "How many USD of BTC market cap movement per peak event
  are attributable to media pressure?"

  Note: per-day R² is low (~0.01) by design — media explains a small fraction
  of total daily residual noise. The meaningful metric is the window-level
  ratio: sum_abs(media_effect) / sum_abs(resid) over ±30 day event windows
  (paper: ~26.2% on average across 29 balanced-preset events).

Input:  data/processed/model_dataset_daily.parquet  (from step06)

Output: data/processed/news_effect_daily.csv
        data/processed/news_effect_summary.json
        data/processed/news_effect_coefficients.csv

Next step: step08_event_level_impact.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

MODEL_DATASET    = DATA_PROCESSED / "model_dataset_daily.parquet"
OUT_PREDICTIONS  = DATA_PROCESSED / "news_effect_daily.csv"
OUT_SUMMARY      = DATA_PROCESSED / "news_effect_summary.json"
OUT_COEFFICIENTS = DATA_PROCESSED / "news_effect_coefficients.csv"

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

# Target: log-space residual from SARIMAX baseline (step04)
# Fitting in log space gives correct scale for media_effect
# media_effect_usd = media_effect_log * btc_mcap (first-order approximation)
TARGET_COL = "resid_log_btc_mcap"

CONTROL_COLS_CANDIDATES = [
    "eth_mcap", "nasdaq_close", "dxy_close", "gold_close",
    "log_eth_mcap", "log_nasdaq", "log_usd_idx", "log_gold",
    "eth_kapitalizacja_usd", "nasdaq_zamkniecie",
    "usd_indeks_szeroki", "zloto_zamkniecie_usd",
]

NEWS_COLS_CANDIDATES = [
    "gdelt_mentions_all",
    "gdelt_tone_all",
    "gdelt_log_mentions_all",
    "gdelt_tone_x_log_mentions_all",
    "gdelt_log_mentions_all_lag1",
    "gdelt_log_mentions_all_lag3",
    "gdelt_log_mentions_all_lag7",
    "gdelt_tone_all_lag1",
    "gdelt_tone_all_lag3",
    "gdelt_tone_all_lag7",
    "gdelt_tone_x_log_mentions_all_lag1",
    "gdelt_tone_x_log_mentions_all_lag3",
    "gdelt_tone_x_log_mentions_all_lag7",
    "gdelt_log_mentions_all_ma3",
    "gdelt_log_mentions_all_ma7",
    "gdelt_tone_all_ma3",
    "gdelt_tone_all_ma7",
    "gdelt_tone_x_log_mentions_all_ma3",
    "gdelt_tone_x_log_mentions_all_ma7",
]

# alpha=1.0 selected via cross-validation (paper Section 3.3)
RIDGE_ALPHA = 1.0
N_SPLITS    = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def time_series_cv(X: pd.DataFrame, y: pd.Series,
                   model: Pipeline, n_splits: int) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        m = compute_metrics(y.iloc[test_idx].to_numpy(), pred)
        maes.append(m["MAE"])
        rmses.append(m["RMSE"])
        r2s.append(m["R2"])
    return {
        "cv_splits": n_splits,
        "MAE_mean":  float(np.mean(maes)),
        "RMSE_mean": float(np.mean(rmses)),
        "R2_mean":   float(np.mean(r2s)),
        "MAE_std":   float(np.std(maes)),
        "RMSE_std":  float(np.std(rmses)),
        "R2_std":    float(np.std(r2s)),
    }


def compute_media_effect(model: Pipeline, X: pd.DataFrame,
                         news_cols: List[str]) -> np.ndarray:
    """
    Isolate the news-feature contribution to the Ridge prediction.
    media_effect_usd = sum(beta_j * z_j) for j in news_cols only.
    Units: same as TARGET_COL (USD).
    """
    scaler: StandardScaler = model.named_steps["scaler"]
    reg:    Ridge           = model.named_steps["ridge"]
    X_scaled  = scaler.transform(X)
    col_names = list(X.columns)
    news_idx  = [col_names.index(c) for c in news_cols if c in col_names]
    if not news_idx:
        return np.zeros(len(X), dtype=float)
    return (X_scaled[:, news_idx] @ reg.coef_[news_idx]).astype(float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 07 — NEWS EFFECT MODEL (Ridge regression)")
    print("=" * 60 + "\n")

    if not MODEL_DATASET.exists():
        raise FileNotFoundError(
            f"File not found: {MODEL_DATASET}\n"
            "Run step06_merge_news_market.py first."
        )

    df = pd.read_parquet(MODEL_DATASET).copy()
    date_col = "date" if "date" in df.columns else "data"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", TARGET_COL]).sort_values("date").reset_index(drop=True)

    control_cols = pick_cols(df, CONTROL_COLS_CANDIDATES)
    news_cols    = pick_cols(df, NEWS_COLS_CANDIDATES)

    if not news_cols:
        raise ValueError("No GDELT news columns found. Check model_dataset_daily.")

    # news-only features — controls already removed by SARIMAX in step04
    feature_cols = news_cols
    print(f"  Rows:              {len(df)}")
    print(f"  Control features:  {len(control_cols)} (removed by step04 SARIMAX)")
    print(f"  News features:     {len(news_cols)}")
    print(f"  Ridge alpha:       {RIDGE_ALPHA}")

    # Prepare features
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0.0)
    df[TARGET_COL]   = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Filter to rows where GDELT signal exists
    mentions_col = next(
        (c for c in ["gdelt_mentions_all"] if c in df.columns), None
    )
    if mentions_col:
        mask = df[mentions_col] > 0
        n_before = len(df)
        df = df[mask].reset_index(drop=True)
        print(f"  Rows with GDELT:   {len(df)} / {n_before}")

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Build and cross-validate model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=RIDGE_ALPHA, random_state=42)),
    ])
    print("\n  Running TimeSeriesSplit CV...")
    cv = time_series_cv(X, y, model, N_SPLITS)
    print(f"  CV R²: {cv['R2_mean']:.4f} ± {cv['R2_std']:.4f}")

    # Fit on full sample
    model.fit(X, y)
    y_hat            = model.predict(X)
    media_effect_usd = compute_media_effect(model, X, news_cols)

    in_sample = compute_metrics(y.to_numpy(), y_hat)  # in log space

    tone_col = next((c for c in ["gdelt_tone_all"] if c in df.columns), None)

    # Convert log-space media effect to USD
    # media_effect_log * btc_mcap ≈ USD impact (first-order approximation)
    mcap_col = next(
        (c for c in ["btc_mcap", "btc_kapitalizacja_usd"] if c in df.columns), None
    )
    btc_mcap_arr = (
        pd.to_numeric(df[mcap_col], errors="coerce").fillna(1.0).to_numpy()
        if mcap_col else np.ones(len(df))
    )
    media_effect_usd_final = media_effect_usd * btc_mcap_arr

    # Also get resid_btc_mcap_usd for step08 (already in USD from step04)
    resid_usd_col = next(
        (c for c in ["resid_btc_mcap_usd"] if c in df.columns), None
    )
    resid_usd_arr = (
        pd.to_numeric(df[resid_usd_col], errors="coerce").fillna(0.0).to_numpy()
        if resid_usd_col else np.zeros(len(df))
    )

    out = pd.DataFrame({
        "date":                 df["date"],
        "data":                 df["date"],
        "resid_log_btc_mcap":   y.to_numpy(dtype=float),
        "resid_btc_mcap_usd":   resid_usd_arr,
        "pred_resid_log":       y_hat.astype(float),
        "media_effect_log":     media_effect_usd,
        "media_effect_usd":     media_effect_usd_final,
        "news_effect_usd":      media_effect_usd_final,
        "abs_media_effect_usd": np.abs(media_effect_usd_final),
        "gdelt_mentions_all":   df[mentions_col].to_numpy() if mentions_col else 0.0,
        "gdelt_wzmianki_all":   df[mentions_col].to_numpy() if mentions_col else 0.0,
        "gdelt_tone_all":       df[tone_col].to_numpy()     if tone_col     else 0.0,
        "gdelt_ton_all":        df[tone_col].to_numpy()     if tone_col     else 0.0,
    })

    summary = {
        "rows":     int(len(out)),
        "date_min": str(out["date"].min()),
        "date_max": str(out["date"].max()),
        "model": {
            "type":              "Pipeline(StandardScaler + Ridge)",
            "alpha":             RIDGE_ALPHA,
            "features_total":    len(feature_cols),
            "features_news":     len(news_cols),
            "news_used":         news_cols,
            "note": "Controls not included — SARIMAX in step04 already removes macro effects",
        },
        "fit_metrics_in_sample": in_sample,
        "cv_metrics":            cv,
        "media_effect_totals": {
            "sum_media_effect_usd":      float(out["media_effect_usd"].sum()),
            "sum_abs_media_effect_usd":  float(out["abs_media_effect_usd"].sum()),
            "mean_media_effect_usd":     float(out["media_effect_usd"].mean()),
            "mean_abs_media_effect_usd": float(out["abs_media_effect_usd"].mean()),
            "max_abs_media_effect_usd":  float(out["abs_media_effect_usd"].max()),
        },
    }

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PREDICTIONS, index=False)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    reg: Ridge = model.named_steps["ridge"]
    coefs = pd.DataFrame({
        "feature":     feature_cols,
        "coef_scaled": reg.coef_.astype(float),
        "is_news":     [1] * len(feature_cols),
    }).sort_values("coef_scaled", key=lambda s: np.abs(s), ascending=False)
    coefs.to_csv(OUT_COEFFICIENTS, index=False)

    print("\n" + "=" * 60)
    print(f"  ✓  news_effect_daily.csv     : {len(out)} rows")
    print(f"  ✓  news_effect_summary.json")
    print(f"  ✓  news_effect_coefficients.csv")
    print(f"\n  In-sample R²:             {in_sample['R2']:.4f}")
    print(f"  CV R² (mean ± std):       {cv['R2_mean']:.4f} ± {cv['R2_std']:.4f}")
    print(f"  SUM(media_effect_usd$):   ${summary['media_effect_totals']['sum_media_effect_usd']:>20,.0f}")
    print(f"  SUM(|media_effect|):      ${summary['media_effect_totals']['sum_abs_media_effect_usd']:>20,.0f}")
    print(f"  MAX(|media_effect|/day):  ${summary['media_effect_totals']['max_abs_media_effect_usd']:>20,.0f}")
    print(f"\n  Note: per-day R²≈0.01 is expected — media explains ~1% of daily")
    print(f"  residual noise. The paper's 26.2% is the window-level ratio")
    print(f"  sum_abs(media)/sum_abs(resid) over ±30 day event windows.")
    print("=" * 60)
    print("Next step: step08_event_level_impact.py\n")


if __name__ == "__main__":
    main()