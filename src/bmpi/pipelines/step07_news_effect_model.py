# -*- coding: utf-8 -*-
"""
pipelines/step07_news_effect_model.py
========================================
BMPI v2: news effect model on abnormal BTC returns.

What changed vs BMPI v1:
- Target is abnormal BTC log-return from step04, not raw residual-level market cap.
- Uses ALL days (including zero-news days) to avoid selection bias.
- Uses only news features as explanatory variables because baseline market factors
  were already removed in step04.
- Applies feature filtering + Ridge regression with time-series cross-validation.
- Produces daily predicted media-driven abnormal return and its USD approximation.

Input:
  data/processed/model_dataset_daily.parquet  (from step06)

Expected key columns:
  date
  abnormal_btc_logret
  btc_mcap
  gdelt_* feature columns created in step06

Output:
  data/processed/news_effect_daily.csv
  data/processed/news_effect_summary.json
  data/processed/news_effect_coefficients.csv

Key outputs:
  predicted_media_abnormal_logret
  predicted_media_effect_usd
  bmip_v2_daily
  bmip_v2_daily_abs
  model_target
  model_type

Notes:
- predicted_media_abnormal_logret is the model-estimated media component
  of abnormal BTC return for each day.
- predicted_media_effect_usd is approximated as:
      predicted_media_abnormal_logret * btc_mcap
  which is acceptable as first-order approximation for daily moves.
- abnormal return != media effect
- predicted media effect is still model-based, not direct causality proof;
  causal interpretation improves when combined with event study and OOS checks.

Next step:
  step08_event_level_impact.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

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

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

MODEL_DATASET = DATA_PROCESSED / "model_dataset_daily.parquet"
OUT_PREDICTIONS = DATA_PROCESSED / "news_effect_daily.csv"
OUT_SUMMARY = DATA_PROCESSED / "news_effect_summary.json"
OUT_COEFFICIENTS = DATA_PROCESSED / "news_effect_coefficients.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_COL = "abnormal_btc_logret"
MCAP_COL_CANDIDATES = ["btc_mcap", "btc_kapitalizacja_usd"]

RIDGE_ALPHA = 10.0
N_SPLITS = 5

MIN_NON_NULL_RATIO = 0.80
MAX_ABS_CORR = 0.97
TOP_K_BY_ABS_CORR = 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def get_news_feature_candidates(df: pd.DataFrame) -> List[str]:
    """
    Take only BMPI v2 news features from step06.
    Exclude raw identifiers and obvious helper flags if needed.
    """
    exclude_exact = {
        "date",
        TARGET_COL,
        "btc_price",
        "btc_mcap",
        "btc_logret",
        "expected_btc_logret",
        "abnormal_btc_logret_zscore_60d",
        "abnormal_btc_return_pct",
        "baseline_btc_mcap_hat_usd",
        "abnormal_btc_mcap_usd",
        "btc_return_pct",
        "expected_btc_return_pct",
        "weekday",
        "month",
        "is_weekend",
        "abnormal_x_volatility",
        "abnormal_x_regime",
        "has_any_news_all",
        "has_bad_news_all",
    }

    cols = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if not col.startswith("gdelt_"):
            continue
        cols.append(col)
    return cols


def filter_numeric_features(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """
    Keep numeric-ish columns with enough non-null coverage and non-zero variance.
    """
    selected: List[str] = []
    n = len(df)

    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        non_null_ratio = float(s.notna().mean())
        if non_null_ratio < MIN_NON_NULL_RATIO:
            continue
        if float(s.fillna(0.0).std(ddof=0)) == 0.0:
            continue
        selected.append(col)

    if not selected:
        raise ValueError("No usable news features after numeric filtering.")

    return selected


def select_top_features_by_target_corr(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    top_k: int,
) -> List[str]:
    """
    Keep strongest features by absolute correlation with target.
    """
    corrs: List[Tuple[str, float]] = []
    y = pd.to_numeric(df[target_col], errors="coerce")

    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < 50:
            continue
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if pd.isna(corr):
            continue
        corrs.append((col, abs(float(corr))))

    if not corrs:
        raise ValueError("Failed to compute feature-target correlations.")

    corrs.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in corrs[:top_k]]


def drop_highly_correlated_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float,
    target_col: str,
) -> List[str]:
    """
    Greedy removal of highly collinear features.
    Keep the one with stronger absolute correlation to target.
    """
    if len(feature_cols) <= 1:
        return feature_cols

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)

    target_corr = {}
    for col in feature_cols:
        corr = X[col].corr(y)
        target_corr[col] = abs(float(corr)) if pd.notna(corr) else 0.0

    corr_matrix = X.corr().abs()
    keep = list(feature_cols)

    for i, col_i in enumerate(feature_cols):
        if col_i not in keep:
            continue
        for col_j in feature_cols[i + 1:]:
            if col_j not in keep:
                continue
            cij = corr_matrix.loc[col_i, col_j]
            if pd.isna(cij):
                continue
            if float(cij) >= threshold:
                if target_corr[col_i] >= target_corr[col_j]:
                    keep.remove(col_j)
                else:
                    keep.remove(col_i)
                    break

    return keep


def time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: Pipeline,
    n_splits: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    TimeSeriesSplit CV with out-of-fold predictions.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, r2s = [], [], []
    oof_pred = np.full(len(y), np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        oof_pred[test_idx] = pred
        m = compute_metrics(y_test.to_numpy(), pred)
        maes.append(m["MAE"])
        rmses.append(m["RMSE"])
        r2s.append(m["R2"])

        print(
            f"  Fold {fold}: "
            f"test_n={len(test_idx)}  "
            f"MAE={m['MAE']:.6f}  RMSE={m['RMSE']:.6f}  R2={m['R2']:.6f}"
        )

    summary = {
        "cv_splits": int(n_splits),
        "MAE_mean": float(np.nanmean(maes)),
        "RMSE_mean": float(np.nanmean(rmses)),
        "R2_mean": float(np.nanmean(r2s)),
        "MAE_std": float(np.nanstd(maes)),
        "RMSE_std": float(np.nanstd(rmses)),
        "R2_std": float(np.nanstd(r2s)),
    }
    return summary, oof_pred


def make_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42)),
    ])


def build_coefficients_table(
    model: Pipeline,
    feature_cols: List[str],
) -> pd.DataFrame:
    scaler: StandardScaler = model.named_steps["scaler"]
    ridge: Ridge = model.named_steps["ridge"]

    # coefficients are already on scaled feature space
    df_coef = pd.DataFrame({
        "feature": feature_cols,
        "coefficient_scaled": ridge.coef_,
        "coefficient_abs": np.abs(ridge.coef_),
        "feature_mean_before_scaling": scaler.mean_,
        "feature_std_before_scaling": scaler.scale_,
    }).sort_values("coefficient_abs", ascending=False).reset_index(drop=True)

    return df_coef


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 07 — NEWS EFFECT MODEL (BMPI v2, abnormal returns)")
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

    mcap_col = pick_first_existing(df, MCAP_COL_CANDIDATES)
    if mcap_col is None:
        raise ValueError("BTC market cap column not found in model dataset.")

    df[mcap_col] = pd.to_numeric(df[mcap_col], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # -----------------------------------------------------------------------
    # Feature selection
    # -----------------------------------------------------------------------
    news_feature_candidates = get_news_feature_candidates(df)
    print(f"  Raw GDELT candidate features: {len(news_feature_candidates)}")

    usable_features = filter_numeric_features(df, news_feature_candidates)
    print(f"  Usable numeric features:      {len(usable_features)}")

    top_features = select_top_features_by_target_corr(
        df=df,
        feature_cols=usable_features,
        target_col=TARGET_COL,
        top_k=TOP_K_BY_ABS_CORR,
    )
    print(f"  Top features by |corr|:       {len(top_features)}")

    final_features = drop_highly_correlated_features(
        df=df,
        feature_cols=top_features,
        threshold=MAX_ABS_CORR,
        target_col=TARGET_COL,
    )
    print(f"  Final selected features:      {len(final_features)}")
    print(f"  Ridge alpha:                  {RIDGE_ALPHA}")

    if not final_features:
        raise ValueError("No final features selected for step07.")

    # Build model matrix
    X = df[final_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[TARGET_COL].copy()

    # -----------------------------------------------------------------------
    # Time-series CV
    # -----------------------------------------------------------------------
    model = make_pipeline(RIDGE_ALPHA)

    print("\n  Running TimeSeriesSplit CV...")
    cv_summary, oof_pred = time_series_cv(X, y, model, N_SPLITS)

    valid_oof_mask = ~np.isnan(oof_pred)
    if valid_oof_mask.sum() > 0:
        oof_metrics = compute_metrics(y.to_numpy()[valid_oof_mask], oof_pred[valid_oof_mask])
    else:
        oof_metrics = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    print(
        f"\n  OOF metrics: "
        f"MAE={oof_metrics['MAE']:.6f}  "
        f"RMSE={oof_metrics['RMSE']:.6f}  "
        f"R2={oof_metrics['R2']:.6f}"
    )

    # -----------------------------------------------------------------------
    # Fit full model
    # -----------------------------------------------------------------------
    model.fit(X, y)
    pred_full = model.predict(X)

    in_sample_metrics = compute_metrics(y.to_numpy(), pred_full)
    print(
        f"  In-sample: "
        f"MAE={in_sample_metrics['MAE']:.6f}  "
        f"RMSE={in_sample_metrics['RMSE']:.6f}  "
        f"R2={in_sample_metrics['R2']:.6f}"
    )

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------
    out = df.copy()
    out["predicted_media_abnormal_logret"] = pred_full
    out["predicted_media_effect_usd"] = (
        out["predicted_media_abnormal_logret"] *
        pd.to_numeric(out[mcap_col], errors="coerce").fillna(0.0)
    )

    out["bmip_v2_daily"] = out["predicted_media_effect_usd"]
    out["bmip_v2_daily_abs"] = out["bmip_v2_daily"].abs()

    # OOF predictions for honest validation downstream
    out["predicted_media_abnormal_logret_oof"] = oof_pred
    out["predicted_media_effect_usd_oof"] = (
        out["predicted_media_abnormal_logret_oof"] *
        pd.to_numeric(out[mcap_col], errors="coerce").fillna(0.0)
    )

    # Helpful ratios
    if "abnormal_btc_mcap_usd" in out.columns:
        denom = pd.to_numeric(out["abnormal_btc_mcap_usd"], errors="coerce").replace(0, np.nan).abs()
        out["bmip_v2_share_of_abnormal_move"] = out["bmip_v2_daily_abs"] / denom

    out["model_target"] = TARGET_COL
    out["model_type"] = "Ridge_news_to_abnormal_returns"
    out["model_alpha"] = RIDGE_ALPHA
    out["n_selected_features"] = len(final_features)

    out.to_csv(OUT_PREDICTIONS, index=False)

    coef_df = build_coefficients_table(model, final_features)
    coef_df.to_csv(OUT_COEFFICIENTS, index=False)

    summary = {
        "model_type": "Ridge_news_to_abnormal_returns",
        "target": TARGET_COL,
        "alpha": RIDGE_ALPHA,
        "n_rows": int(len(out)),
        "n_features_raw": int(len(news_feature_candidates)),
        "n_features_usable": int(len(usable_features)),
        "n_features_top_corr": int(len(top_features)),
        "n_features_final": int(len(final_features)),
        "selected_features": final_features,
        "in_sample_metrics": in_sample_metrics,
        "cv_summary": cv_summary,
        "oof_metrics": oof_metrics,
        "bmip_v2_daily_abs_sum_usd": float(out["bmip_v2_daily_abs"].sum()),
        "bmip_v2_daily_sum_usd": float(out["bmip_v2_daily"].sum()),
        "mean_abs_predicted_media_logret": float(np.abs(out["predicted_media_abnormal_logret"]).mean()),
        "mean_abs_predicted_media_usd": float(out["bmip_v2_daily_abs"].mean()),
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Saved predictions:   {OUT_PREDICTIONS}")
    print(f"  Saved coefficients:  {OUT_COEFFICIENTS}")
    print(f"  Saved summary:       {OUT_SUMMARY}")
    print(f"  Final features used: {len(final_features)}")
    print(f"  Mean OOF R2:         {cv_summary['R2_mean']:.6f}")
    print("\nNext step: step08_event_level_impact.py")


if __name__ == "__main__":
    main()