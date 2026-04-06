# -*- coding: utf-8 -*-
"""
pipelines/step14_oos_validation.py
==============================================
Out-of-Sample Validation of the BMPI index (fixed console-only version).

  TRAIN: < 2022-01-01
  TEST:  >= 2022-01-01

Tests:
  1. Distribution stability (KS-test)
  2. Predictive performance BMPI -> excess_media_usd
  3. Event-level validation
  4. Rolling walk-forward validation
  5. Regime stability

Input:
  data/processed/excess_media_effect_daily.csv
  data/processed/events_peaks_balanced.csv
  data/processed/features_daily.parquet

Output:
  data/processed/oos_validation_results.json
  data/processed/oos_validation_events.csv
  data/processed/oos_rolling_metrics.csv

Next step:
  step15_benchmark_comparison.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.special import expit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

EXCESS_CSV = DATA_PROCESSED / "excess_media_effect_daily.csv"
PEAKS_CSV = DATA_PROCESSED / "events_peaks_balanced.csv"
FEATURES_PAR = DATA_PROCESSED / "features_daily.parquet"

OUT_JSON = DATA_PROCESSED / "oos_validation_results.json"
OUT_EVENTS = DATA_PROCESSED / "oos_validation_events.csv"
OUT_ROLLING = DATA_PROCESSED / "oos_rolling_metrics.csv"

SPLIT_DATE = pd.Timestamp("2022-01-01")
ROLLING_TRAIN_WINDOW = 365
ROLLING_TEST_WINDOW = 30

BMPI_WEIGHTS = {"w1": 0.25, "w2": 0.20, "w3": 0.20, "w4": 0.20, "w5": 0.15}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load CSV/Parquet, normalise columns, always create 'date' column."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        df.columns = [str(c).lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if c in ("date", "data", "day")), None)
        if date_col:
            df["date"] = pd.to_datetime(
                df[date_col].astype(str).str.strip().str[:10],
                format="%Y-%m-%d",
                errors="coerce",
            ).dt.normalize()
        return df
    except Exception as e:
        print(f"  [WARN] {path.name}: {e}")
        return None


def fmt(x, d: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.{d}f}"


def stars(p) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def pearsonr(x, y) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    if np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan
    r, p = scipy_stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def z_apply(arr: np.ndarray, mu: float, sd: float) -> np.ndarray:
    sd = max(sd, 1e-9)
    return np.clip((arr - mu) / sd, -3, 3)


# ---------------------------------------------------------------------------
# BMPI computation
# ---------------------------------------------------------------------------

def compute_bmpi_from_components(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    Compute daily BMPI from z-score components.
    Uses TRAIN params when provided to avoid look-ahead bias.
    Missing components are treated as zero.
    """
    def _z(col: str) -> np.ndarray:
        arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values if col in df.columns else np.zeros(len(df))
        if params and f"mu_{col}" in params and f"sd_{col}" in params:
            return z_apply(arr, params[f"mu_{col}"], params[f"sd_{col}"])
        mu = float(np.nanmean(arr)) if len(arr) else 0.0
        sd = float(np.nanstd(arr)) if len(arr) else 1.0
        return z_apply(arr, mu, sd)

    raw = (
        BMPI_WEIGHTS["w1"] * _z("z_volume") +
        BMPI_WEIGHTS["w2"] * _z("z_tone") +
        BMPI_WEIGHTS["w4"] * _z("z_resid")
    )
    return pd.Series(expit(raw), index=df.index)


def compute_bmpi_from_train_scaling(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    """
    Fallback when z-components are absent.
    Uses raw proxies available in step09 output:
      - raw_abs_media_effect_usd
      - bmpi_tone_used
      - media_share_of_abnormal_move_pct
    Scales them on TRAIN only and applies to target_df.
    """
    def extract_series(df: pd.DataFrame, name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros(len(df)), index=df.index)

    train_volume = extract_series(train_df, "raw_abs_media_effect_usd")
    train_tone = extract_series(train_df, "bmpi_tone_used").abs()
    train_resid = extract_series(train_df, "media_share_of_abnormal_move_pct")

    mu_vol, sd_vol = float(train_volume.mean()), float(train_volume.std(ddof=0))
    mu_tone, sd_tone = float(train_tone.mean()), float(train_tone.std(ddof=0))
    mu_resid, sd_resid = float(train_resid.mean()), float(train_resid.std(ddof=0))

    target_volume = extract_series(target_df, "raw_abs_media_effect_usd").values
    target_tone = extract_series(target_df, "bmpi_tone_used").abs().values
    target_resid = extract_series(target_df, "media_share_of_abnormal_move_pct").values

    z_volume = z_apply(target_volume, mu_vol, sd_vol if sd_vol > 1e-9 else 1.0)
    z_tone = z_apply(target_tone, mu_tone, sd_tone if sd_tone > 1e-9 else 1.0)
    z_resid = z_apply(target_resid, mu_resid, sd_resid if sd_resid > 1e-9 else 1.0)

    raw = (
        BMPI_WEIGHTS["w1"] * z_volume +
        BMPI_WEIGHTS["w2"] * z_tone +
        BMPI_WEIGHTS["w4"] * z_resid
    )
    return pd.Series(expit(raw), index=target_df.index)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    print("=" * 72)
    print("LOADING DATA")
    print("=" * 72)

    excess = load_csv(EXCESS_CSV)
    if excess is None:
        raise FileNotFoundError(
            f"Required file not found: {EXCESS_CSV}\n"
            "Run step09_fake_classification.py first."
        )

    for c in (
        "excess_media_effect_usd",
        "bmpi_score",
        "resid_btc_mcap_usd",
        "media_effect_usd",
        "media_effect_used",
        "raw_abs_media_effect_usd",
        "media_share_of_abnormal_move_pct",
        "bmpi_tone_used",
        "z_volume",
        "z_tone",
        "z_resid",
    ):
        if c in excess.columns:
            excess[c] = pd.to_numeric(excess[c], errors="coerce")

    if "excess_media_effect_usd" not in excess.columns:
        if "media_effect_used" in excess.columns:
            excess["excess_media_effect_usd"] = pd.to_numeric(excess["media_effect_used"], errors="coerce")
        elif "media_effect_usd" in excess.columns:
            excess["excess_media_effect_usd"] = pd.to_numeric(excess["media_effect_usd"], errors="coerce")
        else:
            excess["excess_media_effect_usd"] = np.nan

    feat = load_csv(FEATURES_PAR)
    if feat is not None:
        logret_col = next((c for c in feat.columns if "logret" in c), None)
        mcap_col = next(
            (c for c in feat.columns if ("mcap" in c or "kapitaliz" in c) and "btc" in c),
            None,
        )

        sub = feat[["date"]].copy()
        if logret_col:
            sub["btc_logret"] = pd.to_numeric(feat[logret_col], errors="coerce")
        if mcap_col:
            sub["btc_mcap"] = pd.to_numeric(feat[mcap_col], errors="coerce")

        excess = excess.merge(sub, on="date", how="left")
        print(f"  [OK] features: logret={logret_col}")

    if "btc_logret" not in excess.columns or excess["btc_logret"].notna().sum() < 100:
        if "btc_mcap" in excess.columns:
            excess["btc_logret"] = np.log(
                excess["btc_mcap"].clip(lower=1) /
                excess["btc_mcap"].clip(lower=1).shift(1)
            )
            print("  [INFO] btc_logret computed from btc_mcap")

    peaks = load_csv(PEAKS_CSV)
    if peaks is not None:
        pk_col = next((c for c in peaks.columns if c in ("peak_date", "data_piku")), None)
        if pk_col:
            peaks["peak_date"] = pd.to_datetime(
                peaks[pk_col].astype(str).str[:10],
                format="%Y-%m-%d",
                errors="coerce",
            ).dt.normalize()

    excess = excess.sort_values("date").reset_index(drop=True)

    print(
        f"  [OK] excess_daily: {len(excess)} days "
        f"({excess['date'].min().date()} -> {excess['date'].max().date()})"
    )
    if peaks is not None:
        print(f"  [OK] peaks: {len(peaks)} events")
    print()

    return excess, peaks


# ---------------------------------------------------------------------------
# Split & calibrate
# ---------------------------------------------------------------------------

def split_and_calibrate(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict, float]:
    print("─" * 72)
    print(f"TRAIN / TEST SPLIT  (boundary: {SPLIT_DATE.date()})")
    print("─" * 72)

    train = df[df["date"] < SPLIT_DATE].copy()
    test = df[df["date"] >= SPLIT_DATE].copy()

    print(
        f"  TRAIN: {len(train)} days "
        f"({train['date'].min().date()} -> {train['date'].max().date()})"
    )
    print(
        f"  TEST:  {len(test)} days "
        f"({test['date'].min().date()} -> {test['date'].max().date()})"
    )
    print()

    params: Dict = {}
    for col in ("z_volume", "z_tone", "z_resid"):
        if col in train.columns:
            vals = train[col].dropna()
            if len(vals) > 0:
                params[f"mu_{col}"] = float(vals.mean())
                params[f"sd_{col}"] = float(vals.std())

    print("  Calibration parameters (from TRAIN):")
    if params:
        for k, v in params.items():
            print(f"    {k:<25} {v:.6f}")
        print()
        train["bmpi_oos"] = compute_bmpi_from_components(train)
        test["bmpi_oos"] = compute_bmpi_from_components(test, params)
        print("  BMPI recomputed from z-score components")
    else:
        print("    [INFO] z-score components not found; using TRAIN-based fallback scaling")
        print()
        train["bmpi_oos"] = compute_bmpi_from_train_scaling(train, train)
        test["bmpi_oos"] = compute_bmpi_from_train_scaling(train, test)
        print("  BMPI recomputed from TRAIN-based fallback scaling")

    # Keep original score too if present
    if "bmpi_score" in train.columns and train["bmpi_score"].notna().sum() > 10:
        train["bmpi"] = train["bmpi_score"]
        test["bmpi"] = test["bmpi_score"]
        print("  Original step09 bmpi_score preserved as reference")
    else:
        train["bmpi"] = train["bmpi_oos"]
        test["bmpi"] = test["bmpi_oos"]

    threshold = float(train["bmpi_oos"].quantile(0.70))
    print(f"  BMPI threshold (70th pct of TRAIN OOS score): {threshold:.4f}")
    print()

    return train, test, params, threshold


# ---------------------------------------------------------------------------
# Check 1: Distribution stability
# ---------------------------------------------------------------------------

def check_distribution_stability(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    print("─" * 72)
    print("TEST 1: BMPI Distribution Stability (KS-test)")
    print("─" * 72)

    tr = train["bmpi_oos"].dropna().values
    te = test["bmpi_oos"].dropna().values

    ks_stat, ks_p = scipy_stats.ks_2samp(tr, te)
    mw_stat, mw_p = scipy_stats.mannwhitneyu(tr, te, alternative="two-sided")

    def _stats(arr):
        return {
            k: float(v) for k, v in zip(
                ("mean", "std", "p25", "p50", "p75", "p90"),
                (
                    arr.mean(),
                    arr.std(),
                    np.percentile(arr, 25),
                    np.percentile(arr, 50),
                    np.percentile(arr, 75),
                    np.percentile(arr, 90),
                ),
            )
        }

    s_tr, s_te = _stats(tr), _stats(te)

    print(f"\n  {'metric':<15} {'TRAIN':>10} {'TEST':>10} {'diff':>10}")
    print("  " + "─" * 52)

    for key in ("mean", "std", "p25", "p50", "p75", "p90"):
        diff = s_te[key] - s_tr[key]
        flag = " ⚠" if abs(diff) > 0.05 else ""
        print(f"  {key:<15} {s_tr[key]:>10.4f} {s_te[key]:>10.4f} {diff:>+10.4f}{flag}")

    print(
        f"\n  KS-test:      stat={ks_stat:.4f}  p={ks_p:.6f}  "
        f"{'distributions SIMILAR ✓' if ks_p > 0.05 else 'distributions DIFFER ⚠'}"
    )
    print(
        f"  Mann-Whitney: stat={mw_stat:.0f}  p={mw_p:.6f}  "
        f"{'medians SIMILAR ✓' if mw_p > 0.05 else 'medians DIFFER ⚠'}"
    )

    mean_diff = abs(s_te["mean"] - s_tr["mean"])
    verdict = "✓ STABLE" if mean_diff < 0.03 else "~ moderate" if mean_diff < 0.07 else "✗ unstable"
    print(f"\n  Mean difference: {mean_diff:.4f} -> BMPI {verdict}\n")

    return {
        **{f"train_{k}": s_tr[k] for k in s_tr},
        **{f"test_{k}": s_te[k] for k in s_te},
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "mw_p": float(mw_p),
        "mean_diff": float(mean_diff),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Check 2: Predictive performance
# ---------------------------------------------------------------------------

def check_predictive_performance(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    print("─" * 72)
    print("TEST 2: Predictive Performance BMPI -> excess_media_effect_usd")
    print("─" * 72)

    results: Dict = {}

    for name, df in (("TRAIN", train), ("TEST", test)):
        x = df["bmpi_oos"].values
        y = df["excess_media_effect_usd"].fillna(0).values

        r, p = pearsonr(x, y)
        mask = np.isfinite(x) & np.isfinite(y)

        if mask.sum() > 10 and np.nanstd(x[mask]) > 1e-9:
            sl, ic, rv, pv, _ = scipy_stats.linregress(x[mask], y[mask])
            yp = sl * x[mask] + ic
            rmse = float(np.sqrt(np.mean((y[mask] - yp) ** 2)))
            mae = float(np.mean(np.abs(y[mask] - yp)))
            r2 = float(rv ** 2)
        else:
            sl = ic = rmse = mae = r2 = np.nan
            if mask.sum() > 10 and np.nanstd(x[mask]) <= 1e-9:
                print(f"  [WARN] {name}: BMPI is constant")

        print(
            f"\n  {name}:  r={fmt(r)} {stars(p)}  "
            f"R²={fmt(r2)}  RMSE={fmt(rmse)}  MAE={fmt(mae)}"
        )

        results[name.lower()] = {
            "r": r,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "p": p,
        }

    r2_tr = results.get("train", {}).get("r2", np.nan)
    r2_te = results.get("test", {}).get("r2", np.nan)
    degrad = (
        float((r2_tr - r2_te) / r2_tr * 100)
        if not (np.isnan(r2_tr) or np.isnan(r2_te)) and r2_tr > 0
        else np.nan
    )

    print(
        f"\n  R² degradation: {fmt(degrad,1)}%  "
        f"{'✓ acceptable (<20%)' if not np.isnan(degrad) and degrad < 20 else '⚠ high'}\n"
    )

    results["degradation_r2_pct"] = degrad
    results["r_train"] = results.get("train", {}).get("r", np.nan)
    results["r_test"] = results.get("test", {}).get("r", np.nan)
    return results


# ---------------------------------------------------------------------------
# Check 3: Event-level validation
# ---------------------------------------------------------------------------

def check_event_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    peaks: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    print("─" * 72)
    print("TEST 3: Event-Level Validation")
    print("─" * 72)

    all_df = pd.concat([train, test], ignore_index=True)
    rows: List[dict] = []

    for i, pk in peaks.iterrows():
        eid = pk.get("event_id", f"EVT_{i}")
        pik = pd.Timestamp(pk["peak_date"])
        split = "TEST" if pik >= SPLIT_DATE else "TRAIN"

        mask = (
            (all_df["date"] >= pik - pd.Timedelta(3, "D")) &
            (all_df["date"] <= pik + pd.Timedelta(2, "D"))
        )
        w = all_df[mask]
        if len(w) == 0:
            continue

        rows.append({
            "event_id": eid,
            "peak_date": pik.date(),
            "split": split,
            "bmpi_mean": float(w["bmpi_oos"].mean()),
            "bmpi_max": float(w["bmpi_oos"].max()),
            "excess_usd_sum": float(w["excess_media_effect_usd"].sum()),
            "above_threshold": float(w["bmpi_oos"].mean()) > threshold,
            "n_days": len(w),
        })

    df_ev = pd.DataFrame(rows)

    for split in ("TRAIN", "TEST"):
        sub = df_ev[df_ev["split"] == split]
        print(f"\n  {split} EVENTS ({len(sub)}):")
        print(f"  {'event_id':<26} {'BMPI_mean':>10} {'excess_B':>10} {'>thr?':>6}")
        print("  " + "─" * 58)

        for _, row in sub.sort_values("bmpi_mean", ascending=False).iterrows():
            flag = " ★" if row["above_threshold"] else ""
            print(
                f"  {str(row['event_id']):<26} {row['bmpi_mean']:>10.4f} "
                f"{row['excess_usd_sum'] / 1e9:>9.3f}B "
                f"{'YES' if row['above_threshold'] else 'no':>6}{flag}"
            )

    df_ev.to_csv(OUT_EVENTS, index=False)
    print()
    return df_ev


# ---------------------------------------------------------------------------
# Check 4: Rolling walk-forward
# ---------------------------------------------------------------------------

def check_rolling_validation(df: pd.DataFrame) -> pd.DataFrame:
    print("─" * 72)
    print("TEST 4: Rolling Walk-Forward Validation")
    print("─" * 72)
    print(
        f"  TRAIN window: {ROLLING_TRAIN_WINDOW} days  |  "
        f"TEST window: {ROLLING_TEST_WINDOW} days\n"
    )

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    rows = []
    w_n = 0

    use_components = all(c in df.columns for c in ("z_volume", "z_tone", "z_resid"))

    for start in range(0, n - ROLLING_TRAIN_WINDOW - ROLLING_TEST_WINDOW, ROLLING_TEST_WINDOW):
        tr_end = start + ROLLING_TRAIN_WINDOW
        te_end = tr_end + ROLLING_TEST_WINDOW
        if te_end > n:
            break

        tr = df.iloc[start:tr_end].copy()
        te = df.iloc[tr_end:te_end].copy()

        if use_components:
            params: Dict = {}
            for col in ("z_volume", "z_tone", "z_resid"):
                vals = pd.to_numeric(tr[col], errors="coerce").dropna()
                if len(vals) > 0:
                    params[f"mu_{col}"] = float(vals.mean())
                    params[f"sd_{col}"] = float(vals.std())
            te["bmpi_roll"] = compute_bmpi_from_components(te, params)
        else:
            te["bmpi_roll"] = compute_bmpi_from_train_scaling(tr, te)

        r, p = pearsonr(te["bmpi_roll"].values, te["excess_media_effect_usd"].fillna(0).values)

        rows.append({
            "window": w_n,
            "test_start": df.iloc[tr_end]["date"],
            "test_end": df.iloc[te_end - 1]["date"],
            "r": r,
            "p": p,
            "sig": stars(p),
            "r2": r ** 2 if not np.isnan(r) else np.nan,
            "n_test": len(te),
            "period": "TEST" if df.iloc[tr_end]["date"] >= SPLIT_DATE else "TRAIN_PERIOD",
        })
        w_n += 1

    df_roll = pd.DataFrame(rows)
    n_total = len(df_roll)
    n_sig = int((df_roll["p"] < 0.05).sum()) if n_total > 0 else 0
    mean_r = float(df_roll["r"].mean()) if n_total > 0 else np.nan
    oos = df_roll["period"] == "TEST"
    oos_r = float(df_roll.loc[oos, "r"].mean()) if oos.sum() > 0 else np.nan

    print(
        f"  Total windows: {n_total}  |  Significant: {n_sig}/{n_total}  "
        f"|  Mean r: {fmt(mean_r)}  |  OOS r: {fmt(oos_r)}\n"
    )

    if n_total > 0:
        df_roll["year"] = pd.to_datetime(df_roll["test_start"]).dt.year
        print(f"  {'year':>6} {'windows':>8} {'mean r':>8} {'sig%':>8}")
        print("  " + "─" * 36)

        for year, grp in df_roll.groupby("year"):
            sig_pct = (grp["p"] < 0.05).mean() * 100
            print(f"  {year:>6} {len(grp):>8} {grp['r'].mean():>8.4f} {sig_pct:>7.0f}%")

    df_roll.to_csv(OUT_ROLLING, index=False)
    print()
    return df_roll


# ---------------------------------------------------------------------------
# Check 5: Regime stability
# ---------------------------------------------------------------------------

def check_regime_stability(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    print("─" * 72)
    print("TEST 5: BMPI Stability Across Market Regimes")
    print("─" * 72)

    all_df = pd.concat([train, test], ignore_index=True)
    if "btc_logret" not in all_df.columns or all_df["btc_logret"].notna().sum() < 50:
        print("  [WARN] btc_logret not available\n")
        return {}

    all_df["regime"] = all_df["btc_logret"].apply(
        lambda r: (
            "bull" if r > 0.01 else
            "bear" if r < -0.01 else
            "flat"
        ) if not pd.isna(r) else "unknown"
    )

    results: Dict = {}
    print(f"\n  {'regime':<10} {'n_days':>8} {'mean_bmpi':>11} {'r->excess':>10} {'sig':>5} {'split':>8}")
    print("  " + "─" * 62)

    for split_name, df_part in (("TRAIN", train), ("TEST", test)):
        df_part = df_part.copy()
        df_part["regime"] = all_df.loc[df_part.index, "regime"].values

        for regime in ("bull", "bear", "flat"):
            sub = df_part[df_part["regime"] == regime]
            if len(sub) < 20:
                continue

            bm = float(sub["bmpi_oos"].mean())
            r, p = pearsonr(sub["bmpi_oos"].values, sub["excess_media_effect_usd"].fillna(0).values)

            print(
                f"  {regime:<10} {len(sub):>8} {bm:>11.4f} "
                f"{fmt(r):>10} {stars(p):>5} {split_name:>8}"
            )

            results[f"{split_name}_{regime}"] = {
                "n": len(sub),
                "bmpi_mean": bm,
                "r": r,
                "p": p,
            }

    print()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    excess, peaks = load_data()
    train, test, params, threshold = split_and_calibrate(excess)

    print("=" * 72)
    print("RUNNING VALIDATION TESTS")
    print("=" * 72 + "\n")

    dist_res = check_distribution_stability(train, test)
    perf_res = check_predictive_performance(train, test)
    ev_df = check_event_validation(train, test, peaks, threshold) if peaks is not None else pd.DataFrame()
    roll_df = check_rolling_validation(excess)
    regime_res = check_regime_stability(train, test)

    summary = {
        "split_date": str(SPLIT_DATE.date()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "ks_p": dist_res.get("ks_p"),
        "mean_diff": dist_res.get("mean_diff"),
        "distribution_verdict": dist_res.get("verdict"),
        "r_train": perf_res.get("r_train"),
        "r_test": perf_res.get("r_test"),
        "degradation_r2_pct": perf_res.get("degradation_r2_pct"),
        "rolling_oos_r": float(roll_df.loc[roll_df["period"] == "TEST", "r"].mean()) if len(roll_df) > 0 else None,
        "n_train_events": int((ev_df["split"] == "TRAIN").sum()) if len(ev_df) > 0 else 0,
        "n_test_events": int((ev_df["split"] == "TEST").sum()) if len(ev_df) > 0 else 0,
        "threshold": threshold,
        "regime_results": regime_res,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[OK] Saved: {OUT_JSON}")
    print(f"[OK] Saved: {OUT_EVENTS}")
    print(f"[OK] Saved: {OUT_ROLLING}")

    r_tr = perf_res.get("r_train", np.nan)
    r_te = perf_res.get("r_test", np.nan)
    degrad = perf_res.get("degradation_r2_pct", np.nan)
    ks_p = dist_res.get("ks_p", np.nan)
    oos = roll_df["period"] == "TEST"
    roll_oos = float(roll_df.loc[oos, "r"].mean()) if oos.sum() > 0 else np.nan

    W = 78
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║  OUT-OF-SAMPLE VALIDATION — SUMMARY                                    ║")
    print("╠" + "═" * (W - 2) + "╣")
    print(f"║  TRAIN: < 2022-01-01  ({len(train)} days){' ' * 36}║")
    print(f"║  TEST:  >= 2022-01-01 ({len(test)} days){' ' * 36}║")
    print("║                                                                          ║")
    print("║  Correlation BMPI -> excess_usd:                                         ║")
    print(f"║    TRAIN r = {fmt(r_tr):<10} TEST r = {fmt(r_te):<10}{' ' * 27}║")
    print(f"║    R² degradation = {fmt(degrad,1)}%{' ' * 45}║")
    print("║                                                                          ║")
    dist_text = "distributions SIMILAR ✓" if not np.isnan(ks_p) and ks_p > 0.05 else "distributions DIFFER ⚠"
    print(f"║  KS-test p = {fmt(ks_p,4):<8} {dist_text:<47}║")
    print(f"║  Rolling OOS mean r = {fmt(roll_oos):<10}{' ' * 45}║")
    print("║                                                                          ║")

    ok = not np.isnan(degrad) and degrad < 20 and not np.isnan(r_te) and r_te > 0.5
    if ok:
        print("║  ✓ VALIDATION PASSED                                                     ║")
        print("║    BMPI is a stable indicator on unseen 2022–2025 data                  ║")
    else:
        print("║  ~ PARTIAL VALIDATION                                                    ║")
        print("║    BMPI works but with some degradation — discuss regime shift           ║")

    print("╚" + "═" * (W - 2) + "╝")
    print("\nPipeline complete. All steps finished.")


if __name__ == "__main__":
    main()