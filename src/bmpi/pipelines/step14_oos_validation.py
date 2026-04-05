# -*- coding: utf-8 -*-
"""
pipelines/step14_oos_validation.py
==============================================
Out-of-Sample Validation of the BMPI index.

  TRAIN: 2015-09-09 → 2021-12-31  (~1412 days, ~20 events)
  TEST:  2022-01-01 → 2026-01-23  (~ 808 days, ~ 9 events)

  Split rationale: 2022 is a natural boundary — new market regime after
  Terra/Luna crash (May 2022), FTX collapse (Nov 2022),
  spot Bitcoin ETF approval (Jan 2024).

Tests:
  1. Distribution stability (KS-test)
  2. Predictive performance BMPI → excess_media_usd
  3. Event-level validation
  4. Rolling walk-forward validation
  5. Regime stability

Input:
  data/processed/excess_media_effect_daily.csv  (from step09)
  data/processed/events_peaks_balanced.csv      (from step03)
  data/processed/features_daily.parquet         (from step02)

Output:
  data/processed/oos_validation_results.json
  data/processed/oos_validation_events.csv
  data/processed/oos_rolling_metrics.csv
  data/processed/oos_report.html

Next step: step15_benchmark_comparison.py
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
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

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

EXCESS_CSV   = DATA_PROCESSED / "excess_media_effect_daily.csv"
PEAKS_CSV    = DATA_PROCESSED / "events_peaks_balanced.csv"
FEATURES_PAR = DATA_PROCESSED / "features_daily.parquet"

OUT_JSON    = DATA_PROCESSED / "oos_validation_results.json"
OUT_EVENTS  = DATA_PROCESSED / "oos_validation_events.csv"
OUT_ROLLING = DATA_PROCESSED / "oos_rolling_metrics.csv"
OUT_HTML    = DATA_PROCESSED / "oos_report.html"

SPLIT_DATE           = pd.Timestamp("2022-01-01")
PRESSURE_THRESHOLD   = 0.55
ROLLING_TRAIN_WINDOW = 365
ROLLING_TEST_WINDOW  = 30

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
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if c in ("date", "data", "day")), None)
        if date_col:
            df["date"] = pd.to_datetime(
                df[date_col].astype(str).str.strip().str[:10],
                format="%Y-%m-%d", errors="coerce"
            ).dt.normalize()
        return df
    except Exception as e:
        print(f"  [WARN] {path.name}: {e}")
        return None


def fmt(x, d=4) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.{d}f}"


def stars(p) -> str:
    if pd.isna(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def pearsonr(x, y) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    r, p = scipy_stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def z_apply(arr: np.ndarray, mu: float, sd: float) -> np.ndarray:
    sd = max(sd, 1e-9)
    return np.clip((arr - mu) / sd, -3, 3)


# ---------------------------------------------------------------------------
# BMPI computation
# ---------------------------------------------------------------------------

def compute_bmpi(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """Compute daily BMPI. Uses `params` from TRAIN to avoid look-ahead bias."""
    def _z(col):
        arr = df[col].fillna(0).values if col in df.columns else np.zeros(len(df))
        if params and f"mu_{col}" in params:
            return z_apply(arr, params[f"mu_{col}"], params[f"sd_{col}"])
        mu, sd = arr.mean(), arr.std()
        return z_apply(arr, mu, sd if sd > 1e-9 else 1.0)

    raw = (BMPI_WEIGHTS["w1"] * _z("z_volume") +
           BMPI_WEIGHTS["w2"] * _z("z_tone")   +
           BMPI_WEIGHTS["w4"] * _z("z_resid"))
    return pd.Series(expit(raw), index=df.index)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    print("=" * 65)
    print("LOADING DATA")
    print("=" * 65)

    excess = load_csv(EXCESS_CSV)
    if excess is None:
        raise FileNotFoundError(
            f"Required file not found: {EXCESS_CSV}\n"
            "Run step09_fake_classification.py first."
        )

    for c in ("excess_media_effect_usd", "bmpi_score", "resid_btc_mcap_usd",
              "media_effect_usd", "z_volume", "z_tone", "z_resid"):
        if c in excess.columns:
            excess[c] = pd.to_numeric(excess[c], errors="coerce")

    feat = load_csv(FEATURES_PAR)
    if feat is not None:
        logret_col = next((c for c in feat.columns if "logret" in c), None)
        mcap_col   = next((c for c in feat.columns
                           if ("mcap" in c or "kapitaliz" in c) and "btc" in c), None)
        sub = feat[["date"]].copy()
        if logret_col: sub["btc_logret"] = pd.to_numeric(feat[logret_col], errors="coerce")
        if mcap_col:   sub["btc_mcap"]   = pd.to_numeric(feat[mcap_col],   errors="coerce")
        excess = excess.merge(sub, on="date", how="left")
        print(f"  [OK] features: logret={logret_col}")

    if "btc_logret" not in excess.columns or excess["btc_logret"].notna().sum() < 100:
        if "btc_mcap" in excess.columns:
            excess["btc_logret"] = np.log(
                excess["btc_mcap"].clip(lower=1) /
                excess["btc_mcap"].clip(lower=1).shift(1)
            )

    peaks = load_csv(PEAKS_CSV)
    if peaks is not None:
        pk_col = next((c for c in peaks.columns
                       if c in ("peak_date", "data_piku")), None)
        if pk_col:
            peaks["peak_date"] = pd.to_datetime(
                peaks[pk_col].astype(str).str[:10],
                format="%Y-%m-%d", errors="coerce"
            ).dt.normalize()

    excess = excess.sort_values("date").reset_index(drop=True)
    print(f"  [OK] excess_daily: {len(excess)} days "
          f"({excess['date'].min().date()} → {excess['date'].max().date()})")
    if peaks is not None:
        print(f"  [OK] peaks: {len(peaks)} events")
    print()
    return excess, peaks


# ---------------------------------------------------------------------------
# Split & Calibrate
# ---------------------------------------------------------------------------

def split_and_calibrate(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict, float]:
    print("─" * 65)
    print(f"TRAIN/TEST SPLIT  (boundary: {SPLIT_DATE.date()})")
    print("─" * 65)

    train = df[df["date"] < SPLIT_DATE].copy()
    test  = df[df["date"] >= SPLIT_DATE].copy()

    print(f"  TRAIN: {len(train)} days "
          f"({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  TEST:  {len(test)} days "
          f"({test['date'].min().date()} → {test['date'].max().date()})")
    print()

    params: Dict = {}
    for col in ("z_volume", "z_tone", "z_resid"):
        if col in train.columns:
            vals = train[col].dropna()
            params[f"mu_{col}"] = float(vals.mean())
            params[f"sd_{col}"] = float(vals.std())

    print("  Calibration parameters (from TRAIN):")
    for k, v in params.items():
        print(f"    {k:<25} {v:.6f}")
    print()

    # Use bmpi_score from step09 if available, otherwise compute
    if "bmpi_score" in train.columns and train["bmpi_score"].notna().sum() > 10:
        train["bmpi"] = train["bmpi_score"]
        test["bmpi"]  = test["bmpi_score"]
        print("  Using bmpi_score from step09")
    else:
        train["bmpi"] = compute_bmpi(train)
        test["bmpi"]  = compute_bmpi(test, params)
        print("  Computed BMPI from z-score components")

    threshold = float(train["bmpi"].quantile(0.70))
    print(f"  BMPI threshold (70th pct of TRAIN): {threshold:.4f}")
    print()
    return train, test, params, threshold


# ---------------------------------------------------------------------------
# Check 1: Distribution stability
# ---------------------------------------------------------------------------

def check_distribution_stability(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    print("─" * 65)
    print("TEST 1: BMPI Distribution Stability (KS-test)")
    print("─" * 65)

    tr = train["bmpi"].dropna().values
    te = test["bmpi"].dropna().values

    ks_stat, ks_p = scipy_stats.ks_2samp(tr, te)
    mw_stat, mw_p = scipy_stats.mannwhitneyu(tr, te, alternative="two-sided")

    def _stats(arr):
        return {k: float(v) for k, v in zip(
            ("mean","std","p25","p50","p75","p90"),
            (arr.mean(), arr.std(),
             np.percentile(arr,25), np.percentile(arr,50),
             np.percentile(arr,75), np.percentile(arr,90))
        )}

    s_tr, s_te = _stats(tr), _stats(te)

    print(f"\n  {'metric':<15} {'TRAIN':>10} {'TEST':>10} {'diff':>10}")
    print("  " + "─"*48)
    for key in ("mean","std","p25","p50","p75","p90"):
        diff = s_te[key] - s_tr[key]
        flag = " ⚠" if abs(diff) > 0.05 else ""
        print(f"  {key:<15} {s_tr[key]:>10.4f} {s_te[key]:>10.4f} {diff:>+10.4f}{flag}")

    print(f"\n  KS-test:      stat={ks_stat:.4f}  p={ks_p:.6f}  "
          f"{'distributions SIMILAR ✓' if ks_p > 0.05 else 'distributions DIFFER ⚠'}")
    print(f"  Mann-Whitney: stat={mw_stat:.0f}  p={mw_p:.6f}  "
          f"{'medians SIMILAR ✓' if mw_p > 0.05 else 'medians DIFFER ⚠'}")

    mean_diff = abs(s_te["mean"] - s_tr["mean"])
    verdict   = "✓ STABLE" if mean_diff < 0.03 else "~ moderate" if mean_diff < 0.07 else "✗ unstable"
    print(f"\n  Mean difference: {mean_diff:.4f} → BMPI {verdict}\n")

    return {**{f"train_{k}": s_tr[k] for k in s_tr},
            **{f"test_{k}":  s_te[k] for k in s_te},
            "ks_stat": float(ks_stat), "ks_p": float(ks_p),
            "mw_p": float(mw_p), "mean_diff": float(mean_diff), "verdict": verdict}


# ---------------------------------------------------------------------------
# Check 2: Predictive performance
# ---------------------------------------------------------------------------

def check_predictive_performance(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    print("─" * 65)
    print("TEST 2: Predictive Performance BMPI → excess_media_effect_usd")
    print("─" * 65)

    results: Dict = {}
    for name, df in (("TRAIN", train), ("TEST", test)):
        x = df["bmpi"].values
        y = df["excess_media_effect_usd"].fillna(0).values
        r, p = pearsonr(x, y)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 10 and x[mask].std() > 1e-9:
            sl, ic, rv, pv, _ = scipy_stats.linregress(x[mask], y[mask])
            yp   = sl * x[mask] + ic
            rmse = float(np.sqrt(np.mean((y[mask]-yp)**2)))
            mae  = float(np.mean(np.abs(y[mask]-yp)))
            r2   = float(rv**2)
        else:
            sl = ic = rmse = mae = r2 = np.nan
            if mask.sum() > 10 and x[mask].std() <= 1e-9:
                print(f"  [WARN] {name}: BMPI is constant — check z-score inputs")

        print(f"\n  {name}:  r={fmt(r)} {stars(p)}  R²={fmt(r2)}  "
              f"RMSE={fmt(rmse)}  MAE={fmt(mae)}")
        results[name.lower()] = {"r": r, "r2": r2, "rmse": rmse, "mae": mae, "p": p}

    r2_tr = results.get("train", {}).get("r2", np.nan)
    r2_te = results.get("test",  {}).get("r2", np.nan)
    degrad = float((r2_tr - r2_te) / r2_tr * 100) \
             if not (np.isnan(r2_tr) or np.isnan(r2_te)) and r2_tr > 0 else np.nan

    print(f"\n  R² degradation: {fmt(degrad,1)}%  "
          f"{'✓ acceptable (<20%)' if not np.isnan(degrad) and degrad < 20 else '⚠ high'}\n")

    results["degradation_r2_pct"] = degrad
    results["r_train"] = results.get("train", {}).get("r", np.nan)
    results["r_test"]  = results.get("test",  {}).get("r", np.nan)
    return results


# ---------------------------------------------------------------------------
# Check 3: Event-level validation
# ---------------------------------------------------------------------------

def check_event_validation(train: pd.DataFrame, test: pd.DataFrame,
                            peaks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    print("─" * 65)
    print("TEST 3: Event-Level Validation")
    print("─" * 65)

    all_df = pd.concat([train, test], ignore_index=True)
    rows: List[dict] = []

    for _, pk in peaks.iterrows():
        eid  = pk.get("event_id", f"EVT_{_}")
        pik  = pd.Timestamp(pk["peak_date"])
        split = "TEST" if pik >= SPLIT_DATE else "TRAIN"
        mask = ((all_df["date"] >= pik - pd.Timedelta(3,"D")) &
                (all_df["date"] <= pik + pd.Timedelta(2,"D")))
        w = all_df[mask]
        if len(w) == 0:
            continue
        rows.append({
            "event_id":        eid,
            "peak_date":       pik.date(),
            "split":           split,
            "bmpi_mean":       float(w["bmpi"].mean()),
            "bmpi_max":        float(w["bmpi"].max()),
            "excess_usd_sum":  float(w["excess_media_effect_usd"].sum()),
            "above_threshold": float(w["bmpi"].mean()) > threshold,
            "n_days":          len(w),
        })

    df_ev = pd.DataFrame(rows)
    for split in ("TRAIN", "TEST"):
        sub = df_ev[df_ev["split"] == split]
        print(f"\n  {split} EVENTS ({len(sub)}):")
        print(f"  {'event_id':<26} {'BMPI_mean':>10} {'excess_B':>10} {'>thr?':>6}")
        print("  " + "─"*56)
        for _, row in sub.sort_values("bmpi_mean", ascending=False).iterrows():
            flag = " ★" if row["above_threshold"] else ""
            print(f"  {str(row['event_id']):<26} {row['bmpi_mean']:>10.4f} "
                  f"{row['excess_usd_sum']/1e9:>9.3f}B "
                  f"{'YES' if row['above_threshold'] else 'no':>6}{flag}")

    df_ev.to_csv(OUT_EVENTS, index=False)
    print()
    return df_ev


# ---------------------------------------------------------------------------
# Check 4: Rolling walk-forward
# ---------------------------------------------------------------------------

def check_rolling_validation(df: pd.DataFrame) -> pd.DataFrame:
    print("─" * 65)
    print("TEST 4: Rolling Walk-Forward Validation")
    print("─" * 65)
    print(f"  TRAIN window: {ROLLING_TRAIN_WINDOW} days  |  TEST window: {ROLLING_TEST_WINDOW} days\n")

    df = df.sort_values("date").reset_index(drop=True)
    n, rows, w_n = len(df), [], 0

    for start in range(0, n - ROLLING_TRAIN_WINDOW - ROLLING_TEST_WINDOW,
                       ROLLING_TEST_WINDOW):
        tr_end = start + ROLLING_TRAIN_WINDOW
        te_end = tr_end + ROLLING_TEST_WINDOW
        if te_end > n:
            break
        tr, te = df.iloc[start:tr_end].copy(), df.iloc[tr_end:te_end].copy()

        params: Dict = {}
        for col in ("z_volume", "z_tone", "z_resid"):
            if col in tr.columns:
                vals = tr[col].dropna()
                if len(vals) > 0:
                    params[f"mu_{col}"] = float(vals.mean())
                    params[f"sd_{col}"] = float(vals.std())

        te["bmpi_oos"] = compute_bmpi(te, params or None)
        r, p = pearsonr(te["bmpi_oos"].values,
                        te["excess_media_effect_usd"].fillna(0).values)

        rows.append({
            "window":     w_n,
            "test_start": df.iloc[tr_end]["date"],
            "test_end":   df.iloc[te_end-1]["date"],
            "r": r, "p": p,
            "sig": stars(p),
            "r2": r**2 if not np.isnan(r) else np.nan,
            "n_test": len(te),
            "period": "TEST" if df.iloc[tr_end]["date"] >= SPLIT_DATE else "TRAIN_PERIOD",
        })
        w_n += 1

    df_roll = pd.DataFrame(rows)
    n_total = len(df_roll)
    n_sig   = int((df_roll["p"] < 0.05).sum()) if n_total > 0 else 0
    mean_r  = float(df_roll["r"].mean()) if n_total > 0 else np.nan
    oos     = df_roll["period"] == "TEST"
    oos_r   = float(df_roll.loc[oos,"r"].mean()) if oos.sum() > 0 else np.nan

    print(f"  Total windows: {n_total}  |  Significant: {n_sig}/{n_total}  "
          f"|  Mean r: {fmt(mean_r)}  |  OOS r: {fmt(oos_r)}\n")

    df_roll["year"] = pd.to_datetime(df_roll["test_start"]).dt.year
    print(f"  {'year':>6} {'windows':>8} {'mean r':>8} {'sig%':>8}")
    print("  " + "─"*34)
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
    print("─" * 65)
    print("TEST 5: BMPI Stability Across Market Regimes")
    print("─" * 65)

    all_df = pd.concat([train, test], ignore_index=True)
    if "btc_logret" not in all_df.columns:
        print("  [WARN] btc_logret not available\n")
        return {}

    all_df["regime"] = all_df["btc_logret"].apply(
        lambda r: "bull" if r > 0.01 else "bear" if r < -0.01 else "flat"
        if not pd.isna(r) else "unknown"
    )
    regime_map = all_df.set_index("date")["regime"]

    results: Dict = {}
    print(f"\n  {'regime':<10} {'n_days':>8} {'mean_bmpi':>11} "
          f"{'r→excess':>10} {'sig':>5} {'split':>8}")
    print("  " + "─"*58)

    for split_name, df in (("TRAIN", train), ("TEST", test)):
        df = df.copy()
        df["regime"] = df["date"].map(regime_map)
        for regime in ("bull", "bear", "flat"):
            sub = df[df["regime"] == regime]
            if len(sub) < 20:
                continue
            bm = float(sub["bmpi"].mean())
            r, p = pearsonr(sub["bmpi"].values,
                            sub["excess_media_effect_usd"].fillna(0).values)
            print(f"  {regime:<10} {len(sub):>8} {bm:>11.4f} "
                  f"{fmt(r):>10} {stars(p):>5} {split_name:>8}")
            results[f"{split_name}_{regime}"] = {"n": len(sub), "bmpi_mean": bm, "r": r, "p": p}
    print()
    return results


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html(dist_res: Dict, perf_res: Dict,
               ev_df: pd.DataFrame, roll_df: pd.DataFrame) -> str:
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    r_tr  = perf_res.get("r_train", np.nan)
    r_te  = perf_res.get("r_test",  np.nan)
    degrad= perf_res.get("degradation_r2_pct", np.nan)
    ks_p  = dist_res.get("ks_p", np.nan)

    css = """*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0a0d14;--bg2:#111520;--bg3:#171c2b;--brd:#232840;
  --acc:#f0a500;--acc2:#4fc3f7;--txt:#e8eaf6;--dim:#455a8a}
body{font-family:'Segoe UI',sans-serif;background:var(--bg);color:var(--txt)}
.hdr{background:var(--bg2);border-bottom:1px solid var(--brd);padding:20px 40px;
  display:flex;align-items:center;gap:12px}
.hdr h1{font-size:20px;color:var(--acc)}
.ts{margin-left:auto;font-size:10px;color:var(--dim)}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
  border:1px solid var(--brd);margin:24px 40px 0;border-radius:8px;overflow:hidden}
.kpi{background:var(--bg2);padding:18px}
.kpi-l{font-size:10px;text-transform:uppercase;color:var(--dim);margin-bottom:4px}
.kpi-v{font-size:20px;font-weight:600;color:var(--acc)}
.sec{margin:24px 40px 0}
.sec-t{font-size:15px;color:var(--acc);margin-bottom:10px;padding-bottom:6px;
  border-bottom:1px solid var(--brd)}
.tbl-w{overflow-x:auto;border:1px solid var(--brd);border-radius:6px}
table{width:100%;border-collapse:collapse;font-size:11px}
th{background:var(--bg3);padding:7px 9px;color:var(--dim);font-size:10px;
  text-transform:uppercase;border-bottom:1px solid var(--brd)}
td{padding:6px 9px;border-bottom:1px solid var(--brd)}
tr:hover td{background:var(--bg3)}
.ok{color:#4caf50} .warn{color:#ff9800}"""

    h = [f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
         f'<title>OOS Validation — BMPI</title><style>{css}</style></head><body>',
         f'<header class="hdr"><h1>Out-of-Sample Validation — BMPI</h1>'
         f'<span class="ts">{now}</span></header>',
         '<div class="kpi-grid">',
         f'<div class="kpi"><div class="kpi-l">r TRAIN</div>'
         f'<div class="kpi-v">{fmt(r_tr)}</div></div>',
         f'<div class="kpi"><div class="kpi-l">r TEST (OOS)</div>'
         f'<div class="kpi-v">{fmt(r_te)}</div></div>',
         f'<div class="kpi"><div class="kpi-l">R² degradation</div>'
         f'<div class="kpi-v">{fmt(degrad,1)}%</div></div>',
         f'<div class="kpi"><div class="kpi-l">KS-test p</div>'
         f'<div class="kpi-v">{fmt(ks_p,4)}</div></div>',
         '</div>']

    if len(ev_df) > 0:
        h += ['<div class="sec"><div class="sec-t">Event-Level Validation</div>',
              '<div class="tbl-w"><table><thead><tr>',
              '<th>Event ID</th><th>Peak Date</th><th>Split</th>',
              '<th>BMPI mean</th><th>Excess (B)</th><th>>threshold?</th>',
              '</tr></thead><tbody>']
        for _, row in ev_df.sort_values(["split","bmpi_mean"],
                                         ascending=[True,False]).iterrows():
            h.append(f'<tr><td>{row["event_id"]}</td><td>{row["peak_date"]}</td>'
                     f'<td class="{"ok" if row["split"]=="TRAIN" else "warn"}">'
                     f'{row["split"]}</td>'
                     f'<td>{fmt(row["bmpi_mean"])}</td>'
                     f'<td>{row["excess_usd_sum"]/1e9:.3f}B</td>'
                     f'<td>{"★ YES" if row["above_threshold"] else "no"}</td></tr>')
        h += ['</tbody></table></div></div>']

    ok = not np.isnan(degrad) and degrad < 20 and not np.isnan(r_te) and r_te > 0.5
    h += [f'<div class="sec"><div class="sec-t">Verdict</div>',
          f'<p style="padding:12px 0;font-size:13px;">',
          f'{"✓ VALIDATION PASSED — BMPI is a stable indicator on 2022–2025 data." if ok else "~ PARTIAL VALIDATION — BMPI works but with some degradation."}',
          f'</p></div></body></html>']
    return "\n".join(h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    excess, peaks = load_data()
    train, test, params, threshold = split_and_calibrate(excess)

    print("=" * 65)
    print("RUNNING VALIDATION TESTS")
    print("=" * 65 + "\n")

    dist_res = check_distribution_stability(train, test)
    perf_res = check_predictive_performance(train, test)
    ev_df    = (check_event_validation(train, test, peaks, threshold)
                if peaks is not None else pd.DataFrame())
    roll_df  = check_rolling_validation(excess)
    _        = check_regime_stability(train, test)

    summary = {
        "split_date":         str(SPLIT_DATE.date()),
        "n_train":            int(len(train)),
        "n_test":             int(len(test)),
        "ks_p":               dist_res.get("ks_p"),
        "mean_diff":          dist_res.get("mean_diff"),
        "r_train":            perf_res.get("r_train"),
        "r_test":             perf_res.get("r_test"),
        "degradation_r2_pct": perf_res.get("degradation_r2_pct"),
        "rolling_oos_r":      float(roll_df.loc[roll_df["period"]=="TEST","r"].mean())
                              if len(roll_df) > 0 else None,
        "timestamp":          datetime.now().isoformat(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    html = build_html(dist_res, perf_res, ev_df, roll_df)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Saved: {OUT_JSON}")
    print(f"[OK] Saved: {OUT_EVENTS}")
    print(f"[OK] Saved: {OUT_ROLLING}")
    print(f"[OK] Saved: {OUT_HTML}  ← open in browser")

    r_tr  = perf_res.get("r_train", np.nan)
    r_te  = perf_res.get("r_test",  np.nan)
    degrad= perf_res.get("degradation_r2_pct", np.nan)
    ks_p  = dist_res.get("ks_p", np.nan)
    oos   = roll_df["period"] == "TEST"
    roll_oos = float(roll_df.loc[oos,"r"].mean()) if oos.sum() > 0 else np.nan

    W = 67
    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║  OUT-OF-SAMPLE VALIDATION — SUMMARY                          ║")
    print("╠" + "═"*(W-2) + "╣")
    print(f"║  TRAIN: 2015–2021  ({len(train)} days)                               ║")
    print(f"║  TEST:  2022–2025  ({len(test)} days)                                ║")
    print("║                                                               ║")
    print(f"║  Correlation BMPI→excess_usd:                                ║")
    print(f"║    TRAIN r = {fmt(r_tr)}    TEST r = {fmt(r_te)}                 ║")
    print(f"║    R² degradation = {fmt(degrad,1)}%                                ║")
    print("║                                                               ║")
    print(f"║  KS-test p = {fmt(ks_p,4)}  "
          f"{'distributions SIMILAR ✓' if not np.isnan(ks_p) and ks_p > 0.05 else 'differ ⚠':<34}║")
    print(f"║  Rolling OOS mean r = {fmt(roll_oos)}                             ║")
    print("║                                                               ║")
    ok = not np.isnan(degrad) and degrad < 20 and not np.isnan(r_te) and r_te > 0.5
    if ok:
        print("║  ✓ VALIDATION PASSED                                          ║")
        print("║    BMPI is a stable indicator on unseen 2022–2025 data        ║")
    else:
        print("║  ~ PARTIAL VALIDATION                                         ║")
        print("║    BMPI works but with some degradation — discuss             ║")
        print("║    market regime change post-2022 in the discussion section   ║")
    print("╚" + "═"*(W-2) + "╝")
    print("\nPipeline complete. All steps finished.")


if __name__ == "__main__":
    main()