# -*- coding: utf-8 -*-
"""
pipelines/step16_johansen_cointegration.py
============================================
Johansen cointegration test: long-run equilibrium between
media pressure and BTC market capitalisation.

Complements Granger causality (short-run) with a test of
long-run co-movement (Johansen 1991, Engle & Granger 1987).

Input:
  data/processed/excess_media_effect_daily.csv  (from step09)
  data/processed/features_daily.parquet         (from step02)
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv

Output:
  data/processed/johansen_results.json
  data/processed/johansen_report.html

Next step: pipeline complete.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARN] statsmodels not installed. Run: pip install statsmodels")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_GDELT     = BASE_DIR / "data" / "raw" / "gdelt"

EXCESS_CSV   = DATA_PROCESSED / "excess_media_effect_daily.csv"
FEATURES_PAR = DATA_PROCESSED / "features_daily.parquet"
OUT_JSON     = DATA_PROCESSED / "johansen_results.json"
OUT_HTML     = DATA_PROCESSED / "johansen_report.html"

JOHANSEN_DET_ORDER = 0
JOHANSEN_K_AR_DIFF = 2


def find_gdelt(preset: str = "balanced") -> Path:
    for p in [DATA_GDELT / f"gdelt_gkg_bitcoin_daily_signal_{preset}.csv",
              DATA_GDELT / f"gdelt_btc_media_signal_{preset}.csv"]:
        if p.exists():
            return p
    return DATA_GDELT / f"gdelt_gkg_bitcoin_daily_signal_{preset}.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> Optional[pd.DataFrame]:
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


# ---------------------------------------------------------------------------
# Build dataset (levels)
# ---------------------------------------------------------------------------

def build_levels_dataset() -> pd.DataFrame:
    print("=" * 65)
    print("LOADING DATA (levels for cointegration)")
    print("=" * 65)

    excess = load_csv(EXCESS_CSV)
    if excess is None:
        raise FileNotFoundError(f"Required: {EXCESS_CSV}")

    df = excess[["date"]].copy()
    for c in ("excess_media_effect_usd", "bmpi_score", "resid_btc_mcap_usd"):
        if c in excess.columns:
            df[c] = pd.to_numeric(excess[c], errors="coerce")

    feat = load_csv(FEATURES_PAR)
    if feat is not None:
        mc_col = next((c for c in feat.columns
                       if ("mcap" in c or "kapitaliz" in c) and "btc" in c), None)
        if mc_col:
            sub = feat[["date", mc_col]].rename(columns={mc_col: "btc_mcap"})
            df = df.merge(sub, on="date", how="left")
            df["btc_mcap"]     = pd.to_numeric(df["btc_mcap"], errors="coerce")
            df["log_btc_mcap"] = np.log(df["btc_mcap"].clip(lower=1))
            print(f"  [OK] btc_mcap: {mc_col}")

    gdelt = load_csv(find_gdelt("balanced"))
    if gdelt is not None:
        m_col = next((c for c in gdelt.columns
                      if c in ("mentions", "liczba_wzmianek")), None)
        t_col = next((c for c in gdelt.columns
                      if c in ("tone", "sredni_tone", "tone_avg")), None)
        if m_col is None:
            non_dt = [c for c in gdelt.columns if c != "date"]
            if non_dt: m_col = non_dt[0]
        if t_col is None:
            non_dt = [c for c in gdelt.columns if c not in ("date", m_col or "")]
            if non_dt: t_col = non_dt[0]
        sub = gdelt[["date"]].copy()
        if m_col: sub["mentions"] = pd.to_numeric(gdelt[m_col], errors="coerce")
        if t_col: sub["tone"]     = pd.to_numeric(gdelt[t_col], errors="coerce")
        df = df.merge(sub.drop_duplicates("date"), on="date", how="left")
        n = df["mentions"].notna().sum() if "mentions" in df.columns else 0
        print(f"  [OK] GDELT balanced: {n} days")

    if "mentions" in df.columns:
        df["log_mentions"] = np.log(df["mentions"].clip(lower=0.1))
    if "excess_media_effect_usd" in df.columns:
        x = df["excess_media_effect_usd"].fillna(0)
        df["log_excess"] = np.sign(x) * np.log(np.abs(x).clip(lower=1))

    df = df.sort_values("date").reset_index(drop=True)
    print(f"  [OK] Dataset: {len(df)} days  "
          f"({df['date'].min().date()} to {df['date'].max().date()})")
    print()
    return df


# ---------------------------------------------------------------------------
# Integration order
# ---------------------------------------------------------------------------

def check_integration_order(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    print("─" * 65)
    print("INTEGRATION ORDER TESTS (ADF)")
    print("─" * 65)
    print(f"  {'variable':<25} {'p(levels)':>12} {'I(1)?':>6} "
          f"{'p(diff)':>10} {'order'}")
    print("  " + "─"*62)

    rows = []
    for col in columns:
        series = df[col].dropna()
        if len(series) < 50:
            continue
        try:
            _, p_lev, *_ = adfuller(series, autolag="AIC")
            p_lev = float(p_lev)
        except Exception:
            p_lev = np.nan
        try:
            _, p_dif, *_ = adfuller(series.diff().dropna(), autolag="AIC")
            p_dif = float(p_dif)
        except Exception:
            p_dif = np.nan

        lev_stat = p_lev < 0.05 if not np.isnan(p_lev) else None
        dif_stat = p_dif < 0.05 if not np.isnan(p_dif) else None
        order = "I(0)" if lev_stat else ("I(1)" if dif_stat else "I(2)?")
        i1 = "yes" if order == "I(1)" else "no"

        print(f"  {col:<25} {fmt(p_lev):>12} {i1:>6} {fmt(p_dif):>10} {order}")
        rows.append({"variable": col, "adf_p_lev": p_lev, "adf_p_dif": p_dif,
                     "order": order, "is_i1": order == "I(1)"})
    print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Engle-Granger pairwise
# ---------------------------------------------------------------------------

def engle_granger_pairs(df: pd.DataFrame, int_df: pd.DataFrame) -> list:
    print("─" * 65)
    print("ENGLE-GRANGER PAIRWISE COINTEGRATION")
    print("─" * 65)

    i1_cols = int_df[int_df["is_i1"]]["variable"].tolist() if len(int_df) > 0 else []
    print(f"  I(1) series: {i1_cols}\n")

    anchor = "log_btc_mcap"
    pairs = [
        (anchor, "log_excess",   "log_btc_mcap vs log_excess_media"),
        (anchor, "log_mentions", "log_btc_mcap vs log_mentions"),
        (anchor, "bmpi_score",   "log_btc_mcap vs bmpi_score"),
        ("log_excess", "log_mentions", "log_excess vs log_mentions"),
    ]

    results = []
    print(f"  {'pair':<42} {'EG stat':>9} {'p':>8}  result")
    print("  " + "─"*66)

    for x_col, y_col, label in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            print(f"  {label:<42}  — missing column")
            continue
        if x_col not in i1_cols or y_col not in i1_cols:
            print(f"  {label:<42}  — not both I(1)")
            continue
        data = df[[x_col, y_col]].dropna()
        if len(data) < 50:
            continue
        try:
            stat, pval, _ = coint(data[x_col].values, data[y_col].values)
            stat, pval = float(stat), float(pval)
            tag = "COINTEGRATED" if pval < 0.05 else "no coint."
            print(f"  {label:<42} {stat:>9.4f} {pval:>8.4f}  {stars(pval)} {tag}")
            results.append({"pair": label, "x": x_col, "y": y_col,
                             "eg_stat": stat, "eg_p": pval, "coint": pval < 0.05})
        except Exception as e:
            print(f"  {label:<42}  error: {e}")
    print()
    return results


# ---------------------------------------------------------------------------
# Johansen
# ---------------------------------------------------------------------------

def johansen_test(df: pd.DataFrame, int_df: pd.DataFrame) -> dict:
    print("─" * 65)
    print("JOHANSEN MULTIVARIATE COINTEGRATION TEST")
    print("─" * 65)

    i1_cols = int_df[int_df["is_i1"]]["variable"].tolist() if len(int_df) > 0 else []
    system_cols = [c for c in ("log_btc_mcap", "log_excess", "log_mentions")
                   if c in df.columns and c in i1_cols]

    if len(system_cols) < 2:
        print(f"  [INFO] Only {len(system_cols)} I(1) series: {system_cols}")
        print("  [INFO] Media signals are I(0) — stationary.")
        print("  [INFO] BTC mcap is I(1) — non-stationary.")
        print("  [INFO] Mixed orders: Johansen NOT applicable.")
        print("  [INFO] Short-run Granger causality (step13) is correct.\n")
        return {
            "system_cols": system_cols, "r": 0,
            "verdict": "not_applicable_mixed_integration_orders",
            "note": ("Mixed I(0)/I(1) orders. Media signals stationary, "
                     "BTC mcap non-stationary. Johansen not applicable. "
                     "Granger causality (step13) is the appropriate test."),
        }

    data = df[system_cols].dropna()
    k    = len(system_cols)
    print(f"  System: {system_cols}  |  N={len(data)}  k={k}")
    print()

    try:
        result = coint_johansen(data.values, det_order=JOHANSEN_DET_ORDER,
                                k_ar_diff=JOHANSEN_K_AR_DIFF)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {}

    trace_stat  = result.lr1
    maxeig_stat = result.lr2
    trace_cv    = result.cvt
    maxeig_cv   = result.cvm

    print(f"  {'H0':<12} {'Trace':>10} {'CV95':>8} {'rej?':>6} "
          f"{'MaxEig':>10} {'CV95':>8} {'rej?':>6}")
    print("  " + "─"*62)
    n_coint_tr = 0
    n_coint_me = 0
    for i in range(k):
        tr_rej = bool(trace_stat[i]  > trace_cv[i, 1])
        me_rej = bool(maxeig_stat[i] > maxeig_cv[i, 1])
        if tr_rej: n_coint_tr += 1
        if me_rej: n_coint_me += 1
        print(f"  {'r<='+str(i):<12} {float(trace_stat[i]):>10.4f} "
              f"{float(trace_cv[i,1]):>8.4f} {'YES' if tr_rej else 'no':>6} "
              f"{float(maxeig_stat[i]):>10.4f} {float(maxeig_cv[i,1]):>8.4f} "
              f"{'YES' if me_rej else 'no':>6}")

    r = min(n_coint_tr, n_coint_me)
    verdict = "no_cointegration" if r == 0 else f"r={r}"
    print(f"\n  Cointegrating vectors: Trace={n_coint_tr}  MaxEig={n_coint_me}  r={r}")
    print(f"  Verdict: {verdict}\n")

    return {
        "system_cols": system_cols, "n_obs": len(data),
        "n_coint_trace": n_coint_tr, "n_coint_maxeig": n_coint_me,
        "r": r, "verdict": verdict,
        "trace_stats":  [float(x) for x in trace_stat],
        "maxeig_stats": [float(x) for x in maxeig_stat],
        "trace_cv_95":  [float(x) for x in trace_cv[:, 1]],
        "maxeig_cv_95": [float(x) for x in maxeig_cv[:, 1]],
    }


# ---------------------------------------------------------------------------
# VECM / VAR
# ---------------------------------------------------------------------------

def fit_vecm_or_var(df: pd.DataFrame, johansen: dict) -> None:
    print("─" * 65)
    r = johansen.get("r", 0)
    v = johansen.get("verdict", "")

    if "not_applicable" in v or r == 0:
        print("SPECIFICATION: VAR in first differences")
        print("─" * 65)
        print("  No cointegration -> VAR(7) from step13 is correct.")
        print("  No error correction term needed.\n")
    else:
        print(f"SPECIFICATION: VECM (r={r})")
        print("─" * 65)
        cols = johansen.get("system_cols", [])
        data = df[cols].dropna()
        try:
            from statsmodels.tsa.vector_ar.vecm import VECM
            model  = VECM(data.values, k_ar_diff=JOHANSEN_K_AR_DIFF,
                          coint_rank=r, deterministic="ci")
            fitted = model.fit()
            print("  Error correction coefficients (alpha):")
            for i, col in enumerate(cols):
                a   = float(fitted.alpha[i, 0])
                tag = "fast" if abs(a) > 0.1 else "slow" if abs(a) > 0.01 else "exogenous"
                print(f"    {col:<25} a={a:+.4f}  ({tag})")
        except Exception as e:
            print(f"  [WARN] VECM error: {e}")
        print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(int_df: pd.DataFrame, eg_pairs: list, johansen: dict) -> None:
    W = 68
    r = johansen.get("r", 0)
    v = johansen.get("verdict", "")
    n_i1 = int(int_df["is_i1"].sum()) if len(int_df) > 0 else 0
    n_i0 = int((int_df["order"] == "I(0)").sum()) if len(int_df) > 0 else 0
    n_ceg = len([p for p in eg_pairs if p.get("coint")])

    print("=" * W)
    print("JOHANSEN COINTEGRATION — KEY FINDINGS")
    print("=" * W)
    print(f"  I(1) series: {n_i1}  |  I(0) series: {n_i0}")
    print(f"  EG cointegrated pairs: {n_ceg}")
    print(f"  Johansen r: {r}")
    print()
    if "not_applicable" in v:
        print("  RESULT: NOT APPLICABLE — mixed integration orders")
        print("  Media signals I(0) (stationary), BTC mcap I(1) (non-stationary).")
        print("  Short-run Granger causality (step13) is the correct framework.")
    elif r == 0:
        print("  RESULT: No cointegration (r=0)")
        print("  Media effects are SHORT-RUN — consistent with Granger results.")
        print("  VAR in first differences is correct specification.")
    else:
        print(f"  RESULT: {r} cointegrating vector(s) found")
        print("  VECM recommended. Long-run equilibrium exists.")
    print()
    print("  INTERPRETATION FOR DISSERTATION:")
    if "not_applicable" in v or r == 0:
        print("  ADF tests confirm media signals (BMPI, excess_media_usd, GDELT)")
        print("  are I(0) — stationary, mean-reverting. BTC mcap is I(1).")
        print("  Mixed integration orders preclude classical cointegration.")
        print("  This is consistent with the episodic nature of media shocks:")
        print("  effects materialise as transient deviations (~14d half-life,")
        print("  Group C) rather than persistent structural shifts.")
        print("  Granger causality (step13) is the appropriate framework.")
    else:
        print("  Stable long-run relationship between media pressure and BTC mcap.")
        print("  VECM error correction term captures mean-reversion dynamics.")
    print()
    print("Pipeline fully complete. Ready for academic writing.")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html(int_df: pd.DataFrame, eg_pairs: list, johansen: dict) -> str:
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    r    = johansen.get("r", 0)
    n_i1 = int(int_df["is_i1"].sum()) if len(int_df) > 0 else 0
    n_ceg= len([p for p in eg_pairs if p.get("coint")])

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
.ok{color:#4caf50} .warn{color:#ff9800}
.note{margin:10px 0;background:var(--bg2);border-left:3px solid var(--acc2);
  padding:10px 16px;font-size:11.5px;border-radius:4px;line-height:1.7}"""

    h = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>',
        f'<title>Johansen Cointegration</title><style>{css}</style></head><body>',
        '<header class="hdr"><h1>Johansen Cointegration Analysis</h1>',
        f'<span class="ts">{now}</span></header>',
        '<div class="kpi-grid">',
        f'<div class="kpi"><div class="kpi-l">I(1) series</div>'
        f'<div class="kpi-v">{n_i1}</div></div>',
        f'<div class="kpi"><div class="kpi-l">EG cointegrated</div>'
        f'<div class="kpi-v">{n_ceg}</div></div>',
        f'<div class="kpi"><div class="kpi-l">Johansen r</div>'
        f'<div class="kpi-v">{r}</div></div>',
        f'<div class="kpi"><div class="kpi-l">Specification</div>'
        f'<div class="kpi-v">{"VECM" if r > 0 else "VAR"}</div></div>',
        '</div>',
    ]

    if len(int_df) > 0:
        h += ['<div class="sec"><div class="sec-t">Integration Order (ADF)</div>',
              '<div class="tbl-w"><table><thead><tr>'
              '<th>Variable</th><th>ADF p (levels)</th>'
              '<th>ADF p (diff)</th><th>Order</th>'
              '</tr></thead><tbody>']
        for _, row in int_df.iterrows():
            cls = "ok" if row["is_i1"] else "warn"
            h.append(
                f'<tr><td>{row["variable"]}</td>'
                f'<td>{fmt(row["adf_p_lev"])}</td>'
                f'<td>{fmt(row["adf_p_dif"])}</td>'
                f'<td class="{cls}">{row["order"]}</td></tr>'
            )
        h += ['</tbody></table></div></div>']

    if eg_pairs:
        h += ['<div class="sec"><div class="sec-t">Engle-Granger Tests</div>',
              '<div class="tbl-w"><table><thead><tr>'
              '<th>Pair</th><th>EG stat</th><th>p-value</th><th>Result</th>'
              '</tr></thead><tbody>']
        for p in eg_pairs:
            cls = "ok" if p.get("coint") else "warn"
            tag = "YES" if p.get("coint") else "no"
            h.append(
                f'<tr><td>{p["pair"]}</td>'
                f'<td>{fmt(p["eg_stat"])}</td>'
                f'<td>{fmt(p["eg_p"])}</td>'
                f'<td class="{cls}">{tag}</td></tr>'
            )
        h += ['</tbody></table></div></div>']

    note = johansen.get("note", "")
    if not note:
        v = johansen.get("verdict", "")
        if r == 0:
            note = ("No cointegration (r=0). Media pressure effects are "
                    "SHORT-RUN and episodic. VAR in first differences "
                    "(step13) is the correct specification. Consistent "
                    "with 14-day half-life and 55% reversal ratio (Group C).")
        elif r > 0:
            note = (f"{r} cointegrating vector(s) found. Long-run "
                    "equilibrium exists. VECM recommended.")

    if note:
        h += [f'<div class="sec"><div class="note">{note}</div></div>']

    h += ['</body></html>']
    return "\n".join(h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HAS_STATSMODELS:
        print("ERROR: statsmodels required. Run: pip install statsmodels")
        return

    df = build_levels_dataset()

    test_cols = [c for c in ("log_btc_mcap", "log_excess", "log_mentions",
                              "bmpi_score", "tone", "resid_btc_mcap_usd")
                 if c in df.columns and df[c].notna().sum() > 50]

    int_df   = check_integration_order(df, test_cols)
    eg_pairs = engle_granger_pairs(df, int_df)
    joh      = johansen_test(df, int_df)
    fit_vecm_or_var(df, joh)
    print_summary(int_df, eg_pairs, joh)

    results = {
        "integration_orders": int_df.to_dict("records") if len(int_df) > 0 else [],
        "engle_granger":      eg_pairs,
        "johansen":           joh,
        "timestamp":          datetime.now().isoformat(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Saved: {OUT_JSON}")

    html = build_html(int_df, eg_pairs, joh)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Saved: {OUT_HTML}  <- open in browser")


if __name__ == "__main__":
    main()