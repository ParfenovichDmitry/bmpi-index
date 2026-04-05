# -*- coding: utf-8 -*-
"""
pipelines/step13_granger_causality.py
========================================
Granger causality test: does media activity PRECEDE BTC price moves?

Tests X Granger-causes Y if past values of X improve prediction of Y
beyond Y's own history. Standard approach in financial econometrics
(Sims 1980, Hamilton 1994, Lütkepohl 2005).

Directions tested:
  mentions        → btc_logret           (media volume → BTC price)
  mentions        → excess_media_usd     (media → anomalous pressure)
  tone            → btc_logret           (sentiment → BTC price)
  tone            → excess_media_usd     (sentiment → anomalous pressure)
  excess_media_usd → btc_logret          (anomalous pressure → price)
  btc_logret      → mentions             (reverse: price → media?)
  bmpi_score      → btc_logret           (BMPI → BTC price)

Input:
  data/processed/excess_media_effect_daily.csv  (from step09)
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv
  data/processed/features_daily.parquet         (from step02)

Output:
  data/processed/granger_results.csv
  data/processed/granger_stationarity.csv
  data/processed/granger_report.html

Next step: step14_oos_validation.py
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
    from statsmodels.tsa.api import VAR
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
OUT_RESULTS  = DATA_PROCESSED / "granger_results.csv"
OUT_STATION  = DATA_PROCESSED / "granger_stationarity.csv"
OUT_HTML     = DATA_PROCESSED / "granger_report.html"

LAGS         = [1, 2, 3, 5, 7, 14]
VAR_MAX_LAGS = 10


def find_gdelt(preset: str = "balanced") -> Path:
    for fname in [
        f"gdelt_gkg_bitcoin_daily_signal_{preset}.csv",
        f"gdelt_btc_media_signal_{preset}.csv",
    ]:
        p = DATA_GDELT / fname
        if p.exists():
            return p
    return DATA_GDELT / f"gdelt_gkg_bitcoin_daily_signal_{preset}.csv"


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
    if pd.isna(p): return "   "
    return "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    print("=" * 65)
    print("LOADING AND PREPARING DATA")
    print("=" * 65)

    # Load excess media effect (required)
    excess = load_csv(EXCESS_CSV)
    if excess is None:
        raise FileNotFoundError(
            f"Required file not found: {EXCESS_CSV}\n"
            "Run step09_fake_classification.py first."
        )
    print(f"  [OK] excess_daily:  {len(excess)} days")

    df = excess[["date"]].copy()
    for c in ("excess_media_effect_usd", "bmpi_score",
              "resid_btc_mcap_usd", "media_effect_usd"):
        if c in excess.columns:
            df[c] = pd.to_numeric(excess[c], errors="coerce")

    # Load GDELT presets
    for preset in ("balanced", "sensitive", "strong"):
        gdf = load_csv(find_gdelt(preset))
        if gdf is None:
            print(f"  [SKIP] GDELT {preset}: not found")
            continue
        m_col = next((c for c in gdf.columns
                      if c in ("mentions", "liczba_wzmianek", "mention_count")), None)
        t_col = next((c for c in gdf.columns
                      if c in ("tone", "sredni_tone", "avg_tone", "tone_avg")), None)
        # Fallback: positional
        if m_col is None:
            non_dt = [c for c in gdf.columns if c != "date"]
            if non_dt: m_col = non_dt[0]
        if t_col is None:
            non_dt = [c for c in gdf.columns if c not in ("date", m_col or "")]
            if non_dt: t_col = non_dt[0]

        sub = gdf[["date"]].copy()
        if m_col: sub[f"mentions_{preset}"] = pd.to_numeric(gdf[m_col], errors="coerce")
        if t_col: sub[f"tone_{preset}"]     = pd.to_numeric(gdf[t_col], errors="coerce")
        sub = sub.drop_duplicates("date")
        df = df.merge(sub, on="date", how="left")
        n = df[f"mentions_{preset}"].notna().sum() if f"mentions_{preset}" in df.columns else 0
        print(f"  [OK] GDELT {preset:<10} {n} days matched")

    # Primary signal: first preset with enough data
    for preset in ("balanced", "sensitive", "strong"):
        mc = f"mentions_{preset}"
        tc = f"tone_{preset}"
        if mc in df.columns and df[mc].notna().sum() > 100:
            df["mentions"] = df[mc]
            df["tone"]     = df.get(tc, pd.Series(dtype=float))
            print(f"  [OK] Primary signal: {preset} ({df['mentions'].notna().sum()} days)")
            break

    # Load features (btc_logret, btc_price)
    feat = load_csv(FEATURES_PAR)
    if feat is not None:
        logret_col = next((c for c in feat.columns if "logret" in c), None)
        price_col  = next((c for c in feat.columns
                           if "btc" in c and any(k in c for k in ("price","cena","close"))), None)
        mcap_col   = next((c for c in feat.columns
                           if ("mcap" in c or "kapitaliz" in c) and "btc" in c), None)
        sub = feat[["date"]].copy()
        if logret_col: sub["btc_logret"] = pd.to_numeric(feat[logret_col], errors="coerce")
        if price_col:  sub["btc_price"]  = pd.to_numeric(feat[price_col],  errors="coerce")
        if mcap_col:   sub["btc_mcap"]   = pd.to_numeric(feat[mcap_col],   errors="coerce")
        df = df.merge(sub, on="date", how="left")
        print(f"  [OK] Features: logret={logret_col}, price={price_col}")

    # Compute btc_logret if missing
    if "btc_logret" not in df.columns or df["btc_logret"].notna().sum() < 100:
        for src in ("btc_mcap", "btc_price"):
            if src in df.columns and df[src].notna().sum() > 100:
                df["btc_logret"] = np.log(df[src] / df[src].shift(1))
                print(f"  [INFO] btc_logret computed from {src}")
                break

    # Ensure columns exist
    for c in ("btc_logret", "excess_media_effect_usd", "bmpi_score",
              "mentions", "tone", "resid_btc_mcap_usd"):
        if c not in df.columns:
            df[c] = np.nan

    # Remove inf and clip extremes
    for c in df.select_dtypes(include=[float]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        q01, q99 = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(q01, q99)

    df = df.sort_values("date").reset_index(drop=True)
    print(f"  [OK] Dataset: {len(df)} days | "
          f"btc_logret: {df['btc_logret'].notna().sum()} | "
          f"mentions: {df['mentions'].notna().sum()}")
    print()
    return df


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def check_stationarity(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """ADF + KPSS stationarity tests. Required before Granger causality."""
    print("─" * 65)
    print("STATIONARITY TESTS (ADF + KPSS)")
    print("─" * 65)
    print(f"  {'variable':<30} {'ADF stat':>10} {'ADF p':>8} "
          f"{'verdict':>18} {'KPSS p':>8}")
    print("  " + "─" * 72)

    rows = []
    for col in columns:
        series = df[col].dropna()
        if len(series) < 30:
            continue
        try:
            adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
            adf_stat, adf_p = float(adf_stat), float(adf_p)
        except Exception:
            adf_stat, adf_p = np.nan, np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
            kpss_p = float(kpss_p)
        except Exception:
            kpss_p = np.nan

        adf_ok  = bool(adf_p < 0.05)  if not np.isnan(adf_p)  else None
        kpss_ok = bool(kpss_p >= 0.05) if not np.isnan(kpss_p) else None
        needs_diff = not adf_ok if adf_ok is not None else False
        verdict = ("✓ stationary" if adf_ok and kpss_ok else
                   "✗ non-stat → diff" if not adf_ok else "~ ambiguous")

        print(f"  {col:<30} {fmt(adf_stat):>10} {fmt(adf_p):>8} "
              f"{verdict:>18} {fmt(kpss_p):>8}")
        rows.append({"variable": col, "adf_stat": adf_stat, "adf_p": adf_p,
                     "kpss_p": kpss_p, "stationary": adf_ok,
                     "needs_diff": needs_diff})
    print()
    return pd.DataFrame(rows)


def make_stationary(df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for _, row in station_df.iterrows():
        col = row["variable"]
        if row.get("needs_diff") and col in result.columns:
            result[col + "_d"] = result[col].diff(1)
            print(f"  [DIFF] {col} → {col}_d")
    return result


# ---------------------------------------------------------------------------
# Granger causality tests
# ---------------------------------------------------------------------------

def run_granger_tests(df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    print("─" * 65)
    print("GRANGER CAUSALITY TESTS")
    print("─" * 65)
    print("  H0: X does NOT Granger-cause Y")
    print("  p < 0.05: reject H0 → X precedes Y ★\n")

    def statcol(name):
        row = station_df[station_df["variable"] == name] if len(station_df) else pd.DataFrame()
        if len(row) > 0 and row.iloc[0].get("needs_diff") and name + "_d" in df.columns:
            return name + "_d"
        return name

    pairs = [
        ("mentions",               "btc_logret",             "media volume → BTC price"),
        ("mentions",               "excess_media_effect_usd","media volume → anomalous pressure"),
        ("tone",                   "btc_logret",             "media sentiment → BTC price"),
        ("tone",                   "excess_media_effect_usd","sentiment → anomalous pressure"),
        ("excess_media_effect_usd","btc_logret",             "anomalous pressure → BTC price"),
        ("btc_logret",             "mentions",               "↩ BTC price → media (reverse)"),
        ("btc_logret",             "excess_media_effect_usd","↩ BTC price → pressure (reverse)"),
        ("bmpi_score",             "btc_logret",             "BMPI index → BTC price"),
    ]

    rows = []
    for x_name, y_name, label in pairs:
        x_col, y_col = statcol(x_name), statcol(y_name)
        if x_col not in df.columns or y_col not in df.columns:
            continue
        data = df[[y_col, x_col]].dropna()
        if len(data) < 50:
            print(f"  {label:<40} — too few obs ({len(data)})")
            continue

        print(f"\n  {label}")
        print(f"  {'lag':>4} {'F-stat':>10} {'p-value':>10} {'result'}")
        print(f"  {'':>4} {'─'*40}")

        for lag in LAGS:
            if lag >= len(data) // 4:
                continue
            try:
                res    = grangercausalitytests(data.values, maxlag=lag, verbose=False)
                f_stat = float(res[lag][0]["ssr_ftest"][0])
                p_val  = float(res[lag][0]["ssr_ftest"][1])
                sig    = stars(p_val)
                result = "★ PRECEDES" if p_val < 0.05 else "no effect"
                print(f"  lag={lag:>2}  {f_stat:>10.4f}  {p_val:>10.6f}  {sig}{result}")
                rows.append({
                    "direction": f"{x_name} → {y_name}",
                    "description": label, "x": x_name, "y": y_name,
                    "lag": lag, "f_stat": f_stat, "p_value": p_val,
                    "sig": sig.strip(), "granger_effect": p_val < 0.05, "n": len(data),
                })
            except Exception as e:
                print(f"  lag={lag:>2}  error: {e}")
    print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# VAR model
# ---------------------------------------------------------------------------

def run_var_model(df: pd.DataFrame, station_df: pd.DataFrame) -> dict:
    print("─" * 65)
    print("VAR MODEL (Vector Autoregression)")
    print("─" * 65)

    def statcol(name):
        row = station_df[station_df["variable"] == name] if len(station_df) else pd.DataFrame()
        if len(row) > 0 and row.iloc[0].get("needs_diff") and name + "_d" in df.columns:
            return name + "_d"
        return name

    cols = [statcol(v) for v in ("btc_logret","mentions","excess_media_effect_usd")
            if statcol(v) in df.columns]
    if len(cols) < 2:
        print("  [WARN] Not enough variables for VAR model")
        return {}

    data = df[cols].dropna()
    print(f"  Variables: {cols}  |  N: {len(data)}")
    try:
        model     = VAR(data.values)
        lag_order = model.select_order(maxlags=VAR_MAX_LAGS)
        opt_aic   = int(lag_order.aic)
        opt_bic   = int(lag_order.bic)
        print(f"  Optimal lag (AIC): {opt_aic}  |  (BIC): {opt_bic}")

        use_lag = max(1, min(opt_aic, 7))
        fitted  = model.fit(use_lag)
        print(f"  VAR({use_lag}): AIC={fitted.aic:.4f}  BIC={fitted.bic:.4f}")

        # IRF
        try:
            irf   = fitted.irf(10)
            i_m   = cols.index(statcol("mentions"))     if statcol("mentions")     in cols else None
            i_r   = cols.index(statcol("btc_logret"))   if statcol("btc_logret")   in cols else None
            if i_m is not None and i_r is not None:
                irf_vals = irf.irfs[:, i_r, i_m]
                print(f"\n  Impulse Response: media shock → BTC price")
                print(f"  {'day':>5} {'effect':>12}  direction")
                print(f"  {'─'*35}")
                for i, v in enumerate(irf_vals[:8]):
                    print(f"  t+{i:<3}  {v:>12.6f}  {'↑' if v > 0 else '↓'}")
                peak = max(abs(irf_vals))
                hl   = next((i for i,v in enumerate(irf_vals) if abs(v) <= peak*0.5 and i>0), None)
                print(f"\n  Peak: {peak:.6f}  |  Half-life: ~{hl} days" if hl else
                      f"\n  Peak: {peak:.6f}")
        except Exception as e:
            print(f"  [WARN] IRF error: {e}")

        return {"opt_lag_aic": opt_aic, "opt_lag_bic": opt_bic,
                "aic": fitted.aic, "bic": fitted.bic, "n_obs": len(data)}
    except Exception as e:
        print(f"  [ERROR] VAR error: {e}")
        return {}


# ---------------------------------------------------------------------------
# Simplified analysis (without statsmodels)
# ---------------------------------------------------------------------------

def run_simplified(df: pd.DataFrame) -> pd.DataFrame:
    print("─" * 65)
    print("SIMPLIFIED LEAD-LAG ANALYSIS (proxy for Granger causality)")
    print("─" * 65)
    pairs = [
        ("mentions",               "btc_logret",             "media volume → BTC price"),
        ("mentions",               "excess_media_effect_usd","media volume → anomalous pressure"),
        ("tone",                   "btc_logret",             "media sentiment → BTC price"),
        ("excess_media_effect_usd","btc_logret",             "anomalous pressure → BTC price"),
        ("btc_logret",             "mentions",               "↩ BTC price → media (reverse)"),
    ]
    rows = []
    for x_col, y_col, label in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        print(f"  {label}")
        best_r, best_lag = 0.0, 0
        for lag in LAGS:
            x = df[x_col].fillna(0)
            y = df[y_col].shift(-lag)
            mask = ~(x.isna() | y.isna())
            if mask.sum() < 20:
                continue
            r, p = scipy_stats.pearsonr(x[mask], y[mask])
            sig  = stars(p)
            mark = " ← BEST" if abs(r) > abs(best_r) else ""
            print(f"    lag={lag:>2}  r={r:+.4f}  {sig}{mark}")
            if abs(r) > abs(best_r):
                best_r, best_lag = r, lag
            rows.append({"direction": f"{x_col} → {y_col}", "description": label,
                         "lag": lag, "pearson_r": r, "p_value": p, "sig": sig.strip()})
        print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(granger_df: pd.DataFrame, var_results: dict) -> None:
    W = 68
    print("╔" + "═"*(W-2) + "╗")
    print("║  KEY GRANGER CAUSALITY RESULTS                               ║")
    print("╠" + "═"*(W-2) + "╣")

    if len(granger_df) > 0:
        sig = granger_df[granger_df.get("granger_effect", granger_df.get("p_value", pd.Series()) < 0.05).fillna(False)]
        print("║                                                              ║")
        print("║  Significant Granger effects (p < 0.05):                    ║")
        seen = set()
        for _, row in sig.iterrows():
            key = row.get("description", "")
            if key in seen: continue
            seen.add(key)
            print(f"║    ★ {str(key)[:44]:<44} lag={int(row.get('lag',0))} ║")
        if not seen:
            print("║    (no significant effects found)                            ║")

        print("║                                                              ║")
        if "description" in granger_df.columns:
            rev = granger_df[granger_df["description"].str.contains("↩", na=False)]
            rev_sig = rev[rev["p_value"] < 0.05] if len(rev) else pd.DataFrame()
            print("║  Reverse direction (BTC price → media):                     ║")
            if len(rev_sig):
                print("║    ⚠ ALSO SIGNIFICANT — possible bidirectional causality     ║")
            else:
                print("║    ✓ NOT SIGNIFICANT — unidirectional: media → price         ║")

    if var_results:
        print("║                                                              ║")
        print(f"║  VAR optimal lag: AIC={var_results.get('opt_lag_aic','?')} "
              f"BIC={var_results.get('opt_lag_bic','?')}                         ║")

    print("╠" + "═"*(W-2) + "╣")
    print("║                                                              ║")
    print("║  INTERPRETATION FOR DISSERTATION:                            ║")
    print("║  Granger causality is NOT structural causality.             ║")
    print("║  It means: X has predictive value for Y beyond Y's own     ║")
    print("║  history. Standard in financial econometrics (Sims 1980).  ║")
    print("║                                                              ║")
    print("╚" + "═"*(W-2) + "╝")
    print("\nNext step: step14_oos_validation.py")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html(granger_df: pd.DataFrame, station_df: pd.DataFrame,
               var_results: dict) -> str:
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_sig = int((granger_df["granger_effect"] == True).sum()) \
            if "granger_effect" in granger_df.columns else 0
    best_p = granger_df["p_value"].min() if "p_value" in granger_df.columns else np.nan

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
.kpi-l{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin-bottom:4px}
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
.sig{color:#f0a500;font-weight:bold} .nosig{color:var(--dim)}"""

    h = [f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>',
         f'<title>Granger Causality</title><style>{css}</style></head><body>',
         f'<header class="hdr"><h1>Granger Causality Analysis</h1>'
         f'<span class="ts">{now}</span></header>',
         f'<div class="kpi-grid">',
         f'<div class="kpi"><div class="kpi-l">Significant directions</div>'
         f'<div class="kpi-v">{n_sig}</div></div>',
         f'<div class="kpi"><div class="kpi-l">Best p-value</div>'
         f'<div class="kpi-v">{fmt(best_p,6)}</div></div>',
         f'<div class="kpi"><div class="kpi-l">VAR optimal lag (AIC)</div>'
         f'<div class="kpi-v">{var_results.get("opt_lag_aic","—")}</div></div>',
         f'<div class="kpi"><div class="kpi-l">N observations</div>'
         f'<div class="kpi-v">{var_results.get("n_obs","—")}</div></div>',
         '</div>']

    if len(granger_df) > 0:
        sort_by = ["description","lag"] if "description" in granger_df.columns else []
        h += ['<div class="sec"><div class="sec-t">Granger Results</div>',
              '<div class="tbl-w"><table><thead><tr>',
              '<th>Direction</th><th>Lag</th><th>F-stat</th>'
              '<th>p-value</th><th>Sig</th><th>Effect</th>',
              '</tr></thead><tbody>']
        for _, row in (granger_df.sort_values(sort_by) if sort_by else granger_df).iterrows():
            p   = float(row.get("p_value", 1.0))
            sig = p < 0.05
            cls = "sig" if sig else "nosig"
            h.append(f'<tr><td>{row.get("description","")}</td>'
                     f'<td>{int(row.get("lag",0))}</td>'
                     f'<td>{fmt(row.get("f_stat",np.nan))}</td>'
                     f'<td>{fmt(p,6)}</td>'
                     f'<td class="{cls}">{row.get("sig","")}</td>'
                     f'<td class="{cls}">{"★ precedes" if sig else "—"}</td></tr>')
        h += ['</tbody></table></div></div>']

    h += ['</body></html>']
    return "\n".join(h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = build_dataset()

    test_cols = [c for c in ("btc_logret", "excess_media_effect_usd", "bmpi_score",
                              "mentions", "tone", "resid_btc_mcap_usd")
                 if c in df.columns and df[c].notna().sum() > 50]

    if HAS_STATSMODELS:
        station_df  = check_stationarity(df, test_cols)
        station_df.to_csv(OUT_STATION, index=False)
        print(f"[OK] Saved: {OUT_STATION}")
        df_stat     = make_stationary(df, station_df)
        granger_df  = run_granger_tests(df_stat, station_df)
        var_results = run_var_model(df_stat, station_df)
    else:
        station_df  = pd.DataFrame()
        granger_df  = run_simplified(df)
        var_results = {}

    granger_df.to_csv(OUT_RESULTS, index=False)
    print(f"\n[OK] Saved: {OUT_RESULTS}")

    html = build_html(granger_df, station_df, var_results)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Saved: {OUT_HTML}  ← open in browser")

    print_summary(granger_df, var_results)


if __name__ == "__main__":
    main()