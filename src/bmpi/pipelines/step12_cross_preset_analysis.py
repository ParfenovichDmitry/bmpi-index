# -*- coding: utf-8 -*-
"""
pipelines/step12_cross_preset_analysis.py
==========================================
Cross-preset comparative analysis of GDELT signals.

Analyses:
  1. Correlations: GDELT mentions/tone → excess_media / bmpi / anomaly
  2. Signal convergence across presets
  3. Lead-lag analysis (does GDELT precede price moves?)
  4. Robustness check across presets

Input:
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv   (from downloader)
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_sensitive.csv
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_strong.csv
  data/processed/excess_media_effect_daily.csv                 (from step09)
  data/processed/events_peaks_balanced.csv                     (from step03)
  data/processed/features_daily.parquet                        (from step02)

Output:
  data/processed/cross_preset_correlations.csv
  data/processed/cross_preset_convergence.csv
  data/processed/cross_preset_granger.csv
  data/processed/cross_preset_robustness.csv
  data/processed/cross_preset_report.html

Next step: step13_granger_causality.py
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_GDELT     = BASE_DIR / "data" / "raw" / "gdelt"

EXCESS_CSV   = DATA_PROCESSED / "excess_media_effect_daily.csv"
PEAKS_CSV    = DATA_PROCESSED / "events_peaks_balanced.csv"
FEATURES_PAR = DATA_PROCESSED / "features_daily.parquet"

OUT_CORR    = DATA_PROCESSED / "cross_preset_correlations.csv"
OUT_CONV    = DATA_PROCESSED / "cross_preset_convergence.csv"
OUT_GRANGER = DATA_PROCESSED / "cross_preset_granger.csv"
OUT_ROBUST  = DATA_PROCESSED / "cross_preset_robustness.csv"
OUT_HTML    = DATA_PROCESSED / "cross_preset_report.html"

MAX_LAG = 14


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
            df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        return df
    except Exception as e:
        print(f"  [WARN] {path.name}: {e}")
        return None


def load_gdelt(name: str) -> Optional[pd.DataFrame]:
    """Load GDELT signal file, resolve mentions/tone columns."""
    for fname in [
        f"gdelt_gkg_bitcoin_daily_signal_{name}.csv",
        f"gdelt_btc_media_signal_{name}.csv",
    ]:
        path = DATA_GDELT / fname
        df = load_csv(path)
        if df is not None:
            # Resolve mentions column
            m_col = next((c for c in df.columns
                          if c in ("mentions", "liczba_wzmianek", "mention_count")), None)
            t_col = next((c for c in df.columns
                          if c in ("tone", "sredni_tone", "avg_tone", "tone_avg")), None)
            # Fallback: use positional columns after date
            if m_col is None:
                non_date = [c for c in df.columns if c != "date"]
                if len(non_date) >= 1:
                    m_col = non_date[0]
            if t_col is None:
                non_date = [c for c in df.columns if c not in ("date", m_col or "")]
                if len(non_date) >= 1:
                    t_col = non_date[0]
            if m_col:
                df["mentions"] = pd.to_numeric(df[m_col], errors="coerce").fillna(0)
            else:
                df["mentions"] = 0.0
            if t_col:
                df["tone"] = pd.to_numeric(df[t_col], errors="coerce").fillna(0)
            else:
                df["tone"] = 0.0
            return df[["date", "mentions", "tone"]].copy()
    return None


def pearsonr(x: pd.Series, y: pd.Series):
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan, np.nan
    r, p = scipy_stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def spearmanr(x: pd.Series, y: pd.Series):
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan, np.nan
    r, p = scipy_stats.spearmanr(x[mask], y[mask])
    return float(r), float(p)


def fmt(x, d=4) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.{d}f}"


def stars(p) -> str:
    if pd.isna(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all():
    print("=" * 65)
    print("LOADING DATA")
    print("=" * 65)

    gdelt: Dict[str, pd.DataFrame] = {}
    for name in ("balanced", "sensitive", "strong"):
        df = load_gdelt(name)
        if df is not None:
            gdelt[name] = df
            print(f"  [OK] GDELT {name:<10} {len(df)} days  "
                  f"mentions mean={df['mentions'].mean():.0f}")
        else:
            print(f"  [SKIP] GDELT {name}: not found in {DATA_GDELT}")

    if not gdelt:
        raise FileNotFoundError(
            f"No GDELT signal files found.\nExpected in: {DATA_GDELT}\n"
            "Run: python -m bmpi.utils.gdelt_btc_downloader --preset balanced"
        )

    excess = load_csv(EXCESS_CSV)
    if excess is None:
        raise FileNotFoundError(f"Required file not found: {EXCESS_CSV}")
    for c in ("excess_media_effect_usd", "media_effect_usd",
              "resid_btc_mcap_usd", "bmpi_score"):
        if c in excess.columns:
            excess[c] = pd.to_numeric(excess[c], errors="coerce")
    print(f"  [OK] excess_daily     {len(excess)} days")

    peaks = load_csv(PEAKS_CSV)
    if peaks is not None:
        pk_col = next((c for c in peaks.columns
                       if c in ("peak_date", "data_piku")), None)
        if pk_col:
            peaks["peak_date"] = pd.to_datetime(peaks[pk_col], errors="coerce").dt.normalize()
        print(f"  [OK] peaks          {len(peaks)} events")

    feat = load_csv(FEATURES_PAR)
    if feat is not None:
        price_col  = next((c for c in feat.columns
                           if "btc" in c and any(k in c for k in ("price","cena","close"))), None)
        logret_col = next((c for c in feat.columns if "logret" in c), None)
        mcap_col   = next((c for c in feat.columns
                           if ("mcap" in c or "kapitaliz" in c) and "btc" in c), None)
        if price_col:  feat["btc_price"]  = pd.to_numeric(feat[price_col],  errors="coerce")
        if logret_col: feat["btc_logret"] = pd.to_numeric(feat[logret_col], errors="coerce")
        if mcap_col:   feat["btc_mcap"]   = pd.to_numeric(feat[mcap_col],   errors="coerce")
        print(f"  [OK] features       {len(feat)} days "
              f"(price={price_col}, logret={logret_col})")

    print()
    return gdelt, excess, peaks, feat


# ---------------------------------------------------------------------------
# Analysis 1 — Correlations
# ---------------------------------------------------------------------------

def analysis_correlations(gdelt: dict, excess: pd.DataFrame) -> pd.DataFrame:
    print("─" * 65)
    print("ANALYSIS 1: Correlations GDELT → excess_media / bmpi / anomaly")
    print("─" * 65)

    targets    = [c for c in ("bmpi_score", "excess_media_effect_usd",
                               "resid_btc_mcap_usd", "media_effect_usd")
                  if c in excess.columns]
    rows: List[dict] = []

    for preset, gdf in gdelt.items():
        merged = gdf.merge(excess[["date"] + targets], on="date", how="inner")
        n = len(merged)
        print(f"  [{preset}] {n} common days")
        if n < 10:
            continue

        for pred in ("mentions", "tone"):
            for tgt in targets:
                r_p, p_p = pearsonr(merged[pred], merged[tgt])
                r_s, p_s = spearmanr(merged[pred], merged[tgt])
                rows.append({
                    "preset": preset, "predictor": pred, "target": tgt, "n": n,
                    "pearson_r": r_p, "pearson_p": p_p,
                    "spearman_r": r_s, "spearman_p": p_s,
                    "r2": r_p**2 if pd.notna(r_p) else np.nan,
                    "sig": stars(p_p),
                })
                print(f"  {preset:<10} {pred:<12} → {tgt:<28} "
                      f"r={fmt(r_p)}  ρ={fmt(r_s)}  {stars(p_p)}")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        best = df.loc[df["pearson_r"].abs().idxmax()]
        print(f"\n  STRONGEST: {best['preset']} | {best['predictor']} → "
              f"{best['target']} | r={fmt(best['pearson_r'])}  R²={fmt(best['r2'])}")
    print()
    return df


# ---------------------------------------------------------------------------
# Analysis 2 — Convergence
# ---------------------------------------------------------------------------

def analysis_convergence(gdelt: dict, excess: pd.DataFrame,
                          feat: Optional[pd.DataFrame]) -> pd.DataFrame:
    print("─" * 65)
    print("ANALYSIS 2: Signal Convergence (cross-preset agreement)")
    print("─" * 65)

    wide = None
    for preset, gdf in gdelt.items():
        sub = gdf[["date", "mentions", "tone"]].rename(columns={
            "mentions": f"mentions_{preset}", "tone": f"tone_{preset}"})
        wide = sub if wide is None else wide.merge(sub, on="date", how="outer")

    if wide is None:
        return pd.DataFrame()

    wide = wide.sort_values("date").reset_index(drop=True)
    wide["conv_score"] = 0
    for preset in gdelt:
        thr = wide[f"mentions_{preset}"].median()
        wide["conv_score"] += (wide[f"mentions_{preset}"].fillna(0) > thr).astype(int)
        wide["conv_score"] += (wide[f"tone_{preset}"].fillna(0) < -1.0).astype(int)

    targets = [c for c in ("excess_media_effect_usd", "bmpi_score", "resid_btc_mcap_usd")
               if c in excess.columns]
    merged = wide.merge(excess[["date"] + targets], on="date", how="left")
    if feat is not None and "btc_logret" in feat.columns:
        merged = merged.merge(feat[["date", "btc_logret"]], on="date", how="left")

    print(f"\n  {'score':>6} {'n_days':>7} {'mean_excess_B':>14} "
          f"{'mean_bmpi':>10} {'mean_resid_B':>13}")
    print("  " + "─"*55)
    rows = []
    for score in sorted(merged["conv_score"].dropna().unique()):
        sub = merged[merged["conv_score"] == score]
        n   = len(sub)
        exc = sub["excess_media_effect_usd"].mean() if "excess_media_effect_usd" in sub else np.nan
        bmp = sub["bmpi_score"].mean()              if "bmpi_score"              in sub else np.nan
        res = sub["resid_btc_mcap_usd"].abs().mean()if "resid_btc_mcap_usd"     in sub else np.nan
        print(f"  {int(score):>6} {n:>7} {exc/1e9:>13.3f}B  "
              f"{fmt(bmp):>10}  {res/1e9:>12.3f}B")
        rows.append({"conv_score": int(score), "n_days": n,
                     "mean_excess_usd": exc, "mean_bmpi_score": bmp, "mean_abs_resid": res})

    r, p = pearsonr(merged["conv_score"],
                    merged.get("excess_media_effect_usd", pd.Series(dtype=float)).fillna(0))
    print(f"\n  Corr conv_score → excess_usd: r={fmt(r)} {stars(p)}")
    merged.to_csv(OUT_CONV, index=False)
    print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 3 — Lead-lag
# ---------------------------------------------------------------------------

def analysis_lead_lag(gdelt: dict, excess: pd.DataFrame,
                       feat: Optional[pd.DataFrame]) -> pd.DataFrame:
    print("─" * 65)
    print("ANALYSIS 3: Lead-Lag Analysis (does GDELT precede price?)")
    print("─" * 65)

    rows: List[dict] = []
    exc_cols = ["date"] + [c for c in ("excess_media_effect_usd", "bmpi_score",
                                        "resid_btc_mcap_usd") if c in excess.columns]

    for preset, gdf in gdelt.items():
        base = gdf.merge(excess[exc_cols], on="date", how="left")
        if feat is not None and "btc_logret" in feat.columns:
            base = base.merge(feat[["date", "btc_logret"]], on="date", how="left")
        else:
            base["btc_logret"] = np.nan
        base = base.sort_values("date").reset_index(drop=True)

        print(f"\n  Preset: {preset.upper()}")
        print(f"  {'lag':>4} {'mentions→excess':>16} {'mentions→ret':>13} "
              f"{'tone→excess':>13} {'tone→ret':>10}")
        print("  " + "─" * 58)

        best = {"lag": 0, "r": 0.0}
        for lag in range(0, MAX_LAG + 1):
            exc_s = base["excess_media_effect_usd"].shift(-lag) \
                if "excess_media_effect_usd" in base.columns \
                else pd.Series(np.nan, index=base.index)
            ret_s = base["btc_logret"].shift(-lag)

            r_me, p_me = pearsonr(base["mentions"], exc_s)
            r_mr, p_mr = pearsonr(base["mentions"], ret_s)
            r_te, p_te = pearsonr(base["tone"],     exc_s)
            r_tr, p_tr = pearsonr(base["tone"],     ret_s)

            if abs(r_me or 0) > abs(best["r"]):
                best = {"lag": lag, "r": r_me}

            print(f"  {lag:>4}  "
                  f"{fmt(r_me)}{stars(p_me):>3}  "
                  f"{fmt(r_mr)}{stars(p_mr):>3}  "
                  f"{fmt(r_te)}{stars(p_te):>3}  "
                  f"{fmt(r_tr)}{stars(p_tr):>3}")

            rows.append({"preset": preset, "lag": lag,
                         "r_mentions_excess": r_me, "p_mentions_excess": p_me,
                         "r_mentions_return": r_mr, "p_mentions_return": p_mr,
                         "r_tone_excess": r_te, "p_tone_excess": p_te,
                         "r_tone_return": r_tr, "p_tone_return": p_tr})

        print(f"\n  Optimal lag mentions→excess: lag={best['lag']} days  r={fmt(best['r'])}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_GRANGER, index=False)
    print()
    return df


# ---------------------------------------------------------------------------
# Analysis 4 — Robustness
# ---------------------------------------------------------------------------

def analysis_robustness(gdelt: dict, excess: pd.DataFrame,
                         peaks: Optional[pd.DataFrame]) -> pd.DataFrame:
    print("─" * 65)
    print("ANALYSIS 4: Robustness Check (stability across presets)")
    print("─" * 65)

    rows: List[dict] = []
    exc_cols = ["date"] + [c for c in ("excess_media_effect_usd", "bmpi_score",
                                        "resid_btc_mcap_usd", "media_effect_usd")
                           if c in excess.columns]

    for preset, gdf in gdelt.items():
        merged = gdf.merge(excess[exc_cols], on="date", how="inner")
        n   = len(merged)
        bmp = float(merged["bmpi_score"].mean())   if "bmpi_score" in merged.columns else np.nan
        exc = float(merged["excess_media_effect_usd"].sum()) if "excess_media_effect_usd" in merged.columns else np.nan
        res = float(merged["resid_btc_mcap_usd"].abs().sum()) if "resid_btc_mcap_usd" in merged.columns else np.nan
        ano = exc / res if res and res > 0 else np.nan

        r_me, p_me = pearsonr(
            merged["mentions"].fillna(0) if "mentions" in merged.columns else pd.Series(dtype=float),
            merged["excess_media_effect_usd"].fillna(0) if "excess_media_effect_usd" in merged.columns else pd.Series(dtype=float),
        )

        print(f"\n  Preset: {preset.upper()} ({n} common days)")
        print(f"    Mean bmpi_score:       {fmt(bmp)}")
        if pd.notna(ano):
            print(f"    Anomaly share global: {ano*100:.2f}%")
        print(f"    Corr mentions→excess: r={fmt(r_me)} {stars(p_me)}")

        rows.append({"preset": preset, "n_days": n,
                     "mean_bmpi_score": bmp, "anomaly_share_global": ano,
                     "r_mentions_excess": r_me, "p_mentions_excess": p_me})

    df = pd.DataFrame(rows)
    if len(df) >= 2:
        print(f"\n  {'Metric':<35} {'balanced':>10} {'sensitive':>10} {'strong':>10}  {'spread':>8}")
        print("  " + "─" * 75)
        for col in ("mean_bmpi_score", "anomaly_share_global", "r_mentions_excess"):
            vals = {r["preset"]: r[col] for _, r in df.iterrows()}
            vs   = [v for v in vals.values() if pd.notna(v)]
            sprd = max(vs) - min(vs) if len(vs) >= 2 else np.nan
            tag  = "✓ STABLE" if pd.notna(sprd) and sprd < 0.05 else \
                   "~ moderate" if pd.notna(sprd) and sprd < 0.15 else "✗ unstable"
            print(f"  {col:<35} "
                  f"{fmt(vals.get('balanced')):>10} "
                  f"{fmt(vals.get('sensitive')):>10} "
                  f"{fmt(vals.get('strong')):>10}  "
                  f"{fmt(sprd):>8}  {tag}")

    df.to_csv(OUT_ROBUST, index=False)
    print()
    return df


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html(corr_df, conv_df, lag_df, robust_df) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    best_r = corr_df["pearson_r"].abs().max() if len(corr_df) > 0 else np.nan
    best   = corr_df.loc[corr_df["pearson_r"].abs().idxmax()] if len(corr_df) > 0 else {}

    if len(lag_df) > 0:
        opt     = lag_df.loc[lag_df["r_mentions_excess"].abs().idxmax()]
        opt_lag = int(opt["lag"])
        opt_r   = float(opt["r_mentions_excess"])
        opt_pre = str(opt["preset"])
    else:
        opt_lag, opt_r, opt_pre = "—", np.nan, "—"

    spread = np.nan
    if len(robust_df) > 0 and "anomaly_share_global" in robust_df.columns:
        vs = robust_df["anomaly_share_global"].dropna()
        spread = float(vs.max() - vs.min()) if len(vs) >= 2 else np.nan

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
th{background:var(--bg3);padding:7px 9px;text-align:left;color:var(--dim);
  font-size:10px;text-transform:uppercase;border-bottom:1px solid var(--brd)}
td{padding:6px 9px;border-bottom:1px solid var(--brd)}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--bg3)}"""

    h = [f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>',
         f'<title>Cross-Preset Analysis</title><style>{css}</style></head><body>',
         f'<header class="hdr"><h1>Cross-Preset GDELT Analysis</h1>',
         f'<span class="ts">{now}</span></header>',
         f'<div class="kpi-grid">',
         f'<div class="kpi"><div class="kpi-l">Strongest r</div>'
         f'<div class="kpi-v">{fmt(best_r)}</div></div>',
         f'<div class="kpi"><div class="kpi-l">Optimal lag</div>'
         f'<div class="kpi-v">{opt_lag} days</div></div>',
         f'<div class="kpi"><div class="kpi-l">Anomaly spread</div>'
         f'<div class="kpi-v">{fmt(spread)}</div></div>',
         f'<div class="kpi"><div class="kpi-l">Stability</div>'
         f'<div class="kpi-v">{"ROBUST" if pd.notna(spread) and spread < 0.05 else "moderate"}</div></div>',
         '</div>']

    # Correlations table
    if len(corr_df) > 0:
        h += ['<div class="sec"><div class="sec-t">Analysis 1 — Correlations</div>',
              '<div class="tbl-w"><table><thead><tr>',
              '<th>Preset</th><th>Predictor</th><th>Target</th>',
              '<th>N</th><th>Pearson r</th><th>Spearman rho</th><th>R2</th><th>Sig</th>',
              '</tr></thead><tbody>']
        for _, row in corr_df.sort_values("pearson_r", key=abs, ascending=False).iterrows():
            h.append(f'<tr><td>{row["preset"]}</td><td>{row["predictor"]}</td>'
                     f'<td>{row["target"]}</td><td>{int(row["n"])}</td>'
                     f'<td>{fmt(row["pearson_r"])}</td><td>{fmt(row["spearman_r"])}</td>'
                     f'<td>{fmt(row["r2"])}</td><td>{row["sig"]}</td></tr>')
        h += ['</tbody></table></div></div>']

    # Robustness table
    if len(robust_df) > 0:
        h += ['<div class="sec"><div class="sec-t">Analysis 4 — Robustness</div>',
              '<div class="tbl-w"><table><thead><tr>',
              '<th>Preset</th><th>N days</th><th>Mean BMPI</th>',
              '<th>Anomaly share</th><th>r mentions→excess</th>',
              '</tr></thead><tbody>']
        for _, row in robust_df.iterrows():
            h.append(f'<tr><td>{row["preset"]}</td><td>{int(row["n_days"])}</td>'
                     f'<td>{fmt(row["mean_bmpi_score"])}</td>'
                     f'<td>{fmt(row.get("anomaly_share_global", np.nan))}</td>'
                     f'<td>{fmt(row["r_mentions_excess"])}{stars(row["p_mentions_excess"])}</td></tr>')
        h += ['</tbody></table></div>']

    h += ['</div></body></html>']
    return "\n".join(h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    gdelt, excess, peaks, feat = load_all()

    print("RUNNING ANALYSES...\n")

    corr_df  = analysis_correlations(gdelt, excess)
    conv_df  = analysis_convergence(gdelt, excess, feat)
    lag_df   = analysis_lead_lag(gdelt, excess, feat)
    rob_df   = analysis_robustness(gdelt, excess, peaks)

    corr_df.to_csv(OUT_CORR, index=False)

    html = build_html(corr_df, conv_df, lag_df, rob_df)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Saved: {OUT_CORR}")
    print(f"[OK] Saved: {OUT_CONV}")
    print(f"[OK] Saved: {OUT_GRANGER}")
    print(f"[OK] Saved: {OUT_ROBUST}")
    print(f"[OK] Saved: {OUT_HTML}  <- open in browser")

    print()
    print("╔" + "═"*63 + "╗")
    print("║  KEY FINDINGS                                                 ║")
    if len(corr_df) > 0:
        best = corr_df.loc[corr_df["pearson_r"].abs().idxmax()]
        r2 = best["pearson_r"]**2
        print(f"║  1. Strongest correlation:                                    ║")
        print(f"║     {best['predictor']} → {best['target']:<32}       ║")
        print(f"║     r={fmt(best['pearson_r'])}  R²={fmt(r2)}  {best['sig']:<3}               ║")
    if len(lag_df) > 0:
        opt = lag_df.loc[lag_df["r_mentions_excess"].abs().idxmax()]
        print(f"║  2. Optimal lag GDELT → excess_usd:                          ║")
        print(f"║     lag={int(opt['lag'])} days  r={fmt(opt['r_mentions_excess'])}  ({opt['preset']})         ║")
    if len(rob_df) >= 2:
        vs = rob_df["anomaly_share_global"].dropna()
        sprd = float(vs.max() - vs.min()) if len(vs) >= 2 else np.nan
        tag  = "ROBUST" if pd.notna(sprd) and sprd < 0.05 else "moderate"
        print(f"║  3. Robustness: spread={fmt(sprd)}  {tag:<20}        ║")
    print("╚" + "═"*63 + "╝")
    print("\nNext step: step13_granger_causality.py")


if __name__ == "__main__":
    main()