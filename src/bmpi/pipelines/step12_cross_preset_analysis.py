# -*- coding: utf-8 -*-
"""
pipelines/step12_cross_preset_analysis.py
==========================================
Cross-preset comparative analysis of GDELT signals (console-only version).

Analyses:
  1. Correlations: GDELT mentions/tone -> excess_media / bmpi / anomaly
  2. Signal convergence across presets
  3. Lead-lag analysis (does GDELT precede price moves?)
  4. Robustness check across presets

Input:
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_sensitive.csv
  data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_strong.csv
  data/processed/excess_media_effect_daily.csv
  data/processed/events_peaks_balanced.csv
  data/processed/features_daily.parquet
  data/processed/news_effect_daily.csv   (optional but recommended)

Output:
  data/processed/cross_preset_correlations.csv
  data/processed/cross_preset_convergence.csv
  data/processed/cross_preset_granger.csv
  data/processed/cross_preset_robustness.csv

Next step:
  step13_granger_causality.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_GDELT = BASE_DIR / "data" / "raw" / "gdelt"

EXCESS_CSV = DATA_PROCESSED / "excess_media_effect_daily.csv"
PEAKS_CSV = DATA_PROCESSED / "events_peaks_balanced.csv"
FEATURES_PAR = DATA_PROCESSED / "features_daily.parquet"
NEWS_EFFECT_CSV = DATA_PROCESSED / "news_effect_daily.csv"

OUT_CORR = DATA_PROCESSED / "cross_preset_correlations.csv"
OUT_CONV = DATA_PROCESSED / "cross_preset_convergence.csv"
OUT_GRANGER = DATA_PROCESSED / "cross_preset_granger.csv"
OUT_ROBUST = DATA_PROCESSED / "cross_preset_robustness.csv"

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
        df.columns = [str(c).lower().strip() for c in df.columns]
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
            m_col = next(
                (c for c in df.columns if c in ("mentions", "liczba_wzmianek", "mention_count")),
                None
            )
            t_col = next(
                (c for c in df.columns if c in ("tone", "sredni_tone", "avg_tone", "tone_avg")),
                None
            )

            if m_col is None:
                non_date = [c for c in df.columns if c != "date"]
                if len(non_date) >= 1:
                    m_col = non_date[0]

            if t_col is None:
                non_date = [c for c in df.columns if c not in ("date", m_col or "")]
                if len(non_date) >= 1:
                    t_col = non_date[0]

            if m_col:
                df["mentions"] = pd.to_numeric(df[m_col], errors="coerce").fillna(0.0)
            else:
                df["mentions"] = 0.0

            if t_col:
                df["tone"] = pd.to_numeric(df[t_col], errors="coerce").fillna(0.0)
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
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all():
    print("=" * 72)
    print("LOADING DATA")
    print("=" * 72)

    gdelt: Dict[str, pd.DataFrame] = {}
    for name in ("balanced", "sensitive", "strong"):
        df = load_gdelt(name)
        if df is not None:
            gdelt[name] = df
            print(
                f"  [OK] GDELT {name:<10} {len(df)} days  "
                f"mentions mean={df['mentions'].mean():.0f}"
            )
        else:
            print(f"  [SKIP] GDELT {name}: not found in {DATA_GDELT}")

    if not gdelt:
        raise FileNotFoundError(
            f"No GDELT signal files found.\nExpected in: {DATA_GDELT}"
        )

    excess = load_csv(EXCESS_CSV)
    if excess is None:
        raise FileNotFoundError(f"Required file not found: {EXCESS_CSV}")

    for c in (
        "excess_media_effect_usd",
        "media_effect_used",
        "raw_abs_media_effect_usd",
        "bmpi_score",
        "media_share_of_abnormal_move_pct",
    ):
        if c in excess.columns:
            excess[c] = pd.to_numeric(excess[c], errors="coerce")

    print(f"  [OK] excess_daily        {len(excess)} days")

    news_effect = load_csv(NEWS_EFFECT_CSV)
    if news_effect is not None:
        for c in (
            "predicted_media_effect_usd_oof",
            "predicted_media_effect_usd",
            "abnormal_btc_mcap_usd",
        ):
            if c in news_effect.columns:
                news_effect[c] = pd.to_numeric(news_effect[c], errors="coerce")
        print(f"  [OK] news_effect         {len(news_effect)} days")
    else:
        print("  [WARN] news_effect_daily.csv not found")

    peaks = load_csv(PEAKS_CSV)
    if peaks is not None:
        pk_col = next((c for c in peaks.columns if c in ("peak_date", "data_piku")), None)
        if pk_col:
            peaks["peak_date"] = pd.to_datetime(peaks[pk_col], errors="coerce").dt.normalize()
        print(f"  [OK] peaks               {len(peaks)} events")

    feat = load_csv(FEATURES_PAR)
    if feat is not None:
        price_col = next(
            (c for c in feat.columns if "btc" in c and any(k in c for k in ("price", "cena", "close"))),
            None
        )
        logret_col = next((c for c in feat.columns if "logret" in c), None)
        mcap_col = next(
            (c for c in feat.columns if ("mcap" in c or "kapitaliz" in c) and "btc" in c),
            None
        )

        if price_col:
            feat["btc_price"] = pd.to_numeric(feat[price_col], errors="coerce")
        if logret_col:
            feat["btc_logret"] = pd.to_numeric(feat[logret_col], errors="coerce")
        if mcap_col:
            feat["btc_mcap"] = pd.to_numeric(feat[mcap_col], errors="coerce")

        print(
            f"  [OK] features            {len(feat)} days "
            f"(price={price_col}, logret={logret_col})"
        )

    print()
    return gdelt, excess, news_effect, peaks, feat


# ---------------------------------------------------------------------------
# Analysis 1 — Correlations
# ---------------------------------------------------------------------------

def analysis_correlations(
    gdelt: Dict[str, pd.DataFrame],
    excess: pd.DataFrame,
    news_effect: Optional[pd.DataFrame],
) -> pd.DataFrame:
    print("─" * 72)
    print("ANALYSIS 1: Correlations GDELT -> BMPI / excess / abnormal / media effect")
    print("─" * 72)

    merged_target_source = excess.copy()

    if news_effect is not None:
        keep_cols = ["date"]
        for c in ("predicted_media_effect_usd_oof", "predicted_media_effect_usd", "abnormal_btc_mcap_usd"):
            if c in news_effect.columns:
                keep_cols.append(c)
        merged_target_source = merged_target_source.merge(
            news_effect[keep_cols].drop_duplicates(subset=["date"]),
            on="date",
            how="left",
        )

    targets = [
        c for c in (
            "bmpi_score",
            "excess_media_effect_usd",
            "media_share_of_abnormal_move_pct",
            "predicted_media_effect_usd_oof",
            "predicted_media_effect_usd",
            "abnormal_btc_mcap_usd",
        )
        if c in merged_target_source.columns
    ]

    rows: List[dict] = []

    for preset, gdf in gdelt.items():
        merged = gdf.merge(merged_target_source[["date"] + targets], on="date", how="inner")
        n = len(merged)
        print(f"  [{preset}] {n} common days")
        if n < 10:
            continue

        for pred in ("mentions", "tone"):
            for tgt in targets:
                r_p, p_p = pearsonr(merged[pred], merged[tgt])
                r_s, p_s = spearmanr(merged[pred], merged[tgt])

                rows.append({
                    "preset": preset,
                    "predictor": pred,
                    "target": tgt,
                    "n": n,
                    "pearson_r": r_p,
                    "pearson_p": p_p,
                    "spearman_r": r_s,
                    "spearman_p": p_s,
                    "r2": r_p ** 2 if pd.notna(r_p) else np.nan,
                    "sig": stars(p_p),
                })

                print(
                    f"  {preset:<10} {pred:<12} -> {tgt:<30} "
                    f"r={fmt(r_p)}  rho={fmt(r_s)}  {stars(p_p)}"
                )

    df = pd.DataFrame(rows)

    if len(df) > 0:
        best = df.loc[df["pearson_r"].abs().idxmax()]
        print(
            f"\n  STRONGEST: {best['preset']} | {best['predictor']} -> "
            f"{best['target']} | r={fmt(best['pearson_r'])}  R²={fmt(best['r2'])}"
        )
    print()
    return df


# ---------------------------------------------------------------------------
# Analysis 2 — Convergence
# ---------------------------------------------------------------------------

def analysis_convergence(
    gdelt: Dict[str, pd.DataFrame],
    excess: pd.DataFrame,
    feat: Optional[pd.DataFrame],
) -> pd.DataFrame:
    print("─" * 72)
    print("ANALYSIS 2: Signal Convergence (cross-preset agreement)")
    print("─" * 72)

    wide = None
    for preset, gdf in gdelt.items():
        sub = gdf[["date", "mentions", "tone"]].rename(
            columns={
                "mentions": f"mentions_{preset}",
                "tone": f"tone_{preset}",
            }
        )
        wide = sub if wide is None else wide.merge(sub, on="date", how="outer")

    if wide is None:
        return pd.DataFrame()

    wide = wide.sort_values("date").reset_index(drop=True)
    wide["conv_score"] = 0

    for preset in gdelt:
        thr = wide[f"mentions_{preset}"].median()
        wide["conv_score"] += (wide[f"mentions_{preset}"].fillna(0) > thr).astype(int)
        wide["conv_score"] += (wide[f"tone_{preset}"].fillna(0) < -1.0).astype(int)

    targets = [c for c in ("excess_media_effect_usd", "bmpi_score", "media_share_of_abnormal_move_pct") if c in excess.columns]
    merged = wide.merge(excess[["date"] + targets], on="date", how="left")

    if feat is not None and "btc_logret" in feat.columns:
        merged = merged.merge(feat[["date", "btc_logret"]], on="date", how="left")

    print(f"\n  {'score':>6} {'n_days':>7} {'mean_excess_B':>14} {'mean_bmpi':>10} {'mean_share%':>12}")
    print("  " + "─" * 60)

    rows = []
    for score in sorted(merged["conv_score"].dropna().unique()):
        sub = merged[merged["conv_score"] == score]
        n = len(sub)
        exc = sub["excess_media_effect_usd"].mean() if "excess_media_effect_usd" in sub else np.nan
        bmp = sub["bmpi_score"].mean() if "bmpi_score" in sub else np.nan
        shr = sub["media_share_of_abnormal_move_pct"].mean() if "media_share_of_abnormal_move_pct" in sub else np.nan

        print(
            f"  {int(score):>6} {n:>7} "
            f"{(exc / 1e9 if pd.notna(exc) else np.nan):>13.3f}B  "
            f"{fmt(bmp):>10}  {fmt(shr):>12}"
        )

        rows.append({
            "conv_score": int(score),
            "n_days": n,
            "mean_excess_usd": exc,
            "mean_bmpi_score": bmp,
            "mean_media_share_pct": shr,
        })

    r, p = pearsonr(
        merged["conv_score"],
        merged.get("excess_media_effect_usd", pd.Series(dtype=float)).fillna(0),
    )
    print(f"\n  Corr conv_score -> excess_usd: r={fmt(r)} {stars(p)}")
    print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 3 — Lead-lag
# ---------------------------------------------------------------------------

def analysis_lead_lag(
    gdelt: Dict[str, pd.DataFrame],
    excess: pd.DataFrame,
    feat: Optional[pd.DataFrame],
    news_effect: Optional[pd.DataFrame],
) -> pd.DataFrame:
    print("─" * 72)
    print("ANALYSIS 3: Lead-Lag Analysis (does GDELT precede market effect?)")
    print("─" * 72)

    rows: List[dict] = []

    exc_cols = ["date"] + [
        c for c in (
            "excess_media_effect_usd",
            "bmpi_score",
            "media_share_of_abnormal_move_pct",
        ) if c in excess.columns
    ]

    for preset, gdf in gdelt.items():
        base = gdf.merge(excess[exc_cols], on="date", how="left")

        if feat is not None and "btc_logret" in feat.columns:
            base = base.merge(feat[["date", "btc_logret"]], on="date", how="left")
        else:
            base["btc_logret"] = np.nan

        if news_effect is not None:
            keep_cols = ["date"] + [c for c in ("predicted_media_effect_usd_oof", "abnormal_btc_mcap_usd") if c in news_effect.columns]
            base = base.merge(news_effect[keep_cols], on="date", how="left")

        base = base.sort_values("date").reset_index(drop=True)

        print(f"\n  Preset: {preset.upper()}")
        print(f"  {'lag':>4} {'m->excess':>12} {'m->media_oof':>14} {'m->ret':>10} {'t->excess':>12} {'t->ret':>10}")
        print("  " + "─" * 70)

        best = {"lag": 0, "r": 0.0}

        for lag in range(0, MAX_LAG + 1):
            exc_s = base["excess_media_effect_usd"].shift(-lag) if "excess_media_effect_usd" in base.columns else pd.Series(np.nan, index=base.index)
            med_s = base["predicted_media_effect_usd_oof"].shift(-lag) if "predicted_media_effect_usd_oof" in base.columns else pd.Series(np.nan, index=base.index)
            ret_s = base["btc_logret"].shift(-lag)

            r_me, p_me = pearsonr(base["mentions"], exc_s)
            r_mm, p_mm = pearsonr(base["mentions"], med_s)
            r_mr, p_mr = pearsonr(base["mentions"], ret_s)
            r_te, p_te = pearsonr(base["tone"], exc_s)
            r_tr, p_tr = pearsonr(base["tone"], ret_s)

            if abs(r_me or 0.0) > abs(best["r"]):
                best = {"lag": lag, "r": r_me}

            print(
                f"  {lag:>4}  "
                f"{fmt(r_me)}{stars(p_me):>3}  "
                f"{fmt(r_mm)}{stars(p_mm):>3}  "
                f"{fmt(r_mr)}{stars(p_mr):>3}  "
                f"{fmt(r_te)}{stars(p_te):>3}  "
                f"{fmt(r_tr)}{stars(p_tr):>3}"
            )

            rows.append({
                "preset": preset,
                "lag": lag,
                "r_mentions_excess": r_me,
                "p_mentions_excess": p_me,
                "r_mentions_media_oof": r_mm,
                "p_mentions_media_oof": p_mm,
                "r_mentions_return": r_mr,
                "p_mentions_return": p_mr,
                "r_tone_excess": r_te,
                "p_tone_excess": p_te,
                "r_tone_return": r_tr,
                "p_tone_return": p_tr,
            })

        print(f"\n  Optimal lag mentions -> excess_usd: lag={best['lag']} days  r={fmt(best['r'])}")

    print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 4 — Robustness
# ---------------------------------------------------------------------------

def analysis_robustness(
    gdelt: Dict[str, pd.DataFrame],
    excess: pd.DataFrame,
    news_effect: Optional[pd.DataFrame],
) -> pd.DataFrame:
    print("─" * 72)
    print("ANALYSIS 4: Robustness Check (stability across presets)")
    print("─" * 72)

    rows: List[dict] = []
    base_df = excess.copy()

    if news_effect is not None:
        keep_cols = ["date"] + [c for c in ("predicted_media_effect_usd_oof", "abnormal_btc_mcap_usd") if c in news_effect.columns]
        base_df = base_df.merge(news_effect[keep_cols], on="date", how="left")

    for preset, gdf in gdelt.items():
        merged = gdf.merge(base_df, on="date", how="inner")
        n = len(merged)

        bmp = float(merged["bmpi_score"].mean()) if "bmpi_score" in merged.columns else np.nan
        exc = float(merged["excess_media_effect_usd"].sum()) if "excess_media_effect_usd" in merged.columns else np.nan

        if "media_share_of_abnormal_move_pct" in merged.columns:
            ano_share = float(merged["media_share_of_abnormal_move_pct"].mean())
        elif "abnormal_btc_mcap_usd" in merged.columns and "excess_media_effect_usd" in merged.columns:
            denom = merged["abnormal_btc_mcap_usd"].abs().sum()
            ano_share = float(100.0 * merged["excess_media_effect_usd"].sum() / (denom + 1e-9))
        else:
            ano_share = np.nan

        r_me, p_me = pearsonr(
            merged["mentions"].fillna(0),
            merged["excess_media_effect_usd"].fillna(0) if "excess_media_effect_usd" in merged.columns else pd.Series(dtype=float),
        )

        print(f"\n  Preset: {preset.upper()} ({n} common days)")
        print(f"    Mean BMPI score:        {fmt(bmp)}")
        print(f"    Mean anomaly share %:   {fmt(ano_share)}")
        print(f"    Corr mentions->excess:  r={fmt(r_me)} {stars(p_me)}")

        rows.append({
            "preset": preset,
            "n_days": n,
            "mean_bmpi_score": bmp,
            "anomaly_share_global_pct": ano_share,
            "r_mentions_excess": r_me,
            "p_mentions_excess": p_me,
        })

    df = pd.DataFrame(rows)

    if len(df) >= 2:
        print(f"\n  {'Metric':<35} {'balanced':>10} {'sensitive':>10} {'strong':>10}  {'spread':>8}")
        print("  " + "─" * 80)

        for col in ("mean_bmpi_score", "anomaly_share_global_pct", "r_mentions_excess"):
            vals = {r["preset"]: r[col] for _, r in df.iterrows()}
            vs = [v for v in vals.values() if pd.notna(v)]
            sprd = max(vs) - min(vs) if len(vs) >= 2 else np.nan

            if pd.notna(sprd) and sprd < 0.05:
                tag = "✓ STABLE"
            elif pd.notna(sprd) and sprd < 0.15:
                tag = "~ moderate"
            else:
                tag = "✗ unstable"

            print(
                f"  {col:<35} "
                f"{fmt(vals.get('balanced')):>10} "
                f"{fmt(vals.get('sensitive')):>10} "
                f"{fmt(vals.get('strong')):>10}  "
                f"{fmt(sprd):>8}  {tag}"
            )

    print()
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    gdelt, excess, news_effect, peaks, feat = load_all()

    print("RUNNING ANALYSES...\n")

    corr_df = analysis_correlations(gdelt, excess, news_effect)
    conv_df = analysis_convergence(gdelt, excess, feat)
    lag_df = analysis_lead_lag(gdelt, excess, feat, news_effect)
    rob_df = analysis_robustness(gdelt, excess, news_effect)

    corr_df.to_csv(OUT_CORR, index=False)
    conv_df.to_csv(OUT_CONV, index=False)
    lag_df.to_csv(OUT_GRANGER, index=False)
    rob_df.to_csv(OUT_ROBUST, index=False)

    print(f"[OK] Saved: {OUT_CORR}")
    print(f"[OK] Saved: {OUT_CONV}")
    print(f"[OK] Saved: {OUT_GRANGER}")
    print(f"[OK] Saved: {OUT_ROBUST}")

    print()
    print("╔" + "═" * 70 + "╗")
    print("║  KEY FINDINGS                                                         ║")

    if len(corr_df) > 0:
        best = corr_df.loc[corr_df["pearson_r"].abs().idxmax()]
        r2 = best["pearson_r"] ** 2 if pd.notna(best["pearson_r"]) else np.nan
        line1 = f"  1. Strongest correlation: {best['predictor']} -> {best['target']}"
        line2 = f"     {best['preset']} | r={fmt(best['pearson_r'])}  R²={fmt(r2)}  {best['sig']}"
        print(f"║{line1:<70}║")
        print(f"║{line2:<70}║")

    if len(lag_df) > 0:
        opt = lag_df.loc[lag_df["r_mentions_excess"].abs().idxmax()]
        line1 = "  2. Optimal lag GDELT -> excess_usd:"
        line2 = f"     lag={int(opt['lag'])} days  r={fmt(opt['r_mentions_excess'])}  ({opt['preset']})"
        print(f"║{line1:<70}║")
        print(f"║{line2:<70}║")

    if len(rob_df) >= 2:
        vs = rob_df["anomaly_share_global_pct"].dropna()
        sprd = float(vs.max() - vs.min()) if len(vs) >= 2 else np.nan
        tag = "ROBUST" if pd.notna(sprd) and sprd < 5.0 else "moderate"
        line1 = f"  3. Robustness spread (anomaly share): {fmt(sprd)} pp"
        line2 = f"     Stability tag: {tag}"
        print(f"║{line1:<70}║")
        print(f"║{line2:<70}║")

    print("╚" + "═" * 70 + "╝")
    print("\nNext step: step13_granger_causality.py")


if __name__ == "__main__":
    main()