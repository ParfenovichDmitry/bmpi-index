# -*- coding: utf-8 -*-
"""
pipelines/step15_benchmark_comparison.py
==========================================
Benchmark comparison: BMPI vs Crypto Fear & Greed Index (CFGI).
Console-only version (no HTML output).

What we compare:
  BMPI  — Bitcoin Media Pressure Index
  CFGI  — Crypto Fear & Greed Index (Alternative.me)

Key hypotheses:
  H1: BMPI and CFGI measure different constructs
  H2: BMPI has superior predictive power for excess_media_usd
  H3: BMPI captures media-specific pressure that CFGI misses

Input:
  data/processed/excess_media_effect_daily.csv
  data/processed/features_daily.parquet

Output:
  data/processed/cfgi_daily.csv
  data/processed/benchmark_comparison.json
  data/processed/benchmark_comparison.csv

Next step:
  step16_johansen_cointegration.py
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

EXCESS_CSV = DATA_PROCESSED / "excess_media_effect_daily.csv"
FEATURES_PAR = DATA_PROCESSED / "features_daily.parquet"

OUT_CFGI_CSV = DATA_PROCESSED / "cfgi_daily.csv"
OUT_JSON = DATA_PROCESSED / "benchmark_comparison.json"
OUT_CSV = DATA_PROCESSED / "benchmark_comparison.csv"

CFGI_API_URL = "https://api.alternative.me/fng/?limit=0&format=json"
CFGI_API_TIMEOUT = 30

CFGI_ZONES = [
    (0, 25, "Extreme Fear"),
    (25, 45, "Fear"),
    (45, 55, "Neutral"),
    (55, 75, "Greed"),
    (75, 100, "Extreme Greed"),
]

GRANGER_LAGS = [1, 2, 3, 5, 7, 14]


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


def pearsonr(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return np.nan, np.nan
    if np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan
    r, p = scipy_stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def spearmanr(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return np.nan, np.nan
    r, p = scipy_stats.spearmanr(x[mask], y[mask])
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Download CFGI
# ---------------------------------------------------------------------------

def download_cfgi() -> pd.DataFrame:
    """Download Crypto Fear & Greed Index. Falls back to cache."""
    if OUT_CFGI_CSV.exists():
        df = pd.read_csv(OUT_CFGI_CSV)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        print(
            f"  [OK] CFGI from cache: {len(df)} days  "
            f"({df['date'].min().date()} -> {df['date'].max().date()})"
        )
        return df

    if not HAS_REQUESTS:
        raise ImportError("requests not installed. Run: pip install requests")

    print("  Downloading CFGI from Alternative.me API...")
    try:
        resp = requests.get(CFGI_API_URL, timeout=CFGI_API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ConnectionError(
            f"Cannot download CFGI: {e}\n"
            f"Manual URL: {CFGI_API_URL}\n"
            f"Save as: {OUT_CFGI_CSV}"
        )

    rows = []
    for rec in data.get("data", []):
        try:
            rows.append({
                "date": pd.Timestamp(int(rec["timestamp"]), unit="s").normalize(),
                "cfgi_value": int(rec["value"]),
                "cfgi_label": str(rec.get("value_classification", "")),
            })
        except Exception:
            continue

    df = (
        pd.DataFrame(rows)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    df["cfgi_norm"] = df["cfgi_value"] / 100.0
    df.to_csv(OUT_CFGI_CSV, index=False)
    print(f"  [OK] CFGI downloaded: {len(df)} days")
    return df


# ---------------------------------------------------------------------------
# Build merged dataset
# ---------------------------------------------------------------------------

def build_merged(cfgi: pd.DataFrame) -> pd.DataFrame:
    excess = load_csv(EXCESS_CSV)
    if excess is None:
        raise FileNotFoundError(
            f"Required: {EXCESS_CSV}\nRun step09 first."
        )

    df = excess[["date"]].copy()
    for c in ("excess_media_effect_usd", "bmpi_score", "resid_btc_mcap_usd"):
        if c in excess.columns:
            df[c] = pd.to_numeric(excess[c], errors="coerce")

    feat = load_csv(FEATURES_PAR)
    if feat is not None:
        lr = next((c for c in feat.columns if "logret" in c), None)
        mc = next(
            (c for c in feat.columns if ("mcap" in c or "kapitaliz" in c) and "btc" in c),
            None,
        )
        sub = feat[["date"]].copy()
        if lr:
            sub["btc_logret"] = pd.to_numeric(feat[lr], errors="coerce")
        if mc:
            sub["btc_mcap"] = pd.to_numeric(feat[mc], errors="coerce")
        df = df.merge(sub, on="date", how="left")

    if "btc_logret" not in df.columns and "btc_mcap" in df.columns:
        df["btc_logret"] = np.log(df["btc_mcap"] / df["btc_mcap"].shift(1))

    df = df.merge(
        cfgi[["date", "cfgi_value", "cfgi_norm", "cfgi_label"]],
        on="date",
        how="left",
    )

    n_ov = df["cfgi_norm"].notna().sum()
    print(
        f"  [OK] Merged: {len(df)} days  |  CFGI overlap: {n_ov} days "
        f"({n_ov / len(df) * 100:.0f}%)"
    )
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 1: Descriptive comparison
# ---------------------------------------------------------------------------

def descriptive_comparison(df: pd.DataFrame) -> dict:
    print("─" * 72)
    print("ANALYSIS 1: Descriptive comparison BMPI vs CFGI")
    print("─" * 72)

    bmpi = df["bmpi_score"].dropna()
    cfgi = df["cfgi_norm"].dropna()

    def _s(s: pd.Series):
        return {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.median()),
            "p75": float(s.quantile(0.75)),
        }

    s_b, s_c = _s(bmpi), _s(cfgi)

    print(f"\n  {'metric':<12} {'BMPI':>10} {'CFGI_norm':>12}")
    print("  " + "─" * 38)
    for k in ("n", "mean", "std", "p25", "p50", "p75"):
        print(f"  {k:<12} {fmt(s_b[k]):>10} {fmt(s_c[k]):>12}")

    print("\n  BMPI zones:")
    for lo, hi, label in ((0, .45, "LOW"), (0.45, .55, "LOW-MOD"), (0.55, .65, "MOD"), (0.65, 1, "HIGH")):
        n = int(((bmpi >= lo) & (bmpi < hi)).sum())
        print(f"    {label:<10} {n:>5} days ({n / len(bmpi) * 100:.1f}%)")

    print("\n  CFGI zones (original 0-100):")
    total_cfgi = df["cfgi_value"].notna().sum()
    for lo, hi, label in CFGI_ZONES:
        n = int(((df["cfgi_value"] >= lo) & (df["cfgi_value"] < hi)).sum())
        if n > 0:
            print(f"    {label:<15} {n:>5} days ({n / total_cfgi * 100:.1f}%)")

    return {"bmpi": s_b, "cfgi": s_c}


# ---------------------------------------------------------------------------
# Analysis 2: Correlation
# ---------------------------------------------------------------------------

def correlation_analysis(df: pd.DataFrame) -> dict:
    print("─" * 72)
    print("ANALYSIS 2: BMPI vs CFGI correlation")
    print("─" * 72)

    ov = df[df["cfgi_norm"].notna() & df["bmpi_score"].notna()].copy()
    print(
        f"  Overlap: {len(ov)} days  "
        f"({ov['date'].min().date()} -> {ov['date'].max().date()})"
    )

    r_p, p_p = pearsonr(ov["bmpi_score"], ov["cfgi_norm"])
    r_s, p_s = spearmanr(ov["bmpi_score"], ov["cfgi_norm"])

    print("\n  BMPI vs CFGI_norm:")
    print(f"    Pearson  r = {fmt(r_p)}  {stars(p_p)}")
    print(f"    Spearman r = {fmt(r_s)}  {stars(p_s)}")

    excess = ov["excess_media_effect_usd"].fillna(0)
    r_bm, p_bm = pearsonr(ov["bmpi_score"], excess)
    r_cf, p_cf = pearsonr(ov["cfgi_norm"], excess)

    print("\n  Predictive power for excess_media_usd:")
    print(f"    BMPI r = {fmt(r_bm)}  {stars(p_bm)}")
    print(f"    CFGI r = {fmt(r_cf)}  {stars(p_cf)}")
    print(f"    BMPI advantage: Δr = {fmt(r_bm - r_cf)}")

    r_brl = r_crl = np.nan
    if "btc_logret" in ov.columns:
        ret = ov["btc_logret"].fillna(0)
        r_brl, _ = pearsonr(ov["bmpi_score"], ret)
        r_crl, _ = pearsonr(ov["cfgi_norm"], ret)
        print("\n  Predictive power for btc_logret:")
        print(f"    BMPI r = {fmt(r_brl)}")
        print(f"    CFGI r = {fmt(r_crl)}")

    inc_r2 = np.nan
    mask = ov["bmpi_score"].notna() & ov["cfgi_norm"].notna() & excess.notna()
    if mask.sum() > 50:
        y = excess[mask].values
        _, _, r_c, _, _ = scipy_stats.linregress(ov.loc[mask, "cfgi_norm"].values, y)
        r2_cfgi = float(r_c ** 2)
        X = np.column_stack([
            ov.loc[mask, "bmpi_score"].values,
            ov.loc[mask, "cfgi_norm"].values,
        ])
        try:
            corr_mat = np.corrcoef(np.vstack([X.T, y]))
            r_xy = corr_mat[:2, 2]
            R_xx = corr_mat[:2, :2]
            r2_both = float(r_xy @ np.linalg.solve(R_xx, r_xy))
            inc_r2 = max(0.0, r2_both - r2_cfgi)
            print(
                f"\n  Incremental R²: CFGI={r2_cfgi:.4f}  "
                f"BMPI+CFGI≈{r2_both:.4f}  Δ=+{inc_r2:.4f}"
            )
        except Exception:
            pass

    return {
        "n_overlap": len(ov),
        "bmpi_cfgi_pearson": r_p,
        "bmpi_cfgi_spearman": r_s,
        "bmpi_excess_r": r_bm,
        "cfgi_excess_r": r_cf,
        "bmpi_ret_r": r_brl,
        "cfgi_ret_r": r_crl,
        "incremental_r2": inc_r2,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Granger causality comparison
# ---------------------------------------------------------------------------

def granger_comparison(df: pd.DataFrame) -> dict:
    if not HAS_STATSMODELS:
        print("  [SKIP] statsmodels not available")
        return {}

    print("─" * 72)
    print("ANALYSIS 3: Granger causality — BMPI vs CFGI")
    print("─" * 72)

    ov = df[df["cfgi_norm"].notna() & df["btc_logret"].notna()].copy()
    data = ov[["btc_logret", "bmpi_score", "cfgi_norm", "excess_media_effect_usd"]].dropna()
    print(f"  N = {len(data)} observations\n")

    results = {}
    pairs = [
        ("BMPI -> btc_logret", "bmpi_score", "btc_logret"),
        ("CFGI -> btc_logret", "cfgi_norm", "btc_logret"),
        ("BMPI -> excess_usd", "bmpi_score", "excess_media_effect_usd"),
        ("CFGI -> excess_usd", "cfgi_norm", "excess_media_effect_usd"),
    ]

    for label, x_col, y_col in pairs:
        td = data[[y_col, x_col]].dropna()
        if len(td) < 50:
            continue

        print(f"  {label}")
        print(f"  {'lag':>4} {'F':>10} {'p':>10} result")
        print("  " + "─" * 38)

        row_res = []
        for lag in GRANGER_LAGS:
            if lag >= len(td) // 4:
                continue
            try:
                res = grangercausalitytests(td.values, maxlag=lag, verbose=False)
                f_v = float(res[lag][0]["ssr_ftest"][0])
                p_v = float(res[lag][0]["ssr_ftest"][1])
                print(
                    f"  lag={lag:>2}  {f_v:>10.4f}  {p_v:>10.6f}  "
                    f"{stars(p_v)} {'★' if p_v < 0.05 else '—'}"
                )
                row_res.append({"lag": lag, "f": f_v, "p": p_v})
            except Exception as e:
                print(f"  lag={lag:>2}  error: {e}")

        if row_res:
            best_p = min(r["p"] for r in row_res)
            sig_lags = [r["lag"] for r in row_res if r["p"] < 0.05]
            results[label] = {"best_p": best_p, "sig_lags": sig_lags}
        print()

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_verdict(corr: dict, granger: dict) -> None:
    W = 74
    print("╔" + "═" * (W - 2) + "╗")
    print("║  BMPI vs CFGI — KEY FINDINGS                                          ║")
    print("╠" + "═" * (W - 2) + "╣")
    print("║                                                                        ║")

    r_bc = corr.get("bmpi_cfgi_pearson", np.nan)
    r_bm = corr.get("bmpi_excess_r", np.nan)
    r_cf = corr.get("cfgi_excess_r", np.nan)
    inc = corr.get("incremental_r2", np.nan)

    print(f"║  1. BMPI–CFGI correlation: r = {fmt(r_bc):<10}{' ' * 32}║")
    if not np.isnan(r_bc):
        tag = (
            "DIFFERENT constructs ✓" if abs(r_bc) < 0.3 else
            "Moderate overlap" if abs(r_bc) < 0.6 else
            "High overlap ⚠"
        )
        print(f"║     -> {tag:<63}║")

    print("║                                                                        ║")
    print("║  2. Predictive power for excess_media_usd:                             ║")
    print(f"║     BMPI r = {fmt(r_bm):<10} | CFGI r = {fmt(r_cf):<10}{' ' * 21}║")
    if not (np.isnan(r_bm) or np.isnan(r_cf)):
        adv = r_bm - r_cf
        tag = (
            f"BMPI outperforms CFGI by Δr={fmt(adv)} ✓" if adv > 0.05 else
            f"BMPI marginally better (Δr={fmt(adv)})" if adv > 0 else
            f"CFGI comparable (Δr={fmt(adv)}) ⚠"
        )
        print(f"║     -> {tag:<63}║")

    if not np.isnan(inc):
        print("║                                                                        ║")
        print(f"║  3. Incremental R² = +{inc:.4f}{' ' * 46}║")
        tag = "unique information ✓" if inc > 0.01 else "limited increment ⚠"
        print(f"║     -> BMPI adds {tag:<48}║")

    print("║                                                                        ║")
    print("╠" + "═" * (W - 2) + "╣")
    print("║  ACADEMIC POSITIONING:                                                 ║")
    print("║  CFGI = composite sentiment (volatility, social, momentum)             ║")
    print("║  BMPI = media-specific pressure (GDELT + econometric residuals).       ║")
    print("║  These are complementary indices measuring different aspects of        ║")
    print("║  information-driven BTC price formation.                               ║")
    print("╚" + "═" * (W - 2) + "╝")
    print("\nNext step: step16_johansen_cointegration.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("BMPI vs CRYPTO FEAR & GREED INDEX — BENCHMARK COMPARISON")
    print("=" * 72 + "\n")

    print("Step 1: Loading CFGI data...")
    cfgi = download_cfgi()

    print("\nStep 2: Building merged dataset...")
    df = build_merged(cfgi)

    print()
    desc = descriptive_comparison(df)
    print()
    corr = correlation_analysis(df)
    print()
    granger = granger_comparison(df)

    print_verdict(corr, granger)

    overlap_cols = [
        c for c in [
            "date",
            "bmpi_score",
            "cfgi_value",
            "cfgi_norm",
            "cfgi_label",
            "excess_media_effect_usd",
            "btc_logret",
        ] if c in df.columns
    ]
    overlap = df[df["cfgi_norm"].notna()][overlap_cols].copy()
    overlap.to_csv(OUT_CSV, index=False)

    summary = {
        "n_overlap": corr.get("n_overlap"),
        "bmpi_cfgi_pearson": corr.get("bmpi_cfgi_pearson"),
        "bmpi_cfgi_spearman": corr.get("bmpi_cfgi_spearman"),
        "bmpi_excess_r": corr.get("bmpi_excess_r"),
        "cfgi_excess_r": corr.get("cfgi_excess_r"),
        "incremental_r2": corr.get("incremental_r2"),
        "granger": granger,
        "timestamp": datetime.now().isoformat(),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[OK] Saved: {OUT_CFGI_CSV}")
    print(f"[OK] Saved: {OUT_CSV}")
    print(f"[OK] Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()