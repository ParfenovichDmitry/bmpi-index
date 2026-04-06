# -*- coding: utf-8 -*-
"""
STEP 16 — JOHANSEN COINTEGRATION (CONSOLE VERSION, FIXED)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime  # ← ДОБАВИТЬ

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ---------------- PATHS ----------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_GDELT = BASE_DIR / "data" / "raw" / "gdelt"

EXCESS_CSV = DATA_PROCESSED / "excess_media_effect_daily.csv"
FEATURES_PAR = DATA_PROCESSED / "features_daily.parquet"
OUT_JSON = DATA_PROCESSED / "johansen_results.json"

# ---------------- CONFIG ----------------

JOHANSEN_DET_ORDER = 0
JOHANSEN_K_AR_DIFF = 2

# ---------------- HELPERS ----------------

def load_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).lower().strip() for c in df.columns]

    date_col = next((c for c in df.columns if c in ("date", "data", "day")), None)
    if date_col:
        df["date"] = pd.to_datetime(
            df[date_col].astype(str).str.strip().str[:10],
            format="%Y-%m-%d",
            errors="coerce"
        ).dt.normalize()

    return df


def find_gdelt_file() -> Path:
    candidates = [
        DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_balanced.csv",
        DATA_GDELT / "gdelt_btc_media_signal_balanced.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Balanced GDELT file not found.")


def fmt(x, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{d}f}"


def to_python_scalar(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ---------------- DATA ----------------

def build_dataset() -> pd.DataFrame:
    print("\nLOADING DATA...\n")

    excess = load_file(EXCESS_CSV)
    feat = load_file(FEATURES_PAR)
    gdelt = load_file(find_gdelt_file())

    df = excess[["date"]].copy()

    if "excess_media_effect_usd" in excess.columns:
        df["excess"] = pd.to_numeric(excess["excess_media_effect_usd"], errors="coerce")
    elif "media_effect_used" in excess.columns:
        df["excess"] = pd.to_numeric(excess["media_effect_used"], errors="coerce")
    else:
        raise ValueError("No excess_media_effect_usd or media_effect_used column found.")

    if "bmpi_score" in excess.columns:
        df["bmpi"] = pd.to_numeric(excess["bmpi_score"], errors="coerce")

    mc_col = next(
        (c for c in feat.columns if "btc" in c and ("mcap" in c or "kapitaliz" in c)),
        None
    )
    if mc_col is None:
        raise ValueError("BTC market cap column not found in features_daily.parquet")

    df["mcap"] = pd.to_numeric(feat[mc_col], errors="coerce")
    df["log_mcap"] = np.log(df["mcap"].clip(lower=1))

    m_col = next((c for c in gdelt.columns if c in ("mentions", "liczba_wzmianek", "mention_count")), None)
    if m_col is None:
        non_dt = [c for c in gdelt.columns if c != "date"]
        if not non_dt:
            raise ValueError("No mentions column found in GDELT file")
        m_col = non_dt[0]

    df["mentions"] = pd.to_numeric(gdelt[m_col], errors="coerce")
    df["log_mentions"] = np.log(df["mentions"].clip(lower=0.1))

    # symmetric log transform for signed excess
    x = df["excess"].fillna(0.0)
    df["log_excess"] = np.sign(x) * np.log(np.abs(x).clip(lower=1.0))

    df = df.dropna(subset=["date", "log_mcap", "log_excess", "log_mentions"]).sort_values("date").reset_index(drop=True)

    print(f"[OK] Dataset: {len(df)} rows")
    print()
    return df


# ---------------- ADF ----------------

def check_integration(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    print("INTEGRATION TESTS (ADF)\n")

    rows = []

    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(s) < 50:
            continue

        try:
            p_level = float(adfuller(s, autolag="AIC")[1])
        except Exception:
            p_level = np.nan

        try:
            p_diff = float(adfuller(s.diff().dropna(), autolag="AIC")[1])
        except Exception:
            p_diff = np.nan

        if pd.notna(p_level) and p_level < 0.05:
            order = "I(0)"
        elif pd.notna(p_diff) and p_diff < 0.05:
            order = "I(1)"
        else:
            order = "I(2)?"

        print(f"{col:20} p(level)={fmt(p_level)}  p(diff)={fmt(p_diff)}  -> {order}")

        rows.append({
            "variable": col,
            "p_level": p_level,
            "p_diff": p_diff,
            "order": order,
            "is_i1": order == "I(1)",
        })

    print()
    return pd.DataFrame(rows)


# ---------------- ENGLE-GRANGER ----------------

def engle_granger(df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> List[Dict]:
    print("ENGLE-GRANGER TEST\n")

    res = []

    for x, y in pairs:
        data = df[[x, y]].dropna()
        if len(data) < 50:
            continue

        stat, p, _ = coint(data[x], data[y])

        print(f"{x} vs {y}  stat={stat:.4f}  p={p:.4f}")

        res.append({
            "x": x,
            "y": y,
            "stat": float(stat),
            "p_value": float(p),
            "cointegrated": bool(p < 0.05),
        })

    print()
    return res


# ---------------- JOHANSEN ----------------

def johansen_test(df: pd.DataFrame, i1_cols: List[str]) -> Dict:
    print("JOHANSEN TEST\n")

    # Johansen only for I(1) variables
    if len(i1_cols) < 2:
        print("Not enough I(1) variables for Johansen.")
        print()
        return {
            "applicable": False,
            "reason": "Less than two I(1) variables",
            "system_cols": i1_cols,
            "rank": None,
        }

    data = df[i1_cols].dropna()

    if len(data) < 100:
        print("Too few observations for Johansen.")
        print()
        return {
            "applicable": False,
            "reason": "Too few observations",
            "system_cols": i1_cols,
            "rank": None,
        }

    result = coint_johansen(data, JOHANSEN_DET_ORDER, JOHANSEN_K_AR_DIFF)

    trace_stats = result.lr1
    trace_cv95 = result.cvt[:, 1]

    print(f"{'H0':10} {'stat':>10} {'cv95':>10} {'reject?':>10}")
    print("-" * 46)

    rank = 0
    for i in range(len(i1_cols)):
        reject = bool(trace_stats[i] > trace_cv95[i])
        if reject:
            rank += 1
        print(f"r <= {i:<6} {trace_stats[i]:>10.2f} {trace_cv95[i]:>10.2f} {str(reject):>10}")

    # Important interpretation fix:
    # full rank in a system of supposedly I(1) variables usually means
    # the specification is not interpretable as standard cointegration.
    full_rank = rank >= len(i1_cols)

    print(f"\nCointegration rank raw = {rank}")

    if full_rank:
        print("Warning: full rank detected.")
        print("This is NOT interpreted as meaningful long-run cointegration.")
        print("It suggests the system/specification is not suitable for standard Johansen interpretation.")
        print()
        return {
            "applicable": False,
            "reason": "Full rank / non-interpretable Johansen outcome",
            "system_cols": i1_cols,
            "rank": int(rank),
            "full_rank": True,
            "trace_stats": [float(x) for x in trace_stats],
            "trace_cv95": [float(x) for x in trace_cv95],
        }

    print()
    return {
        "applicable": True,
        "reason": None,
        "system_cols": i1_cols,
        "rank": int(rank),
        "full_rank": False,
        "trace_stats": [float(x) for x in trace_stats],
        "trace_cv95": [float(x) for x in trace_cv95],
    }


# ---------------- MAIN ----------------

def main():
    df = build_dataset()

    cols = ["log_mcap", "log_excess", "log_mentions"]
    integration_df = check_integration(df, cols)

    i1_cols = integration_df.loc[integration_df["is_i1"], "variable"].tolist()

    eg_res = engle_granger(df, [
        ("log_mcap", "log_excess"),
        ("log_mcap", "log_mentions"),
    ])

    johansen_res = johansen_test(df, i1_cols)

    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(f"I(1) variables: {i1_cols}")

    eg_main = next((r for r in eg_res if r["x"] == "log_mcap" and r["y"] == "log_excess"), None)

    if eg_main is not None:
        print(f"Engle-Granger log_mcap vs log_excess: p = {eg_main['p_value']:.4f}")

    if not johansen_res["applicable"]:
        print("Johansen: NOT APPLICABLE / NOT INTERPRETABLE")
        print(f"Reason: {johansen_res['reason']}")
        print("Conclusion: no reliable evidence of long-run equilibrium.")
        print("Interpretation: BMPI / media pressure acts primarily through short-run dynamics.")
    else:
        print(f"Johansen rank: {johansen_res['rank']}")
        if johansen_res["rank"] == 0:
            print("Conclusion: no cointegration, short-run only.")
        else:
            print("Conclusion: long-run relationship exists, VECM may be justified.")

    results = {
        "timestamp": datetime.now().isoformat(),
        "integration": [
            {
                "variable": str(row["variable"]),
                "p_level": None if pd.isna(row["p_level"]) else float(row["p_level"]),
                "p_diff": None if pd.isna(row["p_diff"]) else float(row["p_diff"]),
                "order": str(row["order"]),
                "is_i1": bool(row["is_i1"]),
            }
            for _, row in integration_df.iterrows()
        ],
        "engle_granger": eg_res,
        "johansen": {
            "applicable": bool(johansen_res.get("applicable", False)),
            "reason": johansen_res.get("reason"),
            "system_cols": [str(x) for x in johansen_res.get("system_cols", [])],
            "rank": None if johansen_res.get("rank") is None else int(johansen_res.get("rank")),
            "full_rank": bool(johansen_res.get("full_rank", False)),
            "trace_stats": [float(x) for x in johansen_res.get("trace_stats", [])],
            "trace_cv95": [float(x) for x in johansen_res.get("trace_cv95", [])],
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[OK] Saved results")


if __name__ == "__main__":
    main()