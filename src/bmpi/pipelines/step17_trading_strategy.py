# -*- coding: utf-8 -*-
"""
STEP 17 — BMPI TRADING STRATEGY (CONSOLE, FIXED)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ================= PATHS =================

BASE_DIR = Path(__file__).resolve().parents[3]
DATA = BASE_DIR / "data" / "processed"

EXCESS_FILE = DATA / "excess_media_effect_daily.csv"
FEATURES_FILE = DATA / "features_daily.parquet"


# ================= HELPERS =================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = next((c for c in df.columns if c in ("date", "data", "day")), None)
    if date_col is None:
        raise ValueError("Date column not found.")
    df["date"] = pd.to_datetime(
        df[date_col].astype(str).str.strip().str[:10],
        format="%Y-%m-%d",
        errors="coerce"
    ).dt.normalize()
    df = df.dropna(subset=["date"])
    return df


def find_price_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if "btc" in c and any(k in c for k in ("price", "close", "cena"))
    ]
    if candidates:
        return candidates[0]

    candidates = [
        c for c in df.columns
        if any(k in c for k in ("price", "close", "cena"))
    ]
    if candidates:
        return candidates[0]

    return None


def safe_sharpe(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2 or x.std(ddof=0) == 0:
        return np.nan
    return float(x.mean() / x.std(ddof=0) * np.sqrt(365))


def safe_max_drawdown(cum_returns: pd.Series) -> float:
    cum_returns = pd.to_numeric(cum_returns, errors="coerce").dropna()
    if len(cum_returns) == 0:
        return np.nan
    peak = cum_returns.cummax()
    drawdown = cum_returns - peak
    return float(drawdown.min())


def safe_total_return(log_returns: pd.Series) -> float:
    log_returns = pd.to_numeric(log_returns, errors="coerce").dropna()
    if len(log_returns) == 0:
        return np.nan
    return float(log_returns.sum())


def print_metrics(title: str, m: Dict[str, float]) -> None:
    print(f"\n{title}:")
    print(f"  Total return:   {m['total_return']:.4f}" if pd.notna(m["total_return"]) else "  Total return:   —")
    print(f"  Sharpe ratio:   {m['sharpe']:.4f}" if pd.notna(m["sharpe"]) else "  Sharpe ratio:   —")
    print(f"  Max drawdown:   {m['max_drawdown']:.4f}" if pd.notna(m["max_drawdown"]) else "  Max drawdown:   —")
    print(f"  N trades days:  {int(m['active_days'])}" if pd.notna(m["active_days"]) else "  N trades days:  —")


# ================= LOAD =================

def load_data() -> pd.DataFrame:
    if not EXCESS_FILE.exists():
        raise FileNotFoundError(f"File not found: {EXCESS_FILE}")
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"File not found: {FEATURES_FILE}")

    df_excess = pd.read_csv(EXCESS_FILE)
    df_feat = pd.read_parquet(FEATURES_FILE)

    df_excess = normalize_columns(df_excess)
    df_feat = normalize_columns(df_feat)

    df_excess = normalize_date_column(df_excess)
    df_feat = normalize_date_column(df_feat)

    keep_excess = ["date"]
    for c in (
        "bmpi_score",
        "excess_media_effect_usd",
        "media_effect_used",
        "raw_abs_media_effect_usd",
        "media_share_of_abnormal_move_pct",
    ):
        if c in df_excess.columns:
            keep_excess.append(c)

    keep_feat = ["date"]
    for c in (
        "btc_logret",
        "btc_price",
        "btc_close",
        "btc_mcap",
    ):
        if c in df_feat.columns:
            keep_feat.append(c)

    price_col = find_price_col(df_feat)
    if price_col and price_col not in keep_feat:
        keep_feat.append(price_col)

    df_excess = df_excess[keep_excess].drop_duplicates(subset=["date"]).copy()
    df_feat = df_feat[keep_feat].drop_duplicates(subset=["date"]).copy()

    df = df_excess.merge(df_feat, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ================= PREP =================

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "bmpi_score" not in df.columns:
        raise ValueError("Column bmpi_score not found.")

    df["bmpi_score"] = pd.to_numeric(df["bmpi_score"], errors="coerce")

    if "btc_logret" in df.columns and df["btc_logret"].notna().sum() > 10:
        df["ret"] = pd.to_numeric(df["btc_logret"], errors="coerce")
    else:
        price_col = None
        for c in ("btc_price", "btc_close"):
            if c in df.columns and df[c].notna().sum() > 10:
                price_col = c
                break

        if price_col is None:
            price_col = find_price_col(df)

        if price_col is None:
            raise ValueError("No BTC price column found to compute returns.")

        price = pd.to_numeric(df[price_col], errors="coerce")
        df["ret"] = np.log(price / price.shift(1))

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["date", "bmpi_score", "ret"]).reset_index(drop=True)

    return df


# ================= STRATEGIES =================

def run_long_short(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = np.where(out["bmpi_score"] < threshold, 1.0, -1.0)
    out["strategy_ret"] = out["signal"].shift(1) * out["ret"]
    out = out.dropna(subset=["strategy_ret"]).reset_index(drop=True)
    return out


def run_long_only(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = np.where(out["bmpi_score"] < threshold, 1.0, 0.0)
    out["strategy_ret"] = out["signal"].shift(1) * out["ret"]
    out = out.dropna(subset=["strategy_ret"]).reset_index(drop=True)
    return out


def run_cash_on_high_bmpi(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = np.where(out["bmpi_score"] >= threshold, 0.0, 1.0)
    out["strategy_ret"] = out["signal"].shift(1) * out["ret"]
    out = out.dropna(subset=["strategy_ret"]).reset_index(drop=True)
    return out


# ================= METRICS =================

def compute_metrics(df: pd.DataFrame, ret_col: str) -> Dict[str, float]:
    rets = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    if len(rets) == 0:
        return {
            "total_return": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "active_days": np.nan,
        }

    cum = rets.cumsum()

    return {
        "total_return": safe_total_return(rets),
        "sharpe": safe_sharpe(rets),
        "max_drawdown": safe_max_drawdown(cum),
        "active_days": float((rets != 0).sum()),
    }


# ================= MAIN =================

def main() -> None:
    print("=" * 60)
    print("STEP 17 — BMPI TRADING STRATEGY")
    print("=" * 60)

    df = load_data()

    print(f"\nLoaded merged rows: {len(df)}")
    if len(df) == 0:
        raise ValueError(
            "Merged DataFrame is empty.\n"
            "Most likely dates do not overlap after normalization."
        )

    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Columns: {list(df.columns)}")

    df = prepare(df)

    print(f"\nRows after prepare: {len(df)}")
    if len(df) == 0:
        raise ValueError("No data left after prepare(). Check bmpi_score / ret columns.")

    threshold = float(df["bmpi_score"].quantile(0.70))
    print(f"BMPI threshold (70th pct): {threshold:.4f}")

    # Baseline BTC
    btc_df = df.copy()
    btc_df["btc_ret"] = btc_df["ret"]
    btc_metrics = compute_metrics(btc_df, "btc_ret")

    # Strategy 1: Long/Short
    ls_df = run_long_short(df, threshold)
    ls_metrics = compute_metrics(ls_df, "strategy_ret")

    # Strategy 2: Long only below threshold
    lo_df = run_long_only(df, threshold)
    lo_metrics = compute_metrics(lo_df, "strategy_ret")

    # Strategy 3: Cash on high BMPI
    cash_df = run_cash_on_high_bmpi(df, threshold)
    cash_metrics = compute_metrics(cash_df, "strategy_ret")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print_metrics("BTC BUY & HOLD", btc_metrics)
    print_metrics("STRATEGY 1 — LONG/SHORT", ls_metrics)
    print_metrics("STRATEGY 2 — LONG ONLY", lo_metrics)
    print_metrics("STRATEGY 3 — CASH ON HIGH BMPI", cash_metrics)

    print("\n" + "=" * 60)
    print("COMPARISON VS BTC")
    print("=" * 60)

    for name, m in [
        ("LONG/SHORT", ls_metrics),
        ("LONG ONLY", lo_metrics),
        ("CASH ON HIGH BMPI", cash_metrics),
    ]:
        better_sharpe = (
            pd.notna(m["sharpe"]) and pd.notna(btc_metrics["sharpe"]) and m["sharpe"] > btc_metrics["sharpe"]
        )
        lower_dd = (
            pd.notna(m["max_drawdown"]) and pd.notna(btc_metrics["max_drawdown"]) and abs(m["max_drawdown"]) < abs(btc_metrics["max_drawdown"])
        )

        print(f"\n{name}:")
        print("  ✓ Better Sharpe than BTC" if better_sharpe else "  ⚠ Sharpe not better than BTC")
        print("  ✓ Lower drawdown than BTC" if lower_dd else "  ⚠ Drawdown not lower than BTC")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()