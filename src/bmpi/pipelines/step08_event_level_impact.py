# -*- coding: utf-8 -*-
"""
pipelines/step08_event_level_impact.py
========================================
BMPI v2: compute event-level media impact from daily predicted media effect
and event abnormal-return statistics.

What changed vs old step08:
- Compatible with BMPI v2 outputs from step05 + step07.
- Uses predicted_media_effect_usd_oof as the PRIMARY honest signal.
- Uses predicted_media_effect_usd as supplementary in-sample signal.
- Measures media share relative to abnormal event move, not old raw residual logic.
- Integrates event-level outputs already computed in step05.

Inputs:
  data/processed/news_effect_daily.csv
  data/processed/residuals_by_event_balanced.csv
  data/processed/residuals_by_event_strong.csv
  data/processed/residuals_by_event_sensitive.csv

Outputs:
  data/processed/news_effect_by_event.csv
  data/processed/news_effect_top_events.csv
  data/processed/news_effect_by_event_summary.json

Key concepts:
- event_window_sum_abs_abnormal_mcap_usd    from step05
- predicted_media_effect_usd_oof            from step07
- media_share_of_event_window_pct           = |media OOF| / |abnormal move| * 100

Next step:
  step09_fake_classification.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

NEWS_EFFECT_CSV = DATA_PROCESSED / "news_effect_daily.csv"

EVENT_FILES = {
    "balanced": DATA_PROCESSED / "residuals_by_event_balanced.csv",
    "strong": DATA_PROCESSED / "residuals_by_event_strong.csv",
    "sensitive": DATA_PROCESSED / "residuals_by_event_sensitive.csv",
}

OUT_BY_EVENT = DATA_PROCESSED / "news_effect_by_event.csv"
OUT_TOP_EVENTS = DATA_PROCESSED / "news_effect_top_events.csv"
OUT_SUMMARY = DATA_PROCESSED / "news_effect_by_event_summary.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOP_EVENTS_N = 50

PRIMARY_EFFECT_COL = "predicted_media_effect_usd_oof"
SECONDARY_EFFECT_COL = "predicted_media_effect_usd"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_sum(series: pd.Series) -> float:
    s = _safe_num(series).fillna(0.0)
    return float(s.sum())


def _safe_abs_sum(series: pd.Series) -> float:
    s = _safe_num(series).fillna(0.0)
    return float(s.abs().sum())


def _safe_max_abs(series: pd.Series) -> float:
    s = _safe_num(series).fillna(0.0)
    if len(s) == 0:
        return 0.0
    return float(s.abs().max())


def _safe_mean(series: pd.Series) -> float:
    s = _safe_num(series).dropna()
    if len(s) == 0:
        return float("nan")
    return float(s.mean())


def _safe_median(series: pd.Series) -> float:
    s = _safe_num(series).dropna()
    if len(s) == 0:
        return float("nan")
    return float(s.median())


def _safe_share_pct(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return 0.0
    return float(100.0 * numerator / denominator)


def load_news_effect_daily() -> pd.DataFrame:
    if not NEWS_EFFECT_CSV.exists():
        raise FileNotFoundError(
            f"File not found: {NEWS_EFFECT_CSV}\n"
            "Run step07_news_effect_model.py first."
        )

    df = pd.read_csv(NEWS_EFFECT_CSV)

    date_col = "date" if "date" in df.columns else "data"
    if date_col not in df.columns:
        raise ValueError("news_effect_daily.csv: date column not found")

    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    required_any = [PRIMARY_EFFECT_COL, SECONDARY_EFFECT_COL]
    if not any(col in df.columns for col in required_any):
        raise ValueError(
            "news_effect_daily.csv does not contain media effect columns.\n"
            f"Expected one of: {required_any}"
        )

    # Normalize missing columns if needed
    if PRIMARY_EFFECT_COL not in df.columns and SECONDARY_EFFECT_COL in df.columns:
        df[PRIMARY_EFFECT_COL] = _safe_num(df[SECONDARY_EFFECT_COL])

    if SECONDARY_EFFECT_COL not in df.columns and PRIMARY_EFFECT_COL in df.columns:
        df[SECONDARY_EFFECT_COL] = _safe_num(df[PRIMARY_EFFECT_COL])

    numeric_cols = [
        PRIMARY_EFFECT_COL,
        SECONDARY_EFFECT_COL,
        "predicted_media_abnormal_logret",
        "predicted_media_abnormal_logret_oof",
        "abnormal_btc_mcap_usd",
        "bmip_v2_daily",
        "bmip_v2_daily_abs",
        "bmip_v2_share_of_abnormal_move",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _safe_num(df[col])

    return df


def load_event_file(path: Path, preset: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")

    df = pd.read_csv(path)

    if "peak_date" not in df.columns:
        raise ValueError(f"{path.name}: required column 'peak_date' not found")

    df["peak_date"] = pd.to_datetime(df["peak_date"], errors="coerce").dt.normalize()

    for col in ["window_start", "window_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()

    df["preset"] = preset
    df = df.dropna(subset=["peak_date"]).sort_values("peak_date").reset_index(drop=True)
    return df


def get_effect_window(df_daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df_daily[
        (df_daily["date"] >= start) &
        (df_daily["date"] <= end)
    ].copy()


def get_relative_window(df_daily: pd.DataFrame, peak_date: pd.Timestamp, days_before: int, days_after: int) -> pd.DataFrame:
    start = peak_date - pd.Timedelta(days=days_before)
    end = peak_date + pd.Timedelta(days=days_after)
    return get_effect_window(df_daily, start, end)


def choose_event_window(row: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Prefer step05 event window if available, otherwise fallback to peak-centered ±30 days.
    """
    if "window_start" in row.index and "window_end" in row.index:
        ws = row["window_start"]
        we = row["window_end"]
        if pd.notna(ws) and pd.notna(we):
            return ws, we

    peak = row["peak_date"]
    return peak - pd.Timedelta(days=30), peak + pd.Timedelta(days=30)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def build_event_impact_table(df_daily: pd.DataFrame, df_events: pd.DataFrame, preset: str) -> pd.DataFrame:
    rows: List[Dict] = []

    for idx, row in df_events.iterrows():
        peak_date = row["peak_date"]
        window_start, window_end = choose_event_window(row)

        event_df = get_effect_window(df_daily, window_start, window_end)
        w7_df = get_relative_window(df_daily, peak_date, 7, 7)
        w14_df = get_relative_window(df_daily, peak_date, 14, 14)
        w30_df = get_relative_window(df_daily, peak_date, 30, 30)

        event_id = row["event_id"] if "event_id" in row.index and pd.notna(row["event_id"]) else f"{preset.upper()}_EVT_{idx+1:04d}"

        out: Dict = {
            "preset": preset,
            "event_number": int(idx + 1),
            "event_id": str(event_id),
            "peak_date": str(peak_date.date()),
            "window_start": str(window_start.date()),
            "window_end": str(window_end.date()),
            "event_window_days_effect": int(len(event_df)),
            "w7_days_effect": int(len(w7_df)),
            "w14_days_effect": int(len(w14_df)),
            "w30_days_effect": int(len(w30_df)),
        }

        # ---------------------------------------------------------------
        # Primary OOF media effect
        # ---------------------------------------------------------------
        out["event_window_sum_media_effect_usd_oof"] = _safe_sum(event_df[PRIMARY_EFFECT_COL])
        out["event_window_sum_abs_media_effect_usd_oof"] = _safe_abs_sum(event_df[PRIMARY_EFFECT_COL])
        out["event_window_max_abs_media_effect_usd_oof"] = _safe_max_abs(event_df[PRIMARY_EFFECT_COL])

        out["w7_sum_abs_media_effect_usd_oof"] = _safe_abs_sum(w7_df[PRIMARY_EFFECT_COL])
        out["w14_sum_abs_media_effect_usd_oof"] = _safe_abs_sum(w14_df[PRIMARY_EFFECT_COL])
        out["w30_sum_abs_media_effect_usd_oof"] = _safe_abs_sum(w30_df[PRIMARY_EFFECT_COL])

        # ---------------------------------------------------------------
        # Secondary in-sample media effect
        # ---------------------------------------------------------------
        out["event_window_sum_media_effect_usd_in"] = _safe_sum(event_df[SECONDARY_EFFECT_COL])
        out["event_window_sum_abs_media_effect_usd_in"] = _safe_abs_sum(event_df[SECONDARY_EFFECT_COL])
        out["event_window_max_abs_media_effect_usd_in"] = _safe_max_abs(event_df[SECONDARY_EFFECT_COL])

        # ---------------------------------------------------------------
        # Bring forward step05 abnormal-event metrics if available
        # ---------------------------------------------------------------
        passthrough_cols = [
            "peak_abnormal_logret",
            "peak_abnormal_return_pct",
            "peak_abnormal_zscore_60d",
            "peak_abnormal_mcap_usd",
            "pre_event_car_log",
            "pre_event_car_pct",
            "post_event_car_log",
            "post_event_car_pct",
            "reversal_ratio",
            "is_reversal",
            "event_window_days",
            "event_window_car_log",
            "event_window_car_pct",
            "event_window_mean_abnormal_logret",
            "event_window_mean_abnormal_return_pct",
            "event_window_sum_abnormal_mcap_usd",
            "event_window_sum_abs_abnormal_mcap_usd",
            "car_3d_log",
            "car_3d_pct",
            "car_3d_sum_abnormal_mcap_usd",
            "car_3d_sum_abs_abnormal_mcap_usd",
            "car_7d_log",
            "car_7d_pct",
            "car_7d_sum_abnormal_mcap_usd",
            "car_7d_sum_abs_abnormal_mcap_usd",
            "car_14d_log",
            "car_14d_pct",
            "car_14d_sum_abnormal_mcap_usd",
            "car_14d_sum_abs_abnormal_mcap_usd",
            "car_30d_log",
            "car_30d_pct",
            "car_30d_sum_abnormal_mcap_usd",
            "car_30d_sum_abs_abnormal_mcap_usd",
        ]
        for col in passthrough_cols:
            if col in row.index:
                out[col] = row[col]

        # ---------------------------------------------------------------
        # Shares relative to abnormal event move
        # ---------------------------------------------------------------
        event_abs_abnormal = row["event_window_sum_abs_abnormal_mcap_usd"] if "event_window_sum_abs_abnormal_mcap_usd" in row.index else np.nan
        car7_abs_abnormal = row["car_7d_sum_abs_abnormal_mcap_usd"] if "car_7d_sum_abs_abnormal_mcap_usd" in row.index else np.nan
        car14_abs_abnormal = row["car_14d_sum_abs_abnormal_mcap_usd"] if "car_14d_sum_abs_abnormal_mcap_usd" in row.index else np.nan
        car30_abs_abnormal = row["car_30d_sum_abs_abnormal_mcap_usd"] if "car_30d_sum_abs_abnormal_mcap_usd" in row.index else np.nan

        out["media_share_of_event_window_pct_oof"] = _safe_share_pct(
            out["event_window_sum_abs_media_effect_usd_oof"],
            event_abs_abnormal,
        )
        out["media_share_of_event_window_pct_in"] = _safe_share_pct(
            out["event_window_sum_abs_media_effect_usd_in"],
            event_abs_abnormal,
        )

        out["media_share_of_car7_pct_oof"] = _safe_share_pct(
            out["w7_sum_abs_media_effect_usd_oof"],
            car7_abs_abnormal,
        )
        out["media_share_of_car14_pct_oof"] = _safe_share_pct(
            out["w14_sum_abs_media_effect_usd_oof"],
            car14_abs_abnormal,
        )
        out["media_share_of_car30_pct_oof"] = _safe_share_pct(
            out["w30_sum_abs_media_effect_usd_oof"],
            car30_abs_abnormal,
        )

        # ---------------------------------------------------------------
        # Additional diagnostic ratios
        # ---------------------------------------------------------------
        peak_abs_abnormal = abs(float(row["peak_abnormal_mcap_usd"])) if "peak_abnormal_mcap_usd" in row.index and pd.notna(row["peak_abnormal_mcap_usd"]) else np.nan
        out["media_share_of_peak_day_pct_oof"] = _safe_share_pct(
            out["event_window_max_abs_media_effect_usd_oof"],
            peak_abs_abnormal,
        )

        rows.append(out)

    result = pd.DataFrame(rows).sort_values(["preset", "peak_date"]).reset_index(drop=True)
    return result


def build_summary(by_event: pd.DataFrame) -> Dict:
    summary: Dict = {
        "by_preset": {},
        "global": {},
    }

    for preset in sorted(by_event["preset"].dropna().unique()):
        sub = by_event[by_event["preset"] == preset].copy()

        summary["by_preset"][preset] = {
            "n_events": int(len(sub)),
            "sum_abs_media_effect_usd_oof": float(_safe_num(sub["event_window_sum_abs_media_effect_usd_oof"]).sum()),
            "mean_abs_media_effect_usd_oof": _safe_mean(sub["event_window_sum_abs_media_effect_usd_oof"]),
            "max_abs_media_effect_usd_oof": float(_safe_num(sub["event_window_sum_abs_media_effect_usd_oof"]).max()),
            "mean_media_share_event_window_pct_oof": _safe_mean(sub["media_share_of_event_window_pct_oof"]),
            "median_media_share_event_window_pct_oof": _safe_median(sub["media_share_of_event_window_pct_oof"]),
            "mean_media_share_car7_pct_oof": _safe_mean(sub["media_share_of_car7_pct_oof"]),
            "mean_media_share_car14_pct_oof": _safe_mean(sub["media_share_of_car14_pct_oof"]),
            "mean_media_share_car30_pct_oof": _safe_mean(sub["media_share_of_car30_pct_oof"]),
            "mean_event_window_car_pct": _safe_mean(sub["event_window_car_pct"]) if "event_window_car_pct" in sub.columns else float("nan"),
            "mean_car_7d_pct": _safe_mean(sub["car_7d_pct"]) if "car_7d_pct" in sub.columns else float("nan"),
            "share_reversal_events": _safe_mean(sub["is_reversal"]) if "is_reversal" in sub.columns else float("nan"),
        }

    summary["global"] = {
        "total_events": int(len(by_event)),
        "sum_abs_media_effect_usd_oof": float(_safe_num(by_event["event_window_sum_abs_media_effect_usd_oof"]).sum()),
        "mean_abs_media_effect_usd_oof": _safe_mean(by_event["event_window_sum_abs_media_effect_usd_oof"]),
        "max_abs_media_effect_usd_oof": float(_safe_num(by_event["event_window_sum_abs_media_effect_usd_oof"]).max()),
        "mean_media_share_event_window_pct_oof": _safe_mean(by_event["media_share_of_event_window_pct_oof"]),
        "median_media_share_event_window_pct_oof": _safe_median(by_event["media_share_of_event_window_pct_oof"]),
    }

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("STEP 08 — EVENT-LEVEL MEDIA IMPACT (BMPI v2)")
    print("=" * 60 + "\n")

    df_daily = load_news_effect_daily()
    print(
        f"  Daily media-effect data: {len(df_daily)} rows  "
        f"({df_daily['date'].min().date()} -> {df_daily['date'].max().date()})"
    )

    event_tables: List[pd.DataFrame] = []
    loaded_presets: List[str] = []

    for preset, path in EVENT_FILES.items():
        if not path.exists():
            print(f"  [{preset}] event file not found — skipping: {path.name}")
            continue

        df_events = load_event_file(path, preset)
        print(f"  [{preset}] loaded events: {len(df_events)}")

        event_table = build_event_impact_table(df_daily, df_events, preset)
        event_tables.append(event_table)
        loaded_presets.append(preset)

    if not event_tables:
        raise FileNotFoundError(
            "No residuals_by_event_*.csv files found.\n"
            "Run step05_residuals.py first."
        )

    by_event = pd.concat(event_tables, ignore_index=True)
    by_event = by_event.sort_values(["preset", "peak_date"]).reset_index(drop=True)
    by_event.to_csv(OUT_BY_EVENT, index=False)

    top_events = by_event.sort_values("event_window_sum_abs_media_effect_usd_oof", ascending=False).head(TOP_EVENTS_N).copy()
    top_events.to_csv(OUT_TOP_EVENTS, index=False)

    summary = build_summary(by_event)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Loaded presets:            {', '.join(loaded_presets)}")
    print(f"  Event rows saved:          {len(by_event)}")
    print(f"  Top events saved:          {len(top_events)}")
    print(f"  Saved by-event CSV:        {OUT_BY_EVENT}")
    print(f"  Saved top-events CSV:      {OUT_TOP_EVENTS}")
    print(f"  Saved summary JSON:        {OUT_SUMMARY}")

    print("\n  Quick summary:")
    for preset, stats in summary["by_preset"].items():
        print(
            f"    [{preset}] "
            f"n={stats['n_events']}  "
            f"mean_share_event={stats['mean_media_share_event_window_pct_oof']:.2f}%  "
            f"mean_share_car7={stats['mean_media_share_car7_pct_oof']:.2f}%  "
            f"sum_abs_oof=${stats['sum_abs_media_effect_usd_oof']:,.0f}"
        )

    print("\nNext step: step09_fake_classification.py")


if __name__ == "__main__":
    main()