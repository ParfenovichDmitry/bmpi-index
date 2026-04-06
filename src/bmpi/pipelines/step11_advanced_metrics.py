# -*- coding: utf-8 -*-
"""
pipelines/step11_advanced_metrics.py
======================================
BMPI v2 — advanced metrics (console-only version, no HTML output).

What it does:
  Computes advanced event-level and global metrics for BMPI v2
  using the current pipeline outputs:
    - step04 baseline_predictions.csv
    - step05 residuals_by_event_balanced.csv
    - step08 news_effect_by_event.csv
    - step09 excess_media_effect_daily.csv
    - step07 news_effect_daily.csv
    - optional JSONL article file for source analysis

Outputs:
  data/processed/advanced_metrics_per_event.csv
  data/processed/advanced_metrics_global.json

Console output only:
  - no HTML report
  - prints compact summary tables to stdout

Next step:
  step12_cross_preset_analysis.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_RAW_NEWS = BASE_DIR / "data" / "raw" / "news_raw"

EVENT_METRICS_CSV = DATA_PROCESSED / "residuals_by_event_balanced.csv"
EVENT_IMPACT_CSV = DATA_PROCESSED / "news_effect_by_event.csv"
EXCESS_MEDIA_CSV = DATA_PROCESSED / "excess_media_effect_daily.csv"
NEWS_EFFECT_CSV = DATA_PROCESSED / "news_effect_daily.csv"
BASELINE_CSV = DATA_PROCESSED / "baseline_predictions.csv"

JSONL_PATHS = [
    DATA_RAW_NEWS / "gdelt_events_peaks_articles" / "gdelt_events_peak_articles.jsonl",
    DATA_RAW_NEWS / "gdelt_events_peaks" / "gdelt_events_peak_articles.jsonl",
]

OUT_CSV = DATA_PROCESSED / "advanced_metrics_per_event.csv"
OUT_JSON = DATA_PROCESSED / "advanced_metrics_global.json"

# ---------------------------------------------------------------------------
# BMPI weights
# ---------------------------------------------------------------------------

BMPI_W1_NEWS_SPIKE = 0.25
BMPI_W2_EXTREME_TONE = 0.20
BMPI_W3_SOURCE_CONC = 0.20
BMPI_W4_PEAK_ABNORMAL = 0.20
BMPI_W5_REVERSAL = 0.15

LOW_CRED = {
    "news.bitcoin.com", "newsbtc.com", "cryptocoinsnews.com", "ccn.com",
    "altcointoday.com", "thebitcoinnews.com", "coinreport.net", "bravenewcoin.com",
    "coinspeaker.com", "4-traders.com", "econotimes.com", "zerohedge.com",
    "cryptonewsz.com", "insidebitcoins.com", "nulltx.com", "zycrypto.com",
    "ambcrypto.com", "coingape.com", "cryptodaily.co.uk", "u.today",
    "coinpedia.org", "cryptopotato.com", "beincrypto.com",
    "ibtimes.com", "ibtimes.co.uk", "beforeitsnews.com",
}

TRUSTED = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com", "forbes.com",
    "fortune.com", "businessinsider.com", "marketwatch.com", "finance.yahoo.com",
    "nasdaq.com", "bbc.com", "bbc.co.uk", "coindesk.com", "cointelegraph.com",
    "bitcoinmagazine.com", "theblock.co", "decrypt.co", "apnews.com", "theguardian.com",
    "bis.org", "ecb.europa.eu", "sec.gov", "cftc.gov",
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def z_norm(s: pd.Series) -> pd.Series:
    s = _safe_num(s)
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd < 1e-9:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-3, 3)


def classify_domain(d: str) -> str:
    d = str(d or "").lower().replace("www.", "").strip()
    for ld in LOW_CRED:
        if d == ld or d.endswith("." + ld):
            return "low_cred"
    for td in TRUSTED:
        if d == td or d.endswith("." + td):
            return "trusted"
    return "unknown"


def fmt_usd(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = float(x)
    if abs(x) >= 1e12:
        return f"{x / 1e12:.3f} T USD"
    if abs(x) >= 1e9:
        return f"{x / 1e9:.3f} B USD"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f} M USD"
    return f"{x:,.0f} USD"


def fmt_pct(x, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{decimals}f}%"


def find_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------


def load_csv(path: Path, required: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"File not found: {path}")
        return None
    df = pd.read_csv(path)
    df = _norm_cols(df)
    return df


def load_articles() -> List[dict]:
    jsonl_path = next((p for p in JSONL_PATHS if p.exists()), None)
    if jsonl_path is None:
        return []

    articles: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(json.loads(line))
            except Exception:
                continue
    return articles


# ---------------------------------------------------------------------------
# BUILD ARTICLE STATS
# ---------------------------------------------------------------------------


def build_article_stats(articles: List[dict]) -> Dict[str, dict]:
    """
    event_id -> article/source stats
    """
    by_event = defaultdict(lambda: {
        "total": 0.0,
        "low_cred": 0.0,
        "trusted": 0.0,
        "unknown": 0.0,
        "source_counts": defaultdict(float),
        "tones": [],
    })

    for a in articles:
        eid = str(a.get("event_id", "")).strip()
        if not eid:
            continue

        dom = str(a.get("source_domain", "")).lower().replace("www.", "").strip()
        mentions = float(a.get("num_articles") or a.get("num_mentions") or 1.0)
        tone = a.get("avg_tone")

        cred = classify_domain(dom)
        ev = by_event[eid]
        ev["total"] += mentions
        ev[cred] += mentions
        ev["source_counts"][dom] += mentions

        if tone is not None:
            try:
                ev["tones"].append(float(tone))
            except Exception:
                pass

    return dict(by_event)


def compute_hhi_for_articles(art_stats: Dict[str, dict]) -> Dict[str, float]:
    """
    HHI per event from source distribution.
    """
    out: Dict[str, float] = {}
    for eid, ev in art_stats.items():
        total = float(ev.get("total", 0.0))
        counts = ev.get("source_counts", {})
        if total <= 0 or not counts:
            out[eid] = np.nan
            continue
        shares = np.array([v / total for v in counts.values()], dtype=float)
        out[eid] = float(np.sum(shares ** 2))
    return out


# ---------------------------------------------------------------------------
# SHOCK / REVERSAL
# ---------------------------------------------------------------------------


def compute_shock_reversal_from_daily(
    df_daily: pd.DataFrame,
    event_date: pd.Timestamp,
    return_col: str,
    fallback_col: Optional[str] = None,
) -> float:
    """
    shock_reversal_index = |t+3..t+7| / |t0..t+2|
    """
    mask_up = (df_daily["date"] >= event_date) & (df_daily["date"] <= event_date + pd.Timedelta(days=2))
    mask_dn = (df_daily["date"] > event_date + pd.Timedelta(days=2)) & (df_daily["date"] <= event_date + pd.Timedelta(days=7))

    up = _safe_num(df_daily.loc[mask_up, return_col]).dropna()
    dn = _safe_num(df_daily.loc[mask_dn, return_col]).dropna()

    if len(up) > 0 and len(dn) > 0:
        up_mag = float(abs(up.sum()))
        dn_mag = float(abs(dn.sum()))
        return float(dn_mag / up_mag) if up_mag > 0 else np.nan

    if fallback_col is not None and fallback_col in df_daily.columns:
        up2 = _safe_num(df_daily.loc[mask_up, fallback_col]).abs().dropna()
        dn2 = _safe_num(df_daily.loc[mask_dn, fallback_col]).abs().dropna()
        if len(up2) > 0 and len(dn2) > 0:
            up_mag = float(up2.mean())
            dn_mag = float(dn2.mean())
            return float(dn_mag / up_mag) if up_mag > 0 else np.nan

    return np.nan


# ---------------------------------------------------------------------------
# MAIN COMPUTATION
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 72)
    print("STEP 11 — ADVANCED METRICS (BMPI v2, CONSOLE VERSION)")
    print("=" * 72 + "\n")

    event_metrics = load_csv(EVENT_METRICS_CSV, required=True)
    event_impact = load_csv(EVENT_IMPACT_CSV, required=True)
    excess_daily = load_csv(EXCESS_MEDIA_CSV, required=True)
    news_daily = load_csv(NEWS_EFFECT_CSV, required=True)
    baseline = load_csv(BASELINE_CSV, required=True)

    articles = load_articles()

    # Normalize dates
    for df in [event_metrics, event_impact, excess_daily, news_daily, baseline]:
        date_col = find_first_existing(list(df.columns), ["date", "data"])
        if date_col:
            df["date"] = _to_date(df[date_col])

    if "peak_date" in event_metrics.columns:
        event_metrics["peak_date"] = _to_date(event_metrics["peak_date"])
    if "window_start" in event_metrics.columns:
        event_metrics["window_start"] = _to_date(event_metrics["window_start"])
    if "window_end" in event_metrics.columns:
        event_metrics["window_end"] = _to_date(event_metrics["window_end"])

    if "peak_date" in event_impact.columns:
        event_impact["peak_date"] = _to_date(event_impact["peak_date"])

    # Merge event tables
    key_cols = ["event_id", "preset", "peak_date"]
    merge_keys = [k for k in key_cols if k in event_metrics.columns and k in event_impact.columns]

    if not merge_keys:
        raise ValueError("Could not find common event merge keys between step05 and step08 outputs.")

    event_df = event_metrics.merge(
        event_impact,
        on=merge_keys,
        how="left",
        suffixes=("", "_impact"),
    )

    # Daily merged table for reversals / extra diagnostics
    daily = news_daily.merge(
        excess_daily[[c for c in ["date", "excess_media_effect_usd", "bmpi_score"] if c in excess_daily.columns]],
        on="date",
        how="left",
        suffixes=("", "_excess"),
    )

    # Detect columns
    return_col = find_first_existing(list(baseline.columns), ["btc_logret", "abnormal_btc_logret", "btc_return_pct"])
    if return_col is None:
        return_col = find_first_existing(list(daily.columns), ["predicted_media_abnormal_logret_oof", "predicted_media_abnormal_logret"])
    fallback_col = find_first_existing(list(daily.columns), ["abnormal_btc_mcap_usd", "predicted_media_effect_usd_oof", "predicted_media_effect_usd"])

    # Join baseline return column onto daily by date if needed
    if return_col is not None and return_col in baseline.columns:
        baseline_small = baseline[["date", return_col]].copy()
        daily = daily.merge(baseline_small, on="date", how="left", suffixes=("", "_baseline"))

    # Article stats
    art_stats = build_article_stats(articles)
    hhi_map = compute_hhi_for_articles(art_stats)

    # Prepare advanced metrics
    rows: List[Dict] = []

    for _, row in event_df.iterrows():
        eid = str(row["event_id"])
        peak_date = pd.Timestamp(row["peak_date"]) if "peak_date" in row.index and pd.notna(row["peak_date"]) else pd.NaT

        ev_art = art_stats.get(eid, {})
        total_mentions = float(ev_art.get("total", 0.0))
        low_cred_mentions = float(ev_art.get("low_cred", 0.0))
        trusted_mentions = float(ev_art.get("trusted", 0.0))
        unknown_mentions = float(ev_art.get("unknown", 0.0))

        low_cred_ratio = low_cred_mentions / total_mentions if total_mentions > 0 else np.nan
        trusted_ratio = trusted_mentions / total_mentions if total_mentions > 0 else np.nan

        tones = ev_art.get("tones", [])
        mean_tone = float(np.mean(tones)) if len(tones) > 0 else np.nan
        extreme_tone = abs(mean_tone) if pd.notna(mean_tone) else np.nan

        source_hhi = hhi_map.get(eid, np.nan)

        peak_abnormal = row["peak_abnormal_mcap_usd"] if "peak_abnormal_mcap_usd" in row.index else np.nan
        event_abs_abnormal = row["event_window_sum_abs_abnormal_mcap_usd"] if "event_window_sum_abs_abnormal_mcap_usd" in row.index else np.nan
        car7_abs_abnormal = row["car_7d_sum_abs_abnormal_mcap_usd"] if "car_7d_sum_abs_abnormal_mcap_usd" in row.index else np.nan

        media_event_oof = row["event_window_sum_abs_media_effect_usd_oof"] if "event_window_sum_abs_media_effect_usd_oof" in row.index else np.nan
        media_share_event = row["media_share_of_event_window_pct_oof"] if "media_share_of_event_window_pct_oof" in row.index else np.nan
        media_share_car7 = row["media_share_of_car7_pct_oof"] if "media_share_of_car7_pct_oof" in row.index else np.nan

        reversal = np.nan
        if pd.notna(peak_date) and return_col is not None:
            reversal = compute_shock_reversal_from_daily(
                df_daily=daily,
                event_date=peak_date,
                return_col=return_col,
                fallback_col=fallback_col,
            )

        permanent_effect_share = np.nan
        temporary_effect_share = np.nan
        if pd.notna(reversal):
            # cap to [0,1+] interpretation
            temporary_effect_share = min(max(reversal, 0.0), 1.0) if reversal <= 1 else 1.0
            permanent_effect_share = max(0.0, 1.0 - temporary_effect_share)

        rows.append({
            "event_id": eid,
            "preset": row["preset"] if "preset" in row.index else "balanced",
            "peak_date": str(peak_date.date()) if pd.notna(peak_date) else None,

            # Core event metrics
            "peak_abnormal_mcap_usd": peak_abnormal,
            "event_window_sum_abs_abnormal_mcap_usd": event_abs_abnormal,
            "car_7d_sum_abs_abnormal_mcap_usd": car7_abs_abnormal,

            "event_window_sum_abs_media_effect_usd_oof": media_event_oof,
            "media_share_of_event_window_pct_oof": media_share_event,
            "media_share_of_car7_pct_oof": media_share_car7,

            # Article / credibility metrics
            "article_total_mentions": total_mentions,
            "article_low_cred_mentions": low_cred_mentions,
            "article_trusted_mentions": trusted_mentions,
            "article_unknown_mentions": unknown_mentions,
            "article_low_cred_ratio": low_cred_ratio,
            "article_trusted_ratio": trusted_ratio,
            "article_mean_tone": mean_tone,
            "article_extreme_tone_abs": extreme_tone,
            "source_hhi": source_hhi,

            # Reversal / stability
            "shock_reversal_index": reversal,
            "temporary_effect_share": temporary_effect_share,
            "permanent_effect_share": permanent_effect_share,
        })

    adv = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Compute BMPI composite per event
    # -----------------------------------------------------------------------
    component_df = adv.copy()

    component_df["_s1_news_spike"] = _safe_num(component_df["article_total_mentions"]).fillna(0.0)
    component_df["_s2_extreme_tone"] = _safe_num(component_df["article_extreme_tone_abs"]).fillna(0.0)
    component_df["_s3_hhi"] = _safe_num(component_df["source_hhi"]).fillna(0.0)
    component_df["_s4_peak_abnormal"] = _safe_num(component_df["peak_abnormal_mcap_usd"]).abs().fillna(0.0)
    component_df["_s5_reversal"] = _safe_num(component_df["shock_reversal_index"]).fillna(0.0)

    component_df["_z1"] = z_norm(component_df["_s1_news_spike"]) * BMPI_W1_NEWS_SPIKE
    component_df["_z2"] = z_norm(component_df["_s2_extreme_tone"]) * BMPI_W2_EXTREME_TONE
    component_df["_z3"] = z_norm(component_df["_s3_hhi"]) * BMPI_W3_SOURCE_CONC
    component_df["_z4"] = z_norm(component_df["_s4_peak_abnormal"]) * BMPI_W4_PEAK_ABNORMAL
    component_df["_z5"] = z_norm(component_df["_s5_reversal"]) * BMPI_W5_REVERSAL

    component_df["bmpi_raw_event"] = component_df[["_z1", "_z2", "_z3", "_z4", "_z5"]].sum(axis=1)
    component_df["bmpi_event"] = sigmoid(component_df["bmpi_raw_event"].to_numpy())

    def bmpi_label(v: float) -> str:
        if pd.isna(v):
            return "—"
        if v >= 0.70:
            return "HIGH"
        if v >= 0.55:
            return "MODERATE"
        if v >= 0.45:
            return "LOW-MODERATE"
        return "LOW"

    component_df["bmpi_event_label"] = component_df["bmpi_event"].apply(bmpi_label)

    # Credibility-weighted impact
    component_df["credible_impact_usd"] = (
        _safe_num(component_df["event_window_sum_abs_media_effect_usd_oof"]).fillna(0.0) *
        _safe_num(component_df["article_trusted_ratio"]).fillna(0.20)
    )
    component_df["low_credibility_impact_usd"] = (
        _safe_num(component_df["event_window_sum_abs_media_effect_usd_oof"]).fillna(0.0) *
        _safe_num(component_df["article_low_cred_ratio"]).fillna(0.25)
    )
    component_df["credibility_ratio"] = (
        component_df["low_credibility_impact_usd"] /
        (component_df["low_credibility_impact_usd"] + component_df["credible_impact_usd"] + 1e-9)
    )

    # Save per-event CSV
    out_cols = [
        "event_id", "preset", "peak_date",
        "peak_abnormal_mcap_usd",
        "event_window_sum_abs_abnormal_mcap_usd",
        "car_7d_sum_abs_abnormal_mcap_usd",
        "event_window_sum_abs_media_effect_usd_oof",
        "media_share_of_event_window_pct_oof",
        "media_share_of_car7_pct_oof",
        "article_total_mentions",
        "article_low_cred_ratio",
        "article_trusted_ratio",
        "article_mean_tone",
        "article_extreme_tone_abs",
        "source_hhi",
        "shock_reversal_index",
        "temporary_effect_share",
        "permanent_effect_share",
        "credible_impact_usd",
        "low_credibility_impact_usd",
        "credibility_ratio",
        "bmpi_raw_event",
        "bmpi_event",
        "bmpi_event_label",
    ]
    component_df[out_cols].to_csv(OUT_CSV, index=False)

    # -----------------------------------------------------------------------
    # Global summary
    # -----------------------------------------------------------------------
    global_summary = {
        "n_events": int(len(component_df)),
        "mean_media_share_event_window_pct_oof": float(_safe_num(component_df["media_share_of_event_window_pct_oof"]).mean()),
        "median_media_share_event_window_pct_oof": float(_safe_num(component_df["media_share_of_event_window_pct_oof"]).median()),
        "mean_media_share_car7_pct_oof": float(_safe_num(component_df["media_share_of_car7_pct_oof"]).mean()),
        "mean_bmpi_event": float(_safe_num(component_df["bmpi_event"]).mean()),
        "median_bmpi_event": float(_safe_num(component_df["bmpi_event"]).median()),
        "high_bmpi_events": int((_safe_num(component_df["bmpi_event"]) >= 0.70).sum()),
        "moderate_or_higher_bmpi_events": int((_safe_num(component_df["bmpi_event"]) >= 0.55).sum()),
        "mean_source_hhi": float(_safe_num(component_df["source_hhi"]).mean()),
        "mean_shock_reversal_index": float(_safe_num(component_df["shock_reversal_index"]).mean()),
        "sum_credible_impact_usd": float(_safe_num(component_df["credible_impact_usd"]).sum()),
        "sum_low_credibility_impact_usd": float(_safe_num(component_df["low_credibility_impact_usd"]).sum()),
        "mean_credibility_ratio": float(_safe_num(component_df["credibility_ratio"]).mean()),
        "top_5_events_by_bmpi": component_df.sort_values("bmpi_event", ascending=False)[["event_id", "peak_date", "bmpi_event"]].head(5).to_dict(orient="records"),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # Console report only
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("FILES")
    print("=" * 72)
    print(f"  ✓ advanced_metrics_per_event.csv : {len(component_df)} rows")
    print(f"  ✓ advanced_metrics_global.json")

    print("\n" + "=" * 72)
    print("GLOBAL SUMMARY")
    print("=" * 72)
    print(f"  Events analyzed:                  {global_summary['n_events']}")
    print(f"  Mean media share (event window):  {fmt_pct(global_summary['mean_media_share_event_window_pct_oof'])}")
    print(f"  Median media share (event window):{fmt_pct(global_summary['median_media_share_event_window_pct_oof'])}")
    print(f"  Mean media share (CAR7):          {fmt_pct(global_summary['mean_media_share_car7_pct_oof'])}")
    print(f"  Mean BMPI event score:            {global_summary['mean_bmpi_event']:.4f}")
    print(f"  Median BMPI event score:          {global_summary['median_bmpi_event']:.4f}")
    print(f"  High BMPI events (>=0.70):        {global_summary['high_bmpi_events']}")
    print(f"  Moderate+ BMPI events (>=0.55):   {global_summary['moderate_or_higher_bmpi_events']}")
    print(f"  Mean source HHI:                  {global_summary['mean_source_hhi']:.4f}")
    print(f"  Mean shock-reversal index:        {global_summary['mean_shock_reversal_index']:.4f}")
    print(f"  Sum credible impact:              {fmt_usd(global_summary['sum_credible_impact_usd'])}")
    print(f"  Sum low-cred impact:              {fmt_usd(global_summary['sum_low_credibility_impact_usd'])}")
    print(f"  Mean credibility ratio:           {global_summary['mean_credibility_ratio']:.4f}")

    print("\n" + "=" * 72)
    print("TOP-5 EVENTS BY BMPI")
    print("=" * 72)
    top5 = component_df.sort_values("bmpi_event", ascending=False)[
        ["event_id", "peak_date", "preset", "bmpi_event", "media_share_of_event_window_pct_oof", "credibility_ratio"]
    ].head(5)

    for _, r in top5.iterrows():
        print(
            f"  {r['event_id']:<18}  {str(r['peak_date']):<12}  "
            f"{str(r['preset']):<10}  BMPI={float(r['bmpi_event']):.4f}  "
            f"share={fmt_pct(r['media_share_of_event_window_pct_oof'])}  "
            f"cred_ratio={float(r['credibility_ratio']):.4f}"
        )

    print("\n" + "=" * 72)
    print("PRESET SUMMARY")
    print("=" * 72)
    if "preset" in component_df.columns:
        preset_summary = (
            component_df.groupby("preset", dropna=False)
            .agg(
                n_events=("event_id", "count"),
                mean_bmpi=("bmpi_event", "mean"),
                mean_media_share=("media_share_of_event_window_pct_oof", "mean"),
                mean_cred_ratio=("credibility_ratio", "mean"),
                mean_reversal=("shock_reversal_index", "mean"),
            )
            .reset_index()
        )

        for _, r in preset_summary.iterrows():
            print(
                f"  {str(r['preset']):<10}  "
                f"n={int(r['n_events']):<3}  "
                f"mean_bmpi={float(r['mean_bmpi']):.4f}  "
                f"mean_share={fmt_pct(r['mean_media_share'])}  "
                f"mean_cred_ratio={float(r['mean_cred_ratio']):.4f}  "
                f"mean_reversal={float(r['mean_reversal']):.4f}"
            )

    print("\nNext step: step12_cross_preset_analysis.py\n")


if __name__ == "__main__":
    main()