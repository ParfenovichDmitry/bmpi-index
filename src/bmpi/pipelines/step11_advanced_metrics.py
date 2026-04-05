# -*- coding: utf-8 -*-
"""
pipelines/step11_advanced_metrics.py
======================================
Advanced metrics for BMPI — 8 metric groups (A, B, C, E, F, G, H).

Input:
  data/processed/events_peaks_balanced.csv       (from step03)
  data/processed/excess_media_effect_daily.csv   (from step09)
  data/processed/news_effect_daily.csv           (from step07)
  data/processed/baseline_predictions.csv        (from step04)
  data/processed/features_daily.parquet          (from step02)

Output:
  data/processed/advanced_metrics_per_event.csv
  data/processed/advanced_metrics_global.json
  data/processed/advanced_metrics_report.html

Next step: step12_cross_preset_analysis.py
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_RAW_NEWS  = BASE_DIR / "data" / "raw" / "news_raw"

PEAKS_CSV         = DATA_PROCESSED / "events_peaks_balanced.csv"
EXCESS_MEDIA_CSV    = DATA_PROCESSED / "excess_media_effect_daily.csv"
NEWS_EFFECT_CSV   = DATA_PROCESSED / "news_effect_daily.csv"
BASELINE_CSV      = DATA_PROCESSED / "baseline_predictions.csv"
FEATURES_PARQUET  = DATA_PROCESSED / "features_daily.parquet"
MODEL_DATASET_PAR = DATA_PROCESSED / "model_dataset_daily.parquet"

JSONL_PATHS = [
    DATA_RAW_NEWS / "gdelt_events_peaks_articles" / "gdelt_events_peak_articles.jsonl",
    DATA_RAW_NEWS / "gdelt_events_peaks" / "gdelt_events_peak_articles.jsonl",
]

OUT_CSV  = DATA_PROCESSED / "advanced_metrics_per_event.csv"
OUT_JSON = DATA_PROCESSED / "advanced_metrics_global.json"
OUT_HTML = DATA_PROCESSED / "advanced_metrics_report.html"

# BMPI weights (sum = 1.0)
BMPI_W1_NEWS_SPIKE    = 0.25   # news volume intensity
BMPI_W2_NEG_TONE      = 0.20   # extreme sentiment weight
BMPI_W3_SOURCE_CONC   = 0.20   # source concentration (HHI)
BMPI_W4_RESID_JUMP    = 0.20   # residual jump weight
BMPI_W5_REVERSAL      = 0.15   # post-shock reversal weight

# Windows for Shock-Reversal Index
SHOCK_WINDOW_BEFORE = 3   # days t0..t+2 (wzrost)
SHOCK_WINDOW_AFTER  = 5   # days t+3..t+7 (reversal)

# Domain credibility classification
LOW_CRED = {
    "news.bitcoin.com","newsbtc.com","cryptocoinsnews.com","ccn.com",
    "altcointoday.com","thebitcoinnews.com","coinreport.net","bravenewcoin.com",
    "coinspeaker.com","4-traders.com","econotimes.com","zerohedge.com",
    "cryptonewsz.com","insidebitcoins.com","nulltx.com","zycrypto.com",
    "ambcrypto.com","coingape.com","cryptodaily.co.uk","u.today",
    "coinpedia.org","cryptopotato.com","beincrypto.com",
    "ibtimes.com","ibtimes.co.uk","beforeitsnews.com",
}
TRUSTED = {
    "reuters.com","bloomberg.com","ft.com","wsj.com","cnbc.com","forbes.com",
    "fortune.com","businessinsider.com","marketwatch.com","finance.yahoo.com",
    "nasdaq.com","bbc.com","bbc.co.uk","coindesk.com","cointelegraph.com",
    "bitcoinmagazine.com","theblock.co","decrypt.co","apnews.com","theguardian.com",
    "bis.org","ecb.europa.eu","sec.gov","cftc.gov",
}

def classify_domain(d: str) -> str:
    d = str(d or "").lower().replace("www.","").strip()
    for ld in LOW_CRED:
        if d == ld or d.endswith("."+ld): return "low_cred"
    for td in TRUSTED:
        if d == td or d.endswith("."+td): return "trusted"
    return "unknown"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def z_norm(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd < 1e-9: return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-3, 3)

def fmt_usd(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    x = float(x)
    if abs(x) >= 1e12: return f"{x/1e12:.4f} T USD"
    if abs(x) >= 1e9:  return f"{x/1e9:.3f} B USD"
    if abs(x) >= 1e6:  return f"{x/1e6:.1f} M USD"
    return f"{x:,.0f} USD"

def fmt_pct(x, decimals=4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{float(x):.{decimals}f}%"


# ─────────────────────────────────────────────────────────────────────────────
# LOADING DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_or_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists(): return None
    try:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if c in ["data","date","day"]), None)
        if date_col:
            df["data"] = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")
        return df
    except Exception as e:
        print(f"  [WARN] {path.name}: {e}")
        return None

def load_all_data():
    print("═"*65)
    print("LOADING DATA")
    print("═"*65)

    data = {}

    # Peaks
    peaks = load_csv_or_parquet(PEAKS_CSV)
    if peaks is None:
        raise FileNotFoundError(f"File not found: {PEAKS_CSV}")
    peaks["data_piku"]   = pd.to_datetime(peaks["data_piku"], errors="coerce").dt.floor("D")
    peaks["okno_start"]  = pd.to_datetime(peaks["okno_start"],  errors="coerce").dt.floor("D")
    peaks["okno_koniec"] = pd.to_datetime(peaks["okno_koniec"], errors="coerce").dt.floor("D")
    data["peaks"] = peaks
    print(f"  [OK] peaks:          {len(peaks)} events")

    # Fake daily
    fake = load_csv_or_parquet(EXCESS_MEDIA_CSV)
    if fake is not None:
        for c in ["excess_media_effect_usd","news_effect_usd","resid_btc_mcap_usd","bmpi_score"]:
            if c in fake.columns:
                fake[c] = pd.to_numeric(fake[c], errors="coerce").fillna(0.0)
        data["fake"] = fake
        print(f"  [OK] excess_daily:     {len(fake)} days, "
              f"excess_usd sum={fmt_usd(fake['excess_media_effect_usd'].abs().sum())}")
    else:
        print("  [WARN] excess_media_effect_daily.csv not found — Groups A/B/C limited")

    # News effect daily
    news_eff = load_csv_or_parquet(NEWS_EFFECT_CSV)
    if news_eff is not None:
        data["news_eff"] = news_eff
        print(f"  [OK] news_effect:    {len(news_eff)} days")
    else:
        print("  [WARN] news_effect_daily.csv not found")

    # Baseline predictions
    baseline = load_csv_or_parquet(BASELINE_CSV)
    if baseline is None:
        baseline = load_csv_or_parquet(FEATURES_PARQUET)
    if baseline is None:
        baseline = load_csv_or_parquet(MODEL_DATASET_PAR)
    if baseline is not None:
        data["baseline"] = baseline
        mcap_col = next((c for c in baseline.columns
                         if "mcap" in c or "kapitaliz" in c or "market_cap" in c), None)
        btc_col  = next((c for c in baseline.columns
                         if ("btc" in c or "bitcoin" in c) and
                         any(kw in c for kw in ["price","cena","close","zamk"])), None)
        # Also check for pre-computed BTC log-returns
        logret_col = next((c for c in baseline.columns
                           if "logret" in c or "log_ret" in c or
                           ("btc" in c and "ret" in c)), None)
        print(f"  [OK] baseline/feat:  {len(baseline)} days "
              f"(mcap_col={mcap_col}, price_col={btc_col}, logret_col={logret_col})")
        data["mcap_col"]     = mcap_col
        data["btc_price_col"]= btc_col
        data["btc_logret_col"]= logret_col
    else:
        print("  [WARN] features_daily / baseline_predictions not found — reversal from peaks")

    # JSONL
    jsonl_path = next((p for p in JSONL_PATHS if p.exists()), None)
    if jsonl_path:
        print(f"  [OK] JSONL:          {jsonl_path.name}", end="")
        articles = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: articles.append(json.loads(line))
                except: continue
        data["articles"] = articles
        print(f" → {len(articles)} records")
    else:
        print("  [WARN] JSONL not found — Groups E/H use fallback estimates")
        data["articles"] = []

    print()
    return data


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DAILY DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_df(data: dict) -> pd.DataFrame:
    """
    Merge all sources into one daily DataFrame:
    date | btc_mcap | btc_price | baseline_mcap |
    news_effect_usd | excess_usd | resid | bmpi_score
    """
    # Base: excess_daily or news_effect
    _excess    = data.get("fake")
    _news    = data.get("news_eff")
    base     = _excess if _excess is not None else _news
    if base is None:
        # Fallback: expand peaks to daily index
        peaks = data["peaks"]
        date_range = pd.date_range(peaks["data_piku"].min(),
                                   peaks["data_piku"].max(), freq="D")
        base = pd.DataFrame({"data": date_range})

    df = base[["data"]].copy()

    def safe_merge(right, cols):
        if right is None: return
        # Keep only needed columns not already in df (avoid _x/_y duplicates)
        new_cols = [c for c in cols if c in right.columns and c not in df.columns]
        if not new_cols: return
        nonlocal df
        df = df.merge(right[["data"] + new_cols], on="data", how="left")

    safe_merge(data.get("fake"), ["excess_media_effect_usd","news_effect_usd",
                                   "resid_btc_mcap_usd","bmpi_score",
                                   "s1_low_cred_ratio","gdelt_ton_all","gdelt_wzmianki_all"])
    # merge news_eff only for missing columns
    safe_merge(data.get("news_eff"), ["news_effect_usd","resid_btc_mcap_usd"])

    # btc_mcap, btc_price, btc_logret z features_daily/baseline
    bl = data.get("baseline")
    if bl is not None:
        mcap_col    = data.get("mcap_col")
        price_col   = data.get("btc_price_col")
        logret_col  = data.get("btc_logret_col")
        bl_merge    = bl[["data"]].copy()
        if mcap_col and mcap_col in bl.columns:
            bl_merge["btc_mcap"] = pd.to_numeric(bl[mcap_col], errors="coerce")
        if price_col and price_col in bl.columns:
            bl_merge["btc_price"] = pd.to_numeric(bl[price_col], errors="coerce")
        # Use log-returns if available — more precise
        if logret_col and logret_col in bl.columns:
            bl_merge["btc_logret"] = pd.to_numeric(bl[logret_col], errors="coerce")
        baseline_col = next((c for c in bl.columns
                              if "baseline" in c and "pred" in c), None) or \
                       next((c for c in bl.columns if "pred" in c), None)
        if baseline_col:
            bl_merge["baseline_mcap"] = pd.to_numeric(bl[baseline_col], errors="coerce")
        df = df.merge(bl_merge, on="data", how="left")

    # Ensure all expected columns exist
    for c in ["excess_media_effect_usd","news_effect_usd","resid_btc_mcap_usd",
              "bmpi_score","btc_mcap","btc_price","btc_logret","baseline_mcap",
              "gdelt_ton_all","gdelt_wzmianki_all","s1_low_cred_ratio"]:
        if c not in df.columns:
            df[c] = np.nan

    df["excess_media_effect_usd"]  = df["excess_media_effect_usd"].abs()
    df["total_change_usd"] = df["resid_btc_mcap_usd"].abs()

    # Daily BTC return — prefer log-returns
    if df["btc_logret"].notna().sum() > 10:
        df["btc_return"] = df["btc_logret"]
        print(f"  [INFO] btc_return: using btc_logret "
              f"({df['btc_logret'].notna().sum()} days)")
    elif df["btc_price"].notna().sum() > 10:
        df["btc_return"] = df["btc_price"].pct_change()
        print(f"  [INFO] btc_return: pct_change from btc_price")
    elif df["btc_mcap"].notna().sum() > 10:
        df["btc_return"] = df["btc_mcap"].pct_change()
        print(f"  [INFO] btc_return: pct_change from btc_mcap")
    else:
        df["btc_return"] = np.nan
        print(f"  [WARN] btc_return: no data — SRI/reversal unavailable")

    return df.sort_values("data").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD ARTICLE STATS PER EVENT
# ─────────────────────────────────────────────────────────────────────────────

def build_article_stats(articles: list, peaks: pd.DataFrame) -> Dict[str, dict]:
    """
    Per event_id builds:
      total_mentions, low_cred_n, trusted_n, unknown_n,
      source_counts (dict domena→n), tones (lista), goldsteins (lista)
    """
    by_event = defaultdict(lambda: {
        "total": 0.0, "low_cred": 0.0, "trusted": 0.0, "unknown": 0.0,
        "source_counts": defaultdict(float),
        "tones": [], "goldsteins": [],
    })

    for a in articles:
        eid = str(a.get("event_id","")).strip()
        if not eid: continue
        dom = str(a.get("source_domain","")).lower().replace("www.","").strip()
        na  = float(a.get("num_articles") or a.get("num_mentions") or 1)
        tone = a.get("avg_tone")
        gold = a.get("goldstein")
        cred = classify_domain(dom)

        ev = by_event[eid]
        ev["total"]          += na
        ev[cred]             += na
        ev["source_counts"][dom] += na
        if tone is not None: ev["tones"].append(float(tone))
        if gold is not None: ev["goldsteins"].append(float(gold))

    return dict(by_event)


# ─────────────────────────────────────────────────────────────────────────────
# GRUPA A: DECOMPOSITION KAPITALIZACJI
# ─────────────────────────────────────────────────────────────────────────────

def compute_decomposition(df: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    """
    Per event computes:
      total_change_usd  = Σ|resid| in window [-3,+2]
      excess_usd          = Σexcess_media_effect_usd in event window
      info_usd          = Σnews_effect_usd in event window (news effect without excess)
      baseline_usd      = total_change - info_usd (approximation when baseline_mcap unavailable)

      fundamental_share = baseline_usd / total_change_usd
      info_share        = info_usd     / total_change_usd
      anomaly_share     = excess_usd     / total_change_usd
    """
    rows = []
    for _, pk in peaks.iterrows():
        eid  = pk["event_id"]
        pik  = pd.Timestamp(pk["data_piku"])
        ns   = pik - pd.Timedelta(days=3)
        ne   = pik + pd.Timedelta(days=2)

        mask = (df["data"] >= ns) & (df["data"] <= ne)
        w = df[mask]

        total  = float(w["total_change_usd"].sum())
        fake   = float(w["excess_media_effect_usd"].sum())
        news   = float(w["news_effect_usd"].abs().sum())

        # baseline_usd = part of change not explained by media
        if "baseline_mcap" in df.columns and w["baseline_mcap"].notna().any():
            baseline = float((w["btc_mcap"] - w["baseline_mcap"]).abs().sum())
        else:
            baseline = max(0.0, total - news)

        info_only = max(0.0, news - fake)  # pure informational effect (without anomalous/excess component)

        rows.append({
            "event_id":         eid,
            "data_piku":        str(pik.date()),
            "btc_cena_usd":     float(pk.get("btc_cena_usd", np.nan)),
            "btc_mcap_piku":    float(pk.get("btc_kapitalizacja_usd", np.nan)),
            "zscore":           float(pk.get("zscore_anomalii", np.nan)),
            # USD components
            "total_change_usd": total,
            "excess_usd":         fake,
            "info_usd":         info_only,
            "baseline_usd":     baseline,
            # Decomposition shares (Group A)
            "anomaly_share":    fake     / total if total > 0 else np.nan,
            "info_share":       info_only / total if total > 0 else np.nan,
            "fundamental_share":baseline / total if total > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# GRUPA E: SOURCE CONCENTRATION INDEX (HHI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hhi(art_stats: Dict[str, dict]) -> Dict[str, float]:
    """
    Herfindahl-Hirschman Index per event:
      HHI = Σ(share_i²) gdzie share_i = n_i / total_n

    HHI = 1.0 → one dominant source (max concentration — suspicious)
    HHI → 0   → evenly distributed sources (organic)

    High HHI = effect driven by 1-2 sources = suspicious
    """
    result = {}
    for eid, ev in art_stats.items():
        counts = ev["source_counts"]
        total  = ev["total"]
        if total <= 0:
            result[eid] = np.nan
            continue
        shares = np.array([n / total for n in counts.values()])
        result[eid] = float(np.sum(shares ** 2))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GRUPA F: SHOCK-REVERSAL INDEX
# ─────────────────────────────────────────────────────────────────────────────

def compute_shock_reversal(df: pd.DataFrame, peaks: pd.DataFrame) -> Dict[str, float]:
    """
    shock_reversal_index = |return[t+3..t+7]| / |return[t0..t+2]|

    > 1.0 → reversal stronger than rise = pump & dump
    ~ 1.0 → symmetric
    < 1.0 → rise sustained (organic)
    """
    result = {}
    for _, pk in peaks.iterrows():
        eid = pk["event_id"]
        pik = pd.Timestamp(pk["data_piku"])

        # Rise window: t0 to t+2
        mask_up  = (df["data"] >= pik) & (df["data"] <= pik + pd.Timedelta(SHOCK_WINDOW_BEFORE,"D"))
        # Reversal window: t+3 to t+7
        mask_dn  = (df["data"] > pik + pd.Timedelta(SHOCK_WINDOW_BEFORE,"D")) & \
                   (df["data"] <= pik + pd.Timedelta(SHOCK_WINDOW_BEFORE+SHOCK_WINDOW_AFTER,"D"))

        ret_up = df[mask_up]["btc_return"].dropna()
        ret_dn = df[mask_dn]["btc_return"].dropna()

        if len(ret_up) == 0 or len(ret_dn) == 0:
            # Fallback: z mcap
            mask_up2  = (df["data"] >= pik) & \
                        (df["data"] <= pik + pd.Timedelta(SHOCK_WINDOW_BEFORE,"D"))
            mask_dn2  = (df["data"] > pik + pd.Timedelta(SHOCK_WINDOW_BEFORE,"D")) & \
                        (df["data"] <= pik + pd.Timedelta(SHOCK_WINDOW_BEFORE+SHOCK_WINDOW_AFTER,"D"))
            # Fallback: use residuals when returns unavailable
            resid_up = df[mask_up2]["resid_btc_mcap_usd"].abs().mean()
            resid_dn = df[mask_dn2]["resid_btc_mcap_usd"].abs().mean()
            if pd.notna(resid_up) and resid_up > 0 and pd.notna(resid_dn):
                result[eid] = float(resid_dn / resid_up)
            else:
                result[eid] = np.nan
            continue

        up_mag = abs(ret_up.sum())
        dn_mag = abs(ret_dn.sum())
        result[eid] = float(dn_mag / up_mag) if up_mag > 0 else np.nan

    return result


# ─────────────────────────────────────────────────────────────────────────────
# GRUPA C: STABILITY EFEKTU (HALF-LIFE + REVERSAL RATIO)
# ─────────────────────────────────────────────────────────────────────────────

def compute_effect_stability(df: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    """
    C1. effect_half_life_days: by fitting exponential decay to excess_media_effect_usd
        after peak date (exponential decay fit)

    C2. reversal_ratio = sum of negative returns / peak_effect_usd
        (how much of the effect was reversed)

    C3. permanent_effect_share = 1 - reversal_ratio
    C4. temporary_effect_share = reversal_ratio
    """
    rows = []
    for _, pk in peaks.iterrows():
        eid = pk["event_id"]
        pik = pd.Timestamp(pk["data_piku"])

        # 14-day post-peak window
        mask_post = (df["data"] > pik) & (df["data"] <= pik + pd.Timedelta(14,"D"))
        w = df[mask_post].copy()
        w["days_after"] = (w["data"] - pik).dt.days

        half_life = np.nan
        reversal_ratio = np.nan
        permanent = np.nan
        temporary = np.nan

        # Half-life: exponential decay fit to excess_media_effect_usd
        if "excess_media_effect_usd" in w.columns:
            vals = w["excess_media_effect_usd"].fillna(0).values
            days = w["days_after"].values
            if len(vals) >= 4 and vals.max() > 0:
                try:
                    # y = a * exp(-k * t), solve for k
                    # linearisation: log(y) = log(a) - k*t
                    log_y = np.log(np.clip(vals, 1e-9, None))
                    slope, intercept, r, p, se = scipy_stats.linregress(days, log_y)
                    k = -slope
                    if k > 0:
                        half_life = float(np.log(2) / k)
                except Exception:
                    pass

        # Reversal ratio: negative return fraction in [t+3, t+7]
        mask_rev = (df["data"] > pik + pd.Timedelta(2,"D")) & \
                   (df["data"] <= pik + pd.Timedelta(7,"D"))
        rev_w = df[mask_rev]

        peak_excess = float(df[(df["data"] >= pik - pd.Timedelta(3,"D")) &
                             (df["data"] <= pik + pd.Timedelta(2,"D"))
                             ]["excess_media_effect_usd"].sum())

        if "btc_return" in rev_w.columns and rev_w["btc_return"].notna().any():
            neg_returns = rev_w["btc_return"].clip(upper=0).abs().sum()
            pos_returns = rev_w["btc_return"].clip(lower=0).sum()
            if (neg_returns + pos_returns) > 0:
                reversal_ratio = float(neg_returns / (neg_returns + pos_returns))
        elif peak_excess > 0:
            # Fallback: z resid
            resid_after = rev_w["resid_btc_mcap_usd"].fillna(0)
            reversed_usd = resid_after.clip(upper=0).abs().sum()
            reversal_ratio = float(min(reversed_usd / peak_excess, 1.0))

        if pd.notna(reversal_ratio):
            permanent = 1.0 - reversal_ratio
            temporary = reversal_ratio

        rows.append({
            "event_id":              eid,
            "effect_half_life_days": half_life,
            "reversal_ratio":        reversal_ratio,
            "permanent_effect_share":permanent,
            "temporary_effect_share":temporary,
            "peak_excess_usd":         peak_excess,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP G: MEDIA-TO-MARKET ELASTICITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_elasticity(df: pd.DataFrame, peaks: pd.DataFrame) -> Dict[str, float]:
    """
    media_market_elasticity = %Δmcap / %Δnews_volume

    Measures: %Δmcap when news_volume increases by 1%.
    High elasticity = market strongly reacts to media stimulation.
    """
    result = {}
    for _, pk in peaks.iterrows():
        eid = pk["event_id"]
        pik = pd.Timestamp(pk["data_piku"])

        # Window: 3 days pre-peak + 2 days post-peak
        mask_pre  = (df["data"] >= pik - pd.Timedelta(3,"D")) & (df["data"] < pik)
        mask_peak = (df["data"] >= pik) & (df["data"] <= pik + pd.Timedelta(2,"D"))

        pre  = df[mask_pre]
        peak = df[mask_peak]

        if "gdelt_wzmianki_all" not in df.columns or "btc_mcap" not in df.columns:
            result[eid] = np.nan
            continue

        vol_pre  = pre["gdelt_wzmianki_all"].mean()
        vol_peak = peak["gdelt_wzmianki_all"].mean()
        mcap_pre  = pre["btc_mcap"].mean()
        mcap_peak = peak["btc_mcap"].mean()

        if (pd.isna(vol_pre) or vol_pre <= 0 or
            pd.isna(mcap_pre) or mcap_pre <= 0 or
            pd.isna(vol_peak) or pd.isna(mcap_peak)):
            result[eid] = np.nan
            continue

        pct_news = (vol_peak - vol_pre) / vol_pre
        pct_mcap = (mcap_peak - mcap_pre) / mcap_pre

        result[eid] = float(pct_mcap / pct_news) if abs(pct_news) > 1e-9 else np.nan

    return result


# ─────────────────────────────────────────────────────────────────────────────
# GRUPA H: CREDIBILITY-WEIGHTED IMPACT
# ─────────────────────────────────────────────────────────────────────────────

def compute_credibility_impact(art_stats: Dict[str, dict],
                                decomp: pd.DataFrame) -> pd.DataFrame:
    """
    Splits the news effect (info_usd + excess_usd) into:
      credible_impact_usd        = news_effect × trusted_ratio
      low_credibility_impact_usd = news_effect × low_cred_ratio
      credibility_ratio          = low_cred / (low_cred + credible)

    Interpretation:
      credibility_ratio > 0.5 → effect mainly from low-credibility sources
      credibility_ratio < 0.3 → effect mainly from credible sources
    """
    rows = []
    for _, row in decomp.iterrows():
        eid       = row["event_id"]
        total_news= (row.get("excess_usd",0) or 0) + (row.get("info_usd",0) or 0)
        ev        = art_stats.get(eid, {})
        total_art = ev.get("total", 0)

        lc_ratio = ev.get("low_cred",0) / total_art if total_art > 0 else 0.25
        tr_ratio = ev.get("trusted", 0) / total_art if total_art > 0 else 0.20
        uk_ratio = 1.0 - lc_ratio - tr_ratio

        credible_usd  = total_news * tr_ratio
        low_cred_usd  = total_news * lc_ratio
        unknown_usd   = total_news * uk_ratio

        cred_ratio = (low_cred_usd / (low_cred_usd + credible_usd)
                      if (low_cred_usd + credible_usd) > 0 else np.nan)

        rows.append({
            "event_id":                  eid,
            "total_news_usd":            total_news,
            "credible_impact_usd":       credible_usd,
            "low_credibility_impact_usd":low_cred_usd,
            "unknown_impact_usd":        unknown_usd,
            "low_cred_article_ratio":    lc_ratio,
            "trusted_article_ratio":     tr_ratio,
            "credibility_ratio":         cred_ratio,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# GRUPA B: BMPI (BTC Information Manipulation Index)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bmpi(peaks: pd.DataFrame,
                 df: pd.DataFrame,
                 art_stats: Dict[str, dict],
                 hhi: Dict[str, float],
                 shock_rev: Dict[str, float]) -> pd.DataFrame:
    """
    BMPI = sigmoid( w1*z(news_spike) + w2*z(neg_tone) + w3*z(hhi)
                   + w4*z(resid_jump) + w5*z(reversal) )

    Components:
      news_spike:   avg mention count in event window (from GDELT)
      neg_tone:     |avg_tone| gdy ujemny (FUD/panika) lub
                    both hype and FUD are anomalous — use absolute value
      hhi:          source concentration (Herfindahl)
      resid_jump:   zscore_anomalii z peaks
      reversal:     shock_reversal_index

    BMPI ∈ (0, 1):
      > 0.7 → high media manipulation pressure
      0.5-0.7 → umiarkowana
      < 0.5 → niska
    """
    bmpi_rows = []
    for _, pk in peaks.iterrows():
        eid = pk["event_id"]
        pik = pd.Timestamp(pk["data_piku"])
        mask = (df["data"] >= pik - pd.Timedelta(3,"D")) & \
               (df["data"] <= pik + pd.Timedelta(2,"D"))
        w = df[mask]

        # S1: news_spike = avg GDELT mention count in event window
        news_spike = float(w["gdelt_wzmianki_all"].mean()) \
                     if "gdelt_wzmianki_all" in df.columns else \
                     float(art_stats.get(eid, {}).get("total", 0))

        # S2: extreme_tone = |avg_tone| — both FUD and hype are anomalous
        ev_tones = art_stats.get(eid, {}).get("tones", [])
        if ev_tones:
            mean_tone = np.mean(ev_tones)
            # Both hype and FUD are anomalous — use |tone|
            extreme_tone = abs(mean_tone)
        else:
            extreme_tone = abs(float(w["gdelt_ton_all"].mean())) \
                           if "gdelt_ton_all" in df.columns else 0.0

        # S3: HHI
        hhi_val = hhi.get(eid, 0.0) or 0.0

        # S4: resid_jump = zscore anomalii
        resid_jump = float(pk.get("zscore_anomalii", 0) or 0)

        # S5: reversal
        reversal = float(shock_rev.get(eid, 0) or 0)

        bmpi_rows.append({
            "event_id":    eid,
            "data_piku":   str(pik.date()),
            "_s1_news_spike":   news_spike,
            "_s2_extreme_tone": extreme_tone,
            "_s3_hhi":          hhi_val,
            "_s4_resid_jump":   resid_jump,
            "_s5_reversal":     reversal,
        })

    bmpi_df = pd.DataFrame(bmpi_rows)

    # Z-normalise and weight each component
    for col, w in [
        ("_s1_news_spike",   BMPI_W1_NEWS_SPIKE),
        ("_s2_extreme_tone", BMPI_W2_NEG_TONE),
        ("_s3_hhi",          BMPI_W3_SOURCE_CONC),
        ("_s4_resid_jump",   BMPI_W4_RESID_JUMP),
        ("_s5_reversal",     BMPI_W5_REVERSAL),
    ]:
        bmpi_df[col + "_z"] = z_norm(bmpi_df[col].fillna(0)) * w

    z_cols = [c for c in bmpi_df.columns if c.endswith("_z")]
    bmpi_df["bmpi_raw"] = bmpi_df[z_cols].sum(axis=1)
    bmpi_df["bmpi"]     = sigmoid(bmpi_df["bmpi_raw"].values)

    # Interpretacja
    def bmpi_label(b):
        if pd.isna(b): return "—"
        if b >= 0.70: return "🔴 HIGH manipulation"
        if b >= 0.55: return "🟠 MODERATE"
        if b >= 0.45: return "🟡 LOW-MODERATE"
        return "🟢 LOW"

    bmpi_df["bmpi_label"] = bmpi_df["bmpi"].apply(bmpi_label)

    return bmpi_df[["event_id","data_piku","bmpi","bmpi_label",
                    "_s1_news_spike","_s2_extreme_tone","_s3_hhi",
                    "_s4_resid_jump","_s5_reversal"]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_console_report(final_df: pd.DataFrame, global_stats: dict):
    W = 72

    def box(title):
        print("\n╔" + "═"*(W-2) + "╗")
        print("║  " + title.ljust(W-4) + "║")
        print("╚" + "═"*(W-2) + "╝")

    def row(label, value, note=""):
        label_str = str(label)
        value_str = str(value)
        note_str  = f"  ← {note}" if note else ""
        line = f"  {label_str:<38} {value_str}{note_str}"
        print(line)

    def sep():
        print("  " + "─"*(W-4))

    # ═══════════════════════════════════════════════════════
    box("GROUP A — BTC MARKET CAP DECOMPOSITION")
    print(f"  {'event_id':<22} {'total_usd':>14} {'fund%':>7} {'info%':>7} {'anomaly%':>9}")
    print("  " + "─"*62)
    for _, r in final_df.sort_values("data_piku").iterrows():
        fs  = r.get("fundamental_share")
        is_ = r.get("info_share")
        ans = r.get("anomaly_share")
        print(f"  {r['event_id']:<22} "
              f"{fmt_usd(r.get('total_change_usd',np.nan)):>14}  "
              f"{fmt_pct(fs*100 if pd.notna(fs) else None,1):>7}  "
              f"{fmt_pct(is_*100 if pd.notna(is_) else None,1):>7}  "
              f"{fmt_pct(ans*100 if pd.notna(ans) else None,1):>8}")
    sep()
    gs_a = global_stats.get("group_a",{})
    row("GLOBAL fundamental_share:",  fmt_pct(gs_a.get("mean_fundamental",np.nan)*100), "avg per event")
    row("GLOBAL info_share:",         fmt_pct(gs_a.get("mean_info",np.nan)*100))
    row("GLOBAL anomaly_share:",      fmt_pct(gs_a.get("mean_anomaly",np.nan)*100))

    # ═══════════════════════════════════════════════════════
    box("GROUP B — BMPI (Bitcoin Media Pressure Index)")
    print(f"  {'event_id':<22} {'bmpi':>6}  label")
    print("  " + "─"*62)
    for _, r in final_df.sort_values("bmpi", ascending=False).iterrows():
        b = r.get("bmpi", np.nan)
        lbl = r.get("bmpi_label","—")
        print(f"  {r['event_id']:<22} {b:.4f}  {lbl}")
    sep()
    gs_b = global_stats.get("group_b",{})
    row("BMPI mean:",   f"{gs_b.get('mean',np.nan):.4f}")
    row("BMPI median:",  f"{gs_b.get('median',np.nan):.4f}")
    row("BMPI max:",      f"{gs_b.get('max',np.nan):.4f}")
    row("Events BMPI>0.70:", str(gs_b.get("n_high",0)))

    # ═══════════════════════════════════════════════════════
    box("GROUP C — EFFECT STABILITY (HALF-LIFE + REVERSAL)")
    print(f"  {'event_id':<22} {'half_life':>10} {'reversal_r':>11} {'perm%':>7} {'temp%':>7}")
    print("  " + "─"*62)
    for _, r in final_df.sort_values("data_piku").iterrows():
        hl  = r.get("effect_half_life_days", np.nan)
        rr  = r.get("reversal_ratio",        np.nan)
        pe  = r.get("permanent_effect_share",np.nan)
        te  = r.get("temporary_effect_share",np.nan)
        hl_s = f"{hl:.1f} days" if pd.notna(hl) else "—"
        print(f"  {r['event_id']:<22} {hl_s:>10}  "
              f"{fmt_pct(rr*100 if pd.notna(rr) else None,1):>10}  "
              f"{fmt_pct(pe*100 if pd.notna(pe) else None,1):>7}  "
              f"{fmt_pct(te*100 if pd.notna(te) else None,1):>7}")
    sep()
    gs_c = global_stats.get("group_c",{})
    row("Mean half-life:",         f"{gs_c.get('mean_half_life',np.nan):.1f} days")
    row("Mean reversal_ratio:",    fmt_pct(gs_c.get("mean_reversal",np.nan)*100,1))
    row("Mean permanent_share:",   fmt_pct(gs_c.get("mean_permanent",np.nan)*100,1))

    # ═══════════════════════════════════════════════════════
    box("GROUP E — SOURCE CONCENTRATION INDEX (HHI)")
    print(f"  {'event_id':<22} {'HHI':>8}  interpretation")
    print("  " + "─"*62)
    for _, r in final_df.sort_values("source_hhi", ascending=False).iterrows():
        hhi_v = r.get("source_hhi", np.nan)
        if pd.isna(hhi_v):
            interp = "—"
        elif hhi_v >= 0.25:
            interp = "⚠ HIGH concentration (oligopol)"
        elif hhi_v >= 0.10:
            interp = "○ moderate"
        else:
            interp = "✓ low (diversified)"
        print(f"  {r['event_id']:<22} {hhi_v:.4f}   {interp}" if pd.notna(hhi_v) else
              f"  {r['event_id']:<22}     —    no data")
    sep()
    gs_e = global_stats.get("group_e",{})
    row("Mean HHI:", f"{gs_e.get('avg',np.nan):.4f}")
    row("Max HHI:", f"{gs_e.get('max',np.nan):.4f}")
    row("Events HHI>0.25:", str(gs_e.get("n_high_concentration",0)))

    # ═══════════════════════════════════════════════════════
    box("GROUP F — SHOCK-REVERSAL INDEX (SRI)")
    print(f"  {'event_id':<22} {'SRI':>8}  interpretation")
    print("  " + "─"*62)
    for _, r in final_df.sort_values("shock_reversal_index", ascending=False).iterrows():
        sri = r.get("shock_reversal_index", np.nan)
        if pd.isna(sri):
            interp = "—"
        elif sri >= 1.5:
            interp = "🔴 PUMP & DUMP (reversal >> rise)"
        elif sri >= 1.0:
            interp = "🟠 reversal ≈ rise"
        else:
            interp = "🟢 rise sustained"
        print(f"  {r['event_id']:<22} {sri:.4f}   {interp}" if pd.notna(sri) else
              f"  {r['event_id']:<22}     —    no data")
    sep()
    gs_f = global_stats.get("group_f",{})
    row("Mean SRI:", f"{gs_f.get('avg',np.nan):.4f}", "1.0 = symmetric rise/reversal")
    row("Events SRI>1.5 (pump&dump):", str(gs_f.get("n_pump_dump",0)))

    # ═══════════════════════════════════════════════════════
    box("GROUP G — MEDIA-TO-MARKET ELASTICITY")
    print(f"  {'event_id':<22} {'elasticity':>12}  interpretation")
    print("  " + "─"*62)
    for _, r in final_df.sort_values("media_market_elasticity",
                                     key=lambda s: s.abs(), ascending=False).iterrows():
        el = r.get("media_market_elasticity", np.nan)
        if pd.isna(el):
            interp = "no data"
        elif el > 2.0:
            interp = "⚠ very high market reactivity"
        elif el > 1.0:
            interp = "○ elevated"
        elif el > 0:
            interp = "✓ normal"
        else:
            interp = "↓ market did not react to news"
        print(f"  {r['event_id']:<22} {el:>12.4f}  {interp}" if pd.notna(el) else
              f"  {r['event_id']:<22}           —   {interp}")
    sep()
    gs_g = global_stats.get("group_g",{})
    row("Mean elasticity:", f"{gs_g.get('avg',np.nan):.4f}",
        "%Δmcap / %Δnews_volume")

    # ═══════════════════════════════════════════════════════
    box("GROUP H — CREDIBILITY-WEIGHTED IMPACT")
    print(f"  {'event_id':<22} {'low_cred%art':>13} {'low_cred_usd':>16} {'cred_ratio':>11}")
    print("  " + "─"*62)
    for _, r in final_df.sort_values("credibility_ratio", ascending=False).iterrows():
        lc_r  = r.get("low_cred_article_ratio", np.nan)
        lc_u  = r.get("low_credibility_impact_usd", np.nan)
        cr    = r.get("credibility_ratio", np.nan)
        print(f"  {r['event_id']:<22} "
              f"{fmt_pct(lc_r*100 if pd.notna(lc_r) else None,1):>13}  "
              f"{fmt_usd(lc_u):>16}  "
              f"{fmt_pct(cr*100 if pd.notna(cr) else None,1):>10}")
    sep()
    gs_h = global_stats.get("group_h",{})
    row("Mean low_cred_article_ratio:", fmt_pct(gs_h.get("mean_lc_ratio",np.nan)*100,1))
    row("SUM credible_impact_usd:",    fmt_usd(gs_h.get("sum_credible",np.nan)))
    row("SUM low_cred_impact_usd:",    fmt_usd(gs_h.get("sum_low_cred",np.nan)))
    row("Global credibility_ratio:", fmt_pct(gs_h.get("global_ratio",np.nan)*100,1),
        "low_cred/(low_cred+credible)")

    # ═══════════════════════════════════════════════════════
    box("GLOBAL SUMMARY")
    row("Number of events:", str(len(final_df)))
    row("Period:", global_stats.get("period","?"))
    row("SUM excess_media_effect_usd:", fmt_usd(global_stats.get("sum_excess_usd",np.nan)))
    row("SUM total_change_usd:", fmt_usd(global_stats.get("sum_total_change",np.nan)))
    sep()
    row("→ Anomaly share (global):", fmt_pct(global_stats.get("global_anomaly_share",np.nan)*100,4),
        "excess_usd / total_change_usd")
    row("→ BMPI mean:", f"{global_stats.get('group_b',{}).get('mean',np.nan):.4f}")
    row("→ Pump&dump events (SRI>1.5):",
        str(global_stats.get("group_f",{}).get("n_pump_dump","?")))
    row("→ Mean permanent effect:",
        fmt_pct(global_stats.get("group_c",{}).get("mean_permanent",np.nan)*100,1))


# ─────────────────────────────────────────────────────────────────────────────
# HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_html_report(final_df: pd.DataFrame, global_stats: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows_html = []
    for _, r in final_df.sort_values("data_piku").iterrows():
        bmpi_v = r.get("bmpi", np.nan)
        bmpi_color = ("#f44336" if bmpi_v >= 0.70 else
                      "#ff9800" if bmpi_v >= 0.55 else
                      "#ffd740" if bmpi_v >= 0.45 else "#4caf50")
        rows_html.append(f"""
        <tr>
          <td style="font-family:monospace;font-size:11px">{r['event_id']}</td>
          <td>{r.get('data_piku','')}</td>
          <td style="text-align:right">{fmt_usd(r.get('total_change_usd',np.nan))}</td>
          <td style="text-align:right">{fmt_pct((r.get('anomaly_share',np.nan) or 0)*100,1)}</td>
          <td style="text-align:right">{fmt_pct((r.get('info_share',np.nan) or 0)*100,1)}</td>
          <td style="text-align:right">{fmt_pct((r.get('fundamental_share',np.nan) or 0)*100,1)}</td>
          <td style="text-align:center;color:{bmpi_color};font-weight:bold">
            {f"{bmpi_v:.4f}" if pd.notna(bmpi_v) else "—"}</td>
          <td>{r.get('bmpi_label','—')}</td>
          <td style="text-align:right">{f"{r.get('effect_half_life_days',np.nan):.1f} days" if pd.notna(r.get('effect_half_life_days')) else "—"}</td>
          <td style="text-align:right">{fmt_pct((r.get('reversal_ratio',np.nan) or np.nan)*100,1)}</td>
          <td style="text-align:right">{f"{r.get('source_hhi',np.nan):.4f}" if pd.notna(r.get('source_hhi')) else "—"}</td>
          <td style="text-align:right">{f"{r.get('shock_reversal_index',np.nan):.3f}" if pd.notna(r.get('shock_reversal_index')) else "—"}</td>
          <td style="text-align:right">{f"{r.get('media_market_elasticity',np.nan):.3f}" if pd.notna(r.get('media_market_elasticity')) else "—"}</td>
          <td style="text-align:right">{fmt_pct((r.get('credibility_ratio',np.nan) or np.nan)*100,1)}</td>
        </tr>""")

    gs_a = global_stats.get("group_a",{})
    gs_b = global_stats.get("group_b",{})
    gs_c = global_stats.get("group_c",{})
    gs_h = global_stats.get("group_h",{})
    gs_f = global_stats.get("group_f",{})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Advanced Metrics — BTC Manipulation Analysis</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  :root{{--bg:#0a0d14;--bg2:#111520;--bg3:#171c2b;--border:#232840;
    --acc:#f0a500;--acc2:#4fc3f7;--text:#e8eaf6;--dim:#455a8a;--muted:#7986cb}}
  body{{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);padding:0}}
  .hdr{{background:var(--bg2);border-bottom:1px solid var(--border);
    padding:24px 48px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
  .hdr h1{{font-family:'Playfair Display',serif;font-size:22px;color:var(--acc)}}
  .badge{{background:var(--bg3);border:1px solid var(--border);color:var(--acc2);
    font-family:'DM Mono',monospace;font-size:10px;padding:2px 10px;border-radius:20px}}
  .ts{{margin-left:auto;font-size:10px;color:var(--dim);font-family:'DM Mono',monospace}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
    border:1px solid var(--border);margin:28px 48px 0;border-radius:10px;overflow:hidden}}
  .kpi{{background:var(--bg2);padding:20px 18px}}
  .kpi-lbl{{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin-bottom:6px}}
  .kpi-val{{font-family:'DM Mono',monospace;font-size:20px;font-weight:500;color:var(--acc)}}
  .kpi-sub{{font-size:11px;color:var(--muted);margin-top:3px}}
  .section{{margin:28px 48px 0}}
  .section-title{{font-family:'Playfair Display',serif;font-size:16px;color:var(--acc);
    margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid var(--border)}}
  .tbl-wrap{{overflow-x:auto;border:1px solid var(--border);border-radius:8px}}
  table{{width:100%;border-collapse:collapse;font-size:11.5px}}
  th{{background:var(--bg3);padding:8px 10px;text-align:left;color:var(--dim);
    font-size:10px;text-transform:uppercase;letter-spacing:.7px;
    border-bottom:1px solid var(--border);white-space:nowrap}}
  td{{padding:7px 10px;border-bottom:1px solid var(--border)}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:var(--bg3)}}
  .method{{margin:16px 48px;background:var(--bg2);border:1px solid var(--border);
    border-left:4px solid var(--acc2);border-radius:6px;padding:14px 20px;font-size:12px;line-height:1.7}}
  .method code{{color:var(--acc2)}}
</style>
</head>
<body>
<header class="hdr">
  <h1>📊 Advanced BTC Manipulation Metrics</h1>
  <span class="badge">balanced preset</span>
  <span class="badge">v1.0</span>
  <span class="ts">Generated: {now}</span>
</header>

<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-lbl">Mean BMPI</div>
    <div class="kpi-val">{gs_b.get('mean',float('nan')):.4f}</div>
    <div class="kpi-sub">BTC Info Manipulation Index</div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">Mean anomaly_share</div>
    <div class="kpi-val">{fmt_pct((gs_a.get('mean_anomaly',float('nan')) or 0)*100,2)}</div>
    <div class="kpi-sub">excess_usd / total_change</div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">Pump&amp;Dump events (SRI&gt;1.5)</div>
    <div class="kpi-val">{gs_f.get('n_pump_dump','—')}</div>
    <div class="kpi-sub">fast reversal after rise</div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">Mean permanent effect</div>
    <div class="kpi-val">{fmt_pct((gs_c.get('mean_permanent',float('nan')) or 0)*100,1)}</div>
    <div class="kpi-sub">permanent_effect_share</div>
  </div>
</div>

<div class="method">
  <strong style="color:var(--acc2)">Methodology:</strong>
  <code>BMPI = sigmoid(w1·z(news_spike) + w2·z(tone) + w3·z(HHI) + w4·z(resid) + w5·z(reversal))</code>
  &nbsp;|&nbsp; window: [-3,+2] days from peak_date &nbsp;|&nbsp;
  <code>anomaly_share = excess_usd / total_change_usd</code> &nbsp;|&nbsp;
  <code>SRI = |ret[t+3..t+7]| / |ret[t0..t+2]|</code>
</div>

<div class="section">
  <div class="section-title">All metrics per event</div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>Event ID</th><th>Peak Date</th>
        <th>Total ΔUSD</th><th>anomaly%</th><th>info%</th><th>fund%</th>
        <th>BMPI</th><th>Label</th>
        <th>Half-life</th><th>Reversal%</th>
        <th>HHI</th><th>SRI</th><th>Elasticity</th><th>Cred.Ratio</th>
      </tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
  </div>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data = load_all_data()

    peaks    = data["peaks"]
    fake     = data.get("fake")
    articles = data.get("articles", [])

    daily_df   = build_daily_df(data)
    art_stats  = build_article_stats(articles, peaks)

    print("COMPUTING METRICS...")
    print("  [A] Market cap decomposition...")
    decomp   = compute_decomposition(daily_df, peaks)

    print("  [E] Source Concentration Index (HHI)...")
    hhi      = compute_hhi(art_stats)

    print("  [F] Shock-Reversal Index (SRI)...")
    shock_rev = compute_shock_reversal(daily_df, peaks)

    print("  [C] Effect stability (half-life, reversal ratio)...")
    stability = compute_effect_stability(daily_df, peaks)

    print("  [G] Media-to-Market Elasticity (appendix)...")
    elasticity = compute_elasticity(daily_df, peaks)

    print("  [H] Credibility-Weighted Impact...")
    cred_impact = compute_credibility_impact(art_stats, decomp)

    print("  [B] BMPI (BTC Information Manipulation Index)...")
    bmpi_df  = compute_bmpi(peaks, daily_df, art_stats, hhi, shock_rev)

    # ── MERGE all metrics ──────────────────────────────────────────────
    final = decomp.copy()
    final["source_hhi"]             = final["event_id"].map(hhi)
    final["shock_reversal_index"]   = final["event_id"].map(shock_rev)
    final["media_market_elasticity"]= final["event_id"].map(elasticity)

    for col in ["effect_half_life_days","reversal_ratio",
                "permanent_effect_share","temporary_effect_share"]:
        final = final.merge(stability[["event_id",col]], on="event_id", how="left")

    for col in ["credible_impact_usd","low_credibility_impact_usd",
                "low_cred_article_ratio","trusted_article_ratio","credibility_ratio"]:
        final = final.merge(cred_impact[["event_id",col]], on="event_id", how="left")

    for col in ["bmpi","bmpi_label"]:
        final = final.merge(bmpi_df[["event_id",col]], on="event_id", how="left")

    # ── GLOBAL STATS ───────────────────────────────────────────────────
    def safe_mean(s): return float(s.dropna().mean()) if s.dropna().any() else float("nan")
    def safe_sum(s):  return float(s.fillna(0).sum())

    peaks_sorted = peaks.sort_values("data_piku")

    global_stats = {
        "period": f"{peaks_sorted['data_piku'].min()} → {peaks_sorted['data_piku'].max()}",
        "n_events": len(final),
        "sum_excess_usd":      safe_sum(final["excess_usd"]),
        "sum_total_change":  safe_sum(final["total_change_usd"]),
        "global_anomaly_share": (safe_sum(final["excess_usd"]) /
                                  safe_sum(final["total_change_usd"])
                                  if safe_sum(final["total_change_usd"]) > 0 else float("nan")),

        "group_a": {
            "mean_anomaly":    safe_mean(final["anomaly_share"]),
            "mean_info":       safe_mean(final["info_share"]),
            "mean_fundamental":safe_mean(final["fundamental_share"]),
        },
        "group_b": {
            "mean":   safe_mean(final["bmpi"]),
            "median": float(final["bmpi"].dropna().median()) if final["bmpi"].dropna().any() else float("nan"),
            "max":    float(final["bmpi"].dropna().max()) if final["bmpi"].dropna().any() else float("nan"),
            "n_high": int((final["bmpi"] >= 0.70).sum()),
        },
        "group_c": {
            "mean_half_life":      safe_mean(final["effect_half_life_days"]),
            "mean_reversal": safe_mean(final["reversal_ratio"]),
            "mean_permanent":      safe_mean(final["permanent_effect_share"]),
            "mean_temporary":      safe_mean(final["temporary_effect_share"]),
        },
        "group_e": {
            "avg": safe_mean(final["source_hhi"]),
            "max": float(final["source_hhi"].dropna().max()) if final["source_hhi"].dropna().any() else float("nan"),
            "n_high_concentration": int((final["source_hhi"] >= 0.25).sum()),
        },
        "group_f": {
            "avg":               safe_mean(final["shock_reversal_index"]),
            "n_pump_dump":  int((final["shock_reversal_index"] >= 1.5).sum()),
        },
        "group_g": {
            "avg": safe_mean(final["media_market_elasticity"]),
        },
        "group_h": {
            "mean_lc_ratio":      safe_mean(final["low_cred_article_ratio"]),
            "sum_credible":  safe_sum(final["credible_impact_usd"]),
            "sum_low_cred":  safe_sum(final["low_credibility_impact_usd"]),
            "global_ratio": (safe_sum(final["low_credibility_impact_usd"]) /
                                   (safe_sum(final["low_credibility_impact_usd"]) +
                                    safe_sum(final["credible_impact_usd"]))
                                   if (safe_sum(final["low_credibility_impact_usd"]) +
                                       safe_sum(final["credible_impact_usd"])) > 0
                                   else float("nan")),
        },
    }

    # ── SAVE OUTPUTS ─────────────────────────────────────────────────────────
    final.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] {OUT_CSV}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"[OK] {OUT_JSON}")

    html = build_html_report(final, global_stats)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] {OUT_HTML}  ← open in browser")

    # ── CONSOLE REPORT ─────────────────────────────────────────────────
    print_console_report(final, global_stats)


if __name__ == "__main__":
    main()