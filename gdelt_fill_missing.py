# -*- coding: utf-8 -*-
"""
gdelt_fill_missing.py  [v3 — no inner ThreadPool, no hangs]
=============================================================
Downloads missing GDELT balanced dates. Sequential file fetching
per day — no ThreadPoolExecutor inside _process_day, so no hangs.

Usage:
    python gdelt_fill_missing.py --workers 6
    python gdelt_fill_missing.py --start 2016-01-01 --end 2016-12-31 --workers 6
    python gdelt_fill_missing.py --status
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import certifi

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── config ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
GDELT_FILE = BASE_DIR / "data" / "raw" / "gdelt" / \
             "gdelt_gkg_bitcoin_daily_signal_balanced.csv"

TARGET_START = date(2015, 10,  1)
TARGET_END   = date(2026,  1, 23)

KEYWORDS  = ["bitcoin", "btc"]
THEMES    = ["WEB_BITCOIN", "BITCOIN"]

GDELT_BASE   = "http://data.gdeltproject.org/gdeltv2/"
TIMEOUT_SEC  = 6      # per HTTP request
MAX_FILES    = 48     # only even hours (00,30 min) → 48 files/day instead of 96
DAY_TIMEOUT  = 90     # hard limit per day in seconds
FLUSH_EVERY  = 10
MAX_WORKERS  = 6      # day-level parallelism only

KW_PAT = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.IGNORECASE)
TH_PAT = re.compile("|".join(re.escape(t) for t in THEMES),   re.IGNORECASE)
TONE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)")


# ── HTTP ──────────────────────────────────────────────────────────────
def _get(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=TIMEOUT_SEC, verify=certifi.where())
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def _unzip(raw: bytes) -> Optional[bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next((n for n in z.namelist() if n.endswith(".csv")), z.namelist()[0])
            return z.read(name)
    except Exception:
        return None


def _parse_tone(s: str) -> Optional[float]:
    if not s:
        return None
    m = TONE_RE.match(s.strip())
    if not m:
        return None
    try:
        v = float(m.group(1))
        return v if -100 <= v <= 100 else None
    except ValueError:
        return None


def _parse_gkg(raw: bytes) -> tuple[int, Optional[float]]:
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return 0, None
    tones, matches = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        line_lower = line.lower()
        parts = line.split("\t")
        kw = KW_PAT.search(line_lower)
        th = ((TH_PAT.search(parts[7]) if len(parts) > 7 else False) or
              (TH_PAT.search(parts[8]) if len(parts) > 8 else False))
        if not (kw or th):
            continue
        matches += 1
        t = _parse_tone(parts[15] if len(parts) > 15 else "")
        if t is not None:
            tones.append(t)
    return matches, (float(np.mean(tones)) if tones else None)


# ── process one day — SEQUENTIAL, no inner threadpool ─────────────────
def _process_day(d: date) -> tuple[date, int, Optional[float]]:
    """
    Fetch GKG files for one day SEQUENTIALLY.
    Only on-the-hour files (00 min) — 24 files instead of 96.
    Hard deadline: stops after DAY_TIMEOUT seconds.
    """
    base     = d.strftime("%Y%m%d")
    # Only :00 files — 24 per day, much faster, still representative
    urls     = [f"{GDELT_BASE}{base}{h:02d}0000.gkg.csv.zip" for h in range(24)]
    deadline = time.time() + DAY_TIMEOUT

    total_mentions = 0
    all_tones: list[tuple[float, int]] = []

    for url in urls:
        if time.time() > deadline:
            break
        raw = _get(url)
        if raw is None:
            continue
        csv_bytes = _unzip(raw)
        if csv_bytes is None:
            continue
        mentions, tone = _parse_gkg(csv_bytes)
        total_mentions += mentions
        if tone is not None and mentions > 0:
            all_tones.append((tone, mentions))

    if total_mentions == 0:
        return d, 0, None

    if all_tones:
        total_w  = sum(w for _, w in all_tones)
        avg_tone = sum(t * w for t, w in all_tones) / total_w
    else:
        avg_tone = None

    return d, total_mentions, avg_tone


# ── file I/O ──────────────────────────────────────────────────────────
def _load_existing() -> pd.DataFrame:
    if not GDELT_FILE.exists():
        return pd.DataFrame(columns=["date", "mentions", "tone", "preset"])
    df = pd.read_csv(GDELT_FILE)
    df.columns = [c.lower().strip() for c in df.columns]
    renames = {}
    if 'data'            in df.columns and 'date'     not in df.columns: renames['data']            = 'date'
    if 'liczba_wzmianek' in df.columns and 'mentions' not in df.columns: renames['liczba_wzmianek'] = 'mentions'
    if 'sredni_tone'     in df.columns and 'tone'     not in df.columns: renames['sredni_tone']     = 'tone'
    if renames:
        df = df.rename(columns=renames)
    df['date'] = pd.to_datetime(df['date'].astype(str).str[:10], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=['date']).sort_values('date').drop_duplicates('date', keep='last')
    logger.info("Loaded %d rows (%s → %s)", len(df), df['date'].min().date(), df['date'].max().date())
    return df


def _save(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return existing
    df_new = pd.DataFrame(new_rows)[["date", "mentions", "tone", "preset"]]
    # Force everything to datetime before concat
    existing = existing.copy()
    existing["date"] = pd.to_datetime(existing["date"].astype(str).str[:10], errors="coerce")
    df_new["date"]   = pd.to_datetime(df_new["date"].astype(str).str[:10],   errors="coerce")
    combined = pd.concat([existing, df_new], ignore_index=True)
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    GDELT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(GDELT_FILE, index=False)
    return combined


def _print_summary(df: pd.DataFrame) -> None:
    if len(df) == 0:
        print("Empty.")
        return
    if df["date"].dtype == object:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    m = df["mentions"].dropna() if "mentions" in df.columns else pd.Series()
    t = df["tone"].dropna()     if "tone"     in df.columns else pd.Series()
    zeros = int((df.get("mentions", pd.Series()) == 0).sum())
    print()
    print("=" * 55)
    print("  GDELT BALANCED — STATUS")
    print("=" * 55)
    print(f"  Rows:        {len(df)}")
    print(f"  Dates:       {pd.to_datetime(df['date']).min().date()} → {pd.to_datetime(df['date']).max().date()}")
    if len(m): print(f"  mentions:    mean={m.mean():.1f}  std={m.std():.1f}")
    if len(m): print(f"  μM≈379:      {'✓' if abs(m.mean()-379)<30 else '✗ CHECK!'}")
    if len(t): print(f"  tone mean:   {t.mean():.4f}")
    print(f"  Zero rows:   {zeros} ({zeros/len(df)*100:.1f}%)")
    remaining = max(0, 2195 - len(df))
    print(f"  Still need:  {remaining} more rows")
    if remaining == 0:
        print("  ✓ READY — run pipeline!")
    print("=" * 55)


# ── main ──────────────────────────────────────────────────────────────
def main(target_start: date, target_end: date, workers: int) -> None:
    existing   = _load_existing()
    have_dates = set(existing["date"].dt.date)
    all_target = [target_start + timedelta(days=i)
                  for i in range((target_end - target_start).days + 1)]
    missing    = sorted(d for d in all_target if d not in have_dates)

    if not missing:
        logger.info("Nothing to download.")
        _print_summary(existing)
        return

    logger.info("Missing: %d | Workers: %d (day-level only)", len(missing), workers)
    logger.info("Est. @ %ds/day × %d days / %d workers = ~%.0f min",
                DAY_TIMEOUT, len(missing), workers,
                len(missing) * DAY_TIMEOUT / workers / 60)

    pbar = tqdm(total=len(missing), unit="day", desc="GDELT", ncols=75) if HAS_TQDM else None
    buf: list[dict] = []
    current_df = existing.copy()
    skipped = 0

    # Day-level parallelism only — no nested ThreadPool inside
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_process_day, d): d for d in missing}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                day, mentions, tone = fut.result(timeout=DAY_TIMEOUT + 10)
                buf.append({
                    "date":     str(day),
                    "mentions": int(mentions),
                    "tone":     float(tone) if tone is not None else float("nan"),
                    "preset":   "balanced",
                })
            except Exception:
                skipped += 1
                buf.append({"date": str(d), "mentions": 0,
                            "tone": float("nan"), "preset": "balanced"})

            if len(buf) >= FLUSH_EVERY:
                current_df = _save(current_df, buf)
                buf = []
                logger.info("Saved. Rows: %d | Skipped: %d", len(current_df), skipped)

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()
    if buf:
        current_df = _save(current_df, buf)

    logger.info("Done. Rows: %d | Skipped: %d", len(current_df), skipped)
    _print_summary(current_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill missing GDELT balanced dates (no-hang version)")
    parser.add_argument("--start",   default=str(TARGET_START))
    parser.add_argument("--end",     default=str(TARGET_END))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--status",  action="store_true")
    args = parser.parse_args()
    if args.status:
        _print_summary(_load_existing())
    else:
        main(date.fromisoformat(args.start), date.fromisoformat(args.end), args.workers)