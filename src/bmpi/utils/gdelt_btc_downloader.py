# -*- coding: utf-8 -*-
"""
utils/gdelt_btc_downloader.py  [FIXED VERSION for article reproduction]
========================================================================
GDELT GKG downloader for the BTC media signal.

CHANGES vs original:
  - balanced preset uses ["bitcoin","btc"] (2 keywords) to match
    the article calibration parameters: μM=379.0, σM=305.8, μT=-0.912
  - CheckpointWriter reads existing files regardless of column name format
    (handles both Polish: data/liczba_wzmianek/sredni_tone
     and English: date/mentions/tone)

Usage:
    # Resume downloading balanced preset (run after convert_balanced_checkpoint.py)
    python -m bmpi.utils.gdelt_btc_downloader --mode full --start 2015-10-01 --end 2026-01-31 --preset balanced

    # All presets
    python -m bmpi.utils.gdelt_btc_downloader --mode full --start 2015-10-01 --end 2026-01-31 --all-presets

    # Status
    python -m bmpi.utils.gdelt_btc_downloader --status
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
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

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR / "src"))

try:
    from bmpi.utils.checkpoint import CheckpointWriter, status_report
except ImportError:
    chk_path = Path(__file__).parent / "checkpoint.py"
    if chk_path.exists():
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("checkpoint", chk_path)
        _mod  = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        CheckpointWriter = _mod.CheckpointWriter
        status_report    = _mod.status_report
    else:
        raise

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_GDELT     = BASE_DIR / "data" / "raw" / "gdelt"
DATA_GDELT.mkdir(parents=True, exist_ok=True)

PEAK_FILES = {
    "sensitive": DATA_PROCESSED / "events_peaks_sensitive.csv",
    "balanced":  DATA_PROCESSED / "events_peaks_balanced.csv",
    "strong":    DATA_PROCESSED / "events_peaks_strong.csv",
}

OUTPUT_FILES = {
    "all":       DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_ALL.csv",
    "balanced":  DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_balanced.csv",
    "sensitive": DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_sensitive.csv",
    "strong":    DATA_GDELT / "gdelt_gkg_bitcoin_daily_signal_strong.csv",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GDELT_BASE    = "http://data.gdeltproject.org/gdeltv2/"
TIMEOUT_SEC   = 45
MAX_WORKERS   = 4
FLUSH_DAYS    = 7
SLEEP_BETWEEN = 0.15
WINDOW_BEFORE = 30
WINDOW_AFTER  = 30
TONE_MIN, TONE_MAX = -100.0, 100.0
TONE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)")

# ---------------------------------------------------------------------------
# Keyword presets
# *** FIXED: balanced uses 2 keywords to reproduce article μM=379 ***
# ---------------------------------------------------------------------------
KEYWORDS = {
    "all":       ["bitcoin", "btc", "cryptocurrency", "crypto currency",
                  "satoshi", "blockchain", "coinbase", "binance",
                  "cryptomarket", "digital currency", "digitalcurrency"],
    "balanced":  ["bitcoin", "btc"],                          # FIXED: was 6 keywords
    "sensitive": ["bitcoin", "btc", "cryptocurrency"],
    "strong":    ["bitcoin", "btc"],
}
THEMES = {
    "all":       ["WEB_BITCOIN", "ECON_CRYPTOCURRENCY", "ECON_DIGITALCURRENCY",
                  "BITCOIN", "CRYPTO", "BTC"],
    "balanced":  ["WEB_BITCOIN", "BITCOIN"],                  # FIXED: was 4 themes
    "sensitive": ["WEB_BITCOIN", "BITCOIN"],
    "strong":    ["WEB_BITCOIN"],
}

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _get(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=TIMEOUT_SEC, verify=certifi.where())
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content
    except requests.exceptions.SSLError:
        try:
            r = requests.get(url.replace("https://", "http://"), timeout=TIMEOUT_SEC)
            return None if r.status_code == 404 else (r.raise_for_status() or r.content)
        except Exception:
            return None
    except Exception:
        return None


def _unzip(raw: bytes) -> Optional[bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next((n for n in z.namelist() if n.endswith(".csv")), z.namelist()[0])
            return z.read(name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GKG parsing
# ---------------------------------------------------------------------------
def _parse_tone(s: str) -> Optional[float]:
    if not s:
        return None
    m = TONE_RE.match(s.strip())
    if not m:
        return None
    try:
        v = float(m.group(1))
        return v if TONE_MIN <= v <= TONE_MAX else None
    except ValueError:
        return None


def parse_gkg_bytes(raw: bytes, kw_pat: re.Pattern,
                    th_pat: re.Pattern) -> tuple[int, Optional[float]]:
    try:
        raw_text = raw.decode("utf-8", errors="replace")
    except Exception:
        return 0, None

    tones, matches = [], 0
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        line_lower = line.lower()
        kw_found   = kw_pat.search(line_lower)
        parts      = line.split("\t")
        th_found   = (
            (th_pat.search(parts[7]) if len(parts) > 7 else False) or
            (th_pat.search(parts[8]) if len(parts) > 8 else False)
        )
        if not (kw_found or th_found):
            continue
        matches += 1
        tone_raw = parts[15] if len(parts) > 15 else ""
        t = _parse_tone(tone_raw)
        if t is not None:
            tones.append(t)

    return matches, (float(np.mean(tones)) if tones else None)


# ---------------------------------------------------------------------------
# Single day
# ---------------------------------------------------------------------------
def _urls_for_day(d: date) -> list[str]:
    base = d.strftime("%Y%m%d")
    return [
        f"{GDELT_BASE}{base}{h:02d}{m:02d}00.gkg.csv.zip"
        for h in range(24) for m in (0, 15, 30, 45)
    ]


def _fetch_one_url(url: str, kw_pat: re.Pattern,
                   th_pat: re.Pattern) -> tuple[int, Optional[float]]:
    raw = _get(url)
    if raw is None:
        return 0, None
    csv_bytes = _unzip(raw)
    if csv_bytes is None:
        return 0, None
    return parse_gkg_bytes(csv_bytes, kw_pat, th_pat)


def _process_day(day: date, kw_pat: re.Pattern,
                 th_pat: re.Pattern,
                 file_workers: int = 12) -> tuple[date, int, Optional[float]]:
    urls           = _urls_for_day(day)
    total_mentions = 0
    all_tones: list[tuple[float, int]] = []

    with ThreadPoolExecutor(max_workers=file_workers) as pool:
        futs = {pool.submit(_fetch_one_url, url, kw_pat, th_pat): url for url in urls}
        for fut in as_completed(futs):
            try:
                mentions, tone = fut.result()
                total_mentions += mentions
                if tone is not None and mentions > 0:
                    all_tones.append((tone, mentions))
            except Exception:
                pass

    if total_mentions == 0:
        return day, 0, None

    if all_tones:
        total_w  = sum(w for _, w in all_tones)
        avg_tone = (sum(t * w for t, w in all_tones) / total_w
                    if total_w > 0
                    else sum(t for t, _ in all_tones) / len(all_tones))
    else:
        avg_tone = None

    return day, total_mentions, avg_tone


# ---------------------------------------------------------------------------
# Checkpoint — reads existing file regardless of column name format
# ---------------------------------------------------------------------------
def _read_existing_dates(path: Path) -> set[date]:
    """
    Read existing dates from checkpoint file.
    Handles both old format (data/liczba_wzmianek/sredni_tone)
    and new format (date/mentions/tone).
    """
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=[0])   # first column is always date
        col = df.columns[0]
        dates = pd.to_datetime(
            df[col].astype(str).str.strip().str[:10],
            format="%Y-%m-%d", errors="coerce"
        ).dropna()
        return set(d.date() for d in dates)
    except Exception as e:
        logger.warning("Could not read existing checkpoint: %s", e)
        return set()


def _append_rows(path: Path, rows: list[dict]) -> None:
    """Append rows to CSV. Creates file with header if it doesn't exist."""
    df_new = pd.DataFrame(rows)[["date", "mentions", "tone", "preset"]]
    if path.exists():
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------
def download_signal(days: list[date], preset: str,
                    workers: int = MAX_WORKERS) -> pd.DataFrame:
    out_path = OUTPUT_FILES[preset]
    kw_pat   = re.compile("|".join(re.escape(k) for k in KEYWORDS[preset]),
                          re.IGNORECASE)
    th_pat   = re.compile("|".join(re.escape(t) for t in THEMES[preset]),
                          re.IGNORECASE)

    # Use our custom reader that handles both column formats
    done = _read_existing_dates(out_path)
    todo = sorted(d for d in days if d not in done)

    if not todo:
        logger.info("[%s] All %d days already done.", preset, len(done))
        return pd.read_csv(out_path)

    logger.info("[%s] To download: %d | Already done: %d | Workers: %d",
                preset, len(todo), len(done), workers)

    pbar = tqdm(total=len(todo), unit="day",
                desc=f"GDELT [{preset}]", ncols=88) if HAS_TQDM else None

    buf: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_day, d, kw_pat, th_pat): d for d in todo}

        for future in as_completed(futures):
            try:
                day, mentions, tone = future.result()
                buf.append({
                    "date":     str(day),
                    "mentions": mentions,
                    "tone":     tone if tone is not None else np.nan,
                    "preset":   preset,
                })
                if len(buf) >= FLUSH_DAYS:
                    _append_rows(out_path, buf)
                    buf = []
            except Exception as e:
                d = futures[future]
                logger.debug("Error %s: %s", d, e)
                buf.append({
                    "date": str(d), "mentions": 0,
                    "tone": np.nan, "preset": preset,
                })
            finally:
                if pbar:
                    pbar.update(1)

            time.sleep(SLEEP_BETWEEN / max(workers, 1))

    if pbar:
        pbar.close()
    if buf:
        _append_rows(out_path, buf)

    # Load, deduplicate, sort and resave
    result = pd.read_csv(out_path)
    result.columns = [c.lower().strip() for c in result.columns]
    date_col = next((c for c in result.columns if c in ('date','data','day')),
                    result.columns[0])
    result["date"] = pd.to_datetime(
        result[date_col].astype(str).str[:10],
        format="%Y-%m-%d", errors="coerce"
    )
    result = (result.dropna(subset=["date"])
                    .sort_values("date")
                    .drop_duplicates(subset=["date"], keep="last")
                    .reset_index(drop=True))
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result.to_csv(out_path, index=False)

    m = result["mentions"].dropna() if "mentions" in result.columns else pd.Series()
    t = result["tone"].dropna()     if "tone"     in result.columns else pd.Series()
    logger.info("[%s] Done: %d days | mentions mean=%.0f | tone mean=%.4f",
                preset, len(result),
                m.mean() if len(m) > 0 else 0,
                t.mean() if len(t) > 0 else 0)
    return result


# ---------------------------------------------------------------------------
# Peaks helper
# ---------------------------------------------------------------------------
def _build_peak_days(preset: str) -> list[date]:
    path = PEAK_FILES.get(preset, PEAK_FILES["balanced"])
    if not path.exists():
        raise FileNotFoundError(
            f"Peak file not found: {path}\n"
            "Run first: python -m bmpi.pipelines.step03_peak_detection"
        )
    df       = pd.read_csv(path)
    date_col = next(
        (c for c in df.columns
         if c.lower() in ("peak_date", "date", "event_date")),
        df.columns[0],
    )
    peak_dates = pd.to_datetime(
        df[date_col].astype(str).str.replace(" UTC", "", regex=False),
        errors="coerce",
    ).dropna()

    days: set[date] = set()
    for d in peak_dates.unique():
        for i in range(-WINDOW_BEFORE, WINDOW_AFTER + 1):
            days.add((pd.Timestamp(d) + pd.Timedelta(days=i)).date())

    result = sorted(days)
    logger.info("[%s] Peaks: %d peaks → %d unique days",
                preset, len(peak_dates), len(result))
    return result


def _date_range(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BMPI — GDELT BTC downloader (fixed for article reproduction)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW to reproduce article numbers:

  Step 1 — convert old file:
    python convert_balanced_checkpoint.py

  Step 2 — resume download:
    python -m bmpi.utils.gdelt_btc_downloader ^
        --mode full --start 2015-10-01 --end 2026-01-31 ^
        --preset balanced --workers 4

  Step 3 — run pipeline:
    python src/bmpi/pipelines/step01_data_normalisation.py
    ... (steps 02-16)
        """,
    )
    parser.add_argument("--mode",        default="full", choices=["full", "peaks"])
    parser.add_argument("--start",       default=None,   help="Start date YYYY-MM-DD")
    parser.add_argument("--end",         default=None,   help="End date YYYY-MM-DD")
    parser.add_argument("--preset",      default="balanced",
                        choices=["all", "balanced", "sensitive", "strong"])
    parser.add_argument("--all-presets", action="store_true")
    parser.add_argument("--workers",     type=int, default=MAX_WORKERS)
    parser.add_argument("--status",      action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    DATA_GDELT.mkdir(parents=True, exist_ok=True)

    if args.status:
        print("\n=== GDELT BTC Signal — status ===")
        for name, path in OUTPUT_FILES.items():
            if path.exists():
                done = _read_existing_dates(path)
                df   = pd.read_csv(path)
                print(f"  {name:<12} {len(df):>5} rows  "
                      f"({min(done)} → {max(done) if done else '—'})")
            else:
                print(f"  {name:<12} not found")
        return

    presets = (["all", "balanced", "sensitive", "strong"]
               if args.all_presets else [args.preset])

    for preset in presets:
        print(f"\n{'='*55}\n  Preset: {preset}\n{'='*55}")
        print(f"  Keywords: {KEYWORDS[preset]}")
        print(f"  Themes:   {THEMES[preset]}")

        if args.mode == "peaks":
            days = _build_peak_days(preset)
        else:
            if not args.start:
                parser.error("--start required for --mode full")
            start = date.fromisoformat(args.start)
            end   = date.fromisoformat(args.end) if args.end else date.today()
            days  = _date_range(start, end)
            logger.info("[%s] Full: %d days (%s – %s)",
                        preset, len(days), start, end)

        result = download_signal(days, preset, args.workers)

        print(f"\n  ✓ {preset}: {len(result)} days → {OUTPUT_FILES[preset]}")
        m = result["mentions"].dropna() if "mentions" in result.columns else pd.Series()
        t = result["tone"].dropna()     if "tone"     in result.columns else pd.Series()
        if len(m):
            print(f"  mentions: mean={m.mean():.0f}  std={m.std():.0f}")
            ok = abs(m.mean() - 379) < 20
            print(f"  Target μM≈379: {'✓' if ok else '✗ recalibration needed'}")
        if len(t):
            print(f"  tone:     mean={t.mean():.4f}  std={t.std():.4f}")


if __name__ == "__main__":
    main()