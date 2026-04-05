# -*- coding: utf-8 -*-
"""
pipelines/step15_download_gdelt_events_peak_articles.py
=========================================================
Download article-level GDELT Events records for BTC peak event windows.

This is a corrected version of the original downloader with:
  [FIX-1] Improved Bitcoin keyword filter — avoids false positives like
          "BTC sale" (company abbreviation), "blockchain consulting" etc.
          Now requires context: bitcoin/btc must appear with crypto context
          OR url must contain specific crypto domains.
  [FIX-2] Path uses Path(__file__).resolve() instead of hardcoded Windows path.
  [FIX-3] Supports both English and Polish column names in peak files.
  [FIX-4] Checkpoint/resume via SQLite — restart = continues from last point.

Output: data/raw/news_raw/gdelt_events_peaks_articles/
          gdelt_events_peak_articles.jsonl

Usage:
  python -m bmpi.pipelines.step15_download_gdelt_events_peak_articles
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
from urllib.parse import urlparse

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_RAW_NEWS  = BASE_DIR / "data" / "raw" / "news_raw"
OUT_DIR        = DATA_RAW_NEWS / "gdelt_events_peaks_articles"

PEAKS_FILES = [
    DATA_PROCESSED / "events_peaks_sensitive.csv",
    DATA_PROCESSED / "events_peaks_balanced.csv",
    DATA_PROCESSED / "events_peaks_strong.csv",
]

GDELT_EVENTS_BASE = "http://data.gdeltproject.org/events/"
OUT_JSONL         = OUT_DIR / "gdelt_events_peak_articles.jsonl"
STATE_DB          = OUT_DIR / "gdelt_events_download_state.sqlite"
RUN_LOG           = OUT_DIR / "gdelt_events_download.log"

# ---------------------------------------------------------------------------
# [FIX-1] Improved Bitcoin keyword filter
# ---------------------------------------------------------------------------
#
# Problem with simple r"\b(bitcoin|btc|...)\b":
#   - "btc-sale" matches → VTB Capital deal (false positive)
#   - "blockchain consulting" matches → unrelated company (false positive)
#
# Solution: two-tier filter
#   Tier 1 (STRONG): unambiguous Bitcoin indicators → always accept
#   Tier 2 (CONTEXT): ambiguous terms → require additional crypto context
#
# Tier 1 examples: "bitcoin-price", "coinbase", "satoshi", "crypto-market"
# Tier 2 examples: "btc" alone, "blockchain" alone → check for context

# Tier 1: unambiguous — always a Bitcoin article
STRONG_KEYWORDS = re.compile(
    r"\b(bitcoin|satoshi|coinbase|binance|cryptocurrency|crypto[-_]market|"
    r"bitcoin[-_]price|bitcoin[-_]news|btc[-_]usd|btcusd|"
    r"bitcoin[-_]mining|bitcoin[-_]wallet|bitcoin[-_]exchange|"
    r"crypto[-_]currency|cryptomarket|altcoin|defi|"
    r"web3|nft|ethereum|litecoin|ripple|dogecoin)\b",
    re.IGNORECASE,
)

# Tier 2: ambiguous — only accept if also has crypto context
AMBIGUOUS_KEYWORDS = re.compile(r"\b(btc|blockchain|crypto)\b", re.IGNORECASE)

# Crypto context indicators
CRYPTO_CONTEXT = re.compile(
    r"\b(bitcoin|coin|wallet|mining|exchange|crypto|"
    r"blockchain|satoshi|hodl|altcoin|defi|token|"
    r"market[-_]cap|mcap|digital[-_]currency|"
    r"price|usd|eur|bull|bear|pump|dump)\b",
    re.IGNORECASE,
)

# Known crypto domains → always accept
CRYPTO_DOMAINS = {
    "coindesk.com", "cointelegraph.com", "bitcoin.com", "bitcoinmagazine.com",
    "decrypt.co", "theblock.co", "coinbase.com", "binance.com",
    "cryptoslate.com", "ambcrypto.com", "newsbtc.com", "u.today",
    "cryptonews.com", "bitcoinist.com", "cryptobriefing.com",
}


def is_bitcoin_article(url: str) -> bool:
    """
    Two-tier filter to determine if a URL is about Bitcoin/crypto.
    Avoids false positives like 'btc-sale' (company abbreviation).
    """
    if not url:
        return False

    url_lower = url.lower()

    # Check known crypto domains first (fastest path)
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if any(cd in domain for cd in CRYPTO_DOMAINS):
            return True
    except Exception:
        pass

    # Tier 1: strong indicators → accept immediately
    if STRONG_KEYWORDS.search(url_lower):
        return True

    # Tier 2: ambiguous terms → require crypto context
    if AMBIGUOUS_KEYWORDS.search(url_lower):
        # Count crypto context signals in the URL
        context_matches = len(CRYPTO_CONTEXT.findall(url_lower))
        # Need at least 2 crypto signals OR "bitcoin" somewhere
        if context_matches >= 2 or "bitcoin" in url_lower:
            return True

    return False


# ---------------------------------------------------------------------------
# HTTP / download settings
# ---------------------------------------------------------------------------

TIMEOUT_SEC               = 30
RETRIES                   = 2
SLEEP_BETWEEN_REQUESTS    = 0.15
USER_AGENT                = "BMPI-Research/1.0 (academic; non-commercial)"
DELETE_CSV_AFTER          = True
DELETE_ZIP_AFTER          = False
LOG_EVERY_ROWS            = 50000
STATE_COMMIT_EVERY        = 1000

# GDELT Events column indices (61 columns, tab-separated)
IDX_GLOBALEVENTID = 0
IDX_SQLDATE       = 1
IDX_EVENTCODE     = 26
IDX_EVENTROOTCODE = 28
IDX_GOLDSTEIN     = 30
IDX_NUMMENTIONS   = 31
IDX_NUMSOURCES    = 32
IDX_NUMARTICLES   = 33
IDX_AVGTONE       = 34
IDX_SOURCEURL     = 57


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventWindow:
    preset:   str
    event_id: str
    start:    datetime
    end:      datetime

    @property
    def key(self) -> str:
        return f"{self.preset}:{self.event_id}:{self.start.date()}_{self.end.date()}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _log(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} | {message}"
    print(line)
    _ensure_dir(RUN_LOG.parent)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except Exception:
        return None


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _domain_from_url(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _infer_preset(file_path: Path) -> str:
    name = file_path.stem.lower()
    for p in ["sensitive", "balanced", "strong"]:
        if p in name:
            return p
    return name


def _find_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None


def _record_key(event_id: str, preset: str, date_str: str,
                url: str, gdelt_id: str) -> str:
    return "||".join([event_id, preset, date_str, url, gdelt_id])


# ---------------------------------------------------------------------------
# SQLite state store (checkpoint/resume + deduplication)
# ---------------------------------------------------------------------------

class _StateDB:
    def __init__(self, path: Path):
        _ensure_dir(path.parent)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._pending = 0
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS completed_days (
                day TEXT PRIMARY KEY, ts TEXT);
            CREATE TABLE IF NOT EXISTS written_records (
                record_key TEXT PRIMARY KEY,
                event_id TEXT, preset TEXT, day TEXT, ts TEXT);
            CREATE INDEX IF NOT EXISTS idx_wr_day ON written_records(day);
        """)
        self.conn.commit()

    def day_done(self, day: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM completed_days WHERE day=? LIMIT 1", (day,)
        ).fetchone() is not None

    def mark_day(self, day: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO completed_days(day,ts) VALUES(?,?)",
            (day, datetime.now().isoformat()),
        )
        self._flush()

    def add_record(self, key: str, event_id: str, preset: str, day: str) -> bool:
        try:
            self.conn.execute(
                "INSERT INTO written_records(record_key,event_id,preset,day,ts)"
                " VALUES(?,?,?,?,?)",
                (key, event_id, preset, day, datetime.now().isoformat()),
            )
            self._pending += 1
            if self._pending >= STATE_COMMIT_EVERY:
                self.conn.commit()
                self._pending = 0
            return True
        except sqlite3.IntegrityError:
            return False

    def _flush(self) -> None:
        self.conn.commit()
        self._pending = 0

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Load peak windows
# ---------------------------------------------------------------------------

def load_windows() -> List[EventWindow]:
    windows: List[EventWindow] = []
    for path in PEAKS_FILES:
        if not path.exists():
            _log(f"[WARN] Peak file not found: {path}")
            continue
        df = pd.read_csv(path)
        col_id    = _find_col(df, ["event_id", "zdarzenie_id"])
        col_start = _find_col(df, ["window_start", "okno_start", "start"])
        col_end   = _find_col(df, ["window_end", "okno_koniec", "end"])
        col_preset= _find_col(df, ["preset", "okno_preset"])
        inferred  = _infer_preset(path)

        if not (col_id and col_start and col_end):
            raise ValueError(
                f"Required columns missing in {path.name}: {list(df.columns)}"
            )

        for _, row in df.iterrows():
            try:
                start = pd.to_datetime(str(row[col_start])).to_pydatetime().replace(
                    hour=0, minute=0, second=0, microsecond=0)
                end   = pd.to_datetime(str(row[col_end])).to_pydatetime().replace(
                    hour=0, minute=0, second=0, microsecond=0)
            except Exception:
                continue
            windows.append(EventWindow(
                preset   = str(row[col_preset]).strip() if col_preset else inferred,
                event_id = str(row[col_id]).strip(),
                start    = start,
                end      = end,
            ))

    return sorted(windows, key=lambda w: (w.start, w.preset, w.event_id))


def _union_days(windows: List[EventWindow]) -> List[datetime]:
    days: Set[object] = set()
    for w in windows:
        d = w.start
        while d <= w.end:
            days.add(d.date())
            d += timedelta(days=1)
    return sorted(datetime.combine(d, datetime.min.time()) for d in days)


def _windows_for_day(windows: List[EventWindow], day: datetime) -> List[EventWindow]:
    return [w for w in windows if w.start <= day <= w.end]


# ---------------------------------------------------------------------------
# GDELT download + parse
# ---------------------------------------------------------------------------

def _download_zip(day: datetime, session: requests.Session) -> Optional[Path]:
    ymd  = day.strftime("%Y%m%d")
    name = f"{ymd}.export.CSV.zip"
    url  = GDELT_EVENTS_BASE + name
    out  = OUT_DIR / name
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        r = session.get(url, timeout=TIMEOUT_SEC,
                        headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        out.write_bytes(r.content)
        return out
    except Exception as e:
        _log(f"[WARN] Cannot download {name}: {e}")
        return None


def _extract_csv(zip_path: Path) -> Optional[Path]:
    csv_path = zip_path.with_suffix("")   # remove .zip
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(OUT_DIR)
        if csv_path.exists():
            return csv_path
        # fallback: first file in archive
        with zipfile.ZipFile(zip_path) as z:
            first = OUT_DIR / z.namelist()[0]
        return first if first.exists() else None
    except Exception as e:
        _log(f"[WARN] ZIP extraction failed {zip_path.name}: {e}")
        return None


def _parse_row(row: List[str]) -> Optional[dict]:
    if len(row) <= IDX_SOURCEURL:
        return None
    url = row[IDX_SOURCEURL].strip()
    if not url:
        return None

    # [FIX-1] improved filter
    if not is_bitcoin_article(url):
        return None

    sql_date = row[IDX_SQLDATE].strip()
    try:
        dt = datetime.strptime(sql_date, "%Y%m%d")
    except Exception:
        return None

    domain = _domain_from_url(url)
    date_str = dt.strftime("%Y-%m-%d")

    return {
        "gdelt_globaleventid": row[IDX_GLOBALEVENTID].strip(),
        "date":         date_str,
        "data":         date_str,          # legacy alias
        "event_code":   row[IDX_EVENTCODE].strip()     if len(row) > IDX_EVENTCODE     else "",
        "event_root_code": row[IDX_EVENTROOTCODE].strip() if len(row) > IDX_EVENTROOTCODE else "",
        "goldstein":    _safe_float(row[IDX_GOLDSTEIN])  if len(row) > IDX_GOLDSTEIN  else None,
        "num_mentions": _safe_int(row[IDX_NUMMENTIONS])  if len(row) > IDX_NUMMENTIONS else None,
        "num_sources":  _safe_int(row[IDX_NUMSOURCES])   if len(row) > IDX_NUMSOURCES  else None,
        "num_articles": _safe_int(row[IDX_NUMARTICLES])  if len(row) > IDX_NUMARTICLES else None,
        "avg_tone":     _safe_float(row[IDX_AVGTONE])    if len(row) > IDX_AVGTONE     else None,
        "source_url":   url,
        "url":          url,
        "source_domain":domain,
        "domain":       domain,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_dir(OUT_DIR)

    windows = load_windows()
    if not windows:
        raise RuntimeError("No event windows found — run step03 first.")

    all_days = _union_days(windows)
    _log(f"[INFO] {len(windows)} windows | {len(all_days)} unique days "
         f"| {all_days[0].date()} → {all_days[-1].date()}")
    _log(f"[INFO] Output: {OUT_JSONL}")

    state   = _StateDB(STATE_DB)
    session = requests.Session()

    total_scanned = total_written = total_dup = 0

    with open(OUT_JSONL, "a", encoding="utf-8") as out_fp:
        for idx, day in enumerate(all_days, 1):
            day_str = day.strftime("%Y-%m-%d")

            if state.day_done(day_str):
                continue

            day_windows = _windows_for_day(windows, day)
            if not day_windows:
                state.mark_day(day_str)
                continue

            zip_path = _download_zip(day, session)
            if not zip_path:
                continue

            csv_path = _extract_csv(zip_path)
            if not csv_path:
                continue

            scanned = written = dup = 0
            try:
                with open(csv_path, encoding="utf-8", errors="replace",
                          newline="") as f:
                    reader = csv.reader(f, delimiter="\t")
                    for row in reader:
                        scanned += 1
                        record = _parse_row(row)
                        if not record:
                            continue

                        for w in day_windows:
                            rkey = _record_key(
                                w.event_id, w.preset,
                                record["date"], record["source_url"],
                                record["gdelt_globaleventid"],
                            )
                            if not state.add_record(rkey, w.event_id, w.preset, day_str):
                                dup += 1
                                continue

                            obj = dict(record)
                            obj.update({
                                "preset":       w.preset,
                                "event_id":     w.event_id,
                                "window_start": w.start.strftime("%Y-%m-%d"),
                                "window_end":   w.end.strftime("%Y-%m-%d"),
                                "window_key":   w.key,
                                # legacy aliases
                                "okno_preset":  w.preset,
                                "okno_start":   w.start.strftime("%Y-%m-%d"),
                                "okno_koniec":  w.end.strftime("%Y-%m-%d"),
                                "okno_key":     w.key,
                                "article_domain": record["source_domain"],
                                "article_url":    record["source_url"],
                                "article_title":  "",
                            })
                            out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            written += 1

                        if written % 100 == 0:
                            out_fp.flush()

                out_fp.flush()
                state.mark_day(day_str)
                total_scanned += scanned
                total_written += written
                total_dup += dup

                _log(f"[{idx:4d}/{len(all_days)}] {day_str} | "
                     f"scanned={scanned:,} written={written:,} dup={dup:,}")

            except Exception as e:
                out_fp.flush()
                _log(f"[ERROR] {day_str}: {e}")
                _log("[INFO] Progress saved — restart to continue.")
                raise

            finally:
                if DELETE_CSV_AFTER and csv_path.exists():
                    try:
                        csv_path.unlink()
                    except Exception:
                        pass
                if DELETE_ZIP_AFTER and zip_path.exists():
                    try:
                        zip_path.unlink()
                    except Exception:
                        pass

    state.close()
    session.close()

    _log("")
    _log("DONE")
    _log(f"  Total rows scanned:  {total_scanned:,}")
    _log(f"  Total JSONL written: {total_written:,}")
    _log(f"  Total duplicates:    {total_dup:,}")
    _log(f"  Output:              {OUT_JSONL}")
    _log("Next step: step14_attribution_fake_usd_to_sources.py")


if __name__ == "__main__":
    main()