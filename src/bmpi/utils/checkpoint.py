# -*- coding: utf-8 -*-
"""
utils/checkpoint.py
====================
Atomic checkpoint writer for the GDELT downloader.
Saves progress every N days so downloads can be resumed after crashes.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd


class CheckpointWriter:
    """
    Writes rows to a CSV file incrementally.
    On re-run, reads existing dates so already-downloaded days are skipped.
    """

    def __init__(self,
                 output_path: Path,
                 key_col: str = "date",
                 flush_every: int = 7):
        self.output_path = Path(output_path)
        self.key_col     = key_col
        self.flush_every = flush_every
        self._buf: list[dict] = []

    # ------------------------------------------------------------------
    def existing_dates(self) -> set[date]:
        """Return set of dates already in the output file."""
        if not self.output_path.exists():
            return set()
        try:
            df = pd.read_csv(self.output_path, usecols=[0])
            col = df.columns[0]
            dates = pd.to_datetime(
                df[col].astype(str).str.strip().str[:10],
                format="%Y-%m-%d", errors="coerce"
            ).dropna()
            return set(d.date() for d in dates)
        except Exception:
            return set()

    # ------------------------------------------------------------------
    def add_many(self, rows: list[dict]) -> None:
        """Buffer rows and flush to disk when buffer is large enough."""
        self._buf.extend(rows)
        if len(self._buf) >= self.flush_every:
            self._flush()

    # ------------------------------------------------------------------
    def _flush(self) -> None:
        if not self._buf:
            return
        df_new = pd.DataFrame(self._buf)
        # Ensure date column is first
        cols = [self.key_col] + [c for c in df_new.columns if c != self.key_col]
        df_new = df_new[[c for c in cols if c in df_new.columns]]

        if self.output_path.exists():
            df_new.to_csv(self.output_path, mode="a", header=False, index=False)
        else:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            df_new.to_csv(self.output_path, index=False)
        self._buf = []

    # ------------------------------------------------------------------
    def finish(self) -> pd.DataFrame:
        """Flush remaining buffer, deduplicate and sort. Returns final DataFrame."""
        if self._buf:
            self._flush()

        if not self.output_path.exists():
            return pd.DataFrame(columns=[self.key_col, "mentions", "tone", "preset"])

        df = pd.read_csv(self.output_path)
        df.columns = [c.lower().strip() for c in df.columns]

        # Normalise date column (handles Polish: data, or English: date)
        date_col = next((c for c in df.columns if c in ("date", "data", "day")),
                        df.columns[0])
        df["date"] = pd.to_datetime(
            df[date_col].astype(str).str.strip().str[:10],
            format="%Y-%m-%d", errors="coerce"
        )

        # Normalise mentions column (handles Polish: liczba_wzmianek)
        m_col = next((c for c in df.columns
                      if c in ("mentions", "liczba_wzmianek")), None)
        if m_col and m_col != "mentions":
            df["mentions"] = df[m_col]

        # Normalise tone column (handles Polish: sredni_tone)
        t_col = next((c for c in df.columns
                      if c in ("tone", "sredni_tone", "tone_avg")), None)
        if t_col and t_col != "tone":
            df["tone"] = df[t_col]

        # Keep only needed columns
        keep = ["date", "mentions", "tone"]
        if "preset" in df.columns:
            keep.append("preset")
        df = df[[c for c in keep if c in df.columns]]

        # Deduplicate and sort
        df = (df.dropna(subset=["date"])
                .sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .reset_index(drop=True))

        # Re-save clean version
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df.to_csv(self.output_path, index=False)

        return df


# ---------------------------------------------------------------------------

def status_report(folder: Path) -> None:
    """Print download status for all CSV files in the GDELT folder."""
    folder = Path(folder)
    if not folder.exists():
        print(f"  Folder not found: {folder}")
        return

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"  No CSV files in {folder}")
        return

    for path in csv_files:
        try:
            df = pd.read_csv(path, usecols=[0])
            col = df.columns[0]
            dates = pd.to_datetime(
                df[col].astype(str).str.strip().str[:10],
                format="%Y-%m-%d", errors="coerce"
            ).dropna()
            n    = len(dates)
            dmin = dates.min().date() if n > 0 else "—"
            dmax = dates.max().date() if n > 0 else "—"
            print(f"  {path.name:<55} {n:>5} rows  ({dmin} → {dmax})")
        except Exception as e:
            print(f"  {path.name:<55} ERROR: {e}")