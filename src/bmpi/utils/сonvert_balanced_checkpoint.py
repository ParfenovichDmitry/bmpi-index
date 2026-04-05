# -*- coding: utf-8 -*-
"""
convert_balanced_checkpoint.py
================================
Converts the old balanced GDELT file (Polish column names) to the
format expected by the current CheckpointWriter / downloader.

Run this ONCE before running the downloader to resume downloading.

Usage:
    python convert_balanced_checkpoint.py

Input:  old_balanced.csv  (put it in the same folder as this script)
Output: data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv
"""

from pathlib import Path
import pandas as pd
import sys

# ---------------------------------------------------------------------------
# Paths — adjust if needed
# ---------------------------------------------------------------------------
HERE        = Path(__file__).resolve().parent
OLD_FILE    = HERE / "old_balanced.csv"           # put your old file here

# Auto-detect project root (3 levels up from src/bmpi/utils)
# If running from project root, use:
PROJECT_ROOT = HERE                                # adjust if needed

OUT_FILE = PROJECT_ROOT / "data" / "raw" / "gdelt" / \
           "gdelt_gkg_bitcoin_daily_signal_balanced.csv"

# ---------------------------------------------------------------------------

def main():
    if not OLD_FILE.exists():
        print(f"[ERROR] Old file not found: {OLD_FILE}")
        print(f"  Put your old balanced CSV file next to this script")
        print(f"  and rename it to: old_balanced.csv")
        sys.exit(1)

    print(f"[INFO] Reading: {OLD_FILE}")
    df = pd.read_csv(OLD_FILE)
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"  Columns found: {list(df.columns)}")
    print(f"  Rows: {len(df)}")

    # Detect column names (handles both Polish and English)
    date_col     = next((c for c in df.columns if c in ('date','data','day')), None)
    mentions_col = next((c for c in df.columns if c in ('mentions','liczba_wzmianek')), None)
    tone_col     = next((c for c in df.columns if c in ('tone','sredni_tone','tone_avg')), None)

    if not date_col:
        print("[ERROR] Cannot find date column. Expected: date / data / day")
        sys.exit(1)

    # Build output DataFrame in new format
    out = pd.DataFrame()
    out["date"]    = pd.to_datetime(
        df[date_col].astype(str).str.strip().str[:10],
        format="%Y-%m-%d", errors="coerce"
    ).dt.normalize()
    out["mentions"] = pd.to_numeric(df[mentions_col], errors="coerce") if mentions_col else 0
    out["tone"]     = pd.to_numeric(df[tone_col],     errors="coerce") if tone_col else float('nan')
    out["preset"]   = "balanced"

    # Drop invalid dates
    n_before = len(out)
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    if n_before != len(out):
        print(f"  [WARN] Dropped {n_before - len(out)} rows with invalid dates")

    # Verify parameters match article
    m = out["mentions"].dropna()
    t = out["tone"].dropna()
    print(f"\n[INFO] Converted file stats:")
    print(f"  Rows:     {len(out)}")
    print(f"  Dates:    {out['date'].min()} → {out['date'].max()}")
    print(f"  mentions: mean={m.mean():.1f}  std={m.std():.1f}")
    print(f"  tone:     mean={t.mean():.4f}  std={t.std():.4f}")
    print()
    print(f"  Expected (article): μM=379.0 σM=305.8 μT=-0.912 σT=0.714")
    ok_m = abs(m.mean() - 379.0) < 10
    ok_s = abs(m.std()  - 305.8) < 10
    ok_t = abs(t.mean() - (-0.912)) < 0.02
    ok_ts= abs(t.std()  - 0.714)   < 0.02
    print(f"  μM match: {'✓' if ok_m else '✗'}  σM match: {'✓' if ok_s else '✗'}  "
          f"μT match: {'✓' if ok_t else '✗'}  σT match: {'✓' if ok_ts else '✗'}")

    if not all([ok_m, ok_s, ok_t, ok_ts]):
        print("\n[WARN] Parameters do not match article values exactly.")
        print("  This may be the wrong input file.")
        ans = input("  Continue anyway? [y/N]: ").strip().lower()
        if ans != 'y':
            print("Aborted.")
            sys.exit(0)

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"\n[OK] Saved: {OUT_FILE}")
    print(f"     {len(out)} rows ready for checkpoint resume")
    print()
    print("Next step:")
    print("  python -m bmpi.utils.gdelt_btc_downloader \\")
    print("      --mode full --start 2015-10-01 --end 2026-01-31 \\")
    print("      --preset balanced --workers 4")
    print()
    print("The downloader will detect existing dates and download only missing ones.")
    print(f"  Already done: {len(out)} days")
    all_days = (pd.Timestamp("2026-01-31") - pd.Timestamp("2015-10-01")).days + 1
    print(f"  Target total: ~{all_days} days")
    print(f"  Still to download: ~{all_days - len(out)} days")


if __name__ == "__main__":
    main()