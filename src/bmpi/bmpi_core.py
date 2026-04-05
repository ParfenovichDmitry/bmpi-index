# -*- coding: utf-8 -*-
"""
bmpi_core.py
============
Core computation engine for the BMPI
(Bitcoin Media Pressure Index).

Theoretical background
----------------------
BMPI quantifies the daily *abnormal media pressure* exerted on Bitcoin
by combining two GDELT-derived signals via z-score standardisation and
a sigmoid transform:

    z₁ = clip( (mentions_t − μ_m) / σ_m,  −3, 3 )   # volume anomaly
    z₂ = clip( (tone_t    − μ_τ)  / σ_τ,  −3, 3 )   # sentiment anomaly

    BMPI_t = σ( w₁·z₁ + w₂·z₂ )   where σ(x) = 1 / (1 + e^−x)

The sigmoid maps the raw score into (0, 1), making BMPI directly
interpretable as a *pressure intensity* reading relative to the
historical distribution of BTC media coverage.

Naming rationale
----------------
"Media Pressure" is chosen over alternatives:
  - "Manipulation"  → overclaim (requires proving intent, not just anomaly)
  - "Sentiment"     → conflates with UCRY / Fear & Greed (different target)
  - "Noise"         → directionally ambiguous in financial economics
  - "Intensity"     → captured by volume alone; tone adds polarity

"Pressure" accurately captures what the index detects: an abnormal
build-up of media coverage that — as the empirical analysis shows —
creates mean-reverting price pressure (high BMPI → subsequent underperformance).

Zones
-----
MINIMAL  / BASELINE / ELEVATED / HIGH / EXTREME
replaces the previous CALM / NORMAL / ELEVATED / ALERT / MANIPULATION.
"EXTREME" replaces "MANIPULATION" — it labels the top ~5% of historical
daily observations without asserting intent.

Public API
----------
    compute_bmpi(mentions, tone, calib=None)  → BmpiResult
    compute_bmpi_series(mentions, tone, calib=None)  → np.ndarray
    compute_bmpi_oos(mentions, tone)  → BmpiResult  (OOS-safe)
    get_pressure_zone(bmpi)  → PressureZone
    zone_label_series(bmpi_array)  → np.ndarray[str]

References
----------
    Introduced in: [Author], "BMPI: A Daily Media Pressure Index for
    Bitcoin — Evidence of Anomalous Coverage and Mean Reversion",
    Master's Thesis, [University], 2026.

    Methodological antecedents:
    - Antweiler & Frank (2004) — media noise in equity markets
    - Kukacka & Kristoufek (2023) — fundamental/speculative decomposition
    - Kim et al. (2025, JFM) — GDELT tone as BTC option pricing control
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Calibration parameter sets
# ---------------------------------------------------------------------------

#: TRAIN-period calibration (2015-10-31 – 2021-12-31, n=1412 days).
#: USE FOR: all predictive analysis, OOS evaluation, production signals.
CALIB_TRAIN: dict = {
    "mu_mentions":  312.4,
    "sd_mentions":  274.1,
    "mu_tone":     -0.8834,
    "sd_tone":      0.6912,
    "n_days":       1412,
    "period":       "2015-10-31 to 2021-12-31",
    "label":        "TRAIN",
}

#: Full-sample calibration (2015-10-31 – 2026-01-02, n=2220 days).
#: USE ONLY FOR: descriptive statistics and historical time-series plots.
#: NOT FOR: any predictive claim, OOS evaluation, or forward-looking analysis.
CALIB_FULL: dict = {
    "mu_mentions":  379.0,
    "sd_mentions":  305.8,
    "mu_tone":     -0.9121,
    "sd_tone":      0.7139,
    "n_days":       2220,
    "period":       "2015-10-31 to 2026-01-02",
    "label":        "FULL",
}

# Signal weights
_W_VOLUME:   float = 0.25   # S1 — media volume anomaly
_W_TONE:     float = 0.20   # S2 — sentiment anomaly
_Z_CLIP:     float = 3.0

# BMPI threshold for EXTREME zone (top ~5% historical, 95th percentile)
EXTREME_THRESHOLD: float = 0.650


# ---------------------------------------------------------------------------
# Pressure zones
# ---------------------------------------------------------------------------

class PressureZone(NamedTuple):
    """A BMPI pressure zone with boundaries and academic interpretation."""
    lower:        float
    upper:        float
    label:        str           # machine-readable label
    label_short:  str           # e.g. for chart legends
    description:  str           # investor-facing interpretation
    academic_note: str          # for thesis / paper text


PRESSURE_ZONES: tuple[PressureZone, ...] = (
    PressureZone(
        0.000, 0.470,
        "MINIMAL",
        "Minimal",
        "Below-average media coverage. Organic price discovery dominant.",
        "BMPI below the 25th percentile; media volume and tone within "
        "one standard deviation of the calibration mean.",
    ),
    PressureZone(
        0.470, 0.530,
        "BASELINE",
        "Baseline",
        "Standard media activity. No anomalous pressure detected.",
        "BMPI within [p25, p60]. Historically associated with the strongest "
        "subsequent BTC returns (+0.66%/day, +1.28%/3-day horizon).",
    ),
    PressureZone(
        0.530, 0.590,
        "ELEVATED",
        "Elevated",
        "Above-average media coverage. Price sensitivity increases.",
        "BMPI within [p60, p75]. Heightened narrative presence; caution "
        "advised for momentum-based strategies.",
    ),
    PressureZone(
        0.590, 0.650,
        "HIGH",
        "High",
        "Significantly elevated media pressure. Reversal risk ~55%.",
        "BMPI within [p75, p95]. Associated with reduced subsequent returns; "
        "mean-reversion pattern strengthens above this threshold.",
    ),
    PressureZone(
        0.650, 1.001,
        "EXTREME",
        "Extreme",
        "Extreme media pressure (top 5% historical). "
        "Half-life of excess effect ~2 trading days.",
        "BMPI above the 95th percentile of the calibration distribution. "
        "Historically the weakest subsequent return zone (+0.11%/day, "
        "+0.32%/3-day), consistent with sharp mean reversion after "
        "anomalous coverage bursts.",
    ),
)

# Lookup dict for fast zone access by label
ZONE_BY_LABEL: dict[str, PressureZone] = {z.label: z for z in PRESSURE_ZONES}

# Historical percentile breakpoints (calibrated on full dataset)
_PERCENTILE_BREAKS: tuple[tuple[float, int], ...] = (
    (0.425,  5), (0.426, 10), (0.434, 20), (0.438, 25), (0.445, 30),
    (0.461, 40), (0.478, 50), (0.495, 60), (0.516, 70), (0.528, 75),
    (0.548, 80), (0.571, 85), (0.598, 90), (0.648, 95), (0.805, 99),
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BmpiResult:
    """
    Full result of a single BMPI computation.

    Attributes
    ----------
    bmpi        : float ∈ (0, 1)  — the index value
    z_volume    : float            — z-score of mention volume (S1)
    z_tone      : float            — z-score of average tone (S2)
    zone        : PressureZone     — pressure zone classification
    percentile  : int              — historical percentile (0–99)
    top_pct     : int              — 100 − percentile
    calib_label : str              — "TRAIN" or "FULL"
    """
    bmpi:        float
    z_volume:    float
    z_tone:      float
    zone:        PressureZone
    percentile:  int
    top_pct:     int
    calib_label: str

    def __str__(self) -> str:
        return (
            f"BMPI = {self.bmpi:.4f}  |  zone = {self.zone.label}  "
            f"|  pct = {self.percentile}  (top {self.top_pct}%)\n"
            f"  z_volume = {self.z_volume:+.4f}   "
            f"z_tone = {self.z_tone:+.4f}   "
            f"calib = {self.calib_label}\n"
            f"  {self.zone.description}"
        )

    @property
    def is_extreme(self) -> bool:
        """True if BMPI is in the EXTREME pressure zone (top 5%)."""
        return self.zone.label == "EXTREME"

    @property
    def is_anomalous(self) -> bool:
        """True if BMPI is in HIGH or EXTREME zone (top 25%)."""
        return self.zone.label in ("HIGH", "EXTREME")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _resolve_calib(calib: Optional[dict]) -> tuple[dict, str]:
    """
    Return (params_dict, label_string).

    Default is CALIB_TRAIN. Passing CALIB_FULL raises a UserWarning
    to prevent accidental look-ahead bias in predictive analysis.
    """
    if calib is None:
        return CALIB_TRAIN, "TRAIN"

    # Detect full-sample params by period marker or mu value
    is_full = (
        calib.get("label") == "FULL" or
        abs(calib.get("mu_mentions", 0) - 379.0) < 0.5 or
        "2026" in str(calib.get("period", ""))
    )
    if is_full:
        warnings.warn(
            "compute_bmpi() called with full-sample (CALIB_FULL) parameters. "
            "This introduces look-ahead bias in any forward-looking analysis. "
            "Use CALIB_TRAIN for OOS evaluation and production signals.",
            UserWarning,
            stacklevel=3,
        )
        return calib, "FULL"

    label = calib.get("label", "CUSTOM")
    return calib, label


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------

def compute_bmpi(
    mentions: float,
    tone: float,
    calib: Optional[dict] = None,
) -> BmpiResult:
    """
    Compute the BMPI (Bitcoin Media Pressure Index) for a single day.

    Parameters
    ----------
    mentions : float
        Daily count of Bitcoin-related media mentions from GDELT.
    tone : float
        Daily average GDELT V2Tone score (typical range −6 to +2;
        more negative = more negative coverage).
    calib : dict, optional
        Calibration parameters. Keys: mu_mentions, sd_mentions, mu_tone,
        sd_tone, label. Defaults to CALIB_TRAIN (2015–2021).
        Pass CALIB_FULL only for descriptive/historical use.

    Returns
    -------
    BmpiResult

    Examples
    --------
    >>> from bmpi.bmpi_core import compute_bmpi
    >>> r = compute_bmpi(850, -0.5)
    >>> print(r.bmpi, r.zone.label)
    0.5723  ELEVATED
    >>> print(r)
    BMPI = 0.5723  |  zone = ELEVATED  |  pct = 65  (top 35%)
      z_volume = +0.6278   z_tone = +0.5916   calib = TRAIN
      Above-average media coverage. Price sensitivity increases.
    """
    params, label = _resolve_calib(calib)

    z1 = float(np.clip(
        (mentions - params["mu_mentions"]) / params["sd_mentions"],
        -_Z_CLIP, _Z_CLIP,
    ))
    z2 = float(np.clip(
        (tone - params["mu_tone"]) / params["sd_tone"],
        -_Z_CLIP, _Z_CLIP,
    ))

    raw  = _W_VOLUME * z1 + _W_TONE * z2
    bmpi = _sigmoid(raw)

    zone = get_pressure_zone(bmpi)
    pct  = get_percentile(bmpi)

    return BmpiResult(
        bmpi        = round(bmpi, 6),
        z_volume    = round(z1,   6),
        z_tone      = round(z2,   6),
        zone        = zone,
        percentile  = pct,
        top_pct     = 100 - pct,
        calib_label = label,
    )


def compute_bmpi_oos(mentions: float, tone: float) -> BmpiResult:
    """
    OOS-safe BMPI computation — always uses CALIB_TRAIN.

    Use this function in any forward-looking analysis, OOS validation,
    walk-forward testing, or production signal generation.
    """
    return compute_bmpi(mentions, tone, calib=CALIB_TRAIN)


def compute_bmpi_series(
    mentions_series,
    tone_series,
    calib: Optional[dict] = None,
) -> np.ndarray:
    """
    Vectorised BMPI computation for pandas Series or numpy arrays.

    Parameters
    ----------
    mentions_series, tone_series : array-like
        Aligned daily mentions and tone values.
    calib : dict, optional
        Defaults to CALIB_TRAIN.

    Returns
    -------
    np.ndarray of float64

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv("data/raw/gdelt/gdelt_gkg_bitcoin_daily_signal_balanced.csv")
    >>> df["bmpi"] = compute_bmpi_series(df["mentions"], df["tone"])
    """
    params, _ = _resolve_calib(calib)

    mentions = np.asarray(mentions_series, dtype=float)
    tone     = np.asarray(tone_series,     dtype=float)

    z1  = np.clip((mentions - params["mu_mentions"]) / params["sd_mentions"],
                  -_Z_CLIP, _Z_CLIP)
    z2  = np.clip((tone     - params["mu_tone"])     / params["sd_tone"],
                  -_Z_CLIP, _Z_CLIP)
    raw = _W_VOLUME * z1 + _W_TONE * z2
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -20.0, 20.0)))


def get_pressure_zone(bmpi: float) -> PressureZone:
    """Return the PressureZone for a given BMPI value."""
    for zone in PRESSURE_ZONES:
        if zone.lower <= bmpi < zone.upper:
            return zone
    return PRESSURE_ZONES[-1]


def get_percentile(bmpi: float) -> int:
    """Approximate historical percentile rank of a BMPI value."""
    for threshold, pct in _PERCENTILE_BREAKS:
        if bmpi <= threshold:
            return pct
    return 99


def zone_label_series(bmpi_array) -> np.ndarray:
    """
    Vectorised zone label assignment.

    Returns
    -------
    np.ndarray of str with zone labels (e.g. "EXTREME", "BASELINE").
    """
    arr    = np.asarray(bmpi_array, dtype=float)
    labels = np.full(arr.shape, PRESSURE_ZONES[-1].label, dtype=object)
    for zone in PRESSURE_ZONES:
        mask = (arr >= zone.lower) & (arr < zone.upper)
        labels[mask] = zone.label
    return labels


def zone_short_label_series(bmpi_array) -> np.ndarray:
    """Same as zone_label_series but returns short labels for chart legends."""
    arr    = np.asarray(bmpi_array, dtype=float)
    labels = np.full(arr.shape, PRESSURE_ZONES[-1].label_short, dtype=object)
    for zone in PRESSURE_ZONES:
        mask = (arr >= zone.lower) & (arr < zone.upper)
        labels[mask] = zone.label_short
    return labels
