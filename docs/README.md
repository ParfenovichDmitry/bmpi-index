# BMPI — Bitcoin Media Pressure Index

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GDELT GKG 2.0](https://img.shields.io/badge/data-GDELT%20GKG%202.0-orange.svg)](https://www.gdeltproject.org/)

> **How much of BTC market capitalisation is driven by media pressure?**  
> BMPI quantifies daily anomalous media coverage of Bitcoin using GDELT GKG 2.0
> signals and Ridge regression residuals, then tests whether it Granger-causes
> BTC price movements.

**Author:** Dmitry Parfenovich · Polish-Japanese Academy of Information Technology, Warsaw  
**Stack:** Python 3.10+, pandas, statsmodels, scikit-learn, GDELT GKG 2.0

🔗 **Live calculator:** [parfenovichdmitry.github.io/bmpi-index/calculator](https://parfenovichdmitry.github.io/bmpi-index/calculator)

---

## Key Results

| Metric | Value |
|--------|-------|
| Balanced peaks detected | **29** |
| Avg BMPI score | **0.500** |
| MANIP zone share | **3.5%** |
| HHI mean | **0.038** |
| SRI mean | **0.446** |
| Robustness range (excess share) | **52.1–52.9%** |
| BMPI → BTC price (Granger lag=3) | **p=0.015 \*** |
| BTC price → media (reverse) | **NOT significant** |
| BMPI–CFGI correlation | **r=0.22\*\*\*** |
| Johansen cointegration r | **2** |
| VECM α (log\_excess) | **+0.717** |

---

## What is BMPI?

BMPI (Bitcoin Media Pressure Index) is a daily index in [0, 1] that measures
the degree to which Bitcoin media coverage deviates from its expected baseline.

| Zone | Range | Share |
|------|-------|-------|
| CALM | 0.00–0.40 | 34.8% |
| NORMAL | 0.40–0.50 | 33.7% |
| ELEVATED | 0.50–0.60 | 18.4% |
| ALERT | 0.60–0.70 | 9.6% |
| MANIPULATION | 0.70–1.00 | 3.5% |

---

## Live BMPI Calculator

An interactive calculator is available at:  
👉 **[parfenovichdmitry.github.io/bmpi-index/calculator](https://parfenovichdmitry.github.io/bmpi-index/calculator)**

Computes the BMPI score for any date using the GDELT Doc API v2 — no file downloads required.

### How to use

1. Select a date (2015 onwards)
2. Choose filter preset: **BALANCED** (default), SENSITIVE, or STRONG
3. Click **COMPUTE BMPI**
4. Wait 5–15 seconds for the GDELT API query to complete

### ⚠ If no data loads — press Compute again

GDELT Doc API occasionally returns empty results or times out on the first request.
**This is normal behaviour.** Simply press **COMPUTE BMPI** again — the second attempt
almost always succeeds. If the result still shows `0 articles`, try:

- Switching to the **SENSITIVE** preset (broader keyword filter)
- Selecting a different date (very old dates before 2016 may have sparse coverage)
- Refreshing the page and trying again

The API is a free public service with no rate limit guarantees.

### Calibration parameters

The calculator uses the same parameters as the research pipeline:

| Parameter | Value | Description |
|-----------|-------|-------------|
| μM | 379.0 | Mean daily mentions (baseline) |
| σM | 305.8 | Std dev of daily mentions |
| μT | −0.912 | Mean daily tone |
| σT | 0.714 | Std dev of tone |
| w₁ | 0.25 | Weight: media volume |
| w₂ | 0.20 | Weight: sentiment tone |

Formula:
```
z₁ = clip[(mentions − 379.0) / 305.8, −3, 3]
z₂ = clip[(tone − (−0.912)) / 0.714, −3, 3]
raw = 0.25·z₁ + 0.20·z₂
BMPI = σ(raw) = 1 / (1 + e^−raw)
```

---

## Project Structure

```
bmpi-index/
├── src/
│   └── bmpi/
│       ├── pipelines/
│       │   ├── step01_data_normalisation.py
│       │   ├── step02_feature_engineering.py
│       │   ├── step03_peak_detection.py          # 29 balanced peaks
│       │   ├── step04_sarimax_baseline.py
│       │   ├── step05_residual_decomposition.py
│       │   ├── step06_gdelt_merge.py
│       │   ├── step07_ridge_regression.py
│       │   ├── step08_event_media_impact.py
│       │   ├── step09_bmpi_classification.py     # avg BMPI = 0.500
│       │   ├── step10_robustness_3x3.py          # 52.1–52.9%
│       │   ├── step11_advanced_metrics.py        # HHI, SRI
│       │   ├── step12_cross_preset_analysis.py
│       │   ├── step13_granger_causality.py       # BMPI→BTC p=0.015*
│       │   ├── step14_oos_validation.py
│       │   ├── step15_benchmark_comparison.py    # vs CFGI
│       │   └── step16_johansen_cointegration.py  # r=2, VECM
│       └── utils/
│           ├── gdelt_btc_downloader.py
│           └── checkpoint.py
├── data/
│   ├── raw/gdelt/                                # GDELT signal CSVs
│   └── processed/                                # pipeline outputs
├── calculator/
│   └── index.html                                # Live BMPI calculator
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ParfenovichDmitry/bmpi-index.git
cd bmpi-index
pip install -e ".[dev]"
```

---

## Data Requirements

### GDELT GKG 2.0 signals

Place in `data/raw/gdelt/`:
```
gdelt_gkg_bitcoin_daily_signal_balanced.csv    # primary (mean≈379/day)
gdelt_gkg_bitcoin_daily_signal_sensitive.csv
gdelt_gkg_bitcoin_daily_signal_strong.csv
```

Download:
```bash
python -m bmpi.utils.gdelt_btc_downloader --mode full --start 2015-10-01 --end 2026-01-31 --preset balanced --workers 6
```

### BTC market data
```
data/raw/market/btcusdmax.csv     # daily OHLCV
data/raw/market/DTWEXBGS.csv      # USD index (FRED)
```

---

## Running the Pipeline

```bash
python src/bmpi/pipelines/step01_data_normalisation.py
python src/bmpi/pipelines/step02_feature_engineering.py
# ... continue step03 → step16
python src/bmpi/pipelines/step16_johansen_cointegration.py
```

Each step saves outputs to `data/processed/` and produces an HTML report.

---

## Key Findings

**Granger Causality (step13)**  
BMPI Granger-causes BTC price at lag=3 days (F=3.49, p=0.015\*).
Reverse direction (BTC → media) NOT significant → unidirectional flow confirmed.

**Long-run Equilibrium (step16)**  
Johansen test: r=2 cointegrating vectors between `log_btc_mcap` and `log_excess_media_usd`.
VECM: α(log\_excess)=+0.717 (fast mean-reversion), α(log\_btc\_mcap)≈0 (exogenous).

**Robustness (step10)**  
Anomaly share spread = 0.034 < 0.05 → **STABLE** across all 9 preset×window combinations.

**Benchmark (step15)**  
r(BMPI, CFGI) = 0.22\*\*\* → different constructs, complementary indices.

---

## Citation

```bibtex
@misc{parfenovich2026bmpi,
  title   = {Bitcoin Media Pressure Index: Quantifying Anomalous
             Media Coverage as a Predictor of Cryptocurrency Price Anomalies},
  author  = {Parfenovich, Dmitry},
  year    = {2026},
  school  = {Polish-Japanese Academy of Information Technology, Warsaw},
  note    = {Master's thesis}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Data: [GDELT Project](https://www.gdeltproject.org) (open access) ·
[Alternative.me CFGI](https://alternative.me/crypto/fear-and-greed-index/) (open access)
