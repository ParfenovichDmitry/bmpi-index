# BMPI — Bitcoin Media Pressure Index

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **How much of BTC market capitalisation is driven by media pressure?**
> BMPI quantifies daily anomalous media coverage of Bitcoin using GDELT GKG 2.0
> signals and Ridge regression residuals, then tests whether it Granger-causes
> BTC price movements.

**Author:** Dmitry Parfenovich · Polish-Japanese Academy of Information Technology, Warsaw  
**Stack:** Python 3.10+, pandas, statsmodels, scikit-learn, GDELT GKG 2.0

---

## Key Results

| Metric | Value | Target |
|--------|-------|--------|
| Balanced peaks detected | **29** | 29 ✓ |
| Avg BMPI score | **0.500** | 0.499 ✓ |
| MANIP zone share | **3.5%** | ✓ |
| HHI mean | **0.038** | 0.038 ✓ |
| SRI mean | **0.446** | 0.446 ✓ |
| low_cred ratio | **13.2%** | 13.2% ✓ |
| Robustness range (excess share) | **52.1–52.9%** | 51.2–53.5% ✓ |
| BMPI → BTC price (Granger lag=3) | **p=0.015 \*** | significant ✓ |
| BTC price → media (reverse) | **NOT significant** | unidirectional ✓ |
| BMPI–CFGI correlation | **r=0.22\*\*\*** | low (different constructs) ✓ |
| Johansen cointegration r | **2** | long-run EQ found |
| VECM α (log\_excess) | **+0.717** | fast adjustment |

---

## What is BMPI?

BMPI (Bitcoin Media Pressure Index) is a daily index in [0, 1] that measures
the degree to which Bitcoin media coverage deviates from its expected baseline.
It is constructed from:

1. **GDELT GKG 2.0** — daily mentions and tone of Bitcoin-related news articles
2. **SARIMAX residuals** — unexplained BTC market cap movements after controlling
   for macro factors
3. **Ridge regression** — decomposition of residuals into media-attributable share
4. **Sigmoid normalisation** — mapping the composite z-score to [0, 1]

BMPI zones:

| Zone | Range | Share |
|------|-------|-------|
| CALM | 0.00–0.40 | 34.8% |
| NORMAL | 0.40–0.50 | 33.7% |
| ELEVATED | 0.50–0.60 | 18.4% |
| ALERT | 0.60–0.70 | 9.6% |
| MANIPULATION | 0.70–1.00 | 3.5% |

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
│       │   ├── step04_sarimax_baseline.py         # AIC = -8513.75
│       │   ├── step05_residual_decomposition.py
│       │   ├── step06_gdelt_merge.py
│       │   ├── step07_ridge_regression.py         # R² = 0.013
│       │   ├── step08_event_media_impact.py       # 16.7% news share
│       │   ├── step09_bmpi_classification.py      # avg BMPI = 0.500
│       │   ├── step10_robustness_3x3.py           # 52.1–52.9%
│       │   ├── step11_advanced_metrics.py         # HHI, SRI, low_cred
│       │   ├── step12_cross_preset_analysis.py    # tone→bmpi r=0.91***
│       │   ├── step13_granger_causality.py        # BMPI→BTC p=0.015*
│       │   ├── step14_oos_validation.py           # partial (regime change)
│       │   ├── step15_benchmark_comparison.py     # vs CFGI r=0.22***
│       │   └── step16_johansen_cointegration.py   # r=2, VECM α=0.717
│       └── utils/
│           └── gdelt_btc_downloader.py
├── data/
│   ├── raw/
│   │   └── gdelt/                                 # GDELT signal CSVs
│   └── processed/                                 # pipeline outputs
├── tests/
├── pyproject.toml
└── README.md
```

---

## Pipeline Overview

```
Raw data          GDELT GKG 2.0 + BTC price/mcap
     │
step01  Normalisation & date alignment
step02  Feature engineering (log-returns, z-scores)
step03  Peak detection (balanced: 29 events, threshold z=2.8, gap=21d)
step04  SARIMAX baseline (AIC=-8513.75, controls for macro)
step05  Residual decomposition
step06  GDELT merge (balanced/sensitive/strong presets)
step07  Ridge regression (α=1.0, news-only, TARGET=resid_btc_mcap_usd)
step08  Event-level media impact (mean news_share=16.7%)
step09  BMPI classification (auto-calibrated, avg=0.500, MANIP=3.5%)
step10  Robustness 3×3 grid (52.1–52.9%, paper range 51.2–53.5%)
step11  Advanced metrics: HHI=0.038, SRI=0.446, low_cred=13.2%
step12  Cross-preset analysis (tone→bmpi_score r=0.91***)
step13  Granger causality (BMPI→BTC lag=3 p=0.015*, reverse NOT sig)
step14  OOS validation (partial — 2022+ regime change)
step15  Benchmark vs CFGI (r=0.22***, incremental R²=+0.009)
step16  Johansen cointegration (r=2, VECM α_excess=+0.717)
```

---

## Installation

```bash
git clone https://github.com/ParfenovichDmitry/bmpi-index.git
cd bmpi-index
pip install -e ".[dev]"
```

For visualisations:
```bash
pip install -e ".[viz]"
```

---

## Data Requirements

### GDELT GKG 2.0 signals
Place CSV files in `data/raw/gdelt/`:

```
gdelt_gkg_bitcoin_daily_signal_balanced.csv    # 3777 rows (primary)
gdelt_gkg_bitcoin_daily_signal_sensitive.csv   # 2308 rows
gdelt_gkg_bitcoin_daily_signal_strong.csv      #  991 rows
```

Download using the bundled downloader:
```bash
bmpi-download --preset balanced
bmpi-download --preset sensitive
bmpi-download --preset strong
```

### BTC market data
Place in `data/raw/`:
```
btc_price_daily.csv      # columns: date, btc_price, btc_mcap
```

### Crypto Fear & Greed Index (step15)
Downloaded automatically from Alternative.me API on first run,
then cached at `data/processed/cfgi_daily.csv`.

---

## Running the Pipeline

Run steps sequentially:

```bash
python src/bmpi/pipelines/step01_data_normalisation.py
python src/bmpi/pipelines/step02_feature_engineering.py
# ... continue through step16
python src/bmpi/pipelines/step16_johansen_cointegration.py
```

Each step saves outputs to `data/processed/` and prints a summary table.
All steps produce an HTML report viewable in any browser.

---

## Key Findings

### Granger Causality (step13)
BMPI Granger-causes BTC price at lag=3 days (F=3.49, p=0.015*).
The reverse direction (BTC price → media) is NOT significant,
confirming a **unidirectional** information flow: media → market.

### Long-run Equilibrium (step16)
Johansen test finds r=2 cointegrating vectors between
`log_btc_mcap` and `log_excess_media_usd`.
VECM error correction coefficients:
- `log_btc_mcap`: α=−0.004 (near-exogenous — BTC does not adjust)
- `log_excess`: α=+0.717 (fast adjustment — media effect mean-reverts)

### Robustness (step10)
Results are stable across all 9 preset×window combinations
(anomaly share spread = 0.034 < 0.05 → **STABLE**).

### Benchmark (step15)
BMPI and CFGI measure different constructs (r=0.22***).
BMPI adds incremental R²=+0.009 beyond CFGI for excess_media_usd.
The indices are **complementary**, not redundant.

### OOS Validation (step14)
Partial validation — r degrades from 0.267 (TRAIN) to 0.055 (TEST).
Explained by the 2022+ structural regime change:
Terra/Luna crash (May 2022), FTX collapse (Nov 2022),
spot Bitcoin ETF approval (Jan 2024).

---

## Outputs

After running the full pipeline, `data/processed/` contains:

| File | Description |
|------|-------------|
| `excess_media_effect_daily.csv` | Daily BMPI scores and excess media effect |
| `events_peaks_balanced.csv` | 29 detected price anomaly peaks |
| `granger_results.csv` | Granger causality test results |
| `johansen_results.json` | Johansen cointegration results |
| `benchmark_comparison.csv` | BMPI vs CFGI comparison |
| `oos_validation_results.json` | Out-of-sample validation metrics |
| `cross_preset_correlations.csv` | Cross-preset stability analysis |
| `*_report.html` | HTML reports for each step (open in browser) |

---

## Academic Context

**Research question:** How much of BTC market capitalisation is driven
by media pressure as measured by GDELT GKG 2.0?

**Methodology:**
- SARIMAX residuals isolate price movements not explained by macro factors
- Ridge regression attributes residuals to media signals (news volume, tone)
- Composite BMPI index normalised via sigmoid transformation
- Granger causality and Johansen cointegration for causal inference

**Positioning vs existing literature:**
- CFGI (Alternative.me) measures general market sentiment;
  BMPI measures media-specific pressure (r=0.22, different constructs)
- Unlike social-media sentiment indices, BMPI uses a structured global
  news database (GDELT) with source credibility weighting (HHI)

**Type:** Master's thesis — Polish-Japanese Academy of Information Technology, Warsaw

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