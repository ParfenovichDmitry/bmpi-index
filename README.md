# 🚀 BMPI — Bitcoin Media Pressure Index

> A quantitative framework to measure how much Bitcoin market movements are driven by media pressure rather than fundamentals

---

## 📌 Overview

BMPI (Bitcoin Media Pressure Index) is a research-grade pipeline that quantifies **information-driven price movements** in the Bitcoin market using:

* 🌐 GDELT GKG 2.0 (global news dataset)
* 📈 Market data (BTC, ETH, NASDAQ, DXY, Gold)
* 🧠 Statistical modeling (SARIMAX, Ridge, Granger, Johansen)

👉 The goal:

> Detect, quantify and validate media-driven anomalies in BTC market capitalization

---

## 🧠 Key Insight

Markets are not purely fundamental — they are partially driven by:

* narratives
* media cycles
* attention shocks

👉 BMPI isolates this effect.

---

## 📊 Key Results (Pipeline v2)

* 📅 Dataset: **3,800 days (2015–2026)**
* ⚡ Events detected: **29 major BTC peaks**
* 📈 Avg BMPI: **0.4985**

### Media impact

* Media explains: **~29.75% of abnormal moves**
* Excess share: **52.76%**

### Event-level

* Mean impact: **16.86%**
* Max impact: **49.68%**

---

## 📊 BMPI Distribution

| Zone         | Share  | Meaning            |
| ------------ | ------ | ------------------ |
| CALM         | 31.63% | Organic market     |
| NORMAL       | 38.13% | Neutral            |
| ELEVATED     | 22.21% | Narrative building |
| ALERT        | 5.87%  | High risk          |
| MANIPULATION | 2.16%  | Extreme            |

---

## 📉 Correlation Insights

* mentions → BMPI: **r ≈ 0.79**
* mentions → media effect: **r ≈ 0.69**
* tone → weak signal

👉 **Main driver = media volume, not sentiment**

---

## 🧪 Robustness

Across all presets and windows:

* Excess/media: **53% – 60%**
* Excess/abnormal: **9.5% – 16.8%**

👉 Results are **stable and consistent**

---

## 🔬 Methodology

### Pipeline steps

```text
step01  Normalize datasets
step02  Feature engineering
step03  Peak detection
step04  Baseline model (SARIMAX)
step05  Residual extraction
step06  Merge news + market
step07  News effect model
step08  Event-level impact
step09  BMPI calculation
step10  Robustness analysis
step11  Advanced metrics
step12  Cross-preset analysis
step13  Granger causality
step14  Out-of-sample validation
step15  Benchmark comparison
step16  Johansen cointegration
step17  Trading strategy
```

---

## 📈 Statistical Results

### Granger Causality

* BMPI → BTC: **p = 0.015 (significant)**
* BTC → media: **not significant**

👉 Evidence supports **media → market causality**

---

### Cointegration

* Johansen rank: **r = 2**
* VECM α (media): **+0.717**

👉 Strong long-term relationship

---

### Benchmark Comparison

* BMPI vs CFGI: **r = 0.22**

👉 BMPI captures a **different signal space**

---

## 🌐 BMPI Web Calculator

The interactive BMPI calculator is **not included in this repository**.

It is available here:

👉 https://parfenovichdmitry.github.io/bmpi-index_calculator/

### Features

* browser-based BMPI calculation
* no backend required
* built with HTML, CSS, JavaScript
* uses GDELT Doc API v2
* suitable for demos and research validation

---

## 🗂 Project Structure

```text
bmpi-index/

├── bmpi/
│   ├── pipelines/
│   ├── utils/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── tests/
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ParfenovichDmitry/bmpi-index_calculator.git
cd bmpi-index_calculator
pip install -r requirements.txt
```

---

## ▶️ Run Pipeline

```bash
python step01_normalize_datasets.py
python step02_align_and_features.py
...
python step17_trading_strategy.py
```

---

## 📦 Data Availability

Data is **not included** due to size.

All datasets can be reconstructed from:

* GDELT GKG 2.0
* CoinGecko
* FRED

👉 The pipeline is fully reproducible.

---

## 🧠 Scientific Contribution

BMPI introduces:

* a quantitative measure of media-driven price distortion
* separation of fundamental vs informational movement
* integration of econometrics + NLP + market data

---

## 👨‍💻 Author

Dmitry Parfenovich
Polish-Japanese Academy of Information Technology (PJATK)
Warsaw, Poland

---

## 📜 License

MIT License
