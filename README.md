# COVID-19 Weekly Case Forecasting in India

**State-level spatio-temporal disease forecasting with multi-source data integration**

## Objective

Develop an analytical workflow to forecast weekly COVID-19 confirmed cases across 33 Indian states/UTs, integrating epidemiological case data, Google Community Mobility Reports, and Census 2011 demographics.

## Quick Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Disease | COVID-19 | Richest data availability; strong mobility–transmission link |
| Spatial scale | State/UT (33 units) | Balances granularity with data completeness |
| Temporal scale | Weekly (ISO weeks) | Smooths daily reporting noise |
| Forecast horizon | 4 weeks ahead | Actionable for public health planning |
| Best model | **Gradient Boosting (R²=0.79)** | Outperforms decomposition and neural network baselines |

## Data Sources

1. **COVID-19 Case Data** — [Kaggle](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india) / [covid19india.org](https://data.covid19india.org/)
2. **Google Community Mobility Reports** — [Google](https://www.google.com/covid19/mobility/)
3. **Census of India 2011** — Population, density, urbanisation, literacy (embedded in code)

## Project Structure

```
covid19-forecasting-india/
├── README.md
├── requirements.txt
├── run_all.sh                      # One-command full pipeline
├── data/
│   ├── raw/                        # Raw CSVs (downloaded or simulated)
│   └── processed/                  # Cleaned weekly panel + ML features
├── src/
│   ├── utils.py                    # Shared paths, metrics, state data
│   ├── 01_download_data.py         # Data download / simulation
│   ├── 02_preprocess.py            # Cleaning → weekly aggregation → merge
│   ├── 03_eda.py                   # Exploratory analysis & plots
│   ├── 04_feature_engineering.py   # 31 features: lags, rolling, mobility, demographics
│   ├── 05_model_xgboost.py        # Gradient Boosting with CV grid search
│   ├── 06_model_prophet.py        # STL decomposition + Ridge (Prophet fallback)
│   ├── 07_model_lstm.py           # MLP sliding-window neural network
│   ├── 08_evaluate_compare.py     # Cross-model comparison
│   └── generate_report.py         # PDF report generator
├── outputs/
│   ├── figures/                    # All generated plots
│   └── models/                     # Saved models + metrics JSON
└── report/
    └── report.pdf                  # 2-page analysis report
```

## Setup & Run

```bash
# 1. Clone and enter
git clone https://github.com/<your-username>/covid19-forecasting-india.git
cd covid19-forecasting-india

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (uses simulated data)
bash run_all.sh

# OR run step-by-step:
python src/01_download_data.py --simulate
python src/02_preprocess.py
python src/03_eda.py
python src/04_feature_engineering.py
python src/05_model_xgboost.py
python src/06_model_prophet.py
python src/07_model_lstm.py
python src/08_evaluate_compare.py
python src/generate_report.py
```

### Using Real Data

1. Download `covid_19_india.csv` from [Kaggle](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india) → `data/raw/`
2. Download Google Mobility CSV from [Google](https://www.google.com/covid19/mobility/) → `data/raw/` (as `Global_Mobility_Report.csv` or India-filtered as `mobility_india.csv`)
3. Run without `--simulate`: `python src/01_download_data.py` then continue from step 2

## Features Engineered (31 total)

- **Lags** (4): weekly cases at t-1, t-2, t-3, t-4
- **Rolling stats** (8): 4-week and 8-week rolling mean, std, min, max
- **Epidemiological** (5): growth rate, Rt proxy, CFR, recovery rate, active cases
- **Temporal** (4): week sin/cos, month, weeks since start
- **Mobility** (6): retail, grocery, parks, transit, workplaces, residential
- **Demographic** (4): log population, log density, urbanisation %, literacy %

## Results

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|----|
| **Gradient Boosting** | **415,486** | **223,192** | **51.5%** | **0.790** |
| STL + Ridge | 676,741 | 535,680 | 126.7% | 0.684 |
| MLP Neural Network | 1,030,060 | 607,636 | 612.3% | -0.289 |

Gradient Boosting is the clear winner. Lag features (lag-1, lag-2) and rolling statistics dominate importance, followed by mobility and demographic features.

## Key Findings

1. **Autoregressive structure dominates**: lag-1 and rolling mean are the most predictive features
2. **Mobility matters**: transit and workplace mobility changes correlate with case dynamics
3. **Demographics help generalisation**: population density and urbanisation improve cross-state predictions
4. **Delta wave is hardest**: rapid acceleration in Apr–May 2021 challenges all models

## Limitations

- Simulated data used for demonstration; validate with real datasets
- Under-reporting and testing rate variability not modelled
- No vaccination data incorporated
- Multi-step forecast errors accumulate through autoregressive features

## Requirements

- Python ≥ 3.9
- pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, reportlab
- Optional: xgboost, prophet, tensorflow (code auto-detects and uses fallbacks)
