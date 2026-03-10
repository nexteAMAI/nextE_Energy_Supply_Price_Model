# RO Energy Supply Pricing Model — Layer 1: Python Data Engine

**Version:** 2.0 | **Architecture:** Three-Layer (Python Engine → Excel Workbook → Streamlit Dashboard)

## Overview

Layer 1 of the Romanian Energy Supply Pricing Model. Handles API data extraction, heavy computation (SRMC, merit order, statistics), and exports clean datasets for Layer 2 (Excel) and Layer 3 (Streamlit).

## Repository Structure

```
ro-energy-pricing-engine/
├── config/                     # Configuration files
│   ├── assumptions.yaml        # Model parameters (fuel efficiency, tariffs, scenarios)
│   ├── curves.yaml             # EQ curve name registry
│   ├── datasets.yaml           # Pre-loaded file parsing metadata
│   ├── schedule.yaml           # Refresh schedule
│   ├── settings.py             # Central config loader (singleton)
│   └── api_keys.env.template   # API key template (copy to api_keys.env)
├── extractors/                 # API connectors + data loaders
│   ├── data_loader.py          # Unified loader for all pre-loaded CSV datasets
│   ├── eq_client.py            # Energy Quantified (Montel) API
│   ├── entsoe_client.py        # ENTSO-E Transparency Platform API
│   ├── balancing_client.py     # Balancing Services API
│   ├── jao_client.py           # Joint Allocation Office API
│   └── fx_client.py            # BNR/ECB FX rate extraction
├── processors/                 # Analytical computation modules
│   ├── dam_analysis.py         # DAM: Base/Peak/Off-Peak, monthly avg, percentiles
│   ├── idm_analysis.py         # IDM: VWAP, IDM-DAM spread, SIDC flows
│   ├── srmc.py                 # Gas-SRMC, Coal-SRMC, clean spark/dark spread
│   ├── merit_order.py          # Residual demand, price regime, generation mix
│   ├── imbalance.py            # Imbalance cost P50/P90, Long/Short spread
│   ├── forward_curve.py        # Forward curve, contango/backwardation, Aurora comparison
│   ├── sensitivity.py          # Price elasticity, tornado chart inputs
│   └── statistics.py           # Volatility, correlations, VaR/CVaR
├── outputs/                    # Export modules
│   ├── excel_export.py         # CSV exports for Layer 2 (Excel)
│   ├── streamlit_data.py       # JSON/Parquet exports for Layer 3 (Streamlit)
│   └── validation.py           # Cross-check protocol (Section 14.1)
├── tests/                      # Test suite (pytest)
├── streamlit_app/              # Layer 3 dashboard (7 pages)
├── data/                       # Data storage (gitignored)
│   ├── raw/                    # Raw API extractions
│   ├── processed/              # Clean outputs for L2 + L3
│   └── static/                 # Aurora forecast, tariff reference
├── .github/workflows/          # CI/CD: daily + weekly refresh
├── pipeline.py                 # Main orchestrator
├── Makefile                    # Convenience targets
└── requirements.txt            # Python dependencies
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/nexteAMAI/nextE_Energy_Supply_Price_Model.git
cd nextE_Energy_Supply_Price_Model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp config/api_keys.env.template config/api_keys.env
# Edit config/api_keys.env with actual credentials

# 4. Run backtest pipeline (uses pre-loaded CSVs)
make backtest

# 5. Run tests
make test

# 6. Launch Streamlit dashboard
make dashboard
```

## Data Sources

| Source | API | Coverage | Modules |
|--------|-----|----------|---------|
| Energy Quantified (Montel) | Python client | Spot, futures, fundamentals, forecasts | M1, M2, M3, M5, M6 |
| ENTSO-E Transparency | entsoe-py | Generation, load, prices, imbalance (2015+) | M1, M3, M5 |
| Balancing Services | REST API | Imbalance + balancing (Jul 2024+, 15-min) | M5 |
| JAO | REST API | Cross-border capacity, FBMC | M1, M3 |
| BNR / ECB | XML feeds | FX rates (EUR/RON, USD/EUR) | M2 |

## Eight Model Modules

| # | Module | Layer 1 Role |
|---|--------|-------------|
| M1 | Wholesale Procurement | DAM/IDM/forward price processing |
| M2 | Fuel & Commodities | SRMC calculation from TTF, coal, EUA |
| M3 | Generation Stack | Merit order, residual demand |
| M4 | Grid Tariffs | Reads from Layer 2 (manual) |
| M5 | Balancing & Imbalance | P50/P90 imbalance cost |
| M6 | Risk Premium | Sensitivity analysis, volatility |
| M7 | Final Price Assembly | Owned by Layer 2 |
| M8 | Contract Register | Owned by Layer 2 |

## Scheduled Refresh

| Schedule | Time | Content |
|----------|------|---------|
| Daily | 06:00 EET | DAM prices, generation, load, imbalance, FX |
| Weekly | Sunday 08:00 EET | Commodity futures, forward curve, SRMC |

## License

Proprietary — NEXTE Energy. All rights reserved.
