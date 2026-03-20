# Integrated Energy Supply Extension — Architecture Specification
## Extension to ro-energy-pricing-engine (nextE_Energy_Supply_Price_Model)

### Version: 1.0 | Date: 2026-03-19

---

## 1. Overview

This extension adds integrated energy supply pricing, procurement optimization, portfolio management, and risk analytics to the existing RO Energy Pricing Engine. It extends the Layer 1-2-3 architecture without modifying existing modules.

### New Capabilities

| Module | Purpose | Granularity |
|--------|---------|-------------|
| `supply_pricing` | Blended supply cost calculation with 3 PV purchase mechanisms | 15-min / hourly |
| `procurement_optimizer` | Forward procurement strategy optimization across 4 channels | Quarterly / monthly |
| `supply_portfolio` | Multi-customer portfolio P&L tracking | Per-contract × 15-min |
| `supply_risk` | VaR, shape risk, volume risk, credit risk quantification | Daily / monthly |
| `generation_profiles` | PV generation curve forecasting (multi-probability P10-P90) | 15-min |
| `consumption_profiles` | Customer load curve modeling (multi-probability) | 15-min |

### New Data Sources

| Source | Method | Data | Frequency |
|--------|--------|------|-----------|
| OPCOM (opcom.ro) | Web scraping (requests + BeautifulSoup) | Bilateral contract prices, GC prices, market reports | Daily |
| BRM/Nord Pool (data.nordpoolgroup.com) | REST API | DAM/IDM prices, volumes for TEL area | Hourly |
| Transelectrica DAMAS (newmarkets.transelectrica.ro) | Web scraping | Balancing market reports, BRP positions | Daily |
| Montel EQ | Python API client (existing) | Forward curves, BESS indices, commodity OHLC | Daily |
| ENTSO-E | Python API client (existing) | Generation, load, NTC, outages | Hourly |

---

## 2. Module Architecture

```
ro-energy-pricing-engine/
├── extractors/                     # DATA LAYER
│   ├── [existing clients]          # entsoe, eq, balancing, jao, fx, data_loader
│   ├── opcom_scraper.py            # NEW: OPCOM bilateral prices, GC market, reports
│   ├── nordpool_client.py          # NEW: BRM/NordPool DAM+IDM for TEL area
│   └── damas_client.py             # NEW: Transelectrica DAMAS public reports
│
├── processors/                     # COMPUTATION LAYER
│   ├── [existing processors]       # dam, idm, srmc, merit_order, imbalance, etc.
│   ├── supply_pricing.py           # NEW: Blended supply cost engine
│   ├── procurement_optimizer.py    # NEW: Forward procurement strategy optimization
│   ├── supply_portfolio.py         # NEW: Multi-customer portfolio P&L
│   ├── supply_risk.py              # NEW: Supply-specific risk metrics
│   ├── generation_profiles.py      # NEW: PV generation curve forecasting
│   └── consumption_profiles.py     # NEW: Customer load curve modeling
│
├── streamlit_app/pages/            # PRESENTATION LAYER
│   ├── [existing pages 01-07]
│   ├── 08_supply_pricing.py        # NEW: Supply price builder with waterfall
│   ├── 09_supply_portfolio.py      # NEW: Portfolio dashboard with P&L
│   └── 10_procurement.py           # NEW: Procurement strategy & execution
│
├── config/
│   ├── [existing configs]
│   └── supply_config.yaml          # NEW: Supply-specific parameters
│
├── supply_pipeline.py              # NEW: Supply-specific pipeline orchestrator
└── data/
    └── supply/                     # NEW: Supply-specific data storage
        ├── contracts/              # Customer contract register (JSON/CSV)
        ├── procurement/            # Forward position register
        └── generation/             # PV generation forecasts per asset
```

---

## 3. Core Processor: supply_pricing.py

### 3.1 Supply Price Waterfall

The fundamental calculation engine. For each supply contract:

```
Final Supply Price = Energy Cost + Regulated Costs + Risk Premia + Margin

Where:
  Energy Cost = PV_Share × PV_Cost + NonSolar_Share × Forward_Cost + Residual × Spot_Cost

  Regulated Costs = GC_Quota + Balancing_Cost + Grid_Tariffs(pass-through)

  Risk Premia = Shape_Risk + Volume_Risk + Credit_Risk + Imbalance_Buffer

  Margin = nextE_Target_Margin
```

### 3.2 PV Purchase Pricing Mechanisms

Three mechanisms, selectable per generator contract:

```python
class PVPricingMechanism(Enum):
    FIXED = "fixed"           # X EUR/MWh guaranteed
    DAM_INDEXED = "indexed"   # (1 - discount%) × DAM_hourly_clearing
    HYBRID = "hybrid"         # max(floor, floor + share% × (DAM - floor))

def calculate_pv_cost(
    mechanism: PVPricingMechanism,
    dam_prices_15min: pd.Series,       # 15-min DAM clearing prices
    pv_generation_15min: pd.Series,    # 15-min PV generation (MW)
    params: dict                        # {fixed_price, discount_pct, floor, share_pct}
) -> pd.DataFrame:
    """Returns 15-min PV procurement cost series."""
```

### 3.3 Forward Purchase Cost

```python
def calculate_forward_cost(
    consumption_15min: pd.Series,       # Customer consumption profile (MW)
    pv_generation_15min: pd.Series,     # PV generation covering solar hours (MW)
    forward_prices: dict,               # {Q2: 95.12, Q3: 119.31, ...} EUR/MWh
    procurement_strategy: str,          # 'annual_strip', 'quarterly', 'monthly', 'shaped'
) -> pd.DataFrame:
    """
    Calculates forward procurement cost for non-solar gap.

    Gap = max(0, consumption - pv_generation) for each 15-min interval
    Cost = Gap × applicable forward price for that delivery period
    """
```

### 3.4 Multi-Probability Scenarios

All calculations run across probability distributions:

```python
PROBABILITY_LEVELS = ['P10', 'P25', 'P50', 'P75', 'P90']

def run_multi_scenario(
    generation_scenarios: Dict[str, pd.Series],  # P10..P90 generation curves
    consumption_scenarios: Dict[str, pd.Series],  # P10..P90 consumption curves
    price_scenarios: Dict[str, dict],             # P10..P90 forward curves
    pv_mechanism: PVPricingMechanism,
    pv_params: dict,
    gc_cost: float,
    balancing_cost: float,
    risk_margin: float,
    nexte_margin: float,
) -> Dict[str, pd.DataFrame]:
    """Returns supply P&L for each probability combination."""
```

---

## 4. Core Processor: procurement_optimizer.py

### 4.1 Procurement Channel Selection

For each non-solar MWh gap, the optimizer selects the cheapest procurement channel:

```python
class ProcurementChannel(Enum):
    BRM_FORWARD = "brm_forward"           # Physical quarterly/annual strips
    OPCOM_BILATERAL = "opcom_bilateral"   # PC-OTC / PCCB-NC
    DIRECT_BILATERAL = "direct_bilateral" # Hidroelectrica / Nuclearelectrica
    EEX_FINANCIAL = "eex_financial"       # Financial futures hedge
    SPOT_DAM = "spot_dam"                 # Residual on DAM
    SPOT_IDM = "spot_idm"                 # Residual on IDM

def optimize_procurement(
    gap_profile_15min: pd.Series,         # Non-solar gap (MW) per 15-min
    channels: List[ProcurementChannel],    # Available channels
    channel_prices: Dict[str, float],      # Price per channel
    channel_limits: Dict[str, float],      # Max volume per channel (MW)
    channel_costs: Dict[str, float],       # Transaction costs per channel
    collateral_budget: float,              # Available collateral (EUR)
) -> pd.DataFrame:
    """Returns optimal procurement allocation per channel."""
```

### 4.2 Hedging Strategy

```python
def design_hedging_strategy(
    supply_contracts: List[dict],          # Signed supply contracts
    pv_portfolio: pd.DataFrame,            # PV generation portfolio
    forward_curve: pd.DataFrame,           # Current forward curve (Q/Y)
    risk_limits: dict,                     # Max unhedged, max tenor, etc.
) -> dict:
    """
    Outputs:
    - Required forward volume per delivery period
    - Recommended channel allocation
    - Collateral requirement
    - Residual unhedged exposure
    """
```

---

## 5. Core Processor: supply_portfolio.py

### 5.1 Contract Register

```python
@dataclass
class SupplyContract:
    contract_id: str
    customer_name: str
    annual_volume_gwh: float
    supply_price_eur_mwh: float          # Fixed price to customer
    start_date: date
    end_date: date
    load_profile: str                     # 'flat_baseload', 'industrial_2shift', 'custom'
    pv_source: str                        # Generator or Nofar SPV
    pv_mechanism: PVPricingMechanism
    pv_params: dict
    forward_channel: ProcurementChannel
    forward_price_locked: float
    gc_treatment: str                     # 'quota_pass_through', 'own_generation_offset'
    credit_rating: str
    bank_guarantee_eur: float
    status: str                           # 'active', 'pending', 'expired'
```

### 5.2 Portfolio P&L Engine

```python
def calculate_portfolio_pnl(
    contracts: List[SupplyContract],
    dam_prices_15min: pd.Series,
    pv_generation_15min: Dict[str, pd.Series],  # Per generator
    actual_consumption_15min: Dict[str, pd.Series],  # Per customer
    gc_market_price: float,
    period: Tuple[date, date],
) -> pd.DataFrame:
    """
    Calculates P&L at 15-min granularity for each contract and portfolio total.

    Returns DataFrame with columns:
    - contract_id, timestamp_15min
    - revenue (supply price × consumption)
    - pv_cost (per mechanism)
    - forward_cost
    - gc_cost
    - balancing_cost
    - gross_margin
    - net_margin (after nextE overhead allocation)
    """
```

---

## 6. Core Processor: supply_risk.py

### 6.1 Risk Metrics

```python
def calculate_supply_var(
    portfolio: List[SupplyContract],
    price_scenarios: pd.DataFrame,        # Monte Carlo price paths (1000+)
    generation_scenarios: pd.DataFrame,   # Monte Carlo PV paths
    consumption_scenarios: pd.DataFrame,  # Monte Carlo load paths
    confidence: float = 0.95,
    horizon_days: int = 30,
) -> dict:
    """
    Returns:
    - VaR_95_30d: 95% 30-day Value at Risk (EUR)
    - CVaR_95_30d: Expected Shortfall (EUR)
    - Shape_risk: EUR impact of generation-consumption mismatch
    - Volume_risk: EUR impact of consumption deviation
    - Price_risk: EUR impact of forward curve movement
    - Credit_risk: EUR exposure by counterparty
    """

def calculate_shape_risk(
    pv_profile_15min: pd.Series,
    consumption_profile_15min: pd.Series,
    forward_price: float,
    dam_price_distribution: pd.Series,
) -> float:
    """Quantifies the cost of hourly mismatch between PV and consumption."""
```

---

## 7. New Extractors

### 7.1 opcom_scraper.py

```python
class OPCOMScraper:
    """Scrapes public data from opcom.ro"""

    BASE_URL = "https://www.opcom.ro"

    def get_bilateral_prices(self, date_from, date_to) -> pd.DataFrame:
        """Scrape PCCB-NC/PC-OTC transaction summaries."""

    def get_gc_market_prices(self, date_from, date_to) -> pd.DataFrame:
        """Scrape Green Certificate market clearing prices."""

    def get_monthly_report(self, year, month) -> dict:
        """Parse OPCOM monthly market report (PDF)."""
```

### 7.2 nordpool_client.py

```python
class NordPoolClient:
    """Fetches BRM/NordPool data for Romania (TEL delivery area)."""

    BASE_URL = "https://data.nordpoolgroup.com"

    def get_dam_prices(self, delivery_date='latest', area='TEL') -> pd.DataFrame:
        """Fetch DAM clearing prices for Romania."""
        # Endpoint: /auction/day-ahead/prices

    def get_idm_statistics(self, delivery_date='latest', area='TEL') -> pd.DataFrame:
        """Fetch IDM hourly statistics."""
        # Endpoint: /intraday/intraday-hourly-statistics
```

### 7.3 damas_client.py

```python
class DAMASClient:
    """Scrapes Transelectrica DAMAS public reports."""

    BASE_URL = "https://newmarkets.transelectrica.ro"

    def get_balancing_results(self, date) -> pd.DataFrame:
        """Fetch daily balancing market activation results."""

    def get_system_imbalance(self, date) -> pd.DataFrame:
        """Fetch 15-min system imbalance data."""
```

---

## 8. Streamlit Pages

### 08_supply_pricing.py
- **Supply Price Builder**: Input customer volume, load profile, PV source, procurement channel
- **Waterfall Chart**: Visual cost decomposition (PV → Forward → GC → Balancing → Risk → Margin → Price)
- **Sensitivity Table**: Tornado chart for key variables
- **Scenario Comparison**: Base/Upside/Downside/Stress
- **Export**: Generate offer sheet (Excel) for customer

### 09_supply_portfolio.py
- **Portfolio Dashboard**: All active contracts with real-time P&L
- **Heat Map**: Margin by contract × month
- **Risk Dashboard**: VaR, shape risk, credit exposure
- **Cash Flow Tracker**: Working capital and collateral positions

### 10_procurement.py
- **Forward Position Monitor**: Current hedged vs. unhedged volumes
- **Channel Allocation**: BRM / OPCOM / Direct / EEX positions
- **Forward Curve Tracker**: Live forward prices with historical overlay
- **Execution Alerts**: Gate closures, margin calls, position limits

---

## 9. Configuration: supply_config.yaml

```yaml
supply:
  gc_quota_coefficient: 0.499387    # CV/MWh (ANRE Order 81/2025)
  gc_unit_price_ron: 146.2532       # RON/CV (reference)
  gc_cost_eur_mwh: 14.50            # Derived EUR/MWh at current FX
  balancing_cost_eur_mwh: 3.00      # Fixed balancing cost per MWh
  default_risk_margin_eur_mwh: 5.00
  default_nexte_margin_eur_mwh: 12.00
  minimum_margin_eur_mwh: 8.00      # Walk-away threshold
  vat_rate: 0.19                    # Romanian VAT

  pv_pricing:
    default_mechanism: hybrid
    hybrid_floor_eur_mwh: 40.00
    hybrid_share_pct: 0.70
    fixed_default_eur_mwh: 50.00
    indexed_discount_pct: 0.10

  procurement:
    channels:
      brm_forward:
        enabled: true
        max_tenor_months: 24
        min_volume_mw: 1.0
        commission_eur_mwh: 0.03    # BRM trading fee (0.125 RON ≈ 0.025 EUR)
      opcom_bilateral:
        enabled: true
        max_tenor_months: 12
      direct_bilateral:
        enabled: true
        counterparties: [hidroelectrica, nuclearelectrica]
      eex_financial:
        enabled: false              # Requires EEX membership
        margin_pct: 0.10

  risk:
    max_unhedged_gwh: 0.0           # Zero tolerance for unhedged supply
    max_single_customer_gwh: 50.0
    max_counterparty_exposure_eur: 2000000
    var_confidence: 0.95
    var_horizon_days: 30
    monte_carlo_paths: 10000

  solar_hours:
    summer: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # CET, Apr-Sep
    winter: [9, 10, 11, 12, 13, 14, 15]                   # CET, Oct-Mar

  forward_curve_source: montel_eq   # or 'brm', 'eex'
```

---

## 10. Data Flow

```
[Live Sources]                    [Processed]              [Outputs]
OPCOM scraper     ─┐
NordPool API      ─┤
DAMAS scraper     ─┤              supply_pricing.py  ──→  supply_price_waterfall.csv
Montel EQ API     ─┤──→ 15-min ──→ procurement_opt.py ──→  procurement_position.csv
ENTSO-E API       ─┤   aligned    supply_portfolio.py ──→  portfolio_pnl.csv
BAL_SERV API      ─┤              supply_risk.py     ──→  risk_metrics.json
Customer profiles ─┤                                       ↓
PV gen forecasts  ─┘                                  Streamlit pages 08-10
```

---

## 11. Implementation Priority

| Phase | Module | Effort | Business Impact |
|-------|--------|--------|-----------------|
| **Phase 1** | supply_pricing.py + supply_config.yaml | 2-3 days | Core pricing engine — enables supply offers |
| **Phase 2** | 08_supply_pricing.py (Streamlit) | 1-2 days | Visual price builder for commercial team |
| **Phase 3** | nordpool_client.py + opcom_scraper.py | 2-3 days | Live data feeds |
| **Phase 4** | procurement_optimizer.py | 2-3 days | Procurement strategy optimization |
| **Phase 5** | supply_portfolio.py + 09_supply_portfolio.py | 3-4 days | Portfolio P&L tracking |
| **Phase 6** | supply_risk.py | 2-3 days | Risk quantification |
| **Phase 7** | generation_profiles.py + consumption_profiles.py | 3-4 days | Multi-probability forecasting |
| **Phase 8** | damas_client.py + integration | 1-2 days | DAMAS data integration |

**Total estimated effort: 16-24 development days**

Phase 1-2 should be delivered first — they enable the commercial team to generate supply offers immediately.
