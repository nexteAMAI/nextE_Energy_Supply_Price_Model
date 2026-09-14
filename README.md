# nextE Energy Supply Bid Management Tool

Python engine and Streamlit application that replace the Excel model
`Energy_Supply_Portfolio_Tracking_v03` for the nextE retail supply portfolio
(workflow `GW-ESB-01`, task `T12`).

Status: Phase 3 - engine complete (imbalance rule book, sources, merit order, P&L, cash flow, daily ledger, guarantees, pricing, overview). Parity with the frozen Reference Case: 20.092 of 20.245 cached cells tied at 1e-6, 0 failures, 153 layout cells not applicable (`tools/parity_report.py`). Streamlit application follows in Phase 5.

## What this repository will contain

| Folder | Purpose | Phase |
|---|---|---|
| `config/` | Parameter register (YAML, Reference Case values) and its schema | 3 (done) |
| `esb/` | Engine: grid and importer (Phase 2); imbalance rule book, scenarios, sources, merit order, monthly layer, P&L, guarantees, cash flow, pricing, reporting, engine, parity (Phase 3) | 2-3 (done) |
| `tests/` | Unit, edge and parity suites (77 tests) | 3-4 |
| `data/reference/` | Reference Case input series and expected output cells (parquet, coded) | 2-3 (done) |
| `docs/` | Methodology, parameters, data contract, user guide, decisions, open items | 3-6 |
| `app/` | Streamlit application | 5 |
| `tools/` | Build scripts (fixture extraction, parity report) | 2-3 |
| `extractors/` | Market data connectors (reviewed from the `legacy` branch) | 7 |

## Governance

- Exact parity with the frozen Reference Case is required before any enhancement (gate G4).
- The `legacy` branch holds the previous code base unchanged; it is not a port target.
- No counterparty names, credentials or bulk time series are committed to this repository.
- All parameters are dynamic and set by the user in the application; the YAML files carry
  the Reference Case values only.

## Development

```
pip install -e ".[dev]"
pytest                                   # about 65 s; parity suite included
python tools/parity_report.py out/       # PARITY_REPORT_<ddmmyyyy>.md + parity.json
```

Run the engine on the Reference Case:

```
import pandas as pd
from esb.engine import run
r = run(pd.read_parquet("data/reference/rc_v03_series.parquet"))
r.pnl.portfolio.y("nm_forecast")       # net margin, forecast, year
```

Upload templates: `python -c "from esb.importer import build_template; build_template('offtaker_load', 2027, 'TPL_offtaker_load_QH_2027.xlsx')"`
(input classes: offtaker_load, pv_generation, baseload_nomination, wholesale_prices). Contract: docs/DATA_CONTRACT.md.

Number and date formats in documents follow the Romanian convention (84,50 EUR/MWh; 1.490 MW; dd.mm.yyyy).
Times are CET/CEST unless stated; the engine grid is labelled in EET.

Proprietary - nextE Asset Management SRL. All rights reserved.
