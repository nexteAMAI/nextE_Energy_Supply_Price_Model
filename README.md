# nextE Energy Supply Bid Management Tool

Python engine and Streamlit application that replace the Excel model
`Energy_Supply_Portfolio_Tracking_v03` for the nextE retail supply portfolio
(workflow `GW-ESB-01`, task `T12`).

Status: Phase 6 - deployed. Live application: https://nexte-esb.streamlit.app (Streamlit Community Cloud, `main`, `app.py`, Python 3.12; sign-in gate with credentials held in the cloud secrets). Engine parity-proven against the frozen Reference Case (Phase 3-4: 20.092 of 20.245 cached cells tied at 1e-6, 0 failures, 153 layout cells not applicable; gates G3 and G4 accepted 14.09.2026); twelve-page application (Phase 5); run-time parameters in a versioned scenario file (ruling PSTORE); runs reproducible from a case bundle. Gate G5 (CEO walk-through) runs on the live application. Deployment details: docs/DEPLOYMENT.md.

## Why this exists - the architectural inversion

The Excel model was the calculation; the earlier Python code base (branch `legacy`) was a data
layer that fed it. This repository inverts that: the Python engine is the calculation and the
institutional record (docs/METHODOLOGY.md), Excel becomes an export format, and the Streamlit
application is the only user surface. The inversion is what justifies every constraint below:
exact parity before any enhancement (nothing changes silently), configuration over
hard-coding (every regulatory constant is a register value with its source), blank is never
zero and the sign convention k is never inferred (the importer refuses instead of guessing),
auditable arithmetic (every figure traces to its inputs through keys and workbook row maps),
determinism and reproducibility (the case bundle), and no names in the repository (the
register is coded; names live in scenario files outside it).

## What this repository contains

| Folder | Purpose | Phase |
|---|---|---|
| `config/` | Parameter register (YAML, Reference Case values) and its schema | 3 (done) |
| `esb/` | Engine: grid and importer (Phase 2); imbalance rule book, scenarios, sources, merit order, monthly layer, P&L, guarantees, cash flow, pricing, reporting, engine, parity (Phase 3); assembly of uploads, scenario file, case bundle, exports, labels (Phase 5) | 2-5 (done) |
| `tests/` | Unit, edge, parity and application suites | 3-5 |
| `data/reference/` | Reference Case input series and expected output cells (parquet, coded) | 2-3 (done) |
| `docs/` | Methodology, parameters, data contract, user guide, deployment, decisions, open items | 3-6 (done) |
| `app/` + `app.py` | Streamlit application: sign-in gate, brand layer, session model, twelve pages | 5-6 (done) |
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
pip install -e ".[dev,app]"
pytest                                   # engine, parity and application suites
python tools/parity_report.py out/       # PARITY_REPORT_<ddmmyyyy>.md + parity.json
streamlit run app.py                     # the application (or Start_ESB_App.cmd on the desk machine)
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
