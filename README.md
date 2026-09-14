# nextE Energy Supply Bid Management Tool

Python engine and Streamlit application that replace the Excel model
`Energy_Supply_Portfolio_Tracking_v03` for the nextE retail supply portfolio
(workflow `GW-ESB-01`, task `T12`).

Status: Phase 1 - repository skeleton. No engine logic yet.

## What this repository will contain

| Folder | Purpose | Phase |
|---|---|---|
| `config/` | Parameter, tariff and calendar registers (YAML) and their schema | 3 |
| `esb/` | Engine: grid, importer, imbalance rule book, sources, merit order, P&L, cash flow, guarantees, pricing, scenarios, reporting | 3 |
| `tests/` | Unit, edge and parity suites | 3-4 |
| `data/reference/` | Reference Case parity fixtures (compressed, coded) | 4 |
| `docs/` | Methodology, parameters, data contract, user guide, decisions, open items | 3-6 |
| `app/` | Streamlit application | 5 |
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
pytest
```

Number and date formats in documents follow the Romanian convention (84,50 EUR/MWh; 1.490 MW; dd.mm.yyyy).
Times are CET/CEST unless stated; the engine grid is labelled in EET.

Proprietary - nextE Asset Management SRL. All rights reserved.
