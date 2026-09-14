# nextE Energy Supply Bid Management Tool

Python engine and Streamlit application that replace the Excel model
`Energy_Supply_Portfolio_Tracking_v03` for the nextE retail supply portfolio
(workflow `GW-ESB-01`, task `T12`).

Status: Phase 2 - data layer (grid, upload contract, importer, Reference Case input fixtures). Engine modules follow in Phase 3.

## What this repository will contain

| Folder | Purpose | Phase |
|---|---|---|
| `config/` | Parameter, tariff and calendar registers (YAML) and their schema | 3 |
| `esb/` | Engine: grid and importer (Phase 2, done); imbalance rule book, sources, merit order, P&L, cash flow, guarantees, pricing, scenarios, reporting (Phase 3) | 2-3 |
| `tests/` | Unit, edge and parity suites | 3-4 |
| `data/reference/` | Reference Case input series (parquet, coded) and, from Phase 4, expected values | 2-4 |
| `docs/` | Methodology, parameters, data contract, user guide, decisions, open items | 3-6 |
| `app/` | Streamlit application | 5 |
| `tools/` | Build scripts (fixture extraction) | 2 |
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

Upload templates: `python -c "from esb.importer import build_template; build_template('offtaker_load', 2027, 'TPL_offtaker_load_QH_2027.xlsx')"`
(input classes: offtaker_load, pv_generation, baseload_nomination, wholesale_prices). Contract: docs/DATA_CONTRACT.md.

Number and date formats in documents follow the Romanian convention (84,50 EUR/MWh; 1.490 MW; dd.mm.yyyy).
Times are CET/CEST unless stated; the engine grid is labelled in EET.

Proprietary - nextE Asset Management SRL. All rights reserved.
