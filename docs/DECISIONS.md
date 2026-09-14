# DECISIONS

Decision numbers continue the D-series of the Reference Case workbook's `_CLAUDE_LOG`
(D01-D86 in the workbook; D87/D88 are the `Pricing_Calc` corrections applied in code, see
OPEN_ITEMS.md). Gate rulings are referenced by their G-ids from the Phase 0 specification.

| ID | Date | Decision | Basis |
|---|---|---|---|
| G0-D2 | 14.09.2026 | The engine grid is positional (96 rows per day, EET labels, fixed 1 h offset to CET); daylight-saving rules are applied at import; the Reference Case series are frozen as they sit | CEO ruling on X-02 |
| G0-D4 | 14.09.2026 | Numerical tolerance `abs(py - xl) <= 1e-6 x max(abs(xl), 1)` | CEO ruling |
| G0-D6 | 14.09.2026 | Fixed evaluation order with an acyclicity assertion; no convergence iteration | CEO ruling on X-14 |
| G0-D7 | 14.09.2026 | Reference Case keeps the placeholder off-takers (positions 3 and 4) Active | CEO ruling |
| G0-F035 | 14.09.2026 | No counterparty names in the repository; entities are coded (`OT1`.. by merit-order position, `PV1`); names live in the application and in Excel files | CEO ruling |
| G0-PAR | 14.09.2026 | All parameters, including off-taker names and count, are dynamic and set in the application; YAML registers carry Reference Case values only | CEO ruling |
| G0-TS | 14.09.2026 | All time series are uploaded through the application against the standard template of docs/DATA_CONTRACT.md | CEO ruling |
| D89 | 14.09.2026 | The 35.136-row workbook container is not reproduced: the engine grid has 365 x 96 rows in a common year (366 x 96 in a leap year); placeholder rows (29.02 in v03, `Outside` rows in the STD template) are dropped | every workbook aggregation is date-keyed; the two containers place the placeholder block differently (X-29) |
| D90 | 14.09.2026 | Upload contract carries `time_basis` (`local_clock` / `fixed_96`) in `Std_Control` and `k` per volume slot in `Series_Registry`; both are declared, never inferred | SPEC section 4.3, X-24 |
| D91-ESB | 14.09.2026 | Fixture workbooks are written by openpyxl (16 significant digits); the parquet built from the workbook's stored text is the canonical fixture | build_rc_fixtures.py |
| D87 | 14.09.2026 | `Pricing_Calc!C46` (forecast premium) is the off-taker's total forecast premium; the workbook's +94 row offset (Collateral component) is a defect reproduced only under `pricing_as_cached=True` for the tie-out | X-15; `esb.pricing` |
| D88 | 14.09.2026 | `Pricing_Calc!C39` (cost to serve) is `(portfolio OPEX + variable OPEX + own BGL fee + memo BGL share + memo interest share) / metered`; the workbook's retired labels return 0 and are reproduced only under `pricing_as_cached=True` | X-16; `esb.pricing` |
| D92 | 14.09.2026 | Grid charges (TSO and DSO) settle on the TSO payment terms, as the workbook links `Input!C23` to `G80` | `esb.cashflow.build_cashflow`; tie-out of `CF_Mth` |
| D93 | 14.09.2026 | The parity fixture `rc_v03_expected.parquet` holds every cached numeric cell of the five result sheets; 153 layout cells (label self-check column, anchors, blank manual column) are declared not applicable and listed in the parity report rather than mapped | `esb.parity`; `tests/parity` |
| D94 | 14.09.2026 | The engine computes the price build-up for every off-taker; only the position-3 view is cached in the workbook and can be tied. The other views are validated by the same code path and the module tests | `esb.pricing` |
| G3 / G4 | 14.09.2026 | Gates G3 (edge suites) and G4 (parity tie-out, corrected `Pricing_Calc` values D87 / D88) accepted by the CEO | CEO ruling | — |
| PSTORE | 14.09.2026 | Run-time parameters persist in a **versioned scenario file** (`ESB-SCN 1.0`, JSON: name, integer version, timestamp, engine version, md5 of the register, history, the full register incl. display names); no database in this phase. The file lives outside the repository; names are allowed in it (F-035) | CEO ruling | Closes OPEN_ITEMS PSTORE |
| D95 | 14.09.2026 | Display names of off-takers and counterparties are fields of the register (`name`, empty in the repository); the application shows the name when set and the code otherwise. The engine never reads a name | F-035; PSTORE | — |
| D96 | 14.09.2026 | Uploads are joined by `esb.assemble`: the Reference Case fixture is an optional base layer, each accepted delivery replaces its series, a wholesale delivery is keyed to a scenario by the application's choice or the `Std_Control` label; an Active off-taker or source without a complete series, or an active scenario without prices, blocks the run with a named message. Blank stays blank | rules 2, 3 and 6 | — |
| D97 | 14.09.2026 | The case bundle (`ESB-CASE 1.0`, zip) carries the scenario file, every accepted upload byte-identical, the importer provenance and the headline results; re-opening re-imports through the same checks, re-runs, and compares the headline values at the parity tolerance; a member with a changed md5 is refused | prompt section 8.3 | — |
| D98 | 14.09.2026 | The application never shows a traceback: importer refusals, coverage problems and engine errors are shown as named messages; the session log records every upload, refusal, change, run and export | rule 6 | — |
| D99 | 14.09.2026 | Baseload nomination uploads are accepted and listed but not consumed: the engine sources Baseload from the strips of the register, as the Reference Case does | parity first | Phase 7 may consume them |
| D100 | 14.09.2026 | "Iteration count" on the cash-flow page is shown as the calculation-order trace with the statement that the count is 0 by construction | G0-D6 | — |
| D101 | 14.09.2026 | The retail NM pre-tax KPI target (3,00 EUR/MWh) is a register value (`meta.kpi_retail_nm_target_eur_per_mwh`), editable on the Parameters page | prompt section 10.1 page 10; house benchmark EW-NFR-01 | — |
