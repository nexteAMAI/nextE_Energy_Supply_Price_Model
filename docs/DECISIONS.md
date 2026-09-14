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
