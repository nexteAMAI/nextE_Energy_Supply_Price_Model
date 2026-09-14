# OPEN_ITEMS

Carried from the Phase 0 discrepancy register (`X-REGISTER_GW-ESB-01_14092026.md`, CEO folder)
and from Phase 2. Items are closed here when the code or a ruling settles them.

| ID | Item | Phase | Status |
|---|---|---|---|
| X-15 / X-16 | `Pricing_Calc!C46` (+94 row offset) and `C39` (retired labels) return wrong values in the frozen workbook; the engine implements the corrected logic (D87 / D88) and reproduces the cached values under `pricing_as_cached=True` for the tie-out; `tools/parity_report.py` lists both values per cell (C39, C41, C46, C47, C48 and `Portf Overview!E139`) | 3 | implemented; CEO acceptance of the corrected values at G4 |
| X-19 | Surplus = Deficit price in both loaded scenarios; provenance of the loaded Aurora series to be stated by the CEO; rule-book tests use a synthetic dual-price case | 3 | open |
| X-05 | Regulatory citations quoted from the workbook (Transelectrica PO 01.13, ANRE Order 129/2015, art. 331 alin. (2) lit. e), Legea 227/2015 art. 41) are unverified; `source_status: unverified` in the parameter register | 3 | open (external) |
| X-03 | Single tariff vector for all off-takers (DEER MV set with T_LV = 0); real DSO and voltage level per off-taker to be supplied before G5 | 5 | open |
| X-29 | The workbook carries values on its 96 undated placeholder rows (prices; volumes 0). Not consumed by any date-keyed aggregation; not in the fixtures | 2 | noted |
| G2-real | No real daylight-saving-bearing delivery available; G2 met on synthetic deliveries only | 2 | open until a real file is placed in `_sources/_internal/` |
| PSTORE | Run-time parameter store for user-set parameters (all dynamic, incl. off-taker names): Supabase per stack vs versioned scenario file | 5 | decide at G4 |
| AF:AG | Portfolio budget series `FW_Retail_Volume!AF:AG` not in the fixtures (only scale the placeholder off-takers, whose series are extracted directly) | 4 | noted |
| PC-views | `Pricing_Calc` views of positions 1, 2 and 4 have no cached workbook values (the sheet shows one off-taker at a time); they cannot be tied cell by cell | 4 | noted (D94) |
| G3 | Module tests cover the rule book, sources, guarantees, cash-flow keys, tax and eleven edge scenarios (77 tests); CEO review of the edge list against the specification | 3 | for G3 ruling |
