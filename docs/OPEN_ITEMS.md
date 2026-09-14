# OPEN_ITEMS

Carried from the Phase 0 discrepancy register (`X-REGISTER_GW-ESB-01_14092026.md`, CEO folder)
and from Phase 2. Items are closed here when the code or a ruling settles them.

| ID | Item | Phase | Status |
|---|---|---|---|
| X-15 / X-16 | `Pricing_Calc!C46` (+94 row offset) and `C39` (retired labels) return wrong values in the frozen workbook; the engine implements the correct logic and parity targets for the affected lines are re-derived from the workbook's inputs (`expected: corrected (D87/D88)`) | 3-4 | open until G4 |
| X-19 | Surplus = Deficit price in both loaded scenarios; provenance of the loaded Aurora series to be stated by the CEO; rule-book tests use a synthetic dual-price case | 3 | open |
| X-05 | Regulatory citations quoted from the workbook (Transelectrica PO 01.13, ANRE Order 129/2015, art. 331 alin. (2) lit. e), Legea 227/2015 art. 41) are unverified; `source_status: unverified` in the parameter register | 3 | open (external) |
| X-03 | Single tariff vector for all off-takers (DEER MV set with T_LV = 0); real DSO and voltage level per off-taker to be supplied before G5 | 5 | open |
| X-29 | The workbook carries values on its 96 undated placeholder rows (prices; volumes 0). Not consumed by any date-keyed aggregation; not in the fixtures | 2 | noted |
| G2-real | No real daylight-saving-bearing delivery available; G2 met on synthetic deliveries only | 2 | open until a real file is placed in `_sources/_internal/` |
| PSTORE | Run-time parameter store for user-set parameters (all dynamic, incl. off-taker names): Supabase per stack vs versioned scenario file | 5 | decide at G4 |
| AF:AG | Portfolio budget series `FW_Retail_Volume!AF:AG` not in the fixtures (only scale the placeholder off-takers, whose series are extracted directly) | 4 | noted |
