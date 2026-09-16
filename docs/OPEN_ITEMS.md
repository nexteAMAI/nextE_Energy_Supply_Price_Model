# OPEN_ITEMS

Carried from the Phase 0 discrepancy register (`X-REGISTER_GW-ESB-01_14092026.md`, CEO folder)
and from Phase 2. Items are closed here when the code or a ruling settles them.

| ID | Item | Phase | Status |
|---|---|---|---|
| X-15 / X-16 | `Pricing_Calc!C46` (+94 row offset) and `C39` (retired labels) return wrong values in the frozen workbook; the engine implements the corrected logic (D87 / D88) and reproduces the cached values under `pricing_as_cached=True` for the tie-out; `tools/parity_report.py` lists both values per cell (C39, C41, C46, C47, C48 and `Portf Overview!E139`) | 3 | implemented; CEO acceptance of the corrected values at G4 |
| X-19 | Surplus = Deficit price in both loaded scenarios; provenance of the loaded Aurora series to be stated by the CEO; rule-book tests use a synthetic dual-price case | 3 | open |
| X-05 | Regulatory citations quoted from the workbook (Transelectrica PO 01.13, ANRE Order 129/2015, art. 331 alin. (2) lit. e), Legea 227/2015 art. 41) are unverified; `source_status: unverified` in the parameter register | 3 | open (external) |
| X-03 | Single tariff vector for all off-takers (DEER MV set with T_LV = 0); real DSO and voltage level per off-taker to be supplied before G5 | 5 | mechanism delivered 15.09.2026 (D109: per off-taker DSO + voltage level on the Parameters page); the actual DSO / level per off-taker and the 2027 tariff values remain to be supplied |
| TAR-SRC | Grid tariff table values (RON/MWh, 2026) come from the v1.1 template sheet `Grid_Taxes_RO`, unverified against the ANRE orders; Retele Electrice Romania has no distribution rows there; the v1.1 sheet labels LV "(6÷20kV)" (typo, shown as 0,4 kV) | 6 | open (external verification) |
| X-29 | The workbook carries values on its 96 undated placeholder rows (prices; volumes 0). Not consumed by any date-keyed aggregation; not in the fixtures | 2 | noted |
| G2-real | No real daylight-saving-bearing delivery available; G2 met on synthetic deliveries only | 2 | open until a real file is placed in `_sources/_internal/` |
| PSTORE | Run-time parameter store | 5 | closed - versioned scenario file (CEO 14.09.2026; `esb.scenario_file`) |
| AF:AG | Portfolio budget series `FW_Retail_Volume!AF:AG` not in the fixtures (only scale the placeholder off-takers, whose series are extracted directly) | 4 | noted |
| PC-views | `Pricing_Calc` views of positions 1, 2 and 4 have no cached workbook values (the sheet shows one off-taker at a time); they cannot be tied cell by cell | 4 | noted (D94) |
| G3 | Edge list accepted by the CEO 14.09.2026 | 3 | closed |
| G2-real | (carried) still no real daylight-saving-bearing delivery; the Data page accepts `local_clock` files and will process one when placed | 5 | open |
| C3 | Price types other than Fixed (Cap+Excess, DAM-indexed, IDM-indexed, floors, caps, index deltas, contract currency, invoice FX) of the C3 dashboard | 7 | not modelled; stated on the Sources and Contracts page |
| C1 | Aurora scenario library 2026-2031 and multi-year runs | 7 | not modelled; stated on the Scenarios page |
| NAMES | Names ever typed into the application persist only in scenario files and case bundles outside the repository; a scenario file must never be committed | 5 | standing rule (F-035) |
| INPUT-FMT | Numeric entry and display are Romanian on every surface (text-based fields and grids, D106) | 5 | closed 15.09.2026 (0.6.0) |
| TPL-CANON | Canonical templates: input templates are generated register-sized in the CEO's styling (D113 input styling, contract 1.1 - D112); the output workbook is rendered from `data/layout/run_export.json` extracted from the CEO's formatted template (D113, D114). Remaining: CEO rulings D-D (Claude corrects the template files, removes the Claude Log sheets, moves the originals to `02_superseded/`), D-F (engine-key row in QH_daily - rendered), D-G (one QH column order; CSV = the engine keys in grid order); re-extraction after any template change (`python tools/extract_layout.py <template>`) | 7 | in progress (audit 16.09.2026) |
| G5 | Walk-through 15.09.2026: sidebar (D104), inactive off-taker without series and duplicate QH column (D107), leg tags (D105), Romanian formats (D106) fixed in 0.6.0; CEO acceptance of G5 on the live app outstanding | 6 | open |
| DEPLOY | Deployed 14.09.2026 (D102). Remaining: production access model (private slot, paid plan or SSO) and rotation of the initial sign-in credential | 6-7 | open |
| O-19 | The CEO's output template carries `Is_Weekend_or_RO_public_holiday_flag_EET` twice (columns O and T of `QH_full`); the renderer writes it once | 7 | noted (audit 16.09.2026); template correction under D-D |
| O-20 | Calendar block field meanings are interpretations (`esb.calendar_ro.CALENDAR_INTERPRETATION`): `Week_year` = ISO week, `Day_year` = day of month, `Weekday` = ISO number (Mon = 1), `Season` = meteorological, `Day_hour_interval` = start hour, `Day_interval` = 1-96 within the day (CET twin within the CET day). To be confirmed by the CEO | 7 | open |
| HOL-RO | Romanian public-holiday rules in `config/calendar_ro.yaml` cite Codul muncii art. 139 alin. (1) with `source_status: to_verify` (article numbering, the current list and the act that added 6-7 January to be confirmed against Monitorul Oficial); the flag is descriptive only, nothing in the engine reads it | 7 | open |
| PAR-CAT | Parameter catalogue (Unit / Standard-default / Source-vintage per register path) for the `Parameters` sheet: only the calendar and the grid tariff rows carry a source today (D-I) | 7 | open (T12.7) |
| QH-FULL | The full quarter-hour export is 54 MB and takes ~25 s on the reference machine; on Streamlit Community Cloud the build runs in the request and may hit the platform's memory or time limits - to be measured on the live app before it is offered by default | 6 | open |
