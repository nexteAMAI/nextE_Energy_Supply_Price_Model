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
| G2-real | G2 accepted conditionally by the CEO 17.09.2026; closes on the first real local-clock delivery uploaded on the Data page | 5 | conditional |
| C3 | Price types other than Fixed (Cap+Excess, DAM-indexed, IDM-indexed, floors, caps, index deltas, contract currency, invoice FX) of the C3 dashboard | 7 | not modelled; stated on the Sources and Contracts page |
| C1 | Aurora scenario library 2026-2031 and multi-year runs | 7 | not modelled; stated on the Scenarios page |
| NAMES | Names ever typed into the application persist only in scenario files and case bundles outside the repository; a scenario file must never be committed | 5 | standing rule (F-035) |
| INPUT-FMT | Numeric entry and display are Romanian on every surface (text-based fields and grids, D106) | 5 | closed 15.09.2026 (0.6.0) |
| TPL-CANON | Canonical templates: input templates are generated register-sized in the CEO's styling (D113 input styling, contract 1.1 - D112); the output workbook is rendered from `data/layout/run_export.json` extracted from the CEO's formatted template (D113, D114). Remaining: CEO rulings D-D (Claude corrects the template files, removes the Claude Log sheets, moves the originals to `02_superseded/`), D-F (engine-key row in QH_daily - rendered), D-G (one QH column order; CSV = the engine keys in grid order); re-extraction after any template change (`python tools/extract_layout.py <template>`) | 7 | in progress (audit 16.09.2026) |
| G5 | Walk-through 15.09.2026: sidebar (D104), inactive off-taker without series and duplicate QH column (D107), leg tags (D105), Romanian formats (D106) fixed in 0.6.0; CEO acceptance of G5 on the live app outstanding | 6 | open |
| DEPLOY | Deployed 14.09.2026 (D102). Remaining: production access model (private slot, paid plan or SSO) and rotation of the initial sign-in credential | 6-7 | open |
| O-19 | The CEO's output template carries `Is_Weekend_or_RO_public_holiday_flag_EET` twice (columns O and T of `QH_full`); the renderer writes it once | 7 | noted (audit 16.09.2026); template correction under D-D |
| O-20 | Calendar block field meanings (`esb.calendar_ro.CALENDAR_INTERPRETATION`) confirmed by the CEO 17.09.2026 as coded | 7 | closed |
| HOL-RO | Romanian public-holiday rules in `config/calendar_ro.yaml` cite Codul muncii art. 139 alin. (1) with `source_status: to_verify` (article numbering, the current list and the act that added 6-7 January to be confirmed against Monitorul Oficial); the flag is descriptive only, nothing in the engine reads it | 7 | open |
| PAR-CAT | Parameter catalogue `config/parameter_catalogue.yaml` (D-I): units for every numeric path by rule, standard / source / status for the regulatory constants and house assumptions; sources completed by the regulatory verification pass | 7 | in progress |
| QH-FULL | Full quarter-hour export on Community Cloud: measured 17.09.2026 on the live app - 57,3 MB built in ~40 s without error (run log) | 6 | closed 17.09.2026 |
| HIST-035 | The counterparty name of W-1 sits in the git history of `main` (commit `31d30e9`, `data/layout/run_export.json`); removed from the working tree in 0.7.1; a history rewrite (force-push of `main`) needs the CEO's explicit word | 7 | open - CEO |
| PREP-BY | With one shared sign-in user, `Requested by` on the Provenance sheet identifies the account, not the person - an optional 'Prepared by' initials field on the Exports page | 7 | open |
