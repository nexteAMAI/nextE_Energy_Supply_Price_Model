# USER GUIDE - nextE Energy Supply Bid Management Tool

Application v0.7.6 (engine v0.4.0, unchanged since gate G4). Written for the person who prices and manages the retail
supply portfolio, not for a developer. Every capability of the Excel model
`Energy_Supply_Portfolio_Tracking_v03` is reachable through the twelve pages described here;
nothing requires a command line.

## 1. Purpose

The tool prices energy supply offers and tracks the portfolio's margin, cash and guarantees
for one contract year at quarter-hour resolution. It reproduces the frozen Reference Case
workbook exactly (20.092 cached cells tied at one part in a million, gate G4 of 14.09.2026)
and replaces it as the working instrument: parameters are set on screen, time series are
uploaded in a standard template, results are read on screen and exported to Excel.

## 2. Opening the application

The application opens in the browser at https://nexte-esb.streamlit.app and asks for the
username and password issued by nextE; "Sign out" is at the bottom of the sidebar. It works in
Chrome, Edge and Firefox and needs no installation. On the desk machine, `Start_ESB_App.cmd`
in the repository folder starts a local copy without the sign-in gate. The left sidebar lists the twelve pages, shows the engine version, the
scenario file in use and the state of the last run, and holds the "Run engine" button.

The application starts with the coded Reference Case: the register of the frozen workbook
(off-takers OT1 to OT4, no names) and the Reference Case series of 2027. Nothing is stored on
a server: what is not saved as a scenario file or a case bundle (page 11) is lost when the
browser tab closes.

## 3. The pages

### 1 - Overview
Year KPIs of the forecast case (retail revenue, GM2, net margin before and after tax, volumes,
guarantees, peak funding, interest), the margins-by-leg table of the Portf Overview sheet
(portfolio, every off-taker, resell; budget, forecast and delta), the cash-flow summary, the
pricing summary of the selected off-taker, and the checks. Every check is listed with its
value and a PASS or FAIL mark; a failing check means the figures are not to be used.

### 2 - Data
Where series enter. The switch "Use the Reference Case fixture" keeps the coded 2027 series as
the base layer; uploads replace series one by one. A delivery is one workbook in the standard
template (at the bottom of the page: off-taker load, PV generation, baseload nomination,
wholesale prices - press "Prepare the template" once, about 20 seconds, then "Download"; the
prepared file is kept for the register it was built for). On "Validate and load" the file passes the six checks (spine,
layout, surface, registry, energy, gaps); the result is shown check by check, and a refused
file is not loaded. For a wholesale-prices delivery the price scenario is taken from the
file's Std_Control or chosen in the box next to the uploader. The coverage table shows which
off-takers, PV series and scenarios are available; an Active off-taker without series blocks
the run with a message naming it.

### 3 - Parameters
The Input sheet. The top block manages the scenario file: name, version note, "Save as new
version", download, load, and reset to the Reference Case register. Below, the sections as in
the workbook: A General (scenario, year, case window, FX, VAT, CIT, tax day, reverse charge,
opening cash, OPEX, shareholder loan rate, PV resell factors), B Premium standard and the NM
KPI target, C Off-takers (one tab each: display name, Active switch, contract window, terms,
advance, contract and PV prices, premium components budget and forecast, target GM, strips
and product prices by month, own guarantee, optional tariff set; "Add off-taker" appends a
new inactive position, the last position can be removed), D Counterparties (PV, Baseload,
Spot, BRP, TSO, DSO: Active, terms, advance, sign convention k, guarantee), E Market
guarantees (regulatory formula inputs) and the admin tab for the regulated tariff components
and green-certificate values. Every section applies with its own "Apply changes" button.
Number fields are typed and shown in the Romanian convention (84,50; 1.234,56; 0,21): the dot
groups thousands and the comma is the decimal separator. A single dot with other than three
digits after it (0.21) is read as a decimal point; anything that is not a number is refused in
place and the previous value is kept. Month-by-product grids follow the same rule.
Every table shows the unit of measurement of every number: a Unit column after the line label, or the
unit in the column header. Off-takers take their regulated tariff components from the grid tariff
table by naming their distribution operator and metering-point voltage level (admin tab holds the
table in RON/MWh); the portfolio set and a manual override remain available. The fixed guarantee
amount of every counterparty is typed by the user; for the PV source the field shows the workbook
derivation from the last run until a value is entered.
Every metric label ends with the leg it belongs to: Retail (the off-taker book), Wholesale spot
resell (surplus PV and Baseload sold to the market) or Total (both legs, and the cost-to-serve,
financing and tax lines below Total GM2). Checks, market series and parameters carry no leg.

### 4 - Sources and Contracts
The contractual position as modelled: per source the nomination basis, price basis,
settlement volume, terms, advance, k, guarantee and fee; per off-taker the contract window,
prices, strips, premiums, target margins, terms and guarantee; strips and product prices by
month. This release models fixed prices for PV and Baseload and MIN(DAM, IDCT) for Spot; the
Cap+Excess and indexed price types of the C3 dashboard are Phase 7 scope.

### 5 - Scenarios
The three price scenarios (Aurora Central, Aurora Low, User Forecast): which are loaded,
their annual statistics, the active one, its monthly profile and daily shape. "Compare
loaded scenarios" runs the engine on each loaded scenario and tabulates the year values side
by side. Multi-year libraries are Phase 7 scope.

### 6 - Engine (QH)
The quarter-hour merit order. Monthly sourcing stack (PV, Baseload, Spot delivered; PV and
Baseload resold), monthly margins and imbalance, the off-takers in the cascade with their
LIFO resell attribution, the five row checks, and a drill-down: choose a day and a view
(sourcing, prices, imbalance, resell, one off-taker's block) to see the 96 quarter-hours
with the workbook column letters. The full 35.040-row frame is loaded only on request.

### 7 - P&L
The Cons_P&L sheet by month and year. Choose the portfolio blocks to show (volumes, costs
and prices, retail margins, wholesale resell, totals, cost to serve, net margin and tax,
green certificates, checks); each line carries its workbook row number. Below, one
off-taker's section with the same structure, including the risk premium and reserve lines
and the retail net margin.

### 8 - Cash Flow
The CF_Mth sheet (accruals, receipts, payments, VAT, financing; beyond-December column) and
the daily ledger (loan outstanding, free cash, restricted floor by day; the summary with the
reconciliation to the monthly table). The calculation order is shown in place of an
iteration count: the engine has no iteration by construction (ruling G0-D6).

### 9 - Guarantees
Outstanding amounts by month per counterparty and per off-taker, the per-counterparty table
(type, sizing, peak, months outstanding, window, fee), the required amount under each of
the five sizing methods on this run's bases, the regulatory formulas with the inputs of the
run, and the own guarantees received. The regulatory citations are carried as quoted in the
workbook and marked unverified until confirmed.

### 10 - Pricing / Bid
The Pricing_Calc sheet for the selected off-taker: energy price, offer excluding and
including VAT, contract minus offer, the retail NM KPI against the target of section B, the
build-up chart (purchase price, imbalance, premium, target GM, pass-through), the year and
monthly table with row numbers, re-pricing at the current forecast, and the manual case: a
form with the year values pre-filled, any of which can be overridden to price a new
off-taker or a what-if. Cost to serve and the forecast premium use the corrected logic of
decisions D87 and D88.

### 11 - Exports
Excel workbook of the run in the house output layout (Portf Overview, Cons_P&L, CF_Mth,
CF_Daily_Ledger, one pricing sheet per off-taker, guarantees, QH_daily, parameters,
provenance; the full quarter-hour frame QH_full on request - about 54 MB, allow half a
minute). Every sheet carries the CONFIDENTIAL banner, the sheet title, a subtitle and, in
row 4, the engine version, scenario, year and export time; the header row is row 6, column B
is the unit of every line, values are formatted with their unit (1.234 MWh, 84,50 €/MWh),
totals are navy, reconciliation checks show six decimals, and the off-taker blocks repeat
once per off-taker with the display name in the block header. CSV files (semicolon, decimal
comma) now carry key, label and unit per line with decimals by unit, dates as dd.mm.yyyy; the quarter-hour CSV of the Engine page uses the workbook's column order (calendar block, sequence, engine keys); the case bundle and the parity report are unchanged. The case bundle is one zip with the scenario file, every accepted upload, the
provenance and the headline results; re-uploading it on this page re-imports the deliveries,
re-runs the engine and states whether the headline values were reproduced. A number that
cannot be reproduced is not put in front of an off-taker.

### 12 - Audit and Log
The session's run log (uploads, refusals, parameter changes, runs, exports), the provenance
of the current series with the assembled frame's md5, the calculation-order trace, the
scenario file history, the verification status of the regulatory constants, and the
registers (decisions, open items, methodology) as shipped with this version.

The verification table (since 0.7.2) shows, for every regulatory constant of the register, the
primary source, its status, the validity period, the date it was checked and a note. A status
`contradicted` means the primary source gives another value or rule than the workbook: the
Reference Case keeps the workbook value (the parity gate depends on it) and the note names
the open item awaiting the CEO's ruling (TAR-2026 for the regulated tariff set, BRP-GF for the
BRP guarantee rule, RC-2027 for the reverse charge that ends on 31.12.2026). Values whose
validity ends before the case year are the ones to refresh before a bid.

### Bid-scenario defaults (since 0.7.3)
The Reference Case register reproduces the frozen workbook and keeps its settings for the parity
gate. A bid scenario starts from it and then takes the rulings of 17.09.2026 with one button on
the Parameters page, "Apply the bid-scenario defaults (D119)": the reverse charge on source
purchases off for a case year after 2026 (the measure ends 31.12.2026), the BRP guarantee by the
delegated-PRE rule (initial guarantee of the PRE service contract, then months of average
imbalance value), and the shipped ANRE 2026 grid tariff table. Under the delegated rule the
PRE service fee of the contract (2.500 RON a month plus 5 % of the aggregation gain) appears
as its own line in the cost-to-serve block and in the cash flow; the aggregation gain itself is
not modelled unless you enter an assumed share on the Parameters page (section E). While a scenario still contradicts
the verified state the page says so in a note; nothing is blocked.

## 4. Working sequence

1. Parameters: load a scenario file or start from the Reference Case register; set names,
   activity, contracts, strips, guarantees; apply each section; save as a new version.
2. Data: upload the deliveries of the case year (load, PV, prices per scenario) or keep the
   Reference Case fixture; confirm coverage is complete.
3. Scenarios: set the active price scenario.
4. Run engine (sidebar or any results page).
5. Read Overview, P&L, Cash Flow, Guarantees, Pricing; check that every check passes.
6. Exports: download the Excel workbook and the case bundle; keep both with the offer.

## 5. Known limitations of this release

- One spine year per run; multi-year runs and the 2026-2031 scenario library are Phase 7.
- Price types other than fixed (Cap+Excess, DAM- and IDM-indexed, floors, caps, index
  deltas, invoice FX) are Phase 7 (C3 dashboard).
- Baseload nomination uploads are informative; the engine sources Baseload from the strips
  of the Parameters page, as the Reference Case does.
- Off-taker codes follow the merit-order position (OT1 first served); reordering is done by
  editing the positions' contents, not by dragging.
- The Pricing_Calc tie-out covers the position-3 view only, the one the workbook caches (D94);
  the other views run the same code.
- The regulated constants were verified against their primary sources on 17.09.2026
  (parameter catalogue, Audit page: status, validity, check date, note). Values the sources
  contradict stay in the Reference Case for parity; bid scenarios take the verified set
  through "Apply the bid-scenario defaults" on the Parameters page. A value that changes
  inside the spine year is carried as one value per scenario (D122). The delegated-PRE
  residuals (pro-rata TSO-guarantee contribution, negative-price VAT treatment, 4-working-day
  service term, aggregation gain) are stated, not modelled.
- Daylight-saving handling of uploads is proven on synthetic deliveries; a real local-clock
  delivery spanning the March and October changes has not yet been processed (gate G2
  conditional).
- Session state lives in the browser tab; closing it discards unsaved work.
