# METHODOLOGY - engine v0.4.0

How the Python engine reproduces the frozen Reference Case workbook
`Energy_Supply_Portfolio_Tracking_v03_[base].xlsx` (md5 `1c873718bcd08957229f1f46d5449b2a`).
The Phase 0 specification (CEO folder, `00_Spec/01_current/`) remains the normative source for
the formulas; this document maps each engine module to the workbook region it replicates and
records the choices made where the workbook and the code differ.

## 1. Evaluation order (ruling G0-D6)

`esb.engine.run(series, params, selected_offtaker, pricing_as_cached)` executes eight stages in a
fixed order; every stage reads only stages before it, there is no iteration and no circular
reference. `RunResult.trace` records the order and cumulative timing of a run (about 1,3 s on
the Reference Case).

| Stage | Module | Workbook region | Output |
|---|---|---|---|
| 1 grid | `esb.grid` | `QH_P&L` date / interval spine | 365 x 96 dated rows (366 x 96 in a leap year), EET labels, Peak flag on intervals 37-84 |
| 2 qh_engine | `esb.merit_order` (+ `scenarios`, `sources`, `imbalance`) | `QH_P&L` columns AA:PL | one frame of 276 named columns per quarter-hour |
| 3 pnl_stage1 | `esb.pnl.build_pnl` | `Cons_P&L` rows 6-223 and the four sections | monthly tables (12 months + year) for the portfolio and each off-taker |
| 4 cashflow_stage_a | `esb.cashflow.build_cashflow` | `CF_Mth` rows 6-73, 78-89 | monthly cash flow up to the financing need and interest |
| 5 pnl_stage2 | `esb.pnl.finalize_pnl` | `Cons_P&L` rows 220-264 | interest, unallocated costs, net margin, corporate income tax |
| 6 cashflow_stage_b_daily | `esb.cashflow.finalize_cashflow`, `build_daily_ledger` | `CF_Mth` rows 74-77, 90-99; `CF_Daily_Ledger` | tax outflows, closing checks, 365-row daily ledger and its summary |
| 7 pricing | `esb.pricing.price_offtaker` | `Pricing_Calc` | price build-up per off-taker (year + 12 months) |
| 8 overview | `esb.reporting.build_overview` | `Portf Overview` | executive table, cash-flow, pricing and check sections |

The two-stage split of P&L and cash flow reproduces the workbook's dependency: financing
interest is computed on the monthly cash position (`CF_Mth` row 89) and only then enters the
net margin (`Cons_P&L` row 220), whose tax outflows return to the cash flow the month after
each quarter.

## 2. Grid (`esb.grid`)

`make_grid(year)` builds the positional quarter-hour spine: `seq`, `date`, `interval` 1-96,
Peak / Off-Peak. Peak is intervals 37-84 (09:00-21:00 EET), verified against all 35.040 rows of
the workbook. The workbook's 35.136-row container (96 undated rows at 29.02 in v03) is not
reproduced (D89); every workbook aggregation is date-keyed, so the placeholder rows never enter
a result. Daylight-saving handling belongs to the importer (docs/DATA_CONTRACT.md); the engine
receives a full, gap-free grid.

## 3. Quarter-hourly engine (`esb.merit_order`, sheet `QH_P&L`)

Inputs: the coded series frame (docs/DATA_CONTRACT.md section 9), the grid and the parameter
register. Steps per quarter-hour, vectorised over the whole year:

1. Scenario prices (`esb.scenarios.active_prices`): DAM, IDCT VWAP15, Surplus and Deficit
   imbalance prices and the system direction of the active scenario; curtailed variants
   `MAX(price, 0)`.
2. Sources (`esb.sources`): PV availability from the PV forecast and its deviation percentages
   through the generation imbalance block (`esb.imbalance.generation_block`); Baseload
   availability as the sum of the active off-takers' strips (`strip_mwh_per_qh`: (BL24 MW +
   Peak MW or Off-Peak MW) / 4, gated by `active`); Baseload product prices per off-taker,
   MW-weighted between BL24 and Peak / Off-Peak with the BL24 price as fallback when the
   strip is 0 (`product_price_per_qh`).
3. Merit order: for each off-taker in position order, the notified demand is served first from
   the remaining PV, then from the remaining Baseload, then from Spot. Buys of PV and Baseload
   and all retail sells settle on metered volume; the Spot buy settles on notified volume at
   `MIN(DAM, IDCT)`; the source imbalance of each source is carried to each off-taker as its
   notified buy x the source's specific imbalance per notified MWh.
4. Consumption imbalance per off-taker (`esb.imbalance.consumption_block`, DSO convention
   `k = -1`): metered minus notified in BRP sign, split into surplus and deficit, settled at
   the Surplus and Deficit price respectively.
5. Resell of the surplus after the last off-taker, at the resell price
   `MAX(curtailed DAM, curtailed IDCT)`: PV resell cost and revenue are `volume x factor x resell
   price` (factors 0,93 / 1,05), the PV source imbalance is set to 0 when both DAM and IDCT are
   negative; Baseload surplus is attributed LIFO to the off-takers' strips (position 1 takes the
   remainder) and costed at the originating off-taker's product price (budget and forecast),
   its revenue is `volume x resell price`.
6. Checks: `check_demand`, `check_pv`, `check_bl`, `check_imb`, `check_origin` are 0 on every
   row (the workbook's BV:BY and DX columns).

`WORKBOOK_COLUMNS` (71 portfolio columns AA:DY) and `OFFTAKER_COLUMNS` (50 columns per
off-taker block; blocks DZ:GV, GW:JS, JT:MP, MQ:PL, 75 columns each) give the workbook letter
of every named column for the four-off-taker layout. Row-3 totals (SUM and AVERAGEIF) and
rows 7-8 of every column were tied to the workbook at 1e-9 (723 checks) before the monthly
layer was built.

## 4. Imbalance rule book (`esb.imbalance`)

Implements the identities 2.1-2.6 of the rule book (`Imblance_settlement_overview_RO`):
`IMB_% = k x (M - N) / N` with IFERROR -> 0, the inversions `M = N x (1 + k x IMB_%)` and
`N = M / (1 + k x IMB_%)`, the surplus / deficit split in BRP sign and the settlement
`surplus x Surplus price + deficit x Deficit price`. The sign convention `k` is declared per
volume slot in the upload contract and never inferred. The loaded scenarios carry
`Surplus = Deficit` (X-19); the dual-price path is covered by a synthetic test.

## 5. Monthly layer (`esb.monthly`, `esb.pnl`; sheet `Cons_P&L`)

`MonthlyTable` holds one row per P&L line with 12 months and a year value; `year_rule`
(`sum`, `last`, `max`, `mean`) reproduces the workbook's column O per row. `QHAggregator` sums
or averages quarter-hour columns by month (AVERAGEIF on positive values for prices).
`WORKBOOK_ROWS` maps every portfolio line to its `Cons_P&L` row; `SECTION_ROWS` maps the
off-taker section lines relative to the position-1 anchor (row 266, pitch 136 rows; positions 3 and 4 carry one extra row after relative row 55).

Stage 1 builds volumes, costs, revenues, GM1 / GM2 (budget and forecast), the resell legs,
guarantees (section 6), the retail and resell reconciliation legs and the regulatory inputs.
Stage 2 adds interest, unallocated costs (market BGL fees + interest), the net margin, the
corporate income tax (quarterly, on cumulative year-to-date net margin, floored at 0, less tax
already accrued) and the after-tax margins; the section memo shares of BGL fees and interest
are allocated on metered volume.

## 6. Guarantees and regulatory formulas (`esb.guarantees`)

Counterparty guarantees follow the five sizing methods of the workbook: Fixed, % of Contract
Value, Coverage Months, Dynamic (monthly, X-26) and Regulatory formula; the outstanding amount
is placed on the months whose first day lies in the guarantee window and is 0 for inactive
counterparties or type `None`. The PV Fixed amount is derived (`mean(cost_pv_budget) +
mean(rs_pv_cost)`, workbook `Input!C85`). BGL fees are `bgl_fee_pa / 12 x outstanding` per
month (`Monthly`) or `bgl_fee_pa x outstanding x days / 365` in the issuance month
(`One-time at issuance`); Cash Collateral carries no fee. Regulatory amounts: Spot
`buffer_days x peak daily notified spot buy x peak DAM price`, BRP
`rate x (generation MW + consumption peak MW) / FX`, TSO `Vtm x 4 x (TL + SS) x metered / 12`
per off-taker, DSO `Vdm x 4 x (T_HV + T_MV + T_LV) x metered / 12 + overdue add-on`. The
citations quoted in the workbook for these formulas are carried with `source_status:
unverified` (docs/PARAMETERS.md).

## 7. Cash flow (`esb.cashflow`; sheets `CF_Mth`, `CF_Daily_Ledger`)

Stage A places every P&L amount on its settlement month with `settlement_keys(year, terms)`:
the month containing `month end + payment terms` (December settlements with terms > 0 fall in
the beyond-December column N). Receipts and payments carry VAT (21 %); source purchases are
reverse-charged when `general.reverse_charge_vat_on_sources` is true; the VAT credit is carried
forward; the grid charges settle with the TSO terms (`Input!C23 = G80`). The financing need is
`MAX(0, restricted cash - (opening + net cash flow))` (restricted cash = risk reserve balance + collateral backing of guarantees), interest is `shareholder loan rate / 12 x financing balance` and is
returned to the P&L (section 5). Stage B places the corporate income tax the month after each
quarter (Q4 in column N) and closes the balance checks.

The daily ledger spreads each monthly settlement on its settlement day (month end + terms,
`tax_payment_day` for VAT and CIT), keeps a daily floor with reserves pro rata, injects
financing daily and accrues interest on a 365-day basis. `daily_summary` returns the peak
funding, the cash trough, the minimum free cash before and after tax, the interest on the
daily basis and the reconciliation checks against the monthly table (`CF_Daily_Ledger` rows
372-379, tied to the workbook).

## 8. Pricing (`esb.pricing`; sheet `Pricing_Calc`)

`price_offtaker(pnl, params, code, manual, as_cached)` rebuilds the price build-up of one
off-taker: sourcing cost, imbalance, regulated components, cost to serve, target margin,
premium components and the repriced energy price, year and 12 months. The workbook caches only
the position-3 view (`Pricing_Calc!C4`); the engine computes every off-taker. Two defects of
the frozen sheet are corrected (D87 / D88, OPEN_ITEMS.md X-15 / X-16):

- `C39` cost to serve: the workbook's retired labels return 0; the engine uses
  `(opex + variable opex + own BGL fee + memo BGL share + memo interest share) / metered`.
- `C46` forecast premium: the workbook's +94 row offset picks the Collateral premium
  component; the engine uses the off-taker's total forecast premium.

`as_cached=True` reproduces the workbook values for the parity tie-out; the production default
is the corrected logic. `tools/parity_report.py` lists both values per affected cell.

## 9. Overview (`esb.reporting`; sheet `Portf Overview`)

`build_overview` assembles the executive table (rows 12-57, per off-taker columns from H and
the resell column T, `OVERVIEW_ROWS`), the cash-flow section (5), the pricing section (6, the
selected off-taker) and the checks section (7). `strip_price_tripwire` (`Portf Overview!E148`) counts strips with MW > 0 and a
0 product price in the budget or forecast table.

## 10. Parity (`esb.parity`, `tools/parity_report.py`)

`data/reference/rc_v03_expected.parquet` holds every cached numeric cell of the five result
sheets of the frozen workbook (20.245 cells). `mapped_cells` addresses each engine value by
sheet and cell through the row and column maps above; `compare` applies the tolerance of
ruling G0-D4 (`abs(py - xl) <= 1e-6 x max(abs(xl), 1)`). Result on the Reference Case, engine
v0.3.0 and unchanged in v0.4.0: 20.092 cells mapped and tied, 0 failures, 153 cells not applicable (layout artefacts:
`Portf Overview` W150:W427 label self-check and B10; `Pricing_Calc` C4, D4, G4, I4 and the
blank manual column D). `tests/parity/test_parity_v03.py` enforces this on every test run.

## 11. Conventions

- Blank is not zero: a missing input is `NaN` and stops the run; the importer never fills gaps.
- Currency EUR throughout; RON amounts converted at `general.fx_ron_per_eur`.
- Times are EET on the grid (fixed 1 h offset to CET); documents use CET/CEST.
- Regulatory constants are never hard-coded; they live in `config/parameters.yaml` with their
  workbook origin and verification status.

## 12. Application layer (Phase 5)

The application (`app.py`, package `app/`) adds no calculation: every number on screen comes
from `esb.engine.run` on the assembled series. Four engine-side modules support it:

- `esb.assemble` (D96) joins the accepted uploads and the optional Reference Case base layer
  into the engine frame and reports coverage (off-takers, PV, scenarios) and problems.
- `esb.scenario_file` (PSTORE) serialises the register with display names as the versioned
  scenario file `ESB-SCN 1.0` and refuses files whose md5 does not match their register.
- `esb.bundle` (D97) writes and re-opens the case bundle `ESB-CASE 1.0`.
- `esb.export` writes the Excel workbook and CSV files of a run; `esb.labels` gives every
  engine key its user-facing label.

`app.state` holds the session (scenario file, uploads, assembled series, last run, log);
`app.brand` holds the presentation layer (Montserrat, navy and neutrals, Romanian formats,
tables, charts). Pages are `app/pages/*.py`, one `render()` each, in the order of the
execution prompt section 10.1. Tests: `tests/unit/test_persistence.py` (round trips and
refusals) and `tests/app/test_pages.py` (every page rendered headlessly on the Reference Case,
the refusal path, a scenario switch, the manual case form, adding an off-taker).
