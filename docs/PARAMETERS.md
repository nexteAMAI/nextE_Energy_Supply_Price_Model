# PARAMETERS - register and verification status

The parameter register is `config/parameters.yaml`, loaded by `config.schema.load_parameters()`
into the `Parameters` dataclass tree (`General`, `Offtaker`, `Counterparty`, `Guarantee`). The
file carries the Reference Case values of the frozen workbook (`Input` sheet and the price and
strip tables) with the workbook cell of origin as a comment on every line. Every value is
dynamic in the application (ruling G0-PAR): the YAML is the Reference Case scenario, not a
configuration users edit. Off-takers are coded by merit-order position (`OT1`..`OT4`); names
never appear in the repository (ruling G0-F035).

## 1. Blocks

| Block | Workbook origin | Content |
|---|---|---|
| `meta` | - | Reference Case identity (name, md5), spine year 2027, currency EUR, `kpi_retail_nm_target_eur_per_mwh` (3,0; D101) |
| `scenario` | `Input!C6:C7` | active scenario (`Aurora Central`, `Aurora Low`, `User Forecast`) and the scenario names; the engine reads the series columns of the active prefix (`esb.scenarios.SCENARIO_PREFIX`) |
| `resell` | `Input!C9:C10` | PV resell cost and revenue factors on the curtailed DAM (0,93 / 1,05) |
| `general` | `Input` section A | case window, FX RON/EUR 5,5, VAT 21 %, tax payment day 25, CIT 16 %, opening cash, portfolio and variable OPEX per metered MWh (0,12 / 0,10), shareholder loan rate 7 % p.a., reverse charge on source purchases, debt facility (present, not modelled - D62) |
| `premium_standard` | `Input` section B | reference premium components and target GM 11; the engine prices with the per-off-taker components |
| `green_certificates` | `Cons_P&L!B6:B11` | GC quota per MWh 0,499387, reference price 148,2201 RON/GC, spot share 0,5; `Parameters.gc_unit_cost` = quota x price / FX |
| `tariff_components_eur_per_mwh` | `Input!C54:F63` | TL, TG, SS, T_HV, T_MV, T_LV, cogeneration, CfD, excise (EUR/MWh at FX 5,5; DEER MV set, T_LV = 0); identical for all four off-takers in the Reference Case (X-03), overridable per off-taker (`Offtaker.tariff_components`) |
| `grid_tariffs` | `config/tariffs_ro.yaml` (ANRE Ordinele nr. 73-78/2025, validity 2026, `basis: specific`; D119) | RON/MWh rows by charge owner (TSO, ANRE, ANAF, each DSO) and component with applicability flags per voltage level; the distribution rows are the specific tariffs of the orders and the cascade sums them to the applied tariff of the connection level (D109); an off-taker with `dso` and `voltage_level` set takes its EUR/MWh components from here; the superseded v1.1 table is `config/tariffs_ro_v11_template_superseded.yaml` |
| `offtakers[]` | `Input` section C, B2 rows 97-105, `FW_Purch_Sell_Price`, strip tables | per off-taker: code, `name` (display name, empty in the repository - D95), active, contract window, contract and PV prices, premium components, target GM, strips (BL24 / Peak / OffPeak MW by month), budget and forecast product prices by month, own guarantee, optional tariff set or `dso` + `voltage_level` (D109) |
| `counterparties` | `Input` section D | pv, baseload, spot, brp, tso, dso: active, payment terms (30 / 15 / 0 / 15 / 15 / 15 days), advance (baseload 50 %), guarantee (type, direction, sizing, amount, fee, window), `k` sign convention, baseload deviation percentages |
| `market_guarantees` | `Input` section E | regulatory formula inputs: spot buffer days 4; BRP `method` (`rate_per_mw` = the workbook's 9.000 RON/MW x (160 MW generation + peak retail MW), Reference Case; `pre_delegated` = D119: max(`pre_initial_ron` 100.000, `pre_months_of_imbalance` 2 x average monthly imbalance value x (1 + VAT if `pre_vat_inclusive`))); TSO Vtm 2; DSO Vdm 1 and overdue add-on 0 |

## 2. Guarantee record

`type` (`None`, `Bank Guarantee Letter`, `Cash Collateral`), `direction`, `sizing` (`Fixed`,
`% of Contract Value`, `Coverage Months`, `Dynamic`, `Regulatory formula`), `fixed_amount`,
`coverage_months`, `pct_of_contract_value`, `bgl_fee_pa`, `bgl_fee_type` (`Monthly`,
`One-time at issuance`), `cash_backing_pct`, `start`, `end`. The PV `fixed_amount` is `null`
in the register: the workbook derives it (`Input!C85`) and so does the engine.

## 3. Verification status of regulatory values

The regulatory verification pass of 17.09.2026 (T12.7, D118) checked the register's regulatory
values against the primary sources; the result per register path - source, `source_status`,
`validity`, `checked`, `note` - lives in `config/parameter_catalogue.yaml` (ruling D-I) and is
rendered on the `Parameters` sheet of every export and on the Audit page. Statuses: `verified`
(read on the primary source), `verified_secondary` (consistent secondary copies, primary text not
reachable), `contradicted` (the primary source says otherwise - the register value is kept for
parity with the frozen workbook until the CEO rules on the open item named), `unverified` (quoted
from the workbook, not covered by the pass), `to_verify`, `not_published`, `assumption`.

| Parameter | Register value | Primary source (17.09.2026) | Validity | Status |
|---|---|---|---|---|
| VAT rate | 21 % | Codul fiscal art. 291 alin. (1), as amended by Legea nr. 141/2025 (M. Of. nr. 699/25.07.2025), from 01.08.2025 | open | verified |
| CIT rate | 16 % | Codul fiscal art. 17 (the workbook cites art. 41 - the declaration and payment article) | open | verified, citation corrected |
| Reverse charge on electricity purchases | applied | Codul fiscal art. 331 alin. (2) lit. e) and alin. (1); sunset art. 331 alin. (6) - 31.12.2026 inclusive (Directive (EU) 2022/890) | until 31.12.2026 | verified - **RC-2027**: the spine year 2027 lies outside |
| Tax payment day | 25 | art. 41 alin. (1) for CIT (checked); VAT art. 326 not checked | open | to_verify |
| GC quota | 0,499387 GC/MWh | ANRE Ordinul nr. 81/16.12.2025 art. 1 (M. Of. nr. 1176/18.12.2025); final 2025 quota 0,49983 (Ordinul nr. 3/2026) | 2026 (estimate) | verified - GC-2027 |
| GC reference price | 148,2201 RON/GC | minimum trading value 2026, Legea nr. 220/2008 art. 11 (29,4 EUR at BNR 2025 average 5,0415; maximum 176,4525), OPCOM notice 08.01.2026 | 2026 | verified - GC-2027 |
| BRP guarantee rate | 9.000 RON/MW | no such rule: Transelectrica PO cod TEL 00.45 ed. I rev. 3 (28.11.2025) applies to a PRE facing the TSO - nextE delegates to a PRE service provider (D119); the delegated rule: initial 100.000 RON (CINTA template contract art. 9.8), then 1-3 average monthly imbalance values (CINTA PRE procedure pct. 5.2.5) | contract term | contradicted for the workbook rule (Reference Case parity); `pre_delegated` for bid scenarios |
| TSO guarantee multiplier Vtm | 2 | Transelectrica PO TEL 01.13 ed. I rev. 0 (TEL nr. 52004/25.11.2021, aviz ANRE nr. 22/2021) pct. 8.2.1: GF = 2 x Vtm (6-month average monthly transmission + system services value) | in force | verified |
| DSO guarantee multiplier Vdm | 1 (+ overdue add-on) | ANRE Ordinul nr. 129/2015 (M. Of. nr. 628/18.08.2015) art. 8: GF = 1 x Vdm + max(V1, V2) | in force | verified |
| Spot collateral buffer | 4 days | OPCOM practice - not covered | - | unverified |
| TG (injection) | 3,63 RON/MWh | ANRE Ordinul nr. 74/2025 (M. Of. nr. 1173/18.12.2025) | 2026 | verified |
| TL (extraction) | 36,54 RON/MWh | ANRE Ordinul nr. 74/2025: **36,45** | 2026 | contradicted (Reference Case parity); the shipped grid table carries 36,45 (TAR-2026 (a)) |
| SS | 14,70 RON/MWh | ANRE Ordinul nr. 73/2025 (Ordinele nr. 12/2026 and 53/2026 not read) | 2026 | verified |
| T_HV / T_MV (DEER MV set) | 39,37 / 122,80 RON/MWh, summed | the applied 2026 tariffs of Distributie Oltenia (Ordinul nr. 75/2025), not DEER's; DEER at MV = 31,96 + 83,36 = 115,32 (Ordinul nr. 77/2025); the cascade double-counts HV | 2026 | **contradicted - TAR-2026** |
| Grid tariff table (`config/tariffs_ro.yaml`) | ANRE 2026 specific tariffs | ANRE Ordinele nr. 73-78/2025 (M. Of. nr. 1173/18.12.2025); Retele Electrice rows from secondary copies. The v1.1 table (applied tariffs under shifted operator names, TL 36,54, Delgaz MT 125,17) is superseded | 2026 | verified (RER rows verified_secondary) - TAR-2026 (a) |
| Cogeneration, CfD, excise | 13,60 / 0,14 / 3,84 RON/MWh | not covered by the pass | 2026 | unverified |
| Public holidays | Codul muncii art. 139 alin. (1) | list matches the consolidated text; 6-7 January by Legea nr. 52/2023 (M. Of. nr. 186/06.03.2023) - secondary copies | in force | verified_secondary |
| FX RON/EUR | 5,5 | workbook D54, Forecast Q3 2026 | - | assumption |
| 2027 values | - | GC quota 2027, GC bounds 2027, network tariffs 2027: not published as of 17.09.2026 | - | not_published |

No regulatory constant is hard-coded in the engine; a change of any value above is a change
of the register (or, in the application, of the user's scenario), never of the code. The
contradicted values stay in the Reference Case register because the parity gate ties that
register to the frozen workbook; bid scenarios adopt the corrections through
`Parameters.apply_bid_defaults()` (D119: reverse charge off after 2026, delegated-PRE BRP
guarantee, shipped ANRE grid table), offered as one button on the Parameters page, and the
page shows non-blocking `regulatory_warnings()` while a scenario still contradicts the verified
state.

## 4. Loading

```
from config.schema import load_parameters
params = load_parameters()            # config/parameters.yaml
params = load_parameters(path)        # any register with the same schema
```

`Parameters.offtaker(code)`, `Parameters.tariff_total_for(o)`, `tso_tariff_for(o)`,
`dso_tariff_for(o)` and `scenario_index` are the accessors the engine uses. Dates are ISO in
the file and `datetime.date` in the dataclasses.

## 5. Scenario file (run-time store, ruling PSTORE)

The application persists the register - with display names - as a versioned scenario file
(`esb.scenario_file`, format `ESB-SCN 1.0`, JSON): `name`, integer `version` (+1 on every
save), `saved_at_utc`, `engine_version`, `notes`, `md5` of the canonical register, `history`
of versions and `parameters` (the register as `Parameters.to_dict()`). A file whose md5 does
not match its register, or whose register fails validation, is refused. Scenario files live
outside the repository; they may carry names (F-035) and must never be committed.
