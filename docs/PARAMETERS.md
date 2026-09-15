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
| `grid_tariffs` | `config/tariffs_ro.yaml` (v1.1 template sheet `Grid_Taxes_RO`, validity 2026) | RON/MWh rows by charge owner (TSO, ANRE, ANAF, each DSO) and component with applicability flags per voltage level; an off-taker with `dso` and `voltage_level` set takes its EUR/MWh components from here by the cascading rule (D109); `source_status: unverified` |
| `offtakers[]` | `Input` section C, B2 rows 97-105, `FW_Purch_Sell_Price`, strip tables | per off-taker: code, `name` (display name, empty in the repository - D95), active, contract window, contract and PV prices, premium components, target GM, strips (BL24 / Peak / OffPeak MW by month), budget and forecast product prices by month, own guarantee, optional tariff set or `dso` + `voltage_level` (D109) |
| `counterparties` | `Input` section D | pv, baseload, spot, brp, tso, dso: active, payment terms (30 / 15 / 0 / 15 / 15 / 15 days), advance (baseload 50 %), guarantee (type, direction, sizing, amount, fee, window), `k` sign convention, baseload deviation percentages |
| `market_guarantees` | `Input` section E | regulatory formula inputs: spot buffer days 4; BRP 9.000 RON/MW and 160 MW generation in the BRP; TSO Vtm 2; DSO Vdm 1 and overdue add-on 0 |

## 2. Guarantee record

`type` (`None`, `Bank Guarantee Letter`, `Cash Collateral`), `direction`, `sizing` (`Fixed`,
`% of Contract Value`, `Coverage Months`, `Dynamic`, `Regulatory formula`), `fixed_amount`,
`coverage_months`, `pct_of_contract_value`, `bgl_fee_pa`, `bgl_fee_type` (`Monthly`,
`One-time at issuance`), `cash_backing_pct`, `start`, `end`. The PV `fixed_amount` is `null`
in the register: the workbook derives it (`Input!C85`) and so does the engine.

## 3. Verification status of regulatory values

Values quoted from the workbook and not yet checked against a primary source carry
`source_status: unverified` in the register. They are to be verified before gate G5 and
recorded here with the primary source.

| Parameter | Value | Workbook citation | Status |
|---|---|---|---|
| VAT rate | 21 % | Legea nr. 227/2015 (Codul fiscal) | unverified |
| CIT rate | 16 % | Legea nr. 227/2015, art. 41 per workbook label | unverified |
| Reverse charge on electricity purchases | applied | art. 331 alin. (2) lit. e) Codul fiscal per workbook | unverified; adviser confirmation outstanding |
| GC quota | 0,499387 GC/MWh | "current regulation, 2026 basis" (workbook D85) | unverified |
| GC reference price | 148,2201 RON/GC | workbook `Cons_P&L!B7` | unverified |
| BRP guarantee rate | 9.000 RON/MW | "Transelectrica BRP rule" | unverified |
| TSO guarantee multiplier Vtm | 2 | "Transelectrica PO 01.13" | unverified |
| DSO guarantee multiplier Vdm | 1 | "ANRE Order 129/2015" | unverified |
| Regulated tariff components | see block | `Input!C54:F63`, vintage "Forecast Q3 2026 basis" (workbook D55 / D60) | unverified |
| FX RON/EUR | 5,5 | workbook D54, Forecast Q3 2026 | assumption |

No regulatory constant is hard-coded in the engine; a change of any value above is a change
of the register (or, in the application, of the user's scenario), never of the code.

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
