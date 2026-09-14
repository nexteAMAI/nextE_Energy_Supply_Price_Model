# Reference Case fixtures

`rc_v03_series.parquet` - every input series of the frozen Reference Case workbook
(`Energy_Supply_Portfolio_Tracking_v03`, md5 `1c873718bcd08957229f1f46d5449b2a`) on the engine
grid (35.040 rows, 2027): off-taker consumption `OT1`..`OT4` (metered, notified), PV `PV1`
(forecast, forecast deviation, imbalance deviation), wholesale and imbalance prices for the
Central and Low scenarios with the system direction, and the workbook's Peak flag. Values are
exactly as stored in the workbook; zstd-compressed; entities coded, no names.

`rc_v03_series_manifest.json` - provenance, column map to the workbook, sums, fixture md5s.

Built by `tools/build_rc_fixtures.py` (see docs/DATA_CONTRACT.md section 9).

`rc_v03_expected.parquet` - every cached numeric cell of the workbook's five result sheets
(`Portf Overview`, `Cons_P&L`, `CF_Mth`, `CF_Daily_Ledger`, `Pricing_Calc`; 20.245 rows:
sheet, cell, col, row, value). Values are the workbook's stored results, no labels, no
formulas. Consumed by `esb.parity` and `tests/parity/test_parity_v03.py`; the `Pricing_Calc`
values are the position-3 view and contain the two known defects (X-15 / X-16), which the
engine reproduces only under `pricing_as_cached=True`.
