# Reference Case fixtures

`rc_v03_series.parquet` - every input series of the frozen Reference Case workbook
(`Energy_Supply_Portfolio_Tracking_v03`, md5 `1c873718bcd08957229f1f46d5449b2a`) on the engine
grid (35.040 rows, 2027): off-taker consumption `OT1`..`OT4` (metered, notified), PV `PV1`
(forecast, forecast deviation, imbalance deviation), wholesale and imbalance prices for the
Central and Low scenarios with the system direction, and the workbook's Peak flag. Values are
exactly as stored in the workbook; zstd-compressed; entities coded, no names.

`rc_v03_series_manifest.json` - provenance, column map to the workbook, sums, fixture md5s.

Built by `tools/build_rc_fixtures.py` (see docs/DATA_CONTRACT.md section 9). Expected output
values for the parity gate are added in Phase 4.
