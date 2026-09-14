"""esb.importer - standard time-series upload contract (Phase 2).

Validates uploads against the STD_EET_QH contract, applies the DST rules at ingest,
runs the reconciliation checks, never fills a data gap, and returns a typed 366 x 96 frame
with provenance. See docs/DATA_CONTRACT.md.
"""
