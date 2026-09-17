"""Phase 1 smoke test: the package imports and every engine module is importable."""

import importlib

import esb

MODULES = [
    "esb.grid",
    "esb.importer",
    "esb.imbalance",
    "esb.scenarios",
    "esb.sources",
    "esb.merit_order",
    "esb.pnl",
    "esb.cashflow",
    "esb.guarantees",
    "esb.pricing",
    "esb.reporting",
    "config.schema",
]


def test_version():
    assert esb.__version__ == "0.7.1"


def test_modules_import():
    for name in MODULES:
        importlib.import_module(name)
