"""Build the Reference Case input fixtures from the cell-level extracts of the frozen workbook.

Inputs (not in the repository): three CSVs produced by streaming the sheet XML of
`Energy_Supply_Portfolio_Tracking_v03_[base].xlsx` (md5 1c873718bcd08957229f1f46d5449b2a),
rows 1..35142, one column per extracted sheet column, values as stored:
  whol.csv   - Whol_Sport_Imb_Fcst: A..Z, AJ..AP
  src.csv    - FW_Source_Purch_Volume: I, U, AH.., BW
  retail.csv - FW_Retail_Volume: I, U, AF, AG, AV.., CV

Outputs:
  <repo>/data/reference/rc_v03_series.parquet       - all input series on the engine grid (coded)
  <repo>/data/reference/rc_v03_series_manifest.json - provenance, column map, sums
  <out>/RC_v03_<class>_2027.xlsx                    - the same series as standard upload workbooks
                                                      (time_basis fixed_96), one per input class
  <out>/RC_v03_derived_series.csv.gz                - workbook-derived columns kept for Phase 4
                                                      verification (not in the repository)

Usage: python tools/build_rc_fixtures.py <csv_dir> <out_dir> [<repo_root>]
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from esb import grid
from esb.importer import Control, import_workbook, preset_registry, write_delivery

YEAR = 2027
V03_MD5 = "1c873718bcd08957229f1f46d5449b2a"
BLANK_BLOCK = (5671, 5766)  # the 96 placeholder rows of 29.02 in the workbook grid
FIRST_DATA_ROW = 7

# workbook column -> fixture series name (off-takers by merit-order position: OT1..OT4)
RETAIL_MAP = {
    "AV": "OT1_metered_consumption_MWh",
    "AW": "OT1_notified_consumption_MWh",
    "BJ": "OT2_metered_consumption_MWh",
    "BK": "OT2_notified_consumption_MWh",
    "BX": "OT3_metered_consumption_MWh",
    "BY": "OT3_notified_consumption_MWh",
    "CL": "OT4_metered_consumption_MWh",
    "CM": "OT4_notified_consumption_MWh",
}
PV_MAP = {
    "AH": "PV1_forecast_generation_uncurtailed_MWh",
    "AJ": "PV1_forecast_generation_deviation_pct",
    "AQ": "PV1_imbalance_deviation_pct",
}
PRICE_MAP = {
    "central": {"L": "DAM_price_EUR_MWh", "M": "IDCT_VWAP15_price_EUR_MWh", "P": "Surplus_imbalance_price_EUR_MWh", "Q": "Deficit_imbalance_price_EUR_MWh", "R": "System_imbalance_direction"},
    "low": {"T": "DAM_price_EUR_MWh", "U": "IDCT_VWAP15_price_EUR_MWh", "X": "Surplus_imbalance_price_EUR_MWh", "Y": "Deficit_imbalance_price_EUR_MWh", "Z": "System_imbalance_direction"},
}
DERIVED_RETAIL = ["AF", "AG", "AX", "BB", "BF", "BT", "CH", "CV"]
DERIVED_SRC = ["AK", "AL", "AM", "AN", "AO", "AP", "AS", "AT", "AU", "AW", "BA", "BC", "BG", "BH", "BJ", "BN", "BP", "BT", "BU", "BV", "BW"]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dated_rows(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Keep the 35.040 dated engine rows, attach date and interval (position within the day)."""
    d = df.copy()
    d["row"] = d["row"].astype(int)
    d = d[(d["row"] >= FIRST_DATA_ROW) & ~d["row"].between(*BLANK_BLOCK)].copy()
    assert (d[date_col] != "").all(), "blank date on a dated row"
    d["date"] = [grid.excel_serial_to_date(float(v)) for v in d[date_col]]
    d["interval"] = d.groupby("date").cumcount() + 1
    assert len(d) == 365 * 96 and d.groupby("date").size().eq(96).all()
    return d.reset_index(drop=True)


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="raise")


def main(csv_dir: Path, out_dir: Path, repo: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (repo / "data" / "reference").mkdir(parents=True, exist_ok=True)
    whol = dated_rows(pd.read_csv(csv_dir / "whol.csv", dtype=str, keep_default_na=False), "C")
    src = dated_rows(pd.read_csv(csv_dir / "src.csv", dtype=str, keep_default_na=False), "I")
    retail = dated_rows(pd.read_csv(csv_dir / "retail.csv", dtype=str, keep_default_na=False), "I")
    # the three sheets share the row layout: same dates and the workbook's own interval column agrees
    assert (whol["date"].values == src["date"].values).all() and (whol["date"].values == retail["date"].values).all()
    assert (whol["D"].astype(int).values == whol["interval"].values).all()

    g = grid.make_grid(YEAR)
    assert (g["date"].dt.date.values == whol["date"].values).all() and (g["interval"].values == whol["interval"].values).all()
    assert (g["peak"].values == whol["F"].values).all(), "workbook Peak flag differs from the grid"
    assert (whol["F"].values == src["U"].values).all() and (whol["F"].values == retail["U"].values).all()

    series = pd.DataFrame({"seq": g["seq"], "date": g["date"], "interval": g["interval"], "peak_v03": whol["F"].values})
    for col, name in RETAIL_MAP.items():
        series[name] = numeric(retail[col]).values
    for col, name in PV_MAP.items():
        series[name] = numeric(src[col]).values
    for scen, m in PRICE_MAP.items():
        for col, name in m.items():
            series[f"{scen}__{name}"] = whol[col].values if name.endswith("direction") else numeric(whol[col]).values

    # --- parquet + manifest --------------------------------------------------------------
    pq = repo / "data" / "reference" / "rc_v03_series.parquet"
    series.to_parquet(pq, engine="pyarrow", compression="zstd", index=False)
    sums = {c: float(series[c].sum()) for c in series.columns if series[c].dtype.kind == "f"}
    manifest = {
        "source_workbook": "Energy_Supply_Portfolio_Tracking_v03_[base].xlsx",
        "source_md5": V03_MD5,
        "spine_year": YEAR,
        "rows": int(len(series)),
        "grid": "365 x 96 dated rows of the workbook (rows 7..35142 minus the 96 placeholder rows 5671..5766)",
        "entity_codes": {"OT1": "merit-order position 1", "OT2": "position 2", "OT3": "position 3", "OT4": "position 4", "PV1": "Forward Source 1 (PV)"},
        "column_map": {
            "FW_Retail_Volume": RETAIL_MAP,
            "FW_Source_Purch_Volume": PV_MAP,
            "Whol_Sport_Imb_Fcst": {scen: m for scen, m in PRICE_MAP.items()},
        },
        "sums": sums,
        "note": "The parquet is the canonical fixture (values exactly as stored in the workbook). The xlsx fixtures are "
                "written with openpyxl (16 significant digits) and may differ from the parquet by 1 ulp (relative 1e-16).",
        "built_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }

    # --- fixture workbooks (fixed_96) -------------------------------------------------------
    base = pd.DataFrame(
        {
            "Date_EET": [d.date() for d in g["date"]],
            "Start_EET": [t.time() for t in g["start_eet"]],
            "End_EET": [t.time() for t in g["end_eet"]],
        }
    )
    fixtures = {}

    def fixture(name: str, input_class: str, reg, cols: dict[str, str], scenario: str = ""):
        data = base.copy()
        for series_col, slot_name in cols.items():
            data[slot_name] = series[series_col].values
        ctrl = Control(input_class=input_class, spine_year=YEAR, time_basis="fixed_96", scenario=scenario,
                       provider="Reference Case v03 (workbook extract)", delivery_date=date.today().strftime("%d.%m.%Y"),
                       notes=f"Extracted from the frozen Reference Case workbook, md5 {V03_MD5}")
        path = write_delivery(out_dir / f"{name}.xlsx", ctrl, reg, data)
        res = import_workbook(path)
        assert res.ok, res.summary()
        for series_col, slot_name in cols.items():
            a, b = res.frame[slot_name], series[series_col]
            if a.dtype.kind == "f":
                # openpyxl serialises floats with 16 significant digits ("%.16g"), so a workbook
                # written here can differ from the parquet by one unit in the last place
                # (relative 1e-16); Excel itself writes 17 digits. Far below the 1e-6 gate.
                same = np.isnan(a.values) & np.isnan(b.values)
                assert np.allclose(a.values[~same], b.values[~same], rtol=1e-12, atol=0), slot_name
            else:
                assert (a.values == b.values).all(), slot_name
        fixtures[path.name] = {"md5": md5(path), "size": path.stat().st_size, "checks": [c.status for c in res.checks]}
        print(res.summary())

    fixture("RC_v03_offtaker_load_2027", "offtaker_load", preset_registry("offtaker_load", ["OT1", "OT2", "OT3", "OT4"]), {v: v for v in RETAIL_MAP.values()})
    fixture("RC_v03_pv_generation_2027", "pv_generation", preset_registry("pv_generation", ["PV1"]), {v: v for v in PV_MAP.values()})
    for scen in ("central", "low"):
        fixture(f"RC_v03_wholesale_prices_{scen}_2027", "wholesale_prices", preset_registry("wholesale_prices"),
                {f"{scen}__{n}": n for n in PRICE_MAP[scen].values()}, scenario=f"Aurora {scen.capitalize()} (as loaded in v03)")
    manifest["fixture_workbooks"] = fixtures
    (repo / "data" / "reference" / "rc_v03_series_manifest.json").write_text(json.dumps(manifest, indent=2))

    # --- derived columns for Phase 4 (CEO folder only) --------------------------------------
    derived = pd.DataFrame({"seq": g["seq"], "date": g["date"].dt.date, "interval": g["interval"]})
    for c in DERIVED_RETAIL:
        derived[f"FW_Retail_Volume!{c}"] = numeric(retail[c]).values
    for c in DERIVED_SRC:
        derived[f"FW_Source_Purch_Volume!{c}"] = numeric(src[c]).values
    for c in ["N", "O", "V", "W", "AJ", "AK", "AL", "AM", "AN", "AO"]:
        derived[f"Whol_Sport_Imb_Fcst!{c}"] = numeric(whol[c]).values
    derived["Whol_Sport_Imb_Fcst!AP"] = whol["AP"].values
    derived.to_csv(out_dir / "RC_v03_derived_series.csv.gz", index=False, compression="gzip")
    print("parquet", pq, pq.stat().st_size, "bytes; md5", md5(pq))


if __name__ == "__main__":
    csv_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    repo = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[1]
    main(csv_dir, out_dir, repo)
