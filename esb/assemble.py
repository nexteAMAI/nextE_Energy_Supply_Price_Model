"""esb.assemble - build the engine's input series frame from accepted uploads.

The engine (esb.engine.run) reads one frame on the grid with the columns of
data/reference/rc_v03_series.parquet: `<code>_metered_consumption_MWh` / `<code>_notified_consumption_MWh`
per off-taker, the three `PV1_*` series and, per scenario, `<prefix>__<price series>`. Uploads
arrive as one ImportResult per workbook (input classes offtaker_load, pv_generation,
baseload_nomination, wholesale_prices). This module joins them, checks that the active
off-takers, the PV source and the active scenario are covered, and refuses with a specific
message otherwise. Missing values stay missing (blank is not zero); a gap in a required series
is a refusal, not a fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.schema import Parameters
from esb import grid
from esb.importer import ImportResult
from esb.scenarios import DIRECTION_COLUMN, PRICE_COLUMNS, SCENARIO_PREFIX

GRID_COLUMNS = ("seq", "date", "interval")


@dataclass
class Coverage:
    offtakers: dict[str, dict[str, bool]] = field(default_factory=dict)  # code -> {metered, notified}
    pv: dict[str, bool] = field(default_factory=dict)
    scenarios: dict[str, bool] = field(default_factory=dict)  # scenario name -> loaded
    notes: list[str] = field(default_factory=list)  # informational (not refusals)
    baseload_nomination: list[str] = field(default_factory=list)  # slot names present (not consumed by the engine)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class AssembledSeries:
    frame: pd.DataFrame
    coverage: Coverage
    sources: list[dict]  # provenance per contributing upload


def scenario_prefix_of(control_scenario: str, explicit: str | None = None) -> str | None:
    """The engine prefix (central / low / user) for a wholesale delivery: the explicit choice from
    the application, else the Std_Control scenario label when it names one unambiguously."""
    if explicit:
        return SCENARIO_PREFIX.get(explicit, explicit if explicit in SCENARIO_PREFIX.values() else None)
    label = (control_scenario or "").lower()
    if "central" in label:
        return "central"
    if "low" in label:
        return "low"
    if "user" in label:
        return "user"
    return None


def assemble(params: Parameters, imports: list[tuple[ImportResult, str | None]],
             reference: pd.DataFrame | None = None) -> AssembledSeries:
    """Join accepted uploads into the engine frame.

    imports: (ImportResult, scenario choice or None) - the choice matters for wholesale_prices only.
    reference: an optional complete frame (the Reference Case fixture) used as the base layer;
    uploads override its columns series by series.
    """
    year = params.spine_year
    g = grid.make_grid(year)
    out = g[list(GRID_COLUMNS)].copy()
    sources: list[dict] = []
    cov = Coverage()
    if reference is not None:
        if len(reference) != len(g):
            raise ValueError(f"reference frame has {len(reference)} rows; the {year} grid has {len(g)}")
        for c in reference.columns:
            if c not in GRID_COLUMNS and c != "peak_v03":
                out[c] = reference[c].to_numpy()
        sources.append({"kind": "reference", "detail": "Reference Case fixture (data/reference/rc_v03_series.parquet)"})

    for res, choice in imports:
        if not res.ok or res.frame is None:
            cov.problems.append(f"{res.provenance.filename}: refused by the importer, not loaded")
            continue
        if res.provenance.spine_year != year:
            cov.problems.append(f"{res.provenance.filename}: spine year {res.provenance.spine_year} differs from the case year {year}")
            continue
        f = res.frame
        if len(f) != len(g):
            cov.problems.append(f"{res.provenance.filename}: {len(f)} rows on a {len(g)}-row grid")
            continue
        cls = res.provenance.input_class
        slot_names = [s.name for s in res.registry.slots]
        if cls == "wholesale_prices":
            blocks = res.provenance.scenario_blocks
            if blocks:  # 1.1 (D112): one file, several scenario blocks keyed by scenario_name
                unmapped = []
                for block in blocks:
                    prefix = scenario_prefix_of(block, choice if len(blocks) == 1 else None)
                    if prefix is None:
                        unmapped.append(block)
                        continue
                    for s in res.registry.slots:
                        if s.scenario_name == block and s.frame_name in f.columns:
                            out[f"{prefix}__{s.name}"] = f[s.frame_name].to_numpy()
                if unmapped:  # informational, not a refusal: the blocks are kept in the upload, unused until the scenario library (Phase 7)
                    cov.notes.append(f"{res.provenance.filename}: scenario block(s) {unmapped} are not one of the application's scenarios "
                                     f"({', '.join(SCENARIO_PREFIX)}) and are not used")
            else:
                prefix = scenario_prefix_of(res.provenance.scenario, choice)
                if prefix is None:
                    cov.problems.append(f"{res.provenance.filename}: scenario not identified (Std_Control scenario "
                                        f"'{res.provenance.scenario}') - choose Aurora Central, Aurora Low or User Forecast")
                    continue
                for name in slot_names:
                    if name in f.columns:
                        out[f"{prefix}__{name}"] = f[name].to_numpy()
        else:
            for name in slot_names:
                if name in f.columns:
                    out[name] = f[name].to_numpy()
            if cls == "baseload_nomination":
                cov.baseload_nomination += [n for n in slot_names if n in f.columns]
        sources.append({"kind": cls, "filename": res.provenance.filename, "md5": res.provenance.md5,
                        "scenario": res.provenance.scenario or ", ".join(res.provenance.scenario_blocks), "time_basis": res.provenance.time_basis,
                        "imported_at_utc": res.provenance.imported_at_utc, "k": res.provenance.k,
                        "not_delivered": list(res.provenance.not_delivered), "rows": int(len(f))})

    # ---- coverage --------------------------------------------------------------------------
    for o in params.offtakers:
        m, n = f"{o.code}_metered_consumption_MWh", f"{o.code}_notified_consumption_MWh"
        cov.offtakers[o.code] = {"metered": m in out.columns, "notified": n in out.columns}
        if o.active:
            for col, what in ((m, "metered"), (n, "notified")):
                if col not in out.columns:
                    cov.problems.append(f"{o.label} ({o.code}) is Active but has no {what} consumption series")
                elif out[col].isna().any():
                    cov.problems.append(f"{o.label} ({o.code}): {int(out[col].isna().sum())} blank quarter-hours in the {what} series")
    pv_cols = ("PV1_forecast_generation_uncurtailed_MWh", "PV1_forecast_generation_deviation_pct", "PV1_imbalance_deviation_pct")
    for c in pv_cols:
        cov.pv[c] = c in out.columns
    if params.counterparties["pv"].active:
        for c in pv_cols:
            if c not in out.columns:
                cov.problems.append(f"PV source is Active but the series {c} is not loaded")
            elif out[c].isna().any():
                cov.problems.append(f"PV series {c}: {int(out[c].isna().sum())} blank quarter-hours")
    for name, prefix in SCENARIO_PREFIX.items():
        have = all(f"{prefix}__{c}" in out.columns for c in PRICE_COLUMNS)
        cov.scenarios[name] = have
    active_prefix = SCENARIO_PREFIX.get(params.scenario_active)
    if active_prefix is None:
        cov.problems.append(f"active scenario '{params.scenario_active}' is not one of {list(SCENARIO_PREFIX)}")
    elif not cov.scenarios.get(params.scenario_active, False):
        cov.problems.append(f"active scenario '{params.scenario_active}' has no wholesale price series loaded")
    else:
        for c in PRICE_COLUMNS:
            col = f"{active_prefix}__{c}"
            if out[col].isna().any():
                cov.problems.append(f"{params.scenario_active}: {int(out[col].isna().sum())} blank quarter-hours in {c}")
        dcol = f"{active_prefix}__{DIRECTION_COLUMN}"
        if dcol not in out.columns:
            out[dcol] = None  # derived by esb.scenarios when absent
    return AssembledSeries(frame=out, coverage=cov, sources=sources)
