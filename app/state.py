"""app.state - the session model of the application.

Everything a run needs lives in st.session_state under one key: the scenario file (parameters
with names), the accepted uploads, the assembled series, the last run and the session log.
Nothing is persisted by the application itself; the user saves scenario files and case bundles
(execution prompt section 8.3; ruling PSTORE = versioned scenario file).
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from esb.assemble import AssembledSeries, assemble
from esb.bundle import UploadRecord
from esb.engine import RunResult, run
from esb.importer import import_workbook
from esb.scenario_file import ScenarioFile, reference_case

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SERIES = ROOT / "data" / "reference" / "rc_v03_series.parquet"
KEY = "esb"


@dataclass
class LogEntry:
    at_utc: str
    kind: str
    text: str


@dataclass
class AppState:
    scenario: ScenarioFile
    uploads: list[UploadRecord] = field(default_factory=list)
    use_reference_fixture: bool = True
    series: AssembledSeries | None = None
    result: RunResult | None = None
    selected_offtaker: str = ""
    dirty: bool = True
    log: list[LogEntry] = field(default_factory=list)
    last_error: str = ""
    scenario_notes: str = ""

    # ---- log -------------------------------------------------------------------------------
    def add_log(self, kind: str, text: str) -> None:
        self.log.append(LogEntry(datetime.now(UTC).strftime("%d.%m.%Y %H:%M:%S"), kind, text))

    # ---- parameters --------------------------------------------------------------------------
    @property
    def params(self):
        return self.scenario.parameters

    def mark_dirty(self, why: str = "") -> None:
        self.dirty = True
        if why:
            self.add_log("change", why)

    # ---- series --------------------------------------------------------------------------------
    def reference_frame(self) -> pd.DataFrame | None:
        if not self.use_reference_fixture or not REFERENCE_SERIES.exists():
            return None
        return _load_reference()

    def rebuild_series(self) -> AssembledSeries:
        self.series = assemble(self.params, [(u.result, u.scenario_choice) for u in self.uploads], reference=self.reference_frame())
        return self.series

    def add_upload(self, filename: str, data: bytes, scenario_choice: str | None) -> UploadRecord:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / filename
            p.write_bytes(data)
            res = import_workbook(p)
        rec = UploadRecord(filename=filename, data=data, scenario_choice=scenario_choice, result=res)
        if res.ok:
            # a new delivery of the same class and scenario replaces the previous one
            self.uploads = [u for u in self.uploads
                            if not (u.result.provenance.input_class == res.provenance.input_class
                                    and (u.scenario_choice or "") == (scenario_choice or ""))]
            self.uploads.append(rec)
            self.add_log("upload", f"{filename} accepted ({res.provenance.input_class}, md5 {res.provenance.md5[:8]})")
            self.mark_dirty()
        else:
            self.add_log("refusal", f"{filename} refused: " + "; ".join(c.detail for c in res.checks if not c.passed))
        return rec

    def remove_upload(self, md5: str) -> None:
        before = len(self.uploads)
        self.uploads = [u for u in self.uploads if u.md5 != md5]
        if len(self.uploads) != before:
            self.add_log("upload", f"upload {md5[:8]} removed")
            self.mark_dirty()

    # ---- run -----------------------------------------------------------------------------------
    def run_engine(self, pricing_as_cached: bool = False) -> RunResult | None:
        self.last_error = ""
        try:
            series = self.rebuild_series()
            if not series.coverage.ok:
                self.last_error = "Inputs incomplete: " + " · ".join(series.coverage.problems)
                self.add_log("refusal", self.last_error)
                self.result = None
                return None
            sel = self.selected_offtaker or (self.params.offtakers[0].code if self.params.offtakers else "")
            if sel not in [o.code for o in self.params.offtakers]:
                sel = self.params.offtakers[0].code if self.params.offtakers else ""
            self.selected_offtaker = sel
            self.result = run(series.frame, self.params.copy(), selected_offtaker=sel, pricing_as_cached=pricing_as_cached)
            self.dirty = False
            s = self.result.summary()
            self.add_log("run", f"engine run: scenario '{self.params.scenario_active}', NM forecast {s['nm_forecast']:,.2f} EUR, "
                                f"{self.result.trace[-1][1]:.2f} s")
            return self.result
        except Exception as e:  # the UI never shows a traceback (rule 6)
            self.last_error = f"Run refused: {type(e).__name__}: {e}"
            self.add_log("error", self.last_error)
            self.result = None
            return None

    def series_md5(self) -> str:
        if self.series is None:
            return ""
        h = hashlib.md5()
        h.update(pd.util.hash_pandas_object(self.series.frame, index=False).values.tobytes())
        return h.hexdigest()


@st.cache_data(show_spinner=False)
def _load_reference() -> pd.DataFrame:
    return pd.read_parquet(REFERENCE_SERIES)


def get() -> AppState:
    if KEY not in st.session_state:
        s = AppState(scenario=reference_case())
        s.add_log("session", "session opened; Reference Case register loaded (coded, no names)")
        st.session_state[KEY] = s
    return st.session_state[KEY]


def require_result(state: AppState) -> RunResult | None:
    """Run on demand when the inputs changed; return the result or None (the page shows why)."""
    if state.result is None or state.dirty:
        with st.spinner("Running the engine"):
            state.run_engine()
    return state.result
