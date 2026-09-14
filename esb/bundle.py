"""esb.bundle - the case bundle (execution prompt section 8.3).

A case is the set of uploads + the scenario file + the run selection that produced a result.
The bundle is one zip the user downloads and can re-upload to reproduce the run exactly:

    manifest.json        format, created_at, engine version, md5 of every member, run selection
    scenario.json        the versioned scenario file (esb.scenario_file)
    uploads/<file>.xlsx  every accepted upload, byte-identical, with its scenario choice in the manifest
    provenance.json      importer provenance of every upload (checks, k, time basis)
    summary.json         headline results of the run the bundle was taken from (for the tie-out on reload)

Re-opening a bundle re-imports every upload through the importer (same checks) and re-runs the
engine; `reproduce` compares the new headline values with summary.json at the parity tolerance.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from esb import __version__
from esb.importer import ImportResult, import_workbook
from esb.parity import TOL_FLOOR, TOL_REL
from esb.scenario_file import ScenarioFile

FORMAT = "ESB-CASE"
FORMAT_VERSION = "1.0"


@dataclass
class UploadRecord:
    filename: str
    data: bytes
    scenario_choice: str | None
    result: ImportResult

    @property
    def md5(self) -> str:
        return hashlib.md5(self.data).hexdigest()


@dataclass
class CaseBundle:
    scenario: ScenarioFile
    uploads: list[UploadRecord]
    selected_offtaker: str
    use_reference_fixture: bool
    summary: dict[str, float] = field(default_factory=dict)
    created_at_utc: str = ""
    engine_version: str = __version__
    manifest: dict = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        now = datetime.now(UTC).replace(microsecond=0).isoformat()
        scn = self.scenario.to_bytes()
        members: dict[str, bytes] = {"scenario.json": scn}
        uploads_manifest = []
        for u in self.uploads:
            name = f"uploads/{u.filename}"
            members[name] = u.data
            uploads_manifest.append({"member": name, "filename": u.filename, "md5": u.md5,
                                     "input_class": u.result.provenance.input_class, "scenario_choice": u.scenario_choice})
        members["provenance.json"] = json.dumps([json.loads(u.result.provenance.to_json()) for u in self.uploads],
                                                indent=2, ensure_ascii=False).encode("utf-8")
        members["summary.json"] = json.dumps(self.summary, indent=2).encode("utf-8")
        manifest = {
            "format": FORMAT, "format_version": FORMAT_VERSION, "created_at_utc": now, "engine_version": self.engine_version,
            "scenario": {"name": self.scenario.name, "version": self.scenario.version, "md5": hashlib.md5(scn).hexdigest()},
            "uploads": uploads_manifest,
            "use_reference_fixture": bool(self.use_reference_fixture),
            "selected_offtaker": self.selected_offtaker,
            "members_md5": {k: hashlib.md5(v).hexdigest() for k, v in members.items()},
            "confidential": "CONFIDENTIAL - nextE",
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            for k, v in members.items():
                z.writestr(k, v)
        return buf.getvalue()

    @property
    def filename(self) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.scenario.name).strip("_") or "case"
        return f"case_{safe}_v{self.scenario.version:03d}_{stamp}.esb-case.zip"


def read_bundle(raw: bytes) -> CaseBundle:
    """Open a bundle: verify member md5s, load the scenario file, re-import every upload."""
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as e:
        raise ValueError(f"case bundle refused: not a zip archive ({e})") from e
    names = set(z.namelist())
    if "manifest.json" not in names:
        raise ValueError("case bundle refused: manifest.json missing")
    manifest = json.loads(z.read("manifest.json").decode("utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"case bundle refused: format '{manifest.get('format')}' is not {FORMAT}")
    for member, md5 in manifest.get("members_md5", {}).items():
        if member not in names:
            raise ValueError(f"case bundle refused: member {member} missing")
        actual = hashlib.md5(z.read(member)).hexdigest()
        if actual != md5:
            raise ValueError(f"case bundle refused: member {member} altered (md5 {actual} vs manifest {md5})")
    scenario = ScenarioFile.from_bytes(z.read("scenario.json"), "scenario.json")
    uploads: list[UploadRecord] = []
    with tempfile.TemporaryDirectory() as tmp:
        for u in manifest.get("uploads", []):
            data = z.read(u["member"])
            path = Path(tmp) / u["filename"]
            path.write_bytes(data)
            res = import_workbook(path)
            uploads.append(UploadRecord(filename=u["filename"], data=data, scenario_choice=u.get("scenario_choice"), result=res))
    summary = json.loads(z.read("summary.json").decode("utf-8")) if "summary.json" in names else {}
    return CaseBundle(scenario=scenario, uploads=uploads, selected_offtaker=str(manifest.get("selected_offtaker", "")),
                      use_reference_fixture=bool(manifest.get("use_reference_fixture", False)), summary=summary,
                      created_at_utc=str(manifest.get("created_at_utc", "")), engine_version=str(manifest.get("engine_version", "")),
                      manifest=manifest)


def compare_summary(stored: dict[str, float], fresh: dict[str, float]) -> list[dict]:
    """Headline values of the stored run vs the reproduced run at the parity tolerance."""
    rows = []
    for k, v in stored.items():
        f = fresh.get(k)
        if f is None:
            rows.append({"key": k, "stored": v, "reproduced": None, "status": "MISSING"})
            continue
        tol = TOL_REL * max(abs(float(v)), TOL_FLOOR)
        rows.append({"key": k, "stored": float(v), "reproduced": float(f), "status": "PASS" if abs(float(f) - float(v)) <= tol else "FAIL"})
    return rows
