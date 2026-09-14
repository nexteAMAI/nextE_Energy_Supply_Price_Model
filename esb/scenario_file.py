"""esb.scenario_file - the versioned scenario file (ruling PSTORE, CEO 14.09.2026).

Every run-time parameter of the application - the full register of config/parameters.yaml plus
the display names of off-takers and counterparties - persists in one JSON document the user
saves and re-loads through the application. The file lives outside the repository (names are
allowed in it, F-035); the repository ships only the coded Reference Case register.

Layout (format ESB-SCN 1.0):
    format, format_version, name, version (integer, +1 on every save of the same name),
    saved_at_utc, engine_version, parameters (dict as Parameters.to_dict()), md5 (of the
    canonical JSON of `parameters`), notes
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from config.schema import Parameters
from esb import __version__

FORMAT = "ESB-SCN"
FORMAT_VERSION = "1.0"


@dataclass
class ScenarioFile:
    name: str
    parameters: Parameters
    version: int = 1
    saved_at_utc: str = ""
    engine_version: str = __version__
    notes: str = ""
    md5: str = ""
    source_filename: str = ""
    history: list[dict] = field(default_factory=list)

    # ---- serialisation -------------------------------------------------------------------
    def to_dict(self) -> dict:
        params = self.parameters.to_dict()
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "version": int(self.version),
            "saved_at_utc": self.saved_at_utc,
            "engine_version": self.engine_version,
            "notes": self.notes,
            "md5": parameters_md5(params),
            "history": list(self.history),
            "parameters": params,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, d: dict, source_filename: str = "") -> ScenarioFile:
        errs = validate_document(d)
        if errs:
            raise ValueError("scenario file refused: " + "; ".join(errs))
        params = Parameters.from_dict(d["parameters"])
        perrs = params.validate()
        if perrs:
            raise ValueError("scenario file refused: " + "; ".join(perrs))
        return cls(
            name=str(d["name"]),
            parameters=params,
            version=int(d.get("version", 1)),
            saved_at_utc=str(d.get("saved_at_utc", "")),
            engine_version=str(d.get("engine_version", "")),
            notes=str(d.get("notes", "")),
            md5=str(d.get("md5", "")),
            source_filename=source_filename,
            history=list(d.get("history", [])),
        )

    @classmethod
    def from_bytes(cls, raw: bytes, source_filename: str = "") -> ScenarioFile:
        try:
            d = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"scenario file refused: not a JSON document ({e})") from e
        return cls.from_dict(d, source_filename)

    @classmethod
    def load(cls, path: str | Path) -> ScenarioFile:
        path = Path(path)
        return cls.from_bytes(path.read_bytes(), path.name)

    # ---- versioning ----------------------------------------------------------------------
    def stamped(self, note: str = "") -> ScenarioFile:
        """A new version of this scenario: version + 1, timestamp, md5, history entry."""
        now = datetime.now(UTC).replace(microsecond=0).isoformat()
        params = self.parameters.to_dict()
        entry = {"version": self.version + 1, "saved_at_utc": now, "md5": parameters_md5(params), "note": note}
        return ScenarioFile(
            name=self.name,
            parameters=self.parameters,
            version=self.version + 1,
            saved_at_utc=now,
            engine_version=__version__,
            notes=self.notes,
            md5=entry["md5"],
            source_filename=self.source_filename,
            history=[*self.history, entry],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self.to_bytes())
        return path

    @property
    def filename(self) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.name).strip("_") or "scenario"
        return f"{safe}_v{self.version:03d}.esb-scn.json"


def parameters_md5(params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def validate_document(d: dict) -> list[str]:
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["document is not an object"]
    if d.get("format") != FORMAT:
        errs.append(f"format '{d.get('format')}' is not {FORMAT}")
    if str(d.get("format_version", "")).split(".")[0] != FORMAT_VERSION.split(".")[0]:
        errs.append(f"format_version '{d.get('format_version')}' not supported (expected {FORMAT_VERSION})")
    if not d.get("name"):
        errs.append("name missing")
    if "parameters" not in d or not isinstance(d["parameters"], dict):
        errs.append("parameters block missing")
    else:
        stored = d.get("md5")
        if stored and stored != parameters_md5(d["parameters"]):
            errs.append("md5 does not match the parameters block (file edited outside the application)")
        for key in ("meta", "scenario", "general", "offtakers", "counterparties", "tariff_components_eur_per_mwh"):
            if key not in d["parameters"]:
                errs.append(f"parameters.{key} missing")
    return errs


def reference_case(name: str = "Reference Case v03") -> ScenarioFile:
    """The coded Reference Case register as scenario version 1 (no names)."""
    from config.schema import load_parameters

    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    p = load_parameters()
    return ScenarioFile(name=name, parameters=p, version=1, saved_at_utc=now, md5=parameters_md5(p.to_dict()),
                        history=[{"version": 1, "saved_at_utc": now, "md5": parameters_md5(p.to_dict()), "note": "Reference Case register"}])
