"""esb.catalogue - the parameter catalogue (ruling D-I): unit, standard / default, source and source status per
register path, from config/parameter_catalogue.yaml. Used by the Parameters sheet of the export and the
Parameters page. A path without a rule or entry gets an empty unit - never a guessed one."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

CATALOGUE_YAML = Path(__file__).resolve().parents[1] / "config" / "parameter_catalogue.yaml"


STATUSES = ("verified", "verified_secondary", "contradicted", "unverified", "to_verify", "not_published", "assumption")


@dataclass(frozen=True)
class Entry:
    unit: str = ""
    standard: str = ""
    source: str = ""
    source_status: str = ""
    validity: str = ""      # the period the source gives the value for
    checked: str = ""       # date of the verification pass that read the source
    note: str = ""          # open item or correction pending a ruling

    @property
    def source_text(self) -> str:
        """Source with its status, validity and check date in brackets - the Parameters sheet column."""
        if not self.source:
            return ""
        tags = [t for t in (self.source_status, f"validity {self.validity}" if self.validity else "",
                            f"checked {self.checked}" if self.checked else "") if t]
        text = f"{self.source} [{'; '.join(tags)}]" if tags else self.source
        return f"{text} - {self.note}" if self.note else text


@cache
def _load() -> tuple[list[dict], list[tuple[re.Pattern, dict]]]:
    with open(CATALOGUE_YAML, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    rules = d.get("rules", []) or []
    entries = [(re.compile("^" + e["path"] + "$"), e) for e in d.get("entries", []) or []]
    bad = [e["path"] for _, e in entries if e.get("source_status") and e["source_status"] not in STATUSES]
    if bad:
        raise ValueError(f"unknown source_status in {CATALOGUE_YAML.name}: {bad}")
    return rules, entries


def leaf_of(path: str) -> str:
    return re.sub(r"\[.*?\]", "", path).split(".")[-1]


def unit_by_rule(path: str) -> str:
    rules, _ = _load()
    leaf = leaf_of(path)
    for r in rules:
        if "leaf" in r and leaf == r["leaf"]:
            return str(r["unit"])
    for r in rules:
        if "suffix" in r and leaf.endswith(r["suffix"]):
            return str(r["unit"])
    for r in rules:
        if "contains" in r and r["contains"] in leaf:
            return str(r["unit"])
    for r in rules:
        if "prefix" in r and leaf.startswith(r["prefix"]):
            return str(r["unit"])
    return ""


def entry_for(path: str) -> Entry:
    """The catalogue entry of a register path: explicit entry first (unit falls back to the rule when empty)."""
    _, entries = _load()
    for pat, e in entries:
        if pat.match(path):
            return Entry(unit=str(e.get("unit") or unit_by_rule(path)), standard=str(e.get("standard", "") or ""),
                         source=str(e.get("source", "") or ""), source_status=str(e.get("source_status", "") or ""),
                         validity=str(e.get("validity", "") or ""), checked=str(e.get("checked", "") or ""),
                         note=str(e.get("note", "") or ""))
    return Entry(unit=unit_by_rule(path))


def catalogue_for(paths) -> dict[str, Entry]:
    return {p: entry_for(p) for p in paths}
