"""esb.importer - standard time-series upload contract (D-TPL / D-IMP).

Public API:
    build_template(input_class, spine_year, path, ...)  -> write a blank upload template
    write_delivery(path, control, registry, data)       -> write a filled delivery (fixtures, tests)
    import_workbook(path)                                -> ImportResult (checks, frame, provenance)
    preset_registry(input_class, entity_codes)           -> the pre-declared registry per input class

The importer validates the control block, the registry and the raw surface, applies the
daylight-saving rules of the contract for local_clock deliveries, runs the six checks and
refuses on any failure. It never fills a data gap. See docs/DATA_CONTRACT.md.
"""

from esb.importer.contract import Control, Registry, SlotSpec, preset_registry
from esb.importer.importer import ImportResult, Provenance, import_workbook
from esb.importer.template import build_template, raw_frame_fixed_96, write_delivery

__all__ = [
    "Control",
    "Registry",
    "SlotSpec",
    "preset_registry",
    "ImportResult",
    "Provenance",
    "import_workbook",
    "build_template",
    "write_delivery",
    "raw_frame_fixed_96",
]
