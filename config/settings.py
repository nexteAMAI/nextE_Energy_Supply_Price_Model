"""
Central configuration loader for the RO Energy Pricing Engine.

Reads YAML configs (assumptions, curves, datasets, schedule) and environment
variables (API keys). Provides typed access to all model parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_RAW_DIR = _DATA_DIR / "raw"
_PROCESSED_DIR = _DATA_DIR / "processed"
_STATIC_DIR = _DATA_DIR / "static"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"


def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML config file from the config directory."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Lazy singleton — loaded once on first access
# ---------------------------------------------------------------------------

class _Settings:
    """Lazy-loaded configuration singleton."""

    def __init__(self):
        self._loaded = False
        self._assumptions: Dict[str, Any] = {}
        self._curves: Dict[str, Any] = {}
        self._datasets: Dict[str, Any] = {}
        self._schedule: Dict[str, Any] = {}
        self._supply: Dict[str, Any] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # Load API keys from .env
        env_path = _CONFIG_DIR / "api_keys.env"
        if env_path.exists():
            load_dotenv(env_path)

        self._assumptions = _load_yaml("assumptions.yaml")
        self._curves = _load_yaml("curves.yaml")
        self._datasets = _load_yaml("datasets.yaml")
        self._schedule = _load_yaml("schedule.yaml")

        # Supply extension config (optional — graceful fallback if not present)
        supply_path = _CONFIG_DIR / "supply_config.yaml"
        if supply_path.exists():
            self._supply = _load_yaml("supply_config.yaml")
        else:
            self._supply = {}

        self._loaded = True

    # ---- Accessors ----

    @property
    def assumptions(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return self._assumptions

    @property
    def curves(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return self._curves

    @property
    def datasets(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return self._datasets

    @property
    def schedule(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return self._schedule

    @property
    def supply(self) -> Dict[str, Any]:
        """Supply extension configuration (supply_config.yaml)."""
        self._ensure_loaded()
        return self._supply

    def get_supply_param(self, *keys: str, default: Any = None) -> Any:
        """Traverse nested supply config dict by keys."""
        self._ensure_loaded()
        node = self._supply
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    # ---- Convenience getters ----

    def get_assumption(self, *keys: str, default: Any = None) -> Any:
        """Traverse nested assumption dict by keys. E.g. get_assumption('fuel', 'gas', 'ccgt_efficiency')."""
        self._ensure_loaded()
        node = self._assumptions
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def get_dataset_meta(self, name: str) -> Optional[Dict[str, Any]]:
        """Get parsing metadata for a named pre-loaded dataset."""
        self._ensure_loaded()
        return self._datasets.get("datasets", {}).get(name)

    # ---- API Keys (from environment) ----

    @property
    def eq_api_key(self) -> str:
        return os.environ.get("EQ_API_KEY", "")

    @property
    def entsoe_api_key(self) -> str:
        return os.environ.get("ENTSOE_API_KEY", "")

    @property
    def balancing_services_api_key(self) -> str:
        return os.environ.get("BALANCING_SERVICES_API_KEY", "")

    @property
    def jao_api_key(self) -> str:
        return os.environ.get("JAO_API_KEY", "")

    # ---- Directory paths ----

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return _DATA_DIR

    @property
    def raw_dir(self) -> Path:
        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        return _RAW_DIR

    @property
    def processed_dir(self) -> Path:
        _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        return _PROCESSED_DIR

    @property
    def static_dir(self) -> Path:
        _STATIC_DIR.mkdir(parents=True, exist_ok=True)
        return _STATIC_DIR

    @property
    def outputs_dir(self) -> Path:
        _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        return _OUTPUTS_DIR

    # ---- Model parameters (typed shortcuts) ----

    @property
    def timezone(self) -> str:
        return self.get_assumption("model", "timezone", default="Europe/Bucharest")

    @property
    def romania_eic(self) -> str:
        return self.get_assumption("model", "romania_eic", default="10YRO-TEL------P")

    @property
    def peak_hours(self) -> list:
        return self.get_assumption("temporal", "peak_hours", default=list(range(9, 20)))

    @property
    def offpeak_hours(self) -> list:
        return self.get_assumption("temporal", "offpeak_hours",
                                   default=[0,1,2,3,4,5,6,7,20,21,22,23])

    @property
    def vat_rate(self) -> float:
        return self.get_assumption("tariffs", "vat_rate", default=0.21)

    @property
    def ccgt_efficiency(self) -> float:
        return self.get_assumption("fuel", "gas", "ccgt_efficiency", default=0.55)

    @property
    def ocgt_efficiency(self) -> float:
        return self.get_assumption("fuel", "gas", "ocgt_efficiency", default=0.38)

    @property
    def gas_co2_intensity(self) -> float:
        return self.get_assumption("fuel", "gas", "co2_intensity_tco2_per_mwh_th", default=0.37)

    @property
    def hard_coal_efficiency(self) -> float:
        return self.get_assumption("fuel", "coal", "hard_coal_efficiency", default=0.42)

    @property
    def lignite_efficiency(self) -> float:
        return self.get_assumption("fuel", "coal", "lignite_efficiency", default=0.38)

    @property
    def hard_coal_co2_intensity(self) -> float:
        return self.get_assumption("fuel", "coal", "hard_coal_co2_intensity_tco2_per_mwh_th",
                                   default=0.34)

    @property
    def lignite_co2_intensity(self) -> float:
        return self.get_assumption("fuel", "coal", "lignite_co2_intensity_tco2_per_mwh_th",
                                   default=0.40)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
settings = _Settings()
