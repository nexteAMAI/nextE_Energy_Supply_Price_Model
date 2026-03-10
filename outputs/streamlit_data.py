"""
Streamlit Data Export Module.

Exports processed data as JSON and Parquet files optimized for
Layer 3 Streamlit dashboard consumption.

Output files (Section E.4):
  - streamlit_kpis.json
  - streamlit_dam_timeseries.parquet
  - streamlit_generation_stack.parquet
  - streamlit_imbalance.parquet
  - streamlit_forward_curve.parquet
  - streamlit_cross_border.parquet
  - streamlit_sensitivity.parquet
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tz_strip_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone info for Parquet compatibility (store as UTC-naive after conversion)."""
    result = df.copy()
    if hasattr(result.index, 'tz') and result.index.tz is not None:
        result.index = result.index.tz_convert("UTC").tz_localize(None)
    return result


def export_kpis(
    kpi_data: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Export KPI summary as JSON for the Streamlit dashboard.

    Expected KPI keys:
      - blended_cost_eur_mwh
      - blended_cost_ron_mwh
      - hedging_ratio_pct
      - margin_indicator
      - dam_trailing_6m_avg
      - forward_curve_cal1
      - imbalance_cost_p50
      - total_risk_premium
      - last_updated
    """
    if output_dir is None:
        output_dir = settings.processed_dir
    _ensure_dir(output_dir)

    filepath = output_dir / "streamlit_kpis.json"
    with open(filepath, "w") as f:
        json.dump(kpi_data, f, indent=2, default=str)

    logger.info("Exported KPIs: %d metrics → %s", len(kpi_data), filepath)
    return filepath


def export_timeseries_parquet(
    data: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Export time-series data as Parquet files for Streamlit.

    Parameters
    ----------
    data : dict
        Keys map to output filenames (without extension).
        Expected keys: 'dam_timeseries', 'generation_stack', 'imbalance',
                        'forward_curve', 'cross_border', 'sensitivity'
    """
    if output_dir is None:
        output_dir = settings.processed_dir
    _ensure_dir(output_dir)

    file_map = {
        "dam_timeseries": "streamlit_dam_timeseries.parquet",
        "generation_stack": "streamlit_generation_stack.parquet",
        "imbalance": "streamlit_imbalance.parquet",
        "forward_curve": "streamlit_forward_curve.parquet",
        "cross_border": "streamlit_cross_border.parquet",
        "sensitivity": "streamlit_sensitivity.parquet",
    }

    exported = {}
    for key, filename in file_map.items():
        if key in data and data[key] is not None and not data[key].empty:
            filepath = output_dir / filename
            df = _tz_strip_for_parquet(data[key])
            df.to_parquet(filepath, engine="pyarrow", compression="snappy")
            exported[filename] = filepath
            logger.info("Exported %s: %d rows → %s", key, len(df), filepath)

    logger.info("Streamlit export complete: %d Parquet files", len(exported))
    return exported


def export_contract_summary(
    summary: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Export aggregated contract register summary for Layer 3.
    This is the ONLY data that flows from Layer 2 → Layer 3.

    Expected keys (Section E.5):
      - total_contracted_volume_mwh
      - portfolio_weighted_price_eur
      - hedging_ratio_pct
      - residual_unhedged_volume_mwh
      - contract_count_active
      - breakdown_by_type: {PPA_Fixed: X%, ...}
    """
    if output_dir is None:
        output_dir = settings.processed_dir
    _ensure_dir(output_dir)

    filepath = output_dir / "contract_summary.json"
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Exported contract summary → %s", filepath)
    return filepath
