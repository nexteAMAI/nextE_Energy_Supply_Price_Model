"""
Excel Export Module.

Writes Layer 1 processed outputs as clean CSVs for Layer 2 (Excel) consumption.
Files are designed to be imported via Excel Power Query or pasted into named ranges.

Output files (Section E.4):
  - dam_monthly_summary.csv
  - dam_hourly_latest.csv
  - idm_monthly_spread.csv
  - commodities_daily.csv
  - srmc_daily.csv
  - forward_curve_latest.csv
  - imbalance_monthly_stats.csv
  - fx_daily.csv
  - aurora_forecast.csv
  - generation_monthly.csv
  - cross_border_monthly.csv
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_for_excel(
    results: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Export all processed datasets as CSVs for Layer 2 Excel consumption.

    Parameters
    ----------
    results : dict
        Keys map to the standardized output filenames.
        Expected keys: 'dam_monthly', 'dam_hourly', 'idm_monthly',
                        'srmc_daily', 'imbalance_monthly', 'fx_daily',
                        'aurora_forecast', 'generation_monthly', 'cross_border_monthly',
                        'forward_curve'.
    output_dir : Path, optional
        Output directory. Defaults to data/processed/.

    Returns
    -------
    Dict mapping filename → absolute Path of exported file.
    """
    if output_dir is None:
        output_dir = settings.processed_dir
    _ensure_dir(output_dir)

    file_map = {
        "dam_monthly": "dam_monthly_summary.csv",
        "dam_hourly": "dam_hourly_latest.csv",
        "idm_monthly": "idm_monthly_spread.csv",
        "commodities_daily": "commodities_daily.csv",
        "srmc_daily": "srmc_daily.csv",
        "forward_curve": "forward_curve_latest.csv",
        "imbalance_monthly": "imbalance_monthly_stats.csv",
        "fx_daily": "fx_daily.csv",
        "aurora_forecast": "aurora_forecast.csv",
        "generation_monthly": "generation_monthly.csv",
        "cross_border_monthly": "cross_border_monthly.csv",
    }

    exported = {}
    for key, filename in file_map.items():
        if key in results and results[key] is not None and not results[key].empty:
            filepath = output_dir / filename
            df = results[key]

            # Ensure index is written
            df.to_csv(filepath, index=True, float_format="%.4f")
            exported[filename] = filepath
            logger.info("Exported %s: %d rows → %s", key, len(df), filepath)
        else:
            logger.debug("Skipped %s (not in results or empty)", key)

    logger.info("Excel export complete: %d files written to %s",
                len(exported), output_dir)
    return exported


def export_dam_hourly_latest(
    dam: pd.DataFrame,
    output_dir: Optional[Path] = None,
    days: int = 90,
) -> Path:
    """Export last N days of hourly DAM prices for Layer 2."""
    if output_dir is None:
        output_dir = settings.processed_dir
    _ensure_dir(output_dir)

    cutoff = dam.index.max() - pd.Timedelta(days=days)
    recent = dam[dam.index >= cutoff].copy()

    filepath = output_dir / "dam_hourly_latest.csv"
    recent.to_csv(filepath, float_format="%.2f")
    logger.info("Exported dam_hourly_latest: %d rows (last %d days)", len(recent), days)
    return filepath
