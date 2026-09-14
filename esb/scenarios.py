"""esb.scenarios - wholesale spot and imbalance price scenarios (workbook `Whol_Sport_Imb_Fcst`).

Three scenario blocks (Aurora Central L:R, Aurora Low T:Z, User Forecast AB:AH) and the ACTIVE
block AJ:AP selected by Input!C6 / C7 (CHOOSE). Curtailed prices are IF(p > 0, p, 0).
The User Forecast direction is derived: IF(Surplus > Deficit, "Positive (Long)", "Negative (Short)").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import Parameters

SCENARIO_PREFIX = {"Aurora Central": "central", "Aurora Low": "low", "User Forecast": "user"}
PRICE_COLUMNS = ("DAM_price_EUR_MWh", "IDCT_VWAP15_price_EUR_MWh", "Surplus_imbalance_price_EUR_MWh", "Deficit_imbalance_price_EUR_MWh")
DIRECTION_COLUMN = "System_imbalance_direction"


def curtailed(p: np.ndarray) -> np.ndarray:
    return np.where(p > 0, p, 0.0)


def scenario_block(series: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """One scenario as the workbook lays it out: DAM, IDCT, curtailed twins, Surplus, Deficit, direction."""
    cols = {c: f"{prefix}__{c}" for c in PRICE_COLUMNS}
    missing = [c for c in cols.values() if c not in series.columns]
    if missing:
        raise KeyError(f"scenario '{prefix}' lacks {missing}")
    dam = series[cols["DAM_price_EUR_MWh"]].to_numpy(dtype=float, na_value=np.nan)
    idct = series[cols["IDCT_VWAP15_price_EUR_MWh"]].to_numpy(dtype=float, na_value=np.nan)
    sur = series[cols["Surplus_imbalance_price_EUR_MWh"]].to_numpy(dtype=float, na_value=np.nan)
    dfc = series[cols["Deficit_imbalance_price_EUR_MWh"]].to_numpy(dtype=float, na_value=np.nan)
    dcol = f"{prefix}__{DIRECTION_COLUMN}"
    if dcol in series.columns and series[dcol].notna().any():
        direction = series[dcol].astype(object).to_numpy()
    else:
        direction = np.where(sur > dfc, "Positive (Long)", "Negative (Short)")
    return pd.DataFrame(
        {
            "dam": dam,
            "idct": idct,
            "dam_curtailed": curtailed(np.nan_to_num(dam)),
            "idct_curtailed": curtailed(np.nan_to_num(idct)),
            "surplus_price": sur,
            "deficit_price": dfc,
            "direction": direction,
        },
        index=series.index,
    )


def active_prices(series: pd.DataFrame, params: Parameters) -> pd.DataFrame:
    """The ACTIVE block (Whol_Sport_Imb_Fcst!AJ:AP) for the selected scenario."""
    prefix = SCENARIO_PREFIX.get(params.scenario_active)
    if prefix is None:
        raise ValueError(f"unknown scenario '{params.scenario_active}'")
    block = scenario_block(series, prefix)
    block.attrs["scenario"] = params.scenario_active
    block.attrs["scenario_index"] = params.scenario_index
    return block


def annual_mean_nonzero(x: pd.Series) -> float:
    """Whol_Sport_Imb_Fcst row 3: IFERROR(AVERAGEIF(range, "<>0"), 0)."""
    v = x.to_numpy(dtype=float)
    v = v[~np.isnan(v) & (v != 0)]
    return float(v.mean()) if len(v) else 0.0
