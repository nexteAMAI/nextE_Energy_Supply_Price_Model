"""esb.imbalance - the imbalance rule book (workbook sheet `Imblance_settlement_overview_RO`).

Conventions
    DSO metering:   consumption (+), generation (-), k = -1
    BRP imbalance:  generation (+) Sell, consumption (-) Buy, k = +1
    k is declared once per dataset and never inferred from the values.

Identities (sheet rows 2.1-2.6)
    IMB_MWh  = METERED - NOTIFIED
    IMB_%    = k x (METERED - NOTIFIED) / |NOTIFIED|          convention-invariant: < 0 Deficit, > 0 Surplus
    METERED  = NOTIFIED + k x IMB_% x |NOTIFIED|
    NOTIFIED = METERED / (1 + k x IMB_% x SIGN(METERED))
    BRP volumes = -(DSO volumes)

Settlement (rule 9.5, BRP convention, both operands signed)
    SETTLEMENT_EUR = surplus_volume x Surplus_price + deficit_volume x Deficit_price
    with surplus_volume = max(IMB_MWh, 0) and deficit_volume = min(IMB_MWh, 0), IMB_MWh in BRP sign.
    Price selection follows the party's own direction; the system state sets level and sign; a
    negative price inverts the outcome; > 0 is REVENUE, < 0 is COST.

Every function is vectorised over numpy arrays / pandas Series and reproduces the workbook
formulas cell for cell (FW_Retail_Volume blocks AV:BF, FW_Source_Purch_Volume blocks AH:AW and
BA:BP). Division-by-zero cases follow the workbook's IFERROR(..., 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

K_DSO = -1
K_BRP = 1


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """IFERROR(num/den, 0): 0 where den == 0 (Excel also returns 0 for 0/0 through IFERROR)."""
    den = _arr(den)
    out = np.zeros_like(_arr(num))
    ok = den != 0
    out[ok] = _arr(num)[ok] / den[ok]
    return out


# ---- identities --------------------------------------------------------------------------
def imbalance_mwh(metered, notified) -> np.ndarray:
    return _arr(metered) - _arr(notified)


def imbalance_pct(metered, notified, k: int) -> np.ndarray:
    """IFERROR(k x (M - N) / ABS(N), 0)  - FW_Retail_Volume!BB, sheet identity 2.2."""
    m, n = _arr(metered), _arr(notified)
    return _safe_div(k * (m - n), np.abs(n))


def metered_from_pct(notified, pct, k: int) -> np.ndarray:
    """N + k x pct x |N| - identity 2.5."""
    n = _arr(notified)
    return n + k * _arr(pct) * np.abs(n)


def notified_from_pct(metered, pct, k: int) -> np.ndarray:
    """IFERROR(M / (1 + k x pct x SIGN(M)), 0) - identity 2.6, FW_Source_Purch_Volume!AL."""
    m = _arr(metered)
    return _safe_div(m, 1 + k * _arr(pct) * np.sign(m))


def split_surplus_deficit(imb_mwh) -> tuple[np.ndarray, np.ndarray]:
    """(IF(x>0,x,0), IF(x<0,x,0))."""
    x = _arr(imb_mwh)
    return np.where(x > 0, x, 0.0), np.where(x < 0, x, 0.0)


def settlement_eur(imb_mwh_brp, surplus_price, deficit_price) -> np.ndarray:
    """surplus x Surplus_price + deficit x Deficit_price (BRP convention, signed)."""
    s, d = split_surplus_deficit(imb_mwh_brp)
    return s * _arr(surplus_price) + d * _arr(deficit_price)


def direction(imb_pct) -> np.ndarray:
    x = _arr(imb_pct)
    return np.where(x < 0, "Deficit", np.where(x > 0, "Surplus", "Balanced"))


# ---- workbook blocks ---------------------------------------------------------------------
@dataclass
class ConsumptionImbalance:
    """FW_Retail_Volume off-taker block (columns AV:BF): DSO metered/notified in, settlement out."""

    metered_dso: np.ndarray  # AV (+)
    notified_dso: np.ndarray  # AW (+)
    metered_brp: np.ndarray  # AY = -AV
    notified_brp: np.ndarray  # AZ = -AW
    imb_mwh: np.ndarray  # BA = AY - AZ (BRP sign)
    imb_pct: np.ndarray  # BB = k(AV-AW)/|AW|
    surplus_mwh: np.ndarray  # BD
    deficit_mwh: np.ndarray  # BE
    settlement_eur: np.ndarray  # BF = BD x Surplus + BE x Deficit


def consumption_block(metered_dso, notified_dso, surplus_price, deficit_price, k: int = K_DSO) -> ConsumptionImbalance:
    m, n = _arr(metered_dso), _arr(notified_dso)
    m_brp, n_brp = -m, -n
    imb = m_brp - n_brp
    s, d = split_surplus_deficit(imb)
    return ConsumptionImbalance(
        metered_dso=m,
        notified_dso=n,
        metered_brp=m_brp,
        notified_brp=n_brp,
        imb_mwh=imb,
        imb_pct=imbalance_pct(m, n, k),
        surplus_mwh=s,
        deficit_mwh=d,
        settlement_eur=s * _arr(surplus_price) + d * _arr(deficit_price),
    )


@dataclass
class GenerationImbalance:
    """FW_Source_Purch_Volume source block (PV: AH:AW; Baseload: BA:BP): forecast + deviations in,
    BRP metered/notified and settlement out."""

    forecast_uncurtailed: np.ndarray  # AH / BA (+)
    metered_dso: np.ndarray  # AK = -(AH + AJ x |AH|)
    notified_dso: np.ndarray  # AL = IFERROR(AK / (1 + k x AQ x SIGN(AK)), 0)
    metered_brp: np.ndarray  # AN = -AK
    notified_brp: np.ndarray  # AO = -AL
    imb_mwh: np.ndarray  # AP = AN - AO
    surplus_mwh: np.ndarray  # AS
    deficit_mwh: np.ndarray  # AT
    settlement_eur: np.ndarray  # AU = AS x Surplus + AT x Deficit
    specific_per_notified: np.ndarray  # AW = IFERROR(AU / AO, 0)
    specific_per_metered: np.ndarray  # AV = IFERROR(AU / AN, 0) (info)


def generation_block(forecast, forecast_deviation_pct, imbalance_deviation_pct, surplus_price, deficit_price, k: int = K_DSO) -> GenerationImbalance:
    f = _arr(forecast)
    m_dso = -(f + _arr(forecast_deviation_pct) * np.abs(f))
    n_dso = notified_from_pct(m_dso, imbalance_deviation_pct, k)
    m_brp, n_brp = -m_dso, -n_dso
    imb = m_brp - n_brp
    s, d = split_surplus_deficit(imb)
    settle = s * _arr(surplus_price) + d * _arr(deficit_price)
    return GenerationImbalance(
        forecast_uncurtailed=f,
        metered_dso=m_dso,
        notified_dso=n_dso,
        metered_brp=m_brp,
        notified_brp=n_brp,
        imb_mwh=imb,
        surplus_mwh=s,
        deficit_mwh=d,
        settlement_eur=settle,
        specific_per_notified=_safe_div(settle, n_brp),
        specific_per_metered=_safe_div(settle, m_brp),
    )


# ---- the 16-case matrix (sheet rows 64-71 producer, 75-82 consumer) -------------------------
def case_outcome(imb_mwh_brp: float, price: float) -> dict:
    """One row of the rule-book matrix: settlement = DEZ x PRICE, outcome by sign."""
    settlement = imb_mwh_brp * price
    return {
        "brp_direction": "Surplus" if imb_mwh_brp > 0 else ("Deficit" if imb_mwh_brp < 0 else "Balanced"),
        "price_sign": int(np.sign(price)),
        "settlement_eur": settlement,
        "outcome": "REVENUE" if settlement > 0 else ("COST" if settlement < 0 else "NEUTRAL"),
    }


def as_frame(block: ConsumptionImbalance | GenerationImbalance, index=None) -> pd.DataFrame:
    return pd.DataFrame({k: v for k, v in block.__dict__.items()}, index=index)
