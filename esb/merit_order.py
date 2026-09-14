"""esb.merit_order - the quarter-hourly engine (workbook sheet `QH_P&L`).

Merit order per quarter-hour: Forward Source 1 (PV) -> Forward Source 2 (Baseload) -> Wholesale
Spot, cascading through the off-takers in their merit-order position (1..n). Surplus PV and
Baseload after the last off-taker are resold at portfolio level; Baseload surplus is attributed
LIFO to the off-takers' own strips (last position first) and costed at the originating
off-taker's product price. Settlement bases follow the workbook: PV / Baseload buys and retail
sells on metered volume, spot buy on notified volume, source imbalance allocated on notified buys.

Column naming: the returned frame carries one column per workbook column that matters, named
by meaning and prefixed with the off-taker code for the per-off-taker blocks
(`OT1_pv_buy_notified` ...). WORKBOOK_COLUMNS maps names to the workbook letters of the
Reference Case layout for traceability (SPEC section 6, X-27).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.schema import Parameters
from esb import imbalance
from esb.scenarios import active_prices
from esb.sources import SourceResult, build_sources

POWER_FACTOR = 4.0  # MW = MWh x 4 (quarter-hour)

# workbook letters (Reference Case, four off-takers) for the portfolio-level columns
WORKBOOK_COLUMNS = {
    "dam": "AA", "idct": "AB", "dam_curtailed": "AC", "idct_curtailed": "AD",
    "surplus_price": "AE", "deficit_price": "AF", "direction": "AG",
    "retail_buy_notified": "AI", "retail_buy_notified_mw": "AJ", "retail_buy_metered": "AK",
    "retail_pv_buy_metered": "AM", "retail_bl_buy_metered": "AO", "retail_spot_settlement": "AQ",
    "retail_cost_budget": "AS", "retail_cost_forecast": "AT", "retail_sell_notified": "AU",
    "retail_sell_metered": "AW", "retail_revenue": "AY", "retail_source_imb": "AZ",
    "retail_offtaker_imb": "BA", "retail_gm1_budget": "BB", "retail_gm1_forecast": "BC",
    "retail_gm2_budget": "BD", "retail_gm2_forecast": "BE",
    "resell_volume": "BF", "resell_cost": "BH", "resell_revenue": "BI", "resell_source_imb": "BJ", "resell_gm2": "BK",
    "total_buy_notified": "BL", "total_buy_metered": "BN", "total_cost_budget": "BP", "total_cost_forecast": "BQ",
    "total_revenue": "BR", "total_imb_all_legs": "BS", "total_gm2_budget": "BT", "total_gm2_forecast": "BU",
    "check_demand": "BV", "check_pv": "BW", "check_bl": "BX", "check_imb": "BY",
    "pv_avail_notified": "CA", "pv_metered": "CC", "pv_ratio": "CE", "pv_specific_imb": "CF",
    "pv_delivered": "CG", "pv_remaining": "CI",
    "bl_avail_notified": "CK", "bl_metered": "CM", "bl_ratio": "CO", "bl_specific_imb": "CP",
    "bl_delivered": "CQ", "bl_remaining": "CS", "spot_buy_notified": "CU",
    "resell_pv_volume": "CX", "resell_pv_cost": "CZ", "resell_pv_revenue": "DA", "resell_pv_source_imb": "DB", "resell_pv_gm2": "DC",
    "resell_bl_volume": "DD", "resell_bl_cost_forecast": "DF", "resell_bl_revenue": "DG", "resell_bl_source_imb": "DH", "resell_bl_gm2": "DI",
    "resell_total_volume": "DJ", "resell_total_cost": "DL", "resell_total_revenue": "DM", "resell_total_source_imb": "DN", "resell_total_gm2": "DO",
    "check_origin": "DX", "resell_bl_cost_budget": "DY",
}
# per off-taker block (letters for position 1; positions 2..4 are shifted by 75 columns each (DZ:GV, GW:JS, JT:MP, MQ:PL))
OFFTAKER_COLUMNS = {
    "price_pv_budget": "DZ", "price_pv_forecast": "EA", "price_bl_budget": "EB", "price_bl_forecast": "EC", "price_sell": "ED",
    "demand_notified": "EE", "demand_metered": "EG", "ratio": "EI", "offtaker_imb": "EJ",
    "buy_notified": "EK", "buy_metered": "EM", "cost_budget": "EO", "cost_forecast": "EP",
    "sell_notified": "EQ", "sell_metered": "ES", "revenue": "EU", "source_imb": "EV",
    "gm1_budget": "EW", "gm1_forecast": "EX", "gm2_budget": "EY", "gm2_forecast": "EZ",
    "pv_avail_before": "FA", "pv_buy_notified": "FC", "pv_buy_metered": "FE", "pv_cost_budget": "FG", "pv_cost_forecast": "FH",
    "pv_sell_metered": "FK", "pv_revenue": "FM", "pv_source_imb": "FN", "pv_gm2_budget": "FO", "pv_gm2_forecast": "FP",
    "bl_avail_before": "FQ", "bl_buy_notified": "FS", "bl_buy_metered": "FU", "bl_cost_budget": "FW", "bl_cost_forecast": "FX",
    "bl_sell_metered": "GA", "bl_revenue": "GC", "bl_source_imb": "GD", "bl_gm2_budget": "GE", "bl_gm2_forecast": "GF",
    "spot_buy_notified": "GG", "spot_cost": "GK", "spot_sell_metered": "GN", "spot_revenue": "GP", "spot_gm": "GQ",
    "pv_remaining_after": "GR", "bl_remaining_after": "GT",
    "strip_notified": "DP",  # own-strip notified volume (DP..DS in the origin-layering block)
    "resell_attributed": "DW",  # LIFO attribution (DW, DV, DU, DT for positions 1..4)
}


@dataclass
class QHResult:
    qh: pd.DataFrame  # one row per quarter-hour
    sources: SourceResult
    offtaker_imbalance: dict[str, imbalance.ConsumptionImbalance]
    codes: list[str]

    def col(self, code: str, name: str) -> pd.Series:
        return self.qh[f"{code}_{name}"]


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return imbalance._safe_div(num, den)


def run_qh(series: pd.DataFrame, grid: pd.DataFrame, params: Parameters, pv_code: str = "PV1") -> QHResult:
    """Run the quarter-hourly engine on standardised series (one row per grid row)."""
    n = len(grid)
    prices = active_prices(series, params)
    src = build_sources(series, grid, prices, params, pv_code=pv_code)
    q: dict[str, np.ndarray] = {}
    dam = prices["dam"].to_numpy(dtype=float)
    idct = prices["idct"].to_numpy(dtype=float)
    dam_c = prices["dam_curtailed"].to_numpy(dtype=float)
    idct_c = prices["idct_curtailed"].to_numpy(dtype=float)
    sur = prices["surplus_price"].to_numpy(dtype=float)
    dfc = prices["deficit_price"].to_numpy(dtype=float)
    q.update(dam=dam, idct=idct, dam_curtailed=dam_c, idct_curtailed=idct_c, surplus_price=sur, deficit_price=dfc)
    q["direction"] = prices["direction"].to_numpy()
    max_curt = np.maximum(dam_c, idct_c)
    both_negative = (dam < 0) & (idct < 0)

    pv_on = 1.0 if params.counterparties["pv"].active else 0.0
    bl_on = 1.0 if params.counterparties["baseload"].active else 0.0

    # ---- sources (CA:CV) ------------------------------------------------------------------
    pv_avail = pv_on * src.pv.notified_brp
    pv_met = pv_on * src.pv.metered_brp
    pv_ratio = _safe_ratio(pv_met, pv_avail)
    pv_spec = pv_on * src.pv.specific_per_notified
    bl_avail = bl_on * src.baseload.notified_brp
    bl_met = bl_on * src.baseload.metered_brp
    bl_ratio = _safe_ratio(bl_met, bl_avail)
    bl_spec = bl_on * src.baseload.specific_per_notified
    q.update(pv_avail_notified=pv_avail, pv_metered=pv_met, pv_ratio=pv_ratio, pv_specific_imb=pv_spec,
             bl_avail_notified=bl_avail, bl_metered=bl_met, bl_ratio=bl_ratio, bl_specific_imb=bl_spec)

    # ---- off-taker cascade ------------------------------------------------------------------
    codes = [o.code for o in params.offtakers]
    imb_blocks: dict[str, imbalance.ConsumptionImbalance] = {}
    pv_before, bl_before = pv_avail, bl_avail
    sums = {k: np.zeros(n) for k in ("pv_delivered", "bl_delivered", "spot", "buy_notified", "buy_metered", "pv_buy_metered",
                                     "bl_buy_metered", "cost_budget", "cost_forecast", "sell_notified", "sell_metered",
                                     "revenue", "source_imb", "offtaker_imb", "demand_notified")}
    for o in params.offtakers:
        c = o.code
        a = 1.0 if o.active else 0.0
        met_dso = series[f"{c}_metered_consumption_MWh"].to_numpy(dtype=float)
        not_dso = series[f"{c}_notified_consumption_MWh"].to_numpy(dtype=float)
        blk = imbalance.consumption_block(met_dso, not_dso, sur, dfc, k=imbalance.K_DSO)
        imb_blocks[c] = blk
        p_pv_b = np.full(n, float(o.pv_price_budget_eur_per_mwh))
        p_pv_f = np.full(n, float(o.pv_price_forecast_eur_per_mwh))
        p_bl_b = src.bl_price_budget[c]
        p_bl_f = src.bl_price_forecast[c]
        p_sell = np.full(n, float(o.contract_price_eur_per_mwh))
        ee = a * not_dso
        eg = a * met_dso
        ei = _safe_ratio(eg, ee)
        ej = a * blk.settlement_eur
        fa = pv_before
        fc = np.minimum(ee, fa)
        fe = fc * pv_ratio
        fg = fe * p_pv_b
        fh = fe * p_pv_f
        fk = fc * ei
        fm = fk * p_sell
        fn = fc * pv_spec
        fo = fm - fg + fn
        fp = fm - fh + fn
        fq = bl_before
        fs = np.minimum(ee - fc, fq)
        fu = fs * bl_ratio
        fw = fu * p_bl_b
        fx = fu * p_bl_f
        ga = fs * ei
        gc = ga * p_sell
        gd = fs * bl_spec
        ge = gc - fw + gd
        gf = gc - fx + gd
        gg = ee - fc - fs
        gk = gg * np.minimum(dam, idct)
        gn = gg * ei
        gp = gn * p_sell
        gq = gp - gk
        gr = fa - fc
        gt = fq - fs
        ek = fc + fs + gg
        em = fe + fu + gg
        eo = fg + fw + gk
        ep = fh + fx + gk
        eq = fc + fs + gg
        es = fk + ga + gn
        eu = fm + gc + gp
        ev = fn + gd
        ew = eu - eo
        ex = eu - ep
        ey = ew + ev + ej
        ez = ex + ev + ej
        block = dict(price_pv_budget=p_pv_b, price_pv_forecast=p_pv_f, price_bl_budget=p_bl_b, price_bl_forecast=p_bl_f,
                     price_sell=p_sell, demand_notified=ee, demand_metered=eg, ratio=ei, offtaker_imb=ej,
                     buy_notified=ek, buy_metered=em, cost_budget=eo, cost_forecast=ep, sell_notified=eq, sell_metered=es,
                     revenue=eu, source_imb=ev, gm1_budget=ew, gm1_forecast=ex, gm2_budget=ey, gm2_forecast=ez,
                     pv_avail_before=fa, pv_buy_notified=fc, pv_buy_metered=fe, pv_cost_budget=fg, pv_cost_forecast=fh,
                     pv_sell_metered=fk, pv_revenue=fm, pv_source_imb=fn, pv_gm2_budget=fo, pv_gm2_forecast=fp,
                     bl_avail_before=fq, bl_buy_notified=fs, bl_buy_metered=fu, bl_cost_budget=fw, bl_cost_forecast=fx,
                     bl_sell_metered=ga, bl_revenue=gc, bl_source_imb=gd, bl_gm2_budget=ge, bl_gm2_forecast=gf,
                     spot_buy_notified=gg, spot_cost=gk, spot_sell_metered=gn, spot_revenue=gp, spot_gm=gq,
                     pv_remaining_after=gr, bl_remaining_after=gt)
        for k, v in block.items():
            q[f"{c}_{k}"] = v
        for k, v in (("pv_delivered", fc), ("bl_delivered", fs), ("spot", gg), ("buy_notified", ek), ("buy_metered", em),
                     ("pv_buy_metered", fe), ("bl_buy_metered", fu), ("cost_budget", eo), ("cost_forecast", ep),
                     ("sell_notified", eq), ("sell_metered", es), ("revenue", eu), ("source_imb", ev),
                     ("offtaker_imb", ej), ("demand_notified", ee)):
            sums[k] = sums[k] + v
        pv_before, bl_before = gr, gt

    pv_remaining, bl_remaining = pv_before, bl_before
    q.update(pv_delivered=sums["pv_delivered"], pv_remaining=pv_remaining, bl_delivered=sums["bl_delivered"],
             bl_remaining=bl_remaining, spot_buy_notified=sums["spot"])

    # ---- wholesale resell (CX:DO) and origin layering (DP:DY) --------------------------------
    cx = pv_remaining
    cz = cx * params.resell_pv_cost_factor * max_curt
    da = cx * params.resell_pv_revenue_factor * max_curt
    db = np.where(both_negative, 0.0, cx * pv_spec)
    dc = da - cz + db
    dd = bl_remaining
    ratio_bl = _safe_ratio(src.baseload.notified_brp, src.baseload.forecast_uncurtailed)
    strip_notified = {c: bl_on * src.strips[c] * ratio_bl for c in codes}
    # LIFO: last position first with MIN; position 1 takes the remainder
    attributed: dict[str, np.ndarray] = {}
    remaining = dd.copy()
    for c in reversed(codes[1:]):
        att = np.minimum(remaining, strip_notified[c])
        attributed[c] = att
        remaining = remaining - att
    if codes:
        attributed[codes[0]] = remaining
    df_cost = np.zeros(n)
    dy_cost = np.zeros(n)
    for c in codes:
        df_cost = df_cost + attributed[c] * src.bl_price_forecast[c]
        dy_cost = dy_cost + attributed[c] * src.bl_price_budget[c]
        q[f"{c}_strip_notified"] = strip_notified[c]
        q[f"{c}_resell_attributed"] = attributed[c]
    dg = dd * max_curt
    dh = dd * bl_spec
    di = dg - df_cost + dh
    total_strip_notified = np.sum(np.vstack([strip_notified[c] for c in codes]), axis=0) if codes else np.zeros(n)
    first_strip = strip_notified[codes[0]] if codes else np.zeros(n)
    dx = np.round(np.abs(total_strip_notified - bl_avail) + np.maximum(0.0, (attributed[codes[0]] if codes else 0.0) - first_strip), 6)
    q.update(resell_pv_volume=cx, resell_pv_cost=cz, resell_pv_revenue=da, resell_pv_source_imb=db, resell_pv_gm2=dc,
             resell_bl_volume=dd, resell_bl_cost_forecast=df_cost, resell_bl_revenue=dg, resell_bl_source_imb=dh, resell_bl_gm2=di,
             resell_bl_cost_budget=dy_cost, check_origin=dx,
             resell_total_volume=cx + dd, resell_total_cost=cz + df_cost, resell_total_revenue=da + dg,
             resell_total_source_imb=db + dh, resell_total_gm2=(da + dg) - (cz + df_cost) + (db + dh))

    # ---- totals (AI:BU) and checks (BV:BY) ----------------------------------------------------
    ai = sums["buy_notified"]
    ak = sums["buy_metered"]
    as_ = sums["cost_budget"]
    at = sums["cost_forecast"]
    au = sums["sell_notified"]
    aw = sums["sell_metered"]
    ay = sums["revenue"]
    az = sums["source_imb"]
    ba = sums["offtaker_imb"]
    bb = ay - as_
    bc = ay - at
    bd = bb + az + ba
    be = bc + az + ba
    bf = q["resell_total_volume"]
    bh = q["resell_total_cost"]
    bi = q["resell_total_revenue"]
    bj = q["resell_total_source_imb"]
    bk = q["resell_total_gm2"]
    bp = as_ + cz + dy_cost
    bq = at + bh
    br = ay + bi
    bs = az + ba + bj
    q.update(retail_buy_notified=ai, retail_buy_notified_mw=ai * POWER_FACTOR, retail_buy_metered=ak,
             retail_pv_buy_metered=sums["pv_buy_metered"], retail_bl_buy_metered=sums["bl_buy_metered"],
             retail_spot_settlement=sums["spot"], retail_cost_budget=as_, retail_cost_forecast=at,
             retail_sell_notified=au, retail_sell_metered=aw, retail_revenue=ay, retail_source_imb=az,
             retail_offtaker_imb=ba, retail_gm1_budget=bb, retail_gm1_forecast=bc, retail_gm2_budget=bd, retail_gm2_forecast=be,
             resell_volume=bf, resell_cost=bh, resell_revenue=bi, resell_source_imb=bj, resell_gm2=bk,
             total_buy_notified=ai + bf, total_buy_metered=ak + bf, total_cost_budget=bp, total_cost_forecast=bq,
             total_revenue=br, total_imb_all_legs=bs, total_gm2_budget=br - bp + bs, total_gm2_forecast=br - bq + bs)
    q["check_demand"] = sums["demand_notified"] - ai
    q["check_pv"] = pv_avail - sums["pv_delivered"] - cx
    q["check_bl"] = bl_avail - sums["bl_delivered"] - dd
    q["check_imb"] = bs - (pv_on * src.pv.settlement_eur - np.where(both_negative, pv_remaining * pv_spec, 0.0)
                           + bl_on * src.baseload.settlement_eur + ba)

    qh = pd.DataFrame(q, index=grid.index)
    for col in ("seq", "date", "interval", "month", "peak"):
        qh.insert(list(("seq", "date", "interval", "month", "peak")).index(col), col, grid[col].values)
    return QHResult(qh=qh, sources=src, offtaker_imbalance=imb_blocks, codes=codes)
