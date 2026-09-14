"""Rule book (Imblance_settlement_overview_RO): identities, the 16-case matrix, the workbook blocks,
and a synthetic dual-price case (the loaded scenarios carry Surplus = Deficit, X-19)."""

import numpy as np
import pytest

from esb import imbalance as ib


def test_identities_round_trip_both_conventions():
    for k in (ib.K_DSO, ib.K_BRP):
        metered = np.array([120.0, 80.0, -50.0, 100.0])
        notified = np.array([100.0, 100.0, -40.0, 100.0])  # M = 0 is not invertible by 2.6 (SIGN(0) = 0), as in the workbook
        pct = ib.imbalance_pct(metered, notified, k)
        assert np.allclose(ib.metered_from_pct(notified, pct, k), metered)  # 2.5
        assert np.allclose(ib.notified_from_pct(metered, pct, k), notified)  # 2.6
    # IMB_% is convention-invariant in sign meaning: DSO consumption over-consumption (M > N, +) -> k = -1 -> negative = Deficit
    assert ib.imbalance_pct(np.array([120.0]), np.array([100.0]), ib.K_DSO)[0] < 0
    # the same physical situation in BRP sign (consumption -): M = -120, N = -100 -> k = +1 -> also negative
    assert ib.imbalance_pct(np.array([-120.0]), np.array([-100.0]), ib.K_BRP)[0] < 0


def test_division_by_zero_follows_iferror():
    assert ib.imbalance_pct(np.array([5.0]), np.array([0.0]), -1)[0] == 0.0
    assert ib.notified_from_pct(np.array([0.0]), np.array([0.5]), -1)[0] == 0.0


@pytest.mark.parametrize("dez", [1.0, -1.0])
@pytest.mark.parametrize("price", [50.0, -50.0])
def test_sixteen_case_matrix_outcome_by_sign(dez, price):
    c = ib.case_outcome(dez, price)
    assert c["settlement_eur"] == dez * price
    assert c["outcome"] == ("REVENUE" if dez * price > 0 else "COST")
    assert c["brp_direction"] == ("Surplus" if dez > 0 else "Deficit")


def test_consumption_block_matches_workbook_formulas():
    # FW_Retail_Volume row 7 of position 1 in the Reference Case: metered 0,259, notified 0,2125, prices 160,23 / 160,23
    b = ib.consumption_block([0.259], [0.2125], [160.23], [160.23])
    assert np.isclose(b.imb_mwh[0], -0.0465)  # BRP: (-0,259) - (-0,2125)
    assert np.isclose(b.imb_pct[0], -0.21882352941176478)
    assert np.isclose(b.settlement_eur[0], -7.4506950000000014)  # deficit x Deficit price
    assert b.surplus_mwh[0] == 0.0 and np.isclose(b.deficit_mwh[0], -0.0465)


def test_generation_block_signs_and_specific_settlement():
    g = ib.generation_block([10.0], [-0.1], [0.05], [100.0], [120.0])
    # metered DSO = -(10 + (-0,1) x 10) = -9 ; notified DSO = -9 / (1 + (-1)(0,05)(-1)) = -9/1,05
    assert np.isclose(g.metered_dso[0], -9.0)
    assert np.isclose(g.notified_dso[0], -9.0 / 1.05)
    assert np.isclose(g.metered_brp[0], 9.0) and np.isclose(g.notified_brp[0], 9.0 / 1.05)
    imb = 9.0 - 9.0 / 1.05
    assert np.isclose(g.imb_mwh[0], imb) and imb > 0  # surplus -> Surplus price
    assert np.isclose(g.settlement_eur[0], imb * 100.0)
    assert np.isclose(g.specific_per_notified[0], imb * 100.0 / (9.0 / 1.05))


def test_dual_price_selection_follows_own_direction():
    sur, dfc = np.array([100.0, 100.0]), np.array([180.0, 180.0])
    b = ib.consumption_block([90.0, 110.0], [100.0, 100.0], sur, dfc)
    # row 0: under-consumption -> BRP imbalance +10 (surplus) -> Surplus price; row 1: deficit -> Deficit price
    assert np.isclose(b.settlement_eur[0], 10.0 * 100.0)
    assert np.isclose(b.settlement_eur[1], -10.0 * 180.0)
    # a negative price inverts the outcome
    b2 = ib.consumption_block([90.0], [100.0], [-30.0], [-30.0])
    assert b2.settlement_eur[0] < 0
