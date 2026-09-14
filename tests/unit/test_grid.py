from datetime import date, time

from esb import grid


def test_dst_dates_eu_rule():
    assert grid.dst_dates(2026) == (date(2026, 3, 29), date(2026, 10, 25))
    assert grid.dst_dates(2027) == (date(2027, 3, 28), date(2027, 10, 31))
    assert grid.dst_dates(2028) == (date(2028, 3, 26), date(2028, 10, 29))


def test_grid_shape_common_and_leap_year():
    g27 = grid.make_grid(2027)
    assert len(g27) == 365 * 96
    assert g27["seq"].iloc[0] == 1 and g27["seq"].iloc[-1] == 35040
    g28 = grid.make_grid(2028)
    assert len(g28) == 366 * 96


def test_peak_flag_is_positional_intervals_37_to_84():
    g = grid.make_grid(2027)
    per_day = g.groupby("date")["peak"].apply(lambda s: (s == "Peak").sum())
    assert (per_day == 48).all()
    on = sorted(g.loc[g["peak"] == "Peak", "interval"].unique())
    assert on == list(range(37, 85))
    # 08:00 CET = 09:00 EET = interval 37; 20:00 CET = 21:00 EET = interval 85 is Off-Peak
    assert grid.interval_start(37) == time(9, 0)
    assert grid.peak_flag(85) == "Off-Peak"


def test_dst_status_marks_the_four_transition_intervals():
    g = grid.make_grid(2027)
    inj = g[g["dst_status"] == "Injected"]
    mer = g[g["dst_status"] == "Merged"]
    assert list(inj["interval"]) == [13, 14, 15, 16] and inj["date"].dt.date.unique().tolist() == [date(2027, 3, 28)]
    assert list(mer["interval"]) == [13, 14, 15, 16] and mer["date"].dt.date.unique().tolist() == [date(2027, 10, 31)]
    assert list(inj["seq"]) == [8269, 8270, 8271, 8272]  # (87 days) x 96 + 13..16


def test_excel_serial_round_trip():
    assert grid.excel_serial_to_date(46388) == date(2027, 1, 1)
    assert grid.excel_serial_to_date(46752) == date(2027, 12, 31)
    assert grid.date_to_excel_serial(date(2027, 1, 1)) == 46388


def test_interval_of_rejects_off_raster():
    assert grid.interval_of(time(0, 0)) == 1 and grid.interval_of(time(23, 45)) == 96
    try:
        grid.interval_of(time(0, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
