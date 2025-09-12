import numpy as np
import pandas as pd

from IntervalML import extract_lap_intervals_from_data


def _make_df_with_laps():
    n = 600
    idx = pd.date_range("2025-01-01 06:00:00", periods=n, freq="1s")
    rng = np.random.default_rng(7)

    # Create two laps: 0-299 and 300-599
    lap = np.concatenate([np.full(300, 1), np.full(300, 2)])

    power = 160 + rng.normal(0, 10, n)
    # Make lap 2 harder
    power[300:600] += 80
    cadence = 80 + rng.normal(0, 5, n)

    df = pd.DataFrame({"power": power, "cadence": cadence, "lap": lap}, index=idx)
    return df


def test_extract_lap_intervals_returns_work_laps():
    df = _make_df_with_laps()
    intervals = extract_lap_intervals_from_data(df)
    # Expect at least one interval (the hard lap)
    assert isinstance(intervals, list)
    assert len(intervals) >= 1
    for start, end in intervals:
        assert isinstance(start, pd.Timestamp)
        assert isinstance(end, pd.Timestamp)
        assert end > start

