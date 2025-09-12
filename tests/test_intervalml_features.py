import numpy as np
import pandas as pd

from IntervalML import IntervalDetector


def _make_dummy_df(n=600):
    # 10 minutes of 1 Hz data
    idx = pd.date_range("2025-01-01 06:00:00", periods=n, freq="1s")
    rng = np.random.default_rng(42)
    power = 180 + rng.normal(0, 20, n)
    cadence = 85 + rng.normal(0, 5, n)
    torque = power / (cadence * np.pi / 30)
    return pd.DataFrame({"power": power, "cadence": cadence, "torque": torque}, index=idx)


def test_create_features_shape_and_no_nans():
    df = _make_dummy_df()
    det = IntervalDetector()
    feats = det.create_features(df, estimated_ftp=250)

    # Basic expectations
    assert len(feats) == len(df)
    assert feats.isna().sum().sum() == 0

    # Presence of some key engineered columns
    expected_cols = [
        "power_normalized",
        "power_relative_to_avg",
        "cadence_normalized",
        "power_cadence_product",
    ]
    for col in expected_cols:
        assert col in feats.columns


def test_estimate_ftp_is_positive_and_reasonable():
    df = _make_dummy_df(n=3600)  # 1 hour
    det = IntervalDetector()
    ftp = det._estimate_ftp_from_best_efforts(df)
    assert ftp > 0
    # Should be within a plausible athletic range for this synthetic data
    assert 100 <= ftp <= 400

