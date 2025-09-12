#!/usr/bin/env python3
"""Interval detection adapters for the new package.

Provides a thin wrapper around existing ML- and lap-based methods in
`IntervalML.py` to ease migration while enabling modular use.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from ..core.processing import clean_ride_data


def detect_intervals_from_laps(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Extract work intervals from lap numbers using existing logic if available.

    Returns a list of (start_time, end_time) timestamps for laps classified as work.
    """
    try:
        from IntervalML import extract_lap_intervals_from_data  # reuse existing logic
    except Exception:
        return []

    df_clean = clean_ride_data(df)
    return extract_lap_intervals_from_data(df_clean)


def detect_intervals_ml(df: pd.DataFrame, ftp: Optional[float] = None):
    """Call the existing ML-based detector, keeping FTP dynamic if provided.

    Returns (intervals, probabilities, original_df)
    """
    try:
        from interval_detection import find_intervals_ml
    except Exception:
        return [], None, df

    if ftp is None:
        # Estimate FTP dynamically from the ride data to honor user preference
        try:
            from IntervalML import IntervalDetector
            detector = IntervalDetector()
            ftp = detector._estimate_ftp_from_best_efforts(df)
        except Exception:
            ftp = float(np.nan)

    return find_intervals_ml(df, ftp)

