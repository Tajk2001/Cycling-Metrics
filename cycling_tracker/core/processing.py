#!/usr/bin/env python3
"""General data cleaning and processing utilities for cycling rides."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_ride_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize a ride DataFrame consistently with training code.

    - Keep rows with valid power; drop NaN power, keep zeros
    - Clip extreme outliers (power < 0 or > 2000; cadence >= 250)
    - Fill missing cadence with 0
    - Compute torque when cadence > 0, else 0
    """
    df_clean = df.copy()

    df_clean = df_clean.dropna(subset=["power"])  # must have power for analysis
    df_clean = df_clean[(df_clean["power"] >= 0) & (df_clean["power"] < 2000)]

    if "cadence" in df_clean.columns:
        df_clean["cadence"] = df_clean["cadence"].fillna(0)
        df_clean = df_clean[df_clean["cadence"] < 250]
    else:
        df_clean["cadence"] = 0

    # Compute torque with safe division
    df_clean["torque"] = np.where(
        (df_clean["cadence"].fillna(0) > 0),
        df_clean["power"].fillna(0) / (df_clean["cadence"].fillna(0) * np.pi / 30.0),
        0.0,
    )

    # Ensure monotonic datetime index if present
    if isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean = df_clean[~df_clean.index.duplicated(keep="first")]
        df_clean = df_clean.sort_index()

    return df_clean

