#!/usr/bin/env python3
"""
FIT file parsing utilities.

This module provides functions to parse Garmin/ANT+ FIT files into a clean
pd.DataFrame with a datetime index and standard columns such as 'power',
'cadence', 'heart_rate', 'speed', 'lap', and derived 'torque'.

Design goals:
- Depend only on fitparse and pandas/numpy
- Be robust to missing fields; fill sensible defaults
- Avoid hardcoded thresholds; compute from ride data when needed
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    from fitparse import FitFile
except Exception as exc:  # pragma: no cover
    FitFile = None  # type: ignore


_RECORD_FIELDS = [
    "timestamp",
    "power",
    "cadence",
    "heart_rate",
    "speed",
    "distance",
    "altitude",
    "temperature",
    "lap",
]


def parse_fit_file(file_path: str) -> pd.DataFrame:
    """Parse a FIT file into a raw DataFrame without heavy cleaning.

    Returns a DataFrame indexed by timestamp (second resolution if available).
    Missing metrics are included as columns with NaN values.
    """
    if FitFile is None:
        raise ImportError("fitparse is not installed. Please install it from requirements.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FIT file not found: {file_path}")

    fit = FitFile(file_path)

    records: list[Dict[str, Any]] = []
    current_lap_number: Optional[int] = None

    # Map lap start/stop to records by timestamp if present
    lap_events: list[Dict[str, Any]] = []

    for msg in fit.get_messages():
        name = msg.name
        if name == "lap":
            data = {d.name: d.value for d in msg}
            lap_events.append(data)
        elif name == "record":
            row: Dict[str, Any] = {d.name: d.value for d in msg}
            records.append(row)

    if not records:
        return pd.DataFrame(columns=["power", "cadence", "heart_rate", "speed", "distance", "altitude", "temperature", "lap"])  # empty

    df = pd.DataFrame.from_records(records)

    # Convert timestamp to pandas datetime and set as index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        df = df.sort_index()
    else:
        # Create a synthetic timeline at 1s steps
        df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

    # Attach lap numbers if lap messages exist
    if lap_events and isinstance(df.index, pd.DatetimeIndex):
        # Assign incremental lap numbers based on event timestamps
        lap_numbers = np.full(len(df), np.nan)
        for i, lap in enumerate(lap_events, start=1):
            start_ts = pd.to_datetime(lap.get("start_time")) if lap.get("start_time") is not None else None
            end_ts = pd.to_datetime(lap.get("end_time")) if lap.get("end_time") is not None else None
            if start_ts is None:
                # Try using timestamp and total_elapsed_time
                start_ts = pd.to_datetime(lap.get("timestamp")) if lap.get("timestamp") is not None else None
                if start_ts is not None and lap.get("total_elapsed_time") is not None:
                    end_ts = start_ts + pd.to_timedelta(float(lap["total_elapsed_time"]), unit="s")
            if start_ts is None or end_ts is None:
                continue
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            lap_numbers[mask.values] = i
        df["lap"] = lap_numbers
    else:
        if "lap" not in df.columns:
            df["lap"] = np.nan

    # Ensure standard columns exist
    for col in ["power", "cadence", "heart_rate", "speed", "distance", "altitude", "temperature"]:
        if col not in df.columns:
            df[col] = np.nan

    # Convert types and sanitize
    for col in ["power", "cadence", "heart_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "speed" in df.columns:
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce")

    # Derive torque with safe handling
    df["torque"] = np.where(
        (df["cadence"].fillna(0) > 0),
        df["power"].fillna(0) / (df["cadence"].fillna(0) * np.pi / 30.0),
        0.0,
    )

    return df


def load_fit_to_dataframe(file_path: str) -> pd.DataFrame:
    """Parse and lightly clean a FIT file into an analysis-ready DataFrame.

    Cleaning steps:
    - Drop rows with missing power; fill cadence with 0
    - Remove extreme outliers (power < 0 or > 2000; cadence >= 250)
    - Ensure datetime index is monotonic
    - Compute torque if missing
    """
    df = parse_fit_file(file_path)

    if df.empty:
        return df

    # Drop rows with missing power but keep zeros (coasting)
    df = df.dropna(subset=["power"])
    df = df[(df["power"] >= 0) & (df["power"] < 2000)]

    # Cadence cleaning
    if "cadence" in df.columns:
        df["cadence"] = df["cadence"].fillna(0)
        df = df[df["cadence"] < 250]
    else:
        df["cadence"] = 0

    # Recompute torque consistently
    df["torque"] = np.where(
        (df["cadence"].fillna(0) > 0),
        df["power"].fillna(0) / (df["cadence"].fillna(0) * np.pi / 30.0),
        0.0,
    )

    # Ensure strictly increasing index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

    return df

