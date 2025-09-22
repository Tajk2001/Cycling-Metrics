from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
from fitparse import FitFile


def _extract_record_fields(record) -> Dict[str, Optional[float]]:
    data = {"timestamp": None, "power": None, "heart_rate": None, "cadence": None, "speed": None, "distance": None, "altitude": None}
    alt_raw = None
    alt_enh = None
    for field in record:
        name = field.name
        value = field.value
        if name == "timestamp":
            # Ensure timezone-aware then convert to naive UTC for consistency
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                value = value.astimezone(timezone.utc).replace(tzinfo=None)
            data["timestamp"] = value
        elif name == "power":
            data["power"] = float(value) if value is not None else None
        elif name == "heart_rate":
            data["heart_rate"] = float(value) if value is not None else None
        elif name == "cadence":
            data["cadence"] = float(value) if value is not None else None
        elif name == "speed":
            data["speed"] = float(value) if value is not None else None
        elif name == "distance":
            data["distance"] = float(value) if value is not None else None
        elif name == "altitude":
            alt_raw = float(value) if value is not None else None
        elif name == "enhanced_altitude":
            alt_enh = float(value) if value is not None else None
    # Prefer enhanced altitude deterministically
    data["altitude"] = alt_enh if alt_enh is not None else alt_raw
    return data


def load_fit_to_dataframe(file_path: str) -> pd.DataFrame:
    """Load a FIT file into a second-by-second pandas DataFrame.

    Columns: timestamp (datetime), power (W), heart_rate (bpm), cadence (rpm), speed (m/s), distance (m), altitude (m)
    """
    try:
        fit = FitFile(file_path)
    except Exception as e:
        print(f"Warning: failed to open FIT file: {file_path} ({e})")
        return pd.DataFrame(columns=["timestamp", "power", "heart_rate", "cadence", "speed", "distance", "altitude"]).astype({"timestamp": "datetime64[ns]"})

    records = []
    try:
        for message in fit.get_messages("record"):
            row = _extract_record_fields(message)
            if row["timestamp"] is not None:
                records.append(row)
    except Exception as e:
        print(f"Warning: failed to parse records in {file_path} ({e})")

    if not records:
        return pd.DataFrame(columns=["timestamp", "power", "heart_rate", "cadence", "speed", "distance", "altitude"]).astype({"timestamp": "datetime64[ns]"})

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Resample to 1 Hz. Use bounded, signal-appropriate gap handling for research-grade accuracy
    df = df.set_index("timestamp")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1s")
    df = df.reindex(full_index)
    numeric_cols = ["power", "heart_rate", "cadence", "speed", "distance", "altitude"]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if col == "power":
            # Zero-order hold up to 3s; leave longer gaps as NaN
            df[col] = df[col].ffill(limit=3)
        elif col in ("heart_rate", "cadence", "speed"):
            # Time-based interpolation for short gaps up to 5s only; do not extrapolate long gaps
            df[col] = df[col].interpolate(method="time", limit=5, limit_direction="both", limit_area="inside")
        elif col in ("distance", "altitude"):
            # Interpolate continuous signals for gaps up to 10s; avoid extrapolation
            df[col] = df[col].interpolate(method="time", limit=10, limit_direction="both", limit_area="inside")
    df.index.name = "timestamp"
    return df.reset_index()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist even if missing in the source."""
    required = ["timestamp", "power", "heart_rate", "cadence", "speed", "distance", "altitude"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    return df[required]
