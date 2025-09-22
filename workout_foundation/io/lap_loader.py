from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fitparse import FitFile

from ..models.types import Interval


def _normalize_ts(dt_val: Optional[datetime]) -> Optional[datetime]:
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=timezone.utc)
        return dt_val.astimezone(timezone.utc).replace(tzinfo=None)
    return None


def load_fit_laps(file_path: str) -> List[Dict[str, Optional[datetime]]]:
    """Parse FIT 'lap' messages and return a list of dicts with start and end times.

    Fields returned per lap: {start_time, end_time}
    end_time is derived from total_elapsed_time if available; otherwise uses message timestamp.
    """
    try:
        fit = FitFile(file_path)
    except Exception as e:
        print(f"Warning: failed to open FIT file for laps: {file_path} ({e})")
        return []
    laps: List[Dict[str, Optional[datetime]]] = []
    try:
        for msg in fit.get_messages("lap"):
            start_time = None
            end_time = None
            total_elapsed_time = None
            end_ts_field = None
            for field in msg:
                if field.name == "start_time":
                    start_time = _normalize_ts(field.value)
                elif field.name == "total_elapsed_time":
                    total_elapsed_time = float(field.value) if field.value is not None else None
                elif field.name == "timestamp":
                    end_ts_field = _normalize_ts(field.value)
            if start_time is not None:
                if total_elapsed_time is not None and total_elapsed_time > 0:
                    end_time = start_time + timedelta(seconds=float(total_elapsed_time))
                else:
                    end_time = end_ts_field
                laps.append({"start_time": start_time, "end_time": end_time})
    except Exception as e:
        print(f"Warning: failed to parse laps in {file_path} ({e})")
    return laps


def laps_to_intervals(df: pd.DataFrame, laps: List[Dict[str, Optional[datetime]]]) -> List[Interval]:
    """Convert lap times to Intervals aligned with the timeline DataFrame indices."""
    if df.empty or not laps:
        return []
    ts = pd.to_datetime(df["timestamp"])  # already normalized in loader
    intervals: List[Interval] = []
    for lap in laps:
        st = lap.get("start_time")
        et = lap.get("end_time")
        if st is None or et is None:
            continue
        # Find nearest indices within the range
        start_idx = int(ts.searchsorted(st, side="left"))
        end_idx = int(ts.searchsorted(et, side="right") - 1)
        if start_idx < 0 or end_idx < 0 or start_idx >= len(df) or end_idx >= len(df):
            continue
        if end_idx <= start_idx:
            continue
        duration_s = float(end_idx - start_idx + 1)
        intervals.append(
            Interval(
                start_time=df.loc[start_idx, "timestamp"],
                end_time=df.loc[end_idx, "timestamp"],
                start_index=start_idx,
                end_index=end_idx,
                duration_s=duration_s,
                label="lap",
            )
        )
    return intervals

