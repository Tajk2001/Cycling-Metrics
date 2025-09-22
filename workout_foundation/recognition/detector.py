from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models.types import Interval, IntervalSet


def _smooth(series: pd.Series, window_s: int) -> pd.Series:
    if series.empty:
        return series
    return series.rolling(window=window_s, min_periods=max(1, window_s // 2), center=False).mean()


def detect_work_intervals(
    df: pd.DataFrame,
    ftp_watts: float,
    min_work_s: int = 20,
    min_gap_s: int = 10,
    work_threshold_relative_ftp: float = 0.85,
    smoothing_s: int = 5,
) -> List[Interval]:
    """Detect work intervals using an FTP-relative threshold (no hardcoded absolute watts).

    - Work begins when smoothed power crosses above threshold
    - Ends when smoothed power drops below threshold for at least min_gap_s
    - Merge short gaps within intervals
    """
    if df.empty or "power" not in df:
        return []

    df = df.copy()
    df["power_smooth"] = _smooth(df["power"], smoothing_s)
    threshold = ftp_watts * work_threshold_relative_ftp
    above = df["power_smooth"] >= threshold

    intervals: List[Interval] = []
    in_interval = False
    start_idx: Optional[int] = None
    gap_counter = 0

    for i, is_above in enumerate(above.values):
        if not in_interval:
            if is_above:
                in_interval = True
                start_idx = i
                gap_counter = 0
        else:
            if is_above:
                gap_counter = 0
            else:
                gap_counter += 1
                if gap_counter >= min_gap_s:
                    end_idx = i - gap_counter
                    if start_idx is not None and end_idx > start_idx:
                        duration_s = end_idx - start_idx + 1
                        if duration_s >= min_work_s:
                            start_time = df.loc[start_idx, "timestamp"]
                            end_time = df.loc[end_idx, "timestamp"]
                            intervals.append(
                                Interval(
                                    start_time=start_time,
                                    end_time=end_time,
                                    start_index=int(start_idx),
                                    end_index=int(end_idx),
                                    duration_s=float(duration_s),
                                    label="work",
                                )
                            )
                    in_interval = False
                    start_idx = None
                    gap_counter = 0

    # Close trailing interval
    if in_interval and start_idx is not None:
        end_idx = len(df) - 1
        duration_s = end_idx - start_idx + 1
        if duration_s >= min_work_s:
            intervals.append(
                Interval(
                    start_time=df.loc[start_idx, "timestamp"],
                    end_time=df.loc[end_idx, "timestamp"],
                    start_index=int(start_idx),
                    end_index=int(end_idx),
                    duration_s=float(duration_s),
                    label="work",
                )
            )

    return intervals


def _group_by_duration_and_intensity(
    df: pd.DataFrame,
    intervals: List[Interval],
    power_tolerance_pct: float = 15.0,
    duration_tolerance_pct: float = 20.0,
) -> List[List[Interval]]:
    groups: List[List[Interval]] = []
    for itv in intervals:
        seg = df.iloc[itv.start_index : itv.end_index + 1]
        avg_power = float(seg["power"].mean()) if "power" in seg else np.nan
        placed = False
        for g in groups:
            g0 = g[0]
            seg0 = df.iloc[g0.start_index : g0.end_index + 1]
            avg0 = float(seg0["power"].mean()) if "power" in seg0 else np.nan
            dur_match = abs(itv.duration_s - g0.duration_s) <= (duration_tolerance_pct / 100.0) * g0.duration_s
            pwr_match = (
                (np.isnan(avg_power) and np.isnan(avg0))
                or (avg0 == 0 and avg_power == 0)
                or (avg0 != 0 and abs(avg_power - avg0) <= (power_tolerance_pct / 100.0) * avg0)
            )
            if dur_match and pwr_match:
                g.append(itv)
                placed = True
                break
        if not placed:
            groups.append([itv])
    # Sort groups by time of first interval
    groups.sort(key=lambda g: g[0].start_index)
    return groups


def recognize_repeated_structures(
    df: pd.DataFrame,
    work_intervals: List[Interval],
    min_reps: int = 3,
) -> List[IntervalSet]:
    """Group similar intervals into repeated structures, e.g., 5x5, 3x20, 10x30/30."""
    if not work_intervals:
        return []

    groups = _group_by_duration_and_intensity(df, work_intervals)
    sets: List[IntervalSet] = []
    for g in groups:
        if len(g) >= min_reps:
            # Build simple template signature by duration
            g_sorted = sorted(g, key=lambda itv: itv.start_index)
            avg_duration = float(np.mean([itv.duration_s for itv in g_sorted]))

            # Estimate average off (recovery) duration from gaps between intervals (1 Hz index spacing)
            gaps: List[int] = []
            for a, b in zip(g_sorted, g_sorted[1:]):
                gap = int(b.start_index) - int(a.end_index) - 1
                if gap >= 0:
                    gaps.append(gap)
            avg_off = float(np.mean(gaps)) if gaps else 0.0

            # Short-on/off set labeling (e.g., 8x40/15, 8x30/30, 8x20/40)
            if avg_duration <= 120:  # on-phase <= 2 minutes → express in seconds (rounded to nearest 5s)
                on_sec = int(round(avg_duration / 5.0) * 5)
                off_sec = int(round(avg_off / 5.0) * 5) if avg_off > 0 else 0
                if off_sec > 0:
                    set_label = f"{len(g_sorted)}x{on_sec}/{off_sec}"
                    template_signature = f"{len(g_sorted)}x{on_sec}/{off_sec}"
                else:
                    set_label = f"{len(g_sorted)}x{on_sec}s"
                    template_signature = f"{len(g_sorted)}x{on_sec}s"
            elif 280 <= avg_duration <= 340:
                set_label = f"{len(g)}x5 VO2"
                template_signature = f"{len(g)}x5@VO2"
            elif 1100 <= avg_duration <= 1300:
                set_label = f"{len(g)}x20 THR"
                template_signature = f"{len(g)}x20@THR"
            else:
                minutes = max(1, int(round(avg_duration / 60.0)))
                set_label = f"{len(g)}x{minutes}"
                template_signature = f"{len(g)}x{minutes}"
            sets.append(IntervalSet(intervals=g_sorted, set_label=set_label, template_signature=template_signature))
    return sets
