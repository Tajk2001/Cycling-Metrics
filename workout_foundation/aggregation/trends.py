from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..models.types import IntervalMetrics, SetSummary


def summarize_over_time(interval_metrics: List[IntervalMetrics]) -> pd.DataFrame:
    rows = []
    for m in interval_metrics:
        rows.append(
            {
                "date": pd.to_datetime(m.interval.start_time).date(),
                "avg_power_w": m.average_power_w,
                "power_wkg": m.power_wkg,
                "normalized_power_w": m.normalized_power_w,
                "avg_hr_bpm": m.average_heart_rate_bpm,
                "power_hr_ratio": m.power_to_hr_ratio,
                "work_kj": m.work_kj,
                "pct_in_target": m.percent_time_within_target,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    grouped = df.groupby("date").agg(
        avg_power_w=("avg_power_w", "mean"),
        power_wkg=("power_wkg", "mean"),
        normalized_power_w=("normalized_power_w", "mean"),
        avg_hr_bpm=("avg_hr_bpm", "mean"),
        power_hr_ratio=("power_hr_ratio", "mean"),
        work_kj=("work_kj", "sum"),
        pct_in_target=("pct_in_target", "mean"),
    )
    return grouped.reset_index()


