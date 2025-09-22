from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class Interval:
    start_time: datetime
    end_time: datetime
    start_index: int
    end_index: int
    duration_s: float
    average_power_w: Optional[float] = None
    average_heart_rate_bpm: Optional[float] = None
    label: Optional[str] = None  # e.g., "work", "recovery"


@dataclass
class IntervalSet:
    intervals: List[Interval]
    set_label: Optional[str] = None  # e.g., "5x5 VO2", "3x20 THR", "10x30/30"
    template_signature: Optional[str] = None  # concise signature like "5x5@VO2"


@dataclass
class Ride:
    source_path: str
    timeline: pd.DataFrame  # second-by-second data
    ftp_watts: float
    weight_kg: Optional[float] = None
    lthr_bpm: Optional[float] = None


@dataclass
class IntervalMetrics:
    interval: Interval
    average_power_w: Optional[float]
    power_wkg: Optional[float]
    peak_5s_w: Optional[float]
    peak_30s_w: Optional[float]
    normalized_power_w: Optional[float]
    average_heart_rate_bpm: Optional[float]
    max_heart_rate_bpm: Optional[float]
    power_to_hr_ratio: Optional[float]
    avg_hr_pct_lthr: Optional[float]
    coggan_power_zone: Optional[str]
    work_kj: Optional[float]
    percent_time_within_target: Optional[float]
    variability_index_vi: Optional[float] = None
    intensity_factor_if: Optional[float] = None
    training_stress_score_tss: Optional[float] = None
    hr_drift_within_interval_pct: Optional[float] = None
    pw_hr_decoupling_pct: Optional[float] = None
    efficiency_factor_ef: Optional[float] = None
    hr_recovery_60s_bpm: Optional[float] = None
    duration_bucket_s: Optional[int] = None
    effort_key: Optional[str] = None
    effort_family: Optional[str] = None
    pr_flag: Optional[bool] = None
    kj_per_min: Optional[float] = None
    average_cadence_rpm: Optional[float] = None
    coasting_percent: Optional[float] = None
    distance_m: Optional[float] = None
    average_speed_mps: Optional[float] = None
    elevation_gain_m: Optional[float] = None


@dataclass
class SetSummary:
    interval_set: IntervalSet
    coefficient_of_variation_pct: Optional[float]
    fade_pct: Optional[float]
    hr_drift_pct: Optional[float]
    recovery_hr_drop_bpm: Optional[float] = None
    total_work_kj: Optional[float] = None
    avg_within_interval_hr_drift_pct: Optional[float] = None
    avg_pw_hr_decoupling_pct: Optional[float] = None


@dataclass
class SetBlockSummary:
    """Summary for an entire on/off set treated as one block interval."""
    interval_set: IntervalSet
    block_interval: Interval
    block_metrics: IntervalMetrics
    on_time_s: float
    off_time_s: float
    avg_on_s: Optional[float] = None
    avg_off_s: Optional[float] = None
    set_family: Optional[str] = None   # e.g., "30_30", "45_15"


@dataclass
class WorkoutSummary:
    ride: Ride
    interval_metrics: List[IntervalMetrics] = field(default_factory=list)
    set_summaries: List[SetSummary] = field(default_factory=list)
