from __future__ import annotations

from typing import List

import pandas as pd

from ..models.types import IntervalMetrics, SetSummary, SetBlockSummary


def export_interval_metrics_csv(metrics: List[IntervalMetrics], path: str) -> None:
    rows = []
    for m in metrics:
        row = {
            "start_time": m.interval.start_time,
            "end_time": m.interval.end_time,
            "duration_s": m.interval.duration_s,
            "avg_power_w": m.average_power_w,
            "power_wkg": m.power_wkg,
            "peak_5s_w": m.peak_5s_w,
            "peak_30s_w": m.peak_30s_w,
            "normalized_power_w": m.normalized_power_w,
            "avg_hr_bpm": m.average_heart_rate_bpm,
            "max_hr_bpm": m.max_heart_rate_bpm,
            "power_hr_ratio": m.power_to_hr_ratio,
            "avg_hr_pct_lthr": m.avg_hr_pct_lthr,
            "power_zone": m.coggan_power_zone,
            "work_kj": m.work_kj,
            "pct_in_target": m.percent_time_within_target,
            "variability_index_vi": m.variability_index_vi,
            "intensity_factor_if": m.intensity_factor_if,
            "tss": m.training_stress_score_tss,
            "hr_drift_pct": m.hr_drift_within_interval_pct,
            "pw_hr_decoupling_pct": m.pw_hr_decoupling_pct,
            "efficiency_factor_ef": m.efficiency_factor_ef,
            "hr_recovery_60s_bpm": m.hr_recovery_60s_bpm,
            "duration_bucket_s": m.duration_bucket_s,
            "effort_key": m.effort_key,
            "effort_family": m.effort_family,
            "pr_flag": m.pr_flag,
            "kj_per_min": m.kj_per_min,
            "avg_cadence_rpm": m.average_cadence_rpm,
            "coasting_percent": m.coasting_percent,
            "distance_m": m.distance_m,
            "avg_speed_mps": m.average_speed_mps,
            "elevation_gain_m": m.elevation_gain_m,
            "label": m.interval.label,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def export_set_summaries_csv(summaries: List[SetSummary], path: str) -> None:
    rows = []
    for s in summaries:
        row = {
            "set_label": s.interval_set.set_label,
            "template_signature": s.interval_set.template_signature,
            "num_intervals": len(s.interval_set.intervals),
            "cv_pct": s.coefficient_of_variation_pct,
            "fade_pct": s.fade_pct,
            "hr_drift_pct": s.hr_drift_pct,
            "recovery_hr_drop_bpm": s.recovery_hr_drop_bpm,
            "total_work_kj": s.total_work_kj,
            "avg_within_interval_hr_drift_pct": s.avg_within_interval_hr_drift_pct,
            "avg_pw_hr_decoupling_pct": s.avg_pw_hr_decoupling_pct,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def export_set_blocks_csv(blocks: List[SetBlockSummary], path: str) -> None:
    rows = []
    for b in blocks:
        m = b.block_metrics
        row = {
            "set_label": b.interval_set.set_label,
            "template_signature": b.interval_set.template_signature,
            "num_intervals": len(b.interval_set.intervals),
            "block_start_time": b.block_interval.start_time,
            "block_end_time": b.block_interval.end_time,
            "block_duration_s": b.block_interval.duration_s,
            "on_time_s": b.on_time_s,
            "off_time_s": b.off_time_s,
            "avg_on_s": b.avg_on_s,
            "avg_off_s": b.avg_off_s,
            "set_family": b.set_family,
            # Flatten key interval metrics for the block
            "block_avg_power_w": m.average_power_w,
            "block_normalized_power_w": m.normalized_power_w,
            "block_variability_index_vi": m.variability_index_vi,
            "block_intensity_factor_if": m.intensity_factor_if,
            "block_tss": m.training_stress_score_tss,
            "block_avg_hr_bpm": m.average_heart_rate_bpm,
            "block_efficiency_factor_ef": m.efficiency_factor_ef,
            "block_hr_drift_pct": m.hr_drift_within_interval_pct,
            "block_pw_hr_decoupling_pct": m.pw_hr_decoupling_pct,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
