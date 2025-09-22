from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..models.types import Interval, IntervalMetrics, IntervalSet, SetSummary, SetBlockSummary, Ride

# Minimum duration to compute within-interval HR drift / Pw:HR decoupling
MIN_DRIFT_DURATION_S = 600  # 10 minutes


def _normalized_power_w(power_series: pd.Series) -> Optional[float]:
    """Normalized Power using 30-second rolling average of power^4.

    Research-grade handling:
    - Require 30 valid seconds for the rolling average; if fewer than 30s available,
      fall back to mean power (common practical convention) rather than using smaller windows.
    - Ignore NaNs; do not forward-fill here.
    """
    if power_series.empty:
        return None
    ps = power_series.dropna()
    if ps.empty:
        return None
    if len(ps) < 30:
        return float(ps.mean())
    rolling = ps.rolling(window=30, min_periods=30).mean()
    fourth = rolling.pow(4)
    mean_fourth = fourth.mean(skipna=True)
    if pd.isna(mean_fourth):
        return None
    return float(np.power(mean_fourth, 1.0 / 4.0))


def _peak_power(power_series: pd.Series, window_s: int) -> Optional[float]:
    if power_series.empty:
        return None
    if len(power_series) < window_s:
        return float(power_series.max())
    rolling = power_series.rolling(window=window_s, min_periods=window_s).mean()
    return float(rolling.max()) if not rolling.empty else None


def _coggan_power_zone_label(avg_power: Optional[float], ftp_watts: Optional[float]) -> Optional[str]:
    """Return Coggan power zone label based on average power and FTP.

    Zones:
    - Z1 Active Recovery: <55%
    - Z2 Endurance: 56–75%
    - Z3 Tempo: 76–90%
    - Z4 Threshold: 91–105%
    - Z5 VO2max: 106–120%
    - Z6 Anaerobic: 121–150%
    - Z7 Neuromuscular: >150%
    """
    if avg_power is None or ftp_watts is None or ftp_watts <= 0:
        return None
    pct = (avg_power / ftp_watts) * 100.0
    if pct < 55:
        return "Z1 Active Recovery"
    if pct <= 75:
        return "Z2 Endurance"
    if pct <= 90:
        return "Z3 Tempo"
    if pct <= 105:
        return "Z4 Threshold"
    if pct <= 120:
        return "Z5 VO2max"
    if pct <= 150:
        return "Z6 Anaerobic"
    return "Z7 Neuromuscular"


def _round_to_nearest(value: float, base: int) -> int:
    return int(base * round(float(value) / float(base)))


def _duration_bucket_s(duration_s: Optional[float]) -> Optional[int]:
    if duration_s is None:
        return None
    d = float(duration_s)
    if d < 300:
        bucket = _round_to_nearest(d, 30)
    else:
        bucket = _round_to_nearest(d, 60)
    # Ensure non-zero sensible minimum
    return int(max(10, bucket))


def _effort_key(zone_label: Optional[str], bucket_s: Optional[int]) -> Optional[str]:
    if zone_label is None or bucket_s is None:
        return None
    zone_code = zone_label.split()[0] if zone_label else None
    if zone_code is None:
        return None
    if bucket_s < 60:
        return f"{zone_code}_{int(bucket_s)}s"
    else:
        minutes = int(round(bucket_s / 60.0))
        return f"{zone_code}_{minutes}m"


def _effort_family_key(zone_label: Optional[str], duration_s: Optional[float]) -> Optional[str]:
    """Coarser, more stable grouping for like-to-like comparisons.

    Rules:
    - < 60s: round to nearest 5s → Zx_30s
    - 60–300s: round to nearest 1m → Zx_3m
    - 5–15min: round to nearest 2m → Zx_14m
    - >=15min: round to nearest 5m → Zx_30m
    """
    if zone_label is None or duration_s is None:
        return None
    zone_code = zone_label.split()[0] if zone_label else None
    if zone_code is None:
        return None
    d = float(duration_s)
    if d < 60:
        sec = int(round(d / 5.0) * 5)
        sec = max(5, sec)
        return f"{zone_code}_{sec}s"
    if d < 300:
        minutes = int(round(d / 60.0))
        minutes = max(1, minutes)
        return f"{zone_code}_{minutes}m"
    if d < 900:
        minutes = int(round(d / 120.0) * 2)
        minutes = max(4, minutes)
        return f"{zone_code}_{minutes}m"
    minutes = int(round(d / 300.0) * 5)
    minutes = max(15, minutes)
    return f"{zone_code}_{minutes}m"


def compute_interval_metrics(df: pd.DataFrame, interval: Interval, ride: Ride) -> IntervalMetrics:
    seg = df.iloc[interval.start_index : interval.end_index + 1]
    avg_power = float(seg["power"].mean()) if "power" in seg else None
    wkg = (avg_power / ride.weight_kg) if (avg_power is not None and ride.weight_kg and ride.weight_kg > 0) else None
    peak5 = _peak_power(seg["power"], 5) if "power" in seg else None
    peak30 = _peak_power(seg["power"], 30) if "power" in seg else None
    npw = _normalized_power_w(seg["power"]) if "power" in seg else None
    avg_hr = float(seg["heart_rate"].mean()) if "heart_rate" in seg else None
    max_hr = float(seg["heart_rate"].max()) if "heart_rate" in seg else None
    p_hr = (avg_power / avg_hr) if (avg_power is not None and avg_hr and avg_hr > 0) else None
    avg_hr_pct_lthr = (avg_hr / ride.lthr_bpm * 100.0) if (avg_hr is not None and ride.lthr_bpm and ride.lthr_bpm > 0) else None
    work_kj = float((seg["power"].sum() / 1000.0)) if "power" in seg else None

    # Derived training metrics
    vi = float(npw / avg_power) if (npw is not None and avg_power and avg_power > 0) else None
    intensity_factor = float(npw / ride.ftp_watts) if (npw is not None and ride.ftp_watts and ride.ftp_watts > 0) else None
    duration_s = int(interval.duration_s) if interval.duration_s is not None else len(seg)
    tss = (
        float((duration_s * npw * intensity_factor) / (ride.ftp_watts * 3600.0) * 100.0)
        if (npw is not None and intensity_factor is not None and ride.ftp_watts and ride.ftp_watts > 0)
        else None
    )
    ef = float(npw / avg_hr) if (npw is not None and avg_hr and avg_hr > 0) else None

    # HR drift and Pw:HR decoupling (first half vs second half)
    hr_drift_pct = None
    pw_hr_decouple_pct = None
    if len(seg) >= 2 and "heart_rate" in seg and "power" in seg and duration_s >= MIN_DRIFT_DURATION_S:
        half = len(seg) // 2
        first = seg.iloc[:half]
        second = seg.iloc[half:]
        if not first.empty and not second.empty:
            hr1 = float(first["heart_rate"].mean()) if not first["heart_rate"].isna().all() else None
            hr2 = float(second["heart_rate"].mean()) if not second["heart_rate"].isna().all() else None
            p1 = float(first["power"].mean()) if not first["power"].isna().all() else None
            p2 = float(second["power"].mean()) if not second["power"].isna().all() else None
            if hr1 and hr1 > 0 and hr2 is not None:
                hr_drift_pct = float((hr2 - hr1) / hr1 * 100.0)
            if hr1 and hr1 > 0 and hr2 and hr2 > 0 and p1 is not None and p2 is not None and p1 > 0:
                ratio1 = p1 / hr1
                ratio2 = p2 / hr2
                if ratio1 > 0:
                    pw_hr_decouple_pct = float((ratio2 / ratio1 - 1.0) * 100.0)

    coggan_zone = _coggan_power_zone_label(avg_power, ride.ftp_watts)
    bucket = _duration_bucket_s(duration_s)
    effort_key = _effort_key(coggan_zone, bucket)
    effort_family = _effort_family_key(coggan_zone, duration_s)

    # HR recovery in next 60s (end avg last 5s minus min of next 60s)
    hr_recovery_60s = None
    if "heart_rate" in df and interval.end_index is not None:
        end_idx = interval.end_index
        end_window = df.iloc[max(interval.start_index, end_idx - 4) : end_idx + 1]
        post_start = end_idx + 1
        post_end = min(len(df) - 1, end_idx + 60)
        post_window = df.iloc[post_start : post_end + 1] if post_start <= post_end else None
        if not end_window.empty and post_window is not None and not post_window.empty:
            hr_end = float(end_window["heart_rate"].mean()) if not end_window["heart_rate"].isna().all() else None
            hr_min_next_minute = float(post_window["heart_rate"].min()) if not post_window["heart_rate"].isna().all() else None
            if hr_end is not None and hr_min_next_minute is not None:
                hr_recovery_60s = float(hr_end - hr_min_next_minute)

    # % time within target: define target as within ±5% of the set's mean if set is known; fallback to within ±10% of interval avg
    if avg_power is not None and "power" in seg:
        lower = avg_power * 0.90
        upper = avg_power * 1.10
        within = seg[(seg["power"] >= lower) & (seg["power"] <= upper)]
        pct_in_target = float(len(within) / len(seg) * 100.0) if len(seg) > 0 else None
    else:
        pct_in_target = None

    # New requested metrics
    kj_per_min = float(work_kj / (duration_s / 60.0)) if (work_kj is not None and duration_s > 0) else None
    avg_cadence = float(seg["cadence"].mean()) if "cadence" in seg and not seg["cadence"].isna().all() else None
    
    # Coasting percent (power = 0 or very low)
    coasting_pct = None
    if "power" in seg and not seg["power"].isna().all():
        coasting_count = (seg["power"] <= 5).sum()  # Power <= 5W considered coasting
        coasting_pct = float(coasting_count / len(seg) * 100.0) if len(seg) > 0 else None
    
    # Distance and speed
    distance = None
    avg_speed = None
    if "distance" in seg and not seg["distance"].isna().all():
        start_dist = float(seg["distance"].iloc[0]) if not pd.isna(seg["distance"].iloc[0]) else 0
        end_dist = float(seg["distance"].iloc[-1]) if not pd.isna(seg["distance"].iloc[-1]) else 0
        distance = abs(end_dist - start_dist)
    if "speed" in seg and not seg["speed"].isna().all():
        avg_speed = float(seg["speed"].mean())
    
    # Elevation gain
    elevation_gain = None
    if "altitude" in seg and not seg["altitude"].isna().all():
        alt_diff = seg["altitude"].diff()
        positive_gains = alt_diff[alt_diff > 0]
        elevation_gain = float(positive_gains.sum()) if not positive_gains.empty else 0.0

    return IntervalMetrics(
        interval=interval,
        average_power_w=avg_power,
        power_wkg=wkg,
        peak_5s_w=peak5,
        peak_30s_w=peak30,
        normalized_power_w=npw,
        average_heart_rate_bpm=avg_hr,
        max_heart_rate_bpm=max_hr,
        power_to_hr_ratio=p_hr,
        avg_hr_pct_lthr=avg_hr_pct_lthr,
        coggan_power_zone=coggan_zone,
        work_kj=work_kj,
        percent_time_within_target=pct_in_target,
        variability_index_vi=vi,
        intensity_factor_if=intensity_factor,
        training_stress_score_tss=tss,
        hr_drift_within_interval_pct=hr_drift_pct,
        pw_hr_decoupling_pct=pw_hr_decouple_pct,
        efficiency_factor_ef=ef,
        hr_recovery_60s_bpm=hr_recovery_60s,
        duration_bucket_s=bucket,
        effort_key=effort_key,
        effort_family=effort_family,
        kj_per_min=kj_per_min,
        average_cadence_rpm=avg_cadence,
        coasting_percent=coasting_pct,
        distance_m=distance,
        average_speed_mps=avg_speed,
        elevation_gain_m=elevation_gain,
    )


def summarize_interval_set(df: pd.DataFrame, interval_set: IntervalSet, ride: Ride, min_drift_duration_s: int = MIN_DRIFT_DURATION_S) -> SetSummary:
    metrics = [compute_interval_metrics(df, itv, ride) for itv in interval_set.intervals]
    powers = [m.average_power_w for m in metrics if m.average_power_w is not None]
    hrs = [m.average_heart_rate_bpm for m in metrics if m.average_heart_rate_bpm is not None]
    coefficient_of_variation_pct: Optional[float]
    if len(powers) >= 2:
        coefficient_of_variation_pct = float(np.std(powers, ddof=1) / np.mean(powers) * 100.0) if np.mean(powers) > 0 else None
    else:
        coefficient_of_variation_pct = None

    fade_pct: Optional[float]
    if len(powers) >= 2:
        first, last = powers[0], powers[-1]
        fade_pct = float((first - last) / first * 100.0) if first else None
    else:
        fade_pct = None

    hr_drift_pct: Optional[float]
    if len(hrs) >= 2:
        hr_first, hr_last = hrs[0], hrs[-1]
        hr_drift_pct = float((hr_last - hr_first) / hr_first * 100.0) if hr_first else None
    else:
        hr_drift_pct = None

    total_work_kj = float(np.nansum([m.work_kj for m in metrics])) if metrics else None

    # Average within-interval HR drift and Pw:HR decoupling across reps (respect duration threshold)
    drift_vals = []
    decouple_vals = []
    for itv, m in zip(interval_set.intervals, metrics):
        if itv.duration_s and itv.duration_s >= min_drift_duration_s:
            if m.hr_drift_within_interval_pct is not None:
                drift_vals.append(m.hr_drift_within_interval_pct)
            if m.pw_hr_decoupling_pct is not None:
                decouple_vals.append(m.pw_hr_decoupling_pct)
    avg_within_drift = float(np.nanmean(drift_vals)) if drift_vals else None
    avg_decoupling = float(np.nanmean(decouple_vals)) if decouple_vals else None

    # Recovery HR drop between reps: measure HR at last 5s of work and min HR in next 60s
    recovery_drops = []
    for idx, itv in enumerate(interval_set.intervals[:-1]):
        next_itv = interval_set.intervals[idx + 1]
        end_idx = itv.end_index
        end_window = df.iloc[max(itv.start_index, end_idx - 4) : end_idx + 1]
        post_start = itv.end_index + 1
        post_end = min(len(df) - 1, itv.end_index + 60)
        post_window = df.iloc[post_start : post_end + 1] if post_start <= post_end else None
        if "heart_rate" in df and not end_window.empty and post_window is not None and not post_window.empty:
            hr_end = float(end_window["heart_rate"].mean()) if not end_window["heart_rate"].isna().all() else None
            hr_min_next_minute = float(post_window["heart_rate"].min()) if not post_window["heart_rate"].isna().all() else None
            if hr_end and hr_min_next_minute is not None:
                recovery_drops.append(hr_end - hr_min_next_minute)
    recovery_hr_drop = float(np.nanmean(recovery_drops)) if recovery_drops else None

    return SetSummary(
        interval_set=interval_set,
        coefficient_of_variation_pct=coefficient_of_variation_pct,
        fade_pct=fade_pct,
        hr_drift_pct=hr_drift_pct,
        recovery_hr_drop_bpm=recovery_hr_drop,
        total_work_kj=total_work_kj,
        avg_within_interval_hr_drift_pct=avg_within_drift,
        avg_pw_hr_decoupling_pct=avg_decoupling,
    )


def summarize_set_block(df: pd.DataFrame, interval_set: IntervalSet, ride: Ride) -> SetBlockSummary:
    """Treat a repeated-interval set as a single block (on + off) and compute metrics.

    - Block interval spans from first interval start to last interval end.
    - on_time_s is sum of on-interval durations; off_time_s = block - on_time.
    - set_family is derived from average on/off durations rounded to nearest 5s.
    """
    if not interval_set.intervals:
        # Empty set fallback: return a zeroed block with minimal info
        empty_interval = Interval(start_time=None, end_time=None, start_index=0, end_index=0, duration_s=0.0, label="set_block")
        empty_metrics = IntervalMetrics(
            interval=empty_interval,
            average_power_w=None,
            power_wkg=None,
            peak_5s_w=None,
            peak_30s_w=None,
            normalized_power_w=None,
            average_heart_rate_bpm=None,
            max_heart_rate_bpm=None,
            power_to_hr_ratio=None,
            avg_hr_pct_lthr=None,
            coggan_power_zone=None,
            work_kj=None,
            percent_time_within_target=None,
        )
        return SetBlockSummary(interval_set=interval_set, block_interval=empty_interval, block_metrics=empty_metrics, on_time_s=0.0, off_time_s=0.0)

    first = interval_set.intervals[0]
    last = interval_set.intervals[-1]
    start_idx = int(first.start_index)
    end_idx = int(last.end_index)
    duration_s = float(end_idx - start_idx + 1)
    block_interval = Interval(
        start_time=first.start_time,
        end_time=last.end_time,
        start_index=start_idx,
        end_index=end_idx,
        duration_s=duration_s,
        label="set_block",
    )

    # Compute block metrics using the existing interval pipeline
    block_metrics = compute_interval_metrics(df, block_interval, ride)

    # Compute on/off composition
    on_time = float(sum(itv.duration_s for itv in interval_set.intervals if itv.duration_s))
    off_time = max(0.0, duration_s - on_time)

    # Average on/off durations
    num = len(interval_set.intervals)
    avg_on_s = on_time / num if num > 0 else None
    # For off, use average gap between successive intervals
    gaps: list[float] = []
    for a, b in zip(interval_set.intervals, interval_set.intervals[1:]):
        gap = float(int(b.start_index) - int(a.end_index) - 1)
        if gap >= 0:
            gaps.append(gap)
    avg_off_s = float(np.mean(gaps)) if gaps else (off_time / num if num > 0 and off_time > 0 else None)

    # Family key rounding to nearest 5s
    def _round5(x: Optional[float]) -> Optional[int]:
        if x is None:
            return None
        return int(round(x / 5.0) * 5)

    on5 = _round5(avg_on_s)
    off5 = _round5(avg_off_s)
    set_family = f"{on5}_{off5}" if (on5 and off5) else (f"{on5}_x" if on5 else None)

    return SetBlockSummary(
        interval_set=interval_set,
        block_interval=block_interval,
        block_metrics=block_metrics,
        on_time_s=on_time,
        off_time_s=off_time,
        avg_on_s=avg_on_s,
        avg_off_s=avg_off_s,
        set_family=set_family,
    )
