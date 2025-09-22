from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

# Allow running as a script by falling back to absolute imports
try:
    from .config import TEST_INPUT_PATH, TEST_OUTPUT_DIR, FTP_WATTS, RIDER_MASS_KG, LTHR_BPM
    from .io.fit_loader import load_fit_to_dataframe, ensure_columns
    from .io.lap_loader import load_fit_laps, laps_to_intervals
    from .models.types import Ride, IntervalMetrics, SetSummary
    from .metrics.compute import compute_interval_metrics, summarize_interval_set
    from .recognition.detector import recognize_repeated_structures
    from .storage.export import export_set_summaries_csv, export_interval_metrics_csv
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from workout_foundation.config import TEST_INPUT_PATH, TEST_OUTPUT_DIR, FTP_WATTS, RIDER_MASS_KG, LTHR_BPM
    from workout_foundation.io.fit_loader import load_fit_to_dataframe, ensure_columns
    from workout_foundation.io.lap_loader import load_fit_laps, laps_to_intervals
    from workout_foundation.models.types import Ride, IntervalMetrics, SetSummary
    from workout_foundation.metrics.compute import compute_interval_metrics, summarize_interval_set
    from workout_foundation.recognition.detector import recognize_repeated_structures
    from workout_foundation.storage.export import export_set_summaries_csv, export_interval_metrics_csv


def _iter_fit_files(input_path: str) -> List[str]:
    p = Path(input_path)
    # Accept direct file path regardless of extension (some FIT exports lack .fit suffix)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = [str(fp) for fp in p.rglob("*.fit")]
        files += [str(fp) for fp in p.rglob("*.FIT")]
        return sorted(list(set(files)))
    return []


def _metrics_to_rows(metrics: List[IntervalMetrics], source_file: str) -> List[Dict]:
    rows: List[Dict] = []
    for idx, m in enumerate(metrics, start=1):
        rows.append(
            {
                "source_file": source_file,
                "lap_index": idx,
                "workout_date": pd.to_datetime(m.interval.start_time).date() if m.interval.start_time else None,
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
                "avg_hr_pct_lthr": m.avg_hr_pct_lthr,
                "power_zone": m.coggan_power_zone,
                "power_hr_ratio": m.power_to_hr_ratio,
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
                "label": m.interval.label,
            }
        )
    return rows


def _append_pr_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pr_flag"] = False
    if df.empty or "effort_key" not in df.columns:
        return df
    # Sort within each effort key by date/time, compute prior cumulative max
    df = df.sort_values(["effort_key", "workout_date", "start_time"]).copy()
    prev_cummax = df.groupby("effort_key")["avg_power_w"].cummax().shift(1)
    df["pr_flag"] = df["avg_power_w"] > prev_cummax.fillna(-np.inf)
    return df


def _write_effort_trends(all_df: pd.DataFrame, output_dir: str) -> str:
    if all_df.empty:
        return ""
    cols_needed = [
        "workout_date",
        "effort_key",
        "avg_power_w",
        "efficiency_factor_ef",
        "hr_drift_pct",
        "pw_hr_decoupling_pct",
        "pct_in_target",
    ]
    for c in cols_needed:
        if c not in all_df.columns:
            all_df[c] = pd.NA
    daily = (
        all_df.groupby(["workout_date", "effort_key"]).agg(
            best_avg_power_w=("avg_power_w", "max"),
            mean_ef=("efficiency_factor_ef", "mean"),
            mean_hr_drift_pct=("hr_drift_pct", "mean"),
            mean_pw_hr_decoupling_pct=("pw_hr_decoupling_pct", "mean"),
            mean_pct_in_target=("pct_in_target", "mean"),
        ).reset_index()
    )

    # Add trends without GroupBy.apply (avoid FutureWarning)
    daily = daily.sort_values(["effort_key", "workout_date"]).copy()
    daily["best_power_28d_ma"] = (
        daily.groupby("effort_key")["best_avg_power_w"].transform(lambda s: s.rolling(28, min_periods=7).mean())
    )
    daily["ef_28d_ma"] = (
        daily.groupby("effort_key")["mean_ef"].transform(lambda s: s.rolling(28, min_periods=7).mean())
    )
    daily["best_power_wow_pct"] = daily.groupby("effort_key")["best_avg_power_w"].pct_change(periods=7) * 100.0
    daily["ef_wow_pct"] = daily.groupby("effort_key")["mean_ef"].pct_change(periods=7) * 100.0

    out = os.path.join(output_dir, "effort_trends.csv")
    daily.to_csv(out, index=False)
    return out


def _write_sprint_trends(all_df: pd.DataFrame, output_dir: str) -> str:
    if all_df.empty:
        return ""
    # Filter sprint-like efforts: Z6/Z7 and <= 60s duration bucket
    sprint = all_df[(all_df["effort_key"].notna()) & (all_df["duration_bucket_s"].notna())].copy()
    sprint = sprint[sprint["effort_key"].str.startswith(("Z6", "Z7")) & (sprint["duration_bucket_s"] <= 60)]
    if sprint.empty:
        return ""
    daily = (
        sprint.groupby(["workout_date", "effort_key"]).agg(
            best_peak_5s_w=("peak_5s_w", "max"),
            best_peak_30s_w=("peak_30s_w", "max"),
            best_avg_power_w=("avg_power_w", "max"),
        ).reset_index()
    )
    # Add trends without GroupBy.apply
    daily = daily.sort_values(["effort_key", "workout_date"]).copy()
    daily["peak5_28d_ma"] = (
        daily.groupby("effort_key")["best_peak_5s_w"].transform(lambda s: s.rolling(28, min_periods=7).mean())
    )
    daily["peak30_28d_ma"] = (
        daily.groupby("effort_key")["best_peak_30s_w"].transform(lambda s: s.rolling(28, min_periods=7).mean())
    )
    daily["avgp_28d_ma"] = (
        daily.groupby("effort_key")["best_avg_power_w"].transform(lambda s: s.rolling(28, min_periods=7).mean())
    )
    daily["peak5_wow_pct"] = daily.groupby("effort_key")["best_peak_5s_w"].pct_change(periods=7) * 100.0
    daily["peak30_wow_pct"] = daily.groupby("effort_key")["best_peak_30s_w"].pct_change(periods=7) * 100.0
    daily["avgp_wow_pct"] = daily.groupby("effort_key")["best_avg_power_w"].pct_change(periods=7) * 100.0
    out = os.path.join(output_dir, "sprint_trends.csv")
    daily.to_csv(out, index=False)
    return out


def run_all() -> None:
    files = _iter_fit_files(TEST_INPUT_PATH)
    if not files:
        print(f"No .fit files found under {TEST_INPUT_PATH}")
        return

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    all_rows: List[Dict] = []

    for file_path in files:
        df = ensure_columns(load_fit_to_dataframe(file_path))
        ride = Ride(source_path=file_path, timeline=df, ftp_watts=FTP_WATTS, weight_kg=RIDER_MASS_KG, lthr_bpm=LTHR_BPM)
        laps = load_fit_laps(file_path)
        lap_intervals = laps_to_intervals(df, laps)
        lap_metrics = [compute_interval_metrics(df, itv, ride) for itv in lap_intervals]
        all_rows.extend(_metrics_to_rows(lap_metrics, source_file=Path(file_path).name))

        print(f"Processed {Path(file_path).stem}: {len(lap_intervals)} lap intervals")

        # Per-ride lap metrics export
        base = Path(file_path).stem
        if lap_metrics:
            export_interval_metrics_csv(lap_metrics, os.path.join(TEST_OUTPUT_DIR, f"{base}_lap_interval_metrics.csv"))

        # Basic set grouping over lap intervals and export per-ride set summaries
        sets = recognize_repeated_structures(df, lap_intervals, min_reps=2)
        if sets:
            set_summaries: List[SetSummary] = [summarize_interval_set(df, s, ride) for s in sets]
            export_set_summaries_csv(set_summaries, os.path.join(TEST_OUTPUT_DIR, f"{base}_set_summaries.csv"))

    if not all_rows:
        print("No lap intervals found to export.")
        return

    out_path = os.path.join(TEST_OUTPUT_DIR, "lap_metrics_history.csv")
    all_df = pd.DataFrame(all_rows)
    all_df = _append_pr_flags(all_df)
    all_df.to_csv(out_path, index=False)
    print(f"Wrote consolidated CSV: {out_path} ({len(all_df)} rows)")

    # Trends and comparison exports
    trends_path = _write_effort_trends(all_df, TEST_OUTPUT_DIR)
    if trends_path:
        print(f"Wrote effort trends CSV: {trends_path}")
    sprint_path = _write_sprint_trends(all_df, TEST_OUTPUT_DIR)
    if sprint_path:
        print(f"Wrote sprint trends CSV: {sprint_path}")

    # Optional: write set summaries history by scanning per-ride set CSVs just written
    try:
        set_csvs = sorted([str(fp) for fp in Path(TEST_OUTPUT_DIR).glob("*_set_summaries.csv")])
        if set_csvs:
            frames = []
            for p in set_csvs:
                df_set = pd.read_csv(p)
                df_set["source_file"] = Path(p).name.replace("_set_summaries.csv", ".fit")
                frames.append(df_set)
            set_hist = pd.concat(frames, ignore_index=True)
            set_hist.to_csv(os.path.join(TEST_OUTPUT_DIR, "set_summaries_history.csv"), index=False)
            print(f"Wrote set summaries history CSV: {os.path.join(TEST_OUTPUT_DIR, 'set_summaries_history.csv')}")
    except Exception as e:
        print(f"Warning: could not build set summaries history: {e}")


if __name__ == "__main__":
    run_all()
