from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import re

from .io.fit_loader import load_fit_to_dataframe, ensure_columns
from .io.lap_loader import load_fit_laps, laps_to_intervals
from .models.types import Ride, IntervalMetrics, SetSummary
from .models.athlete_profile import load_athlete_profile
from .metrics.compute import compute_interval_metrics, summarize_interval_set, summarize_set_block
from .recognition.detector import recognize_repeated_structures
from .recognition.categorize import categorize_workout
from .storage.export import export_interval_metrics_csv, export_set_summaries_csv
from .config import FTP_WATTS, RIDER_MASS_KG, LTHR_BPM


def _iter_fit_files_many(inputs: List[str]) -> List[str]:
    files: List[str] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            files.append(str(p))
        elif p.is_dir():
            # Include all files except hidden ones (no extension requirement)
            files.extend([str(fp) for fp in p.rglob("*") if fp.is_file() and not fp.name.startswith('.')])
    # Deduplicate and sort
    return sorted(list(dict.fromkeys(files)))


def _metrics_to_rows(
    metrics: List[IntervalMetrics],
    source_file: str,
    ride_tags: Optional[str] = None,
    athlete: Optional[str] = None,
    workout_category: Optional[str] = None,
) -> List[Dict]:
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
                "effort_family": m.effort_family,
                "ride_tags": ride_tags,
                "athlete": athlete,
                "workout_category": workout_category,
                "kj_per_min": m.kj_per_min,
                "avg_cadence_rpm": m.average_cadence_rpm,
                "coasting_percent": m.coasting_percent,
                "distance_m": m.distance_m,
                "avg_speed_mps": m.average_speed_mps,
                "elevation_gain_m": m.elevation_gain_m,
                "label": m.interval.label,
            }
        )
    return rows


def _append_pr_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pr_flag"] = False
    if df.empty or "effort_key" not in df.columns:
        return df
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
        "effort_family",
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
            effort_family=("effort_family", "first"),
        ).reset_index()
    )
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


def _sanitize_tag(tag: str) -> str:
    tag = tag.strip().lower().replace(" ", "-")
    tag = re.sub(r"[^a-z0-9_\-+]+", "", tag)
    return tag


def _build_ride_tags(file_path: str, sets: List[SetSummary] | List, manual_tags: List[str] | None, tag_parent_dir: bool) -> List[str]:
    tags: List[str] = []
    if tag_parent_dir:
        tags.append(Path(file_path).parent.name)
    # Add set labels if available (either SetSummary or IntervalSet)
    seen = set()
    for s in sets or []:
        label = getattr(s, "interval_set", s)
        set_label = getattr(label, "set_label", None)
        if set_label and set_label not in seen:
            tags.append(set_label)
            seen.add(set_label)
    if manual_tags:
        tags.extend(manual_tags)
    # Sanitize and dedupe order-preserving
    clean: List[str] = []
    used = set()
    for t in tags:
        st = _sanitize_tag(str(t))
        if st and st not in used:
            clean.append(st)
            used.add(st)
    return clean[:5]


def _iter_athletes_layout(root: str) -> List[Dict]:
    tasks: List[Dict] = []
    root_path = Path(root)
    if not root_path.exists():
        return tasks
    # Support selecting an athlete folder directly as the root
    root_fits = [str(fp) for fp in root_path.glob("*.fit")] + [str(fp) for fp in root_path.glob("*.FIT")]
    if root_fits:
        athlete = root_path.name
        # Direct files as uncategorized
        tasks.append({"athlete": athlete, "category": "uncategorized", "input_dir": str(root_path), "files": sorted(root_fits)})
        # Subdirectories as categories
        for cat_dir in sorted([p for p in root_path.iterdir() if p.is_dir() and not p.name.startswith(".")]):
            category = cat_dir.name
            files = [str(fp) for fp in cat_dir.rglob("*.fit")] + [str(fp) for fp in cat_dir.rglob("*.FIT")]
            if files:
                tasks.append({"athlete": athlete, "category": category, "input_dir": str(cat_dir), "files": sorted(files)})
        return tasks
    for athlete_dir in sorted([p for p in root_path.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        athlete = athlete_dir.name
        # Files directly under athlete folder → "uncategorized"
        direct = [str(fp) for fp in athlete_dir.glob("*.fit")] + [str(fp) for fp in athlete_dir.glob("*.FIT")]
        if direct:
            tasks.append({"athlete": athlete, "category": "uncategorized", "input_dir": str(athlete_dir), "files": sorted(direct)})
        for cat_dir in sorted([p for p in athlete_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]):
            category = cat_dir.name
            files = [str(fp) for fp in cat_dir.rglob("*.fit")] + [str(fp) for fp in cat_dir.rglob("*.FIT")]
            if not files:
                continue
            tasks.append({"athlete": athlete, "category": category, "input_dir": str(cat_dir), "files": sorted(files)})
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Workout Foundation CLI: per-ride metrics, trends, and set summaries")
    parser.add_argument("--input", nargs="+", help="One or more input FIT files or directories")
    parser.add_argument("--output", required=True, help="Output directory for exports (or output root for athletes layout)")
    parser.add_argument("--athletes-root", help="Root folder with athlete/workout_type subfolders containing FIT files")
    parser.add_argument("--min-reps", type=int, default=2, help="Minimum reps to recognize a set (default: 2)")
    parser.add_argument("--tag", action="append", help="Tag to include in per-ride filenames (can be repeated or comma-separated)")
    parser.add_argument("--tag-parent-dir", action="store_true", help="Include parent directory name as a tag")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    # Athletes layout mode
    if args.athletes_root:
        tasks = _iter_athletes_layout(args.athletes_root)
        if not tasks:
            print("No .fit files found under athletes root.")
            return
        all_rows: List[Dict] = []
        for task in tasks:
            athlete = task["athlete"]
            category = task["category"]
            files = task["files"]
            # Export directly to athlete's category directory
            athlete_dir = Path(args.athletes_root) / athlete / category
            athlete_dir.mkdir(parents=True, exist_ok=True)
            
            # Load athlete's personal profile, allow category-specific overrides
            athlete_profile = load_athlete_profile(Path(args.athletes_root) / athlete)
            cat_profile_path = Path(args.athletes_root) / athlete / category / "profile.json"
            if cat_profile_path.exists():
                try:
                    with open(cat_profile_path, "r") as f:
                        cat_prof = json.load(f)
                    # Override core fields if provided
                    athlete_profile.ftp_watts = cat_prof.get("ftp_watts", athlete_profile.ftp_watts)
                    athlete_profile.lthr_bpm = cat_prof.get("lthr_bpm", athlete_profile.lthr_bpm)
                    athlete_profile.weight_kg = cat_prof.get("weight_kg", athlete_profile.weight_kg)
                except Exception:
                    pass

            for file_path in files:
                df = ensure_columns(load_fit_to_dataframe(file_path))
                ride = Ride(
                    source_path=file_path, 
                    timeline=df, 
                    ftp_watts=athlete_profile.ftp_watts, 
                    weight_kg=athlete_profile.weight_kg, 
                    lthr_bpm=athlete_profile.lthr_bpm
                )

                laps = load_fit_laps(file_path)
                lap_intervals = laps_to_intervals(df, laps)
                lap_metrics = [compute_interval_metrics(df, itv, ride) for itv in lap_intervals]

                # Recognize sets and build tagging
                sets = recognize_repeated_structures(df, lap_intervals, min_reps=args.min_reps)
                try:
                    date_str = pd.to_datetime(df["timestamp"].iloc[0]).strftime("%Y%m%d") if not df.empty else "unknown"
                except Exception:
                    date_str = "unknown"
                stem = Path(file_path).stem
                # Use folder category directly (no prediction override)
                applied_category = category
                tag_str = _sanitize_tag(applied_category)
                base = f"{date_str}_{stem}"
                base_with_tags = f"{base}__{tag_str}" if tag_str else base

                # Per-ride exports saved directly in athlete's category directory
                if lap_metrics:
                    export_interval_metrics_csv(lap_metrics, str(athlete_dir / f"{base_with_tags}_lap_interval_metrics.csv"))
                if sets:
                    set_summaries: List[SetSummary] = [summarize_interval_set(df, s, ride) for s in sets]
                    export_set_summaries_csv(set_summaries, str(athlete_dir / f"{base_with_tags}_set_summaries.csv"))
                    # Set blocks (on+off block per set)
                    set_blocks = [summarize_set_block(df, s, ride) for s in sets]
                    from .storage.export import export_set_blocks_csv
                    export_set_blocks_csv(set_blocks, str(athlete_dir / f"{base_with_tags}_set_blocks.csv"))

                # Accumulate consolidated rows with metadata
                all_rows.extend(
                    _metrics_to_rows(
                        lap_metrics,
                        source_file=Path(file_path).name,
                        ride_tags=tag_str,
                        athlete=athlete,
                        workout_category=applied_category,
                    )
                )

                print(f"Processed {athlete}/{category}/{stem}: {len(lap_intervals)} lap intervals (FTP={athlete_profile.ftp_watts}W, Wt={athlete_profile.weight_kg}kg)")

        if not all_rows:
            print("No lap intervals found to export.")
            return
        all_df = pd.DataFrame(all_rows)
        all_df = _append_pr_flags(all_df)
        # Save consolidated files per athlete
        for athlete in all_df['athlete'].unique():
            athlete_consolidated_dir = Path(args.athletes_root) / athlete / "consolidated"
            athlete_consolidated_dir.mkdir(parents=True, exist_ok=True)
            
            # Filter data for this athlete
            athlete_df = all_df[all_df['athlete'] == athlete]
            hist_path = athlete_consolidated_dir / "lap_metrics_history.csv"
            athlete_df.to_csv(hist_path, index=False)
            print(f"Wrote consolidated CSV for {athlete}: {hist_path} ({len(athlete_df)} rows)")

            trends_path = _write_effort_trends(athlete_df, str(athlete_consolidated_dir))
            if trends_path:
                print(f"Wrote effort trends CSV for {athlete}: {trends_path}")
            sprint_path = _write_sprint_trends(athlete_df, str(athlete_consolidated_dir))
            if sprint_path:
                print(f"Wrote sprint trends CSV for {athlete}: {sprint_path}")

        # Build set summaries and set blocks history per athlete
        try:
            for athlete in all_df['athlete'].unique():
                athlete_consolidated_dir = Path(args.athletes_root) / athlete / "consolidated"
                athlete_consolidated_dir.mkdir(parents=True, exist_ok=True)
                
                # Find set summary files for this athlete
                athlete_dir = Path(args.athletes_root) / athlete
                set_csvs = [str(p) for p in athlete_dir.rglob("*_set_summaries.csv")]
                block_csvs = [str(p) for p in athlete_dir.rglob("*_set_blocks.csv")]
                
                if set_csvs:
                    frames = []
                    for p in set_csvs:
                        df_set = pd.read_csv(p)
                        # Extract category from path
                        category = Path(p).parent.name
                        df_set["workout_category"] = category
                        frames.append(df_set)
                    
                    if frames:
                        set_hist = pd.concat(frames, ignore_index=True)
                        out_path = athlete_consolidated_dir / "set_summaries_history.csv"
                        set_hist.to_csv(out_path, index=False)
                        print(f"Wrote set summaries history CSV for {athlete}: {out_path}")

                if block_csvs:
                    frames_b = []
                    for p in block_csvs:
                        df_block = pd.read_csv(p)
                        category = Path(p).parent.name
                        df_block["workout_category"] = category
                        frames_b.append(df_block)
                    if frames_b:
                        block_hist = pd.concat(frames_b, ignore_index=True)
                        out_path_b = athlete_consolidated_dir / "set_blocks_history.csv"
                        block_hist.to_csv(out_path_b, index=False)
                        print(f"Wrote set blocks history CSV for {athlete}: {out_path_b}")
        except Exception as e:
            print(f"Warning: could not build set summaries history: {e}")
        return

    # Default multi-input mode
    files = _iter_fit_files_many(args.input or [])
    if not files:
        print("No .fit files found.")
        return

    # Parse manual tags (allow comma-separated values)
    manual_tags: List[str] = []
    if args.tag:
        for v in args.tag:
            manual_tags.extend([t.strip() for t in v.split(",") if t.strip()])

    all_rows: List[Dict] = []
    for file_path in files:
        df = ensure_columns(load_fit_to_dataframe(file_path))
        ride = Ride(source_path=file_path, timeline=df, ftp_watts=FTP_WATTS, weight_kg=RIDER_MASS_KG, lthr_bpm=LTHR_BPM)

        # Lap-based intervals and metrics
        laps = load_fit_laps(file_path)
        lap_intervals = laps_to_intervals(df, laps)
        lap_metrics = [compute_interval_metrics(df, itv, ride) for itv in lap_intervals]

        # Recognize sets (used for tagging and per-ride set summaries)
        sets = recognize_repeated_structures(df, lap_intervals, min_reps=args.min_reps)

        # Build tags and dated base filename
        try:
            date_str = pd.to_datetime(df["timestamp"].iloc[0]).strftime("%Y%m%d") if not df.empty else "unknown"
        except Exception:
            date_str = "unknown"
        # Predict category/label and include as a tag
        pred_category, pred_label = categorize_workout(lap_metrics)
        ride_tags_list = _build_ride_tags(file_path, sets, manual_tags, args.tag_parent_dir)
        if pred_category:
            ride_tags_list.insert(0, _sanitize_tag(pred_category))
        tag_str = "+".join(ride_tags_list)
        stem = Path(file_path).stem
        base = f"{date_str}_{stem}"
        base_with_tags = f"{base}__{tag_str}" if tag_str else base

        # Per-ride export
        if lap_metrics:
            export_interval_metrics_csv(lap_metrics, os.path.join(args.output, f"{base_with_tags}_lap_interval_metrics.csv"))

        # Basic set grouping and per-ride set summary export
        if sets:
            set_summaries: List[SetSummary] = [summarize_interval_set(df, s, ride) for s in sets]
            export_set_summaries_csv(set_summaries, os.path.join(args.output, f"{base_with_tags}_set_summaries.csv"))

        # Accumulate for consolidated exports
        ride_tags_str = tag_str if tag_str else None
        all_rows.extend(
            _metrics_to_rows(
                lap_metrics,
                source_file=Path(file_path).name,
                ride_tags=ride_tags_str,
                workout_category=pred_category,
                workout_label=pred_label,
            )
        )

        # Console summary
        print(f"Processed {base}: {len(lap_intervals)} lap intervals (FTP={FTP_WATTS}W, Wt={RIDER_MASS_KG}kg)")

    # Consolidated history + PR flags
    if not all_rows:
        print("No lap intervals found to export.")
        return
    all_df = pd.DataFrame(all_rows)
    all_df = _append_pr_flags(all_df)
    # Save consolidated files per athlete (if athletes_root is available)
    if hasattr(args, 'athletes_root') and args.athletes_root:
        for athlete in all_df['athlete'].unique():
            athlete_consolidated_dir = Path(args.athletes_root) / athlete / "consolidated"
            athlete_consolidated_dir.mkdir(parents=True, exist_ok=True)
            
            # Filter data for this athlete
            athlete_df = all_df[all_df['athlete'] == athlete]
            hist_path = athlete_consolidated_dir / "lap_metrics_history.csv"
            trends_path = _write_effort_trends(athlete_df, str(athlete_consolidated_dir))
            sprint_path = _write_sprint_trends(athlete_df, str(athlete_consolidated_dir))
            
            athlete_df.to_csv(hist_path, index=False)
            print(f"Wrote consolidated CSV for {athlete}: {hist_path} ({len(athlete_df)} rows)")
            
            if trends_path:
                print(f"Wrote effort trends CSV for {athlete}: {trends_path}")
            if sprint_path:
                print(f"Wrote sprint trends CSV for {athlete}: {sprint_path}")
    else:
        # Fallback to output directory if no athletes_root
        hist_path = os.path.join(args.output, "lap_metrics_history.csv")
        trends_path = _write_effort_trends(all_df, args.output)
        sprint_path = _write_sprint_trends(all_df, args.output)
        
        all_df.to_csv(hist_path, index=False)
        print(f"Wrote consolidated CSV: {hist_path} ({len(all_df)} rows)")
        
        if trends_path:
            print(f"Wrote effort trends CSV: {trends_path}")
        if sprint_path:
            print(f"Wrote sprint trends CSV: {sprint_path}")

    # Set summaries history per athlete
    try:
        if hasattr(args, 'athletes_root') and args.athletes_root:
            for athlete in all_df['athlete'].unique():
                athlete_consolidated_dir = Path(args.athletes_root) / athlete / "consolidated"
                athlete_consolidated_dir.mkdir(parents=True, exist_ok=True)
                
                # Find set summary files for this athlete
                athlete_dir = Path(args.athletes_root) / athlete
                set_csvs = [str(p) for p in athlete_dir.rglob("*_set_summaries.csv")]
                block_csvs = [str(p) for p in athlete_dir.rglob("*_set_blocks.csv")]
                
                if set_csvs:
                    frames = []
                    for p in set_csvs:
                        df_set = pd.read_csv(p)
                        # Extract category from path
                        category = Path(p).parent.name
                        df_set["workout_category"] = category
                        frames.append(df_set)
                    
                    if frames:
                        set_hist = pd.concat(frames, ignore_index=True)
                        out_path = athlete_consolidated_dir / "set_summaries_history.csv"
                        set_hist.to_csv(out_path, index=False)
                        print(f"Wrote set summaries history CSV for {athlete}: {out_path}")
                # Consolidate set blocks history
                if block_csvs:
                    frames_b = []
                    for p in block_csvs:
                        df_block = pd.read_csv(p)
                        # Extract category from path
                        category = Path(p).parent.name
                        df_block["workout_category"] = category
                        frames_b.append(df_block)
                    if frames_b:
                        block_hist = pd.concat(frames_b, ignore_index=True)
                        out_path_b = athlete_consolidated_dir / "set_blocks_history.csv"
                        block_hist.to_csv(out_path_b, index=False)
                        print(f"Wrote set blocks history CSV for {athlete}: {out_path_b}")
        else:
            # Fallback to output directory
            set_csvs = sorted([str(fp) for fp in Path(args.output).glob("*_set_summaries.csv")])
            if set_csvs:
                frames = []
                for p in set_csvs:
                    df_set = pd.read_csv(p)
                    df_set["source_file"] = Path(p).name.replace("_set_summaries.csv", ".fit")
                    frames.append(df_set)
                set_hist = pd.concat(frames, ignore_index=True)
                out_path = os.path.join(args.output, "set_summaries_history.csv")
                set_hist.to_csv(out_path, index=False)
                print(f"Wrote set summaries history CSV: {out_path}")
    except Exception as e:
        print(f"Warning: could not build set summaries history: {e}")


if __name__ == "__main__":
    main()
