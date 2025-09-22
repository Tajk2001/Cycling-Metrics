from __future__ import annotations

import os
from pathlib import Path

# Allow running as a script by falling back to absolute imports
try:
    from .config import TEST_INPUT_PATH, TEST_OUTPUT_DIR, FTP_WATTS, RIDER_MASS_KG, LTHR_BPM
    from .io.fit_loader import load_fit_to_dataframe, ensure_columns
    from .io.lap_loader import load_fit_laps, laps_to_intervals
    from .models.types import Ride
    from .metrics.compute import compute_interval_metrics
    from .storage.export import export_interval_metrics_csv
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from workout_foundation.config import TEST_INPUT_PATH, TEST_OUTPUT_DIR, FTP_WATTS, RIDER_MASS_KG, LTHR_BPM
    from workout_foundation.io.fit_loader import load_fit_to_dataframe, ensure_columns
    from workout_foundation.io.lap_loader import load_fit_laps, laps_to_intervals
    from workout_foundation.models.types import Ride
    from workout_foundation.metrics.compute import compute_interval_metrics
    from workout_foundation.storage.export import export_interval_metrics_csv


def run_single() -> None:
    input_path = TEST_INPUT_PATH
    output_dir = TEST_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Resolve directory to first .fit file if a directory is provided
    p = Path(input_path)
    if p.is_dir():
        fit_files = sorted([str(fp) for fp in p.rglob("*.fit")])
        if not fit_files:
            print(f"No .fit files found in {input_path}")
            return
        input_file = fit_files[0]
    else:
        input_file = str(p)

    df = ensure_columns(load_fit_to_dataframe(input_file))
    ride = Ride(source_path=input_file, timeline=df, ftp_watts=FTP_WATTS, weight_kg=RIDER_MASS_KG, lthr_bpm=LTHR_BPM)

    laps = load_fit_laps(input_file)
    lap_intervals = laps_to_intervals(df, laps)
    lap_interval_metrics = [compute_interval_metrics(df, itv, ride) for itv in lap_intervals]

    base = Path(input_file).stem
    if lap_interval_metrics:
        export_interval_metrics_csv(lap_interval_metrics, os.path.join(output_dir, f"{base}_lap_interval_metrics.csv"))

    print(f"Processed {base}: {len(lap_intervals)} lap intervals (FTP={FTP_WATTS}W, Wt={RIDER_MASS_KG}kg)")


if __name__ == "__main__":
    run_single()
