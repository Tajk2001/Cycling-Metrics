## Workout Foundation (Lap-only Core)

Minimal, clean system for:
- Computing interval-level metrics based on laps from FIT files
- Tracking performance and physiological response per lap

### Design
- Input: FIT files → DataFrame (1 Hz) via `io/fit_loader.py`
- Metrics: interval-level power/HR/efficiency → `metrics/compute.py`
- Storage: CSV export → `storage/export.py`
- CLI: orchestrates pipeline → `cli.py`

FTP, LTHR, and weight are hardcoded in `config.py` (SprintV1-style). Adjust as needed.

### Quick Start
Research‑grade pipeline with bounded gap handling and strict NP.

1) Activate your venv and install deps
```bash
pip install -r requirements.txt
```

2) Run the converged CLI on one or more FIT files or directories (recursive)
```bash
python -m workout_foundation.cli \
  --input /path/to/fits_or_dirs \
  --output /path/to/exports \
  --tag 2x20-thr                # optional single workout tag
# Optional: add one context tag or use parent dir as a tag
# --tag outdoor      # or indoor, race
# --tag-parent-dir   # include parent folder name as a tag
```

3) Outputs (written to `--output`)
- Per ride:
  - `<YYYYMMDD>_<fit-stem>__<tags>_lap_interval_metrics.csv`
  - `<YYYYMMDD>_<fit-stem>__<tags>_set_summaries.csv`
- Consolidated history:
  - `lap_metrics_history.csv` (includes `ride_tags`, `effort_key`, `pr_flag`)
- Trends:
  - `effort_trends.csv` (best per day per effort_key + 28d MA, WoW)
  - `sprint_trends.csv` (Z6/Z7 ≤60s bests + 28d MA, WoW)
- Sets history:
  - `set_summaries_history.csv`

### Athletes Folder Layout (optional)
Organize your data as:
```
/data/athletes/
  alice/
    z2/
      ride1.fit
      ride2.fit
    thr/
      zwift-activity-...fit
  bob/
    vo2-short/
      ...
```

Run on the root and mirror outputs per athlete/category:
```bash
python -m workout_foundation.cli \
  --athletes-root /data/athletes \
  --output /data/exports
```

This will:
- Auto-tag each ride by its workout_category (the subfolder name)
- Write per-ride CSVs under `/data/exports/<athlete>/<category>/...`
- Build consolidated `lap_metrics_history.csv` with `athlete` and `workout_category` columns
- Write trends CSVs and a global `set_summaries_history.csv`

### Simple UI (optional)
A minimal web UI to inspect folders, upload FITs, and run analysis.
```bash
python -m workout_foundation.ui.app \
  --athletes-root /data/athletes \
  --output /data/exports \
  --port 8050
```
Then open http://127.0.0.1:8050

UI features:
- View athletes/categories and FIT files
- Upload FITs into a selected athlete/category folder
- Trigger the full analysis in one click and view logs

### Labeling (Simple & Consistent)
- Use one compact workout tag; optional one context tag.
- Units: `m` (minutes), `s` (seconds). Examples:
  - Threshold: `2x20-thr`, `3x12-thr`
  - Tempo: `60m-tempo`, `3x20-tempo`
  - Endurance: `90m-z2`, `150m-z2`
  - VO2 short (1–4m): `5x3-vo2-short`, `10x30-30-vo2-short`
  - VO2 long (5–8m): `4x8-vo2-long`, `5x6-vo2-long`
  - Anaerobic (30–90s): `8x1-anaerobic`, `10x45s-anaerobic`
  - Sprint (≤15s): `10x10s-sprint`, `8x12s-sprint`
- Context (only if special): `race`, `indoor`, `outdoor`.

Examples:
```bash
python -m workout_foundation.cli \
  --input /Users/you/Desktop/Training \
  --output /path/to/exports \
  --tag 2x20-thr                      # simple threshold day

python -m workout_foundation.cli \
  --input /Users/you/Desktop/Training \
  --output /path/to/exports \
  --tag 5x3-vo2-short,outdoor \
  --min-reps 3
```

### What Gets Computed
- Power: avg, W/kg, NP, peak 5s/30s, work (kJ), VI, IF, TSS
- Heart: avg/max HR, Pw:HR ratio, HR drift (≥10m), Pw:HR decoupling (≥10m), 60s HR recovery
- Execution: % time within ±10% of avg power
- Comparison: effort_key (e.g., `Z4_20m`), `pr_flag` per effort_key, `ride_tags`

### Data Fidelity
- Power: hold up to 3s gaps, longer gaps remain NaN
- HR/Cadence/Speed: interpolate time up to 5s; no extrapolation
- Distance/Altitude: interpolate time up to 10s; no extrapolation
- NP: strict 30s rolling; if <30s valid, falls back to mean power
