from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Dict, List

# Ensure package imports work when running this file directly
if (__package__ is None or __package__ == "") and "workout_foundation" not in sys.modules:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from workout_foundation.config import TEST_OUTPUT_DIR
from workout_foundation.models.athlete_profile import AthleteProfile, save_athlete_profile
from workout_foundation.io.fit_loader import load_fit_to_dataframe, ensure_columns


def scan_athletes_layout(root: str) -> pd.DataFrame:
    rows: List[Dict] = []
    root_path = Path(root)
    if not root_path.exists():
        return pd.DataFrame(columns=["athlete", "category", "file", "path"])
    
    # Scan all athlete directories
    for athlete_dir in sorted([p for p in root_path.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        athlete = athlete_dir.name
        has_files = False
        
        # FIT files directly under the athlete folder → "uncategorized"
        direct_fits = list(athlete_dir.glob("*.fit")) + list(athlete_dir.glob("*.FIT"))
        for fp in direct_fits:
            rows.append({"athlete": athlete, "category": "uncategorized", "file": fp.name, "path": str(fp)})
            has_files = True
            
        # Check category directories
        for cat_dir in sorted([p for p in athlete_dir.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name != "consolidated"]):
            category = cat_dir.name
            cat_fits = list(cat_dir.rglob("*.fit")) + list(cat_dir.rglob("*.FIT"))
            for fp in cat_fits:
                rows.append({"athlete": athlete, "category": category, "file": fp.name, "path": str(fp)})
                has_files = True
        
        # If athlete has no FIT files, add a placeholder row to show they exist
        if not has_files:
            rows.append({"athlete": athlete, "category": "no-files", "file": "No FIT files found", "path": ""})
    
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["athlete", "category", "file", "path"])
    return df


def build_app(athletes_root: str, output_root: str) -> Dash:
    app: Dash = dash.Dash(__name__)
    if not output_root:
        output_root = TEST_OUTPUT_DIR

    app.layout = html.Div([
        html.H3("Taj's Interval Analysis – Version 1"),
        
        # File Management Section
        html.Div([
            html.Div([
                html.Label("Athletes Root"),
                dcc.Input(id="athletes-root", value=athletes_root, style={"width": "100%"}),
                html.Div(
                    "Exports: Per-athlete CSVs in category folders, consolidated files in each athlete/consolidated/",
                    style={"marginTop": 6, "fontFamily": "monospace", "color": "green"},
                ),
            ], style={"flex": 1}),
            html.Button("Refresh", id="refresh-btn", n_clicks=0, style={"marginLeft": 10, "height": 36}),
        ], style={"display": "flex", "alignItems": "end", "marginBottom": 10}),
        
        html.Div(id="scan-summary", style={"marginBottom": 10, "fontFamily": "monospace"}),
        
        # Athlete Management
        html.Div([
            html.Div([
                html.Label("New Athlete Name"),
                dcc.Input(id="new-athlete-name", placeholder="Enter athlete name", style={"width": "200px"}),
                html.Button("Add Athlete", id="add-athlete-btn", n_clicks=0, style={"marginLeft": 10}),
                html.Button("Clear CSV Files", id="clear-csv-btn", n_clicks=0, style={"marginLeft": 10, "backgroundColor": "#dc3545", "color": "white", "border": "none"}),
                html.Button("Clear FIT Files", id="clear-fit-btn", n_clicks=0, style={"marginLeft": 10, "backgroundColor": "#fd7e14", "color": "white", "border": "none"}),
                html.Div(id="add-athlete-log", style={"marginLeft": 10, "color": "green", "fontFamily": "monospace"}),
            ], style={"display": "flex", "alignItems": "end", "marginBottom": 10}),
            
            html.Button("Run Analysis (All)", id="run-analysis-btn", n_clicks=0),
            html.Div(id="run-log", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace", "marginTop": 10}),
        ], style={"marginBottom": 20}),
        
        # File Table
        html.H4("Folder Contents"),
        dash_table.DataTable(id="files-table", columns=[
            {"name": "athlete", "id": "athlete"},
            {"name": "category", "id": "category"},
            {"name": "file", "id": "file"},
        ], page_size=10, style_table={"overflowX": "auto"}),
        dcc.Store(id="files-store"),

        # Analysis Section
        html.Hr(style={"margin": "20px 0"}),
        html.H4("Analysis"),
        html.Div([
            html.Div([
                html.Label("Athlete"),
                dcc.Dropdown(id="viz-athlete", placeholder="Select athlete"),
            ], style={"width": "45%", "marginRight": 20}),
            html.Div([
                html.Label("Category"),
                dcc.Dropdown(id="viz-category", placeholder="Select category"),
            ], style={"width": "45%"}),
        ], style={"display": "flex", "marginBottom": 20}),
        
         html.Div(id="analysis-summary", style={"fontFamily": "monospace", "color": "#666", "marginBottom": 10}),
         
         dcc.Tabs(id="analysis-tabs", value="rides", children=[
             dcc.Tab(label="Rides", value="rides"),
             dcc.Tab(label="Laps", value="laps"),
             dcc.Tab(label="Intervals", value="intervals"),
             dcc.Tab(label="Compare", value="compare"),
         ]),
         
         html.Div(id="analysis-content", style={"marginTop": 15}),
         
    ], style={"maxWidth": 1000, "margin": "0 auto", "padding": 20})

    # Callbacks
    @app.callback(
        Output("files-store", "data"),
        Output("files-table", "data"),
        Output("scan-summary", "children"),
        Input("refresh-btn", "n_clicks"),
        Input("athletes-root", "value"),
        prevent_initial_call=False,
    )
    def refresh(_n, root):
        if root and len(root) > 200:  # Likely CLI output text, not a path
            root = athletes_root
        df = scan_athletes_layout(root or athletes_root)
        summary = "No FIT files found."
        if not df.empty:
            summary = f"Found {len(df)} files across {df['athlete'].nunique()} athlete(s) and {df['category'].nunique()} category(ies)."
        data = df[["athlete", "category", "file"]].to_dict("records") if not df.empty else []
        return (df.to_dict("records"), data, summary)

    @app.callback(
        Output("add-athlete-log", "children"),
        Input("add-athlete-btn", "n_clicks"),
        Input("clear-csv-btn", "n_clicks"),
        Input("clear-fit-btn", "n_clicks"),
        State("new-athlete-name", "value"),
        State("athletes-root", "value"),
        prevent_initial_call=True,
    )
    def manage_athletes(n_add, n_clear, n_clear_fit, athlete_name, root):
        ctx = dash.callback_context
        if not ctx.triggered:
            return ""
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "clear-csv-btn":
            try:
                root = root or athletes_root
                athletes_path = Path(root)
                csv_count = 0
                for csv_file in athletes_path.rglob("*.csv"):
                    csv_file.unlink()
                    csv_count += 1
                exports_path = Path(athletes_root).parent / "cycling_data" / "exports"
                if exports_path.exists():
                    for csv_file in exports_path.rglob("*.csv"):
                        csv_file.unlink()
                        csv_count += 1
                return f"🗑️ Cleared {csv_count} CSV files"
            except Exception as e:
                return f"❌ Error clearing CSV files: {e}"
        
        elif button_id == "clear-fit-btn":
            try:
                root = root or athletes_root
                athletes_path = Path(root)
                fit_count = 0
                for fit_file in athletes_path.rglob("*.fit"):
                    fit_file.unlink()
                    fit_count += 1
                for fit_file in athletes_path.rglob("*.FIT"):
                    fit_file.unlink()
                    fit_count += 1
                return f"🗑️ Cleared {fit_count} FIT files (profiles preserved)"
            except Exception as e:
                return f"❌ Error clearing FIT files: {e}"
        
        elif button_id == "add-athlete-btn":
            if not athlete_name or not athlete_name.strip():
                return "Please enter an athlete name"
            
            athlete_name = athlete_name.strip().lower()
            root = root or athletes_root
            
            try:
                athletes_path = Path(root)
                athlete_path = athletes_path / athlete_name
                
                if athlete_path.exists():
                    return f"Athlete '{athlete_name}' already exists"
                
                athlete_path.mkdir(parents=True, exist_ok=True)
                
                # Create default profile
                default_profile = AthleteProfile(
                    name=athlete_name,
                    ftp_watts=250.0,
                    lthr_bpm=170.0,
                    weight_kg=70.0,
                    notes="Default profile - please update with actual values"
                )
                save_athlete_profile(athlete_path, default_profile)
                
                # Create standard workout categories
                categories = ["thr", "vo2-short", "vo2-long", "tempo", "sprint", "anaerobic", "z2"]
                for category in categories:
                    category_path = athlete_path / category
                    category_path.mkdir(exist_ok=True)
                    (category_path / ".gitkeep").touch()
                
                return f"✅ Created athlete '{athlete_name}' with {len(categories)} categories + profile.json"
            except Exception as e:
                return f"❌ Error creating athlete: {e}"
        
        return ""

    @app.callback(
        Output("run-log", "children"),
        Input("run-analysis-btn", "n_clicks"),
        State("athletes-root", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(_n, root):
        root = root or athletes_root
        out = output_root
        cmd = [sys.executable, "-m", "workout_foundation.cli", "--athletes-root", root, "--output", out]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            log = (proc.stdout or "") + "\n" + (proc.stderr or "")
            return log.strip()
        except Exception as e:
            return f"Error running analysis: {e}"

    # Analysis section callbacks
    @app.callback(
        Output("viz-athlete", "options"),
        Input("files-store", "data"),
    )
    def update_athlete_dropdown(files_data):
        if not files_data:
            return []
        df = pd.DataFrame(files_data)
        athletes = sorted(df["athlete"].unique()) if not df.empty else []
        return [{"label": athlete, "value": athlete} for athlete in athletes]

    @app.callback(
        Output("viz-category", "options"),
        Input("viz-athlete", "value"),
        State("files-store", "data"),
    )
    def update_category_dropdown(selected_athlete, files_data):
        if not selected_athlete or not files_data:
            return []
        
        # Standard workout categories
        standard_cats = [
            {"label": "Threshold (THR)", "value": "thr"},
            {"label": "VO2 Short", "value": "vo2-short"},
            {"label": "VO2 Long", "value": "vo2-long"},
            {"label": "Tempo", "value": "tempo"},
            {"label": "Sprint", "value": "sprint"},
            {"label": "Anaerobic", "value": "anaerobic"},
            {"label": "Z2 Endurance", "value": "z2"},
        ]
        
        # Add any additional categories found in files
        df = pd.DataFrame(files_data)
        athlete_files = df[df["athlete"] == selected_athlete]
        file_categories = athlete_files["category"].unique() if not athlete_files.empty else []
        
        # Add file categories that aren't in standard list
        standard_values = [cat["value"] for cat in standard_cats]
        for cat in file_categories:
            if cat not in standard_values and cat != "no-files":
                standard_cats.append({"label": cat, "value": cat})
        
        return standard_cats

    @app.callback(
        Output("analysis-summary", "children"),
        Input("viz-athlete", "value"),
        Input("viz-category", "value"),
        State("files-store", "data"),
    )
    def update_analysis_summary(selected_athlete, selected_category, files_data):
        if not selected_athlete:
            return "Select an athlete to view analysis"
        
        if not selected_category:
            return f"Select a category for {selected_athlete}"
        
        # Count files for this athlete/category combination
        if files_data:
            df = pd.DataFrame(files_data)
            athlete_cat_files = df[(df["athlete"] == selected_athlete) & (df["category"] == selected_category)]
            file_count = len(athlete_cat_files) if not athlete_cat_files.empty else 0
        else:
            file_count = 0
        
        return f"Analysis: {selected_athlete} - {selected_category} ({file_count} files)"

    @app.callback(
        Output("analysis-content", "children"),
        Input("analysis-tabs", "value"),
        Input("viz-athlete", "value"),
        Input("viz-category", "value"),
    )
    def update_analysis_content(tab, selected_athlete, selected_category):
        if not selected_athlete or not selected_category:
            return html.Div("Select athlete and category to view analysis")
        
        # Build paths to consolidated CSV files
        athletes_root_path = Path("/Users/tajkrieger/Projects/cycling_analysis/athletes")
        athlete_dir = athletes_root_path / selected_athlete
        consolidated_dir = athlete_dir / "consolidated"
        
        if tab == "rides":
            return _render_rides_table(consolidated_dir, selected_athlete, selected_category)
        elif tab == "laps":
            return _render_laps_table(consolidated_dir, selected_athlete, selected_category)
        elif tab == "intervals":
            return _render_intervals_table(consolidated_dir, selected_athlete, selected_category)
        elif tab == "compare":
            return _render_compare_interface(consolidated_dir, selected_athlete, selected_category)
        
        return html.Div("Select a tab to view analysis")

    def _render_rides_table(consolidated_dir: Path, athlete: str, category: str):
        """Show basic ride metrics - one row per workout."""
        lap_history_path = consolidated_dir / "lap_metrics_history.csv"
        
        if not lap_history_path.exists():
            return html.Div("No lap metrics history found. Run analysis first.")
        
        try:
            df = pd.read_csv(lap_history_path)
            if df.empty:
                return html.Div("No data in lap metrics history.")
            
            # Filter by category if specified
            if category != "all":
                df = df[df["workout_category"] == category]
            
            if df.empty:
                return html.Div(f"No data for {category} category.")
            
            # Group by workout (source_file + workout_date) to create ride summaries
            ride_summaries = []
            for (source_file, workout_date), group in df.groupby(["source_file", "workout_date"]):
                total_work = group["work_kj"].sum() if "work_kj" in group else 0
                total_time = group["duration_s"].sum() if "duration_s" in group else 0
                avg_power = group["avg_power_w"].mean() if "avg_power_w" in group else 0
                max_power = group["peak_5s_w"].max() if "peak_5s_w" in group else 0
                avg_hr = group["avg_hr_bpm"].mean() if "avg_hr_bpm" in group else 0
                total_distance = group["distance_m"].sum() if "distance_m" in group else 0
                total_elevation = group["elevation_gain_m"].sum() if "elevation_gain_m" in group else 0
                num_intervals = len(group)
                
                ride_summaries.append({
                    "date": workout_date,
                    "file": source_file,
                    "intervals": num_intervals,
                    "total_time_min": round(total_time / 60, 1),
                    "total_work_kj": round(total_work, 1),
                    "avg_power_w": round(avg_power, 1),
                    "max_power_w": round(max_power, 1),
                    "avg_hr_bpm": round(avg_hr, 1),
                    "distance_km": round(total_distance / 1000, 2),
                    "elevation_m": round(total_elevation, 1),
                })
            
            rides_df = pd.DataFrame(ride_summaries).sort_values("date", ascending=False)
            
            # Create table
            table = dash_table.DataTable(
                columns=[
                    {"name": "Date", "id": "date"},
                    {"name": "File", "id": "file"},
                    {"name": "Intervals", "id": "intervals"},
                    {"name": "Time (min)", "id": "total_time_min"},
                    {"name": "Work (kJ)", "id": "total_work_kj"},
                    {"name": "Avg Power (W)", "id": "avg_power_w"},
                    {"name": "Max Power (W)", "id": "max_power_w"},
                    {"name": "Avg HR (bpm)", "id": "avg_hr_bpm"},
                    {"name": "Distance (km)", "id": "distance_km"},
                    {"name": "Elevation (m)", "id": "elevation_m"},
                ],
                data=rides_df.to_dict("records"),
                page_size=15,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "fontSize": "14px"},
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
            )
            
            return html.Div([
                html.H5(f"Rides Summary - {athlete} ({category})"),
                html.P(f"Showing {len(rides_df)} rides with basic metrics"),
                table
            ])
            
        except Exception as e:
            return html.Div(f"Error loading ride data: {e}")

    def _render_laps_table(consolidated_dir: Path, athlete: str, category: str):
        """Show all lap details - one row per lap, grouped by ride."""
        lap_history_path = consolidated_dir / "lap_metrics_history.csv"
        
        if not lap_history_path.exists():
            return html.Div("No lap metrics history found. Run analysis first.")
        
        try:
            df = pd.read_csv(lap_history_path)
            if df.empty:
                return html.Div("No data in lap metrics history.")
            
            # Filter by category if specified
            if category != "all":
                df = df[df["workout_category"] == category]
            
            if df.empty:
                return html.Div(f"No data for {category} category.")
            
            # Sort by date and lap index
            df = df.sort_values(["workout_date", "source_file", "lap_index"])
            
            # Select key columns for interval display
            interval_cols = [
                "workout_date", "source_file", "lap_index", "duration_s", 
                "avg_power_w", "power_wkg", "normalized_power_w", "avg_hr_bpm", 
                "avg_hr_pct_lthr", "power_zone", "work_kj", "kj_per_min",
                "avg_cadence_rpm", "distance_m", "elevation_gain_m"
            ]
            
            # Filter to only columns that exist
            available_cols = [col for col in interval_cols if col in df.columns]
            display_df = df[available_cols].copy()
            
            # Round numeric columns
            numeric_cols = ["duration_s", "avg_power_w", "power_wkg", "normalized_power_w", 
                          "avg_hr_bpm", "avg_hr_pct_lthr", "work_kj", "kj_per_min", 
                          "avg_cadence_rpm", "distance_m", "elevation_gain_m"]
            for col in numeric_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(1)
            
            # Create table with nice column names
            column_mapping = {
                "workout_date": "Date",
                "source_file": "Workout File",
                "lap_index": "Lap #",
                "duration_s": "Duration (s)",
                "avg_power_w": "Avg Power (W)",
                "power_wkg": "W/kg",
                "normalized_power_w": "NP (W)",
                "avg_hr_bpm": "Avg HR",
                "avg_hr_pct_lthr": "% LTHR",
                "power_zone": "Zone",
                "work_kj": "Work (kJ)",
                "kj_per_min": "kJ/min",
                "avg_cadence_rpm": "Cadence",
                "distance_m": "Distance (m)",
                "elevation_gain_m": "Elevation (m)",
            }
            
            table_columns = []
            for col in available_cols:
                table_columns.append({
                    "name": column_mapping.get(col, col),
                    "id": col
                })
            
            table = dash_table.DataTable(
                columns=table_columns,
                data=display_df.to_dict("records"),
                page_size=20,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "fontSize": "13px", "padding": "8px"},
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "#f9f9f9"
                    }
                ]
            )
            
            return html.Div([
                html.H5(f"Laps Detail - {athlete} ({category})"),
                html.P(f"Showing {len(display_df)} laps grouped by workout date"),
                table
            ])
            
        except Exception as e:
            return html.Div(f"Error loading interval data: {e}")

    def _render_intervals_table(consolidated_dir: Path, athlete: str, category: str):
        """Show zone-matched intervals - only intervals that match the category's target zone."""
        lap_history_path = consolidated_dir / "lap_metrics_history.csv"
        
        if not lap_history_path.exists():
            return html.Div("No lap metrics history found. Run analysis first.")
        
        try:
            df = pd.read_csv(lap_history_path)
            if df.empty:
                return html.Div("No data in lap metrics history.")
            
            # Filter by category first
            if category != "all":
                df = df[df["workout_category"] == category]
            
            if df.empty:
                return html.Div(f"No data for {category} category.")
            
            # Zone mapping for categories
            zone_mapping = {
                "tempo": ["Z3 Tempo"],
                "thr": ["Z4 Threshold"],
                "vo2-short": ["Z5 VO2max", "Z6 Anaerobic", "Z7 Neuromuscular"],
                "vo2-long": ["Z5 VO2max"],
                "sprint": ["Z6 Anaerobic", "Z7 Neuromuscular"],
                "anaerobic": ["Z6 Anaerobic", "Z7 Neuromuscular"],
                "z2": ["Z2 Endurance"],
            }
            
            target_zones = zone_mapping.get(category, [])
            # If category has no explicit target zones (e.g., 'all'), don't filter by zone
            if target_zones and "power_zone" in df.columns:
                zone_filtered = df[df["power_zone"].isin(target_zones)]
            else:
                zone_filtered = df.copy()
            
            if zone_filtered.empty:
                return html.Div(f"No intervals found in target zones {target_zones} for {category}.")
            
            # Sort by date and lap index
            zone_filtered = zone_filtered.sort_values(["workout_date", "source_file", "lap_index"])
            
            # Create interval identifier and concise label for selection
            zone_filtered["interval_id"] = zone_filtered.apply(lambda r: f"{r['source_file']}_L{int(r['lap_index'])}", axis=1)
            def _label_row(r):
                date_str = str(r.get('workout_date'))
                lap = int(r.get('lap_index')) if pd.notna(r.get('lap_index')) else 0
                zone = r.get('power_zone') or ''
                pw = r.get('avg_power_w')
                dur = r.get('duration_s')
                bits = []
                if pd.notna(pw):
                    try:
                        bits.append(f"{int(round(float(pw)))}W")
                    except Exception:
                        pass
                if pd.notna(dur):
                    try:
                        ds = float(dur)
                        if ds < 60:
                            dstr = f"{int(ds)}s"
                        else:
                            dm = ds / 60.0
                            dstr = f"{int(round(dm))}min" if abs(dm - round(dm)) < 0.1 else f"{dm:.1f}min"
                        bits.append(dstr)
                    except Exception:
                        pass
                info = f" ({', '.join(bits)})" if bits else ""
                zshort = zone.replace('Z', '').replace(' Tempo', 'T').replace(' Threshold', 'Th').replace(' VO2max', 'V').replace(' Anaerobic', 'A').replace(' Neuromuscular', 'N')
                return f"{date_str} L{lap} - {zshort}{info}"
            zone_filtered["interval_label"] = zone_filtered.apply(_label_row, axis=1)
            
            # Select key columns for interval display
            interval_cols = [
                "workout_date", "source_file", "lap_index", "duration_s", 
                "avg_power_w", "power_wkg", "normalized_power_w", "power_zone",
                "avg_hr_bpm", "avg_hr_pct_lthr", "work_kj", "kj_per_min",
                "avg_cadence_rpm", "distance_m", "elevation_gain_m", "pr_flag"
            ]
            
            # Filter to only columns that exist
            available_cols = [col for col in interval_cols if col in zone_filtered.columns]
            display_df = zone_filtered[available_cols].copy()
            
            # Round numeric columns
            numeric_cols = ["duration_s", "avg_power_w", "power_wkg", "normalized_power_w", 
                          "avg_hr_bpm", "avg_hr_pct_lthr", "work_kj", "kj_per_min", 
                          "avg_cadence_rpm", "distance_m", "elevation_gain_m"]
            for col in numeric_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(1)
            
            # Create table with nice column names
            column_mapping = {
                "workout_date": "Date",
                "source_file": "Workout File",
                "lap_index": "Lap #",
                "duration_s": "Duration (s)",
                "avg_power_w": "Avg Power (W)",
                "power_wkg": "W/kg",
                "normalized_power_w": "NP (W)",
                "power_zone": "Zone",
                "avg_hr_bpm": "Avg HR",
                "avg_hr_pct_lthr": "% LTHR",
                "work_kj": "Work (kJ)",
                "kj_per_min": "kJ/min",
                "avg_cadence_rpm": "Cadence",
                "distance_m": "Distance (m)",
                "elevation_gain_m": "Elevation (m)",
                "pr_flag": "PR",
            }
            
            table_columns = []
            for col in available_cols:
                table_columns.append({
                    "name": column_mapping.get(col, col),
                    "id": col
                })
            
            table = dash_table.DataTable(
                columns=table_columns,
                data=display_df.to_dict("records"),
                page_size=20,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "fontSize": "13px", "padding": "8px"},
                style_header={"backgroundColor": "#e3f2fd", "fontWeight": "bold"},
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "#f3f8ff"
                    },
                    {
                        "if": {"filter_query": "{pr_flag} = True"},
                        "backgroundColor": "#fff3cd",
                        "color": "#856404"
                    }
                ]
            )
            
            zone_list = ", ".join(target_zones)
            # Build interval options and store
            interval_options = [
                {"label": row.interval_label, "value": row.interval_id}
                for _, row in zone_filtered.iterrows()
            ]
            # Default selection: first two intervals if present
            default_select = [opt["value"] for opt in interval_options[:2]]
            
            # Keep a lightweight store for callback
            diff_cols = [
                "interval_id", "interval_label", "workout_date", "source_file", "lap_index",
                "duration_s", "avg_power_w", "power_wkg", "normalized_power_w", "avg_hr_bpm",
                "avg_hr_pct_lthr", "work_kj", "kj_per_min", "avg_cadence_rpm", "distance_m", "elevation_gain_m"
            ]
            diff_df = zone_filtered[[c for c in diff_cols if c in zone_filtered.columns]].copy()
            
            return html.Div([
                html.H5(f"Target Zone Intervals - {athlete} ({category})"),
                html.P(f"Showing {len(display_df)} intervals in target zones: {zone_list}"),
                html.P("These are the intervals that match the intended training zone for this category."),
                table,
                html.Hr(),
                html.H5("Compare Selected Intervals"),
                html.P("Pick two or more intervals. The first selected is the baseline; values show ±Δ vs baseline."),
                dcc.Dropdown(id="intervals-diff-select", options=interval_options, value=default_select, multi=True, placeholder="Choose intervals to compare...", style={"maxWidth": 700}),
                html.Div(id="intervals-diff-output", style={"marginTop": 10}),
                dcc.Store(id="intervals-data", data=diff_df.to_dict("records")),
            ])
            
        except Exception as e:
            return html.Div(f"Error loading interval data: {e}")

    def _render_compare_interface(consolidated_dir: Path, athlete: str, category: str):
        """Show interactive comparison interface for selecting intervals and variables to plot."""
        lap_history_path = consolidated_dir / "lap_metrics_history.csv"
        
        if not lap_history_path.exists():
            return html.Div("No lap metrics history found. Run analysis first.")
        
        try:
            df = pd.read_csv(lap_history_path)
            if df.empty:
                return html.Div("No data in lap metrics history.")
            
            # Filter by category if specified
            if category != "all":
                df = df[df["workout_category"] == category]
            
            if df.empty:
                return html.Div(f"No data for {category} category.")
            
            # Convert workout_date to datetime for proper sorting
            df["workout_date"] = pd.to_datetime(df["workout_date"])
            df = df.sort_values(["workout_date", "source_file", "lap_index"])
            
            # Create interval identifiers for selection - use simpler format
            df["interval_id"] = df.apply(lambda row: f"{row['source_file']}_L{row['lap_index']}", axis=1)
            
            # Define plottable numeric columns that actually exist in the data
            numeric_columns = [
                "avg_power_w", "power_wkg", "normalized_power_w", "peak_5s_w", "peak_30s_w",
                "avg_hr_bpm", "avg_hr_pct_lthr", "max_hr_bpm", 
                "work_kj", "kj_per_min", "duration_s",
                "avg_cadence_rpm", "distance_m", "elevation_gain_m",
                "variability_index_vi", "intensity_factor_if", "efficiency_factor_ef"
            ]
            
            # Filter to only columns that exist in the data
            available_columns = [col for col in numeric_columns if col in df.columns and df[col].notna().any()]
            
            # Create simplified column display names
            column_labels = {
                "avg_power_w": "Power (W)",
                "power_wkg": "W/kg",
                "normalized_power_w": "NP (W)",
                "peak_5s_w": "Peak 5s (W)",
                "peak_30s_w": "Peak 30s (W)",
                "avg_hr_bpm": "HR",
                "avg_hr_pct_lthr": "HR %",
                "max_hr_bpm": "Max HR",
                "work_kj": "Work (kJ)",
                "kj_per_min": "kJ/min",
                "duration_s": "Duration (s)",
                "avg_cadence_rpm": "Cadence",
                "distance_m": "Distance (m)",
                "elevation_gain_m": "Elevation (m)",
                "variability_index_vi": "VI",
                "intensity_factor_if": "IF",
                "efficiency_factor_ef": "EF"
            }
            
            column_options = [{"label": column_labels.get(col, col), "value": col} for col in available_columns]
            
            # Create interval selection options with concise labels, grouped by workout
            interval_options = []
            
            # Group by workout (date + file) for better organization
            workouts = df.groupby(['workout_date', 'source_file'])
            
            for (workout_date, source_file), workout_df in workouts:
                date_str = workout_date.strftime('%m/%d')
                
                # Create workout header (disabled option for grouping)
                clean_name = source_file.replace('zwift-activity-', '').replace('.fit', '').replace('.FIT', '')
                if len(clean_name) > 12:
                    clean_name = clean_name[:12]
                
                workout_header = f"{date_str} - {clean_name}"
                interval_options.append({"label": workout_header, "value": f"header_{workout_date}_{source_file}", "disabled": True})
                
                # Add intervals for this workout
                for _, row in workout_df.iterrows():
                    if row.get('power_zone'):
                        # Use zone abbreviation for cleaner look
                        zone_short = row['power_zone'].replace('Z', '').replace(' Tempo', 'T').replace(' Threshold', 'Th').replace(' VO2max', 'V').replace(' Anaerobic', 'A').replace(' Neuromuscular', 'N')[:4]
                        label = f"   L{row['lap_index']} - {zone_short}"
                    else:
                        # Fallback without zone
                        label = f"   L{row['lap_index']}"
                    
                    # Add power/duration info for quick reference with smart duration formatting
                    if pd.notna(row.get('avg_power_w')) and pd.notna(row.get('duration_s')):
                        power = int(row['avg_power_w'])
                        duration_s = row['duration_s']
                        
                        # Format duration smartly
                        if duration_s < 60:
                            # Less than 1 minute: show seconds
                            duration_str = f"{int(duration_s)}s"
                        else:
                            # 1 minute or more: show minutes with .5 decimals if needed
                            duration_min = duration_s / 60
                            if abs(duration_min - round(duration_min)) < 0.1:
                                # Close to whole number
                                duration_str = f"{int(round(duration_min))}min"
                            else:
                                # Show .5 precision
                                duration_str = f"{duration_min:.1f}min"
                        
                        label += f" ({power}W, {duration_str})"
                    
                    interval_options.append({"label": label, "value": row["interval_id"]})
            
            return html.Div([
                html.H5(f"Compare Intervals - {athlete} ({category})"),
                html.P("Select intervals and choose mode: trend over dates or overlay time-series."),

                html.Div([
                    html.Div([
                        html.Label("Compare Mode", style={"fontWeight": "bold", "marginBottom": 6}),
                        dcc.RadioItems(
                            id="compare-mode",
                            options=[
                                {"label": "Trend over dates", "value": "trend"},
                                {"label": "Overlay intervals", "value": "overlay"},
                                {"label": "Table compare", "value": "table"},
                            ],
                            value="overlay",
                            labelStyle={"display": "inline-block", "marginRight": 15},
                            style={"marginBottom": 12}
                        ),
                        html.Label("Trend Variables (trend mode)", style={"fontWeight": "bold", "marginBottom": 6}),
                        dcc.Dropdown(
                            id="compare-variables",
                            options=column_options,
                            value=["avg_power_w", "avg_hr_bpm"] if len(column_options) >= 2 else [column_options[0]["value"]] if column_options else [],
                            multi=True,
                            placeholder="Choose variables to compare (trend mode)"
                        ),
                        html.Div(style={"height": 8}),
                        dcc.RadioItems(
                            id="trend-group-mode",
                            options=[
                                {"label": "Group by Effort Family", "value": "family"},
                                {"label": "Selected intervals only", "value": "selected"},
                            ],
                            value="family",
                            labelStyle={"display": "inline-block", "marginRight": 15}
                        ),
                        html.Div(style={"height": 12}),
                        html.Label("Series Variable (overlay mode)", style={"fontWeight": "bold", "marginBottom": 6}),
                        dcc.Dropdown(
                            id="overlay-series",
                            options=[
                                {"label": "Power (W)", "value": "power"},
                                {"label": "Heart Rate (bpm)", "value": "heart_rate"},
                                {"label": "Cadence (rpm)", "value": "cadence"},
                                {"label": "Speed (m/s)", "value": "speed"},
                                {"label": "Altitude (m)", "value": "altitude"},
                            ],
                            value="power",
                            clearable=False
                        ),
                        html.Div(style={"height": 8}),
                        dcc.RadioItems(
                            id="overlay-xmode",
                            options=[
                                {"label": "X axis: Time (s)", "value": "time"},
                                {"label": "X axis: Percent (%)", "value": "percent"},
                            ],
                            value="time",
                            labelStyle={"display": "inline-block", "marginRight": 15}
                        ),
                        html.Div(style={"height": 8}),
                        html.Label("Smoothing (s)", style={"fontWeight": "bold", "marginBottom": 6}),
                        dcc.Slider(id="overlay-smooth", min=1, max=20, step=1, value=5,
                                   marks={1: "1", 5: "5", 10: "10", 15: "15", 20: "20"}),
                    ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
                    
                    html.Div([
                        html.Label("Select Intervals to Compare:", style={"fontWeight": "bold", "marginBottom": 10}),
                        dcc.Dropdown(
                            id="compare-intervals",
                            options=interval_options,
                            value=[opt["value"] for opt in interval_options if not opt.get("disabled", False)][:5],
                            multi=True,
                            placeholder="Choose intervals to compare..."
                        ),
                    ], style={"width": "48%", "float": "right", "display": "inline-block", "verticalAlign": "top"}),
                ], style={"marginBottom": 20}),
                
                html.Div(id="comparison-plot"),
                
                # Store the data for the callback
                dcc.Store(id="comparison-data", data=df.to_dict("records")),
            ])
            
        except Exception as e:
            return html.Div(f"Error loading comparison data: {e}")

    # Add callback for the comparison plot with trend lines and overlay mode
    @app.callback(
        Output("comparison-plot", "children"),
        [Input("compare-mode", "value"),
         Input("compare-variables", "value"),
         Input("trend-group-mode", "value"),
         Input("overlay-series", "value"),
         Input("overlay-xmode", "value"),
         Input("overlay-smooth", "value"),
         Input("compare-intervals", "value"),
         Input("comparison-data", "data")]
    )
    def update_comparison_plot(compare_mode, selected_variables, trend_group_mode, overlay_series, overlay_xmode, overlay_smooth, selected_intervals, data):
        if not selected_intervals or not data:
            return html.Div("Select intervals to generate comparison plot.")
        
        try:
            # Convert data back to DataFrame early for both modes
            df = pd.DataFrame(data)
            if "workout_date" in df.columns:
                df["workout_date"] = pd.to_datetime(df["workout_date"], errors="coerce")
            # Table compare mode: build a table of absolute values and % differences vs baseline
            if compare_mode == "table":
                # Filter valid intervals
                valid_intervals = [i for i in selected_intervals if not str(i).startswith("header_")]
                if not valid_intervals:
                    return html.Div("Select one or more intervals.")
                # Choose metrics to show; fallback to a compact default set if none selected
                var_list = selected_variables if selected_variables else [
                    "avg_power_w", "avg_hr_bpm", "duration_s", "normalized_power_w"
                ]
                var_list = [v for v in var_list if v in df.columns]
                if not var_list:
                    return html.Div("No comparable metrics available in data.")
                # Labels for display
                column_labels = {
                    "avg_power_w": "Power (W)", "power_wkg": "W/kg", "normalized_power_w": "NP (W)",
                    "peak_5s_w": "Peak 5s (W)", "peak_30s_w": "Peak 30s (W)", "avg_hr_bpm": "HR",
                    "avg_hr_pct_lthr": "HR %", "max_hr_bpm": "Max HR", "work_kj": "Work (kJ)",
                    "kj_per_min": "kJ/min", "duration_s": "Duration (s)", "avg_cadence_rpm": "Cadence",
                    "distance_m": "Distance (m)", "elevation_gain_m": "Elevation (m)",
                    "variability_index_vi": "VI", "intensity_factor_if": "IF", "efficiency_factor_ef": "EF"
                }
                # Utility formatters
                def _fmt_val(x):
                    try:
                        return f"{float(x):.1f}" if pd.notna(x) else ""
                    except Exception:
                        return ""
                def _fmt_delta(a, b):
                    if pd.isna(a) or pd.isna(b):
                        return ""
                    try:
                        a_f = float(a); b_f = float(b)
                        d = b_f - a_f
                        return f"{d:+.1f}"
                    except Exception:
                        return ""
                def _fmt_pct(a, b):
                    if pd.isna(a) or pd.isna(b):
                        return ""
                    try:
                        a_f = float(a); b_f = float(b)
                        pct = (b_f - a_f) / a_f * 100.0 if a_f != 0 else float('nan')
                        return f"{pct:+.1f}%" if not np.isnan(pct) else ""
                    except Exception:
                        return ""
                # Build columns: Interval, then for each variable: Value, Δ, %Δ
                columns = [{"name": "Interval", "id": "interval"}]
                for v in var_list:
                    lab = column_labels.get(v, v)
                    columns.append({"name": lab, "id": f"{v}__val"})
                    columns.append({"name": f"{lab} Δ", "id": f"{v}__delta"})
                    columns.append({"name": f"{lab} %Δ", "id": f"{v}__pct"})
                # Build rows
                base_id = valid_intervals[0]
                base_row = df[df["interval_id"] == base_id].iloc[0]
                def _interval_label(row):
                    lap_idx = row.get("lap_index")
                    zone = row.get("power_zone") or ""
                    avg_power = row.get("avg_power_w")
                    duration_s = row.get("duration_s")
                    bits = []
                    if pd.notna(avg_power):
                        try:
                            bits.append(f"{int(round(float(avg_power)))}W")
                        except Exception:
                            pass
                    if pd.notna(duration_s):
                        try:
                            ds = float(duration_s)
                            bits.append(f"{int(ds)}s" if ds < 60 else (f"{int(round(ds/60))}min" if abs(ds/60 - round(ds/60)) < 0.1 else f"{ds/60:.1f}min"))
                        except Exception:
                            pass
                    suffix = f" ({', '.join(bits)})" if bits else ""
                    name = f"L{int(lap_idx)}" if pd.notna(lap_idx) else str(row.get("interval_id"))
                    if zone:
                        name += f" - {zone}"
                    return name + suffix
                rows = []
                for iid in valid_intervals:
                    row = df[df["interval_id"] == iid].iloc[0]
                    r = {"interval": _interval_label(row)}
                    for v in var_list:
                        base_v = base_row.get(v)
                        cur_v = row.get(v)
                        r[f"{v}__val"] = _fmt_val(cur_v)
                        r[f"{v}__delta"] = _fmt_delta(base_v, cur_v)
                        r[f"{v}__pct"] = _fmt_pct(base_v, cur_v)
                    rows.append(r)
                table = dash_table.DataTable(
                    columns=columns,
                    data=rows,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "fontSize": "13px", "padding": "8px"},
                    style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
                )
                # Baseline label
                base_label = _interval_label(base_row)
                return html.Div([
                    html.H5("Table Compare", style={"marginBottom": 8}),
                    html.Div([html.Span("Baseline:", style={"fontWeight": "bold", "marginRight": 6}), html.Span(base_label)], style={"marginBottom": 8}),
                    table
                ])

            # Overlay mode: load FIT time-series and overlay selected intervals
            if compare_mode == "overlay":
                athletes_root_path = Path("/Users/tajkrieger/Projects/cycling_analysis/athletes")
                valid_intervals = [i for i in selected_intervals if not str(i).startswith("header_")]
                if not valid_intervals:
                    return html.Div("Select one or more intervals.")
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                color_map = {interval: colors[i % len(colors)] for i, interval in enumerate(valid_intervals)}
                fig = go.Figure()
                traces = 0
                # smoothing window from slider
                try:
                    smooth_win = max(1, int(overlay_smooth)) if overlay_smooth is not None else 5
                except Exception:
                    smooth_win = 5
                for interval_id in valid_intervals:
                    row_df = df[df["interval_id"] == interval_id]
                    if row_df.empty:
                        continue
                    row = row_df.iloc[0]
                    source_file = row.get("source_file")
                    athlete_name = row.get("athlete")
                    category = row.get("workout_category")
                    start_dt = pd.to_datetime(row.get("start_time"), errors="coerce")
                    end_dt = pd.to_datetime(row.get("end_time"), errors="coerce")
                    if pd.isna(start_dt) or pd.isna(end_dt) or not source_file or not athlete_name:
                        continue
                    candidate = athletes_root_path / str(athlete_name) / str(category) / str(source_file)
                    file_path = candidate if candidate.exists() else None
                    if file_path is None or not file_path.exists():
                        try:
                            for p in (athletes_root_path / str(athlete_name)).rglob(str(source_file)):
                                file_path = p
                                break
                        except Exception:
                            file_path = None
                    if file_path is None or not file_path.exists():
                        continue
                    try:
                        df_raw = ensure_columns(load_fit_to_dataframe(str(file_path)))
                    except Exception:
                        continue
                    if "timestamp" not in df_raw.columns:
                        continue
                    if not np.issubdtype(df_raw["timestamp"].dtype, np.datetime64):
                        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
                    seg = df_raw[(df_raw["timestamp"] >= start_dt) & (df_raw["timestamp"] <= end_dt)].copy()
                    if seg.empty or overlay_series not in seg.columns:
                        continue
                    seg["rel_s"] = (seg["timestamp"] - start_dt).dt.total_seconds()
                    y_raw = seg[overlay_series]
                    y = y_raw.rolling(window=smooth_win, min_periods=1).mean()
                    mask = y.notna() & seg["rel_s"].notna()
                    seg = seg[mask]
                    y = y[mask]
                    if seg.empty:
                        continue
                    if overlay_xmode == "percent":
                        dur = max(seg["rel_s"].max(), 1e-9)
                        x = seg["rel_s"] / dur * 100.0
                        x_title = "Percent of interval (%)"
                    else:
                        x = seg["rel_s"]
                        x_title = "Time from start (s)"
                    lap_idx = row.get("lap_index")
                    zone = row.get("power_zone") or ""
                    avg_power = row.get("avg_power_w")
                    duration_s = row.get("duration_s")
                    info_bits = []
                    if pd.notna(avg_power):
                        try:
                            info_bits.append(f"{int(round(float(avg_power)))}W")
                        except Exception:
                            pass
                    if pd.notna(duration_s):
                        try:
                            ds = float(duration_s)
                            if ds < 60:
                                dstr = f"{int(ds)}s"
                            else:
                                dm = ds / 60.0
                                dstr = f"{int(round(dm))}min" if abs(dm - round(dm)) < 0.1 else f"{dm:.1f}min"
                            info_bits.append(dstr)
                        except Exception:
                            pass
                    suffix = f" ({', '.join(info_bits)})" if info_bits else ""
                    name = f"L{int(lap_idx)}" if pd.notna(lap_idx) else str(interval_id)
                    if zone:
                        name += f" - {zone}"
                    name += suffix
                    color = color_map[interval_id]
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=2)))
                    traces += 1
                if traces == 0:
                    return html.Div("No time-series data found for selected intervals.")
                y_titles = {
                    "power": "Power (W)",
                    "heart_rate": "Heart Rate (bpm)",
                    "cadence": "Cadence (rpm)",
                    "speed": "Speed (m/s)",
                    "altitude": "Altitude (m)",
                }
                fig.update_layout(
                    title=dict(text=f"Overlay: {y_titles.get(overlay_series, overlay_series)}", font=dict(size=20, color="#2c3e50"), x=0.5),
                    xaxis=dict(title=x_title, tickfont=dict(size=12, color="#7f8c8d"), gridcolor="#ecf0f1", showgrid=True),
                    yaxis=dict(title=y_titles.get(overlay_series, overlay_series), tickfont=dict(size=12, color="#7f8c8d"), gridcolor="#ecf0f1", showgrid=True),
                    legend=dict(font=dict(size=11, color="#2c3e50"), bgcolor="rgba(255,255,255,0.8)", bordercolor="#bdc3c7", borderwidth=1),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(l=60, r=60, t=80, b=60)
                )
                return dcc.Graph(figure=fig)
            # Define simplified column labels for cleaner graphs
            column_labels = {
                "avg_power_w": "Power (W)", "power_wkg": "W/kg", "normalized_power_w": "NP (W)",
                "peak_5s_w": "Peak 5s (W)", "peak_30s_w": "Peak 30s (W)", "avg_hr_bpm": "HR",
                "avg_hr_pct_lthr": "HR %", "max_hr_bpm": "Max HR", "work_kj": "Work (kJ)",
                "kj_per_min": "kJ/min", "duration_s": "Duration (s)", "avg_cadence_rpm": "Cadence",
                "distance_m": "Distance (m)", "elevation_gain_m": "Elevation (m)",
                "variability_index_vi": "VI", "intensity_factor_if": "IF", "efficiency_factor_ef": "EF"
            }
            if not selected_variables:
                return html.Div("Select one or more variables for trend mode.")
            
            # Convert data back to DataFrame
            df = pd.DataFrame(data)
            df["workout_date"] = pd.to_datetime(df["workout_date"])
            
            # Filter to selected intervals - handle both header and actual interval selections
            valid_intervals = [interval for interval in selected_intervals if not interval.startswith("header_")]
            filtered_df = df[df["interval_id"].isin(valid_intervals)]
            
            if filtered_df.empty:
                return html.Div("No data found for selected intervals.")
            
            # Sort by date for proper line connections
            filtered_df = filtered_df.sort_values("workout_date")
            
            # Calculate summary statistics per selection or by effort_family
            summary_stats = []
            def _trend_for(df_in, var):
                d = df_in[["workout_date", var]].dropna().copy()
                if d.empty:
                    return None
                # Average values by date to avoid duplicate points per day
                d = d.groupby(d["workout_date"].dt.date)[var].mean().reset_index()
                if len(d) < 2:
                    return None
                dates = pd.to_datetime(d["workout_date"])  # ensure datetime index
                dn = pd.to_numeric(dates)
                vals = d[var].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(dn, vals)
                return {
                    "dates": dates,
                    "trend_y": slope * dn + intercept,
                    "first_value": float(vals[0]),
                    "last_value": float(vals[-1]),
                    "change": float(vals[-1] - vals[0]),
                    "pct_change": float(((vals[-1] - vals[0]) / vals[0] * 100.0) if vals[0] != 0 else 0.0),
                    "slope": float(slope),
                    "r2": float(r_value**2),
                }
            for variable in selected_variables:
                if trend_group_mode == "family" and "effort_family" in filtered_df.columns:
                    for fam, grp in filtered_df.groupby("effort_family"):
                        res = _trend_for(grp, variable)
                        if res:
                            summary_stats.append({
                                "variable": variable,
                                "interval": f"Family {fam}",
                                "first_value": res["first_value"],
                                "last_value": res["last_value"],
                                "change": res["change"],
                                "pct_change": res["pct_change"],
                                "slope": res["slope"],
                                "r_squared": res["r2"],
                                "trend": 'Improving' if res["slope"] > 0 else 'Declining' if res["slope"] < 0 else 'Stable'
                            })
                else:
                    res = _trend_for(filtered_df, variable)
                    if res:
                        summary_stats.append({
                            "variable": variable,
                            "interval": "All selected",
                            "first_value": res["first_value"],
                            "last_value": res["last_value"],
                            "change": res["change"],
                            "pct_change": res["pct_change"],
                            "slope": res["slope"],
                            "r_squared": res["r2"],
                            "trend": 'Improving' if res["slope"] > 0 else 'Declining' if res["slope"] < 0 else 'Stable'
                        })
            
            # Create plots with trend lines
            if len(selected_variables) == 1:
                var_name = column_labels.get(selected_variables[0], selected_variables[0])
                
                # Create a professional color palette
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                
                fig = go.Figure()
                
                # Add data points and trend lines for each interval
                for i, interval_id in enumerate(valid_intervals):
                    interval_data = filtered_df[filtered_df["interval_id"] == interval_id]
                    if not interval_data.empty:
                        color = colors[i % len(colors)]
                        
                        # Sort by date for proper line connections
                        interval_data = interval_data.sort_values("workout_date")
                        
                        # Add data points with connecting lines
                        fig.add_trace(go.Scatter(
                            x=interval_data["workout_date"],
                            y=interval_data[selected_variables[0]],
                            mode="markers+lines",
                            name=interval_id.replace('_L', ' - Lap ').replace('.fit', ''),
                            line=dict(color=color, width=4, shape='linear'),  # Thicker, more visible lines
                            marker=dict(color=color, size=10, line=dict(width=2, color='white')),
                            hovertemplate=f"<b>{var_name}</b><br>" +
                                        "Date: %{x}<br>" +
                                        "Value: %{y}<br>" +
                                        "<extra></extra>",
                            connectgaps=False  # Don't connect across gaps
                        ))
                        
                        # Add trend line if we have at least 2 points
                        if len(interval_data) >= 2:
                            # Remove NaN values for trend calculation
                            clean_data = interval_data[["workout_date", selected_variables[0]]].dropna()
                            if len(clean_data) >= 2:
                                # Calculate trend line
                                dates_numeric = pd.to_numeric(clean_data["workout_date"])
                                values = clean_data[selected_variables[0]].values
                                slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, values)
                                
                                # Create trend line points
                                trend_y = slope * dates_numeric + intercept
                                
                                fig.add_trace(go.Scatter(
                                    x=clean_data["workout_date"],
                                    y=trend_y,
                                    mode="lines",
                                    name=f"Trend - {interval_id.replace('_L', ' - Lap ').replace('.fit', '')}",
                                    line=dict(color=color, dash="dash", width=2),
                                    opacity=0.7,
                                    hovertemplate=f"<b>Trend Line</b><br>" +
                                                "Date: %{x}<br>" +
                                                "Predicted: %{y}<br>" +
                                                f"R²: {r_value**2:.3f}<br>" +
                                                "<extra></extra>"
                                ))
                
                # Add overall or family trend line(s) atop all intervals
                def _trend_series(df_in, var):
                    d = df_in[["workout_date", var]].dropna().copy()
                    d = d.groupby(d["workout_date"].dt.date)[var].mean().reset_index()
                    if len(d) < 2:
                        return None
                    dates = pd.to_datetime(d["workout_date"]) ; dn = pd.to_numeric(dates)
                    vals = d[var].values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(dn, vals)
                    return dates, (slope * dn + intercept)
                if summary_stats:
                    if trend_group_mode == "family" and "effort_family" in filtered_df.columns:
                        fam_palette = ['#2c3e50', '#16a085', '#8e44ad', '#c0392b']
                        for idx, (fam, grp) in enumerate(filtered_df.groupby("effort_family")):
                            res = _trend_series(grp, selected_variables[0])
                            if res:
                                dates, trend_y = res
                                fig.add_trace(go.Scatter(x=dates, y=trend_y, mode="lines", name=f"Trend - Family {fam}", line=dict(color=fam_palette[idx % len(fam_palette)], dash="dash", width=3)))
                    else:
                        res = _trend_series(filtered_df, selected_variables[0])
                        if res:
                            dates, trend_y = res
                            fig.add_trace(go.Scatter(x=dates, y=trend_y, mode="lines", name="Trend - All selected", line=dict(color="#2c3e50", dash="dash", width=3)))

                # Professional styling
                fig.update_layout(
                    title=dict(
                        text=f"{var_name} Over Time",
                        font=dict(size=20, color="#2c3e50"),
                        x=0.5
                    ),
                    xaxis=dict(
                        title=dict(text="Date", font=dict(size=14, color="#34495e")),
                        tickfont=dict(size=12, color="#7f8c8d"),
                        gridcolor="#ecf0f1",
                        showgrid=True,
                        tickformat="%m/%d/%y",
                        tickangle=-45,
                        nticks=min(8, len(filtered_df["workout_date"].unique()))
                    ),
                    yaxis=dict(
                        title=dict(text=var_name, font=dict(size=14, color="#34495e")),
                        tickfont=dict(size=12, color="#7f8c8d"),
                        gridcolor="#ecf0f1",
                        showgrid=True
                    ),
                    legend=dict(
                        font=dict(size=11, color="#2c3e50"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#bdc3c7",
                        borderwidth=1
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=600,
                    margin=dict(l=60, r=60, t=80, b=60),
                    hovermode="x unified"
                )
            else:
                # Create subplot for each variable with clean labels
                clean_titles = [column_labels.get(var, var) for var in selected_variables]
                
                fig = make_subplots(
                    rows=len(selected_variables), 
                    cols=1,
                    subplot_titles=clean_titles,
                    vertical_spacing=0.08,
                    shared_xaxes=True
                )
                
                # Professional color palette
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                interval_color_map = {interval: colors[i % len(colors)] for i, interval in enumerate(valid_intervals)}
                
                for i, variable in enumerate(selected_variables, 1):
                    clean_var_name = column_labels.get(variable, variable)
                    
                    for j, interval_id in enumerate(valid_intervals):
                        interval_data = filtered_df[filtered_df["interval_id"] == interval_id]
                        if not interval_data.empty:
                            # Sort by date
                            interval_data = interval_data.sort_values("workout_date")
                            color = interval_color_map[interval_id]
                            
                            # Add data points with connecting lines
                            fig.add_trace(
                                go.Scatter(
                                    x=interval_data["workout_date"],
                                    y=interval_data[variable],
                                    mode="lines+markers",
                                    name=interval_id.replace('_L', ' - Lap ').replace('.fit', '') if i == 1 else "",  # Only show legend for first subplot
                                    showlegend=(i == 1),
                                    line=dict(color=color, width=4, shape='linear'),  # Thicker, more visible lines
                                    marker=dict(color=color, size=8, line=dict(width=2, color='white')),
                                    hovertemplate=f"<b>{clean_var_name}</b><br>" +
                                                "Date: %{x}<br>" +
                                                "Value: %{y}<br>" +
                                                "<extra></extra>",
                                    connectgaps=False  # Don't connect across gaps
                                ),
                                row=i, col=1
                            )
                            
                            # Add trend line if enough data points
                            clean_data = interval_data[["workout_date", variable]].dropna()
                            if len(clean_data) >= 2:
                                dates_numeric = pd.to_numeric(clean_data["workout_date"])
                                values = clean_data[variable].values
                                slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, values)
                                trend_y = slope * dates_numeric + intercept
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=clean_data["workout_date"],
                                        y=trend_y,
                                        mode="lines",
                                        name="",
                                        showlegend=False,
                                        line=dict(color=color, dash="dash", width=2),
                                        opacity=0.7,
                                        hovertemplate=f"<b>Trend Line</b><br>" +
                                                    "Date: %{x}<br>" +
                                                    "Predicted: %{y}<br>" +
                                                    f"R²: {r_value**2:.3f}<br>" +
                                                    "<extra></extra>"
                                    ),
                                    row=i, col=1
                                )
                    # Add overall/family trend per subplot
                    def _trend_series_subplot(df_in, var):
                        d = df_in[["workout_date", var]].dropna().copy()
                        d = d.groupby(d["workout_date"].dt.date)[var].mean().reset_index()
                        if len(d) < 2:
                            return None
                        dates = pd.to_datetime(d["workout_date"]) ; dn = pd.to_numeric(dates)
                        vals = d[var].values
                        slope, intercept, r_value, p_value, std_err = stats.linregress(dn, vals)
                        return dates, (slope * dn + intercept)
                    if summary_stats:
                        if trend_group_mode == "family" and "effort_family" in filtered_df.columns:
                            fam_palette = ['#2c3e50', '#16a085', '#8e44ad', '#c0392b']
                            for idx, (fam, grp) in enumerate(filtered_df.groupby("effort_family")):
                                res = _trend_series_subplot(grp, variable)
                                if res:
                                    dates, trend_y = res
                                    fig.add_trace(go.Scatter(x=dates, y=trend_y, mode="lines", name="" if i > 1 else f"Trend - Family {fam}", showlegend=(i == 1), line=dict(color=fam_palette[idx % len(fam_palette)], dash="dash", width=3)), row=i, col=1)
                        else:
                            res = _trend_series_subplot(filtered_df, variable)
                            if res:
                                dates, trend_y = res
                                fig.add_trace(go.Scatter(x=dates, y=trend_y, mode="lines", name="" if i > 1 else "Trend - All selected", showlegend=(i == 1), line=dict(color="#2c3e50", dash="dash", width=3)), row=i, col=1)
                    
                    # Use clean variable name for y-axis
                    fig.update_yaxes(
                        title=dict(text=clean_var_name, font=dict(size=12, color="#34495e")), 
                        row=i, col=1, 
                        tickfont=dict(size=10, color="#7f8c8d"),
                        gridcolor="#ecf0f1",
                        showgrid=True
                    )
                
                # Update x-axis styling
                fig.update_xaxes(
                    title=dict(text="Date", font=dict(size=12, color="#34495e")), 
                    row=len(selected_variables), 
                    col=1, 
                    tickfont=dict(size=10, color="#7f8c8d"),
                    tickformat="%m/%d/%y",
                    tickangle=-45,
                    nticks=min(8, len(filtered_df["workout_date"].unique())),
                    gridcolor="#ecf0f1",
                    showgrid=True
                )
                
                # Professional layout
                fig.update_layout(
                    height=350 * len(selected_variables), 
                    title=dict(
                        text="Interval Comparison Over Time",
                        font=dict(size=20, color="#2c3e50"),
                        x=0.5
                    ),
                    legend=dict(
                        font=dict(size=11, color="#2c3e50"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#bdc3c7",
                        borderwidth=1
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(l=60, r=60, t=80, b=60),
                    hovermode="x unified"
                )
            
            # Create summary statistics table
            summary_table = None
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats)
                
                # Create display columns without modifying original data
                summary_df = summary_df.copy()
                summary_df['variable_display'] = summary_df['variable'].map(column_labels).fillna(summary_df['variable'])
                summary_df['first_display'] = summary_df['first_value'].apply(lambda x: f"{x:.1f}")
                summary_df['last_display'] = summary_df['last_value'].apply(lambda x: f"{x:.1f}")
                summary_df['change_display'] = summary_df['change'].apply(lambda x: f"{x:+.1f}")
                summary_df['pct_display'] = summary_df['pct_change'].apply(lambda x: f"{x:+.1f}%")
                summary_df['trend_display'] = summary_df['trend']
                summary_df['r2_display'] = summary_df['r_squared'].apply(lambda x: f"{x:.2f}")
                
                # Create table columns
                table_columns = [
                    {"name": "Variable", "id": "variable_display"},
                    {"name": "Interval", "id": "interval"},
                    {"name": "First", "id": "first_display"},
                    {"name": "Last", "id": "last_display"},
                    {"name": "Change", "id": "change_display"},
                    {"name": "% Change", "id": "pct_display"},
                    {"name": "Trend", "id": "trend_display"},
                    {"name": "R²", "id": "r2_display"}
                ]
                
                summary_table = html.Div([
                    html.H5("📊 Trend Analysis Summary", style={
                        "marginTop": 30, 
                        "marginBottom": 15, 
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                        "textAlign": "center",
                        "borderBottom": "2px solid #3498db",
                        "paddingBottom": "10px"
                    }),
                    dash_table.DataTable(
                        columns=table_columns,
                        data=summary_df.to_dict("records"),
                        style_table={
                            "overflowX": "auto",
                            "border": "1px solid #bdc3c7",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                        },
                        style_cell={
                            "textAlign": "center", 
                            "fontSize": "13px", 
                            "padding": "12px 8px",
                            "fontFamily": "Arial, sans-serif",
                            "border": "1px solid #ecf0f1"
                        },
                        style_header={
                            "backgroundColor": "#3498db", 
                            "color": "white",
                            "fontWeight": "bold",
                            "fontSize": "14px",
                            "textAlign": "center",
                            "border": "1px solid #2980b9"
                        },
                        style_data_conditional=[
                            {
                                "if": {"row_index": "odd"},
                                "backgroundColor": "#f8f9fa"
                            },
                            {
                                "if": {"filter_query": "{trend_display} = Improving"},
                                "backgroundColor": "#d4edda",
                                "color": "#155724",
                                "fontWeight": "bold"
                            },
                            {
                                "if": {"filter_query": "{trend_display} = Declining"},
                                "backgroundColor": "#f8d7da",
                                "color": "#721c24",
                                "fontWeight": "bold"
                            },
                            {
                                "if": {"filter_query": "{trend_display} = Stable"},
                                "backgroundColor": "#fff3cd",
                                "color": "#856404"
                            }
                        ]
                    ),
                    html.Div([
                        html.P("💡 ", style={"display": "inline", "fontSize": "16px"}),
                        html.Span("Improving", style={"backgroundColor": "#d4edda", "color": "#155724", "padding": "2px 6px", "borderRadius": "3px", "fontSize": "12px", "marginRight": "10px"}),
                        html.Span("Declining", style={"backgroundColor": "#f8d7da", "color": "#721c24", "padding": "2px 6px", "borderRadius": "3px", "fontSize": "12px", "marginRight": "10px"}),
                        html.Span("Stable", style={"backgroundColor": "#fff3cd", "color": "#856404", "padding": "2px 6px", "borderRadius": "3px", "fontSize": "12px"}),
                    ], style={"textAlign": "center", "marginTop": "15px", "fontSize": "12px"})
                ])
            
            # Return summary table first, then graph for better visibility
            components = []
            if summary_table:
                components.append(summary_table)
            components.append(dcc.Graph(figure=fig))
            
            return html.Div(components)
            
        except Exception as e:
            import traceback
            return html.Div(f"Error creating plot: {e}<br><br>Traceback: {traceback.format_exc()}")

    # Intervals tab difference table callback
    @app.callback(
        Output("intervals-diff-output", "children"),
        [Input("intervals-diff-select", "value"), Input("intervals-data", "data")]
    )
    def build_intervals_diff(selected_ids, data):
        if not data:
            return html.Div()
        df = pd.DataFrame(data)
        if not selected_ids or len(selected_ids) < 2:
            return html.Div("Select at least two intervals to compare.", style={"color": "#666"})
        # Ensure unique and preserve order
        selected = []
        for i in selected_ids:
            if i not in selected:
                selected.append(i)
        base_id = selected[0]
        others = selected[1:]
        base_row = df[df["interval_id"] == base_id]
        if base_row.empty:
            return html.Div("Baseline interval not found.")
        base = base_row.iloc[0]
        metrics = [
            ("avg_power_w", "Avg Power (W)"),
            ("normalized_power_w", "NP (W)"),
            ("power_wkg", "W/kg"),
            ("avg_hr_bpm", "Avg HR"),
            ("avg_hr_pct_lthr", "% LTHR"),
            ("duration_s", "Duration (s)"),
            ("work_kj", "Work (kJ)"),
            ("kj_per_min", "kJ/min"),
            ("avg_cadence_rpm", "Cadence"),
            ("distance_m", "Distance (m)"),
            ("elevation_gain_m", "Elevation (m)")
        ]
        rows = []
        def _fmt_delta(a, b):
            if pd.isna(a) or pd.isna(b):
                return ""
            try:
                a_f = float(a); b_f = float(b)
                d = b_f - a_f
                pct = (d / a_f * 100.0) if a_f != 0 else float('nan')
                pct_str = f" ({pct:+.1f}%)" if not np.isnan(pct) else ""
                return f"{d:+.1f}{pct_str}"
            except Exception:
                return ""
        for oid in others:
            row_df = df[df["interval_id"] == oid]
            if row_df.empty:
                continue
            r = row_df.iloc[0]
            out = {"compare": r.get("interval_label", oid)}
            for key, label in metrics:
                out[label] = _fmt_delta(base.get(key), r.get(key))
            rows.append(out)
        if not rows:
            return html.Div("No comparable intervals found.")
        columns = [{"name": "Interval", "id": "compare"}] + [{"name": lab, "id": lab} for _, lab in metrics]
        table = dash_table.DataTable(
            columns=columns,
            data=rows,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "fontSize": "13px", "padding": "8px"},
            style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        )
        return html.Div([
            html.Div([html.Span("Baseline:", style={"fontWeight": "bold", "marginRight": 6}), html.Span(base.get("interval_label", base_id))], style={"marginBottom": 6}),
            table
        ])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple UI for Workout Foundation")
    parser.add_argument("--athletes-root", required=True, help="Root folder (athlete folder or multi-athlete root)")
    parser.add_argument("--output", help="Output root folder (defaults to config TEST_OUTPUT_DIR)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    app = build_app(args.athletes_root, args.output or "")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
