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
from workout_foundation.config import TEST_OUTPUT_DIR
from workout_foundation.models.athlete_profile import AthleteProfile, save_athlete_profile


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
        html.H3("Workout Foundation – Simple UI"),
        
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
        html.Div(id="analysis-content"),
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
        Output("analysis-content", "children"),
        Input("viz-athlete", "value"),
        Input("viz-category", "value"),
        State("files-store", "data"),
    )
    def update_analysis(selected_athlete, selected_category, files_data):
        if not selected_athlete:
            return "Select an athlete to view analysis", ""
        
        if not selected_category:
            return f"Select a category for {selected_athlete}", ""
        
        # Count files for this athlete/category combination
        if files_data:
            df = pd.DataFrame(files_data)
            athlete_cat_files = df[(df["athlete"] == selected_athlete) & (df["category"] == selected_category)]
            file_count = len(athlete_cat_files) if not athlete_cat_files.empty else 0
        else:
            file_count = 0
        
        summary = f"Analysis: {selected_athlete} - {selected_category} ({file_count} files)"
        
        content = html.Div([
            html.P(f"Athlete: {selected_athlete}"),
            html.P(f"Category: {selected_category}"),
            html.P(f"Files found: {file_count}"),
            html.P("Run analysis to generate metrics and view results in the consolidated CSV files."),
        ], style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"})
        
        return summary, content

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
