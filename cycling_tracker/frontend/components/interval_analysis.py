"""
Interval Analysis Component
=========================

This component provides detailed analysis of intervals detected from lap data,
including interval metrics, power curve analysis, and interval comparison.

Focuses on lap-based interval detection rather than power-based detection
like in the original SprintV1.py.
"""

from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class IntervalAnalysisComponent:
    """
    Component for detailed interval analysis and visualization.
    
    Features:
    - Lap-based interval detection and display
    - Individual interval analysis
    - Power curve generation
    - Interval comparison and ranking
    - Evolution tracking within intervals
    """
    
    def __init__(self):
        """Initialize the interval analysis component."""
        self.component_id = 'interval-analysis'
        logger.info("Interval Analysis Component initialized")
    
    def get_layout(self):
        """Return the layout for the interval analysis page."""
        return dbc.Container([
            
            # Page Header
            dbc.Row([
                dbc.Col([
                    html.H2("Interval Analysis", className="mb-3"),
                    html.P("Detailed analysis of intervals detected from lap data",
                           className="text-muted")
                ])
            ], className="mb-4"),
            
            # Interval Detection Settings
            dbc.Row([
                dbc.Col([
                    self._create_interval_settings()
                ], width=12)
            ], className="mb-4"),
            
            # Interval Summary Cards
            dbc.Row([
                dbc.Col([
                    self._create_interval_summary_cards()
                ], width=12)
            ], className="mb-4"),
            
            # Main Analysis Section
            dbc.Row([
                # Interval List
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Detected Intervals"),
                        dbc.CardBody([
                            html.Div(id='intervals-table')
                        ])
                    ])
                ], width=4),
                
                # Selected Interval Details
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Interval Details"),
                        dbc.CardBody([
                            html.Div(id='selected-interval-details')
                        ])
                    ])
                ], width=8)
            ], className="mb-4"),
            
            # Interval Visualizations
            dbc.Row([
                # Individual Interval Graph
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Selected Interval Analysis"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='interval-detail-graph',
                                style={'height': '400px'}
                            )
                        ])
                    ])
                ], width=6),
                
                # Power Curve
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Interval Power Curve"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='interval-power-curve',
                                style={'height': '400px'}
                            )
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Interval Comparison Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Interval Comparison"),
                        dbc.CardBody([
                            # Comparison Controls
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Group by:"),
                                    dcc.Dropdown(
                                        id='interval-grouping',
                                        options=[
                                            {'label': 'Duration Range', 'value': 'duration'},
                                            {'label': 'Power Zone', 'value': 'power_zone'},
                                            {'label': 'Lap Type', 'value': 'lap_type'}
                                        ],
                                        value='duration'
                                    )
                                ], width=4),
                                
                                dbc.Col([
                                    html.Label("Sort by:"),
                                    dcc.Dropdown(
                                        id='interval-sorting',
                                        options=[
                                            {'label': 'Average Power', 'value': 'avg_power'},
                                            {'label': 'Duration', 'value': 'duration'},
                                            {'label': 'Intensity Factor', 'value': 'intensity'},
                                            {'label': 'Start Time', 'value': 'start_time'}
                                        ],
                                        value='avg_power'
                                    )
                                ], width=4)
                            ], className="mb-3"),
                            
                            # Comparison Graph
                            dcc.Graph(
                                id='interval-comparison-graph',
                                style={'height': '500px'}
                            )
                        ])
                    ])
                ], width=12)
            ]),
            
            # Hidden store for selected interval
            dcc.Store(id='selected-interval-store')
            
        ], fluid=True)
    
    def _create_interval_settings(self):
        """Create settings panel for interval detection parameters."""
        return dbc.Card([
            dbc.CardBody([
                html.H6("Interval Detection Settings:", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Minimum Interval Duration (seconds):"),
                        dcc.Input(
                            id='min-interval-duration',
                            type='number',
                            value=60,
                            min=10,
                            max=3600,
                            step=10
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Label("Detection Method:"),
                        dcc.Dropdown(
                            id='detection-method',
                            options=[
                                {'label': 'Lap Data (Recommended)', 'value': 'lap_based'},
                                {'label': 'Power Threshold', 'value': 'power_based'},
                                {'label': 'Heart Rate Zones', 'value': 'hr_based'}
                            ],
                            value='lap_based'
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Label("Power Threshold (% of FTP):"),
                        dcc.Input(
                            id='power-threshold',
                            type='number',
                            value=80,
                            min=50,
                            max=150,
                            step=5,
                            disabled=True  # Enabled when power_based is selected
                        )
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Button(
                            "Re-detect Intervals",
                            id='redetect-button',
                            color='primary',
                            className='mt-4'
                        )
                    ], width=3)
                ])
            ])
        ], color="light")
    
    def _create_interval_summary_cards(self):
        """Create summary cards for interval statistics."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0", id="total-intervals", className="card-title"),
                        html.P("Total Intervals", className="card-text text-muted")
                    ])
                ], color="primary", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0:00", id="total-interval-time", className="card-title"),
                        html.P("Total Interval Time", className="card-text text-muted")
                    ])
                ], color="info", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0W", id="avg-interval-power", className="card-title"),
                        html.P("Avg Interval Power", className="card-text text-muted")
                    ])
                ], color="success", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0W", id="best-interval-power", className="card-title"),
                        html.P("Best Interval Power", className="card-text text-muted")
                    ])
                ], color="warning", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0:00", id="avg-interval-duration", className="card-title"),
                        html.P("Avg Duration", className="card-text text-muted")
                    ])
                ], color="secondary", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0.0", id="avg-intensity-factor", className="card-title"),
                        html.P("Avg Intensity Factor", className="card-text text-muted")
                    ])
                ], color="dark", outline=True)
            ], width=2)
        ])
    
    def register_callbacks(self, app):
        """Register callbacks for the interval analysis component."""
        
        @app.callback(
            [Output('total-intervals', 'children'),
             Output('total-interval-time', 'children'),
             Output('avg-interval-power', 'children'),
             Output('best-interval-power', 'children'),
             Output('avg-interval-duration', 'children'),
             Output('avg-intensity-factor', 'children')],
            Input('global-data-context', 'data')
        )
        def update_interval_summary(global_context):
            """Update interval summary cards."""
            if not global_context or not global_context.get('current_ride'):
                return "0", "0:00", "0W", "0W", "0:00", "0.0"
            
            # TODO: Calculate actual interval statistics
            return "8", "45:32", "287W", "342W", "5:41", "0.91"
        
        @app.callback(
            Output('intervals-table', 'children'),
            Input('global-data-context', 'data')
        )
        def update_intervals_table(global_context):
            """Update the intervals table."""
            if not global_context or not global_context.get('current_ride'):
                return html.P("No interval data available")
            
            return self._create_demo_intervals_table()
        
        @app.callback(
            Output('selected-interval-details', 'children'),
            Input('selected-interval-store', 'data')
        )
        def update_interval_details(selected_interval):
            """Update details for the selected interval."""
            if not selected_interval:
                return html.P("Select an interval to view details")
            
            return self._create_interval_details(selected_interval)
        
        @app.callback(
            Output('interval-detail-graph', 'figure'),
            Input('selected-interval-store', 'data')
        )
        def update_interval_graph(selected_interval):
            """Update the detailed interval graph."""
            if not selected_interval:
                return go.Figure()
            
            return self._create_demo_interval_graph(selected_interval)
        
        @app.callback(
            Output('interval-power-curve', 'figure'),
            Input('selected-interval-store', 'data')
        )
        def update_power_curve(selected_interval):
            """Update the interval power curve."""
            if not selected_interval:
                return go.Figure()
            
            return self._create_demo_power_curve(selected_interval)
        
        @app.callback(
            Output('interval-comparison-graph', 'figure'),
            [Input('global-data-context', 'data'),
             Input('interval-grouping', 'value'),
             Input('interval-sorting', 'value')]
        )
        def update_comparison_graph(global_context, grouping, sorting):
            """Update the interval comparison graph."""
            if not global_context or not global_context.get('current_ride'):
                return go.Figure()
            
            return self._create_demo_comparison_graph(grouping, sorting)
        
        # Callback for interval selection from table
        @app.callback(
            Output('selected-interval-store', 'data'),
            Input('intervals-table', 'children')
        )
        def handle_interval_selection(intervals_table):
            """Handle interval selection from the table."""
            # TODO: Implement actual interval selection logic
            # For now, return first interval as selected
            return {'interval_id': 0, 'start_time': '0:15:30', 'duration': '8:45'}
    
    def _create_demo_intervals_table(self):
        """Create a demo intervals table."""
        intervals_data = [
            ["1", "0:15:30", "8:45", "315W", "162 bpm", "0.95", "Work"],
            ["2", "0:26:15", "5:30", "285W", "158 bpm", "0.87", "Work"],
            ["3", "0:45:10", "12:20", "342W", "168 bpm", "1.05", "Work"],
            ["4", "1:02:45", "6:15", "298W", "161 bpm", "0.91", "Work"],
            ["5", "1:15:30", "4:45", "265W", "155 bpm", "0.81", "Work"],
            ["6", "1:32:15", "15:30", "378W", "173 bpm", "1.15", "Work"],
            ["7", "1:55:20", "7:40", "308W", "165 bpm", "0.94", "Work"],
            ["8", "2:05:45", "3:20", "245W", "148 bpm", "0.75", "Recovery"]
        ]
        
        table_header = html.Thead([
            html.Tr([
                html.Th("#"),
                html.Th("Start Time"),
                html.Th("Duration"),
                html.Th("Avg Power"),
                html.Th("Avg HR"),
                html.Th("IF"),
                html.Th("Type")
            ])
        ])
        
        table_rows = []
        for i, row in enumerate(intervals_data):
            table_rows.append(
                html.Tr([html.Td(cell) for cell in row],
                       id=f'interval-row-{i}',
                       style={'cursor': 'pointer'},
                       className='table-row-hover')
            )
        
        table_body = html.Tbody(table_rows)
        
        return dbc.Table(
            [table_header, table_body],
            bordered=True,
            striped=True,
            hover=True,
            responsive=True,
            size="sm"
        )
    
    def _create_interval_details(self, selected_interval):
        """Create detailed view of selected interval."""
        return dbc.Row([
            dbc.Col([
                html.H6(f"Interval {selected_interval.get('interval_id', 0) + 1}"),
                html.P(f"Start: {selected_interval.get('start_time', 'N/A')}"),
                html.P(f"Duration: {selected_interval.get('duration', 'N/A')}")
            ], width=4),
            dbc.Col([
                html.P("Average Power: 315W"),
                html.P("Max Power: 425W"),
                html.P("Power Fade: 8.5%")
            ], width=4),
            dbc.Col([
                html.P("Average HR: 162 bpm"),
                html.P("Max HR: 171 bpm"), 
                html.P("Intensity Factor: 0.95")
            ], width=4)
        ])
    
    def _create_demo_interval_graph(self, selected_interval):
        """Create a demo graph for the selected interval."""
        time_seconds = np.linspace(0, 525, 525)  # 8:45 interval
        
        # Generate realistic interval data with some power fade
        base_power = 315
        power_fade_factor = 0.085  # 8.5% fade
        power_data = base_power * (1 - power_fade_factor * time_seconds / max(time_seconds))
        power_data += np.random.normal(0, 15, len(time_seconds))
        
        hr_data = 162 + 5 * np.sin(time_seconds/60) + np.random.normal(0, 3, len(time_seconds))
        cadence_data = 88 + 3 * np.sin(time_seconds/30) + np.random.normal(0, 2, len(time_seconds))
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Power', 'Heart Rate', 'Cadence'),
            vertical_spacing=0.08
        )
        
        fig.add_trace(
            go.Scatter(x=time_seconds, y=power_data, name='Power (W)', 
                      line=dict(color='#1f77b4')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_seconds, y=hr_data, name='Heart Rate (bpm)',
                      line=dict(color='#d62728')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_seconds, y=cadence_data, name='Cadence (rpm)',
                      line=dict(color='#2ca02c')),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Selected Interval Analysis',
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
        
        return fig
    
    def _create_demo_power_curve(self, selected_interval):
        """Create a demo power curve for the interval."""
        durations = [5, 10, 20, 30, 60, 120, 300, 600]  # seconds
        powers = [425, 410, 395, 380, 350, 330, 320, 315]  # watts
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=durations, y=powers, mode='lines+markers',
                      name='Power Curve', line=dict(color='#ff7f0e', width=3))
        )
        
        fig.update_layout(
            title='Interval Power Curve',
            xaxis_title='Duration (seconds)',
            yaxis_title='Power (W)',
            xaxis_type='log'
        )
        
        return fig
    
    def _create_demo_comparison_graph(self, grouping, sorting):
        """Create a demo interval comparison graph."""
        intervals = [f'Interval {i+1}' for i in range(8)]
        powers = [315, 285, 342, 298, 265, 378, 308, 245]
        durations = [525, 330, 740, 375, 285, 930, 460, 200]  # seconds
        
        fig = go.Figure()
        
        if sorting == 'avg_power':
            # Sort by power
            sorted_data = sorted(zip(intervals, powers, durations), key=lambda x: x[1], reverse=True)
        else:
            # Default order
            sorted_data = list(zip(intervals, powers, durations))
        
        intervals_sorted, powers_sorted, durations_sorted = zip(*sorted_data)
        
        fig.add_trace(
            go.Bar(x=intervals_sorted, y=powers_sorted, name='Average Power (W)',
                  marker_color='#2E86AB')
        )
        
        fig.update_layout(
            title=f'Interval Comparison - Grouped by {grouping.replace("_", " ").title()}',
            xaxis_title='Intervals',
            yaxis_title='Average Power (W)',
            height=500
        )
        
        return fig