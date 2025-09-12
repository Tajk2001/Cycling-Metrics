"""
Ride Overview Component
=====================

This component provides comprehensive visualization and analysis of individual rides,
including ride-level metrics, time-series data visualization, and summary statistics.

Based on SprintV1.py analysis but redesigned for complete ride analysis.
"""

from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RideOverviewComponent:
    """
    Component for displaying comprehensive ride overview and metrics.
    
    Features:
    - Ride summary statistics
    - Multi-metric time series visualization  
    - Interactive metric toggles
    - Ride segment analysis
    """
    
    def __init__(self):
        """Initialize the ride overview component."""
        self.component_id = 'ride-overview'
        logger.info("Ride Overview Component initialized")
    
    def get_layout(self):
        """Return the layout for the ride overview page."""
        return dbc.Container([
            
            # Page Header
            dbc.Row([
                dbc.Col([
                    html.H2("Ride Overview", className="mb-3"),
                    html.P("Comprehensive analysis of your cycling performance",
                           className="text-muted")
                ])
            ], className="mb-4"),
            
            # Ride Summary Cards
            dbc.Row([
                dbc.Col([
                    self._create_summary_cards()
                ], width=12)
            ], className="mb-4"),
            
            # Metric Controls
            dbc.Row([
                dbc.Col([
                    self._create_metric_controls()
                ], width=12)
            ], className="mb-3"),
            
            # Main Time Series Visualization
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Workout Overview - Metrics Over Time"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='ride-overview-graph',
                                style={'height': '600px'}
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Additional Analysis Sections
            dbc.Row([
                # Power Analysis
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Power Analysis"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='power-distribution-graph',
                                style={'height': '300px'}
                            )
                        ])
                    ])
                ], width=6),
                
                # Cadence Analysis  
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Cadence Analysis"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='cadence-distribution-graph', 
                                style={'height': '300px'}
                            )
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Ride Segments Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Ride Segments"),
                        dbc.CardBody([
                            html.Div(id='ride-segments-table')
                        ])
                    ])
                ], width=12)
            ])
            
        ], fluid=True)
    
    def _create_summary_cards(self):
        """Create summary metric cards for the ride."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0.0", id="total-distance", className="card-title"),
                        html.P("Total Distance (km)", className="card-text text-muted")
                    ])
                ], color="primary", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0:00:00", id="total-time", className="card-title"),
                        html.P("Total Time", className="card-text text-muted")
                    ])
                ], color="info", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0W", id="avg-power", className="card-title"),
                        html.P("Average Power", className="card-text text-muted")
                    ])
                ], color="success", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0W", id="max-power", className="card-title"),
                        html.P("Max Power", className="card-text text-muted")
                    ])
                ], color="warning", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0m", id="elevation-gain", className="card-title"),
                        html.P("Elevation Gain", className="card-text text-muted")
                    ])
                ], color="secondary", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0.0", id="intensity-factor", className="card-title"),
                        html.P("Intensity Factor", className="card-text text-muted")
                    ])
                ], color="dark", outline=True)
            ], width=2)
        ])
    
    def _create_metric_controls(self):
        """Create controls for selecting which metrics to display."""
        return dbc.Card([
            dbc.CardBody([
                html.H6("Select Metrics to Display:", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Checklist(
                            id="metric-toggles",
                            options=[
                                {"label": "Power", "value": "power"},
                                {"label": "Heart Rate", "value": "heart_rate"},
                                {"label": "Cadence", "value": "cadence"},
                                {"label": "Speed", "value": "speed"},
                            ],
                            value=["power", "heart_rate"],
                            inline=True,
                            style={'margin-right': '20px'}
                        )
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Checklist(
                            id="secondary-metric-toggles",
                            options=[
                                {"label": "Torque", "value": "torque"},
                                {"label": "Altitude", "value": "altitude"},
                                {"label": "Grade", "value": "grade"},
                                {"label": "Temperature", "value": "temperature"}
                            ],
                            value=["altitude"],
                            inline=True
                        )
                    ], width=6)
                ])
            ])
        ], color="light")
    
    def register_callbacks(self, app):
        """Register callbacks for the ride overview component."""
        
        @app.callback(
            [Output('total-distance', 'children'),
             Output('total-time', 'children'),
             Output('avg-power', 'children'),
             Output('max-power', 'children'),
             Output('elevation-gain', 'children'),
             Output('intensity-factor', 'children')],
            Input('global-data-context', 'data')
        )
        def update_summary_cards(global_context):
            """Update the summary metric cards based on current ride data."""
            if not global_context or not global_context.get('current_ride'):
                return "0.0", "0:00:00", "0W", "0W", "0m", "0.0"
            
            # TODO: Calculate actual metrics from ride data
            # For now, return placeholder values
            return "45.2", "2:15:33", "245W", "892W", "1250m", "0.82"
        
        @app.callback(
            Output('ride-overview-graph', 'figure'),
            [Input('global-data-context', 'data'),
             Input('metric-toggles', 'value'),
             Input('secondary-metric-toggles', 'value')]
        )
        def update_main_graph(global_context, primary_metrics, secondary_metrics):
            """Update the main time-series graph based on selected metrics."""
            if not global_context or not global_context.get('current_ride'):
                return self._create_empty_graph()
            
            # TODO: Use actual ride data
            # For now, create demo data
            return self._create_demo_graph(primary_metrics + secondary_metrics)
        
        @app.callback(
            Output('power-distribution-graph', 'figure'),
            Input('global-data-context', 'data')
        )
        def update_power_distribution(global_context):
            """Update the power distribution histogram."""
            if not global_context or not global_context.get('current_ride'):
                return go.Figure()
            
            # TODO: Create actual power distribution from ride data
            return self._create_demo_power_distribution()
        
        @app.callback(
            Output('cadence-distribution-graph', 'figure'),
            Input('global-data-context', 'data')
        )
        def update_cadence_distribution(global_context):
            """Update the cadence distribution histogram."""
            if not global_context or not global_context.get('current_ride'):
                return go.Figure()
            
            # TODO: Create actual cadence distribution from ride data
            return self._create_demo_cadence_distribution()
        
        @app.callback(
            Output('ride-segments-table', 'children'),
            Input('global-data-context', 'data')
        )
        def update_segments_table(global_context):
            """Update the ride segments table."""
            if not global_context or not global_context.get('current_ride'):
                return html.P("No ride data available")
            
            # TODO: Create actual segments table from ride data
            return self._create_demo_segments_table()
    
    def _create_empty_graph(self):
        """Create an empty placeholder graph."""
        fig = go.Figure()
        fig.add_annotation(
            text="Upload a FIT file to view ride data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16
        )
        fig.update_layout(
            showlegend=False,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    
    def _create_demo_graph(self, selected_metrics):
        """Create a demo graph with sample data."""
        # Generate sample time series data
        time_minutes = np.linspace(0, 135, 1000)  # 2h 15min ride
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Primary Metrics', 'Secondary Metrics'),
            vertical_spacing=0.1
        )
        
        # Primary metrics
        if 'power' in selected_metrics:
            power_data = 200 + 50 * np.sin(time_minutes/10) + np.random.normal(0, 20, len(time_minutes))
            fig.add_trace(
                go.Scatter(x=time_minutes, y=power_data, name='Power (W)', 
                          line=dict(color='#1f77b4')), 
                row=1, col=1
            )
        
        if 'heart_rate' in selected_metrics:
            hr_data = 150 + 20 * np.sin(time_minutes/15) + np.random.normal(0, 5, len(time_minutes))
            fig.add_trace(
                go.Scatter(x=time_minutes, y=hr_data, name='Heart Rate (bpm)',
                          line=dict(color='#d62728'), yaxis='y2'),
                row=1, col=1
            )
        
        # Secondary metrics
        if 'altitude' in selected_metrics:
            altitude_data = 100 + 50 * np.sin(time_minutes/20) + np.random.normal(0, 10, len(time_minutes))
            fig.add_trace(
                go.Scatter(x=time_minutes, y=altitude_data, name='Altitude (m)',
                          line=dict(color='#8c564b')),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Ride Metrics Over Time',
            xaxis_title='Time (minutes)',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_demo_power_distribution(self):
        """Create a demo power distribution histogram."""
        power_data = np.random.normal(250, 60, 1000)
        power_data = power_data[power_data > 0]  # Remove negative values
        
        fig = go.Figure(data=[go.Histogram(x=power_data, nbinsx=30)])
        fig.update_layout(
            title='Power Distribution',
            xaxis_title='Power (W)',
            yaxis_title='Frequency'
        )
        return fig
    
    def _create_demo_cadence_distribution(self):
        """Create a demo cadence distribution histogram."""
        cadence_data = np.random.normal(85, 10, 1000)
        cadence_data = cadence_data[(cadence_data > 0) & (cadence_data < 120)]
        
        fig = go.Figure(data=[go.Histogram(x=cadence_data, nbinsx=25)])
        fig.update_layout(
            title='Cadence Distribution',
            xaxis_title='Cadence (rpm)',
            yaxis_title='Frequency'
        )
        return fig
    
    def _create_demo_segments_table(self):
        """Create a demo ride segments table."""
        segments_data = [
            ["Warm-up", "0:00:00", "0:15:00", "3.2", "180", "128", "12.8"],
            ["Main Set", "0:15:00", "1:45:00", "28.5", "268", "157", "18.4"],
            ["Recovery", "1:45:00", "2:00:00", "4.8", "165", "142", "19.2"],
            ["Cool-down", "2:00:00", "2:15:33", "8.7", "145", "125", "20.1"]
        ]
        
        table_header = html.Thead([
            html.Tr([
                html.Th("Segment"),
                html.Th("Start Time"),
                html.Th("End Time"), 
                html.Th("Distance (km)"),
                html.Th("Avg Power (W)"),
                html.Th("Avg HR (bpm)"),
                html.Th("Avg Speed (km/h)")
            ])
        ])
        
        table_rows = []
        for row in segments_data:
            table_rows.append(
                html.Tr([html.Td(cell) for cell in row])
            )
        
        table_body = html.Tbody(table_rows)
        
        return dbc.Table(
            [table_header, table_body],
            bordered=True,
            striped=True,
            hover=True,
            responsive=True
        )