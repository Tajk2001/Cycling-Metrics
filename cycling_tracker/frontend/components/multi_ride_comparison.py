"""
Multi-Ride Comparison Component
=============================

This component enables comparison of metrics across multiple rides,
allowing users to track performance changes and identify patterns over time.
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


class MultiRideComparisonComponent:
    """
    Component for comparing metrics across multiple rides.
    
    Features:
    - Side-by-side ride comparison
    - Metric trend analysis across rides
    - Statistical comparison (mean, max, improvement)
    - Ride filtering and selection
    """
    
    def __init__(self):
        """Initialize the multi-ride comparison component."""
        self.component_id = 'multi-ride-comparison'
        logger.info("Multi-Ride Comparison Component initialized")
    
    def get_layout(self):
        """Return the layout for the multi-ride comparison page."""
        return dbc.Container([
            
            # Page Header
            dbc.Row([
                dbc.Col([
                    html.H2("Multi-Ride Comparison", className="mb-3"),
                    html.P("Compare performance across multiple rides and track improvements",
                           className="text-muted")
                ])
            ], className="mb-4"),
            
            # Ride Selection and Filters
            dbc.Row([
                dbc.Col([
                    self._create_ride_filters()
                ], width=12)
            ], className="mb-4"),
            
            # Comparison Summary Cards
            dbc.Row([
                dbc.Col([
                    self._create_comparison_summary()
                ], width=12)
            ], className="mb-4"),
            
            # Main Comparison Visualizations
            dbc.Row([
                # Ride Comparison Chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Ride Metrics Comparison"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='ride-comparison-chart',
                                style={'height': '500px'}
                            )
                        ])
                    ])
                ], width=8),
                
                # Selected Rides List
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Selected Rides"),
                        dbc.CardBody([
                            html.Div(id='selected-rides-list')
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Detailed Comparison Sections
            dbc.Row([
                # Power Comparison
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Power Analysis Comparison"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='power-comparison-chart',
                                style={'height': '400px'}
                            )
                        ])
                    ])
                ], width=6),
                
                # Interval Comparison
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Interval Performance Comparison"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='interval-comparison-chart',
                                style={'height': '400px'}
                            )
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Improvement Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Improvement Analysis"),
                        dbc.CardBody([
                            html.Div(id='improvement-analysis')
                        ])
                    ])
                ], width=12)
            ]),
            
            # Hidden stores
            dcc.Store(id='selected-rides-store', data=[]),
            dcc.Store(id='available-rides-store', data=[])
            
        ], fluid=True)
    
    def _create_ride_filters(self):
        """Create ride selection and filtering controls."""
        return dbc.Card([
            dbc.CardBody([
                html.H6("Ride Selection & Filters:", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='date-range-picker',
                            start_date=datetime.now() - timedelta(days=90),
                            end_date=datetime.now(),
                            display_format='YYYY-MM-DD'
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Label("Ride Type:"),
                        dcc.Dropdown(
                            id='ride-type-filter',
                            options=[
                                {'label': 'All Rides', 'value': 'all'},
                                {'label': 'Training Rides', 'value': 'training'},
                                {'label': 'Race/Event', 'value': 'race'},
                                {'label': 'Recovery', 'value': 'recovery'}
                            ],
                            value='all',
                            multi=True
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Label("Minimum Duration (min):"),
                        dcc.Input(
                            id='min-duration-filter',
                            type='number',
                            value=30,
                            min=0,
                            max=480
                        )
                    ], width=2),
                    
                    dbc.Col([
                        html.Label("Maximum Rides:"),
                        dcc.Input(
                            id='max-rides-filter',
                            type='number',
                            value=10,
                            min=2,
                            max=50
                        )
                    ], width=2),
                    
                    dbc.Col([
                        dbc.Button(
                            "Update Comparison",
                            id='update-comparison-button',
                            color='primary',
                            className='mt-4'
                        )
                    ], width=2)
                ])
            ])
        ], color="light")
    
    def _create_comparison_summary(self):
        """Create summary cards for ride comparison metrics."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0", id="total-compared-rides", className="card-title"),
                        html.P("Rides Compared", className="card-text text-muted")
                    ])
                ], color="primary", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("+0%", id="avg-power-improvement", className="card-title"),
                        html.P("Avg Power Improvement", className="card-text text-muted")
                    ])
                ], color="success", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0", id="total-distance-compared", className="card-title"),
                        html.P("Total Distance (km)", className="card-text text-muted")
                    ])
                ], color="info", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0:00", id="total-time-compared", className="card-title"),
                        html.P("Total Time", className="card-text text-muted")
                    ])
                ], color="warning", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("+0%", id="endurance-improvement", className="card-title"),
                        html.P("Endurance Improvement", className="card-text text-muted")
                    ])
                ], color="secondary", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("0.0", id="consistency-score", className="card-title"),
                        html.P("Consistency Score", className="card-text text-muted")
                    ])
                ], color="dark", outline=True)
            ], width=2)
        ])
    
    def register_callbacks(self, app):
        """Register callbacks for the multi-ride comparison component."""
        
        @app.callback(
            Output('available-rides-store', 'data'),
            [Input('date-range-picker', 'start_date'),
             Input('date-range-picker', 'end_date'),
             Input('ride-type-filter', 'value'),
             Input('min-duration-filter', 'value')]
        )
        def update_available_rides(start_date, end_date, ride_types, min_duration):
            """Update available rides based on filters."""
            # TODO: Query actual ride database
            # For now, return demo data
            demo_rides = [
                {'id': 1, 'date': '2025-01-10', 'name': 'Morning Training', 'duration': 95, 'type': 'training'},
                {'id': 2, 'date': '2025-01-08', 'name': 'Hill Intervals', 'duration': 120, 'type': 'training'},
                {'id': 3, 'date': '2025-01-05', 'name': 'Recovery Ride', 'duration': 60, 'type': 'recovery'},
                {'id': 4, 'date': '2025-01-03', 'name': 'Time Trial', 'duration': 45, 'type': 'race'},
                {'id': 5, 'date': '2025-01-01', 'name': 'Base Training', 'duration': 135, 'type': 'training'}
            ]
            return demo_rides
        
        @app.callback(
            [Output('total-compared-rides', 'children'),
             Output('avg-power-improvement', 'children'),
             Output('total-distance-compared', 'children'),
             Output('total-time-compared', 'children'),
             Output('endurance-improvement', 'children'),
             Output('consistency-score', 'children')],
            Input('selected-rides-store', 'data')
        )
        def update_comparison_summary(selected_rides):
            """Update comparison summary metrics."""
            if not selected_rides:
                return "0", "+0%", "0", "0:00", "+0%", "0.0"
            
            # TODO: Calculate actual metrics from selected rides
            return str(len(selected_rides)), "+12.5%", "425", "8:45:30", "+8.2%", "7.8"
        
        @app.callback(
            Output('selected-rides-list', 'children'),
            Input('available-rides-store', 'data')
        )
        def update_rides_list(available_rides):
            """Update the list of selectable rides."""
            if not available_rides:
                return html.P("No rides found for selected criteria")
            
            return self._create_rides_list(available_rides)
        
        @app.callback(
            Output('ride-comparison-chart', 'figure'),
            Input('selected-rides-store', 'data')
        )
        def update_comparison_chart(selected_rides):
            """Update the main ride comparison chart."""
            if not selected_rides or len(selected_rides) < 2:
                return self._create_empty_comparison_chart()
            
            return self._create_demo_comparison_chart(selected_rides)
        
        @app.callback(
            Output('power-comparison-chart', 'figure'),
            Input('selected-rides-store', 'data')
        )
        def update_power_comparison(selected_rides):
            """Update the power comparison chart."""
            if not selected_rides or len(selected_rides) < 2:
                return go.Figure()
            
            return self._create_demo_power_comparison(selected_rides)
        
        @app.callback(
            Output('interval-comparison-chart', 'figure'),
            Input('selected-rides-store', 'data')
        )
        def update_interval_comparison(selected_rides):
            """Update the interval comparison chart."""
            if not selected_rides or len(selected_rides) < 2:
                return go.Figure()
            
            return self._create_demo_interval_comparison(selected_rides)
        
        @app.callback(
            Output('improvement-analysis', 'children'),
            Input('selected-rides-store', 'data')
        )
        def update_improvement_analysis(selected_rides):
            """Update the improvement analysis section."""
            if not selected_rides or len(selected_rides) < 2:
                return html.P("Select at least 2 rides to see improvement analysis")
            
            return self._create_improvement_analysis(selected_rides)
    
    def _create_rides_list(self, available_rides):
        """Create a list of selectable rides."""
        rides_items = []
        for ride in available_rides[:10]:  # Show max 10 rides
            rides_items.append(
                dbc.ListGroupItem([
                    dbc.Row([
                        dbc.Col([
                            html.H6(ride['name'], className="mb-1"),
                            html.P(f"{ride['date']} • {ride['duration']} min", 
                                  className="mb-1 text-muted small")
                        ], width=8),
                        dbc.Col([
                            dbc.Checkbox(
                                id=f"ride-select-{ride['id']}",
                                label="",
                                value=False
                            )
                        ], width=2)
                    ])
                ], action=True, id=f"ride-item-{ride['id']}")
            )
        
        return dbc.ListGroup(rides_items)
    
    def _create_empty_comparison_chart(self):
        """Create an empty comparison chart with instructions."""
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least 2 rides to compare<br>Use the checkboxes on the right",
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
    
    def _create_demo_comparison_chart(self, selected_rides):
        """Create a demo comparison chart."""
        # Demo data for selected rides
        rides = ['Ride 1', 'Ride 2', 'Ride 3', 'Ride 4']
        avg_power = [245, 268, 252, 275]
        max_power = [892, 945, 878, 932]
        distance = [42.5, 38.2, 45.8, 51.2]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Power', 'Max Power', 'Distance', 'Duration'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average Power
        fig.add_trace(
            go.Bar(x=rides, y=avg_power, name='Avg Power (W)', marker_color='#1f77b4'),
            row=1, col=1
        )
        
        # Max Power
        fig.add_trace(
            go.Bar(x=rides, y=max_power, name='Max Power (W)', marker_color='#ff7f0e'),
            row=1, col=2
        )
        
        # Distance
        fig.add_trace(
            go.Bar(x=rides, y=distance, name='Distance (km)', marker_color='#2ca02c'),
            row=2, col=1
        )
        
        # Duration (demo data)
        duration = [95, 120, 85, 140]
        fig.add_trace(
            go.Bar(x=rides, y=duration, name='Duration (min)', marker_color='#d62728'),
            row=2, col=2
        )
        
        fig.update_layout(height=500, showlegend=False, title_text="Ride Metrics Comparison")
        return fig
    
    def _create_demo_power_comparison(self, selected_rides):
        """Create a demo power comparison chart."""
        fig = go.Figure()
        
        # Demo power distributions for different rides
        for i, ride in enumerate(['Ride 1', 'Ride 2', 'Ride 3'][:len(selected_rides)]):
            power_data = np.random.normal(250 + i*15, 45, 500)
            power_data = power_data[power_data > 0]
            
            fig.add_trace(go.Histogram(
                x=power_data,
                name=ride,
                opacity=0.7,
                nbinsx=30
            ))
        
        fig.update_layout(
            title='Power Distribution Comparison',
            xaxis_title='Power (W)',
            yaxis_title='Frequency',
            barmode='overlay'
        )
        
        return fig
    
    def _create_demo_interval_comparison(self, selected_rides):
        """Create a demo interval comparison chart."""
        rides = ['Ride 1', 'Ride 2', 'Ride 3'][:len(selected_rides)]
        interval_types = ['Short (1-3 min)', 'Medium (3-8 min)', 'Long (8+ min)']
        
        fig = go.Figure()
        
        # Demo data for different interval types across rides
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, ride in enumerate(rides):
            powers = [320 + i*10, 295 + i*8, 275 + i*12]
            fig.add_trace(go.Bar(
                x=interval_types,
                y=powers,
                name=ride,
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            title='Interval Power Comparison by Duration',
            xaxis_title='Interval Type',
            yaxis_title='Average Power (W)',
            barmode='group'
        )
        
        return fig
    
    def _create_improvement_analysis(self, selected_rides):
        """Create improvement analysis content."""
        return dbc.Row([
            dbc.Col([
                html.H6("Power Improvements", className="mb-2"),
                html.P("Average Power: +12.5% improvement"),
                html.P("Peak Power: +8.3% improvement"),
                html.P("Power Consistency: +5.2% improvement")
            ], width=4),
            
            dbc.Col([
                html.H6("Endurance Metrics", className="mb-2"),
                html.P("Ride Duration: +15.8% increase"),
                html.P("Total Distance: +18.2% increase"),
                html.P("Average Speed: +6.4% improvement")
            ], width=4),
            
            dbc.Col([
                html.H6("Training Load", className="mb-2"),
                html.P("TSS Progression: +22.1%"),
                html.P("Intensity Factor: +0.05 improvement"),
                html.P("Work Capacity: +19.7% increase")
            ], width=4)
        ])