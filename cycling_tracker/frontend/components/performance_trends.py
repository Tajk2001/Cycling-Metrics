"""
Performance Trends Component
==========================

This component provides long-term performance trend analysis,
tracking improvements and patterns over time across multiple rides.
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


class PerformanceTrendsComponent:
    """
    Component for analyzing performance trends over time.
    
    Features:
    - Long-term performance tracking
    - Trend line analysis
    - Seasonal performance patterns
    - Goal tracking and progress monitoring
    """
    
    def __init__(self):
        """Initialize the performance trends component."""
        self.component_id = 'performance-trends'
        logger.info("Performance Trends Component initialized")
    
    def get_layout(self):
        """Return the layout for the performance trends page."""
        return dbc.Container([
            
            # Page Header
            dbc.Row([
                dbc.Col([
                    html.H2("Performance Trends", className="mb-3"),
                    html.P("Track your cycling performance improvements over time",
                           className="text-muted")
                ])
            ], className="mb-4"),
            
            # Trend Analysis Controls
            dbc.Row([
                dbc.Col([
                    self._create_trend_controls()
                ], width=12)
            ], className="mb-4"),
            
            # Performance Summary Cards
            dbc.Row([
                dbc.Col([
                    self._create_trend_summary()
                ], width=12)
            ], className="mb-4"),
            
            # Main Trend Visualizations
            dbc.Row([
                # Power Trends
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Power Performance Trends"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='power-trends-chart',
                                style={'height': '400px'}
                            )
                        ])
                    ])
                ], width=6),
                
                # Volume Trends
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Volume Trends"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='volume-trends-chart',
                                style={'height': '400px'}
                            )
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Detailed Analysis Sections
            dbc.Row([
                # Seasonal Analysis
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Seasonal Performance Analysis"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='seasonal-analysis-chart',
                                style={'height': '400px'}
                            )
                        ])
                    ])
                ], width=8),
                
                # Goals Progress
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Goals Progress"),
                        dbc.CardBody([
                            html.Div(id='goals-progress')
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Performance Metrics Over Time
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Comprehensive Performance Timeline"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='comprehensive-timeline-chart',
                                style={'height': '600px'}
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Analysis Insights
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Insights & Recommendations"),
                        dbc.CardBody([
                            html.Div(id='performance-insights')
                        ])
                    ])
                ], width=12)
            ])
            
        ], fluid=True)
    
    def _create_trend_controls(self):
        """Create controls for trend analysis parameters."""
        return dbc.Card([
            dbc.CardBody([
                html.H6("Trend Analysis Settings:", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Time Period:"),
                        dcc.Dropdown(
                            id='time-period-selector',
                            options=[
                                {'label': 'Last 3 Months', 'value': '3m'},
                                {'label': 'Last 6 Months', 'value': '6m'},
                                {'label': 'Last Year', 'value': '1y'},
                                {'label': 'Last 2 Years', 'value': '2y'},
                                {'label': 'All Time', 'value': 'all'}
                            ],
                            value='6m'
                        )
                    ], width=2),
                    
                    dbc.Col([
                        html.Label("Aggregation:"),
                        dcc.Dropdown(
                            id='aggregation-selector',
                            options=[
                                {'label': 'Daily', 'value': 'daily'},
                                {'label': 'Weekly', 'value': 'weekly'},
                                {'label': 'Monthly', 'value': 'monthly'}
                            ],
                            value='weekly'
                        )
                    ], width=2),
                    
                    dbc.Col([
                        html.Label("Smoothing:"),
                        dcc.Dropdown(
                            id='smoothing-selector',
                            options=[
                                {'label': 'None', 'value': 'none'},
                                {'label': '7-day Average', 'value': '7day'},
                                {'label': '14-day Average', 'value': '14day'},
                                {'label': '30-day Average', 'value': '30day'}
                            ],
                            value='14day'
                        )
                    ], width=2),
                    
                    dbc.Col([
                        html.Label("Primary Metric:"),
                        dcc.Dropdown(
                            id='primary-metric-selector',
                            options=[
                                {'label': 'Average Power', 'value': 'avg_power'},
                                {'label': 'FTP', 'value': 'ftp'},
                                {'label': 'Training Load', 'value': 'tss'},
                                {'label': 'Distance', 'value': 'distance'}
                            ],
                            value='avg_power'
                        )
                    ], width=2),
                    
                    dbc.Col([
                        dbc.Checklist(
                            id='trend-options',
                            options=[
                                {'label': 'Show Trend Line', 'value': 'trendline'},
                                {'label': 'Show Goals', 'value': 'goals'}
                            ],
                            value=['trendline', 'goals'],
                            inline=True
                        )
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Button(
                            "Update Analysis",
                            id='update-trends-button',
                            color='primary',
                            className='mt-4'
                        )
                    ], width=1)
                ])
            ])
        ], color="light")
    
    def _create_trend_summary(self):
        """Create summary cards for trend metrics."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("+15.2%", id="overall-improvement", className="card-title text-success"),
                        html.P("Overall Improvement", className="card-text text-muted")
                    ])
                ], color="success", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("287W", id="current-ftp", className="card-title"),
                        html.P("Current FTP", className="card-text text-muted")
                    ])
                ], color="primary", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("+12W", id="ftp-gain", className="card-title text-success"),
                        html.P("FTP Gain (6 months)", className="card-text text-muted")
                    ])
                ], color="info", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("85%", id="consistency-rate", className="card-title"),
                        html.P("Training Consistency", className="card-text text-muted")
                    ])
                ], color="warning", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("125", id="total-rides", className="card-title"),
                        html.P("Total Rides", className="card-text text-muted")
                    ])
                ], color="secondary", outline=True)
            ], width=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("92%", id="goal-progress", className="card-title text-success"),
                        html.P("Goal Progress", className="card-text text-muted")
                    ])
                ], color="dark", outline=True)
            ], width=2)
        ])
    
    def register_callbacks(self, app):
        """Register callbacks for the performance trends component."""
        
        @app.callback(
            Output('power-trends-chart', 'figure'),
            [Input('time-period-selector', 'value'),
             Input('aggregation-selector', 'value'),
             Input('smoothing-selector', 'value'),
             Input('trend-options', 'value')]
        )
        def update_power_trends(time_period, aggregation, smoothing, options):
            """Update the power trends chart."""
            return self._create_demo_power_trends(time_period, smoothing, 'trendline' in options)
        
        @app.callback(
            Output('volume-trends-chart', 'figure'),
            [Input('time-period-selector', 'value'),
             Input('aggregation-selector', 'value'),
             Input('smoothing-selector', 'value')]
        )
        def update_volume_trends(time_period, aggregation, smoothing):
            """Update the training volume trends chart."""
            return self._create_demo_volume_trends(time_period)
        
        @app.callback(
            Output('seasonal-analysis-chart', 'figure'),
            Input('time-period-selector', 'value')
        )
        def update_seasonal_analysis(time_period):
            """Update the seasonal analysis chart."""
            return self._create_demo_seasonal_analysis()
        
        @app.callback(
            Output('comprehensive-timeline-chart', 'figure'),
            [Input('primary-metric-selector', 'value'),
             Input('time-period-selector', 'value')]
        )
        def update_comprehensive_timeline(primary_metric, time_period):
            """Update the comprehensive timeline chart."""
            return self._create_demo_comprehensive_timeline(primary_metric)
        
        @app.callback(
            Output('goals-progress', 'children'),
            Input('time-period-selector', 'value')
        )
        def update_goals_progress(time_period):
            """Update the goals progress section."""
            return self._create_goals_progress()
        
        @app.callback(
            Output('performance-insights', 'children'),
            Input('time-period-selector', 'value')
        )
        def update_performance_insights(time_period):
            """Update the performance insights section."""
            return self._create_performance_insights()
    
    def _create_demo_power_trends(self, time_period, smoothing, show_trendline):
        """Create a demo power trends chart."""
        # Generate sample data based on time period
        if time_period == '3m':
            days = 90
        elif time_period == '6m':
            days = 180
        elif time_period == '1y':
            days = 365
        else:
            days = 180  # Default to 6 months
        
        dates = pd.date_range(end=datetime.now(), periods=days//7, freq='W')  # Weekly data
        
        # Generate realistic power progression with some variability
        base_power = 250
        improvement_rate = 0.15 / (days/365)  # 15% improvement per year
        power_data = []
        
        for i, date in enumerate(dates):
            # Base improvement trend
            trend_power = base_power * (1 + improvement_rate * i/len(dates))
            # Add seasonal variation (winter dip, summer peak)
            seasonal_factor = 0.9 + 0.1 * np.sin(2 * np.pi * (date.dayofyear / 365.25))
            # Add random variation
            noise = np.random.normal(0, 15)
            final_power = trend_power * seasonal_factor + noise
            power_data.append(max(final_power, 180))  # Minimum reasonable power
        
        fig = go.Figure()
        
        # Add main power data
        fig.add_trace(go.Scatter(
            x=dates,
            y=power_data,
            mode='lines+markers',
            name='Average Power',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        # Add smoothed trend line if requested
        if show_trendline:
            # Simple linear trend
            x_numeric = np.arange(len(dates))
            coeffs = np.polyfit(x_numeric, power_data, 1)
            trend_line = np.polyval(coeffs, x_numeric)
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=trend_line,
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Power Performance Trends',
            xaxis_title='Date',
            yaxis_title='Average Power (W)',
            hovermode='x unified'
        )
        
        return fig
    
    def _create_demo_volume_trends(self, time_period):
        """Create a demo training volume trends chart."""
        # Generate sample weekly training data
        weeks = 26 if time_period == '6m' else 52  # 6 months or 1 year
        dates = pd.date_range(end=datetime.now(), periods=weeks, freq='W')
        
        # Generate realistic training volume data
        base_hours = 8  # hours per week
        volume_data = []
        distance_data = []
        
        for i, date in enumerate(dates):
            # Seasonal variation in training
            seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (date.dayofyear / 365.25 + 0.25))
            # Progressive volume increase with some variation
            progress_factor = 1 + 0.3 * i / len(dates)
            # Random variation
            noise = np.random.normal(1, 0.2)
            
            hours = base_hours * seasonal_factor * progress_factor * noise
            distance = hours * (25 + np.random.normal(0, 3))  # ~25 km/h average
            
            volume_data.append(max(hours, 2))  # Minimum 2 hours
            distance_data.append(max(distance, 50))  # Minimum 50km
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Weekly Training Hours', 'Weekly Distance'),
            vertical_spacing=0.1
        )
        
        # Training hours
        fig.add_trace(
            go.Scatter(x=dates, y=volume_data, mode='lines+markers',
                      name='Hours/Week', line=dict(color='#2ca02c')),
            row=1, col=1
        )
        
        # Weekly distance
        fig.add_trace(
            go.Scatter(x=dates, y=distance_data, mode='lines+markers',
                      name='Distance/Week (km)', line=dict(color='#ff7f0e')),
            row=2, col=1
        )
        
        fig.update_layout(title='Training Volume Trends', height=400, showlegend=False)
        return fig
    
    def _create_demo_seasonal_analysis(self):
        """Create a demo seasonal performance analysis."""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Demo seasonal data (winter lower, summer higher)
        power_2024 = [235, 240, 250, 265, 275, 285, 290, 285, 280, 270, 255, 245]
        power_2025 = [245, 252, 265, 280, 290, 300, 305, 295, 288, 275, 260, 250]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months, y=power_2024,
            mode='lines+markers',
            name='2024',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=months, y=power_2025,
            mode='lines+markers',
            name='2025',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        fig.update_layout(
            title='Seasonal Performance Comparison',
            xaxis_title='Month',
            yaxis_title='Average Power (W)'
        )
        
        return fig
    
    def _create_demo_comprehensive_timeline(self, primary_metric):
        """Create a demo comprehensive timeline chart."""
        dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
        
        # Generate multiple metrics over time
        base_values = {
            'avg_power': 250,
            'ftp': 275,
            'tss': 350,
            'distance': 45
        }
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Power', 'FTP Estimates', 'Training Stress Score', 'Distance per Ride'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Generate data for each metric
        for i, (metric, base) in enumerate(base_values.items()):
            # Add progressive improvement with noise
            trend = base * (1 + 0.15 * np.linspace(0, 1, len(dates)))
            noise = np.random.normal(0, base * 0.1, len(dates))
            values = trend + noise
            
            row = 1 if i < 2 else 2
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(x=dates, y=values, mode='lines', name=metric.replace('_', ' ').title()),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False, title='Comprehensive Performance Timeline')
        return fig
    
    def _create_goals_progress(self):
        """Create goals progress display."""
        goals = [
            {"name": "FTP Improvement", "current": 287, "target": 300, "unit": "W"},
            {"name": "Weekly Distance", "current": 285, "target": 300, "unit": "km"},
            {"name": "Training Consistency", "current": 85, "target": 90, "unit": "%"}
        ]
        
        goal_items = []
        for goal in goals:
            progress = min(100, (goal['current'] / goal['target']) * 100)
            color = "success" if progress >= 90 else "warning" if progress >= 70 else "danger"
            
            goal_items.append(
                html.Div([
                    html.H6(goal['name'], className="mb-2"),
                    dbc.Progress(
                        value=progress,
                        color=color,
                        striped=True,
                        animated=True,
                        className="mb-2"
                    ),
                    html.P(f"{goal['current']}{goal['unit']} / {goal['target']}{goal['unit']} ({progress:.0f}%)",
                           className="small text-muted")
                ], className="mb-3")
            )
        
        return goal_items
    
    def _create_performance_insights(self):
        """Create performance insights and recommendations."""
        return dbc.Row([
            dbc.Col([
                html.H6("🎯 Key Insights", className="mb-3"),
                html.Ul([
                    html.Li("Power output has improved 15.2% over the last 6 months"),
                    html.Li("Training consistency remains strong at 85%"),
                    html.Li("Peak performance typically occurs in summer months"),
                    html.Li("Recovery rides show good adherence to target zones")
                ])
            ], width=6),
            
            dbc.Col([
                html.H6("💡 Recommendations", className="mb-3"),
                html.Ul([
                    html.Li("Focus on winter base training to maintain fitness"),
                    html.Li("Consider adding more interval training sessions"),
                    html.Li("Track sleep and recovery metrics for optimization"),
                    html.Li("Plan periodization around key events")
                ])
            ], width=6)
        ])