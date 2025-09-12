"""
Main Dashboard Application
========================

This is the primary Dash application for the Cycling Performance Tracking System.
It creates and manages the main dashboard interface with comprehensive ride analysis
and multi-ride comparison capabilities.

Based on analysis of SprintV1.py but redesigned for:
- Complete ride analysis (not just sprints)
- Lap-based interval detection
- Multi-ride performance tracking
- Improved modularity and maintainability
"""

import dash
from dash import html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .layouts.main_layout import create_main_layout
from .components.ride_overview import RideOverviewComponent
from .components.interval_analysis import IntervalAnalysisComponent  
from .components.multi_ride_comparison import MultiRideComparisonComponent
from .components.performance_trends import PerformanceTrendsComponent
from .utils.data_formatter import DataFormatter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CyclingDashboard:
    """
    Main Dash application for cycling performance analysis.
    
    This class manages the overall dashboard structure, component integration,
    and high-level callback coordination.
    """
    
    def __init__(self, app_name="Cycling Performance Tracker"):
        """Initialize the dashboard application."""
        self.app = dash.Dash(__name__, 
                           suppress_callback_exceptions=True,
                           meta_tags=[
                               {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                           ])
        
        self.app.title = app_name
        
        # Initialize components
        self.ride_overview = RideOverviewComponent()
        self.interval_analysis = IntervalAnalysisComponent()
        self.multi_ride_comparison = MultiRideComparisonComponent()
        self.performance_trends = PerformanceTrendsComponent()
        self.data_formatter = DataFormatter()
        
        # Data storage
        self.current_ride_data = None
        self.historical_data = None
        
        # Set up the dashboard
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info("Cycling Dashboard initialized successfully")
    
    def _setup_layout(self):
        """Set up the main dashboard layout."""
        self.app.layout = create_main_layout()
        logger.info("Dashboard layout configured")
    
    def _setup_callbacks(self):
        """Register all dashboard callbacks."""
        self._register_navigation_callbacks()
        self._register_data_loading_callbacks()
        self._register_component_callbacks()
        logger.info("Dashboard callbacks registered")
    
    def _register_navigation_callbacks(self):
        """Register callbacks for navigation and page switching."""
        
        @self.app.callback(
            Output('page-content', 'children'),
            Input('main-tabs', 'active_tab')
        )
        def display_page_content(active_tab):
            """Switch between different dashboard pages."""
            if active_tab == 'ride-overview':
                return self.ride_overview.get_layout()
            elif active_tab == 'interval-analysis':
                return self.interval_analysis.get_layout()
            elif active_tab == 'multi-ride-comparison':
                return self.multi_ride_comparison.get_layout()
            elif active_tab == 'performance-trends':
                return self.performance_trends.get_layout()
            else:
                return self._get_welcome_page()
    
    def _register_data_loading_callbacks(self):
        """Register callbacks for data loading and management."""
        
        @self.app.callback(
            [Output('data-status', 'children'),
             Output('ride-data-store', 'data')],
            Input('upload-fit-file', 'contents'),
            State('upload-fit-file', 'filename')
        )
        def handle_file_upload(contents, filename):
            """Handle FIT file upload and processing."""
            if contents is None:
                return "No file uploaded", {}
            
            try:
                # TODO: Integrate with backend FIT file processing
                # For now, return placeholder data structure
                ride_data = {
                    'filename': filename,
                    'upload_time': datetime.now().isoformat(),
                    'status': 'uploaded',
                    'processed': False
                }
                
                status_message = html.Div([
                    html.I(className="fas fa-check-circle", 
                          style={'color': 'green', 'margin-right': '5px'}),
                    f"Successfully uploaded: {filename}"
                ])
                
                return status_message, ride_data
                
            except Exception as e:
                logger.error(f"Error processing file upload: {e}")
                error_message = html.Div([
                    html.I(className="fas fa-exclamation-triangle", 
                          style={'color': 'red', 'margin-right': '5px'}),
                    f"Error uploading file: {str(e)}"
                ])
                return error_message, {}
    
    def _register_component_callbacks(self):
        """Register callbacks for individual components."""
        # Register callbacks for each component
        self.ride_overview.register_callbacks(self.app)
        self.interval_analysis.register_callbacks(self.app)
        self.multi_ride_comparison.register_callbacks(self.app)
        self.performance_trends.register_callbacks(self.app)
        
        @self.app.callback(
            Output('global-data-context', 'data'),
            [Input('ride-data-store', 'data'),
             Input('historical-data-store', 'data')]
        )
        def update_global_context(ride_data, historical_data):
            """Update global data context for cross-component communication."""
            return {
                'current_ride': ride_data or {},
                'historical': historical_data or {},
                'last_updated': datetime.now().isoformat()
            }
    
    def _get_welcome_page(self):
        """Get the welcome/landing page content."""
        return html.Div([
            html.H2("Welcome to Cycling Performance Tracker", 
                   className="text-center mb-4"),
            
            html.Div([
                html.Div([
                    html.I(className="fas fa-upload fa-3x mb-3"),
                    html.H4("Get Started"),
                    html.P("Upload a FIT file to begin analyzing your ride data"),
                    dcc.Upload(
                        id='welcome-upload',
                        children=html.Button('Upload FIT File',
                                           className='btn btn-primary'),
                        style={'width': '100%', 'textAlign': 'center'}
                    )
                ], className="col-md-4 text-center"),
                
                html.Div([
                    html.I(className="fas fa-chart-line fa-3x mb-3"),
                    html.H4("Comprehensive Analysis"),
                    html.P("Analyze complete rides with automatic interval detection from lap data")
                ], className="col-md-4 text-center"),
                
                html.Div([
                    html.I(className="fas fa-history fa-3x mb-3"),
                    html.H4("Track Progress"),
                    html.P("Compare multiple rides and track performance improvements over time")
                ], className="col-md-4 text-center")
                
            ], className="row justify-content-center mt-5"),
            
            html.Hr(className="mt-5"),
            
            html.Div([
                html.H5("Features:"),
                html.Ul([
                    html.Li("Automatic interval detection from lap data"),
                    html.Li("Comprehensive ride metrics calculation"),
                    html.Li("Multi-ride comparison and trend analysis"),
                    html.Li("Power zone analysis and interval grouping"),
                    html.Li("Performance evolution tracking"),
                    html.Li("CSV data export and management")
                ])
            ], className="mt-4")
            
        ], className="container")
    
    def run_server(self, debug=True, host='127.0.0.1', port=8050, **kwargs):
        """Run the dashboard server."""
        logger.info(f"Starting dashboard server on {host}:{port}")
        self.app.run_server(debug=debug, host=host, port=port, **kwargs)
    
    def get_app(self):
        """Return the underlying Dash app instance."""
        return self.app


# For direct execution
if __name__ == '__main__':
    dashboard = CyclingDashboard()
    dashboard.run_server(debug=True)