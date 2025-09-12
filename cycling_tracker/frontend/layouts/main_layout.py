"""
Main Dashboard Layout
==================

Defines the overall structure and layout of the cycling performance dashboard.
Creates a responsive, modern interface with navigation tabs and content areas.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_main_layout():
    """Create the main dashboard layout with navigation and content areas."""
    
    return dbc.Container([
        
        # Header Section
        dbc.Row([
            dbc.Col([
                html.H1("Cycling Performance Tracker", 
                       className="display-4 text-center mb-0"),
                html.P("Comprehensive ride analysis and performance monitoring",
                       className="lead text-center text-muted")
            ])
        ], className="mb-4 pt-3"),
        
        # Data Upload Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Data Upload", className="card-title"),
                        dcc.Upload(
                            id='upload-fit-file',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                                html.Br(),
                                'Drag & Drop or Click to Upload FIT File'
                            ]),
                            style={
                                'width': '100%',
                                'height': '80px',
                                'lineHeight': '80px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'cursor': 'pointer'
                            },
                            multiple=False
                        ),
                        html.Div(id='data-status', className="mt-2")
                    ])
                ], color="light")
            ], width=12)
        ], className="mb-4"),
        
        # Navigation Tabs
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="Ride Overview", tab_id="ride-overview", 
                           tab_style={"margin-left": "auto"}),
                    dbc.Tab(label="Interval Analysis", tab_id="interval-analysis"),
                    dbc.Tab(label="Multi-Ride Comparison", tab_id="multi-ride-comparison"),
                    dbc.Tab(label="Performance Trends", tab_id="performance-trends"),
                ], 
                id="main-tabs", 
                active_tab="ride-overview",
                className="mb-4"
                )
            ])
        ]),
        
        # Main Content Area
        dbc.Row([
            dbc.Col([
                html.Div(id="page-content")
            ])
        ]),
        
        # Hidden data stores for cross-component communication
        dcc.Store(id='ride-data-store'),
        dcc.Store(id='historical-data-store'), 
        dcc.Store(id='global-data-context'),
        
        # Footer
        html.Hr(className="mt-5"),
        dbc.Row([
            dbc.Col([
                html.P("Cycling Performance Tracker v1.0 | Built with Dash & Plotly",
                      className="text-center text-muted small")
            ])
        ], className="mb-3")
        
    ], fluid=True, className="px-4")


def create_loading_spinner(component_id):
    """Create a loading spinner for components."""
    return dbc.Spinner([
        html.Div(id=component_id)
    ], color="primary", spinner_style={"width": "3rem", "height": "3rem"})


def create_error_alert(message, alert_type="danger"):
    """Create an error alert component."""
    return dbc.Alert(
        [
            html.I(className="fas fa-exclamation-triangle me-2"),
            message
        ],
        color=alert_type,
        dismissable=True
    )


def create_success_alert(message):
    """Create a success alert component.""" 
    return create_error_alert(message, alert_type="success")