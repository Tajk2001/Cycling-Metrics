"""UI layout for the Cycling Tracker Dash app."""

from dash import dcc, html


def build_layout():
    """Construct the base application layout."""
    return html.Div(
        className="container",
        children=[
            dcc.Store(id="app-store"),

            html.Header(
                role="banner",
                children=[
                    html.H1("Cycling Tracker"),
                    html.P("Upload ride FIT files to analyze performance and intervals."),
                ],
            ),

            html.Main(
                role="main",
                children=[
                    html.Section(
                        id="upload-section",
                        children=[
                            html.Div(
                                id="upload-placeholder",
                                className="placeholder",
                                children="File upload coming next...",
                            )
                        ],
                    ),
                    html.Hr(),
                    html.Section(
                        id="overview",
                        children=[
                            html.H2("Overview"),
                            html.Div(
                                className="grid two-col",
                                children=[
                                    dcc.Graph(
                                        id="power-over-time",
                                        figure={},
                                        config={"displayModeBar": False},
                                    ),
                                    dcc.Graph(
                                        id="heart-rate-over-time",
                                        figure={},
                                        config={"displayModeBar": False},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            html.Footer(role="contentinfo", children=[html.Small("v0.1.0")]),
        ],
    )

