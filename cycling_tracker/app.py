"""Dash app factory for the Cycling Tracker UI."""

from dash import Dash

from .layout import build_layout
from .callbacks import register_callbacks


def create_app() -> Dash:
    """Create and configure the Dash application instance.

    Returns:
        Dash: Configured Dash application.
    """
    app = Dash(
        __name__,
        title="Cycling Tracker",
        suppress_callback_exceptions=True,
    )

    app.layout = build_layout()
    register_callbacks(app)

    return app


if __name__ == "__main__":
    _app = create_app()
    _app.run_server(debug=True, host="0.0.0.0", port=8050)

