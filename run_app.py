"""Entrypoint to run the Cycling Tracker Dash application."""

from cycling_tracker.app import create_app


app = create_app()


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)

