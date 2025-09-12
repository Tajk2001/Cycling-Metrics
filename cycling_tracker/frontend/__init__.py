"""
Frontend module for the Cycling Performance Tracking System
=======================================================

This module contains the Dash-based web dashboard for visualizing and analyzing
cycling performance data. It includes:

- Main dashboard application
- Component definitions for different visualizations
- Layout management for responsive design
- Callback functions for interactivity
- Utility functions for data formatting and processing

Components:
- components/: Individual visualization components
- layouts/: Page layouts and structure definitions  
- callbacks/: Interactive callback functions
- utils/: Helper functions and utilities

Usage:
    from cycling_tracker.frontend import CyclingDashboard
    
    app = CyclingDashboard()
    app.run_server(debug=True)
"""

from .main_dashboard import CyclingDashboard

__all__ = ['CyclingDashboard']