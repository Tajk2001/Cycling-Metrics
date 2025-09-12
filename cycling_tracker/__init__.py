"""
Cycling Tracker - Comprehensive cycling performance analysis system.

A modular system for analyzing cycling FIT files, detecting intervals,
calculating metrics, and tracking performance over time.
"""

from .main import (
    CyclingTracker, 
    setup_cycling_tracker, 
    process_single_ride
)

from .core.fit_parser import parse_fit_file
from .core.interval_detector import detect_intervals
from .metrics.ride_metrics import calculate_ride_metrics
from .storage.csv_manager import CSVStorageManager
from .storage.data_models import RideMetrics, IntervalMetrics, RideData
from .utils.config import get_config, reset_config

__version__ = "1.0.0"
__author__ = "Cycling Tracker System"

# Main interface classes
__all__ = [
    # Main interfaces
    "CyclingTracker",
    "setup_cycling_tracker", 
    "process_single_ride",
    
    # Core functionality
    "parse_fit_file",
    "detect_intervals",
    "calculate_ride_metrics",
    
    # Data management
    "CSVStorageManager",
    "RideMetrics",
    "IntervalMetrics", 
    "RideData",
    
    # Configuration
    "get_config",
    "reset_config"
]