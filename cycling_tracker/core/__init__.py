"""Core processing modules for FIT parsing and interval detection."""

from .fit_parser import FITParser, parse_fit_file
from .interval_detector import IntervalDetector, detect_intervals

__all__ = [
    "FITParser",
    "parse_fit_file", 
    "IntervalDetector",
    "detect_intervals"
]