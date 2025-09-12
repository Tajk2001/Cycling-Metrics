"""
Cycling Tracker - modular backend for cycling data processing and analysis.

Public exports kept minimal; core utilities are available under subpackages.
"""

from .core.fit_parser import load_fit_to_dataframe, parse_fit_file  # re-export for convenience

__all__ = [
    "load_fit_to_dataframe",
    "parse_fit_file",
]

