"""Foundation package for workout structure recognition and metrics.

Modules:
- io: Loading ride files into pandas DataFrames
- models: Typed domain objects
- recognition: Interval detection and repeated structure recognition
- metrics: Interval and workout metrics
- aggregation: Cross-workout summaries and trends
- storage: Export helpers
- cli: Command line interface
"""

__all__ = [
    "io",
    "models",
    "recognition",
    "metrics",
    "aggregation",
    "storage",
    "cli",
]


