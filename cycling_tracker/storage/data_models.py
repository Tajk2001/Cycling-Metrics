"""
Data models for the cycling tracker system.
Defines the structure for ride and interval data.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

@dataclass
class RideMetrics:
    """Comprehensive ride-level metrics."""
    # Basic ride information
    ride_id: str
    date: datetime
    filename: str
    
    # Duration and distance
    total_time_seconds: int
    moving_time_seconds: int
    total_distance_km: float = 0.0
    
    # Power metrics
    avg_power_watts: float = 0.0
    max_power_watts: float = 0.0
    normalized_power_watts: float = 0.0
    avg_power_per_kg: float = 0.0
    max_power_per_kg: float = 0.0
    total_work_kj: float = 0.0
    
    # Training stress metrics
    intensity_factor: float = 0.0
    training_stress_score: float = 0.0
    
    # Heart rate metrics
    avg_heart_rate: Optional[int] = None
    max_heart_rate: Optional[int] = None
    heart_rate_zones: Dict[str, float] = field(default_factory=dict)
    
    # Speed and elevation
    avg_speed_kph: float = 0.0
    max_speed_kph: float = 0.0
    total_elevation_gain_m: float = 0.0
    
    # Cadence metrics
    avg_cadence: Optional[float] = None
    max_cadence: Optional[float] = None
    
    # Temperature
    avg_temperature_c: Optional[float] = None
    
    # Interval summary
    total_intervals_detected: int = 0
    intervals_by_zone: Dict[str, int] = field(default_factory=dict)
    
    # Power curve data (for comparison across rides)
    power_curve_5s: float = 0.0
    power_curve_10s: float = 0.0
    power_curve_30s: float = 0.0
    power_curve_1min: float = 0.0
    power_curve_5min: float = 0.0
    power_curve_10min: float = 0.0
    power_curve_20min: float = 0.0
    power_curve_60min: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV storage."""
        return {
            'ride_id': self.ride_id,
            'date': self.date.isoformat(),
            'filename': self.filename,
            'total_time_seconds': self.total_time_seconds,
            'moving_time_seconds': self.moving_time_seconds,
            'total_distance_km': self.total_distance_km,
            'avg_power_watts': self.avg_power_watts,
            'max_power_watts': self.max_power_watts,
            'normalized_power_watts': self.normalized_power_watts,
            'avg_power_per_kg': self.avg_power_per_kg,
            'max_power_per_kg': self.max_power_per_kg,
            'total_work_kj': self.total_work_kj,
            'intensity_factor': self.intensity_factor,
            'training_stress_score': self.training_stress_score,
            'avg_heart_rate': self.avg_heart_rate,
            'max_heart_rate': self.max_heart_rate,
            'avg_speed_kph': self.avg_speed_kph,
            'max_speed_kph': self.max_speed_kph,
            'total_elevation_gain_m': self.total_elevation_gain_m,
            'avg_cadence': self.avg_cadence,
            'max_cadence': self.max_cadence,
            'avg_temperature_c': self.avg_temperature_c,
            'total_intervals_detected': self.total_intervals_detected,
            'power_curve_5s': self.power_curve_5s,
            'power_curve_10s': self.power_curve_10s,
            'power_curve_30s': self.power_curve_30s,
            'power_curve_1min': self.power_curve_1min,
            'power_curve_5min': self.power_curve_5min,
            'power_curve_10min': self.power_curve_10min,
            'power_curve_20min': self.power_curve_20min,
            'power_curve_60min': self.power_curve_60min
        }

@dataclass
class IntervalMetrics:
    """Comprehensive interval-level metrics."""
    # Basic interval information
    interval_id: str
    ride_id: str
    interval_number: int
    
    # Timing
    start_time: datetime
    end_time: datetime
    duration_seconds: int
    
    # Source and classification
    detection_source: str  # 'lap', 'power_threshold', 'ml_detection'
    interval_type: str  # 'work', 'recovery', 'test', 'race'
    power_zone: str  # zone1-zone7
    
    # Power metrics
    avg_power_watts: float = 0.0
    max_power_watts: float = 0.0
    min_power_watts: float = 0.0
    avg_power_per_kg: float = 0.0
    max_power_per_kg: float = 0.0
    power_cv: float = 0.0  # Coefficient of variation
    power_fade_pct: float = 0.0  # Power fade percentage
    normalized_power_watts: float = 0.0
    work_kj: float = 0.0
    
    # Heart rate metrics
    avg_heart_rate: Optional[int] = None
    max_heart_rate: Optional[int] = None
    min_heart_rate: Optional[int] = None
    heart_rate_cv: Optional[float] = None
    heart_rate_fade_pct: Optional[float] = None
    
    # Speed and cadence
    avg_speed_kph: float = 0.0
    max_speed_kph: float = 0.0
    distance_km: float = 0.0
    avg_cadence: Optional[float] = None
    max_cadence: Optional[float] = None
    
    # Quality and consistency metrics
    quality_score: float = 0.0  # Overall interval quality (0-100)
    consistency_score: float = 0.0  # Power/HR consistency (0-100)
    
    # Relative to ride
    power_relative_to_ride_avg: float = 0.0  # Ratio to ride average power
    power_relative_to_ftp: float = 0.0  # Ratio to FTP
    
    # Recovery metrics (for intervals following this one)
    recovery_time_after_seconds: Optional[int] = None
    recovery_power_ratio: Optional[float] = None
    
    # Set information (if part of a repeating set)
    is_part_of_set: bool = False
    set_id: Optional[str] = None
    set_position: Optional[int] = None
    set_total_intervals: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV storage."""
        return {
            'interval_id': self.interval_id,
            'ride_id': self.ride_id,
            'interval_number': self.interval_number,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'detection_source': self.detection_source,
            'interval_type': self.interval_type,
            'power_zone': self.power_zone,
            'avg_power_watts': self.avg_power_watts,
            'max_power_watts': self.max_power_watts,
            'min_power_watts': self.min_power_watts,
            'avg_power_per_kg': self.avg_power_per_kg,
            'max_power_per_kg': self.max_power_per_kg,
            'power_cv': self.power_cv,
            'power_fade_pct': self.power_fade_pct,
            'normalized_power_watts': self.normalized_power_watts,
            'work_kj': self.work_kj,
            'avg_heart_rate': self.avg_heart_rate,
            'max_heart_rate': self.max_heart_rate,
            'min_heart_rate': self.min_heart_rate,
            'heart_rate_cv': self.heart_rate_cv,
            'heart_rate_fade_pct': self.heart_rate_fade_pct,
            'avg_speed_kph': self.avg_speed_kph,
            'max_speed_kph': self.max_speed_kph,
            'distance_km': self.distance_km,
            'avg_cadence': self.avg_cadence,
            'max_cadence': self.max_cadence,
            'quality_score': self.quality_score,
            'consistency_score': self.consistency_score,
            'power_relative_to_ride_avg': self.power_relative_to_ride_avg,
            'power_relative_to_ftp': self.power_relative_to_ftp,
            'recovery_time_after_seconds': self.recovery_time_after_seconds,
            'recovery_power_ratio': self.recovery_power_ratio,
            'is_part_of_set': self.is_part_of_set,
            'set_id': self.set_id,
            'set_position': self.set_position,
            'set_total_intervals': self.set_total_intervals
        }

@dataclass
class RideData:
    """Container for raw ride data and processed metrics."""
    # Raw data
    dataframe: pd.DataFrame
    lap_dataframe: Optional[pd.DataFrame] = None
    
    # Metadata
    filename: str = ""
    file_path: str = ""
    processed_at: datetime = field(default_factory=datetime.now)
    
    # Processed metrics
    ride_metrics: Optional[RideMetrics] = None
    interval_metrics: List[IntervalMetrics] = field(default_factory=list)
    
    # Processing flags
    is_processed: bool = False
    has_power_data: bool = False
    has_heart_rate_data: bool = False
    has_lap_data: bool = False
    
    def get_interval_count_by_zone(self) -> Dict[str, int]:
        """Get count of intervals by power zone."""
        zone_counts = {}
        for interval in self.interval_metrics:
            zone = interval.power_zone
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        return zone_counts
    
    def get_intervals_by_duration(self, min_duration: int, max_duration: int) -> List[IntervalMetrics]:
        """Get intervals within a duration range (seconds)."""
        return [
            interval for interval in self.interval_metrics
            if min_duration <= interval.duration_seconds <= max_duration
        ]
    
    def get_work_intervals(self) -> List[IntervalMetrics]:
        """Get only work intervals (exclude recovery)."""
        return [
            interval for interval in self.interval_metrics
            if interval.interval_type == 'work'
        ]

class PowerCurveCalculator:
    """Calculate power curve data points from ride data."""
    
    STANDARD_DURATIONS = [5, 10, 30, 60, 300, 600, 1200, 3600]  # seconds
    
    @staticmethod
    def calculate_power_curve(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate power curve for standard durations."""
        if 'power' not in df.columns or df['power'].isna().all():
            return {f'{d}s': 0.0 for d in PowerCurveCalculator.STANDARD_DURATIONS}
        
        power_curve = {}
        power_series = df['power'].dropna()
        
        for duration in PowerCurveCalculator.STANDARD_DURATIONS:
            if len(power_series) >= duration:
                # Calculate rolling mean and take maximum
                rolling_mean = power_series.rolling(window=duration, min_periods=duration).mean()
                max_avg_power = rolling_mean.max()
                power_curve[f'{duration}s'] = max_avg_power if not pd.isna(max_avg_power) else 0.0
            else:
                power_curve[f'{duration}s'] = 0.0
        
        return power_curve

def create_ride_id(date: datetime, filename: str) -> str:
    """Create a unique ride ID based on date and filename."""
    date_str = date.strftime("%Y%m%d_%H%M%S")
    file_base = filename.replace('.fit', '').replace('.FIT', '')
    return f"{date_str}_{file_base}"

def create_interval_id(ride_id: str, interval_number: int) -> str:
    """Create a unique interval ID."""
    return f"{ride_id}_INT_{interval_number:03d}"