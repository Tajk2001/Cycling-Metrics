"""
Ride-level metrics calculation for the cycling tracker system.
Calculates comprehensive metrics for entire rides based on SprintV1.py algorithms.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

from ..storage.data_models import RideData, RideMetrics, PowerCurveCalculator, create_ride_id
from ..utils.config import get_config

class RideMetricsCalculator:
    """
    Calculate comprehensive ride-level metrics.
    """
    
    def __init__(self):
        self.config = get_config()
    
    def calculate_ride_metrics(self, ride_data: RideData, intervals: Optional[List] = None) -> RideMetrics:
        """
        Calculate comprehensive ride metrics from ride data.
        
        Args:
            ride_data: RideData object containing parsed FIT data
            intervals: Optional list of detected intervals for summary stats
            
        Returns:
            RideMetrics object with calculated metrics
        """
        df = ride_data.dataframe
        
        if df.empty:
            raise ValueError("Cannot calculate metrics from empty dataframe")
        
        print("📊 Calculating ride-level metrics...")
        
        # Create ride ID
        start_time = df.index[0]
        ride_id = create_ride_id(start_time, ride_data.filename)
        
        # Basic timing metrics
        timing_metrics = self._calculate_timing_metrics(df)
        
        # Power metrics
        power_metrics = self._calculate_power_metrics(df)
        
        # Training stress metrics
        stress_metrics = self._calculate_training_stress_metrics(df, power_metrics)
        
        # Heart rate metrics
        hr_metrics = self._calculate_heart_rate_metrics(df)
        
        # Speed and distance metrics
        speed_metrics = self._calculate_speed_distance_metrics(df)
        
        # Elevation metrics
        elevation_metrics = self._calculate_elevation_metrics(df)
        
        # Cadence metrics
        cadence_metrics = self._calculate_cadence_metrics(df)
        
        # Environmental metrics
        env_metrics = self._calculate_environmental_metrics(df)
        
        # Interval summary metrics
        interval_summary = self._calculate_interval_summary_metrics(intervals) if intervals else {}
        
        # Power curve metrics
        power_curve = PowerCurveCalculator.calculate_power_curve(df)
        
        # Create RideMetrics object
        ride_metrics = RideMetrics(
            ride_id=ride_id,
            date=start_time,
            filename=ride_data.filename,
            **timing_metrics,
            **power_metrics,
            **stress_metrics,
            **hr_metrics,
            **speed_metrics,
            **elevation_metrics,
            **cadence_metrics,
            **env_metrics,
            **interval_summary,
            power_curve_5s=power_curve.get('5s', 0.0),
            power_curve_10s=power_curve.get('10s', 0.0),
            power_curve_30s=power_curve.get('30s', 0.0),
            power_curve_1min=power_curve.get('60s', 0.0),
            power_curve_5min=power_curve.get('300s', 0.0),
            power_curve_10min=power_curve.get('600s', 0.0),
            power_curve_20min=power_curve.get('1200s', 0.0),
            power_curve_60min=power_curve.get('3600s', 0.0)
        )
        
        print(f"✅ Calculated ride metrics for {ride_id}")
        print(f"   • Duration: {timing_metrics['total_time_seconds'] / 3600:.2f} hours")
        print(f"   • Distance: {speed_metrics.get('total_distance_km', 0):.1f} km")
        print(f"   • Average Power: {power_metrics.get('avg_power_watts', 0):.0f}W")
        print(f"   • TSS: {stress_metrics.get('training_stress_score', 0):.0f}")
        
        return ride_metrics
    
    def _calculate_timing_metrics(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate timing-related metrics."""
        total_time = (df.index[-1] - df.index[0]).total_seconds()
        
        # Calculate moving time (exclude stationary periods)
        moving_time = self._calculate_moving_time(df)
        
        return {
            'total_time_seconds': int(total_time),
            'moving_time_seconds': int(moving_time)
        }
    
    def _calculate_moving_time(self, df: pd.DataFrame) -> float:
        """Calculate moving time excluding stationary periods."""
        # Use speed threshold to determine moving vs stationary
        speed_threshold = self.config.processing.speed_threshold
        
        # Check multiple speed columns
        speed_col = None
        for col in ['speed', 'enhanced_speed', 'speed_kph', 'enhanced_speed_kph']:
            if col in df.columns:
                speed_col = col
                break
        
        if speed_col is None:
            # If no speed data, return total time
            return (df.index[-1] - df.index[0]).total_seconds()
        
        # Convert to m/s if needed
        speed_data = df[speed_col].fillna(0)
        if 'kph' in speed_col:
            speed_data = speed_data / 3.6
        
        # Count periods above speed threshold
        moving_mask = speed_data > speed_threshold
        moving_periods = moving_mask.sum()
        
        # Estimate sampling rate
        if len(df) > 1:
            avg_sampling_rate = len(df) / (df.index[-1] - df.index[0]).total_seconds()
        else:
            avg_sampling_rate = 1.0
        
        return moving_periods / avg_sampling_rate
    
    def _calculate_power_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate power-related metrics."""
        metrics = {
            'avg_power_watts': 0.0,
            'max_power_watts': 0.0,
            'normalized_power_watts': 0.0,
            'avg_power_per_kg': 0.0,
            'max_power_per_kg': 0.0,
            'total_work_kj': 0.0
        }
        
        if 'power' not in df.columns or df['power'].isna().all():
            return metrics
        
        power = df['power'].dropna()
        if power.empty:
            return metrics
        
        # Basic power metrics
        metrics['avg_power_watts'] = power.mean()
        metrics['max_power_watts'] = power.max()
        
        # Power-to-weight ratios
        if self.config.rider.mass_kg > 0:
            metrics['avg_power_per_kg'] = metrics['avg_power_watts'] / self.config.rider.mass_kg
            metrics['max_power_per_kg'] = metrics['max_power_watts'] / self.config.rider.mass_kg
        
        # Total work (kJ)
        # Assume 1-second sampling rate for simplicity
        metrics['total_work_kj'] = power.sum() / 1000
        
        # Normalized Power calculation (30-second rolling average, then 4th root of mean of 4th powers)
        if len(power) >= 30:
            # Calculate 30-second rolling averages
            rolling_30s = power.rolling(window=30, min_periods=30).mean().dropna()
            if not rolling_30s.empty:
                # Calculate 4th root of mean of 4th powers
                fourth_powers = rolling_30s ** 4
                mean_fourth_power = fourth_powers.mean()
                normalized_power = mean_fourth_power ** 0.25
                metrics['normalized_power_watts'] = normalized_power
            else:
                metrics['normalized_power_watts'] = metrics['avg_power_watts']
        else:
            metrics['normalized_power_watts'] = metrics['avg_power_watts']
        
        return metrics
    
    def _calculate_training_stress_metrics(self, df: pd.DataFrame, power_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate training stress metrics (IF, TSS)."""
        metrics = {
            'intensity_factor': 0.0,
            'training_stress_score': 0.0
        }
        
        try:
            ftp = self.config.get_ftp_for_interval_detection()
        except ValueError:
            return metrics
        
        if ftp <= 0 or power_metrics['normalized_power_watts'] <= 0:
            return metrics
        
        # Intensity Factor (IF) = NP / FTP
        metrics['intensity_factor'] = power_metrics['normalized_power_watts'] / ftp
        
        # Training Stress Score (TSS) = (duration_hours × IF² × 100)
        duration_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
        if duration_hours > 0:
            metrics['training_stress_score'] = (
                duration_hours * (metrics['intensity_factor'] ** 2) * 100
            )
        
        return metrics
    
    def _calculate_heart_rate_metrics(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate heart rate metrics."""
        metrics = {
            'avg_heart_rate': None,
            'max_heart_rate': None,
            'heart_rate_zones': {}
        }
        
        if 'heart_rate' not in df.columns or df['heart_rate'].isna().all():
            return metrics
        
        hr = df['heart_rate'].dropna()
        if hr.empty:
            return metrics
        
        metrics['avg_heart_rate'] = int(hr.mean())
        metrics['max_heart_rate'] = int(hr.max())
        
        # Calculate time in heart rate zones (if LTHR is set)
        if self.config.rider.lthr > 0:
            lthr = self.config.rider.lthr
            hr_zones = self._calculate_hr_zones(hr, lthr)
            metrics['heart_rate_zones'] = hr_zones
        
        return metrics
    
    def _calculate_hr_zones(self, hr_data: pd.Series, lthr: int) -> Dict[str, float]:
        """Calculate time spent in heart rate zones."""
        zones = {
            'zone1': (0, 0.68 * lthr),       # Active Recovery
            'zone2': (0.69 * lthr, 0.83 * lthr),  # Aerobic Base
            'zone3': (0.84 * lthr, 0.94 * lthr),  # Aerobic Build
            'zone4': (0.95 * lthr, 1.05 * lthr),  # Lactate Threshold
            'zone5': (1.06 * lthr, float('inf'))   # VO2 Max+
        }
        
        zone_times = {}
        total_points = len(hr_data)
        
        for zone_name, (min_hr, max_hr) in zones.items():
            in_zone = ((hr_data >= min_hr) & (hr_data <= max_hr)).sum()
            zone_times[zone_name] = (in_zone / total_points) * 100 if total_points > 0 else 0
        
        return zone_times
    
    def _calculate_speed_distance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate speed and distance metrics."""
        metrics = {
            'total_distance_km': 0.0,
            'avg_speed_kph': 0.0,
            'max_speed_kph': 0.0
        }
        
        # Try to get distance from multiple possible columns
        distance_col = None
        for col in ['distance', 'total_distance']:
            if col in df.columns and not df[col].isna().all():
                distance_col = col
                break
        
        if distance_col is not None:
            distance_data = df[distance_col].dropna()
            if not distance_data.empty:
                # Total distance (assuming distance is cumulative)
                metrics['total_distance_km'] = (distance_data.iloc[-1] - distance_data.iloc[0]) / 1000
        
        # Try to get speed from multiple possible columns
        speed_col = None
        for col in ['speed_kph', 'enhanced_speed_kph', 'speed', 'enhanced_speed']:
            if col in df.columns and not df[col].isna().all():
                speed_col = col
                break
        
        if speed_col is not None:
            speed_data = df[speed_col].dropna()
            if not speed_data.empty:
                # Convert to km/h if needed
                if speed_col in ['speed', 'enhanced_speed']:
                    # Assume m/s, convert to km/h
                    speed_data = speed_data * 3.6
                
                metrics['avg_speed_kph'] = speed_data.mean()
                metrics['max_speed_kph'] = speed_data.max()
        
        # If no distance from GPS but we have speed, estimate distance
        if metrics['total_distance_km'] == 0.0 and metrics['avg_speed_kph'] > 0:
            duration_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
            metrics['total_distance_km'] = metrics['avg_speed_kph'] * duration_hours
        
        return metrics
    
    def _calculate_elevation_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate elevation gain metrics."""
        metrics = {
            'total_elevation_gain_m': 0.0
        }
        
        # Try multiple elevation columns
        elevation_col = None
        for col in ['altitude', 'elevation', 'enhanced_altitude']:
            if col in df.columns and not df[col].isna().all():
                elevation_col = col
                break
        
        if elevation_col is not None:
            elevation_data = df[elevation_col].dropna()
            if len(elevation_data) > 1:
                # Calculate positive elevation changes
                elevation_diff = elevation_data.diff()
                positive_gains = elevation_diff[elevation_diff > 0]
                metrics['total_elevation_gain_m'] = positive_gains.sum()
        
        return metrics
    
    def _calculate_cadence_metrics(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate cadence metrics."""
        metrics = {
            'avg_cadence': None,
            'max_cadence': None
        }
        
        if 'cadence' not in df.columns or df['cadence'].isna().all():
            return metrics
        
        cadence = df['cadence'].dropna()
        if cadence.empty:
            return metrics
        
        # Filter out unrealistic cadence values
        valid_cadence = cadence[
            (cadence >= self.config.processing.min_cadence) & 
            (cadence <= self.config.processing.max_cadence)
        ]
        
        if not valid_cadence.empty:
            metrics['avg_cadence'] = valid_cadence.mean()
            metrics['max_cadence'] = valid_cadence.max()
        
        return metrics
    
    def _calculate_environmental_metrics(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate environmental metrics."""
        metrics = {
            'avg_temperature_c': None
        }
        
        # Try multiple temperature columns
        temp_col = None
        for col in ['temperature', 'air_temperature', 'temp']:
            if col in df.columns and not df[col].isna().all():
                temp_col = col
                break
        
        if temp_col is not None:
            temp_data = df[temp_col].dropna()
            if not temp_data.empty:
                metrics['avg_temperature_c'] = temp_data.mean()
        
        return metrics
    
    def _calculate_interval_summary_metrics(self, intervals: List) -> Dict[str, object]:
        """Calculate summary metrics from detected intervals."""
        metrics = {
            'total_intervals_detected': len(intervals),
            'intervals_by_zone': {}
        }
        
        if not intervals:
            return metrics
        
        # Count intervals by power zone
        zone_counts = {}
        for interval in intervals:
            zone = getattr(interval, 'power_zone', 'unknown')
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        metrics['intervals_by_zone'] = zone_counts
        
        return metrics

def calculate_ride_metrics(ride_data: RideData, intervals: Optional[List] = None) -> RideMetrics:
    """
    Convenience function to calculate ride metrics.
    
    Args:
        ride_data: RideData object containing parsed FIT data
        intervals: Optional list of detected intervals
        
    Returns:
        RideMetrics object with calculated metrics
    """
    calculator = RideMetricsCalculator()
    return calculator.calculate_ride_metrics(ride_data, intervals)