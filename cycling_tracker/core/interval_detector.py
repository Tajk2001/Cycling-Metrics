"""
Interval detection module for the cycling tracker system.
Implements multiple detection methods: lap-based, power-based, and ML-enhanced detection.
Based on SprintV1.py algorithms with enhancements for comprehensive ride analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings

from ..storage.data_models import RideData, IntervalMetrics, create_interval_id
from ..utils.config import get_config

class IntervalDetector:
    """
    Comprehensive interval detection system supporting multiple detection methods.
    """
    
    def __init__(self):
        self.config = get_config()
    
    def detect_intervals(self, ride_data: RideData) -> List[IntervalMetrics]:
        """
        Main interval detection method that uses multiple approaches.
        
        Args:
            ride_data: RideData object containing parsed FIT data
            
        Returns:
            List of IntervalMetrics objects
        """
        if not ride_data.has_power_data:
            print("⚠️ No power data available for interval detection")
            return []
        
        ftp = self.config.get_ftp_for_interval_detection()
        print(f"🔍 Starting interval detection with FTP: {ftp}W")
        
        all_intervals = []
        
        # 1. LAP-BASED DETECTION (Primary method)
        if ride_data.has_lap_data:
            print("🏁 Detecting intervals from lap data...")
            lap_intervals = self._detect_lap_intervals(ride_data, ftp)
            all_intervals.extend(lap_intervals)
            print(f"   Found {len(lap_intervals)} lap-based intervals")
        
        # 2. POWER-BASED DETECTION (Fallback or supplementary)
        if len(all_intervals) < 2:
            print("⚡ Using power-based interval detection as fallback...")
            power_intervals = self._detect_power_intervals(ride_data, ftp)
            if power_intervals:
                all_intervals = power_intervals  # Replace lap intervals if insufficient
                print(f"   Found {len(power_intervals)} power-based intervals")
        else:
            # Supplement with additional power-based intervals if not overlapping
            print("⚡ Detecting additional power-based intervals...")
            power_intervals = self._detect_power_intervals(ride_data, ftp)
            filtered_power = self._filter_overlapping_intervals(power_intervals, all_intervals)
            all_intervals.extend(filtered_power)
            print(f"   Added {len(filtered_power)} additional power-based intervals")
        
        # 3. INTERVAL SET IDENTIFICATION
        if all_intervals:
            print("🔄 Identifying repeating interval sets...")
            all_intervals = self._identify_interval_sets(all_intervals)
        
        # 4. QUALITY SCORING
        if all_intervals:
            print("⭐ Computing interval quality scores...")
            all_intervals = self._score_intervals(all_intervals, ride_data, ftp)
        
        # Sort by start time
        all_intervals.sort(key=lambda x: x.start_time)
        
        print(f"✅ Total intervals detected: {len(all_intervals)}")
        return all_intervals
    
    def _detect_lap_intervals(self, ride_data: RideData, ftp: int) -> List[IntervalMetrics]:
        """Detect intervals based on lap data."""
        if not ride_data.has_lap_data or ride_data.lap_dataframe is None:
            return []
        
        intervals = []
        lap_df = ride_data.lap_dataframe
        df = ride_data.dataframe
        
        for i, lap_row in lap_df.iterrows():
            if not pd.notna(lap_row.get('start_time')) or not pd.notna(lap_row.get('end_time')):
                continue
            
            start_time = lap_row['start_time']
            end_time = lap_row['end_time']
            duration = (end_time - start_time).total_seconds()
            
            # Skip very short laps (likely not intervals)
            if duration < self.config.detection.min_interval_duration:
                continue
            
            # Get lap data
            lap_data = df.loc[(df.index >= start_time) & (df.index <= end_time)]
            if lap_data.empty:
                continue
            
            # Check if this is a work interval (high power) or recovery
            avg_power = lap_data['power'].mean()
            power_threshold = ftp * self.config.detection.power_threshold_pct
            
            interval_type = 'work' if avg_power >= power_threshold else 'recovery'
            
            # Create interval metrics
            interval = self._create_interval_metrics(
                ride_data=ride_data,
                start_time=start_time,
                end_time=end_time,
                interval_number=i + 1,
                detection_source='lap',
                interval_type=interval_type
            )
            
            if interval:
                intervals.append(interval)
        
        return intervals
    
    def _detect_power_intervals(self, ride_data: RideData, ftp: int) -> List[IntervalMetrics]:
        """Detect intervals based on power thresholds."""
        df = ride_data.dataframe
        if 'power' not in df.columns:
            return []
        
        power_threshold = ftp * self.config.detection.power_threshold_pct
        min_duration = self.config.detection.min_interval_duration
        
        # Find periods above power threshold
        above_threshold = df['power'] >= power_threshold
        
        # Find start and end points of intervals
        intervals = []
        in_interval = False
        start_idx = None
        
        for idx in df.index:
            if above_threshold.loc[idx] and not in_interval:
                # Start of interval
                in_interval = True
                start_idx = idx
            elif not above_threshold.loc[idx] and in_interval:
                # End of interval
                in_interval = False
                if start_idx is not None:
                    duration = (idx - start_idx).total_seconds()
                    if duration >= min_duration:
                        intervals.append((start_idx, idx, duration))
                start_idx = None
        
        # Handle case where interval continues to end of data
        if in_interval and start_idx is not None:
            end_idx = df.index[-1]
            duration = (end_idx - start_idx).total_seconds()
            if duration >= min_duration:
                intervals.append((start_idx, end_idx, duration))
        
        # Create IntervalMetrics objects
        interval_metrics = []
        for i, (start_time, end_time, duration) in enumerate(intervals):
            interval = self._create_interval_metrics(
                ride_data=ride_data,
                start_time=start_time,
                end_time=end_time,
                interval_number=i + 1,
                detection_source='power_threshold',
                interval_type='work'
            )
            if interval:
                interval_metrics.append(interval)
        
        return interval_metrics
    
    def _create_interval_metrics(
        self,
        ride_data: RideData,
        start_time: datetime,
        end_time: datetime,
        interval_number: int,
        detection_source: str,
        interval_type: str
    ) -> Optional[IntervalMetrics]:
        """Create IntervalMetrics object from interval data."""
        df = ride_data.dataframe
        
        # Get interval data
        interval_data = df.loc[(df.index >= start_time) & (df.index <= end_time)]
        if interval_data.empty:
            return None
        
        duration = (end_time - start_time).total_seconds()
        
        # Create ride ID if not set
        if ride_data.ride_metrics and ride_data.ride_metrics.ride_id:
            ride_id = ride_data.ride_metrics.ride_id
        else:
            ride_id = f"ride_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        interval_id = create_interval_id(ride_id, interval_number)
        
        # Calculate power metrics
        power_metrics = self._calculate_power_metrics(interval_data)
        
        # Calculate heart rate metrics
        hr_metrics = self._calculate_heart_rate_metrics(interval_data)
        
        # Calculate speed/distance metrics
        speed_metrics = self._calculate_speed_metrics(interval_data)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(interval_data)
        
        # Determine power zone
        avg_power = power_metrics.get('avg_power', 0)
        power_zone = self._determine_power_zone(avg_power)
        
        # Create interval metrics object
        interval = IntervalMetrics(
            interval_id=interval_id,
            ride_id=ride_id,
            interval_number=interval_number,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=int(duration),
            detection_source=detection_source,
            interval_type=interval_type,
            power_zone=power_zone,
            **power_metrics,
            **hr_metrics,
            **speed_metrics,
            **quality_metrics
        )
        
        return interval
    
    def _calculate_power_metrics(self, interval_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate power-related metrics for an interval."""
        metrics = {}
        
        if 'power' not in interval_data.columns:
            return metrics
        
        power = interval_data['power'].dropna()
        if power.empty:
            return metrics
        
        metrics['avg_power_watts'] = power.mean()
        metrics['max_power_watts'] = power.max()
        metrics['min_power_watts'] = power.min()
        
        # Power per kg
        if self.config.rider.mass_kg > 0:
            metrics['avg_power_per_kg'] = metrics['avg_power_watts'] / self.config.rider.mass_kg
            metrics['max_power_per_kg'] = metrics['max_power_watts'] / self.config.rider.mass_kg
        
        # Coefficient of variation (consistency)
        if power.std() > 0 and power.mean() > 0:
            metrics['power_cv'] = (power.std() / power.mean()) * 100
        
        # Power fade (comparison of first and last quarters)
        if len(power) >= 4:
            quarter_size = len(power) // 4
            first_quarter_avg = power.iloc[:quarter_size].mean()
            last_quarter_avg = power.iloc[-quarter_size:].mean()
            if first_quarter_avg > 0:
                metrics['power_fade_pct'] = ((first_quarter_avg - last_quarter_avg) / first_quarter_avg) * 100
        
        # Normalized power (approximation using 30s rolling average)
        if len(power) > 30:
            normalized_power = (power.rolling(30, min_periods=30).mean() ** 4).mean() ** 0.25
            metrics['normalized_power_watts'] = normalized_power
        else:
            metrics['normalized_power_watts'] = metrics['avg_power_watts']
        
        # Work (kJ)
        metrics['work_kj'] = metrics['avg_power_watts'] * len(interval_data) / 1000
        
        return metrics
    
    def _calculate_heart_rate_metrics(self, interval_data: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate heart rate metrics for an interval."""
        metrics = {
            'avg_heart_rate': None,
            'max_heart_rate': None,
            'min_heart_rate': None,
            'heart_rate_cv': None,
            'heart_rate_fade_pct': None
        }
        
        if 'heart_rate' not in interval_data.columns:
            return metrics
        
        hr = interval_data['heart_rate'].dropna()
        if hr.empty:
            return metrics
        
        metrics['avg_heart_rate'] = int(hr.mean())
        metrics['max_heart_rate'] = int(hr.max())
        metrics['min_heart_rate'] = int(hr.min())
        
        # Heart rate CV
        if hr.std() > 0 and hr.mean() > 0:
            metrics['heart_rate_cv'] = (hr.std() / hr.mean()) * 100
        
        # Heart rate drift (fade)
        if len(hr) >= 4:
            quarter_size = len(hr) // 4
            first_quarter_avg = hr.iloc[:quarter_size].mean()
            last_quarter_avg = hr.iloc[-quarter_size:].mean()
            if first_quarter_avg > 0:
                metrics['heart_rate_fade_pct'] = ((last_quarter_avg - first_quarter_avg) / first_quarter_avg) * 100
        
        return metrics
    
    def _calculate_speed_metrics(self, interval_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate speed and distance metrics for an interval."""
        metrics = {
            'avg_speed_kph': 0.0,
            'max_speed_kph': 0.0,
            'distance_km': 0.0,
            'avg_cadence': None,
            'max_cadence': None
        }
        
        # Speed metrics
        if 'speed_kph' in interval_data.columns:
            speed = interval_data['speed_kph'].dropna()
            if not speed.empty:
                metrics['avg_speed_kph'] = speed.mean()
                metrics['max_speed_kph'] = speed.max()
        elif 'enhanced_speed_kph' in interval_data.columns:
            speed = interval_data['enhanced_speed_kph'].dropna()
            if not speed.empty:
                metrics['avg_speed_kph'] = speed.mean()
                metrics['max_speed_kph'] = speed.max()
        
        # Distance calculation
        if 'distance' in interval_data.columns:
            if not interval_data['distance'].empty:
                distance_start = interval_data['distance'].iloc[0]
                distance_end = interval_data['distance'].iloc[-1]
                metrics['distance_km'] = (distance_end - distance_start) / 1000
        elif metrics['avg_speed_kph'] > 0:
            # Estimate distance from speed
            duration_hours = len(interval_data) / 3600  # Assuming 1-second sampling
            metrics['distance_km'] = metrics['avg_speed_kph'] * duration_hours
        
        # Cadence metrics
        if 'cadence' in interval_data.columns:
            cadence = interval_data['cadence'].dropna()
            if not cadence.empty:
                metrics['avg_cadence'] = cadence.mean()
                metrics['max_cadence'] = cadence.max()
        
        return metrics
    
    def _calculate_quality_metrics(self, interval_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality and consistency scores for an interval."""
        metrics = {
            'quality_score': 0.0,
            'consistency_score': 0.0
        }
        
        if 'power' not in interval_data.columns:
            return metrics
        
        power = interval_data['power'].dropna()
        if power.empty:
            return metrics
        
        # Quality score based on power consistency and level
        quality_factors = []
        
        # Power consistency (lower CV = better quality)
        if power.std() > 0 and power.mean() > 0:
            power_cv = (power.std() / power.mean()) * 100
            consistency_score = max(0, 100 - power_cv * 5)  # Scale CV to 0-100
            quality_factors.append(consistency_score * 0.4)
            metrics['consistency_score'] = consistency_score
        
        # Power level relative to FTP
        try:
            ftp = self.config.get_ftp_for_interval_detection()
            power_ratio = power.mean() / ftp
            power_score = min(100, power_ratio * 50)  # Scale to 0-100
            quality_factors.append(power_score * 0.3)
        except ValueError:
            pass
        
        # Duration factor (longer intervals get slight quality bonus)
        duration_minutes = len(interval_data) / 60
        duration_score = min(100, 50 + duration_minutes * 2)
        quality_factors.append(duration_score * 0.3)
        
        # Overall quality score
        if quality_factors:
            metrics['quality_score'] = sum(quality_factors)
        
        return metrics
    
    def _determine_power_zone(self, avg_power: float) -> str:
        """Determine power zone based on average power and FTP."""
        try:
            power_zones = self.config.get_power_zones()
            
            for zone_name, zone_info in power_zones.items():
                if zone_info['min'] <= avg_power <= zone_info['max']:
                    return zone_name
            
            return 'zone7'  # Default to highest zone if above all thresholds
        except ValueError:
            return 'unknown'
    
    def _filter_overlapping_intervals(
        self, 
        new_intervals: List[IntervalMetrics], 
        existing_intervals: List[IntervalMetrics]
    ) -> List[IntervalMetrics]:
        """Filter out intervals that overlap with existing ones."""
        overlap_threshold = self.config.detection.overlap_threshold
        filtered = []
        
        for new_interval in new_intervals:
            is_overlapping = False
            
            for existing in existing_intervals:
                # Check for time overlap
                time_overlap = max(0, (
                    min(new_interval.end_time, existing.end_time) - 
                    max(new_interval.start_time, existing.start_time)
                ).total_seconds())
                
                if time_overlap >= overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered.append(new_interval)
        
        return filtered
    
    def _identify_interval_sets(self, intervals: List[IntervalMetrics]) -> List[IntervalMetrics]:
        """Identify and mark intervals that are part of repeating sets."""
        if len(intervals) < 2:
            return intervals
        
        # Group intervals by similar duration and power
        interval_groups = {}
        
        for interval in intervals:
            # Create grouping key based on duration (±20%) and power zone
            duration_key = round(interval.duration_seconds / 60)  # Round to nearest minute
            power_key = interval.power_zone
            group_key = f"{power_key}_{duration_key}min"
            
            if group_key not in interval_groups:
                interval_groups[group_key] = []
            interval_groups[group_key].append(interval)
        
        # Mark sets with 2 or more intervals
        set_counter = 1
        for group_key, group_intervals in interval_groups.items():
            if len(group_intervals) >= 2:
                # Sort by start time
                group_intervals.sort(key=lambda x: x.start_time)
                
                # Create set ID
                set_id = f"SET_{set_counter:03d}_{group_key}"
                
                # Mark all intervals in this set
                for i, interval in enumerate(group_intervals):
                    interval.is_part_of_set = True
                    interval.set_id = set_id
                    interval.set_position = i + 1
                    interval.set_total_intervals = len(group_intervals)
                
                set_counter += 1
        
        return intervals
    
    def _score_intervals(
        self, 
        intervals: List[IntervalMetrics], 
        ride_data: RideData, 
        ftp: int
    ) -> List[IntervalMetrics]:
        """Calculate relative scores and additional metrics for intervals."""
        if not intervals:
            return intervals
        
        # Calculate ride-level averages for relative scoring
        df = ride_data.dataframe
        if 'power' in df.columns:
            ride_avg_power = df['power'].mean()
        else:
            ride_avg_power = 0
        
        # Calculate recovery metrics and relative scores
        sorted_intervals = sorted(intervals, key=lambda x: x.start_time)
        
        for i, interval in enumerate(sorted_intervals):
            # Power relative to ride average
            if ride_avg_power > 0:
                interval.power_relative_to_ride_avg = interval.avg_power_watts / ride_avg_power
            
            # Power relative to FTP
            if ftp > 0:
                interval.power_relative_to_ftp = interval.avg_power_watts / ftp
            
            # Recovery time and power ratio (for next interval)
            if i < len(sorted_intervals) - 1:
                next_interval = sorted_intervals[i + 1]
                recovery_time = (next_interval.start_time - interval.end_time).total_seconds()
                interval.recovery_time_after_seconds = int(max(0, recovery_time))
                
                if interval.avg_power_watts > 0:
                    interval.recovery_power_ratio = next_interval.avg_power_watts / interval.avg_power_watts
        
        return intervals

def detect_intervals(ride_data: RideData) -> List[IntervalMetrics]:
    """
    Convenience function to detect intervals from ride data.
    
    Args:
        ride_data: RideData object containing parsed FIT data
        
    Returns:
        List of IntervalMetrics objects
    """
    detector = IntervalDetector()
    return detector.detect_intervals(ride_data)