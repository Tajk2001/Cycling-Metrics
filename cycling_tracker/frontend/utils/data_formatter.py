"""
Data Formatting Utilities
========================

Utility functions for formatting and processing data for display in dashboard components.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataFormatter:
    """
    Utility class for formatting cycling data for dashboard display.
    
    Provides methods for:
    - Time formatting (seconds to HH:MM:SS)
    - Power zone calculations
    - Metric aggregations
    - Display formatting
    """
    
    def __init__(self):
        """Initialize the data formatter."""
        self.power_zones = {
            1: {'name': 'Active Recovery', 'min_pct': 0, 'max_pct': 55, 'color': '#87CEEB'},
            2: {'name': 'Endurance', 'min_pct': 56, 'max_pct': 75, 'color': '#90EE90'},
            3: {'name': 'Tempo', 'min_pct': 76, 'max_pct': 90, 'color': '#FFD700'},
            4: {'name': 'Lactate Threshold', 'min_pct': 91, 'max_pct': 105, 'color': '#FFA500'},
            5: {'name': 'VO2 Max', 'min_pct': 106, 'max_pct': 120, 'color': '#FF6347'},
            6: {'name': 'Anaerobic Capacity', 'min_pct': 121, 'max_pct': 150, 'color': '#FF1493'},
            7: {'name': 'Neuromuscular Power', 'min_pct': 151, 'max_pct': 999, 'color': '#8A2BE2'}
        }
    
    def format_duration(self, seconds: float) -> str:
        """
        Format duration from seconds to HH:MM:SS format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if pd.isna(seconds) or seconds <= 0:
            return "0:00:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def format_power(self, power: float, include_unit: bool = True) -> str:
        """
        Format power value for display.
        
        Args:
            power: Power value in watts
            include_unit: Whether to include 'W' unit
            
        Returns:
            Formatted power string
        """
        if pd.isna(power):
            return "N/A"
        
        unit = "W" if include_unit else ""
        return f"{int(power)}{unit}"
    
    def format_distance(self, distance: float, unit: str = 'km') -> str:
        """
        Format distance for display.
        
        Args:
            distance: Distance value
            unit: Distance unit ('km' or 'mi')
            
        Returns:
            Formatted distance string
        """
        if pd.isna(distance):
            return "0.0"
        
        if unit == 'mi':
            distance = distance * 0.621371  # km to miles
        
        return f"{distance:.1f}"
    
    def format_speed(self, speed: float, unit: str = 'kmh') -> str:
        """
        Format speed for display.
        
        Args:
            speed: Speed value
            unit: Speed unit ('kmh' or 'mph')
            
        Returns:
            Formatted speed string
        """
        if pd.isna(speed):
            return "0.0"
        
        if unit == 'mph':
            speed = speed * 0.621371  # km/h to mph
            unit_suffix = " mph"
        else:
            unit_suffix = " km/h"
        
        return f"{speed:.1f}{unit_suffix}"
    
    def format_elevation(self, elevation: float, unit: str = 'm') -> str:
        """
        Format elevation for display.
        
        Args:
            elevation: Elevation value
            unit: Elevation unit ('m' or 'ft')
            
        Returns:
            Formatted elevation string
        """
        if pd.isna(elevation):
            return "0"
        
        if unit == 'ft':
            elevation = elevation * 3.28084  # meters to feet
            unit_suffix = "ft"
        else:
            unit_suffix = "m"
        
        return f"{int(elevation)}{unit_suffix}"
    
    def format_percentage(self, value: float, decimal_places: int = 1) -> str:
        """
        Format percentage for display.
        
        Args:
            value: Percentage value (0.85 for 85%)
            decimal_places: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        if pd.isna(value):
            return "0%"
        
        return f"{value * 100:.{decimal_places}f}%"
    
    def get_power_zone(self, power: float, ftp: float = 275) -> Dict[str, Any]:
        """
        Get power zone information for a given power value.
        
        Args:
            power: Power value in watts
            ftp: Functional Threshold Power
            
        Returns:
            Dictionary with zone information
        """
        if pd.isna(power) or power <= 0:
            return {'zone': 0, 'name': 'No Data', 'color': '#CCCCCC'}
        
        power_pct = (power / ftp) * 100
        
        for zone_num, zone_info in self.power_zones.items():
            if zone_info['min_pct'] <= power_pct <= zone_info['max_pct']:
                return {
                    'zone': zone_num,
                    'name': zone_info['name'],
                    'color': zone_info['color'],
                    'pct_ftp': power_pct
                }
        
        # If no zone matches, return zone 1
        return {
            'zone': 1,
            'name': self.power_zones[1]['name'],
            'color': self.power_zones[1]['color'],
            'pct_ftp': power_pct
        }
    
    def calculate_intensity_factor(self, avg_power: float, ftp: float = 275) -> float:
        """
        Calculate Intensity Factor (IF).
        
        Args:
            avg_power: Average power for the session
            ftp: Functional Threshold Power
            
        Returns:
            Intensity Factor value
        """
        if pd.isna(avg_power) or ftp <= 0:
            return 0.0
        
        return avg_power / ftp
    
    def calculate_tss(self, avg_power: float, duration_hours: float, ftp: float = 275) -> float:
        """
        Calculate Training Stress Score (TSS).
        
        Args:
            avg_power: Average power for the session
            duration_hours: Duration in hours
            ftp: Functional Threshold Power
            
        Returns:
            TSS value
        """
        if pd.isna(avg_power) or pd.isna(duration_hours) or ftp <= 0:
            return 0.0
        
        intensity_factor = self.calculate_intensity_factor(avg_power, ftp)
        return (duration_hours * intensity_factor ** 2) * 100
    
    def format_ride_summary(self, ride_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Format ride data for summary display.
        
        Args:
            ride_data: Dictionary containing ride metrics
            
        Returns:
            Dictionary with formatted display values
        """
        return {
            'distance': self.format_distance(ride_data.get('distance', 0)),
            'duration': self.format_duration(ride_data.get('duration', 0)),
            'avg_power': self.format_power(ride_data.get('avg_power', 0)),
            'max_power': self.format_power(ride_data.get('max_power', 0)),
            'elevation_gain': self.format_elevation(ride_data.get('elevation_gain', 0)),
            'avg_speed': self.format_speed(ride_data.get('avg_speed', 0)),
            'intensity_factor': f"{ride_data.get('intensity_factor', 0):.2f}",
            'tss': f"{ride_data.get('tss', 0):.0f}"
        }
    
    def format_interval_data(self, intervals: List[Dict[str, Any]], ftp: float = 275) -> List[Dict[str, Any]]:
        """
        Format interval data for display.
        
        Args:
            intervals: List of interval dictionaries
            ftp: Functional Threshold Power
            
        Returns:
            List of formatted interval data
        """
        formatted_intervals = []
        
        for i, interval in enumerate(intervals):
            power_zone = self.get_power_zone(interval.get('avg_power', 0), ftp)
            
            formatted_intervals.append({
                'id': i,
                'start_time': self.format_duration(interval.get('start_time', 0)),
                'duration': self.format_duration(interval.get('duration', 0)),
                'avg_power': self.format_power(interval.get('avg_power', 0)),
                'max_power': self.format_power(interval.get('max_power', 0)),
                'avg_hr': f"{int(interval.get('avg_hr', 0))} bpm" if interval.get('avg_hr') else "N/A",
                'intensity_factor': f"{self.calculate_intensity_factor(interval.get('avg_power', 0), ftp):.2f}",
                'power_zone': power_zone['zone'],
                'zone_name': power_zone['name'],
                'zone_color': power_zone['color']
            })
        
        return formatted_intervals
    
    def create_power_zone_colors(self) -> Dict[int, str]:
        """
        Get power zone colors for visualization.
        
        Returns:
            Dictionary mapping zone numbers to colors
        """
        return {zone: info['color'] for zone, info in self.power_zones.items()}
    
    def format_comparison_data(self, rides: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Format multiple rides for comparison display.
        
        Args:
            rides: List of ride data dictionaries
            
        Returns:
            Dictionary with formatted comparison data
        """
        comparison_data = {
            'ride_names': [],
            'distances': [],
            'durations': [],
            'avg_powers': [],
            'max_powers': [],
            'tss_values': []
        }
        
        for ride in rides:
            comparison_data['ride_names'].append(ride.get('name', 'Unnamed Ride'))
            comparison_data['distances'].append(ride.get('distance', 0))
            comparison_data['durations'].append(ride.get('duration', 0) / 3600)  # Convert to hours
            comparison_data['avg_powers'].append(ride.get('avg_power', 0))
            comparison_data['max_powers'].append(ride.get('max_power', 0))
            comparison_data['tss_values'].append(ride.get('tss', 0))
        
        return comparison_data