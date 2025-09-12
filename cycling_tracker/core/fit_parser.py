"""
FIT file parser for the cycling tracker system.
Based on SprintV1.py implementation with enhancements for comprehensive ride analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import warnings
from pathlib import Path

# Import fitparse with error handling
try:
    from fitparse import FitFile
except ImportError:
    FitFile = None
    warnings.warn("fitparse not available. Install with: pip install fitparse")

from ..storage.data_models import RideData, create_ride_id
from ..utils.config import get_config

class FITParser:
    """
    FIT file parser that extracts comprehensive ride data including laps.
    """
    
    def __init__(self):
        self.config = get_config()
        
        if FitFile is None:
            raise ImportError("fitparse is not installed. Please install via `pip install fitparse`.")
    
    def parse_fit_file(self, file_path: str) -> RideData:
        """
        Parse a FIT file and return structured ride data.
        
        Args:
            file_path: Path to the FIT file
            
        Returns:
            RideData object containing parsed data and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"FIT file not found: {file_path}")
        
        print(f"📁 Parsing FIT file: {file_path.name}")
        
        # Load FIT file
        fitfile = FitFile(str(file_path))
        
        # Extract records and laps
        df = self._extract_records(fitfile)
        lap_df = self._extract_laps(fitfile)
        
        # Add lap information to records
        if not lap_df.empty:
            df = self._associate_laps_with_records(df, lap_df)
        
        # Apply data cleaning and processing
        df = self._process_data(df)
        
        # Create ride data object
        ride_data = RideData(
            dataframe=df,
            lap_dataframe=lap_df if not lap_df.empty else None,
            filename=file_path.name,
            file_path=str(file_path),
            has_power_data='power' in df.columns and not df['power'].isna().all(),
            has_heart_rate_data='heart_rate' in df.columns and not df['heart_rate'].isna().all(),
            has_lap_data=not lap_df.empty
        )
        
        print(f"✅ Successfully parsed {len(df)} records")
        print(f"   • Power data: {'✓' if ride_data.has_power_data else '✗'}")
        print(f"   • Heart rate data: {'✓' if ride_data.has_heart_rate_data else '✗'}")
        print(f"   • Lap data: {'✓' if ride_data.has_lap_data else '✗'} ({len(lap_df) if not lap_df.empty else 0} laps)")
        
        return ride_data
    
    def _extract_records(self, fitfile: FitFile) -> pd.DataFrame:
        """Extract record messages from FIT file."""
        records = []
        
        for msg in fitfile.get_messages('record'):
            row = {}
            for field in msg:
                row[field.name] = field.value
            records.append(row)
        
        if not records:
            raise ValueError("No record messages found in FIT file")
        
        df = pd.DataFrame(records)
        
        # Handle timestamp and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            raise ValueError("No timestamp data found in FIT file")
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        return df
    
    def _extract_laps(self, fitfile: FitFile) -> pd.DataFrame:
        """Extract lap messages from FIT file."""
        laps = []
        
        for lap_msg in fitfile.get_messages('lap'):
            lap_data = {}
            for field in lap_msg:
                lap_data[field.name] = field.value
            
            # Process lap timing
            if 'start_time' in lap_data:
                lap_data['start_time'] = pd.to_datetime(lap_data['start_time'])
            
            if 'timestamp' in lap_data:
                lap_data['end_time'] = pd.to_datetime(lap_data['timestamp'])
            elif 'total_elapsed_time' in lap_data and 'start_time' in lap_data:
                lap_data['end_time'] = lap_data['start_time'] + pd.to_timedelta(
                    lap_data.get('total_elapsed_time', 0), unit='s'
                )
            
            laps.append(lap_data)
        
        if not laps:
            return pd.DataFrame()
        
        lap_df = pd.DataFrame(laps)
        return lap_df
    
    def _associate_laps_with_records(self, df: pd.DataFrame, lap_df: pd.DataFrame) -> pd.DataFrame:
        """Associate lap information with record data."""
        if lap_df.empty or not {'start_time', 'end_time'}.issubset(lap_df.columns):
            df['lap'] = 1
            return df
        
        df['lap'] = np.nan
        
        for i, lap_row in lap_df.iterrows():
            start_time = lap_row['start_time']
            end_time = lap_row['end_time']
            
            if pd.notna(start_time) and pd.notna(end_time):
                # Find records within this lap's time range
                mask = (df.index >= start_time) & (df.index <= end_time)
                df.loc[mask, 'lap'] = i + 1
        
        # Handle any records without lap assignment
        if df['lap'].isna().all():
            df['lap'] = 1
        else:
            # Forward fill and backward fill to assign all records to a lap
            df['lap'] = df['lap'].ffill().bfill()
        
        return df
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data processing, cleaning, and derived field calculation."""
        # Apply data limits and smoothing
        df = self._apply_data_limits(df)
        
        # Calculate derived fields
        df = self._calculate_derived_fields(df)
        
        # Calculate enhanced speed if position data available
        df = self._calculate_enhanced_speed(df)
        
        return df
    
    def _apply_data_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply hard limits and data cleaning."""
        # Power limits
        if 'power' in df.columns:
            df.loc[df['power'] < 0, 'power'] = 0
            df.loc[df['power'] > 3000, 'power'] = np.nan  # Reasonable upper limit
        
        # Heart rate limits
        if 'heart_rate' in df.columns:
            df.loc[df['heart_rate'] < 40, 'heart_rate'] = np.nan
            df.loc[df['heart_rate'] > 220, 'heart_rate'] = np.nan
        
        # Cadence limits
        if 'cadence' in df.columns:
            df.loc[df['cadence'] < 0, 'cadence'] = np.nan
            df.loc[df['cadence'] > 200, 'cadence'] = np.nan
        
        # Speed limits
        if 'speed' in df.columns:
            df.loc[df['speed'] < 0, 'speed'] = 0
            df.loc[df['speed'] > 30, 'speed'] = np.nan  # 30 m/s = 108 km/h
        
        return df
    
    def _calculate_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived fields from raw data."""
        # Calculate torque if power and cadence are available
        if 'power' in df.columns and 'cadence' in df.columns:
            # Torque = Power / (Cadence * 2π / 60)
            cadence_rad_s = df['cadence'] * 2 * np.pi / 60
            df['torque'] = df['power'] / cadence_rad_s
            df.loc[df['cadence'] <= 0, 'torque'] = np.nan
        
        # Convert speed from m/s to km/h if needed
        if 'speed' in df.columns:
            if df['speed'].max() < 50:  # Assume m/s if max < 50
                df['speed_kph'] = df['speed'] * 3.6
            else:
                df['speed_kph'] = df['speed']  # Already in km/h
        
        # Calculate power-to-weight ratio
        if 'power' in df.columns and self.config.rider.mass_kg > 0:
            df['power_per_kg'] = df['power'] / self.config.rider.mass_kg
        
        # Calculate grade if elevation and distance data available
        if 'altitude' in df.columns and 'distance' in df.columns:
            df['grade'] = self._calculate_grade(df['altitude'], df['distance'])
        
        return df
    
    def _calculate_enhanced_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced speed from position data if available."""
        if 'position_lat' in df.columns and 'position_long' in df.columns:
            # Calculate distance between consecutive points
            distances = self._calculate_distances(
                df['position_lat'].values,
                df['position_long'].values
            )
            
            # Calculate time intervals
            time_diffs = df.index.to_series().diff().dt.total_seconds().values[1:]
            
            # Calculate speeds (m/s)
            speeds = np.zeros(len(df))
            speeds[1:] = distances / np.maximum(time_diffs, 0.1)  # Avoid division by zero
            
            df['enhanced_speed'] = speeds
            df['enhanced_speed_kph'] = speeds * 3.6
        
        return df
    
    def _calculate_grade(self, altitude: pd.Series, distance: pd.Series) -> pd.Series:
        """Calculate grade percentage from altitude and distance data."""
        elevation_diff = altitude.diff()
        distance_diff = distance.diff()
        
        grade = np.zeros(len(altitude))
        mask = distance_diff > 0
        grade[mask] = (elevation_diff[mask] / distance_diff[mask]) * 100
        
        # Smooth grade data to remove noise
        grade_series = pd.Series(grade, index=altitude.index)
        return grade_series.rolling(window=10, min_periods=1).mean()
    
    def _calculate_distances(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Calculate distances between consecutive GPS points using Haversine formula."""
        if len(lats) < 2:
            return np.array([])
        
        # Convert to radians
        lat1 = np.radians(lats[:-1])
        lon1 = np.radians(lons[:-1])
        lat2 = np.radians(lats[1:])
        lon2 = np.radians(lons[1:])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in meters
        R = 6371000
        distances = R * c
        
        return distances

def parse_fit_file(file_path: str) -> RideData:
    """
    Convenience function to parse a FIT file.
    
    Args:
        file_path: Path to the FIT file
        
    Returns:
        RideData object containing parsed data
    """
    parser = FITParser()
    return parser.parse_fit_file(file_path)