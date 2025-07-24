#!/usr/bin/env python3
"""
Enhanced Cycling Analysis Script
Comprehensive analysis of cycling FIT files with personalized metrics and visualizations.

This module provides:
- Advanced data processing pipeline
- Personalized metrics calculation
- Comprehensive visualization generation
- Critical power and W' balance analysis
- Heat stress and fatigue pattern analysis

Author: Cycling Analysis Team
Version: 1.0.0
"""

# Standard library imports
import os
import json
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy import stats
from scipy.signal import savgol_filter
import seaborn as sns
import fitparse
import gpxpy
import gpxpy.gpx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONSTANTS ---
SMOOTHING_WINDOW = 30
MOVING_AVG_WINDOW = 30
POWER_OUTLIER_THRESHOLD = 5.0
HR_OUTLIER_THRESHOLD = 3.0
CADENCE_OUTLIER_THRESHOLD = 3.0
SIGNIFICANT_POWER_CHANGE = 20
FORWARD_FILL_LIMIT = 5

class DataQualityLevel(Enum):
    """Enum for data quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class DataQualityReport:
    """Data structure for data quality assessment results."""
    overall_quality: DataQualityLevel
    missing_data_percentage: float
    outlier_percentage: float
    sampling_rate_consistency: bool
    required_fields_present: Dict[str, bool]
    data_gaps: List[Tuple[float, float]]
    recommendations: List[str]

# Set style for modern, minimalist plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class CyclingAnalyzer:
    """
    Main class for cycling data analysis with personalized configuration.
    
    This class provides comprehensive analysis capabilities including:
    - Data ingestion and cleaning
    - Metrics calculation
    - Visualization generation
    - Advanced modeling (CP, W' balance)
    - Performance analysis
    
    Use set_lactate_coeffs(a, b, c) to personalize the lactate-power curve for your physiology.
    """
    
    def __init__(self, save_figures: bool = True, ftp: int = 250, max_hr: int = 195, 
                 rest_hr: int = 51, weight_kg: float = 70, height_cm: float = 175, 
                 athlete_name: str = "Cyclist", save_dir: str = "figures", 
                 analysis_id: Optional[str] = None):
        """
        Initialize the Cycling Analyzer with comprehensive data processing capabilities.
        
        Args:
            save_figures: Whether to save generated figures
            ftp: Functional Threshold Power in watts
            max_hr: Maximum heart rate
            rest_hr: Resting heart rate
            weight_kg: Athlete weight in kg
            height_cm: Athlete height in cm
            athlete_name: Athlete name
            save_dir: Directory to save figures
            analysis_id: Unique identifier for this analysis
        """
        self.save_figures = save_figures
        self.ftp = ftp
        self.max_hr = max_hr
        self.rest_hr = rest_hr
        self.weight_kg = weight_kg
        self.height_cm = height_cm
        self.athlete_name = athlete_name
        self.save_dir = save_dir
        self.analysis_id = analysis_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize data structures
        self.df = None  # Main dataframe
        self.raw_data = None
        self.clean_data = None
        self.processed_data = None
        self.quality_report = None
        self.metrics = {}
        self.power_bests = {}
        
        # W' balance parameters (Skiba et al., 2012)
        self.w_prime_tau = 228  # seconds, recovery time constant
        
        # Create save directory if it doesn't exist
        if save_figures and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Define zone colors for visualization
        self.zone_colors = {
            'Z1 (Active Recovery)': '#4CAF50',
            'Z2 (Endurance)': '#8BC34A', 
            'Z3 (Tempo)': '#FFEB3B',
            'Z4 (Threshold)': '#FF9800',
            'Z5 (VO2 Max)': '#F44336',
            'Z6 (Anaerobic Capacity)': '#9C27B0',
            'Z7 (Neuromuscular Power)': '#E91E63'
        }
        
        # Initialize multi-ride attributes for compatibility
        self.multi_ride_zones = []
        self.multi_ride_labels = []
        
        print(f"🚴 Cycling Analyzer initialized for {athlete_name}")
        print(f"📊 FTP: {ftp}W, Max HR: {max_hr}, Weight: {weight_kg}kg")
        print(f"💾 Analysis ID: {self.analysis_id}")
    
    def process_activity_data(self, file_path: str) -> bool:
        """
        Main data processing pipeline following the logical flow
        
        Args:
            file_path (str): Path to the activity file (FIT, TCX, CSV)
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        print("\n" + "="*80)
        print("🚴 CYCLING ANALYSIS PIPELINE")
        print("="*80)
        
        try:
            # Step 1: 📥 DATA INGESTION
            print("\n📥 STEP 1: DATA INGESTION")
            print("-" * 40)
            if not self._ingest_data(file_path):
                return False
            
            # Step 2: 🔍 INITIAL DATA CHECKS
            print("\n🔍 STEP 2: INITIAL DATA CHECKS")
            print("-" * 40)
            self._perform_initial_checks()
            
            # Step 3: 🧹 DATA CLEANING
            print("\n🧹 STEP 3: DATA CLEANING")
            print("-" * 40)
            self._clean_data()
            
            # Step 4: 🧮 FEATURE ENGINEERING
            print("\n🧮 STEP 4: FEATURE ENGINEERING")
            print("-" * 40)
            self._engineer_features()
            
            # Step 5: 📊 DATA AGGREGATION
            print("\n📊 STEP 5: DATA AGGREGATION")
            print("-" * 40)
            self._aggregate_data()
            
            # Step 6: 📈 VISUALIZATION
            print("\n📈 STEP 6: VISUALIZATION")
            print("-" * 40)
            self._create_visualizations()
            
            # Step 7: 🧪 MODELING & INTERPRETATION
            print("\n🧪 STEP 7: MODELING & INTERPRETATION")
            print("-" * 40)
            self._perform_modeling()
            
            print("\n✅ Data processing pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error in data processing pipeline: {str(e)}")
            return False
    
    def _ingest_data(self, file_path: str) -> bool:
        """
        Step 1: Data Ingestion - Import FIT/TCX/CSV file(s)
        
        Args:
            file_path (str): Path to the activity file
            
        Returns:
            bool: True if ingestion successful
        """
        print(f"📁 Loading file: {file_path}")
        
        try:
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'fit':
                self.raw_data = self._load_fit_file(file_path)
            elif file_extension == 'tcx':
                self.raw_data = self._load_tcx_file(file_path)
            elif file_extension == 'csv':
                self.raw_data = self._load_csv_file(file_path)
            else:
                print(f"❌ Unsupported file format: {file_extension}")
                return False
            
            if self.raw_data is None or self.raw_data.empty:
                print("❌ No data loaded from file")
                return False
            
            print(f"✅ Successfully loaded {len(self.raw_data)} data points")
            print(f"📊 Data columns: {list(self.raw_data.columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Error during data ingestion: {str(e)}")
            return False
    
    def _load_fit_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from FIT file
        
        Args:
            file_path (str): Path to FIT file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            fitfile = fitparse.FitFile(file_path)
            
            # Extract data from FIT file
            data = []
            for record in fitfile.get_messages('record'):
                record_data = {}
                
                # Get timestamp
                if record.get_value('timestamp'):
                    record_data['timestamp'] = record.get_value('timestamp')
                
                # Get power data
                if record.get_value('power'):
                    record_data['power'] = record.get_value('power')
                
                # Get heart rate data
                if record.get_value('heart_rate'):
                    record_data['heart_rate'] = record.get_value('heart_rate')
                
                # Get cadence data
                if record.get_value('cadence'):
                    record_data['cadence'] = record.get_value('cadence')
                
                # Get speed data
                if record.get_value('speed'):
                    record_data['speed'] = record.get_value('speed') * 3.6  # Convert m/s to km/h
                
                # Get elevation data
                if record.get_value('altitude'):
                    record_data['elevation'] = record.get_value('altitude')
                
                # Get distance data
                if record.get_value('distance'):
                    record_data['distance'] = record.get_value('distance') / 1000  # Convert to km
                
                if record_data:  # Only add if we have some data
                    data.append(record_data)
            
            if not data:
                print("❌ No valid data found in FIT file")
                return None
            
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading FIT file: {str(e)}")
            return None
    
    def _load_tcx_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from TCX file
        
        Args:
            file_path (str): Path to TCX file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            with open(file_path, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)
            
            data = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        point_data = {}
                        
                        # Get timestamp
                        if point.time:
                            point_data['timestamp'] = point.time
                        
                        # Get elevation
                        if point.elevation:
                            point_data['elevation'] = point.elevation
                        
                        # Get speed (if available)
                        if point.speed:
                            point_data['speed'] = point.speed * 3.6  # Convert m/s to km/h
                        
                        # Get heart rate (if available in extensions)
                        if point.extensions:
                            for extension in point.extensions:
                                if 'heartrate' in extension.tag.lower():
                                    point_data['heart_rate'] = float(extension.text)
                        
                        if point_data:
                            data.append(point_data)
            
            if not data:
                print("❌ No valid data found in TCX file")
                return None
            
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading TCX file: {str(e)}")
            return None
    
    def _load_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(file_path)
            
            # Try to identify timestamp column
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if timestamp_cols:
                df['timestamp'] = pd.to_datetime(df[timestamp_cols[0]])
            
            # Try to identify other columns
            column_mapping = {
                'power': ['power', 'watts', 'w'],
                'heart_rate': ['heart_rate', 'hr', 'heartrate', 'bpm'],
                'cadence': ['cadence', 'rpm'],
                'speed': ['speed', 'velocity', 'kmh', 'mph'],
                'elevation': ['elevation', 'altitude', 'alt'],
                'distance': ['distance', 'dist']
            }
            
            for target_col, possible_names in column_mapping.items():
                for col in df.columns:
                    if col.lower() in possible_names:
                        df[target_col] = df[col]
                        break
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading CSV file: {str(e)}")
            return None
    
    def _perform_initial_checks(self):
        """
        Step 2: Initial Data Checks - Verify required fields and data quality
        """
        print("🔍 Performing initial data quality checks...")
        
        # Define required fields
        required_fields = {
            'timestamp': 'Time',
            'power': 'Power', 
            'heart_rate': 'Heart Rate',
            'cadence': 'Cadence',
            'speed': 'Speed',
            'elevation': 'Elevation'
        }
        
        # Check field presence
        field_status = {}
        for field, display_name in required_fields.items():
            field_status[display_name] = field in self.raw_data.columns
            status = "✅" if field_status[display_name] else "❌"
            print(f"   {status} {display_name}")
        
        # Check data quality
        missing_data = self.raw_data.isnull().sum()
        total_points = len(self.raw_data)
        
        print(f"\n📊 Data Quality Summary:")
        print(f"   Total data points: {total_points}")
        print(f"   Missing data percentage: {(missing_data.sum() / (total_points * len(self.raw_data.columns))) * 100:.1f}%")
        
        # Check for zero-duration segments
        if 'timestamp' in self.raw_data.columns:
            time_diffs = self.raw_data['timestamp'].diff().dt.total_seconds()
            zero_duration_segments = (time_diffs == 0).sum()
            print(f"   Zero-duration segments: {zero_duration_segments}")
        
        # Create quality report
        missing_percentage = (missing_data.sum() / (total_points * len(self.raw_data.columns))) * 100
        
        if missing_percentage < 5 and all(field_status.values()):
            quality_level = DataQualityLevel.EXCELLENT
        elif missing_percentage < 15 and sum(field_status.values()) >= 4:
            quality_level = DataQualityLevel.GOOD
        elif missing_percentage < 30 and sum(field_status.values()) >= 3:
            quality_level = DataQualityLevel.FAIR
        else:
            quality_level = DataQualityLevel.POOR
        
        self.quality_report = DataQualityReport(
            overall_quality=quality_level,
            missing_data_percentage=missing_percentage,
            outlier_percentage=0,  # Will be calculated in cleaning step
            sampling_rate_consistency=True,  # Will be checked in cleaning step
            required_fields_present=field_status,
            data_gaps=[],
            recommendations=[]
        )
        
        print(f"   Overall quality: {quality_level.value.upper()}")
    
    def _clean_data(self):
        """
        Step 3: Data Cleaning - Handle missing values, outliers, and artifacts
        """
        print("🧹 Cleaning and preprocessing data...")
        
        # Create a copy for cleaning
        self.clean_data = self.raw_data.copy()
        
        # Handle missing values
        print("   📝 Handling missing values...")
        for column in self.clean_data.columns:
            if column == 'timestamp':
                continue  # Don't interpolate timestamps
            
            missing_count = self.clean_data[column].isnull().sum()
            if missing_count > 0:
                print(f"     {column}: {missing_count} missing values")
                
                # Use forward fill for small gaps, interpolation for larger gaps
                if missing_count < len(self.clean_data) * 0.1:  # Less than 10% missing
                    self.clean_data[column] = self.clean_data[column].fillna(method='ffill').fillna(method='bfill')
                else:
                    self.clean_data[column] = self.clean_data[column].interpolate(method='linear')
        
        # Remove zero-duration or artifact segments
        print("   🗑️ Removing artifacts...")
        if 'timestamp' in self.clean_data.columns:
            time_diffs = self.clean_data['timestamp'].diff().dt.total_seconds()
            artifact_mask = (time_diffs == 0) | (time_diffs > 60)  # Gaps > 60 seconds
            artifacts_removed = artifact_mask.sum()
            self.clean_data = self.clean_data[~artifact_mask]
            print(f"     Removed {artifacts_removed} artifact segments")
        
        # Smooth extreme outliers
        print("   📊 Smoothing outliers...")
        numeric_columns = self.clean_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['power', 'heart_rate', 'cadence', 'speed']:
                # Use IQR method to detect outliers
                Q1 = self.clean_data[column].quantile(0.25)
                Q3 = self.clean_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.clean_data[column] < lower_bound) | 
                           (self.clean_data[column] > upper_bound)).sum()
                
                if outliers > 0:
                    print(f"     {column}: {outliers} outliers detected")
                    # Replace outliers with rolling median
                    outlier_mask = (self.clean_data[column] < lower_bound) | (self.clean_data[column] > upper_bound)
                    self.clean_data.loc[outlier_mask, column] = self.clean_data[column].rolling(window=5, center=True, min_periods=1).median()
        
        # Resample to consistent 1Hz if necessary
        print("   ⏱️ Resampling to consistent rate...")
        if 'timestamp' in self.clean_data.columns:
            # Check if we need resampling
            time_diffs = self.clean_data['timestamp'].diff().dt.total_seconds()
            unique_intervals = time_diffs.value_counts()
            
            if len(unique_intervals) > 1:
                print(f"     Inconsistent sampling detected: {unique_intervals.index.tolist()}")
                # Resample to 1Hz
                self.clean_data = self.clean_data.set_index('timestamp').resample('1S').mean().reset_index()
                print("     Resampled to 1Hz")
            else:
                print(f"     Consistent sampling rate: {unique_intervals.index[0]}s")
        
        print(f"✅ Data cleaning completed. Clean dataset: {len(self.clean_data)} points")
    
    def _engineer_features(self):
        """
        Step 4: Feature Engineering - Calculate core and derived metrics
        """
        print("🧮 Engineering features and calculating metrics...")
        
        # Create processed data copy
        self.processed_data = self.clean_data.copy()
        
        # Core metrics calculation
        print("   📊 Calculating core metrics...")
        
        # Power metrics
        if 'power' in self.processed_data.columns:
            self.avg_power = self.processed_data['power'].mean()
            self.max_power = self.processed_data['power'].max()
            
            # Normalized Power calculation
            power_series = self.processed_data['power'].fillna(0)
            # 30-second rolling average
            rolling_30s = power_series.rolling(window=30, center=True, min_periods=1).mean()
            # 4th power average
            power_4th = rolling_30s ** 4
            avg_power_4th = power_4th.mean()
            self.np_calc = avg_power_4th ** 0.25
            
            # Intensity Factor
            self.IF = self.np_calc / self.ftp if self.ftp > 0 else 0
            
            # Variability Index
            self.VI = self.np_calc / self.avg_power if self.avg_power > 0 else 0
            
            # Training Stress Score
            duration_hours = len(self.processed_data) / 3600  # Assuming 1Hz data
            self.TSS = (duration_hours * self.np_calc * self.IF) / self.ftp if self.ftp > 0 else 0
            
            print(f"     Avg Power: {self.avg_power:.0f}W")
            print(f"     Max Power: {self.max_power:.0f}W")
            print(f"     Normalized Power: {self.np_calc:.0f}W")
            print(f"     Intensity Factor: {self.IF:.2f}")
            print(f"     Variability Index: {self.VI:.2f}")
            print(f"     Training Stress Score: {self.TSS:.0f}")
        
        # Heart Rate metrics
        if 'heart_rate' in self.processed_data.columns:
            self.avg_hr = self.processed_data['heart_rate'].mean()
            self.max_hr = self.processed_data['heart_rate'].max()
            print(f"     Avg HR: {self.avg_hr:.0f} bpm")
            print(f"     Max HR: {self.max_hr:.0f} bpm")
        
        # Cadence metrics
        if 'cadence' in self.processed_data.columns:
            self.avg_cadence = self.processed_data['cadence'].mean()
            self.max_cadence = self.processed_data['cadence'].max()
            print(f"     Avg Cadence: {self.avg_cadence:.0f} rpm")
            print(f"     Max Cadence: {self.max_cadence:.0f} rpm")
        
        # Speed metrics
        if 'speed' in self.processed_data.columns:
            self.avg_speed = self.processed_data['speed'].mean()
            self.max_speed = self.processed_data['speed'].max()
            print(f"     Avg Speed: {self.avg_speed:.1f} km/h")
            print(f"     Max Speed: {self.max_speed:.1f} km/h")
        
        # Energy expenditure
        if 'power' in self.processed_data.columns:
            # Calculate total kJ (assuming 1Hz data)
            self.total_kj = self.processed_data['power'].sum() / 1000  # Convert to kJ
            print(f"     Total Energy: {self.total_kj:.0f} kJ")
        
        # Elevation metrics
        if 'elevation' in self.processed_data.columns:
            elevation_changes = self.processed_data['elevation'].diff()
            self.total_elevation_m = elevation_changes[elevation_changes > 0].sum()
            print(f"     Total Elevation Gain: {self.total_elevation_m:.0f}m")
        
        # Duration calculation
        if 'timestamp' in self.processed_data.columns:
            self.duration_hr = (self.processed_data['timestamp'].max() - 
                               self.processed_data['timestamp'].min()).total_seconds() / 3600
            print(f"     Duration: {self.duration_hr:.2f} hours")
        
        # Distance calculation
        if 'distance' in self.processed_data.columns:
            # Use distance column if available
            self.total_distance = self.processed_data['distance'].iloc[-1] / 1000  # Convert to km
        elif 'speed' in self.processed_data.columns:
            # Calculate distance from speed
            self.total_distance = self.processed_data['speed'].sum() / 3600  # km
        else:
            self.total_distance = 0
        print(f"     Total Distance: {self.total_distance:.2f} km")
        
        # Power zones calculation
        print("   🎯 Calculating power zones...")
        if 'power' in self.processed_data.columns:
            self._calculate_power_zones()
        
        # Heart rate zones calculation
        print("   ❤️ Calculating heart rate zones...")
        if 'heart_rate' in self.processed_data.columns:
            self._calculate_hr_zones()
        
        # Rolling averages
        print("   📈 Calculating rolling averages...")
        self._calculate_rolling_averages()
        
        # Power bests calculation
        print("   🏆 Calculating power bests...")
        self._calculate_power_bests()
        
        print("✅ Feature engineering completed")
    
    def _calculate_power_zones(self):
        """Calculate power zones and time in zones"""
        # Standard power zones based on FTP
        zone_thresholds = {
            'Z1': self.ftp * 0.55,  # Active Recovery
            'Z2': self.ftp * 0.75,  # Endurance
            'Z3': self.ftp * 0.90,  # Tempo
            'Z4': self.ftp * 1.05,  # Threshold
            'Z5': self.ftp * 1.20,  # VO2 Max
            'Z6': self.ftp * 1.50,  # Anaerobic Capacity
            'Z7': float('inf')       # Neuromuscular Power
        }
        
        power_series = self.processed_data['power']
        total_time = len(power_series)
        
        zone_times = {}
        for zone, threshold in zone_thresholds.items():
            if zone == 'Z7':
                zone_mask = power_series >= threshold
            else:
                zone_mask = power_series < threshold
            
            zone_times[zone] = zone_mask.sum()
        
        # Convert to percentages
        zone_percentages = {zone: (time / total_time) * 100 for zone, time in zone_times.items()}
        
        self.power_zones = {
            'thresholds': zone_thresholds,
            'times': zone_times,
            'percentages': zone_percentages
        }
        
        # Set zone_percentages for compatibility with existing methods
        zone_names = ['Active Recovery', 'Endurance', 'Tempo', 'Threshold', 'VO2 Max', 'Anaerobic Capacity', 'Neuromuscular Power']
        self.zone_percentages = {}
        for i, (zone, percentage) in enumerate(zone_percentages.items()):
            zone_name = zone_names[i] if i < len(zone_names) else zone
            self.zone_percentages[f'Z{i+1} ({zone_name})'] = percentage
        
        print("     Power zones calculated")
    
    def _calculate_hr_zones(self):
        """Calculate heart rate zones and time in zones"""
        # Standard HR zones based on max HR
        hr_thresholds = {
            'Z1': self.rest_hr + (self.max_hr - self.rest_hr) * 0.60,  # Active Recovery
            'Z2': self.rest_hr + (self.max_hr - self.rest_hr) * 0.70,  # Endurance
            'Z3': self.rest_hr + (self.max_hr - self.rest_hr) * 0.80,  # Tempo
            'Z4': self.rest_hr + (self.max_hr - self.rest_hr) * 0.90,  # Threshold
            'Z5': self.max_hr  # VO2 Max
        }
        
        hr_series = self.processed_data['heart_rate']
        total_time = len(hr_series)
        
        zone_times = {}
        for zone, threshold in hr_thresholds.items():
            if zone == 'Z5':
                zone_mask = hr_series >= threshold
            else:
                zone_mask = hr_series < threshold
            
            zone_times[zone] = zone_mask.sum()
        
        # Convert to percentages
        zone_percentages = {zone: (time / total_time) * 100 for zone, time in zone_times.items()}
        
        self.hr_zones = {
            'thresholds': hr_thresholds,
            'times': zone_times,
            'percentages': zone_percentages
        }
        
        # Set hr_zone_percentages for compatibility with existing methods
        hr_zone_names = ['Active Recovery', 'Endurance', 'Tempo', 'Threshold', 'VO2 Max']
        self.hr_zone_percentages = {}
        for i, (zone, percentage) in enumerate(zone_percentages.items()):
            zone_name = hr_zone_names[i] if i < len(hr_zone_names) else zone
            self.hr_zone_percentages[f'Z{i+1} ({zone_name})'] = percentage
        
        print("     Heart rate zones calculated")
    
    def _calculate_rolling_averages(self):
        """Calculate rolling averages for various metrics"""
        rolling_periods = [5, 30, 60, 300, 600]  # 5s, 30s, 1min, 5min, 10min
        
        for period in rolling_periods:
            if 'power' in self.processed_data.columns:
                self.processed_data[f'power_{period}s_avg'] = (
                    self.processed_data['power'].rolling(window=period, center=True, min_periods=1).mean()
                )
            
            if 'heart_rate' in self.processed_data.columns:
                self.processed_data[f'hr_{period}s_avg'] = (
                    self.processed_data['heart_rate'].rolling(window=period, center=True, min_periods=1).mean()
                )
        
        print("     Rolling averages calculated")
    
    def _calculate_power_bests(self):
        """Calculate power bests for various intervals"""
        if 'power' not in self.processed_data.columns:
            return
        
        intervals = [1, 5, 10, 30, 60, 180, 300, 480, 600, 720, 1200, 3600, 5400]  # seconds
        interval_names = ['1s', '5s', '10s', '30s', '1min', '3min', '5min', '8min', '10min', '12min', '20min', '60min', '90min']
        
        self.power_bests = {}
        
        for interval, name in zip(intervals, interval_names):
            if interval <= len(self.processed_data):
                # Calculate rolling average for this interval
                rolling_avg = self.processed_data['power'].rolling(window=interval, center=True, min_periods=interval).mean()
                
                # Find the maximum
                max_power = rolling_avg.max()
                
                if not pd.isna(max_power) and max_power > 0:
                    self.power_bests[name] = {
                        'power': max_power,
                        'duration': interval
                    }
        
        print(f"     Power bests calculated for {len(self.power_bests)} intervals")
    
    def _aggregate_data(self):
        """
        Step 5: Data Aggregation - Create ride totals and summaries
        """
        print("📊 Aggregating data and creating summaries...")
        
        # Store all metrics in a comprehensive dictionary
        self.metrics = {
            'power': {
                'avg': self.avg_power if hasattr(self, 'avg_power') else 0,
                'max': self.max_power if hasattr(self, 'max_power') else 0,
                'np': self.np_calc if hasattr(self, 'np_calc') else 0,
                'if': self.IF if hasattr(self, 'IF') else 0,
                'vi': self.VI if hasattr(self, 'VI') else 0,
                'tss': self.TSS if hasattr(self, 'TSS') else 0,
                'zones': self.power_zones if hasattr(self, 'power_zones') else {}
            },
            'heart_rate': {
                'avg': self.avg_hr if hasattr(self, 'avg_hr') else 0,
                'max': self.max_hr if hasattr(self, 'max_hr') else 0,
                'zones': self.hr_zones if hasattr(self, 'hr_zones') else {}
            },
            'cadence': {
                'avg': self.avg_cadence if hasattr(self, 'avg_cadence') else 0,
                'max': self.max_cadence if hasattr(self, 'max_cadence') else 0
            },
            'speed': {
                'avg': self.avg_speed if hasattr(self, 'avg_speed') else 0,
                'max': self.max_speed if hasattr(self, 'max_speed') else 0
            },
            'ride': {
                'duration_hr': self.duration_hr if hasattr(self, 'duration_hr') else 0,
                'total_distance': self.total_distance if hasattr(self, 'total_distance') else 0,
                'total_kj': self.total_kj if hasattr(self, 'total_kj') else 0,
                'total_elevation_m': self.total_elevation_m if hasattr(self, 'total_elevation_m') else 0
            },
            'power_bests': self.power_bests
        }
        
        print("✅ Data aggregation completed")
    
    def _create_visualizations(self):
        """
        Step 6: Visualization - Create comprehensive charts and plots
        """
        print("📈 Creating visualizations...")
        
        # Set the processed data as the main dataframe for compatibility
        self.df = self.processed_data
        
        if self.save_figures:
            # Create dashboard
            self.create_dashboard()
            
            # Create detailed analysis plots
            self.analyze_fatigue_patterns()
            self.analyze_heat_stress()
            self.analyze_power_hr_efficiency()
            self.analyze_variable_relationships()
            self.analyze_torque()
            # Create dual axis analysis
            self.create_professional_dual_axis_graph()
            # Calculate W' balance with proper error handling
            try:
                cp_est, w_prime_est = self.estimate_critical_power()
                if cp_est is not None and w_prime_est is not None:
                    self.calculate_w_prime_balance(cp_est, w_prime_est)
                else:
                    # Use FTP as fallback for CP estimation
                    cp_est = self.ftp
                    w_prime_est = 20000  # Default W' estimate
                    self.calculate_w_prime_balance(cp_est, w_prime_est)
            except Exception as e:
                print(f"     W' balance calculation failed: {str(e)}")
                # Use FTP as fallback
                cp_est = self.ftp
                w_prime_est = 20000  # Default W' estimate
                self.calculate_w_prime_balance(cp_est, w_prime_est)
            self.estimate_lactate()
        
        print("✅ Visualizations completed")
    
    def _perform_modeling(self):
        """
        Step 7: Modeling & Interpretation - CP & W' estimation, trends analysis
        """
        print("🧪 Performing advanced modeling and interpretation...")
        
        # Critical Power estimation
        if hasattr(self, 'power_bests') and len(self.power_bests) >= 3:
            self._estimate_critical_power()
        
        # Fatigue modeling (if historical data available)
        self._analyze_fatigue_patterns()
        
        # Performance trends
        self._analyze_performance_trends()
        
        print("✅ Modeling and interpretation completed")
    
    def _estimate_critical_power(self):
        """Estimate Critical Power using power-duration relationship"""
        # This is a simplified CP estimation
        # In practice, you'd need multiple rides for accurate CP estimation
        print("   📊 Estimating Critical Power...")
        
        # Use power bests for CP estimation
        if len(self.power_bests) >= 3:
            durations = []
            powers = []
            
            for interval, data in self.power_bests.items():
                if data['power'] > 0:
                    durations.append(data['duration'])
                    powers.append(data['power'])
            
            if len(durations) >= 3:
                # Simple linear regression on power-duration relationship
                # In practice, you'd use more sophisticated models
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(durations, powers)
                    estimated_cp = intercept  # CP is the y-intercept
                    
                    self.critical_power = estimated_cp
                    self.cp_r_squared = r_value ** 2
                    
                    print(f"     Estimated CP: {estimated_cp:.0f}W (R² = {r_value**2:.3f})")
                except:
                    print("     CP estimation failed - insufficient data")
    
    def _analyze_fatigue_patterns(self):
        """Analyze fatigue patterns during the ride"""
        print("   🔄 Analyzing fatigue patterns...")
        
        if 'power' in self.processed_data.columns:
            # Split ride into segments and analyze power decline
            segment_count = 10
            segment_length = len(self.processed_data) // segment_count
            
            segment_powers = []
            for i in range(segment_count):
                start_idx = i * segment_length
                end_idx = start_idx + segment_length
                segment_power = self.processed_data['power'].iloc[start_idx:end_idx].mean()
                segment_powers.append(segment_power)
            
            # Calculate fatigue index
            if len(segment_powers) >= 2:
                fatigue_index = (segment_powers[0] - segment_powers[-1]) / segment_powers[0] * 100
                self.fatigue_index = fatigue_index
                print(f"     Fatigue Index: {fatigue_index:.1f}%")
    
    def _analyze_performance_trends(self):
        """Analyze performance trends and patterns"""
        print("   📈 Analyzing performance trends...")
        
        # This would typically involve comparing with historical data
        # For now, we'll just note that this analysis is available
        print("     Performance trend analysis ready for historical comparison")
    
    # Legacy methods for backward compatibility
    def load_fit_file(self, file_path):
        """Legacy method - now uses the new pipeline"""
        return self.process_activity_data(file_path)
    
    def clean_and_smooth_data(self):
        """Legacy method - now handled in the pipeline"""
        pass  # Already done in _clean_data()
    
    def calculate_metrics(self):
        """Legacy method - now handled in the pipeline"""
        pass  # Already done in _engineer_features()
    
    def _print_performance_analysis(self):
        """Print performance analysis with context."""
        # Performance analysis completed
        intensity_level = 'High' if hasattr(self, 'TSS') and self.TSS is not None and self.TSS > 150 else 'Medium' if hasattr(self, 'TSS') and self.TSS is not None and self.TSS > 80 else 'Low'
        pacing_quality = 'Poor'
        if hasattr(self, 'avg_power') and hasattr(self, 'np_calc') and self.np_calc is not None and self.avg_power is not None:
            try:
                ratio = float(self.avg_power) / float(self.np_calc) if float(self.np_calc) != 0 else 0
                if ratio > 0.9:
                    pacing_quality = 'Good'
                elif ratio > 0.8:
                    pacing_quality = 'Moderate'
            except Exception:
                pacing_quality = 'Poor'
        tss_vs_target = "Above" if hasattr(self, 'TSS') and hasattr(self, 'target_tss') and self.TSS > self.target_tss else "Below" if hasattr(self, 'TSS') and hasattr(self, 'target_tss') and self.TSS < self.target_tss * 0.8 else "On Target"
        
        # Performance metrics calculated
    
    def create_professional_multi_axis_graph(self):
        """Create a professional multi-axis graph with multiple variables on different axes."""
        if self.df is None:
            return
            
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60 if 'timestamp' in self.df.columns else np.arange(len(self.df))
        
        # Create figure with multiple subplots sharing x-axis
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, 
                                gridspec_kw={'height_ratios': [2, 1, 1, 1], 'hspace': 0.1})
        
        # Set main title
        fig.suptitle(f'{self.athlete_name} - Professional Multi-Axis Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Power (Primary metric - largest subplot)
        if 'power' in self.df.columns:
            # Raw power as background
            axes[0].plot(time_minutes, self.df['power'], alpha=0.3, linewidth=0.8, color='#1f77b4', label='Raw Power')
            
            # Smoothed power as main line
            if 'power_smoothed' in self.df.columns:
                axes[0].plot(time_minutes, self.df['power_smoothed'], alpha=0.9, linewidth=2, color='#1f77b4', label='Power')
            
            # Add average and NP lines
            if hasattr(self, 'avg_power') and self.avg_power is not None:
                axes[0].axhline(y=self.avg_power, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Avg: {self.avg_power:.0f}W')
            if hasattr(self, 'np_calc') and self.np_calc is not None:
                axes[0].axhline(y=self.np_calc, color='orange', linestyle='--', alpha=0.8, linewidth=1.5, label=f'NP: {self.np_calc:.0f}W')
            
            axes[0].set_ylabel('Power (W)', fontweight='bold')
            axes[0].legend(loc='upper right', fontsize=10)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('Power Profile', fontweight='bold', fontsize=12, pad=10)
            
            # Set y-axis limits with some padding
            if 'power_smoothed' in self.df.columns:
                power_data = self.df['power_smoothed'].dropna()
                if len(power_data) > 0:
                    power_min, power_max = power_data.min(), power_data.max()
                    power_range = power_max - power_min
                    axes[0].set_ylim(max(0, power_min - power_range * 0.1), power_max + power_range * 0.1)
        else:
            axes[0].text(0.5, 0.5, 'No power data available', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Power Profile (No data)', fontweight='bold', fontsize=12)
        
        # 2. Heart Rate
        if 'heart_rate' in self.df.columns:
            # Raw HR as background
            axes[1].plot(time_minutes, self.df['heart_rate'], alpha=0.3, linewidth=0.8, color='#d62728', label='Raw HR')
            
            # Smoothed HR as main line
            if 'heart_rate_smoothed' in self.df.columns:
                axes[1].plot(time_minutes, self.df['heart_rate_smoothed'], alpha=0.9, linewidth=2, color='#d62728', label='Heart Rate')
            
            # Add average HR line
            if 'heart_rate' in self.metrics and 'avg' in self.metrics['heart_rate']:
                avg_hr = self.metrics['heart_rate']['avg']
                axes[1].axhline(y=avg_hr, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Avg: {avg_hr:.0f}bpm')
            
            axes[1].set_ylabel('Heart Rate (bpm)', fontweight='bold')
            axes[1].legend(loc='upper right', fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title('Heart Rate Profile', fontweight='bold', fontsize=12, pad=10)
            
            # Set y-axis limits
            if 'heart_rate_smoothed' in self.df.columns:
                hr_data = self.df['heart_rate_smoothed'].dropna()
                if len(hr_data) > 0:
                    hr_min, hr_max = hr_data.min(), hr_data.max()
                    hr_range = hr_max - hr_min
                    axes[1].set_ylim(max(0, hr_min - hr_range * 0.1), hr_max + hr_range * 0.1)
        else:
            axes[1].text(0.5, 0.5, 'No heart rate data available', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Heart Rate Profile (No data)', fontweight='bold', fontsize=12)
        
        # 3. Cadence
        if 'cadence' in self.df.columns:
            # Raw cadence as background
            axes[2].plot(time_minutes, self.df['cadence'], alpha=0.3, linewidth=0.8, color='#2ca02c', label='Raw Cadence')
            
            # Smoothed cadence as main line
            if 'cadence_smoothed' in self.df.columns:
                axes[2].plot(time_minutes, self.df['cadence_smoothed'], alpha=0.9, linewidth=2, color='#2ca02c', label='Cadence')
            
            # Add average cadence line
            if 'cadence' in self.metrics and 'avg' in self.metrics['cadence']:
                avg_cadence = self.metrics['cadence']['avg']
                axes[2].axhline(y=avg_cadence, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Avg: {avg_cadence:.0f}rpm')
            
            axes[2].set_ylabel('Cadence (rpm)', fontweight='bold')
            axes[2].legend(loc='upper right', fontsize=10)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_title('Cadence Profile', fontweight='bold', fontsize=12, pad=10)
            
            # Set y-axis limits
            if 'cadence_smoothed' in self.df.columns:
                cadence_data = self.df['cadence_smoothed'].dropna()
                if len(cadence_data) > 0:
                    cadence_min, cadence_max = cadence_data.min(), cadence_data.max()
                    cadence_range = cadence_max - cadence_min
                    axes[2].set_ylim(max(0, cadence_min - cadence_range * 0.1), cadence_max + cadence_range * 0.1)
        else:
            axes[2].text(0.5, 0.5, 'No cadence data available', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Cadence Profile (No data)', fontweight='bold', fontsize=12)
        
        # 4. Speed (if available) or Elevation
        if 'speed' in self.df.columns:
            # Convert speed to km/h if it's in m/s
            speed_data = self.df['speed']
            if speed_data.max() < 50:  # Likely in m/s, convert to km/h
                speed_data = speed_data * 3.6
            
            axes[3].plot(time_minutes, speed_data, alpha=0.9, linewidth=2, color='#9467bd', label='Speed')
            
            # Add average speed line
            avg_speed = speed_data.mean()
            axes[3].axhline(y=avg_speed, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Avg: {avg_speed:.1f}km/h')
            
            axes[3].set_ylabel('Speed (km/h)', fontweight='bold')
            axes[3].legend(loc='upper right', fontsize=10)
            axes[3].grid(True, alpha=0.3)
            axes[3].set_title('Speed Profile', fontweight='bold', fontsize=12, pad=10)
            
            # Set y-axis limits
            speed_min, speed_max = speed_data.min(), speed_data.max()
            speed_range = speed_max - speed_min
            axes[3].set_ylim(max(0, speed_min - speed_range * 0.1), speed_max + speed_range * 0.1)
        elif 'altitude' in self.df.columns:
            axes[3].plot(time_minutes, self.df['altitude'], alpha=0.9, linewidth=2, color='#8c564b', label='Elevation')
            
            # Add average elevation line
            avg_altitude = self.df['altitude'].mean()
            axes[3].axhline(y=avg_altitude, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Avg: {avg_altitude:.0f}m')
            
            axes[3].set_ylabel('Elevation (m)', fontweight='bold')
            axes[3].legend(loc='upper right', fontsize=10)
            axes[3].grid(True, alpha=0.3)
            axes[3].set_title('Elevation Profile', fontweight='bold', fontsize=12, pad=10)
        else:
            axes[3].text(0.5, 0.5, 'No speed or elevation data available', ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Speed/Elevation Profile (No data)', fontweight='bold', fontsize=12)
        
        # Set x-axis label and limits for all subplots
        axes[-1].set_xlabel('Time (minutes)', fontweight='bold')
        axes[-1].set_xlim(0, time_minutes.max())
        
        # Add ride summary text box
        summary_text = f"Duration: {self.duration_hr:.1f}h | Distance: {self.total_distance:.1f}km"
        if hasattr(self, 'avg_power') and self.avg_power is not None:
            summary_text += f" | Avg Power: {self.avg_power:.0f}W"
        if hasattr(self, 'np_calc') and self.np_calc is not None:
            summary_text += f" | NP: {self.np_calc:.0f}W"
        if 'heart_rate' in self.metrics and 'avg' in self.metrics['heart_rate']:
            summary_text += f" | Avg HR: {self.metrics['heart_rate']['avg']:.0f}bpm"
        
        fig.text(0.02, 0.02, summary_text, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        plt.tight_layout()
        
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_training_peaks_style")
            fig.savefig(fig_path + ".png", dpi=300, bbox_inches='tight')
            fig.savefig(fig_path + ".svg", bbox_inches='tight')
        else:
            plt.show()

    def create_dashboard(self):
        """Create simplified dashboard with 3 key graphs."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60 if 'timestamp' in self.df.columns else np.arange(len(self.df))
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
        ax_bar = fig.add_subplot(gs[1, 2])  # Bar chart below the pie
        fig.suptitle(f'{self.athlete_name} - Ride Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.97)
        plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.92, bottom=0.08, left=0.05, right=0.95)
        # 1. Power Profile (Main metric)
        lines = []
        labels = []
        if 'power' in self.df.columns:
            l1, = axes[0].plot(time_minutes, self.df['power'], alpha=0.3, linewidth=1, color='#1f77b4', label='Raw Power')
            lines.append(l1)
            labels.append('Raw Power')
            if 'power_smoothed' in self.df.columns:
                l2, = axes[0].plot(time_minutes, self.df['power_smoothed'], alpha=0.9, linewidth=2.5, color='#1f77b4', label='Smoothed Power')
                lines.append(l2)
                labels.append('Smoothed Power')
                avg_power_val = float(self.avg_power) if hasattr(self, 'avg_power') and self.avg_power is not None and not isinstance(self.avg_power, pd.Series) else 0.0
                np_calc_val = float(self.np_calc) if hasattr(self, 'np_calc') and self.np_calc is not None and not isinstance(self.np_calc, pd.Series) else 0.0
                l3 = axes[0].axhline(y=avg_power_val, color='red', linestyle='--', alpha=0.8, label=f'Avg: {avg_power_val:.0f} W')
                l4 = axes[0].axhline(y=np_calc_val, color='orange', linestyle='--', alpha=0.8, label=f'NP: {np_calc_val:.0f} W')
                lines.extend([l3, l4])
                labels.extend([f'Avg: {avg_power_val:.0f} W', f'NP: {np_calc_val:.0f} W'])
            axes[0].set_xlabel('Time (minutes)')
            axes[0].set_ylabel('Power (W)')
            axes[0].set_title('Power Profile', fontweight='bold', fontsize=14)
            axes[0].legend(lines, labels)
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].set_title('Power Profile (No power data)')
            axes[0].text(0.5, 0.5, 'No power data available', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_axis_off()
        # 2. Heart Rate Profile (Enhanced)
        if 'heart_rate' in self.df.columns:
            sample_rate = max(1, len(time_minutes) // 200)
            time_sampled = time_minutes[::sample_rate]
            hr_sampled = self.df['heart_rate'].iloc[::sample_rate]
            axes[1].plot(time_sampled, hr_sampled, alpha=0.3, linewidth=1, color='#d62728', label='Raw HR')
            if 'heart_rate_smoothed' in self.df.columns:
                axes[1].plot(time_minutes, self.df['heart_rate_smoothed'], alpha=0.9, linewidth=2.5, color='#d62728', label='Smoothed HR')
            axes[1].axhline(y=self.metrics.get('hr', {}).get('avg', 0), color='red', linestyle='--', alpha=0.8, 
                           label=f"Avg: {self.metrics.get('hr', {}).get('avg', 0):.0f} bpm")
            axes[1].set_xlabel('Time (minutes)')
            axes[1].set_ylabel('Heart Rate (bpm)')
            axes[1].set_title('Heart Rate Profile', fontweight='bold', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].set_title('Heart Rate Profile (No HR data)')
            axes[1].text(0.5, 0.5, 'No heart rate data available', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_axis_off()
        # 3. Power Zone Distribution (Key metric) - Bigger pie chart with zone numbers
        if 'power' in self.df.columns and self.zone_percentages:
            zone_labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7']
            zone_names = ['Recovery', 'Endurance', 'Tempo', 'Threshold', 'VO2max', 'Anaerobic', 'Neuromuscular']
            zone_full_labels = [f'Z{i} ({zone_names[i-1]})' for i in range(1, 8)]
            zone_values = [self.zone_percentages.get(f'Z{i} ({zone_names[i-1]})', 0) for i in range(1, 8)]
            colors = [self.zone_colors.get(f'Z{i} ({zone_names[i-1]})', '#cccccc') for i in range(1, 8)]
            explode = [0.05 if val > 0 else 0 for val in zone_values]
            pie_result = axes[2].pie(zone_values, labels=zone_labels, autopct='', 
                                                   colors=colors, startangle=90, explode=explode, shadow=True,
                                                   textprops={'fontsize': 12, 'fontweight': 'bold'})
            if len(pie_result) == 3:
                wedges, texts, autotexts = pie_result
            else:
                wedges, texts = pie_result
                autotexts = []
            for autotext in autotexts:
                autotext.set_text('')
            for i, text in enumerate(texts):
                if zone_values[i] > 0:
                    text.set_color(colors[i])
                    text.set_fontweight('bold')
                    text.set_fontsize(11)
                else:
                    text.set_text('')
            axes[2].set_title('Power Zone Distribution', fontweight='bold', fontsize=14)
            table_data = []
            for i, (zone_label, zone_name, value) in enumerate(zip(zone_labels, zone_names, zone_values)):
                if value > 0:
                    table_data.append([zone_label, zone_name, f"{value:.1f}%"])
            if table_data:
                table_text = "Zone Distribution:\n"
                table_text += "=" * 35 + "\n"
                table_text += f"{'Zone':<4} {'Name':<12} {'%':<6}\n"
                table_text += "-" * 35 + "\n"
                for zone_label, zone_name, percentage in table_data:
                    table_text += f"{zone_label:<4} {zone_name:<12} {percentage:<6}\n"
                axes[2].text(1.2, 0.5, table_text, transform=axes[2].transAxes, 
                           fontsize=9, verticalalignment='center', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            if self.multi_ride_zones is not None and isinstance(self.multi_ride_zones, list) and len(self.multi_ride_zones) > 1:
                zone_percentages_list = self.multi_ride_zones
                ride_labels = getattr(self, 'multi_ride_labels', [f'Ride {i+1}' for i in range(len(zone_percentages_list))])
                zones = list(self.zone_colors.keys())
                data = np.array([[ride.get(zone, 0) for zone in zones] for ride in zone_percentages_list])
                left = np.zeros(len(zone_percentages_list))
                for i, zone in enumerate(zones):
                    ax_bar.barh(np.arange(len(zone_percentages_list)), data[:, i], left=left, color=self.zone_colors[zone], label=zone)
                    left += data[:, i]
                ax_bar.set_yticks(np.arange(len(zone_percentages_list)))
                ax_bar.set_yticklabels(ride_labels)
                ax_bar.set_xlabel('Time in Zone (%)')
                ax_bar.set_title('Zone Distribution Comparison', fontweight='bold', fontsize=12)
                ax_bar.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            else:
                ax_bar.axis('off')
        else:
            axes[2].set_title('Power Zone Distribution (No power data)')
            axes[2].text(0.5, 0.5, 'No power data available', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_axis_off()
        plt.tight_layout()
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_dashboard")
            fig.savefig(fig_path + ".png", dpi=300)
            fig.savefig(fig_path + ".svg")
        else:
            plt.show()
    
    def estimate_critical_power(self):
        """Estimate Critical Power and W'."""
        if self.df is None:
            return None, None
        
        def cp_model(t, CP, W_prime):
            return CP + (W_prime / t)
        
        durations = np.array([5, 15, 30, 60, 120, 300, 600])
        mmp = [self.df['power'].rolling(window=d, min_periods=1).mean().max() for d in durations]
        
        # Filter out any NaN or invalid values
        valid_indices = [i for i, val in enumerate(mmp) if not pd.isna(val) and val > 0]
        if len(valid_indices) < 3:
            # Not enough data for CP estimation, use FTP as fallback
            return self.ftp, 20000
        
        durations = durations[valid_indices]
        mmp = [mmp[i] for i in valid_indices]
        
        try:
            popt, _ = curve_fit(cp_model, durations, mmp, bounds=(0, [600, 100000]))
            cp_est, w_prime_est = popt
            
            # Validate results
            if cp_est < 50 or cp_est > 600 or w_prime_est < 1000 or w_prime_est > 50000:
                # Results out of reasonable range, use FTP as fallback
                return self.ftp, 20000
            
            return cp_est, w_prime_est
            
        except Exception as e:
            # Curve fitting failed, use FTP as fallback
            return self.ftp, 20000
    
    def calculate_w_prime_balance(self, cp_est, w_prime_est):
        """Calculate W' balance over time with enhanced smooth visualization."""
        if self.df is None or cp_est is None:
            return
        
        tau = self.w_prime_tau
        w_bal = []
        current_w_prime = w_prime_est
        timestamps = pd.to_datetime(self.df['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds().fillna(1).clip(lower=0.01)
        
        # Apply smoothing to power data for smoother W' balance calculation
        power_smoothed = self.df['power'].rolling(window=5, center=True, min_periods=1).mean()
        
        for idx, row in self.df.iterrows():
            power = power_smoothed.iloc[idx] if not pd.isna(power_smoothed.iloc[idx]) else row['power']
            dt = time_diffs.iloc[idx]
            if power > cp_est:
                current_w_prime -= (power - cp_est) * dt
            else:
                recovery = (w_prime_est - current_w_prime) * (1 - np.exp(-dt / tau))
                current_w_prime += recovery
            current_w_prime = max(0, min(current_w_prime, w_prime_est))
            w_bal.append(current_w_prime)
        
        self.df['w_prime_bal'] = w_bal
        
        # Apply additional smoothing to W' balance for visualization
        w_bal_smoothed = pd.Series(w_bal).rolling(window=10, center=True, min_periods=1).mean()
        
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        
        # Create enhanced W' balance plot with smoother curves
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Main W' balance plot with smoothed data
        ax1.plot(time_minutes, w_bal_smoothed, color='purple', linewidth=2.5, label='W\' Balance (Smoothed)', alpha=0.9)
        ax1.plot(time_minutes, self.df['w_prime_bal'], color='purple', linewidth=1, alpha=0.3, label='W\' Balance (Raw)')
        
        # Add gradient fill for visual appeal
        ax1.fill_between(time_minutes, w_bal_smoothed, alpha=0.2, color='purple')
        
        # Add reference lines with better styling
        ax1.axhline(y=w_prime_est, color='green', linestyle='--', alpha=0.8, linewidth=2)
        ax1.text(time_minutes.iloc[-1], w_prime_est, "Full W′ (100%)", color='green', fontsize=11, fontweight='bold', 
                va='bottom', ha='right', backgroundcolor='white', alpha=0.9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax1.axhline(y=w_prime_est * 0.5, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, 
                    label='50% W\' Depleted')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='W\' Depleted')
        
        # Enhanced zone shading with better colors
        ax1.fill_between(time_minutes, w_prime_est * 0.8, w_prime_est, alpha=0.15, color='green', label='High W\' (>80%)')
        ax1.fill_between(time_minutes, w_prime_est * 0.4, w_prime_est * 0.8, alpha=0.15, color='yellow', label='Moderate W\' (40-80%)')
        ax1.fill_between(time_minutes, 0, w_prime_est * 0.4, alpha=0.15, color='red', label='Low W\' (<40%)')
        
        ax1.set_title("Enhanced W' Balance Over Time", fontweight='bold', fontsize=14)
        ax1.set_xlabel('Time (minutes)', fontsize=12)
        ax1.set_ylabel('W\' Balance (J)', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, w_prime_est * 1.1)
        
        # Enhanced power vs W' balance scatter with smoother data
        # Use smoothed power data for better visualization
        power_for_scatter = power_smoothed.fillna(self.df['power'])
        
        scatter = ax2.scatter(power_for_scatter, w_bal_smoothed, 
                             alpha=0.7, c=time_minutes, cmap='viridis', s=25, edgecolors='white', linewidth=0.5)
        ax2.set_xlabel('Power (W)', fontsize=12)
        ax2.set_ylabel('W\' Balance (J)', fontsize=12)
        ax2.set_title('Power vs W\' Balance Relationship (Smoothed)', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add CP reference line with better styling
        ax2.axvline(x=cp_est, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'CP ({cp_est:.0f}W)')
        ax2.legend(fontsize=10)
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Time (minutes)', fontsize=10)
        
        plt.tight_layout()
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_w_prime_balance")
            fig.savefig(fig_path + ".png", dpi=300, bbox_inches='tight')
            fig.savefig(fig_path + ".svg", bbox_inches='tight')
        else:
            plt.show()
        
        # Print W' balance statistics
        w_bal_min = min(w_bal)
        w_bal_max = max(w_bal)
        w_bal_final = w_bal[-1]
        depletion_pct = ((w_prime_est - w_bal_final) / w_prime_est) * 100
        
        print(f"\n📊 W' Balance Analysis:")
        print(f"   Initial W': {w_prime_est:.0f}J")
        print(f"   Final W': {w_bal_final:.0f}J")
        print(f"   W' Depletion: {depletion_pct:.1f}%")
        print(f"   Min W': {w_bal_min:.0f}J")
        print(f"   Max W': {w_bal_max:.0f}J")
    
    def calculate_hr_strain(self):
        """Analyze HR response using practical training metrics."""
        if self.df is None or 'heart_rate' not in self.df.columns:
            # Heart rate data required
            return
        
        # Calculate HR reserve utilization (Karvonen method)
        hr_reserve = self.max_hr - self.rest_hr
        self.df['hr_reserve_pct'] = (self.df['heart_rate'] - self.rest_hr) / hr_reserve * 100
        
        # Calculate time in minutes
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        self.df['time_minutes'] = time_minutes
        
        # 1. Aerobic Decoupling (Pw:Hr Drift) - Enhanced with shorter, normalized efforts
        aerobic_decoupling = []
        
        # Multiple window sizes for different effort types
        window_sizes = [120, 180, 240]  # 2, 3, 4 minutes
        step_sizes = [30, 45, 60]       # Check every 30s, 45s, 60s respectively
        
        if self.df is not None:
            for window_size, step_size in zip(window_sizes, step_sizes):
                for i in range(0, len(self.df) - window_size, step_size):
                    window_data = self.df.iloc[i:i+window_size]
                    if len(window_data) == window_size:
                        power_std = window_data['power'].std()
                        power_mean = window_data['power'].mean()
                        hr_mean = window_data['heart_rate'].mean()
                        
                        # Enhanced steady effort criteria:
                        # - Power CV < 8% (more realistic)
                        # - Power > 50% FTP (lower threshold for more efforts)
                        # - HR > 60% max HR (ensure meaningful HR response)
                        # - Power > 100W (minimum meaningful effort)
                        power_cv = power_std / power_mean if power_mean > 0 else float('inf')
                        hr_reserve_pct = (hr_mean - self.rest_hr) / (self.max_hr - self.rest_hr) * 100
                        
                        if (power_cv < 0.08 and 
                            power_mean > self.ftp * 0.5 and 
                            power_mean > 100 and
                            hr_reserve_pct > 20):  # At least 20% HR reserve
                            
                            # Split into thirds for more granular analysis
                            third_size = len(window_data) // 3
                            first_third = window_data.iloc[:third_size]
                            last_third = window_data.iloc[-third_size:]
                            
                            # Calculate Efficiency Factor (power/HR) for each third
                            ef1 = first_third['power'].mean() / first_third['heart_rate'].mean()
                            ef2 = last_third['power'].mean() / last_third['heart_rate'].mean()
                            
                            # Calculate drift percentage
                            drift_pct = (ef2 - ef1) / ef1 * 100 if ef1 > 0 else 0
                            
                            # Only include if we have meaningful efficiency values
                            if ef1 > 0 and ef2 > 0:
                                aerobic_decoupling.append({
                                    'start_time': time_minutes.iloc[i],
                                    'duration': window_size / 60,  # Duration in minutes
                                    'avg_power': power_mean,
                                    'avg_hr': hr_mean,
                                    'power_cv': power_cv * 100,  # Store CV as percentage
                                    'ef1': ef1,
                                    'ef2': ef2,
                                    'drift_pct': drift_pct,
                                    'hr_reserve_pct': hr_reserve_pct
                                })
        
        # 2. Cardiac Cost Index (Beats per kJ or km)
        # Calculate total heartbeats (for 1s data: sum(HR)/60)
        interval_sec = (self.df['timestamp'].iloc[1] - self.df['timestamp'].iloc[0]).total_seconds() if len(self.df) > 1 else 1
        total_beats = self.df['heart_rate'].sum() * (interval_sec / 60)
        total_work_kj = (self.df['power'].sum() * (interval_sec / 3600))  # W to kJ
        total_distance_km = self.df['distance'].iloc[-1] / 1000 if len(self.df) > 0 and 'distance' in self.df.columns else 0
        cci_kj = total_beats / total_work_kj if total_work_kj > 0 else 0
        cci_km = total_beats / total_distance_km if total_distance_km > 0 else 0
        
        # 3. TRIMP (HR-based Training Load)
        # Banister formula: Σ(dt_min × HRr × e^(β·HRr)), dt_min = dt_sec/60
        trimp_score = 0
        beta = 1.92  # For men (use 1.67 for women)
        dt_sec = self.df['timestamp'].diff().dt.total_seconds().fillna(1)
        for i in range(len(self.df)):
            hr_reserve_decimal = self.df['hr_reserve_pct'].iloc[i] / 100
            dt_min = dt_sec.iloc[i] / 60
            trimp_score += dt_min * hr_reserve_decimal * np.exp(beta * hr_reserve_decimal)
        
        # 4. HR-based MET and VO₂ Estimates
        # Formula: MET ≈ 6 × HR_index – 5; VO₂ = MET × 3.5 ml/kg/min
        hr_index = self.df['heart_rate'] / self.rest_hr
        met_values = 6 * hr_index - 5
        vo2_values = met_values * 3.5  # ml/kg/min
        
        avg_met = met_values.mean()
        max_met = met_values.max()
        avg_vo2 = vo2_values.mean()
        max_vo2 = vo2_values.max()
        total_energy = met_values.sum() * 3.5  # Total MET-minutes
        
        # Create practical HR analysis visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('HR Training Load Analysis (Practical Metrics)', fontweight='bold', fontsize=14)
        
        # 1. Aerobic Decoupling - Enhanced visualization
        if aerobic_decoupling:
            drift_percentages = [dec['drift_pct'] for dec in aerobic_decoupling]
            avg_powers = [dec['avg_power'] for dec in aerobic_decoupling]
            durations = [dec['duration'] for dec in aerobic_decoupling]
            power_cvs = [dec['power_cv'] for dec in aerobic_decoupling]
            
            # Color code by duration
            colors = ['blue' if d <= 2 else 'green' if d <= 3 else 'red' for d in durations]
            sizes = [60 + d * 20 for d in durations]  # Size based on duration
            
            scatter = ax1.scatter(avg_powers, drift_percentages, c=colors, s=sizes, alpha=0.7)
            ax1.set_xlabel('Average Power (W)')
            ax1.set_ylabel('Aerobic Decoupling (%)')
            ax1.set_title('Enhanced Aerobic Decoupling Analysis')
            ax1.grid(True, alpha=0.3)
            
            # Add reference lines
            ax1.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Good (<5%)')
            ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Moderate (5-10%)')
            ax1.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Poor (>10%)')
            
            # Add legend for duration colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', alpha=0.7, label='2min efforts'),
                Patch(facecolor='green', alpha=0.7, label='3min efforts'),
                Patch(facecolor='red', alpha=0.7, label='4min efforts')
            ]
            ax1.legend(handles=legend_elements, loc='upper right')
            
            # Add trend line if enough data points
            if len(drift_percentages) > 3:
                z = np.polyfit(avg_powers, drift_percentages, 1)
                p = np.poly1d(z)
                ax1.plot(avg_powers, p(avg_powers), "r--", alpha=0.8, linewidth=2)
        else:
            ax1.text(0.5, 0.5, 'No steady efforts\nfound for decoupling analysis\n(Try longer rides or\nmore consistent efforts)', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=11)
            ax1.set_title('Enhanced Aerobic Decoupling Analysis')
        
        # 2. Cardiac Cost Index
        ax2.bar(['Beats/kJ', 'Beats/km'], [cci_kj, cci_km], color=['blue', 'green'], alpha=0.7)
        ax2.set_ylabel('Cardiac Cost Index (beats/kJ or beats/km)')
        ax2.set_title('Cardiovascular Economy')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate([cci_kj, cci_km]):
            ax2.text(i, v + max(cci_kj, cci_km) * 0.01, f'{v:.1f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 3. TRIMP Score Over Time
        # Calculate cumulative TRIMP
        cumulative_trimp = []
        current_trimp = 0
        dt_sec = self.df['timestamp'].diff().dt.total_seconds().fillna(1)
        for i in range(len(self.df)):
            hr_reserve_decimal = self.df['hr_reserve_pct'].iloc[i] / 100
            dt_min = dt_sec.iloc[i] / 60
            current_trimp += dt_min * hr_reserve_decimal * np.exp(beta * hr_reserve_decimal)
            cumulative_trimp.append(current_trimp)
        
        ax3.plot(time_minutes, cumulative_trimp, color='purple', linewidth=2)
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Cumulative TRIMP Score')
        ax3.set_title('Training Load (TRIMP, unitless)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. MET and VO₂ Over Time
        ax4.plot(time_minutes, met_values, color='orange', linewidth=1.5, alpha=0.8, label='MET')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(time_minutes, vo2_values, color='red', linewidth=1.5, alpha=0.8, label='VO₂')
        
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('MET', color='orange')
        ax4_twin.set_ylabel('VO₂ (ml/kg/min)', color='red')
        ax4.set_title('Metabolic Cost Over Time')
        ax4.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_hr_strain")
            fig.savefig(fig_path + ".png", dpi=300)
            fig.savefig(fig_path + ".svg")
        else:
            plt.show()
        
        # Print practical HR analysis
        # HR training load analysis completed
        
        # Aerobic decoupling analysis completed
        
        return aerobic_decoupling, trimp_score
    
    def analyze_heat_stress(self):
        """Analyze heat stress factors and HR response correlation."""
        if self.df is None or 'heart_rate' not in self.df.columns:
            return
        
        # Calculate time-based heat stress indicators
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        
        # 1. HR Drift Analysis (heat stress indicator)
        # Calculate HR drift over time using rolling windows
        hr_rolling_10min = self.df['heart_rate'].rolling(window=600, min_periods=300).mean()  # 10-min windows
        hr_rolling_5min = self.df['heart_rate'].rolling(window=300, min_periods=150).mean()   # 5-min windows
        
        # Calculate HR drift rate (increase in HR over time)
        hr_drift_rate = []
        drift_times = []
        
        for i in range(300, len(self.df), 300):  # Check every 5 minutes
            if i + 300 <= len(self.df):
                window_data = self.df.iloc[i:i+300]
                if len(window_data) > 150:  # At least 2.5 minutes of data
                    hr_start = window_data['heart_rate'].iloc[:60].mean()  # First minute
                    hr_end = window_data['heart_rate'].iloc[-60:].mean()   # Last minute
                    power_avg = window_data['power'].mean()
                    
                    # Only consider if power is meaningful (>100W)
                    if power_avg > 100:
                        drift_rate = (hr_end - hr_start) / 5  # HR increase per minute
                        hr_drift_rate.append(drift_rate)
                        drift_times.append(time_minutes.iloc[i])
        
        # 2. HR Response Lag Analysis (heat stress indicator)
        # Calculate HR response time to power changes
        hr_response_lag = []
        hr_response_amplitude = []
        hr_response_amplitude_times = []  # Track actual times for amplitude events
        
        # Find power changes and measure HR response
        power_changes = self.df['power'].diff().abs()
        significant_changes = power_changes > SIGNIFICANT_POWER_CHANGE  # 20W change threshold
        
        for i in range(60, len(self.df) - 60):
            if significant_changes.iloc[i]:
                # Look for HR response in next 30 seconds
                hr_before = self.df['heart_rate'].iloc[i-30:i].mean()
                hr_after = self.df['heart_rate'].iloc[i:i+30].mean()
                
                if hr_after > hr_before:
                    response_time = 0
                    for j in range(1, 31):
                        if self.df['heart_rate'].iloc[i+j] > hr_before + 2:  # 2 bpm threshold
                            response_time = j
                            break
                    
                    if response_time > 0:
                        hr_response_lag.append(response_time)
                        hr_response_amplitude.append(hr_after - hr_before)
                        # Store the actual time in minutes for this event
                        hr_response_amplitude_times.append(time_minutes.iloc[i])
        
        # 3. Heat Stress Index (composite metric)
        # Combine HR drift, response lag, and time factors
        heat_stress_index = []
        heat_stress_times = []
        
        for i in range(0, len(self.df), 60):  # Every minute
            if i + 60 <= len(self.df):
                window_data = self.df.iloc[i:i+60]
                hr_mean = window_data['heart_rate'].mean()
                hr_std = window_data['heart_rate'].std()
                time_factor = time_minutes.iloc[i] / 60  # Hours into ride
                
                # Heat stress components
                hr_elevation = (hr_mean - self.rest_hr) / (self.max_hr - self.rest_hr)  # 0-1 scale
                hr_variability = hr_std / hr_mean if hr_mean > 0 else 0
                time_heat_factor = min(time_factor * 0.1, 0.5)  # Heat accumulates over time
                
                # Composite heat stress index (0-100 scale)
                heat_index = (
                    hr_elevation * 40 +           # HR elevation (40 points)
                    hr_variability * 20 +         # HR variability (20 points)
                    time_heat_factor * 40         # Time-based heat (40 points)
                )
                
                heat_stress_index.append(heat_index)
                heat_stress_times.append(time_minutes.iloc[i])
        
        # 4. Create heat stress visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Heat Stress Analysis & HR Response', fontweight='bold', fontsize=14)
        
        # 1. Heat Stress Index over time
        if heat_stress_index:
            ax1.plot(heat_stress_times, heat_stress_index, color='red', linewidth=2, alpha=0.8)
            ax1.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Low Heat Stress')
            ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Moderate Heat Stress')
            ax1.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='High Heat Stress')
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Heat Stress Index (unitless)')
            ax1.set_title('Heat Stress Index Over Time', fontweight='bold', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Insufficient data\nfor heat stress analysis', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Heat Stress Index Over Time', fontweight='bold', fontsize=14)
        
        # 2. HR Drift Rate vs Power
        if hr_drift_rate:
            ax2.scatter([self.df['power'].iloc[int(t*60)] for t in drift_times], 
                       hr_drift_rate, alpha=0.7, s=60, color='orange')
            ax2.set_xlabel('Average Power (W)')
            ax2.set_ylabel('HR Drift Rate (bpm/min)')
            ax2.set_title('HR Drift Rate vs Power', fontweight='bold', fontsize=14)
            # Draw vertical line at CP if available
            # Always draw vertical dashed line at CP=335 W
            ax2.axvline(x=335, ls='--', color='red', lw=1, label='CP: 335 W')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. HR Response Lag Distribution
        if hr_response_lag:
            ax3.hist(hr_response_lag, bins=15, alpha=0.7, color='blue', edgecolor='black')
            ax3.set_xlabel('HR Response Time (seconds)')
            ax3.set_ylabel('Frequency (count)')
            ax3.set_title('HR Response Lag Distribution', fontweight='bold', fontsize=14)
            ax3.grid(True, alpha=0.3)
            # Add statistics
            avg_lag = np.mean(hr_response_lag)
            ax3.axvline(x=avg_lag, color='red', linestyle='--', alpha=0.8, 
                       label=f'Mean: {avg_lag:.1f}s')
            # Annotate HRR60 if available
            if hasattr(self, 'hrr60') and self.hrr60 is not None and not np.isnan(self.hrr60):
                ax3.annotate(f'HRR60: {self.hrr60:.0f} bpm', xy=(avg_lag, ax3.get_ylim()[1]*0.8),
                             xytext=(avg_lag+2, ax3.get_ylim()[1]*0.9),
                             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='black')
            ax3.legend()
        
        # 4. HR Response Amplitude vs Time
        if hr_response_amplitude:
            ax4.scatter(hr_response_amplitude_times, hr_response_amplitude, alpha=0.6, s=40, color='purple')
            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('HR Response Amplitude (bpm)')
            ax4.set_title('HR Response Amplitude Over Time', fontweight='bold', fontsize=14)
            ax4.grid(True, alpha=0.3)
            # Add trend line
            if len(hr_response_amplitude_times) > 5:
                z = np.polyfit(hr_response_amplitude_times, hr_response_amplitude, 1)
                p = np.poly1d(z)
                ax4.plot(hr_response_amplitude_times, p(hr_response_amplitude_times), "r--", alpha=0.8, linewidth=2)
        else:
            ax4.text(0.5, 0.5, 'No HR response\namplitudes detected', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('HR Response Amplitude Over Time', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_heat_stress")
            fig.savefig(fig_path + ".png", dpi=300)
            fig.savefig(fig_path + ".svg")
        else:
            plt.show()
        
        # Print heat stress analysis
        print(f"\n=== HEAT STRESS ANALYSIS ===")
        
        if heat_stress_index:
            avg_heat_stress = np.mean(heat_stress_index)
            max_heat_stress = np.max(heat_stress_index)
            
            print(f"Heat Stress Index:")
            print(f"  - Average: {avg_heat_stress:.1f}/100")
            print(f"  - Maximum: {max_heat_stress:.1f}/100")
            
            # Categorize heat stress
            if avg_heat_stress < 30:
                heat_category = "Low"
            elif avg_heat_stress < 50:
                heat_category = "Moderate"
            elif avg_heat_stress < 70:
                heat_category = "High"
            else:
                heat_category = "Very High"
            print(f"  - Overall Heat Stress: {heat_category}")
        
        if hr_drift_rate:
            avg_drift_rate = np.mean(hr_drift_rate)
            max_drift_rate = np.max(hr_drift_rate)
            
            print(f"\nHR Drift Analysis:")
            print(f"  - Average Drift Rate: {avg_drift_rate:.2f} bpm/min")
            print(f"  - Maximum Drift Rate: {max_drift_rate:.2f} bpm/min")
            
            # Categorize drift
            if avg_drift_rate < 1:
                drift_category = "Minimal"
            elif avg_drift_rate < 3:
                drift_category = "Moderate"
            else:
                drift_category = "Significant"
            print(f"  - Drift Category: {drift_category}")
        
        if hr_response_lag:
            avg_response_lag = np.mean(hr_response_lag)
            avg_response_amplitude = np.mean(hr_response_amplitude)
            
            print(f"\nHR Response Analysis:")
            print(f"  - Average Response Time: {avg_response_lag:.1f} seconds")
            print(f"  - Average Response Amplitude: {avg_response_amplitude:.1f} bpm")
            
            # Categorize response
            if avg_response_lag < 10:
                response_category = "Fast"
            elif avg_response_lag < 20:
                response_category = "Normal"
            else:
                response_category = "Slow"
            print(f"  - Response Category: {response_category}")
        
        # Heat stress recommendations
        print(f"\n--- HEAT STRESS RECOMMENDATIONS ---")
        if heat_stress_index and np.mean(heat_stress_index) > 50:
            print(f"⚠️  High heat stress detected - consider:")
            print(f"  - Hydration monitoring")
            print(f"  - Reduced intensity")
            print(f"  - Cooling strategies")
            print(f"  - Shorter intervals")
        elif hr_drift_rate and np.mean(hr_drift_rate) > 3:
            print(f"⚠️  Significant HR drift detected - consider:")
            print(f"  - Pacing adjustments")
            print(f"  - Recovery periods")
            print(f"  - Heat management")
        else:
            print(f"✅ Heat stress levels appear manageable")
        
        return heat_stress_index, hr_drift_rate, hr_response_lag
    
    def analyze_power_hr_efficiency(self):
        """Analyze power-to-heart rate efficiency over time with slope analysis and filtered data."""
        if self.df is None or 'power' not in self.df.columns or 'heart_rate' not in self.df.columns:
            return
        
        # Calculate power-to-HR ratio (efficiency metric) and filter out zero values
        self.df['power_hr_ratio'] = self.df['power'] / self.df['heart_rate']
        
        # Filter out zero and extreme values for meaningful analysis
        valid_mask = (self.df['power'] > 0) & (self.df['heart_rate'] > 50) & (self.df['power_hr_ratio'] > 0)
        valid_data = self.df[valid_mask].copy()
        
        if len(valid_data) == 0:
            return
        
        # Calculate efficiency over time using rolling averages
        efficiency_rolling = valid_data['power_hr_ratio'].rolling(window=60, min_periods=30).mean()
        
        # Calculate efficiency statistics (using valid data only)
        efficiency_mean = valid_data['power_hr_ratio'].mean()
        efficiency_std = valid_data['power_hr_ratio'].std()
        
        # Calculate efficiency slope over time
        time_minutes = (valid_data['timestamp'] - valid_data['timestamp'].iloc[0]).dt.total_seconds() / 60
        
        # Fit linear trend to efficiency over time
        # Filter out NaN values and align time_minutes with valid efficiency data
        if isinstance(efficiency_rolling, pd.Series):
            efficiency_valid = efficiency_rolling.dropna()
            if len(efficiency_valid) > 10:
                # Get corresponding time values for non-NaN efficiency values
                if hasattr(efficiency_rolling, 'notna'):
                    time_minutes_valid = time_minutes[efficiency_rolling.notna()]
                else:
                    time_minutes_valid = time_minutes
                efficiency_trend = np.polyfit(time_minutes_valid, efficiency_valid, 1)
                efficiency_slope = efficiency_trend[0]  # W/bpm per minute
                trend_label = f"Slope: {efficiency_slope:.3f} W/bpm/min"
            else:
                efficiency_slope = 0
                trend_label = "Insufficient data for trend"
        else:
            efficiency_slope = 0
            trend_label = "Insufficient data for trend"
        
        # Create efficiency zones based on valid data
        efficiency_zones = {
            'Very Low': efficiency_mean - 2*efficiency_std,
            'Low': efficiency_mean - efficiency_std,
            'Moderate': efficiency_mean,
            'High': efficiency_mean + efficiency_std,
            'Very High': efficiency_mean + 2*efficiency_std
        }
        
        # Label efficiency bands with interpretation
        efficiency_band_labels = {
            'Very Low': 'Very Low (fatigue/overreaching)',
            'Low': 'Low (sub-optimal, possible fatigue)',
            'Moderate': 'Moderate (normal endurance)',
            'High': 'High (good aerobic efficiency)',
            'Very High': 'Very High (excellent efficiency)'
        }
        efficiency_categories = []
        for ratio in valid_data['power_hr_ratio']:
            if ratio < efficiency_zones['Low']:
                efficiency_categories.append(efficiency_band_labels['Very Low'])
            elif ratio < efficiency_zones['Moderate']:
                efficiency_categories.append(efficiency_band_labels['Low'])
            elif ratio < efficiency_zones['High']:
                efficiency_categories.append(efficiency_band_labels['Moderate'])
            elif ratio < efficiency_zones['Very High']:
                efficiency_categories.append(efficiency_band_labels['High'])
            else:
                efficiency_categories.append(efficiency_band_labels['Very High'])
        valid_data['efficiency_category'] = efficiency_categories
        efficiency_counts = pd.Series(efficiency_categories).value_counts()
        
        # Create enhanced efficiency analysis visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced Power-to-Heart Rate Efficiency Analysis', fontweight='bold', fontsize=14)
        
        # 1. Efficiency over time with trend line
        ax1.plot(time_minutes, efficiency_rolling, color='blue', linewidth=2, label='Efficiency')
        ax1.axhline(y=efficiency_mean, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean: {efficiency_mean:.2f} W/bpm')
        # Add trend line
        if len(time_minutes) > 10:
            trend_line = efficiency_trend[0] * time_minutes + efficiency_trend[1]
            ax1.plot(time_minutes, trend_line, color='green', linestyle='--', alpha=0.8, 
                    label=trend_label)
        ax1.set_title('Efficiency Over Time with Trend Analysis', fontsize=12)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Efficiency (W/bpm)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Efficiency vs Power (filtered)
        # Use hexbin for density visualization
        hb = ax2.hexbin(valid_data['power'], valid_data['power_hr_ratio'], gridsize=60, cmap='Reds', bins='log')
        ax2.set_xlabel('Power (W)')
        ax2.set_ylabel('Efficiency (W/bpm)')
        ax2.set_title('Efficiency vs Power (Filtered Data)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        cb = plt.colorbar(hb, ax=ax2)
        cb.set_label('log10(N)')
        
        # 3. Efficiency vs Heart Rate (filtered)
        ax3.scatter(valid_data['heart_rate'], valid_data['power_hr_ratio'], alpha=0.5, s=15)
        ax3.set_xlabel('Heart Rate (bpm)')
        ax3.set_ylabel('Efficiency (W/bpm)')
        ax3.set_title('Efficiency vs Heart Rate (Filtered Data)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency zone distribution (filtered data)
        colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90', '#87CEEB']
        ax4.pie(efficiency_counts.values, labels=efficiency_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax4.set_title('Efficiency Zone Distribution (Filtered)')
        
        plt.tight_layout()
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_power_hr_efficiency")
            fig.savefig(fig_path + ".png", dpi=300)
            fig.savefig(fig_path + ".svg")
        else:
            plt.show()
        
        # Print enhanced efficiency analysis
        # Enhanced power-to-HR efficiency analysis completed
        
        # Interpret efficiency trend
                # Efficiency trend analysis completed
        
        # Efficiency zone breakdown
        for zone, count in efficiency_counts.items():
            percentage = (count / len(valid_data)) * 100
            print(f"{zone} Efficiency: {percentage:.1f}% of valid data")
        
        # 3. Efficiency Category Distribution
        efficiency_labels = list(efficiency_band_labels.values())
        efficiency_dist = [efficiency_counts.get(label, 0) for label in efficiency_labels]
        ax3.bar(efficiency_labels, efficiency_dist, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])
        ax3.set_xlabel('Efficiency Category')
        ax3.set_ylabel('Count')
        ax3.set_title('Efficiency Category Distribution', fontsize=12)
        ax3.tick_params(axis='x', rotation=20)
        ax3.grid(True, alpha=0.3)
        
        # Print efficiency band interpretation
        print("\nEfficiency Band Interpretation:")
        for label, desc in efficiency_band_labels.items():
            print(f"  {desc}")
    
    def analyze_fatigue_patterns(self):
        """Analyze fatigue and drift with rolling average and clarify drift sign."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
        
        # Apply 30 s rolling median to power before drift calculation
        self.df['power_median_30s'] = self.df['power'].rolling(window=30, min_periods=1, center=True).median()
        
        # Use at least 20 segments for better resolution
        num_segments = max(20, len(self.df) // 100)  # At least 20 segments, more for longer rides
        segment_length = len(self.df) // num_segments
        segments = []
        
        # Calculate time in minutes
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        
        # Create segments with detailed metrics using median-smoothed power
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length if i < num_segments - 1 else len(self.df)
            segment_data = self.df.iloc[start_idx:end_idx]
            segment_time_mid = segment_data['timestamp'].iloc[len(segment_data)//2]
            time_midpoint = (segment_time_mid - self.df['timestamp'].iloc[0]).total_seconds() / 60
            segments.append({
                'segment': i + 1,
                'avg_power': segment_data['power_median_30s'].mean(),
                'avg_hr': segment_data['heart_rate'].mean() if 'heart_rate' in segment_data.columns else 0,
                'avg_cadence': segment_data['cadence'].mean() if 'cadence' in segment_data.columns else 0,
                'avg_speed': segment_data['speed_kmh'].mean() if 'speed_kmh' in segment_data.columns else 0,
                'time_midpoint': time_midpoint,
                'duration': len(segment_data) / 60  # Duration in minutes
            })
        
        # Calculate HR drift in relation to power output
        if 'heart_rate' in self.df.columns:
            # Calculate power-to-HR ratio (efficiency metric) for each segment
            hr_power_ratios = []
            hr_efficiency_drift = []
            
            for segment in segments:
                start_idx = (segment['segment'] - 1) * segment_length
                end_idx = start_idx + segment_length if segment['segment'] < num_segments else len(self.df)
                segment_data = self.df.iloc[start_idx:end_idx]
                
                # Calculate power-to-HR ratio for this segment
                if segment_data['heart_rate'].mean() > 0:
                    power_hr_ratio = segment_data['power'].mean() / segment_data['heart_rate'].mean()
                    hr_power_ratios.append(power_hr_ratio)
                else:
                    hr_power_ratios.append(np.nan)
                
                # Calculate HR drift relative to power (HR at similar power levels)
                # Use average power as reference point
                target_power = segment_data['power'].mean()
                power_tolerance = target_power * 0.1  # ±10% of segment average power
                
                power_mask = (segment_data['power'] >= target_power - power_tolerance) & \
                            (segment_data['power'] <= target_power + power_tolerance)
                
                if power_mask.sum() > 0:
                    hr_at_similar_power = segment_data.loc[power_mask, 'heart_rate'].mean()
                    hr_efficiency_drift.append(hr_at_similar_power)
                else:
                    hr_efficiency_drift.append(segment_data['heart_rate'].mean())
        
        # Calculate cumulative work done (W/hr) for each segment
        cumulative_work = []
        total_work = 0
        
        for segment in segments:
            # Calculate work done in this segment (power * time in hours)
            segment_work = segment['avg_power'] * segment['duration'] / 60  # Convert to hours
            total_work += segment_work
            cumulative_work.append(total_work)
        
        # Create enhanced fatigue analysis with 4 graphs including W/hr
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Enhanced Fatigue & Drift Analysis (15 Segments)', fontweight='bold', fontsize=16)
        
        # Calculate drift for subtitle
        drift_pct_per_hr = None
        drift_w_per_hr = None
        drift_sign = ''
        if len(segments) > 2:
            time_segments = [seg['time_midpoint'] for seg in segments]
            power_by_segment = [seg['avg_power'] for seg in segments]
            power_trend = np.polyfit(time_segments, power_by_segment, 1)
            drift_sign = '+' if power_trend[0] >= 0 else '−'
            drift_w_per_hr = power_trend[0]
            initial_power = power_by_segment[0]
            final_power = power_by_segment[-1]
            duration_hr = (time_segments[-1] - time_segments[0]) / 60 if (time_segments[-1] - time_segments[0]) > 0 else 1
            drift_pct_per_hr = ((final_power - initial_power) / initial_power * 100) / duration_hr if initial_power > 0 else 0
            subtitle = f"{drift_sign}{abs(drift_pct_per_hr):.1f} % h⁻¹ ({drift_sign}{abs(drift_w_per_hr):.2f} W h⁻¹)"
        else:
            subtitle = None
        fig.suptitle('Enhanced Fatigue & Drift Analysis (15 Segments)' + (f"\n{subtitle}" if subtitle else ''), fontweight='bold', fontsize=16)
        
        # 1. Power drift over time with line of best fit
        time_segments = [seg['time_midpoint'] for seg in segments]
        power_by_segment = [seg['avg_power'] for seg in segments]
        
        ax1.plot(time_segments, power_by_segment, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Power')
        
        # Add line of best fit
        if len(time_segments) > 2:
            power_trend = np.polyfit(time_segments, power_by_segment, 1)
            power_trend_line = power_trend[0] * np.array(time_segments) + power_trend[1]
            drift_sign = '+' if power_trend[0] >= 0 else '−'
            ax1.plot(time_segments, power_trend_line, '--', color='red', linewidth=2, 
                    label=f'Trend: {drift_sign}{abs(power_trend[0]):.1f} W h⁻¹')
        
        ax1.set_title('Power Drift Over Time')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Power (W)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. HR Efficiency Drift (Power-to-HR ratio over time)
        if 'heart_rate' in self.df.columns and len(hr_power_ratios) > 0:
            # Remove NaN values for trend calculation
            valid_indices = [i for i, ratio in enumerate(hr_power_ratios) if not np.isnan(ratio)]
            if len(valid_indices) > 2:
                valid_times = [time_segments[i] for i in valid_indices]
                valid_ratios = [hr_power_ratios[i] for i in valid_indices]
                
                ax2.plot(valid_times, valid_ratios, 'o-', color='#d62728', linewidth=2, markersize=8, label='Power/HR Ratio')
                
                # Add line of best fit
                ratio_trend = np.polyfit(valid_times, valid_ratios, 1)
                ratio_trend_line = ratio_trend[0] * np.array(valid_times) + ratio_trend[1]
                ax2.plot(valid_times, ratio_trend_line, '--', color='red', linewidth=2,
                        label=f'Trend: {ratio_trend[0]:.3f} W/bpm/min')
                
                ax2.set_title('HR Efficiency Drift (Power/HR Ratio)')
                ax2.set_xlabel('Time (minutes)')
                ax2.set_ylabel('Power/HR Ratio (W/bpm)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative Work Done (W/hr) with trend line
        ax3.plot(time_segments, cumulative_work, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Cumulative Work')
        
        # Add line of best fit for cumulative work
        if len(time_segments) > 2:
            work_trend = np.polyfit(time_segments, cumulative_work, 1)
            work_trend_line = work_trend[0] * np.array(time_segments) + work_trend[1]
            ax3.plot(time_segments, work_trend_line, '--', color='red', linewidth=2,
                    label=f'Trend: {work_trend[0]:.1f} kJ/min')
        
        ax3.set_title('Cumulative Work Done (kJ)')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Cumulative Work (kJ)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cadence drift over time with line of best fit
        if 'cadence' in self.df.columns:
            cadence_by_segment = [seg['avg_cadence'] for seg in segments]
            
            ax4.plot(time_segments, cadence_by_segment, 'o-', color='#9467bd', linewidth=2, markersize=8, label='Cadence')
            
            # Add line of best fit
            if len(time_segments) > 2:
                cadence_trend = np.polyfit(time_segments, cadence_by_segment, 1)
                cadence_trend_line = cadence_trend[0] * np.array(time_segments) + cadence_trend[1]
                ax4.plot(time_segments, cadence_trend_line, '--', color='red', linewidth=2,
                        label=f'Trend: {cadence_trend[0]:.2f} rpm/min')
            
            ax4.set_title('Cadence Drift Over Time')
            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('Cadence (rpm)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_fatigue_patterns")
            fig.savefig(fig_path + ".png", dpi=300)
            fig.savefig(fig_path + ".svg")
        else:
            plt.show()
        
        # Print enhanced fatigue analysis
        print(f"\n=== Enhanced Fatigue & Drift Analysis (15 Segments) ===")
        # Calculate overall trends
        if len(time_segments) > 2:
            power_trend = np.polyfit(time_segments, power_by_segment, 1)
            # Power Drift: ±X W h⁻¹ and %·h⁻¹
            drift_sign = '+' if power_trend[0] >= 0 else '−'
            drift_w_per_hr = power_trend[0]
            initial_power = power_by_segment[0]
            final_power = power_by_segment[-1]
            duration_hr = (time_segments[-1] - time_segments[0]) / 60 if (time_segments[-1] - time_segments[0]) > 0 else 1
            drift_pct_per_hr = ((final_power - initial_power) / initial_power * 100) / duration_hr if initial_power > 0 else 0
            print(f"Power Drift: {drift_sign}{abs(drift_w_per_hr):.2f} W h⁻¹ ({drift_pct_per_hr:+.2f}%·h⁻¹)")
            # Calculate cumulative work trend
            work_trend = np.polyfit(time_segments, cumulative_work, 1)
            # Total Work: Y kJ
            total_work_kj = cumulative_work[-1]
            avg_work_rate_kj_min = total_work_kj / time_segments[-1] if time_segments[-1] > 0 else 0
            avg_work_rate_w = avg_work_rate_kj_min * 1000 / 60
            print(f"Total Work: {total_work_kj:.1f} kJ")
            print(f"Average Work Rate: {avg_work_rate_kj_min:.2f} kJ·min⁻¹ (≈{avg_work_rate_w:.0f} W)")
        
        # Segment-by-segment breakdown (first 5 and last 5 segments)
        print(f"\n--- SEGMENT ANALYSIS ---")
        for i, segment in enumerate(segments):
            if i < 5 or i >= len(segments) - 5:  # Show first 5 and last 5 segments
                print(f"Segment {segment['segment']} ({segment['time_midpoint']:.1f} min):")
                print(f"  Power: {segment['avg_power']:.0f}W")
                if 'heart_rate' in self.df.columns:
                    print(f"  HR: {segment['avg_hr']:.0f}bpm")
                if 'cadence' in self.df.columns:
                    print(f"  Cadence: {segment['avg_cadence']:.0f}rpm")
                if 'speed_kmh' in self.df.columns:
                    print(f"  Speed: {segment['avg_speed']:.1f}km/h")
            elif i == 5:
                print("  ... (middle segments omitted for brevity)")
        
        # Apply rolling average to power before drift calculation
        self.df['power_rolling'] = self.df['power'].rolling(window=60, min_periods=30).mean()
        # Calculate drift rate as the difference between last and first segment means, per hour
        segment_size = len(self.df) // 10  # 10 segments
        if segment_size > 0:
            first_mean = self.df['power_rolling'].iloc[:segment_size].mean()
            last_mean = self.df['power_rolling'].iloc[-segment_size:].mean()
            duration_hr = (self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]).total_seconds() / 3600
            drift_rate = (last_mean - first_mean) / duration_hr if duration_hr > 0 else 0
            if drift_rate < 0:
                print(f"Power Drift Rate: {drift_rate:.2f} W/hr (negative: early data may be inflated)")
            else:
                print(f"Power Drift Rate: {drift_rate:.2f} W/hr")
        else:
            print('Not enough data to calculate drift rate.')
    
    def analyze_variable_relationships(self):
        """Analyze relationships between different variables including elevation/grade."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
        
        # Check for elevation/grade data
        elevation_available = 'altitude' in self.df.columns or 'enhanced_altitude' in self.df.columns
        grade_available = 'grade' in self.df.columns or 'enhanced_grade' in self.df.columns
        
        # Calculate grade from altitude if not available
        if not grade_available and elevation_available and 'distance' in self.df.columns:
            alt_col = 'altitude' if 'altitude' in self.df.columns else 'enhanced_altitude'
            self.df['calculated_grade'] = self.df[alt_col].diff() / self.df['distance'].diff() * 100
            grade_available = True
        
        # Calculate correlations
        variables = ['power']
        
        if 'heart_rate' in self.df.columns:
            variables.append('heart_rate')
        if 'cadence' in self.df.columns:
            variables.append('cadence')
        if 'speed_kmh' in self.df.columns:
            variables.append('speed_kmh')
        if grade_available:
            grade_col = 'grade' if 'grade' in self.df.columns else 'enhanced_grade' if 'enhanced_grade' in self.df.columns else 'calculated_grade'
            variables.append(grade_col)
        
        # Calculate correlation matrix
        corr_matrix = None
        if self.df is not None and hasattr(self.df, 'corr'):
            try:
                corr_matrix = self.df[variables].corr(method='pearson')
            except Exception:
                corr_matrix = None
        
        # Create relationship analysis visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Variable Relationship Analysis (with Elevation/Grade)', fontweight='bold', fontsize=14)
        
        # 1. Correlation heatmap
        if corr_matrix is not None:
            im = ax1.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax1.set_xticks(range(len(variables)))
            ax1.set_yticks(range(len(variables)))
            ax1.set_xticklabels(variables)
            ax1.set_yticklabels(variables)
            ax1.set_title('Correlation Matrix')
            
            # Add correlation values
            for i in range(len(variables)):
                for j in range(len(variables)):
                    text = ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=ax1)
        
        # 2. Power vs HR relationship with trend line (if available)
        if 'heart_rate' in self.df.columns:
            # Filter out zero values for meaningful analysis
            hr_mask = (self.df['power'] > 0) & (self.df['heart_rate'] > 50)
            if hr_mask.sum() > 10:
                power_hr = self.df[hr_mask]['power']
                hr_data = self.df[hr_mask]['heart_rate']
                
                ax2.scatter(power_hr, hr_data, alpha=0.5, s=15)
                
                # Add trend line
                hr_trend = np.polyfit(power_hr, hr_data, 1)
                hr_line = hr_trend[0] * power_hr + hr_trend[1]
                ax2.plot(power_hr, hr_line, color='red', linestyle='--', alpha=0.8, 
                        label=f'Slope: {hr_trend[0]:.3f} bpm/W')
                ax2.legend()
            
            ax2.set_xlabel('Power (W)')
            ax2.set_ylabel('Heart Rate (bpm)')
            ax2.set_title('Power vs Heart Rate')
            ax2.grid(True, alpha=0.3)
        
        # 3. Power vs Grade relationship (if available)
        if grade_available:
            grade_col = 'grade' if 'grade' in self.df.columns else 'enhanced_grade' if 'enhanced_grade' in self.df.columns else 'calculated_grade'
            # Filter out extreme values
            grade_mask = (self.df['power'] > 0) & (self.df[grade_col].abs() < 20)  # Exclude extreme grades
            if grade_mask.sum() > 10:
                power_grade = self.df[grade_mask]['power']
                grade_data = self.df[grade_mask][grade_col]
                
                ax3.scatter(grade_data, power_grade, alpha=0.5, s=15)
                
                # Add trend line
                grade_trend = np.polyfit(grade_data, power_grade, 1)
                grade_line = grade_trend[0] * grade_data + grade_trend[1]
                ax3.plot(grade_data, grade_line, color='red', linestyle='--', alpha=0.8,
                        label=f'Slope: {grade_trend[0]:.1f} W/% grade')
                ax3.legend()
            ax3.set_xlabel('Grade (%)')
            ax3.set_ylabel('Power (W)')
            ax3.set_title('Power vs Grade')
            ax3.grid(True, alpha=0.3)

        # 3b. Power/mass vs Grade with slope annotation (if weight_kg and grade available)
        if grade_available and hasattr(self, 'weight_kg') and self.weight_kg > 0:
            grade_col = 'grade' if 'grade' in self.df.columns else 'enhanced_grade' if 'enhanced_grade' in self.df.columns else 'calculated_grade'
            pmask = (self.df['power'] > 0) & (self.df[grade_col].abs() < 20)
            if pmask.sum() > 10:
                power_mass = self.df[pmask]['power'] / self.weight_kg
                grade_vals = self.df[pmask][grade_col]
                # Add to ax3 as overlay (or create a new plot if desired)
                ax3.scatter(grade_vals, power_mass, alpha=0.5, s=15, color='purple', label='Power/mass')
                # Fit and annotate slope
                pm_trend = np.polyfit(grade_vals, power_mass, 1)
                pm_line = pm_trend[0] * grade_vals + pm_trend[1]
                ax3.plot(grade_vals, pm_line, color='purple', linestyle='--', alpha=0.8, label=f'Slope (W/kg/%): {pm_trend[0]:.3f}')
                # Annotate slope on plot
                ax3.annotate(f'Slope: {pm_trend[0]:.3f} W/kg/%', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=16, color='purple', ha='left', va='top', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
                ax3.legend()
        
        # 4. CLEANER Pw:Hr DECOUPLING (20-min window, coasting removed)
        # 1)  Keep only pedalling samples (power ≥ 20 W) to avoid zero-power artefacts
        df_eff = self.df[self.df['power'] >= 20].copy()
        # 2) Rebuild time axis (min from ride start)
        time_min_eff = (df_eff['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        # 3) Rolling Efficiency Factor and baseline after warm-up (≥ 20 min)
        window = 20 * 60                     # 20-min window in seconds
        ef      = df_eff['power'] / df_eff['heart_rate']
        ef_roll = ef.rolling(window, min_periods=window).median()
        if ef_roll[time_min_eff >= 20].dropna().size > 0:
            baseline = ef_roll[time_min_eff >= 20].dropna().iloc[0]
            decoup   = (ef_roll - baseline) / baseline * 100   # % drift
            ax4.clear()
            ax4.plot(time_min_eff, decoup, color='purple', lw=1.2, label='Decoupling')
            ax4.axhline( 5, ls='--', color='red', lw=1)
            ax4.axhline(-5, ls='--', color='red', lw=1, label='±5 % threshold')
            ax4.set_title('Rolling 20-min Pw:Hr Decoupling (P ≥ 20 W)')
            ax4.set_xlabel('Time (min)')
            ax4.set_ylabel('Pw:Hr Decoupling (%)')
            ax4.legend(frameon=False)
            ax4.grid(alpha=0.3)
        fig.tight_layout()
        
        plt.tight_layout()
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_variable_relationships")
            fig.savefig(fig_path + ".png", dpi=300)
            fig.savefig(fig_path + ".svg")
        else:
            plt.show()
        
        # Print relationship analysis
        print(f"\n=== VARIABLE RELATIONSHIP ANALYSIS (with Elevation/Grade) ===")
        print("Correlation Matrix:")
        print(corr_matrix.round(3))
        
        # Key insights
        if 'heart_rate' in self.df.columns:
            hr_power_corr = corr_matrix.loc['power', 'heart_rate']
            print(f"\nPower-HR Correlation: {hr_power_corr:.3f}")
            if hr_power_corr > 0.7:
                print("  Strong positive correlation - HR closely follows power")
            elif hr_power_corr > 0.4:
                print("  Moderate positive correlation - HR generally follows power")
            else:
                print("  Weak correlation - HR and power not strongly related")
        
        if grade_available:
            grade_col = 'grade' if 'grade' in self.df.columns else 'enhanced_grade' if 'enhanced_grade' in self.df.columns else 'calculated_grade'
            grade_power_corr = corr_matrix.loc['power', grade_col]
            print(f"Power-Grade Correlation: {grade_power_corr:.3f}")
            if grade_power_corr > 0.3:
                print("  Positive correlation - higher power on steeper grades")
            elif grade_power_corr < -0.3:
                print("  Negative correlation - lower power on steeper grades")
            else:
                print("  Weak correlation - power and grade not strongly related")
        
        if 'cadence' in self.df.columns:
            cadence_power_corr = corr_matrix.loc['power', 'cadence']
            print(f"Power-Cadence Correlation: {cadence_power_corr:.3f}")
            if cadence_power_corr > 0.3:
                print("  Positive correlation - higher cadence with higher power")
            elif cadence_power_corr < -0.3:
                print("  Negative correlation - lower cadence with higher power")
            else:
                print("  Weak correlation - cadence and power not strongly related")

    def print_comprehensive_metrics_table(self):
        """Print a comprehensive metrics table with proper formatting."""
        print(f"\n{'COMPREHENSIVE METRICS TABLE':^80}")
        print("=" * 80)
        print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
        print("-" * 80)
        
        # Power Metrics
        if hasattr(self, 'avg_power') and self.avg_power is not None:
            print(f"{'Average Power':<30} {self.avg_power:<20.0f} {'W':<15} {'':<15}")
        if hasattr(self, 'np_calc') and self.np_calc is not None:
            print(f"{'Normalized Power':<30} {self.np_calc:<20.0f} {'W':<15} {'':<15}")
        if self.df is not None and 'power' in self.df.columns:
            print(f"{'Max Power':<30} {self.df['power'].max():<20.0f} {'W':<15} {'':<15}")
            print(f"{'Min Power':<30} {self.df['power'].min():<20.0f} {'W':<15} {'':<15}")
        
        # Check if VI is available
        if hasattr(self, 'VI') and self.VI is not None:
            print(f"{'Power Variability Index':<30} {self.VI:<20.2f} {'':<15} {'NP/AP':<15}")
        
        # FTP-based metrics
        if hasattr(self, 'ftp') and self.ftp > 0:
            if hasattr(self, 'IF') and self.IF is not None:
                print(f"{'Intensity Factor':<30} {self.IF:<20.2f} {'':<15} {'NP/FTP':<15}")
            if hasattr(self, 'TSS') and self.TSS is not None:
                print(f"{'Training Stress Score':<30} {self.TSS:<20.0f} {'':<15} {'':<15}")
            if hasattr(self, 'avg_power') and self.avg_power is not None:
                print(f"{'Power at FTP %':<30} {(self.avg_power/self.ftp*100):<20.1f} {'%':<15} {'':<15}")
        
        # Heart Rate Metrics
        if self.df is not None and 'heart_rate' in self.df.columns:
            print(f"\n{'HEART RATE METRICS':^80}")
            print("-" * 80)
            print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
            print("-" * 80)
            avg_hr = self.metrics.get('hr', {}).get('avg', 0) if hasattr(self, 'metrics') else 0
            max_hr = self.metrics.get('hr', {}).get('max', 0) if hasattr(self, 'metrics') else 0
            min_hr = self.metrics.get('hr', {}).get('min', 0) if hasattr(self, 'metrics') else 0
            print(f"{'Average HR':<30} {avg_hr:<20.0f} {'bpm':<15} {'':<15}")
            print(f"{'Max HR':<30} {max_hr:<20.0f} {'bpm':<15} {'':<15}")
            print(f"{'Min HR':<30} {min_hr:<20.0f} {'bpm':<15} {'':<15}")
            if hasattr(self, 'max_hr') and self.max_hr > 0:
                hr_reserve = (max_hr - self.rest_hr) / (self.max_hr - self.rest_hr) * 100 if (self.max_hr - self.rest_hr) > 0 else 0
                avg_hr_reserve = (avg_hr - self.rest_hr) / (self.max_hr - self.rest_hr) * 100 if (self.max_hr - self.rest_hr) > 0 else 0
                print(f"{'Avg HR Reserve':<30} {avg_hr_reserve:<20.1f} {'%':<15} {'':<15}")
        
        # Cadence Metrics
        if self.df is not None and 'cadence' in self.df.columns:
            print(f"\n{'CADENCE METRICS':^80}")
            print("-" * 80)
            print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
            print("-" * 80)
            avg_cadence = self.metrics.get('cadence', {}).get('avg', 0) if hasattr(self, 'metrics') else 0
            max_cadence = self.metrics.get('cadence', {}).get('max', 0) if hasattr(self, 'metrics') else 0
            min_cadence_val = self.df['cadence'].replace(0, np.nan).min() if 'cadence' in self.df.columns else np.nan
            min_cadence = min_cadence_val if not np.isnan(min_cadence_val) else '—'
            avg_cadence_str = f"{avg_cadence:.0f}" if isinstance(avg_cadence, (int, float)) else ""
            max_cadence_str = f"{max_cadence:.0f}" if isinstance(max_cadence, (int, float)) else ""
            min_cadence_str = f"{min_cadence:.0f}" if isinstance(min_cadence, (int, float)) and min_cadence != '—' else (min_cadence if min_cadence == '—' else "")
            print(f"{'Average Cadence':<30} {avg_cadence_str:<20} {'rpm':<15} {'':<15}")
            print(f"{'Max Cadence':<30} {max_cadence_str:<20} {'rpm':<15} {'':<15}")
            print(f"{'Min Cadence':<30} {min_cadence_str:<20} {'rpm':<15} {'':<15}")
        
        # Speed Metrics
        if self.df is not None and 'speed_kmh' in self.df.columns:
            print(f"\n{'SPEED METRICS':^80}")
            print("-" * 80)
            print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
            print("-" * 80)
            avg_speed = self.metrics.get('speed', {}).get('avg', 0) if hasattr(self, 'metrics') else 0
            max_speed = self.metrics.get('speed', {}).get('max', 0) if hasattr(self, 'metrics') else 0
            min_speed_val = self.df['speed_kmh'].replace(0, np.nan).min() if 'speed_kmh' in self.df.columns else np.nan
            min_speed = min_speed_val if not np.isnan(min_speed_val) else '—'
            avg_speed_str = f"{avg_speed:.1f}" if isinstance(avg_speed, (int, float)) else ""
            max_speed_str = f"{max_speed:.1f}" if isinstance(max_speed, (int, float)) else ""
            min_speed_str = f"{min_speed:.1f}" if isinstance(min_speed, (int, float)) and min_speed != '—' else (min_speed if min_speed == '—' else "")
            print(f"{'Average Speed':<30} {avg_speed_str:<20} {'km/h':<15} {'':<15}")
            print(f"{'Max Speed':<30} {max_speed_str:<20} {'km/h':<15} {'':<15}")
            print(f"{'Min Speed':<30} {min_speed_str:<20} {'km/h':<15} {'':<15}")
        
        # Time and Distance Metrics
        print(f"\n{'TIME & DISTANCE METRICS':^80}")
        print("-" * 80)
        print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
        print("-" * 80)
        if hasattr(self, 'duration_hr') and self.duration_hr is not None:
            print(f"{'Duration':<30} {self.duration_hr:<20.2f} {'hours':<15} {'':<15}")
            print(f"{'Duration':<30} {(self.duration_hr*60):<20.0f} {'minutes':<15} {'':<15}")
        if self.df is not None and 'distance_km' in self.df.columns:
            print(f"{'Distance':<30} {self.df['distance_km'].max():<20.2f} {'km':<15} {'':<15}")
            if hasattr(self, 'duration_hr') and self.duration_hr:
                print(f"{'Average Speed':<30} {(self.df['distance_km'].max()/self.duration_hr):<20.1f} {'km/h':<15} {'':<15}")
        
        # Zone Analysis
        if hasattr(self, 'zone_percentages') and self.zone_percentages:
            print(f"\n{'ZONE ANALYSIS':^80}")
            print("-" * 80)
            print(f"{'Zone':<20} {'Time':<15} {'%':<10} {'Description':<35}")
            print("-" * 80)
            for zone_name, percentage in self.zone_percentages.items():
                if percentage > 0:
                    time_minutes = (percentage / 100) * self.duration_hr * 60 if hasattr(self, 'duration_hr') and self.duration_hr else 0
                    print(f"{zone_name:<20} {time_minutes:<15.1f} {percentage:<10.1f} {'':<35}")
        
        # HR Zone Analysis
        if self.df is not None and 'heart_rate' in self.df.columns and hasattr(self, 'hr_zones') and hasattr(self, 'hr_zone_percentages'):
            print(f"\n{'HR ZONE ANALYSIS':^80}")
            print("-" * 80)
            print(f"{'Zone':<20} {'Time':<15} {'%':<10} {'Description':<35}")
            print("-" * 80)
            for zone_name, percentage in self.hr_zone_percentages.items():
                if percentage > 0:
                    time_minutes = (percentage / 100) * self.duration_hr * 60 if hasattr(self, 'duration_hr') and self.duration_hr else 0
                    print(f"{zone_name:<20} {time_minutes:<15.1f} {percentage:<10.1f} {'':<35}")
        
        print("=" * 80)

    def estimate_lactate(self):
        """Estimate lactate levels throughout the ride with proper physiological modeling."""
        if self.df is None or 'power' not in self.df.columns:
            print("No data loaded or missing power data. Please load a FIT file first.")
            return

        def estimate_lactate_func(power, ftp):
            """Physiologically accurate lactate estimation based on power zones."""
            lactate_rest = 1.2
            if power <= ftp * 0.55:
                return lactate_rest + (power / (ftp * 0.55)) * 0.3
            elif power <= ftp * 0.75:
                return lactate_rest + 0.3 + (power - ftp * 0.55) / (ftp * 0.2) * 0.7
            elif power <= ftp * 0.9:
                return lactate_rest + 1.0 + (power - ftp * 0.75) / (ftp * 0.15) * 1.0
            elif power <= ftp:
                return lactate_rest + 2.0 + (power - ftp * 0.9) / (ftp * 0.1) * 1.5
            elif power <= ftp * 1.05:
                return lactate_rest + 3.5 + (power - ftp) / (ftp * 0.05) * 1.0
            elif power <= ftp * 1.2:
                return lactate_rest + 4.5 + (power - ftp * 1.05) / (ftp * 0.15) * 2.0
            else:
                return lactate_rest + 6.5 + (power - ftp * 1.2) / (ftp * 0.3) * 3.0

        # Calculate raw lactate estimates
        self.df['lactate_est'] = self.df['power'].apply(lambda p: estimate_lactate_func(p, self.ftp) if pd.notnull(p) and self.ftp > 0 else np.nan)
        
        # Apply exponential moving average (EMA) smoothing to simulate physiological lactate delay
        # Based on Newell et al., 2007 (Comput Methods Programs Biomed) and Beneke et al., 2001 (Med Sci Sports Exerc)
        # EMA with span=60 reflects realistic lactate accumulation and clearance lag
        # This models the physiological delay in blood lactate response to exercise intensity changes
        self.df['lactate_smoothed'] = self.df['lactate_est'].ewm(span=60, adjust=False).mean()

        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60 if 'timestamp' in self.df.columns else np.arange(len(self.df))

        # Create enhanced lactate plot
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Main lactate plot with zones
        ax1.plot(time_minutes, self.df['lactate_smoothed'], color='#d62728', linewidth=2, label='Estimated Lactate')
        ax1.fill_between(time_minutes, self.df['lactate_smoothed'], alpha=0.3, color='#d62728')
        ax1.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Aerobic Threshold (~2.0)')
        ax1.axhline(y=4.0, color='orange', linestyle='--', alpha=0.7, label='Lactate Threshold (~4.0)')
        ax1.axhline(y=8.0, color='red', linestyle='--', alpha=0.7, label='Onset of Blood Lactate (~8.0)')
        ax1.set_title('Estimated Blood Lactate Response', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Lactate (mmol/L)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 12)
        # Only shade area >=8 mmol/L for OBLA
        ax1.fill_between(time_minutes, 8, 12, alpha=0.15, color='red', label='OBLA (≥8 mmol/L)')
        # Add on-plot note about modelled values
        ax1.text(0.98, 0.02, 'Lactate values modelled (not blood-sampled)',
                 transform=ax1.transAxes, fontsize=10, color='gray', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Power vs Lactate scatter
        scatter = ax2.scatter(self.df['power'], self.df['lactate_smoothed'], alpha=0.6, c=time_minutes, cmap='viridis', s=20)
        ax2.set_xlabel('Power (W)')
        ax2.set_ylabel('Lactate (mmol/L)')
        ax2.set_title('Power vs Lactate Relationship')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=self.ftp, color='red', linestyle='--', alpha=0.7, label=f'FTP ({self.ftp}W)')
        ax2.legend()
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Time (minutes)')
        # Limit x-axis to max(power)+10 W
        max_power = np.nanmax(self.df['power']) if 'power' in self.df.columns else None
        if max_power is not None:
            ax2.set_xlim(0, max_power + 10)
        plt.tight_layout()
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_lactate")
            fig.savefig(fig_path + ".png", dpi=300)
            fig.savefig(fig_path + ".svg")
        else:
            plt.show()

        # Print lactate statistics
        print(f"\n=== LACTATE ANALYSIS ===")
        print(f"Average Lactate: {self.df['lactate_smoothed'].mean():.2f} mmol/L")
        print(f"Peak Lactate: {self.df['lactate_smoothed'].max():.2f} mmol/L")
        print(f"Time above 4.0 mmol/L: {((self.df['lactate_smoothed'] > 4.0).sum() / len(self.df) * 100):.1f}%")
        print(f"Time above 8.0 mmol/L: {((self.df['lactate_smoothed'] > 8.0).sum() / len(self.df) * 100):.1f}%")
        print("Note: RMSSD is skipped unless RR intervals are available. Consider DFA-α1 if RR is present.")

    def analyze_torque(self):
        """Analyze torque vs cadence relationship."""
        if self.df is None or 'power' not in self.df.columns or 'cadence' not in self.df.columns:
            print("No data loaded or missing power/cadence data.")
            return

        # Filter out rows where cadence < 5 rpm
        valid_mask = (self.df['cadence'] >= 5)
        df_valid = self.df[valid_mask].copy()
        # Avoid division by zero for cadence
        df_valid['torque'] = (df_valid['power'] * 60) / (2 * np.pi * df_valid['cadence'])

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Torque vs Cadence
        scatter = ax1.scatter(df_valid['cadence'], df_valid['torque'], alpha=0.3, c=df_valid['power'], cmap='viridis')
        ax1.set_xlabel('Cadence (rpm)')
        ax1.set_ylabel('Torque (Nm)')
        ax1.set_title('Torque vs Cadence')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Power (W)')

        # Power distribution histogram
        ax2.hist(df_valid['power'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(self.avg_power, color='red', linestyle='--', label=f'Avg: {self.avg_power:.0f}W')
        ax2.axvline(self.np_calc, color='orange', linestyle='--', label=f'NP: {self.np_calc:.0f}W')
        ax2.set_xlabel('Power (W)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Power Distribution')
        ax2.legend()

        plt.tight_layout()
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_torque")
            fig.savefig(fig_path + ".png", dpi=300)
            fig.savefig(fig_path + ".svg")
        else:
            plt.show()

    def print_insights(self):
        """Print performance insights and recommendations."""
        if not hasattr(self, 'TSS'):
            print("Metrics not calculated. Please run calculate_metrics() first.")
            return

        print("=== PERFORMANCE INSIGHTS ===")

        # Training load assessment
        if self.TSS < 50:
            load_assessment = "Recovery ride - good for active recovery"
        elif self.TSS < 100:
            load_assessment = "Moderate training load - good for base building"
        elif self.TSS < 150:
            load_assessment = "High training load - good for fitness building"
        else:
            load_assessment = "Very high training load - ensure adequate recovery"

        # Pacing assessment
        pacing_quality = 'Poor'
        if hasattr(self, 'avg_power') and hasattr(self, 'np_calc') and self.np_calc is not None and self.avg_power is not None:
            try:
                ratio = float(self.avg_power) / float(self.np_calc) if float(self.np_calc) != 0 else 0
                if ratio > 0.9:
                    pacing_quality = 'Good'
                elif ratio > 0.8:
                    pacing_quality = 'Moderate'
            except Exception:
                pacing_quality = 'Poor'
        else:
            pacing_assessment = "Insufficient data for pacing assessment"

        # Zone distribution insights
        if hasattr(self, 'zone_percentages') and self.zone_percentages:
            primary_zone = max(self.zone_percentages, key=self.zone_percentages.get)
            zone_percentage = self.zone_percentages[primary_zone]
        else:
            primary_zone = "N/A"
            zone_percentage = 0

        print(f"\n📊 Training Load: {load_assessment}")
        print(f"📈 Pacing Quality: {pacing_quality}")
        print(f"🎯 Primary Zone: {primary_zone} ({zone_percentage:.1f}% of time)")

        if 'heart_rate' in getattr(self, 'df', {}).columns:
            hr_avg = self.metrics.get('hr', {}).get('avg', None)
            if hr_avg is not None and hasattr(self, 'rest_hr') and hasattr(self, 'max_hr') and (self.max_hr - self.rest_hr) > 0:
                hr_intensity = (hr_avg - self.rest_hr) / (self.max_hr - self.rest_hr) * 100
                print(f"💓 Heart Rate Intensity: {hr_intensity:.1f}% of HR reserve")

        print(f"\n💡 Recommendations:")
        if hasattr(self, 'TSS') and hasattr(self, 'target_tss'):
            if self.TSS > self.target_tss * 1.2:
                print("   - Consider reducing intensity in future sessions")
            elif self.TSS < self.target_tss * 0.8:
                print("   - Could increase intensity for target training load")

        if hasattr(self, 'avg_power') and hasattr(self, 'np_calc') and self.np_calc > 0 and (self.avg_power / self.np_calc) < 0.8:
            print("   - Focus on more consistent pacing to improve efficiency")

        if hasattr(self, 'zone_percentages') and self.zone_percentages.get('Z7 (Neuromuscular)', 0) > 5:
            print("   - High neuromuscular load - ensure adequate recovery")

        print("\n" + "="*50)

    def set_lactate_coeffs(self, a, b, c):
        """Set user-supplied lactate-power curve coefficients for personalisation."""
        self.lactate_coeffs = (a, b, c)

    def create_interactive_professional_graph(self):
        """Create an interactive professional multi-axis graph using Plotly with multiple variables on different axes."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
            
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
        except ImportError:
            print("Plotly not available. Please install with: pip install plotly")
            return
            
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60 if 'timestamp' in self.df.columns else np.arange(len(self.df))
        
        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,  # Increased spacing
            subplot_titles=('Power Profile', 'Heart Rate Profile', 'Cadence Profile', 'Speed/Elevation Profile'),
            row_heights=[0.45, 0.2, 0.2, 0.15]  # Power gets even more space
        )
        
        # 1. Power (Primary metric - largest subplot)
        if 'power' in self.df.columns:
            # Raw power as background
            fig.add_trace(
                go.Scatter(
                    x=time_minutes,
                    y=self.df['power'],
                    mode='lines',
                    name='Raw Power',
                    line=dict(color='#1f77b4', width=1),
                    opacity=0.3,
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Smoothed power as main line
            if 'power_smoothed' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=time_minutes,
                        y=self.df['power_smoothed'],
                        mode='lines',
                        name='Power',
                        line=dict(color='#1f77b4', width=2),
                        opacity=0.9,
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Add average and NP lines
            if hasattr(self, 'avg_power') and self.avg_power is not None:
                fig.add_hline(
                    y=self.avg_power,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.8,
                    annotation_text=f"Avg: {self.avg_power:.0f}W",
                    row=1, col=1
                )
            
            if hasattr(self, 'np_calc') and self.np_calc is not None:
                fig.add_hline(
                    y=self.np_calc,
                    line_dash="dash",
                    line_color="orange",
                    opacity=0.8,
                    annotation_text=f"NP: {self.np_calc:.0f}W",
                    row=1, col=1
                )
        
        # 2. Heart Rate
        if 'heart_rate' in self.df.columns:
            # Raw HR as background
            fig.add_trace(
                go.Scatter(
                    x=time_minutes,
                    y=self.df['heart_rate'],
                    mode='lines',
                    name='Raw HR',
                    line=dict(color='#d62728', width=1),
                    opacity=0.3,
                    showlegend=True
                ),
                row=2, col=1
            )
            
            # Smoothed HR as main line
            if 'heart_rate_smoothed' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=time_minutes,
                        y=self.df['heart_rate_smoothed'],
                        mode='lines',
                        name='Heart Rate',
                        line=dict(color='#d62728', width=2),
                        opacity=0.9,
                        showlegend=True
                    ),
                    row=2, col=1
                )
            
            # Add average HR line
            if 'heart_rate' in self.metrics and 'avg' in self.metrics['heart_rate']:
                avg_hr = self.metrics['heart_rate']['avg']
                fig.add_hline(
                    y=avg_hr,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.8,
                    annotation_text=f"Avg: {avg_hr:.0f}bpm",
                    row=2, col=1
                )
        
        # 3. Cadence
        if 'cadence' in self.df.columns:
            # Raw cadence as background
            fig.add_trace(
                go.Scatter(
                    x=time_minutes,
                    y=self.df['cadence'],
                    mode='lines',
                    name='Raw Cadence',
                    line=dict(color='#2ca02c', width=1),
                    opacity=0.3,
                    showlegend=True
                ),
                row=3, col=1
            )
            
            # Smoothed cadence as main line
            if 'cadence_smoothed' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=time_minutes,
                        y=self.df['cadence_smoothed'],
                        mode='lines',
                        name='Cadence',
                        line=dict(color='#2ca02c', width=2),
                        opacity=0.9,
                        showlegend=True
                    ),
                    row=3, col=1
                )
            
            # Add average cadence line
            if 'cadence' in self.metrics and 'avg' in self.metrics['cadence']:
                avg_cadence = self.metrics['cadence']['avg']
                fig.add_hline(
                    y=avg_cadence,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.8,
                    annotation_text=f"Avg: {avg_cadence:.0f}rpm",
                    row=3, col=1
                )
        
        # 4. Speed (if available) or Elevation
        if 'speed' in self.df.columns:
            # Convert speed to km/h if it's in m/s
            speed_data = self.df['speed']
            if speed_data.max() < 50:  # Likely in m/s, convert to km/h
                speed_data = speed_data * 3.6
            
            fig.add_trace(
                go.Scatter(
                    x=time_minutes,
                    y=speed_data,
                    mode='lines',
                    name='Speed',
                    line=dict(color='#9467bd', width=2),
                    opacity=0.9,
                    showlegend=True
                ),
                row=4, col=1
            )
            
            # Add average speed line
            avg_speed = speed_data.mean()
            fig.add_hline(
                y=avg_speed,
                line_dash="dash",
                line_color="red",
                opacity=0.8,
                annotation_text=f"Avg: {avg_speed:.1f}km/h",
                row=4, col=1
            )
            
        elif 'altitude' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=time_minutes,
                    y=self.df['altitude'],
                    mode='lines',
                    name='Elevation',
                    line=dict(color='#8c564b', width=2),
                    opacity=0.9,
                    showlegend=True
                ),
                row=4, col=1
            )
            
            # Add average elevation line
            avg_altitude = self.df['altitude'].mean()
            fig.add_hline(
                y=avg_altitude,
                line_dash="dash",
                line_color="red",
                opacity=0.8,
                annotation_text=f"Avg: {avg_altitude:.0f}m",
                row=4, col=1
            )
        
        # Update layout for better interactivity and full-width display
        fig.update_layout(
            title=f'{self.athlete_name} - Interactive Professional Multi-Axis Analysis',
            title_x=0.5,
            title_font_size=18,
            title_font_color='black',
            height=900,  # Increased height for better visibility
            width=None,  # Auto-width for full container
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            hovermode='x unified',  # Show all traces at same x position
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial",
                bordercolor="black"
            ),
            margin=dict(l=50, r=50, t=80, b=50),  # Better margins
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes labels and styling
        fig.update_xaxes(title_text="Time (minutes)", row=4, col=1)
        fig.update_yaxes(title_text="Power (W)", row=1, col=1)
        fig.update_yaxes(title_text="Heart Rate (bpm)", row=2, col=1)
        fig.update_yaxes(title_text="Cadence (rpm)", row=3, col=1)
        
        if 'speed' in self.df.columns:
            fig.update_yaxes(title_text="Speed (km/h)", row=4, col=1)
        elif 'altitude' in self.df.columns:
            fig.update_yaxes(title_text="Elevation (m)", row=4, col=1)
        
        # Add grid to all subplots
        for i in range(1, 5):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1)
        
        # Add ride summary as annotation
        summary_text = f"Duration: {self.duration_hr:.1f}h | Distance: {self.total_distance:.1f}km"
        if hasattr(self, 'avg_power') and self.avg_power is not None:
            summary_text += f" | Avg Power: {self.avg_power:.0f}W"
        if hasattr(self, 'np_calc') and self.np_calc is not None:
            summary_text += f" | NP: {self.np_calc:.0f}W"
        if 'heart_rate' in self.metrics and 'avg' in self.metrics['heart_rate']:
            summary_text += f" | Avg HR: {self.metrics['heart_rate']['avg']:.0f}bpm"
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        # Save or display
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_interactive_training_peaks")
            fig.write_html(fig_path + ".html")
            fig.write_image(fig_path + ".png", width=1600, height=800)
            print(f"Interactive graph saved to {fig_path}.html")
        else:
            fig.show()
        
        return fig

    def create_normalized_interactive_graph(self):
        """Create a single interactive graph with all variables normalized and plotted together."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
            
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Plotly not available. Please install with: pip install plotly")
            return
            
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60 if 'timestamp' in self.df.columns else np.arange(len(self.df))
        
        # Create figure
        fig = go.Figure()
        
        # Color scheme for different variables
        colors = {
            'power': '#1f77b4',
            'heart_rate': '#d62728', 
            'cadence': '#2ca02c',
            'speed': '#9467bd',
            'altitude': '#8c564b',
            'power_smoothed': '#1f77b4',
            'heart_rate_smoothed': '#d62728',
            'cadence_smoothed': '#2ca02c'
        }
        
        # Normalize function
        def normalize_data(data, name):
            """Normalize data to 0-1 range with mean at 0.5"""
            if data is None or len(data) == 0:
                return None
            data_clean = data.dropna()
            if len(data_clean) == 0:
                return None
            min_val = data_clean.min()
            max_val = data_clean.max()
            if max_val == min_val:
                return np.full(len(data), 0.5)
            normalized = (data_clean - min_val) / (max_val - min_val)
            # Fill NaN values with 0.5 (middle)
            result = np.full(len(data), 0.5)
            result[data_clean.index] = normalized
            return result
        
        # Add traces for each variable
        traces_added = 0
        
        # 1. Power (primary metric)
        if 'power' in self.df.columns:
            power_data = self.df['power']
            if 'power_smoothed' in self.df.columns:
                power_data = self.df['power_smoothed']
            
            power_norm = normalize_data(power_data, 'power')
            if power_norm is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_minutes,
                        y=power_norm,
                        mode='lines',
                        name='Power',
                        line=dict(color=colors['power'], width=4),
                        opacity=1.0,
                        hovertemplate='<b>Power</b><br>' +
                                    'Time: %{x:.1f} min<br>' +
                                    'Power: %{text} W<br>' +
                                    '<extra></extra>',
                        text=[f"{val:.0f}" for val in power_data],
                        yaxis='y'
                    )
                )
                traces_added += 1
        
        # 2. Heart Rate
        if 'heart_rate' in self.df.columns:
            hr_data = self.df['heart_rate']
            if 'heart_rate_smoothed' in self.df.columns:
                hr_data = self.df['heart_rate_smoothed']
            
            hr_norm = normalize_data(hr_data, 'heart_rate')
            if hr_norm is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_minutes,
                        y=hr_norm,
                        mode='lines',
                        name='Heart Rate',
                        line=dict(color=colors['heart_rate'], width=3.5),
                        opacity=1.0,
                        hovertemplate='<b>Heart Rate</b><br>' +
                                    'Time: %{x:.1f} min<br>' +
                                    'HR: %{text} bpm<br>' +
                                    '<extra></extra>',
                        text=[f"{val:.0f}" for val in hr_data],
                        yaxis='y'
                    )
                )
                traces_added += 1
        
        # 3. Cadence
        if 'cadence' in self.df.columns:
            cadence_data = self.df['cadence']
            if 'cadence_smoothed' in self.df.columns:
                cadence_data = self.df['cadence_smoothed']
            
            cadence_norm = normalize_data(cadence_data, 'cadence')
            if cadence_norm is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_minutes,
                        y=cadence_norm,
                        mode='lines',
                        name='Cadence',
                        line=dict(color=colors['cadence'], width=3),
                        opacity=1.0,
                        hovertemplate='<b>Cadence</b><br>' +
                                    'Time: %{x:.1f} min<br>' +
                                    'Cadence: %{text} rpm<br>' +
                                    '<extra></extra>',
                        text=[f"{val:.0f}" for val in cadence_data],
                        yaxis='y'
                    )
                )
                traces_added += 1
        
        # 4. Speed
        if 'speed' in self.df.columns:
            speed_data = self.df['speed']
            # Convert to km/h if in m/s
            if speed_data.max() < 50:
                speed_data = speed_data * 3.6
            
            speed_norm = normalize_data(speed_data, 'speed')
            if speed_norm is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_minutes,
                        y=speed_norm,
                        mode='lines',
                        name='Speed',
                        line=dict(color=colors['speed'], width=3),
                        opacity=1.0,
                        hovertemplate='<b>Speed</b><br>' +
                                    'Time: %{x:.1f} min<br>' +
                                    'Speed: %{text} km/h<br>' +
                                    '<extra></extra>',
                        text=[f"{val:.1f}" for val in speed_data],
                        yaxis='y'
                    )
                )
                traces_added += 1
        
        # 5. Altitude (if no speed)
        elif 'altitude' in self.df.columns:
            altitude_data = self.df['altitude']
            altitude_norm = normalize_data(altitude_data, 'altitude')
            if altitude_norm is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_minutes,
                        y=altitude_norm,
                        mode='lines',
                        name='Elevation',
                        line=dict(color=colors['altitude'], width=3),
                        opacity=1.0,
                        hovertemplate='<b>Elevation</b><br>' +
                                    'Time: %{x:.1f} min<br>' +
                                    'Elevation: %{text} m<br>' +
                                    '<extra></extra>',
                        text=[f"{val:.0f}" for val in altitude_data],
                        yaxis='y'
                    )
                )
                traces_added += 1
        
        # Add average lines for each variable
        if 'power' in self.df.columns and hasattr(self, 'avg_power') and self.avg_power is not None:
            power_data = self.df['power']
            if 'power_smoothed' in self.df.columns:
                power_data = self.df['power_smoothed']
            power_norm_avg = normalize_data(pd.Series([self.avg_power] * len(power_data)), 'power_avg')
            if power_norm_avg is not None:
                fig.add_hline(
                    y=power_norm_avg[0],
                    line_dash="dash",
                    line_color=colors['power'],
                    opacity=0.6,
                    annotation_text=f"Avg Power: {self.avg_power:.0f}W",
                    annotation_position="top right"
                )
        
        if 'heart_rate' in self.metrics and 'avg' in self.metrics['heart_rate']:
            avg_hr = self.metrics['heart_rate']['avg']
            hr_data = self.df['heart_rate']
            hr_norm_avg = normalize_data(pd.Series([avg_hr] * len(hr_data)), 'hr_avg')
            if hr_norm_avg is not None:
                fig.add_hline(
                    y=hr_norm_avg[0],
                    line_dash="dash",
                    line_color=colors['heart_rate'],
                    opacity=0.6,
                    annotation_text=f"Avg HR: {avg_hr:.0f}bpm",
                    annotation_position="top right"
                )
        
        if 'cadence' in self.metrics and 'avg' in self.metrics['cadence']:
            avg_cadence = self.metrics['cadence']['avg']
            cadence_data = self.df['cadence']
            cadence_norm_avg = normalize_data(pd.Series([avg_cadence] * len(cadence_data)), 'cadence_avg')
            if cadence_norm_avg is not None:
                fig.add_hline(
                    y=cadence_norm_avg[0],
                    line_dash="dash",
                    line_color=colors['cadence'],
                    opacity=0.6,
                    annotation_text=f"Avg Cadence: {avg_cadence:.0f}rpm",
                    annotation_position="top right"
                )
        
        # Update layout with improved text visibility
        fig.update_layout(
            title=f'{self.athlete_name} - Normalized Multi-Variable Analysis',
            title_x=0.5,
            title_font_size=20,
            title_font_color='black',
            title_font_family="Arial Black",
            height=800,
            width=None,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=14, color="black")
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial Black",
                bordercolor="black"
            ),
            margin=dict(l=60, r=60, t=100, b=60),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title="Time (minutes)",
                title_font=dict(size=16, color="black"),
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                tickfont=dict(size=14, color="black"),
                tickmode='auto',
                nticks=10
            ),
            yaxis=dict(
                title="Normalized Values (0-1)",
                title_font=dict(size=16, color="black"),
                range=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                tickmode='array',
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['Min', '25%', 'Avg', '75%', 'Max'],
                tickfont=dict(size=14, color="black")
            )
        )
        
        # Add ride summary as annotation with improved visibility
        summary_text = f"Duration: {self.duration_hr:.1f}h | Distance: {self.total_distance:.1f}km"
        if hasattr(self, 'avg_power') and self.avg_power is not None:
            summary_text += f" | Avg Power: {self.avg_power:.0f}W"
        if hasattr(self, 'np_calc') and self.np_calc is not None:
            summary_text += f" | NP: {self.np_calc:.0f}W"
        if 'heart_rate' in self.metrics and 'avg' in self.metrics['heart_rate']:
            summary_text += f" | Avg HR: {self.metrics['heart_rate']['avg']:.0f}bpm"
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=12, color="black", family="Arial Black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=2
        )
        
        # Save or display
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_normalized_interactive")
            fig.write_html(fig_path + ".html")
            fig.write_image(fig_path + ".png", width=1600, height=800)
            print(f"Normalized interactive graph saved to {fig_path}.html")
        else:
            fig.show()
        
        return fig

    def create_professional_dual_axis_graph(self):
        """Create a professional multi-axis graph with dual y-axes overlaying multiple cycling metrics."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
            
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not available. Please install with: pip install plotly")
            return
            
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60 if 'timestamp' in self.df.columns else np.arange(len(self.df))
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Color scheme for different variables (Professional style)
        colors = {
            'power': '#1f77b4',      # Blue
            'heart_rate': '#d62728',  # Red
            'cadence': '#2ca02c',     # Green
            'speed': '#9467bd',       # Purple
            'altitude': '#8c564b'     # Brown
        }
        
        # Track which axes are used
        left_axis_vars = []
        right_axis_vars = []
        
        # Apply smoothing to data for better visualization
        smoothing_window = 5
        
        # Primary metrics on left y-axis (Power, HR, Cadence)
        if 'power' in self.df.columns:
            power_data = self.df['power'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=time_minutes,
                    y=power_data,
                    mode='lines',
                    name='Power',
                    line=dict(color=colors['power'], width=3),
                    opacity=0.9,
                    hovertemplate='<b>Power</b><br>' +
                                'Time: %{x:.1f} min<br>' +
                                'Power: %{y:.0f} W<br>' +
                                '<extra></extra>',
                    yaxis='y'
                )
            )
            left_axis_vars.append('Power')
        
        if 'heart_rate' in self.df.columns:
            hr_data = self.df['heart_rate'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=time_minutes,
                    y=hr_data,
                    mode='lines',
                    name='Heart Rate',
                    line=dict(color=colors['heart_rate'], width=2.5),
                    opacity=0.8,
                    hovertemplate='<b>Heart Rate</b><br>' +
                                'Time: %{x:.1f} min<br>' +
                                'HR: %{y:.0f} bpm<br>' +
                                '<extra></extra>',
                    yaxis='y'
                )
            )
            left_axis_vars.append('Heart Rate')
        
        if 'cadence' in self.df.columns:
            cadence_data = self.df['cadence'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=time_minutes,
                    y=cadence_data,
                    mode='lines',
                    name='Cadence',
                    line=dict(color=colors['cadence'], width=2),
                    opacity=0.7,
                    hovertemplate='<b>Cadence</b><br>' +
                                'Time: %{x:.1f} min<br>' +
                                'Cadence: %{y:.0f} rpm<br>' +
                                '<extra></extra>',
                    yaxis='y'
                )
            )
            left_axis_vars.append('Cadence')
        
        # Secondary metrics on separate y-axes (Speed on y2, Elevation on y3)
        if 'speed' in self.df.columns:
            speed_data = self.df['speed'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            # Convert to km/h if in m/s
            if speed_data.max() < 50:
                speed_data = speed_data * 3.6
            
            fig.add_trace(
                go.Scatter(
                    x=time_minutes,
                    y=speed_data,
                    mode='lines',
                    name='Speed',
                    line=dict(color=colors['speed'], width=2.5),
                    opacity=0.8,
                    hovertemplate='<b>Speed</b><br>' +
                                'Time: %{x:.1f} min<br>' +
                                'Speed: %{y:.1f} km/h<br>' +
                                '<extra></extra>',
                    yaxis='y2'
                )
            )
            right_axis_vars.append('Speed')
        
        if 'altitude' in self.df.columns:
            altitude_data = self.df['altitude'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=time_minutes,
                    y=altitude_data,
                    mode='lines',
                    name='Elevation',
                    line=dict(color=colors['altitude'], width=2.5),
                    opacity=0.8,
                    hovertemplate='<b>Elevation</b><br>' +
                                'Time: %{x:.1f} min<br>' +
                                'Elevation: %{y:.0f} m<br>' +
                                '<extra></extra>',
                    yaxis='y3'
                )
            )
            right_axis_vars.append('Elevation')
        
        # Add average lines for primary metrics with better positioning
        if 'power' in self.df.columns and hasattr(self, 'avg_power') and self.avg_power is not None:
            fig.add_hline(
                y=self.avg_power,
                line_dash="dash",
                line_color=colors['power'],
                opacity=0.6,
                annotation_text=f"Avg Power: {self.avg_power:.0f}W",
                annotation_position="top right",
                annotation=dict(font_size=12)
            )
        
        if 'heart_rate' in self.metrics and 'avg' in self.metrics['heart_rate']:
            avg_hr = self.metrics['heart_rate']['avg']
            fig.add_hline(
                y=avg_hr,
                line_dash="dash",
                line_color=colors['heart_rate'],
                opacity=0.6,
                annotation_text=f"Avg HR: {avg_hr:.0f}bpm",
                annotation_position="top right",
                annotation=dict(font_size=12)
            )
        
        if 'cadence' in self.metrics and 'avg' in self.metrics['cadence']:
            avg_cadence = self.metrics['cadence']['avg']
            fig.add_hline(
                y=avg_cadence,
                line_dash="dash",
                line_color=colors['cadence'],
                opacity=0.6,
                annotation_text=f"Avg Cadence: {avg_cadence:.0f}rpm",
                annotation_position="top right",
                annotation=dict(font_size=12)
            )
        
        # Update layout with multiple y-axes and improved styling
        fig.update_layout(
            title=f'{self.athlete_name} - Enhanced Multi-Axis Professional Analysis',
            title_x=0.5,
            title_font_size=24,
            title_font_color='black',
            title_font_family="Arial Black",
            height=800,
            width=None,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.98)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=16, color="black")
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Arial Black",
                bordercolor="black"
            ),
            margin=dict(l=80, r=80, t=120, b=80),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title="Time (minutes)",
                title_font=dict(size=18, color="black"),
                title_standoff=25,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                tickfont=dict(size=16, color="black"),
                tickmode='auto',
                nticks=10
            ),
            # Primary y-axis (left)
            yaxis=dict(
                title="Power (W) / Heart Rate (bpm) / Cadence (rpm)",
                title_font=dict(size=18, color="black"),
                title_standoff=30,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                tickfont=dict(size=16, color="black"),
                side='left'
            ),
            # Secondary y-axis (middle-right for speed)
            yaxis2=dict(
                title="Speed (km/h)",
                title_font=dict(size=18, color="black"),
                title_standoff=30,
                showgrid=False,
                tickfont=dict(size=16, color="black"),
                side='right',
                overlaying='y',
                anchor='x',
                position=0.85
            ),
            # Tertiary y-axis (far-right for elevation)
            yaxis3=dict(
                title="Elevation (m)",
                title_font=dict(size=18, color="black"),
                title_standoff=30,
                showgrid=False,
                tickfont=dict(size=16, color="black"),
                side='right',
                overlaying='y',
                anchor='x',
                position=0.95
            )
        )
        
        # Add ride summary as annotation with enhanced info
        summary_text = f"Duration: {self.duration_hr:.1f}h | Distance: {self.total_distance:.1f}km"
        if hasattr(self, 'avg_power') and self.avg_power is not None:
            summary_text += f" | Avg Power: {self.avg_power:.0f}W"
        if hasattr(self, 'np_calc') and self.np_calc is not None:
            summary_text += f" | NP: {self.np_calc:.0f}W"
        if 'heart_rate' in self.metrics and 'avg' in self.metrics['heart_rate']:
            summary_text += f" | Avg HR: {self.metrics['heart_rate']['avg']:.0f}bpm"
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=14, color="black", family="Arial Black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=2
        )
        
        # Add axis information with better formatting
        if left_axis_vars and right_axis_vars:
            # Separate speed and elevation for axis info
            speed_vars = [var for var in right_axis_vars if var == 'Speed']
            elevation_vars = [var for var in right_axis_vars if var == 'Elevation']
            
            axis_info = f"Left: {', '.join(left_axis_vars)} | Middle-Right: {', '.join(speed_vars)} | Far-Right: {', '.join(elevation_vars)}"
            fig.add_annotation(
                text=axis_info,
                xref="paper", yref="paper",
                x=0.02, y=0.95,
                showarrow=False,
                font=dict(size=12, color="black", family="Arial"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        
        # Save or display
        if self.save_figures:
            fig_path = os.path.join(self.save_dir, f"{self.analysis_id}_multi_axis_analysis")
            fig.write_html(fig_path + ".html")
            fig.write_image(fig_path + ".png", width=1600, height=700)
            print(f"Professional multi-axis graph saved to {fig_path}.html")
        else:
            fig.show()
        
        return fig


def main():
    """Main function to run enhanced cycling analysis."""
    import argparse
    import os
    import pathlib
    import pandas as pd
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Enhanced Cycling Analysis')
    parser.add_argument('file_path', help='Path to .fit file')
    parser.add_argument('--ftp', type=int, default=300, help='Functional Threshold Power')
    parser.add_argument('--save_figures', action='store_true', help='Save figures to disk')
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"❌ File not found: {args.file_path}")
        return
    
    print(f"🚴 Loading FIT file: {args.file_path}")
    
    # Extract analysis_id from filename
    analysis_id = args.file_path.split("/")[-1].replace(".fit", "")
    
    analyzer = CyclingAnalyzer(save_figures=args.save_figures, ftp=args.ftp, analysis_id=analysis_id)
    
    if analyzer.load_fit_file(args.file_path):
        print("✅ FIT file loaded successfully!")
        
        # Clean and smooth data
        print("\n🔧 Cleaning and smoothing data...")
        analyzer.clean_and_smooth_data()
        
        # Calculate metrics
        print("\n📊 Calculating metrics...")
        analyzer.calculate_metrics()
        
        # Print summary
        print("\n📋 Printing comprehensive metrics table...")
        analyzer.print_comprehensive_metrics_table()
        
        # Create professional multi-axis graph first
        print("\n📈 Creating professional multi-axis graph...")
        analyzer.create_professional_multi_axis_graph()
        
        # Create interactive professional multi-axis graph
        print("\n📈 Creating interactive professional multi-axis graph...")
        analyzer.create_interactive_professional_graph()
        
        # Create normalized interactive graph
        print("\n📈 Creating normalized interactive graph...")
        analyzer.create_normalized_interactive_graph()
        
        # Create professional dual-axis graph
        print("\n📈 Creating professional dual-axis graph...")
        analyzer.create_professional_dual_axis_graph()
        
        # Advanced physiological analysis
        print("\n🔬 Running advanced physiological analysis...")
        
        # Archive advanced metrics
        advanced_row = {
            "ride_id": args.file_path.split("/")[-1].replace(".fit", ""),
            "date": analyzer.df['timestamp'].iloc[0].date() if (analyzer.df is not None and 'timestamp' in analyzer.df.columns) else '',
            "duration_min": analyzer.duration_hr * 60 if hasattr(analyzer, 'duration_hr') else np.nan,
            "distance_km": analyzer.total_distance if hasattr(analyzer, 'total_distance') else np.nan,
        }
        
        # Only add power-based fields if available
        if hasattr(analyzer, 'avg_power') and analyzer.avg_power is not None and not np.isnan(analyzer.avg_power):
            advanced_row["avg_power_W"] = analyzer.avg_power
        if hasattr(analyzer, 'np_calc') and analyzer.np_calc is not None and not np.isnan(analyzer.np_calc):
            advanced_row["NP_W"] = analyzer.np_calc
        if hasattr(analyzer, 'IF') and analyzer.IF is not None and not np.isnan(analyzer.IF):
            advanced_row["IF"] = analyzer.IF
        if hasattr(analyzer, 'TSS') and analyzer.TSS is not None and not np.isnan(analyzer.TSS):
            advanced_row["TSS"] = analyzer.TSS
        if hasattr(analyzer, 'max_power') and analyzer.max_power is not None and not np.isnan(analyzer.max_power):
            advanced_row["max_power_W"] = analyzer.max_power
        if hasattr(analyzer, 'metrics') and 'hr' in analyzer.metrics:
            advanced_row["avg_hr"] = analyzer.metrics['hr'].get('avg', np.nan)
            advanced_row["max_hr"] = analyzer.metrics['hr'].get('max', np.nan)
        if hasattr(analyzer, 'metrics') and 'cadence' in analyzer.metrics:
            advanced_row["avg_cadence"] = analyzer.metrics['cadence'].get('avg', np.nan)
            advanced_row["max_cadence"] = analyzer.metrics['cadence'].get('max', np.nan)
        if hasattr(analyzer, 'trimp_score'):
            advanced_row["trimp"] = analyzer.trimp_score
        if hasattr(analyzer, 'cci_kj'):
            advanced_row["cci_kj"] = analyzer.cci_kj
        if hasattr(analyzer, 'total_met_min'):
            advanced_row["total_met_min"] = analyzer.total_met_min
        
        # Save to ride history
        hist_path = pathlib.Path("ride_history.csv")
        hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
        if "ride_id" in hist.columns:
            hist = hist[hist["ride_id"] != advanced_row["ride_id"]]  # de-dupe
        hist = pd.concat([hist, pd.DataFrame([advanced_row])], ignore_index=True)
        if 'date' in hist.columns and pd.Series(hist['date']).notna().any():
            hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
            hist = pd.DataFrame(hist).sort_values(by="date")
        hist.to_csv(hist_path, index=False)
        print(f"📁 Advanced metrics archived to {hist_path} ({len(hist)} rides).")
        
        # Critical Power analysis
        cp_est, w_prime_est = analyzer.estimate_critical_power()
        if cp_est and analyzer.show_w_prime_balance:
            analyzer.calculate_w_prime_balance(cp_est, w_prime_est)
        
        # Lactate estimation
        if analyzer.show_lactate_estimation:
            analyzer.estimate_lactate()
        
        # Torque analysis
        if analyzer.show_advanced_plots:
            analyzer.analyze_torque()
        
        # Enhanced drift and fatigue analysis
        print("\n📉 Enhanced drift and fatigue analysis...")
        analyzer.analyze_fatigue_patterns()
        
        # HR-based strain analysis
        if 'heart_rate' in analyzer.df.columns:
            analyzer.calculate_hr_strain()
        
        # Heat stress analysis
        if 'heart_rate' in analyzer.df.columns:
            analyzer.analyze_heat_stress()
        
        # Power-to-HR efficiency analysis
        if 'power' in analyzer.df.columns and 'heart_rate' in analyzer.df.columns:
            analyzer.analyze_power_hr_efficiency()
        
        # Variable relationship analysis
        analyzer.analyze_variable_relationships()
        
        # Performance insights
        analyzer.print_insights()
        
        print("\n✅ Enhanced analysis complete!")
        print("\nKey improvements demonstrated:")
        print("✅ Enhanced graph spacing and readability")
        print("✅ Proper physiological lactate estimation")
        print("✅ Advanced drift analysis with trend detection")
        print("✅ Moving-time-based TSS and IF calculations")
        print("✅ Improved data cleaning with outlier detection")
        print("✅ Comprehensive athlete-specific analysis")
    else:
        print("Failed to load FIT file. Please check the file path.")


if __name__ == "__main__":
    main() 