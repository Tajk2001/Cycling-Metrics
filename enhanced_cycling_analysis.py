#!/usr/bin/env python3
"""
Enhanced Cycling Analysis Script
Comprehensive analysis of cycling FIT files with personalized metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fitparse
import seaborn as sns
from datetime import datetime
import os
from typing import Optional, Dict, List, Tuple, Any

# Set style for modern, minimalist plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class CyclingAnalyzer:
    """Main class for cycling data analysis with personalized configuration."""
    
    def __init__(self, athlete_name="Taj Krieger", ftp=300, max_hr=195, rest_hr=51, 
                 weight_kg=52, height_cm=165, lactate_rest=1.2, lactate_peak=8.0,
                 w_prime_tau=386, max_interpolation_pct=5.0, power_outlier_threshold=5.0,
                 hr_outlier_threshold=3.0, cadence_outlier_threshold=3.0):
        """Initialize analyzer with enhanced athlete profile and data quality parameters."""
        self.athlete_name = athlete_name
        self.ftp = ftp
        self.max_hr = max_hr
        self.rest_hr = rest_hr
        self.weight_kg = weight_kg
        self.height_cm = height_cm
        
        # Enhanced physiological parameters
        self.lactate_rest = lactate_rest  # Resting lactate (mmol/L)
        self.lactate_peak = lactate_peak  # Peak lactate (mmol/L)
        self.w_prime_tau = w_prime_tau    # W' recovery time constant (s)
        
        # Data quality thresholds
        self.max_interpolation_pct = max_interpolation_pct
        self.power_outlier_threshold = power_outlier_threshold
        self.hr_outlier_threshold = hr_outlier_threshold
        self.cadence_outlier_threshold = cadence_outlier_threshold
        
        # Training zones
        self.power_zones = {
            'Z1 (Recovery)': (0, 55),
            'Z2 (Endurance)': (55, 75),
            'Z3 (Tempo)': (75, 90),
            'Z4 (Threshold)': (90, 105),
            'Z5 (VO2max)': (105, 120),
            'Z6 (Anaerobic)': (120, 150),
            'Z7 (Neuromuscular)': (150, 200)
        }
        
        # Heart rate zones
        self.hr_zones = {
            'Z1 (Recovery)': (rest_hr, max_hr * 0.65),
            'Z2 (Endurance)': (max_hr * 0.65, max_hr * 0.75),
            'Z3 (Tempo)': (max_hr * 0.75, max_hr * 0.85),
            'Z4 (Threshold)': (max_hr * 0.85, max_hr * 0.95),
            'Z5 (VO2max)': (max_hr * 0.95, max_hr),
            'Z6 (Anaerobic)': (max_hr, max_hr * 1.05)
        }
        
        # Analysis settings
        self.smoothing_window = 30
        self.moving_avg_window = 30  # For moving-time-based calculations
        
        # Display settings
        self.show_advanced_plots = True
        self.show_lactate_estimation = True
        self.show_w_prime_balance = True
        self.show_zone_analysis = True
        
        # Training goals
        self.session_type = "Endurance"
        self.target_tss = 100
        self.target_duration_hr = 2.0
        
        # Data storage
        self.df = None
        self.metrics = {}
        self.zone_percentages = {}
        self.hr_zone_percentages = {}
        self.cadence_zone_percentages = {}
        self.speed_zone_percentages = {}
        self.data_quality_report = {}
        
    def load_fit_file(self, file_path):
        """Load and parse FIT file."""
        try:
            fitfile = fitparse.FitFile(file_path)
            records = [{field.name: field.value for field in record} 
                      for record in fitfile.get_messages('record')]
            self.df = pd.DataFrame(records)
            
            print(f"Data loaded: {self.df.shape[0]} records, {self.df.shape[1]} fields")
            print(f"Available columns: {list(self.df.columns)}")
            return True
            
        except Exception as e:
            print(f"Error loading FIT file: {e}")
            return False
    
    def clean_and_smooth_data(self):
        """Enhanced data cleaning with outlier detection and quality reporting."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return False
        
        # Initialize data quality report
        self.data_quality_report: Dict[str, Any] = {
            'total_records': len(self.df),
            'interpolated_pct': 0.0,
            'outliers_removed': 0,
            'data_quality_score': 100.0
        }
        
        # Convert units
        if 'speed' in self.df.columns:
            self.df['speed_kmh'] = self.df['speed'] * 3.6
        if 'distance' in self.df.columns:
            self.df['distance_km'] = self.df['distance'] / 1000
        
        # Enhanced outlier detection and removal
        outlier_removed = 0
        for col, threshold in [('power', self.power_outlier_threshold), 
                              ('heart_rate', self.hr_outlier_threshold),
                              ('cadence', self.cadence_outlier_threshold)]:
            if col in self.df.columns:
                # Calculate statistics for outlier detection
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                
                # Define outlier bounds
                lower_bound = mean_val - threshold * std_val
                upper_bound = mean_val + threshold * std_val
                
                # Count outliers
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                outlier_removed += outliers
                
                # Clip outliers to bounds (more conservative than removal)
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
        
        # Enhanced interpolation with limits
        interpolated_count = 0
        for col in ['power', 'cadence', 'heart_rate', 'speed_kmh']:
            if col in self.df.columns:
                # Count missing values before interpolation
                missing_before = self.df[col].isna().sum()
                
                # Interpolate with forward-fill limit of 5 seconds
                self.df[col] = self.df[col].interpolate(method='linear', limit=5, limit_direction='both')
                
                # Count interpolated values
                interpolated_count += missing_before - self.df[col].isna().sum()
        
        # Calculate interpolation percentage
        total_possible = len(self.df) * 4  # 4 main columns
        interpolation_pct = (interpolated_count / total_possible) * 100
        
        # Apply smoothing for visualization only (keep raw data for NP calculation)
        smoothing_windows = {
            'power': self.smoothing_window, 
            'heart_rate': self.smoothing_window, 
            'cadence': self.smoothing_window, 
            'speed_kmh': 10
        }
        
        for col in smoothing_windows:
            if col in self.df.columns:
                self.df[col+'_smoothed'] = self.df[col].rolling(
                    window=smoothing_windows[col], min_periods=1, center=True).mean()
        
        # Update data quality report
        self.data_quality_report['interpolated_pct'] = float(interpolation_pct)
        self.data_quality_report['outliers_removed'] = int(outlier_removed)
        self.data_quality_report['data_quality_score'] = float(max(0, 100 - interpolation_pct - (outlier_removed / len(self.df) * 10)))
        
        # Warn if data quality is poor
        if interpolation_pct > self.max_interpolation_pct:
            print(f"⚠️  Warning: {interpolation_pct:.1f}% of data was interpolated (threshold: {self.max_interpolation_pct}%)")
        
        if self.data_quality_report['data_quality_score'] < 80:
            print(f"⚠️  Warning: Data quality score is {self.data_quality_report['data_quality_score']:.1f}/100")
        
        print('Enhanced data cleaning completed successfully.')
        return True
    
    def calculate_metrics(self):
        """Calculate all performance metrics with moving-time-based algorithms."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return False
        
        # Enhanced core calculations with moving-time-based algorithms
        # Use raw power data for NP calculation (not smoothed)
        power_rolling_30s = self.df['power'].rolling(window=30, min_periods=15).mean()
        self.np_calc = (power_rolling_30s ** 4).mean() ** 0.25
        
        # Basic power metrics
        self.avg_power = self.df['power'].mean()
        self.max_power = self.df['power'].max()
        self.power_std = self.df['power'].std()
        
        # Duration and distance
        self.duration_hr = (self.df['timestamp'].iloc[-1] - 
                           self.df['timestamp'].iloc[0]).total_seconds() / 3600
        self.total_distance = (self.df['distance_km'].iloc[-1] 
                              if 'distance_km' in self.df.columns else 0)
        
        # Enhanced training metrics with moving-time-based calculations
        self.IF = self.np_calc / self.ftp
        self.VI = self.np_calc / self.avg_power
        
        # Moving-time-based TSS calculation
        # Calculate TSS using 30-second moving average power
        moving_avg_power = self.df['power'].rolling(window=30, min_periods=15).mean()
        moving_avg_IF = moving_avg_power / self.ftp
        self.TSS = (moving_avg_IF ** 2 * self.duration_hr * 100).sum() / len(self.df)
        
        # Optional metrics
        self.metrics = {}
        for metric, col in [('speed', 'speed_kmh'), ('hr', 'heart_rate'), ('cadence', 'cadence')]:
            if col in self.df.columns:
                self.metrics[metric] = {
                    'avg': self.df[col].mean(),
                    'max': self.df[col].max(),
                    'min': self.df[col].min() if metric == 'hr' else self.df[col].max()
                }
        
        # Power zones
        if 'power' in self.df.columns:
            self.df['pct_ftp'] = (self.df['power'] / self.ftp) * 100
            zone_counts = {}
            for zone_name, (lower, upper) in self.power_zones.items():
                if upper == 200:  # Z7 has no upper limit
                    mask = (self.df['pct_ftp'] >= lower)
                else:
                    mask = (self.df['pct_ftp'] >= lower) & (self.df['pct_ftp'] < upper)
                zone_counts[zone_name] = mask.sum()
            
            total_samples = len(self.df)
            self.zone_percentages = {zone: (count/total_samples)*100 
                                   for zone, count in zone_counts.items()}
        
        # Heart rate zones
        if 'heart_rate' in self.df.columns:
            hr_zone_counts = {}
            for zone_name, (lower, upper) in self.hr_zones.items():
                mask = (self.df['heart_rate'] >= lower) & (self.df['heart_rate'] < upper)
                hr_zone_counts[zone_name] = mask.sum()
            
            total_samples = len(self.df)
            self.hr_zone_percentages = {zone: (count/total_samples)*100 
                                       for zone, count in hr_zone_counts.items()}
        
        # Cadence zones
        if 'cadence' in self.df.columns:
            cadence_zones = {
                'Very Low (<60)': (0, 60),
                'Low (60-80)': (60, 80),
                'Moderate (80-90)': (80, 90),
                'Optimal (90-100)': (90, 100),
                'High (100-110)': (100, 110),
                'Very High (>110)': (110, 200)
            }
            
            cadence_zone_counts = {}
            for zone_name, (lower, upper) in cadence_zones.items():
                if upper == 200:  # Very High has no upper limit
                    mask = (self.df['cadence'] >= lower)
                else:
                    mask = (self.df['cadence'] >= lower) & (self.df['cadence'] < upper)
                cadence_zone_counts[zone_name] = mask.sum()
            
            total_samples = len(self.df)
            self.cadence_zone_percentages = {zone: (count/total_samples)*100 
                                            for zone, count in cadence_zone_counts.items()}
        
        # Speed zones (if available)
        if 'speed_kmh' in self.df.columns:
            speed_zones = {
                'Recovery (<15)': (0, 15),
                'Easy (15-25)': (15, 25),
                'Moderate (25-35)': (25, 35),
                'Fast (35-45)': (35, 45),
                'Very Fast (>45)': (45, 100)
            }
            
            speed_zone_counts = {}
            for zone_name, (lower, upper) in speed_zones.items():
                if upper == 100:  # Very Fast has no upper limit
                    mask = (self.df['speed_kmh'] >= lower)
                else:
                    mask = (self.df['speed_kmh'] >= lower) & (self.df['speed_kmh'] < upper)
                speed_zone_counts[zone_name] = mask.sum()
            
            total_samples = len(self.df)
            self.speed_zone_percentages = {zone: (count/total_samples)*100 
                                          for zone, count in speed_zone_counts.items()}
        
        return True
    
    def print_summary(self):
        """Print comprehensive session summary."""
        if not hasattr(self, 'np_calc'):
            print("Metrics not calculated. Please run calculate_metrics() first.")
            return
        
        summary_data = {
            'Metric': [
                'Duration', 'Distance (km)', 'Avg Speed (km/h)', 'Max Speed (km/h)',
                'Avg Power (W)', 'Normalized Power (W)', 'Max Power (W)',
                'Intensity Factor', 'Variability Index', 'Training Stress Score',
                'Avg Heart Rate (bpm)', 'Max Heart Rate (bpm)', 'Min Heart Rate (bpm)',
                'Avg Cadence (rpm)', 'Max Cadence (rpm)',
                'Power Z1 (%)', 'Power Z2 (%)', 'Power Z3 (%)', 'Power Z4 (%)', 
                'Power Z5 (%)', 'Power Z6 (%)', 'Power Z7 (%)',
                'HR Z1 (%)', 'HR Z2 (%)', 'HR Z3 (%)', 'HR Z4 (%)', 'HR Z5 (%)', 'HR Z6 (%)',
                'Cadence Very Low (%)', 'Cadence Low (%)', 'Cadence Moderate (%)',
                'Cadence Optimal (%)', 'Cadence High (%)', 'Cadence Very High (%)',
                'Speed Recovery (%)', 'Speed Easy (%)', 'Speed Moderate (%)',
                'Speed Fast (%)', 'Speed Very Fast (%)'
            ],
            'Value': [
                f"{int(self.duration_hr):02d}:{int((self.duration_hr % 1) * 60):02d}",
                f"{self.total_distance:.1f}",
                f"{self.metrics.get('speed', {}).get('avg', 0):.1f}",
                f"{self.metrics.get('speed', {}).get('max', 0):.1f}",
                f"{self.avg_power:.0f}",
                f"{self.np_calc:.0f}",
                f"{self.max_power:.0f}",
                f"{self.IF:.2f}",
                f"{self.VI:.2f}",
                f"{self.TSS:.0f}",
                f"{self.metrics.get('hr', {}).get('avg', 0):.0f}",
                f"{self.metrics.get('hr', {}).get('max', 0):.0f}",
                f"{self.metrics.get('hr', {}).get('min', 0):.0f}",
                f"{self.metrics.get('cadence', {}).get('avg', 0):.0f}",
                f"{self.metrics.get('cadence', {}).get('max', 0):.0f}",
                f"{self.zone_percentages.get('Z1 (Recovery)', 0):.1f}",
                f"{self.zone_percentages.get('Z2 (Endurance)', 0):.1f}",
                f"{self.zone_percentages.get('Z3 (Tempo)', 0):.1f}",
                f"{self.zone_percentages.get('Z4 (Threshold)', 0):.1f}",
                f"{self.zone_percentages.get('Z5 (VO2max)', 0):.1f}",
                f"{self.zone_percentages.get('Z6 (Anaerobic)', 0):.1f}",
                f"{self.zone_percentages.get('Z7 (Neuromuscular)', 0):.1f}",
                f"{self.hr_zone_percentages.get('Z1 (Recovery)', 0):.1f}",
                f"{self.hr_zone_percentages.get('Z2 (Endurance)', 0):.1f}",
                f"{self.hr_zone_percentages.get('Z3 (Tempo)', 0):.1f}",
                f"{self.hr_zone_percentages.get('Z4 (Threshold)', 0):.1f}",
                f"{self.hr_zone_percentages.get('Z5 (VO2max)', 0):.1f}",
                f"{self.hr_zone_percentages.get('Z6 (Anaerobic)', 0):.1f}",
                f"{self.cadence_zone_percentages.get('Very Low (<60)', 0):.1f}",
                f"{self.cadence_zone_percentages.get('Low (60-80)', 0):.1f}",
                f"{self.cadence_zone_percentages.get('Moderate (80-90)', 0):.1f}",
                f"{self.cadence_zone_percentages.get('Optimal (90-100)', 0):.1f}",
                f"{self.cadence_zone_percentages.get('High (100-110)', 0):.1f}",
                f"{self.cadence_zone_percentages.get('Very High (>110)', 0):.1f}",
                f"{self.speed_zone_percentages.get('Recovery (<15)', 0):.1f}",
                f"{self.speed_zone_percentages.get('Easy (15-25)', 0):.1f}",
                f"{self.speed_zone_percentages.get('Moderate (25-35)', 0):.1f}",
                f"{self.speed_zone_percentages.get('Fast (35-45)', 0):.1f}",
                f"{self.speed_zone_percentages.get('Very Fast (>45)', 0):.1f}"
            ]
        }
        
        print(f"=== {self.athlete_name.upper()} - SESSION SUMMARY ===")
        print(pd.DataFrame(summary_data).to_string(index=False))
        print("=" * 50)
        
        # Performance analysis
        self._print_performance_analysis()
    
    def _print_performance_analysis(self):
        """Print performance analysis with context."""
        print(f"\n=== PERFORMANCE ANALYSIS ===")
        intensity_level = 'High' if self.TSS > 150 else 'Medium' if self.TSS > 80 else 'Low'
        pacing_quality = 'Good' if self.avg_power/self.np_calc > 0.9 else 'Moderate' if self.avg_power/self.np_calc > 0.8 else 'Poor'
        tss_vs_target = "Above" if self.TSS > self.target_tss else "Below" if self.TSS < self.target_tss * 0.8 else "On Target"
        
        print(f"Training Stress: {self.TSS:.0f} ({intensity_level} intensity) - {tss_vs_target} target of {self.target_tss}")
        print(f"Pacing Quality: {pacing_quality} (Avg/NP ratio: {self.avg_power/self.np_calc:.2f})")
        print(f"Power Variability: {'High' if self.VI > 1.15 else 'Medium' if self.VI > 1.05 else 'Low'} (VI: {self.VI:.2f})")
        
        if 'heart_rate' in self.df.columns:
            hr_reserve = (self.metrics['hr']['max'] - self.rest_hr) / (self.max_hr - self.rest_hr) * 100
            print(f"Heart Rate Reserve: {hr_reserve:.1f}% (Max HR: {self.metrics['hr']['max']:.0f}bpm)")
    
    def create_dashboard(self):
        """Create simplified dashboard with 3 key graphs."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
        
        # Simplified dashboard with 3 key graphs
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{self.athlete_name} - Ride Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Improved subplot spacing
        plt.subplots_adjust(wspace=0.3, top=0.85, bottom=0.1, left=0.05, right=0.95)
        
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        
        # 1. Power Profile (Main metric)
        axes[0].plot(time_minutes, self.df['power'], alpha=0.8, linewidth=1.2, color='#1f77b4')
        axes[0].axhline(y=self.avg_power, color='red', linestyle='--', alpha=0.8, label=f'Avg: {self.avg_power:.0f}W')
        axes[0].axhline(y=self.np_calc, color='orange', linestyle='--', alpha=0.8, label=f'NP: {self.np_calc:.0f}W')
        axes[0].set_xlabel('Time (minutes)')
        axes[0].set_ylabel('Power (W)')
        axes[0].set_title('Power Profile', fontweight='bold', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Heart Rate Profile (Enhanced)
        if 'heart_rate' in self.df.columns:
            # Downsample for better visualization
            sample_rate = max(1, len(time_minutes) // 200)
            time_sampled = time_minutes[::sample_rate]
            hr_sampled = self.df['heart_rate'].iloc[::sample_rate]
            
            axes[1].plot(time_sampled, hr_sampled, alpha=0.8, linewidth=1.2, color='#d62728')
            axes[1].scatter(time_sampled, hr_sampled, alpha=0.5, s=15, color='#d62728')
            axes[1].axhline(y=self.metrics['hr']['avg'], color='red', linestyle='--', alpha=0.8, 
                           label=f"Avg: {self.metrics['hr']['avg']:.0f}bpm")
            axes[1].set_xlabel('Time (minutes)')
            axes[1].set_ylabel('Heart Rate (bpm)')
            axes[1].set_title('Heart Rate Profile', fontweight='bold', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 3. Power Zone Distribution (Key metric) - Bigger pie chart with zone numbers
        if 'power' in self.df.columns:
            zone_labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7']
            zone_names = ['Recovery', 'Endurance', 'Tempo', 'Threshold', 'VO2max', 'Anaerobic', 'Neuromuscular']
            zone_values = [self.zone_percentages.get(f'Z{i} (Recovery)', 0) if i == 1 else
                          self.zone_percentages.get(f'Z{i} (Endurance)', 0) if i == 2 else
                          self.zone_percentages.get(f'Z{i} (Tempo)', 0) if i == 3 else
                          self.zone_percentages.get(f'Z{i} (Threshold)', 0) if i == 4 else
                          self.zone_percentages.get(f'Z{i} (VO2max)', 0) if i == 5 else
                          self.zone_percentages.get(f'Z{i} (Anaerobic)', 0) if i == 6 else
                          self.zone_percentages.get(f'Z{i} (Neuromuscular)', 0) for i in range(1, 8)]
            
            colors = ['#2E8B57', '#3CB371', '#FFD700', '#FF8C00', '#FF4500', '#DC143C', '#8B0000']
            explode = [0.05 if val > 0 else 0 for val in zone_values]
            
            # Create pie chart with zone numbers only (no percentages)
            wedges, texts, autotexts = axes[2].pie(zone_values, labels=zone_labels, autopct='', 
                                                   colors=colors, startangle=90, explode=explode, shadow=True,
                                                   textprops={'fontsize': 12, 'fontweight': 'bold'})
            
            # Remove all percentage text from pie chart
            for autotext in autotexts:
                autotext.set_text('')
            
            # Customize label colors and positioning
            for i, text in enumerate(texts):
                if zone_values[i] > 0:
                    text.set_color(colors[i])
                    text.set_fontweight('bold')
                    text.set_fontsize(11)
                else:
                    text.set_text('')
            
            axes[2].set_title('Power Zone Distribution', fontweight='bold', fontsize=14)
            
            # Create zone table next to pie chart
            table_data = []
            for i, (zone_label, zone_name, value) in enumerate(zip(zone_labels, zone_names, zone_values)):
                if value > 0:
                    table_data.append([zone_label, zone_name, f"{value:.1f}%"])
            
            if table_data:
                # Create plain table
                table_text = "Zone Distribution:\n"
                table_text += "=" * 35 + "\n"
                table_text += f"{'Zone':<4} {'Name':<12} {'%':<6}\n"
                table_text += "-" * 35 + "\n"
                
                for zone_label, zone_name, percentage in table_data:
                    table_text += f"{zone_label:<4} {zone_name:<12} {percentage:<6}\n"
                
                # Position the table to the right of the pie chart
                axes[2].text(1.2, 0.5, table_text, transform=axes[2].transAxes, 
                           fontsize=9, verticalalignment='center', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        plt.tight_layout()
        plt.show()
    
    def estimate_critical_power(self):
        """Estimate Critical Power and W'."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return None, None
        
        def cp_model(t, CP, W_prime):
            return CP + (W_prime / t)
        
        durations = np.array([5, 15, 30, 60, 120, 300, 600])
        mmp = [self.df['power'].rolling(window=d, min_periods=1).mean().max() for d in durations]
        
        try:
            popt, _ = curve_fit(cp_model, durations, mmp, bounds=(0, [600, 100000]))
            cp_est, w_prime_est = popt
            
            print(f'Estimated CP: {cp_est:.0f} W, W\': {w_prime_est:.0f} J')
            print(f'CP vs FTP: {cp_est/self.ftp:.2f} ratio')
            
            return cp_est, w_prime_est
            
        except Exception as e:
            print(f"Error estimating CP: {e}")
            return None, None
    
    def calculate_w_prime_balance(self, cp_est, w_prime_est):
        """Calculate W' balance over time with enhanced visualization."""
        if self.df is None or cp_est is None:
            print("No data loaded or CP not estimated. Please load data and estimate CP first.")
            return
        
        tau = self.w_prime_tau
        w_bal = []
        current_w_prime = w_prime_est
        timestamps = pd.to_datetime(self.df['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds().fillna(1).clip(lower=0.01)
        
        for idx, row in self.df.iterrows():
            power = row['power']
            dt = time_diffs.iloc[idx]
            if power > cp_est:
                current_w_prime -= (power - cp_est) * dt
            else:
                recovery = (w_prime_est - current_w_prime) * (1 - np.exp(-dt / tau))
                current_w_prime += recovery
            current_w_prime = max(0, min(current_w_prime, w_prime_est))
            w_bal.append(current_w_prime)
        
        self.df['w_prime_bal'] = w_bal
        
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        
        # Create enhanced W' balance plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Main W' balance plot
        ax1.plot(time_minutes, self.df['w_prime_bal'], color='purple', linewidth=2, label='W\' Balance')
        ax1.fill_between(time_minutes, self.df['w_prime_bal'], alpha=0.3, color='purple')
        
        # Add reference lines
        ax1.axhline(y=w_prime_est, color='green', linestyle='--', alpha=0.7, 
                    label=f'Full W\' ({w_prime_est:.0f} J)')
        ax1.axhline(y=w_prime_est * 0.5, color='orange', linestyle='--', alpha=0.7, 
                    label='50% W\' Depleted')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='W\' Depleted')
        
        ax1.set_title("W' Balance Over Time", fontweight='bold', fontsize=14)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('W\' Balance (J)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, w_prime_est * 1.1)
        
        # Add zone shading
        ax1.fill_between(time_minutes, w_prime_est * 0.8, w_prime_est, alpha=0.1, color='green', label='High W\'')
        ax1.fill_between(time_minutes, w_prime_est * 0.4, w_prime_est * 0.8, alpha=0.1, color='yellow', label='Moderate W\'')
        ax1.fill_between(time_minutes, 0, w_prime_est * 0.4, alpha=0.1, color='red', label='Low W\'')
        
        # Power vs W' balance scatter
        scatter = ax2.scatter(self.df['power'], self.df['w_prime_bal'], 
                             alpha=0.6, c=time_minutes, cmap='viridis', s=20)
        ax2.set_xlabel('Power (W)')
        ax2.set_ylabel('W\' Balance (J)')
        ax2.set_title('Power vs W\' Balance Relationship')
        ax2.grid(True, alpha=0.3)
        
        # Add CP reference line
        ax2.axvline(x=cp_est, color='red', linestyle='--', alpha=0.7, label=f'CP ({cp_est:.0f}W)')
        ax2.legend()
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Time (minutes)')
        
        plt.tight_layout()
        plt.show()
        
        # Print W' balance statistics
        print(f"\n=== W' BALANCE ANALYSIS ===")
        print(f"Initial W': {w_prime_est:.0f} J")
        print(f"Minimum W': {self.df['w_prime_bal'].min():.0f} J")
        print(f"Average W': {self.df['w_prime_bal'].mean():.0f} J")
        print(f"Time below 50% W': {((self.df['w_prime_bal'] < w_prime_est * 0.5).sum() / len(self.df) * 100):.1f}%")
        print(f"Time W' depleted: {((self.df['w_prime_bal'] < 100).sum() / len(self.df) * 100):.1f}%")
    
    def analyze_torque(self):
        """Analyze torque vs cadence relationship."""
        if self.df is None or 'power' not in self.df.columns or 'cadence' not in self.df.columns:
            print("No data loaded or missing power/cadence data.")
            return
        
        self.df['torque'] = (self.df['power'] * 60) / (2 * np.pi * self.df['cadence'].replace(0, np.nan))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Torque vs Cadence
        scatter = ax1.scatter(self.df['cadence'], self.df['torque'], alpha=0.3, c=self.df['power'], cmap='viridis')
        ax1.set_xlabel('Cadence (rpm)')
        ax1.set_ylabel('Torque (Nm)')
        ax1.set_title('Torque vs Cadence')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Power (W)')
        
        # Power distribution histogram
        ax2.hist(self.df['power'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(self.avg_power, color='red', linestyle='--', label=f'Avg: {self.avg_power:.0f}W')
        ax2.axvline(self.np_calc, color='orange', linestyle='--', label=f'NP: {self.np_calc:.0f}W')
        ax2.set_xlabel('Power (W)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Power Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def estimate_lactate(self):
        """Estimate lactate levels throughout the ride with proper physiological modeling."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
        
        def estimate_lactate_func(power, ftp):
            """Physiologically accurate lactate estimation based on power zones."""
            # Resting lactate: 1.0-1.5 mmol/L
            lactate_rest = 1.2
            
            if power <= ftp * 0.55:  # Recovery zone (0-55% FTP)
                return lactate_rest + (power / (ftp * 0.55)) * 0.3  # 1.2-1.5 mmol/L
            
            elif power <= ftp * 0.75:  # Endurance zone (55-75% FTP)
                return lactate_rest + 0.3 + (power - ftp * 0.55) / (ftp * 0.2) * 0.7  # 1.5-2.2 mmol/L
            
            elif power <= ftp * 0.9:  # Tempo zone (75-90% FTP)
                return lactate_rest + 1.0 + (power - ftp * 0.75) / (ftp * 0.15) * 1.0  # 2.2-3.2 mmol/L
            
            elif power <= ftp:  # Threshold zone (90-100% FTP)
                return lactate_rest + 2.0 + (power - ftp * 0.9) / (ftp * 0.1) * 1.5  # 3.2-4.7 mmol/L
            
            elif power <= ftp * 1.05:  # VO2max zone (100-105% FTP)
                return lactate_rest + 3.5 + (power - ftp) / (ftp * 0.05) * 1.0  # 4.7-5.7 mmol/L
            
            elif power <= ftp * 1.2:  # Anaerobic zone (105-120% FTP)
                return lactate_rest + 4.5 + (power - ftp * 1.05) / (ftp * 0.15) * 2.0  # 5.7-7.7 mmol/L
            
            else:  # Neuromuscular zone (>120% FTP)
                return lactate_rest + 6.5 + (power - ftp * 1.2) / (ftp * 0.3) * 3.0  # 7.7+ mmol/L
        
        # Calculate raw lactate estimates
        self.df['lactate_est'] = self.df['power'].apply(lambda p: estimate_lactate_func(p, self.ftp))
        
        # Apply smoothing to make it more realistic (physiological response is gradual)
        self.df['lactate_smoothed'] = self.df['lactate_est'].rolling(window=60, min_periods=1, center=True).mean()
        
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        
        # Create enhanced lactate plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Main lactate plot with zones
        ax1.plot(time_minutes, self.df['lactate_smoothed'], color='#d62728', linewidth=2, label='Estimated Lactate')
        ax1.fill_between(time_minutes, self.df['lactate_smoothed'], alpha=0.3, color='#d62728')
        
        # Add physiological zones
        ax1.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Aerobic Threshold (~2.0)')
        ax1.axhline(y=4.0, color='orange', linestyle='--', alpha=0.7, label='Lactate Threshold (~4.0)')
        ax1.axhline(y=8.0, color='red', linestyle='--', alpha=0.7, label='Onset of Blood Lactate (~8.0)')
        
        ax1.set_title('Estimated Blood Lactate Response', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Lactate (mmol/L)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 12)
        
        # Add zone shading
        ax1.fill_between(time_minutes, 0, 2, alpha=0.1, color='green', label='Aerobic Zone')
        ax1.fill_between(time_minutes, 2, 4, alpha=0.1, color='yellow', label='Tempo Zone')
        ax1.fill_between(time_minutes, 4, 8, alpha=0.1, color='orange', label='Threshold Zone')
        ax1.fill_between(time_minutes, 8, 12, alpha=0.1, color='red', label='Anaerobic Zone')
        
        # Power vs Lactate scatter
        scatter = ax2.scatter(self.df['power'], self.df['lactate_smoothed'], 
                             alpha=0.6, c=time_minutes, cmap='viridis', s=20)
        ax2.set_xlabel('Power (W)')
        ax2.set_ylabel('Lactate (mmol/L)')
        ax2.set_title('Power vs Lactate Relationship')
        ax2.grid(True, alpha=0.3)
        
        # Add FTP reference line
        ax2.axvline(x=self.ftp, color='red', linestyle='--', alpha=0.7, label=f'FTP ({self.ftp}W)')
        ax2.legend()
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Time (minutes)')
        
        plt.tight_layout()
        plt.show()
        
        # Print lactate statistics
        print(f"\n=== LACTATE ANALYSIS ===")
        print(f"Average Lactate: {self.df['lactate_smoothed'].mean():.2f} mmol/L")
        print(f"Peak Lactate: {self.df['lactate_smoothed'].max():.2f} mmol/L")
        print(f"Time above 4.0 mmol/L: {((self.df['lactate_smoothed'] > 4.0).sum() / len(self.df) * 100):.1f}%")
        print(f"Time above 8.0 mmol/L: {((self.df['lactate_smoothed'] > 8.0).sum() / len(self.df) * 100):.1f}%")
    
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
        if self.avg_power/self.np_calc > 0.95:
            pacing_assessment = "Excellent pacing - very consistent effort"
        elif self.avg_power/self.np_calc > 0.85:
            pacing_assessment = "Good pacing - relatively consistent effort"
        elif self.avg_power/self.np_calc > 0.75:
            pacing_assessment = "Moderate pacing - some variability in effort"
        else:
            pacing_assessment = "High variability - consider more consistent pacing"
        
        # Zone distribution insights
        primary_zone = max(self.zone_percentages, key=self.zone_percentages.get)
        zone_percentage = self.zone_percentages[primary_zone]
        
        print(f"\n📊 Training Load: {load_assessment}")
        print(f"📈 Pacing Quality: {pacing_assessment}")
        print(f"🎯 Primary Zone: {primary_zone} ({zone_percentage:.1f}% of time)")
        
        if 'heart_rate' in self.df.columns:
            hr_avg = self.metrics['hr']['avg']
            hr_intensity = (hr_avg - self.rest_hr) / (self.max_hr - self.rest_hr) * 100
            print(f"💓 Heart Rate Intensity: {hr_intensity:.1f}% of HR reserve")
        
        print(f"\n💡 Recommendations:")
        if self.TSS > self.target_tss * 1.2:
            print("   - Consider reducing intensity in future sessions")
        elif self.TSS < self.target_tss * 0.8:
            print("   - Could increase intensity for target training load")
        
        if self.avg_power/self.np_calc < 0.8:
            print("   - Focus on more consistent pacing to improve efficiency")
        
        if self.zone_percentages.get('Z7 (Neuromuscular)', 0) > 5:
            print("   - High neuromuscular load - ensure adequate recovery")
        
        print("\n" + "="*50)

    def calculate_hr_strain(self):
        """Analyze HR response using practical training metrics."""
        if 'heart_rate' not in self.df.columns:
            print("Heart rate data required for HR strain analysis.")
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
        # Calculate total heartbeats
        total_beats = self.df['heart_rate'].sum() / 60  # Assuming 1-second sampling
        total_work_kj = (self.df['power'].sum() / 60) / 1000  # Convert to kJ
        total_distance_km = self.df['distance'].iloc[-1] / 1000 if 'distance' in self.df.columns else 0
        
        cci_kj = total_beats / total_work_kj if total_work_kj > 0 else 0
        cci_km = total_beats / total_distance_km if total_distance_km > 0 else 0
        
        # 3. TRIMP (HR-based Training Load)
        # Banister formula: Σ(duration × HRr × e^(β·HRr))
        trimp_score = 0
        beta = 1.92  # For men (use 1.67 for women)
        
        for i in range(len(self.df)):
            hr_reserve_pct = self.df['hr_reserve_pct'].iloc[i]
            # Convert to decimal (0-1 scale)
            hr_reserve_decimal = hr_reserve_pct / 100
            # TRIMP calculation
            trimp_score += hr_reserve_decimal * np.exp(beta * hr_reserve_decimal)
        
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
        ax2.set_ylabel('Cardiac Cost Index')
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
        for i in range(len(self.df)):
            hr_reserve_decimal = self.df['hr_reserve_pct'].iloc[i] / 100
            current_trimp += hr_reserve_decimal * np.exp(beta * hr_reserve_decimal)
            cumulative_trimp.append(current_trimp)
        
        ax3.plot(time_minutes, cumulative_trimp, color='purple', linewidth=2)
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Cumulative TRIMP Score')
        ax3.set_title('Training Load (TRIMP) Over Time')
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
        plt.show()
        
        # Print practical HR analysis
        print(f"\n=== HR TRAINING LOAD ANALYSIS (Practical Metrics) ===")
        print(f"Average HR Reserve: {self.df['hr_reserve_pct'].mean():.1f}%")
        print(f"Peak HR Reserve: {self.df['hr_reserve_pct'].max():.1f}%")
        
        if aerobic_decoupling:
            print(f"\n--- ENHANCED AEROBIC DECOUPLING ANALYSIS ---")
            print(f"Number of Steady Efforts Found: {len(aerobic_decoupling)}")
            
            # Group by duration
            durations_2min = [dec for dec in aerobic_decoupling if dec['duration'] <= 2]
            durations_3min = [dec for dec in aerobic_decoupling if 2 < dec['duration'] <= 3]
            durations_4min = [dec for dec in aerobic_decoupling if dec['duration'] > 3]
            
            print(f"  - 2-minute efforts: {len(durations_2min)}")
            print(f"  - 3-minute efforts: {len(durations_3min)}")
            print(f"  - 4-minute efforts: {len(durations_4min)}")
            
            # Overall statistics
            avg_drift = np.mean([dec['drift_pct'] for dec in aerobic_decoupling])
            avg_power = np.mean([dec['avg_power'] for dec in aerobic_decoupling])
            avg_power_cv = np.mean([dec['power_cv'] for dec in aerobic_decoupling])
            
            print(f"\nOverall Statistics:")
            print(f"  - Average Decoupling: {avg_drift:.1f}%")
            print(f"  - Average Power: {avg_power:.0f}W ({avg_power/self.ftp*100:.1f}% FTP)")
            print(f"  - Average Power CV: {avg_power_cv:.1f}%")
            
            # Categorize decoupling
            if avg_drift < 5:
                decoupling_category = "Good"
            elif avg_drift < 10:
                decoupling_category = "Moderate"
            else:
                decoupling_category = "Poor"
            print(f"  - Decoupling Quality: {decoupling_category}")
            
            # Show best and worst efforts
            if len(aerobic_decoupling) > 1:
                best_effort = min(aerobic_decoupling, key=lambda x: abs(x['drift_pct']))
                worst_effort = max(aerobic_decoupling, key=lambda x: abs(x['drift_pct']))
                
                print(f"\nBest Steady Effort:")
                print(f"  - Duration: {best_effort['duration']:.1f}min")
                print(f"  - Power: {best_effort['avg_power']:.0f}W")
                print(f"  - Decoupling: {best_effort['drift_pct']:.1f}%")
                
                print(f"\nMost Challenging Effort:")
                print(f"  - Duration: {worst_effort['duration']:.1f}min")
                print(f"  - Power: {worst_effort['avg_power']:.0f}W")
                print(f"  - Decoupling: {worst_effort['drift_pct']:.1f}%")
        else:
            print(f"\n--- AEROBIC DECOUPLING ANALYSIS ---")
            print(f"No steady efforts found for decoupling analysis.")
            print(f"Suggestions:")
            print(f"  - Try longer rides (>30 minutes)")
            print(f"  - Include more consistent power efforts")
            print(f"  - Ensure power > 50% FTP for meaningful analysis")
            print(f"  - Check that heart rate data is available and valid")
        
        print(f"\n--- CARDIOVASCULAR ECONOMY ---")
        print(f"Cardiac Cost Index (beats/kJ): {cci_kj:.1f}")
        if total_distance_km > 0:
            print(f"Cardiac Cost Index (beats/km): {cci_km:.1f}")
        
        print(f"\n--- TRAINING LOAD ---")
        print(f"Total TRIMP Score: {trimp_score:.1f}")
        
        # Categorize TRIMP load
        if trimp_score < 100:
            load_category = "Light"
        elif trimp_score < 200:
            load_category = "Moderate"
        elif trimp_score < 300:
            load_category = "Hard"
        else:
            load_category = "Very Hard"
        print(f"Training Load: {load_category}")
        
        print(f"\n--- METABOLIC COST ---")
        print(f"Average MET: {avg_met:.1f}")
        print(f"Maximum MET: {max_met:.1f}")
        print(f"Average VO₂: {avg_vo2:.1f} ml/kg/min")
        print(f"Maximum VO₂: {max_vo2:.1f} ml/kg/min")
        print(f"Total Energy Expenditure: {total_energy:.0f} MET-minutes")
        
        return aerobic_decoupling, trimp_score
    
    def analyze_heat_stress(self):
        """Analyze heat stress factors and HR response correlation."""
        if 'heart_rate' not in self.df.columns:
            print("Heart rate data required for heat stress analysis.")
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
        
        # Find power changes and measure HR response
        power_changes = self.df['power'].diff().abs()
        significant_changes = power_changes > 20  # 20W change threshold
        
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
            ax1.set_ylabel('Heat Stress Index (0-100)')
            ax1.set_title('Heat Stress Index Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Insufficient data\nfor heat stress analysis', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Heat Stress Index')
        
        # 2. HR Drift Rate vs Power
        if hr_drift_rate:
            ax2.scatter([self.df['power'].iloc[int(t*60)] for t in drift_times], 
                       hr_drift_rate, alpha=0.7, s=60, color='orange')
            ax2.set_xlabel('Average Power (W)')
            ax2.set_ylabel('HR Drift Rate (bpm/min)')
            ax2.set_title('HR Drift Rate vs Power')
            ax2.grid(True, alpha=0.3)
            
            # Add reference lines
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='Moderate Drift')
            ax2.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='High Drift')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No significant\nHR drift detected', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('HR Drift Rate vs Power')
        
        # 3. HR Response Lag Distribution
        if hr_response_lag:
            ax3.hist(hr_response_lag, bins=15, alpha=0.7, color='blue', edgecolor='black')
            ax3.set_xlabel('HR Response Time (seconds)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('HR Response Lag Distribution')
            ax3.grid(True, alpha=0.3)
            
            # Add statistics
            avg_lag = np.mean(hr_response_lag)
            ax3.axvline(x=avg_lag, color='red', linestyle='--', alpha=0.8, 
                       label=f'Mean: {avg_lag:.1f}s')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No HR response\nlags detected', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('HR Response Lag Distribution')
        
        # 4. HR Response Amplitude vs Time
        if hr_response_amplitude:
            response_times = np.arange(len(hr_response_amplitude)) * 30  # Approximate times
            ax4.scatter(response_times, hr_response_amplitude, alpha=0.6, s=40, color='purple')
            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('HR Response Amplitude (bpm)')
            ax4.set_title('HR Response Amplitude Over Time')
            ax4.grid(True, alpha=0.3)
            
            # Add trend line
            if len(hr_response_amplitude) > 5:
                z = np.polyfit(response_times, hr_response_amplitude, 1)
                p = np.poly1d(z)
                ax4.plot(response_times, p(response_times), "r--", alpha=0.8, linewidth=2)
        else:
            ax4.text(0.5, 0.5, 'No HR response\namplitudes detected', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('HR Response Amplitude Over Time')
        
        plt.tight_layout()
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
        if 'power' not in self.df.columns or 'heart_rate' not in self.df.columns:
            print("Power and heart rate data required for efficiency analysis.")
            return
        
        # Calculate power-to-HR ratio (efficiency metric) and filter out zero values
        self.df['power_hr_ratio'] = self.df['power'] / self.df['heart_rate']
        
        # Filter out zero and extreme values for meaningful analysis
        valid_mask = (self.df['power'] > 0) & (self.df['heart_rate'] > 50) & (self.df['power_hr_ratio'] > 0)
        valid_data = self.df[valid_mask].copy()
        
        if len(valid_data) == 0:
            print("No valid data for efficiency analysis.")
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
        efficiency_valid = efficiency_rolling.dropna()
        if len(efficiency_valid) > 10:
            # Get corresponding time values for non-NaN efficiency values
            time_minutes_valid = time_minutes[efficiency_rolling.notna()]
            efficiency_trend = np.polyfit(time_minutes_valid, efficiency_valid, 1)
            efficiency_slope = efficiency_trend[0]  # W/bpm per minute
            trend_label = f"Slope: {efficiency_slope:.3f} W/bpm/min"
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
        
        # Categorize efficiency
        efficiency_categories = []
        for ratio in valid_data['power_hr_ratio']:
            if ratio < efficiency_zones['Low']:
                efficiency_categories.append('Very Low')
            elif ratio < efficiency_zones['Moderate']:
                efficiency_categories.append('Low')
            elif ratio < efficiency_zones['High']:
                efficiency_categories.append('Moderate')
            elif ratio < efficiency_zones['Very High']:
                efficiency_categories.append('High')
            else:
                efficiency_categories.append('Very High')
        
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
        
        ax1.set_title('Efficiency Over Time with Trend Analysis')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Efficiency (W/bpm)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Efficiency vs Power (filtered)
        ax2.scatter(valid_data['power'], valid_data['power_hr_ratio'], alpha=0.5, s=15)
        ax2.set_xlabel('Power (W)')
        ax2.set_ylabel('Efficiency (W/bpm)')
        ax2.set_title('Efficiency vs Power (Filtered Data)')
        ax2.grid(True, alpha=0.3)
        
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
        plt.show()
        
        # Print enhanced efficiency analysis
        print(f"\n=== ENHANCED POWER-TO-HR EFFICIENCY ANALYSIS ===")
        print(f"Valid Data Points: {len(valid_data)} / {len(self.df)} ({len(valid_data)/len(self.df)*100:.1f}%)")
        print(f"Average Efficiency: {efficiency_mean:.2f} W/bpm")
        print(f"Efficiency Range: {efficiency_mean - efficiency_std:.2f} to {efficiency_mean + efficiency_std:.2f} W/bpm")
        print(f"Peak Efficiency: {valid_data['power_hr_ratio'].max():.2f} W/bpm")
        print(f"Lowest Efficiency: {valid_data['power_hr_ratio'].min():.2f} W/bpm")
        print(f"Efficiency Trend: {trend_label}")
        
        # Interpret efficiency trend
        if abs(efficiency_slope) > 0.01:
            if efficiency_slope > 0:
                print(f"📈 Efficiency improving over time (+{efficiency_slope:.3f} W/bpm/min)")
            else:
                print(f"📉 Efficiency declining over time ({efficiency_slope:.3f} W/bpm/min)")
        else:
            print("📊 Efficiency relatively stable over time")
        
        # Efficiency zone breakdown
        for zone, count in efficiency_counts.items():
            percentage = (count / len(valid_data)) * 100
            print(f"{zone} Efficiency: {percentage:.1f}% of valid data")
    
    def analyze_fatigue_patterns(self):
        """Analyze fatigue patterns with 15 segments and terrain-normalized drift analysis."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
        
        # Use at least 20 segments for better resolution
        num_segments = max(20, len(self.df) // 100)  # At least 20 segments, more for longer rides
        segment_length = len(self.df) // num_segments
        segments = []
        
        # Calculate time in minutes
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        
        # Create 15 segments with detailed metrics
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length if i < num_segments - 1 else len(self.df)
            
            segment_data = self.df.iloc[start_idx:end_idx]
            segment_time_mid = segment_data['timestamp'].iloc[len(segment_data)//2]
            time_midpoint = (segment_time_mid - self.df['timestamp'].iloc[0]).total_seconds() / 60
            
            segments.append({
                'segment': i + 1,
                'avg_power': segment_data['power'].mean(),
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
        
        # 1. Power drift over time with line of best fit
        time_segments = [seg['time_midpoint'] for seg in segments]
        power_by_segment = [seg['avg_power'] for seg in segments]
        
        ax1.plot(time_segments, power_by_segment, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Power')
        
        # Add line of best fit
        if len(time_segments) > 2:
            power_trend = np.polyfit(time_segments, power_by_segment, 1)
            power_trend_line = power_trend[0] * np.array(time_segments) + power_trend[1]
            ax1.plot(time_segments, power_trend_line, '--', color='red', linewidth=2, 
                    label=f'Trend: {power_trend[0]:.1f} W/min')
        
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
                    label=f'Trend: {work_trend[0]:.1f} W/hr/min')
        
        ax3.set_title('Cumulative Work Done (W/hr)')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Cumulative Work (W/hr)')
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
        plt.show()
        
        # Print enhanced fatigue analysis
        print(f"\n=== ENHANCED FATIGUE & DRIFT ANALYSIS (15 Segments) ===")
        
        # Calculate overall trends
        if len(time_segments) > 2:
            power_trend = np.polyfit(time_segments, power_by_segment, 1)
            print(f"Power Drift Rate: {power_trend[0]:.2f} W/min")
            
            # Calculate cumulative work trend
            work_trend = np.polyfit(time_segments, cumulative_work, 1)
            print(f"Cumulative Work Trend: {work_trend[0]:.1f} W/hr/min")
            print(f"Total Work Done: {cumulative_work[-1]:.1f} W/hr")
            print(f"Average Work Rate: {cumulative_work[-1] / time_segments[-1]:.1f} W/hr")
            
            if 'cadence' in self.df.columns:
                cadence_trend = np.polyfit(time_segments, cadence_by_segment, 1)
                print(f"Cadence Drift Rate: {cadence_trend[0]:.2f} rpm/min")
            
            if 'heart_rate' in self.df.columns and len(hr_power_ratios) > 0:
                valid_ratios = [ratio for ratio in hr_power_ratios if not np.isnan(ratio)]
                if len(valid_ratios) > 2:
                    valid_times = [time_segments[i] for i, ratio in enumerate(hr_power_ratios) if not np.isnan(ratio)]
                    ratio_trend = np.polyfit(valid_times, valid_ratios, 1)
                    print(f"HR Efficiency Drift: {ratio_trend[0]:.3f} W/bpm/min")
                    print(f"Average Power/HR Ratio: {np.mean(valid_ratios):.2f} W/bpm")
        
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
        corr_matrix = self.df[variables].corr()
        
        # Create relationship analysis visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Variable Relationship Analysis (with Elevation/Grade)', fontweight='bold', fontsize=14)
        
        # 1. Correlation heatmap
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
        
        # 4. Speed vs Power relationship with trend line (if available)
        if 'speed_kmh' in self.df.columns:
            # Filter out zero values
            speed_mask = (self.df['power'] > 0) & (self.df['speed_kmh'] > 0)
            if speed_mask.sum() > 10:
                speed_data = self.df[speed_mask]['speed_kmh']
                power_speed = self.df[speed_mask]['power']
                
                ax4.scatter(speed_data, power_speed, alpha=0.5, s=15)
                
                # Add trend line
                speed_trend = np.polyfit(speed_data, power_speed, 1)
                speed_line = speed_trend[0] * speed_data + speed_trend[1]
                ax4.plot(speed_data, speed_line, color='red', linestyle='--', alpha=0.8,
                        label=f'Slope: {speed_trend[0]:.1f} W/(km/h)')
                ax4.legend()
            
            ax4.set_xlabel('Speed (km/h)')
            ax4.set_ylabel('Power (W)')
            ax4.set_title('Speed vs Power')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
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
        """Print a comprehensive metrics table similar to TrainingPeaks."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
        
        print("\n" + "="*80)
        print(f"{'COMPREHENSIVE RIDE METRICS TABLE':^80}")
        print("="*80)
        
        # Power Metrics
        print(f"\n{'POWER METRICS':^80}")
        print("-" * 80)
        print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
        print("-" * 80)
        print(f"{'Average Power':<30} {self.avg_power:<20.0f} {'W':<15} {'':<15}")
        print(f"{'Normalized Power':<30} {self.np_calc:<20.0f} {'W':<15} {'':<15}")
        print(f"{'Max Power':<30} {self.df['power'].max():<20.0f} {'W':<15} {'':<15}")
        print(f"{'Min Power':<30} {self.df['power'].min():<20.0f} {'W':<15} {'':<15}")
        
        # Check if VI is available
        if hasattr(self, 'vi_calc'):
            print(f"{'Power Variability Index':<30} {self.vi_calc:<20.2f} {'':<15} {'NP/AP':<15}")
        
        # FTP-based metrics
        if hasattr(self, 'ftp') and self.ftp > 0:
            if hasattr(self, 'if_calc'):
                print(f"{'Intensity Factor':<30} {self.if_calc:<20.2f} {'':<15} {'NP/FTP':<15}")
            if hasattr(self, 'tss_calc'):
                print(f"{'Training Stress Score':<30} {self.tss_calc:<20.0f} {'':<15} {'':<15}")
            print(f"{'Power at FTP %':<30} {(self.avg_power/self.ftp*100):<20.1f} {'%':<15} {'':<15}")
        
        # Heart Rate Metrics
        if 'heart_rate' in self.df.columns:
            print(f"\n{'HEART RATE METRICS':^80}")
            print("-" * 80)
            print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
            print("-" * 80)
            print(f"{'Average HR':<30} {self.metrics['hr']['avg']:<20.0f} {'bpm':<15} {'':<15}")
            print(f"{'Max HR':<30} {self.metrics['hr']['max']:<20.0f} {'bpm':<15} {'':<15}")
            print(f"{'Min HR':<30} {self.metrics['hr']['min']:<20.0f} {'bpm':<15} {'':<15}")
            
            if hasattr(self, 'max_hr') and self.max_hr > 0:
                hr_reserve = self.max_hr - self.rest_hr
                avg_hr_reserve = (self.metrics['hr']['avg'] - self.rest_hr) / hr_reserve * 100
                print(f"{'Avg HR Reserve':<30} {avg_hr_reserve:<20.1f} {'%':<15} {'':<15}")
        
        # Cadence Metrics
        if 'cadence' in self.df.columns:
            print(f"\n{'CADENCE METRICS':^80}")
            print("-" * 80)
            print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
            print("-" * 80)
            print(f"{'Average Cadence':<30} {self.metrics['cadence']['avg']:<20.0f} {'rpm':<15} {'':<15}")
            print(f"{'Max Cadence':<30} {self.metrics['cadence']['max']:<20.0f} {'rpm':<15} {'':<15}")
            print(f"{'Min Cadence':<30} {self.metrics['cadence']['min']:<20.0f} {'rpm':<15} {'':<15}")
        
        # Speed Metrics
        if 'speed_kmh' in self.df.columns:
            print(f"\n{'SPEED METRICS':^80}")
            print("-" * 80)
            print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
            print("-" * 80)
            print(f"{'Average Speed':<30} {self.metrics['speed']['avg']:<20.1f} {'km/h':<15} {'':<15}")
            print(f"{'Max Speed':<30} {self.metrics['speed']['max']:<20.1f} {'km/h':<15} {'':<15}")
            print(f"{'Min Speed':<30} {self.metrics['speed']['min']:<20.1f} {'km/h':<15} {'':<15}")
        
        # Time and Distance Metrics
        print(f"\n{'TIME & DISTANCE METRICS':^80}")
        print("-" * 80)
        print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
        print("-" * 80)
        print(f"{'Duration':<30} {self.duration_hr:<20.2f} {'hours':<15} {'':<15}")
        print(f"{'Duration':<30} {(self.duration_hr*60):<20.0f} {'minutes':<15} {'':<15}")
        
        if 'distance_km' in self.df.columns:
            print(f"{'Distance':<30} {self.df['distance_km'].max():<20.2f} {'km':<15} {'':<15}")
            print(f"{'Average Speed':<30} {(self.df['distance_km'].max()/self.duration_hr):<20.1f} {'km/h':<15} {'':<15}")
        
        # Zone Analysis
        if hasattr(self, 'zone_percentages'):
            print(f"\n{'ZONE ANALYSIS':^80}")
            print("-" * 80)
            print(f"{'Zone':<20} {'Time':<15} {'%':<10} {'Description':<35}")
            print("-" * 80)
            
            for zone_name, percentage in self.zone_percentages.items():
                if percentage > 0:
                    time_minutes = (percentage / 100) * self.duration_hr * 60
                    print(f"{zone_name:<20} {time_minutes:<15.1f} {percentage:<10.1f} {'':<35}")
        
        # HR Zone Analysis
        if 'heart_rate' in self.df.columns and hasattr(self, 'hr_zones') and hasattr(self, 'hr_zone_percentages'):
            print(f"\n{'HEART RATE ZONE ANALYSIS':^80}")
            print("-" * 80)
            print(f"{'Zone':<20} {'Time':<15} {'%':<10} {'Description':<35}")
            print("-" * 80)
            
            for zone_name, percentage in self.hr_zone_percentages.items():
                if percentage > 0:
                    time_minutes = (percentage / 100) * self.duration_hr * 60
                    print(f"{zone_name:<20} {time_minutes:<15.1f} {percentage:<10.1f} {'':<35}")
        
        # Performance Metrics
        print(f"\n{'PERFORMANCE METRICS':^80}")
        print("-" * 80)
        print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
        print("-" * 80)
        
        # Calculate efficiency metrics
        if 'heart_rate' in self.df.columns:
            power_hr_ratio = self.avg_power / self.metrics['hr']['avg'] if self.metrics['hr']['avg'] > 0 else 0
            print(f"{'Power/HR Ratio':<30} {power_hr_ratio:<20.2f} {'W/bpm':<15} {'Efficiency':<15}")
        
        if 'cadence' in self.df.columns:
            power_cadence_ratio = self.avg_power / self.metrics['cadence']['avg'] if self.metrics['cadence']['avg'] > 0 else 0
            print(f"{'Power/Cadence Ratio':<30} {power_cadence_ratio:<20.2f} {'W/rpm':<15} {'Torque':<15}")
        
        # Data Quality
        if hasattr(self, 'data_quality_report'):
            print(f"\n{'DATA QUALITY':^80}")
            print("-" * 80)
            print(f"{'Metric':<30} {'Value':<20} {'Unit':<15} {'Notes':<15}")
            print("-" * 80)
            print(f"{'Data Quality Score':<30} {self.data_quality_report['data_quality_score']:<20.1f} {'/100':<15} {'':<15}")
            print(f"{'Interpolated Data':<30} {self.data_quality_report['interpolated_pct']:<20.1f} {'%':<15} {'':<15}")
            print(f"{'Outliers Removed':<30} {self.data_quality_report['outliers_removed']:<20.0f} {'':<15} {'':<15}")
        
        print("\n" + "="*80)
        print("End of Comprehensive Metrics Table")
        print("="*80)


def main():
    """Main function to run the enhanced analysis with proper athlete constants."""
    # Initialize analyzer with enhanced athlete profile
    analyzer = CyclingAnalyzer(
        athlete_name="Your Name",
        ftp=280,           # Functional Threshold Power (W)
        max_hr=185,        # Maximum Heart Rate (bpm)
        rest_hr=45,        # Resting Heart Rate (bpm)
        weight_kg=70,      # Weight (kg)
        height_cm=175,     # Height (cm)
        # Enhanced physiological parameters
        lactate_rest=1.2,  # Resting lactate (mmol/L)
        lactate_peak=8.0,  # Peak lactate (mmol/L)
        w_prime_tau=386,   # W' recovery time constant (s)
        # Data quality thresholds
        max_interpolation_pct=5.0,  # Max % interpolated data
        power_outlier_threshold=3.0,  # Power outlier detection (std devs)
        hr_outlier_threshold=3.0,     # HR outlier detection (std devs)
        cadence_outlier_threshold=3.0  # Cadence outlier detection (std devs)
    )
    
    # Load your FIT file
    file_path = "/Users/tajkrieger/Downloads/Truckee_Gravel_6th.fit"  # Update this path
    
    if analyzer.load_fit_file(file_path):
        # Enhanced data cleaning and validation
        print("\n🧹 Enhanced data cleaning and validation...")
        analyzer.clean_and_smooth_data()
        
        # Check data quality
        if hasattr(analyzer, 'data_quality_report'):
            print("📋 Data Quality Report:")
            for metric, value in analyzer.data_quality_report.items():
                print(f"  {metric}: {value}")
        
        # Calculate metrics with moving-time-based algorithms
        print("\n📈 Calculating enhanced metrics...")
        analyzer.calculate_metrics()
        
        # Generate comprehensive analysis
        print("\n📊 Generating comprehensive analysis...")
        analyzer.print_summary()
        analyzer.create_dashboard()
        
        # Print comprehensive metrics table
        print("\n📋 Printing comprehensive metrics table...")
        analyzer.print_comprehensive_metrics_table()
        
        # Advanced physiological analysis
        print("\n🔬 Running advanced physiological analysis...")
        
        # Critical Power analysis
        cp_est, w_prime_est = analyzer.estimate_critical_power()
        if cp_est and analyzer.show_w_prime_balance:
            analyzer.calculate_w_prime_balance(cp_est, w_prime_est)
        
        # Lactate estimation (fixed physiology)
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