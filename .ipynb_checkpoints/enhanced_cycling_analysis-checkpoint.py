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

# Set style for modern, minimalist plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class CyclingAnalyzer:
    """Main class for cycling data analysis with personalized configuration."""
    
    def __init__(self, athlete_name="Your Name", ftp=250, max_hr=185, rest_hr=45, 
                 weight_kg=70, height_cm=175):
        """Initialize analyzer with athlete profile."""
        self.athlete_name = athlete_name
        self.ftp = ftp
        self.max_hr = max_hr
        self.rest_hr = rest_hr
        self.weight_kg = weight_kg
        self.height_cm = height_cm
        
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
        self.w_prime_tau = 546
        self.lactate_rest = 1.0
        self.lactate_peak = 12.0
        
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
        """Clean and smooth the data."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return False
        
        # Convert units
        if 'speed' in self.df.columns:
            self.df['speed_kmh'] = self.df['speed'] * 3.6
        if 'distance' in self.df.columns:
            self.df['distance_km'] = self.df['distance'] / 1000
        
        # Interpolate missing values
        for col in ['power', 'cadence', 'heart_rate', 'speed_kmh']:
            if col in self.df.columns:
                self.df[col] = self.df[col].interpolate(limit_direction='both')
        
        # Apply smoothing
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
        
        print('Data cleaned and smoothed successfully.')
        return True
    
    def calculate_metrics(self):
        """Calculate all performance metrics."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return False
        
        # Core calculations
        self.np_calc = (self.df['power'].rolling(window=self.smoothing_window, 
                                                min_periods=1).mean() ** 4).mean() ** 0.25
        self.avg_power = self.df['power'].mean()
        self.max_power = self.df['power'].max()
        self.power_std = self.df['power'].std()
        
        # Duration and distance
        self.duration_hr = (self.df['timestamp'].iloc[-1] - 
                           self.df['timestamp'].iloc[0]).total_seconds() / 3600
        self.total_distance = (self.df['distance_km'].iloc[-1] 
                              if 'distance_km' in self.df.columns else 0)
        
        # Training metrics
        self.IF = self.np_calc / self.ftp
        self.VI = self.np_calc / self.avg_power
        self.TSS = (self.duration_hr * self.np_calc * self.IF) / self.ftp * 100
        
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
                'HR Z1 (%)', 'HR Z2 (%)', 'HR Z3 (%)', 'HR Z4 (%)', 'HR Z5 (%)', 'HR Z6 (%)'
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
                f"{self.hr_zone_percentages.get('Z6 (Anaerobic)', 0):.1f}"
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
        """Create comprehensive visualization dashboard."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.athlete_name} - Ride Analysis Dashboard', fontsize=16, fontweight='bold')
        
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        
        # 1. Power Zones
        if 'power' in self.df.columns:
            zone_labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7']
            zone_values = [self.zone_percentages.get(f'Z{i} (Recovery)', 0) if i == 1 else
                          self.zone_percentages.get(f'Z{i} (Endurance)', 0) if i == 2 else
                          self.zone_percentages.get(f'Z{i} (Tempo)', 0) if i == 3 else
                          self.zone_percentages.get(f'Z{i} (Threshold)', 0) if i == 4 else
                          self.zone_percentages.get(f'Z{i} (VO2max)', 0) if i == 5 else
                          self.zone_percentages.get(f'Z{i} (Anaerobic)', 0) if i == 6 else
                          self.zone_percentages.get(f'Z{i} (Neuromuscular)', 0) for i in range(1, 8)]
            
            colors = ['#2E8B57', '#3CB371', '#FFD700', '#FF8C00', '#FF4500', '#DC143C', '#8B0000']
            axes[0,0].pie(zone_values, labels=zone_labels, autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0,0].set_title('Power Zone Distribution')
        
        # 2. Power Profile
        axes[0,1].plot(time_minutes, self.df['power'], alpha=0.7, linewidth=0.8, color='#1f77b4')
        axes[0,1].axhline(y=self.avg_power, color='red', linestyle='--', alpha=0.8, label=f'Avg: {self.avg_power:.0f}W')
        axes[0,1].axhline(y=self.np_calc, color='orange', linestyle='--', alpha=0.8, label=f'NP: {self.np_calc:.0f}W')
        axes[0,1].set_xlabel('Time (minutes)')
        axes[0,1].set_ylabel('Power (W)')
        axes[0,1].set_title('Power Profile')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Heart Rate Profile
        if 'heart_rate' in self.df.columns:
            axes[0,2].plot(time_minutes, self.df['heart_rate'], alpha=0.7, linewidth=0.8, color='#d62728')
            axes[0,2].axhline(y=self.metrics['hr']['avg'], color='red', linestyle='--', alpha=0.8, 
                              label=f"Avg: {self.metrics['hr']['avg']:.0f}bpm")
            axes[0,2].set_xlabel('Time (minutes)')
            axes[0,2].set_ylabel('Heart Rate (bpm)')
            axes[0,2].set_title('Heart Rate Profile')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Speed Profile
        if 'speed_kmh' in self.df.columns:
            axes[1,0].plot(time_minutes, self.df['speed_kmh'], alpha=0.7, linewidth=0.8, color='#2ca02c')
            axes[1,0].axhline(y=self.metrics['speed']['avg'], color='red', linestyle='--', alpha=0.8,
                              label=f"Avg: {self.metrics['speed']['avg']:.1f}km/h")
            axes[1,0].set_xlabel('Time (minutes)')
            axes[1,0].set_ylabel('Speed (km/h)')
            axes[1,0].set_title('Speed Profile')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. Cadence Profile
        if 'cadence' in self.df.columns:
            axes[1,1].plot(time_minutes, self.df['cadence'], alpha=0.7, linewidth=0.8, color='#9467bd')
            axes[1,1].axhline(y=self.metrics['cadence']['avg'], color='red', linestyle='--', alpha=0.8,
                              label=f"Avg: {self.metrics['cadence']['avg']:.0f}rpm")
            axes[1,1].set_xlabel('Time (minutes)')
            axes[1,1].set_ylabel('Cadence (rpm)')
            axes[1,1].set_title('Cadence Profile')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Power vs Heart Rate
        if 'heart_rate' in self.df.columns:
            scatter = axes[1,2].scatter(self.df['power'], self.df['heart_rate'], alpha=0.5, c=time_minutes, cmap='viridis')
            axes[1,2].set_xlabel('Power (W)')
            axes[1,2].set_ylabel('Heart Rate (bpm)')
            axes[1,2].set_title('Power vs Heart Rate')
            cbar = plt.colorbar(scatter, ax=axes[1,2])
            cbar.set_label('Time (minutes)')
        
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
        """Calculate W' balance over time."""
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
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_minutes, self.df['w_prime_bal'], color='purple', linewidth=1)
        plt.fill_between(time_minutes, self.df['w_prime_bal'], alpha=0.3, color='purple')
        plt.title("W' Balance Over Time")
        plt.xlabel('Time (minutes)')
        plt.ylabel('W\' Balance (J)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
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
        """Estimate lactate levels throughout the ride."""
        if self.df is None:
            print("No data loaded. Please load a FIT file first.")
            return
        
        def estimate_lactate_func(power, ftp):
            if power <= ftp:
                return self.lactate_rest + (power / ftp) * 2
            else:
                return self.lactate_rest + 2 + (self.lactate_peak - 2) * (1 - np.exp(-(power - ftp) / (0.1 * ftp)))
        
        self.df['lactate_est'] = self.df['power'].apply(lambda p: estimate_lactate_func(p, self.ftp))
        
        time_minutes = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds() / 60
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_minutes, self.df['lactate_est'], color='red', linewidth=1)
        plt.fill_between(time_minutes, self.df['lactate_est'], alpha=0.3, color='red')
        plt.title('Estimated Lactate (mmol/L)')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Lactate (mmol/L)')
        plt.grid(True, alpha=0.3)
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


def main():
    """Main function to run the analysis."""
    # Initialize analyzer with your profile
    analyzer = CyclingAnalyzer(
        athlete_name="Your Name",
        ftp=250,
        max_hr=185,
        rest_hr=45,
        weight_kg=70,
        height_cm=175
    )
    
    # Load your FIT file
    file_path = "/Users/tajkrieger/Downloads/Truckee_Gravel_6th.fit"  # Update this path
    
    if analyzer.load_fit_file(file_path):
        # Process the data
        analyzer.clean_and_smooth_data()
        analyzer.calculate_metrics()
        
        # Generate reports
        analyzer.print_summary()
        analyzer.create_dashboard()
        
        # Advanced analysis (optional)
        if analyzer.show_advanced_plots:
            analyzer.analyze_torque()
        
        if analyzer.show_lactate_estimation:
            analyzer.estimate_lactate()
        
        # Critical Power analysis
        cp_est, w_prime_est = analyzer.estimate_critical_power()
        if cp_est and analyzer.show_w_prime_balance:
            analyzer.calculate_w_prime_balance(cp_est, w_prime_est)
        
        # Performance insights
        analyzer.print_insights()
    else:
        print("Failed to load FIT file. Please check the file path.")


if __name__ == "__main__":
    main() 