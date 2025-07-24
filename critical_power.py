#!/usr/bin/env python3
"""
Enhanced Critical Power (CP) Analysis Module
===========================================

A comprehensive implementation of Critical Power analysis with advanced features:

Improvements:
1. Light smoothing (3s rolling median) before power calculations
2. Smart duration filtering based on sampling rate
3. Tightened parameter bounds for realistic ranges
4. Log-transformed durations for linear regression
5. AIC/BIC metrics for model selection
6. Enhanced W' balance plots with zones
7. Export results to CSV/JSON
8. Track CP/W' over time across sessions

References:
- Jones & Vanhatalo (2010): The 'Critical Power' Concept: Applications to Sports Performance
- Skiba et al. (2012): Modeling the expenditure and reconstitution of work capacity above critical power
- Skiba et al. (2014): The W' balance model: Mathematical and physiological considerations
- Triska et al. (2021): Critical power: A comprehensive review of theory and applications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
import fitparse
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta
import os
import json
from scipy.stats import linregress

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

@dataclass
class CPResults:
    """Enhanced data structure for Critical Power analysis results"""
    cp_watts: float
    w_prime_joules: float
    tte_minutes: float
    r_squared: float
    aic: float
    bic: float
    model_type: str
    power_duration_data: Dict[str, float]
    fitted_curve: Optional[np.ndarray] = None
    curve_durations: Optional[np.ndarray] = None
    sampling_rate: Optional[float] = None
    smoothing_applied: bool = False

@dataclass
class WPrimeBalance:
    """Enhanced data structure for W' balance analysis results"""
    w_prime_initial: float
    w_prime_balance: np.ndarray
    time_points: np.ndarray
    cp_threshold: float
    depletion_zones: List[Tuple[float, float]]
    recovery_zones: List[Tuple[float, float]]
    exhaustion_time: Optional[float] = None

@dataclass
class CPSession:
    """Data structure for tracking CP/W' over time"""
    date: datetime
    cp_watts: float
    w_prime_joules: float
    tte_minutes: float
    r_squared: float
    model_type: str
    rides_analyzed: int
    notes: Optional[str] = None

class CriticalPowerAnalyzer:
    """
    Enhanced Critical Power Analysis Implementation
    
    Implements the hyperbolic model (P = W'/t + CP) and linear work-time model
    with advanced features for better accuracy and visualization.
    """
    
    def __init__(self, ftp: float = 250):
        """
        Initialize the Enhanced Critical Power Analyzer
        
        Args:
            ftp (float): Functional Threshold Power in watts (for validation)
        """
        self.ftp = ftp
        self.results = None
        self.w_prime_balance = None
        self.sessions_history = []
        
        # Standard durations for power-duration curve (in seconds)
        self.standard_durations = [3, 5, 10, 20, 30, 60, 120, 180, 300, 600, 1200, 1800]
        self.duration_names = ['3s', '5s', '10s', '20s', '30s', '1min', '2min', '3min', 
                              '5min', '10min', '20min', '30min']
        
        # Enhanced parameter bounds for more realistic ranges
        self.cp_bounds = (50, 800)  # Watts: 50W to 800W
        self.w_prime_bounds = (1000, 50000)  # Joules: 1kJ to 50kJ
        
        print("🔬 Enhanced Critical Power Analyzer initialized")
        print(f"📊 Standard durations: {self.duration_names}")
        print(f"🔧 CP bounds: {self.cp_bounds[0]}-{self.cp_bounds[1]}W")
        print(f"🔧 W' bounds: {self.w_prime_bounds[0]}-{self.w_prime_bounds[1]}J")
    
    def _apply_smoothing(self, power_series: pd.Series, sampling_rate: float = 1.0) -> pd.Series:
        """
        Apply light smoothing to power series before analysis
        
        Args:
            power_series (pd.Series): Raw power data
            sampling_rate (float): Data sampling rate in Hz
            
        Returns:
            pd.Series: Smoothed power data
        """
        # Calculate smoothing window (3 seconds)
        smoothing_window = int(3 * sampling_rate)
        if smoothing_window < 3:
            smoothing_window = 3  # Minimum 3 points
        
        # Apply rolling median smoothing
        smoothed_power = power_series.rolling(
            window=smoothing_window, 
            center=True, 
            min_periods=1
        ).median()
        
        return smoothed_power
    
    def _estimate_sampling_rate(self, data: pd.DataFrame) -> float:
        """
        Estimate sampling rate from timestamp data
        
        Args:
            data (pd.DataFrame): Data with timestamp column
            
        Returns:
            float: Estimated sampling rate in Hz
        """
        if 'timestamp' not in data.columns:
            return 1.0  # Default assumption
        
        # Calculate time differences
        time_diffs = data['timestamp'].diff().dt.total_seconds()
        
        # Remove outliers (gaps > 60 seconds)
        time_diffs = time_diffs[time_diffs <= 60]
        
        if len(time_diffs) == 0:
            return 1.0
        
        # Calculate median sampling rate
        median_interval = time_diffs.median()
        sampling_rate = 1.0 / median_interval if median_interval > 0 else 1.0
        
        return sampling_rate
    
    def _filter_durations_by_sampling_rate(self, sampling_rate: float) -> Tuple[List[int], List[str]]:
        """
        Filter durations based on sampling rate
        
        Args:
            sampling_rate (float): Data sampling rate in Hz
            
        Returns:
            Tuple[List[int], List[str]]: Filtered durations and names
        """
        filtered_durations = []
        filtered_names = []
        
        for duration, name in zip(self.standard_durations, self.duration_names):
            # Skip durations below 5s unless sampling rate > 1Hz
            if duration < 5 and sampling_rate <= 1.0:
                print(f"   ⚠️ Skipping {name} (duration < 5s, sampling rate ≤ 1Hz)")
                continue
            
            # Ensure we have enough data points for the duration
            min_points = duration * sampling_rate
            if min_points < 3:
                print(f"   ⚠️ Skipping {name} (insufficient data points)")
                continue
            
            filtered_durations.append(duration)
            filtered_names.append(name)
        
        return filtered_durations, filtered_names
    
    def load_activity_data(self, file_path: str) -> bool:
        """
        Enhanced data loading with sampling rate estimation
        
        Args:
            file_path (str): Path to activity file
            
        Returns:
            bool: True if successful
        """
        print(f"📁 Loading activity data from: {file_path}")
        
        try:
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'fit':
                self.data = self._load_fit_file(file_path)
            elif file_extension == 'csv':
                self.data = self._load_csv_file(file_path)
            else:
                print(f"❌ Unsupported file format: {file_extension}")
                return False
            
            if self.data is None or self.data.empty:
                print("❌ No data loaded from file")
                return False
            
            # Estimate sampling rate
            self.sampling_rate = self._estimate_sampling_rate(self.data)
            print(f"📊 Estimated sampling rate: {self.sampling_rate:.2f} Hz")
            
            print(f"✅ Successfully loaded {len(self.data)} data points")
            print(f"📊 Available columns: {list(self.data.columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Error during data ingestion: {str(e)}")
            return False
    
    def extract_power_duration_curve(self) -> Dict[str, float]:
        """
        Enhanced power-duration curve extraction with smoothing
        
        Returns:
            Dict[str, float]: Duration -> Maximal Power mapping
        """
        print("📈 Extracting power-duration curve with smoothing...")
        
        if 'power' not in self.data.columns:
            print("❌ No power data available")
            return {}
        
        # Apply smoothing to power data
        raw_power = self.data['power'].fillna(0)
        smoothed_power = self._apply_smoothing(raw_power, self.sampling_rate)
        
        print(f"   🔧 Applied {3}s rolling median smoothing")
        
        # Filter durations based on sampling rate
        filtered_durations, filtered_names = self._filter_durations_by_sampling_rate(self.sampling_rate)
        
        power_duration_data = {}
        
        for duration, name in zip(filtered_durations, filtered_names):
            if duration <= len(smoothed_power):
                # Calculate rolling average for this duration
                rolling_avg = smoothed_power.rolling(
                    window=duration, 
                    center=True, 
                    min_periods=duration
                ).mean()
                max_power = rolling_avg.max()
                
                if not pd.isna(max_power) and max_power > 0:
                    power_duration_data[name] = max_power
                    print(f"   {name}: {max_power:.0f}W")
        
        print(f"✅ Extracted {len(power_duration_data)} power-duration points")
        return power_duration_data
    
    def _calculate_aic_bic(self, y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> Tuple[float, float]:
        """
        Calculate AIC and BIC for model comparison
        
        Args:
            y_true (np.ndarray): Observed values
            y_pred (np.ndarray): Predicted values
            n_params (int): Number of model parameters
            
        Returns:
            Tuple[float, float]: AIC and BIC values
        """
        n = len(y_true)
        residuals = y_true - y_pred
        mse = np.mean(residuals ** 2)
        
        # AIC = n * ln(MSE) + 2k
        aic = n * np.log(mse) + 2 * n_params
        
        # BIC = n * ln(MSE) + k * ln(n)
        bic = n * np.log(mse) + n_params * np.log(n)
        
        return aic, bic
    
    def fit_hyperbolic_model(self, power_duration_data: Dict[str, float]) -> CPResults:
        """
        Enhanced hyperbolic model fitting with improved bounds and metrics
        
        Args:
            power_duration_data (Dict[str, float]): Duration -> Power mapping
            
        Returns:
            CPResults: Fitted model results
        """
        print("\n🔬 Fitting Enhanced Hyperbolic Model (P = W'/t + CP)...")
        
        # Convert duration names to seconds
        durations_seconds = []
        powers = []
        
        for name, power in power_duration_data.items():
            if name in self.duration_names:
                idx = self.duration_names.index(name)
                duration = self.standard_durations[idx]
                durations_seconds.append(duration)
                powers.append(power)
        
        if len(durations_seconds) < 3:
            print("❌ Insufficient data points for fitting")
            return None
        
        # Convert to numpy arrays
        durations = np.array(durations_seconds)
        powers = np.array(powers)
        
        # Define hyperbolic model function
        def hyperbolic_model(t, cp, w_prime):
            return w_prime / t + cp
        
        try:
            # Improved initial parameter estimates
            # CP estimate: average of longer durations (>5min)
            long_durations = durations >= 300  # 5 minutes
            if np.any(long_durations):
                cp_estimate = np.mean(powers[long_durations])
            else:
                cp_estimate = np.mean(powers)
            
            # W' estimate: (P - CP) * t for shorter durations
            short_durations = durations <= 300  # 5 minutes
            if np.any(short_durations):
                w_prime_estimate = np.mean((powers[short_durations] - cp_estimate) * durations[short_durations])
            else:
                w_prime_estimate = 20000  # Default estimate
            
            # Ensure parameters are within bounds
            cp_estimate = max(self.cp_bounds[0], min(cp_estimate, self.cp_bounds[1]))
            w_prime_estimate = max(self.w_prime_bounds[0], min(w_prime_estimate, self.w_prime_bounds[1]))
            
            # Fit the model with tightened bounds
            popt, pcov = curve_fit(
                hyperbolic_model, 
                durations, 
                powers, 
                p0=[cp_estimate, w_prime_estimate],
                bounds=(self.cp_bounds, self.w_prime_bounds)
            )
            
            cp_fitted, w_prime_fitted = popt
            
            # Calculate predictions
            y_pred = hyperbolic_model(durations, cp_fitted, w_prime_fitted)
            
            # Calculate R-squared
            ss_res = np.sum((powers - y_pred) ** 2)
            ss_tot = np.sum((powers - np.mean(powers)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate AIC and BIC
            aic, bic = self._calculate_aic_bic(powers, y_pred, n_params=2)
            
            # Calculate TTE
            power_above_cp = cp_fitted + 10
            tte_minutes = (w_prime_fitted / (power_above_cp - cp_fitted)) / 60
            
            # Generate fitted curve for plotting
            curve_durations = np.logspace(np.log10(min(durations)), np.log10(max(durations)), 100)
            fitted_curve = hyperbolic_model(curve_durations, cp_fitted, w_prime_fitted)
            
            results = CPResults(
                cp_watts=cp_fitted,
                w_prime_joules=w_prime_fitted,
                tte_minutes=tte_minutes,
                r_squared=r_squared,
                aic=aic,
                bic=bic,
                model_type="Enhanced Hyperbolic",
                power_duration_data=power_duration_data,
                fitted_curve=fitted_curve,
                curve_durations=curve_durations,
                sampling_rate=self.sampling_rate,
                smoothing_applied=True
            )
            
            print(f"✅ Enhanced Hyperbolic Model Results:")
            print(f"   CP: {cp_fitted:.0f}W")
            print(f"   W': {w_prime_fitted:.0f}J")
            print(f"   TTE: {tte_minutes:.1f} minutes")
            print(f"   R²: {r_squared:.3f}")
            print(f"   AIC: {aic:.1f}")
            print(f"   BIC: {bic:.1f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error fitting hyperbolic model: {str(e)}")
            return None
    
    def fit_linear_work_time_model(self, power_duration_data: Dict[str, float], use_log_transform: bool = True) -> CPResults:
        """
        Enhanced linear work-time model with optional log transformation
        
        Args:
            power_duration_data (Dict[str, float]): Duration -> Power mapping
            use_log_transform (bool): Whether to use log-transformed durations
            
        Returns:
            CPResults: Fitted model results
        """
        print(f"\n📊 Fitting Enhanced Linear Work-Time Model (Work = CP × Time + W')...")
        if use_log_transform:
            print("   🔧 Using log-transformed durations")
        
        # Convert duration names to seconds
        durations_seconds = []
        powers = []
        
        for name, power in power_duration_data.items():
            if name in self.duration_names:
                idx = self.duration_names.index(name)
                duration = self.standard_durations[idx]
                durations_seconds.append(duration)
                powers.append(power)
        
        if len(durations_seconds) < 3:
            print("❌ Insufficient data points for fitting")
            return None
        
        # Convert to numpy arrays
        durations = np.array(durations_seconds)
        powers = np.array(powers)
        
        try:
            # Calculate work done (Power × Time)
            work_done = powers * durations
            
            # Use log-transformed durations if requested
            if use_log_transform:
                x_var = np.log(durations)
                model_name = "Linear Work-Time (Log-Transformed)"
            else:
                x_var = durations
                model_name = "Linear Work-Time"
            
            # Linear regression: Work = CP × Time + W'
            slope, intercept, r_value, p_value, std_err = linregress(x_var, work_done)
            
            cp_linear = slope  # CP is the slope
            w_prime_linear = intercept  # W' is the intercept
            
            # Ensure parameters are within bounds
            cp_linear = max(self.cp_bounds[0], min(cp_linear, self.cp_bounds[1]))
            w_prime_linear = max(self.w_prime_bounds[0], min(w_prime_linear, self.w_prime_bounds[1]))
            
            # Calculate predictions
            y_pred = cp_linear * x_var + w_prime_linear
            work_pred = y_pred / durations  # Convert back to power
            
            # Calculate AIC and BIC
            aic, bic = self._calculate_aic_bic(powers, work_pred, n_params=2)
            
            # Calculate TTE
            power_above_cp = cp_linear + 10
            tte_minutes = (w_prime_linear / (power_above_cp - cp_linear)) / 60
            
            # Generate fitted curve for plotting
            curve_durations = np.logspace(np.log10(min(durations)), np.log10(max(durations)), 100)
            if use_log_transform:
                fitted_curve = (cp_linear * np.log(curve_durations) + w_prime_linear) / curve_durations
            else:
                fitted_curve = (cp_linear * curve_durations + w_prime_linear) / curve_durations
            
            results = CPResults(
                cp_watts=cp_linear,
                w_prime_joules=w_prime_linear,
                tte_minutes=tte_minutes,
                r_squared=r_value ** 2,
                aic=aic,
                bic=bic,
                model_type=model_name,
                power_duration_data=power_duration_data,
                fitted_curve=fitted_curve,
                curve_durations=curve_durations,
                sampling_rate=self.sampling_rate,
                smoothing_applied=True
            )
            
            print(f"✅ Enhanced Linear Work-Time Model Results:")
            print(f"   CP: {cp_linear:.0f}W")
            print(f"   W': {w_prime_linear:.0f}J")
            print(f"   TTE: {tte_minutes:.1f} minutes")
            print(f"   R²: {r_value**2:.3f}")
            print(f"   AIC: {aic:.1f}")
            print(f"   BIC: {bic:.1f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error fitting linear model: {str(e)}")
            return None
    
    def calculate_w_prime_balance(self, cp: float, w_prime: float) -> WPrimeBalance:
        """
        Enhanced W' balance calculation with zone tracking
        
        Args:
            cp (float): Critical Power in watts
            w_prime (float): W' in joules
            
        Returns:
            WPrimeBalance: Enhanced W' balance analysis results
        """
        print(f"\n⚡ Calculating Enhanced W' Balance (CP: {cp:.0f}W, W': {w_prime:.0f}J)...")
        
        if 'power' not in self.data.columns:
            print("❌ No power data available for W' balance calculation")
            return None
        
        power_series = self.data['power'].fillna(0)
        time_points = np.arange(len(power_series))  # Assuming 1Hz data
        
        # Initialize W' balance array
        w_prime_balance = np.zeros(len(power_series))
        w_prime_balance[0] = w_prime  # Start with full W'
        
        # Track depletion and recovery zones
        depletion_zones = []
        recovery_zones = []
        current_zone_start = 0
        in_depletion = False
        
        # Calculate W' balance over time
        for i in range(1, len(power_series)):
            current_power = power_series.iloc[i]
            
            if current_power > cp:
                # Above CP: W' is being expended
                w_prime_expenditure = (current_power - cp) * 1  # 1 second intervals
                w_prime_balance[i] = w_prime_balance[i-1] - w_prime_expenditure
                
                # Track depletion zone
                if not in_depletion:
                    current_zone_start = i
                    in_depletion = True
            else:
                # Below CP: W' is being reconstituted
                tau = 228  # seconds (Skiba et al., 2012)
                w_prime_recovery = (w_prime - w_prime_balance[i-1]) * (1 - np.exp(-1/tau))
                w_prime_balance[i] = w_prime_balance[i-1] + w_prime_recovery
                
                # Track recovery zone
                if in_depletion:
                    depletion_zones.append((current_zone_start / 60, i / 60))  # Convert to minutes
                    in_depletion = False
                    current_zone_start = i
                elif not in_depletion and i > 0:
                    # Continue recovery zone
                    pass
            
            # Ensure W' balance stays within bounds
            w_prime_balance[i] = max(0, min(w_prime, w_prime_balance[i]))
        
        # Handle final zone
        if in_depletion:
            depletion_zones.append((current_zone_start / 60, len(power_series) / 60))
        else:
            recovery_zones.append((current_zone_start / 60, len(power_series) / 60))
        
        # Find exhaustion time (when W' balance reaches 0)
        exhaustion_idx = np.where(w_prime_balance <= 0)[0]
        exhaustion_time = exhaustion_idx[0] / 60 if len(exhaustion_idx) > 0 else None
        
        results = WPrimeBalance(
            w_prime_initial=w_prime,
            w_prime_balance=w_prime_balance,
            time_points=time_points,
            cp_threshold=cp,
            depletion_zones=depletion_zones,
            recovery_zones=recovery_zones,
            exhaustion_time=exhaustion_time
        )
        
        print(f"✅ Enhanced W' Balance Analysis Complete:")
        print(f"   Initial W': {w_prime:.0f}J")
        print(f"   Final W': {w_prime_balance[-1]:.0f}J")
        print(f"   W' Expenditure: {w_prime - w_prime_balance[-1]:.0f}J")
        print(f"   Depletion zones: {len(depletion_zones)}")
        print(f"   Recovery zones: {len(recovery_zones)}")
        if exhaustion_time:
            print(f"   Exhaustion Time: {exhaustion_time:.1f} minutes")
        else:
            print(f"   No exhaustion detected")
        
        return results
    
    def plot_power_duration_curve(self, results: CPResults, save_path: Optional[str] = None):
        """
        Plot power-duration curve with fitted model
        
        Args:
            results (CPResults): Fitted model results
            save_path (Optional[str]): Path to save the plot
        """
        print("\n📊 Creating Power-Duration Curve Plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract data for plotting
        durations_seconds = []
        powers = []
        
        for name, power in results.power_duration_data.items():
            if name in self.duration_names:
                idx = self.duration_names.index(name)
                duration = self.standard_durations[idx]
                durations_seconds.append(duration)
                powers.append(power)
        
        # Plot actual data points
        ax.scatter(durations_seconds, powers, color='blue', s=100, alpha=0.7, 
                  label='Actual Efforts', zorder=5)
        
        # Plot fitted curve
        if results.fitted_curve is not None and results.curve_durations is not None:
            ax.plot(results.curve_durations, results.fitted_curve, 'r-', linewidth=2, 
                   label=f'{results.model_type} Fit', zorder=4)
        
        # Add CP line
        ax.axhline(y=results.cp_watts, color='green', linestyle='--', alpha=0.7, 
                  label=f'CP = {results.cp_watts:.0f}W', zorder=3)
        
        # Customize plot
        ax.set_xscale('log')
        ax.set_xlabel('Duration (seconds)', fontsize=12)
        ax.set_ylabel('Power (watts)', fontsize=12)
        ax.set_title(f'Power-Duration Curve: {results.model_type} Model\n'
                    f'CP = {results.cp_watts:.0f}W, W\' = {results.w_prime_joules:.0f}J, R² = {results.r_squared:.3f}',
                    fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add annotations
        ax.text(0.02, 0.98, f'CP: {results.cp_watts:.0f}W\nW\': {results.w_prime_joules:.0f}J\nTTE: {results.tte_minutes:.1f}min',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_w_prime_balance(self, w_prime_results: WPrimeBalance, save_path: Optional[str] = None):
        """
        Enhanced W' balance plot with shaded zones
        
        Args:
            w_prime_results (WPrimeBalance): W' balance analysis results
            save_path (Optional[str]): Path to save the plot
        """
        print("\n⚡ Creating Enhanced W' Balance Plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Convert time points to minutes
        time_minutes = w_prime_results.time_points / 60
        
        # Plot W' balance with shaded zones
        ax1.plot(time_minutes, w_prime_results.w_prime_balance, 'b-', linewidth=2, label='W\' Balance')
        ax1.axhline(y=w_prime_results.w_prime_initial, color='g', linestyle='--', alpha=0.7, label='Initial W\'')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Exhaustion')
        
        # Add shaded zones for depletion and recovery
        for start, end in w_prime_results.depletion_zones:
            ax1.axvspan(start, end, alpha=0.3, color='red', label='Depletion' if start == w_prime_results.depletion_zones[0][0] else "")
        
        for start, end in w_prime_results.recovery_zones:
            ax1.axvspan(start, end, alpha=0.3, color='green', label='Recovery' if start == w_prime_results.recovery_zones[0][0] else "")
        
        ax1.set_xlabel('Time (minutes)', fontsize=12)
        ax1.set_ylabel('W\' Balance (Joules)', fontsize=12)
        ax1.set_title('Enhanced W\' Balance Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot power with CP threshold
        if 'power' in self.data.columns:
            power_series = self.data['power'].fillna(0)
            ax2.plot(time_minutes, power_series, 'orange', linewidth=1, alpha=0.7, label='Power')
            ax2.axhline(y=w_prime_results.cp_threshold, color='g', linestyle='--', alpha=0.7, label=f'CP = {w_prime_results.cp_threshold:.0f}W')
            
            # Add shaded zones to power plot as well
            for start, end in w_prime_results.depletion_zones:
                ax2.axvspan(start, end, alpha=0.2, color='red')
            
            for start, end in w_prime_results.recovery_zones:
                ax2.axvspan(start, end, alpha=0.2, color='green')
            
            ax2.set_xlabel('Time (minutes)', fontsize=12)
            ax2.set_ylabel('Power (watts)', fontsize=12)
            ax2.set_title('Power Output with CP Threshold', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Enhanced W' balance plot saved to: {save_path}")
        
        plt.show()
    
    def print_power_duration_table(self, power_duration_data: Dict[str, float]):
        """
        Print table of best efforts for standard durations
        
        Args:
            power_duration_data (Dict[str, float]): Duration -> Power mapping
        """
        print("\n📋 Power-Duration Table:")
        print("=" * 50)
        print(f"{'Duration':<10} {'Power (W)':<12} {'% of CP':<10}")
        print("-" * 50)
        
        if hasattr(self, 'results') and self.results:
            cp = self.results.cp_watts
            for name in self.duration_names:
                if name in power_duration_data:
                    power = power_duration_data[name]
                    percent_cp = (power / cp) * 100 if cp > 0 else 0
                    print(f"{name:<10} {power:<12.0f} {percent_cp:<10.1f}%")
        else:
            for name in self.duration_names:
                if name in power_duration_data:
                    power = power_duration_data[name]
                    print(f"{name:<10} {power:<12.0f}")
        
        print("=" * 50)
    
    def run_enhanced_analysis(self, file_path: str, save_plots: bool = True, 
                             export_results: bool = True, add_to_history: bool = True) -> Dict:
        """
        Run enhanced Critical Power analysis with all improvements
        
        Args:
            file_path (str): Path to activity file
            save_plots (bool): Whether to save plots
            export_results (bool): Whether to export results
            add_to_history (bool): Whether to add to session history
            
        Returns:
            Dict: Complete analysis results
        """
        print("🚀 Starting Enhanced Critical Power Analysis")
        print("=" * 60)
        print("🔧 Features: Smoothing, Smart filtering, Enhanced bounds")
        print("🔧 Metrics: R², AIC, BIC for model selection")
        print("🔧 Export: CSV/JSON results and curve data")
        print("=" * 60)
        
        # Load data
        if not self.load_activity_data(file_path):
            return None
        
        # Extract power-duration curve with smoothing
        power_duration_data = self.extract_power_duration_curve()
        if not power_duration_data:
            print("❌ No power-duration data extracted")
            return None
        
        # Fit both models with enhanced features
        hyperbolic_results = self.fit_hyperbolic_model(power_duration_data)
        linear_results = self.fit_linear_work_time_model(power_duration_data, use_log_transform=True)
        
        # Choose best model based on BIC (better than R² for model selection)
        if hyperbolic_results and linear_results:
            if hyperbolic_results.bic < linear_results.bic:
                self.results = hyperbolic_results
                print(f"\n🏆 Selected {hyperbolic_results.model_type} model (BIC = {hyperbolic_results.bic:.1f})")
            else:
                self.results = linear_results
                print(f"\n🏆 Selected {linear_results.model_type} model (BIC = {linear_results.bic:.1f})")
        elif hyperbolic_results:
            self.results = hyperbolic_results
        elif linear_results:
            self.results = linear_results
        else:
            print("❌ No models successfully fitted")
            return None
        
        # Calculate enhanced W' balance
        w_prime_results = self.calculate_w_prime_balance(self.results.cp_watts, self.results.w_prime_joules)
        
        # Print enhanced results table
        self.print_power_duration_table(power_duration_data)
        
        # Create enhanced plots
        if save_plots:
            self.plot_power_duration_curve(self.results, "cp_enhanced_curve.png")
            if w_prime_results:
                self.plot_w_prime_balance(w_prime_results, "cp_enhanced_w_prime_balance.png")
        
        # Export results
        if export_results:
            self.export_results(self.results, w_prime_results, format='json')
        
        # Add to session history
        if add_to_history:
            self.add_session_to_history(self.results)
        
        # Compile results
        analysis_results = {
            'cp_watts': self.results.cp_watts,
            'w_prime_joules': self.results.w_prime_joules,
            'tte_minutes': self.results.tte_minutes,
            'r_squared': self.results.r_squared,
            'aic': self.results.aic,
            'bic': self.results.bic,
            'model_type': self.results.model_type,
            'power_duration_data': power_duration_data,
            'w_prime_balance': w_prime_results,
            'sampling_rate': self.results.sampling_rate,
            'smoothing_applied': self.results.smoothing_applied
        }
        
        print("\n✅ Enhanced Critical Power Analysis Complete!")
        print(f"📊 CP: {self.results.cp_watts:.0f}W")
        print(f"⚡ W': {self.results.w_prime_joules:.0f}J")
        print(f"⏱️ TTE: {self.results.tte_minutes:.1f} minutes")
        print(f"📈 R²: {self.results.r_squared:.3f}")
        print(f"📊 AIC: {self.results.aic:.1f}")
        print(f"📊 BIC: {self.results.bic:.1f}")
        print(f"🔧 Sampling rate: {self.results.sampling_rate:.2f} Hz")
        print(f"🔧 Smoothing applied: {self.results.smoothing_applied}")
        
        return analysis_results

    def _load_fit_file(self, file_path: str) -> pd.DataFrame:
        """Load data from FIT file"""
        try:
            fitfile = fitparse.FitFile(file_path)
            
            data = []
            for record in fitfile.get_messages('record'):
                record_data = {}
                
                if record.get_value('timestamp'):
                    record_data['timestamp'] = record.get_value('timestamp')
                
                if record.get_value('power'):
                    record_data['power'] = record.get_value('power')
                
                if record_data:
                    data.append(record_data)
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading FIT file: {str(e)}")
            return None
    
    def _load_csv_file(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Try to identify timestamp and power columns
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            power_cols = [col for col in df.columns if 'power' in col.lower() or 'watts' in col.lower()]
            
            if timestamp_cols:
                df['timestamp'] = pd.to_datetime(df[timestamp_cols[0]])
            
            if power_cols:
                df['power'] = df[power_cols[0]]
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading CSV file: {str(e)}")
            return None

    def load_multiple_rides_for_cp(self, ride_files: List[str]) -> bool:
        """
        Load multiple rides for proper CP analysis
        
        Args:
            ride_files (List[str]): List of ride file paths
            
        Returns:
            bool: True if successful
        """
        print(f"📁 Loading {len(ride_files)} rides for proper CP analysis...")
        
        all_best_efforts = {}
        
        for i, file_path in enumerate(ride_files):
            print(f"\n📊 Processing ride {i+1}/{len(ride_files)}: {os.path.basename(file_path)}")
            
            # Load individual ride
            if not self.load_activity_data(file_path):
                print(f"   ⚠️ Skipping {file_path}")
                continue
            
            # Extract power-duration curve for this ride
            ride_power_data = self.extract_power_duration_curve()
            
            # Merge with overall best efforts
            for duration, power in ride_power_data.items():
                if duration not in all_best_efforts or power > all_best_efforts[duration]:
                    all_best_efforts[duration] = power
                    print(f"   🏆 New best for {duration}: {power:.0f}W")
        
        if not all_best_efforts:
            print("❌ No valid power data extracted from any rides")
            return False
        
        # Store the combined best efforts
        self.combined_best_efforts = all_best_efforts
        self.rides_analyzed = len([f for f in ride_files if os.path.exists(f)])
        
        print(f"\n✅ Successfully processed {self.rides_analyzed} rides")
        print(f"📊 Combined best efforts: {len(all_best_efforts)} durations")
        
        return True
    
    def fit_cp_from_best_efforts(self) -> CPResults:
        """
        Fit CP model using combined best efforts from multiple rides
        
        Returns:
            CPResults: Fitted model results
        """
        if not hasattr(self, 'combined_best_efforts'):
            print("❌ No combined best efforts available")
            return None
        
        print("\n🔬 Fitting CP model to combined best efforts...")
        
        # Convert duration names to seconds
        durations_seconds = []
        powers = []
        
        for name, power in self.combined_best_efforts.items():
            if name in self.duration_names:
                idx = self.duration_names.index(name)
                duration = self.standard_durations[idx]
                durations_seconds.append(duration)
                powers.append(power)
        
        if len(durations_seconds) < 3:
            print("❌ Insufficient data points for fitting")
            return None
        
        # Convert to numpy arrays
        durations = np.array(durations_seconds)
        powers = np.array(powers)
        
        # Fit both models
        hyperbolic_results = self._fit_hyperbolic_to_best_efforts(durations, powers)
        linear_results = self._fit_linear_work_time_to_best_efforts(durations, powers)
        
        # Choose best model based on BIC
        if hyperbolic_results and linear_results:
            if hyperbolic_results.bic < linear_results.bic:
                return hyperbolic_results
            else:
                return linear_results
        elif hyperbolic_results:
            return hyperbolic_results
        elif linear_results:
            return linear_results
        else:
            return None
    
    def _fit_hyperbolic_to_best_efforts(self, durations: np.ndarray, powers: np.ndarray) -> CPResults:
        """Fit hyperbolic model to best efforts"""
        def hyperbolic_model(t, cp, w_prime):
            return w_prime / t + cp
        
        try:
            # Initial parameter estimates
            long_durations = durations >= 300
            if np.any(long_durations):
                cp_estimate = np.mean(powers[long_durations])
            else:
                cp_estimate = np.mean(powers)
            
            short_durations = durations <= 300
            if np.any(short_durations):
                w_prime_estimate = np.mean((powers[short_durations] - cp_estimate) * durations[short_durations])
            else:
                w_prime_estimate = 20000
            
            # Ensure parameters are within bounds
            cp_estimate = max(self.cp_bounds[0], min(cp_estimate, self.cp_bounds[1]))
            w_prime_estimate = max(self.w_prime_bounds[0], min(w_prime_estimate, self.w_prime_bounds[1]))
            
            # Fit the model
            popt, pcov = curve_fit(
                hyperbolic_model, 
                durations, 
                powers, 
                p0=[cp_estimate, w_prime_estimate],
                bounds=(self.cp_bounds, self.w_prime_bounds)
            )
            
            cp_fitted, w_prime_fitted = popt
            
            # Calculate predictions and metrics
            y_pred = hyperbolic_model(durations, cp_fitted, w_prime_fitted)
            
            ss_res = np.sum((powers - y_pred) ** 2)
            ss_tot = np.sum((powers - np.mean(powers)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            aic, bic = self._calculate_aic_bic(powers, y_pred, n_params=2)
            
            # Calculate TTE
            power_above_cp = cp_fitted + 10
            tte_minutes = (w_prime_fitted / (power_above_cp - cp_fitted)) / 60
            
            # Generate fitted curve
            curve_durations = np.logspace(np.log10(min(durations)), np.log10(max(durations)), 100)
            fitted_curve = hyperbolic_model(curve_durations, cp_fitted, w_prime_fitted)
            
            results = CPResults(
                cp_watts=cp_fitted,
                w_prime_joules=w_prime_fitted,
                tte_minutes=tte_minutes,
                r_squared=r_squared,
                aic=aic,
                bic=bic,
                model_type="Enhanced Hyperbolic (Multi-Ride)",
                power_duration_data=self.combined_best_efforts,
                fitted_curve=fitted_curve,
                curve_durations=curve_durations
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error fitting hyperbolic model: {str(e)}")
            return None
    
    def _fit_linear_work_time_to_best_efforts(self, durations: np.ndarray, powers: np.ndarray) -> CPResults:
        """Fit linear work-time model to best efforts"""
        try:
            # Calculate work done
            work_done = powers * durations
            
            # Use log-transformed durations
            x_var = np.log(durations)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x_var, work_done)
            
            cp_linear = slope
            w_prime_linear = intercept
            
            # Ensure parameters are within bounds
            cp_linear = max(self.cp_bounds[0], min(cp_linear, self.cp_bounds[1]))
            w_prime_linear = max(self.w_prime_bounds[0], min(w_prime_linear, self.w_prime_bounds[1]))
            
            # Calculate predictions and metrics
            y_pred = cp_linear * x_var + w_prime_linear
            work_pred = y_pred / durations
            
            aic, bic = self._calculate_aic_bic(powers, work_pred, n_params=2)
            
            # Calculate TTE
            power_above_cp = cp_linear + 10
            tte_minutes = (w_prime_linear / (power_above_cp - cp_linear)) / 60
            
            # Generate fitted curve
            curve_durations = np.logspace(np.log10(min(durations)), np.log10(max(durations)), 100)
            fitted_curve = (cp_linear * np.log(curve_durations) + w_prime_linear) / curve_durations
            
            results = CPResults(
                cp_watts=cp_linear,
                w_prime_joules=w_prime_linear,
                tte_minutes=tte_minutes,
                r_squared=r_value ** 2,
                aic=aic,
                bic=bic,
                model_type="Linear Work-Time (Multi-Ride)",
                power_duration_data=self.combined_best_efforts,
                fitted_curve=fitted_curve,
                curve_durations=curve_durations
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error fitting linear model: {str(e)}")
            return None
    
    def run_proper_cp_analysis(self, ride_files: List[str], save_plots: bool = True) -> Dict:
        """
        Run proper CP analysis using multiple rides
        
        Args:
            ride_files (List[str]): List of ride file paths
            save_plots (bool): Whether to save plots
            
        Returns:
            Dict: Complete analysis results
        """
        print("🚀 Starting Proper CP Analysis (Multiple Rides)")
        print("=" * 60)
        print("🔬 Using best efforts from multiple rides")
        print("🔬 Scientifically validated methodology")
        print("=" * 60)
        
        # Load multiple rides
        if not self.load_multiple_rides_for_cp(ride_files):
            return None
        
        # Fit CP model to combined best efforts
        self.results = self.fit_cp_from_best_efforts()
        if not self.results:
            print("❌ Failed to fit CP model")
            return None
        
        # Calculate W' balance for the most recent ride
        if ride_files:
            latest_ride = ride_files[-1]
            if self.load_activity_data(latest_ride):
                w_prime_results = self.calculate_w_prime_balance(self.results.cp_watts, self.results.w_prime_joules)
            else:
                w_prime_results = None
        else:
            w_prime_results = None
        
        # Print results
        print(f"\n✅ Proper CP Analysis Complete!")
        print(f"📊 CP: {self.results.cp_watts:.0f}W")
        print(f"⚡ W': {self.results.w_prime_joules:.0f}J")
        print(f"⏱️ TTE: {self.results.tte_minutes:.1f} minutes")
        print(f"📈 R²: {self.results.r_squared:.3f}")
        print(f"📊 AIC: {self.results.aic:.1f}")
        print(f"📊 BIC: {self.results.bic:.1f}")
        print(f"🚴 Rides analyzed: {self.rides_analyzed}")
        
        # Create plots
        if save_plots:
            self.plot_power_duration_curve(self.results, "cp_proper_analysis.png")
            if w_prime_results:
                self.plot_w_prime_balance(w_prime_results, "cp_proper_w_prime.png")
        
        # Export results
        self.export_results(self.results, w_prime_results, format='json', filename="cp_proper_analysis")
        
        # Add to history
        self.add_session_to_history(self.results, notes=f"Multi-ride analysis ({self.rides_analyzed} rides)")
        
        return {
            'cp_watts': self.results.cp_watts,
            'w_prime_joules': self.results.w_prime_joules,
            'tte_minutes': self.results.tte_minutes,
            'r_squared': self.results.r_squared,
            'aic': self.results.aic,
            'bic': self.results.bic,
            'model_type': self.results.model_type,
            'rides_analyzed': self.rides_analyzed,
            'w_prime_balance': w_prime_results
        }

    def export_results(self, results: CPResults, w_prime_results: Optional[WPrimeBalance] = None, 
                      format: str = 'json', filename: Optional[str] = None) -> str:
        """
        Export CP analysis results to CSV or JSON
        
        Args:
            results (CPResults): CP analysis results
            w_prime_results (Optional[WPrimeBalance]): W' balance results
            format (str): Export format ('json' or 'csv')
            filename (Optional[str]): Custom filename
            
        Returns:
            str: Path to exported file
        """
        print(f"\n📤 Exporting results to {format.upper()}...")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cp_analysis_{timestamp}"
        
        # Prepare data for export
        export_data = {
            'analysis_info': {
                'date': datetime.now().isoformat(),
                'model_type': results.model_type,
                'sampling_rate': results.sampling_rate,
                'smoothing_applied': results.smoothing_applied,
                'rides_analyzed': getattr(self, 'rides_analyzed', 1)
            },
            'cp_results': {
                'cp_watts': results.cp_watts,
                'w_prime_joules': results.w_prime_joules,
                'tte_minutes': results.tte_minutes,
                'r_squared': results.r_squared,
                'aic': results.aic,
                'bic': results.bic
            },
            'power_duration_data': results.power_duration_data,
            'curve_data': {
                'durations': results.curve_durations.tolist() if results.curve_durations is not None else [],
                'powers': results.fitted_curve.tolist() if results.fitted_curve is not None else []
            }
        }
        
        # Add W' balance data if available
        if w_prime_results:
            export_data['w_prime_balance'] = {
                'initial_w_prime': w_prime_results.w_prime_initial,
                'final_w_prime': w_prime_results.w_prime_balance[-1],
                'exhaustion_time': w_prime_results.exhaustion_time,
                'depletion_zones': w_prime_results.depletion_zones,
                'recovery_zones': w_prime_results.recovery_zones
            }
        
        if format.lower() == 'json':
            filepath = f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == 'csv':
            # Create CSV with multiple sheets
            filepath = f"{filename}.csv"
            
            # Main results
            main_results = pd.DataFrame([{
                'CP_Watts': results.cp_watts,
                'W_Prime_Joules': results.w_prime_joules,
                'TTE_Minutes': results.tte_minutes,
                'R_Squared': results.r_squared,
                'AIC': results.aic,
                'BIC': results.bic,
                'Model_Type': results.model_type
            }])
            main_results.to_csv(filepath, index=False)
            
            # Power-duration data
            pd_data = pd.DataFrame(list(results.power_duration_data.items()), 
                                 columns=['Duration', 'Power_Watts'])
            pd_data.to_csv(f"{filename}_power_duration.csv", index=False)
            
            # Curve data
            if results.curve_durations is not None and results.fitted_curve is not None:
                curve_data = pd.DataFrame({
                    'Duration_Seconds': results.curve_durations,
                    'Power_Watts': results.fitted_curve
                })
                curve_data.to_csv(f"{filename}_curve.csv", index=False)
        
        print(f"✅ Results exported to: {filepath}")
        return filepath
    
    def add_session_to_history(self, results: CPResults, notes: Optional[str] = None):
        """
        Add current session to CP history for tracking over time
        
        Args:
            results (CPResults): Current session results
            notes (Optional[str]): Session notes
        """
        session = CPSession(
            date=datetime.now(),
            cp_watts=results.cp_watts,
            w_prime_joules=results.w_prime_joules,
            tte_minutes=results.tte_minutes,
            r_squared=results.r_squared,
            model_type=results.model_type,
            rides_analyzed=getattr(self, 'rides_analyzed', 1),
            notes=notes
        )
        
        self.sessions_history.append(session)
        print(f"📊 Added session to history (CP: {results.cp_watts:.0f}W)")
    
    def plot_cp_history(self, save_path: Optional[str] = None):
        """
        Plot CP and W' over time across sessions
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        if not self.sessions_history:
            print("❌ No session history available")
            return
        
        print("\n📈 Creating CP History Plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        dates = [session.date for session in self.sessions_history]
        cps = [session.cp_watts for session in self.sessions_history]
        w_primes = [session.w_prime_joules for session in self.sessions_history]
        
        # Plot CP over time
        ax1.plot(dates, cps, 'bo-', linewidth=2, markersize=8)
        ax1.set_ylabel('Critical Power (Watts)', fontsize=12)
        ax1.set_title('CP Evolution Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add FTP reference line
        ax1.axhline(y=self.ftp, color='r', linestyle='--', alpha=0.7, label=f'FTP: {self.ftp}W')
        ax1.legend()
        
        # Plot W' over time
        ax2.plot(dates, w_primes, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('W\' (Joules)', fontsize=12)
        ax2.set_title('W\' Evolution Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 CP history plot saved to: {save_path}")
        
        plt.show()


def main():
    """
    Enhanced Critical Power Analysis Demo
    
    Demonstrates all the new features:
    1. Single ride analysis with smoothing and enhanced metrics
    2. Multi-ride analysis for proper CP calculation
    3. Session history tracking
    4. Export capabilities
    """
    print("🔬 Enhanced Critical Power Analysis Demo")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CriticalPowerAnalyzer(ftp=250)
    
    # Example files (replace with actual paths)
    single_file = "cache/Into_the_clouds.fit"
    multiple_files = [
        "cache/Into_the_clouds.fit",
        # Add more files here for multi-ride analysis
    ]
    
    print("\n1️⃣ Single Ride Enhanced Analysis")
    print("-" * 40)
    
    if os.path.exists(single_file):
        try:
            results = analyzer.run_enhanced_analysis(single_file, save_plots=True, export_results=True, add_to_history=True)
            
            if results:
                print(f"\n✅ Single ride analysis complete!")
                print(f"📊 CP: {results['cp_watts']:.0f}W")
                print(f"⚡ W': {results['w_prime_joules']:.0f}J")
                print(f"📈 R²: {results['r_squared']:.3f}")
                print(f"📊 BIC: {results['bic']:.1f}")
                print(f"🔧 Sampling rate: {results['sampling_rate']:.2f} Hz")
                print(f"🔧 Smoothing applied: {results['smoothing_applied']}")
        except Exception as e:
            print(f"❌ Error in single ride analysis: {str(e)}")
    else:
        print(f"⚠️ File not found: {single_file}")
    
    print("\n2️⃣ Multi-Ride Proper CP Analysis")
    print("-" * 40)
    
    # Check if we have multiple files for proper CP analysis
    available_files = [f for f in multiple_files if os.path.exists(f)]
    
    if len(available_files) >= 2:
        try:
            multi_results = analyzer.run_proper_cp_analysis(available_files, save_plots=True)
            
            if multi_results:
                print(f"\n✅ Multi-ride analysis complete!")
                print(f"📊 CP: {multi_results['cp_watts']:.0f}W")
                print(f"⚡ W': {multi_results['w_prime_joules']:.0f}J")
                print(f"📈 R²: {multi_results['r_squared']:.3f}")
                print(f"📊 BIC: {multi_results['bic']:.1f}")
                print(f"🚴 Rides analyzed: {multi_results['rides_analyzed']}")
        except Exception as e:
            print(f"❌ Error in multi-ride analysis: {str(e)}")
    else:
        print(f"⚠️ Need at least 2 files for proper CP analysis. Found: {len(available_files)}")
    
    print("\n3️⃣ Session History and Tracking")
    print("-" * 40)
    
    if analyzer.sessions_history:
        print(f"📊 Session history: {len(analyzer.sessions_history)} sessions")
        
        # Plot CP history if we have multiple sessions
        if len(analyzer.sessions_history) > 1:
            try:
                analyzer.plot_cp_history("cp_history_evolution.png")
                print("📈 CP history plot created")
            except Exception as e:
                print(f"❌ Error creating history plot: {str(e)}")
        
        # Show session summary
        print("\n📋 Session Summary:")
        for i, session in enumerate(analyzer.sessions_history):
            print(f"   Session {i+1}: CP={session.cp_watts:.0f}W, W'={session.w_prime_joules:.0f}J, R²={session.r_squared:.3f}")
    else:
        print("📊 No session history available")
    
    print("\n4️⃣ Enhanced Features Summary")
    print("-" * 40)
    print("✅ Light smoothing (3s rolling median) applied")
    print("✅ Smart duration filtering based on sampling rate")
    print("✅ Tightened parameter bounds for realistic ranges")
    print("✅ Log-transformed durations for linear regression")
    print("✅ AIC/BIC metrics for model selection")
    print("✅ Enhanced W' balance plots with shaded zones")
    print("✅ Export results to CSV/JSON")
    print("✅ Session history tracking")
    
    print("\n🎯 Enhanced Critical Power Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main() 