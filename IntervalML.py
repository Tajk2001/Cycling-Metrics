#!/usr/bin/env python3
"""
IntervalML.py - Machine Learning Approach to Cycling Interval Detection

This script trains a model to automatically detect intervals in cycling power data
using supervised learning on labeled interval data.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as ndimage
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import SprintV1 functions for FIT file loading
try:
    from SprintV1 import load_fit_to_dataframe
except ImportError:
    # Fallback if SprintV1 is not available
    def load_fit_to_dataframe(path):
        print("❌ SprintV1 not available. Please ensure SprintV1.py is in the same directory.")
        return None

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IntervalDetector:
    """Machine learning-based interval detector for cycling power data."""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the interval detector.
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def create_features(self, df, window_sizes=[5, 10, 15, 30, 60, 120, 300], estimated_ftp=None):
        """
        Create features for interval detection from power, cadence, and torque data.
        
        Args:
            df (pd.DataFrame): DataFrame with 'power', 'cadence', 'torque' columns and datetime index
            window_sizes (list): List of window sizes for rolling features
            estimated_ftp (float, optional): Pre-calculated FTP estimate to use instead of estimating
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        print("🔧 Creating features for interval detection...")
        
        # Ensure we have a copy
        df_copy = df.copy()
        
        # Use provided FTP estimate or estimate from best efforts
        if estimated_ftp is None:
            estimated_ftp = self._estimate_ftp_from_best_efforts(df_copy)
        
        # Normalize power relative to estimated FTP (much more meaningful)
        ride_avg_power = df_copy['power'].mean()
        ride_max_power = df_copy['power'].max()
        ride_std_power = df_copy['power'].std()
        
        # Create normalized power features relative to estimated FTP
        # Zone 4 = 91-105% FTP, Zone 5 = 106-120% FTP, Zone 6 = 121-150% FTP, Zone 7 = 151%+ FTP
        df_copy['power_normalized'] = df_copy['power'] / (estimated_ftp + 1)  # 1.0 = 100% FTP
        df_copy['power_relative_to_avg'] = df_copy['power'] / (ride_avg_power + 1)
        df_copy['power_relative_to_ftp'] = df_copy['power'] / (estimated_ftp + 1)
        df_copy['power_z_score'] = (df_copy['power'] - ride_avg_power) / (ride_std_power + 1)
        
        # Add power zone classification based on estimated FTP
        df_copy['power_zone'] = pd.cut(df_copy['power_relative_to_ftp'], 
                                      bins=[0, 0.55, 0.75, 0.90, 1.05, 1.20, 1.50, 2.0, 10.0],
                                      labels=['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5', 'Zone 6', 'Zone 7', 'Zone 7+'])
        
        # Basic power features (using normalized power)
        df_copy['power_smooth'] = df_copy['power_normalized'].rolling(window=5, min_periods=1).mean()
        df_copy['power_std'] = df_copy['power_normalized'].rolling(window=5, min_periods=1).std()
        df_copy['power_change'] = df_copy['power_normalized'].diff()
        df_copy['power_accel'] = df_copy['power_change'].diff()
        
        # Rolling statistics for different window sizes (using normalized power)
        for window in window_sizes:
            # Rolling averages (normalized)
            df_copy[f'power_avg_{window}s'] = df_copy['power_normalized'].rolling(window=window, min_periods=1).mean()
            df_copy[f'power_std_{window}s'] = df_copy['power_normalized'].rolling(window=window, min_periods=1).std()
            df_copy[f'power_max_{window}s'] = df_copy['power_normalized'].rolling(window=window, min_periods=1).max()
            df_copy[f'power_min_{window}s'] = df_copy['power_normalized'].rolling(window=window, min_periods=1).min()
            
            # Rolling changes (normalized)
            df_copy[f'power_change_{window}s'] = df_copy['power_normalized'].rolling(window=window, min_periods=1).apply(
                lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
            )
            
            # Coefficient of variation (normalized)
            df_copy[f'power_cv_{window}s'] = df_copy[f'power_std_{window}s'] / df_copy[f'power_avg_{window}s']
            df_copy[f'power_cv_{window}s'] = df_copy[f'power_cv_{window}s'].fillna(0)
        
        # Power zones (relative to ride characteristics)
        # Use ride's own power distribution instead of fixed FTP
        df_copy['power_percentile_rank'] = df_copy['power'].rank(pct=True)
        df_copy['power_zone'] = pd.cut(df_copy['power_percentile_rank'], 
                                      bins=[0, 0.30, 0.50, 0.70, 0.85, 0.95, 0.99, 1.0],
                                      labels=['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5', 'Zone 6', 'Zone 7'])
        
        # Time-based features
        df_copy['time_of_day'] = df_copy.index.hour + df_copy.index.minute / 60
        df_copy['elapsed_time'] = (df_copy.index - df_copy.index[0]).total_seconds()
        
        # Normalize cadence relative to this ride
        ride_avg_cadence = df_copy['cadence'].mean()
        ride_max_cadence = df_copy['cadence'].max()
        
        # Use FTP-based scaling for cadence too (cadence often correlates with effort)
        cadence_reference = max(ride_avg_cadence * 1.2, ride_max_cadence * 0.8)
        
        # Cadence features (normalized)
        df_copy['cadence_normalized'] = df_copy['cadence'] / (cadence_reference + 1)
        df_copy['cadence_relative_to_avg'] = df_copy['cadence'] / (ride_avg_cadence + 1)
        df_copy['cadence_smooth'] = df_copy['cadence_normalized'].rolling(window=5, min_periods=1).mean()
        df_copy['cadence_change'] = df_copy['cadence_normalized'].diff()
        df_copy['cadence_power_ratio'] = df_copy['power_normalized'] / (df_copy['cadence_normalized'] + 0.01)
        
        # Rolling cadence statistics (normalized)
        for window in window_sizes:
            df_copy[f'cadence_avg_{window}s'] = df_copy['cadence_normalized'].rolling(window=window, min_periods=1).mean()
            df_copy[f'cadence_std_{window}s'] = df_copy['cadence_normalized'].rolling(window=window, min_periods=1).std()
            df_copy[f'cadence_change_{window}s'] = df_copy['cadence_normalized'].rolling(window=window, min_periods=1).apply(
                lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
            )
        
        # Combined features (normalized)
        df_copy['power_cadence_product'] = df_copy['power_normalized'] * df_copy['cadence_normalized']
        
        # Handle different data types when filling NaN values
        for col in df_copy.columns:
            if df_copy[col].isna().any():
                if df_copy[col].dtype.name == 'category':
                    # For categorical columns, fill with the most frequent value
                    mode_val = df_copy[col].mode()
                    if len(mode_val) > 0:
                        df_copy[col] = df_copy[col].fillna(mode_val[0])
                    else:
                        # Convert to object type and fill with 'unknown'
                        df_copy[col] = df_copy[col].astype('object').fillna('unknown')
                elif pd.api.types.is_numeric_dtype(df_copy[col]):
                    # For numeric columns, fill with 0
                    df_copy[col] = df_copy[col].fillna(0)
                else:
                    # For other types, fill with 'unknown'
                    df_copy[col] = df_copy[col].fillna('unknown')
        
        print(f"✅ Created {len(df_copy.columns)} features")
        print(f"📊 Data points: {len(df)} → {len(df_copy)} ({len(df_copy)/len(df)*100:.1f}% retained)")
        
        return df_copy
    
    def _estimate_ftp_from_best_efforts(self, df):
        """
        Estimate FTP by working backwards from actual effort levels.
        This gives us a much more accurate reference point than arbitrary scaling factors.
        """
        print(f"   🎯 Analyzing effort levels to estimate FTP...")
        
        # Analyze the rider's actual power output patterns
        # Work backwards from their real performance to estimate FTP
        
        # Get power percentiles to understand the rider's capability
        power_95th = df['power'].quantile(0.95)
        power_90th = df['power'].quantile(0.90)
        power_75th = df['power'].quantile(0.75)
        power_50th = df['power'].quantile(0.50)
        max_power = df['power'].max()
        avg_power = df['power'].mean()
        
        print(f"      📊 Power analysis:")
        print(f"         Max: {max_power:.0f}W, 95th: {power_95th:.0f}W, 90th: {power_90th:.0f}W")
        print(f"         75th: {power_75th:.0f}W, 50th: {power_50th:.0f}W, Avg: {avg_power:.0f}W")
        
        # Find sustained efforts at different durations
        effort_analysis = []
        
        # Test different duration windows to find sustained power
        duration_tests = [30, 60, 120, 300, 600, 1200]  # 30s, 1min, 2min, 5min, 10min, 20min
        
        for duration in duration_tests:
            if duration <= len(df):
                # Calculate rolling average power for this duration
                rolling_power = df['power'].rolling(window=duration, min_periods=duration).mean()
                max_sustained_power = rolling_power.max()
                
                if not pd.isna(max_sustained_power) and max_sustained_power > 0:
                    # For FTP estimation, we want efforts that are sustainable
                    # Shorter efforts (30s-2min) are typically 120-150% FTP
                    # Longer efforts (5-20min) are typically 95-110% FTP
                    if duration <= 120:  # Short efforts
                        # These are likely 120-150% of FTP, so FTP ≈ power / 1.35
                        estimated_ftp = max_sustained_power / 1.35
                        effort_type = "short burst"
                    elif duration <= 300:  # Medium efforts (2-5min)
                        # These are likely 110-130% of FTP, so FTP ≈ power / 1.2
                        estimated_ftp = max_sustained_power / 1.2
                        effort_type = "medium effort"
                    else:  # Long efforts (5-20min)
                        # These are likely 95-110% of FTP, so FTP ≈ power / 1.02
                        estimated_ftp = max_sustained_power / 1.02
                        effort_type = "sustained effort"
                    
                    effort_analysis.append({
                        'duration': f"{duration}s",
                        'max_sustained_power': max_sustained_power,
                        'effort_type': effort_type,
                        'estimated_ftp': estimated_ftp,
                        'reliability': duration  # Longer efforts are more reliable
                    })
        
        if not effort_analysis:
            # Fallback: estimate from power distribution
            print(f"      📊 No sustained efforts found, using power distribution...")
            # Use 75th percentile as a reasonable FTP estimate
            estimated_ftp = power_75th
            print(f"         Using 75th percentile: {estimated_ftp:.0f}W")
        else:
            # Sort by reliability (duration) and use the most reliable estimates
            effort_analysis.sort(key=lambda x: x['reliability'], reverse=True)
            
            # Focus on medium to long efforts (more reliable for FTP)
            reliable_efforts = [e for e in effort_analysis if e['reliability'] >= 120]
            
            if reliable_efforts:
                # Use the most reliable effort
                best_effort = reliable_efforts[0]
                estimated_ftp = best_effort['estimated_ftp']
                print(f"      📊 Best effort: {best_effort['duration']} {best_effort['effort_type']} at {best_effort['max_sustained_power']:.0f}W")
                print(f"         Estimated FTP: {estimated_ftp:.0f}W")
            else:
                # Use the longest effort available
                best_effort = effort_analysis[0]
                estimated_ftp = best_effort['estimated_ftp']
                print(f"      📊 Best effort: {best_effort['duration']} {best_effort['effort_type']} at {best_effort['max_sustained_power']:.0f}W")
                print(f"         Estimated FTP: {estimated_ftp:.0f}W")
        
        # Apply realistic constraints based on the rider's actual performance
        print(f"      🎯 Raw FTP estimate: {estimated_ftp:.0f}W")
        
        # Constrain FTP based on the rider's actual power output
        # FTP should be between 70% of their average power and 85% of their max power
        min_ftp = avg_power * 0.7
        max_ftp = max_power * 0.85
        
        # Use more realistic constraints based on actual performance
        estimated_ftp = max(min_ftp, min(estimated_ftp, max_ftp))
        
        print(f"         📊 Constrained range: {min_ftp:.0f}W - {max_ftp:.0f}W")
        print(f"      ✅ Final FTP estimate: {estimated_ftp:.0f}W")
        
        return estimated_ftp
    
    def create_training_data(self, df, interval_labels):
        """
        Create training data with features and labels.
        
        Args:
            df (pd.DataFrame): DataFrame with engineered features
            interval_labels (list): List of (start_time, end_time) tuples for intervals
            
        Returns:
            tuple: (X_features, y_labels)
        """
        print("🏷️ Creating training labels...")
        
        # Create binary labels: 1 for interval, 0 for non-interval
        y_labels = np.zeros(len(df))
        
        successful_intervals = 0
        for start_time, end_time in interval_labels:
            try:
                # Find indices for this interval using nearest timestamp matching
                start_idx = self._find_nearest_timestamp(df.index, start_time)
                end_idx = self._find_nearest_timestamp(df.index, end_time)
                
                if start_idx is not None and end_idx is not None:
                    # Label all points in the interval as 1
                    y_labels[start_idx:end_idx+1] = 1
                    successful_intervals += 1
                    print(f"   ✅ Labeled interval: {start_time} to {end_time}")
                else:
                    print(f"⚠️ Warning: Could not find interval {start_time} to {end_time} in data. Skipping.")
            except Exception as e:
                print(f"⚠️ Warning: Error processing interval {start_time} to {end_time}: {e}")
                continue
        
        print(f"   📊 Successfully labeled {successful_intervals}/{len(interval_labels)} intervals")
        
        # Create engineered features with FTP-based normalization
        print("🔧 Creating features for interval detection...")
        df_with_features = self.create_features(df)
        
        if df_with_features is None:
            print("❌ Failed to create features")
            return None, None
        
        # Get feature columns (exclude non-feature columns)
        exclude_cols = ['power_zone', 'file_path', 'file_name']  # Categorical columns to exclude
        feature_cols = [col for col in df_with_features.columns if col not in exclude_cols and not col.startswith('elapsed_time')]
        
        X_features = df_with_features[feature_cols].values
        self.feature_names = feature_cols
        
        print(f"✅ Created training data: {X_features.shape[0]} samples, {X_features.shape[1]} features")
        print(f"📊 Class distribution: {np.bincount(y_labels.astype(int))}")
        
        return X_features, y_labels
    
    def _find_nearest_timestamp(self, index, target_time, tolerance_seconds=5):
        """
        Find the nearest timestamp within a tolerance.
        
        Args:
            index (pd.DatetimeIndex): Index to search in
            target_time (pd.Timestamp): Target timestamp
            tolerance_seconds (int): Tolerance in seconds
            
        Returns:
            int or None: Index of nearest timestamp, or None if not found
        """
        # Convert to datetime if it's a string
        if isinstance(target_time, str):
            target_time = pd.to_datetime(target_time)
        
        # Find the closest timestamp
        time_diff = abs(index - target_time)
        min_diff_idx = time_diff.argmin()
        min_diff = time_diff[min_diff_idx]
        
        # Check if within tolerance
        if min_diff.total_seconds() <= tolerance_seconds:
            return min_diff_idx
        else:
            print(f"   ⚠️ Closest timestamp {index[min_diff_idx]} is {min_diff.total_seconds():.1f}s away from target {target_time}")
            return None
    
    def train_model(self, X_train, y_train, use_cross_validation=True, enhance_labels=True):
        """
        Train the interval detection model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            use_cross_validation (bool): Whether to use cross-validation for hyperparameter tuning
            enhance_labels (bool): Whether to apply smoothing and amplification to labels
        """
        print(f"🤖 Training {self.model_type} model...")
        
        # Apply label enhancement if requested
        if enhance_labels:
            print("🔧 Applying label smoothing and amplification...")
            original_interval_count = np.sum(y_train)
            y_train_enhanced = self.smooth_and_amplify_interval_labels(y_train)
            enhanced_interval_count = np.sum(y_train_enhanced)
            
            print(f"   📊 Original interval points: {original_interval_count}")
            print(f"   📊 Enhanced interval points: {enhanced_interval_count}")
            print(f"   📊 Enhancement factor: {enhanced_interval_count/max(original_interval_count, 1):.2f}x")
            
            y_train = y_train_enhanced
        
        # Check if we have enough data for cross-validation
        n_samples = len(X_train)
        if use_cross_validation and n_samples < 10:
            print(f"⚠️ Warning: Only {n_samples} training samples. Disabling cross-validation.")
            use_cross_validation = False
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.model_type == 'random_forest':
            if use_cross_validation:
                # Adjust cross-validation based on sample size
                cv_folds = min(5, max(2, n_samples // 2))
                print(f"📊 Using {cv_folds}-fold cross-validation")
                
                # Grid search for hyperparameter tuning
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                
                rf = RandomForestClassifier(random_state=42, class_weight='balanced')
                grid_search = GridSearchCV(rf, param_grid, cv=cv_folds, scoring='f1', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                
                self.model = grid_search.best_estimator_
                print(f"✅ Best parameters: {grid_search.best_params_}")
            else:
                self.model = RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=20, 
                    random_state=42,
                    class_weight='balanced'
                )
                self.model.fit(X_train_scaled, y_train)
                
        elif self.model_type == 'gradient_boosting':
            if use_cross_validation:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
                
                gb = GradientBoostingClassifier(random_state=42)
                grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='f1', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                
                self.model = grid_search.best_estimator_
                print(f"✅ Best parameters: {grid_search.best_params_}")
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                self.model.fit(X_train_scaled, y_train)
                
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
            self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        print(f"✅ Model trained successfully!")
        

    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
        """
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return
        
        print("📊 Evaluating model performance...")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        # Handle case when model has only one class (predict_proba returns one column)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        if y_pred_proba.shape[1] > 1:
            y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = y_pred_proba[:, 0]
        
        # Print classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return y_pred
    

    
    def predict_intervals(self, df):
        """
        Predict intervals in new data.
        
        Args:
            df (pd.DataFrame): DataFrame with power data
            
        Returns:
            list: List of (start_time, end_time) tuples for predicted intervals
        """
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return []
        
        print("🔮 Predicting intervals in new data...")
        
        # Create features
        df_features = self.create_features(df)
        
        # Scale features
        X_new = df_features[self.feature_names].values
        X_new_scaled = self.scaler.transform(X_new)
        
        # Get predictions
        y_pred = self.model.predict(X_new_scaled)
        
        # Handle case when model has only one class
        y_pred_proba_raw = self.model.predict_proba(X_new_scaled)
        if y_pred_proba_raw.shape[1] > 1:
            y_pred_proba = y_pred_proba_raw[:, 1]
        else:
            y_pred_proba = y_pred_proba_raw[:, 0]
        
        # Find continuous intervals
        intervals = self._find_continuous_intervals(y_pred, df_features.index)
        
        print(f"✅ Predicted {len(intervals)} intervals")
        
        return intervals
    
    def _find_continuous_intervals(self, predictions, timestamps):
        """
        Find continuous intervals from binary predictions.
        
        Args:
            predictions (np.array): Binary predictions (0 or 1)
            timestamps (pd.DatetimeIndex): Timestamps for the predictions
            
        Returns:
            list: List of (start_time, end_time) tuples
        """
        intervals = []
        in_interval = False
        start_idx = None
        
        for i, pred in enumerate(predictions):
            if pred == 1 and not in_interval:
                # Start of interval
                start_idx = i
                in_interval = True
            elif pred == 0 and in_interval:
                # End of interval
                end_idx = i - 1
                if end_idx >= start_idx:  # Ensure valid interval
                    start_time = timestamps[start_idx]
                    end_time = timestamps[end_idx]
                    duration = (end_time - start_time).total_seconds()
                    
                    # Only keep intervals longer than 10 seconds
                    if duration >= 10:
                        intervals.append((start_time, end_time))
                
                in_interval = False
                start_idx = None
        
        # Handle case where interval extends to end of data
        if in_interval and start_idx is not None:
            end_idx = len(predictions) - 1
            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx]
            duration = (end_time - start_time).total_seconds()
            
            if duration >= 10:
                intervals.append((start_time, end_time))
        
        return intervals
    
    def detect_intervals_sliding_windows(self, df, ftp=300):
        """
        Detect intervals using sliding windows and power thresholds.
        
        Args:
            df (pd.DataFrame): DataFrame with power data
            ftp (float): Functional Threshold Power in watts
            
        Returns:
            pd.DataFrame: DataFrame with detected intervals
        """
        print(f"🔍 Detecting intervals using sliding windows (FTP = {ftp}W)...")
        
        # Window sizes to test (in seconds)
        window_sizes = [10, 15, 20, 30, 40, 45, 60, 120, 180, 240, 300, 360, 480, 600, 720, 1200, 1800]
        
        intervals_data = []
        
        for window_size in window_sizes:
            print(f"   🔍 Testing {window_size}s window...")
            
            # Calculate rolling statistics
            power_avg = df['power'].rolling(window=window_size, min_periods=window_size).mean()
            power_std = df['power'].rolling(window=window_size, min_periods=window_size).std()
            power_cv = power_std / power_avg
            power_cv = power_cv.fillna(0)
            
            # Find intervals based on criteria
            # Power >= 0.8 * FTP AND CV <= 0.3
            interval_mask = (power_avg >= 0.8 * ftp) & (power_cv <= 0.3)
            
            # Find continuous segments
            in_interval = False
            start_idx = None
            
            for i, is_interval in enumerate(interval_mask):
                if is_interval and not in_interval:
                    start_idx = i
                    in_interval = True
                elif not is_interval and in_interval:
                    end_idx = i - 1
                    if end_idx >= start_idx:
                        start_time = df.index[start_idx]
                        end_time = df.index[end_idx]
                        duration = (end_time - start_time).total_seconds() / 60  # minutes
                        
                        # Only keep intervals longer than 30 seconds
                        if duration >= 0.5:
                            avg_power = power_avg.iloc[start_idx:end_idx+1].mean()
                            max_power = df['power'].iloc[start_idx:end_idx+1].max()
                            power_cv_val = power_cv.iloc[start_idx:end_idx+1].mean()
                            
                            intervals_data.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': duration,
                                'avg_power': avg_power,
                                'max_power': max_power,
                                'power_cv': power_cv_val,
                                'window_size': window_size
                            })
                    
                    in_interval = False
                    start_idx = None
            
            # Handle case where interval extends to end
            if in_interval and start_idx is not None:
                end_idx = len(df) - 1
                start_time = df.index[start_idx]
                end_time = df.index[end_idx]
                duration = (end_time - start_time).total_seconds() / 60
                
                if duration >= 0.5:
                    avg_power = power_avg.iloc[start_idx:end_idx+1].mean()
                    max_power = df['power'].iloc[start_idx:end_idx+1].max()
                    power_cv_val = power_cv.iloc[start_idx:end_idx+1].mean()
                    
                    intervals_data.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'avg_power': avg_power,
                        'max_power': max_power,
                        'power_cv': power_cv_val,
                        'window_size': window_size
                    })
        
        if not intervals_data:
            print("   ❌ No intervals detected")
            return None
        
        # Create DataFrame and remove duplicates
        intervals_df = pd.DataFrame(intervals_data)
        intervals_df = intervals_df.drop_duplicates(subset=['start_time', 'end_time'])
        intervals_df = intervals_df.sort_values('start_time')
        
        print(f"   ✅ Detected {len(intervals_df)} unique intervals")
        return intervals_df
    
    def smooth_and_amplify_interval_labels(self, interval_labels, window_size=3, amplification_factor=1.5):
        """
        Smooth and amplify interval labels during training for better learning.
        
        Args:
            interval_labels: Binary interval labels (0/1)
            window_size: Size of smoothing window
            amplification_factor: Factor to amplify interval regions
        
        Returns:
            Enhanced interval labels
        """
        # Convert to float for processing
        labels_float = interval_labels.astype(float)
        
        # Apply light smoothing to interval labels
        smoothed_labels = ndimage.gaussian_filter1d(labels_float, sigma=window_size/3)
        
        # Amplify interval regions
        amplified_labels = smoothed_labels * amplification_factor
        
        # Apply threshold to create enhanced binary labels
        # Use lower threshold since we amplified the signal
        enhanced_threshold = 0.3
        enhanced_labels = (amplified_labels > enhanced_threshold).astype(int)
        
        return enhanced_labels

    def save_model(self, filepath):
        """Save the trained model and scaler."""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and scaler."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            
            # Handle models that may not have model_type saved
            self.model_type = model_data.get('model_type', 'random_forest')
            self.is_trained = True
            
            print(f"📂 Model loaded from {filepath}")
            print(f"🤖 Model type: {self.model_type}")
            print(f"📊 Features: {len(self.feature_names)}")
            
            return self.model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None


def create_30_30_workout():
    """Create a 30/30 workout (30s on, 30s off) with active recovery."""
    np.random.seed(42)
    n_samples = 3600  # 1 hour
    
    # Base endurance power
    base_power = 160 + np.random.normal(0, 12, n_samples)
    
    # Warm up (10 min)
    warmup_end = 600
    
    # 30/30 intervals (20 sets = 20 minutes total)
    intervals = []
    current_time = warmup_end
    
    for i in range(20):
        # 30s high intensity
        interval_start = current_time
        interval_end = current_time + 30
        
        # High power with realistic fluctuations
        high_power = 320 + np.random.normal(0, 25, interval_end - interval_start)
        # Add power fluctuations within the interval
        for j in range(interval_end - interval_start):
            high_power[j] += 20 * np.sin(j * np.pi / 15) + np.random.normal(0, 15)
        
        base_power[interval_start:interval_end] = high_power
        intervals.append((interval_start, interval_end))
        
        # 30s active recovery (not complete rest)
        recovery_start = interval_end
        recovery_end = recovery_start + 30
        recovery_power = 180 + np.random.normal(0, 15, recovery_end - recovery_start)
        base_power[recovery_start:recovery_end] = recovery_power
        
        current_time = recovery_end
    
    # Cool down
    cooldown_start = current_time
    cooldown_power = 140 + np.random.normal(0, 8, n_samples - cooldown_start)
    base_power[cooldown_start:] = cooldown_power
    
    # Create realistic cadence patterns
    cadence = np.full(n_samples, 80)
    
    # Adjust cadence for intervals and recovery
    for start, end in intervals:
        # High cadence during intervals with variation
        interval_cadence = 95 + np.random.normal(0, 5, end - start)
        # Add cadence fluctuations
        for j in range(end - start):
            interval_cadence[j] += 8 * np.sin(j * np.pi / 10) + np.random.normal(0, 3)
        cadence[start:end] = interval_cadence
        
        # Lower cadence during active recovery
        recovery_start = end
        recovery_end = min(end + 30, n_samples)
        cadence[recovery_start:recovery_end] = 70 + np.random.normal(0, 4, recovery_end - recovery_start)
    
    # Calculate torque (Nm) from power and cadence
    # Torque = Power / (2π × Cadence/60) = Power / (Cadence × π/30)
    torque = base_power / (cadence * np.pi / 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'power': base_power,
        'cadence': cadence,
        'torque': torque
    })
    
    # Create timestamps for the DataFrame
    df.index = pd.date_range('2025-01-01 06:00:00', periods=len(df), freq='1S')
    
    # Define intervals using timestamps
    interval_times = []
    for start_idx, end_idx in intervals:
        start_time = df.index[start_idx]
        end_time = df.index[end_idx - 1]
        interval_times.append((start_time, end_time))
    
    return {'data': df, 'intervals': interval_times, 'type': '30/30 Intervals'}


def create_3min_vo2_workout():
    """Create a 3 min VO2 max workout with active recovery."""
    np.random.seed(43)
    n_samples = 5400  # 1.5 hours
    
    # Base endurance power
    base_power = 170 + np.random.normal(0, 12, n_samples)
    
    # Warm up (15 min)
    warmup_end = 900
    
    # 5x3 min VO2 max intervals with 3 min active recovery
    interval_duration = 180  # 3 min
    recovery_duration = 180  # 3 min
    
    intervals = []
    current_time = warmup_end
    
    for i in range(5):
        # VO2 Max interval
        interval_start = current_time
        interval_end = current_time + interval_duration
        
        # High power with realistic fluctuations and fade
        vo2_power = np.linspace(340, 300, interval_end - interval_start) + np.random.normal(0, 25, interval_end - interval_start)
        # Add power fluctuations
        for j in range(interval_end - interval_start):
            vo2_power[j] += 25 * np.sin(j * np.pi / 90) + np.random.normal(0, 20)
        
        base_power[interval_start:interval_end] = vo2_power
        intervals.append((interval_start, interval_end))
        
        # Active recovery (not complete rest)
        if i < 4:
            recovery_start = interval_end
            recovery_end = recovery_start + recovery_duration
            recovery_power = 160 + np.random.normal(0, 12, recovery_end - recovery_start)
            base_power[recovery_start:recovery_end] = recovery_power
            current_time = recovery_end
        else:
            current_time = interval_end
    
    # Cool down
    cooldown_start = current_time
    cooldown_power = 140 + np.random.normal(0, 8, n_samples - cooldown_start)
    base_power[cooldown_start:] = cooldown_power
    
    # Create realistic cadence patterns
    cadence = np.full(n_samples, 80)
    
    # Adjust for intervals
    for start, end in intervals:
        # High cadence during intervals with variation
        interval_cadence = 95 + np.random.normal(0, 4, end - start)
        # Add cadence fluctuations
        for j in range(end - start):
            interval_cadence[j] += 6 * np.sin(j * np.pi / 60) + np.random.normal(0, 3)
        cadence[start:end] = interval_cadence
    
    # Calculate torque
    torque = base_power / (cadence * np.pi / 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'power': base_power,
        'cadence': cadence,
        'torque': torque
    })
    
    # Define intervals
    interval_times = [(df.index[start], df.index[end-1]) for start, end in intervals]
    
    return {'data': df, 'intervals': interval_times, 'type': '3min VO2 Max 5x3'}


def create_5x5_workout():
    """Create a 5x5 workout (5 min intervals at threshold)."""
    np.random.seed(44)
    n_samples = 6000  # 1.67 hours
    
    # Base endurance power
    base_power = 175 + np.random.normal(0, 14, n_samples)
    
    # Warm up (15 min)
    warmup_end = 900
    
    # 5x5 min threshold intervals with 5 min active recovery
    interval_duration = 300  # 5 min
    recovery_duration = 300  # 5 min
    
    intervals = []
    current_time = warmup_end
    
    for i in range(5):
        # Threshold interval
        interval_start = current_time
        interval_end = current_time + interval_duration
        
        # Threshold power with realistic fluctuations
        threshold_power = 290 + np.random.normal(0, 20, interval_end - interval_start)
        # Add power fluctuations
        for j in range(interval_end - interval_start):
            threshold_power[j] += 15 * np.sin(j * np.pi / 150) + np.random.normal(0, 18)
        
        base_power[interval_start:interval_end] = threshold_power
        intervals.append((interval_start, interval_end))
        
        # Active recovery
        if i < 4:
            recovery_start = interval_end
            recovery_end = recovery_start + recovery_duration
            recovery_power = 170 + np.random.normal(0, 15, recovery_end - recovery_start)
            base_power[recovery_start:recovery_end] = recovery_power
            current_time = recovery_end
        else:
            current_time = interval_end
    
    # Cool down
    cooldown_start = current_time
    cooldown_power = 145 + np.random.normal(0, 8, n_samples - cooldown_start)
    base_power[cooldown_start:] = cooldown_power
    
    # Create realistic cadence patterns
    cadence = np.full(n_samples, 82)
    
    # Adjust for intervals
    for start, end in intervals:
        # Moderate-high cadence during intervals with variation
        interval_cadence = 88 + np.random.normal(0, 3, end - start)
        # Add cadence fluctuations
        for j in range(end - start):
            interval_cadence[j] += 4 * np.sin(j * np.pi / 100) + np.random.normal(0, 2)
        cadence[start:end] = interval_cadence
    
    # Calculate torque
    torque = base_power / (cadence * np.pi / 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'power': base_power,
        'cadence': cadence,
        'torque': torque
    })
    
    # Define intervals
    interval_times = [(df.index[start], df.index[end-1]) for start, end in intervals]
    
    return {'data': df, 'intervals': interval_times, 'type': '5x5 Threshold'}


def create_4x_threshold_variations():
    """Create 4x threshold workout with varying durations (6, 8, 10, 12 min)."""
    np.random.seed(45)
    n_samples = 7200  # 2 hours
    
    # Base endurance power
    base_power = 170 + np.random.normal(0, 15, n_samples)
    
    # Warm up (15 min)
    warmup_end = 900
    
    # 4 threshold intervals with varying durations
    interval_durations = [360, 480, 600, 720]  # 6, 8, 10, 12 min
    recovery_duration = 300  # 5 min active recovery
    
    intervals = []
    current_time = warmup_end
    
    for i, duration in enumerate(interval_durations):
        # Threshold interval
        interval_start = current_time
        interval_end = current_time + duration
        
        # Threshold power with realistic fluctuations
        threshold_power = 285 + np.random.normal(0, 22, interval_end - interval_start)
        # Add power fluctuations
        for j in range(interval_end - interval_start):
            threshold_power[j] += 18 * np.sin(j * np.pi / (duration/2)) + np.random.normal(0, 20)
        
        base_power[interval_start:interval_end] = threshold_power
        intervals.append((interval_start, interval_end))
        
        # Active recovery (except after last interval)
        if i < 3:
            recovery_start = interval_end
            recovery_end = recovery_start + recovery_duration
            recovery_power = 165 + np.random.normal(0, 15, recovery_end - recovery_start)
            base_power[recovery_start:recovery_end] = recovery_power
            current_time = recovery_end
        else:
            current_time = interval_end
    
    # Cool down
    cooldown_start = current_time
    cooldown_power = 140 + np.random.normal(0, 8, n_samples - cooldown_start)
    base_power[cooldown_start:] = cooldown_power
    
    # Create realistic cadence patterns
    cadence = np.full(n_samples, 80)
    
    # Adjust for intervals
    for start, end in intervals:
        # Moderate cadence during intervals with variation
        interval_cadence = 85 + np.random.normal(0, 3, end - start)
        # Add cadence fluctuations
        for j in range(end - start):
            interval_cadence[j] += 5 * np.sin(j * np.pi / (end - start)) + np.random.normal(0, 2)
        cadence[start:end] = interval_cadence
    
    # Calculate torque
    torque = base_power / (cadence * np.pi / 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'power': base_power,
        'cadence': cadence,
        'torque': torque
    })
    
    # Define intervals
    interval_times = [(df.index[start], df.index[end-1]) for start, end in intervals]
    
    return {'data': df, 'intervals': interval_times, 'type': '4x Threshold Variations (6,8,10,12min)'}


def create_over_unders_workout():
    """Create an over-unders workout (alternating above/below threshold)."""
    np.random.seed(46)
    n_samples = 5400  # 1.5 hours
    
    # Base endurance power
    base_power = 175 + np.random.normal(0, 14, n_samples)
    
    # Warm up (15 min)
    warmup_end = 900
    
    # 6 over-unders sets (2 min each: 1 min over, 1 min under)
    set_duration = 120  # 2 min per set
    recovery_duration = 180  # 3 min active recovery
    
    intervals = []
    current_time = warmup_end
    
    for i in range(6):
        # Over-unders set
        set_start = current_time
        
        # 1 min over threshold
        over_start = set_start
        over_end = over_start + 60
        over_power = 310 + np.random.normal(0, 25, over_end - over_start)
        # Add power fluctuations
        for j in range(over_end - over_start):
            over_power[j] += 20 * np.sin(j * np.pi / 30) + np.random.normal(0, 20)
        base_power[over_start:over_end] = over_power
        
        # 1 min under threshold
        under_start = over_end
        under_end = under_start + 60
        under_power = 250 + np.random.normal(0, 20, under_end - under_start)
        # Add power fluctuations
        for j in range(under_end - under_start):
            under_power[j] += 15 * np.sin(j * np.pi / 30) + np.random.normal(0, 18)
        base_power[under_start:under_end] = under_power
        
        # Mark the entire set as an interval
        intervals.append((over_start, under_end - 1))
        
        # Active recovery (except after last set)
        if i < 5:
            recovery_start = under_end
            recovery_end = recovery_start + recovery_duration
            recovery_power = 160 + np.random.normal(0, 12, recovery_end - recovery_start)
            base_power[recovery_start:recovery_end] = recovery_power
            current_time = recovery_end
        else:
            current_time = under_end
    
    # Cool down
    cooldown_start = current_time
    cooldown_power = 140 + np.random.normal(0, 8, n_samples - cooldown_start)
    base_power[cooldown_start:] = cooldown_power
    
    # Create realistic cadence patterns
    cadence = np.full(n_samples, 82)
    
    # Adjust for intervals
    for start, end in intervals:
        # Higher cadence during over sections
        over_start = start
        over_end = start + 60
        cadence[over_start:over_end] = 90 + np.random.normal(0, 4, over_end - over_start)
        
        # Lower cadence during under sections
        under_start = over_end
        under_end = min(end + 1, n_samples)
        cadence[under_start:under_end] = 78 + np.random.normal(0, 3, under_end - under_start)
    
    # Calculate torque
    torque = base_power / (cadence * np.pi / 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'power': base_power,
        'cadence': cadence,
        'torque': torque
    })
    
    # Define intervals
    interval_times = [(df.index[start], df.index[end-1]) for start, end in intervals]
    
    return {'data': df, 'intervals': interval_times, 'type': 'Over-Unders 6x2min'}


def create_free_ride_workout():
    """Create a free ride workout with natural power variations and some intervals."""
    np.random.seed(47)
    n_samples = 7200  # 2 hours
    
    # Base endurance power with natural variations
    base_power = 180 + np.random.normal(0, 15, n_samples)
    
    # Add natural power variations (hills, wind, fatigue)
    for i in range(0, n_samples, 300):  # Every 5 minutes
        # Random power variations
        variation = np.random.normal(0, 25, min(300, n_samples - i))
        base_power[i:i+len(variation)] += variation
    
    # Add some spontaneous intervals (not structured)
    spontaneous_intervals = []
    current_time = 1200  # Start after 20 min warm-up
    
    # 3-5 spontaneous efforts
    num_efforts = np.random.randint(3, 6)
    for i in range(num_efforts):
        # Random effort duration (30s to 3 min)
        effort_duration = np.random.randint(30, 180)
        effort_start = current_time + np.random.randint(300, 900)  # 5-15 min apart
        
        if effort_start + effort_duration < n_samples:
            # High power effort
            effort_power = 280 + np.random.normal(0, 30, effort_duration)
            base_power[effort_start:effort_start+effort_duration] = effort_power
            spontaneous_intervals.append((effort_start, effort_start+effort_duration))
            current_time = effort_start + effort_duration
    
    # Create realistic cadence patterns
    cadence = np.full(n_samples, 85.0, dtype=float)
    
    # Add cadence variations
    for i in range(0, n_samples, 200):  # Every ~3 minutes
        variation = np.random.normal(0, 8, min(200, n_samples - i))
        cadence[i:i+len(variation)] += variation
    
    # Adjust cadence for spontaneous intervals
    for start, end in spontaneous_intervals:
        interval_cadence = 90 + np.random.normal(0, 5, end - start)
        cadence[start:end] = interval_cadence
    
    # Calculate torque
    torque = base_power / (cadence * np.pi / 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'power': base_power,
        'cadence': cadence,
        'torque': torque
    })
    
    # Define intervals
    interval_times = [(df.index[start], df.index[end-1]) for start, end in spontaneous_intervals]
    
    return {'data': df, 'intervals': interval_times, 'type': 'Free Ride with Spontaneous Efforts'}


def create_recovery_ride_workout():
    """Create a recovery ride with mostly easy riding and minimal intervals."""
    np.random.seed(48)
    n_samples = 5400  # 1.5 hours
    
    # Base recovery power (low and steady)
    base_power = 140 + np.random.normal(0, 8, n_samples)
    
    # Add gentle power variations
    for i in range(0, n_samples, 600):  # Every 10 minutes
        variation = np.random.normal(0, 12, min(600, n_samples - i))
        base_power[i:i+len(variation)] += variation
    
    # Only 1-2 very short intervals (for variety)
    intervals = []
    if np.random.random() > 0.5:  # 50% chance of one interval
        interval_start = np.random.randint(1800, 3600)  # Between 30-60 min
        interval_duration = np.random.randint(60, 120)  # 1-2 min
        
        if interval_start + interval_duration < n_samples:
            # Gentle interval (not too hard)
            interval_power = 200 + np.random.normal(0, 15, interval_duration)
            base_power[interval_start:interval_start+interval_duration] = interval_power
            intervals.append((interval_start, interval_start+interval_duration))
    
    # Create steady cadence patterns
    cadence = np.full(n_samples, 75.0, dtype=float)
    
    # Add minimal cadence variations
    for i in range(0, n_samples, 900):  # Every 15 minutes
        variation = np.random.normal(0, 3, min(900, n_samples - i))
        cadence[i:i+len(variation)] += variation
    
    # Calculate torque
    torque = base_power / (cadence * np.pi / 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'power': base_power,
        'cadence': cadence,
        'torque': torque
    })
    
    # Define intervals
    interval_times = [(df.index[start], df.index[end-1]) for start, end in intervals]
    
    return {'data': df, 'intervals': interval_times, 'type': 'Recovery Ride'}


def create_sample_training_data():
    """
    Create realistic sample training data for cycling workouts.
    Generates various workout types with realistic power patterns.
    """
    print("🎯 Creating realistic cycling workout training data...")
    
    workouts = []
    all_interval_labels = []
    
    # Workout 1: 30/30 Intervals
    print("⚡ Creating 30/30 workout...")
    workout1 = create_30_30_workout()
    workouts.append(workout1)
    all_interval_labels.extend(workout1['intervals'])
    
    # Workout 2: 3 min VO2 Max
    print("💨 Creating 3min VO2 Max workout...")
    workout2 = create_3min_vo2_workout()
    workouts.append(workout2)
    all_interval_labels.extend(workout2['intervals'])
    
    # Workout 3: 5x5 Threshold
    print("🔥 Creating 5x5 Threshold workout...")
    workout3 = create_5x5_workout()
    workouts.append(workout3)
    all_interval_labels.extend(workout3['intervals'])
    
    # Workout 4: 4x Threshold Variations
    print("📊 Creating 4x Threshold Variations workout...")
    workout4 = create_4x_threshold_variations()
    workouts.append(workout4)
    all_interval_labels.extend(workout4['intervals'])
    
    # Workout 5: Over-Unders
    print("🔄 Creating Over-Unders workout...")
    workout5 = create_over_unders_workout()
    workouts.append(workout5)
    all_interval_labels.extend(workout5['intervals'])
    
    # Combine all workouts
    combined_df = pd.concat([w['data'] for w in workouts], ignore_index=True)
    combined_df.index = pd.date_range('2025-01-01 06:00:00', periods=len(combined_df), freq='1S')
    
    # Fix interval labels to match the new combined DataFrame index
    fixed_interval_labels = []
    current_offset = 0
    
    for workout in workouts:
        workout_length = len(workout['data'])
        # Get the original interval indices (before timestamp conversion)
        original_intervals = []
        for start_time, end_time in workout['intervals']:
            # Find the original integer indices from the workout data
            start_idx = workout['data'].index.get_loc(start_time)
            end_idx = workout['data'].index.get_loc(end_time)
            original_intervals.append((start_idx, end_idx))
        
        # Now create new timestamps for the combined DataFrame
        for start_idx, end_idx in original_intervals:
            adjusted_start = combined_df.index[current_offset + start_idx]
            adjusted_end = combined_df.index[current_offset + end_idx]
            fixed_interval_labels.append((adjusted_start, adjusted_end))
        
        current_offset += workout_length
    
    print(f"✅ Created {len(workouts)} realistic workouts")
    print(f"🏷️ Total intervals: {len(fixed_interval_labels)}")
    print(f"📊 Total data points: {len(combined_df)}")
    
    return combined_df, fixed_interval_labels


def create_extended_training_dataset(num_workouts=50):
    """
    Create an extended training dataset with many variations of workouts.
    This is useful for training a robust model.
    """
    print(f"🎯 Creating extended training dataset with {num_workouts} workouts...")
    
    all_workouts = []
    all_interval_labels = []
    
    # Create multiple variations of each workout type
    workout_types = [
        create_30_30_workout,
        create_3min_vo2_workout,
        create_5x5_workout,
        create_4x_threshold_variations,
        create_over_unders_workout,
        create_free_ride_workout,
        create_recovery_ride_workout
    ]
    
    for i in range(num_workouts):
        # Randomly select workout type
        workout_func = np.random.choice(workout_types)
        
        # Create workout with slight variations
        workout = workout_func()
        
        # Add some random variation to make each workout unique
        workout['data']['power'] += np.random.normal(0, 5, len(workout['data']))
        workout['data']['cadence'] += np.random.normal(0, 2, len(workout['data']))
        workout['data']['torque'] += np.random.normal(0, 1, len(workout['data'])) # Add torque variation
        
        all_workouts.append(workout)
        all_interval_labels.extend(workout['intervals'])
        
        if (i + 1) % 10 == 0:
            print(f"   Created {i + 1}/{num_workouts} workouts...")
    
    # Combine all workouts
    combined_df = pd.concat([w['data'] for w in all_workouts], ignore_index=True)
    combined_df.index = pd.date_range('2025-01-01 06:00:00', periods=len(combined_df), freq='1S')
    
    # Fix interval labels to match the new combined DataFrame index (same as basic dataset)
    fixed_interval_labels = []
    current_offset = 0
    
    for workout in all_workouts:
        workout_length = len(workout['data'])
        # Get the original interval indices (before timestamp conversion)
        original_intervals = []
        for start_time, end_time in workout['intervals']:
            # Find the original integer indices from the workout data
            start_idx = workout['data'].index.get_loc(start_time)
            end_idx = workout['data'].index.get_loc(end_time)
            original_intervals.append((start_idx, end_idx))
        
        # Now create new timestamps for the combined DataFrame
        for start_idx, end_idx in original_intervals:
            adjusted_start = combined_df.index[current_offset + start_idx]
            adjusted_end = combined_df.index[current_offset + end_idx]
            fixed_interval_labels.append((adjusted_start, adjusted_end))
        
        current_offset += workout_length
    
    print(f"✅ Created extended dataset: {len(all_workouts)} workouts")
    print(f"🏷️ Total intervals: {len(fixed_interval_labels)}")
    print(f"📊 Total data points: {len(combined_df)}")
    
    return combined_df, fixed_interval_labels


def visualize_workout_samples(df, interval_labels, num_samples=3):
    """
    Visualize sample workouts to show the training data quality.
    
    Args:
        df (pd.DataFrame): Combined workout data
        interval_labels (list): List of interval tuples
        num_samples (int): Number of sample workouts to show
    """
    print(f"\n📊 Visualizing {num_samples} sample workouts...")
    
    # Find workout boundaries (approximate by looking for long recovery periods)
    workout_boundaries = find_workout_boundaries(df)
    
    # Show sample workouts
    for i, (start_idx, end_idx) in enumerate(workout_boundaries[:num_samples]):
        if i >= num_samples:
            break
            
        workout_data = df.iloc[start_idx:end_idx]
        workout_intervals = [interval for interval in interval_labels 
                           if isinstance(interval[0], pd.Timestamp) and 
                              isinstance(interval[1], pd.Timestamp) and
                              interval[0] >= workout_data.index[0] and 
                              interval[1] <= workout_data.index[-1]]
        
        # Create subplot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot power
        ax1.plot(workout_data.index, workout_data['power'], 'b-', linewidth=1, alpha=0.7)
        ax1.set_ylabel('Power (W)')
        ax1.set_title(f'Sample Workout {i+1} - Power Profile')
        ax1.grid(True, alpha=0.3)
        
        # Highlight intervals
        for start, end in workout_intervals:
            ax1.axvspan(start, end, alpha=0.3, color='red', label='Interval' if start == workout_intervals[0][0] else "")
        
        # Plot cadence
        ax2.plot(workout_data.index, workout_data['cadence'], 'g-', linewidth=1, alpha=0.7)
        ax2.set_ylabel('Cadence (rpm)')
        ax2.set_title('Cadence Profile')
        ax2.grid(True, alpha=0.3)
        
        # Highlight intervals
        for start, end in workout_intervals:
            ax2.axvspan(start, end, alpha=0.3, color='red')
        
        # Plot torque
        ax3.plot(workout_data.index, workout_data['torque'], 'r-', linewidth=1, alpha=0.7)
        ax3.set_ylabel('Torque (Nm)')
        ax3.set_title('Torque Profile')
        ax3.set_xlabel('Time')
        ax3.grid(True, alpha=0.3)
        
        # Highlight intervals
        for start, end in workout_intervals:
            ax3.axvspan(start, end, alpha=0.3, color='red')
        
        plt.tight_layout()
        plt.show()
        
        print(f"   Workout {i+1}: {len(workout_intervals)} intervals, "
              f"Duration: {(workout_data.index[-1] - workout_data.index[0]).total_seconds() / 60:.1f} min")


def find_workout_boundaries(df, min_recovery_duration=600):
    """
    Find approximate workout boundaries by looking for long recovery periods.
    
    Args:
        df (pd.DataFrame): Workout data
        min_recovery_duration (int): Minimum recovery duration in seconds
    
    Returns:
        list: List of (start_idx, end_idx) tuples for each workout
    """
    # Calculate power rolling average to smooth out noise
    power_smooth = df['power'].rolling(window=60, min_periods=1).mean()
    
    # Find periods where power is consistently low (recovery)
    recovery_mask = power_smooth < 200  # Assuming 200W is recovery threshold
    
    # Find continuous recovery periods
    workout_boundaries = []
    in_recovery = False
    recovery_start = None
    
    for i, is_recovery in enumerate(recovery_mask):
        if is_recovery and not in_recovery:
            recovery_start = i
            in_recovery = True
        elif not is_recovery and in_recovery:
            recovery_duration = i - recovery_start
            if recovery_duration >= min_recovery_duration:
                # This is a significant recovery period, likely between workouts
                if workout_boundaries:
                    # End previous workout
                    workout_boundaries[-1] = (workout_boundaries[-1][0], recovery_start)
                # Start new workout
                workout_boundaries.append((recovery_start, len(df)))
            in_recovery = False
            recovery_start = None
    
    # Handle first workout if no recovery period found
    if not workout_boundaries:
        workout_boundaries = [(0, len(df))]
    
    return workout_boundaries


def load_real_training_data(fit_file_paths):
    """
    Load real FIT files with lap data to create training data.
    
    Args:
        fit_file_paths (list): List of paths to FIT files
        
    Returns:
        tuple: (combined_df, all_interval_labels)
    """
    print("🎯 Loading real training data from FIT files...")
    
    all_rides = []
    all_interval_labels = []
    
    for i, fit_path in enumerate(fit_file_paths):
        print(f"📁 Loading FIT file {i+1}/{len(fit_file_paths)}: {fit_path}")
        
        try:
            # Load the FIT file using SprintV1's function
            df = load_fit_to_dataframe(fit_path)
            
            if df is None or df.empty:
                print(f"⚠️ Warning: Could not load {fit_path}")
                continue
            
            # Check if we have power and cadence data
            required_cols = ['power', 'cadence']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️ Warning: Missing required columns in {fit_path}")
                continue
            
            # Calculate torque if not present
            if 'torque' not in df.columns:
                df['torque'] = df['power'] / (df['cadence'] * np.pi / 30)
                df['torque'] = df['torque'].fillna(0)  # Handle division by zero
            
            # Clean and prepare the data first
            df_clean = clean_ride_data(df)
            
            # Extract lap information AFTER cleaning to ensure timestamp alignment
            if 'lap' in df_clean.columns:
                print(f"   📊 Lap data found: {df_clean['lap'].nunique()} unique laps")
                print(f"   📊 Cleaned data timestamps: {df_clean.index[0]} to {df_clean.index[-1]}")
                
                # Extract intervals directly from lap numbers in the cleaned data
                lap_intervals = extract_lap_intervals_from_data(df_clean)
                print(f"   ✅ Found {len(lap_intervals)} valid intervals")
                all_interval_labels.extend(lap_intervals)
            else:
                print(f"   ⚠️ No lap data found in {fit_path}")
                continue
            
            all_rides.append(df_clean)
            
        except Exception as e:
            print(f"❌ Error loading {fit_path}: {e}")
            continue
    
    if not all_rides:
        raise ValueError("No valid FIT files could be loaded")
    
    # For single ride, keep original timestamps
    if len(all_rides) == 1:
        combined_df = all_rides[0]
    else:
        # For multiple rides, combine and create continuous timestamps
        combined_df = pd.concat(all_rides, ignore_index=True)
        combined_df.index = pd.date_range('2025-01-01 06:00:00', periods=len(combined_df), freq='1S')
        
        # Update interval labels to match new timestamps
        updated_intervals = []
        current_offset = 0
        
        for i, ride in enumerate(all_rides):
            ride_start = current_offset
            ride_end = current_offset + len(ride)
            
            # Find intervals that belong to this ride and update their timestamps
            ride_intervals = [interval for interval in all_interval_labels 
                            if interval[0] >= ride.index[0] and interval[1] <= ride.index[-1]]
            
            for start_time, end_time in ride_intervals:
                # Convert to new timestamps
                start_offset = (start_time - ride.index[0]).total_seconds()
                end_offset = (end_time - ride.index[0]).total_seconds()
                
                new_start = combined_df.index[ride_start + int(start_offset)]
                new_end = combined_df.index[ride_start + int(end_offset)]
                updated_intervals.append((new_start, new_end))
            
            current_offset = ride_end
        
        all_interval_labels = updated_intervals
    
    print(f"✅ Loaded {len(all_rides)} real rides")
    print(f"🏷️ Total intervals: {len(all_interval_labels)}")
    print(f"📊 Total data points: {len(combined_df)}")
    
    return combined_df, all_interval_labels


def extract_lap_intervals(df):
    """
    Extract interval labels from lap data.
    
    Args:
        df (pd.DataFrame): DataFrame with lap column
        
    Returns:
        list: List of (start_time, end_time) tuples for each lap
    """
    intervals = []
    
    # Group by lap and find start/end times
    for lap_num in df['lap'].unique():
        if pd.isna(lap_num):
            continue
            
        lap_data = df[df['lap'] == lap_num]
        if len(lap_data) < 10:  # Skip very short laps
            continue
            
        start_time = lap_data.index[0]
        end_time = lap_data.index[-1]
        
        # Only include laps longer than 30 seconds
        duration = (end_time - start_time).total_seconds()
        if duration >= 30:
            intervals.append((start_time, end_time))
    
    return intervals


def extract_lap_intervals_from_data(df):
    """
    Extract and classify interval labels directly from lap numbers in the processed data.
    Intelligently classifies laps as work vs rest intervals based on power, duration, and cadence patterns.
    
    Args:
        df (pd.DataFrame): DataFrame with lap column
        
    Returns:
        list: List of (start_time, end_time) tuples for WORK intervals only
    """
    work_intervals = []
    rest_intervals = []
    
    # Get unique lap numbers, excluding NaN
    lap_numbers = sorted([lap for lap in df['lap'].unique() if pd.notna(lap)])
    
    # Calculate overall ride statistics for comparison
    overall_avg_power = df['power'].mean()
    overall_avg_cadence = df['cadence'].mean()
    
    # Calculate rolling averages to identify effort patterns
    df['power_rolling_5min'] = df['power'].rolling(window=300, min_periods=1).mean()
    df['cadence_rolling_5min'] = df['cadence'].rolling(window=300, min_periods=1).mean()
    
    print(f"   📊 Overall ride averages: Power {overall_avg_power:.0f}W, Cadence {overall_avg_cadence:.0f} rpm")
    print(f"   📊 Power range: {df['power'].min():.0f}W - {df['power'].max():.0f}W")
    print(f"   📊 Cadence range: {df['cadence'].min():.0f} - {df['cadence'].max():.0f} rpm")
    
    # Calculate and display FTP analysis using best efforts and backwards calculation
    detector = IntervalDetector()
    ftp_estimate = detector._estimate_ftp_from_best_efforts(df)
    print(f"   🎯 FTP Estimate: {ftp_estimate:.0f}W (from best efforts)")
    
    print(f"   📊 Power percentiles: 60th={df['power'].quantile(0.60):.0f}W, 75th={df['power'].quantile(0.75):.0f}W")
    print(f"   📊 Power std dev: {df['power'].std():.0f}W")
    
    for lap_num in lap_numbers:
        lap_data = df[df['lap'] == lap_num]
        
        if len(lap_data) < 10:  # Skip very short laps (less than 10 data points)
            print(f"   Lap {lap_num}: SKIPPED - only {len(lap_data)} data points")
            continue
            
        # Use the actual data timestamps from the DataFrame
        start_time = lap_data.index[0]
        end_time = lap_data.index[-1]
        
        # Only include laps longer than 15 seconds (to capture 30/30 workout intervals)
        duration = (end_time - start_time).total_seconds()
        if duration < 15:
            print(f"   Lap {lap_num}: SKIPPED - only {duration:.1f}s duration")
            continue
            
        # Calculate lap statistics
        lap_avg_power = lap_data['power'].mean()
        lap_avg_cadence = lap_data['cadence'].mean()
        lap_max_power = lap_data['power'].max()
        
        # Use data-driven classification based on relative patterns within the ride
        # Calculate percentile ranks for this lap relative to the entire ride
        power_percentile = (df['power'] < lap_avg_power).mean() * 100
        cadence_percentile = (df['cadence'] < lap_avg_cadence).mean() * 100
        
        # Calculate intensity score (weighted combination of power and cadence)
        # Normalize both to 0-1 scale relative to ride min/max
        power_normalized = (lap_avg_power - df['power'].min()) / (df['power'].max() - df['power'].min())
        cadence_normalized = (lap_avg_cadence - df['cadence'].min()) / (df['cadence'].max() - df['cadence'].min())
        
        # Intensity score: 70% power + 30% cadence (power is more important for effort)
        intensity_score = 0.7 * power_normalized + 0.3 * cadence_normalized
        
        # POWER-BASED CLASSIFICATION: Use % FTP and relative power analysis
        
        # Calculate FTP estimate using best efforts and backwards calculation
        # This gives us a much more accurate baseline for % FTP calculations
        # We already calculated this above, so we can reuse it
        # ftp_estimate = detector._estimate_ftp_from_best_efforts(df)  # Already calculated above
        
        # Calculate % FTP for this lap
        lap_ftp_percent = (lap_avg_power / ftp_estimate) * 100 if ftp_estimate > 0 else 0
        
        # Calculate power statistics for the entire ride
        ride_power_mean = df['power'].mean()
        ride_power_std = df['power'].std()
        ride_power_75th = df['power'].quantile(0.75)
        ride_power_60th = df['power'].quantile(0.60)
        
        # CLASSIFICATION CRITERIA:
        
        # Criterion 1: % FTP-based (most important for cycling)
        ftp_work = lap_ftp_percent > 85  # Above 85% FTP = work interval
        
        # Criterion 2: Relative to ride average (identifies high-effort periods)
        relative_work = lap_avg_power > (ride_power_mean + 0.5 * ride_power_std)
        
        # Criterion 3: Percentile ranking within the ride
        percentile_work = lap_avg_power > ride_power_60th  # Top 40% by power
        
        # Criterion 4: Absolute power threshold (catches very high efforts)
        absolute_work = lap_avg_power > 200  # Above 200W = likely work
        
        # Criterion 5: Pattern-based (for structured workouts like 30/30)
        pattern_work = False
        if len(work_intervals) > 0 and len(rest_intervals) > 0:
            # Look for alternating patterns in recent intervals
            recent_work = len(work_intervals[-3:]) if len(work_intervals) >= 3 else len(work_intervals)
            recent_rest = len(rest_intervals[-3:]) if len(rest_intervals) >= 3 else len(rest_intervals)
            pattern_work = recent_rest > recent_work  # Alternate if more rest recently
        
        # WEIGHTED CLASSIFICATION SCORE:
        # % FTP: 40%, Relative to ride: 25%, Percentile: 20%, Absolute: 10%, Pattern: 5%
        classification_score = (
            0.40 * ftp_work +           # % FTP is most important
            0.25 * relative_work +      # Relative to ride average
            0.20 * percentile_work +    # Percentile ranking
            0.10 * absolute_work +      # Absolute threshold
            0.05 * pattern_work         # Pattern recognition
        )
        
        # Determine if this is a work interval
        is_work_interval = classification_score > 0.5
        
        # Create detailed classification reason
        if is_work_interval:
            reasons = []
            if ftp_work: reasons.append(f"{lap_ftp_percent:.0f}% FTP")
            if relative_work: reasons.append("above ride avg")
            if percentile_work: reasons.append("top 40% power")
            if absolute_work: reasons.append(">200W")
            if pattern_work: reasons.append("pattern")
            
            reason = f"WORK: {', '.join(reasons)} (score: {classification_score:.2f})"
        else:
            reasons = []
            if lap_ftp_percent < 60: reasons.append(f"{lap_ftp_percent:.0f}% FTP")
            if lap_avg_power < ride_power_mean: reasons.append("below ride avg")
            if lap_avg_power < ride_power_60th: reasons.append("bottom 60% power")
            if lap_avg_power < 150: reasons.append("<150W")
            
            reason = f"REST: {', '.join(reasons)} (score: {classification_score:.2f})"
        
        if is_work_interval:
            work_intervals.append((start_time, end_time))
            print(f"   🚴 WORK Lap {lap_num}: {start_time} to {end_time} ({duration:.1f}s) - {len(lap_data)} pts")
            print(f"      Power: {lap_avg_power:.0f}W ({lap_ftp_percent:.0f}% FTP, max: {lap_max_power:.0f}W)")
            print(f"      Cadence: {lap_avg_cadence:.0f} rpm | {reason}")
        else:
            rest_intervals.append((start_time, end_time))
            print(f"   🛌 REST Lap {lap_num}: {start_time} to {end_time} ({duration:.1f}s) - {len(lap_data)} pts")
            print(f"      Power: {lap_avg_power:.0f}W ({lap_ftp_percent:.0f}% FTP, max: {lap_max_power:.0f}W)")
            print(f"      Cadence: {lap_avg_cadence:.0f} rpm | {reason}")
    
    print(f"   📊 Classified: {len(work_intervals)} work intervals, {len(rest_intervals)} rest intervals")
    
    # Return only work intervals for training (rest periods will be labeled as 0)
    return work_intervals


def extract_lap_intervals_from_timestamps(df):
    """
    Extract interval labels from lap data using timestamp-based approach.
    This handles cases where lap timestamps might not match data timestamps exactly.
    
    Args:
        df (pd.DataFrame): DataFrame with lap column
        
    Returns:
        list: List of (start_time, end_time) tuples for each lap
    """
    intervals = []
    
    # Group by lap and find start/end times
    for lap_num in df['lap'].unique():
        if pd.isna(lap_num):
            continue
            
        lap_data = df[df['lap'] == lap_num]
        if len(lap_data) < 10:  # Skip very short laps
            continue
            
        # Use the actual data timestamps from the DataFrame
        start_time = lap_data.index[0]
        end_time = lap_data.index[-1]
        
        # Only include laps longer than 15 seconds (to capture 30/30 workout intervals)
        duration = (end_time - start_time).total_seconds()
        if duration >= 15:
            intervals.append((start_time, end_time))
            print(f"   Lap {lap_num}: {start_time} to {end_time} ({duration:.1f}s)")
    
    return intervals


def clean_ride_data(df):
    """
    Clean and prepare ride data for training.
    
    Args:
        df (pd.DataFrame): Raw ride data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Only remove rows with missing power (we need power for training)
    df_clean = df_clean.dropna(subset=['power'])
    
    # Remove extreme outliers only (keep reasonable power range including 0 for coasting)
    df_clean = df_clean[
        (df_clean['power'] >= 0) & (df_clean['power'] < 2000)
    ]
    
    # Handle cadence - fill missing values with 0 (coasting)
    if 'cadence' in df_clean.columns:
        df_clean['cadence'] = df_clean['cadence'].fillna(0)
        # Remove extreme cadence outliers only
        df_clean = df_clean[df_clean['cadence'] < 250]
    else:
        df_clean['cadence'] = 0
    
    # Calculate torque - handle division by zero properly
    if 'torque' not in df_clean.columns:
        # When cadence is 0, torque should be 0 (not infinity)
        df_clean['torque'] = np.where(
            df_clean['cadence'] > 0,
            df_clean['power'] / (df_clean['cadence'] * np.pi / 30),
            0
        )
    
    print(f"   📊 Data cleaning: {len(df)} → {len(df_clean)} data points ({len(df_clean)/len(df)*100:.1f}% retained)")
    
    # Debug: print some sample timestamps to understand the data structure
    print(f"   🔍 Sample timestamps: {df_clean.index[:5].tolist()}")
    print(f"   🔍 Sample timestamps: {df_clean.index[-5:].tolist()}")
    
    return df_clean


def create_training_data_from_fits():
    """
    Interactive function to create training data from FIT files.
    """
    print("🎯 Training Data Creation from Real FIT Files")
    print("=" * 50)
    
    # Get FIT file paths from user
    print("Enter paths to FIT files (one per line, empty line to finish):")
    fit_paths = []
    
    while True:
        path = input("FIT file path: ").strip()
        if not path:
            break
        fit_paths.append(path)
    
    if not fit_paths:
        print("❌ No FIT files provided")
        return None, None
    
    # Load the real training data
    try:
        df, interval_labels = load_real_training_data(fit_paths)
        return df, interval_labels
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None, None


def test_model_on_new_ride(model_path, fit_file_path):
    """
    Test a trained model on a new FIT file to detect intervals.
    
    Args:
        model_path (str): Path to the trained model file
        fit_file_path (str): Path to the FIT file to analyze
    """
    print(f"🔮 Testing trained model on: {fit_file_path}")
    
    # Load the trained model
    detector = IntervalDetector()
    try:
        detector.load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load the FIT file
    try:
        df = load_fit_to_dataframe(fit_file_path)
        if df is None or df.empty:
            print("❌ Could not load FIT file")
            return
    except Exception as e:
        print(f"❌ Error loading FIT file: {e}")
        return
    
    # Check if we have required data
    if 'power' not in df.columns or 'cadence' not in df.columns:
        print("❌ FIT file missing power or cadence data")
        return
    
    # Calculate torque if not present
    if 'torque' not in df.columns:
        df['torque'] = df['power'] / (df['cadence'] * np.pi / 30)
        df['torque'] = df['torque'].fillna(0)
    
    # Clean the data
    df_clean = clean_ride_data(df)
    
    # Predict intervals
    predicted_intervals = detector.predict_intervals(df_clean)
    
    print(f"\n🎯 Interval Detection Results:")
    print(f"📊 Total ride duration: {(df_clean.index[-1] - df_clean.index[0]).total_seconds() / 60:.1f} minutes")
    print(f"🔍 Detected {len(predicted_intervals)} intervals:")
    
    for i, (start, end) in enumerate(predicted_intervals):
        duration = (end - start).total_seconds()
        start_min = (start - df_clean.index[0]).total_seconds() / 60
        end_min = (end - df_clean.index[0]).total_seconds() / 60
        
        print(f"   Interval {i+1}: {start_min:.1f} - {end_min:.1f} min ({duration:.1f}s)")
    
    # Visualize the results
    visualize_ride_with_intervals(df_clean, predicted_intervals)
    
    return predicted_intervals


def visualize_ride_with_intervals(df, intervals):
    """
    Visualize a ride with detected intervals highlighted.
    
    Args:
        df (pd.DataFrame): Ride data
        intervals (list): List of detected intervals
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot power
    ax1.plot(df.index, df['power'], 'b-', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Ride Power Profile with Detected Intervals')
    ax1.grid(True, alpha=0.3)
    
    # Highlight intervals
    for start, end in intervals:
        ax1.axvspan(start, end, alpha=0.3, color='red')
    
    # Plot cadence
    ax2.plot(df.index, df['cadence'], 'g-', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Cadence (rpm)')
    ax2.set_title('Cadence Profile')
    ax2.grid(True, alpha=0.3)
    
    # Highlight intervals
    for start, end in intervals:
        ax2.axvspan(start, end, alpha=0.3, color='red')
    
    # Plot torque
    ax3.plot(df.index, df['torque'], 'r-', linewidth=1, alpha=0.7)
    ax3.set_ylabel('Torque (Nm)')
    ax3.set_title('Torque Profile')
    ax3.set_xlabel('Time')
    ax3.grid(True, alpha=0.3)
    
    # Highlight intervals
    for start, end in intervals:
        ax3.axvspan(start, end, alpha=0.3, color='red')
    
    plt.tight_layout()
    plt.show()


def visualize_model_predictions(df, actual_intervals, predicted_intervals):
    """
    Visualize the model's predictions compared to actual intervals.
    
    Args:
        df (pd.DataFrame): Ride data
        actual_intervals (list): Actual intervals from FIT file
        predicted_intervals (list): Intervals predicted by the model
    """
    print("\n📊 Visualizing Model Predictions vs Actual Intervals...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot power with actual intervals (green) and predicted intervals (red)
    ax1.plot(df.index, df['power'], 'b-', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Power Profile: Actual Intervals (Green) vs Model Predictions (Red)')
    ax1.grid(True, alpha=0.3)
    
    # Highlight actual intervals in green
    for start, end in actual_intervals:
        ax1.axvspan(start, end, alpha=0.3, color='green', label='Actual Interval' if start == actual_intervals[0][0] else "")
    
    # Highlight predicted intervals in red
    for start, end in predicted_intervals:
        ax1.axvspan(start, end, alpha=0.3, color='red', label='Predicted Interval' if start == predicted_intervals[0][0] else "")
    
    # Plot cadence
    ax2.plot(df.index, df['cadence'], 'g-', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Cadence (rpm)')
    ax2.set_title('Cadence Profile')
    ax2.grid(True, alpha=0.3)
    
    # Highlight intervals
    for start, end in actual_intervals:
        ax2.axvspan(start, end, alpha=0.3, color='green')
    for start, end in predicted_intervals:
        ax2.axvspan(start, end, alpha=0.3, color='red')
    
    # Plot torque
    ax3.plot(df.index, df['torque'], 'r-', linewidth=1, alpha=0.7)
    ax3.set_ylabel('Torque (Nm)')
    ax3.set_title('Torque Profile')
    ax3.set_xlabel('Time')
    ax3.grid(True, alpha=0.3)
    
    # Highlight intervals
    for start, end in actual_intervals:
        ax3.axvspan(start, end, alpha=0.3, color='green')
    for start, end in predicted_intervals:
        ax3.axvspan(start, end, alpha=0.3, color='red')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='Actual Intervals'),
        Patch(facecolor='red', alpha=0.3, label='Model Predictions')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate overlap metrics
    calculate_interval_overlap(actual_intervals, predicted_intervals)


def calculate_interval_overlap(actual_intervals, predicted_intervals):
    """
    Calculate how well the predicted intervals overlap with actual intervals.
    
    Args:
        actual_intervals (list): Actual intervals from FIT file
        predicted_intervals (list): Intervals predicted by the model
    """
    print(f"\n📊 Interval Detection Performance:")
    print(f"   Actual intervals: {len(actual_intervals)}")
    print(f"   Predicted intervals: {len(predicted_intervals)}")
    
    if not predicted_intervals:
        print("   ❌ Model didn't predict any intervals!")
        return
    
    # Calculate overlap for each actual interval
    overlaps = []
    for i, (actual_start, actual_end) in enumerate(actual_intervals):
        actual_duration = (actual_end - actual_start).total_seconds()
        best_overlap = 0
        best_predicted = None
        
        for pred_start, pred_end in predicted_intervals:
            # Calculate overlap
            overlap_start = max(actual_start, pred_start)
            overlap_end = min(actual_end, pred_end)
            
            if overlap_end > overlap_start:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                overlap_ratio = overlap_duration / actual_duration
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_predicted = (pred_start, pred_end)
        
        overlaps.append(best_overlap)
        
        if best_predicted:
            pred_duration = (best_predicted[1] - best_predicted[0]).total_seconds()
            print(f"   Interval {i+1}: {best_overlap:.1%} overlap (Actual: {actual_duration:.0f}s, Predicted: {pred_duration:.0f}s)")
        else:
            print(f"   Interval {i+1}: 0% overlap (Model missed this interval)")
    
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    print(f"\n   📈 Average overlap: {avg_overlap:.1%}")
    
    if avg_overlap > 0.8:
        print("   🎉 Excellent! Model is learning your interval patterns well.")
    elif avg_overlap > 0.6:
        print("   👍 Good! Model is learning but could improve with more data.")
    elif avg_overlap > 0.3:
        print("   🤔 Fair. Model needs more training data or feature tuning.")
    else:
        print("   ❌ Poor performance. Model needs significant improvement.")


def main():
    """Main function to demonstrate the interval detector."""
    print("🚴 Interval Detection with Machine Learning")
    print("=" * 50)
    
    # Choose training dataset size
    print("Choose training dataset size:")
    print("1. Basic (6 workouts)")
    print("2. Extended (50 workouts)")
    print("3. Custom size")
    print("4. Real FIT files")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🏃‍♂️ Creating basic training dataset...")
        df, interval_labels = create_sample_training_data()
    elif choice == "2":
        print("\n🏃‍♂️ Creating extended training dataset...")
        df, interval_labels = create_extended_training_dataset(50)
    elif choice == "3":
        try:
            size = int(input("Enter number of workouts (10-200): "))
            size = max(10, min(200, size))  # Clamp between 10-200
            print(f"\n🏃‍♂️ Creating custom training dataset with {size} workouts...")
            df, interval_labels = create_extended_training_dataset(size)
        except ValueError:
            print("Invalid input, using basic dataset...")
            df, interval_labels = create_sample_training_data()
    elif choice == "4":
        print("\n🏃‍♂️ Creating training dataset from real FIT files...")
        df, interval_labels = create_training_data_from_fits()
    else:
        print("Invalid choice, using basic dataset...")
    df, interval_labels = create_sample_training_data()
    
    if df is None or interval_labels is None:
        print("No valid data loaded. Exiting.")
        return
    
    # Check if we have enough intervals for training
    if len(interval_labels) < 3:
        print(f"⚠️ Warning: Only {len(interval_labels)} intervals found. Need at least 3 for training.")
        print("💡 Consider:")
        print("   - Adding more FIT files with lap data")
        print("   - Using synthetic data first (option 1-3)")
        print("   - Checking if your FIT files have proper lap markers")
        return
    
    # Initialize detector
    detector = IntervalDetector(model_type='random_forest')
    
    # Create features
    df_features = detector.create_features(df)
    
    # Create training data
    X, y = detector.create_training_data(df_features, interval_labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print(f"🏷️ Training intervals: {np.sum(y_train)}")
    print(f"🏷️ Test intervals: {np.sum(y_test)}")
    
    # Visualize sample workouts
    visualize_workout_samples(df, interval_labels, num_samples=3)
    
    # Train model
    detector.train_model(X_train, y_train, use_cross_validation=True)
    
    # Evaluate model
    y_pred = detector.evaluate_model(X_test, y_test)
    
    # Predict intervals on the full dataset to see how well it learned
    print("\n🔮 Testing model on full dataset...")
    full_intervals = detector.predict_intervals(df)
    
    print(f"📊 Model found {len(full_intervals)} intervals in your training data")
    for i, (start, end) in enumerate(full_intervals):
        duration = (end - start).total_seconds()
        start_min = (start - df.index[0]).total_seconds() / 60
        end_min = (end - df.index[0]).total_seconds() / 60
        print(f"   Interval {i+1}: {start_min:.1f} - {end_min:.1f} min ({duration:.1f}s)")
    
    # Compare with actual intervals
    print(f"\n📊 Actual intervals from your FIT file:")
    for i, (start, end) in enumerate(interval_labels):
        duration = (end - start).total_seconds()
        start_min = (start - df.index[0]).total_seconds() / 60
        end_min = (end - df.index[0]).total_seconds() / 60
        print(f"   Actual {i+1}: {start_min:.1f} - {end_min:.1f} min ({duration:.1f}s)")
    
    # Visualize how well the model learned
    visualize_model_predictions(df, interval_labels, full_intervals)
    
    # Save model
    detector.save_model('interval_detector_model.pkl')
    
    print("\n🎉 Training complete!")
    print("💡 Next steps:")
    print("   1. Use the model to detect intervals in new rides")
    print("   2. Test on real FIT files")
    print("   3. Fine-tune with more real data")
    
    # Ask if user wants to test on a new ride
    test_choice = input("\n🧪 Test model on a new FIT file? (y/n): ").strip().lower()
    if test_choice == 'y':
        fit_path = input("Enter path to FIT file: ").strip()
        if fit_path:
            test_model_on_new_ride('interval_detector_model.pkl', fit_path)


if __name__ == "__main__":
    main()
