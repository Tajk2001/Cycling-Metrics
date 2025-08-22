#!/usr/bin/env python3
"""
Interval Detection for Cycling Training
Uses trained ML model with user-provided FTP for interval detection
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from SprintV1 import load_fit_to_dataframe
from IntervalML import IntervalDetector

def find_intervals_ml(df, ftp, min_duration=15, gap_tolerance=10):
    """Find intervals using trained ML model (FTP only for feature creation)"""
    print("   🤖 Using trained ML model for interval detection...")
    
    # Initialize detector and create features using user's FTP
    detector = IntervalDetector()
    
    # Apply 30-second power smoothing before feature creation
    print(f"   🔧 Applying 30-second power smoothing...")
    df_smoothed = df.copy()
    
    # Apply 30-second rolling average to power (standard in cycling analysis)
    window_size = 30  # 30 seconds
    df_smoothed['power_30s'] = df_smoothed['power'].rolling(window=window_size, min_periods=1, center=True).mean()
    
    # Replace original power with smoothed power for more stable analysis
    original_power_std = df_smoothed['power'].std()
    smoothed_power_std = df_smoothed['power_30s'].std()
    
    print(f"      Original power std: {original_power_std:.1f}W")
    print(f"      Smoothed power std: {smoothed_power_std:.1f}W")
    print(f"      Noise reduction: {((original_power_std - smoothed_power_std) / original_power_std * 100):.1f}%")
    
    # Use smoothed power for analysis
    df_smoothed['power'] = df_smoothed['power_30s']
    
    # Create features using the smoothed data
    features = detector.create_features(df_smoothed, estimated_ftp=ftp)
    
    # Load the trained model
    model_path = "trained_models/simple_comprehensive_model_20250820_125836.pkl"
    if not os.path.exists(model_path):
        print("   ❌ Trained model not found. Please ensure the model file exists.")
        return []
    
    # Load and use the model
    model = detector.load_model(model_path)
    if model is None:
        print("   ❌ Failed to load model - cannot proceed without ML model")
        return [], [], df
    else:
        # Get numeric features only (exclude categorical columns)
        exclude_cols = ['power_zone', 'file_path', 'file_name']
        feature_cols = [col for col in features.columns if col not in exclude_cols and not col.startswith('elapsed_time')]
        
        # Ensure all feature columns are numeric
        features_numeric = features[feature_cols].copy()
        for col in features_numeric.columns:
            features_numeric[col] = pd.to_numeric(features_numeric[col], errors='coerce').fillna(0)
        
        # Convert features to numpy array
        X_features = features_numeric.values
        
        # Ensure we have the right number of features for the trained model
        if X_features.shape[1] != len(detector.feature_names):
            # Truncate or pad to match model expectations
            expected_features = len(detector.feature_names)
            if X_features.shape[1] > expected_features:
                X_features = X_features[:, :expected_features]
            else:
                # Pad with zeros if we have fewer features
                padding = np.zeros((X_features.shape[0], expected_features - X_features.shape[1]))
                X_features = np.hstack([X_features, padding])
        
        # Scale the features
        X_scaled = detector.scaler.transform(X_features)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of being an interval
        
        # Debug: Show prediction statistics
        print(f"   📊 ML Model Statistics:")
        print(f"      Predictions: {predictions.sum()} out of {len(predictions)} points flagged as intervals")
        print(f"      Max probability: {probabilities.max():.3f}")
        print(f"      Min probability: {probabilities.min():.3f}")
        print(f"      Mean probability: {probabilities.mean():.3f}")
        print(f"      Points above 0.3: {(probabilities > 0.3).sum()}")
        print(f"      Points above 0.5: {(probabilities > 0.5).sum()}")
        print(f"      Points above 0.7: {(probabilities > 0.7).sum()}")
        
        # Apply RMS smoothing and amplification to ML probabilities
        print(f"   🔧 Applying RMS smoothing and amplification...")
        
        # Calculate RMS over a rolling window (e.g., 10 seconds)
        window_size = 10  # 10-second window for smoothing
        rms_probabilities = np.zeros_like(probabilities)
        
        for i in range(len(probabilities)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(probabilities), i + window_size // 2 + 1)
            window_probs = probabilities[start_idx:end_idx]
            rms_probabilities[i] = np.sqrt(np.mean(window_probs ** 2))
        
        # Amplify the smoothed probabilities
        amplification_factor = 2.0  # Increase sensitivity
        amplified_probs = rms_probabilities * amplification_factor
        
        # Clip to [0, 1] range
        amplified_probs = np.clip(amplified_probs, 0, 1)
        
        print(f"   📊 After RMS smoothing and amplification:")
        print(f"      Max amplified probability: {amplified_probs.max():.3f}")
        print(f"      Mean amplified probability: {amplified_probs.mean():.3f}")
        print(f"      Points above 0.3: {(amplified_probs > 0.3).sum()}")
        print(f"      Points above 0.5: {(amplified_probs > 0.5).sum()}")
        print(f"      Points above 0.7: {(amplified_probs > 0.7).sum()}")
        
        # Apply smoothing to ML confidence for better visualization and predictions
        # Smooth the original probabilities (not amplified) using rolling average
        smoothed_probs = pd.Series(probabilities).rolling(window=10, min_periods=1, center=True).mean().values
        
        # Use smoothed probabilities for predictions (what's shown in graph)
        predictions = (smoothed_probs > 0.3).astype(int)
        
        # Use smoothed probabilities for visualization
        probabilities = smoothed_probs
    
    # Find intervals from predictions
    intervals = []
    in_interval = False
    start_idx = last_idx = 0
    
    for i, pred in enumerate(predictions):
        if pred == 1 and not in_interval:
            # Start of new interval
            in_interval = True
            start_idx = last_idx = i
        elif pred == 1 and in_interval:
            # Continue interval
            last_idx = i
        elif pred == 0 and in_interval:
            gap_size = i - last_idx
            
            if gap_size <= gap_tolerance:
                # Small gap, continue interval
                continue
            else:
                # Gap too large, end interval
                duration = last_idx - start_idx + 1
                if duration >= min_duration:
                    start_time = df_smoothed.index[start_idx]
                    end_time = df_smoothed.index[last_idx]
                    intervals.append((start_time, end_time, duration))
                in_interval = False
    
    # Handle case where interval continues to end of data
    if in_interval:
        duration = last_idx - start_idx + 1
        if duration >= min_duration:
            start_time = df_smoothed.index[start_idx]
            end_time = df_smoothed.index[last_idx]
            intervals.append((start_time, end_time, duration))
    
    # Expand interval boundaries using average power optimization
    print(f"   🔧 Optimizing interval boundaries using average power...")
    expanded_intervals = []
    
    for start_time, end_time, duration in intervals:
        start_idx = df.index.get_loc(start_time)
        end_idx = df.index.get_loc(end_time)
        
        # Calculate current average power of the detected interval
        current_avg_power = df['power'].iloc[start_idx:end_idx + 1].mean()
        
        # Try expanding backwards in 5-10 second increments
        best_start_idx = start_idx
        best_avg_power = current_avg_power
        
        # Look backwards up to 30 seconds in 5-10 second increments
        for expansion_size in [5, 10, 15, 20, 25, 30]:
            test_start_idx = max(0, start_idx - expansion_size)
            if test_start_idx < start_idx:  # Only test if we actually expand
                test_avg_power = df['power'].iloc[test_start_idx:end_idx + 1].mean()
                
                if test_avg_power > best_avg_power:
                    best_start_idx = test_start_idx
                    best_avg_power = test_avg_power
                    print(f"      ⬅️  Expanding start by {expansion_size}s improves avg power: {current_avg_power:.1f}W → {test_avg_power:.1f}W")
                else:
                    break  # Stop expanding if average power decreases
        
        # Try expanding forwards in 5-10 second increments
        best_end_idx = end_idx
        current_best_with_start = df['power'].iloc[best_start_idx:end_idx + 1].mean()
        
        # Look forwards up to 30 seconds in 5-10 second increments
        for expansion_size in [5, 10, 15, 20, 25, 30]:
            test_end_idx = min(len(df) - 1, end_idx + expansion_size)
            if test_end_idx > end_idx:  # Only test if we actually expand
                test_avg_power = df['power'].iloc[best_start_idx:test_end_idx + 1].mean()
                
                if test_avg_power > current_best_with_start:
                    best_end_idx = test_end_idx
                    current_best_with_start = test_avg_power
                    print(f"      ➡️  Expanding end by {expansion_size}s improves avg power: {best_avg_power:.1f}W → {test_avg_power:.1f}W")
                else:
                    break  # Stop expanding if average power decreases
        
        # Create final optimized interval
        optimized_start_time = df.index[best_start_idx]
        optimized_end_time = df.index[best_end_idx]
        optimized_duration = int((optimized_end_time - optimized_start_time).total_seconds())
        final_avg_power = df['power'].iloc[best_start_idx:best_end_idx + 1].mean()
        
        # Check if optimization made a meaningful improvement
        expansion_diff = optimized_duration - duration
        power_improvement = final_avg_power - current_avg_power
        
        if expansion_diff > 3 or power_improvement > 2:  # Meaningful expansion or power improvement
            print(f"      ✅ Optimized interval: +{expansion_diff}s, +{power_improvement:.1f}W avg power")
            print(f"         {start_time.strftime('%H:%M:%S')} → {optimized_start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')} → {optimized_end_time.strftime('%H:%M:%S')}")
            expanded_intervals.append((optimized_start_time, optimized_end_time, optimized_duration))
        else:
            expanded_intervals.append((start_time, end_time, duration))
    
    print(f"   ✅ Power optimization complete: {len(expanded_intervals)} intervals")
    
    # Return expanded intervals, probabilities, and original df for visualization
    return expanded_intervals, probabilities, df

def create_visualization(df, intervals, ftp, probabilities=None, save_plot=True):
    """Create visualization of power data, detected intervals, and ML confidence"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('ML-Based Interval Detection with User FTP', fontsize=16, fontweight='bold')
    
    # Time axis for x-axis
    time_axis = np.arange(len(df)) / 60  # Convert to minutes
    
    # Plot 1: Power data with detected intervals
    ax1.plot(time_axis, df['power'], color='blue', alpha=0.7, linewidth=1, label='Power')
    
    # Add FTP reference lines
    ax1.axhline(y=ftp, color='green', linestyle='--', alpha=0.7, label=f'FTP: {ftp:.0f}W')
    ax1.axhline(y=ftp * 1.1, color='orange', linestyle='--', alpha=0.7, label=f'110% FTP: {ftp*1.1:.0f}W')
    ax1.axhline(y=ftp * 1.2, color='red', linestyle='--', alpha=0.7, label=f'120% FTP: {ftp*1.2:.0f}W')
    
    # Highlight intervals
    for start_time, end_time, duration in intervals:
        start_idx = df.index.get_loc(start_time)
        end_idx = df.index.get_loc(end_time)
        interval_times = np.arange(start_idx, end_idx + 1) / 60
        interval_powers = df['power'].iloc[start_idx:end_idx + 1]
        ax1.plot(interval_times, interval_powers, color='red', linewidth=3, alpha=0.8)
    
    ax1.set_ylabel('Power (W)', fontsize=12)
    ax1.set_title('Power Profile with ML-Detected Intervals', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: ML confidence scores
    if probabilities is not None:
        ax2.plot(time_axis, probabilities, color='purple', alpha=0.7, linewidth=1, label='ML Confidence')
        ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='30% Threshold')
        ax2.set_ylabel('ML Confidence Score', fontsize=12)
        ax2.set_title('Machine Learning Model Confidence', fontsize=14, fontweight='bold')
    else:
        # Fallback: Power distribution analysis (no FTP dependency)
        power_percentile = df['power'].rank(pct=True) * 100
        ax2.plot(time_axis, power_percentile, color='purple', alpha=0.7, linewidth=1, label='Power Percentile')
        ax2.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80th Percentile')
        ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90th Percentile')
        ax2.set_ylabel('Power Percentile (%)', fontsize=12)
        ax2.set_title('Power Distribution (Percentile)', fontsize=14, fontweight='bold')
    
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ml_interval_detection_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   📊 Saved: {filename}")
        print("   ✅ Visualization created successfully!")
    
    plt.show()

def detect_intervals_ml_simple(file_path, ftp, save_plot=True):
    """Hybrid interval detection: Lap-based first, then ML-based fallback"""
    print("🧪 Hybrid Interval Detection (Laps + ML Fallback)")
    print("=" * 50)
    print(f"📁 File: {file_path}")
    print(f"🎯 Strategy: Lap detection first, then ML model if needed")
    
    # Load FIT file
    print("\n📥 Loading FIT file...")
    try:
        df = load_fit_to_dataframe(file_path)
        if df is None or len(df) == 0:
            print("❌ Failed to load FIT file or file is empty")
            return None
        
        print(f"✅ Loaded {len(df)} data points")
        
    except Exception as e:
        print(f"❌ Error loading FIT file: {e}")
        return None
    
    # Calculate power statistics
    ride_avg_power = df['power'].mean()
    ride_max_power = df['power'].max()
    normalized_power = df['power'].rolling(window=30, min_periods=1).mean() ** 4
    normalized_power = normalized_power.mean() ** (1/4)
    
    # Calculate intensity relative to ride's own power distribution (no FTP dependency)
    power_95th = df['power'].quantile(0.95)
    intensity_ratio = normalized_power / power_95th if power_95th > 0 else 0
    
    print(f"   📊 Ride average: {ride_avg_power:.0f}W")
    print(f"   📊 Normalized Power: {normalized_power:.0f}W")
    print(f"   📊 Intensity Ratio (vs 95th percentile): {intensity_ratio:.2f}")
    
    # Try lap-based detection first (using your training logic)
    print(f"\n🎯 Step 1: Attempting lap-based interval detection...")
    from IntervalML import extract_lap_intervals_from_data
    
    lap_intervals = []
    try:
        lap_intervals = extract_lap_intervals_from_data(df)
        print(f"   ✅ Found {len(lap_intervals)} intervals from laps")
        
        # Convert lap intervals to our format (start_time, end_time, duration)
        formatted_lap_intervals = []
        for start_time, end_time in lap_intervals:
            duration = int((end_time - start_time).total_seconds())
            formatted_lap_intervals.append((start_time, end_time, duration))
        
        lap_intervals = formatted_lap_intervals
        
    except Exception as e:
        print(f"   ⚠️  Lap detection failed: {e}")
        lap_intervals = []
    
    # If we have good lap intervals, use them; otherwise fall back to ML
    if len(lap_intervals) >= 2:  # Need at least 2 intervals to be useful
        print(f"   🎯 Using {len(lap_intervals)} lap-based intervals")
        intervals = lap_intervals
        probabilities = None  # No ML probabilities for lap intervals
        original_df = df
        detection_method = "Lap-based"
    else:
        print(f"   🔄 Insufficient lap intervals ({len(lap_intervals)}), falling back to ML model...")
        intervals, probabilities, original_df = find_intervals_ml(df, ftp, min_duration=15, gap_tolerance=10)
        detection_method = "ML-based"
    
    # Display results
    if intervals:
        print(f"   ✅ Found {len(intervals)} intervals using {detection_method} detection")
        total_time = sum(duration for _, _, duration in intervals)
        print(f"   ⏱️  Total interval time: {total_time/60:.1f} minutes")
        
        # Save intervals to file for future analysis (DISABLED - using dashboard instead)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # intervals_filename = f"detected_intervals_{timestamp}.txt"
        
        # with open(intervals_filename, 'w') as f:  # File generation disabled
            f.write(f"File: {file_path}\n")
            f.write(f"Detection Method: {detection_method}\n")
            f.write(f"FTP: {ftp}W\n")
            f.write(f"Total Intervals: {len(intervals)}\n")
            f.write(f"Total Time: {total_time/60:.1f} minutes\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")
            
            for i, (start_time, end_time, duration) in enumerate(intervals, 1):
                start_idx = original_df.index.get_loc(start_time)
                end_idx = original_df.index.get_loc(end_time)
                interval_power = original_df['power'].iloc[start_idx:end_idx + 1]
                avg_power = interval_power.mean()
                max_power = interval_power.max()
                
                f.write(f"Interval {i}: {start_time} - {end_time} ({duration}s)\n")
                f.write(f"  Power: {avg_power:.0f}W avg, {max_power:.0f}W max\n")
                if probabilities is not None:
                    interval_confidence = probabilities[start_idx:end_idx + 1].mean()
                    f.write(f"  ML Confidence: {interval_confidence:.3f}\n")
                f.write("\n")
        
        # print(f"   💾 Saved intervals to: {intervals_filename}")  # File generation disabled
        
        print(f"\n📋 Interval Details:")
        for i, (start_time, end_time, duration) in enumerate(intervals, 1):
            start_idx = original_df.index.get_loc(start_time)
            end_idx = original_df.index.get_loc(end_time)
            interval_power = original_df['power'].iloc[start_idx:end_idx + 1]
            avg_power = interval_power.mean()
            max_power = interval_power.max()
            
            # Calculate power relative to ride's own distribution (no FTP dependency)
            ride_power_95th = original_df['power'].quantile(0.95)
            power_ratio = avg_power / ride_power_95th if ride_power_95th > 0 else 0
            
            # Get average ML confidence for this interval
            if probabilities is not None:
                interval_confidence = probabilities[start_idx:end_idx + 1].mean()
                confidence_str = f", ML Confidence: {interval_confidence:.2f}"
            else:
                confidence_str = ""
            
            # Format duration
            if duration < 60:
                duration_str = f"{duration}s"
            else:
                duration_str = f"{duration/60:.1f}min"
            
            print(f"   {i:2d}. {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')} ({duration_str})")
            print(f"       Power: {avg_power:.0f}W avg ({power_ratio:.1f}x 95th percentile), {max_power:.0f}W max{confidence_str}")
        
        # Create visualization
        if save_plot:
            print(f"\n📊 Creating visualization...")
            create_visualization(original_df, intervals, ftp, probabilities, save_plot=True)
        
        return intervals
    else:
        print("   ❌ No intervals detected")
        return []

if __name__ == "__main__":
    # Get FTP from user input
    try:
        ftp_input = input("Enter your FTP in watts: ")
        ftp = float(ftp_input)
        print(f"Using FTP: {ftp:.0f}W")
    except ValueError:
        print("Invalid input. Using default FTP of 250W")
        ftp = 250.0
    
    # Get file path from user input
    file_path = input("Enter the path to your FIT file: ")
    
    if os.path.exists(file_path):
        detect_intervals_ml_simple(file_path, ftp)
    else:
        print("File not found. Please check the path.")
