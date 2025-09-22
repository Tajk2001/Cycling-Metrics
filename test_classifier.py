#!/usr/bin/env python3
"""
Test script for the workout classifier.
This script demonstrates how to use the classifier with sample data.
"""

import os
import sys
from workout_classifier import WorkoutClassifier

def test_with_sample_data():
    """Test the classifier with some sample FIT files from the athletes folder."""
    
    print("🧪 Testing Workout Classifier")
    print("=" * 40)
    
    # Initialize classifier
    classifier = WorkoutClassifier(ftp=250.0)
    
    # Look for sample FIT files in the athletes folder
    sample_files = []
    
    # Check athletes folders for FIT files
    athletes_dir = "/Users/tajkrieger/Projects/cycling_analysis/athletes"
    if os.path.exists(athletes_dir):
        for athlete in os.listdir(athletes_dir):
            athlete_path = os.path.join(athletes_dir, athlete)
            if os.path.isdir(athlete_path):
                # Search for FIT files recursively
                for root, dirs, files in os.walk(athlete_path):
                    for file in files:
                        if file.endswith('.fit'):
                            sample_files.append(os.path.join(root, file))
                            if len(sample_files) >= 3:  # Limit to 3 files for testing
                                break
                    if len(sample_files) >= 3:
                        break
            if len(sample_files) >= 3:
                break
    
    if not sample_files:
        print("❌ No FIT files found in athletes folder")
        print("💡 To test the classifier:")
        print("   1. Add some FIT files to the athletes folder")
        print("   2. Or use the classifier directly on your own data:")
        print("      python classify_workouts.py /path/to/your/workouts --ftp 280")
        return
    
    print(f"📁 Found {len(sample_files)} sample FIT files")
    print()
    
    # Process each file
    results = []
    for i, file_path in enumerate(sample_files, 1):
        print(f"[{i}/{len(sample_files)}] Testing: {os.path.basename(file_path)}")
        result = classifier.process_single_file(file_path)
        results.append(result)
        print()
    
    # Generate summary
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 30)
    print(f"Files tested: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if successful_results:
        print(f"\nWorkout classifications:")
        for result in successful_results:
            print(f"  {result['file_name']}: {result['workout_type']} (confidence: {result['confidence']:.2f})")
    
    if failed_results:
        print(f"\nFailed files:")
        for result in failed_results:
            print(f"  {result['file_path']}: {result['error']}")
    
    print(f"\n✅ Test completed!")
    print(f"💡 To process a full folder of workouts, use:")
    print(f"   python classify_workouts.py /path/to/your/workouts --ftp 280 --recursive")

def test_classification_logic():
    """Test the classification logic with mock data."""
    
    print("\n🧪 Testing Classification Logic")
    print("=" * 35)
    
    classifier = WorkoutClassifier(ftp=250.0)
    
    # Create mock ride data
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Mock data for different workout types
    test_cases = [
        {
            'name': 'Recovery Ride',
            'avg_power': 150,  # 60% FTP
            'max_power': 200,
            'normalized_power': 155,
            'duration_min': 60,
            'intervals': []
        },
        {
            'name': 'Tempo Ride',
            'avg_power': 225,  # 90% FTP
            'max_power': 280,
            'normalized_power': 230,
            'duration_min': 45,
            'intervals': [(10, 20, 600), (25, 35, 600)]  # 2x10min intervals
        },
        {
            'name': 'VO2 Max Ride',
            'avg_power': 290,  # 116% FTP
            'max_power': 400,
            'normalized_power': 295,
            'duration_min': 30,
            'intervals': [(5, 8, 180), (12, 15, 180), (18, 21, 180)]  # 3x3min intervals
        }
    ]
    
    # Create mock DataFrame for each test case
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['name']}")
        
        # Create mock DataFrame with 1-second intervals
        duration_seconds = int(test_case['duration_min'] * 60)
        timestamps = pd.date_range(start='2025-01-01 06:00:00', periods=duration_seconds, freq='1S')
        
        # Create power data with some variation
        base_power = test_case['avg_power']
        power_data = np.random.normal(base_power, base_power * 0.1, duration_seconds)
        power_data = np.clip(power_data, 0, test_case['max_power'])
        
        df = pd.DataFrame({
            'power': power_data,
            'cadence': np.random.normal(85, 10, duration_seconds)
        }, index=timestamps)
        
        # Create ride metrics
        ride_metrics = {
            'duration_min': test_case['duration_min'],
            'avg_power': test_case['avg_power'],
            'max_power': test_case['max_power'],
            'normalized_power': test_case['normalized_power']
        }
        
        # Convert intervals to proper format (start_time, end_time, duration)
        formatted_intervals = []
        for start_min, end_min, duration_s in test_case['intervals']:
            start_time = timestamps[start_min * 60]
            end_time = timestamps[end_min * 60]
            formatted_intervals.append((start_time, end_time, duration_s))
        
        # Classify
        category, label, confidence = classifier.classify_workout_type(df, formatted_intervals, ride_metrics)
        
        print(f"   Expected: {test_case['name']}")
        print(f"   Classified as: {category}")
        print(f"   Label: {label}")
        print(f"   Confidence: {confidence:.2f}")
        
        # Check if classification makes sense
        expected_lower = test_case['name'].lower()
        if expected_lower in category or category in expected_lower:
            print(f"   ✅ Classification looks good!")
        else:
            print(f"   ⚠️ Classification may need adjustment")

if __name__ == "__main__":
    # Run tests
    test_classification_logic()
    test_with_sample_data()
