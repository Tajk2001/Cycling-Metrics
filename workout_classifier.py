#!/usr/bin/env python3
"""
Workout Classifier - Batch Processing Tool

This script processes entire folders of cycling workouts (FIT files) and determines
the type of workout for each ride using machine learning and rule-based analysis.

Features:
- Batch processing of FIT files
- Multiple classification methods (ML + rule-based)
- Workout type categorization (VO2, Threshold, Tempo, etc.)
- Detailed reporting and statistics
- Export results to CSV/JSON

Author: AI Assistant
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse

warnings.filterwarnings('ignore')

# Import our existing modules
from SprintV1 import load_fit_to_dataframe
from IntervalML import IntervalDetector, extract_lap_intervals_from_data
from interval_detection import detect_intervals_ml_simple

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class WorkoutClassifier:
    """
    A comprehensive workout classifier that can process multiple FIT files
    and determine workout types using multiple analysis methods.
    """
    
    def __init__(self, ftp: float = 250.0, model_path: Optional[str] = None):
        """
        Initialize the workout classifier.
        
        Args:
            ftp (float): Functional Threshold Power for analysis
            model_path (str, optional): Path to trained ML model
        """
        self.ftp = ftp
        self.model_path = model_path or "trained_models/simple_comprehensive_model_20250820_125836.pkl"
        self.detector = IntervalDetector()
        self.results = []
        self.workout_types = {
            'recovery': {'power_threshold': 0.65, 'duration_min': 30, 'max_intervals': 2},
            'z2': {'power_threshold': 0.75, 'duration_min': 45, 'max_intervals': 3},
            'tempo': {'power_threshold': 0.85, 'duration_min': 20, 'interval_duration': 300},
            'thr': {'power_threshold': 0.95, 'duration_min': 15, 'interval_duration': 600},
            'vo2-short': {'power_threshold': 1.15, 'duration_min': 30, 'interval_duration': 180},
            'vo2-long': {'power_threshold': 1.10, 'duration_min': 60, 'interval_duration': 300},
            'anaerobic': {'power_threshold': 1.25, 'duration_min': 20, 'interval_duration': 60},
            'sprint': {'power_threshold': 1.40, 'duration_min': 10, 'interval_duration': 30},
            'mixed': {'description': 'Multiple workout types detected'},
            'endurance': {'power_threshold': 0.70, 'duration_min': 120, 'max_intervals': 5},
            'uncategorized': {'description': 'Could not determine workout type'}
        }
    
    def classify_workout_type(self, df: pd.DataFrame, intervals: List[Tuple], 
                            ride_metrics: Dict[str, Any]) -> Tuple[str, str, float]:
        """
        Classify workout type based on power analysis and intervals.
        
        Args:
            df: Power data DataFrame
            intervals: List of detected intervals
            ride_metrics: Basic ride metrics
            
        Returns:
            Tuple of (category, label, confidence)
        """
        # Calculate power metrics
        avg_power = ride_metrics['avg_power']
        max_power = ride_metrics['max_power']
        normalized_power = ride_metrics['normalized_power']
        duration_min = ride_metrics['duration_min']
        
        # Calculate power relative to FTP
        ftp_ratio = avg_power / self.ftp if self.ftp > 0 else 0
        max_ftp_ratio = max_power / self.ftp if self.ftp > 0 else 0
        
        # Analyze intervals
        interval_analysis = self._analyze_intervals(df, intervals)
        
        # Classification logic
        classification_score = {}
        
        # Recovery rides
        if ftp_ratio < 0.65 and duration_min > 30 and len(intervals) <= 2:
            classification_score['recovery'] = 0.9
        
        # Endurance rides (Z2)
        if 0.65 <= ftp_ratio < 0.75 and duration_min > 45 and len(intervals) <= 3:
            classification_score['z2'] = 0.85
        
        # Tempo rides
        if 0.75 <= ftp_ratio < 0.90 and interval_analysis['avg_interval_duration'] > 300:
            classification_score['tempo'] = 0.8
        
        # Threshold rides
        if 0.90 <= ftp_ratio < 1.05 and interval_analysis['avg_interval_duration'] > 300:
            classification_score['thr'] = 0.85
        
        # VO2 Max rides
        if ftp_ratio >= 1.05:
            if interval_analysis['avg_interval_duration'] < 300:
                classification_score['vo2-short'] = 0.9
            else:
                classification_score['vo2-long'] = 0.85
        
        # Anaerobic rides
        if max_ftp_ratio > 1.25 and interval_analysis['avg_interval_duration'] < 120:
            classification_score['anaerobic'] = 0.8
        
        # Sprint rides
        if max_ftp_ratio > 1.40 and interval_analysis['avg_interval_duration'] < 60:
            classification_score['sprint'] = 0.9
        
        # Mixed workouts (multiple interval types)
        if len(set(interval_analysis['power_zones'])) > 2:
            classification_score['mixed'] = 0.7
        
        # Long endurance rides
        if duration_min > 120 and ftp_ratio < 0.75 and len(intervals) <= 5:
            classification_score['endurance'] = 0.8
        
        # If no clear classification, use uncategorized
        if not classification_score:
            classification_score['uncategorized'] = 0.5
        
        # Get the best classification
        best_category = max(classification_score, key=classification_score.get)
        confidence = classification_score[best_category]
        
        # Create descriptive label
        label = self._create_workout_label(best_category, interval_analysis, ride_metrics)
        
        return best_category, label, confidence
    
    def _analyze_intervals(self, df: pd.DataFrame, intervals: List[Tuple]) -> Dict[str, Any]:
        """Analyze interval characteristics for classification."""
        if not intervals:
            return {
                'count': 0,
                'total_duration': 0,
                'avg_interval_duration': 0,
                'avg_interval_power': 0,
                'power_zones': [],
                'interval_pattern': 'none'
            }
        
        interval_durations = []
        interval_powers = []
        power_zones = []
        
        for start_time, end_time, duration in intervals:
            start_idx = df.index.get_loc(start_time)
            end_idx = df.index.get_loc(end_time)
            interval_power = df['power'].iloc[start_idx:end_idx + 1].mean()
            
            interval_durations.append(duration)
            interval_powers.append(interval_power)
            
            # Determine power zone
            ftp_ratio = interval_power / self.ftp if self.ftp > 0 else 0
            if ftp_ratio < 0.55:
                power_zones.append('Zone 1')
            elif ftp_ratio < 0.75:
                power_zones.append('Zone 2')
            elif ftp_ratio < 0.90:
                power_zones.append('Zone 3')
            elif ftp_ratio < 1.05:
                power_zones.append('Zone 4')
            elif ftp_ratio < 1.20:
                power_zones.append('Zone 5')
            elif ftp_ratio < 1.50:
                power_zones.append('Zone 6')
            else:
                power_zones.append('Zone 7')
        
        # Analyze interval pattern
        pattern = 'none'
        if len(intervals) > 3:
            durations = np.array(interval_durations)
            if np.std(durations) < np.mean(durations) * 0.2:
                pattern = 'uniform'
            elif len(intervals) > 6:
                pattern = 'high_volume'
        
        return {
            'count': len(intervals),
            'total_duration': sum(interval_durations),
            'avg_interval_duration': np.mean(interval_durations) if interval_durations else 0,
            'avg_interval_power': np.mean(interval_powers) if interval_powers else 0,
            'power_zones': power_zones,
            'interval_pattern': pattern
        }
    
    def _create_workout_label(self, category: str, interval_analysis: Dict, 
                            ride_metrics: Dict[str, Any]) -> str:
        """Create a descriptive label for the workout."""
        duration_min = ride_metrics['duration_min']
        interval_count = interval_analysis['count']
        
        if category == 'recovery':
            return f"Recovery Ride ({duration_min:.0f}min)"
        elif category == 'z2':
            return f"Z2 Endurance ({duration_min:.0f}min)"
        elif category == 'tempo':
            avg_duration = interval_analysis['avg_interval_duration']
            return f"Tempo ({interval_count}x{avg_duration/60:.0f}min)"
        elif category == 'thr':
            avg_duration = interval_analysis['avg_interval_duration']
            return f"Threshold ({interval_count}x{avg_duration/60:.0f}min)"
        elif category == 'vo2-short':
            return f"VO2 Max Short ({interval_count} intervals)"
        elif category == 'vo2-long':
            return f"VO2 Max Long ({interval_count} intervals)"
        elif category == 'anaerobic':
            return f"Anaerobic ({interval_count} intervals)"
        elif category == 'sprint':
            return f"Sprint ({interval_count} efforts)"
        elif category == 'mixed':
            return f"Mixed Workout ({interval_count} intervals)"
        elif category == 'endurance':
            return f"Long Endurance ({duration_min:.0f}min)"
        else:
            return f"Uncategorized ({duration_min:.0f}min)"
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single FIT file and classify the workout.
        
        Args:
            file_path: Path to the FIT file
            
        Returns:
            Dictionary with classification results
        """
        print(f"\n📁 Processing: {os.path.basename(file_path)}")
        
        try:
            # Load FIT file
            df = load_fit_to_dataframe(file_path)
            if df is None or len(df) == 0:
                return {
                    'file_path': file_path,
                    'status': 'failed',
                    'error': 'Could not load FIT file',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate basic metrics
            duration_min = (df.index[-1] - df.index[0]).total_seconds() / 60
            avg_power = df['power'].mean()
            max_power = df['power'].max()
            
            # Calculate normalized power
            power_30s = df['power'].rolling(window=30, min_periods=1).mean()
            normalized_power = (power_30s ** 4).mean() ** (1/4)
            
            ride_metrics = {
                'duration_min': duration_min,
                'avg_power': avg_power,
                'max_power': max_power,
                'normalized_power': normalized_power
            }
            
            print(f"   📊 Duration: {duration_min:.1f}min, Avg Power: {avg_power:.0f}W")
            
            # Detect intervals using ML model
            print(f"   🔍 Detecting intervals...")
            intervals = detect_intervals_ml_simple(file_path, self.ftp, save_plot=False)
            
            if not intervals:
                intervals = []
                print(f"   ⚠️ No intervals detected")
            else:
                print(f"   ✅ Found {len(intervals)} intervals")
            
            # Classify workout type
            category, label, confidence = self.classify_workout_type(df, intervals, ride_metrics)
            
            print(f"   🎯 Classification: {category} (confidence: {confidence:.2f})")
            print(f"   🏷️ Label: {label}")
            
            # Prepare result
            result = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'workout_type': category,
                'workout_label': label,
                'confidence': confidence,
                'ride_metrics': ride_metrics,
                'interval_count': len(intervals),
                'intervals': [
                    {
                        'start_time': start.isoformat(),
                        'end_time': end.isoformat(),
                        'duration_s': duration
                    } for start, end, duration in intervals
                ]
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            return {
                'file_path': file_path,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def process_folder(self, folder_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Process all FIT files in a folder.
        
        Args:
            folder_path: Path to folder containing FIT files
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List of classification results
        """
        print(f"🚴 Processing workout folder: {folder_path}")
        print("=" * 60)
        
        # Find all FIT files
        if recursive:
            fit_files = glob.glob(os.path.join(folder_path, "**", "*.fit"), recursive=True)
        else:
            fit_files = glob.glob(os.path.join(folder_path, "*.fit"))
        
        if not fit_files:
            print(f"❌ No FIT files found in {folder_path}")
            return []
        
        print(f"📁 Found {len(fit_files)} FIT files")
        
        # Process each file
        results = []
        for i, file_path in enumerate(fit_files, 1):
            print(f"\n[{i}/{len(fit_files)}] Processing...")
            result = self.process_single_file(file_path)
            results.append(result)
            self.results.append(result)
        
        print(f"\n✅ Completed processing {len(fit_files)} files")
        return results
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary report of all processed workouts."""
        successful_results = [r for r in results if r['status'] == 'success']
        failed_results = [r for r in results if r['status'] == 'failed']
        
        # Workout type distribution
        workout_types = {}
        for result in successful_results:
            workout_type = result['workout_type']
            workout_types[workout_type] = workout_types.get(workout_type, 0) + 1
        
        # Calculate statistics
        total_duration = sum(r['ride_metrics']['duration_min'] for r in successful_results)
        avg_power = np.mean([r['ride_metrics']['avg_power'] for r in successful_results])
        total_intervals = sum(r['interval_count'] for r in successful_results)
        
        summary = {
            'processing_summary': {
                'total_files': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results) / len(results) if results else 0
            },
            'workout_statistics': {
                'total_duration_min': total_duration,
                'average_power': avg_power,
                'total_intervals': total_intervals,
                'workout_type_distribution': workout_types
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return summary
    
    def export_results(self, results: List[Dict[str, Any]], output_dir: str = "workout_classification_results"):
        """Export results to CSV and JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export detailed results to JSON
        json_path = os.path.join(output_dir, f"workout_classification_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Export summary to CSV
        csv_data = []
        for result in results:
            if result['status'] == 'success':
                csv_data.append({
                    'file_name': result['file_name'],
                    'workout_type': result['workout_type'],
                    'workout_label': result['workout_label'],
                    'confidence': result['confidence'],
                    'duration_min': result['ride_metrics']['duration_min'],
                    'avg_power': result['ride_metrics']['avg_power'],
                    'max_power': result['ride_metrics']['max_power'],
                    'interval_count': result['interval_count'],
                    'processed_at': result['timestamp']
                })
        
        if csv_data:
            df_results = pd.DataFrame(csv_data)
            csv_path = os.path.join(output_dir, f"workout_classification_{timestamp}.csv")
            df_results.to_csv(csv_path, index=False)
        
        # Generate and export summary report
        summary = self.generate_summary_report(results)
        summary_path = os.path.join(output_dir, f"workout_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📊 Results exported to {output_dir}/")
        print(f"   📄 Detailed results: {os.path.basename(json_path)}")
        if csv_data:
            print(f"   📊 Summary CSV: {os.path.basename(csv_path)}")
        print(f"   📋 Summary report: {os.path.basename(summary_path)}")
        
        return json_path, csv_path if csv_data else None, summary_path
    
    def create_visualization(self, results: List[Dict[str, Any]], output_dir: str = "workout_classification_results"):
        """Create visualizations of the classification results."""
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Workout Classification Analysis', fontsize=16, fontweight='bold')
        
        # 1. Workout type distribution
        workout_types = [r['workout_type'] for r in successful_results]
        type_counts = pd.Series(workout_types).value_counts()
        
        ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Workout Type Distribution')
        
        # 2. Duration vs Average Power scatter plot
        durations = [r['ride_metrics']['duration_min'] for r in successful_results]
        avg_powers = [r['ride_metrics']['avg_power'] for r in successful_results]
        colors = [r['confidence'] for r in successful_results]
        
        scatter = ax2.scatter(durations, avg_powers, c=colors, cmap='viridis', alpha=0.7)
        ax2.set_xlabel('Duration (minutes)')
        ax2.set_ylabel('Average Power (W)')
        ax2.set_title('Duration vs Average Power')
        plt.colorbar(scatter, ax=ax2, label='Confidence')
        
        # 3. Interval count distribution
        interval_counts = [r['interval_count'] for r in successful_results]
        ax3.hist(interval_counts, bins=min(20, len(set(interval_counts))), alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Number of Intervals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Interval Count Distribution')
        
        # 4. Confidence distribution by workout type
        confidence_by_type = {}
        for result in successful_results:
            workout_type = result['workout_type']
            confidence = result['confidence']
            if workout_type not in confidence_by_type:
                confidence_by_type[workout_type] = []
            confidence_by_type[workout_type].append(confidence)
        
        ax4.boxplot([confidence_by_type[wt] for wt in confidence_by_type.keys()], 
                   labels=list(confidence_by_type.keys()))
        ax4.set_ylabel('Confidence Score')
        ax4.set_title('Confidence Distribution by Workout Type')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"workout_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {os.path.basename(plot_path)}")
        
        plt.show()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Classify cycling workouts from FIT files')
    parser.add_argument('folder_path', help='Path to folder containing FIT files')
    parser.add_argument('--ftp', type=float, default=250.0, help='Functional Threshold Power (default: 250W)')
    parser.add_argument('--model', help='Path to trained ML model')
    parser.add_argument('--recursive', action='store_true', help='Search subdirectories recursively')
    parser.add_argument('--output', default='workout_classification_results', help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = WorkoutClassifier(ftp=args.ftp, model_path=args.model)
    
    # Process folder
    results = classifier.process_folder(args.folder_path, recursive=args.recursive)
    
    if not results:
        print("No files processed. Exiting.")
        return
    
    # Export results
    json_path, csv_path, summary_path = classifier.export_results(results, args.output)
    
    # Create visualizations
    if not args.no_viz:
        classifier.create_visualization(results, args.output)
    
    # Print summary
    summary = classifier.generate_summary_report(results)
    print(f"\n📋 SUMMARY REPORT")
    print("=" * 40)
    print(f"Files processed: {summary['processing_summary']['total_files']}")
    print(f"Successful: {summary['processing_summary']['successful']}")
    print(f"Failed: {summary['processing_summary']['failed']}")
    print(f"Success rate: {summary['processing_summary']['success_rate']:.1%}")
    print(f"\nTotal duration: {summary['workout_statistics']['total_duration_min']:.1f} minutes")
    print(f"Average power: {summary['workout_statistics']['average_power']:.0f}W")
    print(f"Total intervals: {summary['workout_statistics']['total_intervals']}")
    
    print(f"\nWorkout types found:")
    for workout_type, count in summary['workout_statistics']['workout_type_distribution'].items():
        print(f"  {workout_type}: {count}")


if __name__ == "__main__":
    main()
