#!/usr/bin/env python3
"""
Simple CLI script for classifying workouts in a folder.

Usage examples:
    python classify_workouts.py /path/to/workouts --ftp 280
    python classify_workouts.py /path/to/workouts --recursive --output results/
"""

import sys
import os
from workout_classifier import WorkoutClassifier

def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_workouts.py <folder_path> [options]")
        print("\nOptions:")
        print("  --ftp <value>        Your FTP in watts (default: 250)")
        print("  --recursive          Search subdirectories")
        print("  --output <dir>       Output directory (default: workout_classification_results)")
        print("  --no-viz             Skip visualization generation")
        print("\nExample:")
        print("  python classify_workouts.py /Users/john/cycling_data --ftp 280 --recursive")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    # Parse arguments
    ftp = 250.0
    recursive = False
    output_dir = "workout_classification_results"
    no_viz = False
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--ftp" and i + 1 < len(sys.argv):
            ftp = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--recursive":
            recursive = True
            i += 1
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--no-viz":
            no_viz = True
            i += 1
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    print(f"🚴 Workout Classification Tool")
    print(f"📁 Processing folder: {folder_path}")
    print(f"🎯 FTP: {ftp}W")
    print(f"🔍 Recursive: {recursive}")
    print(f"📊 Output: {output_dir}")
    print()
    
    # Initialize classifier
    classifier = WorkoutClassifier(ftp=ftp)
    
    # Process folder
    results = classifier.process_folder(folder_path, recursive=recursive)
    
    if not results:
        print("No FIT files found or processed.")
        return
    
    # Export results
    json_path, csv_path, summary_path = classifier.export_results(results, output_dir)
    
    # Create visualizations
    if not no_viz:
        classifier.create_visualization(results, output_dir)
    
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
    
    print(f"\n✅ Classification complete! Check the '{output_dir}' folder for detailed results.")

if __name__ == "__main__":
    main()
