#!/usr/bin/env python3
"""
Enhanced Cycling Analysis Runner
Implements the comprehensive data processing pipeline with logical flow:
1. 📥 DATA INGESTION
2. 🔍 INITIAL DATA CHECKS
3. 🧹 DATA CLEANING
4. 🧮 FEATURE ENGINEERING
5. 📊 DATA AGGREGATION
6. 📈 VISUALIZATION
7. 🧪 MODELING & INTERPRETATION
"""

import os
import sys
import argparse
from pathlib import Path
from analyzer import CyclingAnalyzer

def main():
    """Main function implementing the comprehensive cycling analysis pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Enhanced Cycling Analysis with Comprehensive Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Logical Flow:
1. 📥 DATA INGESTION - Import FIT/TCX/CSV file(s)
2. 🔍 INITIAL DATA CHECKS - Verify required fields and data quality
3. 🧹 DATA CLEANING - Handle missing values, outliers, artifacts
4. 🧮 FEATURE ENGINEERING - Calculate core and derived metrics
5. 📊 DATA AGGREGATION - Create ride totals and summaries
6. 📈 VISUALIZATION - Generate comprehensive charts and plots
7. 🧪 MODELING & INTERPRETATION - CP & W' estimation, trends analysis

Example usage:
  python run.py --file activity.fit --ftp 250 --max-hr 195
  python run.py --file activity.tcx --ftp 280 --weight 75
        """
    )
    
    parser.add_argument('--file', '-f', required=True,
                       help='Path to activity file (FIT, TCX, or CSV)')
    parser.add_argument('--ftp', type=int, default=250,
                       help='Functional Threshold Power in watts (default: 250)')
    parser.add_argument('--max-hr', type=int, default=195,
                       help='Maximum heart rate (default: 195)')
    parser.add_argument('--rest-hr', type=int, default=51,
                       help='Resting heart rate (default: 51)')
    parser.add_argument('--weight', type=float, default=70,
                       help='Athlete weight in kg (default: 70)')
    parser.add_argument('--height', type=float, default=175,
                       help='Athlete height in cm (default: 175)')
    parser.add_argument('--name', default='Cyclist',
                       help='Athlete name (default: Cyclist)')
    parser.add_argument('--output-dir', default='figures',
                       help='Output directory for figures (default: figures)')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save figures (display only)')
    parser.add_argument('--analysis-id',
                       help='Custom analysis ID (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.file):
        print(f"❌ Error: File '{args.file}' not found")
        sys.exit(1)
    
    # Check file format
    file_ext = Path(args.file).suffix.lower()
    if file_ext not in ['.fit', '.tcx', '.csv']:
        print(f"❌ Error: Unsupported file format '{file_ext}'. Supported: .fit, .tcx, .csv")
        sys.exit(1)
    
    print("🚴 Enhanced Cycling Analysis Pipeline")
    print("=" * 60)
    print(f"📁 Input file: {args.file}")
    print(f"👤 Athlete: {args.name}")
    print(f"⚡ FTP: {args.ftp}W")
    print(f"❤️ Max HR: {args.max_hr} bpm")
    print(f"⚖️ Weight: {args.weight}kg")
    print(f"📏 Height: {args.height}cm")
    print(f"💾 Output: {args.output_dir}")
    print("=" * 60)
    
    try:
        # Initialize the analyzer with comprehensive pipeline
        analyzer = CyclingAnalyzer(
            save_figures=not args.no_save,
            ftp=args.ftp,
            max_hr=args.max_hr,
            rest_hr=args.rest_hr,
            weight_kg=args.weight,
            height_cm=args.height,
            athlete_name=args.name,
            save_dir=args.output_dir,
            analysis_id=args.analysis_id
        )
        
        # Run the comprehensive data processing pipeline
        success = analyzer.process_activity_data(args.file)
        
        if success:
            print("\n✅ Analysis completed successfully!")
            print(f"📊 Results saved to: {args.output_dir}")
            
            # Print summary of key metrics
            if hasattr(analyzer, 'metrics'):
                print("\n📈 Key Metrics Summary:")
                print("-" * 40)
                
                if 'power' in analyzer.metrics:
                    power = analyzer.metrics['power']
                    print(f"⚡ Avg Power: {power['avg']:.0f}W")
                    print(f"⚡ Max Power: {power['max']:.0f}W")
                    print(f"⚡ Normalized Power: {power['np']:.0f}W")
                    print(f"⚡ Intensity Factor: {power['if']:.2f}")
                    print(f"⚡ Training Stress Score: {power['tss']:.0f}")
                
                if 'heart_rate' in analyzer.metrics:
                    hr = analyzer.metrics['heart_rate']
                    print(f"❤️ Avg HR: {hr['avg']:.0f} bpm")
                    print(f"❤️ Max HR: {hr['max']:.0f} bpm")
                
                if 'ride' in analyzer.metrics:
                    ride = analyzer.metrics['ride']
                    print(f"⏱️ Duration: {ride['duration_hr']:.2f} hours")
                    print(f"📏 Distance: {ride['total_distance']:.2f} km")
                    print(f"⚡ Energy: {ride['total_kj']:.0f} kJ")
                    print(f"🏔️ Elevation: {ride['total_elevation_m']:.0f}m")
                
                if 'power_bests' in analyzer.metrics and analyzer.metrics['power_bests']:
                    print("\n🏆 Power Bests:")
                    for interval, data in analyzer.metrics['power_bests'].items():
                        print(f"   {interval}: {data['power']:.0f}W")
            
            # Print quality report if available
            if hasattr(analyzer, 'quality_report') and analyzer.quality_report:
                report = analyzer.quality_report
                print(f"\n📊 Data Quality: {report.overall_quality.value.upper()}")
                print(f"📝 Missing data: {report.missing_data_percentage:.1f}%")
                
                if report.recommendations:
                    print("\n💡 Recommendations:")
                    for rec in report.recommendations:
                        print(f"   • {rec}")
        
        else:
            print("❌ Analysis failed. Check the error messages above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 