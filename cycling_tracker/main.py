"""
Main module for the cycling tracker system.
Provides high-level interface for processing FIT files and managing ride data.
"""
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings

# Core components
from .core.fit_parser import parse_fit_file
from .core.interval_detector import detect_intervals
from .metrics.ride_metrics import calculate_ride_metrics
from .storage.csv_manager import CSVStorageManager, store_ride_analysis
from .storage.data_models import RideData, RideMetrics, IntervalMetrics
from .utils.config import get_config, CyclingConfig

class CyclingTracker:
    """
    Main cycling tracker class that orchestrates all components.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.config = get_config()
        self.storage = CSVStorageManager(data_dir)
        self.processed_rides: Dict[str, RideMetrics] = {}
        
        print("🚴 Cycling Tracker initialized")
        print(f"   📁 Data directory: {self.storage.data_dir}")
    
    def setup_rider_profile(self, ftp: int, lthr: Optional[int] = None, 
                          mass_kg: Optional[float] = None, 
                          crank_length_mm: Optional[int] = None) -> None:
        """
        Setup rider profile for analysis.
        
        Args:
            ftp: Functional Threshold Power (watts)
            lthr: Lactate Threshold Heart Rate (bpm)
            mass_kg: Rider mass in kilograms
            crank_length_mm: Crank length in millimeters
        """
        self.config.set_rider_profile(ftp, lthr, mass_kg, crank_length_mm)
        
        print("👤 Rider profile updated:")
        print(f"   🔋 FTP: {ftp}W")
        if lthr:
            print(f"   💗 LTHR: {lthr} bpm")
        if mass_kg:
            print(f"   ⚖️ Mass: {mass_kg} kg")
        if crank_length_mm:
            print(f"   🔧 Crank length: {crank_length_mm} mm")
    
    def process_fit_file(self, file_path: str, overwrite_existing: bool = False) -> Tuple[RideMetrics, List[IntervalMetrics]]:
        """
        Process a FIT file and extract comprehensive metrics.
        
        Args:
            file_path: Path to the FIT file
            overwrite_existing: Whether to overwrite existing data in storage
            
        Returns:
            Tuple of (RideMetrics, List[IntervalMetrics])
        """
        print(f"\n🔄 Processing FIT file: {Path(file_path).name}")
        print("=" * 60)
        
        # Validate configuration
        try:
            self.config.validate_configuration()
        except ValueError as e:
            raise ValueError(f"Configuration error: {e}")
        
        # 1. Parse FIT file
        print("1️⃣ Parsing FIT file...")
        ride_data = parse_fit_file(file_path)
        
        # 2. Detect intervals
        print("\n2️⃣ Detecting intervals...")
        intervals = detect_intervals(ride_data)
        
        # 3. Calculate ride metrics
        print("\n3️⃣ Calculating ride metrics...")
        ride_metrics = calculate_ride_metrics(ride_data, intervals)
        
        # 4. Store to CSV
        print("\n4️⃣ Storing metrics...")
        storage_success = store_ride_analysis(
            ride_metrics, intervals, self.storage, overwrite_existing
        )
        
        if storage_success:
            self.processed_rides[ride_metrics.ride_id] = ride_metrics
            print(f"✅ Successfully processed and stored ride: {ride_metrics.ride_id}")
        else:
            print(f"⚠️ Ride processed but storage failed")
        
        # 5. Print summary
        self._print_processing_summary(ride_metrics, intervals)
        
        return ride_metrics, intervals
    
    def process_multiple_fit_files(self, file_paths: List[str], overwrite_existing: bool = False) -> Dict[str, Tuple[RideMetrics, List[IntervalMetrics]]]:
        """
        Process multiple FIT files in batch.
        
        Args:
            file_paths: List of FIT file paths
            overwrite_existing: Whether to overwrite existing data
            
        Returns:
            Dictionary mapping file paths to (RideMetrics, IntervalMetrics) tuples
        """
        print(f"\n📦 Batch processing {len(file_paths)} FIT files...")
        print("=" * 60)
        
        results = {}
        failed_files = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                print(f"\n[{i}/{len(file_paths)}] Processing: {Path(file_path).name}")
                ride_metrics, intervals = self.process_fit_file(file_path, overwrite_existing)
                results[file_path] = (ride_metrics, intervals)
                
            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")
                failed_files.append(file_path)
        
        # Print batch summary
        print(f"\n📊 Batch processing complete:")
        print(f"   ✅ Successfully processed: {len(results)} files")
        print(f"   ❌ Failed: {len(failed_files)} files")
        
        if failed_files:
            print(f"\n   Failed files:")
            for failed in failed_files:
                print(f"      • {Path(failed).name}")
        
        return results
    
    def get_ride_comparison(self, ride_ids: List[str]) -> Dict[str, any]:
        """
        Compare multiple rides and return analysis.
        
        Args:
            ride_ids: List of ride IDs to compare
            
        Returns:
            Dictionary with comparison data and analysis
        """
        print(f"\n📊 Comparing {len(ride_ids)} rides...")
        
        # Load comparison data
        comparison_data = self.storage.get_ride_comparison_data(ride_ids)
        rides_df = comparison_data['rides']
        intervals_df = comparison_data['intervals']
        
        if rides_df.empty:
            print("❌ No ride data found for comparison")
            return {}
        
        # Calculate comparison metrics
        comparison = {
            'rides': rides_df,
            'intervals': intervals_df,
            'summary': self._calculate_comparison_summary(rides_df, intervals_df),
            'trends': self._calculate_comparison_trends(rides_df),
            'interval_analysis': self._analyze_intervals_comparison(intervals_df)
        }
        
        print(f"✅ Comparison complete for {len(rides_df)} rides")
        return comparison
    
    def get_performance_trends(self, days_back: int = 90) -> Dict[str, any]:
        """
        Analyze performance trends over time.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        print(f"\n📈 Analyzing performance trends ({days_back} days)...")
        
        # Load trend data
        trend_data = self.storage.get_performance_trend_data(days_back)
        rides_df = trend_data['rides']
        intervals_df = trend_data['intervals']
        
        if rides_df.empty:
            print("❌ No ride data found for trend analysis")
            return {}
        
        # Calculate trends
        trends = {
            'rides': rides_df,
            'intervals': intervals_df,
            'power_trends': self._calculate_power_trends(rides_df),
            'volume_trends': self._calculate_volume_trends(rides_df),
            'interval_trends': self._calculate_interval_trends(intervals_df)
        }
        
        print(f"✅ Trend analysis complete for {len(rides_df)} rides")
        return trends
    
    def get_storage_info(self) -> Dict[str, any]:
        """Get information about stored data."""
        return self.storage.get_storage_stats()
    
    def delete_ride(self, ride_id: str) -> bool:
        """Delete a ride from storage."""
        success = self.storage.delete_ride_data(ride_id)
        if success and ride_id in self.processed_rides:
            del self.processed_rides[ride_id]
        return success
    
    def _print_processing_summary(self, ride_metrics: RideMetrics, intervals: List[IntervalMetrics]):
        """Print a summary of the processed ride."""
        print(f"\n📋 RIDE SUMMARY")
        print("=" * 40)
        print(f"Ride ID: {ride_metrics.ride_id}")
        print(f"Date: {ride_metrics.date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {ride_metrics.total_time_seconds / 3600:.2f} hours")
        print(f"Distance: {ride_metrics.total_distance_km:.1f} km")
        print(f"Average Power: {ride_metrics.avg_power_watts:.0f}W ({ride_metrics.avg_power_per_kg:.1f} W/kg)")
        print(f"TSS: {ride_metrics.training_stress_score:.0f}")
        print(f"IF: {ride_metrics.intensity_factor:.3f}")
        print(f"Intervals Detected: {len(intervals)}")
        
        if intervals:
            # Show interval summary by zone
            zone_counts = {}
            for interval in intervals:
                zone = interval.power_zone
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
            
            print(f"\nInterval Distribution:")
            for zone, count in sorted(zone_counts.items()):
                print(f"  {zone}: {count} intervals")
        
        print("=" * 40)
    
    def _calculate_comparison_summary(self, rides_df, intervals_df):
        """Calculate summary statistics for ride comparison."""
        if rides_df.empty:
            return {}
        
        return {
            'avg_power_mean': rides_df['avg_power_watts'].mean(),
            'avg_power_std': rides_df['avg_power_watts'].std(),
            'distance_mean': rides_df['total_distance_km'].mean(),
            'distance_std': rides_df['total_distance_km'].std(),
            'tss_mean': rides_df['training_stress_score'].mean(),
            'tss_std': rides_df['training_stress_score'].std(),
            'total_rides': len(rides_df),
            'total_intervals': len(intervals_df) if not intervals_df.empty else 0
        }
    
    def _calculate_comparison_trends(self, rides_df):
        """Calculate trends across compared rides."""
        if len(rides_df) < 2:
            return {}
        
        # Sort by date
        rides_df = rides_df.sort_values('date')
        
        return {
            'power_trend': self._calculate_linear_trend(rides_df['avg_power_watts']),
            'distance_trend': self._calculate_linear_trend(rides_df['total_distance_km']),
            'tss_trend': self._calculate_linear_trend(rides_df['training_stress_score'])
        }
    
    def _analyze_intervals_comparison(self, intervals_df):
        """Analyze intervals across compared rides."""
        if intervals_df.empty:
            return {}
        
        # Group by power zone
        zone_analysis = {}
        for zone in intervals_df['power_zone'].unique():
            zone_data = intervals_df[intervals_df['power_zone'] == zone]
            zone_analysis[zone] = {
                'count': len(zone_data),
                'avg_power': zone_data['avg_power_watts'].mean(),
                'avg_duration': zone_data['duration_seconds'].mean()
            }
        
        return zone_analysis
    
    def _calculate_power_trends(self, rides_df):
        """Calculate power-related trends."""
        if len(rides_df) < 2:
            return {}
        
        rides_df = rides_df.sort_values('date')
        
        return {
            'avg_power_trend': self._calculate_linear_trend(rides_df['avg_power_watts']),
            'max_power_trend': self._calculate_linear_trend(rides_df['max_power_watts']),
            'normalized_power_trend': self._calculate_linear_trend(rides_df['normalized_power_watts'])
        }
    
    def _calculate_volume_trends(self, rides_df):
        """Calculate training volume trends."""
        if len(rides_df) < 2:
            return {}
        
        rides_df = rides_df.sort_values('date')
        
        return {
            'distance_trend': self._calculate_linear_trend(rides_df['total_distance_km']),
            'time_trend': self._calculate_linear_trend(rides_df['total_time_seconds'] / 3600),
            'tss_trend': self._calculate_linear_trend(rides_df['training_stress_score'])
        }
    
    def _calculate_interval_trends(self, intervals_df):
        """Calculate interval-related trends."""
        if intervals_df.empty:
            return {}
        
        # Group by ride and calculate ride-level interval metrics
        ride_interval_summary = intervals_df.groupby('ride_id').agg({
            'avg_power_watts': 'mean',
            'duration_seconds': 'mean',
            'quality_score': 'mean'
        }).reset_index()
        
        if len(ride_interval_summary) < 2:
            return {}
        
        return {
            'interval_power_trend': self._calculate_linear_trend(ride_interval_summary['avg_power_watts']),
            'interval_duration_trend': self._calculate_linear_trend(ride_interval_summary['duration_seconds']),
            'interval_quality_trend': self._calculate_linear_trend(ride_interval_summary['quality_score'])
        }
    
    def _calculate_linear_trend(self, data):
        """Calculate linear trend (slope) for a data series."""
        try:
            import numpy as np
            if len(data) < 2:
                return 0.0
            
            x = np.arange(len(data))
            y = data.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return 0.0
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Calculate linear regression slope
            slope = np.polyfit(x_clean, y_clean, 1)[0]
            return float(slope)
            
        except Exception:
            return 0.0

# Convenience functions
def setup_cycling_tracker(ftp: int, data_dir: Optional[str] = None, **rider_params) -> CyclingTracker:
    """
    Setup and configure a cycling tracker instance.
    
    Args:
        ftp: Functional Threshold Power
        data_dir: Data directory path
        **rider_params: Additional rider parameters (lthr, mass_kg, crank_length_mm)
        
    Returns:
        Configured CyclingTracker instance
    """
    tracker = CyclingTracker(data_dir)
    tracker.setup_rider_profile(ftp, **rider_params)
    return tracker

def process_single_ride(file_path: str, ftp: int, **config_params) -> Tuple[RideMetrics, List[IntervalMetrics]]:
    """
    Process a single FIT file with minimal setup.
    
    Args:
        file_path: Path to FIT file
        ftp: Functional Threshold Power
        **config_params: Additional configuration parameters
        
    Returns:
        Tuple of (RideMetrics, List[IntervalMetrics])
    """
    tracker = setup_cycling_tracker(ftp, **config_params)
    return tracker.process_fit_file(file_path)