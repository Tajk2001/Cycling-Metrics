"""
CSV storage manager for the cycling tracker system.
Handles persistence of ride and interval metrics to CSV files.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple
import shutil
import warnings

from .data_models import RideMetrics, IntervalMetrics, RideData
from ..utils.config import get_config

class CSVStorageManager:
    """
    Manages CSV storage for ride and interval metrics with backup functionality.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.config = get_config()
        self.data_dir = Path(data_dir) if data_dir else Path(self.config.storage.data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # CSV file paths
        self.ride_metrics_file = self.data_dir / self.config.storage.csv_ride_metrics
        self.interval_metrics_file = self.data_dir / self.config.storage.csv_interval_metrics
        
        # Backup directory
        if self.config.storage.backup_enabled:
            self.backup_dir = self.data_dir / "backups"
            self.backup_dir.mkdir(exist_ok=True)
        
        print(f"📁 CSV Storage initialized: {self.data_dir}")
    
    def store_ride_metrics(self, ride_metrics: RideMetrics, overwrite_existing: bool = False) -> bool:
        """
        Store ride metrics to CSV file.
        
        Args:
            ride_metrics: RideMetrics object to store
            overwrite_existing: Whether to overwrite existing ride with same ID
            
        Returns:
            True if successfully stored, False otherwise
        """
        try:
            # Convert to DataFrame
            new_data = pd.DataFrame([ride_metrics.to_dict()])
            
            # Load existing data if file exists
            if self.ride_metrics_file.exists():
                existing_df = pd.read_csv(self.ride_metrics_file)
                
                # Check for existing ride
                if ride_metrics.ride_id in existing_df['ride_id'].values:
                    if overwrite_existing:
                        print(f"🔄 Updating existing ride: {ride_metrics.ride_id}")
                        existing_df = existing_df[existing_df['ride_id'] != ride_metrics.ride_id]
                        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                    else:
                        print(f"⚠️ Ride {ride_metrics.ride_id} already exists. Use overwrite_existing=True to update.")
                        return False
                else:
                    print(f"➕ Adding new ride: {ride_metrics.ride_id}")
                    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            else:
                print(f"📝 Creating new ride metrics file with: {ride_metrics.ride_id}")
                combined_df = new_data
            
            # Create backup if enabled
            if self.config.storage.backup_enabled and self.ride_metrics_file.exists():
                self._create_backup(self.ride_metrics_file, "ride_metrics")
            
            # Sort by date and save
            combined_df['date'] = pd.to_datetime(combined_df['date'])
            combined_df = combined_df.sort_values('date')
            combined_df.to_csv(self.ride_metrics_file, index=False)
            
            print(f"✅ Stored ride metrics: {ride_metrics.ride_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing ride metrics: {e}")
            return False
    
    def store_interval_metrics(self, interval_metrics: List[IntervalMetrics], overwrite_existing: bool = False) -> bool:
        """
        Store interval metrics to CSV file.
        
        Args:
            interval_metrics: List of IntervalMetrics objects to store
            overwrite_existing: Whether to overwrite existing intervals with same ride_id
            
        Returns:
            True if successfully stored, False otherwise
        """
        if not interval_metrics:
            print("⚠️ No interval metrics to store")
            return True
        
        try:
            # Convert to DataFrame
            new_data = pd.DataFrame([interval.to_dict() for interval in interval_metrics])
            ride_id = interval_metrics[0].ride_id
            
            # Load existing data if file exists
            if self.interval_metrics_file.exists():
                existing_df = pd.read_csv(self.interval_metrics_file)
                
                # Check for existing intervals from this ride
                existing_ride_intervals = existing_df[existing_df['ride_id'] == ride_id]
                if not existing_ride_intervals.empty:
                    if overwrite_existing:
                        print(f"🔄 Updating existing intervals for ride: {ride_id}")
                        existing_df = existing_df[existing_df['ride_id'] != ride_id]
                        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                    else:
                        print(f"⚠️ Intervals for ride {ride_id} already exist. Use overwrite_existing=True to update.")
                        return False
                else:
                    print(f"➕ Adding {len(interval_metrics)} intervals for ride: {ride_id}")
                    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            else:
                print(f"📝 Creating new interval metrics file with {len(interval_metrics)} intervals")
                combined_df = new_data
            
            # Create backup if enabled
            if self.config.storage.backup_enabled and self.interval_metrics_file.exists():
                self._create_backup(self.interval_metrics_file, "interval_metrics")
            
            # Sort by ride_id and interval_number and save
            combined_df['start_time'] = pd.to_datetime(combined_df['start_time'])
            combined_df = combined_df.sort_values(['ride_id', 'interval_number'])
            combined_df.to_csv(self.interval_metrics_file, index=False)
            
            print(f"✅ Stored {len(interval_metrics)} interval metrics for ride: {ride_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing interval metrics: {e}")
            return False
    
    def load_ride_metrics(self, ride_id: Optional[str] = None, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load ride metrics from CSV file.
        
        Args:
            ride_id: Specific ride ID to load (optional)
            start_date: Filter rides from this date onwards (optional)
            end_date: Filter rides up to this date (optional)
            
        Returns:
            DataFrame with ride metrics
        """
        if not self.ride_metrics_file.exists():
            print("📄 No ride metrics file found")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.ride_metrics_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Apply filters
            if ride_id:
                df = df[df['ride_id'] == ride_id]
            
            if start_date:
                df = df[df['date'] >= start_date]
            
            if end_date:
                df = df[df['date'] <= end_date]
            
            print(f"📊 Loaded {len(df)} ride records")
            return df
            
        except Exception as e:
            print(f"❌ Error loading ride metrics: {e}")
            return pd.DataFrame()
    
    def load_interval_metrics(self, ride_id: Optional[str] = None,
                            power_zone: Optional[str] = None,
                            min_duration: Optional[int] = None,
                            max_duration: Optional[int] = None) -> pd.DataFrame:
        """
        Load interval metrics from CSV file.
        
        Args:
            ride_id: Specific ride ID to filter by (optional)
            power_zone: Filter by power zone (optional)
            min_duration: Minimum duration in seconds (optional)
            max_duration: Maximum duration in seconds (optional)
            
        Returns:
            DataFrame with interval metrics
        """
        if not self.interval_metrics_file.exists():
            print("📄 No interval metrics file found")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.interval_metrics_file)
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            
            # Apply filters
            if ride_id:
                df = df[df['ride_id'] == ride_id]
            
            if power_zone:
                df = df[df['power_zone'] == power_zone]
            
            if min_duration:
                df = df[df['duration_seconds'] >= min_duration]
            
            if max_duration:
                df = df[df['duration_seconds'] <= max_duration]
            
            print(f"📊 Loaded {len(df)} interval records")
            return df
            
        except Exception as e:
            print(f"❌ Error loading interval metrics: {e}")
            return pd.DataFrame()
    
    def get_ride_comparison_data(self, ride_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get data for comparing multiple rides.
        
        Args:
            ride_ids: List of ride IDs to compare
            
        Returns:
            Dictionary with 'rides' and 'intervals' DataFrames
        """
        result = {
            'rides': pd.DataFrame(),
            'intervals': pd.DataFrame()
        }
        
        try:
            # Load ride data
            all_rides = self.load_ride_metrics()
            if not all_rides.empty:
                result['rides'] = all_rides[all_rides['ride_id'].isin(ride_ids)]
            
            # Load interval data
            all_intervals = self.load_interval_metrics()
            if not all_intervals.empty:
                result['intervals'] = all_intervals[all_intervals['ride_id'].isin(ride_ids)]
            
            print(f"📊 Comparison data loaded: {len(result['rides'])} rides, {len(result['intervals'])} intervals")
            
        except Exception as e:
            print(f"❌ Error loading comparison data: {e}")
        
        return result
    
    def get_performance_trend_data(self, days_back: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Get data for performance trend analysis.
        
        Args:
            days_back: Number of days to look back from today
            
        Returns:
            Dictionary with trend analysis data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        rides = self.load_ride_metrics(start_date=start_date, end_date=end_date)
        
        result = {
            'rides': rides,
            'intervals': pd.DataFrame()
        }
        
        # Load intervals for these rides
        if not rides.empty:
            ride_ids = rides['ride_id'].tolist()
            intervals = self.load_interval_metrics()
            if not intervals.empty:
                result['intervals'] = intervals[intervals['ride_id'].isin(ride_ids)]
        
        return result
    
    def delete_ride_data(self, ride_id: str) -> bool:
        """
        Delete all data for a specific ride.
        
        Args:
            ride_id: Ride ID to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            success = True
            
            # Delete ride metrics
            if self.ride_metrics_file.exists():
                ride_df = pd.read_csv(self.ride_metrics_file)
                if ride_id in ride_df['ride_id'].values:
                    ride_df = ride_df[ride_df['ride_id'] != ride_id]
                    ride_df.to_csv(self.ride_metrics_file, index=False)
                    print(f"🗑️ Deleted ride metrics for: {ride_id}")
                else:
                    print(f"⚠️ Ride {ride_id} not found in ride metrics")
                    success = False
            
            # Delete interval metrics
            if self.interval_metrics_file.exists():
                interval_df = pd.read_csv(self.interval_metrics_file)
                if ride_id in interval_df['ride_id'].values:
                    interval_df = interval_df[interval_df['ride_id'] != ride_id]
                    interval_df.to_csv(self.interval_metrics_file, index=False)
                    print(f"🗑️ Deleted interval metrics for: {ride_id}")
                else:
                    print(f"⚠️ Ride {ride_id} not found in interval metrics")
            
            return success
            
        except Exception as e:
            print(f"❌ Error deleting ride data: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, any]:
        """Get statistics about stored data."""
        stats = {
            'rides_count': 0,
            'intervals_count': 0,
            'date_range': None,
            'storage_size_mb': 0,
            'backup_count': 0
        }
        
        try:
            # Ride stats
            if self.ride_metrics_file.exists():
                ride_df = pd.read_csv(self.ride_metrics_file)
                stats['rides_count'] = len(ride_df)
                
                if not ride_df.empty:
                    ride_df['date'] = pd.to_datetime(ride_df['date'])
                    stats['date_range'] = {
                        'earliest': ride_df['date'].min(),
                        'latest': ride_df['date'].max()
                    }
                
                stats['storage_size_mb'] += self.ride_metrics_file.stat().st_size / (1024 * 1024)
            
            # Interval stats
            if self.interval_metrics_file.exists():
                interval_df = pd.read_csv(self.interval_metrics_file)
                stats['intervals_count'] = len(interval_df)
                stats['storage_size_mb'] += self.interval_metrics_file.stat().st_size / (1024 * 1024)
            
            # Backup stats
            if self.config.storage.backup_enabled and self.backup_dir.exists():
                backup_files = list(self.backup_dir.glob("*.csv"))
                stats['backup_count'] = len(backup_files)
        
        except Exception as e:
            print(f"❌ Error getting storage stats: {e}")
        
        return stats
    
    def _create_backup(self, file_path: Path, file_type: str):
        """Create a backup of the specified file."""
        if not file_path.exists():
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_type}_{timestamp}.csv"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(file_path, backup_path)
            
            # Clean up old backups
            self._cleanup_old_backups(file_type)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create backup: {e}")
    
    def _cleanup_old_backups(self, file_type: str):
        """Remove old backup files, keeping only the most recent ones."""
        try:
            backup_files = sorted(
                self.backup_dir.glob(f"{file_type}_*.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Keep only the most recent backups
            max_backups = self.config.storage.max_backup_files
            for old_backup in backup_files[max_backups:]:
                old_backup.unlink()
                
        except Exception as e:
            print(f"⚠️ Warning: Could not cleanup old backups: {e}")

def store_ride_analysis(
    ride_metrics: RideMetrics, 
    interval_metrics: List[IntervalMetrics],
    storage_manager: Optional[CSVStorageManager] = None,
    overwrite_existing: bool = False
) -> bool:
    """
    Convenience function to store complete ride analysis results.
    
    Args:
        ride_metrics: RideMetrics object to store
        interval_metrics: List of IntervalMetrics objects to store
        storage_manager: Optional CSVStorageManager instance
        overwrite_existing: Whether to overwrite existing data
        
    Returns:
        True if successfully stored, False otherwise
    """
    if storage_manager is None:
        storage_manager = CSVStorageManager()
    
    # Store ride metrics
    ride_success = storage_manager.store_ride_metrics(ride_metrics, overwrite_existing)
    
    # Store interval metrics
    interval_success = storage_manager.store_interval_metrics(interval_metrics, overwrite_existing)
    
    return ride_success and interval_success

def load_ride_comparison_data(ride_ids: List[str], storage_manager: Optional[CSVStorageManager] = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load ride comparison data.
    
    Args:
        ride_ids: List of ride IDs to compare
        storage_manager: Optional CSVStorageManager instance
        
    Returns:
        Dictionary with comparison data
    """
    if storage_manager is None:
        storage_manager = CSVStorageManager()
    
    return storage_manager.get_ride_comparison_data(ride_ids)