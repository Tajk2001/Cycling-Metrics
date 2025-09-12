"""
Configuration module for the cycling tracker system.
Provides dynamic configuration without hardcoded values.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class RiderProfile:
    """Rider-specific configuration."""
    ftp: int = 0  # Functional Threshold Power - to be set dynamically
    lthr: int = 0  # Lactate Threshold Heart Rate
    mass_kg: float = 70.0  # Default rider mass
    crank_length_mm: int = 172.5  # Standard crank length

@dataclass 
class DetectionSettings:
    """Interval detection configuration."""
    min_interval_duration: int = 60  # Minimum interval duration in seconds
    power_threshold_pct: float = 0.8  # Percentage of FTP for power-based detection
    cv_threshold: float = 0.15  # Coefficient of variation threshold
    overlap_threshold: int = 30  # Overlap threshold in seconds
    set_time_window: int = 300  # Time window for identifying interval sets
    max_intervals_to_analyze: int = 15  # Maximum intervals to analyze in detail

@dataclass
class ProcessingSettings:
    """Data processing configuration."""
    speed_threshold: float = 0.5  # Minimum speed threshold (m/s)
    min_cadence: int = 40  # Minimum cadence threshold
    max_cadence: int = 140  # Maximum cadence threshold
    smoothing_window: int = 3  # Window for data smoothing

@dataclass
class StorageSettings:
    """Data storage configuration."""
    data_dir: str = "cycling_data"  # Directory for storing ride data
    csv_ride_metrics: str = "ride_metrics.csv"  # Ride-level metrics file
    csv_interval_metrics: str = "interval_metrics.csv"  # Interval-level metrics file
    backup_enabled: bool = True  # Enable data backup
    max_backup_files: int = 10  # Maximum backup files to keep

class CyclingConfig:
    """Main configuration class for the cycling tracker system."""
    
    def __init__(self):
        self.rider = RiderProfile()
        self.detection = DetectionSettings()
        self.processing = ProcessingSettings()
        self.storage = StorageSettings()
        self._user_inputs: Dict[str, Any] = {}
    
    def set_rider_profile(self, ftp: int, lthr: Optional[int] = None, 
                         mass_kg: Optional[float] = None, 
                         crank_length_mm: Optional[int] = None):
        """Set rider profile parameters dynamically."""
        self.rider.ftp = ftp
        self._user_inputs['ftp'] = ftp
        
        if lthr is not None:
            self.rider.lthr = lthr
            self._user_inputs['lthr'] = lthr
        
        if mass_kg is not None:
            self.rider.mass_kg = mass_kg
            self._user_inputs['mass_kg'] = mass_kg
        
        if crank_length_mm is not None:
            self.rider.crank_length_mm = crank_length_mm
            self._user_inputs['crank_length_mm'] = crank_length_mm
    
    def get_ftp_for_interval_detection(self) -> int:
        """Get FTP for interval detection, ensuring it's been set."""
        if self.rider.ftp <= 0:
            raise ValueError("FTP must be set before interval detection. Use set_rider_profile() to set FTP.")
        return self.rider.ftp
    
    def update_detection_settings(self, **kwargs):
        """Update interval detection settings dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.detection, key):
                setattr(self.detection, key, value)
                self._user_inputs[f'detection_{key}'] = value
            else:
                raise ValueError(f"Unknown detection setting: {key}")
    
    def update_processing_settings(self, **kwargs):
        """Update processing settings dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.processing, key):
                setattr(self.processing, key, value)
                self._user_inputs[f'processing_{key}'] = value
            else:
                raise ValueError(f"Unknown processing setting: {key}")
    
    def update_storage_settings(self, **kwargs):
        """Update storage settings dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.storage, key):
                setattr(self.storage, key, value)
                self._user_inputs[f'storage_{key}'] = value
            else:
                raise ValueError(f"Unknown storage setting: {key}")
    
    def get_power_zones(self) -> Dict[str, Dict[str, float]]:
        """Calculate power zones based on FTP."""
        if self.rider.ftp <= 0:
            raise ValueError("FTP must be set to calculate power zones")
        
        ftp = self.rider.ftp
        return {
            'zone1': {'min': 0, 'max': 0.55 * ftp, 'name': 'Active Recovery'},
            'zone2': {'min': 0.55 * ftp, 'max': 0.75 * ftp, 'name': 'Endurance'},
            'zone3': {'min': 0.75 * ftp, 'max': 0.90 * ftp, 'name': 'Tempo'},
            'zone4': {'min': 0.90 * ftp, 'max': 1.05 * ftp, 'name': 'Threshold'},
            'zone5': {'min': 1.05 * ftp, 'max': 1.20 * ftp, 'name': 'VO2 Max'},
            'zone6': {'min': 1.20 * ftp, 'max': 1.50 * ftp, 'name': 'Anaerobic'},
            'zone7': {'min': 1.50 * ftp, 'max': float('inf'), 'name': 'Neuromuscular'}
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'rider_profile': {
                'ftp': self.rider.ftp,
                'lthr': self.rider.lthr,
                'mass_kg': self.rider.mass_kg,
                'crank_length_mm': self.rider.crank_length_mm
            },
            'detection_settings': {
                'min_interval_duration': self.detection.min_interval_duration,
                'power_threshold_pct': self.detection.power_threshold_pct,
                'cv_threshold': self.detection.cv_threshold
            },
            'user_inputs': self._user_inputs
        }
    
    def validate_configuration(self) -> bool:
        """Validate that required configuration is set."""
        errors = []
        
        if self.rider.ftp <= 0:
            errors.append("FTP must be set and greater than 0")
        
        if self.rider.mass_kg <= 0:
            errors.append("Rider mass must be greater than 0")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
        
        return True

# Global configuration instance
config = CyclingConfig()

def get_config() -> CyclingConfig:
    """Get the global configuration instance."""
    return config

def reset_config() -> CyclingConfig:
    """Reset configuration to defaults."""
    global config
    config = CyclingConfig()
    return config