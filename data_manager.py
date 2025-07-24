#!/usr/bin/env python3
"""
Cycling Data Manager
Comprehensive data management system for cycling analysis with error-proofing and clean storage.

This module handles:
- FIT file storage and retrieval
- Analysis results management
- Data caching and validation
- Settings management
- System status monitoring

Author: Cycling Analysis Team
Version: 1.0.0
"""

# Standard library imports
import os
import json
import hashlib
import shutil
import tempfile
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CyclingDataManager:
    """
    Comprehensive data manager for cycling analysis with error-proofing and clean storage.
    
    This class handles FIT file storage, analysis results, caching, and data retrieval
    with robust error handling and data validation.
    
    Attributes:
        data_dir (Path): Directory for storing data files
        cache_dir (Path): Directory for caching files
        figures_dir (Path): Directory for storing generated figures
        ride_history_path (Path): Path to ride history CSV file
        analysis_history_path (Path): Path to analysis history CSV file
        file_registry_path (Path): Path to file registry JSON file
        settings_path (Path): Path to settings JSON file
    """
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "cache"):
        """
        Initialize the data manager with organized storage structure.
        
        Args:
            data_dir: Directory for storing data files
            cache_dir: Directory for caching files
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.figures_dir = Path("figures")
        
        # Create directory structure
        self._create_directories()
        
        # File paths
        self.ride_history_path = self.data_dir / "ride_history.csv"
        self.analysis_history_path = self.data_dir / "analysis_history.csv"
        self.file_registry_path = self.data_dir / "file_registry.json"
        self.settings_path = self.data_dir / "settings.json"
        
        # Initialize session state keys
        self._init_session_state()
        
        # Load existing data
        self._load_existing_data()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [self.data_dir, self.cache_dir, self.figures_dir]
        for directory in directories:
            directory.mkdir(exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def _init_session_state(self) -> None:
        """Initialize Streamlit session state with default values."""
        if 'data_manager_initialized' not in st.session_state:
            st.session_state.data_manager_initialized = True
            st.session_state.uploaded_files = {}
            st.session_state.analysis_cache = {}
            st.session_state.current_analysis = None
            st.session_state.error_messages = []
            st.session_state.success_messages = []
    
    def _load_existing_data(self) -> None:
        """Load existing data files and validate them."""
        try:
            # Load ride history
            if self.ride_history_path.exists():
                self.ride_history = pd.read_csv(self.ride_history_path)
                logger.info(f"Loaded ride history: {len(self.ride_history)} rides")
            else:
                self.ride_history = pd.DataFrame()
                logger.info("Created new ride history")
            
            # Load analysis history
            if self.analysis_history_path.exists():
                self.analysis_history = pd.read_csv(self.analysis_history_path)
                logger.info(f"Loaded analysis history: {len(self.analysis_history)} analyses")
            else:
                self.analysis_history = pd.DataFrame()
                logger.info("Created new analysis history")
            
            # Load file registry
            if self.file_registry_path.exists():
                with open(self.file_registry_path, 'r') as f:
                    self.file_registry = json.load(f)
                logger.info(f"Loaded file registry: {len(self.file_registry)} files")
            else:
                self.file_registry = {}
                logger.info("Created new file registry")
                
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            self._create_backup_and_reset()
    
    def _create_backup_and_reset(self) -> None:
        """Create backup of corrupted files and reset them."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.data_dir / f"backup_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        # Backup existing files
        for file_path in [self.ride_history_path, self.analysis_history_path, self.file_registry_path]:
            if file_path.exists():
                try:
                    shutil.copy2(file_path, backup_dir / file_path.name)
                    logger.info(f"Backed up {file_path} to {backup_dir}")
                except Exception as e:
                    logger.error(f"Failed to backup {file_path}: {e}")
        
        # Reset data structures
        self.ride_history = pd.DataFrame()
        self.analysis_history = pd.DataFrame()
        self.file_registry = {}
        
        logger.info("Reset data structures after corruption")
    
    def save_data(self) -> bool:
        """
        Save all data to disk with error handling.
        
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Save ride history
            if not self.ride_history.empty:
                self.ride_history.to_csv(self.ride_history_path, index=False)
            
            # Save analysis history
            if not self.analysis_history.empty:
                self.analysis_history.to_csv(self.analysis_history_path, index=False)
            
            # Save file registry
            with open(self.file_registry_path, 'w') as f:
                json.dump(self.file_registry, f, indent=2)
            
            logger.info("Data saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
    
    def upload_fit_file(self, uploaded_file) -> Tuple[bool, str, Optional[str]]:
        """
        Upload and store a FIT file with comprehensive error handling.
        
        Returns:
            Tuple[bool, str, Optional[str]]: (success, message, file_path)
        """
        try:
            if uploaded_file is None:
                return False, "No file uploaded", None
            
            # Validate file
            if not uploaded_file.name.lower().endswith('.fit'):
                return False, "Please upload a valid .fit file", None
            
            # Generate unique ride ID from filename
            ride_id = self._generate_ride_id(uploaded_file.name)
            
            # Check if file already exists
            if ride_id in self.file_registry:
                return False, f"File '{ride_id}' already exists. Use a different filename.", None
            
            # Save file to cache directory
            file_path = self.cache_dir / f"{ride_id}.fit"
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            # Calculate file hash for integrity checking
            file_hash = self._calculate_file_hash(file_path)
            
            # Register file
            self.file_registry[ride_id] = {
                'file_path': str(file_path),
                'original_name': uploaded_file.name,
                'upload_date': datetime.now().isoformat(),
                'file_hash': file_hash,
                'file_size': file_path.stat().st_size
            }
            
            # Store in session state for immediate access
            st.session_state.uploaded_files[ride_id] = str(file_path)
            
            # Save updated registry
            self.save_data()
            
            logger.info(f"Successfully uploaded: {ride_id}")
            return True, f"✅ File uploaded successfully: {ride_id}", str(file_path)
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False, f"❌ Upload failed: {str(e)}", None
    
    def _generate_ride_id(self, filename: str) -> str:
        """Generate a clean ride ID from filename."""
        # Remove .fit extension and clean the name
        ride_id = filename.replace('.fit', '').replace('.FIT', '')
        
        # Replace spaces and special characters with underscores
        ride_id = re.sub(r'[^a-zA-Z0-9_-]', '_', ride_id)
        
        # Remove multiple underscores
        ride_id = re.sub(r'_+', '_', ride_id)
        
        # Remove leading/trailing underscores
        ride_id = ride_id.strip('_')
        
        return ride_id
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for integrity checking."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_available_rides(self) -> List[str]:
        """Get list of all available rides (both cached and in history)."""
        cached_rides = list(st.session_state.uploaded_files.keys())
        history_rides = self.ride_history['ride_id'].tolist() if not self.ride_history.empty else []
        
        # Combine and deduplicate
        all_rides = list(set(cached_rides + history_rides))
        return sorted(all_rides)
    
    def get_fit_file_path(self, ride_id: str) -> Optional[str]:
        """Get the file path for a specific ride ID."""
        # Check session state first (recently uploaded)
        if ride_id in st.session_state.uploaded_files:
            return st.session_state.uploaded_files[ride_id]
        
        # Check file registry
        if ride_id in self.file_registry:
            file_path = self.file_registry[ride_id]['file_path']
            if Path(file_path).exists():
                return file_path
        
        # Check if file exists in cache directory
        cache_file = self.cache_dir / f"{ride_id}.fit"
        if cache_file.exists():
            return str(cache_file)
        
        return None
    
    def get_ride_data(self, ride_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive data for a specific ride."""
        ride_data = {
            'ride_id': ride_id,
            'has_fit_file': False,
            'fit_file_path': None,
            'in_history': False,
            'history_data': None,
            'analysis_available': False
        }
        
        # Check if FIT file exists
        fit_path = self.get_fit_file_path(ride_id)
        if fit_path:
            ride_data['has_fit_file'] = True
            ride_data['fit_file_path'] = fit_path
        
        # Check if in history
        if not self.ride_history.empty and ride_id in self.ride_history['ride_id'].values:
            ride_data['in_history'] = True
            ride_data['history_data'] = self.ride_history[self.ride_history['ride_id'] == ride_id].iloc[0].to_dict()
        
        # Check if analysis is available
        if not self.analysis_history.empty and ride_id in self.analysis_history['ride_name'].values:
            ride_data['analysis_available'] = True
        
        return ride_data
    
    def save_analysis_results(self, ride_id: str, analysis_type: str, results: Dict[str, Any], 
                            ftp: int, lthr: int) -> bool:
        """Save analysis results to history."""
        try:
            # Prepare analysis data
            analysis_data = {
                'ride_id': ride_id,
                'analysis_type': analysis_type,
                'ftp': ftp,
                'lthr': lthr,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }
            
            # Store in session cache
            cache_key = f"{ride_id}_{analysis_type}"
            st.session_state.analysis_cache[cache_key] = analysis_data
            
            # Save to analysis history
            history_entry = {
                'analysis_id': f"{ride_id}_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'ride_name': ride_id,
                'analysis_type': analysis_type,
                'ftp': ftp,
                'lthr': lthr,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'figures_saved': True
            }
            
            # Add to analysis history DataFrame
            self.analysis_history = pd.concat([
                self.analysis_history, 
                pd.DataFrame([history_entry])
            ], ignore_index=True)
            
            # Update ride history with metrics
            ride_history_entry = {
                'ride_id': ride_id,
                'analysis_date': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'ftp': ftp,
                'lthr': lthr
            }
            
            # Add all metrics from results to ride history
            for key, value in results.items():
                if key != 'status':  # Skip status field
                    ride_history_entry[key] = value
            
            # Remove existing entry if it exists
            if not self.ride_history.empty and ride_id in self.ride_history['ride_id'].values:
                self.ride_history = self.ride_history[self.ride_history['ride_id'] != ride_id]
            
            # Add new entry to ride history
            self.ride_history = pd.concat([
                self.ride_history,
                pd.DataFrame([ride_history_entry])
            ], ignore_index=True)
            
            # Save data
            self.save_data()
            
            logger.info(f"Saved analysis results for {ride_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            return False
    
    def get_analysis_results(self, ride_id: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis results."""
        cache_key = f"{ride_id}_{analysis_type}"
        return st.session_state.analysis_cache.get(cache_key)
    
    def validate_file_integrity(self, ride_id: str) -> Tuple[bool, str]:
        """Validate that a file hasn't been corrupted."""
        try:
            file_path = self.get_fit_file_path(ride_id)
            if not file_path:
                return False, "File not found"
            
            if ride_id not in self.file_registry:
                return False, "File not in registry"
            
            # Calculate current hash
            current_hash = self._calculate_file_hash(Path(file_path))
            stored_hash = self.file_registry[ride_id]['file_hash']
            
            if current_hash != stored_hash:
                return False, "File integrity check failed - file may be corrupted"
            
            return True, "File integrity validated"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def cleanup_orphaned_files(self) -> List[str]:
        """Remove files that are no longer referenced."""
        orphaned_files = []
        
        # Check cache directory for files not in registry
        for file_path in self.cache_dir.glob("*.fit"):
            ride_id = file_path.stem
            if ride_id not in self.file_registry:
                try:
                    file_path.unlink()
                    orphaned_files.append(str(file_path))
                    logger.info(f"Removed orphaned file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove orphaned file {file_path}: {e}")
        
        return orphaned_files
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'total_rides': len(self.get_available_rides()),
            'cached_files': len(st.session_state.uploaded_files),
            'history_entries': len(self.ride_history),
            'analysis_entries': len(self.analysis_history),
            'registry_entries': len(self.file_registry),
            'data_directory': str(self.data_dir),
            'cache_directory': str(self.cache_dir),
            'figures_directory': str(self.figures_dir)
        }
        
        # Check directory sizes
        try:
            status['data_dir_size'] = sum(f.stat().st_size for f in self.data_dir.rglob('*') if f.is_file())
            status['cache_dir_size'] = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        except Exception as e:
            logger.error(f"Error calculating directory sizes: {e}")
            status['data_dir_size'] = 0
            status['cache_dir_size'] = 0
        
        return status
    
    def export_data(self, export_path: str) -> bool:
        """Export all data to a backup location."""
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all data files
            for file_path in [self.ride_history_path, self.analysis_history_path, self.file_registry_path]:
                if file_path.exists():
                    shutil.copy2(file_path, export_dir / file_path.name)
            
            # Copy cache directory
            cache_export = export_dir / "cache"
            if self.cache_dir.exists():
                shutil.copytree(self.cache_dir, cache_export, dirs_exist_ok=True)
            
            # Copy figures directory
            figures_export = export_dir / "figures"
            if self.figures_dir.exists():
                shutil.copytree(self.figures_dir, figures_export, dirs_exist_ok=True)
            
            logger.info(f"Data exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def import_data(self, import_path: str) -> bool:
        """Import data from a backup location."""
        try:
            import_dir = Path(import_path)
            
            # Import data files
            for file_name in ["ride_history.csv", "analysis_history.csv", "file_registry.json"]:
                source_file = import_dir / file_name
                if source_file.exists():
                    shutil.copy2(source_file, self.data_dir / file_name)
            
            # Import cache
            cache_source = import_dir / "cache"
            if cache_source.exists():
                shutil.copytree(cache_source, self.cache_dir, dirs_exist_ok=True)
            
            # Reload data
            self._load_existing_data()
            
            logger.info(f"Data imported from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False
    
    def delete_ride(self, ride_id: str) -> Tuple[bool, str]:
        """
        Delete a specific ride and all associated data.
        
        Args:
            ride_id (str): The ride ID to delete
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            logger.info(f"Starting deletion of ride: {ride_id}")
            deleted_items = []
            
            # 1. Remove FIT file from cache
            fit_file_path = self.get_fit_file_path(ride_id)
            logger.info(f"FIT file path for {ride_id}: {fit_file_path}")
            if fit_file_path and Path(fit_file_path).exists():
                try:
                    Path(fit_file_path).unlink()
                    deleted_items.append("FIT file")
                    logger.info(f"Deleted FIT file: {fit_file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete FIT file {fit_file_path}: {e}")
            else:
                logger.info(f"No FIT file found for {ride_id}")
            
            # 2. Remove from file registry
            if ride_id in self.file_registry:
                del self.file_registry[ride_id]
                deleted_items.append("file registry entry")
                logger.info(f"Removed from file registry: {ride_id}")
            else:
                logger.info(f"Ride {ride_id} not found in file registry")
            
            # 3. Remove from session state
            if ride_id in st.session_state.uploaded_files:
                del st.session_state.uploaded_files[ride_id]
                deleted_items.append("session cache")
                logger.info(f"Removed from session cache: {ride_id}")
            else:
                logger.info(f"Ride {ride_id} not found in session cache")
            
            # 4. Remove from ride history
            if not self.ride_history.empty and ride_id in self.ride_history['ride_id'].values:
                before_count = len(self.ride_history)
                self.ride_history = self.ride_history[self.ride_history['ride_id'] != ride_id]
                after_count = len(self.ride_history)
                deleted_items.append("ride history entry")
                logger.info(f"Removed from ride history: {ride_id} (before: {before_count}, after: {after_count})")
            else:
                logger.info(f"Ride {ride_id} not found in ride history")
            
            # 5. Remove from analysis history
            if not self.analysis_history.empty and ride_id in self.analysis_history['ride_name'].values:
                before_count = len(self.analysis_history)
                self.analysis_history = self.analysis_history[self.analysis_history['ride_name'] != ride_id]
                after_count = len(self.analysis_history)
                deleted_items.append("analysis history entry")
                logger.info(f"Removed from analysis history: {ride_id} (before: {before_count}, after: {after_count})")
            else:
                logger.info(f"Ride {ride_id} not found in analysis history")
            
            # 6. Remove associated figures
            figure_patterns = [
                f"{ride_id}_dashboard.*",
                f"{ride_id}_fatigue_patterns.*",
                f"{ride_id}_heat_stress.*",
                f"{ride_id}_lactate.*",
                f"{ride_id}_power_hr_efficiency.*",
                f"{ride_id}_torque.*",
                f"{ride_id}_variable_relationships.*",
                f"{ride_id}_w_prime_balance.*"
            ]
            
            figures_deleted = 0
            for pattern in figure_patterns:
                for figure_file in self.figures_dir.glob(pattern):
                    try:
                        figure_file.unlink()
                        figures_deleted += 1
                        logger.info(f"Deleted figure: {figure_file}")
                    except Exception as e:
                        logger.error(f"Failed to delete figure {figure_file}: {e}")
            
            if figures_deleted > 0:
                deleted_items.append(f"{figures_deleted} figure files")
            else:
                logger.info(f"No figures found for {ride_id}")
            
            # 7. Save updated data
            save_success = self.save_data()
            logger.info(f"Data save after deletion: {save_success}")
            
            if deleted_items:
                message = f"✅ Successfully deleted ride '{ride_id}'. Removed: {', '.join(deleted_items)}"
                logger.info(message)
                return True, message
            else:
                message = f"⚠️ Ride '{ride_id}' not found or already deleted"
                logger.warning(message)
                return False, message
                
        except Exception as e:
            error_msg = f"❌ Error deleting ride '{ride_id}': {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def clear_all_rides(self) -> Tuple[bool, str]:
        """
        Clear all rides and associated data.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Create backup before clearing
            backup_path = f"backup_before_clear_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if self.export_data(backup_path):
                logger.info(f"Created backup before clearing: {backup_path}")
            
            deleted_counts = {
                'fit_files': 0,
                'history_entries': 0,
                'analysis_entries': 0,
                'figures': 0
            }
            
            # 1. Clear all FIT files from cache
            for fit_file in self.cache_dir.glob("*.fit"):
                try:
                    fit_file.unlink()
                    deleted_counts['fit_files'] += 1
                    logger.info(f"Deleted FIT file: {fit_file}")
                except Exception as e:
                    logger.error(f"Failed to delete FIT file {fit_file}: {e}")
            
            # 2. Clear file registry
            registry_count = len(self.file_registry)
            self.file_registry.clear()
            logger.info(f"Cleared file registry ({registry_count} entries)")
            
            # 3. Clear session state
            session_count = len(st.session_state.uploaded_files)
            st.session_state.uploaded_files.clear()
            logger.info(f"Cleared session cache ({session_count} entries)")
            
            # 4. Clear ride history
            if not self.ride_history.empty:
                deleted_counts['history_entries'] = len(self.ride_history)
                self.ride_history = pd.DataFrame()
                logger.info(f"Cleared ride history ({deleted_counts['history_entries']} entries)")
            
            # 5. Clear analysis history
            if not self.analysis_history.empty:
                deleted_counts['analysis_entries'] = len(self.analysis_history)
                self.analysis_history = pd.DataFrame()
                logger.info(f"Cleared analysis history ({deleted_counts['analysis_entries']} entries)")
            
            # 6. Clear all figures (except keep directory structure)
            for figure_file in self.figures_dir.glob("*"):
                if figure_file.is_file():
                    try:
                        figure_file.unlink()
                        deleted_counts['figures'] += 1
                        logger.info(f"Deleted figure: {figure_file}")
                    except Exception as e:
                        logger.error(f"Failed to delete figure {figure_file}: {e}")
            
            # 7. Save updated data
            self.save_data()
            
            # Create summary message
            summary_parts = []
            if deleted_counts['fit_files'] > 0:
                summary_parts.append(f"{deleted_counts['fit_files']} FIT files")
            if deleted_counts['history_entries'] > 0:
                summary_parts.append(f"{deleted_counts['history_entries']} history entries")
            if deleted_counts['analysis_entries'] > 0:
                summary_parts.append(f"{deleted_counts['analysis_entries']} analysis entries")
            if deleted_counts['figures'] > 0:
                summary_parts.append(f"{deleted_counts['figures']} figure files")
            
            if summary_parts:
                message = f"✅ Successfully cleared all rides. Removed: {', '.join(summary_parts)}. Backup created: {backup_path}"
                logger.info(message)
                return True, message
            else:
                message = "⚠️ No rides found to clear"
                logger.warning(message)
                return False, message
                
        except Exception as e:
            error_msg = f"❌ Error clearing all rides: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def clear_cache(self, cache_type: str = "all") -> Tuple[bool, str]:
        """
        Clear different types of cache with selective options.
        
        Args:
            cache_type: Type of cache to clear ("fit_files", "session", "analysis", "all")
            
        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            deleted_items = []
            
            if cache_type in ["fit_files", "all"]:
                # Clear FIT files from cache directory
                fit_files_cleared = 0
                for fit_file in self.cache_dir.glob("*.fit"):
                    try:
                        fit_file.unlink()
                        fit_files_cleared += 1
                        logger.info(f"Deleted cached FIT file: {fit_file.name}")
                    except Exception as e:
                        logger.error(f"Failed to delete {fit_file}: {e}")
                
                if fit_files_cleared > 0:
                    deleted_items.append(f"{fit_files_cleared} FIT files")
                
                # Clear file registry entries for deleted files
                registry_cleared = 0
                for ride_id in list(self.file_registry.keys()):
                    file_path = self.file_registry[ride_id]['file_path']
                    if not Path(file_path).exists():
                        del self.file_registry[ride_id]
                        registry_cleared += 1
                
                if registry_cleared > 0:
                    deleted_items.append(f"{registry_cleared} registry entries")
            
            if cache_type in ["session", "all"]:
                # Clear session state cache
                session_count = len(st.session_state.uploaded_files)
                st.session_state.uploaded_files.clear()
                if session_count > 0:
                    deleted_items.append(f"{session_count} session entries")
                logger.info(f"Cleared session cache ({session_count} entries)")
            
            if cache_type in ["analysis", "all"]:
                # Clear analysis cache
                analysis_count = len(st.session_state.analysis_cache)
                st.session_state.analysis_cache.clear()
                if analysis_count > 0:
                    deleted_items.append(f"{analysis_count} analysis cache entries")
                logger.info(f"Cleared analysis cache ({analysis_count} entries)")
            
            # Save updated data
            self.save_data()
            
            if deleted_items:
                message = f"Cache cleared successfully: {', '.join(deleted_items)}"
                logger.info(message)
                return True, message
            else:
                return True, "No cache items found to clear"
                
        except Exception as e:
            error_msg = f"Error clearing cache: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get detailed information about cache usage.
        
        Returns:
            Dict[str, Any]: Cache information including sizes and counts
        """
        try:
            cache_info = {
                'fit_files_count': 0,
                'fit_files_size': 0,
                'session_entries': len(st.session_state.uploaded_files),
                'analysis_cache_entries': len(st.session_state.analysis_cache),
                'registry_entries': len(self.file_registry)
            }
            
            # Calculate FIT file cache info
            for fit_file in self.cache_dir.glob("*.fit"):
                cache_info['fit_files_count'] += 1
                cache_info['fit_files_size'] += fit_file.stat().st_size
            
            # Convert bytes to MB for readability
            cache_info['fit_files_size_mb'] = cache_info['fit_files_size'] / (1024 * 1024)
            
            return cache_info
            
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {}
    
    def load_settings(self) -> Dict[str, Any]:
        """
        Load user settings from JSON file.
        
        Returns:
            Dict[str, Any]: Dictionary containing user settings
        """
        default_settings = {
            'ftp': 250,
            'lthr': 160,
            'max_hr': 195,
            'rest_hr': 51,
            'weight_kg': 70,
            'height_cm': 175,
            'athlete_name': 'Cyclist'
        }
        
        try:
            if self.settings_path.exists():
                with open(self.settings_path, 'r') as f:
                    saved_settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    settings = {**default_settings, **saved_settings}
                    logger.info(f"Loaded settings from {self.settings_path}")
                    return settings
            else:
                logger.info("No settings file found, using defaults")
                return default_settings
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return default_settings
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Save user settings to JSON file.
        
        Args:
            settings: Dictionary containing user settings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            logger.info(f"Settings saved to {self.settings_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a specific setting value.
        
        Args:
            key: Setting key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            Any: Setting value or default
        """
        settings = self.load_settings()
        return settings.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> bool:
        """
        Set a specific setting value.
        
        Args:
            key: Setting key to set
            value: Value to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        settings = self.load_settings()
        settings[key] = value
        return self.save_settings(settings)