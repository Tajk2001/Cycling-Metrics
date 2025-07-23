import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import tempfile
import os
import hashlib
import shutil
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CyclingDataManager:
    """
    Comprehensive data manager for cycling analysis with error-proofing and clean storage.
    Handles FIT file storage, analysis results, caching, and data retrieval.
    """
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "cache"):
        """Initialize the data manager with organized storage structure."""
        self.data_dir = pathlib.Path(data_dir)
        self.cache_dir = pathlib.Path(cache_dir)
        self.figures_dir = pathlib.Path("figures")
        
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
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [self.data_dir, self.cache_dir, self.figures_dir]
        for directory in directories:
            directory.mkdir(exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def _init_session_state(self):
        """Initialize Streamlit session state with default values."""
        if 'data_manager_initialized' not in st.session_state:
            st.session_state.data_manager_initialized = True
            st.session_state.uploaded_files = {}
            st.session_state.analysis_cache = {}
            st.session_state.current_analysis = None
            st.session_state.error_messages = []
            st.session_state.success_messages = []
    
    def _load_existing_data(self):
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
    
    def _create_backup_and_reset(self):
        """Create backup of corrupted files and reset them."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.data_dir / f"backup_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        # Backup corrupted files
        for file_path in [self.ride_history_path, self.analysis_history_path, self.file_registry_path]:
            if file_path.exists():
                try:
                    shutil.copy2(file_path, backup_dir / file_path.name)
                    logger.info(f"Backed up {file_path} to {backup_dir}")
                except Exception as e:
                    logger.error(f"Failed to backup {file_path}: {e}")
        
        # Reset to empty data structures
        self.ride_history = pd.DataFrame()
        self.analysis_history = pd.DataFrame()
        self.file_registry = {}
        
        st.warning(f"⚠️ Data files were corrupted and have been reset. Backup created in {backup_dir}")
    
    def save_data(self):
        """Save all data with error handling."""
        try:
            # Save ride history
            if not self.ride_history.empty:
                self.ride_history.to_csv(self.ride_history_path, index=False)
                logger.info("Saved ride history")
            
            # Save analysis history
            if not self.analysis_history.empty:
                self.analysis_history.to_csv(self.analysis_history_path, index=False)
                logger.info("Saved analysis history")
            
            # Save file registry
            with open(self.file_registry_path, 'w') as f:
                json.dump(self.file_registry, f, indent=2)
            logger.info("Saved file registry")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            st.error(f"❌ Failed to save data: {e}")
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
    
    def _calculate_file_hash(self, file_path: pathlib.Path) -> str:
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
            if pathlib.Path(file_path).exists():
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
            current_hash = self._calculate_file_hash(pathlib.Path(file_path))
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
            export_dir = pathlib.Path(export_path)
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
            import_dir = pathlib.Path(import_path)
            
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