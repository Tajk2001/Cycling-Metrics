"""
Simple Bulk Training Pipeline for Cycling Interval Detection
==========================================================

This simplified pipeline works with the existing IntervalML.py code to train
on multiple FIT files from a single folder.

Usage:
    python simple_bulk_training.py
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

# Import the core functionality
from SprintV1 import load_fit_to_dataframe
from IntervalML import (
    IntervalDetector, 
    extract_lap_intervals_from_data,
    find_workout_boundaries,
    clean_ride_data
)

class SimpleBulkTraining:
    """Simple bulk training pipeline that works with existing code."""
    
    def __init__(self, output_dir="trained_models"):
        """Initialize the simple bulk training pipeline."""
        self.output_dir = output_dir
        self.detector = IntervalDetector()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training data storage
        self.all_features = []
        self.all_labels = []
        self.file_results = []
        
        print(f"🚀 Simple Bulk Training Pipeline Initialized")
        print(f"   📊 FTP: Dynamic estimation per ride")
        print(f"   📁 Output: {output_dir}")
    
    def load_fit_files_from_folder(self, folder_path):
        """Load all FIT files from a specified folder."""
        print(f"\n📁 Loading FIT files from: {folder_path}")
        
        # Find all FIT files (case insensitive, including files without extensions)
        fit_patterns = ["*.fit", "*.FIT", "*.Fit", "Training*", "training*"]
        fit_files = []
        
        for pattern in fit_patterns:
            fit_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        
        # Remove duplicates
        fit_files = list(set(fit_files))
        
        if not fit_files:
            print(f"❌ No FIT files found in {folder_path}")
            return []
        
        print(f"✅ Found {len(fit_files)} FIT files")
        
        # Load each FIT file
        loaded_data = []
        for fit_file in sorted(fit_files):
            try:
                print(f"   📥 Loading: {os.path.basename(fit_file)}")
                df = load_fit_to_dataframe(fit_file)
                
                if df is not None and len(df) > 0:
                    # Add file info
                    df['file_path'] = fit_file
                    df['file_name'] = os.path.basename(fit_file)
                    loaded_data.append(df)
                    print(f"      ✅ Loaded {len(df)} data points")
                else:
                    print(f"      ❌ Empty or invalid data")
                    
            except Exception as e:
                print(f"      ❌ Error loading {os.path.basename(fit_file)}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(loaded_data)} FIT files")
        return loaded_data
    
    def process_file(self, df):
        """Process a single FIT file to extract training data."""
        print(f"   🔄 Processing: {df['file_name'].iloc[0]}")
        
        try:
            # Clean the data
            df_clean = clean_ride_data(df)
            
            # Try to extract lap intervals first
            lap_intervals = extract_lap_intervals_from_data(df_clean)
            
            if lap_intervals and len(lap_intervals) > 0:
                print(f"      📊 Found {len(lap_intervals)} lap intervals")
                return self._process_with_laps(df_clean, lap_intervals)
            else:
                # Fall back to workout boundary detection
                print(f"      🔍 No laps found, using workout boundaries...")
                workout_boundaries = find_workout_boundaries(df_clean)
                
                if workout_boundaries and len(workout_boundaries) > 0:
                    print(f"      📊 Found {len(workout_boundaries)} workout boundaries")
                    return self._process_with_workout_boundaries(df_clean, workout_boundaries)
                else:
                    print(f"      ⚠️ No workout boundaries found, skipping")
                    return None, None, None
                    
        except Exception as e:
            print(f"      ❌ Error processing file: {e}")
            return None, None, None
    
    def _process_with_laps(self, df, lap_intervals):
        """Process file using lap intervals."""
        # Create training data from laps
        X, y = self.detector.create_training_data(df, lap_intervals)
        
        if X is not None and y is not None:
            # Get feature names
            feature_names = [col for col in df.columns if col not in ['power_zone', 'file_path', 'file_name']]
            return X, y, feature_names
        
        return None, None, None
    
    def _process_with_workout_boundaries(self, df, workout_boundaries):
        """Process file using workout boundaries."""
        # Convert workout boundaries to interval format
        intervals = []
        for start, end in workout_boundaries:
            intervals.append((start, end))
        
        # Create training data
        X, y = self.detector.create_training_data(df, intervals)
        
        if X is not None and y is not None:
            # Get feature names
            feature_names = [col for col in df.columns if col not in ['power_zone', 'file_path', 'file_name']]
            return X, y, feature_names
        
        return None, None, None
    
    def train_comprehensive_model(self, X, y, num_features):
        """Train a comprehensive model on all the data."""
        print(f"\n🤖 Training Comprehensive Model")
        print(f"=" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        print(f"📊 Class distribution: {np.bincount(y_train.astype(int))}")
        
        # Ensure all features are numeric
        print(f"🔍 Checking feature data types...")
        
        # Clean and convert features to numeric, handling any non-numeric data
        def clean_features(X):
            # Convert to pandas DataFrame for easier handling
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X)
            else:
                X_df = X.copy()
            
            # Replace any non-numeric values with 0
            for col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
            
            return X_df.values
        
        X_train_clean = clean_features(X_train)
        X_test_clean = clean_features(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test_clean)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Create feature names 
        feature_names = [f'feature_{i}' for i in range(num_features)]
        
        results = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'test_accuracy': accuracy,
            'classification_report': report,
            'feature_importance': dict(zip(feature_names, model.feature_importances_))
        }
        
        print(f"✅ Model training complete!")
        print(f"   📊 Test accuracy: {accuracy:.3f}")
        if '1' in report:
            print(f"   📊 Precision (intervals): {report['1']['precision']:.3f}")
            print(f"   📊 Recall (intervals): {report['1']['recall']:.3f}")
            print(f"   📊 F1-score (intervals): {report['1']['f1-score']:.3f}")
        
        return results
    
    def save_model(self, training_results, filename=None):
        """Save the trained model."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_comprehensive_model_{timestamp}.pkl"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Save model and scaler
        model_data = {
            'model': training_results['model'],
            'scaler': training_results['scaler'],
            'feature_names': training_results['feature_names'],
                    'training_info': {
            'ftp_method': 'Dynamic estimation per ride',
            'total_files': len(self.file_results),
            'total_samples': len(self.all_features),
            'test_accuracy': training_results['test_accuracy'],
            'timestamp': datetime.now().isoformat()
        }
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")
        
        return filepath
    
    def run_bulk_training(self, folder_path):
        """Run the complete bulk training process."""
        print(f"🚀 Starting Simple Bulk Training Process")
        print(f"=" * 60)
        
        # Step 1: Load all FIT files
        loaded_data = self.load_fit_files_from_folder(folder_path)
        
        if not loaded_data:
            print("❌ No valid FIT files found. Exiting.")
            return None
        
        # Step 2: Process each file
        for i, df in enumerate(loaded_data):
            print(f"\n🔄 Processing file {i+1}/{len(loaded_data)}")
            
            X, y, feature_names = self.process_file(df)
            
            if X is not None and y is not None:
                # Store results
                self.all_features.append(X)
                self.all_labels.append(y)
                
                self.file_results.append({
                    'file_name': df['file_name'].iloc[0],
                    'file_path': df['file_path'].iloc[0],
                    'sample_count': len(X),
                    'feature_count': len(feature_names)
                })
                
                print(f"      ✅ Processed {len(X)} samples")
            else:
                print(f"      ⚠️ No training data extracted")
        
        if not self.all_features:
            print("❌ No training data could be extracted. Exiting.")
            return None
        
        # Step 3: Combine all data
        print(f"\n🔧 Combining training data from {len(self.all_features)} files...")
        
        # Ensure all feature matrices have the same number of features
        feature_counts = [X.shape[1] for X in self.all_features]
        max_features = max(feature_counts)
        min_features = min(feature_counts)
        
        print(f"📊 Feature analysis:")
        print(f"   Min features: {min_features}")
        print(f"   Max features: {max_features}")
        
        if max_features != min_features:
            print(f"   ⚠️ Feature count mismatch detected, standardizing to {min_features} features")
            
            # Standardize all feature matrices to minimum size
            standardized_features = []
            for X in self.all_features:
                if X.shape[1] > min_features:
                    # Truncate to minimum size
                    X_truncated = X[:, :min_features]
                    standardized_features.append(X_truncated)
                else:
                    # Keep as is
                    standardized_features.append(X)
            
            self.all_features = standardized_features
            
            # Verify all have same dimensions
            feature_counts = [X.shape[1] for X in self.all_features]
            print(f"   ✅ Standardized feature counts: {feature_counts}")
        
        X_combined = np.vstack(self.all_features)
        y_combined = np.concatenate(self.all_labels)
        
        print(f"📊 Combined dataset:")
        print(f"   Total samples: {len(X_combined)}")
        print(f"   Total features: {X_combined.shape[1]}")
        print(f"   Class distribution: {np.bincount(y_combined.astype(int))}")
        
        # Step 4: Train comprehensive model
        training_results = self.train_comprehensive_model(X_combined, y_combined,
                                                       X_combined.shape[1])
        
        # Step 5: Save model
        model_path = self.save_model(training_results)
        
        # Summary
        print(f"\n🎉 Simple Bulk Training Complete!")
        print(f"=" * 60)
        print(f"📊 Summary:")
        print(f"   Files processed: {len(self.file_results)}")
        print(f"   Total samples: {len(X_combined)}")
        print(f"   Final accuracy: {training_results['test_accuracy']:.3f}")
        print(f"   Model saved to: {model_path}")
        
        return {
            'files_processed': len(self.file_results),
            'total_samples': len(X_combined),
            'final_accuracy': training_results['test_accuracy'],
            'model_path': model_path
        }


def main():
    """Main function to run simple bulk training."""
    print("🏋️ Simple Bulk Training Pipeline for Cycling Interval Detection")
    print("=" * 70)
    
    # Get folder path from user
    folder_path = input("Enter the path to your FIT files folder: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Initialize pipeline
    pipeline = SimpleBulkTraining()
    
    # Run bulk training
    results = pipeline.run_bulk_training(folder_path)
    
    if results:
        print(f"\n🎯 Your comprehensive model is ready!")
        print(f"💡 You can now use this model to detect intervals in new rides.")
    else:
        print(f"\n❌ Training failed. Please check your FIT files and try again.")


if __name__ == "__main__":
    main()
