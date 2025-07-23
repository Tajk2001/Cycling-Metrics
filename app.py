import fitparse as fit
import pandas as pd
import numpy as np
import pathlib

def run_basic_analysis(fit_path, ftp=250, lthr=160, save_figures=False, analysis_id=None):
    """
    Run basic analysis on a FIT file.
    
    Args:
        fit_path (str): Path to the .fit file
        ftp (int): Functional Threshold Power in watts
        lthr (int): Lactate Threshold Heart Rate in bpm
        save_figures (bool): Whether to save figures to disk
        analysis_id (str): Unique identifier for this analysis
    
    Returns:
        tuple: (results_dict, dataframe) or (None, None) if no data
    """
    try:
        # Load FIT file
        fitfile = fit.FitFile(fit_path)
        records = []
        
        # Extract record data
        for record in fitfile.get_messages('record'):
            record_data = {field.name: field.value for field in record}
            records.append(record_data)
        
        if not records:
            return None, None
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        # --- METRIC CONVERSIONS ---
        if 'speed' in df.columns:
            df['speed_kmh'] = df['speed'] * 3.6
        
        if 'distance' in df.columns:
            df['distance_km'] = df['distance'] / 1000
        
        if 'altitude' in df.columns:
            df['altitude_m'] = df['altitude']
        
        if 'temperature' in df.columns:
            df['temperature_C'] = df['temperature']
        
        # Handle gear data
        for col in ['front_gear', 'rear_gear']:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # --- DATA CLEANING ---
        continuous_cols = ['power', 'cadence', 'heart_rate', 'speed_kmh', 'altitude_m', 'distance_km', 'temperature_C', 'torque']
        for col in continuous_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        # --- SCIENTIFIC SMOOTHING ---
        smoothing_windows = {
            'power': 30,
            'heart_rate': 30,
            'cadence': 30,
            'speed_kmh': 10,
            'altitude_m': 10,
            'temperature_C': 10,
            'torque': 10
        }
        
        for col in continuous_cols:
            if col in df.columns:
                window = smoothing_windows.get(col, 10)
                df[col + '_smoothed'] = df[col].rolling(window=window, min_periods=1, center=True).mean()
        
        # --- ADVANCED METRICS ---
        if 'power' in df.columns:
            rolling_power_30s = df['power'].rolling(window=30, min_periods=1).mean()
            np_calc = (rolling_power_30s ** 4).mean() ** 0.25
            np_display = int(np.round(np_calc))
        else:
            np_calc = np.nan
            np_display = np.nan
        
        IF = np_calc / ftp if not np.isnan(np_calc) else np.nan
        IF_display = int(np.round(IF * 100)) if not np.isnan(IF) else np.nan
        
        # Calculate TSS
        if 'power' in df.columns and 'timestamp' in df.columns:
            duration_sec = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
            TSS = ((duration_sec * np_calc * IF) / (ftp * 3600)) * 100
            TSS_display = int(np.round(TSS))
        else:
            TSS = np.nan
            TSS_display = np.nan
        
        # Calculate elevation gain
        if 'altitude_m' in df.columns:
            elevation_gain = df['altitude_m'].diff()
            elevation_gain_sum = elevation_gain[elevation_gain > 0].sum()
            elevation_gain_display = int(np.round(elevation_gain_sum))
        else:
            elevation_gain_display = np.nan
        
        # --- ADVANCED METRICS DICT ---
        advanced_metrics = {}
        if not np.isnan(np_display):
            advanced_metrics["Normalized Power (W)"] = np_display
        if not np.isnan(IF_display):
            advanced_metrics["Intensity Factor (%)"] = IF_display
        if not np.isnan(TSS_display):
            advanced_metrics["Training Stress Score"] = TSS_display
        if not np.isnan(elevation_gain_display):
            advanced_metrics["Elevation Gain (m)"] = elevation_gain_display
        
        # --- SESSION SUMMARY ---
        summary = {}
        if 'power' in df.columns:
            summary['Avg Power (W)'] = int(np.round(df['power'].mean()))
            summary['Max Power (W)'] = int(np.round(df['power'].max()))
        
        if 'heart_rate' in df.columns:
            summary['Avg HR (bpm)'] = int(np.round(df['heart_rate'].mean()))
            summary['Max HR (bpm)'] = int(np.round(df['heart_rate'].max()))
        
        if 'cadence' in df.columns:
            summary['Avg Cadence (rpm)'] = int(np.round(df['cadence'].mean()))
            summary['Max Cadence (rpm)'] = int(np.round(df['cadence'].max()))
        
        if 'speed_kmh' in df.columns:
            summary['Avg Speed (km/h)'] = int(np.round(df['speed_kmh'].mean()))
            summary['Max Speed (km/h)'] = int(np.round(df['speed_kmh'].max()))
        
        if 'temperature_C' in df.columns:
            summary['Avg Temp (°C)'] = int(np.round(df['temperature_C'].mean()))
            summary['Max Temp (°C)'] = int(np.round(df['temperature_C'].max()))
        
        if 'torque' in df.columns:
            summary['Avg Torque (Nm)'] = int(np.round(df['torque'].mean()))
            summary['Max Torque (Nm)'] = int(np.round(df['torque'].max()))
        
        # Duration and distance
        if 'timestamp' in df.columns:
            duration_sec = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
            summary['Duration (min)'] = int(np.round(duration_sec / 60))
        
        if 'distance_km' in df.columns:
            summary['Total Distance (km)'] = int(np.round(df['distance_km'].max()))
        
        # --- POWER ZONES ---
        zone_df = None
        if 'power' in df.columns and ftp > 0:
            zones = {
                'Z1 (Recovery)': (0, ftp * 0.55),
                'Z2 (Endurance)': (ftp * 0.55, ftp * 0.75),
                'Z3 (Tempo)': (ftp * 0.75, ftp * 0.9),
                'Z4 (Threshold)': (ftp * 0.9, ftp * 1.05),
                'Z5 (VO2max)': (ftp * 1.05, ftp * 1.2),
                'Z6 (Anaerobic)': (ftp * 1.2, ftp * 1.5),
                'Z7 (Neuromuscular)': (ftp * 1.5, float('inf'))
            }
            
            zone_counts = {}
            total_records = len(df)
            
            for zone_name, (min_power, max_power) in zones.items():
                count = len(df[(df['power'] >= min_power) & (df['power'] < max_power)])
                percentage = (count / total_records) * 100 if total_records > 0 else 0
                zone_counts[zone_name] = percentage
            
            zone_df = pd.DataFrame(list(zone_counts.items()), columns=['Zone', 'Percentage (%)'])
        
        # --- HR ZONES ---
        hr_zone_df = None
        if 'heart_rate' in df.columns and lthr > 0:
            hr_zones = {
                'Z1 (Recovery)': (0, lthr * 0.68),
                'Z2 (Endurance)': (lthr * 0.68, lthr * 0.83),
                'Z3 (Tempo)': (lthr * 0.83, lthr * 0.94),
                'Z4 (Threshold)': (lthr * 0.94, lthr * 1.05),
                'Z5 (VO2max)': (lthr * 1.05, lthr * 1.17),
                'Z6 (Anaerobic)': (lthr * 1.17, float('inf'))
            }
            
            hr_zone_counts = {}
            total_records = len(df)
            
            for zone_name, (min_hr, max_hr) in hr_zones.items():
                count = len(df[(df['heart_rate'] >= min_hr) & (df['heart_rate'] < max_hr)])
                percentage = (count / total_records) * 100 if total_records > 0 else 0
                hr_zone_counts[zone_name] = percentage
            
            hr_zone_df = pd.DataFrame(list(hr_zone_counts.items()), columns=['Zone', 'Percentage (%)'])
        
        # Save figures if requested
        if save_figures and analysis_id:
            try:
                import matplotlib.pyplot as plt
                import os
                
                # Create figures directory if it doesn't exist
                os.makedirs("figures", exist_ok=True)
                
                # Create a simple summary plot
                if 'power' in df.columns and 'timestamp' in df.columns:
                    plt.figure(figsize=(12, 8))
                    
                    # Power over time
                    plt.subplot(2, 2, 1)
                    plt.plot(df['timestamp'], df['power'], alpha=0.7, label='Power')
                    if 'power_smoothed' in df.columns:
                        plt.plot(df['timestamp'], df['power_smoothed'], 'r-', linewidth=2, label='Smoothed')
                    plt.title('Power Over Time')
                    plt.xlabel('Time')
                    plt.ylabel('Power (W)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Heart rate if available
                    if 'heart_rate' in df.columns:
                        plt.subplot(2, 2, 2)
                        plt.plot(df['timestamp'], df['heart_rate'], alpha=0.7, color='red', label='Heart Rate')
                        if 'heart_rate_smoothed' in df.columns:
                            plt.plot(df['timestamp'], df['heart_rate_smoothed'], 'r-', linewidth=2, label='Smoothed')
                        plt.title('Heart Rate Over Time')
                        plt.xlabel('Time')
                        plt.ylabel('Heart Rate (bpm)')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                    
                    # Power zones if available
                    if zone_df is not None:
                        plt.subplot(2, 2, 3)
                        zone_df.plot(x='Zone', y='Percentage (%)', kind='bar', ax=plt.gca())
                        plt.title('Power Zone Distribution')
                        plt.xlabel('Zone')
                        plt.ylabel('Percentage (%)')
                        plt.xticks(rotation=45)
                        plt.grid(True, alpha=0.3)
                    
                    # Speed if available
                    if 'speed_kmh' in df.columns:
                        plt.subplot(2, 2, 4)
                        plt.plot(df['timestamp'], df['speed_kmh'], alpha=0.7, color='green', label='Speed')
                        if 'speed_kmh_smoothed' in df.columns:
                            plt.plot(df['timestamp'], df['speed_kmh_smoothed'], 'g-', linewidth=2, label='Smoothed')
                        plt.title('Speed Over Time')
                        plt.xlabel('Time')
                        plt.ylabel('Speed (km/h)')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(f"figures/{analysis_id}_basic_analysis.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    
            except Exception as e:
                print(f"Warning: Could not save figures: {e}")
        
        return {
            'summary': summary,
            'advanced_metrics': advanced_metrics,
            'zone_df': zone_df,
            'hr_zone_df': hr_zone_df
        }, df
        
    except Exception as e:
        print(f"Error analyzing FIT file: {e}")
        return None, None

# TODO: Confirmed that run_basic_analysis is robust to missing columns and is importable by main_dashboard.py. No Streamlit UI code remains. All necessary imports are present. No further action needed unless new requirements arise.
