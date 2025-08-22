import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
import matplotlib.pyplot as plt

# Data I/O for FIT
try:
    from fitparse import FitFile
except Exception as e:
    FitFile = None
    warnings.warn("fitparse not available. Install with: pip install fitparse")

# Interactive plotting
try:
    import plotly.graph_objects as go
except Exception as e:
    go = None
    warnings.warn("plotly not available. Install with: pip install plotly")

# Modeling
try:
    from sklearn.linear_model import LinearRegression
except Exception as e:
    LinearRegression = None
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")

# Optional smoothing
try:
    from scipy.signal import savgol_filter
except Exception:
    def savgol_filter(x, *args, **kwargs):
        return np.asarray(x)

pd.options.mode.copy_on_write = True

# User Parameters
file_path = "/Users/tajkrieger/Desktop/Training/Training5"
FTP = 290
LTHR = 181
rider_mass_kg = 52
crank_length_mm = 165
speed_threshold = 0.5

# ML-Based Interval Detection Parameters
max_intervals_to_analyze = 15   # Maximum number of intervals to analyze in detail


def load_fit_to_dataframe(path):
    """Load FIT file and extract records with lap information."""
    if FitFile is None:
        raise ImportError("fitparse is not installed. Please install via `pip install fitparse`.")

    fitfile = FitFile(path)
    
    # Extract records
    recs = []
    for msg in fitfile.get_messages('record'):
        row = {}
        for f in msg:
            row[f.name] = f.value
        recs.append(row)
    
    df = pd.DataFrame(recs)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    df = df.dropna(axis=1, how='all')

    # Extract laps
    laps = []
    for lap in fitfile.get_messages('lap'):
        l = {}
        for f in lap:
            l[f.name] = f.value
        if 'start_time' in l:
            l['start_time'] = pd.to_datetime(l['start_time'])
        if 'timestamp' in l:
            l['end_time'] = pd.to_datetime(l['timestamp'])
        elif 'total_elapsed_time' in l and 'start_time' in l:
            l['end_time'] = l['start_time'] + pd.to_timedelta(l.get('total_elapsed_time', 0), unit='s')
        laps.append(l)

    lap_df = pd.DataFrame(laps)
    if not lap_df.empty and {'start_time','end_time'}.issubset(lap_df.columns):
        df['lap'] = np.nan
        for i, row in lap_df.iterrows():
            st = row['start_time']
            et = row['end_time']
            if pd.notna(st) and pd.notna(et):
                df.loc[(df.index >= st) & (df.index <= et), 'lap'] = i + 1
        if df['lap'].isna().all():
            df['lap'] = 1
        else:
            df['lap'] = df['lap'].ffill().bfill()
    else:
        df['lap'] = 1

    return df


def apply_hard_limits_and_smooth(df):
    """Apply hard limits, smooth data, and compute derived fields."""
    df = df.copy()
    limits = {
        "heart_rate": (30, 230),
        "cadence": (10, 220),
        "temperature": (-10, 60),
        "speed": (-1, 50),
        "speed_kph": (-1, 140),
        "grade": (-40, 40),
        "power": (-1, 2000),
        "altitude": (-100, 6000),
        "torque": (-1, 150)
    }

    # Compute speed_kph if needed
    if "speed" in df.columns and "speed_kph" not in df.columns:
        df["speed_kph"] = df["speed"] * 3.6

    # Create enhanced_speed column (prefer speed_kph, fallback to speed, then speed_kph)
    if "speed_kph" in df.columns:
        df["enhanced_speed"] = df["speed_kph"] / 3.6  # Convert back to m/s for consistency
    elif "speed" in df.columns:
        df["enhanced_speed"] = df["speed"]
    else:
        df["enhanced_speed"] = np.nan

    # Compute torque (Nm) if power & cadence exist
    if "power" in df.columns and "cadence" in df.columns:
        df["torque"] = np.where(df["cadence"] > 0,
                                (60.0 * df["power"]) / (2*np.pi * df["cadence"]),
                                np.nan)

    # Apply limits & flags
    for col, (lo, hi) in limits.items():
        if col in df.columns:
            bad = (df[col] < lo) | (df[col] > hi)
            df.loc[bad, col] = np.nan
            df[f"{col}_out_of_range"] = bad.astype(int)

    # Smoothing function
    def smooth(series, method="rolling", window=21, polyorder=2):
        if series.isna().any() or len(series) < window or window % 2 == 0:
            return series
        if method == "savgol":
            return pd.Series(savgol_filter(series, window_length=window, polyorder=polyorder),
                             index=series.index)
        return series.rolling(window=window, center=True).mean()

    # Apply smoothing
    plan = {
        "power": ("savgol", 21),
        "heart_rate": ("rolling", 31),
        "cadence": ("rolling", 11),
        "speed_kph": ("rolling", 15),
        "grade": ("rolling", 31),
        "altitude": ("rolling", 31),
        "torque": ("rolling", 21),
    }
    for col, (m, w) in plan.items():
        if col in df.columns:
            df[f"{col}_smoothed"] = smooth(df[col], method=m, window=w)

    # Time columns
    df["time_sec"] = (df.index - df.index[0]).total_seconds()
    df["time_min"] = df["time_sec"] / 60.0
    df["time_str"] = df["time_sec"].apply(lambda x: f"{int(x//60):02d}:{int(x%60):02d}")
    
    return df


def plot_torque_vs_rpm(df):
    """Create interactive torque vs RPM plot colored by power."""
    if go is None:
        print("Plotly not installed; cannot show interactive plot.")
        return
    
    need = {"torque", "cadence", "power"}
    if not need.issubset(df.columns):
        print("Missing:", need - set(df.columns))
        return
    
    data = df[(df["cadence"] > 10) & df["torque"].notna()].copy()

    fig = go.Figure(go.Scatter(
        x=data["cadence"], y=data["torque"], mode='markers',
        marker=dict(size=5, color=data["power"], colorscale='Viridis',
                    colorbar=dict(title="Power (W)")),
        hovertemplate=("Cadence: %{x:.1f} rpm<br>"
                       "Torque: %{y:.2f} Nm<br>"
                       "Power: %{marker.color:.0f} W<br>"
                       "Time: %{text}"),
        text=data["time_str"]
    ))
    
    fig.update_layout(
        title="Torque vs RPM (colored by Power)",
        xaxis=dict(title="Cadence (rpm)", range=[0, 200]),
        yaxis=dict(title="Torque (Nm)", range=[0, 120]),
        template="plotly_white",
        height=520
    )
    # Save to HTML file instead of showing
    # fig.write_html("torque_vs_rpm.html")  # Removed file generation
    # print("Plot saved to torque_vs_rpm.html")  # Plots shown in dashboard instead


def best_10s_windows(df, *, n_best=3, win=10, min_gap=30, min_cad=40, max_cad=140):
    """Return up to n_best highest-power 10-s windows with no overlap."""
    cols_out = ['Start', 'Power (W)', 'Torque (Nm)', 'Cadence (rpm)', 'ω (rad·s⁻¹)', 'Err %']
    need = {'power', 'torque', 'cadence'}
    
    if not need.issubset(df.columns):
        print("Missing columns:", need - set(df.columns))
        return pd.DataFrame(columns=cols_out)

    roll = df[['power', 'torque', 'cadence']].rolling(win, min_periods=win).mean()
    tmp = (roll.dropna()
                 .query(f'cadence.between({min_cad},{max_cad})')
                 .copy())
    
    if tmp.empty:
        print("No 10-s windows pass cadence/length criteria.")
        return pd.DataFrame(columns=cols_out)

    tmp['start_ts'] = tmp.index - pd.Timedelta(seconds=win-1)
    chosen = []
    
    while len(chosen) < n_best and not tmp.empty:
        idx = tmp['power'].idxmax()
        chosen.append(tmp.loc[idx])
        span_start = idx - pd.Timedelta(seconds=win-1)
        span_end = idx
        gap_start = span_start - pd.Timedelta(seconds=min_gap)
        gap_end = span_end + pd.Timedelta(seconds=min_gap)
        tmp = tmp[(tmp.index < gap_start) | (tmp.index > gap_end)]

    if not chosen:
        print("No sprint windows separated by the required gap.")
        return pd.DataFrame(columns=cols_out)

    best = (pd.DataFrame(chosen)
              .reset_index(drop=True)
              .rename(columns={'power': 'Power (W)',
                               'torque': 'Torque (Nm)',
                               'cadence': 'Cadence (rpm)'}))
    
    best['ω (rad·s⁻¹)'] = best['Cadence (rpm)'] * 2*np.pi / 60
    best['P_rebuild'] = best['Torque (Nm)'] * best['ω (rad·s⁻¹)']
    best['Err %'] = 100*(best['P_rebuild'] - best['Power (W)']) / best['Power (W)']
    best['Start'] = best['start_ts'].dt.strftime('%H:%M:%S')
    
    # Return both the display columns and the start_ts for analysis
    return best[['Start', 'Power (W)', 'Torque (Nm)', 'Cadence (rpm)', 'ω (rad·s⁻¹)', 'Err %', 'start_ts']]


def sprint_micro_metrics(df, start_ts, win=10):
    """Extract detailed micro-metrics for a sprint window with enhanced timing analysis."""
    seg = df.loc[start_ts : start_ts + pd.Timedelta(seconds=win-1)].copy()
    if seg.empty: 
        return None
    
    t = (seg.index - seg.index[0]).total_seconds()
    out = {}
    
    # Get the time interval between data points (assuming regular sampling)
    if len(seg) > 1:
        time_interval = (seg.index[1] - seg.index[0]).total_seconds()
    else:
        time_interval = 1.0  # Default to 1 second if only one data point
    
    # Enhanced timing analysis for each metric
    for k in ['power', 'torque', 'cadence']:
        if k not in seg.columns: 
            continue
        
        kmax_idx = seg[k].idxmax()
        kmax_time = (kmax_idx - seg.index[0]).total_seconds()
        
        out[f'{k}_max'] = seg[k].max()
        out[f'{k}_avg'] = seg[k].mean()
        out[f'{k}_ttp_s'] = kmax_time  # Time to peak
        out[f'{k}_fade_%'] = 100 * (seg[k].iloc[-2:].mean() - seg[k].max()) / seg[k].max()
        
        # Additional timing metrics
        out[f'{k}_peak_position_%'] = (kmax_time / win) * 100  # Where in the sprint the peak occurs
        out[f'{k}_rise_time_s'] = kmax_time  # Time to reach peak (same as TTP for now)
        
        # Calculate time to 90% of peak (acceleration phase)
        peak_value = seg[k].max()
        threshold_90 = peak_value * 0.9
        above_90 = seg[k] >= threshold_90
        if above_90.any():
            first_above_90_idx = above_90.idxmax()
            out[f'{k}_time_to_90%_s'] = (first_above_90_idx - seg.index[0]).total_seconds()
        else:
            out[f'{k}_time_to_90%_s'] = np.nan
        
        # Calculate acceleration rate (how quickly metric rises)
        if kmax_time > 0:
            out[f'{k}_acceleration_rate'] = peak_value / kmax_time  # units per second
        else:
            out[f'{k}_acceleration_rate'] = np.nan
        
        # Calculate time above 80% of peak (sustained effort)
        threshold_80 = peak_value * 0.8
        above_80 = seg[k] >= threshold_80
        if above_80.any():
            time_above_80 = above_80.sum() * time_interval
            out[f'{k}_time_above_80%_s'] = time_above_80
        else:
            out[f'{k}_time_above_80%_s'] = 0
    
    # Cross-metric timing analysis
    if all(k in seg.columns for k in ['power', 'torque', 'cadence']):
        # Time differences between peaks
        power_peak_time = out['power_ttp_s']
        torque_peak_time = out['torque_ttp_s']
        cadence_peak_time = out['cadence_ttp_s']
        
        out['torque_peak_delay_vs_power_s'] = torque_peak_time - power_peak_time
        out['cadence_peak_delay_vs_power_s'] = cadence_peak_time - power_peak_time
        out['cadence_peak_delay_vs_torque_s'] = cadence_peak_time - torque_peak_time
        
        # Peak sequence analysis
        peak_times = [('power', power_peak_time), ('torque', torque_peak_time), ('cadence', cadence_peak_time)]
        peak_times.sort(key=lambda x: x[1])
        out['peak_sequence'] = ' → '.join([p[0] for p in peak_times])
        
        # Synchronization score (how close peaks are together)
        time_spread = max(peak_times, key=lambda x: x[1])[1] - min(peak_times, key=lambda x: x[1])[1]
        out['peak_synchronization_s'] = time_spread
    
    # Enhanced work and efficiency metrics
    out['work_kJ'] = seg['power'].sum() / 1000
    out['impulse_Nm'] = seg['torque'].sum()
    out['rev_est'] = seg['cadence'].mean() / 60 * win
    out['start'] = start_ts.strftime('%H:%M:%S')
    
    # Power curve analysis
    if 'power' in seg.columns:
        power_curve = seg['power'].values
        # Time to reach 50%, 75%, 90% of max power
        for threshold in [0.5, 0.75, 0.9]:
            threshold_value = seg['power'].max() * threshold
            above_threshold = seg['power'] >= threshold_value
            if above_threshold.any():
                first_above_idx = above_threshold.idxmax()
                out[f'power_time_to_{int(threshold*100)}%_s'] = (first_above_idx - seg.index[0]).total_seconds()
            else:
                out[f'power_time_to_{int(threshold*100)}%_s'] = np.nan
    
    # Torque curve analysis
    if 'torque' in seg.columns:
        torque_curve = seg['torque'].values
        # Time to reach 50%, 75%, 90% of max torque
        for threshold in [0.5, 0.75, 0.9]:
            threshold_value = seg['torque'].max() * threshold
            above_threshold = seg['torque'] >= threshold_value
            if above_threshold.any():
                first_above_idx = above_threshold.idxmax()
                out[f'torque_time_to_{int(threshold*100)}%_s'] = (first_above_idx - seg.index[0]).total_seconds()
            else:
                out[f'torque_time_to_{int(threshold*100)}%_s'] = np.nan
    
    # Cadence curve analysis
    if 'cadence' in seg.columns:
        cadence_curve = seg['cadence'].values
        # Time to reach 50%, 75%, 90% of max cadence
        for threshold in [0.5, 0.75, 0.9]:
            threshold_value = seg['cadence'].max() * threshold
            above_threshold = seg['cadence'] >= threshold_value
            if above_threshold.any():
                first_above_idx = above_threshold.idxmax()
                out[f'cadence_time_to_{int(threshold*100)}%_s'] = (first_above_idx - seg.index[0]).total_seconds()
            else:
                out[f'cadence_time_to_{int(threshold*100)}%_s'] = np.nan
    
    # Speed analysis for sprints
    if 'enhanced_speed' in seg.columns:
        # Convert speed from m/s to km/h for easier interpretation
        speed_kph = seg['enhanced_speed'] * 3.6
        
        # Speed increase during sprint
        start_speed = speed_kph.iloc[0]
        end_speed = speed_kph.iloc[-1]
        speed_increase = end_speed - start_speed
        out['speed_increase_kph'] = speed_increase
        
        # Peak speed and timing
        max_speed = speed_kph.max()
        max_speed_idx = speed_kph.idxmax()
        time_to_max_speed = (max_speed_idx - seg.index[0]).total_seconds()
        out['max_speed_kph'] = max_speed
        out['speed_ttp_s'] = time_to_max_speed
        
        # Speed acceleration rate (km/h per second)
        if time_to_max_speed > 0:
            out['speed_acceleration_rate'] = max_speed / time_to_max_speed
        else:
            out['speed_acceleration_rate'] = np.nan
        
        # Speed consistency (coefficient of variation)
        speed_std = speed_kph.std()
        speed_cv = (speed_std / speed_kph.mean()) * 100 if speed_kph.mean() > 0 else np.nan
        out['speed_consistency_cv'] = speed_cv
        
        # Time above 90% of max speed (sustained high speed)
        threshold_90 = max_speed * 0.9
        above_90 = speed_kph >= threshold_90
        if above_90.any():
            time_above_90 = above_90.sum() * time_interval
            out['speed_time_above_90%_s'] = time_above_90
        else:
            out['speed_time_above_90%_s'] = 0
        
        # Speed vs Power timing relationship
        if 'power' in seg.columns:
            power_peak_time = out.get('power_ttp_s', 0)
            speed_power_timing_diff = time_to_max_speed - power_peak_time
            out['speed_power_timing_diff_s'] = speed_power_timing_diff
            
            # Classify sprint type based on timing
            if abs(speed_power_timing_diff) < 1.0:
                out['sprint_type'] = 'Synchronized'
            elif speed_power_timing_diff > 0:
                out['sprint_type'] = 'Power-First'
            else:
                out['sprint_type'] = 'Speed-First'
        
        # Distance covered during sprint
        distance_covered = seg['enhanced_speed'].sum()  # meters
        if distance_covered > 0:
            out['mechanical_efficiency_proxy'] = out['work_kJ'] / (distance_covered / 1000)  # kJ/km
            out['sprint_distance_m'] = distance_covered
        else:
            out['mechanical_efficiency_proxy'] = np.nan
            out['sprint_distance_m'] = 0
    else:
        out['mechanical_efficiency_proxy'] = np.nan
        # Set default values for speed metrics
        out['speed_increase_kph'] = np.nan
        out['max_speed_kph'] = np.nan
        out['speed_ttp_s'] = np.nan
        out['speed_acceleration_rate'] = np.nan
        out['speed_consistency_cv'] = np.nan
        out['speed_time_above_90%_s'] = np.nan
        out['speed_power_timing_diff_s'] = np.nan
        out['sprint_type'] = 'No Speed Data'
        out['sprint_distance_m'] = np.nan
    
    # Calculate comprehensive sprint efficiency score
    if all(k in out for k in ['power_acceleration_rate', 'power_time_above_80%_s', 'power_fade_%']):
        # Base score starts at 100
        efficiency_score = 100
        
        # Bonus for quick acceleration (faster = better)
        if not np.isnan(out['power_acceleration_rate']) and out['power_acceleration_rate'] > 0:
            # Bonus up to 20 points for very fast acceleration
            accel_bonus = min(20, out['power_acceleration_rate'] / 100)
            efficiency_score += accel_bonus
        
        # Bonus for sustained effort (longer time above 80% = better)
        if out['power_time_above_80%_s'] > 0:
            # Bonus up to 15 points for sustained effort
            sustain_bonus = min(15, out['power_time_above_80%_s'] / 2)
            efficiency_score += sustain_bonus
        
        # Penalty for power fade (less fade = better)
        if not np.isnan(out['power_fade_%']):
            # Penalty up to 25 points for excessive fade
            fade_penalty = min(25, abs(out['power_fade_%']) / 4)
            efficiency_score -= fade_penalty
        
        # Ensure score stays within reasonable bounds
        efficiency_score = max(0, min(150, efficiency_score))
        out['sprint_efficiency_score'] = round(efficiency_score, 1)
    
    return out


def sprint_summary(df, start_ts, win=10):
    """Comprehensive sprint summary with mechanical outputs and alternative metrics."""
    seg = df.loc[start_ts : start_ts + pd.Timedelta(seconds=win-1)].copy()
    if seg.empty: 
        return None
    
    out = {}
    
    # Basic timing
    out['start'] = start_ts.strftime('%H:%M:%S')
    
    # Mechanical outputs
    if 'power' in seg.columns:
        out['avg_power_10s'] = seg['power'].mean()
        out['max_power'] = seg['power'].max()
        out['work_kJ'] = seg['power'].sum() / 1000
        
        # Calculate mechanical efficiency proxy if speed data available
        if 'enhanced_speed' in seg.columns:
            distance_covered = seg['enhanced_speed'].sum()  # meters
            if distance_covered > 0:
                out['mechanical_efficiency_proxy'] = out['work_kJ'] / (distance_covered / 1000)  # kJ/km
            else:
                out['mechanical_efficiency_proxy'] = np.nan
        else:
            out['mechanical_efficiency_proxy'] = np.nan
    
    if 'torque' in seg.columns:
        out['peak_torque'] = seg['torque'].max()
        out['avg_torque'] = seg['torque'].mean()
    
    if 'cadence' in seg.columns:
        out['peak_cadence'] = seg['cadence'].max()
        out['avg_cadence'] = seg['cadence'].mean()
        out['peak_omega'] = out['peak_cadence'] * 2 * np.pi / 60  # rad/s
    
    # Power at peak torque vs peak cadence analysis
    if 'power' in seg.columns and 'torque' in seg.columns and 'cadence' in seg.columns:
        peak_torque_idx = seg['torque'].idxmax()
        peak_cadence_idx = seg['cadence'].idxmax()
        out['power_at_peak_torque'] = seg.loc[peak_torque_idx, 'power']
        out['power_at_peak_cadence'] = seg.loc[peak_cadence_idx, 'power']
        out['torque_early_vs_spin'] = out['power_at_peak_torque'] - out['power_at_peak_cadence']
    
    # Acceleration and speed metrics
    if 'enhanced_speed' in seg.columns:
        start_speed = seg['enhanced_speed'].iloc[0]
        end_speed = seg['enhanced_speed'].iloc[-1]
        out['delta_speed_kph'] = (end_speed - start_speed) * 3.6  # Convert m/s to km/h
        
        # Max acceleration (derivative of speed)
        if len(seg) > 1:
            speed_diff = np.diff(seg['enhanced_speed'])
            time_diff = np.diff((seg.index - seg.index[0]).total_seconds())
            accel = speed_diff / time_diff
            out['max_accel_ms2'] = accel.max()
        
        # Time to +5 km/h gain
        target_gain = 5 / 3.6  # Convert km/h to m/s
        speed_gain = seg['enhanced_speed'] - start_speed
        above_threshold = speed_gain >= target_gain
        if above_threshold.any():
            first_above = above_threshold.idxmax()
            out['time_to_5kph_s'] = (first_above - start_ts).total_seconds()
        else:
            out['time_to_5kph_s'] = np.nan
    
    # Time distribution analysis
    if 'power' in seg.columns and len(seg) >= 6:
        first_3s = seg.iloc[:3]
        last_3s = seg.iloc[-3:]
        out['first_3s_avg_power'] = first_3s['power'].mean()
        out['last_3s_avg_power'] = last_3s['power'].mean()
        out['burst_vs_endurance'] = out['first_3s_avg_power'] - out['last_3s_avg_power']
        
        # Work distribution
        first_half = seg.iloc[:len(seg)//2]
        second_half = seg.iloc[len(seg)//2:]
        first_half_work = first_half['power'].sum() / 1000
        second_half_work = second_half['power'].sum() / 1000
        total_work = out['work_kJ']
        out['first_half_contribution_%'] = (first_half_work / total_work) * 100 if total_work > 0 else np.nan
    
    # Fade analysis
    if 'power' in seg.columns:
        out['fade_%'] = 100 * (out['last_3s_avg_power'] - out['max_power']) / out['max_power']
    
    # Gear proxy metric
    if 'enhanced_speed' in seg.columns and 'cadence' in seg.columns:
        # m/rev = speed (m/s) / (cadence (rpm) / 60)
        gear_proxy = seg['enhanced_speed'] / (seg['cadence'] / 60)
        out['avg_gear_proxy'] = gear_proxy.mean()
        out['gear_proxy_range'] = gear_proxy.max() - gear_proxy.min()
        
        # Check for mid-sprint shifts (step changes in gear proxy)
        if len(gear_proxy) > 1:
            gear_diff = np.diff(gear_proxy)
            out['max_gear_shift'] = abs(gear_diff).max()
    
    return out


def display_enhanced_sprint_metrics(sprint_data):
    """Display enhanced sprint metrics in an organized, readable format."""
    if not sprint_data:
        print("No sprint data to display.")
        return
    
    print("\n🚀 ENHANCED SPRINT ANALYSIS")
    print("=" * 60)
    
    for i, sprint in enumerate(sprint_data, 1):
        print(f"\n📊 Sprint {i} - {sprint.get('start', 'N/A')}")
        print("-" * 40)
        
        # Timing Analysis
        print("⏱️  TIMING ANALYSIS:")
        if 'power_ttp_s' in sprint:
            print(f"   Power Peak: {sprint['power_ttp_s']:.2f}s")
        if 'torque_ttp_s' in sprint:
            print(f"   Torque Peak: {sprint['torque_ttp_s']:.2f}s")
        if 'cadence_ttp_s' in sprint:
            print(f"   Cadence Peak: {sprint['cadence_ttp_s']:.2f}s")
        
        # Peak Sequence
        if 'peak_sequence' in sprint:
            print(f"   Peak Sequence: {sprint['peak_sequence']}")
        if 'peak_synchronization_s' in sprint:
            print(f"   Peak Spread: {sprint['peak_synchronization_s']:.2f}s")
        
        # Power Curve Analysis
        print("\n⚡ POWER CURVE ANALYSIS:")
        for threshold in [50, 75, 90]:
            key = f'power_time_to_{threshold}%_s'
            if key in sprint and not np.isnan(sprint[key]):
                print(f"   Time to {threshold}%: {sprint[key]:.2f}s")
        
        # Torque Curve Analysis
        print("\n🔧 TORQUE CURVE ANALYSIS:")
        for threshold in [50, 75, 90]:
            key = f'torque_time_to_{threshold}%_s'
            if key in sprint and not np.isnan(sprint[key]):
                print(f"   Time to {threshold}%: {sprint[key]:.2f}s")
        
        # Cadence Curve Analysis
        print("\n🔄 CADENCE CURVE ANALYSIS:")
        for threshold in [50, 75, 90]:
            key = f'cadence_time_to_{threshold}%_s'
            if key in sprint and not np.isnan(sprint[key]):
                print(f"   Time to {threshold}%: {sprint[key]:.2f}s")
        
        # Cross-Metric Delays
        print("\n⏰ PEAK DELAYS:")
        if 'torque_peak_delay_vs_power_s' in sprint:
            delay = sprint['torque_peak_delay_vs_power_s']
            if delay > 0:
                print(f"   Torque peaks {delay:.2f}s AFTER power")
            elif delay < 0:
                print(f"   Torque peaks {abs(delay):.2f}s BEFORE power")
            else:
                print(f"   Torque and power peak together")
        
        if 'cadence_peak_delay_vs_power_s' in sprint:
            delay = sprint['cadence_peak_delay_vs_power_s']
            if delay > 0:
                print(f"   Cadence peaks {delay:.2f}s AFTER power")
            elif delay < 0:
                print(f"   Cadence peaks {abs(delay):.2f}s BEFORE power")
            else:
                print(f"   Cadence and power peak together")
        
        # Performance Metrics
        print("\n📈 PERFORMANCE METRICS:")
        if 'work_kJ' in sprint:
            print(f"   Total Work: {sprint['work_kJ']:.2f} kJ")
        if 'mechanical_efficiency_proxy' in sprint and not np.isnan(sprint['mechanical_efficiency_proxy']):
            print(f"   Efficiency: {sprint['mechanical_efficiency_proxy']:.1f} kJ/km")
        if 'fade_%' in sprint:
            print(f"   Power Fade: {sprint['fade_%']:.1f}%")
        
        # Enhanced Metrics
        print("\n🚀 ENHANCED METRICS:")
        if 'power_acceleration_rate' in sprint and not np.isnan(sprint['power_acceleration_rate']):
            print(f"   Power Acceleration: {sprint['power_acceleration_rate']:.0f} W/s")
        if 'sprint_efficiency_score' in sprint:
            score = sprint['sprint_efficiency_score']
            if score >= 120:
                rating = "🔥 EXCELLENT"
            elif score >= 100:
                rating = "✅ GOOD"
            elif score >= 80:
                rating = "⚠️  FAIR"
            else:
                rating = "❌ POOR"
            print(f"   Efficiency Score: {score}/150 ({rating})")
        
        # Speed Analysis
        print("\n🚴 SPEED ANALYSIS:")
        if 'speed_increase_kph' in sprint and not np.isnan(sprint['speed_increase_kph']):
            speed_gain = sprint['speed_increase_kph']
            if speed_gain > 0:
                print(f"   Speed Gain: +{speed_gain:.1f} km/h")
            else:
                print(f"   Speed Loss: {speed_gain:.1f} km/h")
        
        if 'max_speed_kph' in sprint and not np.isnan(sprint['max_speed_kph']):
            print(f"   Peak Speed: {sprint['max_speed_kph']:.1f} km/h")
        
        if 'speed_ttp_s' in sprint and not np.isnan(sprint['speed_ttp_s']):
            print(f"   Time to Max Speed: {sprint['speed_ttp_s']:.1f}s")
        
        if 'speed_acceleration_rate' in sprint and not np.isnan(sprint['speed_acceleration_rate']):
            print(f"   Speed Acceleration: {sprint['speed_acceleration_rate']:.1f} km/h/s")
        
        if 'sprint_type' in sprint:
            print(f"   Sprint Type: {sprint['sprint_type']}")
        
        if 'sprint_distance_m' in sprint and not np.isnan(sprint['sprint_distance_m']):
            print(f"   Distance Covered: {sprint['sprint_distance_m']:.0f}m")
        
        print("-" * 40)


def analyze_sprint_technique_patterns(sprint_data):
    """Analyze sprint technique patterns across multiple sprints."""
    if not sprint_data or len(sprint_data) < 2:
        print("Need at least 2 sprints to analyze patterns.")
        return
    
    print("\n🔍 SPRINT TECHNIQUE PATTERN ANALYSIS")
    print("=" * 60)
    
    # Convert to DataFrame for easier analysis
    df_sprints = pd.DataFrame(sprint_data)
    
    # Consistency analysis
    print("\n📊 CONSISTENCY ANALYSIS:")
    for metric in ['power_ttp_s', 'torque_ttp_s', 'cadence_ttp_s', 'speed_ttp_s']:
        if metric in df_sprints.columns:
            values = df_sprints[metric].dropna()
            if len(values) > 1:
                mean_val = values.mean()
                std_val = values.std()
                cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
                print(f"   {metric.replace('_', ' ').title()}: {mean_val:.2f}s ± {std_val:.2f}s (CV: {cv:.1f}%)")
    
    # Peak timing relationships
    print("\n⏰ PEAK TIMING RELATIONSHIPS:")
    if all(col in df_sprints.columns for col in ['power_ttp_s', 'torque_ttp_s', 'cadence_ttp_s']):
        # Calculate correlations
        power_torque_corr = df_sprints['power_ttp_s'].corr(df_sprints['torque_ttp_s'])
        power_cadence_corr = df_sprints['power_ttp_s'].corr(df_sprints['cadence_ttp_s'])
        torque_cadence_corr = df_sprints['torque_ttp_s'].corr(df_sprints['cadence_ttp_s'])
        
        print(f"   Power-Torque timing correlation: {power_torque_corr:.3f}")
        print(f"   Power-Cadence timing correlation: {power_cadence_corr:.3f}")
        print(f"   Torque-Cadence timing correlation: {torque_cadence_corr:.3f}")
        
        # Interpret correlations
        if abs(power_torque_corr) > 0.7:
            print("   → Strong power-torque timing relationship")
        elif abs(power_torque_corr) > 0.4:
            print("   → Moderate power-torque timing relationship")
        else:
            print("   → Weak power-torque timing relationship")
    
    # Speed analysis
    print("\n🚴 SPEED ANALYSIS:")
    if 'speed_increase_kph' in df_sprints.columns:
        speed_increases = df_sprints['speed_increase_kph'].dropna()
        if len(speed_increases) > 0:
            avg_speed_gain = speed_increases.mean()
            max_speed_gain = speed_increases.max()
            min_speed_gain = speed_increases.min()
            
            print(f"   Average Speed Gain: {avg_speed_gain:.1f} km/h")
            print(f"   Best Speed Gain: {max_speed_gain:.1f} km/h")
            print(f"   Worst Speed Gain: {min_speed_gain:.1f} km/h")
            
            if 'sprint_type' in df_sprints.columns:
                sprint_types = df_sprints['sprint_type'].value_counts()
                print(f"   Sprint Types: {dict(sprint_types)}")
    
    # Speed vs Power timing analysis
    if all(col in df_sprints.columns for col in ['power_ttp_s', 'speed_ttp_s']):
        print("\n⚡ SPEED vs POWER TIMING:")
        power_speed_corr = df_sprints['power_ttp_s'].corr(df_sprints['speed_ttp_s'])
        print(f"   Power-Speed timing correlation: {power_speed_corr:.3f}")
        
        if abs(power_speed_corr) > 0.7:
            print("   → Strong power-speed timing relationship")
        elif abs(power_speed_corr) > 0.4:
            print("   → Moderate power-speed timing relationship")
        else:
            print("   → Weak power-speed timing relationship")
    
    # Sprint progression analysis
    print("\n📈 SPRINT PROGRESSION ANALYSIS:")
    if 'start' in df_sprints.columns:
        # Sort by start time to see progression
        df_sprints_sorted = df_sprints.sort_values('start')
        
        # Check if timing improves over sprints
        if 'power_ttp_s' in df_sprints_sorted.columns:
            power_times = df_sprints_sorted['power_ttp_s'].dropna()
            if len(power_times) > 1:
                first_sprint = power_times.iloc[0]
                last_sprint = power_times.iloc[-1]
                improvement = first_sprint - last_sprint
                if improvement > 0.1:
                    print(f"   Power timing improved by {improvement:.2f}s over sprints")
                elif improvement < -0.1:
                    print(f"   Power timing degraded by {abs(improvement):.2f}s over sprints")
                else:
                    print(f"   Power timing remained consistent (±{abs(improvement):.2f}s)")
    
    # Technique classification
    print("\n🎯 TECHNIQUE CLASSIFICATION:")
    if all(col in df_sprints.columns for col in ['power_ttp_s', 'torque_ttp_s', 'cadence_ttp_s']):
        avg_power_ttp = df_sprints['power_ttp_s'].mean()
        avg_torque_ttp = df_sprints['torque_ttp_s'].mean()
        avg_cadence_ttp = df_sprints['cadence_ttp_s'].mean()
        
        # Classify sprint technique
        if avg_power_ttp < 2.0 and avg_torque_ttp < 2.0:
            print("   → EXPLOSIVE: Quick power and torque development")
        elif avg_power_ttp < 3.0 and avg_torque_ttp < 3.0:
            print("   → MODERATE: Balanced power and torque timing")
        else:
            print("   → GRADUAL: Slower power and torque development")
        
        if avg_cadence_ttp < avg_power_ttp:
            print("   → CADENCE-DRIVEN: Cadence peaks before power")
        else:
            print("   → POWER-DRIVEN: Power peaks before cadence")
    
    print("-" * 60)


def create_sprint_plots(best):
    """Create scatter plots for sprint analysis."""
    if best is None or best.empty:
        print("No valid sprints to plot.")
        return
    
    if go is None:
        print("Plotly not installed; cannot show interactive plots.")
        return

    def scatter_xy(x, y, title, xlab, ylab):
        hover = [f"{s}<br>P: {p:.0f} W" for s, p in zip(best['Start'], best['Power (W)'])]
        fig = go.Figure(go.Scatter(
            x=best[x], y=best[y], mode='markers+text',
            text=best['Start'], textposition='top center',
            marker=dict(size=10, color=best['Power (W)'],
                        colorscale='Viridis', colorbar=dict(title='Power (W)')),
            hovertext=hover, hoverinfo='text'))
        
        fig.update_layout(template='plotly_white', title=title,
                          xaxis_title=xlab, yaxis_title=ylab, height=480)
        # Save to HTML file instead of showing
        # filename = f"{xlab.lower().replace(' ', '_')}_vs_{ylab.lower().replace(' ', '_')}.html"
        # fig.write_html(filename)  # Removed file generation
        # print(f"Plot saved to {filename}")  # Plots shown in dashboard instead

    scatter_xy('Torque (Nm)', 'Power (W)',
               'Torque vs Power — best 10‑s efforts', 'Torque (Nm)', 'Power (W)')
    scatter_xy('Cadence (rpm)', 'Power (W)',
               'Cadence vs Power — best 10‑s efforts', 'Cadence (rpm)', 'Power (W)')


def plot_sprint_trajectories(df, best10):
    """Plot torque-cadence trajectories with time coloring for each sprint."""
    if go is None:
        print("Plotly not installed; cannot show trajectory plots.")
        return
    
    for i, (_, row) in enumerate(best10.iterrows()):
        start_ts = row['start_ts']
        seg = df.loc[start_ts : start_ts + pd.Timedelta(seconds=9)]
        
        if seg.empty or 'torque' not in seg.columns or 'cadence' not in seg.columns:
            continue
        
        # Create time-based coloring
        time_sec = (seg.index - seg.index[0]).total_seconds()
        
        fig = go.Figure(go.Scatter(
            x=seg['cadence'], y=seg['torque'], mode='markers+lines',
            marker=dict(size=8, color=time_sec, colorscale='Viridis',
                        colorbar=dict(title="Time (s)")),
            line=dict(width=2),
            hovertemplate=("Cadence: %{x:.1f} rpm<br>"
                          "Torque: %{y:.2f} Nm<br>"
                          "Time: %{marker.color:.1f}s<br>"
                          "Power: %{text:.0f} W"),
            text=seg['power'] if 'power' in seg.columns else [None] * len(seg)
        ))
        
        fig.update_layout(
            title=f"Sprint {i+1} Trajectory: Torque vs Cadence (colored by time)",
            xaxis=dict(title="Cadence (rpm)", range=[0, 200]),
            yaxis=dict(title="Torque (Nm)", range=[0, 120]),
            template="plotly_white",
            height=500
        )
        
        # Save to HTML file
        # filename = f"sprint_{i+1}_trajectory.html"
        # fig.write_html(filename)  # Removed file generation
        # print(f"Trajectory plot saved to {filename}")  # Plots shown in dashboard instead
        
        # Also plot speed curve
        if 'enhanced_speed' in seg.columns:
            fig_speed = go.Figure(go.Scatter(
                x=time_sec, y=seg['enhanced_speed'] * 3.6, mode='lines+markers',
                line=dict(width=3, color='red'),
                marker=dict(size=6),
                hovertemplate=("Time: %{x:.1f}s<br>"
                              "Speed: %{y:.1f} km/h<br>"
                              "Power: %{text:.0f} W"),
                text=seg['power'] if 'power' in seg.columns else [None] * len(seg)
            ))
            
            fig_speed.update_layout(
                title=f"Sprint {i+1} Speed Curve",
                xaxis=dict(title="Time (s)"),
                yaxis=dict(title="Speed (km/h)"),
                template="plotly_white",
                height=400
            )
            
            # filename_speed = f"sprint_{i+1}_speed_curve.html"
            # fig_speed.write_html(filename_speed)  # Removed file generation
            # print(f"Speed curve saved to {filename_speed}")  # Plots shown in dashboard instead


def display_sprint_analysis(best10, micro_df, sprint_summary_df):
    """Display sprint analysis in clean, organized tables."""
    print("\n" + "="*80)
    print("CYCLING SPRINT ANALYSIS REPORT")
    print("="*80)
    
    # Sprint Performance Summary Table
    print("\nSPRINT PERFORMANCE SUMMARY")
    print("-" * 50)
    
    # Create a clean summary table
    summary_data = []
    for i, row in best10.iterrows():
        summary_data.append({
            'Sprint': f'Sprint {i+1}',
            'Start Time': row['Start'],
            'Power (W)': f"{row['Power (W)']:.0f}",
            'Torque (Nm)': f"{row['Torque (Nm)']:.1f}",
            'Cadence (rpm)': f"{row['Cadence (rpm)']:.1f}",
            'Angular Velocity (rad/s)': f"{row['ω (rad·s⁻¹)']:.2f}",
            'Error (%)': f"{row['Err %']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Sprint Micro-Metrics Table
    if not micro_df.empty:
        print("\nSPRINT MICRO-METRICS")
        print("-" * 50)
        
        # Select and rename key columns for display
        micro_display = micro_df[['start', 'power_max', 'power_ttp_s', 'torque_max', 'cadence_max', 
                                 'power_fade_%', 'work_kJ', 'rev_est']].copy()
        micro_display.columns = ['Start Time', 'Peak Power (W)', 'Time to Peak (s)', 'Peak Torque (Nm)', 
                                'Peak Cadence (rpm)', 'Power Fade (%)', 'Work (kJ)', 'Est. Revolutions']
        
        # Format the display
        for col in ['Peak Power (W)', 'Peak Torque (Nm)', 'Peak Cadence (rpm)', 'Work (kJ)', 'Est. Revolutions']:
            if col in micro_display.columns:
                micro_display[col] = micro_display[col].round(1)
        
        print(micro_display.to_string(index=False))
    
    # Sprint Summary Analysis Table
    if not sprint_summary_df.empty:
        print("\nSPRINT SUMMARY ANALYSIS")
        print("-" * 50)
        
        # Select and rename key columns for display - use available columns with fallbacks
        available_cols = sprint_summary_df.columns.tolist()
        cols_to_use = []
        col_names = []
        
        if 'start' in available_cols:
            cols_to_use.append('start')
            col_names.append('Start Time')
        
        if 'avg_power_10s' in available_cols:
            cols_to_use.append('avg_power_10s')
            col_names.append('Avg Power (W)')
        
        if 'max_power' in available_cols:
            cols_to_use.append('max_power')
            col_names.append('Max Power (W)')
        
        if 'peak_torque' in available_cols:
            cols_to_use.append('peak_torque')
            col_names.append('Peak Torque (Nm)')
        
        if 'peak_cadence' in available_cols:
            cols_to_use.append('peak_cadence')
            col_names.append('Peak Cadence (rpm)')
        
        # Try different possible column names for speed gain
        speed_col = None
        for col in ['speed_gain_kmh', 'delta_speed_kph', 'speed_gain']:
            if col in available_cols:
                speed_col = col
                break
        
        if speed_col:
            cols_to_use.append(speed_col)
            col_names.append('Speed Gain (km/h)')
        
        # Try different possible column names for acceleration
        accel_col = None
        for col in ['max_accel_ms2', 'max_acceleration', 'accel_max']:
            if col in available_cols:
                accel_col = col
                break
        
        if accel_col:
            cols_to_use.append(accel_col)
            col_names.append('Max Accel (m/s²)')
        
        if 'fade_%' in available_cols:
            cols_to_use.append('fade_%')
            col_names.append('Fade (%)')
        
        if cols_to_use:
            summary_display = sprint_summary_df[cols_to_use].copy()
            summary_display.columns = col_names
            
            # Format the display
            for col in ['Avg Power (W)', 'Max Power (W)', 'Peak Torque (Nm)', 'Peak Cadence (rpm)']:
                if col in summary_display.columns:
                    summary_display[col] = summary_display[col].round(0)
            
            for col in ['Speed Gain (km/h)', 'Max Accel (m/s²)', 'Fade (%)']:
                if col in summary_display.columns:
                    summary_display[col] = summary_display[col].round(1)
            
            print(summary_display.to_string(index=False))
        else:
            print("No sprint summary data available to display.")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    return True


def analyze_sprint_context(df, best10, context_seconds=10):
    """Analyze the context around each sprint (before and after)."""
    print(f"\nSPRINT CONTEXT ANALYSIS (±{context_seconds}s)")
    print("-" * 60)
    
    # Create context data for table
    context_data = []
    
    for i, (_, row) in enumerate(best10.iterrows()):
        start_ts = row['start_ts']
        sprint_start = start_ts
        sprint_end = start_ts + pd.Timedelta(seconds=9)
        
        # Get context windows
        pre_sprint = df.loc[start_ts - pd.Timedelta(seconds=context_seconds) : start_ts - pd.Timedelta(seconds=1)]
        post_sprint = df.loc[sprint_end + pd.Timedelta(seconds=1) : sprint_end + pd.Timedelta(seconds=context_seconds)]
        
        # Pre-sprint analysis
        pre_power_avg = 0
        pre_cadence_avg = np.nan
        pre_speed_avg = np.nan
        
        if not pre_sprint.empty and 'power' in pre_sprint.columns:
            pre_power_avg = pre_sprint['power'].mean()
            pre_cadence_avg = pre_sprint['cadence'].mean() if 'cadence' in pre_sprint.columns else np.nan
            pre_speed_avg = pre_sprint['enhanced_speed'].mean() if 'enhanced_speed' in pre_sprint.columns else np.nan
        
        # Sprint performance (summary)
        sprint_power_avg = row['Power (W)']
        sprint_power_max = row.get('max_power', np.nan)
        if np.isnan(sprint_power_max) and 'power' in df.columns:
            sprint_seg = df.loc[sprint_start : sprint_end]
            sprint_power_max = sprint_seg['power'].max()
        
        # Post-sprint analysis
        post_power_avg = 0
        post_cadence_avg = np.nan
        post_speed_avg = np.nan
        
        if not post_sprint.empty and 'power' in post_sprint.columns:
            post_power_avg = post_sprint['power'].mean()
            post_cadence_avg = post_sprint['cadence'].mean() if 'cadence' in post_sprint.columns else np.nan
            post_speed_avg = post_sprint['enhanced_speed'].mean() if 'enhanced_speed' in post_sprint.columns else np.nan
        
        # Calculate power surge
        power_surge = ((sprint_power_max - pre_power_avg) / pre_power_avg * 100) if pre_power_avg > 0 else np.nan
        
        # Add to context data
        context_data.append({
            'Sprint': f'Sprint {i+1}',
            'Start Time': row['Start'],
            'Pre-Power (W)': f"{pre_power_avg:.0f}",
            'Pre-Cadence (rpm)': f"{pre_cadence_avg:.1f}" if not np.isnan(pre_cadence_avg) else "N/A",
            'Pre-Speed (km/h)': f"{pre_speed_avg * 3.6:.1f}" if not np.isnan(pre_speed_avg) else "N/A",
            'Sprint Power (W)': f"{sprint_power_avg:.0f}",
            'Max Power (W)': f"{sprint_power_max:.0f}" if not np.isnan(sprint_power_max) else "N/A",
            'Power Surge (%)': f"+{power_surge:.1f}%" if not np.isnan(power_surge) else "N/A",
            'Post-Power (W)': f"{post_power_avg:.0f}",
            'Post-Cadence (rpm)': f"{post_cadence_avg:.1f}" if not np.isnan(post_cadence_avg) else "N/A",
            'Post-Speed (km/h)': f"{post_speed_avg * 3.6:.1f}" if not np.isnan(post_speed_avg) else "N/A"
        })
    
    # Display as table
    context_df = pd.DataFrame(context_data)
    print(context_df.to_string(index=False))
    
    return True


def calculate_standard_metrics(df, best10, micro_df, sprint_summary_df):
    """Calculate standard cycling metrics used in WKO5 and TrainingPeaks."""
    print("\nSTANDARD CYCLING METRICS (WKO5/TrainingPeaks Style)")
    print("=" * 70)
    
    # Rider inputs
    print(f"RIDER PROFILE:")
    print(f"  • FTP: {FTP}W")
    print(f"  • LTHR: {LTHR} bpm")
    print(f"  • Mass: {rider_mass_kg} kg")
    print()
    
    # Overall ride metrics
    if 'power' in df.columns:
        total_time = (df.index[-1] - df.index[0]).total_seconds() / 3600  # hours
        total_work = df['power'].sum() / 1000  # kJ
        
        # Power metrics
        avg_power = df['power'].mean()
        max_power = df['power'].max()
        
        # Normalized Power (30s rolling average, then 95th percentile)
        if len(df) > 30:
            rolling_30s = df['power'].rolling(30, min_periods=30).mean()
            # Sort by power and take 95th percentile
            sorted_powers = rolling_30s.dropna().sort_values(ascending=False)
            if len(sorted_powers) > 0:
                np_95th = sorted_powers.iloc[int(len(sorted_powers) * 0.05)]
            else:
                np_95th = np.nan
        else:
            np_95th = np.nan
        
        # Intensity Factor (NP / FTP)
        if not np.isnan(np_95th):
            intensity_factor = np_95th / FTP
        else:
            intensity_factor = np.nan
        
        # Training Stress Score (TSS) - proper Coggan formula
        if not np.isnan(intensity_factor) and total_time > 0:
            tss = (total_time * intensity_factor * intensity_factor * 100)
        else:
            tss = np.nan
        
        # Power-to-weight ratios
        avg_power_kg = avg_power / rider_mass_kg
        max_power_kg = max_power / rider_mass_kg
        if not np.isnan(np_95th):
            np_kg = np_95th / rider_mass_kg
        else:
            np_kg = np.nan
        
        print(f"OVERALL RIDE METRICS:")
        print("-" * 50)
        
        # Create overall metrics table
        overall_data = []
        overall_data.append(['Total Time', f"{total_time:.2f} hours"])
        overall_data.append(['Total Work', f"{total_work:.1f} kJ"])
        overall_data.append(['Average Power', f"{avg_power:.0f}W ({avg_power_kg:.1f} W/kg)"])
        overall_data.append(['Max Power', f"{max_power:.0f}W ({max_power_kg:.1f} W/kg)"])
        
        if not np.isnan(np_95th):
            overall_data.append(['Normalized Power', f"{np_95th:.0f}W ({np_kg:.1f} W/kg)"])
        if not np.isnan(intensity_factor):
            overall_data.append(['Intensity Factor', f"{intensity_factor:.2f}"])
        if not np.isnan(tss):
            overall_data.append(['Training Stress Score', f"{tss:.0f}"])
        
        # Display as table
        overall_df = pd.DataFrame(overall_data, columns=['Metric', 'Value'])
        print(overall_df.to_string(index=False))
        print()
        
        # Coggan Power Zones
        print(f"COGGAN POWER ZONES (FTP = {FTP}W):")
        print("-" * 50)
        
        power_zones = {
            'Zone 1 (Active Recovery)': (df['power'] < FTP * 0.55).sum(),
            'Zone 2 (Endurance)': ((df['power'] >= FTP * 0.55) & (df['power'] < FTP * 0.75)).sum(),
            'Zone 3 (Tempo)': ((df['power'] >= FTP * 0.75) & (df['power'] < FTP * 0.90)).sum(),
            'Zone 4 (Lactate Threshold)': ((df['power'] >= FTP * 0.90) & (df['power'] < FTP * 1.05)).sum(),
            'Zone 5 (VO2 Max)': ((df['power'] >= FTP * 1.05) & (df['power'] < FTP * 1.20)).sum(),
            'Zone 6 (Anaerobic Capacity)': ((df['power'] >= FTP * 1.20) & (df['power'] < FTP * 1.50)).sum(),
            'Zone 7 (Neuromuscular Power)': (df['power'] >= FTP * 1.50).sum()
        }
        
        total_samples = len(df['power'].dropna())
        zone_data = []
        for zone, count in power_zones.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            time_minutes = (count * (df.index[1] - df.index[0]).total_seconds()) / 60 if len(df) > 1 else 0
            zone_data.append([zone, f"{percentage:.1f}%", f"{time_minutes:.1f} min"])
        
        # Display as table
        zone_df = pd.DataFrame(zone_data, columns=['Power Zone', 'Percentage', 'Time'])
        print(zone_df.to_string(index=False))
        print()
    
    # Sprint-specific standard metrics
    if not best10.empty:
        print(f"SPRINT-SPECIFIC METRICS:")
        print("-" * 50)
        
        # Best sprint metrics
        best_sprint = best10.iloc[0]  # Highest power sprint
        best_power_kg = best_sprint['Power (W)'] / rider_mass_kg
        
        # Create sprint metrics table
        sprint_metrics_data = []
        sprint_metrics_data.append(['Best 10s Power', f"{best_sprint['Power (W)']:.0f}W ({best_power_kg:.1f} W/kg)"])
        
        # Sprint power curve (5s, 10s, 20s, 30s, 1min, 2min, 5min)
        power_curves = []
        for duration in [5, 10, 20, 30, 60, 120, 300]:  # seconds
            if len(df) >= duration:
                # Find best power for this duration
                rolling_power = df['power'].rolling(duration, min_periods=duration).mean()
                best_power = rolling_power.max()
                if not np.isnan(best_power):
                    power_curves.append((duration, best_power))
        
        if power_curves:
            sprint_metrics_data.append(['Power Curve', ''])
            for duration, power in power_curves:
                power_kg = power / rider_mass_kg
                if duration < 60:
                    sprint_metrics_data.append([f'  {duration}s', f"{power:.0f}W ({power_kg:.1f} W/kg)"])
                else:
                    minutes = duration / 60
                    sprint_metrics_data.append([f'  {minutes:.0f}min', f"{power:.0f}W ({power_kg:.1f} W/kg)"])
        
        # Sprint repeatability
        if len(best10) >= 2:
            power_std = best10['Power (W)'].std()
            power_cv = (power_std / best10['Power (W)'].mean()) * 100  # Coefficient of variation
            sprint_metrics_data.append(['Sprint Repeatability', f"CV = {power_cv:.1f}%"])
            
            if power_cv < 5:
                status = "Excellent consistency"
            elif power_cv < 10:
                status = "Good consistency"
            else:
                status = "Variable performance"
            sprint_metrics_data.append(['Status', status])
        
        # Display as table
        sprint_metrics_df = pd.DataFrame(sprint_metrics_data, columns=['Metric', 'Value'])
        print(sprint_metrics_df.to_string(index=False))
        print()
    
    # Cadence analysis
    if 'cadence' in df.columns:
        print(f"CADENCE ANALYSIS:")
        print("-" * 50)
        
        cadence_data = df['cadence'].dropna()
        if not cadence_data.empty:
            avg_cadence = cadence_data.mean()
            max_cadence = cadence_data.max()
            cadence_std = cadence_data.std()
            
            # Create cadence summary table
            cadence_summary_data = []
            cadence_summary_data.append(['Average Cadence', f"{avg_cadence:.1f} rpm"])
            cadence_summary_data.append(['Max Cadence', f"{max_cadence:.1f} rpm"])
            cadence_summary_data.append(['Cadence Variability', f"{cadence_std:.1f} rpm"])
            
            # Display summary
            cadence_summary_df = pd.DataFrame(cadence_summary_data, columns=['Metric', 'Value'])
            print(cadence_summary_df.to_string(index=False))
            print()
            
            # Cadence zones (common in cycling analysis)
            cadence_zones = {
                'Low (<80 rpm)': (cadence_data < 80).sum(),
                'Medium (80-100 rpm)': ((cadence_data >= 80) & (cadence_data < 100)).sum(),
                'High (100-120 rpm)': ((cadence_data >= 100) & (cadence_data < 120)).sum(),
                'Very High (>120 rpm)': (cadence_data >= 120).sum()
            }
            
            total_samples = len(cadence_data)
            cadence_zone_data = []
            for zone, count in cadence_zones.items():
                percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                cadence_zone_data.append([zone, f"{percentage:.1f}%"])
            
            # Display zones as table
            print("Cadence Distribution:")
            cadence_zone_df = pd.DataFrame(cadence_zone_data, columns=['Zone', 'Percentage'])
            print(cadence_zone_df.to_string(index=False))
        print()
    
    # Heart rate analysis (if available)
    if 'heart_rate' in df.columns:
        print(f"HEART RATE ANALYSIS (LTHR = {LTHR} bpm):")
        print("-" * 50)
        
        hr_data = df['heart_rate'].dropna()
        if not hr_data.empty:
            avg_hr = hr_data.mean()
            max_hr = hr_data.max()
            min_hr = hr_data.min()
            
            # Create HR summary table
            hr_summary_data = []
            hr_summary_data.append(['Average HR', f"{avg_hr:.0f} bpm"])
            hr_summary_data.append(['Max HR', f"{max_hr:.0f} bpm"])
            hr_summary_data.append(['Min HR', f"{min_hr:.0f} bpm"])
            
            # Display summary
            hr_summary_df = pd.DataFrame(hr_summary_data, columns=['Metric', 'Value'])
            print(hr_summary_df.to_string(index=False))
            print()
            
            # Coggan HR zones (using your LTHR)
            hr_zones = {
                'Zone 1 (Active Recovery)': (hr_data < LTHR * 0.85).sum(),
                'Zone 2 (Endurance)': ((hr_data >= LTHR * 0.85) & (hr_data < LTHR * 0.95)).sum(),
                'Zone 3 (Tempo)': ((hr_data >= LTHR * 0.95) & (hr_data < LTHR * 1.05)).sum(),
                'Zone 4 (Lactate Threshold)': ((hr_data >= LTHR * 1.05) & (hr_data < LTHR * 1.15)).sum(),
                'Zone 5 (VO2 Max)': (hr_data >= LTHR * 1.15).sum()
            }
            
            total_samples = len(hr_data)
            hr_zone_data = []
            for zone, count in hr_zones.items():
                percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                time_minutes = (count * (df.index[1] - df.index[0]).total_seconds()) / 60 if len(df) > 1 else 0
                hr_zone_data.append([zone, f"{percentage:.1f}%", f"{time_minutes:.1f} min"])
            
            # Display zones as table
            print("HR Zone Distribution:")
            hr_zone_df = pd.DataFrame(hr_zone_data, columns=['HR Zone', 'Percentage', 'Time'])
            print(hr_zone_df.to_string(index=False))
        print()
    
    print("=" * 70)
    return True


def create_simple_graphs(df, best10, micro_df, sprint_summary_df):
    """Create improved, professional-looking matplotlib graphs."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Set professional style
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    print("\nCreating improved graphs...")
    
    # 1. POWER CURVE - Single focused plot
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    durations = [5, 10, 20, 30, 60, 120, 300]
    best_powers = []
    
    for duration in durations:
        if len(df) >= duration:
            rolling_power = df['power'].rolling(duration, min_periods=duration).mean()
            best_power = rolling_power.max()
            if not np.isnan(best_power):
                best_powers.append(best_power)
            else:
                best_powers.append(0)
        else:
            best_powers.append(0)
    
    # Filter out zero values
    valid_durations = [d for d, p in zip(durations, best_powers) if p > 0]
    valid_powers = [p for p in best_powers if p > 0]
    
    # Main power curve
    line1 = ax1.plot(valid_durations, valid_powers, 'o-', linewidth=3, markersize=10, 
                     color='#1f77b4', label='Power (W)', zorder=3)
    
    # Add FTP reference line
    ax1.axhline(y=FTP, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                label=f'FTP ({FTP}W)', zorder=2)
    
    # Add power-to-weight on secondary y-axis
    ax1_twin = ax1.twinx()
    powers_kg = [p / rider_mass_kg for p in valid_powers]
    line2 = ax1_twin.plot(valid_durations, powers_kg, 's--', linewidth=3, markersize=8, 
                          color='#ff7f0e', label='Power (W/kg)', zorder=3)
    
    # Styling
    ax1.set_xlabel('Duration (seconds)', fontweight='bold')
    ax1.set_ylabel('Power (W)', fontweight='bold', color='#1f77b4')
    ax1_twin.set_ylabel('Power (W/kg)', fontweight='bold', color='#ff7f0e')
    ax1.set_title('Power Curve - Best Efforts at Different Durations', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, zorder=1)
    
    # Color the y-axis labels
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1_twin.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    # 2. COGGAN POWER ZONES - Horizontal bar chart for better readability
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    if 'power' in df.columns:
        power_data = df['power'].dropna()
        zone_boundaries = [0, FTP*0.55, FTP*0.75, FTP*0.90, FTP*1.05, FTP*1.20, FTP*1.50, power_data.max()]
        zone_labels = ['Zone 1\n(Active Recovery)', 'Zone 2\n(Endurance)', 'Zone 3\n(Tempo)', 
                      'Zone 4\n(Lactate Threshold)', 'Zone 5\n(VO2 Max)', 'Zone 6\n(Anaerobic)', 
                      'Zone 7\n(Neuromuscular)']
        zone_colors = ['#87CEEB', '#90EE90', '#FFFFE0', '#FFA500', '#FF4500', '#9370DB', '#8B0000']
        
        zone_counts = []
        zone_percentages = []
        for i in range(len(zone_boundaries)-1):
            count = ((power_data >= zone_boundaries[i]) & (power_data < zone_boundaries[i+1])).sum()
            zone_counts.append(count)
            percentage = (count / len(power_data)) * 100
            zone_percentages.append(percentage)
        
        # Horizontal bar chart
        bars = ax2.barh(zone_labels, zone_counts, color=zone_colors, alpha=0.8, height=0.6)
        
        # Add percentage labels
        for i, (bar, count, percentage) in enumerate(zip(bars, zone_counts, zone_percentages)):
            width = bar.get_width()
            ax2.text(width + max(zone_counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{percentage:.1f}%', ha='left', va='center', fontweight='bold')
            
            # Add zone boundaries
            ax2.text(width + max(zone_counts) * 0.05, bar.get_y() + bar.get_height()/2, 
                    f'<{zone_boundaries[i+1]:.0f}W', ha='left', va='center', 
                    fontsize=9, alpha=0.7)
        
        ax2.set_xlabel('Data Points', fontweight='bold')
        ax2.set_title(f'Coggan Power Zones Distribution (FTP = {FTP}W)', fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add total time information
        total_time = (df.index[-1] - df.index[0]).total_seconds() / 60
        ax2.text(0.02, 0.98, f'Total Time: {total_time:.1f} minutes', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 3. SPRINT POWER COMPARISON - Clear, focused comparison
    if not best10.empty:
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        
        sprint_numbers = [f'Sprint {i+1}' for i in range(len(best10))]
        avg_powers = best10['Power (W)'].values
        max_powers = []
        
        for _, row in best10.iterrows():
            start_ts = row['start_ts']
            seg = df.loc[start_ts : start_ts + pd.Timedelta(seconds=9)]
            max_powers.append(seg['power'].max())
        
        x = np.arange(len(sprint_numbers))
        width = 0.35
        
        # Create bars with better colors and spacing
        bars1 = ax3.bar(x - width/2, avg_powers, width, label='Average Power (10s)', 
                        color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax3.bar(x + width/2, max_powers, width, label='Peak Power', 
                        color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(max_powers) * 0.01,
                    f'{height:.0f}W', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(max_powers) * 0.01,
                    f'{height:.0f}W', ha='center', va='bottom', fontweight='bold')
        
        # Styling
        ax3.set_xlabel('Sprint', fontweight='bold')
        ax3.set_ylabel('Power (W)', fontweight='bold')
        ax3.set_title('Sprint Power Comparison', fontweight='bold', pad=20)
        ax3.set_xticks(x)
        ax3.set_xticklabels(sprint_numbers)
        ax3.legend(framealpha=0.9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add FTP reference line
        ax3.axhline(y=FTP, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                    label=f'FTP ({FTP}W)')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
    
    # 4. SPRINT TRAJECTORIES - Individual plots for clarity
    if not best10.empty:
        for i, (_, row) in enumerate(best10.iterrows()):
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            
            start_ts = row['start_ts']
            seg = df.loc[start_ts : start_ts + pd.Timedelta(seconds=9)]
            
            if not seg.empty and 'torque' in seg.columns and 'cadence' in seg.columns:
                seg_clean = seg.dropna(subset=['torque', 'cadence'])
                if len(seg_clean) >= 2:
                    time_sec = (seg_clean.index - seg_clean.index[0]).total_seconds()
                    
                    # Create scatter plot with time coloring
                    scatter = ax4.scatter(seg_clean['cadence'], seg_clean['torque'], 
                                        c=time_sec, cmap='viridis', s=150, alpha=0.8)
                    
                    # Add trajectory line
                    ax4.plot(seg_clean['cadence'], seg_clean['torque'], 'k-', alpha=0.6, linewidth=3)
                    
                    # Add time labels
                    for j, (cad, tor, t) in enumerate(zip(seg_clean['cadence'], seg_clean['torque'], time_sec)):
                        ax4.annotate(f'{t:.0f}s', (cad, tor), xytext=(5, 5), 
                                    textcoords='offset points', fontsize=9, alpha=0.8)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax4)
                    cbar.set_label('Time (seconds)', fontweight='bold')
                    
                    # Styling
                    ax4.set_xlabel('Cadence (rpm)', fontweight='bold')
                    ax4.set_ylabel('Torque (Nm)', fontweight='bold')
                    ax4.set_title(f'Sprint {i+1} Trajectory: {row["Start"]}\nTorque vs Cadence Over Time', 
                                fontweight='bold', pad=20)
                    ax4.grid(True, alpha=0.3)
                    
                    # Add power information if available
                    if 'power' in seg_clean.columns:
                        avg_power = seg_clean['power'].mean()
                        max_power = seg_clean['power'].max()
                        ax4.text(0.02, 0.98, f'Avg Power: {avg_power:.0f}W\nMax Power: {max_power:.0f}W', 
                                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.tight_layout()
                    plt.show()
    
    # 5. SPRINT PERFORMANCE METRICS - Simple, clear comparison
    if not micro_df.empty:
        fig5, ax5 = plt.subplots(figsize=(12, 8))
        
        metrics = ['power_max', 'torque_max', 'cadence_max', 'work_kJ']
        metric_labels = ['Peak Power\n(W)', 'Peak Torque\n(Nm)', 'Peak Cadence\n(rpm)', 'Work\n(kJ)']
        
        # Normalize metrics to 0-1 scale for comparison
        normalized_metrics = []
        for metric in metrics:
            if metric in micro_df.columns:
                values = micro_df[metric].values
                if metric == 'work_kJ':
                    normalized = values / values.max() if values.max() > 0 else values
                else:
                    normalized = (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else values
                normalized_metrics.append(normalized)
        
        if normalized_metrics:
            x = np.arange(len(metric_labels))
            width = 0.25
            
            # Create grouped bars for each sprint
            for i in range(len(best10)):
                values = [metric[i] for metric in normalized_metrics]
                bars = ax5.bar(x + i * width, values, width, 
                              label=f'Sprint {i+1} ({best10.iloc[i]["Start"]})', 
                              alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax5.set_xlabel('Metrics', fontweight='bold')
            ax5.set_ylabel('Normalized Performance (0-1)', fontweight='bold')
            ax5.set_title('Sprint Performance Metrics Comparison (Normalized)', fontweight='bold', pad=20)
            ax5.set_xticks(x + width)
            ax5.set_xticklabels(metric_labels)
            ax5.legend(framealpha=0.9)
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.set_ylim(0, 1.1)
            
            plt.tight_layout()
            plt.show()
    
    print("All improved graphs displayed! Close them to continue.")
    return True


def detect_intervals_ml_fallback(df, ftp):
    """
    Fallback interval detection using the original SprintV1 logic.
    This is used when the enhanced ML detection is not available.
    """
    if 'power' not in df.columns:
        print("No power data available for interval analysis")
        return pd.DataFrame()
    
    print(f"🔍 Starting fallback interval detection...")
    print(f"📊 FTP: {ftp}W, Power threshold: {power_threshold_pct*100:.0f}% FTP")
    print(f"⏱️  Overlap threshold: {overlap_threshold}s")
    
    all_intervals = []
    
    # 1️⃣ LAP-BASED INTERVAL EXTRACTION
    lap_intervals = []
    if 'lap' in df.columns:
        print(f"\n🏁 Extracting lap-based intervals...")
        lap_intervals = extract_lap_intervals(df, ftp)
        all_intervals.extend(lap_intervals)
        print(f"   Found {len(lap_intervals)} lap-based intervals")
    
    # 2️⃣ CHECK IF WE NEED FALLBACK DETECTION
    if len(lap_intervals) < 2:  # If insufficient laps, use fallback
        print(f"\n⚠️  Insufficient laps detected ({len(lap_intervals)}), using fallback auto-detection...")
        fallback_intervals = detect_fallback_intervals(df, ftp)
        print(f"   🚨 Fallback detection found {len(fallback_intervals)} intervals")
        
        if fallback_intervals:
            all_intervals = fallback_intervals
            print(f"   📊 Using fallback intervals: {len(all_intervals)} total")
        else:
            # Use regular auto-detection as fallback
            print(f"\n⚡ Using regular auto-detection as fallback...")
            auto_intervals = detect_power_based_intervals(df, ftp, min_duration, power_threshold_pct, cv_threshold)
            all_intervals.extend(auto_intervals)
            print(f"   Final auto-detected: {len(auto_intervals)} intervals")
    else:
        # 3️⃣ REGULAR AUTO-DETECTION WHEN SUFFICIENT LAPS
        print(f"\n⚡ Auto-detecting threshold efforts...")
        auto_intervals = detect_power_based_intervals(df, ftp, min_duration, power_threshold_pct, cv_threshold)
        
        # Filter out auto-detected intervals that overlap with lap intervals
        if lap_intervals and auto_intervals:
            print(f"   Checking for overlaps with lap intervals...")
            filtered_auto = filter_overlapping_intervals(auto_intervals, lap_intervals, overlap_threshold)
            print(f"   Removed {len(auto_intervals) - len(filtered_auto)} overlapping intervals")
            auto_intervals = filtered_auto
        
        all_intervals.extend(auto_intervals)
        print(f"   Final auto-detected: {len(auto_intervals)} intervals")
    
    # 3️⃣ DETECT REPEATING INTERVAL SETS
    print(f"\n🔄 Identifying repeating interval sets...")
    set_intervals = identify_interval_sets(all_intervals, set_time_window)
    print(f"   Found {len([i for i in set_intervals if i.get('set_info')])} interval sets")
    
    # 4️⃣ SCORE EACH INTERVAL
    print(f"\n⭐ Computing interval quality scores...")
    scored_intervals = score_intervals(set_intervals, ftp)
    
    # Create final DataFrame
    if not scored_intervals:
        print("❌ No intervals detected")
        return pd.DataFrame()
    
    intervals_df = pd.DataFrame(scored_intervals)
    
    # Sort by quality score (descending) and add ranking
    intervals_df = intervals_df.sort_values('quality_score', ascending=False).reset_index(drop=True)
    intervals_df['rank'] = range(1, len(intervals_df) + 1)
    
    print(f"\n✅ Final result: {len(intervals_df)} intervals detected")
    print(f"   Lap-based: {len([i for i in scored_intervals if i.get('source') == 'lap'])}")
    print(f"   Auto-detected: {len([i for i in scored_intervals if i.get('source') == 'auto'])}")
    print(f"   Fallback detection: {len([i for i in scored_intervals if i.get('source') == 'fallback'])}")
    print(f"   In sets: {len([i for i in scored_intervals if i.get('set_info')])}")
    
    return intervals_df


def detect_intervals_ml(df, ftp):
    """
    Use the standalone interval detection script directly on the original file.
    This ensures we get exactly the same results as the standalone script.
    """
    if 'power' not in df.columns:
        print("No power data available for interval analysis")
        return pd.DataFrame()
    
    print(f"🔍 Starting enhanced ML-based interval detection...")
    print(f"📊 Using FTP: {ftp}W")
    
    # Use the standalone script directly on the original file path
    try:
        from interval_detection import detect_intervals_ml_simple
        print(f"✅ Using standalone interval detection script directly")
        
        # Call the standalone script with the original file path
        # This ensures we get exactly 6 intervals as designed
        intervals = detect_intervals_ml_simple(file_path, ftp, save_plot=False)
        probabilities = None  # The standalone script doesn't return probabilities directly
        detection_method = "Standalone script"
        
        if not intervals:
            print("❌ No intervals detected by standalone script")
            return pd.DataFrame()
        
        # Convert intervals to the format expected by SprintV1
        all_intervals = []
        for start_time, end_time, duration in intervals:
            # Calculate interval metrics
            start_idx = df.index.get_loc(start_time)
            end_idx = df.index.get_loc(end_time)
            interval_data = df.iloc[start_idx:end_idx + 1]
            
            # Basic power metrics
            avg_power = interval_data['power'].mean()
            max_power = interval_data['power'].max()
            work_kj = interval_data['power'].sum() / 1000
            
            # Calculate power consistency (CV)
            power_cv = interval_data['power'].std() / avg_power if avg_power > 0 else 0
            
            # Calculate fade
            power_fade = (max_power - avg_power) / max_power if max_power > 0 else 0
            
            # Get ML confidence if available
            ml_confidence = None
            if probabilities is not None:
                ml_confidence = probabilities[start_idx:end_idx + 1].mean()
            
            # Calculate cadence metrics
            avg_cadence = np.nan
            max_cadence = np.nan
            min_cadence = np.nan
            cadence_drift = 0
            if 'cadence' in interval_data.columns:
                cadence_data = interval_data['cadence'].dropna()
                if not cadence_data.empty:
                    avg_cadence = cadence_data.mean()
                    max_cadence = cadence_data.max()
                    min_cadence = cadence_data.min()
                    if len(cadence_data) > 1:
                        cadence_drift = (cadence_data.iloc[-1] - cadence_data.iloc[0]) / duration
            
            # Calculate heart rate metrics
            avg_hr = np.nan
            max_hr = np.nan
            hr_drift = 0
            if 'heart_rate' in interval_data.columns:
                hr_data = interval_data['heart_rate'].dropna()
                if not hr_data.empty:
                    avg_hr = hr_data.mean()
                    max_hr = hr_data.max()
                    if len(hr_data) > 1:
                        hr_drift = hr_data.iloc[-1] - hr_data.iloc[0]
            
            # Calculate speed metrics
            avg_speed_kph = np.nan
            max_speed_kph = np.nan
            speed_gain_kph = 0
            if 'enhanced_speed' in interval_data.columns:
                speed_data = interval_data['enhanced_speed'].dropna()
                if not speed_data.empty:
                    avg_speed_kph = speed_data.mean() * 3.6  # Convert to km/h
                    max_speed_kph = speed_data.max() * 3.6
                    if len(speed_data) > 1:
                        speed_gain_kph = (speed_data.iloc[-1] - speed_data.iloc[0]) * 3.6
            
            # Calculate torque metrics
            avg_torque = np.nan
            max_torque = np.nan
            torque_stability = 0
            if 'torque' in interval_data.columns:
                torque_data = interval_data['torque'].dropna()
                if not torque_data.empty:
                    avg_torque = torque_data.mean()
                    max_torque = torque_data.max()
                    if len(torque_data) > 1:
                        torque_stability = 1 / (1 + torque_data.std())
            
            # Calculate normalized power (if duration allows)
            normalized_power = avg_power
            if duration >= 30:  # Only for efforts >= 30s
                normalized_power = interval_data['power'].rolling(30, min_periods=30).mean().iloc[-1]
            
            # Calculate altitude metrics if available
            altitude_gain = 0
            if 'altitude' in interval_data.columns:
                alt_data = interval_data['altitude'].dropna()
                if len(alt_data) > 1:
                    altitude_gain = alt_data.iloc[-1] - alt_data.iloc[0]
            
            # Calculate grade metrics if available
            avg_grade = np.nan
            if 'grade' in interval_data.columns:
                grade_data = interval_data['grade'].dropna()
                if not grade_data.empty:
                    avg_grade = grade_data.mean()
            
            # Calculate quality score components
            power_score = min(100, (avg_power / ftp) * 100) if ftp > 0 else 0
            duration_score = min(100, (duration / 300) * 100)  # Max at 5min
            consistency_score = max(0, 100 - (power_cv * 200))
            fade_score = max(0, 100 - (power_fade * 100))
            cadence_score = max(0, 100 - abs(cadence_drift) * 10) if not np.isnan(cadence_drift) else 0
            torque_score = torque_stability * 100
            work_score = min(100, (work_kj / (duration / 60)) * 2) if duration > 0 else 0
            
            # Calculate overall quality score
            quality_score = (
                power_score * 0.25 +
                duration_score * 0.20 +
                consistency_score * 0.20 +
                fade_score * 0.15 +
                cadence_score * 0.10 +
                torque_score * 0.05 +
                work_score * 0.05
            )
            
            interval_info = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'duration_s': duration,  # Add this for SprintV1 compatibility
                'duration_min': duration / 60,  # Add duration in minutes
                'avg_power': avg_power,
                'max_power': max_power,
                'work_kj': work_kj,
                'source': 'ml_enhanced',
                'ml_confidence': ml_confidence,
                'quality_score': quality_score,
                # Add SprintV1 compatibility columns
                'start_str': start_time.strftime('%H:%M:%S'),
                'end_str': end_time.strftime('%H:%M:%S'),
                'duration_str': f"{duration}s" if duration < 60 else f"{duration/60:.1f}min",
                'intensity_factor': avg_power / ftp if ftp > 0 else 0,
                'training_zone': classify_interval(duration, avg_power / ftp if ftp > 0 else 0),
                # Add comprehensive metrics
                'power_cv': power_cv,
                'power_fade': power_fade,
                'avg_cadence': avg_cadence,
                'max_cadence': max_cadence,
                'min_cadence': min_cadence,
                'cadence_drift': cadence_drift,
                'avg_hr': avg_hr,
                'max_hr': max_hr,
                'hr_drift': hr_drift,
                'avg_speed_kph': avg_speed_kph,
                'max_speed_kph': max_speed_kph,
                'speed_gain_kph': speed_gain_kph,
                'avg_torque': avg_torque,
                'max_torque': max_torque,
                'torque_stability': torque_stability,
                'altitude_gain': altitude_gain,
                'avg_grade': avg_grade,
                'normalized_power': normalized_power,
                'power_per_kg': avg_power / rider_mass_kg if rider_mass_kg > 0 else np.nan,
                'np_per_kg': normalized_power / rider_mass_kg if rider_mass_kg > 0 else np.nan,
                'work_per_min': work_kj / (duration / 60) if duration > 0 else 0,
                'power_zone': classify_power_zone(avg_power, ftp),
                'hr_zone': classify_hr_zone(avg_hr, LTHR) if not np.isnan(avg_hr) else 'N/A',
                'sprint_efficiency': calculate_sprint_efficiency(avg_power, max_power, power_fade, duration) if duration <= 60 else np.nan,
                'mechanical_efficiency': calculate_mechanical_efficiency(work_kj, speed_gain_kph, duration) if not np.isnan(speed_gain_kph) and speed_gain_kph > 0 else np.nan,
                'power_consistency_score': calculate_power_consistency_score(power_cv, power_fade),
                'overall_performance_score': calculate_overall_performance_score(quality_score, avg_power, ftp, duration),
                'power_to_weight_ratio': avg_power / rider_mass_kg if rider_mass_kg > 0 else np.nan,
                'np_to_weight_ratio': normalized_power / rider_mass_kg if rider_mass_kg > 0 else np.nan,
                'power_zone_number': get_power_zone_number(avg_power, ftp),
                'hr_zone_number': get_hr_zone_number(avg_hr, LTHR)
            }
            all_intervals.append(interval_info)
        
        print(f"✅ Enhanced ML detection found {len(all_intervals)} intervals")
        
    except ImportError as e:
        print(f"⚠️  Enhanced ML detection not available: {e}")
        print(f"🔄 Falling back to original interval detection...")
        return detect_intervals_ml_fallback(df, ftp)
    
    # Create final DataFrame
    intervals_df = pd.DataFrame(all_intervals)
    
    # Sort by quality score (descending) and add ranking
    intervals_df = intervals_df.sort_values('quality_score', ascending=False).reset_index(drop=True)
    intervals_df['rank'] = intervals_df['quality_score'].rank(ascending=False, method='dense').astype(int)
    
    print(f"\n✅ Final result: {len(intervals_df)} intervals detected")
    print(f"   Enhanced ML: {len(all_intervals)} intervals")
    
    return intervals_df


def compare_similar_intervals(intervals_df, comparison_type='duration'):
    """
    Compare intervals of similar characteristics to analyze consistency and patterns.
    
    Args:
        intervals_df: DataFrame with interval information
        comparison_type: Type of comparison ('duration', 'intensity', 'zone', 'source')
    
    Returns:
        DataFrame with comparison results
    """
    if intervals_df.empty:
        return pd.DataFrame()
    
    comparisons = []
    
    if comparison_type == 'duration':
        # Group intervals by duration ranges
        intervals_df['duration_range'] = pd.cut(intervals_df['duration_s'], 
                                               bins=[0, 60, 300, 600, 1200, float('inf')],
                                               labels=['0-1min', '1-5min', '5-10min', '10-20min', '20min+'])
        
        for duration_range in intervals_df['duration_range'].unique():
            if pd.isna(duration_range):
                continue
                
            similar_intervals = intervals_df[intervals_df['duration_range'] == duration_range]
            if len(similar_intervals) >= 2:
                comparison = analyze_interval_group(similar_intervals, f"Duration: {duration_range}")
                comparisons.append(comparison)
    
    elif comparison_type == 'intensity':
        # Group intervals by intensity factor ranges
        intervals_df['intensity_range'] = pd.cut(intervals_df['intensity_factor'], 
                                                bins=[0, 0.8, 1.0, 1.2, 1.5, float('inf')],
                                                labels=['<80% FTP', '80-100% FTP', '100-120% FTP', '120-150% FTP', '>150% FTP'])
        
        for intensity_range in intervals_df['intensity_range'].unique():
            if pd.isna(intensity_range):
                continue
                
            similar_intervals = intervals_df[intervals_df['intensity_range'] == intensity_range]
            if len(similar_intervals) >= 2:
                comparison = analyze_interval_group(similar_intervals, f"Intensity: {intensity_range}")
                comparisons.append(comparison)
    
    elif comparison_type == 'zone':
        # Group intervals by power zone
        if 'power_zone' in intervals_df.columns:
            for zone in intervals_df['power_zone'].unique():
                if pd.isna(zone) or zone == 'N/A':
                    continue
                    
                similar_intervals = intervals_df[intervals_df['power_zone'] == zone]
                if len(similar_intervals) >= 2:
                    comparison = analyze_interval_group(similar_intervals, f"Zone: {zone}")
                    comparisons.append(comparison)
    
    elif comparison_type == 'source':
        # Group intervals by detection source
        if 'source' in intervals_df.columns:
            for source in intervals_df['source'].unique():
                similar_intervals = intervals_df[intervals_df['source'] == source]
                if len(similar_intervals) >= 2:
                    comparison = analyze_interval_group(similar_intervals, f"Source: {source}")
                    comparisons.append(comparison)
    
    if comparisons:
        return pd.DataFrame(comparisons)
    else:
        return pd.DataFrame()


def analyze_interval_group(intervals, group_name):
    """Analyze a group of similar intervals for consistency and patterns."""
    analysis = {'group_name': group_name, 'interval_count': len(intervals)}
    
    # Basic metrics
    if 'avg_power' in intervals.columns:
        analysis.update({
            'avg_power_mean': intervals['avg_power'].mean(),
            'avg_power_std': intervals['avg_power'].std(),
            'avg_power_cv': (intervals['avg_power'].std() / intervals['avg_power'].mean()) * 100 if intervals['avg_power'].mean() > 0 else 0,
            'avg_power_min': intervals['avg_power'].min(),
            'avg_power_max': intervals['avg_power'].max(),
            'avg_power_range': intervals['avg_power'].max() - intervals['avg_power'].min()
        })
    
    if 'max_power' in intervals.columns:
        analysis.update({
            'max_power_mean': intervals['max_power'].mean(),
            'max_power_std': intervals['max_power'].std(),
            'max_power_cv': (intervals['max_power'].std() / intervals['max_power'].mean()) * 100 if intervals['max_power'].mean() > 0 else 0
        })
    
    if 'duration_s' in intervals.columns:
        analysis.update({
            'duration_mean': intervals['duration_s'].mean(),
            'duration_std': intervals['duration_s'].std(),
            'duration_cv': (intervals['duration_s'].std() / intervals['duration_s'].mean()) * 100 if intervals['duration_s'].mean() > 0 else 0
        })
    
    # Cadence metrics
    if 'avg_cadence' in intervals.columns:
        cadence_data = intervals['avg_cadence'].dropna()
        if not cadence_data.empty:
            analysis.update({
                'avg_cadence_mean': cadence_data.mean(),
                'avg_cadence_std': cadence_data.std(),
                'avg_cadence_cv': (cadence_data.std() / cadence_data.mean()) * 100 if cadence_data.mean() > 0 else 0
            })
    
    # Heart rate metrics
    if 'avg_hr' in intervals.columns:
        hr_data = intervals['avg_hr'].dropna()
        if not hr_data.empty:
            analysis.update({
                'avg_hr_mean': hr_data.mean(),
                'avg_hr_std': hr_data.std(),
                'avg_hr_cv': (hr_data.std() / hr_data.mean()) * 100 if hr_data.mean() > 0 else 0
            })
    
    # Speed metrics
    if 'avg_speed_kph' in intervals.columns:
        speed_data = intervals['avg_speed_kph'].dropna()
        if not speed_data.empty:
            analysis.update({
                'avg_speed_mean': speed_data.mean(),
                'avg_speed_std': speed_data.std(),
                'avg_speed_cv': (speed_data.std() / speed_data.mean()) * 100 if speed_data.mean() > 0 else 0
            })
    
    # Quality metrics
    if 'quality_score' in intervals.columns:
        analysis.update({
            'quality_score_mean': intervals['quality_score'].mean(),
            'quality_score_std': intervals['quality_score'].std(),
            'quality_score_cv': (intervals['quality_score'].std() / intervals['quality_score'].mean()) * 100 if intervals['quality_score'].mean() > 0 else 0
        })
    
    # Consistency metrics
    if 'power_cv' in intervals.columns:
        analysis.update({
            'power_cv_mean': intervals['power_cv'].mean(),
            'power_cv_mean': intervals['power_cv'].mean(),
            'power_cv_std': intervals['power_cv'].std()
        })
    
    if 'power_fade' in intervals.columns:
        analysis.update({
            'power_fade_mean': intervals['power_fade'].mean(),
            'power_fade_std': intervals['power_fade'].std()
        })
    
    # Performance trends
    if len(intervals) >= 3 and 'start_time' in intervals.columns:
        # Sort by time and check for trends
        sorted_intervals = intervals.sort_values('start_time')
        first_half = sorted_intervals.iloc[:len(sorted_intervals)//2]
        second_half = sorted_intervals.iloc[len(sorted_intervals)//2:]
        
        if 'avg_power' in intervals.columns:
            first_avg = first_half['avg_power'].mean()
            second_avg = second_half['avg_power'].mean()
            analysis['power_trend'] = (second_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0
    
    return analysis


def create_interval_comparison_plots(intervals_df):
    """Create comprehensive comparison plots for intervals."""
    if intervals_df.empty:
        return []
    
    plots = []
    
    # 1. Duration-based comparison
    duration_comparison = compare_similar_intervals(intervals_df, 'duration')
    if not duration_comparison.empty:
        plots.append(create_duration_comparison_plot(duration_comparison))
    
    # 2. Intensity-based comparison
    intensity_comparison = compare_similar_intervals(intervals_df, 'intensity')
    if not intensity_comparison.empty:
        plots.append(create_intensity_comparison_plot(intensity_comparison))
    
    # 3. Zone-based comparison
    zone_comparison = compare_similar_intervals(intervals_df, 'zone')
    if not zone_comparison.empty:
        plots.append(create_zone_comparison_plot(zone_comparison))
    
    # 4. Source-based comparison
    source_comparison = compare_similar_intervals(intervals_df, 'source')
    if not source_comparison.empty:
        plots.append(create_source_comparison_plot(source_comparison))
    
    # 5. Performance consistency analysis
    consistency_plot = create_consistency_analysis_plot(intervals_df)
    if consistency_plot:
        plots.append(consistency_plot)
    
    return plots


def create_duration_comparison_plot(duration_comparison):
    """Create plot comparing intervals by duration groups."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Power consistency by duration
    fig.add_trace(go.Bar(
        x=duration_comparison['group_name'],
        y=duration_comparison['avg_power_cv'],
        name='Power CV (%)',
        marker_color='#e74c3c',
        text=[f"{cv:.1f}%" for cv in duration_comparison['avg_power_cv']],
        textposition='auto',
        hovertemplate="Group: %{x}<br>Power CV: %{y:.1f}%<br>Count: %{customdata}<extra></extra>",
        customdata=duration_comparison['interval_count']
    ))
    
    # Add cadence consistency if available
    if 'avg_cadence_cv' in duration_comparison.columns:
        fig.add_trace(go.Bar(
            x=duration_comparison['group_name'],
            y=duration_comparison['avg_cadence_cv'],
            name='Cadence CV (%)',
            marker_color='#3498db',
            text=[f"{cv:.1f}%" for cv in duration_comparison['avg_cadence_cv']],
            textposition='auto',
            hovertemplate="Group: %{x}<br>Cadence CV: %{y:.1f}%<br>Count: %{customdata}<extra></extra>",
            customdata=duration_comparison['interval_count']
        ))
    else:
        # Duration consistency as fallback
        fig.add_trace(go.Bar(
            x=duration_comparison['group_name'],
            y=duration_comparison['duration_cv'],
            name='Duration CV (%)',
            marker_color='#f39c12',
            text=[f"{cv:.1f}%" for cv in duration_comparison['duration_cv']],
            textposition='auto',
            hovertemplate="Group: %{x}<br>Duration CV: %{y:.1f}%<br>Count: %{customdata}<extra></extra>",
            customdata=duration_comparison['interval_count']
        ))
    
    fig.update_layout(
        title='Interval Consistency by Duration Group (Lower CV = More Consistent)',
        xaxis_title='Duration Groups',
        yaxis_title='Coefficient of Variation (%)',
        template='plotly_white',
        height=300,
        barmode='group',
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=11)
    )
    
    return fig


def create_intensity_comparison_plot(intensity_comparison):
    """Create plot comparing intervals by intensity groups."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Power consistency by intensity
    fig.add_trace(go.Bar(
        x=intensity_comparison['group_name'],
        y=intensity_comparison['avg_power_cv'],
        name='Power CV (%)',
        marker_color='#27ae60',
        text=[f"{cv:.1f}%" for cv in intensity_comparison['avg_power_cv']],
        textposition='auto',
        hovertemplate="Group: %{x}<br>Power CV: %{y:.1f}%<br>Avg Power: %{customdata}W<extra></extra>",
        customdata=intensity_comparison['avg_power_mean']
    ))
    
    # Quality score by intensity
    if 'quality_score_cv' in intensity_comparison.columns:
        fig.add_trace(go.Bar(
            x=intensity_comparison['group_name'],
            y=intensity_comparison['quality_score_cv'],
            name='Quality Score CV (%)',
            marker_color='#e67e22',
            text=[f"{cv:.1f}%" for cv in intensity_comparison['quality_score_cv']],
            textposition='auto',
            hovertemplate="Group: %{x}<br>Quality CV: %{y:.1f}%<br>Avg Quality: %{customdata}<extra></extra>",
            customdata=intensity_comparison['quality_score_mean']
        ))
    
    fig.update_layout(
        title='Interval Consistency by Intensity Group (Lower CV = More Consistent)',
        xaxis_title='Intensity Groups',
        yaxis_title='Coefficient of Variation (%)',
        template='plotly_white',
        height=300,
        barmode='group',
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=11)
    )
    
    return fig


def create_zone_comparison_plot(zone_comparison):
    """Create plot comparing intervals by power zones."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Power consistency by zone
    fig.add_trace(go.Bar(
        x=zone_comparison['group_name'],
        y=zone_comparison['avg_power_cv'],
        name='Power CV (%)',
        marker_color='#8e44ad',
        text=[f"{cv:.1f}%" for cv in zone_comparison['avg_power_cv']],
        textposition='auto',
        hovertemplate="Zone: %{x}<br>Power CV: %{y:.1f}%<br>Count: %{customdata}<extra></extra>",
        customdata=zone_comparison['interval_count']
    ))
    
    # Cadence consistency by zone
    if 'avg_cadence_cv' in zone_comparison.columns:
        fig.add_trace(go.Bar(
            x=zone_comparison['group_name'],
            y=zone_comparison['avg_cadence_cv'],
            name='Cadence CV (%)',
            marker_color='#16a085',
            text=[f"{cv:.1f}%" for cv in zone_comparison['avg_cadence_cv']],
            textposition='auto',
            hovertemplate="Zone: %{x}<br>Cadence CV: %{y:.1f}%<br>Count: %{customdata}<extra></extra>",
            customdata=zone_comparison['interval_count']
        ))
    
    fig.update_layout(
        title='Interval Consistency by Power Zone (Lower CV = More Consistent)',
        xaxis_title='Power Zones',
        yaxis_title='Coefficient of Variation (%)',
        template='plotly_white',
        height=300,
        barmode='group',
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=11)
    )
    
    return fig


def create_source_comparison_plot(source_comparison):
    """Create plot comparing intervals by detection source."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Power consistency by source
    fig.add_trace(go.Bar(
        x=source_comparison['group_name'],
        y=source_comparison['avg_power_cv'],
        name='Power CV (%)',
        marker_color='#e377c2',
        text=[f"{cv:.1f}%" for cv in source_comparison['avg_power_cv']],
        textposition='auto'
    ))
    
    # Quality score by source
    if 'quality_score_cv' in source_comparison.columns:
        fig.add_trace(go.Bar(
            x=source_comparison['group_name'],
            y=source_comparison['quality_score_cv'],
            name='Quality Score CV (%)',
            marker_color='#7f7f7f',
            text=[f"{cv:.1f}%" for cv in source_comparison['quality_score_cv']],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Interval Consistency by Detection Source (Lower CV = More Consistent)',
        xaxis_title='Detection Sources',
        yaxis_title='Coefficient of Variation (%)',
        template='plotly_white',
        height=300,
        barmode='group',
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=11)
    )
    
    return fig


def create_consistency_analysis_plot(intervals_df):
    """Create plot showing overall consistency analysis."""
    import plotly.graph_objects as go
    
    if intervals_df.empty:
        return None
    
    fig = go.Figure()
    
    # Calculate consistency metrics for each interval
    consistency_data = []
    for _, interval in intervals_df.iterrows():
        consistency_score = 0
        factors = []
        
        # Power consistency (lower CV = better)
        if 'power_cv' in interval and not pd.isna(interval['power_cv']):
            power_consistency = max(0, 100 - interval['power_cv'] * 10)
            consistency_score += power_consistency * 0.4
            factors.append(f"Power: {power_consistency:.0f}")
        
        # Duration consistency (if multiple similar intervals)
        if 'duration_s' in interval:
            duration_consistency = 100  # Base score, will be adjusted in group analysis
            consistency_score += duration_consistency * 0.2
            factors.append(f"Duration: {duration_consistency:.0f}")
        
        # Quality consistency
        if 'quality_score' in interval and not pd.isna(interval['quality_score']):
            quality_consistency = min(100, interval['quality_score'])
            consistency_score += quality_consistency * 0.4
            factors.append(f"Quality: {quality_consistency:.0f}")
        
        consistency_data.append({
            'interval': f"#{interval.get('rank', 'N/A')}",
            'consistency_score': consistency_score,
            'factors': ' | '.join(factors)
        })
    
    if not consistency_data:
        return None
    
    consistency_df = pd.DataFrame(consistency_data)
    
    fig.add_trace(go.Bar(
        x=consistency_df['interval'],
        y=consistency_df['consistency_score'],
        name='Consistency Score',
        marker_color='#bcbd22',
        text=[f"{score:.0f}" for score in consistency_df['consistency_score']],
        textposition='auto',
        hovertemplate="Interval: %{x}<br>Consistency Score: %{y:.0f}<br>Factors: %{customdata}<extra></extra>",
        customdata=consistency_df['factors']
    ))
    
    fig.update_layout(
        title='Interval Consistency Analysis (Higher = More Consistent)',
        xaxis_title='Intervals',
        yaxis_title='Consistency Score (0-100)',
        template='plotly_white',
        height=300,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=11)
    )
    
    return fig


def display_interval_comparison_analysis(intervals_df):
    """Display comprehensive interval comparison analysis."""
    if intervals_df.empty:
        print("\nNo intervals available for comparison analysis.")
        return
    
    print("\n" + "="*80)
    print("INTERVAL COMPARISON ANALYSIS")
    print("="*80)
    
    # 1. Duration-based comparison
    print("\nDURATION-BASED COMPARISON")
    print("-" * 50)
    duration_comparison = compare_similar_intervals(intervals_df, 'duration')
    if not duration_comparison.empty:
        display_comparison_table(duration_comparison, 'Duration Groups')
    else:
        print("Insufficient intervals for duration-based comparison.")
    
    # 2. Intensity-based comparison
    print("\nINTENSITY-BASED COMPARISON")
    print("-" * 50)
    intensity_comparison = compare_similar_intervals(intervals_df, 'intensity')
    if not intensity_comparison.empty:
        display_comparison_table(intensity_comparison, 'Intensity Groups')
    else:
        print("Insufficient intervals for intensity-based comparison.")
    
    # 3. Zone-based comparison
    print("\nZONE-BASED COMPARISON")
    print("-" * 50)
    zone_comparison = compare_similar_intervals(intervals_df, 'zone')
    if not zone_comparison.empty:
        display_comparison_table(zone_comparison, 'Zone Groups')
    else:
        print("Insufficient intervals for zone-based comparison.")
    
    # 4. Source-based comparison
    print("\nSOURCE-BASED COMPARISON")
    print("-" * 50)
    source_comparison = compare_similar_intervals(intervals_df, 'source')
    if not source_comparison.empty:
        display_comparison_table(source_comparison, 'Detection Sources')
    else:
        print("Insufficient intervals for source-based comparison.")
    
    # 5. Overall consistency summary
    print("\nOVERALL CONSISTENCY SUMMARY")
    print("-" * 50)
    display_consistency_summary(intervals_df)


def display_comparison_table(comparison_df, comparison_type):
    """Display comparison table for a specific comparison type."""
    if comparison_df.empty:
        return
    
    print(f"\n{comparison_type} Analysis:")
    
    # Select key columns for display
    display_cols = ['group_name', 'interval_count']
    
    # Add available metrics
    if 'avg_power_cv' in comparison_df.columns:
        display_cols.extend(['avg_power_mean', 'avg_power_cv'])
    if 'duration_cv' in comparison_df.columns:
        display_cols.extend(['duration_mean', 'duration_cv'])
    if 'quality_score_cv' in comparison_df.columns:
        display_cols.extend(['quality_score_mean', 'quality_score_cv'])
    if 'avg_cadence_cv' in comparison_df.columns:
        display_cols.extend(['avg_cadence_mean', 'avg_cadence_cv'])
    
    # Create display DataFrame
    display_df = comparison_df[display_cols].copy()
    
    # Rename columns for better display
    column_mapping = {
        'group_name': 'Group',
        'interval_count': 'Count',
        'avg_power_mean': 'Avg Power (W)',
        'avg_power_cv': 'Power CV (%)',
        'duration_mean': 'Duration (s)',
        'duration_cv': 'Duration CV (%)',
        'quality_score_mean': 'Quality Score',
        'quality_score_cv': 'Quality CV (%)',
        'avg_cadence_mean': 'Avg Cadence (rpm)',
        'avg_cadence_cv': 'Cadence CV (%)'
    }
    
    display_df.columns = [column_mapping.get(col, col) for col in display_df.columns]
    
    # Format numeric columns
    for col in display_df.columns:
        if 'CV' in col:
            display_df[col] = display_df[col].round(1)
        elif 'Power' in col or 'Duration' in col or 'Quality' in col:
            display_df[col] = display_df[col].round(0)
        elif 'Cadence' in col:
            display_df[col] = display_df[col].round(1)
    
    print(display_df.to_string(index=False))


def display_consistency_summary(intervals_df):
    """Display overall consistency summary."""
    if intervals_df.empty:
        return
    
    print("\nConsistency Analysis Summary:")
    
    # Calculate overall consistency metrics
    consistency_metrics = []
    
    if 'avg_power' in intervals_df.columns:
        power_cv = (intervals_df['avg_power'].std() / intervals_df['avg_power'].mean()) * 100 if intervals_df['avg_power'].mean() > 0 else 0
        consistency_metrics.append(f"Power CV: {power_cv:.1f}%")
    
    if 'duration_s' in intervals_df.columns:
        duration_cv = (intervals_df['duration_s'].std() / intervals_df['duration_s'].mean()) * 100 if intervals_df['duration_s'].mean() > 0 else 0
        consistency_metrics.append(f"Duration CV: {duration_cv:.1f}%")
    
    if 'quality_score' in intervals_df.columns:
        quality_cv = (intervals_df['quality_score'].std() / intervals_df['quality_score'].mean()) * 100 if intervals_df['quality_score'].mean() > 0 else 0
        consistency_metrics.append(f"Quality Score CV: {quality_cv:.1f}%")
    
    if 'avg_cadence' in intervals_df.columns:
        cadence_data = intervals_df['avg_cadence'].dropna()
        if not cadence_data.empty:
            cadence_cv = (cadence_data.std() / cadence_data.mean()) * 100 if cadence_data.mean() > 0 else 0
            consistency_metrics.append(f"Cadence CV: {cadence_cv:.1f}%")
    
    # Display consistency metrics
    for metric in consistency_metrics:
        print(f"   • {metric}")
    
    # Consistency rating
    if consistency_metrics:
        avg_cv = np.mean([float(metric.split(': ')[1].replace('%', '')) for metric in consistency_metrics])
        if avg_cv < 10:
            rating = "Excellent consistency"
        elif avg_cv < 20:
            rating = "Good consistency"
        elif avg_cv < 30:
            rating = "Moderate consistency"
        else:
            rating = "Variable performance"
        
        print(f"\nOverall Rating: {rating} (Average CV: {avg_cv:.1f}%)")


def create_comparison_table_html(comparison_df, comparison_type):
    """Create HTML table for comparison data."""
    try:
        from dash import html
    except ImportError:
        return "HTML table not available (Dash not imported)"
    
    if comparison_df.empty:
        return html.P("No comparison data available")
    
    # Select key columns for display
    display_cols = ['group_name', 'interval_count']
    
    # Add available metrics
    if 'avg_power_cv' in comparison_df.columns:
        display_cols.extend(['avg_power_mean', 'avg_power_cv'])
    if 'duration_cv' in comparison_df.columns:
        display_cols.extend(['duration_mean', 'duration_cv'])
    if 'quality_score_cv' in comparison_df.columns:
        display_cols.extend(['quality_score_mean', 'quality_score_cv'])
    if 'avg_cadence_cv' in comparison_df.columns:
        display_cols.extend(['avg_cadence_mean', 'avg_cadence_cv'])
    
    # Create display DataFrame
    display_df = comparison_df[display_cols].copy()
    
    # Rename columns for better display
    column_mapping = {
        'group_name': 'Group',
        'interval_count': 'Count',
        'avg_power_mean': 'Avg Power (W)',
        'avg_power_cv': 'Power CV (%)',
        'duration_mean': 'Duration (s)',
        'duration_cv': 'Duration CV (%)',
        'quality_score_mean': 'Quality Score',
        'quality_score_cv': 'Quality CV (%)',
        'avg_cadence_mean': 'Avg Cadence (rpm)',
        'avg_cadence_cv': 'Cadence CV (%)'
    }
    
    display_df.columns = [column_mapping.get(col, col) for col in display_df.columns]
    
    # Format numeric columns
    for col in display_df.columns:
        if 'CV' in col:
            display_df[col] = display_df[col].round(1)
        elif 'Power' in col or 'Duration' in col or 'Quality' in col:
            display_df[col] = display_df[col].round(0)
        elif 'Cadence' in col:
            display_df[col] = display_df[col].round(1)
    
    # Create HTML table
    table_data = []
    for _, row in display_df.iterrows():
        table_data.append(html.Tr([html.Td(str(row[col])) for col in display_df.columns]))
    
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in display_df.columns])),
        html.Tbody(table_data)
    ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse', 'marginBottom': '20px'})


def create_consistency_summary_html(intervals_df):
    """Create HTML consistency summary."""
    try:
        from dash import html
    except ImportError:
        return "HTML summary not available (Dash not imported)"
    
    if intervals_df.empty:
        return html.P("No consistency data available")
    
    # Calculate overall consistency metrics
    consistency_metrics = []
    
    if 'avg_power' in intervals_df.columns:
        power_cv = (intervals_df['avg_power'].std() / intervals_df['avg_power'].mean()) * 100 if intervals_df['avg_power'].mean() > 0 else 0
        consistency_metrics.append(f"Power CV: {power_cv:.1f}%")
    
    if 'duration_s' in intervals_df.columns:
        duration_cv = (intervals_df['duration_s'].std() / intervals_df['duration_s'].mean()) * 100 if intervals_df['duration_s'].mean() > 0 else 0
        consistency_metrics.append(f"Duration CV: {duration_cv:.1f}%")
    
    if 'quality_score' in intervals_df.columns:
        quality_cv = (intervals_df['quality_score'].std() / intervals_df['quality_score'].mean()) * 100 if intervals_df['quality_score'].mean() > 0 else 0
        consistency_metrics.append(f"Quality Score CV: {quality_cv:.1f}%")
    
    if 'avg_cadence' in intervals_df.columns:
        cadence_data = intervals_df['avg_cadence'].dropna()
        if not cadence_data.empty:
            cadence_cv = (cadence_data.std() / cadence_data.mean()) * 100 if cadence_data.mean() > 0 else 0
            consistency_metrics.append(f"Cadence CV: {cadence_cv:.1f}%")
    
    # Create HTML list
    metric_items = [html.Li(metric) for metric in consistency_metrics]
    
    # Consistency rating
    if consistency_metrics:
        avg_cv = np.mean([float(metric.split(': ')[1].replace('%', '')) for metric in consistency_metrics])
        if avg_cv < 10:
            rating = "Excellent consistency"
            rating_color = "#27ae60"
        elif avg_cv < 20:
            rating = "Good consistency"
            rating_color = "#f39c12"
        elif avg_cv < 30:
            rating = "Moderate consistency"
            rating_color = "#e67e22"
        else:
            rating = "Variable performance"
            rating_color = "#e74c3c"
        
        rating_text = html.P(f"Overall Rating: {rating} (Average CV: {avg_cv:.1f}%)", 
                           style={'color': rating_color, 'fontWeight': 'bold', 'marginTop': '10px'})
    else:
        rating_text = html.P("No consistency data available")
    
    return html.Div([
        html.Ul(metric_items, style={'marginBottom': '20px'}),
        rating_text
    ])


def extract_lap_intervals(df, ftp):
    """Extract intervals from lap data if available."""
    intervals = []
    
    for lap_num, lap_data in df.groupby('lap'):
        if len(lap_data) < 2:  # Skip laps with insufficient data
            continue
            
        duration = (lap_data.index[-1] - lap_data.index[0]).total_seconds()
        if duration < 10:  # Skip very short laps
            continue
            
        # Calculate lap metrics
        avg_power = lap_data['power'].mean()
        max_power = lap_data['power'].max()
        work_kj = lap_data['power'].sum() / 1000
        
        # Calculate fade (power drop during lap)
        power_fade = (max_power - avg_power) / max_power if max_power > 0 else 0
        
        # Calculate power consistency
        power_cv = lap_data['power'].std() / avg_power if avg_power > 0 else 0
        
        # Calculate cadence drift
        cadence_drift = 0
        if 'cadence' in lap_data.columns:
            cadence_data = lap_data['cadence'].dropna()
            if len(cadence_data) > 1:
                cadence_drift = (cadence_data.iloc[-1] - cadence_data.iloc[0]) / duration
        
        # Calculate speed gain
        speed_gain = 0
        if 'enhanced_speed' in lap_data.columns:
            speed_data = lap_data['enhanced_speed'].dropna()
            if len(speed_data) > 1:
                speed_gain = (speed_data.iloc[-1] - speed_data.iloc[0]) * 3.6  # km/h
        
        # Calculate torque stability
        torque_stability = 0
        if 'torque' in lap_data.columns:
            torque_data = lap_data['torque'].dropna()
            if len(torque_data) > 1:
                torque_stability = 1 / (1 + torque_data.std())
        
        interval_info = {
            'source': 'lap',
            'lap_number': lap_num,
            'start_time': lap_data.index[0],
            'end_time': lap_data.index[-1],
            'duration_s': duration,
            'duration_min': duration / 60,
            'start_str': lap_data.index[0].strftime('%H:%M:%S'),
            'end_str': lap_data.index[-1].strftime('%H:%M:%S'),
            'avg_power': avg_power,
            'max_power': max_power,
            'work_kj': work_kj,
            'intensity_factor': avg_power / ftp,
            'power_fade': power_fade,
            'power_cv': power_cv,
            'cadence_drift': cadence_drift,
            'speed_gain_kph': speed_gain,
            'torque_stability': torque_stability,
            'work_per_min': work_kj / (duration / 60) if duration > 0 else 0
        }
        
        # Add additional metrics if available
        if 'cadence' in lap_data.columns:
            cadence_data = lap_data['cadence'].dropna()
            if not cadence_data.empty:
                interval_info.update({
                    'avg_cadence': cadence_data.mean(),
                    'max_cadence': cadence_data.max(),
                    'min_cadence': cadence_data.min()
                })
        
        if 'heart_rate' in lap_data.columns:
            hr_data = lap_data['heart_rate'].dropna()
            if not hr_data.empty:
                interval_info.update({
                    'avg_hr': hr_data.mean(),
                    'max_hr': hr_data.max(),
                    'hr_drift': hr_data.iloc[-1] - hr_data.iloc[0]
                })
        
        intervals.append(interval_info)
    
    return intervals


def detect_power_based_intervals(df, ftp, min_duration, power_threshold_pct, cv_threshold):
    """Auto-detect intervals using adaptive detection with bias toward common training durations."""
    intervals = []
    
    print(f"      🔍 Adaptive interval detection with common duration bias:")
    print(f"         Looking for intervals matching standard training patterns")
    
    # === Adaptive Interval Detection with Bias Toward Common Durations ===
    
    # 1. Define biased durations (in seconds)
    common_durations = [10, 15, 20, 30, 40, 45, 60, 120, 180, 240, 300, 360, 480, 600, 720, 900, 1200]
    bias_weight = 1.1  # Multiplier for fit_score when duration is in common_durations
    
    print(f"         Common durations: {', '.join([f'{d}s' if d < 60 else f'{d//60}min' for d in common_durations])}")
    print(f"         Bias weight: {bias_weight}x for common durations")
    
    # 2. Smooth power data
    df_copy = df.copy()
    df_copy['power_smooth'] = df_copy['power'].rolling(window=5, min_periods=1).mean()
    
    print(f"      📊 Power range: {df['power'].min():.0f}W - {df['power'].max():.0f}W")
    print(f"         Using 5s rolling average for smooth detection")
    
    # 3. Initialize empty list to store interval candidates
    interval_candidates = []
    
    # 4. Loop over rolling window durations
    for duration in common_durations:
        window_samples = duration  # assumes 1Hz sampling
        rolling_avg = df_copy['power_smooth'].rolling(window=window_samples, min_periods=window_samples).mean()
        rolling_std = df_copy['power_smooth'].rolling(window=window_samples, min_periods=window_samples).std()
        
        print(f"         🔍 Analyzing {duration}s windows...")
        
        for i in range(window_samples, len(df_copy) - 10):
            avg_power = rolling_avg.iloc[i]
            std_power = rolling_std.iloc[i]
            post_window_power = df_copy['power_smooth'].iloc[i+1:i+11].mean()
            
            # Avoid NaNs
            if np.isnan(avg_power) or np.isnan(std_power) or np.isnan(post_window_power):
                continue
            
            # Calculate quality scores
            drop_score = max(0, (avg_power - post_window_power) / avg_power)
            std_score = std_power / avg_power  # normalized std
            fit_score = avg_power * (1 - std_score) * (1 - drop_score)
            
            # Boost score if this is a common interval duration
            if duration in common_durations:
                biased_score = fit_score * bias_weight
                bias_applied = True
            else:
                biased_score = fit_score
                bias_applied = False
            
            # Only consider intervals above FTP threshold
            if avg_power >= (ftp * power_threshold_pct):
                interval_candidates.append({
                    'start_idx': i - window_samples + 1,
                    'end_idx': i,
                    'duration': duration,
                    'avg_power': avg_power,
                    'fit_score': biased_score,
                    'raw_score': fit_score,
                    'bias_applied': bias_applied,
                    'std_score': std_score,
                    'drop_score': drop_score
                })
    
    print(f"         Found {len(interval_candidates)} total candidates")
    
    # 5. Convert to DataFrame and remove overlapping intervals (keep best-fit)
    if interval_candidates:
        interval_df = pd.DataFrame(interval_candidates)
        interval_df = interval_df.sort_values(by='fit_score', ascending=False)
        
        print(f"         Top 5 candidates by fit score:")
        for i, (_, row) in enumerate(interval_df.head().iterrows()):
            start_time = df.index[int(row['start_idx'])]
            print(f"            #{i+1}: {row['duration']}s at {start_time.strftime('%H:%M:%S')} - {row['avg_power']:.0f}W (score: {row['fit_score']:.0f})")
        
        # 6. Non-overlapping selection (greedy)
        final_intervals = []
        used = np.zeros(len(df_copy), dtype=bool)
        
        for _, row in interval_df.iterrows():
            start_idx = int(row['start_idx'])
            end_idx = int(row['end_idx'])
            
            # Check if this interval overlaps with already selected ones
            if not used[start_idx:end_idx].any():
                final_intervals.append(row)
                used[start_idx:end_idx] = True
                print(f"         ✅ Selected: {row['duration']}s interval at {df.index[start_idx].strftime('%H:%M:%S')} - {row['avg_power']:.0f}W")
            else:
                print(f"         ❌ Skipped: {row['duration']}s interval at {df.index[start_idx].strftime('%H:%M:%S')} (overlaps)")
        
        # 7. Final output
        effort_starts = []
        effort_ends = []
        
        for interval in final_intervals:
            if interval['duration'] >= min_duration:
                start_time = df.index[int(interval['start_idx'])]
                end_time = df.index[int(interval['end_idx'])]
                effort_starts.append(start_time)
                effort_ends.append(end_time)
        
        print(f"      🎯 Found {len(effort_starts)} final efforts after adaptive detection")
    else:
        print(f"      ❌ No interval candidates found")
        effort_starts = []
        effort_ends = []
    
    # Process each detected effort
    for start, end in zip(effort_starts, effort_ends):
        duration = (end - start).total_seconds()
        
        # Filter by duration
        if duration < min_duration:
            continue
        
        # Get effort data
        effort_data = df.loc[start:end]
        
        # Calculate power consistency (CV)
        avg_power = effort_data['power'].mean()
        power_cv = effort_data['power'].std() / avg_power if avg_power > 0 else 0
        
        print(f"         ⏱️  Duration: {duration:.0f}s, Avg Power: {avg_power:.0f}W, CV: {power_cv:.3f}")
        
        # Filter by power consistency
        if power_cv > cv_threshold:
            print(f"         ❌ Rejected: CV {power_cv:.3f} > {cv_threshold}")
            continue
        
        # Additional quality checks: filter out intervals with data gaps or flatlines
        if effort_data['power'].isna().sum() > len(effort_data) * 0.1:  # >10% NaN values
            print(f"         ❌ Rejected: too many power data gaps")
            continue
        
        if effort_data['power'].std() < 5:  # Power variation too low (flatline)
            print(f"         ❌ Rejected: power too flat (std: {effort_data['power'].std():.1f}W)")
            continue
        
        # Calculate effort metrics
        max_power = effort_data['power'].max()
        work_kj = effort_data['power'].sum() / 1000
        
        # Calculate fade
        power_fade = (max_power - avg_power) / max_power if max_power > 0 else 0
        
        # Calculate cadence drift
        cadence_drift = 0
        if 'cadence' in effort_data.columns:
            cadence_data = effort_data['cadence'].dropna()
            if len(cadence_data) > 1:
                cadence_drift = (cadence_data.iloc[-1] - cadence_data.iloc[0]) / duration
        
        # Calculate speed gain
        speed_gain = 0
        if 'enhanced_speed' in effort_data.columns:
            speed_data = effort_data['enhanced_speed'].dropna()
            if len(speed_data) > 1:
                speed_gain = (speed_data.iloc[-1] - speed_data.iloc[0]) * 3.6
        
        # Calculate torque stability
        torque_stability = 0
        if 'torque' in effort_data.columns:
            torque_data = effort_data['torque'].dropna()
            if len(torque_data) > 1:
                torque_stability = 1 / (1 + torque_data.std())
        
        # Calculate quality score components
        power_score = min(100, (avg_power / ftp) * 100)
        duration_score = min(100, (duration / 300) * 100)  # Max at 5min
        consistency_score = max(0, 100 - (power_cv * 200))
        fade_score = max(0, 100 - (power_fade * 100))
        cadence_score = max(0, 100 - abs(cadence_drift) * 10)
        torque_score = torque_stability * 100
        work_score = min(100, (work_kj / (duration / 60)) * 2)
        
        # Calculate overall quality score
        quality_score = (
            power_score * 0.25 +
            duration_score * 0.20 +
            consistency_score * 0.20 +
            fade_score * 0.15 +
            cadence_score * 0.10 +
            torque_score * 0.05 +
            work_score * 0.05
        )
        
        interval_info = {
            'source': 'auto',
            'start_time': start,
            'end_time': end,
            'duration_s': duration,
            'duration_min': duration / 60,
            'start_str': start.strftime('%H:%M:%S'),
            'end_str': end.strftime('%H:%M:%S'),
            'avg_power': avg_power,
            'max_power': max_power,
            'work_kj': work_kj,
            'intensity_factor': avg_power / ftp,
            'power_fade': power_fade,
            'power_cv': power_cv,
            'cadence_drift': cadence_drift,
            'speed_gain_kph': speed_gain,
            'torque_stability': torque_stability,
            'work_per_min': work_kj / (duration / 60) if duration > 0 else 0,
            'quality_score': quality_score
        }
        
        # Add additional metrics if available
        if 'cadence' in effort_data.columns:
            cadence_data = effort_data['cadence'].dropna()
            if not cadence_data.empty:
                interval_info.update({
                    'avg_cadence': cadence_data.mean(),
                    'max_cadence': cadence_data.max(),
                    'min_cadence': cadence_data.min()
                })
        
        if 'heart_rate' in effort_data.columns:
            hr_data = effort_data['heart_rate'].dropna()
            if not hr_data.empty:
                interval_info.update({
                    'avg_hr': hr_data.mean(),
                    'max_hr': hr_data.max(),
                    'hr_drift': hr_data.iloc[-1] - hr_data.iloc[0]
                })
        
        intervals.append(interval_info)
        print(f"         ✅ Accepted: {len(intervals)} intervals so far - Q:{quality_score:.1f}")
    
    print(f"      🎉 Final result: {len(intervals)} auto-detected intervals")
    return intervals


def detect_fallback_intervals(df, ftp):
    """Fallback interval detection when laps are missing or insufficient.
    
    Refined Heuristics:
    - 30s rolling average power > 60% FTP
    - Duration >= 30s with quality filtering
    - Power stability score > 50
    - Gap between intervals ≥ 15s
    - Filter out low-quality or noisy sections
    
    Args:
        df: DataFrame with cycling data
        ftp: Functional Threshold Power in watts
    
    Returns:
        List of interval dictionaries with comprehensive stats
    """
    print(f"      🔍 Starting refined fallback detection with 60% FTP threshold...")
    
    intervals = []
    
    # Refined parameters for quality detection
    power_threshold = 0.60 * ftp  # 60% FTP
    min_duration = 30  # 30 seconds minimum (increased from 15s)
    min_gap = 15  # Minimum gap between intervals
    min_quality_score = 50  # Minimum quality score threshold
    
    # Create adaptive fallback detection with common duration bias
    df_copy = df.copy()
    
    print(f"         Fallback adaptive detection:")
    print(f"            Using more sensitive thresholds for comprehensive coverage")
    
    # Use more sensitive parameters for fallback
    fallback_power_threshold = 0.60  # 60% FTP instead of 75%
    fallback_min_duration = 15       # 15s instead of 20s
    
    print(f"         Min duration: {fallback_min_duration}s")
    print(f"         Min gap: {min_gap}s")
    print(f"         Min quality score: {min_quality_score}")
    print(f"         Power threshold: {fallback_power_threshold*100:.0f}% FTP")
    
    # Use the same adaptive detection but with fallback parameters
    effort_starts, effort_ends = detect_adaptive_intervals_fallback(
        df_copy, ftp, fallback_min_duration, fallback_power_threshold
    )
    
    print(f"         Found {len(effort_starts)} final fallback efforts after adaptive detection")
    

    
    # Process each detected effort with quality filtering
    for i, (start, end) in enumerate(zip(effort_starts, effort_ends)):
        duration = (end - start).total_seconds()
        
        # Filter by duration
        if duration < min_duration:
            continue
        
        # Get effort data
        effort_data = df.loc[start:end]
        
        # Calculate comprehensive metrics
        avg_power = effort_data['power'].mean()
        max_power = effort_data['power'].max()
        work_kj = effort_data['power'].sum() / 1000
        
        # Calculate power consistency (CV)
        power_cv = effort_data['power'].std() / avg_power if avg_power > 0 else 0
        
        # Calculate fade
        power_fade = (max_power - avg_power) / max_power if max_power > 0 else 0
        
        # Calculate cadence drift
        cadence_drift = 0
        if 'cadence' in effort_data.columns:
            cadence_data = effort_data['cadence'].dropna()
            if len(cadence_data) > 1:
                cadence_drift = (cadence_data.iloc[-1] - cadence_data.iloc[0]) / duration
        
        # Calculate speed gain
        speed_gain = 0
        if 'enhanced_speed' in effort_data.columns:
            speed_data = effort_data['enhanced_speed'].dropna()
            if len(speed_data) > 1:
                speed_gain = (speed_data.iloc[-1] - speed_data.iloc[0]) * 3.6
        
        # Calculate torque stability
        torque_stability = 0
        if 'torque' in effort_data.columns:
            torque_data = effort_data['torque'].dropna()
            if len(torque_data) > 1:
                torque_stability = 1 / (1 + torque_data.std())
        
        # Calculate normalized power (if duration allows)
        normalized_power = avg_power
        if duration >= 30:  # Only for efforts >= 30s
            normalized_power = effort_data['power'].rolling(30, min_periods=30).mean().iloc[-1]
        
        # Calculate quality score components
        power_score = min(100, (avg_power / ftp) * 100)
        duration_score = min(100, (duration / 300) * 100)  # Max at 5min
        consistency_score = max(0, 100 - (power_cv * 200))
        fade_score = max(0, 100 - (power_fade * 100))
        cadence_score = max(0, 100 - abs(cadence_drift) * 10)
        torque_score = torque_stability * 100
        work_score = min(100, (work_kj / (duration / 60)) * 2)
        
        # Calculate overall quality score
        quality_score = (
            power_score * 0.25 +
            duration_score * 0.20 +
            consistency_score * 0.20 +
            fade_score * 0.15 +
            cadence_score * 0.10 +
            torque_score * 0.05 +
            work_score * 0.05
        )
        
        # Filter by quality score
        if quality_score < min_quality_score:
            print(f"            ❌ Skipping interval: quality score {quality_score:.1f} < {min_quality_score}")
            continue
        
        # Create interval info
        interval_info = {
            'source': 'fallback',
            'start_time': start,
            'end_time': end,
            'duration_s': duration,
            'duration_min': duration / 60,
            'start_str': start.strftime('%H:%M:%S'),
            'end_str': end.strftime('%H:%M:%S'),
            'avg_power': avg_power,
            'max_power': max_power,
            'normalized_power': normalized_power,
            'work_kj': work_kj,
            'intensity_factor': avg_power / ftp,
            'power_fade': power_fade,
            'power_cv': power_cv,
            'cadence_drift': cadence_drift,
            'speed_gain_kph': speed_gain,
            'torque_stability': torque_stability,
            'work_per_min': work_kj / (duration / 60) if duration > 0 else 0
        }
        
        # Add additional metrics if available
        if 'cadence' in effort_data.columns:
            cadence_data = effort_data['cadence'].dropna()
            if not cadence_data.empty:
                interval_info.update({
                    'avg_cadence': cadence_data.mean(),
                    'max_cadence': cadence_data.max(),
                    'min_cadence': cadence_data.min()
                })
        
        if 'heart_rate' in effort_data.columns:
            hr_data = effort_data['heart_rate'].dropna()
            if not hr_data.empty:
                interval_info.update({
                    'avg_hr': hr_data.mean(),
                    'max_hr': hr_data.max(),
                    'hr_drift': hr_data.iloc[-1] - hr_data.iloc[0]
                })
        
        if 'enhanced_speed' in effort_data.columns:
            speed_data = effort_data['enhanced_speed'].dropna()
            if not speed_data.empty:
                interval_info.update({
                    'avg_speed': speed_data.mean() * 3.6,  # Convert to km/h
                    'max_speed': speed_data.max() * 3.6
                })
        
        # Add power zone information
        zone_boundaries = [0, ftp*0.55, ftp*0.75, ftp*0.90, ftp*1.05, ftp*1.20, ftp*1.50, 2000]
        zone_labels = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5', 'Zone 6', 'Zone 7']
        
        for i, (min_bound, max_bound) in enumerate(zip(zone_boundaries[:-1], zone_boundaries[1:])):
            if min_bound <= avg_power < max_bound:
                interval_info['training_zone'] = zone_labels[i]
                break
        
        intervals.append(interval_info)
        print(f"            ✅ Fallback interval: {duration:.0f}s at {avg_power:.0f}W ({avg_power/ftp:.1%} FTP) - Q:{quality_score:.1f}")
    
    print(f"         🎉 Fallback detection complete: {len(intervals)} intervals found")
    return intervals








def detect_adaptive_intervals_fallback(df, ftp, min_duration, power_threshold_pct):
    """Fallback adaptive detection with more sensitive parameters."""
    
    # Use more sensitive common durations for fallback
    fallback_durations = [10, 15, 20, 25, 30, 40, 45, 60, 90, 120, 180, 240, 300]
    bias_weight = 1.2  # Higher bias for fallback to catch more intervals
    
    # Smooth power data
    df_copy = df.copy()
    df_copy['power_smooth'] = df_copy['power'].rolling(window=3, min_periods=1).mean()  # Less smoothing for sensitivity
    
    # Initialize interval candidates
    interval_candidates = []
    
    # Loop over rolling window durations
    for duration in fallback_durations:
        window_samples = duration
        rolling_avg = df_copy['power_smooth'].rolling(window=window_samples, min_periods=window_samples).mean()
        rolling_std = df_copy['power_smooth'].rolling(window=window_samples, min_periods=window_samples).std()
        
        for i in range(window_samples, len(df_copy) - 10):
            avg_power = rolling_avg.iloc[i]
            std_power = rolling_std.iloc[i]
            post_window_power = df_copy['power_smooth'].iloc[i+1:i+11].mean()
            
            # Avoid NaNs
            if np.isnan(avg_power) or np.isnan(std_power) or np.isnan(post_window_power):
                continue
            
            # Calculate quality scores
            drop_score = max(0, (avg_power - post_window_power) / avg_power)
            std_score = std_power / avg_power
            fit_score = avg_power * (1 - std_score) * (1 - drop_score)
            
            # Boost score for common durations
            biased_score = fit_score * bias_weight
            
            # More sensitive power threshold for fallback
            if avg_power >= (ftp * power_threshold_pct):
                interval_candidates.append({
                    'start_idx': i - window_samples + 1,
                    'end_idx': i,
                    'duration': duration,
                    'avg_power': avg_power,
                    'fit_score': biased_score,
                    'raw_score': fit_score,
                    'bias_applied': True,
                    'std_score': std_score,
                    'drop_score': drop_score
                })
    
    # Convert to DataFrame and sort by fit score
    if interval_candidates:
        interval_df = pd.DataFrame(interval_candidates)
        interval_df = interval_df.sort_values(by='fit_score', ascending=False)
        
        # Non-overlapping selection (greedy)
        final_intervals = []
        used = np.zeros(len(df_copy), dtype=bool)
        
        for _, row in interval_df.iterrows():
            start_idx = int(row['start_idx'])
            end_idx = int(row['end_idx'])
            
            # Check if this interval overlaps with already selected ones
            if not used[start_idx:end_idx].any():
                final_intervals.append(row)
                used[start_idx:end_idx] = True
        
        # Convert to start/end times
        effort_starts = []
        effort_ends = []
        
        for interval in final_intervals:
            if interval['duration'] >= min_duration:
                start_time = df.index[int(interval['start_idx'])]
                end_time = df.index[int(interval['end_idx'])]
                effort_starts.append(start_time)
                effort_ends.append(end_time)
        
        return effort_starts, effort_ends
    else:
        return [], []


def identify_interval_sets(intervals, set_time_window):
    """Identify repeating interval sets based on timing and similarity."""
    if not intervals:
        return intervals
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x['start_time'])
    
    # Group intervals by similarity
    sets = []
    current_set = []
    
    for i, interval in enumerate(sorted_intervals):
        if not current_set:
            current_set = [interval]
            continue
        
        # Check if this interval belongs to current set
        last_interval = current_set[-1]
        
        # Time gap check
        time_gap = (interval['start_time'] - last_interval['end_time']).total_seconds()
        
        # Duration similarity check (±5s)
        duration_diff = abs(interval['duration_s'] - last_interval['duration_s'])
        
        # Power similarity check (±20W)
        power_diff = abs(interval['avg_power'] - last_interval['avg_power'])
        
        # Check if this interval belongs to current set
        if (time_gap <= set_time_window and 
            duration_diff <= 5 and 
            power_diff <= 20):
            current_set.append(interval)
        else:
            # End current set if it has 2+ intervals
            if len(current_set) >= 2:
                sets.append(current_set)
            current_set = [interval]
    
    # Don't forget the last set
    if len(current_set) >= 2:
        sets.append(current_set)
    
    # Tag intervals with set information
    for set_idx, interval_set in enumerate(sets):
        set_size = len(interval_set)
        for interval in interval_set:
            interval['set_info'] = f"{set_size}x"
            interval['set_number'] = set_idx + 1
    
    return intervals


def filter_overlapping_intervals(auto_intervals, lap_intervals, overlap_threshold):
    """Filter out auto-detected intervals that overlap with lap intervals."""
    if not auto_intervals or not lap_intervals:
        return auto_intervals
    
    filtered_intervals = []
    
    for auto_interval in auto_intervals:
        auto_start = auto_interval['start_time']
        auto_end = auto_interval['end_time']
        
        # Check if this auto interval overlaps with any lap interval
        overlaps = False
        for lap_interval in lap_intervals:
            lap_start = lap_interval['start_time']
            lap_end = lap_interval['end_time']
            
            # Calculate overlap
            overlap_start = max(auto_start, lap_start)
            overlap_end = min(auto_end, lap_end)
            
            if overlap_end > overlap_start:  # There is an overlap
                # Calculate overlap duration
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                
                # Calculate total duration of both intervals
                auto_duration = (auto_end - auto_start).total_seconds()
                lap_duration = (lap_end - lap_start).total_seconds()
                
                # Calculate overlap percentage
                overlap_percent = overlap_duration / min(auto_duration, lap_duration)
                
                # If overlap is significant (>30% of shorter interval), mark as overlapping
                if overlap_percent > 0.3:
                    overlaps = True
                    break
        
        # Only keep intervals that don't significantly overlap
        if not overlaps:
            filtered_intervals.append(auto_interval)
    
    return filtered_intervals


def score_intervals(intervals, ftp):
    """Compute quality score for each interval."""
    if not intervals:
        return intervals
    
    for interval in intervals:
        # Base score components (0-100 scale)
        power_score = min(100, (interval['avg_power'] / ftp) * 100)  # Power vs FTP
        
        duration_score = min(100, (interval['duration_s'] / 300) * 100)  # Duration (max at 5min)
        
        # Consistency score (lower CV = higher score)
        consistency_score = max(0, 100 - (interval['power_cv'] * 200))
        
        # Fade score (lower fade = higher score)
        fade_score = max(0, 100 - (interval['power_fade'] * 100))
        
        # Cadence control score (lower drift = higher score)
        cadence_score = max(0, 100 - abs(interval['cadence_drift']) * 10)
        
        # Torque stability score
        torque_score = interval['torque_stability'] * 100
        
        # Work efficiency score
        work_score = min(100, interval['work_per_min'] * 2)  # kJ/min
        
        # Calculate weighted quality score
        quality_score = (
            power_score * 0.25 +      # 25% weight
            duration_score * 0.20 +   # 20% weight
            consistency_score * 0.20 + # 20% weight
            fade_score * 0.15 +       # 15% weight
            cadence_score * 0.10 +    # 10% weight
            torque_score * 0.05 +     # 5% weight
            work_score * 0.05         # 5% weight
        )
        
        interval['quality_score'] = round(quality_score, 1)
        
        # Add training zone classification only if not already present
        if 'training_zone' not in interval:
            interval['training_zone'] = classify_training_zone(interval['intensity_factor'])
    
    return intervals


def classify_interval(duration, intensity_factor):
    """Classify interval based on duration and intensity."""
    if duration <= 10:
        return "Sprint"
    elif duration <= 30:
        return "Anaerobic"
    elif duration <= 120:
        return "VO2 Max"
    elif duration <= 300:
        return "Threshold"
    elif duration <= 600:
        return "Tempo"
    elif duration <= 1200:
        return "Endurance"
    else:
        return "Long Endurance"


def classify_training_zone(intensity_factor):
    """Classify training zone based on intensity factor."""
    if intensity_factor >= 1.5:
        return "Zone 7 (Neuromuscular)"
    elif intensity_factor >= 1.2:
        return "Zone 6 (Anaerobic)"
    elif intensity_factor >= 1.05:
        return "Zone 5 (VO2 Max)"
    elif intensity_factor >= 0.9:
        return "Zone 4 (Threshold)"
    elif intensity_factor >= 0.75:
        return "Zone 3 (Tempo)"
    elif intensity_factor >= 0.55:
        return "Zone 2 (Endurance)"
    else:
        return "Zone 1 (Recovery)"


def classify_power_zone(avg_power, ftp):
    """Classify power zone based on FTP."""
    if ftp <= 0:
        return 'N/A'
    
    intensity = avg_power / ftp
    if intensity < 0.55:
        return 'Zone 1 (Active Recovery)'
    elif intensity < 0.75:
        return 'Zone 2 (Endurance)'
    elif intensity < 0.90:
        return 'Zone 3 (Tempo)'
    elif intensity < 1.05:
        return 'Zone 4 (Lactate Threshold)'
    elif intensity < 1.20:
        return 'Zone 5 (VO2 Max)'
    elif intensity < 1.50:
        return 'Zone 6 (Anaerobic Capacity)'
    else:
        return 'Zone 7 (Neuromuscular Power)'


def classify_hr_zone(avg_hr, lthr):
    """Classify heart rate zone based on LTHR."""
    if np.isnan(avg_hr) or lthr <= 0:
        return 'N/A'
    
    intensity = avg_hr / lthr
    if intensity < 0.85:
        return 'Zone 1 (Active Recovery)'
    elif intensity < 0.95:
        return 'Zone 2 (Endurance)'
    elif intensity < 1.05:
        return 'Zone 3 (Tempo)'
    elif intensity < 1.15:
        return 'Zone 4 (Lactate Threshold)'
    else:
        return 'Zone 5 (VO2 Max)'


def calculate_sprint_efficiency(avg_power, max_power, power_fade, duration):
    """Calculate sprint efficiency score for short intervals."""
    if duration > 60 or max_power <= 0:
        return np.nan
    
    # Base score starts at 100
    efficiency_score = 100
    
    # Bonus for high average power relative to max (efficiency)
    power_efficiency = avg_power / max_power if max_power > 0 else 0
    if power_efficiency > 0.8:
        efficiency_score += 20
    elif power_efficiency > 0.7:
        efficiency_score += 10
    
    # Bonus for short duration (sprint-like efforts)
    if duration <= 30:
        efficiency_score += 10
    elif duration <= 45:
        efficiency_score += 5
    
    # Ensure score stays within reasonable bounds
    efficiency_score = max(0, min(150, efficiency_score))
    return round(efficiency_score, 1)


def calculate_mechanical_efficiency(work_kj, speed_gain_kph, duration):
    """Calculate mechanical efficiency proxy (kJ per km/h gained)."""
    if speed_gain_kph <= 0 or duration <= 0:
        return np.nan
    
    # Convert speed gain to m/s for calculation
    speed_gain_ms = speed_gain_kph / 3.6
    
    # Calculate distance covered (assuming constant acceleration)
    # distance = 0.5 * acceleration * time^2
    # acceleration = speed_gain / time
    # distance = 0.5 * speed_gain * time
    distance_km = 0.5 * speed_gain_ms * duration / 1000
    
    if distance_km <= 0:
        return np.nan
    
    # Mechanical efficiency = work / distance
    efficiency = work_kj / distance_km  # kJ/km
    
    return round(efficiency, 1)


def calculate_power_consistency_score(power_cv, power_fade):
    """Calculate power consistency score based on CV and fade."""
    # Base score starts at 100
    consistency_score = 100
    
    # Penalty for high CV (coefficient of variation)
    if not np.isnan(power_cv):
        cv_penalty = min(40, power_cv * 200)
        consistency_score -= cv_penalty
    
    # Penalty for power fade
    if not np.isnan(power_fade):
        fade_penalty = min(30, abs(power_fade) * 100)
        consistency_score -= fade_penalty
    
    # Ensure score stays within reasonable bounds
    consistency_score = max(0, min(100, consistency_score))
    return round(consistency_score, 1)


def calculate_overall_performance_score(quality_score, avg_power, ftp, duration):
    """Calculate overall performance score combining multiple factors."""
    if ftp <= 0:
        return quality_score
    
    # Base score from quality calculation
    base_score = quality_score
    
    # Bonus for high power relative to FTP
    power_bonus = min(20, (avg_power / ftp - 1) * 20) if avg_power > ftp else 0
    
    # Bonus for appropriate duration (not too short, not too long)
    duration_bonus = 0
    if 60 <= duration <= 600:  # 1-10 minutes
        duration_bonus = 10
    elif 30 <= duration <= 1200:  # 30 seconds to 20 minutes
        duration_bonus = 5
    
    # Calculate overall score
    overall_score = base_score + power_bonus + duration_bonus
    
    # Ensure score stays within reasonable bounds
    overall_score = max(0, min(150, overall_score))
    return round(overall_score, 1)


def get_power_zone_number(avg_power, ftp):
    """Get power zone number (1-7) based on FTP."""
    if ftp <= 0:
        return np.nan
    
    intensity = avg_power / ftp
    if intensity < 0.55:
        return 1
    elif intensity < 0.75:
        return 2
    elif intensity < 0.90:
        return 3
    elif intensity < 1.05:
        return 4
    elif intensity < 1.20:
        return 5
    elif intensity < 1.50:
        return 6
    else:
        return 7


def get_hr_zone_number(avg_hr, lthr):
    """Get heart rate zone number (1-5) based on LTHR."""
    if np.isnan(avg_hr) or lthr <= 0:
        return np.nan
    
    intensity = avg_hr / lthr
    if intensity < 0.85:
        return 1
    elif intensity < 0.95:
        return 2
    elif intensity < 1.05:
        return 3
    elif intensity < 1.15:
        return 4
    else:
        return 5


def analyze_interval_evolution(df, interval_df, n_best=5):
    """Analyze how metrics evolve during the best intervals.
    
    Args:
        df: DataFrame with cycling data
        interval_df: DataFrame with interval information
        n_best: Number of best intervals to analyze
    
    Returns:
        Dictionary with evolution analysis for each interval
    """
    if interval_df.empty:
        return {}
    
    # Take top n_best intervals
    top_intervals = interval_df.head(n_best)
    
    evolution_analysis = {}
    
    for _, interval in top_intervals.iterrows():
        start_time = interval['start_time']
        end_time = interval['end_time']
        duration = interval['duration_s']
        
        # Get interval data
        interval_data = df.loc[start_time:end_time].copy()
        
        # Create time segments for analysis
        n_segments = min(10, max(3, int(duration / 30)))  # 3-10 segments based on duration
        segment_duration = duration / n_segments
        
        segments = []
        for i in range(n_segments):
            seg_start = start_time + pd.Timedelta(seconds=i * segment_duration)
            seg_end = start_time + pd.Timedelta(seconds=(i + 1) * segment_duration)
            seg_data = df.loc[seg_start:seg_end]
            
            if not seg_data.empty:
                segment_info = {
                    'segment': i + 1,
                    'time_from_start': i * segment_duration,
                    'time_from_start_min': (i * segment_duration) / 60,
                    'avg_power': seg_data['power'].mean() if 'power' in seg_data.columns else np.nan,
                    'avg_cadence': seg_data['cadence'].mean() if 'cadence' in seg_data.columns else np.nan,
                    'avg_hr': seg_data['heart_rate'].mean() if 'heart_rate' in seg_data.columns else np.nan,
                    'avg_torque': seg_data['torque'].mean() if 'torque' in seg_data.columns else np.nan,
                    'avg_speed_kph': seg_data['enhanced_speed'].mean() * 3.6 if 'enhanced_speed' in seg_data.columns else np.nan
                }
                segments.append(segment_info)
        
        # Calculate evolution metrics
        if len(segments) >= 2:
            # Power evolution
            power_values = [s['avg_power'] for s in segments if not np.isnan(s['avg_power'])]
            if len(power_values) >= 2:
                power_start = power_values[0]
                power_end = power_values[-1]
                power_fade = ((power_end - power_start) / power_start * 100) if power_start > 0 else 0
                power_consistency = np.std(power_values) / np.mean(power_values) * 100 if np.mean(power_values) > 0 else 0
            else:
                power_fade = power_consistency = np.nan
            
            # Cadence evolution
            cadence_values = [s['avg_cadence'] for s in segments if not np.isnan(s['avg_cadence'])]
            if len(cadence_values) >= 2:
                cadence_start = cadence_values[0]
                cadence_end = cadence_values[-1]
                cadence_drift = cadence_end - cadence_start
                cadence_consistency = np.std(cadence_values) / np.mean(cadence_values) * 100 if np.mean(cadence_values) > 0 else np.nan
            else:
                cadence_drift = cadence_consistency = np.nan
            
            # HR evolution
            hr_values = [s['avg_hr'] for s in segments if not np.isnan(s['avg_hr'])]
            if len(hr_values) >= 2:
                hr_start = hr_values[0]
                hr_end = hr_values[-1]
                hr_drift = hr_end - hr_start
                hr_consistency = np.std(hr_values) / np.mean(hr_values) * 100 if np.mean(hr_values) > 0 else np.nan
            else:
                hr_drift = hr_consistency = np.nan
            
            # Torque evolution
            torque_values = [s['avg_torque'] for s in segments if not np.isnan(s['avg_torque'])]
            if len(torque_values) >= 2:
                torque_start = torque_values[0]
                torque_end = torque_values[-1]
                torque_drift = torque_end - torque_start
                torque_consistency = np.std(torque_values) / np.mean(torque_values) * 100 if np.mean(torque_values) > 0 else np.nan
            else:
                torque_drift = torque_consistency = np.nan
            
            # Speed evolution
            speed_values = [s['avg_speed_kph'] for s in segments if not np.isnan(s['avg_speed_kph'])]
            if len(speed_values) >= 2:
                speed_start = speed_values[0]
                speed_end = speed_values[-1]
                speed_gain = speed_end - speed_start
            else:
                speed_gain = np.nan
            
            evolution_metrics = {
                'segments': segments,
                'power_fade_%': power_fade,
                'power_consistency_%': power_consistency,
                'cadence_drift_rpm': cadence_drift,
                'cadence_consistency_%': cadence_consistency,
                'hr_drift_bpm': hr_drift,
                'hr_consistency_%': hr_consistency,
                'torque_drift_nm': torque_drift,
                'torque_consistency_%': torque_consistency,
                'speed_gain_kph': speed_gain
            }
            
            evolution_analysis[f"Interval_{interval['rank']}_{interval['start_str']}"] = evolution_metrics
    
    return evolution_analysis


def create_interval_plots(df, interval_df, evolution_analysis):
    """Create comprehensive plots for interval analysis."""
    if go is None:
        print("Plotly not installed; cannot show interactive interval plots.")
        return
    
    if interval_df.empty:
        print("No intervals to plot.")
        return
    
    print("\nCreating interval analysis plots...")
    
    # 1. Interval Power Comparison
    fig1 = go.Figure()
    
    # Add bars for average power
    fig1.add_trace(go.Bar(
        x=[f"{row['duration_min']:.1f}min\n{row['start_str']}" for _, row in interval_df.iterrows()],
        y=interval_df['avg_power'],
        name='Average Power',
        marker_color='#1f77b4',
        text=[f"{p:.0f}W" for p in interval_df['avg_power']],
        textposition='auto'
    ))
    
    # Add bars for max power
    fig1.add_trace(go.Bar(
        x=[f"{row['duration_min']:.1f}min\n{row['start_str']}" for _, row in interval_df.iterrows()],
        y=interval_df['max_power'],
        name='Max Power',
        marker_color='#ff7f0e',
        text=[f"{p:.0f}W" for p in interval_df['max_power']],
        textposition='auto'
    ))
    
    fig1.update_layout(
        title='Interval Power Comparison',
        xaxis_title='Interval Duration & Start Time',
        yaxis_title='Power (W)',
        template='plotly_white',
        height=500,
        barmode='group'
    )
    
    # Add FTP reference line
    fig1.add_hline(y=FTP, line_dash="dash", line_color="red", 
                   annotation_text=f"FTP ({FTP}W)")
    
    # fig1.write_html("interval_power_comparison.html")  # Removed file generation
    # print("Interval power comparison saved to interval_power_comparison.html")  # Plots shown in dashboard instead
    
    # 2. Interval Evolution Over Time (for best intervals)
    if evolution_analysis:
        for interval_name, evolution in evolution_analysis.items():
            segments = evolution['segments']
            
            if len(segments) < 2:
                continue
            
            # Create subplot with multiple metrics
            fig2 = go.Figure()
            
            # Power evolution
            if any(not np.isnan(s['avg_power']) for s in segments):
                power_values = [s['avg_power'] for s in segments]
                time_values = [s['time_from_start_min'] for s in segments]
                fig2.add_trace(go.Scatter(
                    x=time_values, y=power_values,
                    mode='lines+markers',
                    name='Power (W)',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
            
            # Cadence evolution
            if any(not np.isnan(s['avg_cadence']) for s in segments):
                cadence_values = [s['avg_cadence'] for s in segments]
                time_values = [s['time_from_start_min'] for s in segments]
                fig2.add_trace(go.Scatter(
                    x=time_values, y=cadence_values,
                    mode='lines+markers',
                    name='Cadence (rpm)',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=8),
                    yaxis='y2'
                ))
            
            # HR evolution
            if any(not np.isnan(s['avg_hr']) for s in segments):
                hr_values = [s['avg_hr'] for s in segments]
                time_values = [s['time_from_start_min'] for s in segments]
                fig2.add_trace(go.Scatter(
                    x=time_values, y=hr_values,
                    mode='lines+markers',
                    name='Heart Rate (bpm)',
                    line=dict(color='#d62728', width=3),
                    marker=dict(size=8),
                    yaxis='y3'
                ))
            
            # Torque evolution
            if any(not np.isnan(s['avg_torque']) for s in segments):
                torque_values = [s['avg_torque'] for s in segments]
                time_values = [s['time_from_start_min'] for s in segments]
                fig2.add_trace(go.Scatter(
                    x=time_values, y=torque_values,
                    mode='lines+markers',
                    name='Torque (Nm)',
                    line=dict(color='#9467bd', width=3),
                    marker=dict(size=8),
                    yaxis='y4'
                ))
            
            fig2.update_layout(
                title=f'{interval_name} - Metric Evolution Over Time',
                xaxis_title='Time from Start (minutes)',
                yaxis=dict(title='Power (W)', side='left'),
                yaxis2=dict(title='Cadence (rpm)', side='right', overlaying='y'),
                yaxis3=dict(title='Heart Rate (bpm)', side='right', overlaying='y', position=0.95),
                yaxis4=dict(title='Torque (Nm)', side='right', overlaying='y', position=0.9),
                template='plotly_white',
                height=500,
                showlegend=True
            )
            
            # Add evolution metrics as annotations
            annotations = []
            if 'power_fade_%' in evolution and not np.isnan(evolution['power_fade_%']):
                annotations.append(dict(
                    x=0.02, y=0.98, xref='paper', yref='paper',
                    text=f"Power Fade: {evolution['power_fade_%']:.1f}%",
                    showarrow=False, bgcolor='white', bordercolor='black'
                ))
            
            if 'cadence_drift_rpm' in evolution and not np.isnan(evolution['cadence_drift_rpm']):
                annotations.append(dict(
                    x=0.02, y=0.93, xref='paper', yref='paper',
                    text=f"Cadence Drift: {evolution['cadence_drift_rpm']:.1f} rpm",
                    showarrow=False, bgcolor='white', bordercolor='black'
                ))
            
            if 'hr_drift_bpm' in evolution and not np.isnan(evolution['hr_drift_bpm']):
                annotations.append(dict(
                    x=0.02, y=0.88, xref='paper', yref='paper',
                    text=f"HR Drift: {evolution['hr_drift_bpm']:.1f} bpm",
                    showarrow=False, bgcolor='white', bordercolor='black'
                ))
            
            fig2.update_layout(annotations=annotations)
            
            # filename = f"interval_evolution_{interval_name.replace(' ', '_').replace(':', '_')}.html"
            # fig2.write_html(filename)  # Removed file generation
            # print(f"Interval evolution plot saved to {filename}")  # Plots shown in dashboard instead
    
    # 3. Interval Duration vs Power Relationship
    fig3 = go.Figure()
    
    # Group by duration ranges
    duration_ranges = [(1, 2), (2, 5), (5, 10), (10, 20)]  # minutes
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    
    for i, (min_dur, max_dur) in enumerate(duration_ranges):
        mask = (interval_df['duration_min'] >= min_dur) & (interval_df['duration_min'] < max_dur)
        subset = interval_df[mask]
        
        if not subset.empty:
            fig3.add_trace(go.Scatter(
                x=subset['duration_min'],
                y=subset['avg_power'],
                mode='markers',
                name=f'{min_dur}-{max_dur} min',
                marker=dict(size=10, color=colors[i]),
                text=[f"{row['start_str']}<br>{row['avg_power']:.0f}W" for _, row in subset.iterrows()],
                hovertemplate='%{text}<extra></extra>'
            ))
    
    # Add trend line
    if len(interval_df) > 1:
        z = np.polyfit(interval_df['duration_min'], interval_df['avg_power'], 1)
        p = np.poly1d(z)
        fig3.add_trace(go.Scatter(
            x=interval_df['duration_min'],
            y=p(interval_df['duration_min']),
            mode='lines',
            name='Trend Line',
            line=dict(color='black', dash='dash'),
            showlegend=True
        ))
    
    fig3.update_layout(
        title='Interval Duration vs Average Power',
        xaxis_title='Duration (minutes)',
        yaxis_title='Average Power (W)',
        template='plotly_white',
        height=500
    )
    
    # Add FTP reference line
    fig3.add_hline(y=FTP, line_dash="dash", line_color="red", 
                   annotation_text=f"FTP ({FTP}W)")
    
    # fig3.write_html("interval_duration_vs_power.html")  # Removed file generation
    # print("Interval duration vs power plot saved to interval_duration_vs_power.html")  # Plots shown in dashboard instead
    
    # 4. Metric Consistency Analysis
    if evolution_analysis:
        fig4 = go.Figure()
        
        consistency_data = []
        interval_names = []
        
        for interval_name, evolution in evolution_analysis.items():
            if 'power_consistency_%' in evolution and not np.isnan(evolution['power_consistency_%']):
                consistency_data.append(evolution['power_consistency_%'])
                interval_names.append(interval_name.split('_', 2)[-1])  # Extract time part
        
        if consistency_data:
            fig4.add_trace(go.Bar(
                x=interval_names,
                y=consistency_data,
                name='Power Consistency (%)',
                marker_color='#1f77b4',
                text=[f"{c:.1f}%" for c in consistency_data],
                textposition='auto'
            ))
            
            fig4.update_layout(
                title='Interval Power Consistency (Lower = More Consistent)',
                xaxis_title='Interval',
                yaxis_title='Power Consistency (%)',
                template='plotly_white',
                height=400
            )
            
            # fig4.write_html("interval_power_consistency.html")  # Removed file generation
            # print("Interval power consistency plot saved to interval_power_consistency.html")  # Plots shown in dashboard instead
    
    print("All interval plots created!")


def display_interval_analysis(interval_df, evolution_analysis):
    """Display comprehensive interval analysis in organized tables."""
    if interval_df.empty:
        print("\nNo intervals found for analysis.")
        return
    
    print("\n" + "="*80)
    print("LONGER INTERVAL EFFORT ANALYSIS")
    print("="*80)
    
    # 1. Interval Summary Table
    print("\nINTERVAL PERFORMANCE SUMMARY")
    print("-" * 60)
    
    # Select key columns for display
    display_cols = ['rank', 'start_str', 'duration_min', 'avg_power', 'max_power', 
                   'work_kj', 'intensity_factor']
    
    # Add available metrics
    if 'avg_cadence' in interval_df.columns:
        display_cols.extend(['avg_cadence', 'max_cadence', 'min_cadence'])
    if 'avg_hr' in interval_df.columns:
        display_cols.extend(['avg_hr', 'max_hr'])
    if 'avg_torque' in interval_df.columns:
        display_cols.extend(['avg_torque', 'max_torque'])
    if 'avg_speed_kph' in interval_df.columns:
        display_cols.extend(['avg_speed_kph', 'max_speed_kph', 'speed_gain_kph'])
    if 'power_cv' in interval_df.columns:
        display_cols.extend(['power_cv', 'power_fade'])
    if 'altitude_gain' in interval_df.columns:
        display_cols.extend(['altitude_gain'])
    if 'avg_grade' in interval_df.columns:
        display_cols.extend(['avg_grade'])
    if 'normalized_power' in interval_df.columns:
        display_cols.extend(['normalized_power'])
    if 'power_per_kg' in interval_df.columns:
        display_cols.extend(['power_per_kg', 'np_per_kg'])
    if 'power_zone' in interval_df.columns:
        display_cols.extend(['power_zone'])
    if 'hr_zone' in interval_df.columns:
        display_cols.extend(['hr_zone'])
    if 'sprint_efficiency' in interval_df.columns:
        display_cols.extend(['sprint_efficiency'])
    if 'mechanical_efficiency' in interval_df.columns:
        display_cols.extend(['mechanical_efficiency'])
    if 'power_consistency_score' in interval_df.columns:
        display_cols.extend(['power_consistency_score'])
    if 'overall_performance_score' in interval_df.columns:
        display_cols.extend(['overall_performance_score'])
    if 'power_to_weight_ratio' in interval_df.columns:
        display_cols.extend(['power_to_weight_ratio', 'np_to_weight_ratio'])
    
    # Create display DataFrame
    display_df = interval_df[display_cols].copy()
    
    # Rename columns for better display
    column_mapping = {
        'rank': 'Rank',
        'start_str': 'Start Time',
        'duration_min': 'Duration (min)',
        'avg_power': 'Avg Power (W)',
        'max_power': 'Max Power (W)',
        'work_kj': 'Work (kJ)',
        'intensity_factor': 'Intensity Factor',
        'avg_cadence': 'Avg Cadence (rpm)',
        'max_cadence': 'Max Cadence (rpm)',
        'min_cadence': 'Min Cadence (rpm)',
        'avg_hr': 'Avg HR (bpm)',
        'max_hr': 'Max HR (bpm)',
        'avg_torque': 'Avg Torque (Nm)',
        'max_torque': 'Max Torque (Nm)',
        'avg_speed_kph': 'Avg Speed (km/h)',
        'max_speed_kph': 'Max Speed (km/h)',
        'speed_gain_kph': 'Speed Gain (km/h)',
        'power_cv': 'Power CV (%)',
        'power_fade': 'Power Fade (%)',
        'altitude_gain': 'Altitude Gain (m)',
        'avg_grade': 'Avg Grade (%)',
        'normalized_power': 'Normalized Power (W)',
        'power_per_kg': 'Power (W/kg)',
        'np_per_kg': 'NP (W/kg)',
        'power_zone': 'Power Zone',
        'hr_zone': 'HR Zone',
        'sprint_efficiency': 'Sprint Efficiency',
        'mechanical_efficiency': 'Mech. Efficiency (kJ/km)',
        'power_consistency_score': 'Power Consistency',
        'overall_performance_score': 'Overall Score',
        'power_to_weight_ratio': 'Power (W/kg)',
        'np_to_weight_ratio': 'NP (W/kg)'
    }
    
    display_df.columns = [column_mapping.get(col, col) for col in display_df.columns]
    
    # Format numeric columns
    for col in display_df.columns:
        if 'Power' in col or 'Work' in col or 'Torque' in col:
            display_df[col] = display_df[col].round(1)
        elif 'Cadence' in col or 'HR' in col or 'Speed' in col:
            display_df[col] = display_df[col].round(1)
        elif 'Duration' in col:
            display_df[col] = display_df[col].round(1)
        elif 'Intensity' in col:
            display_df[col] = display_df[col].round(2)
        elif 'CV' in col or 'Fade' in col:
            display_df[col] = display_df[col].round(2)
        elif 'Altitude' in col:
            display_df[col] = display_df[col].round(0)
        elif 'Grade' in col:
            display_df[col] = display_df[col].round(1)
        elif 'Normalized Power' in col:
            display_df[col] = display_df[col].round(0)
        elif 'W/kg' in col:
            display_df[col] = display_df[col].round(2)
        elif 'Sprint Efficiency' in col:
            display_df[col] = display_df[col].round(1)
        elif 'Mech. Efficiency' in col:
            display_df[col] = display_df[col].round(1)
        elif 'Power Consistency' in col or 'Overall Score' in col:
            display_df[col] = display_df[col].round(1)
        elif 'W/kg' in col:
            display_df[col] = display_df[col].round(2)
    
    print(display_df.to_string(index=False))
    
    # 2. Evolution Analysis
    if evolution_analysis:
        print("\nINTERVAL EVOLUTION ANALYSIS")
        print("-" * 60)
        
        evolution_summary = []
        
        for interval_name, evolution in evolution_analysis.items():
            # Extract interval info
            parts = interval_name.split('_')
            rank = parts[1] if len(parts) > 1 else 'N/A'
            start_time = parts[2] if len(parts) > 2 else 'N/A'
            
            evolution_info = {
                'Interval': f"#{rank} ({start_time})",
                'Power Fade (%)': f"{evolution.get('power_fade_%', 'N/A'):.1f}" if not np.isnan(evolution.get('power_fade_%', np.nan)) else 'N/A',
                'Power Consistency (%)': f"{evolution.get('power_consistency_%', 'N/A'):.1f}" if not np.isnan(evolution.get('power_consistency_%', np.nan)) else 'N/A',
                'Cadence Drift (rpm)': f"{evolution.get('cadence_drift_rpm', 'N/A'):.1f}" if not np.isnan(evolution.get('cadence_drift_rpm', np.nan)) else 'N/A',
                'HR Drift (bpm)': f"{evolution.get('hr_drift_bpm', 'N/A'):.1f}" if not np.isnan(evolution.get('hr_drift_bpm', np.nan)) else 'N/A',
                'Torque Drift (Nm)': f"{evolution.get('torque_drift_nm', 'N/A'):.1f}" if not np.isnan(evolution.get('torque_drift_nm', np.nan)) else 'N/A',
                'Speed Gain (km/h)': f"{evolution.get('speed_gain_kph', 'N/A'):.1f}" if not np.isnan(evolution.get('speed_gain_kph', np.nan)) else 'N/A'
            }
            
            evolution_summary.append(evolution_info)
        
        if evolution_summary:
            evolution_df = pd.DataFrame(evolution_summary)
            print(evolution_df.to_string(index=False))
    
    # 3. Performance Insights
    print("\nPERFORMANCE INSIGHTS")
    print("-" * 60)
    
    # Best interval by different metrics
    if not interval_df.empty:
        best_by_power = interval_df.iloc[0]  # Already sorted by power
        
        print(f"Best Interval by Average Power:")
        print(f"   • Duration: {best_by_power['duration_min']:.1f} minutes")
        print(f"   • Average Power: {best_by_power['avg_power']:.0f}W ({best_by_power['avg_power']/rider_mass_kg:.1f} W/kg)")
        print(f"   • Max Power: {best_by_power['max_power']:.0f}W")
        print(f"   • Work: {best_by_power['work_kj']:.1f} kJ")
        print(f"   • Intensity Factor: {best_by_power['intensity_factor']:.2f}")
        
        if 'avg_cadence' in best_by_power and not np.isnan(best_by_power['avg_cadence']):
            print(f"   • Average Cadence: {best_by_power['avg_cadence']:.1f} rpm")
        
        if 'avg_hr' in best_by_power and not np.isnan(best_by_power['avg_hr']):
            print(f"   • Average HR: {best_by_power['avg_hr']:.0f} bpm")
        
        # Find best interval by duration
        if len(interval_df) > 1:
            # Group by duration ranges and find best in each
            duration_ranges = [(1, 2), (2, 5), (5, 10), (10, 20)]
            
            for min_dur, max_dur in duration_ranges:
                mask = (interval_df['duration_min'] >= min_dur) & (interval_df['duration_min'] < max_dur)
                subset = interval_df[mask]
                
                if not subset.empty:
                    best_in_range = subset.iloc[0]
                    print(f"\nBest {min_dur}-{max_dur} min Interval:")
                    print(f"   • Duration: {best_in_range['duration_min']:.1f} minutes")
                    print(f"   • Average Power: {best_in_range['avg_power']:.0f}W")
                    print(f"   • Start Time: {best_in_range['start_str']}")
    
    print("\n" + "="*80)
    print("Interval Analysis Complete!")
    print("="*80)


def analyze_interval_patterns(df, interval_df):
    """Analyze patterns in interval performance and timing."""
    if interval_df.empty:
        return
    
    print("\nINTERVAL PATTERN ANALYSIS")
    print("-" * 60)
    
    # 1. Timing patterns
    if 'start_time' in interval_df.columns:
        # Convert to time of day
        interval_df['hour'] = interval_df['start_time'].dt.hour
        interval_df['minute'] = interval_df['start_time'].dt.minute
        
        # Group by hour
        hourly_performance = interval_df.groupby('hour').agg({
            'avg_power': 'mean',
            'duration_min': 'mean',
            'intensity_factor': 'mean'
        }).round(2)
        
        print("Hourly Performance Patterns:")
        print(hourly_performance.to_string())
        
        # Find best performing hour
        best_hour = hourly_performance['avg_power'].idxmax()
        print(f"\nBest performing hour: {best_hour:02d}:00 ({hourly_performance.loc[best_hour, 'avg_power']:.0f}W avg)")
    
    # 2. Duration vs Performance relationship
    if len(interval_df) > 1:
        print(f"\nDuration vs Performance Analysis:")
        
        # Calculate correlation
        duration_power_corr = interval_df['duration_min'].corr(interval_df['avg_power'])
        print(f"   • Duration-Power Correlation: {duration_power_corr:.3f}")
        
        if duration_power_corr < -0.3:
            print("   • Strong negative correlation: Longer intervals tend to have lower power")
        elif duration_power_corr > 0.3:
            print("   • Strong positive correlation: Longer intervals tend to have higher power")
        else:
            print("   • Weak correlation: No clear relationship between duration and power")
        
        # Power decay analysis
        if len(interval_df) >= 3:
            # Sort by duration and calculate power decay
            sorted_by_duration = interval_df.sort_values('duration_min')
            power_decay = []
            
            for i in range(1, len(sorted_by_duration)):
                prev_power = sorted_by_duration.iloc[i-1]['avg_power']
                curr_power = sorted_by_duration.iloc[i]['avg_power']
                duration_diff = sorted_by_duration.iloc[i]['duration_min'] - sorted_by_duration.iloc[i-1]['duration_min']
                
                if duration_diff > 0:
                    decay_rate = (curr_power - prev_power) / duration_diff
                    power_decay.append(decay_rate)
            
            if power_decay:
                avg_decay = np.mean(power_decay)
                print(f"   • Average Power Decay Rate: {avg_decay:.1f} W/min")
                
                if avg_decay > -10:
                    print("   • Good power maintenance across durations")
                elif avg_decay > -20:
                    print("   • Moderate power decay with duration")
                else:
                    print("   • Significant power decay with duration")
    
    # 3. Recovery analysis (if multiple intervals)
    if len(interval_df) > 1:
        print(f"\nRecovery Analysis:")
        
        # Sort by start time
        sorted_intervals = interval_df.sort_values('start_time')
        recovery_times = []
        
        for i in range(1, len(sorted_intervals)):
            prev_end = sorted_intervals.iloc[i-1]['end_time']
            curr_start = sorted_intervals.iloc[i]['start_time']
            recovery = (curr_start - prev_end).total_seconds() / 60  # minutes
            
            if recovery > 0:  # Only if there's actual recovery time
                recovery_times.append(recovery)
                
                prev_power = sorted_intervals.iloc[i-1]['avg_power']
                curr_power = sorted_intervals.iloc[i]['avg_power']
                power_ratio = curr_power / prev_power if prev_power > 0 else 0
                
                print(f"   • Recovery {i}: {recovery:.1f} min, Power ratio: {power_ratio:.2f}")
        
        if recovery_times:
            avg_recovery = np.mean(recovery_times)
            print(f"   • Average recovery time: {avg_recovery:.1f} minutes")
            
            # Assess recovery adequacy
            if avg_recovery < 2:
                print("   • Warning: Short recovery periods - may impact performance")
            elif avg_recovery < 5:
                print("   • Warning: Moderate recovery periods - monitor fatigue")
            else:
                print("   • Adequate recovery periods")


def create_comprehensive_summary(df, best10, micro_df, sprint_summary_df, long_intervals, evolution_analysis):
    """Create a comprehensive summary of both sprint and interval analysis."""
    print("\n" + "="*80)
    print("COMPREHENSIVE CYCLING ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall ride statistics
    total_time = (df.index[-1] - df.index[0]).total_seconds() / 3600  # hours
    total_distance = 0
    if 'enhanced_speed' in df.columns:
        total_distance = df['enhanced_speed'].sum() / 1000  # km
    
    print(f"\nOVERALL RIDE STATISTICS")
    print(f"   • Total Time: {total_time:.2f} hours")
    if total_distance > 0:
        print(f"   • Total Distance: {total_distance:.1f} km")
    print(f"   • Average Power: {df['power'].mean():.0f}W ({df['power'].mean()/rider_mass_kg:.1f} W/kg)")
    print(f"   • Max Power: {df['power'].max():.0f}W ({df['power'].max()/rider_mass_kg:.1f} W/kg)")
    
    # Sprint summary
    if not best10.empty:
        print(f"\nSPRINT PERFORMANCE SUMMARY")
        print(f"   • Best 10s Power: {best10.iloc[0]['Power (W)']:.0f}W ({best10.iloc[0]['Power (W)']/rider_mass_kg:.1f} W/kg)")
        print(f"   • Number of Sprints Analyzed: {len(best10)}")
        
        if len(best10) > 1:
            power_std = best10['Power (W)'].std()
            power_cv = (power_std / best10['Power (W)'].mean()) * 100
            print(f"   • Sprint Consistency (CV): {power_cv:.1f}%")
            
            if power_cv < 5:
                print("   • Status: Excellent consistency")
            elif power_cv < 10:
                print("   • Status: Good consistency")
            else:
                print("   • Status: Variable performance")
    
    # Interval summary
    if not long_intervals.empty:
        print(f"\nINTERVAL PERFORMANCE SUMMARY")
        print(f"   • Number of Intervals: {len(long_intervals)}")
        print(f"   • Best Interval Power: {long_intervals.iloc[0]['avg_power']:.0f}W ({long_intervals.iloc[0]['avg_power']/rider_mass_kg:.1f} W/kg)")
        print(f"   • Duration Range: {long_intervals['duration_min'].min():.1f} - {long_intervals['duration_min'].max():.1f} minutes")
        
        # Find best intervals by duration category
        duration_categories = {
            'Short (1-2 min)': (1, 2),
            'Medium (2-5 min)': (2, 5),
            'Long (5-10 min)': (5, 10),
            'Extended (10+ min)': (10, 100)
        }
        
        for category, (min_dur, max_dur) in duration_categories.items():
            mask = (long_intervals['duration_min'] >= min_dur) & (long_intervals['duration_min'] < max_dur)
            subset = long_intervals[mask]
            
            if not subset.empty:
                best_in_category = subset.iloc[0]
                print(f"   • {category}: {best_in_category['avg_power']:.0f}W ({best_in_category['duration_min']:.1f} min)")
    
    # Performance insights
    print(f"\nPERFORMANCE INSIGHTS")
    
    # Power curve analysis
    if not best10.empty and not long_intervals.empty:
        sprint_power = best10.iloc[0]['Power (W)']
        interval_power = long_intervals.iloc[0]['avg_power']
        
        if sprint_power > interval_power * 1.5:
            print(f"   • Strong sprint performance relative to intervals")
        elif sprint_power < interval_power * 1.2:
            print(f"   • Sprint power could be improved relative to interval performance")
        else:
            print(f"   • Balanced sprint and interval performance")
    
    # Cadence analysis
    if 'cadence' in df.columns:
        avg_cadence = df['cadence'].mean()
        if avg_cadence < 80:
            print(f"   • Low average cadence ({avg_cadence:.0f} rpm) - consider higher cadence training")
        elif avg_cadence > 100:
            print(f"   • High average cadence ({avg_cadence:.0f} rpm) - good for endurance")
        else:
            print(f"   • Optimal cadence range ({avg_cadence:.0f} rpm)")
    
    # Heart rate analysis
    if 'heart_rate' in df.columns:
        avg_hr = df['heart_rate'].mean()
        hr_zone = "Zone 1-2" if avg_hr < LTHR * 0.95 else "Zone 3+" if avg_hr < LTHR * 1.05 else "Zone 4+"
        print(f"   • Average HR: {avg_hr:.0f} bpm ({hr_zone})")
    
    # Training recommendations
    print(f"\nTRAINING RECOMMENDATIONS")
    
    if not best10.empty and not long_intervals.empty:
        # Analyze gaps in performance
        sprint_power_kg = best10.iloc[0]['Power (W)'] / rider_mass_kg
        interval_power_kg = long_intervals.iloc[0]['avg_power'] / rider_mass_kg
        
        if sprint_power_kg > 12:  # Very strong sprint
            print(f"   • Excellent sprint power - focus on endurance and threshold work")
        elif sprint_power_kg < 8:  # Weak sprint
            print(f"   • Sprint power needs improvement - add neuromuscular and anaerobic training")
        
        if interval_power_kg > 4:  # Strong intervals
            print(f"   • Strong interval performance - maintain with regular threshold work")
        elif interval_power_kg < 3:  # Weak intervals
            print(f"   • Interval power needs work - focus on sweet spot and threshold training")
    
    # Recovery analysis
    if not long_intervals.empty and len(long_intervals) > 1:
        sorted_intervals = long_intervals.sort_values('start_time')
        recovery_times = []
        
        for i in range(1, len(sorted_intervals)):
            prev_end = sorted_intervals.iloc[i-1]['end_time']
            curr_start = sorted_intervals.iloc[i]['start_time']
            recovery = (curr_start - prev_end).total_seconds() / 60
            if recovery > 0:
                recovery_times.append(recovery)
        
        if recovery_times:
            avg_recovery = np.mean(recovery_times)
            if avg_recovery < 2:
                print(f"   • Warning: Recovery periods may be too short - consider longer rest")
            elif avg_recovery > 10:
                print(f"   • Recovery periods are adequate - good training structure")
    
    print(f"\n" + "="*80)
    print("Summary Complete!")
    print("="*80)


def analyze_metric_relationships(df, interval_df, evolution_analysis):
    """Analyze relationships between different metrics during intervals."""
    if interval_df.empty or not evolution_analysis:
        return
    
    print("\nMETRIC RELATIONSHIP ANALYSIS")
    print("-" * 60)
    
    # 1. Power-Cadence relationship
    if 'avg_cadence' in interval_df.columns and 'avg_power' in interval_df.columns:
        power_cadence_corr = interval_df['avg_power'].corr(interval_df['avg_cadence'])
        print(f"Power-Cadence Correlation: {power_cadence_corr:.3f}")
        
        if power_cadence_corr > 0.3:
            print("   • Strong positive correlation: Higher cadence associated with higher power")
        elif power_cadence_corr < -0.3:
            print("   • Strong negative correlation: Lower cadence associated with higher power")
        else:
            print("   • Weak correlation: No clear relationship between power and cadence")
    
    # 2. Power-HR relationship
    if 'avg_hr' in interval_df.columns and 'avg_power' in interval_df.columns:
        power_hr_corr = interval_df['avg_power'].corr(interval_df['avg_hr'])
        print(f"Power-HR Correlation: {power_hr_corr:.3f}")
        
        if power_hr_corr > 0.7:
            print("   • Strong HR response to power - good cardiovascular fitness")
        elif power_hr_corr < 0.5:
            print("   • Weak HR response to power - may indicate fatigue or fitness issues")
    
    # 3. Duration vs Metric relationships
    print(f"\nDuration Impact on Metrics:")
    
    # Group intervals by duration and analyze
    duration_ranges = [(1, 2), (2, 5), (5, 10), (10, 20)]
    
    for min_dur, max_dur in duration_ranges:
        mask = (interval_df['duration_min'] >= min_dur) & (interval_df['duration_min'] < max_dur)
        subset = interval_df[mask]
        
        if not subset.empty:
            print(f"\n{min_dur}-{max_dur} min intervals ({len(subset)} found):")
            
            if 'avg_power' in subset.columns:
                avg_power = subset['avg_power'].mean()
                print(f"   • Average Power: {avg_power:.0f}W")
            
            if 'avg_cadence' in subset.columns:
                avg_cadence = subset['avg_cadence'].mean()
                print(f"   • Average Cadence: {avg_cadence:.1f} rpm")
            
            if 'avg_hr' in subset.columns:
                avg_hr = subset['avg_hr'].mean()
                print(f"   • Average HR: {avg_hr:.0f} bpm")
            
            if 'avg_torque' in subset.columns:
                avg_torque = subset['avg_torque'].mean()
                print(f"   • Average Torque: {avg_torque:.1f} Nm")
    
    # 4. Evolution pattern analysis
    print(f"\nEvolution Pattern Analysis:")
    
    # Analyze power fade patterns
    power_fades = []
    for interval_name, evolution in evolution_analysis.items():
        if 'power_fade_%' in evolution and not np.isnan(evolution['power_fade_%']):
            power_fades.append(evolution['power_fade_%'])
    
    if power_fades:
        avg_fade = np.mean(power_fades)
        fade_std = np.std(power_fades)
        
        print(f"   • Average Power Fade: {avg_fade:.1f}% ± {fade_std:.1f}%")
        
        if avg_fade > -5:
            print("   • Excellent power maintenance across intervals")
        elif avg_fade > -15:
            print("   • Good power maintenance with moderate fade")
        else:
            print("   • Significant power fade - consider pacing strategy")
    
    # Analyze cadence drift patterns
    cadence_drifts = []
    for interval_name, evolution in evolution_analysis.items():
        if 'cadence_drift_rpm' in evolution and not np.isnan(evolution['cadence_drift_rpm']):
            cadence_drifts.append(evolution['cadence_drift_rpm'])
    
    if cadence_drifts:
        avg_drift = np.mean(cadence_drifts)
        print(f"   • Average Cadence Drift: {avg_drift:.1f} rpm")
        
        if abs(avg_drift) < 5:
            print("   • Stable cadence across intervals")
        elif avg_drift > 0:
            print("   • Cadence tends to increase during intervals")
        else:
            print("   • Cadence tends to decrease during intervals")
    
    # 5. Performance consistency analysis
    print(f"\nPerformance Consistency Analysis:")
    
    if len(interval_df) > 1:
        # Power consistency across intervals
        power_cv = (interval_df['avg_power'].std() / interval_df['avg_power'].mean()) * 100
        print(f"   • Power Consistency (CV): {power_cv:.1f}%")
        
        if power_cv < 5:
            print("   • Excellent interval consistency")
        elif power_cv < 10:
            print("   • Good interval consistency")
        else:
            print("   • Variable interval performance")
        
        # Duration consistency
        duration_cv = (interval_df['duration_min'].std() / interval_df['duration_min'].mean()) * 100
        print(f"   • Duration Consistency (CV): {duration_cv:.1f}%")
    
    # 6. Training load analysis
    if 'work_kj' in interval_df.columns:
        total_work = interval_df['work_kj'].sum()
        avg_work_per_interval = interval_df['work_kj'].mean()
        
        print(f"\nTraining Load Analysis:")
        print(f"   • Total Interval Work: {total_work:.1f} kJ")
        print(f"   • Average Work per Interval: {avg_work_per_interval:.1f} kJ")
        
        # Compare to overall ride work
        if 'power' in df.columns:
            total_ride_work = df['power'].sum() / 1000
            interval_work_ratio = (total_work / total_ride_work) * 100 if total_ride_work > 0 else 0
            print(f"   • Intervals as % of Total Work: {interval_work_ratio:.1f}%")
            
            if interval_work_ratio > 50:
                print("   • High-intensity focused workout")
            elif interval_work_ratio > 20:
                print("   • Balanced workout with good interval component")
            else:
                print("   • Endurance-focused workout with minimal intervals")


def create_workout_overview_graph(df):
    """Create a comprehensive workout overview graph showing all key variables over time."""
    if go is None:
        print("Plotly not installed; cannot show interactive workout overview.")
        return
    
    print("\nCreating comprehensive workout overview graph...")
    
    # Create subplots for different metric groups
    fig = go.Figure()
    
    # Convert time to minutes for x-axis
    time_min = (df.index - df.index[0]).total_seconds() / 60
    
    # 1. Power (primary metric)
    if 'power' in df.columns:
        fig.add_trace(go.Scatter(
            x=time_min, y=df['power'],
            mode='lines',
            name='Power (W)',
            line=dict(color='#1f77b4', width=2),
            yaxis='y'
        ))
        
        # Add FTP reference line
        fig.add_hline(y=FTP, line_dash="dash", line_color="red", 
                     annotation_text=f"FTP ({FTP}W)", line_width=1)
    
    # 2. Heart Rate (secondary y-axis)
    if 'heart_rate' in df.columns:
        fig.add_trace(go.Scatter(
            x=time_min, y=df['heart_rate'],
            mode='lines',
            name='Heart Rate (bpm)',
            line=dict(color='#d62728', width=2),
            yaxis='y2'
        ))
        
        # Add LTHR reference line
        fig.add_shape(
            type="line",
            x0=time_min.min(), x1=time_min.max(),
            y0=LTHR, y1=LTHR,
            line=dict(dash="dash", color="orange", width=1),
            yref="y2"
        )
        # Add LTHR annotation
        fig.add_annotation(
            x=time_min.max() * 0.95,
            y=LTHR,
            text=f"LTHR ({LTHR} bpm)",
            showarrow=False,
            yref="y2",
            font=dict(color="orange", size=10),
            bgcolor="white",
            bordercolor="orange"
        )
    
    # 3. Cadence (third y-axis)
    if 'cadence' in df.columns:
        fig.add_trace(go.Scatter(
            x=time_min, y=df['cadence'],
            mode='lines',
            name='Cadence (rpm)',
            line=dict(color='#2ca02c', width=2),
            yaxis='y3'
        ))
    
    # 4. Speed (fourth y-axis)
    if 'enhanced_speed' in df.columns:
        speed_kph = df['enhanced_speed'] * 3.6
        fig.add_trace(go.Scatter(
            x=time_min, y=speed_kph,
            mode='lines',
            name='Speed (km/h)',
            line=dict(color='#9467bd', width=2),
            yaxis='y4'
        ))
    
    # 5. Torque (fifth y-axis)
    if 'torque' in df.columns:
        fig.add_trace(go.Scatter(
            x=time_min, y=df['torque'],
            mode='lines',
            name='Torque (Nm)',
            line=dict(color='#ff7f0e', width=2),
            yaxis='y5'
        ))
    
    # 6. Altitude (sixth y-axis)
    if 'altitude' in df.columns:
        # Normalize altitude to fit on graph
        alt_min = df['altitude'].min()
        alt_max = df['altitude'].max()
        if alt_max > alt_min:
            alt_normalized = (df['altitude'] - alt_min) / (alt_max - alt_min) * 100
            fig.add_trace(go.Scatter(
                x=time_min, y=alt_normalized,
                mode='lines',
                name='Altitude (normalized %)',
                line=dict(color='#8c564b', width=1.5),
                yaxis='y6'
            ))
    
    # 7. Grade (seventh y-axis)
    if 'grade' in df.columns:
        fig.add_trace(go.Scatter(
            x=time_min, y=df['grade'],
            mode='lines',
            name='Grade (%)',
            line=dict(color='#e377c2', width=1.5),
            yaxis='y7'
        ))
    
    # Update layout with multiple y-axes
    fig.update_layout(
        title='Workout Overview - All Metrics Over Time',
        xaxis=dict(
            title='Time (minutes)',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title=dict(text='Power (W)', font=dict(color='#1f77b4')),
            tickfont=dict(color='#1f77b4'),
            side='left',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis2=dict(
            title=dict(text='Heart Rate (bpm)', font=dict(color='#d62728')),
            tickfont=dict(color='#d62728'),
            side='right',
            overlaying='y',
            position=0.95,
            showgrid=False
        ),
        yaxis3=dict(
            title=dict(text='Cadence (rpm)', font=dict(color='#2ca02c')),
            tickfont=dict(color='#2ca02c'),
            side='right',
            overlaying='y',
            position=0.90,
            showgrid=False
        ),
        yaxis4=dict(
            title=dict(text='Speed (km/h)', font=dict(color='#9467bd')),
            tickfont=dict(color='#9467bd'),
            side='right',
            overlaying='y',
            position=0.85,
            showgrid=False
        ),
        yaxis5=dict(
            title=dict(text='Torque (Nm)', font=dict(color='#ff7f0e')),
            tickfont=dict(color='#ff7f0e'),
            side='right',
            overlaying='y',
            position=0.80,
            showgrid=False
        ),
        yaxis6=dict(
            title=dict(text='Altitude (normalized %)', font=dict(color='#8c564b')),
            tickfont=dict(color='#8c564b'),
            side='right',
            overlaying='y',
            position=0.75,
            showgrid=False
        ),
        yaxis7=dict(
            title=dict(text='Grade (%)', font=dict(color='#e377c2')),
            tickfont=dict(color='#e377c2'),
            side='right',
            overlaying='y',
            position=0.70,
            showgrid=False
        ),
        template='plotly_white',
        height=700,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    # Add workout statistics as annotations
    annotations = []
    
    # Overall stats
    if 'power' in df.columns:
        avg_power = df['power'].mean()
        max_power = df['power'].max()
        annotations.append(dict(
            x=0.02, y=0.95, xref='paper', yref='paper',
            text=f"Avg Power: {avg_power:.0f}W<br>Max Power: {max_power:.0f}W",
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ))
    
    if 'heart_rate' in df.columns:
        avg_hr = df['heart_rate'].mean()
        max_hr = df['heart_rate'].max()
        annotations.append(dict(
            x=0.02, y=0.90, xref='paper', yref='paper',
            text=f"Avg HR: {avg_hr:.0f} bpm<br>Max HR: {max_hr:.0f} bpm",
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ))
    
    if 'cadence' in df.columns:
        avg_cadence = df['cadence'].mean()
        max_cadence = df['cadence'].max()
        annotations.append(dict(
            x=0.02, y=0.85, xref='paper', yref='paper',
            text=f"Avg Cadence: {avg_cadence:.0f} rpm<br>Max Cadence: {max_cadence:.0f} rpm",
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ))
    
    # Workout duration
    total_time_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    annotations.append(dict(
        x=0.02, y=0.80, xref='paper', yref='paper',
        text=f"Duration: {total_time_hours:.2f} hours",
        showarrow=False,
        bgcolor='white',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=10)
    ))
    
    fig.update_layout(annotations=annotations)
    
    # Save to HTML file
    # fig.write_html("workout_overview.html")  # Removed to prevent file creation
    # print("Workout overview graph saved to workout_overview.html")  # Removed
    
    # Create a simplified version with just the main metrics (Power, HR, Cadence)
    fig_simple = go.Figure()
    
    if 'power' in df.columns:
        fig_simple.add_trace(go.Scatter(
            x=time_min, y=df['power'],
            mode='lines',
            name='Power (W)',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_simple.add_hline(y=FTP, line_dash="dash", line_color="red", 
                           annotation_text=f"FTP ({FTP}W)")
    
    if 'heart_rate' in df.columns:
        fig_simple.add_trace(go.Scatter(
            x=time_min, y=df['heart_rate'],
            mode='lines',
            name='Heart Rate (bpm)',
            line=dict(color='#d62728', width=2),
            yaxis='y2'
        ))
        # Add LTHR reference line using add_shape
        fig_simple.add_shape(
            type="line",
            x0=time_min.min(), x1=time_min.max(),
            y0=LTHR, y1=LTHR,
            line=dict(dash="dash", color="orange", width=1),
            yref="y2"
        )
        # Add LTHR annotation
        fig_simple.add_annotation(
            x=time_min.max() * 0.95,
            y=LTHR,
            text=f"LTHR ({LTHR} bpm)",
            showarrow=False,
            yref="y2",
            font=dict(color="orange", size=10),
            bgcolor="white",
            bordercolor="orange"
        )
    
    if 'cadence' in df.columns:
        fig_simple.add_trace(go.Scatter(
            x=time_min, y=df['cadence'],
            mode='lines',
            name='Cadence (rpm)',
            line=dict(color='#2ca02c', width=2),
            yaxis='y3'
        ))
    
    fig_simple.update_layout(
        title='Workout Overview - Main Metrics',
        xaxis_title='Time (minutes)',
        yaxis=dict(title=dict(text='Power (W)'), side='left'),
        yaxis2=dict(title=dict(text='Heart Rate (bpm)'), side='right', overlaying='y'),
        yaxis3=dict(title=dict(text='Cadence (rpm)'), side='right', overlaying='y', position=0.95),
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    # fig_simple.write_html("workout_overview_simple.html")  # Removed to prevent file creation
    # print("Simplified workout overview saved to workout_overview_simple.html")  # Removed


def create_dash_dashboard(df, best10, micro_df, sprint_summary_df, long_intervals, evolution_analysis, interval_viz=None, interval_table=None):
    """Create a comprehensive Dash dashboard with all cycling analysis."""
    try:
        import dash
        from dash import dcc, html, Input, Output, State, callback
        import plotly.express as px
        import plotly.graph_objects as go
        from dash.exceptions import PreventUpdate
    except ImportError:
        print("Dash not installed. Install with: pip install dash")
        return
    
    print("\nCreating Dash dashboard...")
    
    # Calculate moving time (excluding stopped time)
    def calculate_moving_time(df):
        """Calculate actual moving time by filtering out stopped periods."""
        # Get the time interval between data points (assuming regular sampling)
        if len(df) > 1:
            time_interval = (df.index[1] - df.index[0]).total_seconds()
        else:
            return 0
        
        if 'enhanced_speed' in df.columns:
            # Consider moving when speed > 0.5 km/h (typical threshold for cycling)
            moving_mask = df['enhanced_speed'] > 0.5
            moving_count = moving_mask.sum()
            moving_time_seconds = moving_count * time_interval
            return moving_time_seconds / 3600  # Convert to hours
        else:
            # Fallback: use power data to estimate moving time
            if 'power' in df.columns:
                # Consider moving when power > 10W (typical threshold for cycling)
                moving_mask = df['power'] > 10
                moving_count = moving_mask.sum()
                moving_time_seconds = moving_count * time_interval
                return moving_time_seconds / 3600
            else:
                # Last resort: use total time
                return (df.index[-1] - df.index[0]).total_seconds() / 3600
    
    # Create Dash app
    app = dash.Dash(__name__, title="Cycling Analysis Dashboard")
    
    # Unit conversion functions
    def convert_to_imperial(value, unit_type):
        """Convert metric values to imperial units."""
        if pd.isna(value) or value is None:
            return value
        
        if unit_type == 'speed':
            return value * 0.621371  # km/h to mph
        elif unit_type == 'distance':
            return value * 0.621371  # km to miles
        else:
            return value
    
    def format_metric(value, unit_type, is_imperial=False):
        """Format metric values with appropriate units."""
        if pd.isna(value) or value is None:
            return "N/A"
        
        if is_imperial and unit_type in ['speed', 'distance']:
            imperial_value = convert_to_imperial(value, unit_type)
            if unit_type == 'speed':
                return f"{imperial_value:.1f} mph"
            elif unit_type == 'distance':
                return f"{imperial_value:.2f} mi"
        else:
            # Keep metric for everything else
            if unit_type == 'speed':
                return f"{value:.1f} km/h"
            elif unit_type == 'distance':
                return f"{value:.2f} km"
            elif unit_type == 'power_kg':
                return f"{value:.2f} W/kg"
            elif unit_type == 'torque':
                return f"{value:.1f} Nm"
            elif unit_type == 'work':
                return f"{value:.1f} kJ"
            else:
                return f"{value:.1f}"
        
        # Fallback for non-converted values
        if unit_type == 'speed':
            return f"{value:.1f} km/h"
        elif unit_type == 'distance':
            return f"{value:.2f} km"
        elif unit_type == 'power_kg':
            return f"{value:.2f} W/kg"
        elif unit_type == 'torque':
            return f"{value:.1f} Nm"
        elif unit_type == 'work':
            return f"{value:.1f} kJ"
        else:
            return f"{value:.1f}"
    
    # Convert time to minutes for x-axis
    time_min = (df.index - df.index[0]).total_seconds() / 60
    
    # Calculate comprehensive metrics like WKO5/TP
    def calculate_comprehensive_metrics():
        metrics = {}
        
        # Basic ride metrics
        total_time = (df.index[-1] - df.index[0]).total_seconds() / 3600  # hours
        moving_time = calculate_moving_time(df)  # Use our new function
        total_distance = 0
        if 'enhanced_speed' in df.columns:
            total_distance = df['enhanced_speed'].sum() / 1000  # km
        
        metrics['total_time_hours'] = total_time
        metrics['moving_time_hours'] = moving_time
        metrics['total_distance_km'] = total_distance
        metrics['avg_speed_kph'] = total_distance / moving_time if moving_time > 0 else 0  # Use moving time for speed
        
        # Power metrics
        if 'power' in df.columns:
            power_data = df['power'].dropna()
            metrics['avg_power'] = power_data.mean()
            metrics['max_power'] = power_data.max()
            metrics['min_power'] = power_data.min()
            metrics['power_std'] = power_data.std()
            metrics['total_work_kj'] = power_data.sum() / 1000
            
            # Normalized Power (Coggan's formula: 30s rolling average, 4th power, mean, 4th root)
            if len(df) > 30:
                rolling_30s = df['power'].rolling(30, min_periods=30).mean()
                # Apply 4th power transformation to penalize high-intensity efforts
                power_4th = rolling_30s ** 4
                # Take mean of 4th power values
                mean_4th_power = power_4th.mean()
                # Take 4th root to get back to watts
                metrics['normalized_power'] = mean_4th_power ** (1/4)
            else:
                metrics['normalized_power'] = np.nan
            
            # Intensity Factor and TSS
            if not np.isnan(metrics['normalized_power']):
                metrics['intensity_factor'] = metrics['normalized_power'] / FTP
                metrics['training_stress_score'] = (moving_time * metrics['intensity_factor'] * metrics['intensity_factor'] * 100)  # Use moving time for TSS
            else:
                metrics['intensity_factor'] = np.nan
                metrics['training_stress_score'] = np.nan
            
            # Power-to-weight ratios
            metrics['avg_power_kg'] = metrics['avg_power'] / rider_mass_kg
            metrics['max_power_kg'] = metrics['max_power'] / rider_mass_kg
            if not np.isnan(metrics['normalized_power']):
                metrics['np_kg'] = metrics['normalized_power'] / rider_mass_kg
            else:
                metrics['np_kg'] = np.nan
            
            # Power zones
            zone_boundaries = [0, FTP*0.55, FTP*0.75, FTP*0.90, FTP*1.05, FTP*1.20, FTP*1.50, power_data.max()]
            zone_labels = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5', 'Zone 6', 'Zone 7']
            
            for i, (min_bound, max_bound) in enumerate(zip(zone_boundaries[:-1], zone_boundaries[1:])):
                zone_count = ((power_data >= min_bound) & (power_data < max_bound)).sum()
                zone_percentage = (zone_count / len(power_data)) * 100
                zone_time = (zone_count * (df.index[1] - df.index[0]).total_seconds()) / 60 if len(df) > 1 else 0
                
                metrics[f'zone_{i+1}_count'] = zone_count
                metrics[f'zone_{i+1}_percentage'] = zone_percentage
                metrics[f'zone_{i+1}_time_min'] = zone_time
        
        # Heart rate metrics
        if 'heart_rate' in df.columns:
            hr_data = df['heart_rate'].dropna()
            if not hr_data.empty:
                metrics['avg_hr'] = hr_data.mean()
                metrics['max_hr'] = hr_data.max()
                metrics['min_hr'] = hr_data.min()
                metrics['hr_std'] = hr_data.std()
                
                # HR zones using LTHR
                hr_zone_boundaries = [0, LTHR*0.85, LTHR*0.95, LTHR*1.05, LTHR*1.15, hr_data.max()]
                hr_zone_labels = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5']
                
                for i, (min_bound, max_bound) in enumerate(zip(hr_zone_boundaries[:-1], hr_zone_boundaries[1:])):
                    zone_count = ((hr_data >= min_bound) & (hr_data < max_bound)).sum()
                    zone_percentage = (zone_count / len(hr_data)) * 100
                    zone_time = (zone_count * (df.index[1] - df.index[0]).total_seconds()) / 60 if len(df) > 1 else 0
                    
                    metrics[f'hr_zone_{i+1}_count'] = zone_count
                    metrics[f'hr_zone_{i+1}_percentage'] = zone_percentage
                    metrics[f'hr_zone_{i+1}_time_min'] = zone_time
        
        # Cadence metrics
        if 'cadence' in df.columns:
            cadence_data = df['cadence'].dropna()
            if not cadence_data.empty:
                metrics['avg_cadence'] = cadence_data.mean()
                metrics['max_cadence'] = cadence_data.max()
                metrics['min_cadence'] = cadence_data.min()
                metrics['cadence_std'] = cadence_data.std()
                
                # Cadence zones
                cadence_zones = {
                    'low': (cadence_data < 80).sum(),
                    'medium': ((cadence_data >= 80) & (cadence_data < 100)).sum(),
                    'high': ((cadence_data >= 100) & (cadence_data < 120)).sum(),
                    'very_high': (cadence_data >= 120).sum()
                }
                
                for zone, count in cadence_zones.items():
                    percentage = (count / len(cadence_data)) * 100
                    metrics[f'cadence_{zone}_count'] = count
                    metrics[f'cadence_{zone}_percentage'] = percentage
        
        # Torque metrics
        if 'torque' in df.columns:
            torque_data = df['torque'].dropna()
            if not torque_data.empty:
                metrics['avg_torque'] = torque_data.mean()
                metrics['max_torque'] = torque_data.max()
                metrics['min_torque'] = torque_data.min()
                metrics['torque_std'] = torque_data.std()
        
        # Speed metrics
        if 'enhanced_speed' in df.columns:
            speed_data = df['enhanced_speed'].dropna()
            if not speed_data.empty:
                metrics['avg_speed_ms'] = speed_data.mean()
                metrics['max_speed_ms'] = speed_data.max()
                metrics['avg_speed_kph'] = speed_data.mean() * 3.6
                metrics['max_speed_kph'] = speed_data.max() * 3.6
        
        # Sprint metrics
        if not best10.empty:
            metrics['best_10s_power'] = best10.iloc[0]['Power (W)']
            metrics['best_10s_power_kg'] = best10.iloc[0]['Power (W)'] / rider_mass_kg
            metrics['sprint_count'] = len(best10)
            
            if len(best10) > 1:
                sprint_powers = best10['Power (W)'].values
                metrics['sprint_power_std'] = np.std(sprint_powers)
                metrics['sprint_power_cv'] = (metrics['sprint_power_std'] / np.mean(sprint_powers)) * 100
        
        # Interval metrics
        if not long_intervals.empty:
            metrics['interval_count'] = len(long_intervals)
            metrics['best_interval_power'] = long_intervals.iloc[0]['avg_power']
            metrics['best_interval_power_kg'] = long_intervals.iloc[0]['avg_power'] / rider_mass_kg
            metrics['total_interval_work_kj'] = long_intervals['work_kj'].sum()
            
            # Interval work as percentage of total
            if 'total_work_kj' in metrics:
                metrics['interval_work_percentage'] = (metrics['total_interval_work_kj'] / metrics['total_work_kj']) * 100
        
        return metrics
    
    # Get comprehensive metrics
    comprehensive_metrics = calculate_comprehensive_metrics()
    
    # Create the main workout overview graph
    def create_workout_overview():
        fig = go.Figure()
        
        # Convert time to minutes for x-axis
        time_min = (df.index - df.index[0]).total_seconds() / 60
        
        # Power (primary metric)
        if 'power' in df.columns:
            fig.add_trace(go.Scatter(
                x=time_min, y=df['power'],
                mode='lines',
                name='Power (W)',
                line=dict(color='#1f77b4', width=2),
                yaxis='y'
            ))
            fig.add_hline(y=FTP, line_dash="dash", line_color="red", 
                         annotation_text=f"FTP ({FTP}W)", line_width=1)
        
        # Heart Rate
        if 'heart_rate' in df.columns:
            fig.add_trace(go.Scatter(
                x=time_min, y=df['heart_rate'],
                mode='lines',
                name='Heart Rate (bpm)',
                line=dict(color='#d62728', width=2),
                yaxis='y2'
            ))
            # Add LTHR reference line using add_shape
            fig.add_shape(
                type="line",
                x0=time_min.min(), x1=time_min.max(),
                y0=LTHR, y1=LTHR,
                line=dict(dash="dash", color="orange", width=1),
                yref="y2"
            )
            # Add LTHR annotation
            fig.add_annotation(
                x=time_min.max() * 0.95,
                y=LTHR,
                text=f"LTHR ({LTHR} bpm)",
                showarrow=False,
                yref="y2",
                font=dict(color="orange", size=10),
                bgcolor="white",
                bordercolor="orange"
            )
        
        # Cadence
        if 'cadence' in df.columns:
            fig.add_trace(go.Scatter(
                x=time_min, y=df['cadence'],
                mode='lines',
                name='Cadence (rpm)',
                line=dict(color='#2ca02c', width=2),
                yaxis='y3'
            ))
        
        # Speed
        if 'enhanced_speed' in df.columns:
            speed_kph = df['enhanced_speed'] * 3.6
            fig.add_trace(go.Scatter(
                x=time_min, y=speed_kph,
                mode='lines',
                name='Speed (km/h)',
                line=dict(color='#9467bd', width=2),
                yaxis='y4'
            ))
        
        # Torque
        if 'torque' in df.columns:
            fig.add_trace(go.Scatter(
                x=time_min, y=df['torque'],
                mode='lines',
                name='Torque (Nm)',
                line=dict(color='#ff7f0e', width=2),
                yaxis='y5'
            ))
        
        # Update layout
        fig.update_layout(
            title='Workout Overview - All Metrics Over Time',
            xaxis=dict(title='Time (minutes)', showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title=dict(text='Power (W)', font=dict(color='#1f77b4')), 
                      tickfont=dict(color='#1f77b4'), side='left'),
            yaxis2=dict(title=dict(text='Heart Rate (bpm)', font=dict(color='#d62728')), 
                       tickfont=dict(color='#d62728'), side='right', overlaying='y', position=0.95),
            yaxis3=dict(title=dict(text='Cadence (rpm)', font=dict(color='#2ca02c')), 
                       tickfont=dict(color='#2ca02c'), side='right', overlaying='y', position=0.90),
            yaxis4=dict(title=dict(text='Speed (km/h)', font=dict(color='#9467bd')), 
                       tickfont=dict(color='#9467bd'), side='right', overlaying='y', position=0.85),
            yaxis5=dict(title=dict(text='Torque (Nm)', font=dict(color='#ff7f0e')), 
                       tickfont=dict(color='#ff7f0e'), side='right', overlaying='y', position=0.80),
            template='plotly_white',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    # Create power curve graph
    def create_power_curve():
        durations = [5, 10, 20, 30, 60, 120, 300]
        best_powers = []
        
        for duration in durations:
            if len(df) >= duration:
                rolling_power = df['power'].rolling(duration, min_periods=duration).mean()
                best_power = rolling_power.max()
                if not np.isnan(best_power):
                    best_powers.append(best_power)
                else:
                    best_powers.append(0)
            else:
                best_powers.append(0)
        
        # Filter out zero values
        valid_durations = [d for d, p in zip(durations, best_powers) if p > 0]
        valid_powers = [p for p in best_powers if p > 0]
        
        fig = go.Figure()
        
        # Main power curve
        fig.add_trace(go.Scatter(
            x=valid_durations, y=valid_powers, 
            mode='lines+markers',
            name='Power (W)',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10, color='#1f77b4')
        ))
        
        # Add FTP reference line
        fig.add_hline(y=FTP, line_dash="dash", line_color="red", 
                     annotation_text=f'FTP ({FTP}W)')
        
        # Add power-to-weight on secondary y-axis
        powers_kg = [p / rider_mass_kg for p in valid_powers]
        fig.add_trace(go.Scatter(
            x=valid_durations, y=powers_kg, 
            mode='lines+markers',
            name='Power (W/kg)',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8, color='#ff7f0e'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Power Curve - Best Efforts at Different Durations',
            xaxis_title='Duration (seconds)',
            yaxis=dict(title='Power (W)', side='left'),
            yaxis2=dict(title='Power (W/kg)', side='right', overlaying='y'),
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    # Create sprint comparison graph
    def create_sprint_comparison():
        if best10.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        sprint_numbers = [f'Sprint {i+1}' for i in range(len(best10))]
        avg_powers = best10['Power (W)'].values
        
        # Get max powers for each sprint
        max_powers = []
        for _, row in best10.iterrows():
            start_ts = row['start_ts']
            seg = df.loc[start_ts : start_ts + pd.Timedelta(seconds=9)]
            max_powers.append(seg['power'].max())
        
        # Average power bars
        fig.add_trace(go.Bar(
            x=sprint_numbers, y=avg_powers,
            name='Average Power (10s)',
            marker_color='#2E86AB',
            text=[f'{p:.0f}W' for p in avg_powers],
            textposition='auto'
        ))
        
        # Max power bars
        fig.add_trace(go.Bar(
            x=sprint_numbers, y=max_powers,
            name='Peak Power',
            marker_color='#A23B72',
            text=[f'{p:.0f}W' for p in max_powers],
            textposition='auto'
        ))
        
        # Add FTP reference line
        fig.add_hline(y=FTP, line_dash="dash", line_color="red", 
                     annotation_text=f'FTP ({FTP}W)')
        
        fig.update_layout(
            title='Sprint Power Comparison',
            xaxis_title='Sprint',
            yaxis_title='Power (W)',
            template='plotly_white',
            height=500,
            showlegend=True,
            barmode='group'
        )
        
        return fig
    
    # Create interval power comparison
    def create_interval_comparison():
        if long_intervals.empty:
            return go.Figure()
        
        # Sort intervals by start time to show chronological order
        sorted_intervals = long_intervals.sort_values('start_time').copy()
        
        fig = go.Figure()
        
        # Create simple x-axis labels: Interval 1, Interval 2, etc.
        x_labels = [f"Interval {i+1}" for i in range(len(sorted_intervals))]
        
        # Add bars for average power
        fig.add_trace(go.Bar(
            x=x_labels,
            y=sorted_intervals['avg_power'],
            name='Average Power',
            marker_color='#1f77b4',
            text=[f"{p:.0f}W" for p in sorted_intervals['avg_power']],
            textposition='auto'
        ))
        
        # Add bars for max power
        fig.add_trace(go.Bar(
            x=x_labels,
            y=sorted_intervals['max_power'],
            name='Max Power',
            marker_color='#ff7f0e',
            text=[f"{p:.0f}W" for p in sorted_intervals['max_power']],
            textposition='auto'
        ))
        
        # Add FTP reference line
        fig.add_hline(y=FTP, line_dash="dash", line_color="red", 
                     annotation_text=f"FTP ({FTP}W)")
        
        fig.update_layout(
            title='Interval Power Comparison (Chronological Order)',
            xaxis_title='Intervals',
            yaxis_title='Power (W)',
            template='plotly_white',
            height=500,
            barmode='group',
            showlegend=True
        )
        
        return fig
    
    # Create torque vs cadence plot
    def create_torque_vs_cadence():
        if 'torque' not in df.columns or 'cadence' not in df.columns or 'power' not in df.columns:
            return go.Figure()
        
        # Filter data
        data = df[(df["cadence"] > 10) & df["torque"].notna()].copy()
        
        fig = go.Figure(go.Scatter(
            x=data["cadence"], y=data["torque"], mode='markers',
            marker=dict(size=5, color=data["power"], colorscale='Viridis',
                        colorbar=dict(title="Power (W)")),
            hovertemplate=("Cadence: %{x:.1f} rpm<br>"
                          "Torque: %{y:.2f} Nm<br>"
                          "Power: %{marker.color:.0f} W<br>"
                          "<extra></extra>")
        ))
        
        fig.update_layout(
            title="Torque vs RPM (colored by Power)",
            xaxis_title="Cadence (rpm)",
            yaxis_title="Torque (Nm)",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    # Create interval table directly (no callback needed)
    def create_interval_table_direct(units='metric'):
        is_imperial = units == 'imperial'
        
        if long_intervals.empty:
            return html.P("No interval data available")
        
        # Create interval table
        table_data = []
        for i, row in long_intervals.iterrows():
            # Format duration: show seconds if < 1 min, otherwise show minutes
            duration_s = row['duration_s']
            if duration_s < 60:
                duration_str = f"{duration_s:.0f}s"
            else:
                duration_str = f"{duration_s/60:.1f} min"
            
            # Keep work in kJ (metric)
            work_kj = row['work_kj']
            work_text = f"{work_kj:.1f} kJ"
            
            table_data.append(html.Tr([
                html.Td(f"#{row['rank']}"),
                html.Td(row['start_str']),
                html.Td(duration_str),
                html.Td(f"{row['avg_power']:.0f}W"),
                html.Td(f"{row['max_power']:.0f}W"),
                html.Td(work_text)
            ]))
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Rank"),
                html.Th("Start Time"),
                html.Th("Duration"),
                html.Th("Avg Power (W)"),
                html.Th("Max Power (W)"),
                html.Th("Work (kJ)")
            ])),
            html.Tbody(table_data)
        ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
    
    # Create coggan power zones
    def create_power_zones():
        if 'power' not in df.columns:
            return go.Figure()
        
        power_data = df['power'].dropna()
        zone_boundaries = [0, FTP*0.55, FTP*0.75, FTP*0.90, FTP*1.05, FTP*1.20, FTP*1.50, power_data.max()]
        zone_labels = ['Zone 1\n(Active Recovery)', 'Zone 2\n(Endurance)', 'Zone 3\n(Tempo)', 
                      'Zone 4\n(Lactate Threshold)', 'Zone 5\n(VO2 Max)', 'Zone 6\n(Anaerobic)', 
                      'Zone 7\n(Neuromuscular)']
        zone_colors = ['#87CEEB', '#90EE90', '#FFFFE0', '#FFA500', '#FF4500', '#9370DB', '#8B0000']
        
        zone_counts = []
        zone_percentages = []
        for i in range(len(zone_boundaries)-1):
            count = ((power_data >= zone_boundaries[i]) & (power_data < zone_boundaries[i+1])).sum()
            zone_counts.append(count)
            percentage = (count / len(power_data)) * 100
            zone_percentages.append(percentage)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=zone_counts,
            y=zone_labels,
            orientation='h',
            marker_color=zone_colors,
            text=[f'{p:.1f}%' for p in zone_percentages],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Coggan Power Zones Distribution (FTP = {FTP}W)',
            xaxis_title='Data Points',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig
    

    
    # Create timing comparison charts
    def create_timing_comparison_charts(units='metric'):
        """Create charts comparing timing metrics across intervals and sprints."""
        is_imperial = units == 'imperial'
        
        if long_intervals.empty and micro_df.empty:
            return go.Figure()
        
        # Create subplots for different timing comparisons
        fig = go.Figure()
        
        # If we have intervals, create interval timing chart
        if not long_intervals.empty:
            # Get time to max power for each interval
            interval_times = []
            interval_names = []
            
            for i, row in long_intervals.iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                interval_data = df.loc[start_time:end_time]
                
                if not interval_data.empty and 'power' in interval_data.columns:
                    max_power_idx = interval_data['power'].idxmax()
                    time_to_max = (max_power_idx - start_time).total_seconds()
                    interval_times.append(time_to_max)
                    interval_names.append(f"Interval {row['rank']}")
            
            if interval_times:
                fig.add_trace(go.Bar(
                    x=interval_names,
                    y=interval_times,
                    name='Time to Max Power',
                    marker_color='#1f77b4',
                    text=[f'{t:.1f}s' for t in interval_times],
                    textposition='auto'
                ))
        
        # If we have sprints, add sprint timing data
        if not micro_df.empty:
            sprint_names = [f"Sprint {i+1}" for i in range(len(micro_df))]
            
            # Power timing
            if 'power_ttp_s' in micro_df.columns:
                fig.add_trace(go.Bar(
                    x=sprint_names,
                    y=micro_df['power_ttp_s'],
                    name='Sprint Power TTP',
                    marker_color='#ff7f0e',
                    text=[f'{t:.1f}s' if not np.isnan(t) else 'N/A' for t in micro_df['power_ttp_s']],
                    textposition='auto'
                ))
        
        fig.update_layout(
            title='Timing Analysis: Time to Maximum Values',
            xaxis_title='Intervals/Sprints',
            yaxis_title='Time (seconds)',
            template='plotly_white',
            height=400,
            barmode='group',
            showlegend=True
        )
        
        return fig
    
    # Create sustained power trend analysis
    def create_sustained_power_trends():
        """Create charts showing sustained power trends and decay curves."""
        if long_intervals.empty:
            return go.Figure()
        
        # Create subplots for different sustained power analyses
        fig = go.Figure()
        
        # 1. Power decay curves for each interval
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, row in long_intervals.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            interval_data = df.loc[start_time:end_time]
            
            if not interval_data.empty and 'power' in interval_data.columns:
                # Calculate time from start for each data point
                time_from_start = (interval_data.index - start_time).total_seconds()
                
                # Normalize power to percentage of max for this interval
                max_power = interval_data['power'].max()
                power_pct = (interval_data['power'] / max_power) * 100
                
                # Add power decay curve
                fig.add_trace(go.Scatter(
                    x=time_from_start,
                    y=power_pct,
                    mode='lines',
                    name=f"Interval {row['rank']} ({row['duration_s']:.0f}s)",
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"Interval {row['rank']}<br>Time: %{{x:.1f}}s<br>Power: %{{y:.1f}}%<extra></extra>"
                ))
        
        # Add reference lines for power thresholds
        fig.add_hline(y=90, line_dash="dash", line_color="green", 
                     annotation_text="90% of Max", annotation_position="right")
        fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                     annotation_text="80% of Max", annotation_position="right")
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="70% of Max", annotation_position="right")
        
        fig.update_layout(
            title='Sustained Power Trends: Power Decay Curves',
            xaxis_title='Time from Start (seconds)',
            yaxis_title='Power (% of Max)',
            template='plotly_white',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    # Create sustained power metrics table
    def create_sustained_power_metrics():
        """Create a table showing sustained power metrics for each interval."""
        if long_intervals.empty:
            return html.P("No interval data available for sustained power analysis")
        
        # Create table data
        table_data = []
        
        for i, row in long_intervals.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            interval_data = df.loc[start_time:end_time]
            
            if not interval_data.empty and 'power' in interval_data.columns:
                max_power = interval_data['power'].max()
                avg_power = interval_data['power'].mean()
                
                # Calculate time above different power thresholds
                time_above_90 = ((interval_data['power'] >= max_power * 0.9).sum() * 
                                (interval_data.index[1] - interval_data.index[0]).total_seconds())
                time_above_80 = ((interval_data['power'] >= max_power * 0.8).sum() * 
                                (interval_data.index[1] - interval_data.index[0]).total_seconds())
                time_above_70 = ((interval_data['power'] >= max_power * 0.7).sum() * 
                                (interval_data.index[1] - interval_data.index[0]).total_seconds())
                
                # Calculate power consistency (coefficient of variation)
                power_std = interval_data['power'].std()
                power_cv = (power_std / avg_power) * 100 if avg_power > 0 else 0
                
                # Calculate pacing strategy
                first_half = interval_data.iloc[:len(interval_data)//2]
                second_half = interval_data.iloc[len(interval_data)//2:]
                first_half_power = first_half['power'].mean()
                second_half_power = second_half['power'].mean()
                
                if first_half_power > second_half_power * 1.1:
                    pacing = "Front-loaded"
                elif second_half_power > first_half_power * 1.1:
                    pacing = "Negative split"
                else:
                    pacing = "Even pacing"
                
                # Calculate power fade percentage
                power_fade = ((interval_data['power'].iloc[-5:].mean() - max_power) / max_power) * 100
                
                table_data.append(html.Tr([
                    html.Td(f"Interval {row['rank']}"),
                    html.Td(f"{row['duration_s']:.0f}s"),
                    html.Td(f"{max_power:.0f}W"),
                    html.Td(f"{avg_power:.0f}W"),
                    html.Td(f"{time_above_90:.1f}s"),
                    html.Td(f"{time_above_80:.1f}s"),
                    html.Td(f"{time_above_70:.1f}s"),
                    html.Td(f"{power_cv:.1f}%"),
                    html.Td(pacing),
                    html.Td(f"{power_fade:.1f}%")
                ]))
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Interval"),
                html.Th("Duration"),
                html.Th("Max Power"),
                html.Th("Avg Power"),
                html.Th("Time Above 90%"),
                html.Th("Time Above 80%"),
                html.Th("Time Above 70%"),
                html.Th("Power CV"),
                html.Th("Pacing Strategy"),
                html.Th("Power Fade")
            ])),
            html.Tbody(table_data)
        ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
    
    # Create power distribution analysis
    def create_power_distribution_analysis():
        """Create charts showing power distribution and consistency analysis."""
        if long_intervals.empty:
            return go.Figure()
        
        # Create subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Power Distribution by Interval', 'Power Consistency (CV)', 
                          'Sustained Power Ratios', 'Pacing Strategy Analysis'),
            specs=[[{"type": "box"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. Power distribution box plots
        for i, row in long_intervals.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            interval_data = df.loc[start_time:end_time]
            
            if not interval_data.empty and 'power' in interval_data.columns:
                fig.add_trace(
                    go.Box(
                        y=interval_data['power'],
                        name=f"Interval {row['rank']}",
                        boxpoints='outliers',
                        jitter=0.3,
                        pointpos=-1.8
                    ),
                    row=1, col=1
                )
        
        # 2. Power consistency (CV) bar chart
        intervals = []
        power_cvs = []
        
        for i, row in long_intervals.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            interval_data = df.loc[start_time:end_time]
            
            if not interval_data.empty and 'power' in interval_data.columns:
                avg_power = interval_data['power'].mean()
                power_std = interval_data['power'].std()
                power_cv = (power_std / avg_power) * 100 if avg_power > 0 else 0
                
                intervals.append(f"Interval {row['rank']}")
                power_cvs.append(power_cv)
        
        fig.add_trace(
            go.Bar(
                x=intervals,
                y=power_cvs,
                name='Power CV (%)',
                marker_color='#ff7f0e'
            ),
            row=1, col=2
        )
        
        # 3. Sustained power ratios
        intervals = []
        sustained_ratios = []
        
        for i, row in long_intervals.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            interval_data = df.loc[start_time:end_time]
            
            if not interval_data.empty and 'power' in interval_data.columns:
                max_power = interval_data['power'].max()
                avg_power = interval_data['power'].mean()
                sustained_ratio = (avg_power / max_power) * 100
                
                intervals.append(f"Interval {row['rank']}")
                sustained_ratios.append(sustained_ratio)
        
        fig.add_trace(
            go.Bar(
                x=intervals,
                y=sustained_ratios,
                name='Sustained Power Ratio (%)',
                marker_color='#2ca02c'
            ),
            row=2, col=1
        )
        
        # 4. Pacing strategy pie chart
        pacing_counts = {'Front-loaded': 0, 'Even pacing': 0, 'Negative split': 0}
        
        for i, row in long_intervals.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            interval_data = df.loc[start_time:end_time]
            
            if not interval_data.empty and 'power' in interval_data.columns:
                first_half = interval_data.iloc[:len(interval_data)//2]
                second_half = interval_data.iloc[len(interval_data)//2:]
                first_half_power = first_half['power'].mean()
                second_half_power = second_half['power'].mean()
                
                if first_half_power > second_half_power * 1.1:
                    pacing_counts['Front-loaded'] += 1
                elif second_half_power > first_half_power * 1.1:
                    pacing_counts['Negative split'] += 1
                else:
                    pacing_counts['Even pacing'] += 1
        
        fig.add_trace(
            go.Pie(
                labels=list(pacing_counts.keys()),
                values=list(pacing_counts.values()),
                name='Pacing Strategy'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Sustained Power Analysis Dashboard',
            template='plotly_white',
            height=800,
            showlegend=False
        )
        
        return fig
    
    # Create comprehensive timing analysis table
    def create_timing_analysis_table():
        """Create a comprehensive table showing all timing metrics."""
        if long_intervals.empty and micro_df.empty:
            return html.P("No timing data available")
        
        # Create table data
        table_data = []
        
        # Add interval timing data
        if not long_intervals.empty:
            for i, row in long_intervals.iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                interval_data = df.loc[start_time:end_time]
                
                # Calculate timing metrics
                if not interval_data.empty:
                    # Power timing
                    if 'power' in interval_data.columns:
                        max_power_idx = interval_data['power'].idxmax()
                        time_to_max_power = (max_power_idx - start_time).total_seconds()
                    else:
                        time_to_max_power = np.nan
                    
                    # Torque timing
                    if 'torque' in interval_data.columns:
                        max_torque_idx = interval_data['torque'].idxmax()
                        time_to_max_torque = (max_torque_idx - start_time).total_seconds()
                    else:
                        time_to_max_torque = np.nan
                    
                    # Cadence timing
                    if 'cadence' in interval_data.columns:
                        max_cadence_idx = interval_data['cadence'].idxmax()
                        time_to_max_cadence = (max_cadence_idx - start_time).total_seconds()
                    else:
                        time_to_max_cadence = np.nan
                    
                    table_data.append(html.Tr([
                        html.Td(f"Interval {row['rank']}"),
                        html.Td(row['start_str']),
                        html.Td(f"{time_to_max_power:.1f}s" if not np.isnan(time_to_max_power) else "N/A"),
                        html.Td(f"{time_to_max_torque:.1f}s" if not np.isnan(time_to_max_torque) else "N/A"),
                        html.Td(f"{time_to_max_cadence:.1f}s" if not np.isnan(time_to_max_cadence) else "N/A"),
                        html.Td(f"{row['avg_power']:.0f}W"),
                        html.Td(f"{row['max_power']:.0f}W")
                    ]))
        
        # Add sprint timing data
        if not micro_df.empty:
            for i, row in micro_df.iterrows():
                table_data.append(html.Tr([
                    html.Td(f"Sprint {i+1}"),
                    html.Td(row.get('start', 'N/A')),
                    html.Td(f"{row.get('power_ttp_s', 'N/A'):.1f}s" if isinstance(row.get('power_ttp_s'), (int, float)) else row.get('power_ttp_s', 'N/A')),
                    html.Td(f"{row.get('torque_ttp_s', 'N/A'):.1f}s" if isinstance(row.get('torque_ttp_s'), (int, float)) else row.get('torque_ttp_s', 'N/A')),
                    html.Td(f"{row.get('cadence_ttp_s', 'N/A'):.1f}s" if isinstance(row.get('cadence_ttp_s'), (int, float)) else row.get('cadence_ttp_s', 'N/A')),
                    html.Td(f"{row.get('power_max', 'N/A'):.0f}W" if isinstance(row.get('power_max'), (int, float)) else row.get('power_max', 'N/A')),
                    html.Td(f"{row.get('power_max', 'N/A'):.0f}W" if isinstance(row.get('power_max'), (int, float)) else row.get('power_max', 'N/A'))
                ]))
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Type"),
                html.Th("Start Time"),
                html.Th("Time to Max Power"),
                html.Th("Time to Max Torque"),
                html.Th("Time to Max Cadence"),
                html.Th("Avg Power"),
                html.Th("Max Power")
            ])),
            html.Tbody(table_data)
        ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
    
    # Create comprehensive metrics table
    def create_metrics_table(units='metric'):
        is_imperial = units == 'imperial'
        
        # Group metrics by category
        ride_metrics = [
            ('Total Time', f"{comprehensive_metrics.get('total_time_hours', 0):.2f} hours"),
            ('Moving Time', f"{comprehensive_metrics.get('moving_time_hours', 0):.2f} hours"),
            ('Total Distance', format_metric(comprehensive_metrics.get('total_distance_km', 0), 'distance', is_imperial)),
            ('Average Speed', format_metric(comprehensive_metrics.get('avg_speed_kph', 0), 'speed', is_imperial))
        ]
        
        power_metrics = [
            ('Average Power', f"{comprehensive_metrics.get('avg_power', 0):.0f}W ({comprehensive_metrics.get('avg_power_kg', 0):.1f} W/kg)"),
            ('Max Power', f"{comprehensive_metrics.get('max_power', 0):.0f}W ({comprehensive_metrics.get('max_power_kg', 0):.1f} W/kg)"),
            ('Normalized Power', f"{comprehensive_metrics.get('normalized_power', 0):.0f}W ({comprehensive_metrics.get('np_kg', 0):.1f} W/kg)"),
            ('Intensity Factor', f"{comprehensive_metrics.get('intensity_factor', 0):.2f}"),
            ('Training Stress Score', f"{comprehensive_metrics.get('training_stress_score', 0):.0f}"),
            ('Total Work', f"{comprehensive_metrics.get('total_work_kj', 0):.1f} kJ")
        ]
        
        hr_metrics = []
        if 'avg_hr' in comprehensive_metrics:
            hr_metrics = [
                ('Average HR', f"{comprehensive_metrics['avg_hr']:.0f} bpm"),
                ('Max HR', f"{comprehensive_metrics['max_hr']:.0f} bpm"),
                ('Min HR', f"{comprehensive_metrics['min_hr']:.0f} bpm")
            ]
        
        cadence_metrics = []
        if 'avg_cadence' in comprehensive_metrics:
            cadence_metrics = [
                ('Average Cadence', f"{comprehensive_metrics['avg_cadence']:.1f} rpm"),
                ('Max Cadence', f"{comprehensive_metrics['max_cadence']:.1f} rpm"),
                ('Min Cadence', f"{comprehensive_metrics['min_cadence']:.1f} rpm")
            ]
        
        sprint_metrics = []
        if 'best_10s_power' in comprehensive_metrics:
            sprint_metrics = [
                ('Best 10s Power', f"{comprehensive_metrics['best_10s_power']:.0f}W ({comprehensive_metrics['best_10s_power_kg']:.1f} W/kg)"),
                ('Sprint Count', f"{comprehensive_metrics['sprint_count']}"),
                ('Sprint Consistency (CV)', f"{comprehensive_metrics.get('sprint_power_cv', 0):.1f}%")
            ]
        
        interval_metrics = []
        if 'best_interval_power' in comprehensive_metrics:
            interval_metrics = [
                ('Best Interval Power', f"{comprehensive_metrics['best_interval_power']:.0f}W ({comprehensive_metrics['best_interval_power_kg']:.1f} W/kg)"),
                ('Interval Count', f"{comprehensive_metrics['interval_count']}"),
                ('Interval Work %', f"{comprehensive_metrics.get('interval_work_percentage', 0):.1f}%")
            ]
        
        # Create table rows
        table_rows = []
        
        # Ride metrics
        table_rows.append(html.Tr([html.Th("RIDE METRICS", colSpan=2, style={'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'})]))
        for label, value in ride_metrics:
            table_rows.append(html.Tr([html.Td(label), html.Td(value)]))
        
        # Power metrics
        table_rows.append(html.Tr([html.Th("POWER METRICS", colSpan=2, style={'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'})]))
        for label, value in power_metrics:
            table_rows.append(html.Tr([html.Td(label), html.Td(value)]))
        
        # HR metrics
        if hr_metrics:
            table_rows.append(html.Tr([html.Th("HEART RATE METRICS", colSpan=2, style={'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'})]))
            for label, value in hr_metrics:
                table_rows.append(html.Tr([html.Td(label), html.Td(value)]))
        
        # Cadence metrics
        if cadence_metrics:
            table_rows.append(html.Tr([html.Th("CADENCE METRICS", colSpan=2, style={'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'})]))
            for label, value in cadence_metrics:
                table_rows.append(html.Tr([html.Td(label), html.Td(value)]))
        
        # Sprint metrics
        if sprint_metrics:
            table_rows.append(html.Tr([html.Th("SPRINT METRICS", colSpan=2, style={'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'})]))
            for label, value in sprint_metrics:
                table_rows.append(html.Tr([html.Td(label), html.Td(value)]))
        
        # Interval metrics
        if interval_metrics:
            table_rows.append(html.Tr([html.Th("INTERVAL METRICS", colSpan=2, style={'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'})]))
            for label, value in interval_metrics:
                table_rows.append(html.Tr([html.Td(label), html.Td(value)]))
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Metric", style={'width': '50%'}),
                html.Th("Value", style={'width': '50%'})
            ])),
            html.Tbody(table_rows)
        ], style={
            'width': '100%', 
            'border': '1px solid #ddd', 
            'borderCollapse': 'collapse',
            'fontSize': '14px'
        })
    
    # Dashboard layout
    app.layout = html.Div([
        # Header with title and info button
        html.Div([
            html.H1("Cycling Analysis Dashboard", 
                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 0, 'flex': 1}),
            html.Button(
                "Metrics Guide",
                id='open-metrics-guide',
                style={
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '16px',
                    'marginLeft': '20px'
                }
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': 30}),
        
        # Unit toggle button
        html.Div([
            html.H4("Units", style={'color': '#34495e', 'marginBottom': '10px'}),
            dcc.RadioItems(
                id='unit-toggle',
                options=[
                    {'label': 'Imperial (mph, mi)', 'value': 'imperial'},
                    {'label': 'Metric (km/h, km)', 'value': 'metric'}
                ],
                value='metric',
                inline=True,
                style={'marginBottom': '20px'}
            )
        ], style={'textAlign': 'center', 'marginBottom': 20}),
        
        # Summary statistics
        html.Div([
            html.H3("Workout Summary", style={'color': '#34495e'}),
            html.Div([
                html.Div([
                    html.H4("Duration"),
                    html.P(id='duration-display')
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                html.Div([
                    html.H4("Total Time"),
                    html.P(id='total-time-display')
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                html.Div([
                    html.H4("Avg Power"),
                    html.P(id='avg-power-display')
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                html.Div([
                    html.H4("Max Power"),
                    html.P(id='max-power-display')
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                html.Div([
                    html.H4("Total Work"),
                    html.P(id='total-work-display')
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})
        ], style={'marginBottom': 30}),
        
        # Comprehensive metrics table
        html.Div([
            html.H3("Comprehensive Metrics (WKO5/TP Style)", style={'color': '#34495e'}),
            html.Div(id='comprehensive-metrics-table', children=create_metrics_table('metric'))
        ], style={'marginBottom': 30}),
        
        # Main workout overview
        html.Div([
            html.H3("Workout Overview", style={'color': '#34495e'}),
            dcc.Graph(id='workout-overview', figure=create_workout_overview())
        ], style={'marginBottom': 30}),
        
        # Power analysis section
        html.Div([
            html.H3("Power Analysis", style={'color': '#34495e'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='power-curve', figure=create_power_curve())
                ], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='power-zones', figure=create_power_zones())
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ], style={'marginBottom': 30}),
        
        # Sprint analysis section
        html.Div([
            html.H3("Sprint Analysis", style={'color': '#34495e'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='sprint-comparison', figure=create_sprint_comparison())
                ], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='torque-vs-cadence', figure=create_torque_vs_cadence())
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ], style={'marginBottom': 30}),
        
        # Timing analysis section
        html.Div([
            html.H3("Timing Analysis", style={'color': '#34495e'}),
            html.Div(id='timing-comparison', children=dcc.Graph(figure=create_timing_comparison_charts('metric'))),
            html.Div([
                html.H4("Comprehensive Timing Metrics", style={'color': '#34495e', 'marginTop': '20px'}),
                html.Div(id='timing-analysis-table', children=create_timing_analysis_table())
            ])
        ], style={'marginBottom': 30}) if (not long_intervals.empty or not micro_df.empty) else html.Div(),
        

        
        # Interval analysis section
        html.Div([
            html.H3("Interval Analysis", style={'color': '#34495e'}),
            dcc.Graph(id='interval-comparison', figure=create_interval_comparison())
        ], style={'marginBottom': 30}),
        
        # Enhanced interval visualization (full width)
        html.Div([
            html.H3("Enhanced Interval Visualization", style={'color': '#34495e'}),
            dcc.Graph(id='enhanced-interval-full', figure=interval_viz if interval_viz else go.Figure())
        ], style={'marginBottom': 30}) if interval_viz else html.Div(),
        
        # Sprint data table
        html.Div([
            html.H3("Sprint Data", style={'color': '#34495e'}),
            html.Div(id='sprint-table')
        ], style={'marginBottom': 30}),
        
        # Interval data table
        html.Div([
            html.H3("Interval Data", style={'color': '#34495e'}),
            html.Div(id='interval-table', children=create_interval_table_direct('metric'))
        ], style={'marginBottom': 30}),
        
        # Interval evolution analysis section
        html.Div([
            html.H3("Interval Evolution Analysis", style={'color': '#34495e'}),
            html.Div(id='interval-evolution-table')
        ], style={'marginBottom': 30}) if evolution_analysis else html.Div(),
        
        # Interval comparison analysis section
        html.Div([
            html.H3("Interval Comparison Analysis", style={'color': '#34495e'}),
            html.Div(id='interval-comparison-content'),
            # Add comparison plots section
            html.Div([
                html.H4("Comparison Plots", style={'color': '#27ae60', 'marginTop': '20px'}),
                html.Div(id='interval-comparison-plots')
            ])
        ], style={'marginBottom': 30}) if not long_intervals.empty else html.Div(),
        
        # Sprint timing analysis section
        html.Div([
            html.H3("Sprint Timing Analysis", style={'color': '#34495e'}),
            html.Div(id='sprint-timing-table')
        ], style={'marginBottom': 30}) if not micro_df.empty else html.Div(),
        
        # Metrics Guide Section (expandable)
        html.Div([
            html.Div([
                html.Button(
                    "Click to Show/Hide Metrics Guide",
                    id='toggle-metrics-guide',
                    style={
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'border': 'none',
                        'padding': '15px 30px',
                        'borderRadius': '8px',
                        'cursor': 'pointer',
                        'fontSize': '18px',
                        'width': '100%',
                        'marginBottom': '20px'
                    }
                ),
                html.Div(
                    id='metrics-guide-content',
                    style={'display': 'none'}
                )
            ])
        ], style={'marginBottom': 30})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'})
    
    # Callbacks for unit conversion
    @app.callback(
        [Output('duration-display', 'children'),
         Output('total-time-display', 'children'),
         Output('avg-power-display', 'children'),
         Output('max-power-display', 'children'),
         Output('total-work-display', 'children')],
        [Input('unit-toggle', 'value')]
    )
    def update_workout_summary_units(units):
        is_imperial = units == 'imperial'
        
        # Duration (moving time)
        moving_time = calculate_moving_time(df)
        duration_text = f"{moving_time:.2f} hours (moving)"
        
        # Total time
        total_time = (df.index[-1] - df.index[0]).total_seconds() / 3600
        total_time_text = f"{total_time:.2f} hours"
        
        # Average power (keep W/kg metric)
        avg_power = df['power'].mean()
        avg_power_kg = avg_power / rider_mass_kg
        avg_power_text = f"{avg_power:.0f}W ({avg_power_kg:.1f} W/kg)"
        
        # Max power (keep W/kg metric)
        max_power = df['power'].max()
        max_power_kg = max_power / rider_mass_kg
        max_power_text = f"{max_power:.0f}W ({max_power_kg:.1f} W/kg)"
        
        # Total work (keep kJ metric)
        total_work = df['power'].sum() / 1000
        total_work_text = f"{total_work:.1f} kJ"
        
        return duration_text, total_time_text, avg_power_text, max_power_text, total_work_text
    
    # Callback to update sprint table
    @app.callback(
        Output('sprint-table', 'children'),
        [Input('workout-overview', 'clickData'),
         Input('unit-toggle', 'value')]
    )
    def update_sprint_table(click_data, units):
        if best10.empty:
            return html.P("No sprint data available")
        
        # Create sprint table (keep torque in Nm, power in W)
        table_data = []
        for i, row in best10.iterrows():
            table_data.append(html.Tr([
                html.Td(f"Sprint {i+1}"),
                html.Td(row['Start']),
                html.Td(f"{row['Power (W)']:.0f}W"),
                html.Td(f"{row['Torque (Nm)']:.1f} Nm"),
                html.Td(f"{row['Cadence (rpm)']:.1f} rpm")
            ]))
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Sprint"),
                html.Th("Start Time"),
                html.Th("Power (W)"),
                html.Th("Torque (Nm)"),
                html.Th("Cadence (rpm)")
            ])),
            html.Tbody(table_data)
        ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
    
    # Callback to update interval evolution table
    @app.callback(
        Output('interval-evolution-table', 'children'),
        Input('workout-overview', 'clickData')
    )
    def update_interval_evolution_table(click_data):
        if not evolution_analysis:
            return html.P("No interval evolution data available")
        
        # Create interval evolution table
        table_data = []
        for interval_name, evolution in evolution_analysis.items():
            table_data.append(html.Tr([
                html.Td(interval_name.replace('_', ' ').title()),
                html.Td(f"{evolution.get('power_fade_%', 'N/A'):.1f}%" if evolution.get('power_fade_%') is not None else 'N/A'),
                html.Td(f"{evolution.get('power_consistency_%', 'N/A'):.1f}%" if evolution.get('power_consistency_%') is not None else 'N/A'),
                html.Td(f"{evolution.get('cadence_drift_rpm', 'N/A'):.1f} rpm" if evolution.get('cadence_drift_rpm') is not None else 'N/A'),
                html.Td(f"{evolution.get('hr_drift_bpm', 'N/A'):.1f} bpm" if evolution.get('hr_drift_bpm') is not None else 'N/A'),
                html.Td(f"{evolution.get('torque_drift_nm', 'N/A'):.1f} Nm" if evolution.get('torque_drift_nm') is not None else 'N/A'),
                html.Td(f"{evolution.get('speed_gain_kph', 'N/A'):.1f} km/h" if evolution.get('speed_gain_kph') is not None else 'N/A')
            ]))
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Interval"),
                html.Th("Power Fade (%)"),
                html.Th("Power Consistency (%)"),
                html.Th("Cadence Drift (rpm)"),
                html.Th("HR Drift (bpm)"),
                html.Th("Torque Drift (Nm)"),
                html.Th("Speed Gain (km/h)")
            ])),
            html.Tbody(table_data)
        ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
    
    # Callback to update sprint timing table
    @app.callback(
        Output('sprint-timing-table', 'children'),
        [Input('workout-overview', 'clickData'),
         Input('unit-toggle', 'value')]
    )
    def update_sprint_timing_table(click_data, units):
        if micro_df.empty:
            return html.P("No sprint timing data available")
        
        is_imperial = units == 'imperial'
        
        # Create sprint timing table with enhanced metrics
        table_data = []
        for i, row in micro_df.iterrows():
            # Get timing metrics
            power_ttp = row.get('power_ttp_s', 'N/A')
            torque_ttp = row.get('torque_ttp_s', 'N/A')
            cadence_ttp = row.get('cadence_ttp_s', 'N/A')
            
            # Get acceleration rates
            power_accel = row.get('power_acceleration_rate', 'N/A')
            
            # Get speed metrics
            speed_ttp = row.get('speed_ttp_s', 'N/A')
            speed_increase = row.get('speed_increase_kph', 'N/A')
            max_speed = row.get('max_speed_kph', 'N/A')
            sprint_type = row.get('sprint_type', 'N/A')
            
            # Get efficiency score
            efficiency_score = row.get('sprint_efficiency_score', 'N/A')
            
            # Convert speed units if needed
            if isinstance(speed_increase, (int, float)) and not pd.isna(speed_increase):
                if is_imperial:
                    speed_increase_text = f"{speed_increase * 0.621371:.1f} mph"
                else:
                    speed_increase_text = f"{speed_increase:.1f} km/h"
            else:
                speed_increase_text = speed_increase
            
            table_data.append(html.Tr([
                html.Td(f"Sprint {i+1}"),
                html.Td(f"{power_ttp:.2f}s" if isinstance(power_ttp, (int, float)) else power_ttp),
                html.Td(f"{torque_ttp:.2f}s" if isinstance(torque_ttp, (int, float)) else torque_ttp),
                html.Td(f"{cadence_ttp:.2f}s" if isinstance(cadence_ttp, (int, float)) else cadence_ttp),
                html.Td(f"{speed_ttp:.2f}s" if isinstance(speed_ttp, (int, float)) else speed_ttp),
                html.Td(f"{power_accel:.0f} W/s" if isinstance(power_accel, (int, float)) else power_accel),
                html.Td(speed_increase_text),
                html.Td(f"{efficiency_score}" if isinstance(efficiency_score, (int, float)) else efficiency_score)
            ]))
        
        # Update header based on units
        if is_imperial:
            speed_header = "Speed Increase (mph)"
        else:
            speed_header = "Speed Increase (km/h)"
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Sprint"),
                html.Th("Time to Max Power"),
                html.Th("Time to Max Torque"),
                html.Th("Time to Max Cadence"),
                html.Th("Time to Max Speed"),
                html.Th("Power Acceleration"),
                html.Th(speed_header),
                html.Th("Efficiency Score")
            ])),
            html.Tbody(table_data)
        ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
    
    # Callback to update interval table
    @app.callback(
        Output('interval-table', 'children'),
        [Input('workout-overview', 'clickData'),
         Input('unit-toggle', 'value')]
    )
    def update_interval_table(click_data, units):
        if long_intervals.empty:
            return html.P("No interval data available")
        
        is_imperial = units == 'imperial'
        
        # Create interval table
        table_data = []
        for i, row in long_intervals.iterrows():
            # Format duration: show seconds if < 1 min, otherwise show minutes
            duration_s = row['duration_s']
            if duration_s < 60:
                duration_str = f"{duration_s:.0f}s"
            else:
                duration_str = f"{duration_s/60:.1f} min"
            
            # Convert work units if needed
            work_kj = row['work_kj']
            if is_imperial:
                work_text = f"{work_kj * 0.737562:.0f} ft-lb"
            else:
                work_text = f"{work_kj:.1f} kJ"
            
            table_data.append(html.Tr([
                html.Td(f"#{row['rank']}"),
                html.Td(row['start_str']),
                html.Td(duration_str),
                html.Td(f"{row['avg_power']:.0f}W"),
                html.Td(f"{row['max_power']:.0f}W"),
                html.Td(work_text)
            ]))
        
        # Update header based on units
        if is_imperial:
            work_header = "Work (ft-lb)"
        else:
            work_header = "Work (kJ)"
        
        return html.Table([
            html.Thead(html.Tr([
                html.Td("Rank"),
                html.Td("Start Time"),
                html.Td("Duration"),
                html.Td("Avg Power (W)"),
                html.Td("Max Power (W)"),
                html.Td(work_header)
            ])),
            html.Tbody(table_data)
        ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
    
    # Callback to update comprehensive metrics table
    @app.callback(
        Output('comprehensive-metrics-table', 'children'),
        [Input('workout-overview', 'clickData'),
         Input('unit-toggle', 'value')]
    )
    def update_comprehensive_metrics_table(click_data, units):
        return create_metrics_table(units)
    
    # Callback to update timing comparison charts
    @app.callback(
        Output('timing-comparison', 'children'),
        [Input('workout-overview', 'clickData'),
         Input('unit-toggle', 'value')]
    )
    def update_timing_comparison_charts(click_data, units):
        return dcc.Graph(figure=create_timing_comparison_charts(units))
    
    # Callback to update interval comparison content
    @app.callback(
        Output('interval-comparison-content', 'children'),
        [Input('workout-overview', 'clickData'),
         Input('unit-toggle', 'value')]
    )
    def update_interval_comparison_content(click_data, units):
        if long_intervals.empty:
            return html.P("No interval data available for comparison")
        
        # Create comparison content
        comparison_content = []
        
        # 1. Duration-based comparison
        duration_comparison = compare_similar_intervals(long_intervals, 'duration')
        if not duration_comparison.empty:
            comparison_content.append(html.H4("Duration-Based Comparison", style={'color': '#27ae60'}))
            comparison_content.append(create_comparison_table_html(duration_comparison, 'Duration Groups'))
            comparison_content.append(html.Hr())
        
        # 2. Intensity-based comparison
        intensity_comparison = compare_similar_intervals(long_intervals, 'intensity')
        if not intensity_comparison.empty:
            comparison_content.append(html.H4("Intensity-Based Comparison", style={'color': '#e74c3c'}))
            comparison_content.append(create_comparison_table_html(intensity_comparison, 'Intensity Groups'))
            comparison_content.append(html.Hr())
        
        # 3. Zone-based comparison
        zone_comparison = compare_similar_intervals(long_intervals, 'zone')
        if not zone_comparison.empty:
            comparison_content.append(html.H4("Zone-Based Comparison", style={'color': '#9b59b6'}))
            comparison_content.append(create_comparison_table_html(zone_comparison, 'Zone Groups'))
            comparison_content.append(html.Hr())
        
        # 4. Source-based comparison
        source_comparison = compare_similar_intervals(long_intervals, 'source')
        if not source_comparison.empty:
            comparison_content.append(html.H4("Source-Based Comparison", style={'color': '#f39c12'}))
            comparison_content.append(create_comparison_table_html(source_comparison, 'Detection Sources'))
            comparison_content.append(html.Hr())
        
        # 5. Overall consistency summary
        comparison_content.append(html.H4("Overall Consistency Summary", style={'color': '#34495e'}))
        consistency_summary = create_consistency_summary_html(long_intervals)
        comparison_content.append(consistency_summary)
        
        return html.Div(comparison_content)
    
    # Callback to update interval comparison plots
    @app.callback(
        Output('interval-comparison-plots', 'children'),
        [Input('workout-overview', 'clickData'),
         Input('unit-toggle', 'value')]
    )
    def update_interval_comparison_plots(click_data, units):
        if long_intervals.empty:
            return html.P("No interval data available for comparison plots")
        
        # Create comparison plots
        comparison_plots = create_interval_comparison_plots(long_intervals)
        if not comparison_plots:
            return html.P("No comparison plots could be created (insufficient data)")
        
        # Create plot components in a grid layout
        plot_components = []
        plot_titles = [
            "Duration-Based Consistency",
            "Intensity-Based Consistency", 
            "Zone-Based Consistency",
            "Source-Based Consistency",
            "Overall Consistency Analysis"
        ]
        
        # Create plots in rows of 2
        for i in range(0, len(comparison_plots), 2):
            row_plots = []
            for j in range(2):
                if i + j < len(comparison_plots):
                    plot_idx = i + j
                    plot = comparison_plots[plot_idx]
                    title = plot_titles[plot_idx]
                    
                    row_plots.append(html.Div([
                        html.H5(title, style={'color': '#34495e', 'marginBottom': '10px', 'fontSize': '14px'}),
                        dcc.Graph(
                            id=f'comparison-plot-{plot_idx}',
                            figure=plot,
                            style={'height': '300px', 'marginBottom': '15px'}
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}))
            
            plot_components.append(html.Div(row_plots, style={'marginBottom': '20px'}))
        
        return html.Div(plot_components)
    
    # Callback to toggle metrics guide
    @app.callback(
        Output('metrics-guide-content', 'children'),
        Output('metrics-guide-content', 'style'),
        Input('toggle-metrics-guide', 'n_clicks'),
        State('metrics-guide-content', 'style')
    )
    def toggle_metrics_guide(n_clicks, current_style):
        if n_clicks is None:
            return [], {'display': 'none'}
        
        if current_style.get('display') == 'none':
            # Show the guide
            guide_content = [
                html.Div([
                    html.H2("Cycling Metrics Guide", 
                            style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
                    
                    # Power Metrics Section
                    html.Div([
                        html.H3("POWER METRICS", style={'color': '#e74c3c', 'borderBottom': '2px solid #e74c3c', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("Average Power"),
                            html.P("Your sustained power output throughout the ride. Key indicator of overall effort and fitness level."),
                            html.H4("Normalized Power (NP)"),
                            html.P("Weighted average power that accounts for the physiological cost of variable power output. More accurate than simple average for training stress."),
                            html.H4("Max Power"),
                            html.P("Peak power output. Useful for understanding your sprint capabilities and power ceiling."),
                            html.H4("Power-to-Weight Ratio (W/kg)"),
                            html.P("Power relative to body weight. Critical for climbing performance and comparing riders of different sizes."),
                            html.H4("Intensity Factor (IF)"),
                            html.P("Normalized Power divided by FTP. Values >1.0 indicate high-intensity efforts, >1.05 suggest race-like intensity."),
                            html.H4("Training Stress Score (TSS)"),
                            html.P("Cumulative training load measure. 100 TSS = 1 hour at FTP. Useful for tracking weekly/monthly training volume.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                    # Timing Metrics Section
                    html.Div([
                        html.H3("TIMING METRICS", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("Time to Max Power (TTP)"),
                            html.P("How quickly you reach peak power. Lower values indicate explosive sprint ability and neuromuscular efficiency."),
                            html.H4("Time to Max Torque"),
                            html.P("Speed of torque development. Important for sprint starts and acceleration from low speeds."),
                            html.H4("Time to Max Cadence"),
                            html.P("How fast you reach optimal pedaling rhythm. Reflects coordination and technique efficiency."),
                            html.H4("Time to Max Speed"),
                            html.P("Acceleration rate to top speed. Key for sprint performance and race tactics.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                    # Sprint Metrics Section
                    html.Div([
                        html.H3("SPRINT METRICS", style={'color': '#f39c12', 'borderBottom': '2px solid #f39c12', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("Power Acceleration Rate"),
                            html.P("Rate of power increase (W/s). Higher values indicate explosive sprint ability and fast-twitch fiber recruitment."),
                            html.H4("Speed Increase"),
                            html.P("Velocity gain during sprint. Important for understanding sprint effectiveness and aerodynamic efficiency."),
                            html.H4("Sprint Efficiency Score"),
                            html.P("Combined metric of power development, maintenance, and fade. Higher scores indicate better sprint technique."),
                            html.H4("Power Fade"),
                            html.P("Percentage drop in power during sprint. Lower fade suggests better anaerobic capacity and pacing."),
                            html.H4("Sprint Consistency (CV)"),
                            html.P("Coefficient of variation across multiple sprints. Lower values indicate more repeatable sprint performance.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                    # Interval Metrics Section
                    html.Div([
                        html.H3("INTERVAL METRICS", style={'color': '#27ae60', 'borderBottom': '2px solid #27ae60', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("Interval Power"),
                            html.P("Sustained power during structured efforts. Key for VO2 max and threshold training assessment."),
                            html.H4("Power Consistency"),
                            html.P("Standard deviation of power during intervals. Lower values indicate better pacing and endurance."),
                            html.H4("Power Fade"),
                            html.P("Power decline during intervals. Important for understanding fatigue resistance and training zone adherence."),
                            html.H4("Interval Work Percentage"),
                            html.P("Proportion of total ride spent in interval efforts. Useful for training load management."),
                            html.H4("Power CV (%)"),
                            html.P("Coefficient of variation in power output. Lower values indicate more consistent power delivery."),
                            html.H4("Normalized Power"),
                            html.P("30-second rolling average power for efforts ≥30s. More physiologically relevant than simple average."),
                            html.H4("Power Zone"),
                            html.P("Training zone classification based on FTP percentage. Helps categorize effort intensity."),
                            html.H4("HR Zone"),
                            html.P("Heart rate zone classification based on LTHR. Essential for understanding cardiovascular stress."),
                            html.H4("Sprint Efficiency"),
                            html.P("Performance score for short intervals (≤60s). Combines power development, maintenance, and technique."),
                            html.H4("Mechanical Efficiency (kJ/km)"),
                            html.P("Energy cost per distance gained. Lower values indicate more efficient power transfer to forward motion."),
                            html.H4("Power Consistency Score"),
                            html.P("Combined metric of power stability and fade resistance. Higher scores indicate better pacing."),
                            html.H4("Overall Performance Score"),
                            html.P("Comprehensive rating combining quality, power, and duration factors. Higher scores indicate better overall performance.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                    # Physiological Metrics Section
                    html.Div([
                        html.H3("PHYSIOLOGICAL METRICS", style={'color': '#e91e63', 'borderBottom': '2px solid #e91e63', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("Heart Rate Zones"),
                            html.P("Time spent in different HR zones based on LTHR. Essential for understanding training intensity distribution."),
                            html.H4("Cadence Analysis"),
                            html.P("Pedaling rhythm patterns. Optimal cadence varies by rider but typically 80-100 rpm for endurance."),
                            html.H4("Torque Analysis"),
                            html.P("Force application patterns. Important for understanding pedaling technique and efficiency."),
                            html.H4("Work (kJ)"),
                            html.P("Total energy output. Useful for comparing ride intensity and caloric expenditure.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                    # Enhanced Cadence & Speed Metrics Section
                    html.Div([
                        html.H3("ENHANCED CADENCE & SPEED METRICS", style={'color': '#16a085', 'borderBottom': '2px solid #16a085', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("Average Cadence"),
                            html.P("Mean pedaling rate during intervals. Key for understanding pedaling efficiency and technique."),
                            html.H4("Max Cadence"),
                            html.P("Peak pedaling rate. Important for sprint performance and neuromuscular coordination."),
                            html.H4("Min Cadence"),
                            html.P("Lowest pedaling rate. Can indicate fatigue or technique breakdown."),
                            html.H4("Cadence Drift"),
                            html.P("Change in cadence over time. Positive values suggest maintaining rhythm, negative values indicate fatigue."),
                            html.H4("Average Speed (km/h)"),
                            html.P("Mean forward velocity. Essential for understanding interval effectiveness and terrain impact."),
                            html.H4("Max Speed (km/h)"),
                            html.P("Peak velocity achieved. Important for sprint and acceleration analysis."),
                            html.H4("Speed Gain (km/h)"),
                            html.P("Velocity increase during interval. Key metric for acceleration and sprint performance."),
                            html.H4("Speed Acceleration Rate"),
                            html.P("Rate of speed increase (km/h per second). Higher values indicate better acceleration ability.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                    # Training Zones Section
                    html.Div([
                        html.H3("TRAINING ZONES", style={'color': '#9b59b6', 'borderBottom': '2px solid #9b59b6', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("Zone 1 (Active Recovery)"),
                            html.P("55-75% FTP. Promotes recovery and builds aerobic base."),
                            html.H4("Zone 2 (Endurance)"),
                            html.P("75-90% FTP. Builds aerobic capacity and fat utilization."),
                            html.H4("Zone 3 (Tempo)"),
                            html.P("90-105% FTP. Improves lactate threshold and sustainable power."),
                            html.H4("Zone 4 (Lactate Threshold)"),
                            html.P("105-120% FTP. Raises anaerobic threshold and race pace."),
                            html.H4("Zone 5 (VO2 Max)"),
                            html.P("120-150% FTP. Improves maximum oxygen consumption."),
                            html.H4("Zone 6 (Anaerobic)"),
                            html.P("150%+ FTP. Develops sprint power and anaerobic capacity.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                    # Power-to-Weight & Performance Metrics Section
                    html.Div([
                        html.H3("POWER-TO-WEIGHT & PERFORMANCE METRICS", style={'color': '#e67e22', 'borderBottom': '2px solid #e67e22', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("Power (W/kg)"),
                            html.P("Power output relative to body weight. Critical for climbing performance and comparing riders of different sizes."),
                            html.H4("NP (W/kg)"),
                            html.P("Normalized power per kilogram. More accurate than simple power-to-weight for training stress assessment."),
                            html.H4("Work per Minute (kJ/min)"),
                            html.P("Energy output rate. Useful for understanding interval intensity and training load."),
                            html.H4("Altitude Gain (m)"),
                            html.P("Vertical elevation change during interval. Important for understanding climbing performance and terrain impact."),
                            html.H4("Average Grade (%)"),
                            html.P("Mean slope gradient. Essential for interpreting power data in context of terrain difficulty."),
                            html.H4("Torque Stability"),
                            html.P("Consistency of force application. Higher values indicate more stable pedaling technique."),
                            html.H4("Quality Score"),
                            html.P("Overall interval quality rating. Combines power, consistency, duration, and technique factors.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                    # Zone Numbering System Section
                    html.Div([
                        html.H3("ZONE NUMBERING SYSTEM", style={'color': '#8e44ad', 'borderBottom': '2px solid #8e44ad', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("Power Zone Numbers (1-7)"),
                            html.P("1: Active Recovery (0-55% FTP), 2: Endurance (55-75% FTP), 3: Tempo (75-90% FTP), 4: Threshold (90-105% FTP), 5: VO2 Max (105-120% FTP), 6: Anaerobic (120-150% FTP), 7: Neuromuscular (150%+ FTP)"),
                            html.H4("HR Zone Numbers (1-5)"),
                            html.P("1: Active Recovery (0-85% LTHR), 2: Endurance (85-95% LTHR), 3: Tempo (95-105% LTHR), 4: Threshold (105-115% LTHR), 5: VO2 Max (115%+ LTHR)"),
                            html.H4("Zone Classification"),
                            html.P("Automatic categorization of intervals into appropriate training zones based on power and heart rate data.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                    html.Hr(style={'margin': '30px 0'}),
                    
                    # Scoring Systems Section
                    html.Div([
                        html.H3("SCORING SYSTEMS", style={'color': '#c0392b', 'borderBottom': '2px solid #c0392b', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("Sprint Efficiency Score (0-150)"),
                            html.P("100 = baseline performance. +20 for high power efficiency (>80% avg/max), +10 for very short duration (≤30s), +5 for short duration (≤45s). Penalties for excessive power fade."),
                            html.H4("Power Consistency Score (0-100)"),
                            html.P("100 = perfect consistency. Penalties for high power variation (CV) and power fade. Lower scores indicate less stable power output."),
                            html.H4("Overall Performance Score (0-150)"),
                            html.P("Combines quality score with power bonuses and duration bonuses. Higher scores indicate better overall interval performance."),
                            html.H4("Quality Score"),
                            html.P("Weighted combination of power, duration, consistency, fade, cadence, torque, and work factors. Higher scores indicate better interval quality.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                    html.Hr(style={'margin': '30px 0'}),
                    html.P("Tip: Use these metrics to track progress, identify weaknesses, and optimize your training program.", 
                           style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#666'}),
                    
                    # Data Sources & Detection Methods Section
                    html.Div([
                        html.H3("DATA SOURCES & DETECTION METHODS", style={'color': '#2c3e50', 'borderBottom': '2px solid #2c3e50', 'paddingBottom': '5px'}),
                        html.Div([
                            html.H4("ML-Enhanced Detection"),
                            html.P("Uses trained machine learning models to identify intervals with high accuracy and confidence scores."),
                            html.H4("Lap-Based Detection"),
                            html.P("Extracts intervals from manual lap markers in your cycling computer data."),
                            html.H4("Auto-Detection"),
                            html.P("Algorithmic detection using power thresholds, duration filters, and quality scoring."),
                            html.H4("Fallback Detection"),
                            html.P("Sensitive detection method when other methods find insufficient intervals."),
                            html.H4("Source Classification"),
                            html.P("Each interval is tagged with its detection method for transparency and analysis.")
                        ], style={'marginLeft': '20px', 'marginBottom': '20px'})
                    ]),
                    
                ], style={'backgroundColor': 'white', 'padding': '30px', 'borderRadius': '10px', 'border': '2px solid #ecf0f1'})
            ]
            return guide_content, {'display': 'block'}
        else:
            # Hide the guide
            return [], {'display': 'none'}
    

    
    # Run the app
    print("Starting Dash dashboard...")
    print("Open your browser and go to: http://127.0.0.1:8052")
    app.run(debug=False, host='127.0.0.1', port=8052)


def create_enhanced_interval_visualization(df, intervals_df, ftp=330):
    """Create enhanced interval visualization with power over time and overlaid intervals.
    
    Args:
        df: DataFrame with DateTime index and 'power' column
        intervals_df: DataFrame with interval data including start_time, end_time, duration_s, avg_power, max_power
        ftp: Functional Threshold Power in watts
    
    Returns:
        plotly.graph_objects.Figure object
    """
    import plotly.graph_objects as go
    
    # Create single figure for power over time with intervals
    fig = go.Figure()
    
    # 1. Main power plot with intervals
    # Power line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['power'],
            mode='lines',
            name='Power',
            line=dict(color='#1f77b4', width=1.5),
            opacity=0.8
        )
    )
    
    # FTP threshold line
    fig.add_hline(
        y=ftp,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"FTP: {ftp}W",
        annotation_position="top right"
    )
    
    # Color palette for intervals
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    # Add interval overlays
    for idx, interval in intervals_df.iterrows():
        # Get interval data
        start_time = interval['start_time']
        end_time = interval['end_time']
        duration = interval['duration_s']
        avg_power = interval['avg_power']
        max_power = interval['max_power']
        source = interval.get('source', 'unknown')
        quality_score = interval.get('quality_score', 0)
        
        # Color based on source
        if source == 'lap':
            color = colors[0]  # Orange for laps
        elif source == 'auto':
            color = colors[1]  # Green for auto-detected
        elif source == 'fallback':
            color = colors[3]  # Purple for fallback detection
        else:
            color = colors[2]  # Red for unknown
        
        # Add interval rectangle
        fig.add_vrect(
            x0=start_time,
            x1=end_time,
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line_width=0
        )
        
        # Add interval annotation
        fig.add_annotation(
            x=start_time + (end_time - start_time) / 2,
            y=avg_power + 50,  # Offset above the interval
            text=f"{duration:.0f}s<br>{avg_power:.0f}W<br>Q:{quality_score:.0f}",
            showarrow=False,
            font=dict(size=10, color='black'),
            bgcolor='white',
            bordercolor=color,
            borderwidth=1
        )
    

    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Enhanced Interval Analysis - {len(intervals_df)} Intervals Detected",
            x=0.5,
            font=dict(size=16, color='#2c3e50')
        ),
        height=600,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update x-axis
    fig.update_xaxes(
        title_text="Time",
        gridcolor='lightgray',
        gridwidth=0.5
    )
    
    # Update y-axis
    fig.update_yaxes(
        title_text="Power (W)",
        gridcolor='lightgray',
        gridwidth=0.5,
        range=[0, max(df['power'].max() * 1.1, ftp * 1.2)]
    )
    
    # Add legend for sources
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=colors[0]),
            name='Lap Intervals',
            showlegend=True
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=colors[1]),
            name='Auto-Detected',
            showlegend=True
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=colors[3]),
            name='Fallback Detection',
            showlegend=True
        )
    )
    
    return fig


def create_interval_summary_table(intervals_df):
    """Create a summary table of all detected intervals."""
    import plotly.graph_objects as go
    
    if intervals_df.empty:
        return go.Figure()
    
    # Prepare data for table
    table_data = []
    for idx, interval in intervals_df.iterrows():
        row = [
            f"{interval['start_str']}",
            f"{interval['duration_s']:.0f}s",
            f"{interval['avg_power']:.0f}W",
            f"{interval['max_power']:.0f}W",
            f"{interval['intensity_factor']:.2f}",
            interval['training_zone'],
            f"{interval['quality_score']:.1f}",
            interval.get('source', 'N/A'),
            interval.get('set_info', 'N/A') if pd.notna(interval.get('set_info')) else 'N/A'
        ]
        table_data.append(row)
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Start', 'Duration', 'Avg Power', 'Max Power', 'IF', 'Zone', 'Quality', 'Source', 'Set'],
            fill_color='#2c3e50',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=list(zip(*table_data)),
            fill_color='white',
            font=dict(size=11),
            align='center',
            height=30
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=f"Interval Summary - {len(intervals_df)} Intervals",
            x=0.5,
            font=dict(size=16, color='#2c3e50')
        ),
        height=400 + len(intervals_df) * 30,
        showlegend=False
    )
    
    return fig





# Main execution
if __name__ == "__main__":
    # Load data
    df = load_fit_to_dataframe(file_path)
    print("Loaded shape:", df.shape)
    print(df.head(3))
    
    # Process data
    df = apply_hard_limits_and_smooth(df)
    print("Columns now:", sorted(df.columns)[:12], "...")
    
    # Find best sprints
    best10 = best_10s_windows(df, n_best=3, win=10, min_gap=30)
    print("\n=== Best 10s Sprint Windows ===")
    # Display without start_ts column for clean output
    display_cols = ['Start', 'Power (W)', 'Torque (Nm)', 'Cadence (rpm)', 'ω (rad·s⁻¹)', 'Err %']
    print(best10[display_cols].round(2))
    
    # Extract micro-metrics for each sprint
    micro_metrics = []
    for _, row in best10.iterrows():
        start_ts = row['start_ts']
        if start_ts:
            metrics = sprint_micro_metrics(df, start_ts, win=10)
            if metrics:
                micro_metrics.append(metrics)
    
    micro_df = pd.DataFrame(micro_metrics)
    if not micro_df.empty:
        print("\n=== Sprint Micro-Metrics ===")
        print(micro_df.round(2))
    
    # Generate comprehensive sprint summary for each sprint
    sprint_summaries = []
    for _, row in best10.iterrows():
        start_ts = row['start_ts']
        if start_ts:
            sprint_summary_result = sprint_summary(df, start_ts, win=10)
            if sprint_summary_result:
                sprint_summary_result['start'] = row['Start']
                sprint_summaries.append(sprint_summary_result)
    
    if sprint_summaries:
        print("\n=== Sprint Summary ===")
        sprint_summary_df = pd.DataFrame(sprint_summaries)
        print(sprint_summary_df.round(2))
        
        # Display enhanced sprint analysis
        display_enhanced_sprint_metrics(micro_metrics)
        
        # Analyze sprint technique patterns across sprints
        analyze_sprint_technique_patterns(micro_metrics)
    
    # Find intervals using ML-based detection system
    print(f"\n🤖 ML-Based Interval Detection:")
    print(f"   FTP: {FTP}W")
    print(f"   Using trained machine learning model")
    
    long_intervals = detect_intervals_ml(df, ftp=FTP)

    # Ensure long_intervals is always a DataFrame
    if long_intervals is None:
        long_intervals = pd.DataFrame()

    print("\n=== ML-Based Interval Detection Results ===")
    if not long_intervals.empty:
        print(f"Found {len(long_intervals)} intervals using ML detection")
        print(f"Duration range: {long_intervals['duration_s'].min():.0f}s to {long_intervals['duration_s'].max():.0f}s")
        print(f"Sources: {long_intervals['source'].value_counts().to_dict()}")
        print(f"Quality scores: {long_intervals['quality_score'].min():.1f} - {long_intervals['quality_score'].max():.1f}")
        print(long_intervals.round(2))
        
        # Create enhanced interval visualization for Dash dashboard
        print("\n📊 Creating enhanced interval visualization for Dash...")
        interval_viz = create_enhanced_interval_visualization(df, long_intervals, ftp=FTP)
        interval_table = create_interval_summary_table(long_intervals)
        print("✅ Enhanced visualizations ready for Dash dashboard")
        
    else:
        print("No intervals detected with ML system.")
        interval_viz = None
        interval_table = None

    # Analyze interval evolution
    evolution_analysis = analyze_interval_evolution(df, long_intervals, n_best=max_intervals_to_analyze)
    if evolution_analysis:
        print("\n=== Interval Evolution Analysis ===")
        for interval_name, evolution in evolution_analysis.items():
            print(f"\n--- {interval_name} ---")
            print(f"Power Fade: {evolution['power_fade_%']:.1f}%")
            print(f"Power Consistency: {evolution['power_consistency_%']:.1f}%")
            print(f"Cadence Drift: {evolution['cadence_drift_rpm']:.1f} rpm")
            print(f"HR Drift: {evolution['hr_drift_bpm']:.1f} bpm")
            print(f"Torque Drift: {evolution['torque_drift_nm']:.1f} Nm")
            print(f"Speed Gain: {evolution['speed_gain_kph']:.1f} km/h")
    else:
        print("\nNo interval evolution data to analyze.")
    
    # Analyze interval comparisons and consistency
    if not long_intervals.empty:
        print("\n=== Interval Comparison Analysis ===")
        display_interval_comparison_analysis(long_intervals)
        
        # Create comparison plots for dashboard
        print("\n=== Creating Interval Comparison Plots for Dashboard ===")
        comparison_plots = create_interval_comparison_plots(long_intervals)
        if comparison_plots:
            print(f"Created {len(comparison_plots)} comparison plots for dashboard integration")
            print("   Plots will be displayed in the Interval Comparison Analysis section")
        else:
            print("No comparison plots could be created (insufficient data)")
        
        # Demonstrate specific comparisons
        print("\n=== Specific Interval Comparisons ===")
        
        # Compare intervals by duration
        print("\n1. Comparing intervals by duration:")
        duration_comparison = compare_similar_intervals(long_intervals, 'duration')
        if not duration_comparison.empty:
            print("   Duration-based groups found:")
            for _, group in duration_comparison.iterrows():
                print(f"   • {group['group_name']}: {group['interval_count']} intervals")
                if 'avg_power_cv' in group:
                    print(f"     Power CV: {group['avg_power_cv']:.1f}%")
                if 'duration_cv' in group:
                    print(f"     Duration CV: {group['duration_cv']:.1f}%")
        else:
            print("   No duration-based groups with sufficient intervals")
        
        # Compare intervals by intensity
        print("\n2. Comparing intervals by intensity:")
        intensity_comparison = compare_similar_intervals(long_intervals, 'intensity')
        if not intensity_comparison.empty:
            print("   Intensity-based groups found:")
            for _, group in intensity_comparison.iterrows():
                print(f"   • {group['group_name']}: {group['interval_count']} intervals")
                if 'avg_power_cv' in group:
                    print(f"     Power CV: {group['avg_power_cv']:.1f}%")
                if 'quality_score_cv' in group:
                    print(f"     Quality Score CV: {group['quality_score_cv']:.1f}%")
        else:
            print("   No intensity-based groups with sufficient intervals")
        
        # Compare intervals by power zone
        print("\n3. Comparing intervals by power zone:")
        zone_comparison = compare_similar_intervals(long_intervals, 'zone')
        if not zone_comparison.empty:
            print("   Zone-based groups found:")
            for _, group in zone_comparison.iterrows():
                print(f"   • {group['group_name']}: {group['interval_count']} intervals")
                if 'avg_power_cv' in group:
                    print(f"     Power CV: {group['avg_power_cv']:.1f}%")
                if 'avg_cadence_cv' in group:
                    print(f"     Cadence CV: {group['avg_cadence_cv']:.1f}%")
        else:
            print("   No zone-based groups with sufficient intervals")
        
        # Compare intervals by detection source
        print("\n4. Comparing intervals by detection source:")
        source_comparison = compare_similar_intervals(long_intervals, 'source')
        if not source_comparison.empty:
            print("   Source-based groups found:")
            for _, group in source_comparison.iterrows():
                print(f"   • {group['group_name']}: {group['interval_count']} intervals")
                if 'avg_power_cv' in group:
                    print(f"     Power CV: {group['avg_power_cv']:.1f}%")
                if 'quality_score_cv' in group:
                    print(f"     Quality Score CV: {group['quality_score_cv']:.1f}%")
        else:
            print("   No source-based groups with sufficient intervals")
    else:
        print("\nNo intervals available for comparison analysis.")

    # Create workout overview graph (for HTML export)
    create_workout_overview_graph(df)

    print("\n" + "="*80)
    print("LAUNCHING DASH DASHBOARD")
    print("="*80)
    print("All analysis complete! Launching interactive dashboard...")
    print("The dashboard will open in your browser automatically.")
    print("If it doesn't open, go to: http://127.0.0.1:8050")
    print("="*80)

    # Create Dash dashboard
    create_dash_dashboard(df, best10, micro_df, sprint_summary_df, long_intervals, evolution_analysis, interval_viz, interval_table)

