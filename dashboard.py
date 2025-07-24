#!/usr/bin/env python3
"""
Enhanced Cycling Analysis Dashboard
A comprehensive Streamlit-based dashboard for cycling data analysis.

This module provides a web interface for:
- Uploading and analyzing FIT files
- Viewing historical data and trends
- Managing analysis settings
- System monitoring and data management

Author: Cycling Analysis Team
Version: 1.0.0
"""

# Standard library imports
import os
import platform
from datetime import datetime
from typing import Optional, Dict, Any

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# Local imports
from data_manager import CyclingDataManager
from analyzer import CyclingAnalyzer

# Page configuration
st.set_page_config(
    page_title="Cycling Analysis",
    layout="wide",
    page_icon="🚴",
    initial_sidebar_state="expanded"
)

# Initialize data manager with caching
@st.cache_resource
def get_data_manager() -> Optional[CyclingDataManager]:
    """
    Initialize and cache the data manager.
    
    Returns:
        Optional[CyclingDataManager]: Initialized data manager or None if failed
    """
    try:
        dm = CyclingDataManager()
        # Test if the object has the required methods
        if hasattr(dm, 'load_settings') and hasattr(dm, 'get_system_status'):
            return dm
        else:
            st.error("Data manager initialized but missing required methods")
            return None
    except Exception as e:
        st.error(f"Error initializing data manager: {e}")
        return None

# Initialize data manager
data_manager = get_data_manager()

# Helper functions for formatting
def format_number(value: Any, decimals: int = 1) -> str:
    """
    Format a number with proper handling of NaN values.
    
    Args:
        value: The value to format
        decimals: Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:.{decimals}f}"

def format_duration(minutes: float) -> str:
    """
    Format duration in minutes to human-readable format.
    
    Args:
        minutes: Duration in minutes
        
    Returns:
        str: Formatted duration string
    """
    if pd.isna(minutes):
        return "N/A"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"

def format_power(power: float) -> str:
    """
    Format power values with proper units.
    
    Args:
        power: Power value in watts
        
    Returns:
        str: Formatted power string
    """
    if pd.isna(power):
        return "N/A"
    return f"{int(power)}W"

def format_distance(km: float) -> str:
    """
    Format distance values with proper units.
    
    Args:
        km: Distance in kilometers
        
    Returns:
        str: Formatted distance string
    """
    if pd.isna(km):
        return "N/A"
    return f"{km:.1f}km"

def display_sidebar_stats() -> None:
    """Display sidebar statistics and settings information."""
    with st.sidebar:
        st.header("📊 Quick Stats")
        
        # Quick stats
        try:
            status = data_manager.get_system_status()
            st.metric("Activities", status['total_rides'])
            st.metric("Analyses", status['analysis_entries'])
        except Exception as e:
            st.error(f"Error loading system status: {e}")
            st.metric("Activities", 0)
            st.metric("Analyses", 0)
        
        st.markdown("---")
        
        # Settings info
        try:
            if hasattr(data_manager, 'load_settings'):
                settings = data_manager.load_settings()
                st.markdown("**⚙️ Current Settings**")
                st.markdown(f"• FTP: {settings.get('ftp', 250)}W")
                st.markdown(f"• LTHR: {settings.get('lthr', 160)}bpm")
                st.markdown(f"• Athlete: {settings.get('athlete_name', 'Cyclist')}")
            else:
                st.markdown("**⚙️ Current Settings**")
                st.markdown("• FTP: 250W (default)")
                st.markdown("• LTHR: 160bpm (default)")
                st.markdown("• Athlete: Cyclist (default)")
                st.error("Data manager missing load_settings method")
        except Exception as e:
            st.error(f"Error loading settings: {e}")
        
        # Quick cache management
        st.markdown("---")
        st.markdown("**🗂️ Quick Cache Management**")
        
        # Show cache info
        try:
            cache_info = data_manager.get_cache_info()
            if cache_info:
                total_size = cache_info.get('fit_files_size_mb', 0)
                total_files = cache_info.get('fit_files_count', 0)
                
                if total_files > 0:
                    st.markdown(f"📁 **Cache:** {total_files} files ({total_size:.1f} MB)")
                    
                    # Quick clear button
                    if st.button("🗑️ Clear Cache", key="sidebar_clear_cache", help="Clear all cached files to free space"):
                        with st.spinner("Clearing cache..."):
                            success, message = data_manager.clear_cache("all")
                            if success:
                                st.success("✅ Cache cleared!")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                else:
                    st.markdown("📁 **Cache:** Empty")
            else:
                st.markdown("📁 **Cache:** Unknown")
        except Exception as e:
            st.error(f"Cache error: {e}")

def main() -> None:
    """
    Main dashboard function that orchestrates the entire application.
    
    This function:
    1. Displays the main title
    2. Validates data manager initialization
    3. Shows sidebar statistics
    4. Creates tabbed interface for different sections
    """
    st.title("🚴 Cycling Analysis")
    
    # Check if data manager is properly initialized
    if data_manager is None:
        st.error("❌ Failed to initialize data manager. Please refresh the page.")
        return
    
    # Display sidebar
    display_sidebar_stats()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analysis", "📊 Totals & History", "⚙️ Settings", "🛠️ System"])
    
    # Tab 1: Analysis
    with tab1:
        show_analysis_tab()
    
    # Tab 2: Totals & History
    with tab2:
        show_totals_history_tab()
    
    # Tab 3: Settings
    with tab3:
        show_settings_tab()
    
    # Tab 4: System
    with tab4:
        show_system_tab()

def show_totals_history_tab():
    """Display comprehensive totals and historical data section."""
    st.header("📊 Totals & Historical Data")
    
    try:
        # Load ride history data
        if hasattr(data_manager, 'ride_history') and not data_manager.ride_history.empty:
            df = data_manager.ride_history.copy()
            
            # Convert date column to datetime
            if 'analysis_date' in df.columns:
                df['date'] = pd.to_datetime(df['analysis_date'], format='%Y%m%d_%H%M%S')
                df = df.sort_values('date', ascending=False)
            
            # Calculate comprehensive totals
            total_rides = len(df)
            total_distance = df['distance_km'].sum() if 'distance_km' in df.columns else 0
            total_duration = df['duration_min'].sum() if 'duration_min' in df.columns else 0
            total_kj = df['total_kj'].sum() if 'total_kj' in df.columns else 0
            total_elevation = df['total_elevation_m'].sum() if 'total_elevation_m' in df.columns else 0
            total_tss = df['TSS'].sum() if 'TSS' in df.columns else 0
            
            # Calculate averages
            avg_distance = total_distance / total_rides if total_rides > 0 else 0
            avg_duration = total_duration / total_rides if total_rides > 0 else 0
            avg_kj = total_kj / total_rides if total_rides > 0 else 0
            avg_elevation = total_elevation / total_rides if total_rides > 0 else 0
            avg_tss = total_tss / total_rides if total_rides > 0 else 0
            
            # Find records
            longest_ride = df['duration_min'].max() if 'duration_min' in df.columns else 0
            longest_ride_name = df.loc[df['duration_min'].idxmax(), 'ride_id'] if 'duration_min' in df.columns and not df.empty else "N/A"
            
            longest_distance = df['distance_km'].max() if 'distance_km' in df.columns else 0
            longest_distance_name = df.loc[df['distance_km'].idxmax(), 'ride_id'] if 'distance_km' in df.columns and not df.empty else "N/A"
            
            most_elevation = df['total_elevation_m'].max() if 'total_elevation_m' in df.columns else 0
            most_elevation_name = df.loc[df['total_elevation_m'].idxmax(), 'ride_id'] if 'total_elevation_m' in df.columns and not df.empty else "N/A"
            
            # Power records
            max_power_record = df['max_power_W'].max() if 'max_power_W' in df.columns else 0
            max_avg_power_record = df['avg_power_W'].max() if 'avg_power_W' in df.columns else 0
            max_np_record = df['NP_W'].max() if 'NP_W' in df.columns else 0
            
            # Heart rate records
            max_hr_record = df['max_hr'].max() if 'max_hr' in df.columns else 0
            avg_hr_all = df['avg_hr'].mean() if 'avg_hr' in df.columns else 0
            
            # Speed records
            max_speed_record = df['max_speed_kmh'].max() if 'max_speed_kmh' in df.columns else 0
            avg_speed_all = df['avg_speed_kmh'].mean() if 'avg_speed_kmh' in df.columns else 0
            
            # Cadence records
            max_cadence_record = df['max_cadence'].max() if 'max_cadence' in df.columns else 0
            avg_cadence_all = df['avg_cadence'].mean() if 'avg_cadence' in df.columns else 0
            
            # Calculate estimated calories (4.2 kJ = 1 kcal)
            total_calories = total_kj / 4.2 if total_kj > 0 else 0
            
            # Calculate time periods
            if 'date' in df.columns and len(df) > 1:
                date_range = (df['date'].max() - df['date'].min()).days
                rides_per_week = (total_rides / (date_range / 7)) if date_range > 0 else 0
                rides_per_month = (total_rides / (date_range / 30)) if date_range > 0 else 0
            else:
                rides_per_week = 0
                rides_per_month = 0
            
            # Display comprehensive summary
            st.subheader("📈 Overall Summary")
            
            # Main metrics in a grid
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rides", f"{total_rides:,}")
                st.metric("Total Distance", f"{total_distance:.1f} km")
                st.metric("Total Duration", format_duration(total_duration))
                st.metric("Total kJ", f"{total_kj:,.0f}")
            
            with col2:
                st.metric("Total Elevation", f"{total_elevation:,.0f} m")
                st.metric("Total TSS", f"{total_tss:.0f}")
                st.metric("Total Calories", f"{total_calories:,.0f}")
                st.metric("Rides/Week", f"{rides_per_week:.1f}")
            
            with col3:
                st.metric("Avg Distance", f"{avg_distance:.1f} km")
                st.metric("Avg Duration", format_duration(avg_duration))
                st.metric("Avg kJ", f"{avg_kj:.0f}")
                st.metric("Avg TSS", f"{avg_tss:.0f}")
            
            with col4:
                st.metric("Avg Elevation", f"{avg_elevation:.0f} m")
                st.metric("Longest Ride", format_duration(longest_ride))
                st.caption(f"Ride: {longest_ride_name}")
                st.metric("Longest Distance", f"{longest_distance:.1f} km")
                st.caption(f"Ride: {longest_distance_name}")
            
            st.markdown("---")
            
            # Records section
            st.subheader("🏆 Records & Achievements")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**💪 Power Records**")
                st.metric("Max Power (1s)", f"{max_power_record:.0f}W")
                st.metric("Max Avg Power", f"{max_avg_power_record:.0f}W")
                st.metric("Max Normalized Power", f"{max_np_record:.0f}W")
            
            with col2:
                st.markdown("**❤️ Heart Rate Records**")
                st.metric("Max HR", f"{max_hr_record:.0f} bpm")
                st.metric("Avg HR (All Rides)", f"{avg_hr_all:.0f} bpm")
                st.metric("Most Elevation", f"{most_elevation:.0f} m")
                st.caption(f"Ride: {most_elevation_name}")
            
            with col3:
                st.markdown("**🏃 Speed & Cadence**")
                st.metric("Max Speed", f"{max_speed_record:.1f} km/h")
                st.metric("Avg Speed", f"{avg_speed_all:.1f} km/h")
                st.metric("Max Cadence", f"{max_cadence_record:.0f} rpm")
                st.metric("Avg Cadence", f"{avg_cadence_all:.0f} rpm")
            
            st.markdown("---")
            
            # Power Bests by Interval
            st.subheader("⚡ Power Bests by Interval")
            
            # Define power best intervals to check
            power_best_intervals = ['1s', '5s', '10s', '30s', '1min', '3min', '5min', '8min', '10min', '12min', '20min', '60min', '90min']
            
            # Find power best columns in the data
            power_best_columns = [col for col in df.columns if col.startswith('power_best_')]
            
            if power_best_columns:
                # Create a grid layout for power bests
                cols = st.columns(4)
                col_idx = 0
                
                for interval in power_best_intervals:
                    col_name = f"power_best_{interval}"
                    if col_name in df.columns:
                        max_power = df[col_name].max()
                        if not pd.isna(max_power) and max_power > 0:
                            # Find which ride had this record
                            max_ride = df.loc[df[col_name].idxmax(), 'ride_id'] if not df.empty else "N/A"
                            
                            with cols[col_idx]:
                                st.metric(f"{interval} Best", f"{max_power:.0f}W")
                                st.caption(f"Ride: {max_ride}")
                            
                            col_idx = (col_idx + 1) % 4
            else:
                st.info("No power best data available. Run analysis on rides to see power bests.")
            
            st.markdown("---")
            
            # Historical Trends
            st.subheader("📈 Historical Trends")
            
            if 'date' in df.columns and len(df) > 1:
                # Calculate average kJ per km for each ride
                if 'total_kj' in df.columns and 'distance_km' in df.columns:
                    df['kj_per_km'] = df['total_kj'] / df['distance_km']
                    df['kj_per_km'] = df['kj_per_km'].fillna(0)
                
                # Time series data with enhanced metrics
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle('Historical Performance Trends', fontsize=16, fontweight='bold')
                
                # Distance over time
                axes[0, 0].plot(df['date'], df['distance_km'], marker='o', linewidth=2, markersize=4, color='blue')
                axes[0, 0].set_title('Distance per Ride')
                axes[0, 0].set_ylabel('Distance (km)')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Average kJ per km over time (NEW)
                if 'kj_per_km' in df.columns:
                    axes[0, 1].plot(df['date'], df['kj_per_km'], marker='o', linewidth=2, markersize=4, color='red')
                    axes[0, 1].set_title('Average kJ per km')
                    axes[0, 1].set_ylabel('kJ/km')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Add trend line
                    if len(df) > 2:
                        z = np.polyfit(range(len(df)), df['kj_per_km'], 1)
                        p = np.poly1d(z)
                        axes[0, 1].plot(df['date'], p(range(len(df))), "--", alpha=0.7, color='red', linewidth=1)
                else:
                    # Duration over time (fallback)
                    axes[0, 1].plot(df['date'], df['duration_min'], marker='o', linewidth=2, markersize=4, color='orange')
                    axes[0, 1].set_title('Duration per Ride')
                    axes[0, 1].set_ylabel('Duration (min)')
                    axes[0, 1].grid(True, alpha=0.3)
                
                # TSS over time
                if 'TSS' in df.columns:
                    axes[1, 0].plot(df['date'], df['TSS'], marker='o', linewidth=2, markersize=4, color='green')
                    axes[1, 0].set_title('Training Stress Score')
                    axes[1, 0].set_ylabel('TSS')
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    axes[1, 0].text(0.5, 0.5, 'No TSS data available', ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Training Stress Score')
                
                # kJ over time
                if 'total_kj' in df.columns:
                    axes[1, 1].plot(df['date'], df['total_kj'], marker='o', linewidth=2, markersize=4, color='purple')
                    axes[1, 1].set_title('Total kJ per Ride')
                    axes[1, 1].set_ylabel('kJ')
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No kJ data available', ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Total kJ per Ride')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Enhanced trend analysis table
                if len(df) > 1:
                    st.subheader("📊 Trend Analysis")
                    
                    # Calculate trend statistics
                    trend_data = []
                    
                    if 'distance_km' in df.columns:
                        recent_avg = df.head(5)['distance_km'].mean() if len(df) >= 5 else df['distance_km'].mean()
                        overall_avg = df['distance_km'].mean()
                        trend_data.append({
                            'Metric': 'Distance (km)',
                            'Recent Avg': f"{recent_avg:.1f}",
                            'Overall Avg': f"{overall_avg:.1f}",
                            'Trend': '↗️ Improving' if recent_avg > overall_avg else '↘️ Declining' if recent_avg < overall_avg else '➡️ Stable'
                        })
                    
                    if 'kj_per_km' in df.columns:
                        recent_avg = df.head(5)['kj_per_km'].mean() if len(df) >= 5 else df['kj_per_km'].mean()
                        overall_avg = df['kj_per_km'].mean()
                        trend_data.append({
                            'Metric': 'kJ per km',
                            'Recent Avg': f"{recent_avg:.1f}",
                            'Overall Avg': f"{overall_avg:.1f}",
                            'Trend': '↗️ Improving' if recent_avg > overall_avg else '↘️ Declining' if recent_avg < overall_avg else '➡️ Stable'
                        })
                    
                    if 'TSS' in df.columns:
                        recent_avg = df.head(5)['TSS'].mean() if len(df) >= 5 else df['TSS'].mean()
                        overall_avg = df['TSS'].mean()
                        trend_data.append({
                            'Metric': 'TSS',
                            'Recent Avg': f"{recent_avg:.0f}",
                            'Overall Avg': f"{overall_avg:.0f}",
                            'Trend': '↗️ Improving' if recent_avg > overall_avg else '↘️ Declining' if recent_avg < overall_avg else '➡️ Stable'
                        })
                    
                    if 'total_kj' in df.columns:
                        recent_avg = df.head(5)['total_kj'].mean() if len(df) >= 5 else df['total_kj'].mean()
                        overall_avg = df['total_kj'].mean()
                        trend_data.append({
                            'Metric': 'Total kJ',
                            'Recent Avg': f"{recent_avg:.0f}",
                            'Overall Avg': f"{overall_avg:.0f}",
                            'Trend': '↗️ Improving' if recent_avg > overall_avg else '↘️ Declining' if recent_avg < overall_avg else '➡️ Stable'
                        })
                    
                    if trend_data:
                        trend_df = pd.DataFrame(trend_data)
                        st.dataframe(trend_df, use_container_width=True)
            
            st.markdown("---")
            
            # Performance Distribution
            st.subheader("📊 Performance Distribution")
            
            if len(df) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distance distribution
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(df['distance_km'], bins=min(10, len(df)//2), alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_title('Distance Distribution')
                    ax.set_xlabel('Distance (km)')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Duration distribution
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(df['duration_min'], bins=min(10, len(df)//2), alpha=0.7, color='lightcoral', edgecolor='black')
                    ax.set_title('Duration Distribution')
                    ax.set_xlabel('Duration (min)')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
            
            st.markdown("---")
            
            # Ride Calendar Heatmap
            st.subheader("📅 Ride Calendar")
            
            if 'date' in df.columns and len(df) > 1:
                # Create a calendar heatmap
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                df['weekday'] = df['date'].dt.dayofweek
                
                # Count rides per day
                daily_rides = df.groupby(['year', 'month', 'day']).size().reset_index(name='rides')
                
                # Create a pivot table for the heatmap
                pivot_data = daily_rides.pivot_table(
                    index='month', 
                    columns='day', 
                    values='rides', 
                    fill_value=0
                )
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(pivot_data, annot=True, fmt='g', cmap='YlOrRd', ax=ax)
                ax.set_title('Rides per Day (Heatmap)')
                ax.set_xlabel('Day of Month')
                ax.set_ylabel('Month')
                st.pyplot(fig)
                plt.close()
            
            st.markdown("---")
            
            # Enhanced Historical Data Table
            st.subheader("📋 Historical Data Table")
            
            # Add comprehensive filters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Date range filter
                if 'date' in df.columns and len(df) > 1:
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    date_range = st.date_input(
                        "Date Range",
                        value=(min_date.date(), max_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date()
                    )
                else:
                    date_range = None
            
            with col2:
                # Distance filter
                if 'distance_km' in df.columns:
                    min_distance = df['distance_km'].min()
                    max_distance = df['distance_km'].max()
                    distance_range = st.slider(
                        "Distance Range (km)",
                        min_value=float(min_distance),
                        max_value=float(max_distance),
                        value=(float(min_distance), float(max_distance))
                    )
                else:
                    distance_range = None
            
            with col3:
                # Duration filter
                if 'duration_min' in df.columns:
                    min_duration = df['duration_min'].min()
                    max_duration = df['duration_min'].max()
                    duration_range = st.slider(
                        "Duration Range (min)",
                        min_value=float(min_duration),
                        max_value=float(max_duration),
                        value=(float(min_duration), float(max_duration))
                    )
                else:
                    duration_range = None
            
            with col4:
                # kJ per km filter
                if 'kj_per_km' in df.columns:
                    min_kj_km = df['kj_per_km'].min()
                    max_kj_km = df['kj_per_km'].max()
                    kj_km_range = st.slider(
                        "kJ/km Range",
                        min_value=float(min_kj_km),
                        max_value=float(max_kj_km),
                        value=(float(min_kj_km), float(max_kj_km))
                    )
                else:
                    kj_km_range = None
            
            # Apply filters
            filtered_df = df.copy()
            
            if date_range and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df['date'].dt.date >= start_date) & 
                    (filtered_df['date'].dt.date <= end_date)
                ]
            
            if distance_range:
                filtered_df = filtered_df[
                    (filtered_df['distance_km'] >= distance_range[0]) & 
                    (filtered_df['distance_km'] <= distance_range[1])
                ]
            
            if duration_range:
                filtered_df = filtered_df[
                    (filtered_df['duration_min'] >= duration_range[0]) & 
                    (filtered_df['duration_min'] <= duration_range[1])
                ]
            
            if kj_km_range:
                filtered_df = filtered_df[
                    (filtered_df['kj_per_km'] >= kj_km_range[0]) & 
                    (filtered_df['kj_per_km'] <= kj_km_range[1])
                ]
            
            # Display filtered results with enhanced info
            st.markdown(f"**Showing {len(filtered_df)} of {len(df)} rides**")
            
            # Enhanced table display with better formatting
            if not filtered_df.empty:
                # Prepare display data
                display_df = filtered_df.copy()
                
                # Format date for display
                if 'date' in display_df.columns:
                    display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                
                # Select and rename columns for better display
                display_columns = []
                column_mapping = {}
                
                if 'ride_id' in display_df.columns:
                    display_columns.append('ride_id')
                    column_mapping['ride_id'] = 'Ride ID'
                
                if 'Date' in display_df.columns:
                    display_columns.append('Date')
                    column_mapping['Date'] = 'Date'
                
                if 'distance_km' in display_df.columns:
                    display_columns.append('distance_km')
                    column_mapping['distance_km'] = 'Distance (km)'
                
                if 'duration_min' in display_df.columns:
                    display_columns.append('duration_min')
                    column_mapping['duration_min'] = 'Duration (min)'
                
                if 'kj_per_km' in display_df.columns:
                    display_columns.append('kj_per_km')
                    column_mapping['kj_per_km'] = 'kJ/km'
                
                if 'total_kj' in display_df.columns:
                    display_columns.append('total_kj')
                    column_mapping['total_kj'] = 'Total kJ'
                
                if 'avg_power_W' in display_df.columns:
                    display_columns.append('avg_power_W')
                    column_mapping['avg_power_W'] = 'Avg Power (W)'
                
                if 'max_power_W' in display_df.columns:
                    display_columns.append('max_power_W')
                    column_mapping['max_power_W'] = 'Max Power (W)'
                
                if 'TSS' in display_df.columns:
                    display_columns.append('TSS')
                    column_mapping['TSS'] = 'TSS'
                
                if 'total_elevation_m' in display_df.columns:
                    display_columns.append('total_elevation_m')
                    column_mapping['total_elevation_m'] = 'Elevation (m)'
                
                # Create final display dataframe
                final_display_df = display_df[display_columns].copy()
                final_display_df.columns = [column_mapping[col] for col in display_columns]
                
                # Format numeric columns
                for col in final_display_df.columns:
                    if 'Distance' in col or 'Duration' in col or 'Elevation' in col:
                        final_display_df[col] = final_display_df[col].round(1)
                    elif 'Power' in col or 'TSS' in col:
                        final_display_df[col] = final_display_df[col].round(0)
                    elif 'kJ' in col:
                        final_display_df[col] = final_display_df[col].round(0)
                
                # Display the table
                st.dataframe(final_display_df, use_container_width=True)
            
            # Enhanced export functionality
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export to CSV"):
                    # Create export data
                    export_df = filtered_df.copy()
                    if 'date' in export_df.columns:
                        export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
                    
                    # Generate CSV
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"cycling_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📈 Export Summary"):
                    # Create enhanced summary statistics
                    summary_data = {
                        'Metric': [
                            'Total Rides', 'Total Distance (km)', 'Total Duration (min)', 
                            'Total kJ', 'Total Elevation (m)', 'Total TSS',
                            'Avg Distance (km)', 'Avg Duration (min)', 'Avg kJ',
                            'Avg kJ/km', 'Max Power (W)', 'Max Speed (km/h)', 'Max HR (bpm)'
                        ],
                        'Value': [
                            len(filtered_df),
                            filtered_df['distance_km'].sum() if 'distance_km' in filtered_df.columns else 0,
                            filtered_df['duration_min'].sum() if 'duration_min' in filtered_df.columns else 0,
                            filtered_df['total_kj'].sum() if 'total_kj' in filtered_df.columns else 0,
                            filtered_df['total_elevation_m'].sum() if 'total_elevation_m' in filtered_df.columns else 0,
                            filtered_df['TSS'].sum() if 'TSS' in filtered_df.columns else 0,
                            filtered_df['distance_km'].mean() if 'distance_km' in filtered_df.columns else 0,
                            filtered_df['duration_min'].mean() if 'duration_min' in filtered_df.columns else 0,
                            filtered_df['total_kj'].mean() if 'total_kj' in filtered_df.columns else 0,
                            filtered_df['kj_per_km'].mean() if 'kj_per_km' in filtered_df.columns else 0,
                            filtered_df['max_power_W'].max() if 'max_power_W' in filtered_df.columns else 0,
                            filtered_df['max_speed_kmh'].max() if 'max_speed_kmh' in filtered_df.columns else 0,
                            filtered_df['max_hr'].max() if 'max_hr' in filtered_df.columns else 0
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Summary",
                        data=csv,
                        file_name=f"cycling_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            st.markdown("---")
            
            # Display filtered data table
            if not filtered_df.empty:
                # Select relevant columns for display
                display_cols = ['ride_id', 'date', 'distance_km', 'duration_min', 'avg_power_W', 'max_power_W', 'NP_W', 'TSS', 'total_kj', 'total_elevation_m', 'avg_hr', 'max_hr', 'avg_speed_kmh', 'max_speed_kmh', 'avg_cadence', 'max_cadence']
                available_cols = [col for col in display_cols if col in filtered_df.columns]
                
                if available_cols:
                    display_df = filtered_df[available_cols].copy()
                    
                    # Format the display
                    if 'date' in display_df.columns:
                        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    
                    # Rename columns for better display
                    column_mapping = {
                        'ride_id': 'Ride',
                        'date': 'Date',
                        'distance_km': 'Distance (km)',
                        'duration_min': 'Duration (min)',
                        'avg_power_W': 'Avg Power (W)',
                        'max_power_W': 'Max Power (W)',
                        'NP_W': 'NP (W)',
                        'TSS': 'TSS',
                        'total_kj': 'kJ',
                        'total_elevation_m': 'Elevation (m)',
                        'avg_hr': 'Avg HR (bpm)',
                        'max_hr': 'Max HR (bpm)',
                        'avg_speed_kmh': 'Avg Speed (km/h)',
                        'max_speed_kmh': 'Max Speed (km/h)',
                        'avg_cadence': 'Avg Cadence (rpm)',
                        'max_cadence': 'Max Cadence (rpm)'
                    }
                    
                    display_df = display_df.rename(columns=column_mapping)
                    
                    # Format numeric columns
                    for col in display_df.columns:
                        if 'Power' in col or 'TSS' in col or 'kJ' in col or 'HR' in col or 'Cadence' in col:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
                        elif 'Distance' in col or 'Speed' in col:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                        elif 'Duration' in col:
                            display_df[col] = display_df[col].apply(lambda x: format_duration(x) if pd.notna(x) else "N/A")
                        elif 'Elevation' in col:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No ride data available for display.")
            else:
                st.info("No rides found matching the selected filters.")
        
        else:
            st.info("No ride history data available. Upload some FIT files to see your totals and historical data.")
    
    except Exception as e:
        st.error(f"Error loading totals and historical data: {e}")
        st.info("Please ensure you have uploaded FIT files and run analyses to see historical data.")

def show_analysis_tab():
    st.header("📊 Analysis")
    
    # File upload section
    st.subheader("📁 Upload Activity")
    
    uploaded_file = st.file_uploader(
        "Choose a .fit file", 
        type=["fit"], 
        help="Upload your cycling activity file"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
    
        with col1:
            st.metric("File", uploaded_file.name)
        with col2:
            st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
        
        if st.button("📤 Upload", type="primary"):
            with st.spinner("Uploading..."):
                success, message, file_path = data_manager.upload_fit_file(uploaded_file)
                if success:
                    st.success("✅ Uploaded successfully!")
                else:
                    st.error(message)
    
    st.markdown("---")

    # Show uploaded files
    st.subheader("📋 Your Activities")
    
    available_rides = data_manager.get_available_rides()
    if available_rides:
        # Create a proper table using st.dataframe for consistent spacing
        table_data = []
        
        for ride in available_rides:
            ride_data = data_manager.get_ride_data(ride)
            
            # Get basic info
            duration = "N/A"
            distance = "N/A"
            power = "N/A"
            
            if ride_data['in_history'] and ride_data['history_data']:
                history = ride_data['history_data']
                duration = format_duration(history.get('duration_min', 0))
                distance = format_distance(history.get('distance_km', 0))
                power = format_power(history.get('avg_power_W', 0))
            
            # Status indicators with more descriptive text
            fit_status = "✅ Available" if ride_data['has_fit_file'] else "❌ Missing"
            analysis_status = "✅ Complete" if ride_data['analysis_available'] else "⏳ Pending"
            
            table_data.append({
                "Activity": ride,
                "Duration": duration,
                "Distance": distance,
                "Power": power,
                "FIT File": fit_status,
                "Analysis": analysis_status
            })
        
        # Create DataFrame and display as table
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)
    else:
        st.info("No activities uploaded yet.")

    st.markdown("---")
    
    # Analysis section
    st.subheader("📊 Run Analysis")
    
    if not available_rides:
        st.info("Upload a FIT file first.")
    else:
        selected_ride = st.selectbox(
            "Select activity to analyze:",
            available_rides
        )
        
        if selected_ride:
            ride_data = data_manager.get_ride_data(selected_ride)
            
            if not ride_data['has_fit_file']:
                st.warning("No FIT file available for this activity.")
            else:
                st.success(f"✅ FIT file available for {selected_ride}")
                
                # Load settings for analysis
                try:
                    settings = data_manager.load_settings()
                except Exception as e:
                    st.error(f"Error loading settings: {str(e)}")
                    settings = {
                        'ftp': 250,
                        'lthr': 160,
                        'max_hr': 195,
                        'rest_hr': 51,
                        'weight_kg': 70,
                        'height_cm': 175,
                        'athlete_name': 'Cyclist'
                    }
                
                # Auto-run analysis if not already done
                if not ride_data['analysis_available']:
                    if st.button("🚀 Run Analysis", type="primary"):
                        with st.spinner("Analyzing..."):
                            try:
                                analyzer = CyclingAnalyzer(
                                    save_figures=True, 
                                    ftp=settings.get('ftp', 250),
                                    max_hr=settings.get('max_hr', 195),
                                    rest_hr=settings.get('rest_hr', 51),
                                    weight_kg=settings.get('weight_kg', 70),
                                    height_cm=settings.get('height_cm', 175),
                                    athlete_name=settings.get('athlete_name', 'Cyclist'),
                                    save_dir="figures",
                                    analysis_id=selected_ride
                                )
                                
                                file_path = ride_data['fit_file_path']
                                
                                if analyzer.load_fit_file(file_path):
                                    analyzer.clean_and_smooth_data()
                                    analyzer.calculate_metrics()
                                    
                                    # Run comprehensive analysis
                                    analyzer.print_summary()
                                    analyzer.create_dashboard()
                                    analyzer.create_training_peaks_dual_axis_graph()  # Add dual axis analysis
                                    analyzer.analyze_fatigue_patterns()
                                    analyzer.analyze_heat_stress()
                                    analyzer.analyze_power_hr_efficiency()
                                    analyzer.analyze_variable_relationships()
                                    analyzer.analyze_torque()
                                    # Calculate W' balance with proper error handling
                                    try:
                                        cp_est, w_prime_est = analyzer.estimate_critical_power()
                                        if cp_est is not None and w_prime_est is not None:
                                            analyzer.calculate_w_prime_balance(cp_est, w_prime_est)
                                        else:
                                            # Use FTP as fallback for CP estimation
                                            cp_est = settings.get('ftp', 250)
                                            w_prime_est = 20000  # Default W' estimate
                                            analyzer.calculate_w_prime_balance(cp_est, w_prime_est)
                                    except Exception as e:
                                        st.warning(f"W' balance calculation failed: {str(e)}")
                                        # Use FTP as fallback
                                        cp_est = settings.get('ftp', 250)
                                        w_prime_est = 20000  # Default W' estimate
                                        analyzer.calculate_w_prime_balance(cp_est, w_prime_est)
                                    analyzer.estimate_lactate()
                                    
                                    # Extract metrics for saving
                                    metrics_to_save = {
                                        "status": "completed",
                                        "avg_power_W": analyzer.avg_power if hasattr(analyzer, 'avg_power') else 0,
                                        "max_power_W": analyzer.max_power if hasattr(analyzer, 'max_power') else 0,
                                        "NP_W": analyzer.np_calc if hasattr(analyzer, 'np_calc') else 0,
                                        "VI": analyzer.VI if hasattr(analyzer, 'VI') else 0,
                                        "TSS": analyzer.TSS if hasattr(analyzer, 'TSS') else 0,
                                        "IF": analyzer.IF if hasattr(analyzer, 'IF') else 0,
                                        "duration_min": analyzer.duration_hr * 60 if hasattr(analyzer, 'duration_hr') else 0,
                                        "distance_km": analyzer.total_distance if hasattr(analyzer, 'total_distance') else 0,
                                        "avg_speed_kmh": analyzer.metrics.get('speed', {}).get('avg', 0) if hasattr(analyzer, 'metrics') and 'speed' in analyzer.metrics else 0,
                                        "max_speed_kmh": analyzer.metrics.get('speed', {}).get('max', 0) if hasattr(analyzer, 'metrics') and 'speed' in analyzer.metrics else 0,
                                        "avg_hr": analyzer.metrics.get('hr', {}).get('avg', 0) if hasattr(analyzer, 'metrics') and 'hr' in analyzer.metrics else 0,
                                        "max_hr": analyzer.metrics.get('hr', {}).get('max', 0) if hasattr(analyzer, 'metrics') and 'hr' in analyzer.metrics else 0,
                                        "avg_cadence": analyzer.metrics.get('cadence', {}).get('avg', 0) if hasattr(analyzer, 'metrics') and 'cadence' in analyzer.metrics else 0,
                                        "max_cadence": analyzer.metrics.get('cadence', {}).get('max', 0) if hasattr(analyzer, 'metrics') and 'cadence' in analyzer.metrics else 0,
                                        # Calories calculation removed - inaccurate
                                        "total_kj": analyzer.total_kj if hasattr(analyzer, 'total_kj') else 0,
                                        "total_elevation_m": analyzer.total_elevation_m if hasattr(analyzer, 'total_elevation_m') else 0
                                    }
                                    
                                    # Add power bests to metrics
                                    if hasattr(analyzer, 'power_bests') and analyzer.power_bests:
                                        for interval, best_data in analyzer.power_bests.items():
                                            power = best_data.get('power', 0)
                                            if not pd.isna(power) and power > 0:
                                                metrics_to_save[f"power_best_{interval}"] = power
                                    
                                    # Save results
                                    data_manager.save_analysis_results(
                                        selected_ride, "Advanced", metrics_to_save, 
                                        settings.get('ftp', 250), settings.get('lthr', 160)
                                    )
                                    
                                    st.success("✅ Analysis completed!")
                                    
                                    # Display power bests summary
                                    if hasattr(analyzer, 'power_bests') and analyzer.power_bests:
                                        st.markdown("---")
                                        st.subheader("🏆 Power Bests Summary")
                                        
                                        # Create a grid layout for power bests
                                        cols = st.columns(4)
                                        col_idx = 0
                                        
                                        # Define display order
                                        display_order = ['1s', '5s', '10s', '30s', '1min', '3min', '5min', '8min', '10min', '12min', '20min', '60min', '90min']
                                        
                                        for interval in display_order:
                                            if interval in analyzer.power_bests:
                                                best_data = analyzer.power_bests[interval]
                                                power = best_data.get('power', 0)
                                                
                                                if not pd.isna(power) and power > 0:
                                                    with cols[col_idx]:
                                                        st.metric(f"{interval} Best", f"{power:.0f}W")
                                                    
                                                    col_idx = (col_idx + 1) % 4
                                                    
                                                    if col_idx == 0:  # Start new row
                                                        cols = st.columns(4)
                                    
                                    st.rerun()
                                else:
                                    st.error("❌ Analysis failed")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                else:
                    st.success("✅ Analysis already completed")
                    
                    # Multi-Axis Analysis Graph - Full Width at Top
                    st.markdown("---")
                    st.subheader("📈 Multi-Axis Analysis")
                    st.markdown("*Multiple metrics with dual y-axes - Professional analysis style*")
                    st.markdown("")
                    
                    # Variable visibility controls
                    st.markdown("**🎛️ Variable Controls**")
                    st.markdown("*Toggle variables on/off to focus your analysis*")
                    
                    # Create columns for variable controls
                    var_cols = st.columns(4)
                    
                    # Variable visibility toggles
                    show_power = var_cols[0].checkbox("Power", value=True, key="show_power_analysis")
                    show_hr = var_cols[1].checkbox("Heart Rate", value=True, key="show_hr_analysis")
                    show_speed = var_cols[2].checkbox("Speed", value=False, key="show_speed_analysis")
                    show_cadence = var_cols[3].checkbox("Cadence", value=False, key="show_cadence_analysis")
                    
                    # Load and display the dashboard figure
                    try:
                        dashboard_path = f"figures/{selected_ride}_dashboard.png"
                        if os.path.exists(dashboard_path):
                            st.image(dashboard_path, use_column_width=True)
                        else:
                            st.warning("Dashboard figure not found. Please re-run the analysis.")
                    except Exception as e:
                        st.error(f"Error loading dashboard: {e}")
                    
                    # Load and display the dual axis analysis
                    try:
                        # Look for dual axis files with the analysis ID pattern
                        import glob
                        dual_axis_files = glob.glob(f"figures/*_multi_axis_analysis.png")
                        if dual_axis_files:
                            # Use the most recent dual axis file
                            dual_axis_path = sorted(dual_axis_files)[-1]
                            st.markdown("---")
                            st.subheader("📈 Enhanced Dual-Axis Analysis")
                            st.markdown("*Smooth, multi-metric visualization with dual y-axes*")
                            st.image(dual_axis_path, use_column_width=True)
                        else:
                            st.info("Dual-axis analysis not available. Please re-run the analysis.")
                    except Exception as e:
                        st.error(f"Error loading dual-axis analysis: {e}")
                    
                    # Additional analysis figures
                    st.markdown("---")
                    st.subheader("📊 Detailed Analysis")
                    
                    # Create tabs for different analysis types
                    analysis_tabs = st.tabs([
                        "🔄 Fatigue Patterns", 
                        "🌡️ Heat Stress", 
                        "💪 Power/HR Efficiency",
                        "📈 Variable Relationships",
                        "⚙️ Torque Analysis",
                        "⚡ W' Balance"
                    ])
                    
                    with analysis_tabs[0]:
                        try:
                            fatigue_path = f"figures/{selected_ride}_fatigue_patterns.png"
                            if os.path.exists(fatigue_path):
                                st.image(fatigue_path, use_column_width=True)
                            else:
                                st.info("Fatigue patterns analysis not available.")
                        except Exception as e:
                            st.error(f"Error loading fatigue analysis: {e}")
                    
                    with analysis_tabs[1]:
                        try:
                            heat_path = f"figures/{selected_ride}_heat_stress.png"
                            if os.path.exists(heat_path):
                                st.image(heat_path, use_column_width=True)
                            else:
                                st.info("Heat stress analysis not available.")
                        except Exception as e:
                            st.error(f"Error loading heat stress analysis: {e}")
                    
                    with analysis_tabs[2]:
                        try:
                            efficiency_path = f"figures/{selected_ride}_power_hr_efficiency.png"
                            if os.path.exists(efficiency_path):
                                st.image(efficiency_path, use_column_width=True)
                            else:
                                st.info("Power/HR efficiency analysis not available.")
                        except Exception as e:
                            st.error(f"Error loading efficiency analysis: {e}")
                    
                    with analysis_tabs[3]:
                        try:
                            relationships_path = f"figures/{selected_ride}_variable_relationships.png"
                            if os.path.exists(relationships_path):
                                st.image(relationships_path, use_column_width=True)
                            else:
                                st.info("Variable relationships analysis not available.")
                        except Exception as e:
                            st.error(f"Error loading relationships analysis: {e}")
                    
                    with analysis_tabs[4]:
                        try:
                            torque_path = f"figures/{selected_ride}_torque.png"
                            if os.path.exists(torque_path):
                                st.image(torque_path, use_column_width=True)
                            else:
                                st.info("Torque analysis not available.")
                        except Exception as e:
                            st.error(f"Error loading torque analysis: {e}")
                    
                    with analysis_tabs[5]:
                        try:
                            wprime_path = f"figures/{selected_ride}_w_prime_balance.png"
                            if os.path.exists(wprime_path):
                                st.image(wprime_path, use_column_width=True)
                            else:
                                st.info("W' balance analysis not available.")
                        except Exception as e:
                            st.error(f"Error loading W' balance analysis: {e}")

def show_settings_tab():
        st.header("⚙️ Analysis Settings")
        
        # Load current settings
        try:
            current_settings = data_manager.load_settings()
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")
            current_settings = {
                'athlete_name': 'Cyclist',
                'weight_kg': 70,
                'height_cm': 175,
                'max_hr': 195,
                'rest_hr': 51,
                'ftp': 250,
                'lthr': 160,
                'ftp_zone_1': 55,
                'ftp_zone_2': 75,
                'ftp_zone_3': 90,
                'ftp_zone_4': 105,
                'ftp_zone_5': 120,
                'ftp_zone_6': 150,
                'ftp_zone_7': 200
            }
        
        # Create settings form
        with st.form("analysis_settings"):
            st.subheader("👤 Athlete Profile")
            
            col1, col2 = st.columns(2)
            
            with col1:
                athlete_name = st.text_input(
                    "Athlete Name",
                    value=current_settings.get('athlete_name', 'Cyclist'),
                    help="Your name for personalization"
                )
                
                weight_kg = st.number_input(
                    "Weight (kg)",
                    min_value=30.0,
                    max_value=200.0,
                    value=float(current_settings.get('weight_kg', 70)),
                    step=0.5,
                    help="Your weight in kilograms"
                )
                
                height_cm = st.number_input(
                    "Height (cm)",
                    min_value=100,
                    max_value=250,
                    value=int(current_settings.get('height_cm', 175)),
                    step=1,
                    help="Your height in centimeters"
                )
            
            with col2:
                max_hr = st.number_input(
                    "Max Heart Rate (bpm)",
                    min_value=150,
                    max_value=220,
                    value=int(current_settings.get('max_hr', 195)),
                    step=1,
                    help="Your maximum heart rate"
                )
                
                rest_hr = st.number_input(
                    "Resting Heart Rate (bpm)",
                    min_value=30,
                    max_value=100,
                    value=int(current_settings.get('rest_hr', 51)),
                    step=1,
                    help="Your resting heart rate"
                )
            
            st.markdown("---")
            st.subheader("💪 Power Thresholds")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ftp = st.number_input(
                    "Functional Threshold Power (W)",
                    min_value=100,
                    max_value=500,
                    value=int(current_settings.get('ftp', 250)),
                    step=5,
                    help="Your FTP in watts"
                )
                
                lthr = st.number_input(
                    "Lactate Threshold Heart Rate (bpm)",
                    min_value=120,
                    max_value=200,
                    value=int(current_settings.get('lthr', 160)),
                    step=1,
                    help="Your lactate threshold heart rate"
                )
            

            
            # Save settings button
            if st.form_submit_button("💾 Save Settings", type="primary"):
                # Prepare settings dictionary
                new_settings = {
                    'athlete_name': athlete_name,
                    'weight_kg': weight_kg,
                    'height_cm': height_cm,
                    'max_hr': max_hr,
                    'rest_hr': rest_hr,
                    'ftp': ftp,
                    'lthr': lthr
                }
                
                # Save settings
                try:
                    if data_manager.save_settings(new_settings):
                        st.success("✅ Settings saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save settings")
                except Exception as e:
                    st.error(f"❌ Error saving settings: {str(e)}")
        
        # Display current settings summary
        st.markdown("---")
        st.subheader("📋 Current Settings Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👤 Athlete Profile**")
            st.markdown(f"• Name: {current_settings.get('athlete_name', 'Not set')}")
            st.markdown(f"• Weight: {current_settings.get('weight_kg', 0)} kg")
            st.markdown(f"• Height: {current_settings.get('height_cm', 0)} cm")
            st.markdown(f"• Max HR: {current_settings.get('max_hr', 0)} bpm")
            st.markdown(f"• Rest HR: {current_settings.get('rest_hr', 0)} bpm")
        
        with col2:
            st.markdown("**💪 Power Thresholds**")
            st.markdown(f"• FTP: {current_settings.get('ftp', 0)} W")
            st.markdown(f"• LTHR: {current_settings.get('lthr', 0)} bpm")
        
        # Settings file info
        st.markdown("---")
        st.subheader("📁 Settings File Information")
        
        settings_file = data_manager.settings_path
        if settings_file.exists():
            file_size = settings_file.stat().st_size
            st.markdown(f"• **File:** {settings_file}")
            st.markdown(f"• **Size:** {file_size} bytes")
            st.markdown(f"• **Last Modified:** {datetime.fromtimestamp(settings_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No settings file found. Default settings will be used.")
        
        # Data Management Section
        st.markdown("---")
        st.subheader("🗑️ Data Management")
        
        # Show current data statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_rides = len(data_manager.get_available_rides())
            st.metric("Total Rides", total_rides)
        
        with col2:
            cached_files = len(st.session_state.uploaded_files)
            st.metric("Cached Files", cached_files)
        
        with col3:
            history_entries = len(data_manager.ride_history)
            st.metric("History Entries", history_entries)
        
        # Delete all data button with confirmation
        st.markdown("---")
        st.subheader("⚠️ Nuclear Option")
        
        st.warning("""
        **⚠️ WARNING: This action cannot be undone!**
        
        This will permanently delete:
        • All uploaded FIT files
        • All analysis results
        • All ride history
        • All generated figures
        • All cached data
        • Settings (will be reset to defaults)
        """)
        
        # Confirmation checkbox
        confirm_delete = st.checkbox(
            "I understand this will permanently delete ALL data and cannot be undone",
            key="confirm_delete_all"
        )
        
        # Confirmation text input
        confirmation_text = st.text_input(
            "Type 'DELETE ALL' to confirm",
            placeholder="DELETE ALL",
            key="delete_confirmation_text"
        )
        
        # Delete button
        if confirm_delete and confirmation_text == "DELETE ALL":
            if st.button("🗑️ DELETE ALL FILES AND DATA", type="secondary", key="delete_all_button"):
                with st.spinner("Deleting all data..."):
                    try:
                        # Get system status before deletion
                        status = data_manager.get_system_status()
                        
                        # Perform deletion
                        success, message = data_manager.clear_all_rides()
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.info(f"Deleted {status['total_rides']} rides, {status['cached_files']} cached files, and {status['history_entries']} history entries.")
                            
                            # Clear session state
                            st.session_state.uploaded_files.clear()
                            st.session_state.analysis_cache.clear()
                            
                            # Reset settings to defaults
                            default_settings = {
                                'athlete_name': 'Cyclist',
                                'weight_kg': 70,
                                'height_cm': 175,
                                'max_hr': 195,
                                'rest_hr': 51,
                                'ftp': 250,
                                'lthr': 160
                            }
                            data_manager.save_settings(default_settings)
                            
                            st.success("✅ All data has been permanently deleted and settings reset to defaults.")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                    except Exception as e:
                        st.error(f"❌ Error during deletion: {str(e)}")
        elif confirm_delete and confirmation_text != "DELETE ALL":
            st.error("❌ Please type 'DELETE ALL' exactly to confirm deletion")
        elif not confirm_delete:
            st.info("🔒 Check the confirmation box and type 'DELETE ALL' to enable deletion")

def show_system_tab():
    st.header("🛠️ System Information")
    
    # Display basic system info
    st.subheader("📋 Basic System Info")
    st.markdown(f"""
    - **Operating System:** {platform.platform()}
    - **Python Version:** {platform.python_version()}
    - **Streamlit Version:** {st.__version__}
    - **Data Manager Version:** {getattr(data_manager, '__version__', 'N/A')}
    """)
    
    # Display data manager status
    st.subheader("📊 Data Manager Status")
    try:
        status = data_manager.get_system_status()
        st.markdown(f"• **Total Rides:** {status.get('total_rides', 0)}")
        st.markdown(f"• **Cached Files:** {status.get('cached_files', 0)}")
        st.markdown(f"• **Analysis Entries:** {status.get('analysis_entries', 0)}")
        st.markdown(f"• **Settings File:** {data_manager.settings_path}")
        st.markdown(f"• **Data Directory:** {data_manager.data_dir}")
    except Exception as e:
        st.error(f"❌ Error getting system status: {e}")
    
    # Cache Management Section
    st.subheader("🗂️ Cache Management")
    
    try:
        # Get detailed cache information
        cache_info = data_manager.get_cache_info()
        
        if cache_info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("FIT Files", cache_info.get('fit_files_count', 0))
                st.caption(f"Size: {cache_info.get('fit_files_size_mb', 0):.1f} MB")
            
            with col2:
                st.metric("Session Entries", cache_info.get('session_entries', 0))
                st.caption("Uploaded files")
            
            with col3:
                st.metric("Analysis Cache", cache_info.get('analysis_cache_entries', 0))
                st.caption("Cached results")
            
            with col4:
                st.metric("Registry Entries", cache_info.get('registry_entries', 0))
                st.caption("File tracking")
            
            st.markdown("---")
            
            # Cache clearing options
            st.markdown("**🧹 Cache Clearing Options**")
            st.markdown("*Select what type of cache to clear*")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🗑️ Clear FIT Files", key="clear_fit_files"):
                    with st.spinner("Clearing FIT files..."):
                        success, message = data_manager.clear_cache("fit_files")
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            with col2:
                if st.button("🗑️ Clear Session Cache", key="clear_session_cache"):
                    with st.spinner("Clearing session cache..."):
                        success, message = data_manager.clear_cache("session")
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            with col3:
                if st.button("🗑️ Clear Analysis Cache", key="clear_analysis_cache"):
                    with st.spinner("Clearing analysis cache..."):
                        success, message = data_manager.clear_cache("analysis")
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            with col4:
                if st.button("🗑️ Clear All Cache", key="clear_all_cache", type="secondary"):
                    with st.spinner("Clearing all cache..."):
                        success, message = data_manager.clear_cache("all")
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            # Cache information details
            st.markdown("---")
            st.subheader("📊 Cache Details")
            
            if cache_info.get('fit_files_count', 0) > 0:
                st.markdown("**📁 Cached FIT Files:**")
                for fit_file in data_manager.cache_dir.glob("*.fit"):
                    file_size = fit_file.stat().st_size / 1024  # KB
                    st.markdown(f"• {fit_file.name} ({file_size:.1f} KB)")
            
            if cache_info.get('session_entries', 0) > 0:
                st.markdown("**📋 Session Cache:**")
                for ride_id in st.session_state.uploaded_files.keys():
                    st.markdown(f"• {ride_id}")
            
            if cache_info.get('analysis_cache_entries', 0) > 0:
                st.markdown("**📈 Analysis Cache:**")
                for cache_key in st.session_state.analysis_cache.keys():
                    st.markdown(f"• {cache_key}")
        
        else:
            st.info("No cache information available.")
    
    except Exception as e:
        st.error(f"❌ Error getting cache information: {e}")
    
    # Display data directory contents
    st.subheader("📁 Data Directory Contents")
    try:
        data_dir_contents = list(data_manager.data_dir.iterdir())
        if data_dir_contents:
            st.markdown("**Files:**")
            for item in data_dir_contents:
                st.markdown(f"- {item.name} ({item.stat().st_size} bytes)")
        else:
            st.info("No files found in the data directory.")
    except Exception as e:
        st.error(f"❌ Error listing data directory: {e}")

if __name__ == "__main__":
    main() 