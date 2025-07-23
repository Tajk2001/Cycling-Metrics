import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime
from data_manager import CyclingDataManager
from app import run_basic_analysis
from enhanced_cycling_analysis import CyclingAnalyzer

# Page configuration
st.set_page_config(
    page_title="Cycling Performance Analysis",
    layout="wide",
    page_icon="🚴",
    initial_sidebar_state="expanded"
)

# Initialize data manager
@st.cache_resource
def get_data_manager():
    return CyclingDataManager()

data_manager = get_data_manager()

# Helper functions for formatting
def format_number(value, decimals=1):
    """Format numbers with appropriate decimal places"""
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:.{decimals}f}"

def format_duration(minutes):
    """Format duration in hours and minutes"""
    if pd.isna(minutes):
        return "N/A"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"

def format_power(power):
    """Format power values"""
    if pd.isna(power):
        return "N/A"
    return f"{int(power)}W"

def format_distance(km):
    """Format distance values"""
    if pd.isna(km):
        return "N/A"
    return f"{km:.1f}km"

def format_percentage(value):
    """Format percentage values"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"

# Main dashboard
def main():
    st.title("🚴 Cycling Performance Analysis")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # FTP and LTHR inputs
        ftp = st.number_input("FTP (W)", min_value=100, max_value=500, value=250, step=5)
        lthr = st.number_input("LTHR (bpm)", min_value=120, max_value=200, value=160, step=1)
        
        st.markdown("---")
        
        # System status
        st.header("📊 System Status")
        status = data_manager.get_system_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Activities", status['total_rides'])
            st.metric("Cached Files", status['cached_files'])
        with col2:
            st.metric("History Entries", status['history_entries'])
            st.metric("Analyses", status['analysis_entries'])
        
        # Data management
        st.markdown("---")
        st.header("🗂️ Data Management")
        
        if st.button("🧹 Cleanup Orphaned Files"):
            orphaned = data_manager.cleanup_orphaned_files()
            if orphaned:
                st.success(f"Cleaned up {len(orphaned)} orphaned files")
            else:
                st.info("No orphaned files found")
        
        if st.button("💾 Save All Data"):
            if data_manager.save_data():
                st.success("Data saved successfully")
            else:
                st.error("Failed to save data")
    
    # Main content tabs - Professional structure
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "📁 Activities", 
        "📈 Analysis", 
        "⚙️ Settings"
    ])
    
    # Tab 1: Overview (Dashboard)
    with tab1:
        st.header("📊 Performance Overview")
        
        available_rides = data_manager.get_available_rides()
        
        if not available_rides:
            st.info("No activities available. Upload FIT files to get started.")
        else:
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            
            # Calculate summary stats from all rides
            all_rides_data = []
            for ride in available_rides:
                ride_data = data_manager.get_ride_data(ride)
                if ride_data['in_history'] and ride_data['history_data']:
                    all_rides_data.append(ride_data['history_data'])
            
            if all_rides_data:
                # Create summary dataframe
                summary_df = pd.DataFrame(all_rides_data)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Activities", len(all_rides_data))
                    st.metric("Total Distance", f"{summary_df['distance_km'].sum():.1f} km")
                
                with col2:
                    st.metric("Total Duration", format_duration(summary_df['duration_min'].sum()))
                    st.metric("Total TSS", f"{summary_df['TSS'].sum():.0f}")
                
                with col3:
                    st.metric("Avg Power", format_power(summary_df['avg_power_W'].mean()))
                    st.metric("Max Power", format_power(summary_df['max_power_W'].max()))
                
                with col4:
                    st.metric("Avg HR", f"{summary_df['avg_hr'].mean():.0f} bpm")
                    st.metric("Total Elevation", f"{summary_df['total_elevation_m'].sum():.0f}m")
            
            # Recent activities
            st.markdown("---")
            st.subheader("🕒 Recent Activities")
            
            # Show last 5 activities
            recent_rides = available_rides[-5:] if len(available_rides) > 5 else available_rides
            
            for ride in reversed(recent_rides):
                ride_data = data_manager.get_ride_data(ride)
                
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                
                with col1:
                    st.write(f"**{ride}**")
                    if ride_data['in_history'] and ride_data['history_data']:
                        history = ride_data['history_data']
                        st.caption(f"{format_duration(history.get('duration_min', 0))} • "
                                 f"{format_distance(history.get('distance_km', 0))} • "
                                 f"{format_power(history.get('avg_power_W', 0))}")
                
                with col2:
                    if ride_data['has_fit_file']:
                        st.success("✅")
                    else:
                        st.error("❌")
                
                with col3:
                    if ride_data['analysis_available']:
                        st.success("✅")
                    else:
                        st.info("⏳")
                
                with col4:
                    if ride_data['in_history'] and ride_data['history_data']:
                        history = ride_data['history_data']
                        st.metric("TSS", f"{history.get('TSS', 0):.0f}")
                
                with col5:
                    if ride_data['in_history'] and ride_data['history_data']:
                        history = ride_data['history_data']
                        st.metric("IF", f"{history.get('IF', 0)*100:.1f}%")
    
    # Tab 2: Activities (File Management)
    with tab2:
        st.header("📁 Activity Management")
        
        # File upload section
        st.subheader("📤 Upload New Activity")
        
        uploaded_file = st.file_uploader(
            "Choose a .fit file", 
            type=["fit"], 
            help="Upload your cycling activity file (.fit format)"
        )
        
        if uploaded_file:
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                ride_id = data_manager._generate_ride_id(uploaded_file.name)
                st.metric("Activity ID", ride_id)
            
            # Upload button
            if st.button("📤 Upload Activity", type="primary"):
                with st.spinner("Uploading activity..."):
                    success, message, file_path = data_manager.upload_fit_file(uploaded_file)
                    
                    if success:
                        st.success(message)
                        st.info("✅ Activity uploaded successfully! Switch to 'Analysis' tab to analyze.")
                    else:
                        st.error(message)
        
        # Activity list
        st.markdown("---")
        st.subheader("📋 Activity Library")
        
        available_rides = data_manager.get_available_rides()
        
        if not available_rides:
            st.info("No activities uploaded yet. Upload a FIT file to get started.")
        else:
            # Activity selection for management
            selected_activity = st.selectbox(
                "Select an activity:",
                available_rides,
                help="Choose an activity to view details or manage"
            )
            
            if selected_activity:
                ride_data = data_manager.get_ride_data(selected_activity)
                
                # Activity details
                st.markdown("---")
                st.subheader(f"📊 Activity Details: {selected_activity}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("FIT File", "✅ Available" if ride_data['has_fit_file'] else "❌ Missing")
                    if ride_data['in_history'] and ride_data['history_data']:
                        history = ride_data['history_data']
                        st.metric("Duration", format_duration(history.get('duration_min', 0)))
                
                with col2:
                    st.metric("Analysis", "✅ Available" if ride_data['analysis_available'] else "❌ None")
                    if ride_data['in_history'] and ride_data['history_data']:
                        history = ride_data['history_data']
                        st.metric("Distance", format_distance(history.get('distance_km', 0)))
                
                with col3:
                    if ride_data['in_history'] and ride_data['history_data']:
                        history = ride_data['history_data']
                        st.metric("Avg Power", format_power(history.get('avg_power_W', 0)))
                        st.metric("TSS", f"{history.get('TSS', 0):.0f}")
                
                with col4:
                    if ride_data['in_history'] and ride_data['history_data']:
                        history = ride_data['history_data']
                        st.metric("IF", f"{history.get('IF', 0)*100:.1f}%")
                        st.metric("VI", f"{history.get('VI', 0):.2f}")
                
                # Activity actions
                st.markdown("---")
                st.subheader("🔧 Activity Actions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Analyze Activity", type="primary"):
                        st.info("Switch to 'Analysis' tab to run analysis on this activity")
                
                with col2:
                    # Confirmation for deletion
                    confirm_delete = st.checkbox(
                        "I understand this will permanently delete this activity and all associated data",
                        key=f"confirm_delete_{selected_activity}"
                    )
                    
                    if st.button("🗑️ Delete Activity", type="secondary", disabled=not confirm_delete):
                        with st.spinner(f"Deleting {selected_activity}..."):
                            success, message = data_manager.delete_ride(selected_activity)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.warning(message)
    
    # Tab 3: Analysis (All Analysis Functions)
    with tab3:
        st.header("📈 Performance Analysis")
        
        available_rides = data_manager.get_available_rides()
        
        if not available_rides:
            st.info("No activities available for analysis. Upload FIT files in the 'Activities' tab first.")
        else:
            # Analysis selection
            st.subheader("🎯 Analysis Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_ride = st.selectbox(
                    "Select activity to analyze:",
                    available_rides,
                    help="Choose an activity to run analysis on"
                )
            
            with col2:
                analysis_type = st.selectbox(
                    "Analysis Type",
                    ["Basic", "Advanced", "Both"],
                    help="Basic: Core metrics. Advanced: Detailed physiological analysis. Both: Complete analysis."
                )
            
            if selected_ride:
                ride_data = data_manager.get_ride_data(selected_ride)
                
                # Check if FIT file is available
                if not ride_data['has_fit_file']:
                    st.warning(f"❌ No FIT file available for '{selected_ride}'. Please upload the file first.")
                else:
                    st.success(f"✅ FIT file available for '{selected_ride}'")
                    
                    # Analysis options
                    st.markdown("---")
                    st.subheader("⚙️ Analysis Settings")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        save_figures = st.checkbox("Save Figures", value=True, help="Save analysis figures to disk")
                    
                    with col2:
                        overwrite_existing = st.checkbox("Overwrite Existing Analysis", value=False, help="Replace existing analysis results")
                    
                    # Run analysis
                    if st.button("🚀 Run Analysis", type="primary"):
                        file_path = ride_data['fit_file_path']
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            # Basic Analysis
                            if analysis_type in ["Basic", "Both"]:
                                status_text.text("Running basic analysis...")
                                progress_bar.progress(25)
                                
                                results, df = run_basic_analysis(
                                    file_path, 
                                    ftp=ftp, 
                                    lthr=lthr, 
                                    save_figures=save_figures, 
                                    analysis_id=selected_ride
                                )
                                
                                if results:
                                    # Save results
                                    data_manager.save_analysis_results(
                                        selected_ride, "Basic", results, ftp, lthr
                                    )
                                    
                                    # Display results
                                    st.success("✅ Basic analysis completed!")
                                    display_basic_results(results)
                                else:
                                    st.error("❌ Basic analysis failed")
                            
                            # Advanced Analysis
                            if analysis_type in ["Advanced", "Both"]:
                                status_text.text("Running advanced analysis...")
                                progress_bar.progress(75)
                                
                                analyzer = CyclingAnalyzer(
                                    save_figures=save_figures, 
                                    ftp=ftp, 
                                    save_dir="figures"
                                )
                                
                                if analyzer.load_fit_file(file_path):
                                    analyzer.clean_and_smooth_data()
                                    analyzer.calculate_metrics()
                                    analyzer.print_summary()
                                    analyzer.create_dashboard()
                                    
                                    # Save results
                                    data_manager.save_analysis_results(
                                        selected_ride, "Advanced", {"status": "completed"}, ftp, lthr
                                    )
                                    
                                    st.success("✅ Advanced analysis completed!")
                                    st.info("📊 Check the 'Analysis Results' section below for visualizations")
                                else:
                                    st.error("❌ Advanced analysis failed")
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Analysis complete!")
                            
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            progress_bar.progress(0)
                
                # Analysis Results Display
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Check for available figures
                figures_dir = pathlib.Path("figures")
                ride_figures = []
                
                if figures_dir.exists():
                    for figure_file in figures_dir.glob(f"{selected_ride}_*"):
                        if figure_file.suffix in ['.png', '.svg']:
                            ride_figures.append(figure_file)
                
                if ride_figures:
                    st.success(f"✅ Found {len(ride_figures)} analysis figures")
                    
                    # Group figures by type
                    figure_types = {}
                    for fig in ride_figures:
                        fig_name = fig.stem.replace(f"{selected_ride}_", "")
                        if fig_name not in figure_types:
                            figure_types[fig_name] = []
                        figure_types[fig_name].append(fig)
                    
                    # Display figures by type
                    for fig_type, figures in figure_types.items():
                        st.markdown(f"**{fig_type.replace('_', ' ').title()}**")
                        
                        cols = st.columns(min(len(figures), 2))
                        for i, fig in enumerate(figures):
                            with cols[i % len(cols)]:
                                st.image(str(fig), caption=fig.name, use_column_width=True)
                        
                        st.markdown("---")
                else:
                    st.info("No analysis figures found. Run analysis to generate visualizations.")
                
                # Detailed metrics display
                if ride_data['in_history'] and ride_data['history_data']:
                    st.markdown("---")
                    st.subheader("📋 Detailed Metrics")
                    
                    history = ride_data['history_data']
                    
                    # Create detailed metrics display
                    metrics_cols = st.columns(4)
                    
                    with metrics_cols[0]:
                        st.markdown("**Power Metrics**")
                        st.metric("Max Power", format_power(history.get('max_power_W', 0)))
                        st.metric("Avg Power", format_power(history.get('avg_power_W', 0)))
                        st.metric("NP", format_power(history.get('NP_W', 0)))
                        st.metric("VI", format_number(history.get('VI', 0)))
                    
                    with metrics_cols[1]:
                        st.markdown("**Heart Rate**")
                        st.metric("Avg HR", format_number(history.get('avg_hr', 0)))
                        st.metric("Max HR", format_number(history.get('max_hr', 0)))
                        st.metric("Min HR", format_number(history.get('min_hr', 0)))
                        st.metric("HR Range", format_number(history.get('max_hr', 0) - history.get('min_hr', 0)))
                    
                    with metrics_cols[2]:
                        st.markdown("**Training Load**")
                        st.metric("IF", format_percentage(history.get('IF', 0) * 100))
                        st.metric("TSS", format_number(history.get('TSS', 0)))
                        st.metric("Duration", format_duration(history.get('duration_min', 0)))
                        st.metric("Distance", format_distance(history.get('distance_km', 0)))
                    
                    with metrics_cols[3]:
                        st.markdown("**Other Metrics**")
                        st.metric("Calories", format_number(history.get('calories', 0)))
                        st.metric("Elevation", f"{history.get('total_elevation_m', 0):.0f}m")
                        st.metric("Avg Speed", f"{history.get('avg_speed_kmh', 0):.1f} km/h")
                        st.metric("Max Speed", f"{history.get('max_speed_kmh', 0):.1f} km/h")
    
    # Tab 4: Settings
    with tab4:
        st.header("⚙️ Settings & System")
        
        # System status
        status = data_manager.get_system_status()
        
        st.subheader("📊 System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Activities", status['total_rides'])
            st.metric("Cached Files", status['cached_files'])
            st.metric("History Entries", status['history_entries'])
            st.metric("Analysis Entries", status['analysis_entries'])
        
        with col2:
            st.metric("Data Directory", status['data_directory'])
            st.metric("Cache Directory", status['cache_directory'])
            st.metric("Figures Directory", status['figures_directory'])
        
        # Data export/import
        st.markdown("---")
        st.subheader("💾 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export All Data"):
                export_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if data_manager.export_data(export_path):
                    st.success(f"✅ Data exported to {export_path}")
                else:
                    st.error("❌ Export failed")
        
        with col2:
            if st.button("📥 Import Data"):
                st.info("Import functionality requires manual file placement in data directory")
        
        # Clear all data
        st.markdown("---")
        st.subheader("🗑️ Data Management")
        
        st.markdown("⚠️ **Warning**: This will delete ALL activities and data!")
        
        # Confirmation checkboxes for clear all
        confirm_clear_all = st.checkbox(
            "I understand this will permanently delete ALL activities and data",
            key="confirm_clear_all"
        )
        
        confirm_clear_all_final = st.checkbox(
            "I am absolutely sure I want to delete everything",
            key="confirm_clear_all_final",
            disabled=not confirm_clear_all
        )
        
        if st.button("🗑️ Clear All Data", type="secondary", disabled=not (confirm_clear_all and confirm_clear_all_final)):
            with st.spinner("Clearing all data..."):
                success, message = data_manager.clear_all_rides()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.warning(message)
        
        if not confirm_clear_all:
            st.info("Please check the first confirmation box above")
        elif not confirm_clear_all_final:
            st.info("Please check both confirmation boxes above")
        
        # File registry
        st.markdown("---")
        st.subheader("📁 File Registry")
        
        if data_manager.file_registry:
            registry_df = pd.DataFrame.from_dict(data_manager.file_registry, orient='index')
            registry_df['ride_id'] = registry_df.index
            registry_df = registry_df[['ride_id', 'original_name', 'upload_date', 'file_size']]
            registry_df['upload_date'] = pd.to_datetime(registry_df['upload_date']).dt.strftime('%Y-%m-%d %H:%M')
            registry_df['file_size'] = registry_df['file_size'].apply(lambda x: f"{x/1024:.1f} KB")
            
            st.dataframe(registry_df, use_container_width=True)
        else:
            st.info("No files in registry")

def display_basic_results(results):
    """Display basic analysis results in a clean format."""
    if not results:
        return
    
    # Session Summary
    if 'summary' in results:
        st.subheader("📈 Session Summary")
        summary_df = pd.DataFrame.from_dict(results['summary'], orient='index', columns=['Value'])
        st.dataframe(summary_df, use_container_width=True)
    
    # Advanced Metrics
    if 'advanced_metrics' in results:
        st.subheader("🎯 Advanced Metrics")
        advanced_df = pd.DataFrame.from_dict(results['advanced_metrics'], orient='index', columns=['Value'])
        st.dataframe(advanced_df, use_container_width=True)
    
    # Power Zones
    if 'zone_df' in results and results['zone_df'] is not None:
        st.subheader("⚡ Power Zone Distribution")
        st.dataframe(results['zone_df'], use_container_width=True)
        st.bar_chart(results['zone_df'].set_index('Zone')['Percentage (%)'])
    
    # HR Zones
    if 'hr_zone_df' in results and results['hr_zone_df'] is not None:
        st.subheader("❤️ HR Zone Distribution")
        st.dataframe(results['hr_zone_df'], use_container_width=True)
        st.bar_chart(results['hr_zone_df'].set_index('Zone')['Percentage (%)'])

if __name__ == "__main__":
    main() 