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
    page_title="Enhanced Cycling Analysis Dashboard",
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
    st.title("🚴 Enhanced Cycling Analysis Dashboard")
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
            st.metric("Total Rides", status['total_rides'])
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
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload", 
        "📊 Analyze", 
        "📈 Data & Figures", 
        "⚙️ Settings"
    ])
    
    # Tab 1: Upload
    with tab1:
        st.header("📁 Upload FIT Files")
        
        # File upload section
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
                st.metric("Ride ID", ride_id)
            
            # Upload button
            if st.button("📤 Upload File", type="primary"):
                with st.spinner("Uploading file..."):
                    success, message, file_path = data_manager.upload_fit_file(uploaded_file)
                    
                    if success:
                        st.success(message)
                        st.info("✅ File uploaded successfully! Switch to 'Analyze' tab to run analysis.")
                    else:
                        st.error(message)
        
        # Show uploaded files
        st.markdown("---")
        st.subheader("📋 Uploaded Files")
        
        available_rides = data_manager.get_available_rides()
        if available_rides:
            for ride in available_rides:
                ride_data = data_manager.get_ride_data(ride)
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{ride}**")
                    if ride_data['in_history'] and ride_data['history_data']:
                        history = ride_data['history_data']
                        st.caption(f"Duration: {format_duration(history.get('duration_min', 0))} | "
                                 f"Distance: {format_distance(history.get('distance_km', 0))} | "
                                 f"Avg Power: {format_power(history.get('avg_power_W', 0))}")
                
                with col2:
                    if ride_data['has_fit_file']:
                        st.success("✅ FIT")
                    else:
                        st.error("❌ Missing")
                
                with col3:
                    if ride_data['analysis_available']:
                        st.success("✅ Analyzed")
                    else:
                        st.info("⏳ Pending")
        else:
            st.info("No files uploaded yet. Upload a FIT file to get started.")
    
    # Tab 2: Analyze
    with tab2:
        st.header("📊 Analyze Rides")
        
        available_rides = data_manager.get_available_rides()
        
        if not available_rides:
            st.info("No rides available for analysis. Upload a FIT file in the 'Upload' tab first.")
        else:
            # Ride selection
            selected_ride = st.selectbox(
                "Select a ride to analyze:",
                available_rides,
                help="Choose a ride from your uploaded files"
            )
            
            if selected_ride:
                ride_data = data_manager.get_ride_data(selected_ride)
                
                # Check if FIT file is available
                if not ride_data['has_fit_file']:
                    st.warning(f"❌ No FIT file available for '{selected_ride}'. Please upload the file first.")
                else:
                    st.success(f"✅ FIT file available for '{selected_ride}'")
                    
                    # Analysis options
                    st.subheader("Analysis Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        analysis_type = st.selectbox(
                            "Analysis Type",
                            ["Basic", "Advanced", "Both"],
                            help="Basic: Core metrics. Advanced: Detailed physiological analysis. Both: Complete analysis."
                        )
                    
                    with col2:
                        save_figures = st.checkbox("Save Figures", value=True, help="Save analysis figures to disk")
                    
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
                                    st.info("📊 Check the 'Data & Figures' tab for detailed visualizations")
                                else:
                                    st.error("❌ Advanced analysis failed")
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Analysis complete!")
                            
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            progress_bar.progress(0)
    
    # Tab 3: Data & Figures
    with tab3:
        st.header("📈 Data & Figures")
        
        available_rides = data_manager.get_available_rides()
        
        if not available_rides:
            st.info("No rides available. Upload and analyze rides to see data and figures.")
        else:
            # Ride selection for viewing data
            selected_ride_data = st.selectbox(
                "Select a ride to view data and figures:",
                available_rides,
                help="Choose a ride to view its analysis results and figures"
            )
            
            if selected_ride_data:
                ride_data = data_manager.get_ride_data(selected_ride_data)
                
                # Display ride summary
                st.subheader(f"📊 Ride Summary: {selected_ride_data}")
                
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
                        st.metric("NP", format_power(history.get('NP_W', 0)))
                
                with col4:
                    if ride_data['in_history'] and ride_data['history_data']:
                        history = ride_data['history_data']
                        st.metric("IF", format_percentage(history.get('IF', 0) * 100))
                        st.metric("TSS", format_number(history.get('TSS', 0)))
                
                # Display detailed history data
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
                
                # Display figures
                st.markdown("---")
                st.subheader("📊 Analysis Figures")
                
                # Check for available figures
                figures_dir = pathlib.Path("figures")
                ride_figures = []
                
                if figures_dir.exists():
                    for figure_file in figures_dir.glob(f"{selected_ride_data}_*"):
                        if figure_file.suffix in ['.png', '.svg']:
                            ride_figures.append(figure_file)
                
                if ride_figures:
                    st.success(f"✅ Found {len(ride_figures)} figures for this ride")
                    
                    # Group figures by type
                    figure_types = {}
                    for fig in ride_figures:
                        fig_name = fig.stem.replace(f"{selected_ride_data}_", "")
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
                    st.info("No figures found for this ride. Run analysis to generate figures.")
                
                # Re-analyze option
                st.markdown("---")
                st.subheader("🔄 Re-analyze This Ride")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    reanalysis_type = st.selectbox(
                        "Analysis Type",
                        ["Basic", "Advanced", "Both"],
                        key="reanalysis_type"
                    )
                
                with col2:
                    reanalysis_save_figures = st.checkbox("Save Figures", value=True, key="reanalysis_save")
                
                if st.button("🔄 Run Re-analysis", type="primary"):
                    if ride_data['has_fit_file']:
                        file_path = ride_data['fit_file_path']
                        
                        with st.spinner(f"Re-analyzing {selected_ride_data}..."):
                            try:
                                # Basic re-analysis
                                if reanalysis_type in ["Basic", "Both"]:
                                    results, df = run_basic_analysis(
                                        file_path, 
                                        ftp=ftp, 
                                        lthr=lthr, 
                                        save_figures=reanalysis_save_figures, 
                                        analysis_id=f"{selected_ride_data}_reanalysis"
                                    )
                                    
                                    if results:
                                        data_manager.save_analysis_results(
                                            selected_ride_data, "Basic", results, ftp, lthr
                                        )
                                        st.success("✅ Basic re-analysis completed!")
                                        st.rerun()  # Refresh to show new figures
                                
                                # Advanced re-analysis
                                if reanalysis_type in ["Advanced", "Both"]:
                                    analyzer = CyclingAnalyzer(
                                        save_figures=reanalysis_save_figures, 
                                        ftp=ftp, 
                                        save_dir="figures"
                                    )
                                    
                                    if analyzer.load_fit_file(file_path):
                                        analyzer.clean_and_smooth_data()
                                        analyzer.calculate_metrics()
                                        analyzer.print_summary()
                                        analyzer.create_dashboard()
                                        
                                        data_manager.save_analysis_results(
                                            selected_ride_data, "Advanced", {"status": "completed"}, ftp, lthr
                                        )
                                        
                                        st.success("✅ Advanced re-analysis completed!")
                                        st.rerun()  # Refresh to show new figures
                                
                            except Exception as e:
                                st.error(f"❌ Re-analysis failed: {str(e)}")
                    else:
                        st.error("❌ No FIT file available for re-analysis")
    
    # Tab 4: Settings
    with tab4:
        st.header("⚙️ Settings & System Info")
        
        # System status
        status = data_manager.get_system_status()
        
        st.subheader("📊 System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Rides", status['total_rides'])
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
        
        # Ride management
        st.markdown("---")
        st.subheader("🗑️ Ride Management")
        
        # Get available rides for deletion
        available_rides = data_manager.get_available_rides()
        
        if available_rides:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Delete Specific Ride**")
                selected_ride_to_delete = st.selectbox(
                    "Select ride to delete:",
                    available_rides,
                    key="delete_ride_select"
                )
                
                # Confirmation checkbox for individual ride deletion
                confirm_delete = st.checkbox(
                    "I understand this will permanently delete this ride and all associated data",
                    key="confirm_delete"
                )
                
                if st.button("🗑️ Delete Selected Ride", type="secondary", disabled=not confirm_delete):
                    with st.spinner(f"Deleting {selected_ride_to_delete}..."):
                        success, message = data_manager.delete_ride(selected_ride_to_delete)
                        if success:
                            st.success(message)
                            st.rerun()  # Refresh the page to update the list
                        else:
                            st.warning(message)
                
                if not confirm_delete:
                    st.info("Please check the confirmation box above to enable deletion")
            
            with col2:
                st.markdown("**Clear All Rides**")
                st.markdown("⚠️ **Warning**: This will delete ALL rides and data!")
                
                # Confirmation checkboxes for clear all
                confirm_clear_all = st.checkbox(
                    "I understand this will permanently delete ALL rides and data",
                    key="confirm_clear_all"
                )
                
                confirm_clear_all_final = st.checkbox(
                    "I am absolutely sure I want to delete everything",
                    key="confirm_clear_all_final",
                    disabled=not confirm_clear_all
                )
                
                if st.button("🗑️ Clear All Rides", type="secondary", disabled=not (confirm_clear_all and confirm_clear_all_final)):
                    with st.spinner("Clearing all rides..."):
                        success, message = data_manager.clear_all_rides()
                        if success:
                            st.success(message)
                            st.rerun()  # Refresh the page
                        else:
                            st.warning(message)
                
                if not confirm_clear_all:
                    st.info("Please check the first confirmation box above")
                elif not confirm_clear_all_final:
                    st.info("Please check both confirmation boxes above")
        else:
            st.info("No rides available to delete")
        
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