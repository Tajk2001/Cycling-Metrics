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
        "📁 Upload & Analyze", 
        "📈 Ride History", 
        "🔍 Re-analyze", 
        "⚙️ System Info"
    ])
    
    # Tab 1: Upload & Analyze
    with tab1:
        st.header("📁 Upload & Analyze New Ride")
        
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
                        st.session_state.current_ride_id = ride_id
                        st.session_state.current_file_path = file_path
                    else:
                        st.error(message)
            
            # Analysis section (if file is uploaded)
            if 'current_ride_id' in st.session_state:
                st.markdown("---")
                st.header("📊 Analysis Options")
                
                analysis_type = st.selectbox(
                    "Analysis Type",
                    ["Basic", "Advanced", "Both"],
                    help="Basic: Core metrics. Advanced: Detailed physiological analysis. Both: Complete analysis."
                )
                
                save_figures = st.checkbox("Save Figures", value=True, help="Save analysis figures to disk")
                
                if st.button("🚀 Run Analysis", type="primary"):
                    ride_id = st.session_state.current_ride_id
                    file_path = st.session_state.current_file_path
                    
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
                                analysis_id=ride_id
                            )
                            
                            if results:
                                # Save results
                                data_manager.save_analysis_results(
                                    ride_id, "Basic", results, ftp, lthr
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
                                    ride_id, "Advanced", {"status": "completed"}, ftp, lthr
                                )
                                
                                st.success("✅ Advanced analysis completed!")
                                st.info("📊 Check the figures directory for detailed visualizations")
                            else:
                                st.error("❌ Advanced analysis failed")
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        # Clear session state
                        if 'current_ride_id' in st.session_state:
                            del st.session_state.current_ride_id
                        if 'current_file_path' in st.session_state:
                            del st.session_state.current_file_path
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        progress_bar.progress(0)
    
    # Tab 2: Ride History
    with tab2:
        st.header("📈 Ride History")
        
        # Get available rides
        available_rides = data_manager.get_available_rides()
        
        if not available_rides:
            st.info("No rides available. Upload a FIT file to get started.")
        else:
            # Ride selection
            selected_ride = st.selectbox(
                "Select a ride to view:",
                available_rides,
                help="Choose a ride from your history"
            )
            
            if selected_ride:
                # Get comprehensive ride data
                ride_data = data_manager.get_ride_data(selected_ride)
                
                # Display ride status
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("FIT File", "✅ Available" if ride_data['has_fit_file'] else "❌ Missing")
                with col2:
                    st.metric("In History", "✅ Yes" if ride_data['in_history'] else "❌ No")
                with col3:
                    st.metric("Analysis", "✅ Available" if ride_data['analysis_available'] else "❌ None")
                
                # File integrity check
                if ride_data['has_fit_file']:
                    is_valid, integrity_msg = data_manager.validate_file_integrity(selected_ride)
                    if is_valid:
                        st.success(f"✅ {integrity_msg}")
                    else:
                        st.warning(f"⚠️ {integrity_msg}")
                
                # Display history data if available
                if ride_data['in_history'] and ride_data['history_data']:
                    st.markdown("---")
                    st.subheader("📊 Ride Summary")
                    
                    history_data = ride_data['history_data']
                    
                    # Create summary display
                    summary_cols = st.columns(4)
                    
                    with summary_cols[0]:
                        st.metric("Duration", format_duration(history_data.get('duration_min', 0)))
                        st.metric("Distance", format_distance(history_data.get('distance_km', 0)))
                    
                    with summary_cols[1]:
                        st.metric("Avg Power", format_power(history_data.get('avg_power_W', 0)))
                        st.metric("NP", format_power(history_data.get('NP_W', 0)))
                    
                    with summary_cols[2]:
                        st.metric("IF", format_percentage(history_data.get('IF', 0) * 100))
                        st.metric("TSS", format_number(history_data.get('TSS', 0)))
                    
                    with summary_cols[3]:
                        st.metric("Avg HR", format_number(history_data.get('avg_hr', 0)))
                        st.metric("Max HR", format_number(history_data.get('max_hr', 0)))
    
    # Tab 3: Re-analyze
    with tab3:
        st.header("🔍 Re-analyze Existing Ride")
        
        available_rides = data_manager.get_available_rides()
        
        if not available_rides:
            st.info("No rides available for re-analysis.")
        else:
            # Ride selection for re-analysis
            reanalyze_ride = st.selectbox(
                "Select ride to re-analyze:",
                available_rides,
                help="Choose a ride to run new analysis on"
            )
            
            if reanalyze_ride:
                ride_data = data_manager.get_ride_data(reanalyze_ride)
                
                # Check if FIT file is available
                if not ride_data['has_fit_file']:
                    st.warning(f"❌ No FIT file available for '{reanalyze_ride}'. Please upload the file first.")
                else:
                    st.success(f"✅ FIT file available for '{reanalyze_ride}'")
                    
                    # Re-analysis options
                    st.subheader("Analysis Options")
                    
                    reanalysis_type = st.selectbox(
                        "Analysis Type",
                        ["Basic", "Advanced", "Both"],
                        key="reanalysis_type"
                    )
                    
                    reanalysis_save_figures = st.checkbox("Save Figures", value=True, key="reanalysis_save")
                    
                    if st.button("🔄 Run Re-analysis", type="primary"):
                        file_path = ride_data['fit_file_path']
                        
                        with st.spinner(f"Re-analyzing {reanalyze_ride}..."):
                            try:
                                # Basic re-analysis
                                if reanalysis_type in ["Basic", "Both"]:
                                    results, df = run_basic_analysis(
                                        file_path, 
                                        ftp=ftp, 
                                        lthr=lthr, 
                                        save_figures=reanalysis_save_figures, 
                                        analysis_id=f"{reanalyze_ride}_reanalysis"
                                    )
                                    
                                    if results:
                                        data_manager.save_analysis_results(
                                            reanalyze_ride, "Basic", results, ftp, lthr
                                        )
                                        st.success("✅ Basic re-analysis completed!")
                                        display_basic_results(results)
                                
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
                                            reanalyze_ride, "Advanced", {"status": "completed"}, ftp, lthr
                                        )
                                        
                                        st.success("✅ Advanced re-analysis completed!")
                                        st.info("📊 Check the figures directory for updated visualizations")
                                
                            except Exception as e:
                                st.error(f"❌ Re-analysis failed: {str(e)}")
    
    # Tab 4: System Info
    with tab4:
        st.header("⚙️ System Information")
        
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