import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime
from data_manager import CyclingDataManager
from enhanced_cycling_analysis import CyclingAnalyzer

# Page configuration
st.set_page_config(
    page_title="Cycling Analysis",
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
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:.{decimals}f}"

def format_duration(minutes):
    if pd.isna(minutes):
        return "N/A"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"

def format_power(power):
    if pd.isna(power):
        return "N/A"
    return f"{int(power)}W"

def format_distance(km):
    if pd.isna(km):
        return "N/A"
    return f"{km:.1f}km"

# Main dashboard
def main():
    st.title("🚴 Cycling Analysis")
    
    # Sidebar - Simple settings
    with st.sidebar:
        st.header("⚙️ Settings")
        ftp = st.number_input("FTP (W)", min_value=100, max_value=500, value=250, step=5)
        lthr = st.number_input("LTHR (bpm)", min_value=120, max_value=200, value=160, step=1)
        
        st.markdown("---")
        
        # Quick stats
        status = data_manager.get_system_status()
        st.metric("Activities", status['total_rides'])
        st.metric("Analyses", status['analysis_entries'])
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "📊 Analysis", "📈 Results"])
    
    # Tab 1: Upload
    with tab1:
        st.header("📁 Upload Activity")
        
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
        
        # Show uploaded files
        st.markdown("---")
        st.subheader("📋 Your Activities")
        
        available_rides = data_manager.get_available_rides()
        if available_rides:
            for ride in available_rides:
                ride_data = data_manager.get_ride_data(ride)
                col1, col2, col3 = st.columns([3, 1, 1])
                
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
        else:
            st.info("No activities uploaded yet.")
    
    # Tab 2: Analysis
    with tab2:
        st.header("📊 Analysis")
        
        available_rides = data_manager.get_available_rides()
        
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
                    
                    # Auto-run analysis if not already done
                    if not ride_data['analysis_available']:
                        if st.button("🚀 Run Analysis", type="primary"):
                            with st.spinner("Analyzing..."):
                                try:
                                    analyzer = CyclingAnalyzer(
                                        save_figures=True, 
                                        ftp=ftp, 
                                        save_dir="figures",
                                        analysis_id=selected_ride
                                    )
                                    
                                    file_path = ride_data['fit_file_path']
                                    
                                    if analyzer.load_fit_file(file_path):
                                        analyzer.clean_and_smooth_data()
                                        analyzer.calculate_metrics()
                                        analyzer.print_summary()
                                        analyzer.create_dashboard()
                                        
                                        # Save results
                                        data_manager.save_analysis_results(
                                            selected_ride, "Advanced", {"status": "completed"}, ftp, lthr
                                        )
                                        
                                        st.success("✅ Analysis completed!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Analysis failed")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                    else:
                        st.success("✅ Analysis already completed")
                        
                        if st.button("🔄 Re-run Analysis", type="secondary"):
                            with st.spinner("Re-analyzing..."):
                                try:
                                    analyzer = CyclingAnalyzer(
                                        save_figures=True, 
                                        ftp=ftp, 
                                        save_dir="figures",
                                        analysis_id=selected_ride
                                    )
                                    
                                    file_path = ride_data['fit_file_path']
                                    
                                    if analyzer.load_fit_file(file_path):
                                        analyzer.clean_and_smooth_data()
                                        analyzer.calculate_metrics()
                                        analyzer.print_summary()
                                        analyzer.create_dashboard()
                                        
                                        # Save results
                                        data_manager.save_analysis_results(
                                            selected_ride, "Advanced", {"status": "completed"}, ftp, lthr
                                        )
                                        
                                        st.success("✅ Re-analysis completed!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Re-analysis failed")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
    
    # Tab 3: Results
    with tab3:
        st.header("📈 Results")
        
        available_rides = data_manager.get_available_rides()
        
        if not available_rides:
            st.info("No activities available.")
        else:
            selected_ride_results = st.selectbox(
                "Select activity to view results:",
                available_rides,
                key="results_select"
            )
            
            if selected_ride_results:
                ride_data = data_manager.get_ride_data(selected_ride_results)
                
                # Show basic metrics
                if ride_data['in_history'] and ride_data['history_data']:
                    history = ride_data['history_data']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Duration", format_duration(history.get('duration_min', 0)))
                        st.metric("Distance", format_distance(history.get('distance_km', 0)))
                    
                    with col2:
                        st.metric("Avg Power", format_power(history.get('avg_power_W', 0)))
                        st.metric("Max Power", format_power(history.get('max_power_W', 0)))
                    
                    with col3:
                        st.metric("TSS", f"{history.get('TSS', 0):.0f}")
                        st.metric("IF", f"{history.get('IF', 0)*100:.1f}%")
                    
                    with col4:
                        st.metric("Avg HR", f"{history.get('avg_hr', 0):.0f} bpm")
                        st.metric("Max HR", f"{history.get('max_hr', 0):.0f} bpm")
                
                # Show figures
                st.markdown("---")
                st.subheader("📊 Analysis Figures")
                
                figures_dir = pathlib.Path("figures")
                ride_figures = []
                
                if figures_dir.exists():
                    for figure_file in figures_dir.glob(f"{selected_ride_results}_*"):
                        if figure_file.suffix in ['.png', '.svg']:
                            ride_figures.append(figure_file)
                
                if ride_figures:
                    # Group figures by type
                    figure_types = {}
                    for fig in ride_figures:
                        fig_name = fig.stem.replace(f"{selected_ride_results}_", "")
                        if fig_name not in figure_types:
                            figure_types[fig_name] = []
                        figure_types[fig_name].append(fig)
                    
                    # Display figures
                    for fig_type, figures in figure_types.items():
                        st.markdown(f"**{fig_type.replace('_', ' ').title()}**")
                        
                        cols = st.columns(min(len(figures), 2))
                        for i, fig in enumerate(figures):
                            with cols[i % len(cols)]:
                                st.image(str(fig), caption=fig.name, use_column_width=True)
                        
                        st.markdown("---")
                else:
                    st.info("No analysis figures found. Run analysis first.")
                
                # Delete option
                st.markdown("---")
                st.subheader("🗑️ Delete Activity")
                
                if st.button("🗑️ Delete This Activity", type="secondary"):
                    with st.spinner("Deleting..."):
                        success, message = data_manager.delete_ride(selected_ride_results)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.warning(message)

if __name__ == "__main__":
    main() 