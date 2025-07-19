# 🚴‍♂️ Advanced Cycling Analysis Dashboard

A comprehensive cycling analysis tool that provides advanced metrics, interactive visualizations, and detailed insights from FIT files. Built with Streamlit and Altair for a modern, interactive experience.

## ✨ Features

### 📊 Interactive Dashboard
- **Real-time FIT file processing** - Upload and analyze cycling data instantly
- **Interactive Altair charts** - Zoom, pan, and explore your data
- **Advanced metrics** - Power zones, heart rate analysis, efficiency metrics
- **Multiple analysis views** - 8 different specialized analysis tabs

### 📈 Analysis Tabs
1. **Dashboard Overview** - Power and heart rate profiles over time
2. **Power/HR Efficiency** - Relationship between power output and heart rate
3. **Fatigue Patterns** - 5-minute rolling power averages to identify fatigue
4. **Heat Stress** - Heat stress index over time
5. **Lactate** - Estimated lactate curves based on power output
6. **Torque** - Torque analysis throughout the ride
7. **Variable Relationships** - Multi-dimensional analysis of power, cadence, and HR
8. **W' Prime Balance** - Anaerobic capacity tracking

### 🎯 Advanced Metrics
- **Functional Threshold Power (FTP)** analysis
- **Normalized Power (NP)** calculations
- **Training Stress Score (TSS)** estimation
- **Power zones** and time-in-zone analysis
- **Heart rate zones** and cardiovascular strain
- **Efficiency metrics** and performance trends

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tajk2001/Cycling-Metrics.git
   cd cycling_analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the interactive dashboard**
   ```bash
   streamlit run altair_full_advanced_dashoboard.py
   ```

### Usage

1. **Upload your FIT file** using the file uploader in the sidebar
2. **Adjust athlete parameters** (FTP, weight, max HR, rest HR) in the sidebar
3. **Explore the analysis** using the interactive tabs
4. **Interact with charts** - zoom, pan, hover for details

## 📁 Project Structure

```
cycling_analysis/
├── altair_full_advanced_dashoboard.py  # Main interactive dashboard
├── main_dashboard.py                    # Ride history dashboard
├── enhanced_cycling_analysis.py         # Advanced analysis engine
├── app.py                              # Basic analysis app
├── requirements.txt                     # Python dependencies
├── ride_history.csv                    # Sample ride data
└── figures/                            # Generated analysis plots
```

## 🛠️ Available Dashboards

### 1. Advanced Interactive Dashboard (`altair_full_advanced_dashoboard.py`)
- **Purpose**: Deep analysis of individual rides
- **Features**: 8 interactive analysis tabs, real-time FIT processing
- **Best for**: Detailed ride analysis, performance insights

### 2. Ride History Dashboard (`main_dashboard.py`)
- **Purpose**: Overview of multiple rides and trends
- **Features**: Ride comparison, cumulative metrics, performance trends
- **Best for**: Long-term training analysis, progress tracking

### 3. Basic Analysis App (`app.py`)
- **Purpose**: Simple, quick analysis
- **Features**: Basic metrics, zone analysis
- **Best for**: Quick ride summaries

## 📊 Example Analysis

The dashboard provides comprehensive analysis including:

- **Power Analysis**: Average, max, normalized power, power zones
- **Heart Rate Analysis**: Cardiovascular strain, HR zones, efficiency
- **Performance Metrics**: TSS, IF, VI, and training load
- **Advanced Physiology**: Lactate estimation, W' balance, heat stress
- **Technical Analysis**: Torque, cadence efficiency, power distribution

## 🔧 Configuration

### Athlete Parameters
- **FTP (Functional Threshold Power)**: Your 20-minute power threshold
- **Max HR**: Maximum heart rate
- **Rest HR**: Resting heart rate
- **Weight**: Body weight for power-to-weight calculations

### Power Zones
- Z1 (Recovery): 0-55% FTP
- Z2 (Endurance): 55-75% FTP
- Z3 (Tempo): 75-90% FTP
- Z4 (Threshold): 90-105% FTP
- Z5 (VO2max): 105-120% FTP
- Z6 (Anaerobic): 120-150% FTP
- Z7 (Neuromuscular): 150%+ FTP

## 📈 Key Metrics Explained

### Normalized Power (NP)
A weighted average power that emphasizes the physiological cost of variable power outputs. More accurate than average power for training load calculation.

### Training Stress Score (TSS)
A measure of training load that accounts for both intensity and duration. 100 TSS = 1 hour at FTP.

### Intensity Factor (IF)
The ratio of normalized power to FTP. IF > 1.0 indicates supra-threshold training.

### Variability Index (VI)
The ratio of normalized power to average power. Higher VI indicates more variable power output.

## 🎯 Use Cases

### For Cyclists
- Analyze individual ride performance
- Track training load and recovery
- Identify strengths and weaknesses
- Plan training sessions

### For Coaches
- Monitor athlete progress
- Analyze training effectiveness
- Identify performance trends
- Plan training blocks

### For Researchers
- Analyze cycling data
- Study performance patterns
- Validate training methodologies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Interactive charts powered by [Altair](https://altair-viz.github.io/)
- FIT file parsing with [fitparse](https://github.com/dtcooper/python-fitparse)
- Data analysis with [pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/)

---

**Happy analyzing! 🚴‍♂️📊** 