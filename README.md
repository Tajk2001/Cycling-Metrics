# 🚴 Cycling Analysis Project

A comprehensive cycling data analysis tool that processes FIT files and provides advanced physiological and performance insights.

## 📋 Features

- **📊 Advanced Analytics**: Power analysis, heart rate zones, training stress scores
- **🎨 Rich Visualizations**: Interactive dashboards and professional charts
- **📈 Historical Tracking**: Long-term performance trends and analysis
- **⚙️ Customizable Settings**: Personalized athlete profiles and thresholds
- **🔄 Real-time Processing**: Live data analysis and visualization
- **💾 Smart Caching**: Efficient file management and storage

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd cycling_analysis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
# Launch web dashboard (recommended)
python start.py

# Or launch command line interface
python start.py --cli

# Or run directly
streamlit run dashboard.py
```

## 🎯 Usage

### Web Dashboard
- Upload FIT files through the web interface
- View comprehensive analysis results
- Track historical performance trends
- Manage athlete settings and preferences

### Command Line
```bash
# Basic analysis
python cli.py --file cache/activity.fit

# Custom settings
python cli.py --file cache/activity.fit --ftp 290 --name "Cyclist"

# Batch processing
for file in cache/*.fit; do
  python cli.py --file "$file" --ftp 290 --name "Cyclist"
done
```

## 📊 Analysis Features

### **Power Analysis**
- Functional Threshold Power (FTP) calculations
- Normalized Power and Intensity Factor
- Training Stress Score (TSS)
- Power-duration curve analysis
- Power bests (1s, 5s, 10s, 30s, 1min, 3min, 5min, 8min, 10min, 12min, 20min, 60min, 90min)

### **Physiological Metrics**
- Critical Power estimation
- W' balance analysis
- Heart rate zone analysis
- Lactate threshold estimation
- Heat stress analysis

### **Performance Tracking**
- Historical trend analysis
- Performance distribution charts
- Training load monitoring
- Efficiency metrics

### **Advanced Visualizations**
- Professional dual-axis charts
- Fatigue pattern analysis
- Power/HR efficiency plots
- Variable relationship analysis
- Torque analysis

## 🛠️ Project Structure

```
cycling_analysis/
├── dashboard.py             # Streamlit web interface
├── analyzer.py              # Core analysis engine
├── data_manager.py          # Data management and caching
├── critical_power.py        # CP/W' analysis
├── cli.py                   # Command-line interface
├── requirements.txt         # Python dependencies
├── cache/                   # Uploaded FIT files
├── data/                    # Settings and history
├── figures/                 # Generated visualizations
└── venv/                   # Virtual environment
```

## 📈 Output Files

Each analysis generates:
- **Dashboard**: Comprehensive overview charts
- **Fatigue Patterns**: Performance degradation analysis
- **Heat Stress**: Temperature and effort correlation
- **Power/HR Efficiency**: Physiological efficiency metrics
- **Variable Relationships**: Multi-metric correlations
- **Torque Analysis**: Force and cadence relationships
- **W' Balance**: Anaerobic work capacity tracking
- **Lactate Estimation**: Metabolic stress indicators
- **Multi-Axis Analysis**: Professional dual-axis charts

## ⚙️ Configuration

### Athlete Settings
- **FTP**: Functional Threshold Power (watts)
- **Max HR**: Maximum heart rate (bpm)
- **Rest HR**: Resting heart rate (bpm)
- **Weight**: Athlete weight (kg)
- **Height**: Athlete height (cm)
- **Name**: Athlete identifier

### Analysis Parameters
- **Power Zones**: Customizable training zones
- **HR Zones**: Heart rate training zones
- **Output Directory**: Custom save locations
- **Cache Management**: File storage options

## 🔧 Advanced Features

### **Cache Management**
- Automatic FIT file caching
- Analysis result storage
- Session state management
- Selective cache clearing

### **Data Export**
- CSV export of analysis results
- Summary statistics export
- Historical data export
- Custom date range filtering

### **Batch Processing**
- Multiple file processing
- Parameter testing workflows
- Automated analysis pipelines
- Progress tracking

## 🚨 Troubleshooting

### **Common Issues**

#### Virtual Environment
```bash
# Activate virtual environment
source venv/bin/activate

# Verify installation
python -c "import pandas, streamlit, matplotlib; print('OK')"
```

#### File Upload Issues
```bash
# Check file permissions
ls -la cache/

# Clear cache if needed
python -c "from data_manager import CyclingDataManager; dm = CyclingDataManager(); dm.clear_cache('all')"
```

#### Memory Issues
```bash
# Use --no-save for large files
python run.py --file large_file.fit --no-save
```

## 📚 Documentation

- **Command Line Usage**: See `COMMAND_LINE_USAGE.md`
- **Warp Terminal Setup**: See `WARP_SETUP.md`
- **API Documentation**: Inline code documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with Streamlit for web interface
- Uses pandas and numpy for data processing
- Matplotlib and seaborn for visualizations
- FIT file parsing with fitparse library

---

**🚀 Ready to analyze your cycling performance!** 