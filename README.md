# 🚴 Enhanced Cycling Analysis Dashboard

A comprehensive, error-proof cycling data analysis system with robust data management and simplified workflows.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Dashboard
```bash
python start.py
```

**Or manually:**
```bash
source venv/bin/activate  # macOS/Linux
streamlit run enhanced_dashboard.py
```

## 📊 Features

- **📁 Upload & Analyze**: Upload FIT files and run analysis
- **📈 Ride History**: View existing rides and data
- **🔍 Re-analyze**: Re-run analysis on existing rides
- **⚙️ System Info**: System status and data management
- **🗑️ Ride Management**: Delete individual rides or clear all data

## 🎯 Usage

1. **Upload FIT file** in "Upload & Analyze" tab
2. **Set FTP/LTHR** values in sidebar
3. **Choose analysis type** (Basic/Advanced/Both)
4. **View results** immediately
5. **Re-analyze** existing rides as needed
6. **Delete rides** when no longer needed

## 🗑️ Ride Management

### Delete Individual Ride
- **Ride History tab**: Select a ride and click "Delete This Ride"
- **System Info tab**: Use the "Delete Specific Ride" section
- **Confirmation required** to prevent accidental deletion

### Clear All Rides
- **System Info tab**: Use the "Clear All Rides" section
- **Double confirmation** required for safety
- **Automatic backup** created before clearing

## 📁 File Structure

```
cycling_analysis/
├── enhanced_dashboard.py    # Main application
├── data_manager.py         # Data management
├── enhanced_cycling_analysis.py  # Analysis engine
├── app.py                  # Basic analysis
├── start.py               # Startup script
├── data/                  # Core data files
├── cache/                 # FIT file storage
├── figures/               # Analysis visualizations
└── README_ENHANCED.md     # Detailed documentation
```

## 🔧 Troubleshooting

### "streamlit: command not found"
```bash
source venv/bin/activate
pip install streamlit
```

### "No FIT data available"
Upload the FIT file in "Upload & Analyze" tab first.

### Data issues
Check "System Info" tab for status and use export/import features.

### Delete functionality
- Individual rides: Use Ride History or System Info tabs
- Clear all: Use System Info tab with double confirmation
- Backups: Automatically created before clearing all rides

## 📖 Documentation

- **README_ENHANCED.md**: Comprehensive documentation
- **SOLUTION_SUMMARY.md**: Implementation details

---

**Version**: 2.0.0  
**Compatibility**: Python 3.8+, Streamlit 1.25+ 