# 🚴 Enhanced Cycling Analysis Dashboard

A comprehensive, error-proof cycling data analysis system with robust data management, clean storage, and simplified workflows.

## ✨ Key Features

### 🔒 Error-Proof Data Management
- **Automatic data validation** and integrity checking
- **Corruption detection** with automatic backup and recovery
- **File hash verification** to ensure data integrity
- **Organized storage structure** with clear separation of concerns

### 📁 Clean Storage Architecture
```
cycling_analysis/
├── data/                    # Core data files
│   ├── ride_history.csv     # Ride analysis results
│   ├── analysis_history.csv # Analysis metadata
│   ├── file_registry.json   # FIT file tracking
│   └── settings.json        # System configuration
├── cache/                   # FIT file storage
│   └── *.fit               # Uploaded FIT files
├── figures/                 # Analysis visualizations
│   └── *.png, *.svg        # Generated charts
└── enhanced_dashboard.py    # Main application
```

### 🎯 Simplified Workflows
- **One-click upload and analysis**
- **Automatic file organization**
- **Smart caching and retrieval**
- **Comprehensive ride history**
- **Easy re-analysis capabilities**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Migrate Existing Data (if any)
```bash
python migrate_data.py
```

### 3. Launch the Dashboard
```bash
streamlit run enhanced_dashboard.py
```

## 📊 Dashboard Features

### 📁 Upload & Analyze Tab
- **Drag-and-drop FIT file upload**
- **Automatic file validation**
- **Real-time progress tracking**
- **Basic and Advanced analysis options**
- **Immediate results display**

### 📈 Ride History Tab
- **Comprehensive ride overview**
- **File availability status**
- **Data integrity validation**
- **Historical metrics display**

### 🔍 Re-analyze Tab
- **Re-run analysis on existing rides**
- **Updated parameters support**
- **Comparison capabilities**
- **Figure regeneration**

### ⚙️ System Info Tab
- **System status monitoring**
- **Data export/import**
- **File registry management**
- **Storage statistics**

## 🔧 Data Management

### File Organization
The system automatically organizes your data:

- **FIT files** → `cache/` directory with clean naming
- **Analysis results** → `data/ride_history.csv`
- **Analysis metadata** → `data/analysis_history.csv`
- **File tracking** → `data/file_registry.json`
- **Generated figures** → `figures/` directory

### Error Handling
- **Automatic corruption detection**
- **Backup creation before reset**
- **File integrity validation**
- **Graceful error recovery**

### Data Validation
- **SHA-256 file hashing**
- **Automatic duplicate detection**
- **Missing file identification**
- **Data consistency checking**

## 🛠️ Advanced Features

### Smart Caching
- **Session-based file caching**
- **Persistent file registry**
- **Automatic cache cleanup**
- **Memory-efficient storage**

### Analysis Types

#### Basic Analysis
- Core power metrics (NP, IF, TSS)
- Heart rate analysis
- Zone distribution
- Session summary

#### Advanced Analysis
- Physiological modeling
- Lactate estimation
- Fatigue patterns
- Heat stress analysis
- W' balance tracking
- Torque analysis

### Data Export/Import
- **Complete data backup**
- **Selective data export**
- **Import from backup**
- **Cross-system compatibility**

## 🔍 Troubleshooting

### Common Issues

#### "No FIT data available for this ride"
**Solution**: Upload the FIT file in the "Upload & Analyze" tab first, then re-analyze.

#### "File integrity check failed"
**Solution**: The system will automatically create a backup and reset corrupted files.

#### "Missing FIT files for rides"
**Solution**: Run the migration script or manually upload missing FIT files.

### Data Recovery
1. Check the `data/backup_*` directories for automatic backups
2. Use the export/import functionality to restore data
3. Re-upload FIT files if needed

## 📈 Performance Tips

### Optimal Settings
- **FTP**: Set your current Functional Threshold Power
- **LTHR**: Set your Lactate Threshold Heart Rate
- **Save Figures**: Enable for detailed visualizations

### Storage Management
- **Regular cleanup**: Use the "Cleanup Orphaned Files" feature
- **Backup strategy**: Export data periodically
- **File organization**: Keep FIT files in a dedicated folder

## 🔄 Migration from Old System

If you have existing data from the previous system:

1. **Run migration script**:
   ```bash
   python migrate_data.py
   ```

2. **Verify data integrity**:
   - Check the migration summary
   - Review any warnings about missing files
   - Upload missing FIT files if needed

3. **Start using the new dashboard**:
   ```bash
   streamlit run enhanced_dashboard.py
   ```

## 📋 File Requirements

### FIT File Format
- **Extension**: `.fit` or `.FIT`
- **Source**: Garmin, Wahoo, or other cycling devices
- **Size**: Typically 1-10 MB per file
- **Content**: Power, heart rate, GPS, and other sensor data

### Supported Metrics
- **Power** (watts)
- **Heart Rate** (bpm)
- **Cadence** (rpm)
- **Speed** (km/h)
- **Distance** (km)
- **Altitude** (m)
- **GPS coordinates**

## 🎯 Best Practices

### File Naming
- Use descriptive names (e.g., `Truckee_Gravel_6th.fit`)
- Avoid special characters
- Include date if helpful

### Analysis Workflow
1. **Upload FIT file**
2. **Set FTP/LTHR values**
3. **Choose analysis type**
4. **Review results**
5. **Save figures if needed**

### Data Maintenance
- **Regular backups**: Export data monthly
- **File organization**: Keep FIT files organized
- **System updates**: Check for updates regularly

## 🔧 Configuration

### Custom Settings
Edit `data/settings.json` for advanced configuration:

```json
{
  "version": "2.0.0",
  "data_structure": "organized",
  "directories": {
    "data": "data",
    "cache": "cache",
    "figures": "figures"
  }
}
```

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Custom port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Custom address (default: localhost)

## 📞 Support

### Getting Help
1. **Check the System Info tab** for status
2. **Review error messages** in the console
3. **Export data** before troubleshooting
4. **Check file permissions** and disk space

### Common Solutions
- **Restart the application** if session issues occur
- **Clear browser cache** if UI problems persist
- **Check file permissions** if upload fails
- **Verify FIT file format** if analysis fails

## 🚀 Future Enhancements

### Planned Features
- **Cloud storage integration**
- **Multi-athlete support**
- **Advanced analytics dashboard**
- **Mobile app companion**
- **API for external integrations**

### Contributing
- **Report bugs** with detailed error messages
- **Suggest features** with use case descriptions
- **Share feedback** on user experience
- **Contribute code** through pull requests

---

**Version**: 2.0.0  
**Last Updated**: December 2024  
**Compatibility**: Python 3.8+, Streamlit 1.25+ 