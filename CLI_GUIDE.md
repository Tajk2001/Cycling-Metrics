# 🚴 Command-Line Usage Guide

## 📋 Quick Start

### Prerequisites
```bash
# Activate virtual environment
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Basic Usage
```bash
python cli.py --file <path_to_fit_file> [options]
```

## 🎯 Common Examples

### 1. **Basic Analysis**
```bash
python cli.py --file cache/Into_the_clouds.fit
```
*Uses default settings: FTP=250W, Max HR=195, Weight=70kg*

### 2. **Custom Athlete Settings**
```bash
python cli.py --file cache/Into_the_clouds.fit \
  --ftp 290 \
  --max-hr 195 \
  --weight 75 \
  --name "Cyclist"
```

### 3. **Different Output Directory**
```bash
python cli.py --file cache/Engine_on_nobody_s_driving.fit \
  --output-dir my_analysis \
  --ftp 280
```

### 4. **Display Only (No Save)**
```bash
python cli.py --file cache/Into_the_clouds.fit \
  --no-save \
  --ftp 290
```

### 5. **Custom Analysis ID**
```bash
python cli.py --file cache/Into_the_clouds.fit \
  --analysis-id "my_custom_analysis" \
  --ftp 290
```

## 📊 Parameter Reference

### **Required Parameters**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `--file, -f` | Path to FIT/TCX/CSV file | `cache/Into_the_clouds.fit` |

### **Optional Parameters**
| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `--ftp` | 250 | Functional Threshold Power (W) | `--ftp 290` |
| `--max-hr` | 195 | Maximum Heart Rate (bpm) | `--max-hr 195` |
| `--rest-hr` | 51 | Resting Heart Rate (bpm) | `--rest-hr 51` |
| `--weight` | 70 | Athlete Weight (kg) | `--weight 75` |
| `--height` | 175 | Athlete Height (cm) | `--height 175` |
| `--name` | "Cyclist" | Athlete Name | `--name "Cyclist"` |
| `--output-dir` | "figures" | Output Directory | `--output-dir my_analysis` |
| `--no-save` | False | Don't save figures | `--no-save` |
| `--analysis-id` | Auto-generated | Custom Analysis ID | `--analysis-id "test_1"` |

## 🔧 Advanced Usage

### **Batch Processing Multiple Files**
```bash
# Process all FIT files in cache directory
for file in cache/*.fit; do
  python cli.py --file "$file" --ftp 290 --name "Cyclist"
done
```

### **Different Settings for Different Rides**
```bash
# Easy ride with lower FTP
python cli.py --file cache/Into_the_clouds.fit --ftp 250

# Hard ride with higher FTP
python cli.py --file cache/Engine_on_nobody_s_driving.fit --ftp 290
```

### **Testing Different Parameters**
```bash
# Test with different FTP values
python cli.py --file cache/Into_the_clouds.fit --ftp 250 --output-dir test_ftp_250
python cli.py --file cache/Into_the_clouds.fit --ftp 290 --output-dir test_ftp_290
python cli.py --file cache/Into_the_clouds.fit --ftp 320 --output-dir test_ftp_320
```

## 📈 Output Files

### **Generated Files**
- **Dashboard**: `{analysis_id}_dashboard.png/svg`
- **Fatigue Patterns**: `{analysis_id}_fatigue_patterns.png/svg`
- **Heat Stress**: `{analysis_id}_heat_stress.png/svg`
- **Power/HR Efficiency**: `{analysis_id}_power_hr_efficiency.png/svg`
- **Variable Relationships**: `{analysis_id}_variable_relationships.png/svg`
- **Torque Analysis**: `{analysis_id}_torque.png/svg`
- **W' Balance**: `{analysis_id}_w_prime_balance.png/svg`
- **Lactate Estimation**: `{analysis_id}_lactate.png/svg`
- **Multi-Axis Analysis**: `{analysis_id}_multi_axis_analysis.html/png`

### **Analysis Pipeline**
1. **📥 Data Ingestion** - Load and validate FIT file
2. **🔍 Initial Checks** - Verify data quality
3. **🧹 Data Cleaning** - Handle missing values and outliers
4. **🧮 Feature Engineering** - Calculate metrics and zones
5. **📊 Data Aggregation** - Create summaries
6. **📈 Visualization** - Generate charts and plots
7. **🧪 Modeling** - CP estimation and advanced analysis

## 🎯 Key Metrics Output

### **Power Metrics**
- **Avg Power**: Average power output
- **Max Power**: Maximum power achieved
- **Normalized Power**: 30-second rolling average
- **Intensity Factor**: NP/FTP ratio
- **Training Stress Score**: Workout intensity measure
- **Variability Index**: NP/AP ratio

### **Physiological Metrics**
- **Critical Power**: Estimated from power-duration curve
- **W' Balance**: Anaerobic work capacity
- **Heart Rate**: Average, max, and zones
- **Lactate**: Estimated from power relationship

### **Performance Metrics**
- **Power Bests**: 1s, 5s, 10s, 30s, 1min, 3min, 5min, 8min, 10min, 12min, 20min, 60min, 90min
- **Duration**: Total ride time
- **Distance**: Total distance covered
- **Elevation**: Total elevation gain
- **Energy**: Total work done (kJ)

## 🚨 Troubleshooting

### **Common Issues**

#### **1. Virtual Environment Not Activated**
```bash
# Error: ModuleNotFoundError: No module named 'pandas'
# Solution: Activate virtual environment
source venv/bin/activate
```

#### **2. File Not Found**
```bash
# Error: File 'activity.fit' not found
# Solution: Check file path and use correct path
python cli.py --file cache/Into_the_clouds.fit
```

#### **3. Insufficient Data**
```bash
# Error: No power data found
# Solution: Ensure FIT file contains power data
# Check file with: fitparse or similar tool
```

#### **4. Memory Issues**
```bash
# For large files, use --no-save to reduce memory usage
python cli.py --file large_file.fit --no-save
```

### **Debug Mode**
```bash
# Enable detailed logging
export PYTHONPATH=.
python -u run.py --file cache/Into_the_clouds.fit --ftp 290
```

## 📊 Performance Tips

### **1. Use Cached Files**
```bash
# Use files already in cache directory
python cli.py --file cache/Into_the_clouds.fit
```

### **2. Custom Output Directory**
```bash
# Organize outputs by date or analysis type
python cli.py --file cache/Into_the_clouds.fit \
  --output-dir "analysis_$(date +%Y%m%d)" \
  --ftp 290
```

### **3. Batch Processing**
```bash
# Process multiple files with same settings
for file in cache/*.fit; do
  basename=$(basename "$file" .fit)
  python cli.py --file "$file" \
    --output-dir "batch_analysis/$basename" \
    --ftp 290 \
    --name "Cyclist"
done
```

## 🎯 Integration with Dashboard

### **Command Line → Dashboard**
1. Run analysis via command line
2. Files are automatically available in dashboard
3. View results in "📈 Analysis" tab

### **Dashboard → Command Line**
1. Upload files via dashboard
2. Use cached files in command line
3. Reference: `cache/{filename}.fit`

## 📝 Example Workflows

### **Workflow 1: Quick Analysis**
```bash
# Fast analysis with defaults
python cli.py --file cache/Into_the_clouds.fit
```

### **Workflow 2: Detailed Analysis**
```bash
# Comprehensive analysis with custom settings
python cli.py --file cache/Into_the_clouds.fit \
  --ftp 290 \
  --max-hr 195 \
  --weight 75 \
  --name "Cyclist" \
  --output-dir "detailed_analysis"
```

### **Workflow 3: Parameter Testing**
```bash
# Test different FTP values
for ftp in 250 270 290 310; do
  python cli.py --file cache/Into_the_clouds.fit \
    --ftp $ftp \
    --output-dir "ftp_test_$ftp"
done
```

### **Workflow 4: Batch Processing**
```bash
# Process all files with same settings
for file in cache/*.fit; do
  basename=$(basename "$file" .fit)
  python cli.py --file "$file" \
    --ftp 290 \
    --name "Cyclist" \
    --output-dir "batch/$basename"
done
```

---

**💡 Pro Tip**: Use the command line for batch processing and automation, and the dashboard for interactive analysis and visualization! 