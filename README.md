# 🚴 Cycling Analysis System

**A comprehensive cycling power analysis system that automatically detects training intervals using machine learning and provides detailed sprint and interval analysis.**

## 🏗️ Project Structure

```
cycling_analysis/
├── SprintV1.py               # 🧠 Main analysis script - use this!
├── interval_detection.py      # 🤖 ML interval detection
├── IntervalML.py             # 🔧 ML utilities and lap detection
├── simple_bulk_training.py   # 🎯 Model training script
├── trained_models/           # 🎯 Pre-trained ML models
│   └── simple_comprehensive_model_20250820_125836.pkl
├── requirements.txt           # 📦 Python dependencies
└── README.md                 # 📖 This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Analyze a Ride
```bash
python SprintV1.py
```

**That's it!** The system will analyze your Test2 file with default parameters.

## 🔧 Customization

To analyze a different file or change parameters, edit the top of `SprintV1.py`:

```python
# User Parameters
file_path = ""  # Set to your FIT file path or leave empty for interactive input
FTP = 290                              # Your FTP in watts
LTHR = 181                             # Your LTHR in bpm
rider_mass_kg = 52.5                   # Your mass in kg
crank_length_mm = 165                  # Your crank length in mm
```

## 🎯 What It Does

- **🏃 Sprint Analysis**: Finds your best 10-second efforts
- **🤖 ML Interval Detection**: Automatically detects training intervals
- **📊 Power Metrics**: Comprehensive power analysis
- **📈 Evolution Analysis**: Tracks interval performance over time
- **📱 Interactive Dashboard**: Beautiful visualization of results

## 🎉 Ready to Go!

Just run `python SprintV1.py` and get comprehensive insights into your training!
