# Cycling Analysis Project

This repository contains cycling performance analysis tools and systems.

## Project Structure

### 🚴‍♂️ **cycling_tracker_system/** - New Comprehensive System
A complete cycling performance analysis system with lap-based interval detection, Andy Coggan power zones, and Normalized Power (NP) classification.

**Key Features:**
- FIT file parsing and data extraction
- Lap-based interval detection using FIT file markers
- Andy Coggan's 7-zone power classification system
- Normalized Power (NP) calculation for accurate training load assessment
- Interactive web dashboard with Dash
- Comprehensive metrics calculation
- Data storage and export capabilities

**Quick Start:**
```bash
cd cycling_tracker_system
pip install -r ../requirements.txt
pip install -e .
python run_app.py  # Launch dashboard
python example_usage.py  # See examples
```

See `cycling_tracker_system/README.md` for detailed documentation.

### 📊 **Original Analysis Scripts** - Legacy Tools
The original cycling analysis scripts and tools:

- **SprintV1.py** - Original FIT file analysis script with lap extraction
- **interval_detection.py** - Basic interval detection algorithms  
- **IntervalML.py** - Machine learning-based interval detection
- **simple_bulk_training.py** - Bulk training data processing
- **trained_models/** - Pre-trained ML models

### 🔧 **Development Environment**
- **venv/** - Python virtual environment
- **requirements.txt** - Python dependencies
- **README.md** - This file

## Getting Started

### For the New System (Recommended)
```bash
cd cycling_tracker_system
pip install -r ../requirements.txt
pip install -e .
python example_usage.py
```

### For Original Scripts
```bash
source venv/bin/activate
python SprintV1.py  # Original analysis script
```

## System Comparison

| Feature | New System | Original Scripts |
|---------|------------|------------------|
| **Architecture** | Modular package | Single scripts |
| **Interval Detection** | Lap-based (FIT markers) | ML algorithms |
| **Power Zones** | Andy Coggan (7 zones) | Basic zones |
| **Zone Classification** | Normalized Power (NP) | Average power |
| **User Interface** | Web dashboard + CLI | Command line only |
| **Data Storage** | Multiple formats | Basic CSV |
| **Testing** | Comprehensive test suite | No tests |
| **Documentation** | Full documentation | Minimal |

## Recommendations

- **Use the new system** (`cycling_tracker_system/`) for comprehensive analysis
- **Reference original scripts** for specific algorithms or legacy approaches
- **Both systems** can coexist and be used independently

## License

This project is licensed under the MIT License.