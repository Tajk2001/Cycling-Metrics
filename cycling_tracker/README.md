# Cycling Performance Tracker

A comprehensive cycling performance analysis system built with Dash and Plotly.

## Features

### 🚴‍♂️ Comprehensive Ride Analysis
- Automatic FIT file processing
- Complete ride metrics calculation
- Multi-metric time series visualization
- Power zone analysis

### 📊 Interval Detection & Analysis
- **Lap-based interval detection** (recommended approach)
- Power threshold detection (alternative method)
- Individual interval analysis with evolution tracking
- Interval comparison and ranking
- Power curve generation

### 🔄 Multi-Ride Comparison
- Side-by-side ride comparisons
- Performance trend analysis
- Statistical improvements tracking
- Ride filtering and selection

### 📈 Performance Trends
- Long-term performance tracking
- Seasonal performance patterns
- Goal tracking and progress monitoring
- Training load analysis

## Project Structure

```
cycling_tracker/
├── frontend/           # Dash dashboard application
│   ├── components/     # Individual UI components
│   ├── layouts/        # Page layouts and structure
│   ├── callbacks/      # Interactive callback functions
│   └── utils/          # Helper utilities and formatters
├── backend/            # Data processing and analysis
├── data/               # Data storage and management
└── tests/              # Test suite
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
python -m cycling_tracker.frontend.main_dashboard
```

3. Open your browser to `http://localhost:8050`

## Usage

1. **Upload FIT Files**: Use the upload area to load cycling data files
2. **Ride Overview**: View comprehensive ride metrics and time-series data
3. **Interval Analysis**: Analyze intervals detected from lap data
4. **Multi-Ride Comparison**: Compare performance across multiple rides
5. **Performance Trends**: Track long-term improvements and patterns

## Key Improvements over SprintV1.py

- **Modular Architecture**: Separated into reusable components
- **Lap-based Intervals**: Uses lap data instead of power thresholds [[memory:6759645]]
- **Multi-ride Support**: Built for comparing multiple rides
- **Modern UI**: Bootstrap-based responsive design
- **Performance Focus**: Optimized for large datasets
- **Extensible**: Easy to add new features and components

## Development Status

This is a new implementation designed to replace and extend the functionality of SprintV1.py. The system is currently in active development with the following status:

- ✅ Frontend structure and components
- ⏳ Backend integration (waiting for Backend Agent)
- ⏳ FIT file processing integration
- ⏳ Database/CSV storage system
- ⏳ Testing and validation

## Dependencies

- **Backend Agent**: Required for data processing and FIT file handling
- **SprintV1.py**: Used as reference for algorithms and approaches (unchanged)

## Configuration

The system supports dynamic configuration including:
- FTP threshold settings (not hardcoded) [[memory:6759645]]
- Interval detection parameters
- Power zone definitions
- Display preferences

## Contributing

This project follows a multi-agent development approach:
- **Frontend Agent**: Dashboard and visualization components
- **Backend Agent**: Data processing and analysis
- **Reviewer Agent**: Quality assurance and testing

Please update the project roadmap coordination document when making changes.