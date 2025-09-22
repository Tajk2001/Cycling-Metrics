# Workout Classifier

A comprehensive tool for automatically classifying cycling workouts from FIT files using machine learning and rule-based analysis.

## Features

- **Batch Processing**: Analyze entire folders of FIT files at once
- **Multiple Classification Methods**: Combines ML-based interval detection with rule-based analysis
- **Comprehensive Workout Types**: Supports all major workout categories (VO2, Threshold, Tempo, etc.)
- **Detailed Reporting**: Generates CSV, JSON reports and visualizations
- **Confidence Scoring**: Provides confidence levels for each classification

## Supported Workout Types

| Type | Description | Power Range | Duration | Characteristics |
|------|-------------|-------------|----------|-----------------|
| Recovery | Easy recovery rides | < 65% FTP | > 30 min | Low power, minimal intervals |
| Z2 | Endurance base rides | 65-75% FTP | > 45 min | Steady state, few intervals |
| Tempo | Moderate intensity | 75-90% FTP | Variable | Sustained efforts > 5 min |
| Threshold | Lactate threshold | 90-105% FTP | Variable | Hard efforts > 5 min |
| VO2 Short | High intensity short | > 115% FTP | < 5 min | Short, intense intervals |
| VO2 Long | High intensity long | > 110% FTP | > 5 min | Longer high-intensity efforts |
| Anaerobic | Very high intensity | > 125% FTP | < 2 min | Short, very hard efforts |
| Sprint | Maximum efforts | > 140% FTP | < 1 min | All-out sprints |
| Mixed | Multiple types | Variable | Variable | Combination of workout types |
| Endurance | Long rides | < 75% FTP | > 2 hours | Long duration, steady effort |

## Installation

Ensure you have the required dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
python classify_workouts.py /path/to/your/workouts
```

#### With Custom FTP
```bash
python classify_workouts.py /path/to/your/workouts --ftp 280
```

#### Recursive Folder Search
```bash
python classify_workouts.py /path/to/your/workouts --recursive
```

#### Custom Output Directory
```bash
python classify_workouts.py /path/to/your/workouts --output my_results/
```

#### Skip Visualizations
```bash
python classify_workouts.py /path/to/your/workouts --no-viz
```

### Programmatic Usage

```python
from workout_classifier import WorkoutClassifier

# Initialize classifier with your FTP
classifier = WorkoutClassifier(ftp=280.0)

# Process a single file
result = classifier.process_single_file("workout.fit")

# Process entire folder
results = classifier.process_folder("/path/to/workouts", recursive=True)

# Export results
classifier.export_results(results, "output_folder/")

# Generate visualizations
classifier.create_visualization(results, "output_folder/")
```

## Output Files

The tool generates several output files:

### 1. Detailed Results (JSON)
`workout_classification_YYYYMMDD_HHMMSS.json`
- Complete analysis results for each file
- Interval details and timing
- Confidence scores
- Raw metrics

### 2. Summary CSV
`workout_classification_YYYYMMDD_HHMMSS.csv`
- Tabular data for easy analysis
- Key metrics per workout
- Classification results

### 3. Summary Report (JSON)
`workout_summary_YYYYMMDD_HHMMSS.json`
- Overall statistics
- Workout type distribution
- Processing summary

### 4. Visualizations (PNG)
`workout_analysis_YYYYMMDD_HHMMSS.png`
- Workout type pie chart
- Duration vs power scatter plot
- Interval count distribution
- Confidence by workout type

## How It Works

### 1. Data Loading
- Loads FIT files using the existing SprintV1 infrastructure
- Extracts power, cadence, and timing data
- Calculates basic ride metrics

### 2. Interval Detection
- Uses the trained ML model from `interval_detection.py`
- Falls back to lap-based detection when available
- Applies power smoothing and optimization

### 3. Classification Logic
- Analyzes power patterns relative to FTP
- Considers interval count, duration, and intensity
- Uses multiple criteria for robust classification

### 4. Confidence Scoring
- Provides confidence levels based on how well the workout matches expected patterns
- Higher confidence = clearer workout type characteristics

## Configuration

### FTP Setting
Your Functional Threshold Power is crucial for accurate classification. Set it correctly:

```python
classifier = WorkoutClassifier(ftp=280.0)  # Your actual FTP
```

### Model Path
By default, uses the trained model at `trained_models/simple_comprehensive_model_20250820_125836.pkl`. You can specify a different model:

```python
classifier = WorkoutClassifier(
    ftp=280.0, 
    model_path="path/to/your/model.pkl"
)
```

## Troubleshooting

### Common Issues

1. **"No FIT files found"**
   - Check that your folder contains `.fit` files
   - Use `--recursive` flag to search subdirectories

2. **"Could not load FIT file"**
   - Ensure FIT files are not corrupted
   - Check file permissions

3. **Low confidence scores**
   - Verify your FTP is set correctly
   - Some unstructured rides may have lower confidence

4. **"No intervals detected"**
   - Normal for recovery or endurance rides
   - ML model may need retraining with more data

### Performance Tips

- For large folders (100+ files), processing may take time
- Use `--no-viz` flag to skip visualizations for faster processing
- Consider processing in smaller batches for very large collections

## Example Output

```
🚴 Processing workout folder: /Users/john/cycling_data
============================================================
📁 Found 15 FIT files

[1/15] Processing...
📁 Processing: 2025-01-15-morning-ride.fit
   📊 Duration: 45.2min, Avg Power: 185W
   🔍 Detecting intervals...
   ✅ Found 3 intervals
   🎯 Classification: tempo (confidence: 0.85)
   🏷️ Label: Tempo (3x5min)

[2/15] Processing...
📁 Processing: 2025-01-16-recovery.fit
   📊 Duration: 60.1min, Avg Power: 142W
   🔍 Detecting intervals...
   ⚠️ No intervals detected
   🎯 Classification: recovery (confidence: 0.90)
   🏷️ Label: Recovery Ride (60min)

...

📋 SUMMARY REPORT
========================================
Files processed: 15
Successful: 14
Failed: 1
Success rate: 93.3%

Total duration: 785.4 minutes
Average power: 178W
Total intervals: 42

Workout types found:
  tempo: 4
  recovery: 3
  z2: 2
  vo2-short: 2
  thr: 2
  endurance: 1

✅ Classification complete! Check the 'workout_classification_results' folder for detailed results.
```

## Integration

This tool integrates with the existing cycling analysis ecosystem:

- Uses `SprintV1.py` for FIT file loading
- Leverages `IntervalML.py` for interval detection
- Compatible with `workout_foundation` package
- Can be extended with additional classification rules

## Contributing

To add new workout types or improve classification:

1. Edit the `workout_types` dictionary in `WorkoutClassifier.__init__()`
2. Add classification logic in `classify_workout_type()`
3. Update the label generation in `_create_workout_label()`
4. Test with sample data

## License

This tool is part of the cycling analysis project and follows the same licensing terms.
