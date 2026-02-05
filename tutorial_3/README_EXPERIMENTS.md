# Activity Recognition Experiments - Usage Guide

This folder contains optimized Python scripts for running activity recognition experiments on the ADL dataset.

## Files

1. **tutorial_3_optimized.py** - Core functionality module
   - Data loading and preprocessing
   - Feature extraction
   - Sliding window implementation
   - Model training and classification
   - CPU optimization utilities

2. **experiments.py** - Experiment runner
   - Experiment 1: Window size sweep (30-150, step 10, overlap=30)
   - Experiment 2: Overlap sweep (15-75, step 15, window=120)
   - Progress bars with TQDM
   - Result visualization with matplotlib/seaborn
   - CSV export functionality

## Requirements

Install required packages:
```bash
pip install numpy pandas scikit-learn scipy tqdm psutil matplotlib seaborn
```

## Quick Start

### Run All Experiments

Simply run the experiments file:
```bash
python experiments.py
```

This will:
1. Configure CPU multi-threading for optimal performance
2. Load the ADL dataset
3. Run Experiment 1 (window size sweep)
4. Run Experiment 2 (overlap sweep)
5. Save results to CSV files
6. Generate visualization plots
7. Print summary statistics

### Run Individual Experiments

Edit `experiments.py` and uncomment the individual experiment code at the bottom:

```python
if __name__ == "__main__":
    # Configure CPU and load data
    configure_cpu_threading()
    df = read_dataset(ZIP_FILE_PATH, EXTRACTED_FOLDER_PATH, SAMPLING_FREQUENCY)
    
    # Run only Experiment 1
    exp1_results = experiment_1_window_sweep(df, overlap_fixed=30, n_jobs=-1)
    print(exp1_results)
    
    # Run only Experiment 2
    exp2_results = experiment_2_overlap_sweep(df, window_fixed=120, n_jobs=-1)
    print(exp2_results)
    
    # Plot results
    plot_experiment_results(exp1_results, exp2_results)
```

## Customizing Experiments

### Change Window Sizes (Experiment 1)

```python
exp1_results = experiment_1_window_sweep(
    df,
    window_sizes=list(range(20, 200, 20)),  # 20 to 200, step 20
    overlap_fixed=30,
    n_jobs=-1
)
```

### Change Overlaps (Experiment 2)

```python
exp2_results = experiment_2_overlap_sweep(
    df,
    overlaps=list(range(10, 100, 10)),  # 10 to 100, step 10
    window_fixed=120,
    n_jobs=-1
)
```

### Control CPU Usage

Use `n_jobs` parameter:
- `n_jobs=-1` : Use all CPU cores (recommended)
- `n_jobs=1` : Use single core
- `n_jobs=4` : Use 4 cores

```python
exp1_results = experiment_1_window_sweep(df, n_jobs=4)
```

## Output Files

After running experiments, you'll find:

1. **experiment1_window_sweep.csv** - Results from window size sweep
   - Columns: window_size, overlap, overlap_percent, num_samples, svm_f1, dt_f1, rf_f1

2. **experiment2_overlap_sweep.csv** - Results from overlap sweep
   - Columns: window_size, overlap, overlap_percent, num_samples, svm_f1, dt_f1, rf_f1

3. **experiment_results.png** - Visualization of F1 scores
   - Left plot: Window size impact
   - Right plot: Overlap impact
   - Best SVM configuration highlighted with gold star

## Performance Notes

- The code is optimized for multi-core CPUs using:
  - BLAS multi-threading for SVM (numpy/scipy operations)
  - Scikit-learn's `n_jobs` parameter for Random Forest
  - Parallel processing where applicable

- On a high-end CPU (like Ryzen 9), you should see significant speedup compared to single-core execution

- Progress bars show:
  - Overall experiment progress (outer bar)
  - Cross-validation progress within each configuration (inner bar)

## Example Output

```
============================================================================
ACTIVITY RECOGNITION WINDOWING PARAMETER OPTIMIZATION
============================================================================
Start: 2026-02-04 10:30:00
============================================================================

Configuring CPU threading...
BLAS threading configured to use 16 cores
This will accelerate SVM's numerical operations

CPU Information:
  Physical cores: 8
  Logical cores (threads): 16
  Current usage: 5.2%
  Available memory: 28.45 GB
  Total memory: 32.00 GB

============================================================================
LOADING DATASET
============================================================================
...

============================================================================
EXPERIMENT 1: Window Size Sweep
============================================================================
Window sizes to test: [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
Fixed overlap: 30
Total configurations: 13
============================================================================

Experiment 1 Progress: 100%|████████████████| 13/13 [15:30<00:00, 71.54s/it]

============================================================================
EXPERIMENT 2: Overlap Sweep
============================================================================
...

Results saved to:
  - experiment1_window_sweep.csv
  - experiment2_overlap_sweep.csv

Plot saved to: experiment_results.png
```

## Troubleshooting

### Dataset not found

Make sure `adl_dataset` file is in the tutorial_3 folder. Download from:
https://archive.ics.uci.edu/static/public/283/dataset+for+adl+recognition+with+wrist+worn+accelerometer.zip

### Memory errors

If you run out of memory, try:
1. Reducing the number of configurations tested
2. Using smaller window sizes
3. Processing fewer volunteers at once

### Import errors

Make sure both Python files are in the same directory:
```
tutorial_3/
├── tutorial_3_optimized.py
├── experiments.py
└── adl_dataset
```

## Using as a Module

You can also import and use functions directly:

```python
from tutorial_3_optimized import read_dataset, train_and_classify, prepare_windowed_data
from experiments import plot_experiment_results

# Load data
df = read_dataset('adl_dataset', 'adl_dataset_extracted', 32)

# Prepare windows
X, y, groups = prepare_windowed_data(df, window_size=60, overlap=30)

# Train models
results = train_and_classify(X, y, groups, n_jobs=-1)
print(f"SVM F1 Score: {results['svm_f1']:.4f}")
```

## Notes

- The experiments focus on **SVM** and **Decision Tree** comparison as per assignment requirements
- **Random Forest** results are included for additional insight
- F1 scores are weighted averages across all activity classes
- Leave-One-Group-Out cross-validation ensures no data leakage between volunteers
