# Assignment 1 – Cognitive Decline Activity Recognition

## Overview

This project trains and evaluates machine learning models on wrist-worn smartwatch accelerometer data to classify seven daily activities (Work, Other, Eat, Travel, Hygiene, Cook, Exercise) with the goal of detecting behavioural changes associated with cognitive decline. The pipeline uses a sliding-window approach to segment the raw 3-axis acceleration signal into fixed-length windows, extracts statistical features (mean, median, mode, max, min, IQR) per window, and trains four classifiers: **LSTM** (PyTorch), **Random Forest**, **SVM**, and **Decision Tree**. The LSTM benefits from GPU acceleration where available, and falls back to maximally threaded CPU via `torch.set_num_threads`. All scikit-learn models leverage multi-core BLAS (OMP/MKL/OpenBLAS threads pinned to all available cores at process start), mirroring the optimisation approach used in Tutorial 3.

## Project Structure

```
assignment_1/
├── df_train.csv                 # Raw smartwatch dataset (not committed to git)
├── README.md                    # This file
├── COEN_498_Assignment_1.ipynb  # Google Colab submission notebook
├── src/
│   ├── data.py        # Loading, filtering, windowing, feature extraction
│   ├── model.py       # LSTM (PyTorch) + sklearn classifiers with CPU MT
│   ├── evaluate.py    # Metrics, classification reports, results persistence
│   ├── plot.py        # EDA and evaluation visualisations
│   ├── main.py        # End-to-end pipeline – run this for a full training run
│   └── experiments.py # Experimentation campaign (window/step/filter sweeps)
├── results/           # Auto-created; metrics.txt + summary.csv per run
└── images/            # Auto-created; all PNG plots
```

## Quick Start

```bash
# From the assignment_1/ directory

# Step-by-step testing (recommended workflow):
python src/data.py        # test data loading + windowing
python src/model.py       # test LSTM + sklearn training
python src/evaluate.py    # test metric computation + saving
python src/plot.py        # test EDA plot generation

# Full pipeline:
python src/main.py

# Experimentation campaign (window/step/filter/model sweeps):
python src/experiments.py
```

## Key Hyperparameters

All tunable values are declared at the top of `src/main.py`:

| Parameter       | Default | Effect |
|-----------------|---------|--------|
| `WINDOW_SIZE`   | 50      | Window length in samples (50 samples @ 10 Hz = 5 s) |
| `STEP_SIZE`     | 25      | Sliding step → 50 % overlap |
| `FILTER_CONFIG` | None    | Optional pre-filter: `lowpass`, `bandpass`, `median`, or `combined` |
| `LSTM_HIDDEN`   | 128     | LSTM hidden units |
| `LSTM_LAYERS`   | 2       | Stacked LSTM layers |
| `LSTM_EPOCHS`   | 30      | Training epochs |
| `LSTM_LR`       | 1e-3    | Adam learning rate |
| `TEST_FRACTION` | 0.20    | Fraction of windows held out for evaluation |

## Dataset

`df_train.csv` – ~1.48 M rows at 10 Hz.

| Column | Description |
|--------|-------------|
| `stamp` | ISO timestamp |
| `user_acceleration_x/y/z` | Raw accelerometer axes (units: g) |
| `user_activity_label` | Ground-truth activity label |

## CPU / GPU Acceleration

- **LSTM**: PyTorch uses CUDA automatically when a GPU is present; otherwise `torch.set_num_threads(N)` pins all logical cores.
- **sklearn (SVM, RF, DT)**: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `BLIS_NUM_THREADS` are set before any import. Random Forest additionally passes `n_jobs=-1`.
- `threadpoolctl` is used for runtime BLAS control when available (`pip install threadpoolctl`).
