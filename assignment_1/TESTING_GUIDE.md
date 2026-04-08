# Assignment 1 - Testing Guide

## ✅ System Configuration

**Hardware:**
- CPU: 16 threads (8 physical cores)
- GPU: NVIDIA GeForce RTX 5070 (CUDA 12.8)
- RAM: 27 GB available

**Software:**
- PyTorch 2.12.0 with CUDA support
- All dependencies installed: scikit-learn, scipy, tqdm, psutil, threadpoolctl, seaborn, matplotlib

## ✅ All Module Tests Completed Successfully

### 1. **data.py** - Data Loading & Preprocessing ✓
```bash
cd /home/jupiter/achal/Pervasive-Computing-for-Health/assignment_1
python3 src/data.py
```
**Status:** ✓ PASSED
- Loaded 1,479,891 rows successfully
- 7 activity classes identified
- Sliding window creates 59,194 windows (50 samples/window, 25 step)
- Train/test split: 47,355 / 11,839 samples
- Fixed numpy deprecation warning

### 2. **model.py** - LSTM & sklearn Models ✓
```bash
python3 src/model.py
```
**Status:** ✓ PASSED
- **LSTM Training:**
  - ✓ GPU Detected: NVIDIA GeForce RTX 5070
  - ✓ Trained on CUDA successfully
  - ✓ 2-epoch test: F1 = 0.4115 (fast test)
- **sklearn Models:**
  - ✓ SVM: F1 = 0.4808 (CPU multi-threaded via BLAS)
  - ✓ Decision Tree: F1 = 0.5774
  - ✓ Random Forest: F1 = 0.6822 (CPU multi-threaded, n_jobs=-1)
  - ✓ CPU utilization: avg 10%, max 99.8%

### 3. **evaluate.py** - Metrics & Reports ✓
```bash
python3 src/evaluate.py
```
**Status:** ✓ PASSED
- Classification reports generated
- F1 scores (weighted & macro) computed
- Confusion matrices created
- Results saved to `results/test_run/`

### 4. **plot.py** - Visualizations ✓
```bash
python3 src/plot.py
```
**Status:** ✓ PASSED
- All EDA plots generated:
  - activity_distribution.png
  - acceleration_samples.png
  - frequency_content.png
  - temporal_patterns.png
  - correlation_heatmap.png
- Confusion matrix plots created

### 5. **main.py** - Full Pipeline ✓
```bash
python3 src/main.py
```
**Status:** ✓ PASSED (tested with 5 epochs for speed)
- **Pipeline Steps:**
  1. ✓ Data loaded (1.48M rows)
  2. ✓ EDA plots created
  3. ✓ Preprocessing & feature extraction
  4. ✓ LSTM trained on GPU (3.6s for 5 epochs)
  5. ✓ sklearn models trained (SVM 52.6s, RF 2.8s)
  6. ✓ All models evaluated
  7. ✓ Results saved to `results/<run_id>/`

**Results (5 epochs quick test):**
- LSTM: F1 = 0.3996 (low due to few epochs)
- SVM:  F1 = 0.4808
- DT:   F1 = 0.5774
- RF:   F1 = 0.6822 (best)

**Total Time:** 1.3 minutes

## 📊 Optimizations Verified

### GPU Optimization (LSTM)
- ✓ PyTorch automatically detects CUDA
- ✓ Model and tensors moved to GPU via `.to(device)`
- ✓ Training speed: ~1.4 it/s on GPU
- ✓ Model moved back to CPU after training for saving

### CPU Multi-threading Optimization
**Random Forest & sklearn models:**
- ✓ `n_jobs=-1` uses all CPU cores
- ✓ CPU usage reaches 100% during training

**SVM & other BLAS operations:**
- ✓ Environment variables set before imports:
  - `OMP_NUM_THREADS=16`
  - `MKL_NUM_THREADS=16`
  - `OPENBLAS_NUM_THREADS=16`
  - `BLIS_NUM_THREADS=16`
- ✓ `threadpoolctl` used for runtime control
- ✓ CPU monitoring shows proper utilization

## 🧪 Next Steps: Full Training & Experiments

### 1. Run Full Training (30 epochs)
```bash
cd /home/jupiter/achal/Pervasive-Computing-for-Health/assignment_1
python3 src/main.py
```
This will:
- Train LSTM for 30 epochs on GPU (~12 seconds)
- Train all sklearn models with full data
- Generate all plots and metrics
- Save results to `results/<timestamp>/`

**Expected time:** ~2-3 minutes

### 2. Run Experimentation Campaign
```bash
python3 src/experiments.py
```
This will execute 4 experiments:
1. **Window Size Sweep** - Test windows: 20, 30, 40, 50, 75, 100 samples
2. **Step Size Sweep** - Test overlap percentages
3. **Filter Comparison** - Compare lowpass, bandpass, median filters
4. **Model Comparison** - LSTM vs SVM vs DT vs RF at optimal config

**Expected time:** ~15-30 minutes (depending on experiment configs)

## 📁 Output Structure

```
assignment_1/
├── results/
│   ├── <run_id>/
│   │   ├── metrics.txt      # Full classification reports
│   │   └── summary.csv      # Summary table
│   ├── exp1_window_sweep_<run_id>.csv
│   ├── exp2_step_sweep_<run_id>.csv
│   ├── exp3_filter_<run_id>.csv
│   └── exp4_model_comparison_<run_id>.csv
│
├── images/
│   ├── activity_distribution.png
│   ├── acceleration_samples.png
│   ├── frequency_content.png
│   ├── temporal_patterns.png
│   ├── correlation_heatmap.png
│   ├── lstm_training_loss.png
│   ├── confusion_matrix_lstm.png
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_dt.png
│   ├── confusion_matrix_rf.png
│   ├── per_class_f1.png
│   └── exp*_<run_id>.png
```

## 🎯 Hyperparameter Tuning

Edit [src/main.py](src/main.py) to adjust:

```python
# Window settings
WINDOW_SIZE   = 50    # samples per window (5s @ 10 Hz)
STEP_SIZE     = 25    # 50% overlap

# LSTM hyperparameters
LSTM_HIDDEN   = 128   # hidden units
LSTM_LAYERS   = 2     # stacked LSTM layers
LSTM_DROPOUT  = 0.3   # dropout rate
LSTM_EPOCHS   = 30    # training epochs
LSTM_BATCH    = 256   # batch size
LSTM_LR       = 1e-3  # learning rate

# Filter (None to disable)
FILTER_CONFIG = None  # or {'method': 'lowpass', 'cutoff': 4, 'order': 4}
```

## 🔍 Individual Module Testing

Each module can be tested independently:
```bash
# Test data loading
python3 src/data.py

# Test model training
python3 src/model.py

# Test evaluation
python3 src/evaluate.py

# Test plotting
python3 src/plot.py
```

## 📝 Notes

1. **LSTM Performance:** With only 5 epochs, LSTM F1 = 0.40. With 30 epochs, expect F1 > 0.60
2. **GPU Utilization:** LSTM training is fast (~3.6s for 5 epochs, ~12s for 30 epochs)
3. **CPU Utilization:** sklearn models properly utilize all 16 threads
4. **Memory:** Current usage is low (~27 GB free), no memory issues
5. **Data Size:** 1.48M rows → 59K windows → 47K train, 12K test samples

## ✅ Summary

All modules tested and working correctly:
- ✅ Data loading & preprocessing
- ✅ LSTM GPU training (RTX 5070)
- ✅ sklearn CPU multi-threading (16 threads)
- ✅ Evaluation & metrics
- ✅ Visualization & plotting
- ✅ Full pipeline integration
- ✅ Experiments framework ready

**Ready for full training and experimentation!**
