# Assignment 1 - Experiment Strategy Guide

## 🎯 Goal
Maximize F1 score for cognitive decline activity recognition by systematically optimizing:
1. Window size
2. Overlap (step size)
3. Signal filtering
4. Filter parameters
5. Model selection (LSTM vs sklearn)

## 📊 Tutorial 3 Analysis - What You Did

Looking at your tutorial_3 results folders and code:

### Phase 1: Window & Overlap Exploration
- **Tested Window**: 170 samples (5.3s @ 32Hz)
- **Tested Overlaps**: 5, 10, 15, 30, 85 samples
- **Best Config from filenames**: Window=170, Overlap=5, F1=0.488

### Phase 2: Filter Comparison
- Compared: No filter vs Lowpass 5Hz
- **Result**: Lowpass filter improved performance

### Phase 3: Filter Parameter Tuning
- Swept different cutoff frequencies for lowpass filter
- Found optimal cutoff frequency

### Phase 4: Full Model Comparison
- Compared SVM, Random Forest, Decision Tree on best config
- **Winner**: Random Forest (typical for HAR tasks)

### Key Insights from Tutorial 3:
1. **Window size matters**: Longer windows (170 samples ~= 5.3s) captured enough temporal context
2. **Low overlap works well**: Overlap=5 samples (~3% overlap) gave best results
3. **Filtering helps**: Lowpass filter removed high-frequency noise
4. **RF wins**: Random Forest outperformed SVM and Decision Tree

---

## 🚀 Assignment 1 Strategy - Adapted Approach

### Key Differences from Tutorial 3:
- **Sampling rate**: 10 Hz (vs 32 Hz in tutorial_3)
- **Activities**: 7 activities (vs 14 in tutorial_3)
- **New model**: LSTM (GPU-trained) in addition to sklearn models
- **Data size**: 1.48M rows (much larger dataset)

### Recommended Window/Overlap Values (for 10 Hz):

| Duration | Samples @ 10Hz | Tutorial_3 Equivalent |
|----------|----------------|----------------------|
| 2 sec    | 20 samples     | 64 samples @ 32Hz    |
| 3 sec    | 30 samples     | 96 samples @ 32Hz    |
| 5 sec    | 50 samples     | 160 samples @ 32Hz   |
| 7 sec    | 70 samples     | 224 samples @ 32Hz   |
| 10 sec   | 100 samples    | 320 samples @ 32Hz   |

**Starting point**: Window=50 (5 seconds) is a good baseline - similar to tutorial_3's successful 170 samples.

---

## 📋 Phased Experiment Plan

### PHASE 1: Baseline Testing ✓ (Already Done)
**Status**: Completed in main.py test

**Config**:
- Window: 50 samples (5s)
- Step: 25 samples (50% overlap)
- Filter: None
- LSTM: 5 epochs (quick test)

**Results**:
- RF: F1 = 0.68 (best sklearn model)
- LSTM: F1 = 0.40 (only 5 epochs, needs more training)
- SVM: F1 = 0.48
- DT: F1 = 0.58

**Action**: ✓ Baseline established

---

### PHASE 2: Window Size Sweep
**Objective**: Find optimal window duration

**Command**:
```bash
python3 src/experiments.py  # Runs experiment_1_window_sweep
```

**Test These Windows**:
- 20 samples (2s) - short activities
- 30 samples (3s)
- 50 samples (5s) - baseline
- 70 samples (7s)
- 100 samples (10s) - long activities

**Fixed Parameters**:
- Step: 25 samples (starting point)
- Filter: None
- Model: Random Forest (fastest, good baseline)

**Expected Insight**: 
- Short windows (20-30) may work for quick activities (Eat, Cook)
- Long windows (70-100) may work for sustained activities (Work, Exercise)
- Likely sweet spot: 40-60 samples

**Time Estimate**: ~20-30 minutes (5 configs × 4-6 min each)

---

### PHASE 3: Overlap (Step Size) Sweep
**Objective**: Optimize overlap percentage

**What to Test**:
- Step=5 (90% overlap) - maximum data, slow
- Step=10 (80% overlap)
- Step=25 (50% overlap) - baseline
- Step=40 (20% overlap)
- Step=window-5 (~minimal overlap) - fast, less data

**Fixed Parameters**:
- Window: **BEST from Phase 2**
- Filter: None
- Model: Random Forest

**Expected Insight**:
- High overlap (small step) = more windows = more training data but redundant
- Low overlap (large step) = fewer windows = faster but may miss patterns
- Tutorial_3 found minimal overlap worked best

**Time Estimate**: ~25 minutes (5 configs)

---

### PHASE 4: Filter Comparison
**Objective**: Determine if filtering helps and which filter type

**Test These Filters**:
1. **None (baseline)**
2. **Lowpass** (cutoff=3 Hz, order=4) - remove high-freq noise
3. **Lowpass** (cutoff=4 Hz, order=4) - less aggressive
4. **Bandpass** (1-5 Hz, order=4) - keep activity frequencies
5. **Median** (kernel=3) - remove spikes

**Fixed Parameters**:
- Window: **BEST from Phase 2**
- Step: **BEST from Phase 3**
- Model: Random Forest

**Expected Insight**:
-Lowpass likely helps (worked in tutorial_3)
- 10 Hz sampling means Nyquist = 5 Hz, so cutoff < 5 Hz
- Activity movements are typically 0.5-3 Hz

**Time Estimate**: ~25 minutes (5 configs)

**Rationale for Filters**:
- **Lowpass 3Hz**: Most human movements are < 3 Hz
- **Lowpass 4Hz**: More permissive, keep more signal
- **Bandpass 1-5Hz**: Remove both DC drift and high-freq noise
- **Median**: Non-linear filter, good for outliers

---

### PHASE 5: Filter Parameter Tuning (If Phase 4 Shows Benefit)
**Objective**: Optimize filter cutoff frequency

**If Lowpass Won Phase 4**:
Test cutoffs: 2.0, 2.5, 3.0, 3.5, 4.0, 4.5 Hz

**If Bandpass Won Phase 4**:
Test variations:
- (0.5-3 Hz)
- (0.5-4 Hz)
- (1-4 Hz)
- (1-5 Hz)

**Fixed Parameters**:
- Window: **BEST from Phase 2**
- Step: **BEST from Phase 3**
- Filter: **BEST TYPE from Phase 4**
- Model: Random Forest

**Time Estimate**: ~30 minutes (6 configs)

---

### PHASE 6: Full Model Comparison at Optimal Config
**Objective**: Compare all models (LSTM, SVM, DT, RF) with best settings

**Command**:
```bash
# Edit src/experiments.py experiment_4_model_comparison params
python3 src/experiments.py  # Runs experiment_4_model_comparison
```

**Models to Test**:
1. **LSTM** (30 epochs, GPU)
2. **Random Forest** (200 trees, CPU multi-threading)
3. **SVM** (RBF kernel, CPU)
4. **Decision Tree** (baseline)

**Fixed Parameters**:
- Window: **BEST from Phase 2**
- Step: **BEST from Phase 3**
- Filter: **BEST from Phases 4-5**

**LSTM Hyperparameters to Test** (optional extended):
- Hidden sizes: 64, 128, 256
- Layers: 1, 2, 3
- Dropout: 0.2, 0.3, 0.4
- Epochs: 20, 30, 50

**Time Estimate**: ~15 minutes (all 4 models)

**Expected Winner**:
- **Option 1**: Random Forest (typical winner for HAR)
- **Option 2**: LSTM (if temporal patterns are complex)

---

## 🎨 Modified Experiments.py Functions

Already implemented in your code:
- `experiment_1_window_sweep()` - Test window sizes
- `experiment_2_step_sweep()` - Test overlaps
- `experiment_3_filter_comparison()` - Test filter types
- `experiment_4_model_comparison()` - Compare all models

**To Add for Phase 5** (filter parameter tuning):

```python
def experiment_5_filter_param_tuning(df, filter_type='lowpass',
                                      window_size=50, step_size=25):
    """
    Sweep filter parameters for best filter type.
    
    For lowpass: sweep cutoff frequencies
    For bandpass: sweep frequency ranges
    """
    if filter_type == 'lowpass':
        cutoffs = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        configs = [
            {'method': 'lowpass', 'cutoff': c, 'order': 4}
            for c in cutoffs
        ]
    elif filter_type == 'bandpass':
        bands = [(0.5, 3), (0.5, 4), (1, 4), (1, 5)]
        configs = [
            {'method': 'bandpass', 'low': l, 'high': h, 'order': 4}
            for l, h in bands
        ]
    
    results = []
    for cfg in configs:
        row = _eval_rf(df, window_size, step_size,
                       filter_config=cfg, verbose=False)
        if row:
            row['filter_config'] = str(cfg)
            results.append(row)
    
    return pd.DataFrame(results)
```

---

## 📊 How to Run Full Campaign

### Option 1: Run All Experiments Automatically
```bash
cd ~/achal/Pervasive-Computing-for-Health/assignment_1
python3 src/experiments.py
```

**Note**: This runs Experiments 1-4. You'll need to manually add Experiment 5 based on Phase 4 results.

### Option 2: Run Phases Individually

```python
# In Python console or script:
from src.experiments import *
import pandas as pd

df = load_data('df_train.csv')

# Phase 2
df1 = experiment_1_window_sweep(df, window_sizes=[20, 30, 50, 70, 100])
print(df1)
best_window = df1.loc[df1['rf_f1'].idxmax(), 'window_size']

# Phase 3
df2 = experiment_2_step_sweep(df, steps=[5, 10, 25, 40, 45],
                               window_fixed=best_window)
print(df2)
best_step = df2.loc[df2['rf_f1'].idxmax(), 'step_size']

# Phase 4
df3 = experiment_3_filter_comparison(df, window_size=best_window,
                                      step_size=best_step)
print(df3)
best_filter = df3.loc[df3['rf_f1'].idxmax(), 'filter']

# Phase 6 (if filter="none", use filter_config=None)
df4 = experiment_4_model_comparison(df, window_size=best_window,
                                     step_size=best_step,
                                     lstm_epochs=30)
print(df4)
```

---

## 🏆 Expected F1 Score Progression

| Phase | Config | Expected RF F1 | Note |
|-------|--------|----------------|------|
| Baseline | w=50, s=25, no filter | 0.68 | ✓ Measured |
| Phase 2 | Best window | 0.70-0.72 | +2-4% from optimal duration |
| Phase 3 | Best window+overlap | 0.73-0.75 | +1-3% from optimal data sampling |
| Phase 4 | +Best filter | 0.76-0.78 | +3-5% from noise reduction |
| Phase 5 | +Optimal filter params | 0.77-0.79 | +1-2% fine-tuning |
| Phase 6 | Best model | 0.78-0.82 | LSTM may win |

**Target**: F1 > 0.78 (best model)
**Stretch**: F1 > 0.80

---

## 💡 Quick Wins to Try First

Before running full campaign, try these manual tweaks in main.py:

### Quick Win 1: Train LSTM Properly
```python
# In main.py, change:
LSTM_EPOCHS = 30  # instead of 5
```
**Expected**: LSTM F1 should jump to 0.55-0.65

### Quick Win 2: Try Larger Window
```python
WINDOW_SIZE = 70  # instead of 50
STEP_SIZE = 35     # keep 50% overlap
```
**Expected**: May capture more temporal context, +2-3% F1

### Quick Win 3: Add Lowpass Filter
```python
FILTER_CONFIG = {'method': 'lowpass', 'cutoff': 3, 'order': 4}
```
**Expected**: +3-5% F1 (based on tutorial_3 results)

### Quick Win 4: Increase RF Trees
```python
# In model.py, change RandomForestClassifier:
n_estimators=300  # instead of 200
max_depth=25      # add depth limit
min_samples_split=10  # prevent overfitting
```

---

## 🔬 Advanced: LSTM Hyperparameter Tuning

If LSTM shows promise, create a dedicated LSTM experiment:

```python
def experiment_lstm_hyperparam_sweep(df, window_size, step_size,
                                      filter_config=None):
    """Test LSTM architecture variations."""
    configs = [
        {'hidden': 64, 'layers': 1, 'dropout': 0.2, 'epochs': 20},
        {'hidden': 128, 'layers': 2, 'dropout': 0.3, 'epochs': 20},
        {'hidden': 128, 'layers': 2, 'dropout': 0.3, 'epochs': 30},  # baseline
        {'hidden': 256, 'layers': 2, 'dropout': 0.3, 'epochs': 30},
        {'hidden': 128, 'layers': 3, 'dropout': 0.4, 'epochs': 30},
        {'hidden': 256, 'layers': 3, 'dropout': 0.4, 'epochs': 50},
    ]
    
    results = []
    data = prepare_ml_data(df, window_size, step_size, filter_config=filter_config)
    
    for cfg in configs:
        lstm_result = train_lstm(data['X_train_seq'], data['y_train'],
                                 num_classes=len(data['label_encoder'].classes_),
                                 **cfg, verbose=False)
        lstm_pred = predict_lstm(lstm_result['model'], data['X_test_seq'])
        f1 = f1_score(data['y_test'], lstm_pred, average='weighted')
        
        results.append({**cfg, 'f1_weighted': f1,
                       'train_time': lstm_result['train_time_sec']})
    
    return pd.DataFrame(results)
```

---

## 📁 Output Structure

After running all experiments:

```
results/
├── exp1_window_sweep_20260310_HHMMSS/
│   ├── results.csv
│   └── window_sweep_plot.png
├── exp2_step_sweep_20260310_HHMMSS/
│   ├── results.csv
│   └── step_sweep_plot.png
├── exp3_filter_20260310_HHMMSS/
│   ├── results.csv
│   └── filter_comparison_plot.png
├── exp4_model_comparison_20260310_HHMMSS/
│   ├── results.csv
│   └── model_comparison_plot.png
└── 20260310_HHMMSS/  # main.py full run
    ├── metrics.txt
    ├── summary.csv
    ├── confusion_matrices_combined.png
    ├── per_class_f1.png
    └── lstm_training_loss.png
```

---

## ⚡ Time Estimates

| Phase | Time | Can Parallelize? |
|-------|------|------------------|
| Phase 2: Window Sweep | 25 min | No |
| Phase 3: Overlap Sweep | 25 min | No |
| Phase 4: Filter Comparison | 25 min | No |
| Phase 5: Filter Tuning | 30 min | No |
| Phase 6: Model Comparison | 15 min | No |
| **Total Sequential** | **~2 hours** | |

**Optimization**: Run experiments overnight or during lunch break.

---

## 🎯 Final Recommendations

### For Best F1 Score:
1. **Start with Quick Wins** (30 min)
   - Change `LSTM_EPOCHS = 30`
   - Add `FILTER_CONFIG = {'method': 'lowpass', 'cutoff': 3, 'order': 4}`
   - Run `python3 src/main.py` with wandb logging

2. **If you have time** (2-3 hours):
   - Run full experiment campaign: `python3 src/experiments.py`
   - Analyze results to find optimal config
   - Re-run main.py with optimal settings

3. **If time is limited** (1 hour):
   - Run Phase 2 (window sweep) manually
   - Run Phase 4 (filter comparison) with best window
   - Run Phase 6 (model comparison) with best window+filter

### Expected Final Config:
```python
# Predicted optimal settings (based on tutorial_3 patterns):
WINDOW_SIZE = 60-70      # slightly longer than baseline
STEP_SIZE = 10-15        # low overlap, more windows
FILTER_CONFIG = {'method': 'lowpass', 'cutoff': 3, 'order': 4}
LSTM_EPOCHS = 30-50
LSTM_HIDDEN = 128-256
Best Model = Random Forest OR LSTM (will know after Phase 6)
```

**Predicted Best F1**: 0.76-0.82

---

## 📝 Next Steps

1. ✅ **Done**: Setup WandB, restructure outputs, create 2x2 confusion matrix
2. **Now**: Run Quick Wins (the main.py test with 30 epochs + filter)
3. **Then**: Decision point:
   - If F1 > 0.75: Submit and move on
   - If F1 < 0.75: Run full experiment campaign
4. **Track Progress**: Use WandB dashboard to monitor all runs

---

## 🚀 Launch Command Summary

```bash
cd ~/achal/Pervasive-Computing-for-Health/assignment_1

# Quick win: Full training with filter
# Edit main.py: LSTM_EPOCHS=30, FILTER_CONFIG={'method': 'lowpass', 'cutoff': 3, 'order': 4}
python3 src/main.py

# Full experiment campaign
python3 src/experiments.py

# Monitor on WandB:
# https://wandb.ai/mimic-robotics/coen498-assignment1
```

Good luck! 🎯
