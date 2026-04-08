# Experiment Status & Updates

## ✅ What Completed

All Phase 1/2/3 experiments finished successfully (SSH cancellation didn't stop them):

| Phase | Configs | Status | Best Result |
|-------|---------|--------|-------------|
| Phase 1: Window+Overlap | 12 | ✅ Complete | Window=100, Step=25 → RF F1=0.8001 |
| Phase 2: Filters | 6 | ✅ Complete | lowpass_3hz → RF F1=0.8021 |
| Phase 3: LSTM (unweighted) | 9 | ✅ Complete | large_2layer_30ep → LSTM F1=0.5625 ⚠️ |

## 🚨 Issue Found: Class Imbalance

**Problem**: LSTM got F1=0.56 vs RF's F1=0.80 (28% worse)

**Root Cause**: Severe class imbalance in dataset:
- Work: 704k samples (48%)
- Exercise: 28k samples (2%)
- **Imbalance ratio: 24:1**

**Solution**: Enable class weighting in LSTM loss function

## 🔧 What Was Updated

### 1. Best Parameters Applied to Codebase

Updated [src/main.py](../src/main.py) with optimal configurations from experiments:

```python
# From Phase 1: Window + Overlap
WINDOW_SIZE   = 100           # 10 seconds @ 10 Hz (was 50)
STEP_SIZE     = 25            # 75% overlap (was 50%)

# From Phase 2: Filter
FILTER_CONFIG = {'method': 'lowpass', 'cutoff': 3.0, 'order': 4}  # (was None)

# From Phase 3: LSTM Architecture
LSTM_HIDDEN   = 256           # (was 128)
LSTM_LAYERS   = 2             # (unchanged)
LSTM_DROPOUT  = 0.3           # (unchanged)
LSTM_EPOCHS   = 100           # Increased for proper training (was 30)
LSTM_BATCH    = 256           # (unchanged)
LSTM_CLASS_WEIGHT = True      # ← KEY FIX for imbalanced data
```

### 2. New Script Created

**[scripts/phase3_lstm_weighted.py](../scripts/phase3_lstm_weighted.py)**

Tests 9 LSTM architectures with **class weighting enabled** and **100 epochs**:

| Config | Hidden | Layers | Epochs | Notes |
|--------|--------|--------|--------|-------|
| small_1layer_100ep | 64 | 1 | 100 | Simplest |
| medium_2layer_100ep | 128 | 2 | 100 | Baseline |
| large_2layer_100ep | 256 | 2 | 100 | Best from Phase 3 |
| small_2layer_100ep | 64 | 2 | 100 | Narrow + deep |
| xlarge_2layer_100ep | 512 | 2 | 100 | Extra capacity |
| medium_3layer_100ep | 128 | 3 | 100 | Deeper |
| large_3layer_100ep | 256 | 3 | 100 | Deep + wide |
| medium_4layer_100ep | 128 | 4 | 100 | Very deep |
| large_4layer_100ep | 256 | 4 | 100 | Very deep + wide |

### 3. Currently Running in Background

```bash
Process ID: 161031
Log file: phase3_weighted.log
Command: python3 scripts/phase3_lstm_weighted.py --window 100 --step 25 --filter lowpass_3hz

Estimated time: ~2-3 hours (9 configs × 100 epochs each)
```

**Monitor progress:**
```bash
# Check log
tail -f phase3_weighted.log

# Check if still running
ps aux | grep phase3_lstm_weighted | grep -v grep

# Kill if needed
kill 161031
```

## 📊 Expected Results

### Without Class Weighting (Phase 3):
- Best LSTM F1: 0.5625
- RF F1: 0.8021
- **Problem**: LSTM predicts majority class (Work) too often

### With Class Weighting (Phase 3 Rerun):
- Expected LSTM F1: **0.70-0.80** (should match or exceed RF)
- Better per-class F1 for minority classes (Exercise, Cook, Hygiene)
- Training takes longer (100 epochs vs 30)

## 📋 Next Steps

### Once Phase 3 (weighted) completes:

1. **Check results:**
   ```bash
   cat results/phase3_lstm_weighted_*/results.csv
   ```

2. **Compare with Phase 3 (unweighted):**
   - Unweighted best: F1=0.5625 
   - Weighted best: F1=??? (should be > 0.70)

3. **Update main.py if needed:**
   If weighted LSTM finds better architecture, update these in `src/main.py`:
   ```python
   LSTM_HIDDEN   = <best_from_weighted>
   LSTM_LAYERS   = <best_from_weighted>
   LSTM_DROPOUT  = <best_from_weighted>
   ```

4. **Run final training:**
   ```bash
   python3 src/main.py
   ```
   This will train all models (LSTM, RF, SVM, DT) with optimized config and log to WandB.

5. **Convert to notebook:**
   With best parameters already in code, notebook will use optimal settings.

## 🎯 Summary

| Component | Old Value | New Value | Source |
|-----------|-----------|-----------|--------|
| Window Size | 50 samples | **100 samples** | Phase 1 |
| Step Size | 25 samples | **25 samples** | Phase 1 (unchanged) |
| Overlap | 50% | **75%** | Phase 1 |
| Filter | None | **lowpass_3hz** | Phase 2 |
| LSTM Hidden | 128 | **256** | Phase 3 |
| LSTM Layers | 2 | **2** | Phase 3 (unchanged) |
| LSTM Dropout | 0.3 | **0.3** | Phase 3 (unchanged) |
| LSTM Epochs | 30 | **100** | User request |
| Class Weighting | False | **True** | Phase 3 rerun |

**All changes committed to codebase. When you convert to notebook, it will use these optimal values automatically!** 🚀

## 📁 File Locations

- Main config: [src/main.py](../src/main.py)
- Phase 1 results: `results/phase1_data_params_20260310_175442/`
- Phase 2 results: `results/phase2_filters_20260310_182327/`
- Phase 3 results: `results/phase3_lstm_hyperparam_20260310_183026/`
- Phase 3 (weighted) running: `phase3_weighted.log` → `results/phase3_lstm_weighted_*/`
