# Hyperparameter Optimization Guide

## 🤔 Why Those Default LSTM Values?

The values in `src/model.py` (`hidden_size=128`, `num_layers=2`, `dropout=0.3`) are **educated defaults from research papers**, NOT optimized for your specific dataset.

**Sources**:
- Hidden size 64-256: Standard for HAR (Hammerla et al. 2016, Ordóñez & Roggen 2016)
- 2 layers: Balance between capacity and overfitting
- 0.3 dropout: Empirical sweet spot for time-series

**These need optimization!**

---

## 🎯 Optimization Strategy: Two-Phase Approach

### Why This Order?

1. **Data Parameters First** (window, overlap, filter)
   - Affect ALL models (RF, SVM, LSTM)
   - RF trains in seconds → fast iteration
   - Once optimized, all models benefit

2. **LSTM Hyperparameters Second**
   - Only affect LSTM
   - Trains in minutes → slower iteration
   - Use optimal data params from Phase 1

**Wrong Approach**: Grid search everything at once (would take days!)

**Right Approach**: Optimize data with RF, then optimize LSTM architecture

---

## 📋 What to Run (Step-by-Step)

### Option 1: Automated (2 hours, set and forget)

```bash
cd ~/achal/Pervasive-Computing-for-Health/assignment_1
python3 scripts/run_all_phases.py
```

This runs everything in sequence:
1. Quick LSTM baseline (2 min)
2. Phase 1: Window + overlap (30 min)
3. Phase 2: Filters (15 min)
4. Phase 3: LSTM hyperparams (60 min)

**Outputs**: `results/phase1_*/`, `results/phase2_*/`, `results/phase3_*/`

---

### Option 2: Manual (Run phases individually)

Gives you more control to inspect results between phases.

#### Step 0: Quick Baseline (2 min)

```bash
python3 scripts/quick_lstm_test.py
```

**What it does**: Train LSTM with 30 epochs (vs 5 in your test)

**Output**: F1 score with current defaults

**Decision**:
- If F1 > 0.70: Already good, but can still improve
- If F1 < 0.70: Definitely need optimization

---

#### Step 1: Data Parameters (30 min)

```bash
python3 scripts/phase1_data_params.py
```

**What it tests**:
- Window sizes: 30, 50, 70, 100 samples (3s, 5s, 7s, 10s @ 10Hz)
- For each window, 3 overlaps: ~15%, ~50%, ~75%
- Total: 12 configurations

**Uses Random Forest** (trains in 4-6 sec per config)

**Output**: 
- `results/phase1_data_params_<timestamp>/results.csv`
- Heatmap showing F1 score for each combination
- Best window & overlap printed

**What to look for**:
- Does longer window help? (captures more temporal context)
- Does more overlap help? (more training data, but slower)

**Example output**:
```
🏆 BEST CONFIGURATION:
Window Size : 70 samples (7.0s)
Step Size   : 10 samples
Overlap     : 60 samples (85.7%)
RF F1 Score : 0.7234
```

---

#### Step 2: Filter Optimization (15 min)

```bash
# Use best window/step from Phase 1
python3 scripts/phase2_filters.py --window 70 --step 10
```

**What it tests**:
- No filter (baseline)
- Lowpass 2Hz, 3Hz, 4Hz
- Bandpass 1-4Hz
- Median filter

**Output**:
- `results/phase2_filters_<timestamp>/results.csv`
- Bar chart comparing all filters
- Improvement over no filter

**What to look for**:
- Does filtering help? (should see +2-5% F1 if data is noisy)
- Which cutoff frequency works best?

**Example output**:
```
🏆 BEST FILTER:
Filter      : lowpass_3hz
RF F1 Score : 0.7489
Improvement over no filter: +2.55 percentage points
```

---

#### Step 3: LSTM Hyperparameters (60 min)

```bash
# Use best window/step/filter from Phase 1 & 2
python3 scripts/phase3_lstm_hyperparam.py --window 70 --step 10 --filter lowpass_3hz
```

**What it tests** (9 configurations):

| Config | Hidden | Layers | Dropout | Epochs | Rationale |
|--------|--------|--------|---------|--------|-----------|
| small_1layer | 64 | 1 | 0.2 | 20 | Simplest, fast |
| medium_2layer | 128 | 2 | 0.3 | 20 | Standard, quick |
| large_2layer | 256 | 2 | 0.3 | 20 | High capacity |
| small_2layer_30ep | 64 | 2 | 0.3 | 30 | More training |
| medium_2layer_30ep | 128 | 2 | 0.3 | 30 | Baseline (current default) |
| large_2layer_30ep | 256 | 2 | 0.3 | 30 | High capacity + training |
| medium_3layer | 128 | 3 | 0.4 | 30 | Deep architecture |
| large_3layer | 256 | 3 | 0.4 | 30 | Very deep |
| medium_2layer_50ep | 128 | 2 | 0.3 | 50 | Extended training |

**Output**:
- `results/phase3_lstm_hyperparam_<timestamp>/results.csv`
- F1 score vs training time scatter plot
- Top 5 configurations

**What to look for**:
- Does more capacity (larger hidden size) help?
- Does depth (3 layers) beat 2 layers?
- Diminishing returns at 50 epochs?

**Example output**:
```
🏆 BEST LSTM CONFIGURATION:
Name        : large_2layer_30ep
Hidden Size : 256
Num Layers  : 2
Dropout     : 0.3
Epochs      : 30
F1 Score    : 0.7812
Train Time  : 45.3s
```

---

## 📊 Expected Results

| Phase | Baseline | After Phase | Improvement |
|-------|----------|-------------|-------------|
| 0. Quick test | RF=0.68, LSTM=0.40 (5 ep) | LSTM=0.65 (30 ep) | +0.25 |
| 1. Window+Overlap | RF=0.68 | RF=0.72 | +0.04 |
| 2. Filter | RF=0.72 | RF=0.75 | +0.03 |
| 3. LSTM hyperparams | LSTM=0.65 | LSTM=0.78-0.82 | +0.13-0.17 |

**Final Target**: F1 > 0.78 (competitive), F1 > 0.80 (excellent)

---

## 🚀 Quick Start (Recommended Path)

### If you have 2 hours:
```bash
# Run everything automated
python3 scripts/run_all_phases.py
```

### If you have 1 hour:
```bash
# Skip Phase 1 (use defaults), focus on filter + LSTM
python3 scripts/phase2_filters.py --window 50 --step 25
# Then use best filter:
python3 scripts/phase3_lstm_hyperparam.py --window 50 --step 25 --filter lowpass_3hz
```

### If you have 30 minutes:
```bash
# Just optimize LSTM with default data params + filter
python3 scripts/phase3_lstm_hyperparam.py --window 50 --step 25 --filter lowpass_3hz
```

### If you have 5 minutes:
```bash
# Just see if 30 epochs helps
python3 scripts/quick_lstm_test.py
```

---

## 🎨 Understanding LSTM Hyperparameters

### Hidden Size (64, 128, 256)
- **What it does**: Size of memory cell in LSTM
- **Too small**: Can't capture complex patterns
- **Too large**: Overfitting, slow training
- **Sweet spot**: Usually 128-256 for HAR

### Num Layers (1, 2, 3)
- **What it does**: Stacked LSTMs learn hierarchical features
- **1 layer**: Simple temporal patterns
- **2 layers**: Most common, good balance
- **3 layers**: Deep features, may overfit on small data
- **Sweet spot**: Usually 2 layers

### Dropout (0.2, 0.3, 0.4)
- **What it does**: Randomly drops neurons during training to prevent overfitting
- **Too low (0.1)**: May overfit
- **Too high (0.5+)**: May underfit
- **Sweet spot**: 0.3 is standard

### Epochs (20, 30, 50)
- **What it does**: Number of passes through training data
- **Too few (10)**: Underfitting, high loss
- **Too many (100)**: Overfitting, diminishing returns
- **Sweet spot**: 30-50 for most HAR tasks

---

## 📈 How to Interpret Results

### Phase 1 (Window + Overlap):

**Heatmap colors**:
- 🟢 Green: Good F1 (> 0.72)
- 🟡 Yellow: Okay F1 (0.68-0.72)
- 🔴 Red: Poor F1 (< 0.68)

**Patterns to look for**:
- Longer windows (70-100) often better for sustained activities
- Low overlap (10-20%) often wins (less redundancy)
- But more overlap = more training samples = may help LSTM

### Phase 2 (Filters):

**Improvements**:
- < 1%: Filtering doesn't help much (clean data)
- 1-3%: Moderate benefit (typical)
- > 3%: Significant benefit (noisy data)

**Filter selection**:
- Lowpass 3Hz usually best for human activities (< 3 Hz motion)
- If bandpass wins: Data has DC drift (sensor bias)

### Phase 3 (LSTM):

**Look at scatter plot**:
- Top-left corner: Best F1, fast training ⭐
- Top-right corner: Best F1, slow training (may not be worth it)
- Bottom: Poor performers (discard)

**Architecture insights**:
- If 3 layers >> 2 layers: Your problem is complex
- If 256 >> 128: Need more capacity
- If 50 epochs >> 30 epochs: Undertrained

---

## 💡 What If Results Are Bad?

### LSTM F1 < 0.65 after Phase 3:
- ❌ LSTM may not be the right model for this data
- ✅ Stick with Random Forest (simpler, often better for tabular features)

### RF F1 < 0.70 after Phase 2:
- ❌ Features may be weak
- ✅ Add more features (FFT, statistical moments) in `data.py`
- ✅ Try XGBoost instead of RF

### No improvement from filters:
- ✅ Data is already clean!
- ✅ Use `filter: none` to save computation

---

## 🔄 After Optimization: Update Main Pipeline

Once you find optimal config, update `src/main.py`:

```python
# Edit these lines based on your results:
WINDOW_SIZE   = 70     # From Phase 1
STEP_SIZE     = 10     # From Phase 1
FILTER_CONFIG = {'method': 'lowpass', 'cutoff': 3, 'order': 4}  # From Phase 2
LSTM_HIDDEN   = 256    # From Phase 3
LSTM_LAYERS   = 2      # From Phase 3
LSTM_DROPOUT  = 0.3    # From Phase 3
LSTM_EPOCHS   = 30     # From Phase 3
```

Then run final training:
```bash
python3 src/main.py
```

This will:
- Train with optimal config
- Log to WandB
- Save combined confusion matrix
- Generate full report in `results/<timestamp>/`

---

## 📝 Summary

**Question**: How did you decide LSTM hyperparameters?
**Answer**: They're research-based defaults, NOT optimized for your data.

**Question**: How do I optimize them?
**Answer**: Run Phase 3 script after optimizing data params in Phases 1-2.

**Question**: What should I run?
**Answer**: 
- Quick: `python3 scripts/run_all_phases.py` (2 hours, automated)
- Manual: Run phase1 → phase2 → phase3 scripts individually

**Best Approach**:
1. Optimize data with RF (fast)
2. Optimize LSTM architecture (slow)
3. Don't grid search everything at once!

**Scripts created**:
- ✅ `scripts/quick_lstm_test.py` - 2 min baseline
- ✅ `scripts/phase1_data_params.py` - 30 min window+overlap
- ✅ `scripts/phase2_filters.py` - 15 min filter comparison
- ✅ `scripts/phase3_lstm_hyperparam.py` - 60 min LSTM tuning
- ✅ `scripts/run_all_phases.py` - 2 hours automated

Ready to run! 🚀
