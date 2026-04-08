# Summary of Changes - Assignment 1 Updates

## ✅ All Tasks Completed

### Task 1: Restructured Image Outputs ✓

**Problem**: Images were being dumped into a shared `images/` folder with too many files

**Solution**:
1. **Main Pipeline (`main.py`)**:
   - Now saves ALL outputs to `results/<run_id>/` folder
   - Creates timestamped run folder (e.g., `results/20260310_161544/`)
   - All plots (confusion matrices, F1 charts, training loss) saved there
   - `images/` folder is now for manual sorting only, not used by scripts

2. **Experiments (`experiments.py`)**:
   - Each experiment gets its own timestamped folder:
     - `results/exp1_window_sweep_<timestamp>/`
     - `results/exp2_step_sweep_<timestamp>/`
     - `results/exp3_filter_<timestamp>/`
     - `results/exp4_model_comparison_<timestamp>/`
   - Each folder contains:
     - `results.csv` - experiment data
     - `*_plot.png` - visualization

3. **Combined Confusion Matrix (2x2 Grid)**:
   - New function: `plot_combined_confusion_matrices()`
   - Displays all 4 models (LSTM, SVM, DT, RF) in one figure
   - Shows F1 score in each subplot title
   - Saved as `confusion_matrices_combined.png`
   - Individual confusion matrices commented out (can be re-enabled)

**Files Modified**:
- ✓ [src/plot.py](src/plot.py) - Added `plot_combined_confusion_matrices()` function
- ✓ [src/main.py](src/main.py) - Changed output directory to `results/<run_id>/`
- ✓ [src/experiments.py](src/experiments.py) - Each experiment creates its own folder

---

### Task 2: WandB Integration ✓

**Problem**: Need to use your WandB account without affecting friend's session on same machine

**Solution**:
1. **Created `src/wandb_config.py`**:
   - Loads API key from `.env` file (not from global login)
   - Functions: `init_wandb()`, `finish_wandb()`, `log_metrics()`
   - Process-isolated: doesn't affect other users' WandB sessions

2. **Created `.env` file**:
   - Stores your API key: `WANDB_API_KEY=wandb_v1_...`
   - Added to `.gitignore` (not pushed to GitHub)

3. **Created `.gitignore`**:
   - Excludes: `.env`, `wandb/`, `results/`, `images/`, etc.

4. **Integrated WandB Logging**:
   - **LSTM Training** ([src/model.py](src/model.py)):
     - Logs loss per epoch: `lstm/train_loss`
     - Logs learning rate: `lstm/learning_rate`
     - Logs final metrics: `lstm/train_time_sec`, `lstm/final_loss`
     - Logs CPU usage if monitored
   
   - **sklearn Models** ([src/model.py](src/model.py)):
     - Logs F1 scores: `svm/f1_weighted`, `dt/f1_weighted`, `rf/f1_weighted`
     - Logs CPU stats: `sklearn/cpu_avg`, `sklearn/cpu_max`
   
   - **Main Pipeline** ([src/main.py](src/main.py)):
     - Initializes WandB at start with config (window size, LSTM params, etc.)
     - Logs final metrics for all models
     - Uploads combined confusion matrix image
     - Finishes run cleanly

5. **WandB Project**:
   - Project name: `coen498-assignment1`
   - Team: `mimic-robotics` (your username: `ac-pate`)
   - URL: https://wandb.ai/mimic-robotics/coen498-assignment1

**Usage**:
```python
# WandB automatically initializes in main.py
# Logs training metrics automatically
# View dashboard: https://wandb.ai/mimic-robotics/coen498-assignment1
```

**Files Created**:
- ✓ [src/wandb_config.py](src/wandb_config.py) - WandB authentication & logging
- ✓ [.env](.env) - API key storage (gitignored)
- ✓ [.gitignore](.gitignore) - Excludes sensitive files

**Files Modified**:
- ✓ [src/model.py](src/model.py) - Added wandb logging to training functions
- ✓ [src/main.py](src/main.py) - Initialize wandb, log final metrics

---

### Task 3: Sampling Frequency Verification ✓

**Question**: How do we know the sampling frequency is 10 Hz?

**Answer**: Analyzed actual timestamp data

**Verification**:
```python
# Analyzed df_train.csv timestamps:
# Timestamp differences: 0.1 seconds (constant)
# Sampling frequency = 1 / 0.1 = 10 Hz
```

**Results**:
- Mean time diff: 0.100000 seconds
- Median time diff: 0.100000 seconds
- **Confirmed: 10 Hz sampling rate**

**This is different from tutorial_3 which had 32 Hz**

**Files Modified**:
- ✓ [src/data.py](src/data.py) - Updated comment to clarify 10 Hz is verified from data analysis

---

### Task 4: Experiment Strategy Guide ✓

**Analysis of Tutorial 3 Approach**:

From commit history and results folders, you followed this progression:
1. ✅ Created initial models (SVM, RF, DT)
2. ✅ Optimized for CPU multi-threading  
3. ✅ Ran window size sweeps (tested various window sizes)
4. ✅ Ran overlap sweeps (found window=170, overlap=5 worked best)
5. ✅ Tested filters (lowpass, bandpass, median)
6. ✅ Found lowpass filter was best
7. ✅ Swept lowpass filter cutoff frequencies
8. ✅ Achieved final F1 ≈ 0.488

**Created Comprehensive Strategy for Assignment 1**:

**Phased Approach**:
- **Phase 1**: Baseline (✓ done) - window=50, step=25, no filter
  - RF F1 = 0.68
  - LSTM F1 = 0.40 (only 5 epochs)

- **Phase 2**: Window Size Sweep
  - Test: 20, 30, 50, 70, 100 samples
  - Find optimal duration for 10 Hz sampling
  
- **Phase 3**: Overlap Sweep  
  - Test: 5, 10, 25, 40, 45 step sizes
  - Balance data quantity vs computation time

- **Phase 4**: Filter Comparison
  - Test: None, Lowpass 3Hz, Lowpass 4Hz, Bandpass, Median
  - Expected: Lowpass helps (based on tutorial_3)

- **Phase 5**: Filter Parameter Tuning
  - If lowpass wins: sweep 2.0-4.5 Hz cutoffs
  - If bandpass wins: sweep frequency ranges

- **Phase 6**: Full Model Comparison
  - Compare LSTM vs sklearn models with optimal config
  - Predict winner: RF or LSTM (need to test)

**Quick Wins Identified**:
1. Train LSTM for 30 epochs (instead of 5): Expected +0.15 F1
2. Add lowpass filter (cutoff=3 Hz): Expected +0.03-0.05 F1
3. Try window=70 samples: May improve temporal context

**Expected Progression**:
- Baseline: 0.68 (✓)
- After Phase 2-3: 0.73-0.75
- After Phase 4-5: 0.76-0.78  
- Final (best model): **0.78-0.82**

**Files Created**:
- ✓ [EXPERIMENT_STRATEGY.md](EXPERIMENT_STRATEGY.md) - Comprehensive experiment guide

---

## 📁 Updated Project Structure

```
assignment_1/
├── .env                    # ✓ NEW - API keys (gitignored)
├── .gitignore              # ✓ NEW - Excludes sensitive files
├── README.md
├── TESTING_GUIDE.md        # ✓ CREATED EARLIER
├── EXPERIMENT_STRATEGY.md  # ✓ NEW - Experiment plan
├── df_train.csv
├── Assignment 1 (2).pdf
├── COEN_498_Assignment_1.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py             # ✓ MODIFIED - Clarified 10 Hz sampling
│   ├── model.py            # ✓ MODIFIED - Added wandb logging
│   ├── evaluate.py
│   ├── plot.py             # ✓ MODIFIED - Added combined confusion matrix
│   ├── main.py             # ✓ MODIFIED - WandB + new output structure
│   ├── experiments.py      # ✓ MODIFIED - Experiment-specific folders
│   └── wandb_config.py     # ✓ NEW - WandB authentication
├── results/                # ✓ MODIFIED - Now contains all outputs
│   ├── <run_id>/          # Main pipeline runs
│   │   ├── metrics.txt
│   │   ├── summary.csv
│   │   ├── confusion_matrices_combined.png  # ✓ NEW - 2x2 grid
│   │   ├── per_class_f1.png
│   │   └── lstm_training_loss.png
│   ├── exp1_window_sweep_<timestamp>/  # ✓ NEW structure
│   │   ├── results.csv
│   │   └── window_sweep_plot.png
│   └── exp*_<timestamp>/  # Other experiments
└── images/                # ✓ NOW - Manual sorting only (not used by scripts)
```

---

## 🎯 What Changed vs Before

| Aspect | Before | After |
|--------|--------|-------|
| **Image Storage** | All in `images/` folder | Each run/experiment in own folder |
| **Confusion Matrices** | 4 separate PNG files | 1 combined 2x2 grid PNG |
| **WandB** | Not integrated | Fully integrated with logging |
| **Experiments** | CSV in `results/`, plots in `images/` | Everything in `results/<exp>_<timestamp>/` |
| **Sampling Freq** | Assumed 10 Hz | Verified from data (10 Hz) |
| **Strategy** | No formal plan | Comprehensive phased plan |

---

## 🚀 How to Use

### 1. Run Main Pipeline with Optimal Settings (Quick Win)

**Edit [src/main.py](src/main.py)**:
```python
WINDOW_SIZE   = 50
STEP_SIZE     = 25
LSTM_EPOCHS   = 30  # Changed from 5
FILTER_CONFIG = {'method': 'lowpass', 'cutoff': 3, 'order': 4}  # Added filter
```

**Run**:
```bash
cd ~/achal/Pervasive-Computing-for-Health/assignment_1
python3 src/main.py
```

**Output**: 
- Results in `results/<timestamp>/`
- WandB dashboard: https://wandb.ai/mimic-robotics/coen498-assignment1
- Expected F1: 0.73-0.76 (RF or LSTM)

### 2. Run Full Experiment Campaign

```bash
python3 src/experiments.py
```

**Output**:
- 4 experiment folders in `results/`
- Each with CSV data + plot
- Time: ~2 hours

### 3. Monitor Progress on WandB

Visit: https://wandb.ai/mimic-robotics/coen498-assignment1

**What's Logged**:
- LSTM training loss per epoch
- Learning rate schedule
- All model F1 scores
- CPU/GPU utilization
- Confusion matrix images
- Final metrics table

---

## 🔐 Security

✅ **API Key Protected**:
- Stored in `.env` file (gitignored)
- Not hardcoded in any Python file
- Won't be pushed to GitHub

✅ **Process Isolated**:
- Uses `WANDB_API_KEY` environment variable
- Doesn't affect friend's global wandb login
- Safe to run on shared machine

---

## 📝 Testing Commands

### Test WandB Integration
```bash
cd ~/achal/Pervasive-Computing-for-Health/assignment_1
python3 src/wandb_config.py
# Should show: "✓ WandB initialization successful"
```

### Test Data Loading
```bash
python3 src/data.py
# Verifies 10 Hz sampling, creates windows
```

### Test Models
```bash
python3 src/model.py
# Tests LSTM (GPU), SVM/RF (CPU multi-threading)
```

### Test Quick Pipeline (2 min)
```bash
python3 -c "
import sys, pathlib
sys.path.insert(0, 'src')
import main
main.LSTM_EPOCHS = 2  # Quick test
main.run_pipeline()
"
```

---

## 🎓 Key Learnings Applied from Tutorial 3

1. ✅ **CPU Multi-threading**: Properly configured BLAS threads
2. ✅ **Experiment Structure**: Phased approach (window → overlap → filter → models)
3. ✅ **Filter Impact**: Lowpass filter significantly improved F1 in tutorial_3
4. ✅ **Minimal Overlap**: Low overlap (high step) worked best in tutorial_3
5. ✅ **RF Strong Baseline**: Random Forest consistently performs well for HAR

**Adapted for Assignment 1**:
- Different sampling rate (10 Hz vs 32 Hz)
- Adjusted window/overlap recommendations
- Added LSTM model (GPU-trained)
- Added WandB tracking
- Better output organization

---

## 📊 Expected Results

| Metric | Baseline (Now) | After Phase 2-3 | After Phase 4-5 | Final (Best Model) |
|--------|---------------|-----------------|-----------------|-------------------|
| RF F1  | 0.68          | 0.73            | 0.76            | 0.76-0.78         |
| LSTM F1 | 0.40 (5 epochs) | 0.60-0.65      | 0.70-0.75       | **0.78-0.82**     |
| SVM F1 | 0.48          | 0.53            | 0.58            | 0.58-0.60         |

**Target**: F1 > 0.78 (competitive)  
**Stretch**: F1 > 0.80 (excellent)

---

## ✅ All Checklist Items Completed

- [x] Verified sampling frequency from actual data (10 Hz)
- [x] Created WandB integration with API key protection
- [x] Restructured outputs to `results/<run_id>/` folders
- [x] Created 2x2 combined confusion matrix grid
- [x] Updated experiments to save everything to timestamped folders
- [x] Integrated WandB logging in LSTM training
- [x] Integrated WandB logging in sklearn training
- [x] Integrated WandB in main pipeline
- [x] Analyzed tutorial_3 commit history and approach
- [x] Created comprehensive experiment strategy guide
- [x] Added .gitignore for security
- [x] Updated all documentation

---

## 🚀 Next Steps

1. **Immediate** (5 min):
   - Review [EXPERIMENT_STRATEGY.md](EXPERIMENT_STRATEGY.md)
   - Decide: Quick wins or full campaign

2. **Quick Win** (30 min):
   - Edit `main.py`: `LSTM_EPOCHS=30`, add filter
   - Run `python3 src/main.py`
   - Check WandB dashboard

3. **Full Campaign** (2 hours):
   - Run `python3 src/experiments.py`
   - Analyze results
   - Re-run with optimal config

4. **Final Submission**:
   - Use best model from experiments
   - Submit F1 score
   - Upload notebook with results

**Good luck! 🎯**
