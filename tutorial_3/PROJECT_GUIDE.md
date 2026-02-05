# Tutorial 3 Quick Guide - HAR Model Optimization

## PROJECT OVERVIEW
**Goal:** Train IMU-based Human Activity Recognition (HAR) model on wrist accelerometer data, maximize F1 score for competition.

**Dataset:** 14 activities from 16 volunteers (Brush_teeth, Climb_stairs, Walk, etc.)
**Current Status:** Notebook complete through Activity 1; remaining work = feature engineering + model optimization

---

## WHAT YOU'RE DOING
Building an **Activity Recognition Chain**:
1. Load accelerometer data (X, Y, Z axes @ 32Hz)
2. Create sliding windows (temporal segments)
3. Extract statistical features from windows
4. Train classifier (SVM/DecisionTree baseline)
5. Evaluate with Leave-One-Subject-Out (LOSO) cross-validation

**Why LOSO?** Prevents data leakage, tests generalization to unseen users (real-world scenario).

---

## SECTIONS BREAKDOWN

### **SECTION 1: Data Loading & EDA** (Lines 1-430)
**Status:** ✅ DONE

**Notes:**
- Magnitude stats are already filled.
- Activity 1 answers live in a separate markdown.

**Key Insight:** Different activities show distinct magnitude patterns. Walk/Climb_stairs = high variance, Drink/Pour = low variance.

---

### **SECTION 2: Feature Extraction (tsfresh)** (Lines 432-520)
**Status:** ✅ DONE (baseline features extracted)

**What's Happening:**
- 7 feature functions × 3 axes = **21 features** per file (X/Y/Z)
- Each file (activity recording) = 1 sample
- ~166 samples total

**TO IMPROVE F1 SCORE:**
- Add more feature functions (see optimization section)
- Current: 21 features → Try: 30–50+ features

---

### **SECTION 3: Baseline Model (no windows)** (Lines 521-641)
**Status:** ✅ DONE

**Results:** SVM & DecisionTree trained on file-level features with LOSO CV.

**Note:** This is weak baseline (1 sample per file = limited data).

---

### **SECTION 4: Sliding Windows** (Lines 643-823)
**Status:** ✅ DONE (code complete)

**What Changed:**
- Window size: 30 samples (1 sec @ 32Hz)
- Overlap: 15 samples (50%)
- Features per window: mean, mode, median, max, min, IQR (6 stats × 3 axes = 18 features)

**Notes:**
- Assertion for `WINDOW_SIZE` is already in place.

---

### **SECTION 5: Homework Tasks** (Line 936)
**Status:** ❌ TODO

**Required:**
1. **Add signal filtering** (e.g., Butterworth low-pass)
2. **Test 3 window sizes** (recommended: 32, 64, 128 samples)
3. **[Optional]** Try Random Forest / XGBoost

---

## OPTIMIZATION STRATEGY (TO WIN F1 COMPETITION)

### **PHASE 1: Feature Engineering** (Biggest Impact)
**Priority 1 - Add Time-Domain Features:**
```python
feature_parameters = {
    'mean': None, 'median': None,
    'standard_deviation': None, 'variance': None,
    'minimum': None, 'maximum': None,
    'quantile': [{'q': 0.25}, {'q': 0.75}],
    'abs_energy': None,  # Signal energy
    'absolute_sum_of_changes': None,
    'count_above_mean': None,
    'count_below_mean': None,
    'kurtosis': None, 'skewness': None,
    'range_count': [{'min': -1, 'max': 1}],
}
```

**Priority 2 - Add Frequency Features (FFT):**
```python
'fft_coefficient': [{'coeff': i} for i in range(10)],
'fft_aggregated': [{'aggtype': 'centroid'}],
'spectral_entropy': [{'normalize': True}],
```

**Priority 3 - Add Magnitude & Jerk:**
- Add `Magnitude = sqrt(X² + Y² + Z²)` column before windowing
- Add `Jerk = diff(acceleration)` (rate of change)

### **PHASE 2: Preprocessing**
**Add Signal Filtering:**
```python
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=10, fs=32, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data, axis=0)

# Apply before windowing
filtered_x = butter_lowpass_filter(x)
```

### **PHASE 3: Window Optimization**
**Test Window Sizes:**
- **Small (32):** Good for short activities (Pour_water, Drink)
- **Medium (64):** Balanced
- **Large (128):** Good for periodic activities (Walk, Climb_stairs)

**Overlap Testing:** Try 25%, 50%, 75%

### **PHASE 4: Model Selection**
**Replace SVM with Better Models:**

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest (usually best for HAR)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced'
)

# XGBoost (if installed)
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8
)
```

### **PHASE 5: Hyperparameter Tuning**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Note: Use nested CV to avoid overfitting
```

---

## QUICK WIN CHECKLIST
- [ ] Add Magnitude column to features
- [ ] Increase feature count to 30-50
- [ ] Add low-pass filter (cutoff=10Hz)
- [ ] Test window size = 64 samples
- [ ] Switch to RandomForest
- [ ] Add class balancing (`class_weight='balanced'`)

**Expected F1 improvement:** 0.60 → 0.75+

---

## WHERE TO SUBMIT
**Location:** Check `link.txt` file or Moodle assignment page

**Required Submission:**
1. **F1 Score** (macro-averaged from LOSO CV)
2. **Activity 1 answers** (doc file with EDA questions)
3. **[Optional]** Notebook with best model

**Format:** Leaderboard submission = single F1 score value

---

## CODE LOCATIONS

**Cell 30:** Complete activity stats
**Cell 62:** Add assertion for window shape
**Cell 51:** Modify `feature_parameters` (add features)
**Cell 55:** Modify `train_and_classify()` (add RF/XGB)
**Cell 62:** Modify `feature_representation()` (add Magnitude/Jerk)

---

## KEY CONCEPTS

**Data Leakage:** Scaling AFTER train/test split (line 601)
**LOSO CV:** Each volunteer left out once = 16 folds
**Window Overlap:** More overlap = more samples but slower training
**Macro F1:** Average F1 across all classes (handles imbalance)

---

## TYPICAL F1 SCORE RANGES
- Baseline (file-level): ~0.50-0.60
- With windows: ~0.65-0.75
- Optimized features: ~0.75-0.85
- Best models: ~0.85-0.92

**Target to beat:** Check class leaderboard (likely ~0.70-0.80)

---

## ERRORS SEEN IN THE NOTEBOOK (FROM SOURCE)

1. **Walk plot title uses undefined variable `activity`**
    - In the walking example plot, replace the title with a fixed label:
      - Use `plt.title('Activity: walk')` or define `activity = 'walk'`.

2. **Dataset download is commented out**
    - The download line is commented, so `adl_dataset` may not be a zip file.
    - Fix options:
      - Download the zip and set `zip_file_path` to the zip file, or
      - Skip extraction and point directly to the already extracted folder.

3. **`zip_file_path` points to a folder**
    - `read_dataset()` expects a zip file for `zip_file_path`.
    - If you already have extracted data, update the function to skip the zip step.
