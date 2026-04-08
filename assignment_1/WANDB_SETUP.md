# WandB Organization Guide

## 🎯 What Changed

**Problem**: Multiple experimental runs cluttered WandB with separate runs, sklearn models showed empty plots.

**Solution**: 
1. **Groups**: Related runs are grouped together for easy comparison
2. **Clean Metrics**: Only LSTM training curves (loss + LR) are logged per-epoch
3. **Summary Values**: Final F1 scores stored as summaries (not time-series)

---

## 📊 What You'll See in WandB

### LSTM Training (Per-Epoch)
✅ **Shows up as nice charts:**
- `lstm/train_loss` - Training loss over epochs
- `lstm/learning_rate` - Learning rate schedule
- `lstm/epoch` - Current epoch

### Final Metrics (Summary Only)
📝 **Shows up as single values (no charts):**
- `lstm_f1`, `rf_f1`, `svm_f1`, `dt_f1` - Final F1 scores
- `total_time_min` - Total training time
- `confusion_matrix` - Confusion matrix image

### Why sklearn metrics are "empty"
They're logged as **summary values**, not time-series. This is correct - sklearn models don't have per-epoch data like LSTM does. They just train once and return a final F1 score.

---

## 🗂️ How Runs Are Organized

### Groups (Keeps Things Tidy)

**Manual Training Runs:**
```
Group: "manual_training"
├─ main_20260310_143022  (your manual run 1)
├─ main_20260310_145510  (your manual run 2)
└─ main_20260310_150912  (your manual run 3)
```
All your `python3 src/main.py` runs group together.

**Quick Tests:**
```
Group: "quick_tests"
├─ quick_baseline_20260310_120000
└─ quick_baseline_20260310_130000
```

**Optimization Campaigns:**
```
Group: "optimization_20260310"  (shared campaign ID)
├─ phase1_data_params_20260310_120000
├─ phase2_filters_20260310_123045
└─ phase3_lstm_20260310_130122
```
All phases of one optimization campaign group together! This lets you:
- Compare Phase 1 window configs side-by-side
- See filter improvements in Phase 2
- Compare LSTM architectures in Phase 3
- Track progression across phases

---

## 🚀 Running Experiments

### Option 1: Manual Training
```bash
python3 src/main.py
```
- Creates run in group `"manual_training"`
- Logs LSTM training curves
- Logs final F1 scores as summary
- Uploads confusion matrix image

### Option 2: Quick Baseline
```bash
python3 scripts/quick_lstm_test.py
```
- Creates run in group `"quick_tests"`
- Logs LSTM training curves (30 epochs)
- Logs final LSTM F1 as summary

### Option 3: Full Optimization (Grouped)
```bash
python3 scripts/run_all_phases.py
```
This runs:
1. **Phase 1**: 12 window+overlap configs → logged to `optimization_<timestamp>`
2. **Phase 2**: 6 filter configs → logged to same group
3. **Phase 3**: 9 LSTM configs → logged to same group

**Result**: All ~27 configs appear in ONE grouped campaign in WandB! 🎉

You can:
- Filter by tags: `phase1`, `phase2`, `phase3`
- Compare all Phase 1 runs side-by-side
- See best config from each phase
- Track improvement across phases

### Option 4: Individual Phases
```bash
# Phase 1 (creates group "optimization_<timestamp>")
python3 scripts/phase1_data_params.py

# Phase 2 (use same campaign group manually or let it create new one)
python3 scripts/phase2_filters.py --window 50 --step 25

# Phase 3
python3 scripts/phase3_lstm_hyperparam.py --window 50 --step 25 --filter lowpass_3hz
```

---

## 📈 WandB Dashboard Views

### View 1: Training Curves (LSTM Only)
**Filter by:** Group = `manual_training`, Metrics = `lstm/train_loss`
**What you see:** Loss curves for all your manual training runs overlaid

### View 2: Compare Window Sizes (Phase 1)
**Filter by:** Group = `optimization_20260310`, Tags = `phase1`
**What you see:** All 12 window+overlap configs with their RF F1 scores

**Create a grouped bar chart:**
- X-axis: `window_size`
- Y-axis: `rf_f1`
- Group by: `overlap_pct`

### View 3: Compare Filters (Phase 2)
**Filter by:** Group = `optimization_20260310`, Tags = `phase2`
**What you see:** All 6 filter types with RF F1 scores

### View 4: Compare LSTM Architectures (Phase 3)
**Filter by:** Group = `optimization_20260310`, Tags = `phase3`
**What you see:** All 9 LSTM configs with F1 scores and training times

**Create scatter plot:**
- X-axis: `train_time_sec`
- Y-axis: `lstm_f1`
- Size: `hidden_size`
- Color: `num_layers`

---

## 🔍 Finding Best Configs

### In WandB UI:
1. Go to project: `coen498-assignment1`
2. Click "Runs" tab
3. Filter by group (e.g., `optimization_20260310`)
4. Sort by `best_rf_f1` (Phase 1/2) or `best_f1` (Phase 3)
5. Top row = best config!

### Programmatically:
```python
import wandb

api = wandb.Api()
runs = api.runs("mimic-robotics/coen498-assignment1", 
                filters={"group": "optimization_20260310", "tags": "phase1"})

# Find best RF F1 from Phase 1
best_run = max(runs, key=lambda r: r.summary.get('best_rf_f1', 0))
print(f"Best window: {best_run.summary['best_window_size']}")
print(f"Best step: {best_run.summary['best_step_size']}")
```

---

## 💡 Pro Tips

### Naming Conventions
- **Manual runs**: `main_<timestamp>`
- **Phase runs**: `phase<N>_<name>_<timestamp>`
- **Groups**: `manual_training`, `quick_tests`, `optimization_<timestamp>`

### Tags Are Your Friend
Add custom tags:
```python
wandb_run = init_wandb(
    name='experiment_final',
    group='manual_training',
    tags=['lstm', 'final', 'tuned', 'submission']
)
```
Then filter by tag in WandB UI!

### Compare Across Groups
Want to compare best from each campaign?
1. Find best run from group `optimization_20260310`
2. Find best run from group `optimization_20260311`
3. Select both → "Compare" → See side-by-side

### Export Results
In WandB:
1. Select runs
2. "Export" → CSV
3. Get all metrics + hyperparameters in spreadsheet

---

## 🛠️ Troubleshooting

### "Sklearn metrics are empty"
✅ **This is normal!** Sklearn models don't have per-epoch data. Their F1 scores are logged as **summary values** (not time-series), so they won't create charts. Only LSTM has training curves.

### "Too many runs in dashboard"
Use **Groups** view:
1. Click "Groups" tab (not "Runs")
2. Now you see campaigns as single items
3. Expand a group to see individual runs within it

### "Can't find my run"
Check the group:
- Manual runs → group `"manual_training"`
- Optimization → group `"optimization_<date>"`
- Quick tests → group `"quick_tests"`

### "Want to delete old runs"
In WandB UI:
1. Select runs (checkbox)
2. "Actions" → "Delete"
3. Or keep them! Storage is free on hobby tier

---

## 📚 Summary

| What | Where | Group | Metrics |
|------|-------|-------|---------|
| Manual training | `python3 src/main.py` | `manual_training` | LSTM curves + final F1s |
| Quick test | `scripts/quick_lstm_test.py` | `quick_tests` | LSTM curves + F1 |
| Phase 1 | `scripts/phase1_data_params.py` | `optimization_<date>` | RF F1 per config |
| Phase 2 | `scripts/phase2_filters.py` | `optimization_<date>` | RF F1 per filter |
| Phase 3 | `scripts/phase3_lstm_hyperparam.py` | `optimization_<date>` | LSTM curves + F1 per config |
| All phases | `scripts/run_all_phases.py` | `optimization_<date>` | Everything above |

**Key Insight**: Groups organize related experiments, tags filter within groups, summaries store final values!

---

## 🎨 Example WandB Workspace Layout

```
Project: coen498-assignment1

Groups View:
├─ manual_training (5 runs)
│  └─ Best F1: 0.78 (main_20260310_150912)
│
├─ quick_tests (3 runs)
│  └─ Best F1: 0.65 (quick_baseline_20260310_130000)
│
└─ optimization_20260310 (27 runs)
   ├─ Phase 1: 12 runs → Best: window=70, step=10, F1=0.72
   ├─ Phase 2: 6 runs → Best: lowpass_3hz, F1=0.75
   └─ Phase 3: 9 runs → Best: large_2layer_30ep, F1=0.81

Click expand → see individual runs with full metrics
```

**Clean. Organized. Easy to navigate.** 🚀

---

## Questions?

**Q: Why not log sklearn training progress?**
**A:** Sklearn models train in one shot (no epochs). There's no "progress" to log.

**Q: Can I disable WandB for a run?**
**A:** Yes! Comment out `init_wandb()` in the script or set `WANDB_MODE=disabled` env var.

**Q: What if I want separate runs instead of groups?**
**A:** Remove the `group=` parameter in `init_wandb()`. But groups are better for organization!

**Q: Can I rename a group after running?**
**A:** No, but you can add tags retroactively in the WandB UI.

Ready to run! 🎉
