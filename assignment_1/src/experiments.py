"""
Assignment 1 - Cognitive Decline Activity Recognition
Experimentation Campaign  (mirrors tutorial_3/experiments.py style)

Experiments:
  1. Window Size Sweep  – vary window size, fixed step, fixed model (RF)
  2. Step (Overlap) Sweep  – vary step, fixed window, fixed model (RF)
  3. Filter Comparison  – compare filter strategies
  4. Model Comparison at best config  – LSTM vs SVM vs DT vs RF

Run from assignment_1/ directory:
    python src/experiments.py

Results are saved to results/<experiment>_<run_id>/
"""

# ============================================================================
# CRITICAL: CPU threading before any data imports
# ============================================================================
import os
import multiprocessing

_num_cores = str(multiprocessing.cpu_count())
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(_v, _num_cores)

print(f"[INIT] BLAS threads → {_num_cores} cores")

# ============================================================================
# Imports
# ============================================================================
import sys
import pathlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

_SRC_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_SRC_DIR))

from data     import load_data, prepare_ml_data, SAMPLING_FREQ
from model    import (get_cpu_info, get_device,
                      train_sklearn_models,
                      train_lstm, predict_lstm)
from evaluate import evaluate_all, save_metrics

# ============================================================================
# PATHS
# ============================================================================
CSV_PATH    = pathlib.Path(__file__).parent.parent / 'df_train.csv'
RESULTS_DIR = pathlib.Path(__file__).parent.parent / 'results'
IMAGES_DIR  = pathlib.Path(__file__).parent.parent / 'images'

N_JOBS = -1   # all cores for sklearn


# ============================================================================
# HELPER: single-config RF evaluation
# ============================================================================

def _eval_rf(df: pd.DataFrame, window_size: int, step_size: int,
             filter_config=None, verbose: bool = False) -> dict:
    """
    Prepare data and evaluate Random Forest for one window/step config.

    Returns dict with: window_size, step_size, num_samples, rf_f1
    """
    try:
        data = prepare_ml_data(df, window_size=window_size,
                               step_size=step_size,
                               filter_config=filter_config,
                               verbose=verbose)
        res = train_sklearn_models(
            data['X_train'], data['y_train'],
            data['X_test'],  data['y_test'],
            n_jobs=N_JOBS, verbose=False
        )
        return {
            'window_size':  window_size,
            'step_size':    step_size,
            'num_samples':  data['X_train'].shape[0] + data['X_test'].shape[0],
            'rf_f1':        res['rf_f1'],
            'dt_f1':        res['dt_f1'],
            'svm_f1':       res['svm_f1'],
        }
    except Exception as e:
        print(f"[WARN] Config w={window_size} s={step_size} failed: {e}")
        return None


# ============================================================================
# EXPERIMENT 1 – Window Size Sweep
# ============================================================================

def experiment_1_window_sweep(df: pd.DataFrame,
                               window_sizes=None,
                               step_fixed: int = 25) -> pd.DataFrame:
    """
    Sweep window sizes with a fixed step.

    Parameters
    ----------
    df : pd.DataFrame
    window_sizes : list[int] or None  (default: 20, 30, 40, 50, 75, 100)
    step_fixed : int

    Returns
    -------
    pd.DataFrame  results table
    """
    if window_sizes is None:
        window_sizes = [20, 30, 40, 50, 75, 100]

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Window Size Sweep")
    print(f"  Window sizes : {window_sizes}")
    print(f"  Fixed step   : {step_fixed}")
    print("=" * 70)

    results = []
    for ws in tqdm(window_sizes, desc="Exp1 Window Sweep"):
        if step_fixed >= ws:
            print(f"[SKIP] step={step_fixed} >= window={ws}")
            continue
        print(f"\n[EXP1] window={ws}, step={step_fixed}")
        row = _eval_rf(df, ws, step_fixed, verbose=False)
        if row:
            results.append(row)
            print(f"  RF F1: {row['rf_f1']:.4f}  |  SVM F1: {row['svm_f1']:.4f}")

    return pd.DataFrame(results)


# ============================================================================
# EXPERIMENT 2 – Step (Overlap) Sweep
# ============================================================================

def experiment_2_step_sweep(df: pd.DataFrame,
                             steps=None,
                             window_fixed: int = 50) -> pd.DataFrame:
    """
    Sweep step sizes (overlap) with a fixed window size.

    Parameters
    ----------
    df : pd.DataFrame
    steps : list[int] or None
    window_fixed : int

    Returns
    -------
    pd.DataFrame
    """
    if steps is None:
        steps = [5, 10, 15, 25, 35, 45]

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Step (Overlap) Sweep")
    print(f"  Fixed window : {window_fixed}")
    print(f"  Steps tested : {steps}")
    print("=" * 70)

    results = []
    for ss in tqdm(steps, desc="Exp2 Step Sweep"):
        if ss >= window_fixed:
            print(f"[SKIP] step={ss} >= window={window_fixed}")
            continue
        print(f"\n[EXP2] window={window_fixed}, step={ss}  "
              f"(overlap={(1-ss/window_fixed)*100:.0f}%)")
        row = _eval_rf(df, window_fixed, ss, verbose=False)
        if row:
            results.append(row)
            print(f"  RF F1: {row['rf_f1']:.4f}")

    return pd.DataFrame(results)


# ============================================================================
# EXPERIMENT 3 – Filter Comparison
# ============================================================================

def experiment_3_filter_comparison(df: pd.DataFrame,
                                    window_size: int = 50,
                                    step_size: int = 25) -> pd.DataFrame:
    """
    Compare different filter strategies on a fixed window/step config.

    Parameters
    ----------
    df : pd.DataFrame
    window_size : int
    step_size : int

    Returns
    -------
    pd.DataFrame
    """
    filter_configs = [
        ('No Filter',           None),
        ('Median k=3',          {'method': 'median',  'kernel_size': 3}),
        ('Median k=9',          {'method': 'median',  'kernel_size': 9}),
        ('Lowpass 4Hz',         {'method': 'lowpass', 'cutoff': 4, 'order': 4}),
        ('Lowpass 2Hz',         {'method': 'lowpass', 'cutoff': 2, 'order': 4}),
        ('Bandpass 0.5-4Hz',    {'method': 'bandpass','low': 0.5, 'high': 4, 'order': 4}),
        ('Median→Lowpass',      {'method': 'combined', 'filters': [
                                    {'method': 'median', 'kernel_size': 3},
                                    {'method': 'lowpass', 'cutoff': 4, 'order': 4},
                                ]}),
    ]

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Filter Comparison")
    print(f"  Window={window_size}, Step={step_size}")
    print("=" * 70)

    results = []
    for name, cfg in tqdm(filter_configs, desc="Exp3 Filters"):
        print(f"\n[EXP3] Filter: {name}")
        row = _eval_rf(df, window_size, step_size, filter_config=cfg, verbose=False)
        if row:
            row['filter'] = name
            results.append(row)
            print(f"  RF F1: {row['rf_f1']:.4f}")

    return pd.DataFrame(results)


# ============================================================================
# EXPERIMENT 4 – Full Model Comparison at best config
# ============================================================================

def experiment_4_model_comparison(df: pd.DataFrame,
                                   window_size: int = 50,
                                   step_size: int = 25,
                                   lstm_epochs: int = 30) -> pd.DataFrame:
    """
    Compare LSTM vs SVM vs DT vs RF at a given window/step config.

    Parameters
    ----------
    df : pd.DataFrame
    window_size : int
    step_size : int
    lstm_epochs : int

    Returns
    -------
    pd.DataFrame
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Full Model Comparison")
    print(f"  Window={window_size}, Step={step_size}, LSTM Epochs={lstm_epochs}")
    print("=" * 70)

    data = prepare_ml_data(df, window_size=window_size, step_size=step_size,
                           verbose=True)
    class_names = list(data['label_encoder'].classes_)
    num_classes  = len(class_names)

    # sklearn
    sk_res = train_sklearn_models(
        data['X_train'], data['y_train'],
        data['X_test'],  data['y_test'],
        n_jobs=N_JOBS, monitor_cpu=True, verbose=True
    )

    # LSTM
    lstm_res = train_lstm(
        data['X_train_seq'], data['y_train'],
        num_classes=num_classes,
        epochs=lstm_epochs,
        monitor_cpu=True,
        verbose=True,
    )
    lstm_pred = predict_lstm(lstm_res['model'], data['X_test_seq'])

    from sklearn.metrics import f1_score
    lstm_f1 = f1_score(data['y_test'], lstm_pred, average='weighted')

    rows = [
        {'model': 'SVM',  'f1_weighted': sk_res['svm_f1']},
        {'model': 'DT',   'f1_weighted': sk_res['dt_f1']},
        {'model': 'RF',   'f1_weighted': sk_res['rf_f1']},
        {'model': 'LSTM', 'f1_weighted': lstm_f1},
    ]

    print("\n[EXP4] Summary:")
    for r in rows:
        print(f"  {r['model']:6s} F1: {r['f1_weighted']:.4f}")

    return pd.DataFrame(rows)


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _save_sweep_plot(results_df: pd.DataFrame,
                     x_col: str, y_col: str,
                     title: str, xlabel: str,
                     filename: str):
    """Generic line plot for sweep results."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results_df[x_col], results_df[y_col],
            marker='o', linewidth=1.8, color='steelblue')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weighted F1')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    out = IMAGES_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


def _save_bar_plot(results_df: pd.DataFrame, x_col: str, y_col: str,
                   title: str, filename: str):
    """Generic bar plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(results_df[x_col].astype(str), results_df[y_col],
           color=sns.color_palette('tab10', len(results_df)))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted F1')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    out = IMAGES_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


# ============================================================================
# MAIN
# ============================================================================

def run_all_experiments():
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("ASSIGNMENT 1 – EXPERIMENTATION CAMPAIGN")
    print(f"Run ID : {run_id}")
    print("=" * 70)

    get_cpu_info()

    # Load once
    df = load_data(str(CSV_PATH), verbose=True)

    # ----- Experiment 1 -----
    df1 = experiment_1_window_sweep(df)
    if not df1.empty:
        out1 = RESULTS_DIR / f'exp1_window_sweep_{run_id}.csv'
        df1.to_csv(out1, index=False)
        print(f"[SAVE] {out1}")
        _save_sweep_plot(df1, 'window_size', 'rf_f1',
                         'Exp 1 – Window Size Sweep (RF)',
                         'Window Size (samples)',
                         f'exp1_window_sweep_{run_id}.png')

    # ----- Experiment 2 -----
    df2 = experiment_2_step_sweep(df)
    if not df2.empty:
        out2 = RESULTS_DIR / f'exp2_step_sweep_{run_id}.csv'
        df2.to_csv(out2, index=False)
        print(f"[SAVE] {out2}")
        _save_sweep_plot(df2, 'step_size', 'rf_f1',
                         'Exp 2 – Step Size Sweep (RF)',
                         'Step Size (samples)',
                         f'exp2_step_sweep_{run_id}.png')

    # ----- Experiment 3 -----
    df3 = experiment_3_filter_comparison(df)
    if not df3.empty:
        out3 = RESULTS_DIR / f'exp3_filter_{run_id}.csv'
        df3.to_csv(out3, index=False)
        print(f"[SAVE] {out3}")
        _save_bar_plot(df3, 'filter', 'rf_f1',
                       'Exp 3 – Filter Comparison (RF)',
                       f'exp3_filter_{run_id}.png')

    # ----- Experiment 4 -----
    df4 = experiment_4_model_comparison(df, window_size=50, step_size=25,
                                         lstm_epochs=30)
    if not df4.empty:
        out4 = RESULTS_DIR / f'exp4_model_comparison_{run_id}.csv'
        df4.to_csv(out4, index=False)
        print(f"[SAVE] {out4}")
        _save_bar_plot(df4, 'model', 'f1_weighted',
                       'Exp 4 – Model Comparison',
                       f'exp4_model_comparison_{run_id}.png')

    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE  –  run_id: {run_id}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
