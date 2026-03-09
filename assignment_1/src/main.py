"""
Assignment 1 - Cognitive Decline Activity Recognition
Main Pipeline Script

Run from the assignment_1/ directory:
    python src/main.py

The script executes the full end-to-end pipeline:
    1.  Load df_train.csv
    2.  EDA plots  → images/
    3.  Preprocess + feature extraction (sliding window)
    4.  Train LSTM (PyTorch, GPU if available else CPU multi-threaded)
    5.  Train sklearn models (SVM, DT, RF – CPU multi-threaded via BLAS)
    6.  Evaluate all models
    7.  Save confusion matrices + per-class F1 bar chart → images/
    8.  Persist metrics → results/<run_id>/

Hyperparameters are declared at the top of this file for easy tuning.
"""

# ============================================================================
# CRITICAL: Set CPU thread counts BEFORE any numpy/scipy/sklearn imports.
# This must be the first executable code in the entry-point script.
# ============================================================================
import os
import multiprocessing

_num_cores = str(multiprocessing.cpu_count())
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(_v, _num_cores)

print(f"[MAIN] BLAS/OpenMP threads configured → {_num_cores} cores")

# ============================================================================
# Standard imports (after env vars)
# ============================================================================
import sys
import pathlib
import time
from datetime import datetime

import numpy as np

# Make sure src/ is on the path when called from the project root
_SRC_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_SRC_DIR))

from data     import load_data, prepare_ml_data
from model    import (get_device, get_cpu_info,
                      train_lstm, predict_lstm,
                      train_sklearn_models)
from evaluate import evaluate_all, save_metrics
from plot     import (run_eda_plots,
                      plot_confusion_matrix,
                      plot_per_class_f1,
                      plot_training_loss)

import matplotlib.pyplot as plt

# ============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │                      HYPERPARAMETER CONFIGURATION                       │
# │  Edit values here to tune the pipeline.                                  │
# └──────────────────────────────────────────────────────────────────────────┘
# ============================================================================

# --- Data ---
CSV_PATH      = pathlib.Path(__file__).parent.parent / 'df_train.csv'
RESULTS_DIR   = pathlib.Path(__file__).parent.parent / 'results'
IMAGES_DIR    = pathlib.Path(__file__).parent.parent / 'images'
TEST_FRACTION = 0.20          # fraction of windowed samples held out

# --- Sliding window ---
WINDOW_SIZE   = 50            # samples per window  (50 samples @ 10 Hz = 5 s)
STEP_SIZE     = 25            # sliding step  (50 % overlap)

# --- Filter (set to None to disable) ---
FILTER_CONFIG = None          # e.g. {'method': 'lowpass', 'cutoff': 4, 'order': 4}

# --- LSTM ---
LSTM_HIDDEN   = 128
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.3
LSTM_EPOCHS   = 30
LSTM_BATCH    = 256
LSTM_LR       = 1e-3

# --- sklearn ---
N_JOBS        = -1            # -1 = all cores

# --- Misc ---
MONITOR_CPU   = True          # track CPU usage during training
VERBOSE       = True


# ============================================================================
# PIPELINE
# ============================================================================

def run_pipeline():
    run_id     = datetime.now().strftime('%Y%m%d_%H%M%S')
    images_dir = str(IMAGES_DIR)
    total_t0   = time.time()

    print("\n" + "=" * 70)
    print("ASSIGNMENT 1 – COGNITIVE DECLINE ACTIVITY RECOGNITION")
    print(f"Run ID : {run_id}")
    print("=" * 70)

    # ---- System info -------------------------------------------------------
    get_cpu_info()
    get_device()   # logs which device LSTM will use

    # ---- 1. Load data -------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1 / 7 : Load Data")
    print("-" * 70)
    if not CSV_PATH.exists():
        print(f"[ERROR] df_train.csv not found at {CSV_PATH}")
        sys.exit(1)

    df = load_data(str(CSV_PATH), verbose=VERBOSE)

    # ---- 2. EDA Plots -------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2 / 7 : EDA Plots")
    print("-" * 70)
    # Use a subsample for plots to keep it snappy
    df_plot = df.sample(min(100_000, len(df)), random_state=42).reset_index(drop=True)
    run_eda_plots(df_plot, images_dir=images_dir)

    # ---- 3. Preprocessing + Feature Extraction ------------------------------
    print("\n" + "-" * 70)
    print("STEP 3 / 7 : Preprocessing + Feature Extraction")
    print("-" * 70)
    data = prepare_ml_data(
        df,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        test_size=TEST_FRACTION,
        filter_config=FILTER_CONFIG,
        verbose=VERBOSE,
    )

    class_names  = list(data['label_encoder'].classes_)
    num_classes  = len(class_names)

    # ---- 4. Train LSTM ------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 4 / 7 : Train LSTM")
    print("-" * 70)
    lstm_result = train_lstm(
        data['X_train_seq'],
        data['y_train'],
        num_classes  = num_classes,
        hidden_size  = LSTM_HIDDEN,
        num_layers   = LSTM_LAYERS,
        dropout      = LSTM_DROPOUT,
        epochs       = LSTM_EPOCHS,
        batch_size   = LSTM_BATCH,
        lr           = LSTM_LR,
        monitor_cpu  = MONITOR_CPU,
        verbose      = VERBOSE,
    )

    lstm_pred = predict_lstm(lstm_result['model'], data['X_test_seq'],
                             batch_size=LSTM_BATCH * 2)

    # Plot LSTM training loss
    loss_fig = plot_training_loss(lstm_result['train_losses'],
                                  images_dir=images_dir)
    plt.close(loss_fig)

    # ---- 5. Train sklearn models --------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 5 / 7 : Train sklearn models (SVM / DT / RF)")
    print("-" * 70)
    sklearn_result = train_sklearn_models(
        data['X_train'], data['y_train'],
        data['X_test'],  data['y_test'],
        n_jobs      = N_JOBS,
        monitor_cpu = MONITOR_CPU,
        verbose     = VERBOSE,
    )

    # ---- 6. Evaluate --------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 6 / 7 : Evaluate")
    print("-" * 70)
    predictions = {
        'LSTM': lstm_pred,
        'SVM':  sklearn_result['svm_pred'],
        'DT':   sklearn_result['dt_pred'],
        'RF':   sklearn_result['rf_pred'],
    }

    eval_results = evaluate_all(
        data['y_test'], predictions, class_names, verbose=VERBOSE
    )

    # ---- 7. Save outputs ----------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 7 / 7 : Save Outputs")
    print("-" * 70)

    extra_info = {
        'csv_path':     str(CSV_PATH),
        'window_size':  WINDOW_SIZE,
        'step_size':    STEP_SIZE,
        'test_fraction': TEST_FRACTION,
        'filter':       str(FILTER_CONFIG),
        'lstm_hidden':  LSTM_HIDDEN,
        'lstm_layers':  LSTM_LAYERS,
        'lstm_epochs':  LSTM_EPOCHS,
        'lstm_batch':   LSTM_BATCH,
        'lstm_lr':      LSTM_LR,
    }

    save_metrics(eval_results,
                 run_id=run_id,
                 results_dir=str(RESULTS_DIR),
                 extra_info=extra_info)

    # Confusion matrices per model
    for model_name, y_pred in predictions.items():
        fig = plot_confusion_matrix(
            data['y_test'], y_pred, class_names,
            model_name=model_name,
            images_dir=images_dir,
        )
        plt.close(fig)

    # Per-class F1 comparison
    f1_fig = plot_per_class_f1(eval_results, class_names, images_dir=images_dir)
    plt.close(f1_fig)

    # ---- Summary ------------------------------------------------------------
    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Total time : {total_elapsed/60:.1f} min")
    print("\nResults summary:")
    for name, m in eval_results.items():
        print(f"  {name:6s} | F1 (weighted): {m['f1_weighted']:.4f} | "
              f"Accuracy: {m['accuracy']:.4f}")
    print(f"\nOutputs written to:")
    print(f"  Metrics : results/{run_id}/")
    print(f"  Plots   : images/")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_pipeline()
