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
                      plot_combined_confusion_matrices,
                      plot_per_class_f1,
                      plot_training_loss)
from wandb_config import init_wandb, finish_wandb, log_metrics

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
# NOTE: images/ folder is for manual sorting only - scripts save to results/<run_id>/
TEST_FRACTION = 0.20          # fraction of windowed samples held out

# --- Sliding window ---
# OPTIMIZED via Phase 1: Window=100 (10s @ 10Hz), Step=25 (75% overlap) → RF F1=0.8001
WINDOW_SIZE   = 100           # samples per window  (100 samples @ 10 Hz = 10 s)
STEP_SIZE     = 25            # sliding step  (75 % overlap)

# --- Filter (set to None to disable) ---
# OPTIMIZED via Phase 2: lowpass_3hz → RF F1=0.8021 (+0.2% improvement)
FILTER_CONFIG = {'method': 'lowpass', 'cutoff': 3.0, 'order': 4}

# --- LSTM ---
# OPTIMIZED via Phase 3 (weighted): Training with 100 epochs for proper convergence
LSTM_HIDDEN   = 256           # Phase 3 best: 256 (testing up to 512)
LSTM_LAYERS   = 2             # Phase 3 best: 2 (testing up to 4)
LSTM_DROPOUT  = 0.3           # Phase 3 best: 0.3
LSTM_EPOCHS   = 100           # Increased to 100 for proper training (30 was too few)
LSTM_BATCH    = 256           # Reduced from 512 for better gradient updates
LSTM_LR       = 1e-3
LSTM_CLASS_WEIGHT = True      # Enable class weighting for imbalanced data (24:1 ratio)

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
    run_dir    = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir = str(run_dir)  # Save all images to results/<run_id>/
    total_t0   = time.time()

    print("\n" + "=" * 70)
    print("ASSIGNMENT 1 – COGNITIVE DECLINE ACTIVITY RECOGNITION")
    print(f"Run ID : {run_id}")
    print("=" * 70)

    # Initialize WandB with group to organize multiple runs
    wandb_run = init_wandb(
        name=f"main_{run_id}",
        group="manual_training",  # Groups all manual runs together
        job_type='full_pipeline',
        config={
            'window_size': WINDOW_SIZE,
            'step_size': STEP_SIZE,
            'test_fraction': TEST_FRACTION,
            'filter': str(FILTER_CONFIG),
            'lstm_hidden': LSTM_HIDDEN,
            'lstm_layers': LSTM_LAYERS,
            'lstm_epochs': LSTM_EPOCHS,
            'lstm_batch': LSTM_BATCH,
            'lstm_lr': LSTM_LR,
        },
        tags=['main', 'full_training']
    )

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
        class_weight = LSTM_CLASS_WEIGHT,  # Use class weights for imbalanced data
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

    # Combined confusion matrix (2x2 grid)
    combined_fig = plot_combined_confusion_matrices(
        predictions, data['y_test'], class_names,
        images_dir=images_dir
    )
    plt.close(combined_fig)

    # Individual confusion matrices (optional, if you want them too)
    # Uncomment if desired:
    # for model_name, y_pred in predictions.items():
    #     fig = plot_confusion_matrix(
    #         data['y_test'], y_pred, class_names,
    #         model_name=model_name,
    #         images_dir=images_dir,
    #     )
    #     plt.close(fig)

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
    print(f"  Results & Plots: results/{run_id}/")
    print("=" * 70)

    # Log summary to WandB
    if wandb_run is not None:
        import wandb
        if wandb.run is not None:
            # Log final F1 scores as summary (not time-series)
            wandb.run.summary.update({
                'lstm_f1': eval_results['LSTM']['f1_weighted'],
                'rf_f1': eval_results['RF']['f1_weighted'],
                'svm_f1': eval_results['SVM']['f1_weighted'],
                'dt_f1': eval_results['DT']['f1_weighted'],
                'total_time_min': total_elapsed / 60,
            })
            # Upload confusion matrix
            wandb.log({"confusion_matrix": wandb.Image(str(run_dir / 'confusion_matrices_combined.png'))})
        
        finish_wandb()
        print("[WANDB] Run finished and logged")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_pipeline()
