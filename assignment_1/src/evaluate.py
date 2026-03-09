"""
Assignment 1 - Cognitive Decline Activity Recognition
Evaluation: Metrics, Classification Reports, Results Persistence

Produces:
  - Per-model weighted F1 scores
  - Full classification reports
  - Persisted results to results/<run_id>/metrics.txt
"""

# ============================================================================
# Imports
# ============================================================================
import os
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    accuracy_score,
)


# ============================================================================
# CORE METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list, model_name: str = "Model",
                    verbose: bool = True) -> dict:
    """
    Compute classification metrics for a single model's predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    class_names : list[str]
        Human-readable class names (ordered by integer encoding).
    model_name : str
        Label used in print output.
    verbose : bool

    Returns
    -------
    dict with keys:
        f1_weighted, f1_macro, accuracy : float
        report      : str   full sklearn classification_report
        confusion   : np.ndarray  raw confusion matrix
        model_name  : str
    """
    f1_w   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m   = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f"\n[EVAL] === {model_name} ===")
        print(f"[EVAL] Weighted F1 : {f1_w:.4f}")
        print(f"[EVAL] Macro F1    : {f1_m:.4f}")
        print(f"[EVAL] Accuracy    : {acc:.4f}")
        print(f"\n{report}")

    return {
        'model_name':   model_name,
        'f1_weighted':  f1_w,
        'f1_macro':     f1_m,
        'accuracy':     acc,
        'report':       report,
        'confusion':    cm,
    }


def evaluate_all(y_true: np.ndarray,
                 predictions: dict,
                 class_names: list,
                 verbose: bool = True) -> dict:
    """
    Evaluate every model in the *predictions* dict.

    Parameters
    ----------
    y_true : np.ndarray
    predictions : dict[str, np.ndarray]
        E.g. {'SVM': svm_pred, 'RF': rf_pred, 'LSTM': lstm_pred}.
    class_names : list[str]
    verbose : bool

    Returns
    -------
    dict[str, dict]
        model_name → metrics dict (from compute_metrics).
    """
    results = {}
    for name, y_pred in predictions.items():
        results[name] = compute_metrics(y_true, y_pred, class_names,
                                        model_name=name, verbose=verbose)
    return results


# ============================================================================
# RESULTS PERSISTENCE
# ============================================================================

def save_metrics(eval_results: dict,
                 run_id: str = None,
                 results_dir: str = None,
                 extra_info: dict = None) -> pathlib.Path:
    """
    Write a human-readable metrics.txt to results/<run_id>/.

    Parameters
    ----------
    eval_results : dict[str, dict]
        Output of evaluate_all().
    run_id : str or None
        Identifier string appended to the folder name.
        Auto-generated from timestamp if None.
    results_dir : str or None
        Root results directory. Defaults to ../results relative to this file.
    extra_info : dict or None
        Additional key-value pairs printed at the top of the file.

    Returns
    -------
    pathlib.Path
        Path to the written metrics.txt file.
    """
    if results_dir is None:
        results_dir = pathlib.Path(__file__).parent.parent / 'results'

    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    out_dir = pathlib.Path(results_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'metrics.txt'

    sep = "=" * 70
    lines = [
        sep,
        "ASSIGNMENT 1 - COGNITIVE DECLINE ACTIVITY RECOGNITION",
        f"Run ID  : {run_id}",
        f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
    ]

    if extra_info:
        lines.append("\n--- Configuration ---")
        for k, v in extra_info.items():
            lines.append(f"  {k:20s}: {v}")

    lines += ["", "--- Summary ---"]
    summary_rows = []
    for name, m in eval_results.items():
        lines.append(
            f"  {name:20s} | Weighted F1: {m['f1_weighted']:.4f} | "
            f"Macro F1: {m['f1_macro']:.4f} | Accuracy: {m['accuracy']:.4f}"
        )
        summary_rows.append({
            'model':        name,
            'f1_weighted':  m['f1_weighted'],
            'f1_macro':     m['f1_macro'],
            'accuracy':     m['accuracy'],
        })

    for name, m in eval_results.items():
        lines += [
            "",
            sep,
            f"Model: {name}",
            sep,
            m['report'],
        ]

    text = "\n".join(lines)
    out_file.write_text(text, encoding='utf-8')
    print(f"[EVAL] Metrics saved to: {out_file}")

    # Also write a compact CSV for easy comparison
    csv_file = out_dir / 'summary.csv'
    pd.DataFrame(summary_rows).to_csv(csv_file, index=False)
    print(f"[EVAL] Summary CSV  : {csv_file}")

    return out_file


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from data import load_data, prepare_ml_data
    from model import train_sklearn_models

    csv_path = pathlib.Path(__file__).parent.parent / 'df_train.csv'

    print("=" * 70)
    print("EVALUATE MODULE - Standalone Test")
    print("=" * 70)

    df   = load_data(str(csv_path))
    data = prepare_ml_data(df, window_size=50, step_size=25, verbose=False)

    res = train_sklearn_models(
        data['X_train'], data['y_train'],
        data['X_test'],  data['y_test'],
        verbose=False,
    )

    preds = {
        'SVM': res['svm_pred'],
        'DT':  res['dt_pred'],
        'RF':  res['rf_pred'],
    }

    class_names = list(data['label_encoder'].classes_)
    eval_results = evaluate_all(data['y_test'], preds, class_names, verbose=True)

    out = save_metrics(
        eval_results,
        run_id='test_run',
        results_dir=str(pathlib.Path(__file__).parent.parent / 'results'),
        extra_info={'window_size': 50, 'step_size': 25},
    )
    print(f"\n[TEST] evaluate.py - ALL CHECKS PASSED  →  {out}")
