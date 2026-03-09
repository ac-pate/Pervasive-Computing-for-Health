"""
Assignment 1 - Cognitive Decline Activity Recognition
Visualisation: EDA Plots and Evaluation Charts

All functions return the figure so callers can either show() or close() them.
Figures are also saved to images/ by default.

EDA plots:
  1. Activity class distribution (bar chart)
  2. Acceleration time-series sample per activity
  3. Frequency content (FFT magnitude spectrum per activity)
  4. Temporal patterns (hourly / rolling activity density)
  5. Correlation heat-map (X, Y, Z, Magnitude per class)

Evaluation plots:
  6. Confusion matrix
  7. Per-class F1 bar chart
  8. LSTM training loss curve
"""

# ============================================================================
# Imports
# ============================================================================
import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Local imports
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import COL_X, COL_Y, COL_Z, COL_LABEL, COL_TIME, SAMPLING_FREQ

# ============================================================================
# HELPERS
# ============================================================================

_PALETTE = "tab10"


def _save(fig: plt.Figure, images_dir: str, filename: str) -> pathlib.Path:
    """Save figure and return path."""
    out_dir = pathlib.Path(images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Saved: {out_path}")
    return out_path


def _default_images_dir() -> str:
    return str(pathlib.Path(__file__).parent.parent / 'images')


# ============================================================================
# 1. ACTIVITY CLASS DISTRIBUTION
# ============================================================================

def plot_activity_distribution(df: pd.DataFrame,
                               images_dir: str = None,
                               filename: str = 'activity_distribution.png') -> plt.Figure:
    """
    Bar chart of sample counts per activity class.

    Parameters
    ----------
    df : pd.DataFrame  (raw, needs COL_LABEL)
    images_dir : str
    filename : str

    Returns
    -------
    plt.Figure
    """
    if images_dir is None:
        images_dir = _default_images_dir()

    counts = df[COL_LABEL].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette(_PALETTE, len(counts)))
    ax.set_title('Activity Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Activity')
    ax.set_ylabel('Sample Count')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    _save(fig, images_dir, filename)
    return fig


# ============================================================================
# 2. ACCELERATION TIME-SERIES SAMPLE
# ============================================================================

def plot_acceleration_samples(df: pd.DataFrame,
                               n_seconds: float = 5.0,
                               images_dir: str = None,
                               filename: str = 'acceleration_samples.png') -> plt.Figure:
    """
    Plot a short acceleration time-series segment for each unique activity.

    Parameters
    ----------
    df : pd.DataFrame
    n_seconds : float
        Duration (seconds) to display per activity.
    images_dir : str
    filename : str

    Returns
    -------
    plt.Figure
    """
    if images_dir is None:
        images_dir = _default_images_dir()

    labels = df[COL_LABEL].unique()
    n_samples = int(n_seconds * SAMPLING_FREQ)
    n_rows = len(labels)

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    palette = sns.color_palette(_PALETTE, n_rows)

    for ax, label, color in zip(axes, labels, palette):
        subset = df[df[COL_LABEL] == label][[COL_X, COL_Y, COL_Z]].head(n_samples)
        t = np.arange(len(subset)) / SAMPLING_FREQ
        ax.plot(t, subset[COL_X].values, label='X', linewidth=0.8, alpha=0.9)
        ax.plot(t, subset[COL_Y].values, label='Y', linewidth=0.8, alpha=0.9)
        ax.plot(t, subset[COL_Z].values, label='Z', linewidth=0.8, alpha=0.9)
        ax.set_title(label, fontsize=10, color=color, fontweight='bold')
        ax.set_ylabel('Accel (g)')
        ax.legend(loc='upper right', fontsize=7, ncol=3)
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Acceleration Time-Series Samples per Activity',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    _save(fig, images_dir, filename)
    return fig


# ============================================================================
# 3. FREQUENCY CONTENT (FFT)
# ============================================================================

def plot_frequency_content(df: pd.DataFrame,
                            channel: str = None,
                            images_dir: str = None,
                            filename: str = 'frequency_content.png') -> plt.Figure:
    """
    Plot average FFT magnitude spectrum for each activity class.

    Parameters
    ----------
    df : pd.DataFrame
    channel : str
        One of COL_X / COL_Y / COL_Z. Defaults to COL_Z.
    images_dir : str
    filename : str

    Returns
    -------
    plt.Figure
    """
    if images_dir is None:
        images_dir = _default_images_dir()
    if channel is None:
        channel = COL_Z

    labels  = df[COL_LABEL].unique()
    segment = int(SAMPLING_FREQ * 5)   # 5-second segments for FFT
    freqs   = np.fft.rfftfreq(segment, d=1.0 / SAMPLING_FREQ)

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette(_PALETTE, len(labels))

    for label, color in zip(labels, palette):
        vals = df[df[COL_LABEL] == label][channel].values
        # Tile into segments and average the FFT
        n_seg  = len(vals) // segment
        if n_seg == 0:
            continue
        mat    = vals[:n_seg * segment].reshape(n_seg, segment)
        mag    = np.abs(np.fft.rfft(mat, axis=1)).mean(axis=0)
        ax.plot(freqs, mag, label=label, color=color, linewidth=1.4)

    ax.set_title(f'Average FFT Magnitude Spectrum per Activity  (channel: {channel})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim([0, SAMPLING_FREQ / 2])
    plt.tight_layout()

    _save(fig, images_dir, filename)
    return fig


# ============================================================================
# 4. TEMPORAL PATTERNS (hourly activity density)
# ============================================================================

def plot_temporal_patterns(df: pd.DataFrame,
                            images_dir: str = None,
                            filename: str = 'temporal_patterns.png') -> plt.Figure:
    """
    Stacked-area chart of activity proportions across the 24-hour clock.

    Parameters
    ----------
    df : pd.DataFrame  (must have COL_TIME parsed as datetime)
    images_dir : str
    filename : str

    Returns
    -------
    plt.Figure
    """
    if images_dir is None:
        images_dir = _default_images_dir()

    df_t = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_t[COL_TIME]):
        df_t[COL_TIME] = pd.to_datetime(df_t[COL_TIME])

    df_t['hour'] = df_t[COL_TIME].dt.hour
    pivot = (df_t.groupby(['hour', COL_LABEL])
               .size()
               .unstack(fill_value=0))
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot_pct.plot.area(ax=ax, colormap=_PALETTE, alpha=0.85)
    ax.set_title('Activity Proportion by Hour of Day',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Proportion')
    ax.set_xticks(range(0, 24))
    ax.legend(title='Activity', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    _save(fig, images_dir, filename)
    return fig


# ============================================================================
# 5. CORRELATION HEAT-MAP
# ============================================================================

def plot_correlation_heatmap(df: pd.DataFrame,
                              images_dir: str = None,
                              filename: str = 'correlation_heatmap.png') -> plt.Figure:
    """
    Pearson correlation heat-map of X, Y, Z channels.

    Parameters
    ----------
    df : pd.DataFrame
    images_dir : str
    filename : str

    Returns
    -------
    plt.Figure
    """
    if images_dir is None:
        images_dir = _default_images_dir()

    corr = df[[COL_X, COL_Y, COL_Z]].corr()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title('Accelerometer Channel Correlation', fontsize=12, fontweight='bold')
    plt.tight_layout()

    _save(fig, images_dir, filename)
    return fig


# ============================================================================
# 6. CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list,
                           model_name: str = 'Model',
                           images_dir: str = None,
                           filename: str = None,
                           normalise: bool = True) -> plt.Figure:
    """
    Annotated heat-map confusion matrix.

    Parameters
    ----------
    y_true, y_pred : np.ndarray  integer labels
    class_names : list[str]
    model_name : str
    images_dir : str
    filename : str  (auto-generated from model_name if None)
    normalise : bool  (divide each row by its sum)

    Returns
    -------
    plt.Figure
    """
    if images_dir is None:
        images_dir = _default_images_dir()
    if filename is None:
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'

    cm = confusion_matrix(y_true, y_pred)
    if normalise:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt     = '.2f'
        vmax    = 1.0
    else:
        cm_plot = cm
        fmt     = 'd'
        vmax    = cm.max()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=vmax, ax=ax, linewidths=0.4)
    ax.set_title(f'Confusion Matrix – {model_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    _save(fig, images_dir, filename)
    return fig


# ============================================================================
# 7. PER-CLASS F1 BAR CHART
# ============================================================================

def plot_per_class_f1(eval_results: dict,
                      class_names: list,
                      images_dir: str = None,
                      filename: str = 'per_class_f1.png') -> plt.Figure:
    """
    Grouped bar chart of per-class F1 across all models.

    Parameters
    ----------
    eval_results : dict[str, dict]   output of evaluate.evaluate_all()
    class_names : list[str]
    images_dir : str
    filename : str

    Returns
    -------
    plt.Figure
    """
    from sklearn.metrics import f1_score

    if images_dir is None:
        images_dir = _default_images_dir()

    model_names = list(eval_results.keys())
    n_classes   = len(class_names)
    x           = np.arange(n_classes)
    width       = 0.8 / len(model_names)
    palette     = sns.color_palette(_PALETTE, len(model_names))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, m) in enumerate(eval_results.items()):
        # Re-derive per-class F1 from the confusion matrix
        cm   = m['confusion']
        tp   = np.diag(cm)
        fp   = cm.sum(axis=0) - tp
        fn   = cm.sum(axis=1) - tp
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0)
        rec  = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1   = np.where(prec + rec > 0,
                        2 * prec * rec / (prec + rec), 0)

        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, f1, width, label=name, color=palette[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score by Model', fontsize=13, fontweight='bold')
    ax.legend(title='Model', fontsize=9)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.tight_layout()

    _save(fig, images_dir, filename)
    return fig


# ============================================================================
# 8. LSTM TRAINING LOSS CURVE
# ============================================================================

def plot_training_loss(train_losses: list,
                       images_dir: str = None,
                       filename: str = 'lstm_training_loss.png') -> plt.Figure:
    """
    Line plot of LSTM cross-entropy loss over epochs.

    Parameters
    ----------
    train_losses : list[float]
    images_dir : str
    filename : str

    Returns
    -------
    plt.Figure
    """
    if images_dir is None:
        images_dir = _default_images_dir()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses,
            marker='o', markersize=3, linewidth=1.5, color='steelblue')
    ax.set_title('LSTM Training Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    _save(fig, images_dir, filename)
    return fig


# ============================================================================
# CONVENIENCE: Run all EDA plots at once
# ============================================================================

def run_eda_plots(df: pd.DataFrame, images_dir: str = None):
    """
    Generate and save all EDA plots.

    Parameters
    ----------
    df : pd.DataFrame  raw data with COL_TIME parsed.
    images_dir : str
    """
    if images_dir is None:
        images_dir = _default_images_dir()

    print("[PLOT] Generating EDA plots …")
    figs = [
        plot_activity_distribution(df, images_dir),
        plot_acceleration_samples(df, images_dir=images_dir),
        plot_frequency_content(df, images_dir=images_dir),
        plot_temporal_patterns(df, images_dir=images_dir),
        plot_correlation_heatmap(df, images_dir=images_dir),
    ]
    for fig in figs:
        plt.close(fig)
    print("[PLOT] All EDA plots saved.")


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from data import load_data

    csv_path = pathlib.Path(__file__).parent.parent / 'df_train.csv'
    images_dir = str(pathlib.Path(__file__).parent.parent / 'images')

    print("=" * 70)
    print("PLOT MODULE - Standalone Test")
    print("=" * 70)

    # Use a small subset for speed
    df = load_data(str(csv_path), verbose=True)
    df_small = df.sample(50_000, random_state=42).reset_index(drop=True)

    run_eda_plots(df_small, images_dir=images_dir)

    # Fake predictions for confusion matrix test
    import numpy as np
    n = 100
    nc = 7
    y_true = np.random.randint(0, nc, n)
    y_pred = np.random.randint(0, nc, n)
    class_names = ['Work', 'Other', 'Eat', 'Travel', 'Hygiene', 'Cook', 'Exercise']

    fig = plot_confusion_matrix(y_true, y_pred, class_names,
                                model_name='Test', images_dir=images_dir)
    plt.close(fig)
    print("\n[TEST] plot.py - ALL CHECKS PASSED")
