"""
Tutorial 3 - Activity Recognition Experiments
Windowing Parameter Optimization

This module runs experiments to find optimal window size and overlap parameters
for activity recognition using the ADL dataset.

Experiments:
1. Window Size Sweep: Test window sizes from 30 to 150 (step 10) with overlap fixed at 30
2. Overlap Sweep: Test overlap from 15 to 75 (step 15) with window size fixed at 120
"""

import os
import multiprocessing

# ============================================================================
# CRITICAL: Configure CPU threading BEFORE importing numpy/scipy/sklearn
# This must be done in the main script before ANY data libraries are loaded
# ============================================================================
try:
    num_cores = str(multiprocessing.cpu_count())
    os.environ['OMP_NUM_THREADS'] = num_cores
    os.environ['MKL_NUM_THREADS'] = num_cores
    os.environ['OPENBLAS_NUM_THREADS'] = num_cores
    os.environ['BLIS_NUM_THREADS'] = num_cores
    print(f"[INIT] Forced BLAS library to use {num_cores} threads (set env vars)")
except Exception as e:
    print(f"[WARNING] Failed to set threading environment variables: {e}")

# Now import data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Import core functions from tutorial_3_optimized
# (CPU threading is already configured in tutorial_3_optimized module)
from tutorial_3_optimized import (
    configure_cpu_threading,
    read_dataset,
    prepare_windowed_data,
    train_and_classify,
    get_cpu_info
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Detect if running in Google Colab or local environment
if os.path.exists('/content'):  # Google Colab environment
    ZIP_FILE_PATH = '/content/adl_dataset'
    EXTRACTED_FOLDER_PATH = '/content/adl_dataset_extracted'
else:  # Local environment
    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ZIP_FILE_PATH = os.path.join(SCRIPT_DIR, 'adl_dataset')
    EXTRACTED_FOLDER_PATH = os.path.join(SCRIPT_DIR, 'adl_dataset_extracted')

# Sampling frequency from the dataset specifications (32 Hz)
SAMPLING_FREQUENCY = 32


# ============================================================================
# EXPERIMENT 1: WINDOW SIZE SWEEP
# ============================================================================

def experiment_1_window_sweep(df, window_sizes=None, overlap_fixed=30, n_jobs=-1):
    """
    Experiment 1: Sweep window size with fixed overlap
    
    Parameters:
    -----------
    df : pd.DataFrame
        Complete dataset
    window_sizes : list, optional
        List of window sizes to test (default: 30 to 150, step 10)
    overlap_fixed : int
        Fixed overlap value (default: 30)
    n_jobs : int
        Number of CPU cores to use (default: -1 for all cores)
        
    Returns:
    --------
    pd.DataFrame
        Results dataframe with columns: window_size, overlap, overlap_percent, 
        num_samples, svm_f1, dt_f1, rf_f1
    """
    if window_sizes is None:
        window_sizes = list(range(30, 151, 10))
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Window Size Sweep")
    print("=" * 80)
    print(f"Window sizes to test: {window_sizes}")
    print(f"Fixed overlap: {overlap_fixed}")
    print(f"Total configurations: {len(window_sizes)}")
    print("=" * 80)
    
    results = []
    
    # Progress bar for the entire experiment
    for window_size in tqdm(window_sizes, desc="Experiment 1 Progress", position=0):
        # Skip if overlap >= window_size
        if overlap_fixed >= window_size:
            print(f"\n[SKIP] Window={window_size}: overlap >= window_size")
            continue
        
        print(f"\n{'-'*80}")
        print(f"Config {window_sizes.index(window_size)+1}/{len(window_sizes)}: Window={window_size}, Overlap={overlap_fixed}")
        print(f"{'-'*80}")
        
        try:
            # Prepare windowed data
            X, y, groups = prepare_windowed_data(df, window_size, overlap_fixed, verbose=False)
            
            print(f"Generated {X.shape[0]} samples, training models...")
            
            # Train and classify (show progress bar but no verbose reports)
            model_results = train_and_classify(X, y, groups, n_jobs=n_jobs, 
                                              monitor_cpu=False, verbose=False, show_progress=True)
            
            # Store results
            results.append({
                'window_size': window_size,
                'overlap': overlap_fixed,
                'overlap_percent': int((overlap_fixed/window_size)*100),
                'num_samples': X.shape[0],
                'svm_f1': model_results['svm_f1'],
                'dt_f1': model_results['dt_f1'],
                'rf_f1': model_results['rf_f1']
            })
            
            print(f"✓ Complete - SVM F1: {model_results['svm_f1']:.4f}, "
                  f"DT F1: {model_results['dt_f1']:.4f}, "
                  f"RF F1: {model_results['rf_f1']:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Configuration failed: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"Experiment 1 Complete - Tested {len(results)} configurations")
    print("=" * 80)
    
    return pd.DataFrame(results)


# ============================================================================
# EXPERIMENT 2: OVERLAP SWEEP
# ============================================================================

def experiment_2_overlap_sweep(df, overlaps=None, window_fixed=170, n_jobs=-1, filter_config=None):
    """
    Experiment 2: Sweep overlap with fixed window size
    
    Parameters:
    -----------
    df : pd.DataFrame
        Complete dataset
    overlaps : list, optional
        List of overlaps to test (default: 15 to 75, step 15)
    window_fixed : int
        Fixed window size (default: 170)
    n_jobs : int
        Number of CPU cores to use (default: -1 for all cores)
    filter_config : dict, optional
        Filter configuration to apply (default: None)
        
    Returns:
    --------
    pd.DataFrame
        Results dataframe with columns: window_size, overlap, overlap_percent,
        num_samples, svm_f1, dt_f1, rf_f1
    """
    if overlaps is None:
        overlaps = [5, 5, 10, 10, 15, 15, 20, 20, 30, 30]
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Overlap Sweep")
    print("=" * 80)
    print(f"Fixed window size: {window_fixed}")
    print(f"Overlaps to test: {overlaps}")
    print(f"Filter: {filter_config['method'] if filter_config else 'None'}")
    print(f"Total configurations: {len(overlaps)}")
    print("=" * 80)
    
    results = []
    
    # Progress bar for the entire experiment
    for overlap in tqdm(overlaps, desc="Experiment 2 Progress", position=0):
        # Skip if overlap >= window_size
        if overlap >= window_fixed:
            print(f"\n[SKIP] Overlap={overlap}: overlap >= window_size")
            continue
        
        print(f"\n{'-'*80}")
        print(f"Config {overlaps.index(overlap)+1}/{len(overlaps)}: Window={window_fixed}, Overlap={overlap}")
        print(f"{'-'*80}")
        
        try:
            # Prepare windowed data
            X, y, groups = prepare_windowed_data(df, window_fixed, overlap, 
                                                verbose=False, filter_config=filter_config)
            
            print(f"Generated {X.shape[0]} samples, training models...")
            
            # Train and classify (show progress bar but no verbose reports)
            model_results = train_and_classify(X, y, groups, n_jobs=n_jobs,
                                              monitor_cpu=False, verbose=False, show_progress=True)
            
            # Store results
            results.append({
                'window_size': window_fixed,
                'overlap': overlap,
                'overlap_percent': int((overlap/window_fixed)*100),
                'num_samples': X.shape[0],
                'svm_f1': model_results['svm_f1'],
                'dt_f1': model_results['dt_f1'],
                'rf_f1': model_results['rf_f1']
            })
            
            print(f"✓ Complete - SVM F1: {model_results['svm_f1']:.4f}, "
                  f"DT F1: {model_results['dt_f1']:.4f}, "
                  f"RF F1: {model_results['rf_f1']:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Configuration failed: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"Experiment 2 Complete - Tested {len(results)} configurations")
    print("=" * 80)
    
    return pd.DataFrame(results)


# ============================================================================
# EXPERIMENT 4: FILTER COMPARISON
# ============================================================================

def experiment_4_filter_comparison(df, window_size=150, overlap=30, n_jobs=-1):
    """
    Experiment 4: Compare different sensor filtering methods
    Tests: None, Median, Lowpass, Bandpass filters
    
    Parameters:
    -----------
    df : pd.DataFrame
        Complete dataset
    window_size : int
        Fixed window size (default: 150)
    overlap : int
        Fixed overlap (default: 30)
    n_jobs : int
        Number of CPU cores to use
        
    Returns:
    --------
    pd.DataFrame
        Results comparing different filters
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: FILTER COMPARISON")
    print("=" * 80)
    print(f"Fixed Configuration: Window={window_size}, Overlap={overlap}")
    print("Testing filters: None, Median, Lowpass, Bandpass")
    print("=" * 80)
    
    # Define filter configurations
    # FINDING: Combined filters performed WORSE than single filters!
    # Lowpass(5Hz) alone: 0.4750
    # Median+Lowpass: 0.4678 (lower!)
    # 
    # Hypothesis: Over-smoothing removes distinguishing features
    # Testing: Does ORDER matter? Lowpass-first vs Median-first
    filter_configs = [
        {'name': 'No Filter (Baseline)', 'config': None},
        # Best individual filters
        {'name': 'Lowpass (5Hz)', 'config': {'method': 'lowpass', 'cutoff': 5, 'order': 4}},
        {'name': 'Median (k=9)', 'config': {'method': 'median', 'kernel_size': 9}},
        
        # Order Test 1: Median THEN Lowpass (original)
        {'name': 'Median→Lowpass (5Hz)', 'config': {
            'method': 'combined',
            'filters': [
                {'method': 'median', 'kernel_size': 9},
                {'method': 'lowpass', 'cutoff': 5, 'order': 4}
            ]
        }},
        
        # Order Test 2: Lowpass THEN Median (reversed)
        {'name': 'Lowpass→Median (k=9)', 'config': {
            'method': 'combined',
            'filters': [
                {'method': 'lowpass', 'cutoff': 5, 'order': 4},
                {'method': 'median', 'kernel_size': 9}
            ]
        }},
        
        # Lighter combined filters (less aggressive)
        {'name': 'Median(k=5) + Lowpass(5Hz)', 'config': {
            'method': 'combined',
            'filters': [
                {'method': 'median', 'kernel_size': 5},
                {'method': 'lowpass', 'cutoff': 5, 'order': 4}
            ]
        }},
        
        {'name': 'Lowpass(5Hz) + Median(k=5)', 'config': {
            'method': 'combined',
            'filters': [
                {'method': 'lowpass', 'cutoff': 5, 'order': 4},
                {'method': 'median', 'kernel_size': 5}
            ]
        }},
    ]
    
    results = []
    
    for i, filter_def in enumerate(filter_configs):
        filter_name = filter_def['name']
        filter_config = filter_def['config']
        
        print(f"\n{'-'*80}")
        print(f"Config {i+1}/{len(filter_configs)}: {filter_name}")
        print(f"{'-'*80}")
        
        try:
            # Prepare windowed data with filtering
            X, y, groups = prepare_windowed_data(df, window_size, overlap, 
                                                verbose=True, 
                                                filter_config=filter_config)
            
            num_volunteers = len(np.unique(groups))
            print(f"Generated {X.shape[0]} samples from {num_volunteers} volunteers, training models...")
            
            # Train models
            model_results = train_and_classify(X, y, groups, n_jobs=n_jobs,
                                             monitor_cpu=False, verbose=False, show_progress=True)
            
            # Store results
            results.append({
                'filter_name': filter_name,
                'filter_method': filter_config['method'] if filter_config else 'none',
                'num_samples': X.shape[0],
                'num_volunteers': num_volunteers,
                'svm_f1': model_results['svm_f1'],
                'dt_f1': model_results['dt_f1'],
                'rf_f1': model_results['rf_f1']
            })
            
            print(f"✓ Complete - SVM F1: {model_results['svm_f1']:.4f}, "
                  f"DT F1: {model_results['dt_f1']:.4f}, "
                  f"RF F1: {model_results['rf_f1']:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Filter configuration failed: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"Filter Comparison Complete - Tested {len(results)} configurations")
    print("=" * 80)
    
    return pd.DataFrame(results)


def plot_filter_comparison(results, output_dir, tag):
    """
    Plot comparison of different filter methods
    """
    if results.empty:
        print("No results to plot.")
        return
    
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========================================================================
    # Plot 1: Bar Chart Comparison
    # ========================================================================
    ax1 = axes[0]
    
    x_pos = np.arange(len(results))
    width = 0.25
    
    ax1.bar(x_pos - width, results['svm_f1'], width, label='SVM', color='#2E86AB', alpha=0.8)
    ax1.bar(x_pos, results['rf_f1'], width, label='Random Forest', color='#5CB85C', alpha=0.8)
    ax1.bar(x_pos + width, results['dt_f1'], width, label='Decision Tree', color='#A23B72', alpha=0.8)
    
    ax1.set_xlabel('Filter Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Filter Impact on Model Performance', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results['filter_name'], rotation=45, ha='right')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(results[['svm_f1', 'rf_f1', 'dt_f1']].max()) * 1.1])
    
    # ========================================================================
    # Plot 2: Line Chart (SVM Focus)
    # ========================================================================
    ax2 = axes[1]
    
    ax2.plot(results['filter_name'], results['svm_f1'], 
            marker='o', linewidth=3, markersize=10, color='#2E86AB', label='SVM')
    ax2.plot(results['filter_name'], results['rf_f1'], 
            marker='^', linewidth=2.5, markersize=8, color='#5CB85C', label='Random Forest')
    
    # Highlight best SVM score
    best_idx = results['svm_f1'].idxmax()
    best_filter = results.loc[best_idx, 'filter_name']
    best_svm = results.loc[best_idx, 'svm_f1']
    
    ax2.scatter([best_idx], [best_svm], marker='*', s=500, 
               color='gold', edgecolors='black', linewidths=2, zorder=5,
               label=f'Best: {best_filter}')
    
    ax2.set_xlabel('Filter Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('SVM Performance Across Filters', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticklabels(results['filter_name'], rotation=45, ha='right')
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'filter_comparison_{tag}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Filter-specific plot saved: {save_path}")
    
    # Also generate universal comparison plots
    plot_results_comparison(results, output_dir, tag, experiment_name="Filter Comparison")


# ============================================================================
# EXPERIMENT 3: GRID SEARCH (OPTIMIZATION)
# ============================================================================

def experiment_3_grid_search(df, window_sizes, overlaps, n_jobs=-1):
    """
    Experiment 3: Grid Search over Window Size and Step Size
    Finds the global optimum combination.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: GRID SEARCH OPTIMIZATION")
    print("=" * 80)
    
    results = []
    total = len(window_sizes) * len(overlaps)
    
    print(f"Searching {total} combinations...")
    print(f"Windows: {window_sizes}")
    print(f"Overlaps: {overlaps}")
    print("=" * 80)
    
    config_num = 0
    
    for win_size in window_sizes:
        for overlap in overlaps:
            config_num += 1
            
            # Constraint: Window must be larger than step to have any data
            # Typically overlap = win - step. If step >= win, overlap <= 0.
            if overlap >= win_size:
                print(f"\n[SKIP] Config {config_num}/{total}: Window={win_size}, Overlap={overlap} (overlap >= window)")
                continue
                
            print(f"\n{'-'*80}")
            print(f"Config {config_num}/{total}: Window={win_size}, Overlap={overlap}")
            print(f"{'-'*80}")
                
            try:
                # prepare_windowed_data 3rd arg 'overlap' acts as 'step_size' in sliding_window
                X, y, groups = prepare_windowed_data(df, win_size, overlap, verbose=False)
                
                num_volunteers = len(np.unique(groups))
                print(f"Generated {X.shape[0]} samples from {num_volunteers} volunteers, training models...")
                
                # Train with progress bar for CV folds
                res = train_and_classify(X, y, groups, n_jobs=n_jobs, 
                                       verbose=False, show_progress=True)
                
                results.append({
                    'window_size': win_size,
                    'overlap': overlap,
                    'num_samples': X.shape[0],
                    'num_volunteers': num_volunteers,
                    'svm_f1': res['svm_f1'],
                    'rf_f1': res['rf_f1'],
                    'dt_f1': res['dt_f1']
                })
                
                print(f"✓ Complete - SVM F1: {res['svm_f1']:.4f}, "
                      f"DT F1: {res['dt_f1']:.4f}, "
                      f"RF F1: {res['rf_f1']:.4f}")
                
            except Exception as e:
                print(f"[ERROR] Configuration failed: {e}")
                continue
    
    # Sort by SVM score
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='svm_f1', ascending=False)
    
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    if not results_df.empty:
        best = results_df.iloc[0]
        print(f"\n✓ Best Configuration:")
        print(f"  Window: {int(best['window_size'])}")
        print(f"  Overlap:   {int(best['overlap'])}")
        print(f"  SVM F1: {best['svm_f1']:.4f}")
        print(f"  RF F1: {best['rf_f1']:.4f}")
        print(f"  DT F1: {best['dt_f1']:.4f}")
    
    return results_df


def plot_results_comparison(results, output_dir, tag, experiment_name="Experiment"):
    """
    Universal plotting function for any experiment results.
    Creates bar chart and model comparison plots.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Results with columns: svm_f1, rf_f1, dt_f1 (required)
    output_dir : str
        Directory to save plots
    tag : str
        Timestamp/identifier for file naming
    experiment_name : str
        Name of the experiment for titles
    """
    if results.empty:
        print("No results to plot.")
        return
    
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # ========================================================================
    # PLOT 1: Best Performance Bar Chart
    # ========================================================================
    max_svm = results['svm_f1'].max()
    max_dt = results['dt_f1'].max()
    max_rf = results['rf_f1'].max()
    
    models = ['SVM', 'Random Forest', 'Decision Tree']
    scores = [max_svm, max_rf, max_dt]
    colors = ['#2E86AB', '#5CB85C', '#A23B72']
    
    plt.figure(figsize=(10, 7))
    bars = plt.bar(models, scores, color=colors, edgecolor='black', alpha=0.85, width=0.6)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    plt.ylim(0, max(scores) * 1.15)
    plt.ylabel('Best F1 Score', fontsize=13, fontweight='bold')
    plt.title(f'{experiment_name}\nBest Performance by Model', fontsize=15, fontweight='bold', pad=15)
    plt.grid(axis='y', alpha=0.3)
    
    bar_plot_path = os.path.join(output_dir, f'model_comparison_{tag}.png')
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved: {bar_plot_path}")
    
    # ========================================================================
    # PLOT 2: Performance Distribution (if multiple configs)
    # ========================================================================
    if len(results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Box plot of all scores
        ax1 = axes[0]
        data_to_plot = [results['svm_f1'], results['rf_f1'], results['dt_f1']]
        bp = ax1.boxplot(data_to_plot, labels=models, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Distribution', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Right: Line plot of all configs
        ax2 = axes[1]
        
        # Check if we have overlap data for x-axis
        if 'overlap' in results.columns and 'overlap_percent' in results.columns:
            x_pos = results['overlap'].values
            x_labels = [f"{o}\n({p}%)" for o, p in zip(results['overlap'], results['overlap_percent'])]
        else:
            x_pos = range(len(results))
            x_labels = [str(i) for i in x_pos]
        
        ax2.plot(x_pos, results['svm_f1'], marker='o', linewidth=2.5, 
                markersize=8, label='SVM', color='#2E86AB')
        ax2.plot(x_pos, results['rf_f1'], marker='^', linewidth=2.5, 
                markersize=8, label='Random Forest', color='#5CB85C')
        ax2.plot(x_pos, results['dt_f1'], marker='s', linewidth=2.5, 
                markersize=8, label='Decision Tree', color='#A23B72')
        
        # Highlight best SVM
        best_idx = results['svm_f1'].idxmax()
        best_svm = results.loc[best_idx, 'svm_f1']
        if 'overlap' in results.columns:
            best_x = results.loc[best_idx, 'overlap']
        else:
            best_x = best_idx
        ax2.scatter([best_x], [best_svm], marker='*', s=500,
                   color='gold', edgecolors='black', linewidths=2, zorder=5)
        
        # Set x-axis label and ticks
        if 'overlap' in results.columns:
            ax2.set_xlabel('Overlap (samples)\n(Overlap %)', fontsize=12, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(x_labels, fontsize=9)
        else:
            ax2.set_xlabel('Configuration Index', fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax2.set_title('All Configurations Performance', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', frameon=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        dist_plot_path = os.path.join(output_dir, f'performance_distribution_{tag}.png')
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Distribution plot saved: {dist_plot_path}")


def plot_grid_search_results(results, output_dir, tag):
    """
    Plot Heatmap of Grid Search Results
    """
    if results.empty:
        print("No results to plot.")
        return

    # Pivot for Heatmap
    try:
        pivot_svm = results.pivot(index='overlap', columns='window_size', values='svm_f1')
        
        plt.figure(figsize=(12, 10))
        sns.set_context("talk")
        
        # Heatmap
        sns.heatmap(pivot_svm, annot=True, fmt=".3f", cmap="viridis", 
                   cbar_kws={'label': 'SVM F1 Score'})
        
        plt.title(f"SVM Performance Landscape\n(Window Size vs Overlap)", pad=20)
        plt.xlabel("Window Size", fontweight='bold')
        plt.ylabel("Overlap (Lower = More Overlap)", fontweight='bold')
        
        # Highlight max
        # Find coordinates of max
        # (This is handled visually by the heatmap, but we can add complex annotation if needed)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'grid_search_heatmap_{tag}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Heatmap saved to: {save_path}")
        
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Also generate universal comparison plots
    plot_results_comparison(results, output_dir, tag, experiment_name="Grid Search")


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def plot_experiment_results(exp1_results, exp2_results, output_dir=None, tag=None):
    """
    Visualize experiment results with professional plots
    
    Parameters:
    -----------
    exp1_results : pd.DataFrame
        Results from Experiment 1 (window size sweep)
    exp2_results : pd.DataFrame
        Results from Experiment 2 (overlap sweep)
    output_dir : str, optional
        Directory to save the figures
    tag : str, optional
        Small tag to differentiate files (e.g., timestamp)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = f"_{tag}" if tag else ""
    
    # Set aesthetic style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # ========================================================================
    # FIGURE 1: Parameter Sweeps (Window Sizes & Overlap)
    # ========================================================================
    fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Subplot 1: Experiment 1 (Window Size Sweep) ---
    ax1 = axes[0]
    
    if not exp1_results.empty:
        # Plot lines for each model
        ax1.plot(exp1_results['window_size'], exp1_results['svm_f1'],
                marker='o', linewidth=2.5, markersize=8, label='SVM', color='#2E86AB')
        ax1.plot(exp1_results['window_size'], exp1_results['dt_f1'],
                marker='s', linewidth=2.5, markersize=8, label='Decision Tree', color='#A23B72')
        ax1.plot(exp1_results['window_size'], exp1_results['rf_f1'],
                marker='^', linewidth=2.5, markersize=7, label='Random Forest',
                color='green', linestyle='-', alpha=1.0)
        
        # Find and mark best SVM score
        best_svm_idx = exp1_results['svm_f1'].idxmax()
        best_svm_window = exp1_results.loc[best_svm_idx, 'window_size']
        best_svm_f1 = exp1_results.loc[best_svm_idx, 'svm_f1']
        
        ax1.scatter([best_svm_window], [best_svm_f1], marker='*', s=500,
                   color='gold', edgecolors='black', linewidths=2, zorder=5,
                   label=f'Best SVM (F1={best_svm_f1:.4f})')
        
        ax1.set_xlabel('Window Size', fontsize=13, fontweight='bold')
        ax1.set_ylabel('F1 Score (Weighted)', fontsize=13, fontweight='bold')
        ax1.set_title(f'Exp 1: Window Size Impact\n(Fixed overlap)',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', frameon=True, shadow=True, fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
    
    # --- Subplot 2: Experiment 2 (Overlap/Step Sweep) ---
    ax2 = axes[1]
    
    if not exp2_results.empty:
        # Plot lines for each model
        ax2.plot(exp2_results['overlap'], exp2_results['svm_f1'],
                marker='o', linewidth=2.5, markersize=8, label='SVM', color='#2E86AB')
        ax2.plot(exp2_results['overlap'], exp2_results['dt_f1'],
                marker='s', linewidth=2.5, markersize=8, label='Decision Tree', color='#A23B72')
        ax2.plot(exp2_results['overlap'], exp2_results['rf_f1'],
                marker='^', linewidth=2.5, markersize=7, label='Random Forest',
                color='green', linestyle='-', alpha=1.0)
        
        # Find and mark best SVM score
        best_svm_idx2 = exp2_results['svm_f1'].idxmax()
        best_svm_step = exp2_results.loc[best_svm_idx2, 'overlap']
        best_svm_f1_2 = exp2_results.loc[best_svm_idx2, 'svm_f1']
        
        ax2.scatter([best_svm_step], [best_svm_f1_2], marker='*', s=500,
                   color='gold', edgecolors='black', linewidths=2, zorder=5,
                   label=f'Best SVM (F1={best_svm_f1_2:.4f})')
        
        ax2.set_xlabel('Step Size (Lower = More Overlap)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('F1 Score (Weighted)', fontsize=13, fontweight='bold')
        ax2.set_title(f'Exp 2: Step Size Impact\n(Fixed Window)',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', frameon=True, shadow=True, fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    sweeps_path = os.path.join(output_dir, f'plot_sweeps{suffix}.png')
    plt.savefig(sweeps_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {sweeps_path}")

    # ========================================================================
    # FIGURE 2: Data Volume Impact (Number of Samples vs F1)
    # ========================================================================
    # We combine data from both experiments to see if "More Data" always equals "Better Score"
    # regardless of whether it came from Window Size or Overlap changes.
    
    merged_results = pd.concat([exp1_results, exp2_results], ignore_index=True)
    
    if not merged_results.empty:
        plt.figure(figsize=(10, 6))
        
        plt.scatter(merged_results['num_samples'], merged_results['svm_f1'], 
                   c='#2E86AB', s=100, alpha=0.7, label='SVM Configs', edgecolors='w')
        plt.scatter(merged_results['num_samples'], merged_results['rf_f1'], 
                   c='#C9C9C9', s=80, alpha=0.5, label='RF Configs', marker='^', edgecolors='k')
        
        # Trend line for SVM
        z = np.polyfit(merged_results['num_samples'], merged_results['svm_f1'], 1)
        p = np.poly1d(z)
        plt.plot(merged_results['num_samples'], p(merged_results['num_samples']), 
                "r--", alpha=0.4, label='Trend (SVM)')

        plt.xlabel('Number of Training Samples', fontsize=12, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
        plt.title('Impact of Training Set Size on Performance', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        data_plot_path = os.path.join(output_dir, f'plot_data_volume{suffix}.png')
        plt.savefig(data_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {data_plot_path}")

    # ========================================================================
    # FIGURE 3: Best Models Comparison Bar Chart
    # ========================================================================
    if not merged_results.empty:
        # Get global maximums for each model type
        max_svm = merged_results['svm_f1'].max()
        max_dt = merged_results['dt_f1'].max()
        max_rf = merged_results['rf_f1'].max()
        
        models = ['SVM', 'Random Forest', 'Decision Tree']
        scores = [max_svm, max_rf, max_dt]
        colors = ['#2E86AB', '#C9C9C9', '#A23B72']
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(models, scores, color=colors, edgecolor='black', alpha=0.8, width=0.6)
        
        # Add value labels on top
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        plt.ylim(0, 1.05)
        plt.ylabel('Best F1 Score Achieved', fontsize=12, fontweight='bold')
        plt.title('Best Performance by Model Architecture', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        bar_plot_path = os.path.join(output_dir, f'plot_model_comparison{suffix}.png')
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {bar_plot_path}")


def print_experiment_summary(exp1_results, exp2_results):
    """
    Print a comprehensive summary of experiment results
    
    Parameters:
    -----------
    exp1_results : pd.DataFrame
        Results from Experiment 1
    exp2_results : pd.DataFrame
        Results from Experiment 2
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    if not exp1_results.empty:
        print("\nExperiment 1 Results:")
        print("-" * 80)
        print(exp1_results.to_string(index=False))
        
        best_svm_idx = exp1_results['svm_f1'].idxmax()
        best_exp1 = exp1_results.loc[best_svm_idx]
        
        print("\nBest Configuration (by SVM F1):")
        print(f"  Window Size: {best_exp1['window_size']}")
        print(f"  Overlap: {best_exp1['overlap']} ({best_exp1['overlap_percent']}%)")
        print(f"  SVM F1: {best_exp1['svm_f1']:.4f}")
        print(f"  Decision Tree F1: {best_exp1['dt_f1']:.4f}")
        print(f"  Random Forest F1: {best_exp1['rf_f1']:.4f}")
    
    if not exp2_results.empty:
        print("\n" + "=" * 80)
        print("\nExperiment 2 Results:")
        print("-" * 80)
        print(exp2_results.to_string(index=False))
        
        best_svm_idx2 = exp2_results['svm_f1'].idxmax()
        best_exp2 = exp2_results.loc[best_svm_idx2]
        
        print("\nBest Configuration (by SVM F1):")
        print(f"  Window Size: {best_exp2['window_size']}")
        print(f"  Overlap: {best_exp2['overlap']} ({best_exp2['overlap_percent']}%)")
        print(f"  SVM F1: {best_exp2['svm_f1']:.4f}")
        print(f"  Decision Tree F1: {best_exp2['dt_f1']:.4f}")
        print(f"  Random Forest F1: {best_exp2['rf_f1']:.4f}")
    
    print("\n" + "=" * 80)


def save_experiment_results(exp1_results, exp2_results, output_dir=None, tag=None):
    """
    Save experiment results to CSV files
    
    Parameters:
    -----------
    exp1_results : pd.DataFrame
        Results from Experiment 1
    exp2_results : pd.DataFrame
        Results from Experiment 2
    output_dir : str, optional
        Directory to save results (default: same as script directory)
    tag : str, optional
        Small tag to differentiate files (e.g., timestamp)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = f"_{tag}" if tag else ""
    
    exp1_path = os.path.join(output_dir, f'experiment1_window_sweep{suffix}.csv')
    exp2_path = os.path.join(output_dir, f'experiment2_overlap_sweep{suffix}.csv')
    
    exp1_results.to_csv(exp1_path, index=False)
    exp2_results.to_csv(exp2_path, index=False)
    
    print("\n" + "=" * 80)
    print("Results saved to:")
    print(f"  - {exp1_path}")
    print(f"  - {exp2_path}")
    print("=" * 80)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run all experiments
    
    You can comment/uncomment sections to run specific experiments
    """
    start_time = datetime.now()
    timestamp_tag = start_time.strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "=" * 80)
    print("ACTIVITY RECOGNITION WINDOWING PARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run ID: {timestamp_tag}")
    print("=" * 80)
    
    # Configure CPU threading
    print("\nConfiguring CPU threading...")
    num_cores = configure_cpu_threading()
    
    # Get CPU info
    get_cpu_info()
    
    # Load dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    print(f"Zip file: {ZIP_FILE_PATH}")
    print(f"Extract to: {EXTRACTED_FOLDER_PATH}")
    print(f"Sampling frequency: {SAMPLING_FREQUENCY} Hz")
    
    # Check if dataset already loaded
    if os.path.exists(EXTRACTED_FOLDER_PATH):
        print("\n[INFO] Dataset already extracted, skipping extraction...")
    
    df = read_dataset(ZIP_FILE_PATH, EXTRACTED_FOLDER_PATH, SAMPLING_FREQUENCY)
    
    print(f"\nDataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Activities: {df['Label'].unique()}")
    print(f"Volunteers: {df['Volunteer'].nunique()}")
    
    # ========================================================================
    # CHOOSE WHICH EXPERIMENT TO RUN (uncomment one)
    # ========================================================================
    
    # --- EXPERIMENT 1: Window Size Sweep ---
    # exp1_results = experiment_1_window_sweep(
    #     df,
    #     window_sizes=list(range(60, 151, 10)),  # 60 to 150, step 10
    #     overlap_fixed=30,
    #     n_jobs=-1
    # )
    
    # --- EXPERIMENT 2: Overlap Sweep with Lowpass 5Hz Filter (ACTIVE) ---
    exp_results = experiment_2_overlap_sweep(
        df,
        overlaps=[15, 16, 17, 18, 19, 20, 21, 22, 25, 30, 35, 40, 50, 70],
        window_fixed=170,
        n_jobs=-1,
        filter_config={'method': 'lowpass', 'cutoff': 5, 'order': 4}
    )
    
    # Define results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results', f'run_{timestamp_tag}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save Results
    csv_path = os.path.join(results_dir, f'experiment2_overlap_sweep_{timestamp_tag}.csv')
    exp_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_results_comparison(exp_results, results_dir, timestamp_tag, experiment_name="Overlap Sweep (Lowpass 5Hz)")
    
    # --- EXPERIMENT 3: Grid Search ---
    # windows = [170]
    # overlaps = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

    # exp_results = experiment_3_grid_search(df, windows, overlaps, n_jobs=-1)
    # # Define results directory
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # results_dir = os.path.join(script_dir, 'results', f'run_{timestamp_tag}')
    # os.makedirs(results_dir, exist_ok=True)

    # # Save Results
    # csv_path = os.path.join(results_dir, f'grid_search_results_{timestamp_tag}.csv')
    # exp_results.to_csv(csv_path, index=False)

    # plot_grid_search_results(exp_results, results_dir, timestamp_tag)
    
    # --- EXPERIMENT 4: Filter Comparison ---
    # exp_results = experiment_4_filter_comparison(df, window_size=150, overlap=30, n_jobs=-1)
    
    # # Define results directory
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # results_dir = os.path.join(script_dir, 'results', f'run_{timestamp_tag}')
    # os.makedirs(results_dir, exist_ok=True)
    
    # # Save Results
    # csv_path = os.path.join(results_dir, f'filter_comparison_{timestamp_tag}.csv')
    # exp_results.to_csv(csv_path, index=False)
    # print(f"\nResults saved to: {csv_path}")

    # plot_filter_comparison(exp_results, results_dir, timestamp_tag)
    
    # Print Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(exp_results.to_string(index=False))
    
    if not exp_results.empty:
        best_idx = exp_results['svm_f1'].idxmax()
        best = exp_results.loc[best_idx]
        print(f"\n✓ Best Configuration:")
        if 'window_size' in best:
            print(f"  Window: {int(best['window_size'])}")
        if 'overlap' in best:
            print(f"  Overlap: {int(best['overlap'])}")
        if 'filter_name' in best:
            print(f"  Filter: {best['filter_name']}")
        print(f"  SVM F1: {best['svm_f1']:.4f}")
        print(f"  RF F1: {best['rf_f1']:.4f}")
        print(f"  DT F1: {best['dt_f1']:.4f}")
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)
    
   
# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the main experiment
    main()
    
    # Or run individual experiments:
    # 
    # # Configure CPU and load data
    # configure_cpu_threading()
    # df = read_dataset(ZIP_FILE_PATH, EXTRACTED_FOLDER_PATH, SAMPLING_FREQUENCY)
    # 
    # # Run only Experiment 1
    # exp1_results = experiment_1_window_sweep(df, overlap_fixed=30, n_jobs=-1)
    # print(exp1_results)
    # 
    # # Run only Experiment 2
    # exp2_results = experiment_2_overlap_sweep(df, window_fixed=120, n_jobs=-1)
    # print(exp2_results)
    # 
    # # Plot results
    # plot_experiment_results(main().exp1_results, main().exp2_results)
