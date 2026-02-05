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

def experiment_2_overlap_sweep(df, overlaps=None, window_fixed=120, n_jobs=-1):
    """
    Experiment 2: Sweep overlap with fixed window size
    
    Parameters:
    -----------
    df : pd.DataFrame
        Complete dataset
    overlaps : list, optional
        List of overlaps to test (default: 15 to 75, step 15)
    window_fixed : int
        Fixed window size (default: 120)
    n_jobs : int
        Number of CPU cores to use (default: -1 for all cores)
        
    Returns:
    --------
    pd.DataFrame
        Results dataframe with columns: window_size, overlap, overlap_percent,
        num_samples, svm_f1, dt_f1, rf_f1
    """
    if overlaps is None:
        overlaps = list(range(15, 76, 15))
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Overlap Sweep")
    print("=" * 80)
    print(f"Fixed window size: {window_fixed}")
    print(f"Overlaps to test: {overlaps}")
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
            X, y, groups = prepare_windowed_data(df, window_fixed, overlap, verbose=False)
            
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
# PLOTTING UTILITIES
# ============================================================================

def plot_experiment_results(exp1_results, exp2_results, save_path=None):
    """
    Visualize experiment results with professional plots
    
    Parameters:
    -----------
    exp1_results : pd.DataFrame
        Results from Experiment 1 (window size sweep)
    exp2_results : pd.DataFrame
        Results from Experiment 2 (overlap sweep)
    save_path : str, optional
        Path to save the figure (if None, figure is displayed)
    """
    # Set aesthetic style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== EXPERIMENT 1: Window Size Sweep ==========
    ax1 = axes[0]
    
    if not exp1_results.empty:
        # Plot lines for each model
        ax1.plot(exp1_results['window_size'], exp1_results['svm_f1'],
                marker='o', linewidth=2.5, markersize=8, label='SVM', color='#2E86AB')
        ax1.plot(exp1_results['window_size'], exp1_results['dt_f1'],
                marker='s', linewidth=2.5, markersize=8, label='Decision Tree', color='#A23B72')
        ax1.plot(exp1_results['window_size'], exp1_results['rf_f1'],
                marker='^', linewidth=2, markersize=7, label='Random Forest',
                color='#C9C9C9', linestyle='--', alpha=0.6)
        
        # Find and mark best SVM score
        best_svm_idx = exp1_results['svm_f1'].idxmax()
        best_svm_window = exp1_results.loc[best_svm_idx, 'window_size']
        best_svm_f1 = exp1_results.loc[best_svm_idx, 'svm_f1']
        
        ax1.scatter([best_svm_window], [best_svm_f1], marker='*', s=500,
                   color='gold', edgecolors='black', linewidths=2, zorder=5,
                   label=f'Best SVM (F1={best_svm_f1:.4f})')
        
        # Add annotation
        ax1.annotate(f'Best: {best_svm_window}',
                    xy=(best_svm_window, best_svm_f1),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'))
    
    # Styling
    ax1.set_xlabel('Window Size', fontsize=13, fontweight='bold')
    ax1.set_ylabel('F1 Score (Weighted)', fontsize=13, fontweight='bold')
    ax1.set_title('Experiment 1: Window Size Impact on Model Performance\n(Overlap fixed at 30)',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])
    
    # ========== EXPERIMENT 2: Overlap Sweep ==========
    ax2 = axes[1]
    
    if not exp2_results.empty:
        # Plot lines for each model
        ax2.plot(exp2_results['overlap_percent'], exp2_results['svm_f1'],
                marker='o', linewidth=2.5, markersize=8, label='SVM', color='#2E86AB')
        ax2.plot(exp2_results['overlap_percent'], exp2_results['dt_f1'],
                marker='s', linewidth=2.5, markersize=8, label='Decision Tree', color='#A23B72')
        ax2.plot(exp2_results['overlap_percent'], exp2_results['rf_f1'],
                marker='^', linewidth=2, markersize=7, label='Random Forest',
                color='#C9C9C9', linestyle='--', alpha=0.6)
        
        # Find and mark best SVM score
        best_svm_idx2 = exp2_results['svm_f1'].idxmax()
        best_svm_overlap_pct = exp2_results.loc[best_svm_idx2, 'overlap_percent']
        best_svm_f1_2 = exp2_results.loc[best_svm_idx2, 'svm_f1']
        
        ax2.scatter([best_svm_overlap_pct], [best_svm_f1_2], marker='*', s=500,
                   color='gold', edgecolors='black', linewidths=2, zorder=5,
                   label=f'Best SVM (F1={best_svm_f1_2:.4f})')
        
        # Add annotation
        ax2.annotate(f'Best: {best_svm_overlap_pct}%',
                    xy=(best_svm_overlap_pct, best_svm_f1_2),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'))
    
    # Styling
    ax2.set_xlabel('Overlap (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('F1 Score (Weighted)', fontsize=13, fontweight='bold')
    ax2.set_title('Experiment 2: Overlap Impact on Model Performance\n(Window size fixed at 120)',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


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


def save_experiment_results(exp1_results, exp2_results, output_dir=None):
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
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    exp1_path = os.path.join(output_dir, 'experiment1_window_sweep.csv')
    exp2_path = os.path.join(output_dir, 'experiment2_overlap_sweep.csv')
    
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
    print("\n" + "=" * 80)
    print("ACTIVITY RECOGNITION WINDOWING PARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    # Run experiments
    # Experiment 1: Window Size Sweep (overlap fixed at 30)
    exp1_results = experiment_1_window_sweep(
        df,
        window_sizes=list(range(30, 151, 10)),  # 30 to 150, step 10
        overlap_fixed=30,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Experiment 2: Overlap Sweep (window size fixed at 120)
    exp2_results = experiment_2_overlap_sweep(
        df,
        overlaps=list(range(15, 76, 15)),  # 15 to 75, step 15
        window_fixed=120,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Save results
    save_experiment_results(exp1_results, exp2_results)
    
    # Print summary
    print_experiment_summary(exp1_results, exp2_results)
    
    # Plot results
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            'experiment_results.png')
    plot_experiment_results(exp1_results, exp2_results, save_path=plot_path)
    
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Uncomment to run all experiments
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
    # plot_experiment_results(exp1_results, exp2_results)
