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
    
    # Run experiments
    # Experiment 1: Window Size Sweep
    # Previous best was 150 (max tested), so we test larger windows now: 150 to 350
    # We fix "overlap" (step size) to 60, as that performed best in Exp 2
    exp1_results = experiment_1_window_sweep(
        df,
        window_sizes=list(range(60, 150, 15)), 
        overlap_fixed=15,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Experiment 2: Overlap Sweep 
    # Testing smaller step sizes (more overlap = more data)
    # We fix window to 150 (a reasonable guess based on upward trend)
    exp2_results = experiment_2_overlap_sweep(
        df,
        overlaps=[5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100],
        window_fixed=150,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Define results directory: results/run_YYYYMMDD_HHMMSS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results', f'run_{timestamp_tag}')
    
    # Save results
    save_experiment_results(exp1_results, exp2_results, output_dir=results_dir, tag=timestamp_tag)
    
    # Print summary
    print_experiment_summary(exp1_results, exp2_results)
    
    # Plot results
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    plot_experiment_results(exp1_results, exp2_results, output_dir=results_dir, tag=timestamp_tag)
    
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Uncomment to run all experiments
    # main()
    
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
    plot_experiment_results(main().exp1_results, main().exp2_results)
