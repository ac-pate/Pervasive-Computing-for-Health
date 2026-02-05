
#!/usr/bin/env python3
"""
Single Experiment Runner - Compare Filtered vs Unfiltered

Runs two specific experiments:
1. Window=170, Overlap=5, No Filter
2. Window=170, Overlap=5, Lowpass 5Hz Filter

Generates comparison plots similar to model_comparison plots.

Usage:
  python single_experiment.py                          # Run experiments
  python single_experiment.py --plot-only <csv_path>   # Just generate plots from existing CSV
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
from tutorial_3_optimized import (
    read_dataset,
    configure_cpu_threading,
    prepare_windowed_data,
    train_and_classify
)

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'adl_dataset')
EXTRACT_DIR = os.path.join(BASE_DIR, 'adl_dataset_extracted')

def plot_comparison(results_df, output_dir, tag):
    """
    Create comparison bar plot similar to model_comparison plot
    Shows SVM, RF, DT for both filtered and unfiltered experiments
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========================================================================
    # PLOT 1: Grouped Bar Chart - All Models, Both Experiments
    # ========================================================================
    ax1 = axes[0]
    
    x_labels = results_df['experiment_name'].tolist()
    x_pos = np.arange(len(x_labels))
    width = 0.25
    
    bars1 = ax1.bar(x_pos - width, results_df['svm_f1'], width, 
                   label='SVM', color='#2E86AB', alpha=0.85, edgecolor='black')
    bars2 = ax1.bar(x_pos, results_df['rf_f1'], width, 
                   label='Random Forest', color='#5CB85C', alpha=0.85, edgecolor='black')
    bars3 = ax1.bar(x_pos + width, results_df['dt_f1'], width, 
                   label='Decision Tree', color='#A23B72', alpha=0.85, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Experiment Configuration', fontsize=13, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax1.set_title('Filter Impact on All Models\n(Window=170, Overlap=5)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=15, ha='right')
    ax1.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    max_f1 = results_df[['svm_f1', 'rf_f1', 'dt_f1']].values.max()
    ax1.set_ylim(0, max_f1 * 1.15)
    
    # ========================================================================
    # PLOT 2: Performance Improvement Chart
    # ========================================================================
    ax2 = axes[1]
    
    # Calculate improvement (Filtered - Unfiltered)
    baseline = results_df.iloc[0]  # No Filter
    filtered = results_df.iloc[1]  # Lowpass 5Hz
    
    improvements = {
        'SVM': (filtered['svm_f1'] - baseline['svm_f1']) * 100,
        'Random Forest': (filtered['rf_f1'] - baseline['rf_f1']) * 100,
        'Decision Tree': (filtered['dt_f1'] - baseline['dt_f1']) * 100
    }
    
    models = list(improvements.keys())
    imp_values = list(improvements.values())
    colors_imp = ['#2E86AB', '#5CB85C', '#A23B72']
    
    bars = ax2.barh(models, imp_values, color=colors_imp, alpha=0.85, edgecolor='black')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, imp_values)):
        ax2.text(val + 0.1 if val > 0 else val - 0.1, i, 
                f'{val:+.2f}%',
                ha='left' if val > 0 else 'right', 
                va='center', fontsize=12, fontweight='bold')
    
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xlabel('F1 Score Change (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Lowpass 5Hz Filter Impact\n(% Change from Baseline)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'filter_vs_nofilter_comparison_{tag}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison plot saved: {save_path}")


def plot_confusion_matrices(y_true, y_pred, model_name, labels, output_dir, tag):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'confusion_matrix_{model_name}_{tag}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")


def main():
    """
    Run two experiments and compare results:
    1. No Filter (Baseline)
    2. Lowpass 5Hz Filter
    """
    # Check for plot-only mode
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        if len(sys.argv) < 3:
            print("Usage: python single_experiment.py --plot-only <csv_path>")
            print("\nAvailable results:")
            results_base = os.path.join(BASE_DIR, 'results')
            if os.path.exists(results_base):
                for folder in sorted(os.listdir(results_base), reverse=True)[:5]:
                    if folder.startswith('comparison_'):
                        csv_file = os.path.join(results_base, folder, f'{folder.replace("comparison", "comparison_results")}.csv')
                        if os.path.exists(csv_file):
                            print(f"  {csv_file}")
            return
        
        csv_path = sys.argv[2]
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            return
        
        print("\n" + "=" * 80)
        print("PLOT-ONLY MODE: Loading results from CSV")
        print("=" * 80)
        print(f"CSV: {csv_path}")
        
        # Load results
        results_df = pd.read_csv(csv_path)
        output_dir = os.path.dirname(csv_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Print summary
        print("\n" + "=" * 80)
        print("LOADED RESULTS")
        print("=" * 80)
        print(results_df.to_string(index=False))
        
        # Generate plots
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)
        
        plot_comparison(results_df, output_dir, timestamp)
        
        print("\n" + "=" * 80)
        print("PLOTTING COMPLETE")
        print("=" * 80)
        print(f"Plots saved to: {output_dir}")
        print("=" * 80)
        return
    
    # Normal experiment mode
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "=" * 80)
    print("FILTER COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Configuration: Window=170, Overlap=10")
    print(f"Experiments: (1) No Filter, (2) Lowpass 5Hz")
    print(f"Run ID: {timestamp}")
    print("=" * 80)
    
    # Configure Threading
    configure_cpu_threading()
    
    # Load Data
    print("\nLoading dataset...")
    df = read_dataset(DATA_DIR, EXTRACT_DIR, 32)
    print(f"Dataset loaded: {df.shape[0]} samples, {df['Volunteer'].nunique()} volunteers")
    
    # Experiment Configuration
    window_size = 170
    overlap = 10
    experiments = [
        {'name': 'No Filter', 'filter_config': None},
        {'name': 'Lowpass 5Hz', 'filter_config': {'method': 'lowpass', 'cutoff': 5, 'order': 4}}
    ]
    
    results = []
    all_predictions = {}  # Store for confusion matrices
    
    # Run both experiments
    for i, exp in enumerate(experiments):
        print("\n" + "=" * 80)
        print(f"EXPERIMENT {i+1}/2: {exp['name']}")
        print("=" * 80)
        print(f"Window={window_size}, Overlap={overlap}, Step={window_size-overlap}")
        
        # Prepare windowed data
        print("\nPreparing windowed data...")
        X, y, groups = prepare_windowed_data(
            df, window_size, overlap, 
            verbose=True, 
            filter_config=exp['filter_config']
        )
        
        print(f"Generated {X.shape[0]} samples from {len(np.unique(groups))} volunteers")
        print(f"Feature shape: {X.shape}")
        
        # Train models
        print("\nTraining models (Leave-One-Group-Out CV)...")
        model_results = train_and_classify(
            X, y, groups, 
            n_jobs=-1, 
            verbose=False, 
            show_progress=True
        )
        
        # Store results
        results.append({
            'experiment_name': exp['name'],
            'window_size': window_size,
            'overlap': overlap,
            'step_size': window_size - overlap,
            'num_samples': X.shape[0],
            'svm_f1': model_results['svm_f1'],
            'rf_f1': model_results['rf_f1'],
            'dt_f1': model_results['dt_f1']
        })
        
        # Store predictions for confusion matrices
        all_predictions[exp['name']] = {
            'y_true': model_results['y_true'],
            'y_pred_svm': model_results['y_pred_svm'],
            'y_pred_rf': model_results['y_pred_rf'],
            'y_pred_dt': model_results['y_pred_dt']
        }
        
        print(f"\n✓ Experiment {i+1} Complete:")
        print(f"  SVM F1: {model_results['svm_f1']:.4f}")
        print(f"  RF  F1: {model_results['rf_f1']:.4f}")
        print(f"  DT  F1: {model_results['dt_f1']:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Calculate improvements
    print("\n" + "=" * 80)
    print("FILTER IMPACT ANALYSIS")
    print("=" * 80)
    baseline = results_df.iloc[0]
    filtered = results_df.iloc[1]
    
    print(f"\nBaseline (No Filter):")
    print(f"  SVM: {baseline['svm_f1']:.4f}")
    print(f"  RF:  {baseline['rf_f1']:.4f}")
    print(f"  DT:  {baseline['dt_f1']:.4f}")
    
    print(f"\nFiltered (Lowpass 5Hz):")
    print(f"  SVM: {filtered['svm_f1']:.4f} ({(filtered['svm_f1']-baseline['svm_f1'])*100:+.2f}%)")
    print(f"  RF:  {filtered['rf_f1']:.4f} ({(filtered['rf_f1']-baseline['rf_f1'])*100:+.2f}%)")
    print(f"  DT:  {filtered['dt_f1']:.4f} ({(filtered['dt_f1']-baseline['dt_f1'])*100:+.2f}%)")
    
    # Save results
    output_dir = os.path.join(BASE_DIR, 'results', f'comparison_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f'comparison_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved: {csv_path}")
    
    # Generate comparison plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    plot_comparison(results_df, output_dir, timestamp)
    
    # Generate confusion matrices for filtered experiment (best performer)
    print("\nGenerating confusion matrices for Lowpass 5Hz...")
    class_names = sorted(df['Label'].unique())
    
    preds = all_predictions['Lowpass 5Hz']
    plot_confusion_matrices(preds['y_true'], preds['y_pred_svm'], 
                          'SVM_Lowpass5Hz', class_names, output_dir, timestamp)
    plot_confusion_matrices(preds['y_true'], preds['y_pred_rf'], 
                          'RF_Lowpass5Hz', class_names, output_dir, timestamp)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
