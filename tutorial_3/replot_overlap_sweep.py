#!/usr/bin/env python3
"""
Regenerate plots for overlap sweep experiment from CSV data
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_overlap_sweep_results(results, output_dir, tag):
    """
    Plot results with overlap values on x-axis
    """
    if results.empty:
        print("No results to plot.")
        return

    models = ['SVM', 'Random Forest', 'Decision Tree']
    
    # ========================================================================
    # PLOT 1: Bar Chart Comparison (Best Configuration)
    # ========================================================================
    best_idx = results['svm_f1'].idxmax()
    best_config = results.loc[best_idx]
    
    best_scores = {
        'SVM': best_config['svm_f1'],
        'Random Forest': best_config['rf_f1'],
        'Decision Tree': best_config['dt_f1']
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(best_scores.keys(), best_scores.values(), 
                   color=['#2E86AB', '#5CB85C', '#A23B72'], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Best Configuration: Overlap={int(best_config["overlap"])} ({int(best_config["overlap_percent"])}%)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(best_scores.values()) * 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    bar_plot_path = os.path.join(output_dir, f'best_config_comparison_{tag}.png')
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar plot saved: {bar_plot_path}")
    
    # ========================================================================
    # PLOT 2: All Configurations Performance (Single Plot)
    # ========================================================================
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x_pos = results['overlap'].values
        x_labels = [f"{int(o)}\n({int(p)}%)" for o, p in zip(results['overlap'], results['overlap_percent'])]
        
        ax.plot(x_pos, results['svm_f1'], marker='o', linewidth=2.5, 
                markersize=8, label='SVM', color='#2E86AB')
        ax.plot(x_pos, results['rf_f1'], marker='^', linewidth=2.5, 
                markersize=8, label='Random Forest', color='#5CB85C')
        ax.plot(x_pos, results['dt_f1'], marker='s', linewidth=2.5, 
                markersize=8, label='Decision Tree', color='#A23B72')
        
        # Highlight best SVM
        best_idx = results['svm_f1'].idxmax()
        best_svm = results.loc[best_idx, 'svm_f1']
        best_x = results.loc[best_idx, 'overlap']
        best_overlap_pct = int(results.loc[best_idx, 'overlap_percent'])
        ax.scatter([best_x], [best_svm], marker='*', s=500,
                   color='gold', edgecolors='black', linewidths=2, zorder=5,
                   label=f'Best: Overlap={int(best_x)} ({best_overlap_pct}%), F1={best_svm:.4f}')
        
        # Get window size from first row
        window_size = int(results['window_size'].iloc[0])
        
        ax.set_xlabel('Overlap (samples)\n(Overlap %)', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Overlap Sweep Performance with Window={window_size}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=10, rotation=0)
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        overlap_plot_path = os.path.join(output_dir, f'overlap_sweep_performance_{tag}.png')
        plt.savefig(overlap_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Overlap sweep plot saved: {overlap_plot_path}")


if __name__ == "__main__":
    # Load the CSV file
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/run_20260205_173221/experiment2_overlap_sweep_20260205_173221.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print(f"Loading results from: {csv_path}")
    results = pd.read_csv(csv_path)
    
    # Get output directory (same as CSV location)
    output_dir = os.path.dirname(csv_path)
    
    # Extract timestamp from filename
    filename = os.path.basename(csv_path)
    tag = filename.replace('experiment2_overlap_sweep_', '').replace('.csv', '')
    
    print(f"\nGenerating plots in: {output_dir}")
    print(f"Tag: {tag}\n")
    
    # Generate plots
    plot_overlap_sweep_results(results, output_dir, tag)
    
    print("\nDone! Plots regenerated successfully.")
