
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tutorial_3_optimized import (
    read_dataset,
    configure_cpu_threading,
    prepare_windowed_data,
    train_and_classify, 
    get_cpu_info
)

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'adl_dataset')
EXTRACT_DIR = os.path.join(BASE_DIR, 'adl_dataset_extracted')

def plot_confusion_matrices(y_true, y_pred, model_name, labels, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run Single Activity Recognition Experiment')
    parser.add_argument('--window', '-w', type=int, required=True, help='Window size (e.g., 250)')
    parser.add_argument('--overlap', '-o', type=int, default=None, help='Overlap size (e.g., 50)')
    parser.add_argument('--step', '-s', type=int, default=None, help='Step size (Alternative to overlap)')
    parser.add_argument('--filter', '-f', type=str, choices=['none', 'median', 'lowpass', 'bandpass'], 
                        default='none', help='Sensor filtering method')
    parser.add_argument('--filter-param', type=float, default=None, 
                        help='Filter parameter (kernel size for median, cutoff for lowpass)')
    
    args = parser.parse_args()
    
    # Configure Threading
    configure_cpu_threading()
    
    # Calculate overlap/step
    if args.step and args.overlap:
        print("Error: Specify only --step or --overlap, not both.")
        return
    
    window_size = args.window
    if args.step:
        overlap = window_size - args.step
    elif args.overlap:
        overlap = args.overlap
    else:
        # Default overlap is window/2 or fixed?
        overlap = 15  # Default from experiment script
        
    print(f"\n=== Configuration ===")
    print(f"Window Size: {window_size}")
    print(f"Overlap: {overlap}")
    print(f"Step Size: {window_size - overlap}")
    
    # Filter Config
    filter_config = None
    if args.filter != 'none':
        filter_config = {'method': args.filter}
        if args.filter == 'median':
            k = int(args.filter_param) if args.filter_param else 3
            filter_config['kernel_size'] = k
            print(f"Filter: Median (Kernel={k})")
        elif args.filter == 'lowpass':
            c = args.filter_param if args.filter_param else 12
            filter_config['cutoff'] = c
            print(f"Filter: Lowpass (Cutoff={c}Hz)")
        elif args.filter == 'bandpass':
            filter_config['low'] = 1
            filter_config['high'] = 5
            print(f"Filter: Bandpass (1-5Hz)")
    else:
        print("Filter: None")

    # Load Data
    print("\nLoading dataset...")
    if not os.path.exists(EXTRACT_DIR):
        print(f"Extracting {DATA_DIR}...")
    
    df = read_dataset(DATA_DIR, EXTRACT_DIR, 32)
    
    # Prepare Windows
    print("\nPreparing windows...")
    X, y, groups = prepare_windowed_data(df, window_size, overlap, 
                                       verbose=True, 
                                       filter_config=filter_config)
    
    print(f"Generated {len(X)} samples.")
    print(f"Features shape: {X.shape}")
    
    # Train
    print("\nTraining models (Leave-One-Group-Out)...")
    results = train_and_classify(X, y, groups, n_jobs=-1, verbose=True)
    
    # Print Summary
    print("\n=== Final Results ===")
    print(f"SVM F1: {results['svm_f1']:.4f}")
    print(f"RF  F1: {results['rf_f1']:.4f}")
    print(f"DT  F1: {results['dt_f1']:.4f}")

    # Plot Confusion Matrices
    print("\nGenerating confusion matrices...")
    
    # Get class names (LabelEncoder sorts alphabetically)
    class_names = sorted(df['Label'].unique())
    output_dir = os.path.join(BASE_DIR, 'results', 'single_run')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_confusion_matrices(results['y_true'], results['y_pred_svm'], 'SVM', class_names, output_dir)
    plot_confusion_matrices(results['y_true'], results['y_pred_rf'], 'RandomForest', class_names, output_dir)
    
    print(f"Plots saved to {output_dir}")
    
if __name__ == "__main__":
    main()
