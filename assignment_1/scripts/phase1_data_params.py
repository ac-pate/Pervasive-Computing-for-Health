"""
Phase 1: Data Parameter Optimization (Window + Overlap)
Use Random Forest (fast) to find optimal window size and overlap

This tests combinations efficiently:
- Window sizes: 30, 50, 70, 100 samples
- For each window, test 3 overlaps: low (10-20%), medium (40-60%), high (75%)

Run from assignment_1/:
    python3 scripts/phase1_data_params.py

Output: results/phase1_data_params_<timestamp>/
"""

import sys
import pathlib
import time
from datetime import datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_data, prepare_ml_data
from model import train_sklearn_models
from wandb_config import init_wandb, finish_wandb

# ============================================================================
# Configuration
# ============================================================================

WINDOW_SIZES = [30, 50, 70, 100]  # samples @ 10 Hz = 3s, 5s, 7s, 10s

# For each window, test 3 overlap percentages
def get_overlaps_for_window(window_size):
    """Return low, medium, high overlap for given window."""
    return [
        int(window_size * 0.15),  # ~15% overlap (low)
        int(window_size * 0.50),  # ~50% overlap (medium)
        int(window_size * 0.75),  # ~75% overlap (high)
    ]

# ============================================================================

def test_config(df, window_size, step_size, verbose=False):
    """Test one window+overlap configuration with RF."""
    try:
        data = prepare_ml_data(df, window_size=window_size, 
                               step_size=step_size, verbose=verbose)
        
        result = train_sklearn_models(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test'],
            n_jobs=-1, verbose=False
        )
        
        overlap = window_size - step_size
        overlap_pct = (overlap / window_size) * 100
        
        return {
            'window_size': window_size,
            'step_size': step_size,
            'overlap': overlap,
            'overlap_pct': overlap_pct,
            'num_windows': data['X_train'].shape[0] + data['X_test'].shape[0],
            'rf_f1': result['rf_f1'],
            'svm_f1': result['svm_f1'],
            'dt_f1': result['dt_f1'],
        }
    except Exception as e:
        print(f"[ERROR] window={window_size}, step={step_size}: {e}")
        return None


def plot_results(results_df, output_dir):
    """Create visualization of results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Heatmap of window vs overlap
    pivot = results_df.pivot(index='overlap_pct', columns='window_size', values='rf_f1')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.6, vmax=0.8, ax=axes[0])
    axes[0].set_title('RF F1 Score: Window vs Overlap', fontweight='bold')
    axes[0].set_xlabel('Window Size (samples)')
    axes[0].set_ylabel('Overlap %')
    
    # Plot 2: Line plot per window size
    for ws in sorted(results_df['window_size'].unique()):
        subset = results_df[results_df['window_size'] == ws].sort_values('overlap_pct')
        axes[1].plot(subset['overlap_pct'], subset['rf_f1'], 
                    marker='o', label=f'Window={ws}')
    
    axes[1].set_title('F1 Score vs Overlap %', fontweight='bold')
    axes[1].set_xlabel('Overlap %')
    axes[1].set_ylabel('RF F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_params_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_dir / 'data_params_heatmap.png'}")


def main():
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = pathlib.Path(__file__).parent.parent / 'results' / f'phase1_data_params_{run_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PHASE 1: DATA PARAMETER OPTIMIZATION")
    print(f"Testing {len(WINDOW_SIZES)} windows × 3 overlaps = {len(WINDOW_SIZES)*3} configs")
    print("=" * 70)
    
    # Initialize WandB for campaign
    campaign_group = f"optimization_{run_id}"
    wandb_run = init_wandb(
        name=f"phase1_data_params_{run_id}",
        group=campaign_group,
        job_type="data_param_sweep",
        config={
            'phase': 1,
            'window_sizes': WINDOW_SIZES,
            'overlap_types': ['low', 'medium', 'high'],
        },
        tags=['phase1', 'window_sweep', 'overlap_sweep']
    )
    
    # Load data once
    csv_path = pathlib.Path(__file__).parent.parent / 'df_train.csv'
    df = load_data(str(csv_path), verbose=True)
    
    # Test all combinations
    results = []
    total = len(WINDOW_SIZES) * 3
    
    with tqdm(total=total, desc="Testing configs") as pbar:
        for window_size in WINDOW_SIZES:
            overlaps = get_overlaps_for_window(window_size)
            
            for overlap in overlaps:
                step_size = window_size - overlap
                
                print(f"\n[TEST] Window={window_size}, Step={step_size}, Overlap={overlap} ({overlap/window_size*100:.0f}%)")
                
                result = test_config(df, window_size, step_size)
                if result:
                    results.append(result)
                    print(f"  RF F1: {result['rf_f1']:.4f} | Windows: {result['num_windows']:,}")
                    
                    # Log to WandB
                    if wandb_run is not None:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                'window_size': result['window_size'],
                                'step_size': result['step_size'],
                                'overlap_pct': result['overlap_pct'],
                                'rf_f1': result['rf_f1'],
                                'num_windows': result['num_windows'],
                            })
                
                pbar.update(1)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rf_f1', ascending=False)
    
    csv_path = output_dir / 'results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] {csv_path}")
    
    # Plot
    plot_results(results_df, output_dir)
    
    # Print top 5
    print("\n" + "=" * 70)
    print("TOP 5 CONFIGURATIONS:")
    print("=" * 70)
    for i, row in results_df.head(5).iterrows():
        print(f"{i+1}. Window={row['window_size']:3.0f}, Step={row['step_size']:3.0f} "
              f"({row['overlap_pct']:4.1f}% overlap) → F1={row['rf_f1']:.4f}")
    
    # Best config
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("🏆 BEST CONFIGURATION:")
    print("=" * 70)
    print(f"Window Size : {best['window_size']:.0f} samples ({best['window_size']/10:.1f}s)")
    print(f"Step Size   : {best['step_size']:.0f} samples")
    print(f"Overlap     : {best['overlap']:.0f} samples ({best['overlap_pct']:.1f}%)")
    print(f"RF F1 Score : {best['rf_f1']:.4f}")
    print(f"Num Windows : {best['num_windows']:,}")
    print("=" * 70)
    
    print(f"\n✅ Phase 1 complete! Results saved to: {output_dir}")
    print(f"\n📋 Next step: Run Phase 2 with filter comparison using:")
    print(f"   Window={best['window_size']:.0f}, Step={best['step_size']:.0f}")
    
    # Log summary to WandB
    if wandb_run is not None:
        import wandb
        if wandb.run is not None:
            wandb.run.summary.update({
                'best_window_size': int(best['window_size']),
                'best_step_size': int(best['step_size']),
                'best_overlap_pct': float(best['overlap_pct']),
                'best_rf_f1': float(best['rf_f1']),
                'campaign_group': campaign_group,
            })
            wandb.log({"phase1_heatmap": wandb.Image(str(output_dir / 'data_params_heatmap.png'))})
        finish_wandb()


if __name__ == "__main__":
    main()
