"""
Phase 2: Filter Optimization
Test filter types and parameters using optimal window/overlap from Phase 1

Run from assignment_1/:
    python3 scripts/phase2_filters.py --window 50 --step 25

Output: results/phase2_filters_<timestamp>/
"""

import sys
import pathlib
import argparse
from datetime import datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import load_data, prepare_ml_data
from model import train_sklearn_models
from wandb_config import init_wandb, finish_wandb

# ============================================================================
# Filter configurations to test
# ============================================================================

FILTER_CONFIGS = [
    {'name': 'none', 'config': None},
    {'name': 'lowpass_2hz', 'config': {'method': 'lowpass', 'cutoff': 2.0, 'order': 4}},
    {'name': 'lowpass_3hz', 'config': {'method': 'lowpass', 'cutoff': 3.0, 'order': 4}},
    {'name': 'lowpass_4hz', 'config': {'method': 'lowpass', 'cutoff': 4.0, 'order': 4}},
    {'name': 'bandpass_1-4hz', 'config': {'method': 'bandpass', 'low': 1.0, 'high': 4.0, 'order': 4}},
    {'name': 'median_k3', 'config': {'method': 'median', 'kernel_size': 3}},
]

# ============================================================================

def test_filter(df, window_size, step_size, filter_config, verbose=False):
    """Test one filter configuration."""
    try:
        data = prepare_ml_data(df, window_size=window_size, 
                               step_size=step_size,
                               filter_config=filter_config,
                               verbose=verbose)
        
        result = train_sklearn_models(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test'],
            n_jobs=-1, verbose=False
        )
        
        return {
            'rf_f1': result['rf_f1'],
            'svm_f1': result['svm_f1'],
            'dt_f1': result['dt_f1'],
        }
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=50, help='Window size from Phase 1')
    parser.add_argument('--step', type=int, default=25, help='Step size from Phase 1')
    args = parser.parse_args()
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = pathlib.Path(__file__).parent.parent / 'results' / f'phase2_filters_{run_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PHASE 2: FILTER OPTIMIZATION")
    print(f"Window={args.window}, Step={args.step}")
    print(f"Testing {len(FILTER_CONFIGS)} filter configurations")
    print("=" * 70)
    
    # Initialize WandB
    campaign_group = f"optimization_{run_id[:8]}"  # Use date portion for campaign group
    wandb_run = init_wandb(
        name=f"phase2_filters_{run_id}",
        group=campaign_group,
        job_type="filter_sweep",
        config={
            'phase': 2,
            'window_size': args.window,
            'step_size': args.step,
            'num_filters': len(FILTER_CONFIGS),
        },
        tags=['phase2', 'filter_sweep']
    )
    
    # Load data
    csv_path = pathlib.Path(__file__).parent.parent / 'df_train.csv'
    df = load_data(str(csv_path), verbose=True)
    
    # Test filters
    results = []
    for fc in tqdm(FILTER_CONFIGS, desc="Testing filters"):
        print(f"\n[TEST] Filter: {fc['name']}")
        
        result = test_filter(df, args.window, args.step, fc['config'])
        if result:
            results.append({
                'filter': fc['name'],
                'filter_config': str(fc['config']),
                **result
            })
            print(f"  RF F1: {result['rf_f1']:.4f}")
            
            # Log to WandB
            if wandb_run is not None:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'filter': fc['name'],
                        'rf_f1': result['rf_f1'],
                        'svm_f1': result['svm_f1'],
                        'dt_f1': result['dt_f1'],
                    })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rf_f1', ascending=False)
    
    csv_path = output_dir / 'results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] {csv_path}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.25
    
    ax.bar(x - width, results_df['rf_f1'], width, label='RF', color='forestgreen')
    ax.bar(x, results_df['svm_f1'], width, label='SVM', color='steelblue')
    ax.bar(x + width, results_df['dt_f1'], width, label='DT', color='orange')
    
    ax.set_xlabel('Filter Type')
    ax.set_ylabel('F1 Score')
    ax.set_title('Filter Comparison (All Models)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['filter'], rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'filter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_dir / 'filter_comparison.png'}")
    
    # Best config
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("🏆 BEST FILTER:")
    print("=" * 70)
    print(f"Filter      : {best['filter']}")
    print(f"RF F1 Score : {best['rf_f1']:.4f}")
    print(f"SVM F1      : {best['svm_f1']:.4f}")
    print(f"DT F1       : {best['dt_f1']:.4f}")
    print("=" * 70)
    
    # Improvement over no filter
    baseline = results_df[results_df['filter'] == 'none'].iloc[0]
    improvement = (best['rf_f1'] - baseline['rf_f1']) * 100
    print(f"\nImprovement over no filter: {improvement:+.2f} percentage points")
    
    print(f"\n✅ Phase 2 complete! Results saved to: {output_dir}")
    print(f"\n📋 Next step: Run Phase 3 (LSTM hyperparameter tuning)")
    
    # Log to WandB
    if wandb_run is not None:
        import wandb
        if wandb.run is not None:
            wandb.run.summary.update({
                'best_filter': best['filter'],
                'best_rf_f1': float(best['rf_f1']),
                'improvement_pct': float(improvement),
                'window_size': args.window,
                'step_size': args.step,
            })
            wandb.log({"phase2_comparison": wandb.Image(str(output_dir / 'filter_comparison.png'))})
        finish_wandb()


if __name__ == "__main__":
    main()
