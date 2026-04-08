"""
Phase 3 Rerun: LSTM Hyperparameter Optimization WITH Class Weighting
Previous run got F1=0.56 due to 24:1 class imbalance. This rerun uses class weights.

Run from assignment_1/:
    nohup python3 scripts/phase3_lstm_weighted.py --window 100 --step 25 --filter lowpass_3hz > phase3_weighted.log 2>&1 &

Output: results/phase3_lstm_weighted_<timestamp>/
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
from model import train_lstm, predict_lstm
from sklearn.metrics import f1_score, classification_report
from wandb_config import init_wandb, finish_wandb

# ============================================================================
# LSTM Configurations to test (ALL WITH 100 EPOCHS - proper training)
# ============================================================================

LSTM_CONFIGS = [
    # Varying architecture depth (all 100 epochs)
    {'name': 'small_1layer_100ep', 'hidden': 64, 'layers': 1, 'dropout': 0.2, 'epochs': 100},
    {'name': 'medium_2layer_100ep', 'hidden': 128, 'layers': 2, 'dropout': 0.3, 'epochs': 100},
    {'name': 'large_2layer_100ep', 'hidden': 256, 'layers': 2, 'dropout': 0.3, 'epochs': 100},
    
    # Varying width (all 100 epochs)
    {'name': 'small_2layer_100ep', 'hidden': 64, 'layers': 2, 'dropout': 0.3, 'epochs': 100},
    {'name': 'xlarge_2layer_100ep', 'hidden': 512, 'layers': 2, 'dropout': 0.3, 'epochs': 100},
    
    # Deep models (3 layers, 100 epochs)
    {'name': 'medium_3layer_100ep', 'hidden': 128, 'layers': 3, 'dropout': 0.4, 'epochs': 100},
    {'name': 'large_3layer_100ep', 'hidden': 256, 'layers': 3, 'dropout': 0.4, 'epochs': 100},
    
    # Very deep (4 layers, 100 epochs)
    {'name': 'medium_4layer_100ep', 'hidden': 128, 'layers': 4, 'dropout': 0.4, 'epochs': 100},
    {'name': 'large_4layer_100ep', 'hidden': 256, 'layers': 4, 'dropout': 0.5, 'epochs': 100},
]

# ============================================================================

FILTER_MAP = {
    'none': None,
    'lowpass_2hz': {'method': 'lowpass', 'cutoff': 2.0, 'order': 4},
    'lowpass_3hz': {'method': 'lowpass', 'cutoff': 3.0, 'order': 4},
    'lowpass_4hz': {'method': 'lowpass', 'cutoff': 4.0, 'order': 4},
}


def test_lstm_config(data, config, verbose=False):
    """Test one LSTM configuration WITH class weighting."""
    try:
        lstm_result = train_lstm(
            data['X_train_seq'], 
            data['y_train'],
            num_classes=len(data['label_encoder'].classes_),
            hidden_size=config['hidden'],
            num_layers=config['layers'],
            dropout=config['dropout'],
            epochs=config['epochs'],
            batch_size=256,
            lr=1e-3,
            class_weight=True,  # ← KEY DIFFERENCE: Enable class weighting!
            verbose=verbose
        )
        
        lstm_pred = predict_lstm(lstm_result['model'], data['X_test_seq'])
        f1 = f1_score(data['y_test'], lstm_pred, average='weighted')
        
        # Also get per-class F1 to verify minority classes improved
        from sklearn.metrics import classification_report
        report = classification_report(
            data['y_test'], 
            lstm_pred, 
            target_names=data['label_encoder'].classes_,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'f1': f1,
            'train_time': lstm_result['train_time_sec'],
            'final_loss': lstm_result['train_losses'][-1],
            'per_class_f1': {cls: report[cls]['f1-score'] for cls in data['label_encoder'].classes_},
        }
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=100, help='Window size from Phase 1')
    parser.add_argument('--step', type=int, default=25, help='Step size from Phase 1')
    parser.add_argument('--filter', type=str, default='lowpass_3hz', help='Filter from Phase 2')
    args = parser.parse_args()
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = pathlib.Path(__file__).parent.parent / 'results' / f'phase3_lstm_weighted_{run_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PHASE 3 RERUN: LSTM WITH CLASS WEIGHTING")
    print(f"Window={args.window}, Step={args.step}, Filter={args.filter}")
    print(f"Testing {len(LSTM_CONFIGS)} LSTM configurations")
    print("🔥 CLASS WEIGHTING ENABLED (fixes 24:1 imbalance issue)")
    print("=" * 70)
    
    # Initialize WandB
    campaign_group = f"optimization_weighted_{run_id[:8]}"
    wandb_run = init_wandb(
        name=f"phase3_lstm_weighted_{run_id}",
        group=campaign_group,
        job_type="lstm_weighted_sweep",
        config={
            'phase': '3_weighted',
            'window_size': args.window,
            'step_size': args.step,
            'filter': args.filter,
            'class_weighting': True,
            'num_configs': len(LSTM_CONFIGS),
        },
        tags=['phase3', 'lstm_sweep', 'class_weighted', 'rerun']
    )
    
    # Load and prepare data once
    csv_path = pathlib.Path(__file__).parent.parent / 'df_train.csv'
    df = load_data(str(csv_path), verbose=True)
    
    filter_config = FILTER_MAP.get(args.filter, None)
    data = prepare_ml_data(df, window_size=args.window, 
                           step_size=args.step,
                           filter_config=filter_config,
                           verbose=True)
    
    print(f"\n[INFO] Class distribution in training data:")
    y_train_labels = data['label_encoder'].inverse_transform(data['y_train'])
    for cls in data['label_encoder'].classes_:
        count = (y_train_labels == cls).sum()
        pct = count / len(y_train_labels) * 100
        print(f"  {cls:10s}: {count:6d} ({pct:5.2f}%)")
    
    # Test LSTM configs
    results = []
    for config in tqdm(LSTM_CONFIGS, desc="Testing LSTM configs"):
        print(f"\n[TEST] {config['name']}: hidden={config['hidden']}, "
              f"layers={config['layers']}, dropout={config['dropout']}, epochs={config['epochs']}")
        
        result = test_lstm_config(data, config, verbose=False)
        if result:
            results.append({
                'name': config['name'],
                'hidden_size': config['hidden'],
                'num_layers': config['layers'],
                'dropout': config['dropout'],
                'epochs': config['epochs'],
                'f1_score': result['f1'],
                'train_time_sec': result['train_time'],
                'final_loss': result['final_loss'],
                **{f'f1_{cls}': result['per_class_f1'][cls] for cls in data['label_encoder'].classes_},
            })
            print(f"  F1: {result['f1']:.4f} | Time: {result['train_time']:.1f}s")
            print(f"  Per-class F1: " + ", ".join([f"{cls}={result['per_class_f1'][cls]:.3f}" for cls in data['label_encoder'].classes_]))
            
            # Log to WandB
            if wandb_run is not None:
                import wandb
                if wandb.run is not None:
                    log_data = {
                        'config_name': config['name'],
                        'hidden_size': config['hidden'],
                        'num_layers': config['layers'],
                        'dropout': config['dropout'],
                        'epochs': config['epochs'],
                        'lstm_f1': result['f1'],
                        'train_time_sec': result['train_time'],
                        'final_loss': result['final_loss'],
                    }
                    # Add per-class metrics
                    for cls in data['label_encoder'].classes_:
                        log_data[f'f1_{cls}'] = result['per_class_f1'][cls]
                    wandb.log(log_data)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    csv_path = output_dir / 'results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] {csv_path}")
    
    # Plot 1: F1 score comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(results_df))
    axes[0].bar(x, results_df['f1_score'], color='steelblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results_df['name'], rotation=45, ha='right')
    axes[0].set_ylabel('F1 Score (Weighted)')
    axes[0].set_title('LSTM F1 Score Comparison (WITH Class Weighting)', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0.8, color='red', linestyle='--', label='RF Baseline (0.80)', alpha=0.7)
    axes[0].legend()
    
    # Plot 2: F1 vs training time
    axes[1].scatter(results_df['train_time_sec'], results_df['f1_score'], 
                   s=100, alpha=0.6, c=results_df['hidden_size'], cmap='viridis')
    for _, row in results_df.iterrows():
        axes[1].annotate(row['name'], 
                        (row['train_time_sec'], row['f1_score']),
                        fontsize=8, alpha=0.7)
    axes[1].set_xlabel('Training Time (seconds)')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 vs Training Time', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lstm_weighted_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_dir / 'lstm_weighted_comparison.png'}")
    
    # Print top 5
    print("\n" + "=" * 70)
    print("TOP 5 LSTM CONFIGURATIONS (WITH CLASS WEIGHTING):")
    print("=" * 70)
    for i, row in results_df.head(5).iterrows():
        print(f"{i+1}. {row['name']:20s} | F1={row['f1_score']:.4f} | "
              f"Time={row['train_time_sec']:4.0f}s | "
              f"h={row['hidden_size']:3.0f}, l={row['num_layers']:1.0f}, "
              f"d={row['dropout']:.1f}, ep={row['epochs']:2.0f}")
    
    # Best config
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("🏆 BEST LSTM CONFIGURATION (WITH CLASS WEIGHTING):")
    print("=" * 70)
    print(f"Name        : {best['name']}")
    print(f"Hidden Size : {best['hidden_size']:.0f}")
    print(f"Num Layers  : {best['num_layers']:.0f}")
    print(f"Dropout     : {best['dropout']:.2f}")
    print(f"Epochs      : {best['epochs']:.0f}")
    print(f"F1 Score    : {best['f1_score']:.4f}")
    print(f"Train Time  : {best['train_time_sec']:.1f}s")
    print("\nPer-Class F1 Scores:")
    for cls in data['label_encoder'].classes_:
        print(f"  {cls:10s}: {best[f'f1_{cls}']:.4f}")
    print("=" * 70)
    
    # Compare with previous run
    print("\n" + "=" * 70)
    print("📊 IMPROVEMENT OVER PHASE 3 (WITHOUT CLASS WEIGHTING):")
    print("=" * 70)
    print(f"Previous best (no weights): F1 = 0.5625")
    print(f"Current best (with weights): F1 = {best['f1_score']:.4f}")
    print(f"Improvement: {(best['f1_score'] - 0.5625):.4f} ({((best['f1_score'] / 0.5625) - 1) * 100:.1f}%)")
    print("=" * 70)
    
    print(f"\n✅ Phase 3 (weighted) complete! Results saved to: {output_dir}")
    print(f"\n📋 Next: Update main.py with best config and run final training")
    
    # Log to WandB
    if wandb_run is not None:
        import wandb
        if wandb.run is not None:
            wandb.run.summary.update({
                'best_config': best['name'],
                'best_hidden_size': int(best['hidden_size']),
                'best_num_layers': int(best['num_layers']),
                'best_dropout': float(best['dropout']),
                'best_epochs': int(best['epochs']),
                'best_f1': float(best['f1_score']),
                'best_train_time': float(best['train_time_sec']),
                'improvement_vs_unweighted': float(best['f1_score'] - 0.5625),
            })
            wandb.log({"phase3_weighted_comparison": wandb.Image(str(output_dir / 'lstm_weighted_comparison.png'))})
        finish_wandb()


if __name__ == "__main__":
    main()
