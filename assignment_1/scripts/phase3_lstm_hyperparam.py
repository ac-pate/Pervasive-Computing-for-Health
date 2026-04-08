"""
Phase 3: LSTM Hyperparameter Optimization
Test LSTM architectures using optimal data params from Phase 1 & 2

Run from assignment_1/:
    python3 scripts/phase3_lstm_hyperparam.py --window 50 --step 25 --filter lowpass_3hz

Output: results/phase3_lstm_hyperparam_<timestamp>/
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
from sklearn.metrics import f1_score
from wandb_config import init_wandb, finish_wandb

# ============================================================================
# LSTM Configurations to test
# ============================================================================

# Rationale for these choices:
# hidden_size: 64 (small), 128 (medium), 256 (large)
# num_layers: 1 (simple), 2 (standard), 3 (deep)
# dropout: 0.2 (light), 0.3 (standard), 0.4 (heavy)
# epochs: 20 (quick), 30 (standard), 50 (thorough)

LSTM_CONFIGS = [
    # Quick baseline configs (20 epochs)
    {'name': 'small_1layer', 'hidden': 64, 'layers': 1, 'dropout': 0.2, 'epochs': 20},
    {'name': 'medium_2layer', 'hidden': 128, 'layers': 2, 'dropout': 0.3, 'epochs': 20},
    {'name': 'large_2layer', 'hidden': 256, 'layers': 2, 'dropout': 0.3, 'epochs': 20},
    
    # Standard training (30 epochs)
    {'name': 'small_2layer_30ep', 'hidden': 64, 'layers': 2, 'dropout': 0.3, 'epochs': 30},
    {'name': 'medium_2layer_30ep', 'hidden': 128, 'layers': 2, 'dropout': 0.3, 'epochs': 30},
    {'name': 'large_2layer_30ep', 'hidden': 256, 'layers': 2, 'dropout': 0.3, 'epochs': 30},
    
    # Deep models (3 layers, 30 epochs)
    {'name': 'medium_3layer', 'hidden': 128, 'layers': 3, 'dropout': 0.4, 'epochs': 30},
    {'name': 'large_3layer', 'hidden': 256, 'layers': 3, 'dropout': 0.4, 'epochs': 30},
    
    # Extended training for best architecture (50 epochs)
    {'name': 'medium_2layer_50ep', 'hidden': 128, 'layers': 2, 'dropout': 0.3, 'epochs': 50},
]

# ============================================================================

FILTER_MAP = {
    'none': None,
    'lowpass_2hz': {'method': 'lowpass', 'cutoff': 2.0, 'order': 4},
    'lowpass_3hz': {'method': 'lowpass', 'cutoff': 3.0, 'order': 4},
    'lowpass_4hz': {'method': 'lowpass', 'cutoff': 4.0, 'order': 4},
}


def test_lstm_config(data, config, verbose=False):
    """Test one LSTM configuration."""
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
            verbose=verbose
        )
        
        lstm_pred = predict_lstm(lstm_result['model'], data['X_test_seq'])
        f1 = f1_score(data['y_test'], lstm_pred, average='weighted')
        
        return {
            'f1': f1,
            'train_time': lstm_result['train_time_sec'],
            'final_loss': lstm_result['train_losses'][-1],
        }
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=50, help='Window size from Phase 1')
    parser.add_argument('--step', type=int, default=25, help='Step size from Phase 1')
    parser.add_argument('--filter', type=str, default='none', help='Filter from Phase 2')
    args = parser.parse_args()
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = pathlib.Path(__file__).parent.parent / 'results' / f'phase3_lstm_hyperparam_{run_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PHASE 3: LSTM HYPERPARAMETER OPTIMIZATION")
    print(f"Window={args.window}, Step={args.step}, Filter={args.filter}")
    print(f"Testing {len(LSTM_CONFIGS)} LSTM configurations")
    print("=" * 70)
    
    # Initialize WandB
    campaign_group = f"optimization_{run_id[:8]}"
    wandb_run = init_wandb(
        name=f"phase3_lstm_{run_id}",
        group=campaign_group,
        job_type="lstm_hyperparam_sweep",
        config={
            'phase': 3,
            'window_size': args.window,
            'step_size': args.step,
            'filter': args.filter,
            'num_configs': len(LSTM_CONFIGS),
        },
        tags=['phase3', 'lstm_sweep', 'hyperparameter_tuning']
    )
    
    # Load and prepare data once
    csv_path = pathlib.Path(__file__).parent.parent / 'df_train.csv'
    df = load_data(str(csv_path), verbose=True)
    
    filter_config = FILTER_MAP.get(args.filter, None)
    data = prepare_ml_data(df, window_size=args.window, 
                           step_size=args.step,
                           filter_config=filter_config,
                           verbose=True)
    
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
            })
            print(f"  F1: {result['f1']:.4f} | Time: {result['train_time']:.1f}s")
            
            # Log to WandB
            if wandb_run is not None:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'config_name': config['name'],
                        'hidden_size': config['hidden'],
                        'num_layers': config['layers'],
                        'dropout': config['dropout'],
                        'epochs': config['epochs'],
                        'lstm_f1': result['f1'],
                        'train_time_sec': result['train_time'],
                        'final_loss': result['final_loss'],
                    })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    csv_path = output_dir / 'results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] {csv_path}")
    
    # Plot 1: F1 score comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    ax.barh(results_df['name'], results_df['f1_score'], color=colors)
    ax.set_xlabel('F1 Score')
    ax.set_title('LSTM Configurations: F1 Score', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: F1 vs training time
    ax = axes[1]
    scatter = ax.scatter(results_df['train_time_sec'], results_df['f1_score'],
                        s=results_df['hidden_size'], c=results_df['num_layers'],
                        cmap='coolwarm', alpha=0.7, edgecolors='black')
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Training Time\n(size=hidden_size, color=num_layers)', 
                fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Num Layers')
    
    # Annotate best
    best_idx = results_df['f1_score'].idxmax()
    best = results_df.loc[best_idx]
    ax.annotate('Best', xy=(best['train_time_sec'], best['f1_score']),
               xytext=(10, 10), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lstm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_dir / 'lstm_comparison.png'}")
    
    # Print top 5
    print("\n" + "=" * 70)
    print("TOP 5 LSTM CONFIGURATIONS:")
    print("=" * 70)
    for i, row in results_df.head(5).iterrows():
        print(f"{i+1}. {row['name']:20s} | F1={row['f1_score']:.4f} | "
              f"Time={row['train_time_sec']:4.0f}s | "
              f"h={row['hidden_size']:3.0f}, l={row['num_layers']:1.0f}, "
              f"d={row['dropout']:.1f}, ep={row['epochs']:2.0f}")
    
    # Best config
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("🏆 BEST LSTM CONFIGURATION:")
    print("=" * 70)
    print(f"Name        : {best['name']}")
    print(f"Hidden Size : {best['hidden_size']:.0f}")
    print(f"Num Layers  : {best['num_layers']:.0f}")
    print(f"Dropout     : {best['dropout']:.2f}")
    print(f"Epochs      : {best['epochs']:.0f}")
    print(f"F1 Score    : {best['f1_score']:.4f}")
    print(f"Train Time  : {best['train_time_sec']:.1f}s")
    print("=" * 70)
    
    print(f"\n✅ Phase 3 complete! Results saved to: {output_dir}")
    print(f"\n📋 Final step: Compare best LSTM vs best RF")
    
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
            })
            wandb.log({"phase3_comparison": wandb.Image(str(output_dir / 'lstm_comparison.png'))})
        finish_wandb()


if __name__ == "__main__":
    main()
