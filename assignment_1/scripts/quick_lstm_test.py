"""
Quick LSTM Baseline Test
Test LSTM with 30 epochs to see if it beats RF baseline

Run from assignment_1/:
    python3 scripts/quick_lstm_test.py
"""

import sys
import pathlib
import time
from datetime import datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))

from data import load_data, prepare_ml_data
from model import train_lstm, predict_lstm
from sklearn.metrics import f1_score, classification_report
from wandb_config import init_wandb, finish_wandb

# Quick test config
WINDOW_SIZE = 50
STEP_SIZE = 25
LSTM_EPOCHS = 30

run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

print("=" * 70)
print("QUICK LSTM BASELINE TEST (30 epochs)")
print("=" * 70)

# Initialize WandB
wandb_run = init_wandb(
    name=f"quick_baseline_{run_id}",
    group="quick_tests",
    job_type="baseline",
    config={
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE,
        'lstm_epochs': LSTM_EPOCHS,
    },
    tags=['quick_test', 'baseline']
)

# Load data
csv_path = pathlib.Path(__file__).parent.parent / 'df_train.csv'
df = load_data(str(csv_path), verbose=True)

# Prepare data
data = prepare_ml_data(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE, verbose=True)

# Train LSTM
lstm_result = train_lstm(
    data['X_train_seq'], 
    data['y_train'],
    num_classes=len(data['label_encoder'].classes_),
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    epochs=LSTM_EPOCHS,
    batch_size=256,
    lr=1e-3,
    verbose=True
)

# Predict
lstm_pred = predict_lstm(lstm_result['model'], data['X_test_seq'])

# Evaluate
f1 = f1_score(data['y_test'], lstm_pred, average='weighted')
print(f"\n{'='*70}")
print(f"LSTM F1 (30 epochs): {f1:.4f}")
print(f"Training time: {lstm_result['train_time_sec']:.1f}s")
print(f"{'='*70}")

print("\nClassification Report:")
print(classification_report(data['y_test'], lstm_pred, 
                           target_names=data['label_encoder'].classes_))

print(f"\n✅ Baseline established: LSTM F1 = {f1:.4f}")
print(f"Next: If F1 < 0.70, optimize data params with RF first")
print(f"      If F1 >= 0.70, you're already doing well!")

# Log to WandB and finish
if wandb_run is not None:
    import wandb
    if wandb.run is not None:
        wandb.run.summary.update({
            'lstm_f1': f1,
            'train_time_sec': lstm_result['train_time_sec'],
        })
    finish_wandb()
