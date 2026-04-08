"""
Master Script: Run All Optimization Phases
Executes complete hyperparameter search in correct order

Run from assignment_1/:
    python3 scripts/run_all_phases.py

This will:
1. Quick LSTM baseline (30 epochs) - 2 min
2. Phase 1: Data params (window+overlap) - 30 min
3. Phase 2: Filter optimization - 15 min
4. Phase 3: LSTM hyperparameters - 60 min
Total time: ~2 hours

Or run phases individually (see OPTIMIZATION_GUIDE.md)
"""

import sys
import pathlib
import subprocess
import time

def run_command(cmd, description):
    """Run a command and return success status."""
    print("\n" + "=" * 70)
    print(f"⚡ {description}")
    print("=" * 70)
    print(f"Command: {cmd}\n")
    
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed in {elapsed/60:.1f} min")
        return True
    else:
        print(f"\n❌ {description} failed!")
        return False


def main():
    project_root = pathlib.Path(__file__).parent.parent
    
    print("=" * 70)
    print("🚀 MASTER OPTIMIZATION PIPELINE")
    print("=" * 70)
    print("This will run all phases sequentially:")
    print("  1. Quick LSTM baseline (~2 min)")
    print("  2. Phase 1: Window + Overlap (~30 min)")
    print("  3. Phase 2: Filters (~15 min)")
    print("  4. Phase 3: LSTM hyperparameters (~60 min)")
    print("\nTotal estimated time: ~2 hours")
    print("=" * 70)
    
    response = input("\nContinue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Quick LSTM baseline
    if not run_command(
        f"cd {project_root} && python3 scripts/quick_lstm_test.py",
        "Quick LSTM Baseline (30 epochs)"
    ):
        return
    
    # Phase 1: Data parameters
    if not run_command(
        f"cd {project_root} && python3 scripts/phase1_data_params.py",
        "Phase 1: Window + Overlap Optimization"
    ):
        return
    
    # Read best config from Phase 1
    import pandas as pd
    import glob
    
    phase1_dirs = sorted(glob.glob(str(project_root / 'results' / 'phase1_data_params_*')))
    if not phase1_dirs:
        print("❌ Could not find Phase 1 results!")
        return
    
    phase1_csv = pathlib.Path(phase1_dirs[-1]) / 'results.csv'
    df1 = pd.read_csv(phase1_csv)
    best_window = int(df1.iloc[0]['window_size'])
    best_step = int(df1.iloc[0]['step_size'])
    
    print(f"\n✅ Best from Phase 1: Window={best_window}, Step={best_step}")
    
    # Phase 2: Filters
    if not run_command(
        f"cd {project_root} && python3 scripts/phase2_filters.py --window {best_window} --step {best_step}",
        f"Phase 2: Filter Optimization (w={best_window}, s={best_step})"
    ):
        return
    
    # Read best filter from Phase 2
    phase2_dirs = sorted(glob.glob(str(project_root / 'results' / 'phase2_filters_*')))
    if not phase2_dirs:
        print("❌ Could not find Phase 2 results!")
        return
    
    phase2_csv = pathlib.Path(phase2_dirs[-1]) / 'results.csv'
    df2 = pd.read_csv(phase2_csv)
    best_filter = df2.iloc[0]['filter']
    
    print(f"\n✅ Best from Phase 2: Filter={best_filter}")
    
    # Phase 3: LSTM hyperparameters
    if not run_command(
        f"cd {project_root} && python3 scripts/phase3_lstm_hyperparam.py --window {best_window} --step {best_step} --filter {best_filter}",
        f"Phase 3: LSTM Hyperparameter Tuning (w={best_window}, s={best_step}, f={best_filter})"
    ):
        return
    
    # Read best LSTM config
    phase3_dirs = sorted(glob.glob(str(project_root / 'results' / 'phase3_lstm_hyperparam_*')))
    if not phase3_dirs:
        print("❌ Could not find Phase 3 results!")
        return
    
    phase3_csv = pathlib.Path(phase3_dirs[-1]) / 'results.csv'
    df3 = pd.read_csv(phase3_csv)
    best_lstm = df3.iloc[0]
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 ALL PHASES COMPLETE!")
    print("=" * 70)
    print("\n📊 OPTIMAL CONFIGURATION:")
    print("-" * 70)
    print(f"Window Size  : {best_window} samples ({best_window/10:.1f}s)")
    print(f"Step Size    : {best_step} samples")
    print(f"Filter       : {best_filter}")
    print(f"LSTM Hidden  : {best_lstm['hidden_size']:.0f}")
    print(f"LSTM Layers  : {best_lstm['num_layers']:.0f}")
    print(f"LSTM Dropout : {best_lstm['dropout']:.2f}")
    print(f"LSTM Epochs  : {best_lstm['epochs']:.0f}")
    print(f"\nBest F1 Score: {best_lstm['f1_score']:.4f}")
    print("=" * 70)
    
    print("\n📋 Next Steps:")
    print("1. Update src/main.py with these optimal values")
    print("2. Run final training: python3 src/main.py")
    print("3. Check WandB dashboard for results")
    print(f"\nResults saved in: results/phase*_{pathlib.Path(phase1_dirs[-1]).name.split('_')[-1][:8]}_*/")


if __name__ == "__main__":
    main()
