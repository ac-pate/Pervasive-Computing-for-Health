
import pandas as pd
import os
import sys
from experiments import plot_experiment_results

def main():
    # Base path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_base = os.path.join(base_dir, 'results')
    
    # Find the latest run folder
    if not os.path.exists(results_base):
        print(f"Results directory not found: {results_base}")
        return

    runs = [d for d in os.listdir(results_base) if d.startswith('run_')]
    if not runs:
        print("No run directories found.")
        return
    
    # Sort to get latest
    runs.sort(reverse=True)
    latest_run = runs[0]
    run_dir = os.path.join(results_base, latest_run)
    
    # Extract timestamp from folder name run_YYYYMMDD_HHMMSS
    try:
        tag = latest_run.split('run_')[1]
    except IndexError:
        tag = "unknown"
        
    print(f"Loading results from: {run_dir}")
    print(f"Run Tag: {tag}")
    
    # Find CSV files
    exp1_file = None
    exp2_file = None
    
    for f in os.listdir(run_dir):
        if f.startswith('experiment1') and f.endswith('.csv'):
            exp1_file = os.path.join(run_dir, f)
        elif f.startswith('experiment2') and f.endswith('.csv'):
            exp2_file = os.path.join(run_dir, f)
            
    if not exp1_file or not exp2_file:
        print("Could not find both experiment CSV files in the folder.")
        print(f"Found Exp1: {exp1_file}")
        print(f"Found Exp2: {exp2_file}")
        return
        
    # Load DataFrames
    print("Loading DataFrames...")
    exp1_df = pd.read_csv(exp1_file)
    exp2_df = pd.read_csv(exp2_file)
    
    # Plot
    print("Generating plots...")
    plot_experiment_results(exp1_df, exp2_df, output_dir=run_dir, tag=tag)
    print("Done!")

if __name__ == "__main__":
    main()
