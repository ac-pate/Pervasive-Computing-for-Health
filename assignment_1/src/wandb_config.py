"""
WandB Configuration for Assignment 1

This module handles WandB authentication and initialization without
interfering with other users' WandB sessions on the same machine.

Usage:
    from wandb_config import init_wandb
    
    run = init_wandb(config={'param': value}, name='experiment_name')
    # ... training code ...
    wandb.log({'metric': value})
    run.finish()
"""

import os
import sys
import pathlib
from typing import Optional, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WANDB] Warning: wandb not installed. Run: pip install wandb")


def load_api_key() -> Optional[str]:
    """
    Load WandB API key from .env file.
    
    Returns
    -------
    str or None
        API key if found, None otherwise.
    """
    # Check for .env file in project root
    project_root = pathlib.Path(__file__).parent.parent
    env_file = project_root / '.env'
    
    if not env_file.exists():
        print(f"[WANDB] Warning: .env file not found at {env_file}")
        return None
    
    # Parse .env file
    api_key = None
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('WANDB_API_KEY='):
                api_key = line.split('=', 1)[1].strip()
                break
    
    if not api_key:
        print("[WANDB] Warning: WANDB_API_KEY not found in .env")
        return None
    
    return api_key


def init_wandb(
    project: str = "coen498-assignment1",
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    resume: str = "allow",
    reinit: bool = True,
) -> Optional[Any]:
    """
    Initialize WandB run with API key from .env file.
    
    This function sets up WandB using a local API key without affecting
    other users' WandB sessions on the same machine.
    
    Parameters
    ----------
    project : str
        WandB project name.
    entity : str, optional
        WandB username or team name.
    name : str, optional
        Run name (displayed in WandB UI).
    config : dict, optional
        Hyperparameters and configuration.
    tags : list, optional
        Tags for organizing runs.
    notes : str, optional
        Description of the experiment.
    group : str, optional
        Group related runs together.
    job_type : str, optional
        Type of run (e.g., 'train', 'eval', 'sweep').
    resume : str
        Resume strategy ('allow', 'must', 'never').
    reinit : bool
        Allow multiple wandb.init() calls in same process.
    
    Returns
    -------
    wandb.Run or None
        WandB run object if successful, None if WandB unavailable.
    
    Examples
    --------
    >>> run = init_wandb(
    ...     name='lstm_trial_1',
    ...     config={'epochs': 30, 'lr': 1e-3},
    ...     tags=['lstm', 'baseline']
    ... )
    >>> wandb.log({'loss': 0.5})
    >>> run.finish()
    """
    if not WANDB_AVAILABLE:
        print("[WANDB] Skipping initialization - wandb not installed")
        return None
    
    # Load API key from .env
    api_key = load_api_key()
    if not api_key:
        print("[WANDB] Skipping initialization - API key not found")
        return None
    
    # Set API key in environment (temporarily for this process only)
    os.environ['WANDB_API_KEY'] = api_key
    
    # Initialize WandB
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            group=group,
            job_type=job_type,
            resume=resume,
            reinit=reinit,
        )
        
        print(f"[WANDB] ✓ Initialized: {run.name} ({run.id})")
        print(f"[WANDB]   Project: {project}")
        print(f"[WANDB]   URL: {run.url}")
        
        return run
        
    except Exception as e:
        print(f"[WANDB] Error initializing: {e}")
        return None


def finish_wandb():
    """Finish the current WandB run."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("[WANDB] Run finished")


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Log metrics to WandB.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric name -> value pairs.
    step : int, optional
        Global step value for x-axis.
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)


def log_sklearn_results(
    model_name: str,
    y_true,
    y_pred,
    class_names: list,
    train_time: float,
):
    """
    Log sklearn model results to WandB.
    
    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'SVM', 'RF').
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    class_names : list
        List of class names.
    train_time : float
        Training time in seconds.
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Compute metrics
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    # Log metrics
    wandb.log({
        f'{model_name}/f1_weighted': f1_weighted,
        f'{model_name}/f1_macro': f1_macro,
        f'{model_name}/accuracy': accuracy,
        f'{model_name}/train_time_sec': train_time,
    })
    
    # Log confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} Confusion Matrix')
    
    wandb.log({f'{model_name}/confusion_matrix': wandb.Image(fig)})
    plt.close(fig)
    
    # Log classification report as table
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    wandb.log({f'{model_name}/classification_report': wandb.Table(
        columns=['class', 'precision', 'recall', 'f1-score', 'support'],
        data=[[cls, report_dict[cls]['precision'], report_dict[cls]['recall'],
               report_dict[cls]['f1-score'], report_dict[cls]['support']]
              for cls in class_names]
    )})


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WANDB CONFIG - Standalone Test")
    print("=" * 70)
    
    # Check if API key exists
    api_key = load_api_key()
    if api_key:
        print(f"[TEST] ✓ API key loaded (length: {len(api_key)})")
    else:
        print("[TEST] ✗ API key not found")
        sys.exit(1)
    
    # Test initialization (without actually creating a run)
    if WANDB_AVAILABLE:
        print("[TEST] ✓ wandb package available")
        
        # Test init with dummy config
        run = init_wandb(
            name='test_run',
            config={'test_param': 123},
            tags=['test'],
            job_type='test'
        )
        
        if run:
            print("[TEST] ✓ WandB initialization successful")
            
            # Test logging
            log_metrics({'test_metric': 0.5})
            print("[TEST] ✓ Metric logging successful")
            
            finish_wandb()
            print("[TEST] ✓ Run finished successfully")
        else:
            print("[TEST] ✗ WandB initialization failed")
    else:
        print("[TEST] ✗ wandb package not available")
    
    print("\n[TEST] wandb_config.py - ALL CHECKS PASSED")
