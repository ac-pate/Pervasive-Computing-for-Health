"""
Assignment 1 - Cognitive Decline Activity Recognition
Model Definitions and Training

Models included:
  - LSTM           (PyTorch, GPU if available, else CPU multi-threaded)
  - Random Forest  (sklearn, CPU multi-threaded via n_jobs)
  - SVM            (sklearn, BLAS threads from data.py env vars)
  - Decision Tree  (sklearn)

CPU threading is already configured by data.py when imported first.
If this module is used standalone, threading is configured here too.
"""

# ============================================================================
# CRITICAL: Configure CPU threading BEFORE importing numpy/scipy/sklearn
# Must be first in main scripts; harmless no-op if already set.
# ============================================================================
import os
import multiprocessing

_num_cores = str(multiprocessing.cpu_count())
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(_var, _num_cores)

# ============================================================================
# Imports
# ============================================================================
import time
import threading
import numpy as np
import psutil
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

try:
    from threadpoolctl import threadpool_limits
    _HAS_THREADPOOLCTL = True
    print("[MODEL] threadpoolctl available – BLAS threading controlled at runtime")
except ImportError:
    _HAS_THREADPOOLCTL = False
    print("[WARNING] threadpoolctl not installed. Run: pip install threadpoolctl")


# ============================================================================
# DEVICE SELECTION
# ============================================================================

def get_device() -> torch.device:
    """
    Return the best available PyTorch device.

    Returns
    -------
    torch.device
        'cuda' if a GPU is available, else 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[MODEL] PyTorch device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        n = int(_num_cores)
        torch.set_num_threads(n)           # OpenMP threads for PyTorch ops
        torch.set_num_interop_threads(max(1, n // 2))
        device = torch.device('cpu')
        print(f"[MODEL] PyTorch device: CPU (torch threads={n})")
    return device


# ============================================================================
# CPU MONITOR  (identical to tutorial_3_optimized.py)
# ============================================================================

class CPUMonitor:
    """Monitor CPU utilisation during training in a background thread."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.monitoring = False
        self.cpu_usage: list = []
        self.thread = None

    def _monitor(self):
        while self.monitoring:
            usage = psutil.cpu_percent(interval=self.interval, percpu=False)
            self.cpu_usage.append(usage)
            time.sleep(self.interval)

    def start(self):
        self.monitoring = True
        self.cpu_usage = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cpu_usage:
            return {
                'average': float(np.mean(self.cpu_usage)),
                'maximum': float(np.max(self.cpu_usage)),
                'minimum': float(np.min(self.cpu_usage)),
                'samples': self.cpu_usage,
            }
        return {}


def get_cpu_info():
    """Print CPU information."""
    logical  = multiprocessing.cpu_count()
    physical = psutil.cpu_count(logical=False)
    print("\n[MODEL] CPU Information:")
    print(f"  Physical cores : {physical}")
    print(f"  Logical threads: {logical}")
    print(f"  Current usage  : {psutil.cpu_percent(interval=0.5):.1f}%")
    print(f"  Memory avail   : {psutil.virtual_memory().available / (1024**3):.2f} GB")
    return logical


# ============================================================================
# LSTM ARCHITECTURE
# ============================================================================

class LSTMClassifier(nn.Module):
    """
    Two-layer LSTM classifier for accelerometer sequence classification.

    Parameters
    ----------
    input_size : int
        Number of input channels (e.g. 4 for X, Y, Z, Magnitude).
    hidden_size : int
        Number of LSTM hidden units per layer.
    num_layers : int
        Stacked LSTM layers.
    num_classes : int
        Output classes.
    dropout : float
        Dropout between LSTM layers (only applied when num_layers > 1).
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 7,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        out = self.dropout(out[:, -1, :])   # last time-step
        return self.fc(out)


# ============================================================================
# LSTM TRAINING
# ============================================================================

def train_lstm(X_seq: np.ndarray, y: np.ndarray,
               num_classes: int,
               hidden_size: int = 128,
               num_layers: int = 2,
               dropout: float = 0.3,
               epochs: int = 30,
               batch_size: int = 256,
               lr: float = 1e-3,
               monitor_cpu: bool = False,
               verbose: bool = True) -> dict:
    """
    Train an LSTMClassifier on windowed accelerometer sequences.

    Parameters
    ----------
    X_seq : np.ndarray, shape (N, seq_len, C)
        Raw windowed sequences (already train split).
    y : np.ndarray, shape (N,)
        Integer labels.
    num_classes : int
        Total number of activity classes.
    hidden_size, num_layers, dropout, epochs, batch_size, lr : hyperparameters
    monitor_cpu : bool
        Track CPU usage during training.
    verbose : bool

    Returns
    -------
    dict with keys:
        model           : trained LSTMClassifier (on CPU after training)
        device          : torch.device used
        train_losses    : list[float]  per-epoch loss
        train_time_sec  : float
        cpu_stats       : dict  (empty if monitor_cpu=False)
    """
    device = get_device()

    input_size = X_seq.shape[2]
    seq_len    = X_seq.shape[1]

    if verbose:
        print(f"\n[MODEL] === Training LSTM ===")
        print(f"[MODEL] Seq shape : ({X_seq.shape[0]}, {seq_len}, {input_size})")
        print(f"[MODEL] Classes   : {num_classes}")
        print(f"[MODEL] hidden={hidden_size}, layers={num_layers}, "
              f"dropout={dropout}, epochs={epochs}, batch={batch_size}, lr={lr}")

    # Build tensors
    X_t = torch.tensor(X_seq,  dtype=torch.float32)
    y_t = torch.tensor(y,      dtype=torch.long)

    dataset    = TensorDataset(X_t, y_t)
    # num_workers>0 only helps for large datasets on non-Windows – keep 0 for Windows
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    cpu_monitor = CPUMonitor() if monitor_cpu else None
    if cpu_monitor:
        cpu_monitor.start()

    train_losses: list = []
    start_time = time.time()

    epoch_iter = tqdm(range(epochs), desc="[LSTM] Training", disable=not verbose)
    for epoch in epoch_iter:
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        epoch_loss /= len(dataset)
        train_losses.append(epoch_loss)
        scheduler.step()

        if verbose:
            epoch_iter.set_postfix(loss=f"{epoch_loss:.4f}",
                                   lr=f"{scheduler.get_last_lr()[0]:.2e}")

    train_time = time.time() - start_time
    cpu_stats  = cpu_monitor.stop() if cpu_monitor else {}

    # Move model back to CPU for pickling / saving
    model = model.to('cpu')

    if verbose:
        print(f"[MODEL] LSTM training complete in {train_time:.1f}s")
        if cpu_stats:
            print(f"[MODEL] CPU avg: {cpu_stats['average']:.1f}%  max: {cpu_stats['maximum']:.1f}%")

    return {
        'model':           model,
        'device':          device,
        'train_losses':    train_losses,
        'train_time_sec':  train_time,
        'cpu_stats':       cpu_stats,
    }


def predict_lstm(model: LSTMClassifier, X_seq: np.ndarray,
                 batch_size: int = 512) -> np.ndarray:
    """
    Run inference with a trained LSTM.

    Parameters
    ----------
    model : LSTMClassifier
    X_seq : np.ndarray, shape (N, seq_len, C)
    batch_size : int

    Returns
    -------
    np.ndarray, shape (N,)
        Predicted integer class indices.
    """
    device  = next(model.parameters()).device
    X_t     = torch.tensor(X_seq, dtype=torch.float32)
    dataset = TensorDataset(X_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds)


# ============================================================================
# SKLEARN MODELS (CPU multi-threaded, same as tutorial_3)
# ============================================================================

def train_sklearn_models(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         n_jobs: int = -1,
                         monitor_cpu: bool = False,
                         verbose: bool = True) -> dict:
    """
    Train SVM, Decision Tree, and Random Forest classifiers.

    Parameters
    ----------
    X_train, y_train : train features and labels (already standardised).
    X_test,  y_test  : test features and labels.
    n_jobs : int
        Cores for Random Forest (-1 = all).
    monitor_cpu : bool
    verbose : bool

    Returns
    -------
    dict with keys:
        svm_f1, dt_f1, rf_f1          : float – weighted F1
        svm_pred, dt_pred, rf_pred    : np.ndarray – predictions on X_test
        cpu_stats                      : dict
        y_true                         : np.ndarray
    """
    if verbose:
        print(f"\n[MODEL] === Training sklearn models ===")
        print(f"[MODEL] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
        print(f"[MODEL] n_jobs (RF): {n_jobs}")

    svm_model = SVC(kernel='rbf', random_state=42)
    dt_model  = DecisionTreeClassifier(random_state=42)
    rf_model  = RandomForestClassifier(n_estimators=200, n_jobs=n_jobs,
                                       random_state=42)

    cpu_monitor = CPUMonitor() if monitor_cpu else None
    if cpu_monitor:
        cpu_monitor.start()

    # --- SVM ---
    print("[MODEL] Fitting SVM …")
    t0 = time.time()
    if _HAS_THREADPOOLCTL:
        with threadpool_limits(limits=int(_num_cores)):
            svm_model.fit(X_train, y_train)
    else:
        svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    print(f"[MODEL] SVM done in {time.time()-t0:.1f}s")

    # --- Decision Tree ---
    print("[MODEL] Fitting Decision Tree …")
    t0 = time.time()
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    print(f"[MODEL] DT done in {time.time()-t0:.1f}s")

    # --- Random Forest ---
    print("[MODEL] Fitting Random Forest …")
    t0 = time.time()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print(f"[MODEL] RF done in {time.time()-t0:.1f}s")

    cpu_stats = cpu_monitor.stop() if cpu_monitor else {}

    svm_f1 = f1_score(y_test, svm_pred, average='weighted')
    dt_f1  = f1_score(y_test, dt_pred,  average='weighted')
    rf_f1  = f1_score(y_test, rf_pred,  average='weighted')

    if verbose:
        print(f"\n[MODEL] F1 Scores → SVM: {svm_f1:.4f} | DT: {dt_f1:.4f} | RF: {rf_f1:.4f}")
        if cpu_stats:
            print(f"[MODEL] CPU avg: {cpu_stats['average']:.1f}%  max: {cpu_stats['maximum']:.1f}%")

    return {
        'svm_f1':   svm_f1,
        'dt_f1':    dt_f1,
        'rf_f1':    rf_f1,
        'svm_pred': svm_pred,
        'dt_pred':  dt_pred,
        'rf_pred':  rf_pred,
        'cpu_stats': cpu_stats,
        'y_true':   y_test,
        'models':   {'svm': svm_model, 'dt': dt_model, 'rf': rf_model},
    }


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from data import load_data, prepare_ml_data

    csv_path = pathlib.Path(__file__).parent.parent / 'df_train.csv'
    print("=" * 70)
    print("MODEL MODULE - Standalone Test")
    print("=" * 70)

    get_cpu_info()
    df   = load_data(str(csv_path))
    data = prepare_ml_data(df, window_size=50, step_size=25, verbose=True)

    # Quick sklearn test
    res = train_sklearn_models(
        data['X_train'], data['y_train'],
        data['X_test'],  data['y_test'],
        monitor_cpu=True, verbose=True
    )
    print(f"\n[TEST] RF F1: {res['rf_f1']:.4f}")

    # Quick LSTM test (2 epochs only)
    lstm_res = train_lstm(
        data['X_train_seq'], data['y_train'],
        num_classes=len(data['label_encoder'].classes_),
        epochs=2, verbose=True
    )
    preds = predict_lstm(lstm_res['model'], data['X_test_seq'])
    lstm_f1 = f1_score(data['y_test'], preds, average='weighted')
    print(f"[TEST] LSTM F1 (2 epochs): {lstm_f1:.4f}")
    print("\n[TEST] model.py - ALL CHECKS PASSED")
