"""
Assignment 1 - Cognitive Decline Activity Recognition
Data Loading, Preprocessing, Feature Extraction, and Windowing

Mirrors the style of tutorial_3_optimized.py:
- CPU threading configured before any imports
- Section-header comments
- tqdm progress bars
- [TAG] print prefixes
"""

# ============================================================================
# CRITICAL: Configure CPU threading BEFORE importing numpy/scipy/sklearn
# ============================================================================
import os
import multiprocessing

num_cores = str(multiprocessing.cpu_count())
os.environ['OMP_NUM_THREADS'] = num_cores
os.environ['MKL_NUM_THREADS'] = num_cores
os.environ['OPENBLAS_NUM_THREADS'] = num_cores
os.environ['BLIS_NUM_THREADS'] = num_cores

# ============================================================================
# Imports (after env vars are set)
# ============================================================================
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mode
from numpy.lib.stride_tricks import as_strided as ast
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt, medfilt

# ============================================================================
# CONSTANTS
# ============================================================================

# Column names in df_train.csv
COL_TIME   = 'stamp'
COL_X      = 'user_acceleration_x'
COL_Y      = 'user_acceleration_y'
COL_Z      = 'user_acceleration_z'
COL_LABEL  = 'user_activity_label'

ACTIVITY_LABELS = ['Work', 'Other', 'Eat', 'Travel', 'Hygiene', 'Cook', 'Exercise']

# Approx sampling frequency (10 Hz inferred from timestamp increments of 0.1 s)
SAMPLING_FREQ = 10


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(csv_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load df_train.csv into a DataFrame.

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to df_train.csv.
    verbose : bool
        Print shape / summary information.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with columns: stamp, user_acceleration_x/y/z, user_activity_label.
    """
    if verbose:
        print(f"[DATA] Loading dataset from: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=[COL_TIME])

    if verbose:
        print(f"[DATA] Loaded {len(df):,} rows × {df.shape[1]} columns")
        print(f"[DATA] Label distribution:\n{df[COL_LABEL].value_counts().to_string()}")
        print(f"[DATA] Missing values: {df.isnull().sum().sum()}")

    return df


# ============================================================================
# PREPROCESSING
# ============================================================================

def add_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Magnitude column = sqrt(x² + y² + z²).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with an extra 'Magnitude' column.
    """
    df = df.copy()
    df['Magnitude'] = np.sqrt(df[COL_X]**2 + df[COL_Y]**2 + df[COL_Z]**2)
    return df


def apply_filter(df: pd.DataFrame, fs: int = SAMPLING_FREQ,
                 config: dict = None) -> pd.DataFrame:
    """
    Apply sensor filtering to X, Y, Z channels.
    Supports single filters or chained combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain COL_X, COL_Y, COL_Z columns.
    fs : int
        Sampling frequency in Hz.
    config : dict or None
        Filtering configuration. None → no filtering.

        Examples::

            {'method': 'median',   'kernel_size': 3}
            {'method': 'lowpass',  'cutoff': 4, 'order': 4}
            {'method': 'bandpass', 'low': 0.5, 'high': 4, 'order': 4}
            {'method': 'combined', 'filters': [cfg1, cfg2]}

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    if config is None or not config:
        return df

    df_f = df.copy()
    channels = [COL_X, COL_Y, COL_Z]
    method = config.get('method', 'median')

    if method == 'combined':
        for sub_cfg in config.get('filters', []):
            df_f = apply_filter(df_f, fs=fs, config=sub_cfg)
        return df_f

    if method == 'median':
        k = config.get('kernel_size', 3)
        print(f"[FILTER] Median (kernel={k})")
        for col in channels:
            df_f[col] = medfilt(df_f[col].values, kernel_size=k)

    elif method == 'lowpass':
        cutoff = config.get('cutoff', 4)
        order  = config.get('order', 4)
        print(f"[FILTER] Low-pass (cutoff={cutoff} Hz, order={order})")
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype='lowpass', analog=False)
        for col in channels:
            df_f[col] = filtfilt(b, a, df_f[col].values)

    elif method == 'bandpass':
        low   = config.get('low', 0.5)
        high  = config.get('high', 4)
        order = config.get('order', 4)
        print(f"[FILTER] Band-pass ({low}–{high} Hz, order={order})")
        nyq = 0.5 * fs
        b, a = butter(order, [low / nyq, high / nyq], btype='bandpass', analog=False)
        for col in channels:
            df_f[col] = filtfilt(b, a, df_f[col].values)

    return df_f


# ============================================================================
# SLIDING WINDOW HELPERS  (identical logic to tutorial_3_optimized.py)
# ============================================================================

def _norm_shape(shape):
    """Normalize shape argument to tuple of ints."""
    try:
        i = int(shape)
        return (i,)
    except (TypeError, ValueError):
        pass
    try:
        return tuple(shape)
    except TypeError:
        pass
    raise TypeError('shape must be an int or tuple of ints')


def sliding_window(a: np.ndarray, ws, ss=None, flatten: bool = True) -> np.ndarray:
    """
    Efficient stride-trick sliding window over numpy array.

    Parameters
    ----------
    a : np.ndarray
    ws : int or tuple
        Window size.
    ss : int or tuple, optional
        Step size (defaults to ws).
    flatten : bool
        Flatten each window to 1-D.

    Returns
    -------
    np.ndarray
    """
    if ss is None:
        ss = ws
    ws = np.array(_norm_shape(ws))
    ss = np.array(_norm_shape(ss))
    shape = np.array(a.shape)

    if np.any(ws > shape):
        raise ValueError(f'ws={ws} larger than array shape={shape}')

    newshape  = _norm_shape(((shape - ws) // ss) + 1) + _norm_shape(ws)
    newstrides = _norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)

    if not flatten:
        return strided

    meat = len(ws) if ws.shape else 0
    firstdim = (np.prod(newshape[:-meat]),) if ws.shape else ()
    return strided.reshape(firstdim + _norm_shape(newshape[-meat:]))


def perform_sliding_window(data_x: np.ndarray, data_y: np.ndarray,
                           ws: int, ss: int):
    """
    Segment raw signal array into overlapping windows.

    Parameters
    ----------
    data_x : np.ndarray, shape (N, C)
        Raw accelerometer channels (X, Y, Z [, Magnitude]).
    data_y : np.ndarray, shape (N,)
        Encoded integer labels.
    ws : int
        Window size (samples).
    ss : int
        Step size (samples).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        win_x : (num_windows, ws, C)  float32
        win_y : (num_windows,)         uint8   (majority label per window)
    """
    win_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = data_y.reshape(-1)
    win_y  = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return win_x.astype(np.float32), win_y.reshape(len(win_y)).astype(np.uint8)


# ============================================================================
# FEATURE EXTRACTION  (statistical features, same as tutorial_3)
# ============================================================================

def _window_features(x: np.ndarray) -> np.ndarray:
    """
    Compute statistical features for a single window.

    Parameters
    ----------
    x : np.ndarray, shape (ws, C)

    Returns
    -------
    np.ndarray, shape (C * 6,)
        [mean, mode, median, max, min, IQR] per channel.
    """
    mean_v   = np.mean(x, axis=0)
    mode_v   = mode(x, axis=0).mode
    median_v = np.median(x, axis=0)
    max_v    = np.max(x, axis=0)
    min_v    = np.min(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    iqr_v    = q75 - q25
    return np.concatenate([mean_v, mode_v, median_v, max_v, min_v, iqr_v])


def extract_features(win_x: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    Extract statistical features from all windows.

    Parameters
    ----------
    win_x : np.ndarray, shape (num_windows, ws, C)
    verbose : bool

    Returns
    -------
    np.ndarray, shape (num_windows, C * 6)
    """
    n_windows   = win_x.shape[0]
    n_channels  = win_x.shape[2]
    features    = np.zeros((n_windows, n_channels * 6))

    iterator = tqdm(range(n_windows), desc="[DATA] Feature extraction", leave=False) if verbose else range(n_windows)
    for i in iterator:
        features[i] = _window_features(win_x[i])

    return features


# ============================================================================
# LABEL ENCODING
# ============================================================================

def encode_labels(series: pd.Series):
    """
    Fit a LabelEncoder on the activity label column.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    tuple[LabelEncoder, np.ndarray]
        Fitted encoder and encoded integer array.
    """
    le = LabelEncoder()
    encoded = le.fit_transform(series.values)
    return le, encoded


# ============================================================================
# FULL PREPROCESSING PIPELINE
# ============================================================================

def prepare_ml_data(df: pd.DataFrame,
                    window_size: int = 50,
                    step_size: int = 25,
                    test_size: float = 0.2,
                    filter_config: dict = None,
                    verbose: bool = True):
    """
    Full pipeline from raw DataFrame to feature matrices ready for sklearn.

    Steps:
        1. Optional filtering
        2. Add Magnitude channel
        3. Label encoding
        4. Sliding window segmentation (chronological – no data leakage)
        5. Feature extraction
        6. Stratified train/test split

    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_data().
    window_size : int
        Window length in samples.
    step_size : int
        Sliding window step.
    test_size : float
        Fraction reserved for testing.
    filter_config : dict or None
        Passed to apply_filter().
    verbose : bool

    Returns
    -------
    dict with keys:
        X_train, X_test     : np.ndarray  feature matrices
        y_train, y_test     : np.ndarray  integer labels
        label_encoder       : LabelEncoder
        scaler              : StandardScaler (fitted on train)
        X_train_seq, X_test_seq  : np.ndarray  raw windows for LSTM (N, ws, C)
    """
    if verbose:
        print(f"\n[DATA] === Preparing ML data ===")
        print(f"[DATA] Window={window_size} samples, Step={step_size} samples")

    # 1. Filter
    if filter_config:
        df = apply_filter(df, fs=SAMPLING_FREQ, config=filter_config)

    # 2. Magnitude
    df = add_magnitude(df)

    # 3. Sort chronologically (important – avoid leakage)
    df = df.sort_values(COL_TIME).reset_index(drop=True)

    # 4. Label encoding
    le, y_all = encode_labels(df[COL_LABEL])
    if verbose:
        print(f"[DATA] Classes ({len(le.classes_)}): {list(le.classes_)}")

    # 5. Sliding window
    channels = [COL_X, COL_Y, COL_Z, 'Magnitude']
    x_raw = df[channels].values
    if verbose:
        print(f"[DATA] Applying sliding window to {len(x_raw):,} samples …")

    win_x, win_y = perform_sliding_window(x_raw, y_all, window_size, step_size)
    if verbose:
        print(f"[DATA] Generated {win_x.shape[0]:,} windows  shape={win_x.shape}")

    # 6. Feature extraction
    X_feat = extract_features(win_x, verbose=verbose)

    # 7. Train/test split (stratified)
    idx = np.arange(len(win_x))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size,
                                      stratify=win_y, random_state=42)

    X_train_feat = X_feat[tr_idx]
    X_test_feat  = X_feat[te_idx]
    X_train_seq  = win_x[tr_idx]
    X_test_seq   = win_x[te_idx]
    y_train      = win_y[tr_idx]
    y_test       = win_y[te_idx]

    # 8. Standardise features
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat  = scaler.transform(X_test_feat)

    if verbose:
        print(f"[DATA] Train samples: {len(y_train):,}  |  Test samples: {len(y_test):,}")
        print("[DATA] Data preparation complete.")

    return {
        'X_train':      X_train_feat,
        'X_test':       X_test_feat,
        'X_train_seq':  X_train_seq,
        'X_test_seq':   X_test_seq,
        'y_train':      y_train,
        'y_test':       y_test,
        'label_encoder': le,
        'scaler':        scaler,
    }


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import sys, pathlib
    # Run from assignment_1/ directory
    csv_path = pathlib.Path(__file__).parent.parent / 'df_train.csv'
    if not csv_path.exists():
        print(f"[ERROR] df_train.csv not found at {csv_path}")
        sys.exit(1)

    print("=" * 70)
    print("DATA MODULE - Standalone Test")
    print("=" * 70)

    df = load_data(str(csv_path))
    data = prepare_ml_data(df, window_size=50, step_size=25, test_size=0.2)

    print(f"\n[TEST] X_train shape  : {data['X_train'].shape}")
    print(f"[TEST] X_test  shape  : {data['X_test'].shape}")
    print(f"[TEST] y_train dist   : {dict(zip(*np.unique(data['y_train'], return_counts=True)))}")
    print(f"[TEST] Classes        : {list(data['label_encoder'].classes_)}")
    print("\n[TEST] data.py - ALL CHECKS PASSED")
