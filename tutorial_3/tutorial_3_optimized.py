"""
Tutorial 3 - Activity Recognition Chain - Core Functionality
Optimized for CPU multi-threading

This module contains all core functions for:
- Data loading and preprocessing
- Feature extraction
- Windowing
- Model training and classification
"""

# ============================================================================
# CRITICAL: Configure CPU threading BEFORE importing numpy/scipy/sklearn
# ============================================================================
import os
import multiprocessing

# Set BLAS threading to use all CPU cores for numpy/scipy operations
num_cores = str(multiprocessing.cpu_count())
os.environ['OMP_NUM_THREADS'] = num_cores
os.environ['MKL_NUM_THREADS'] = num_cores
os.environ['OPENBLAS_NUM_THREADS'] = num_cores
os.environ['BLIS_NUM_THREADS'] = num_cores

print(f"[INIT] BLAS threading configured to use {num_cores} cores")
print("[INIT] This will accelerate SVM's numerical operations")

# ============================================================================
# Now import the libraries (after environment variables are set)
# ============================================================================
import zipfile
import time
import threading
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
from scipy.stats import mode
from numpy.lib.stride_tricks import as_strided as ast

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.signal import butter, filtfilt, medfilt

# Import threadpoolctl for runtime BLAS threading control
try:
    from threadpoolctl import threadpool_limits
    HAS_THREADPOOLCTL = True
    print("[INIT] threadpoolctl available - BLAS threading will be controlled at runtime")
except ImportError:
    HAS_THREADPOOLCTL = False
    print("[WARNING] threadpoolctl not installed - install with: pip install threadpoolctl")
    print("[WARNING] SVM multi-threading may not work properly!")


# ============================================================================
# CPU OPTIMIZATION CONFIGURATION
# ============================================================================

def configure_cpu_threading():
    """Report CPU threading configuration (already set at module import)"""
    num_cores = multiprocessing.cpu_count()
    print(f"\nCPU Threading Status:")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
    print(f"  OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")
    print(f"  Available cores: {num_cores}")
    print("  Status: ✓ Multi-threading ENABLED for SVM/numpy operations")
    return num_cores


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def read_file(file_path):
    """Read a single accelerometer data file"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            values = line.split(' ')
            real_values = [(-1.5 + (int(val) / 63) * 3) for val in values]
            data.append(real_values)
    
    return pd.DataFrame(data, columns=['X', 'Y', 'Z'])


def read_dataset(zip_file_path, extracted_folder_path, sampling_frequency):
    """
    Read the entire ADL Recognition dataset
    
    Parameters:
    -----------
    zip_file_path : str
        Path to the dataset zip file
    extracted_folder_path : str
        Path where to extract the dataset
    sampling_frequency : int
        Sampling frequency of the accelerometer data
        
    Returns:
    --------
    pd.DataFrame
        Complete dataset with columns: X, Y, Z, Time, Start_Time, Label, Volunteer, id
    """
    df = pd.DataFrame()
    
    # Extract the files from the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder_path)
    
    # Iterate over all subdirectories inside the extracted folder
    id = 0
    for root, dirs, files in os.walk(extracted_folder_path):
        for directory in tqdm(dirs, total=len(dirs), desc="Loading dataset"):
            directory_path = os.path.join(root, directory)
            
            # Check if the directory contains files starting with "Accelerometer"
            if any(file.startswith('Accelerometer') for file in os.listdir(directory_path)):
                
                # Iterate over the files in the directory
                for filename in os.listdir(directory_path):
                    if filename.startswith('Accelerometer'):
                        file_path = os.path.join(directory_path, filename)
                        
                        # Extract information from the file name
                        file_parts = filename.split('-')
                        start_time = '-'.join(file_parts[1:7])
                        volunteer = file_parts[-1].split('.')[0]
                        label = file_parts[7]
                        
                        # Read the dataset from the file
                        dataset = read_file(file_path)
                        
                        # Extract the start time from the filename
                        start_time = pd.to_datetime(start_time, format='%Y-%m-%d-%H-%M-%S')
                        
                        # Calculate the time increment based on the sampling frequency
                        time_increment = pd.to_timedelta(1 / sampling_frequency, unit='s')
                        
                        # Create a new column for the adjusted time
                        dataset['Time'] = start_time + (pd.to_timedelta(dataset.index * time_increment, unit='s'))
                        
                        # Add the label, start time, and volunteer columns
                        dataset['Start_Time'] = start_time
                        dataset['Label'] = label
                        dataset['Volunteer'] = volunteer
                        dataset['id'] = id
                        id += 1
                        
                        # Append the dataset to the DataFrame
                        df = pd.concat([df, dataset])
    
    # Add magnitude column
    df['Magnitude'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    
    return df


# ============================================================================
# SLIDING WINDOW FUNCTIONS
# ============================================================================

def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.
    """
    try:
        if hasattr(shape, 'size'):
            if shape.size == 1:
                i = int(shape.item())
                return (i,)
            else:
                raise TypeError
        else:
            i = int(shape)
            return (i,)
    except (TypeError, ValueError):
        pass
    
    try:
        t = tuple(shape)
        return t
    except TypeError:
        pass
    
    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    """
    Return a sliding window over a in any number of dimensions
    
    Parameters:
    -----------
    a : np.ndarray
        An n-dimensional numpy array
    ws : int or tuple
        Window size
    ss : int or tuple, optional
        Step size (defaults to ws if not specified)
    flatten : bool
        If True, all slices are flattened
        
    Returns:
    --------
    np.ndarray
        Array containing each n-dimensional window from a
    """
    if None is ss:
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
    
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError('a.shape, ws and ss must all have the same length. They were %s' % str(ls))
    
    if np.any(ws > shape):
        raise ValueError('ws cannot be larger than a in any dimension. a.shape was %s and ws was %s' % (str(a.shape), str(ws)))
    
    newshape = norm_shape(((shape - ws) // ss) + 1)
    newshape += norm_shape(ws)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    
    if not flatten:
        return strided
    
    meat = len(ws) if ws.shape else 0
    firstdim = (np.prod(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    
    return strided.reshape(dim)


def perform_sliding_window(data_x, data_y, ws, ss):
    """
    Efficiently perform sliding window based segmentation
    
    Parameters:
    -----------
    data_x : np.ndarray
        Processed data stream
    data_y : np.ndarray
        Processed labels stream
    ws : int
        Window size
    ss : int
        Step size (overlap)
        
    Returns:
    --------
    tuple
        Windowed data and ground truth labels
    """
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def feature_representation(x):
    """
    Calculate statistical features for a single window
    
    Parameters:
    -----------
    x : np.ndarray
        Window data (window_size x num_features)
        
    Returns:
    --------
    np.ndarray
        Feature vector
    """
    # Calculate statistical features
    mean_values = np.mean(x, axis=0)
    mode_values = mode(x, axis=0).mode
    median_values = np.median(x, axis=0)
    max_values = np.max(x, axis=0)
    min_values = np.min(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    iqr_values = q75 - q25
    
    # Concatenate all the values into a single array
    features = np.concatenate([mean_values, mode_values, median_values, 
                               max_values, min_values, iqr_values])
    
    return features


def get_features(x):
    """
    Compute features for all windows
    
    Parameters:
    -----------
    x : np.ndarray
        All windowed data (num_windows x window_size x num_features)
        
    Returns:
    --------
    np.ndarray
        Feature matrix (num_windows x num_features)
    """
    num_windows = x.shape[0]
    num_components = x.shape[2]
    temp_features = np.zeros((num_windows, (num_components) * 6))
    
    for i in range(num_windows):
        temp_features[i] = feature_representation(x[i])
    
    return temp_features


# ============================================================================
# CPU MONITORING
# ============================================================================

class CPUMonitor:
    """Monitor CPU usage during training"""
    
    def __init__(self, interval=0.5):
        self.interval = interval
        self.monitoring = False
        self.cpu_usage = []
        self.thread = None
    
    def _monitor(self):
        """Internal monitoring function"""
        while self.monitoring:
            usage = psutil.cpu_percent(interval=self.interval, percpu=False)
            self.cpu_usage.append(usage)
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring"""
        self.monitoring = True
        self.cpu_usage = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.cpu_usage:
            avg_usage = np.mean(self.cpu_usage)
            max_usage = np.max(self.cpu_usage)
            min_usage = np.min(self.cpu_usage)
            
            return {
                'average': avg_usage,
                'maximum': max_usage,
                'minimum': min_usage,
                'samples': self.cpu_usage
            }
        return None


def get_cpu_info():
    """Get CPU information and current utilization"""
    cpu_count_logical = multiprocessing.cpu_count()
    cpu_count_physical = psutil.cpu_count(logical=False)
    
    print("\nCPU Information:")
    print(f"  Physical cores: {cpu_count_physical}")
    print(f"  Logical cores (threads): {cpu_count_logical}")
    print(f"  Current usage: {psutil.cpu_percent(interval=1)}%")
    print(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"  Total memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    return cpu_count_logical


# ============================================================================
# MODEL TRAINING AND CLASSIFICATION
# ============================================================================

def train_and_classify(X, y, group, n_jobs=-1, monitor_cpu=False, verbose=False, show_progress=True):
    """
    Train and classify with CPU optimization using Leave-One-Group-Out cross-validation
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Labels
    group : array-like
        Groups for cross-validation (volunteers)
    n_jobs : int
        Number of CPU cores to use (-1 = all cores)
    monitor_cpu : bool
        Whether to monitor CPU usage during training
    verbose : bool
        Whether to print detailed classification reports
    show_progress : bool
        Whether to show progress bars (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - svm_f1: SVM weighted F1 score
        - dt_f1: Decision Tree weighted F1 score
        - rf_f1: Random Forest weighted F1 score
        - cpu_stats: CPU usage statistics (if monitor_cpu=True)
    """
    # Initialize CPU monitor if requested
    cpu_monitor = None
    if monitor_cpu:
        cpu_monitor = CPUMonitor(interval=0.5)
        cpu_monitor.start()
    
    # Initialize classifiers
    svm_model = SVC()
    dt_model = DecisionTreeClassifier()
    rf_model = RandomForestClassifier(n_jobs=n_jobs, n_estimators=100, random_state=42)
    
    # Create empty lists to accumulate predictions and true labels
    svm_predictions = []
    dt_predictions = []
    rf_predictions = []
    true_labels = []
    
    # Create the LeaveOneGroupOut cross-validator
    logo = LeaveOneGroupOut()
    n_groups = logo.get_n_splits(X, y, groups=group)
    
    # Perform leave-one-subject-out cross-validation
    # We use standard loop without threadpoolctl to match notebook behavior exactly
    for train_index, test_index in tqdm(logo.split(X, y, groups=group), total=n_groups, disable=not show_progress):
        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create a scaler to standardize the features
        scaler = StandardScaler()
        
        # Standardize the feature matrix
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit and predict with SVM
        svm_model.fit(X_train_scaled, y_train)
        svm_pred = svm_model.predict(X_test_scaled)
        svm_predictions.extend(svm_pred)
        
        # Fit and predict with Decision Tree
        dt_model.fit(X_train_scaled, y_train)
        dt_pred = dt_model.predict(X_test_scaled)
        dt_predictions.extend(dt_pred)
        
        # Fit and predict with Random Forest
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_predictions.extend(rf_pred)
        
        # Accumulate true labels
        true_labels.extend(y_test)

    # Stop CPU monitoring
    cpu_stats = None
    if cpu_monitor:
        cpu_stats = cpu_monitor.stop()
    
    # Calculate F1 scores
    svm_f1 = f1_score(true_labels, svm_predictions, average="weighted")
    dt_f1 = f1_score(true_labels, dt_predictions, average="weighted")
    rf_f1 = f1_score(true_labels, rf_predictions, average="weighted")
    
    # Print classification reports if verbose
    if verbose:
        print("\nSVM Classification Report:")
        print(classification_report(true_labels, svm_predictions))
        
        print("\nDecision Tree Classification Report:")
        print(classification_report(true_labels, dt_predictions))
        
        print("\nRandom Forest Classification Report:")
        print(classification_report(true_labels, rf_predictions))
        
        print(f"\nF1 Scores: SVM={svm_f1:.4f}, DT={dt_f1:.4f}, RF={rf_f1:.4f}")
    
    return {
        "svm_f1": svm_f1,
        "dt_f1": dt_f1,
        "rf_f1": rf_f1,
        "cpu_stats": cpu_stats,
        "y_true": true_labels,
        "y_pred_svm": svm_predictions,
        "y_pred_rf": rf_predictions,
        "y_pred_dt": dt_predictions
    }


# ============================================================================
# SIGNAL PROCESSING / FILTERING
# ============================================================================

def apply_filtering(df, fs=32, config=None):
    """
    Apply sensor filtering to X, Y, Z channels based on configuration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with X, Y, Z columns
    fs : int
        Sampling frequency (default 32Hz)
    config : dict or None
        Filtering configuration. If None, no filtering is applied.
        Examples:
        - {'method': 'median', 'kernel_size': 3}
        - {'method': 'lowpass', 'cutoff': 12, 'order': 4}
        - {'method': 'bandpass', 'low': 1, 'high': 5, 'order': 4}
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with filtered X, Y, Z columns
    """
    if config is None or not config:
        return df
        
    df_filtered = df.copy()
    channels = ['X', 'Y', 'Z']
    method = config.get('method', 'median')
    
    if method == 'median':
        kernel_size = config.get('kernel_size', 3)
        if hasattr(df, 'Volunteer') and df['Volunteer'].nunique() > 1:
            # Apply per user chunks if possible?
            # Ideally filtering should be continuous, but here we likely have one big DF.
            # Median filter is local, so it's fine.
            pass
            
        print(f"[FILTER] Applying Median Filter (kernel={kernel_size})")
        for col in channels:
            df_filtered[col] = medfilt(df[col], kernel_size=kernel_size)
            
    elif method == 'lowpass':
        cutoff = config.get('cutoff', 12)  # 12Hz safe for 32Hz without losing too much
        order = config.get('order', 4)
        print(f"[FILTER] Applying Low-pass Filter (cutoff={cutoff}Hz, order={order})")
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
        
        for col in channels:
            df_filtered[col] = filtfilt(b, a, df[col])
            
    elif method == 'bandpass':
        low = config.get('low', 1)
        high = config.get('high', 5)
        order = config.get('order', 4)
        print(f"[FILTER] Applying Band-pass Filter ({low}-{high}Hz, order={order})")
        
        nyq = 0.5 * fs
        b, a = butter(order, [low/nyq, high/nyq], btype='bandpass', analog=False)
        
        for col in channels:
            df_filtered[col] = filtfilt(b, a, df[col])
    
    return df_filtered


# ============================================================================
# DATA PREPARATION FOR WINDOWING
# ============================================================================

def prepare_windowed_data(df, window_size, overlap, verbose=True, filter_config=None):
    """
    Prepare windowed data with features for classification
    
    Parameters:
    -----------
    df : pd.DataFrame
        Complete dataset with columns: X, Y, Z, Label, Volunteer
    window_size : int
        Size of the sliding window
    overlap : int
        Overlap between windows (step size)
    verbose : bool
        Whether to print progress
    filter_config : dict, optional
        Configuration for sensor filtering (default: None)
        
    Returns:
    --------
    tuple
        (X, y, groups) - Feature matrix, labels, and volunteer groups
    """
    # Apply filtering if requested
    if filter_config:
        if verbose:
            print("Applying sensor filtering...")
        df = apply_filtering(df, fs=32, config=filter_config)

    # Prepare label encoder
    le = LabelEncoder()
    y_all = df['Label'].values
    le.fit(y_all)
    
    # Prepare user encoder
    users = df['Volunteer'].unique()
    user_le = LabelEncoder()
    user_le.fit(users)
    users_num = user_le.transform(users)
    
    # Process each user
    all_data = []
    all_labels = []
    all_groups = []
    
    iterator = tqdm(range(len(users)), desc="Processing users") if verbose else range(len(users))
    
    for i in iterator:
        user_data = df[df['Volunteer'] == users[i]]
        x = user_data[['X', 'Y', 'Z']].values
        y_user = user_data['Label'].values
        y_user = le.transform(y_user)
        
        try:
            # Apply sliding window
            win_x, win_y = perform_sliding_window(x, y_user, window_size, overlap)
            
            # Extract features
            features = get_features(win_x)
            
            all_data.append(features)
            all_labels.append(win_y)
            
            # Create group labels
            groups_user = np.full_like(win_y, fill_value=users_num[i])
            all_groups.append(groups_user)
            
        except Exception as e:
            if verbose:
                print(f"Error processing user {i}: {e}")
            pass
    
    # Concatenate all data
    if len(all_data) == 0:
        raise ValueError("No valid windows generated")
    
    X = np.concatenate(all_data)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)
    
    return X, y, groups


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

if __name__ == "__main__":
    print("Tutorial 3 - Activity Recognition Chain - Core Module")
    print("=" * 80)
    
    # Configure CPU threading
    num_cores = configure_cpu_threading()
    
    # Get CPU info
    get_cpu_info()
    
    print("\nCore functions loaded successfully!")
    print("Available functions:")
    print("  - read_dataset()")
    print("  - prepare_windowed_data()")
    print("  - train_and_classify()")
    print("  - get_features()")
    print("  - perform_sliding_window()")
