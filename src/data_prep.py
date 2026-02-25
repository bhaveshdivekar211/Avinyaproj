import pickle
import numpy as np
import os

# Setup Dynamic Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def prepare_wesad_timeseries(window_size=100, step_size=50):
    X_eda_total, X_ecg_total, y_total = [], [], []
    
    for subject in range(2, 18):
        RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', f's{subject}', f's{subject}.pkl')
        
        # --- PATH VERIFICATION ---
        if not os.path.exists(RAW_DATA_PATH):
            print(f"WARNING: Cannot find s{subject}.pkl at: {RAW_DATA_PATH}. Skipping.")
            continue

        print(f"Loading WESAD data for subject {subject} from: {RAW_DATA_PATH}...")
        with open(RAW_DATA_PATH, 'rb') as f:
            # Using latin1 is required for WESAD pickle files
            data = pickle.load(f, encoding='latin1')
        
        # Extracting signals
        eda = data['signal']['chest']['EDA'].flatten()
        ecg = data['signal']['chest']['ECG'].flatten()
        labels = data['label'].flatten()
        
        # 1. Downsample (700Hz -> 100Hz)
        eda = eda[::7]
        ecg = ecg[::7]
        labels = labels[::7]
        
        # 2. Normalize (0 to 1 scaling) per subject
        eda = (eda - np.min(eda)) / (np.max(eda) - np.min(eda))
        ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))
        
        # 3. Create Windows
        X_eda, X_ecg, y = [], [], []
        print(f"Creating time-series sequences for subject {subject}...")
        for i in range(0, len(eda) - window_size, step_size):
            X_eda.append(eda[i : i + window_size])
            X_ecg.append(ecg[i : i + window_size])
            window_labels = labels[i : i + window_size]
            # Label 2 in WESAD is 'Stress'
            y.append(1 if 2 in window_labels else 0)
        
        # Append to totals
        X_eda_total.extend(X_eda)
        X_ecg_total.extend(X_ecg)
        y_total.extend(y)
    
    if not X_eda_total:
        print("ERROR: No data processed. Check file paths.")
        return
    
    X_eda_total = np.array(X_eda_total).reshape(-1, window_size, 1)
    X_ecg_total = np.array(X_ecg_total).reshape(-1, window_size, 1)
    y_total = np.array(y_total)
    
    # 4. Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, 'X_eda.npy'), X_eda_total)
    np.save(os.path.join(PROCESSED_DIR, 'X_ecg.npy'), X_ecg_total)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_total)
    print(f"Done! Saved {len(X_eda_total)} sequences to {PROCESSED_DIR}")

if __name__ == "__main__":
    prepare_wesad_timeseries()