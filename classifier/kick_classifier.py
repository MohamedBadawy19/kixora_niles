"""
Taekwondo Kick Classifier
==========================
Handles two critical cross-device issues:
1. FREQUENCY MISMATCH: Dataset was likely ~100 Hz (MPU6050). Phone may be 50-400 Hz.
   -> All features are computed in TIME units (seconds), not sample counts.
   -> New data is resampled to match training frequency before feature extraction.

2. IMU TYPE MISMATCH: Dataset sensor (MPU6050, +/-2g range, +/-19.6 m/s^2 clipping).
   Phone sensor (various, typically +/-8g or +/-16g, different noise profile).
   -> Z-score normalization per segment removes absolute scale dependence.
   -> Ratio-based features (axis ratios, energy ratios) are scale-invariant.
   -> Gravity direction is used for implicit orientation alignment.

Pipeline: Raw CSV -> Resample -> Segment -> Normalize -> Extract Features -> Classify
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import find_peaks, resample
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_SAMPLING_RATE = 100  # Hz - estimated from peak intervals (~0.85s/kick)
SEGMENT_WINDOW_SEC = 0.5     # seconds - window around each peak for feature extraction
MIN_PEAK_DISTANCE_SEC = 0.3  # seconds - minimum time between detected kicks


# ============================================================
# 1. DATA LOADING
# ============================================================
def load_dataset(base_path):
    """Load all files from the Indonesian dataset."""
    files = {}
    
    # CSV files
    for name in ['Ap Kiri', 'Diam', 'Doylo Kiri']:
        path = os.path.join(base_path, f'{name}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = ['X', 'Y', 'Z']
            files[name] = df
    
    # Excel files
    for name in ['Ap Kanan', 'Doylo Kanan']:
        path = os.path.join(base_path, f'{name}.xlsx')
        if os.path.exists(path):
            df = pd.read_excel(path, sheet_name=0)
            df.columns = ['X', 'Y', 'Z']
            files[name] = df
    
    return files


# ============================================================
# 2. RESAMPLING (handles frequency mismatch)
# ============================================================
def resample_to_target(data, source_fs, target_fs):
    """
    Resample data from source_fs to target_fs.
    This is KEY for phone compatibility - phone may record at 50, 100, 200, or 400 Hz.
    We resample everything to DATASET_SAMPLING_RATE before feature extraction.
    """
    if source_fs == target_fs:
        return data
    
    n_source = len(data)
    n_target = int(n_source * target_fs / source_fs)
    
    if isinstance(data, pd.DataFrame):
        resampled = pd.DataFrame()
        for col in data.columns:
            resampled[col] = resample(data[col].values, n_target)
        return resampled
    else:
        return resample(data, n_target)


# ============================================================
# 3. SEGMENTATION (frequency-aware)
# ============================================================
def segment_kicks(df, fs=DATASET_SAMPLING_RATE, label=None):
    """
    Detect individual kicks using acceleration magnitude peaks.
    All parameters are in SECONDS, not samples, so this works at any sampling rate.
    """
    mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).values
    
    # Smoothing window in samples (5ms equivalent)
    smooth_window = max(3, int(0.05 * fs))
    mag_smooth = pd.Series(mag).rolling(window=smooth_window, center=True).mean()
    mag_smooth = mag_smooth.fillna(pd.Series(mag))
    mag_smooth = mag_smooth.values
    
    # Adaptive threshold
    threshold = np.mean(mag_smooth) + 1.5 * np.std(mag_smooth)
    min_distance_samples = int(MIN_PEAK_DISTANCE_SEC * fs)
    
    peaks, properties = find_peaks(
        mag_smooth,
        height=threshold,
        distance=min_distance_samples,
        prominence=np.std(mag_smooth) * 0.5
    )
    
    # Extract segments around each peak
    half_window = int(SEGMENT_WINDOW_SEC * fs)
    segments = []
    
    for p in peaks:
        start = max(0, p - half_window)
        end = min(len(df), p + half_window)
        
        seg_data = {
            'X': df['X'].values[start:end],
            'Y': df['Y'].values[start:end],
            'Z': df['Z'].values[start:end],
            'mag': mag[start:end],
            'peak_idx': p,
            'label': label,
            'fs': fs,
        }
        segments.append(seg_data)
    
    return segments


# ============================================================
# 4. NORMALIZATION (handles IMU type mismatch)
# ============================================================
def normalize_segment(seg):
    """
    Apply z-score normalization per segment.
    This removes dependence on:
    - Absolute scale (different IMU ranges: +/-2g vs +/-8g)
    - DC offset (different gravity calibrations)
    - Sensitivity differences between IMU chips
    
    After normalization, features are about the SHAPE of the signal,
    not its absolute magnitude.
    """
    normalized = {}
    for axis in ['X', 'Y', 'Z', 'mag']:
        data = seg[axis].copy()
        mean = np.mean(data)
        std = np.std(data)
        if std > 1e-6:  # avoid division by zero for flat signals
            normalized[axis] = (data - mean) / std
        else:
            normalized[axis] = data - mean
        # Also keep raw stats before normalization (for ratio features)
        normalized[f'{axis}_raw_mean'] = mean
        normalized[f'{axis}_raw_std'] = std
    
    normalized['label'] = seg['label']
    normalized['fs'] = seg['fs']
    return normalized


# ============================================================
# 5. FEATURE EXTRACTION (IMU-agnostic features)
# ============================================================
def extract_features(seg, fs=DATASET_SAMPLING_RATE):
    """
    Extract features designed to be ROBUST across different IMUs.
    
    Three categories:
    A) Normalized time-domain: computed on z-scored data -> scale-invariant
    B) Ratio features: axis ratios, energy ratios -> scale-invariant  
    C) Frequency features: computed with correct fs -> frequency-invariant
    """
    features = {}
    norm = normalize_segment(seg)
    
    for axis in ['X', 'Y', 'Z', 'mag']:
        data_norm = norm[axis]
        data_raw = seg[axis]
        
        if len(data_norm) < 5:
            return None
        
        # --- A) Normalized time-domain features ---
        features[f'{axis}_norm_mean'] = float(np.mean(data_norm))
        features[f'{axis}_norm_std'] = float(np.std(data_norm))
        features[f'{axis}_norm_rms'] = float(np.sqrt(np.mean(data_norm**2)))
        features[f'{axis}_norm_peak'] = float(np.max(np.abs(data_norm)))
        features[f'{axis}_norm_skew'] = float(skew(data_norm))
        features[f'{axis}_norm_kurtosis'] = float(kurtosis(data_norm))
        features[f'{axis}_norm_range'] = float(np.max(data_norm) - np.min(data_norm))
        
        # Zero crossing rate (scale-invariant, frequency-aware)
        zero_crossings = np.sum(np.diff(np.sign(data_norm - np.mean(data_norm))) != 0)
        duration_sec = len(data_norm) / fs
        features[f'{axis}_zcr_per_sec'] = float(zero_crossings / duration_sec) if duration_sec > 0 else 0.0
        
        # --- C) Frequency domain features (with correct fs) ---
        n = len(data_norm)
        if n > 10:
            yf = np.abs(fft(data_norm))[:n//2]
            xf = fftfreq(n, 1.0/fs)[:n//2]
            
            if len(yf) > 1:
                # Dominant frequency
                features[f'{axis}_dom_freq'] = float(xf[np.argmax(yf[1:]) + 1])
                
                # Spectral energy distribution (ratio-based, so scale-invariant)
                total_energy = np.sum(yf**2) + 1e-10
                
                # Low freq energy (0-5 Hz) vs high freq (5+ Hz)
                low_mask = xf < 5.0
                high_mask = xf >= 5.0
                features[f'{axis}_low_freq_ratio'] = float(np.sum(yf[low_mask]**2) / total_energy)
                features[f'{axis}_high_freq_ratio'] = float(np.sum(yf[high_mask]**2) / total_energy)
                
                # Spectral centroid (in Hz - scale-invariant)
                features[f'{axis}_spectral_centroid'] = float(
                    np.sum(xf * yf) / (np.sum(yf) + 1e-10)
                )
            else:
                features[f'{axis}_dom_freq'] = 0.0
                features[f'{axis}_low_freq_ratio'] = 0.0
                features[f'{axis}_high_freq_ratio'] = 0.0
                features[f'{axis}_spectral_centroid'] = 0.0
    
    # --- B) Ratio features (completely scale-invariant) ---
    raw_stds = {axis: norm[f'{axis}_raw_std'] for axis in ['X', 'Y', 'Z']}
    raw_means = {axis: norm[f'{axis}_raw_mean'] for axis in ['X', 'Y', 'Z']}
    total_std = sum(raw_stds.values()) + 1e-10
    
    # Which axis has most energy (tells you kick direction)
    features['x_energy_ratio'] = float(raw_stds['X'] / total_std)
    features['y_energy_ratio'] = float(raw_stds['Y'] / total_std)
    features['z_energy_ratio'] = float(raw_stds['Z'] / total_std)
    
    # Gravity direction ratios (which axis gravity is on -> orientation info)
    total_mean = np.sqrt(sum(m**2 for m in raw_means.values())) + 1e-10
    features['gravity_x_ratio'] = float(raw_means['X'] / total_mean)
    features['gravity_y_ratio'] = float(raw_means['Y'] / total_mean)
    features['gravity_z_ratio'] = float(raw_means['Z'] / total_mean)
    
    # Cross-axis correlations (dimensionless, invariant to scale)
    if len(seg['X']) > 5:
        try:
            features['corr_xy'] = float(np.corrcoef(seg['X'], seg['Y'])[0, 1])
            features['corr_xz'] = float(np.corrcoef(seg['X'], seg['Z'])[0, 1])
            features['corr_yz'] = float(np.corrcoef(seg['Y'], seg['Z'])[0, 1])
        except:
            features['corr_xy'] = 0.0
            features['corr_xz'] = 0.0
            features['corr_yz'] = 0.0
    
    # Replace any NaN/inf with 0
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0
    
    return features


# ============================================================
# 6. BUILD TRAINING DATA
# ============================================================
def build_training_data(files, fs=DATASET_SAMPLING_RATE):
    """Segment all files and extract features to create X, y training arrays."""
    
    # Map file names to class labels
    label_map = {
        'Ap Kanan': 'Ap Chagi',       # Front kick (right leg)
        'Ap Kiri': 'Ap Chagi',         # Front kick (left leg)
        'Doylo Kanan': 'Dolyo Chagi',  # Roundhouse kick (right leg)
        'Doylo Kiri': 'Dolyo Chagi',   # Roundhouse kick (left leg)
        'Diam': 'Idle',                # Standing idle
    }
    
    all_features = []
    all_labels = []
    all_sources = []  # Track which file each segment came from
    
    for name, df in files.items():
        label = label_map.get(name, 'Unknown')
        segments = segment_kicks(df, fs=fs, label=label)
        
        for seg in segments:
            feat = extract_features(seg, fs=fs)
            if feat is not None:
                all_features.append(feat)
                all_labels.append(label)
                all_sources.append(name)
    
    # Convert to DataFrame
    X = pd.DataFrame(all_features)
    y = np.array(all_labels)
    sources = np.array(all_sources)
    
    return X, y, sources


# ============================================================
# 7. TRAIN AND EVALUATE
# ============================================================
def train_and_evaluate(X, y, sources):
    """
    Train multiple classifiers and evaluate with:
    1. Stratified K-Fold cross-validation (general performance)
    2. Leave-One-File-Out cross-validation (tests generalization across recordings)
    """
    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION")
    print("=" * 60)
    
    print(f"\nDataset: {len(X)} segments")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Features: {X.shape[1]}")
    print(f"Sources: {dict(zip(*np.unique(sources, return_counts=True)))}")
    
    # Handle any remaining NaN
    X = X.fillna(0)
    
    # --- Test 1: Stratified 5-Fold CV ---
    print("\n--- Stratified 5-Fold Cross-Validation ---")
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=3,
            min_samples_leaf=2, random_state=42, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=3, random_state=42
        ),
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42,
                         class_weight='balanced'),
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf),
        ])
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
        mean_score = scores.mean()
        print(f"  {name}: {mean_score:.4f} (+/- {scores.std():.4f})  "
              f"[{', '.join(f'{s:.3f}' for s in scores)}]")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = pipeline
            best_name = name
    
    # --- Test 2: Leave-One-File-Out CV (generalization test) ---
    print("\n--- Leave-One-File-Out Cross-Validation (Generalization Test) ---")
    
    # Create group indices based on source file
    unique_sources = np.unique(sources)
    group_indices = np.array([np.where(unique_sources == s)[0][0] for s in sources])
    
    logo = LeaveOneGroupOut()
    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf),
        ])
        scores = cross_val_score(pipeline, X, y, cv=logo, groups=group_indices, scoring='accuracy')
        print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})  "
              f"[{', '.join(f'{s:.3f}' for s in scores)}]")
    
    # --- Train final model on ALL data ---
    print(f"\n--- Training Final Model: {best_name} ({best_score:.4f}) ---")
    best_model.fit(X, y)
    
    # Full training report
    y_pred = best_model.predict(X)
    print(f"\nFull Training Classification Report:")
    print(classification_report(y, y_pred))
    
    # Confusion matrix
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=labels)
    print("Confusion Matrix:")
    print(f"{'':>15}", end="")
    for l in labels:
        print(f"{l:>15}", end="")
    print()
    for i, l in enumerate(labels):
        print(f"{l:>15}", end="")
        for j in range(len(labels)):
            print(f"{cm[i,j]:>15}", end="")
        print()
    
    # Feature importances (if Random Forest or GB)
    clf_inner = best_model.named_steps['classifier']
    if hasattr(clf_inner, 'feature_importances_'):
        importances = clf_inner.feature_importances_
        feat_names = X.columns
        top_indices = np.argsort(importances)[::-1][:15]
        print(f"\nTop 15 Most Important Features:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. {feat_names[idx]}: {importances[idx]:.4f}")
    
    return best_model, best_name, best_score


# ============================================================
# 8. SAVE MODEL
# ============================================================
def save_model(model, feature_names, output_dir):
    """Save the trained model and metadata for phone validation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'kick_classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'training_sampling_rate': DATASET_SAMPLING_RATE,
        'segment_window_sec': SEGMENT_WINDOW_SEC,
        'min_peak_distance_sec': MIN_PEAK_DISTANCE_SEC,
        'feature_names': list(feature_names),
        'classes': ['Ap Chagi', 'Dolyo Chagi', 'Idle'],
        'notes': {
            'frequency_handling': 'Resample phone data to training_sampling_rate before feature extraction',
            'imu_handling': 'Z-score normalization per segment + ratio features make this IMU-agnostic',
            'gravity': 'Dataset includes gravity (raw accelerometer, not linear acceleration)',
        }
    }
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return model_path, metadata_path


# ============================================================
# 9. PHONE DATA PROCESSOR (for validation)
# ============================================================
def process_phone_data(csv_path, model_path, phone_fs, gravity_included=True):
    """
    Process accelerometer data recorded on a phone and classify kicks.
    
    This handles the two key mismatches:
    1. Resamples from phone_fs to training_fs
    2. Uses the same z-score + ratio features that are IMU-agnostic
    
    Args:
        csv_path: Path to CSV file from phone (must have X/Y/Z or similar columns)
        model_path: Path to saved model (.pkl)
        phone_fs: Sampling rate of the phone recording (Hz)
        gravity_included: True if phone recorded raw accel (with gravity).
                         The training data includes gravity, so this should match.
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata
    meta_path = model_path.replace('.pkl', '').replace('kick_classifier', 'model_metadata') + '.json'
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    target_fs = metadata['training_sampling_rate']
    
    # Load phone data
    df = pd.read_csv(csv_path)
    
    # Try to identify columns (different apps use different names)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'accel' in col_lower and 'x' in col_lower:
            col_mapping['X'] = col
        elif 'accel' in col_lower and 'y' in col_lower:
            col_mapping['Y'] = col
        elif 'accel' in col_lower and 'z' in col_lower:
            col_mapping['Z'] = col
        elif col_lower == 'x':
            col_mapping['X'] = col
        elif col_lower == 'y':
            col_mapping['Y'] = col
        elif col_lower == 'z':
            col_mapping['Z'] = col
    
    if len(col_mapping) < 3:
        # Fallback: assume first 3 numeric columns are X, Y, Z
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            col_mapping = {'X': numeric_cols[0], 'Y': numeric_cols[1], 'Z': numeric_cols[2]}
        else:
            raise ValueError(f"Cannot find X/Y/Z columns. Found: {list(df.columns)}")
    
    phone_df = pd.DataFrame({
        'X': df[col_mapping['X']].values,
        'Y': df[col_mapping['Y']].values,
        'Z': df[col_mapping['Z']].values,
    })
    
    print(f"Phone data: {len(phone_df)} samples at {phone_fs} Hz ({len(phone_df)/phone_fs:.1f} seconds)")
    
    # Step 1: RESAMPLE to training frequency
    if phone_fs != target_fs:
        print(f"Resampling: {phone_fs} Hz -> {target_fs} Hz")
        phone_df = resample_to_target(phone_df, phone_fs, target_fs)
        print(f"After resampling: {len(phone_df)} samples")
    
    # Step 2: Segment kicks
    segments = segment_kicks(phone_df, fs=target_fs, label='unknown')
    print(f"Detected {len(segments)} kick/movement events")
    
    if len(segments) == 0:
        print("No kicks detected! Check if the phone was recording during kicks.")
        return []
    
    # Step 3: Extract features and classify
    results = []
    for i, seg in enumerate(segments):
        feat = extract_features(seg, fs=target_fs)
        if feat is None:
            continue
        
        feat_df = pd.DataFrame([feat])
        
        # Ensure same columns as training
        for col in metadata['feature_names']:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[metadata['feature_names']]
        
        prediction = model.predict(feat_df)[0]
        
        # Get confidence if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feat_df)[0]
            confidence = max(proba)
            class_probas = dict(zip(model.classes_, proba))
        else:
            confidence = None
            class_probas = {}
        
        time_sec = seg['peak_idx'] / target_fs
        
        result = {
            'segment': i + 1,
            'time_sec': round(time_sec, 2),
            'prediction': prediction,
            'confidence': round(confidence, 3) if confidence else None,
            'probabilities': {k: round(v, 3) for k, v in class_probas.items()},
        }
        results.append(result)
        
        conf_str = f" ({confidence:.1%})" if confidence else ""
        print(f"  Segment {i+1} at {time_sec:.1f}s: {prediction}{conf_str}")
    
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    BASE_PATH = r'd:\Downloads\Dataset\raw_data'
    OUTPUT_DIR = r'd:\Downloads\Dataset\classifier\output'
    
    print("=" * 60)
    print("TAEKWONDO KICK CLASSIFIER")
    print("Handles: Frequency mismatch + IMU type mismatch")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading dataset...")
    files = load_dataset(BASE_PATH)
    print(f"  Loaded {len(files)} files: {list(files.keys())}")
    
    # Build training data
    print("\n[2/4] Segmenting kicks & extracting IMU-agnostic features...")
    X, y, sources = build_training_data(files, fs=DATASET_SAMPLING_RATE)
    print(f"  Extracted {len(X)} segments with {X.shape[1]} features each")
    
    # Train and evaluate
    print("\n[3/4] Training classifiers...")
    model, model_name, model_score = train_and_evaluate(X, y, sources)
    
    # Save
    print("\n[4/4] Saving model...")
    save_model(model, X.columns, OUTPUT_DIR)
    
    # --- Generate visualization ---
    # Confusion matrix heatmap
    y_pred = model.predict(X)
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'Confusion Matrix - {model_name}\n(5-Fold CV Accuracy: {model_score:.1%})', 
                 fontsize=13, fontweight='bold')
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=14)
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    print(f"\n{'=' * 60}")
    print(f"DONE - Model: {model_name} | CV Accuracy: {model_score:.1%}")
    print(f"{'=' * 60}")
    print(f"\nTo validate with your phone, run:")
    print(f"  python kick_classifier.py --phone your_recording.csv --phone_fs 100")
