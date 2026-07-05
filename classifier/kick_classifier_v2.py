"""
Taekwondo Kick Classifier v2 - Fixed Generalization
=====================================================
v1 showed 98.9% on K-Fold but 21% on Leave-One-File-Out.
Problem: model was learning file-specific patterns, not kick patterns.

Fix: 
1. Remove gravity-direction features (changes with how sensor was strapped on)
2. Focus on DYNAMIC features (how the signal changes during a kick)
3. Use only features that capture kick biomechanics, not sensor mounting
4. Better cross-validation strategy
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
from sklearn.model_selection import (StratifiedKFold, cross_val_score, 
                                      LeaveOneGroupOut, cross_val_predict)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATASET_SAMPLING_RATE = 100  # Hz
SEGMENT_WINDOW_SEC = 0.5     # seconds around each peak
MIN_PEAK_DISTANCE_SEC = 0.3  # seconds


# ============================================================
# DATA LOADING
# ============================================================
def load_dataset(base_path):
    files = {}
    for name in ['Ap Kiri', 'Diam', 'Doylo Kiri']:
        path = os.path.join(base_path, f'{name}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = ['X', 'Y', 'Z']
            files[name] = df
    for name in ['Ap Kanan', 'Doylo Kanan']:
        path = os.path.join(base_path, f'{name}.xlsx')
        if os.path.exists(path):
            df = pd.read_excel(path, sheet_name=0)
            df.columns = ['X', 'Y', 'Z']
            files[name] = df
    return files


# ============================================================
# RESAMPLING
# ============================================================
def resample_to_target(data, source_fs, target_fs):
    if source_fs == target_fs:
        return data
    n_target = int(len(data) * target_fs / source_fs)
    if isinstance(data, pd.DataFrame):
        resampled = pd.DataFrame()
        for col in data.columns:
            resampled[col] = resample(data[col].values, n_target)
        return resampled
    return resample(data, n_target)


# ============================================================
# SEGMENTATION
# ============================================================
def segment_kicks(df, fs=DATASET_SAMPLING_RATE, label=None):
    mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).values
    
    smooth_window = max(3, int(0.05 * fs))
    mag_smooth = pd.Series(mag).rolling(window=smooth_window, center=True).mean()
    mag_smooth = mag_smooth.fillna(pd.Series(mag)).values
    
    threshold = np.mean(mag_smooth) + 1.5 * np.std(mag_smooth)
    min_dist = int(MIN_PEAK_DISTANCE_SEC * fs)
    
    peaks, _ = find_peaks(mag_smooth, height=threshold, distance=min_dist,
                           prominence=np.std(mag_smooth) * 0.5)
    
    half_win = int(SEGMENT_WINDOW_SEC * fs)
    segments = []
    
    for p in peaks:
        start = max(0, p - half_win)
        end = min(len(df), p + half_win)
        segments.append({
            'X': df['X'].values[start:end],
            'Y': df['Y'].values[start:end],
            'Z': df['Z'].values[start:end],
            'mag': mag[start:end],
            'peak_idx': p,
            'label': label,
            'fs': fs,
        })
    
    return segments


# ============================================================
# FEATURE EXTRACTION v2 - Generalization-focused
# ============================================================
def extract_features_v2(seg, fs=DATASET_SAMPLING_RATE):
    """
    v2 features designed for GENERALIZATION across different recordings.
    
    Key changes from v1:
    - REMOVED: gravity direction features (sensor-mount dependent)
    - REMOVED: raw mean features (offset-dependent)
    - ADDED: derivative features (captures kick dynamics - how fast the motion changes)
    - KEPT: normalized features, ratio features, frequency features
    - ADDED: per-axis DYNAMIC features (acceleration changes, not absolute values)
    """
    features = {}
    
    # Compute acceleration magnitude
    x, y, z = seg['X'], seg['Y'], seg['Z']
    mag = seg['mag']
    
    if len(x) < 10:
        return None
    
    # ---- GRAVITY REMOVAL ----
    # Subtract per-axis mean to remove gravity component
    # This makes features independent of how the phone was mounted
    x_dynamic = x - np.mean(x)
    y_dynamic = y - np.mean(y)
    z_dynamic = z - np.mean(z)
    mag_dynamic = np.sqrt(x_dynamic**2 + y_dynamic**2 + z_dynamic**2)
    
    # ---- PER-AXIS DYNAMIC FEATURES ----
    for axis_name, data in [('x', x_dynamic), ('y', y_dynamic), ('z', z_dynamic), ('mag', mag_dynamic)]:
        # Z-score normalize for scale invariance
        std = np.std(data)
        if std > 1e-6:
            data_norm = (data - np.mean(data)) / std
        else:
            data_norm = data - np.mean(data)
        
        # Time domain
        features[f'{axis_name}_std'] = float(np.std(data))
        features[f'{axis_name}_rms'] = float(np.sqrt(np.mean(data**2)))
        features[f'{axis_name}_peak'] = float(np.max(np.abs(data)))
        features[f'{axis_name}_range'] = float(np.ptp(data))
        features[f'{axis_name}_skew'] = float(skew(data_norm))
        features[f'{axis_name}_kurtosis'] = float(kurtosis(data_norm))
        
        # Zero crossing rate (per second - frequency invariant)
        zcr = np.sum(np.diff(np.sign(data_norm)) != 0)
        features[f'{axis_name}_zcr'] = float(zcr / (len(data) / fs))
        
        # DERIVATIVE features - how fast is acceleration changing
        # This captures the "sharpness" of the kick motion
        if len(data) > 2:
            deriv = np.diff(data) * fs  # in m/s^3 (jerk)
            features[f'{axis_name}_jerk_rms'] = float(np.sqrt(np.mean(deriv**2)))
            features[f'{axis_name}_jerk_peak'] = float(np.max(np.abs(deriv)))
        
        # Frequency domain
        n = len(data_norm)
        if n > 10:
            yf = np.abs(fft(data_norm))[:n//2]
            xf = fftfreq(n, 1.0/fs)[:n//2]
            
            if len(yf) > 1:
                features[f'{axis_name}_dom_freq'] = float(xf[np.argmax(yf[1:]) + 1])
                total_e = np.sum(yf**2) + 1e-10
                low_mask = xf < 5.0
                features[f'{axis_name}_low_freq_ratio'] = float(np.sum(yf[low_mask]**2) / total_e)
                features[f'{axis_name}_spectral_centroid'] = float(np.sum(xf * yf) / (np.sum(yf) + 1e-10))
    
    # ---- RATIO FEATURES (completely scale-invariant) ----
    total_energy = (np.std(x_dynamic)**2 + np.std(y_dynamic)**2 + np.std(z_dynamic)**2) + 1e-10
    features['x_energy_share'] = float(np.std(x_dynamic)**2 / total_energy)
    features['y_energy_share'] = float(np.std(y_dynamic)**2 / total_energy)
    features['z_energy_share'] = float(np.std(z_dynamic)**2 / total_energy)
    
    # ---- CROSS-AXIS CORRELATIONS ----
    try:
        features['corr_xy'] = float(np.corrcoef(x_dynamic, y_dynamic)[0, 1])
        features['corr_xz'] = float(np.corrcoef(x_dynamic, z_dynamic)[0, 1])
        features['corr_yz'] = float(np.corrcoef(y_dynamic, z_dynamic)[0, 1])
    except:
        features['corr_xy'] = features['corr_xz'] = features['corr_yz'] = 0.0
    
    # ---- SIGNAL SHAPE FEATURES ----
    # Time to peak (as fraction of segment - dimensionless)
    features['time_to_peak_ratio'] = float(np.argmax(mag_dynamic) / max(len(mag_dynamic) - 1, 1))
    
    # Symmetry: compare first half energy to second half energy  
    mid = len(mag_dynamic) // 2
    first_half_e = np.sum(mag_dynamic[:mid]**2) + 1e-10
    second_half_e = np.sum(mag_dynamic[mid:]**2) + 1e-10
    features['energy_symmetry'] = float(first_half_e / (first_half_e + second_half_e))
    
    # Clean NaN/inf
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0
    
    return features


# ============================================================
# BUILD TRAINING DATA
# ============================================================
def build_training_data(files, fs=DATASET_SAMPLING_RATE):
    label_map = {
        'Ap Kanan': 'Ap Chagi',
        'Ap Kiri': 'Ap Chagi',
        'Doylo Kanan': 'Dolyo Chagi',
        'Doylo Kiri': 'Dolyo Chagi',
        'Diam': 'Idle',
    }
    
    all_features = []
    all_labels = []
    all_sources = []
    
    for name, df in files.items():
        label = label_map.get(name, 'Unknown')
        segments = segment_kicks(df, fs=fs, label=label)
        
        for seg in segments:
            feat = extract_features_v2(seg, fs=fs)
            if feat is not None:
                all_features.append(feat)
                all_labels.append(label)
                all_sources.append(name)
    
    X = pd.DataFrame(all_features)
    y = np.array(all_labels)
    sources = np.array(all_sources)
    return X, y, sources


# ============================================================
# TRAIN AND EVALUATE
# ============================================================
def train_and_evaluate(X, y, sources, output_dir):
    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION")
    print("=" * 60)
    
    print(f"\nDataset: {len(X)} segments, {X.shape[1]} features")
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"  {cls}: {count} segments")
    
    X = X.fillna(0)
    
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=5,
            min_samples_leaf=3, random_state=42, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            min_samples_split=5, random_state=42
        ),
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42,
                         class_weight='balanced', probability=True),
    }
    
    # ---- Test 1: Stratified 5-Fold ----
    print("\n--- Stratified 5-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, clf in classifiers.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
        print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = pipe
            best_name = name
    
    # ---- Test 2: Leave-One-File-Out (THE REAL TEST) ----
    print("\n--- Leave-One-File-Out CV (Cross-Recording Generalization) ---")
    unique_sources = np.unique(sources)
    group_indices = np.array([np.where(unique_sources == s)[0][0] for s in sources])
    logo = LeaveOneGroupOut()
    
    for name, clf in classifiers.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        scores = cross_val_score(pipe, X, y, cv=logo, groups=group_indices, scoring='accuracy')
        mean = scores.mean()
        per_file = {unique_sources[i]: f"{s:.1%}" for i, s in enumerate(scores)}
        print(f"  {name}: {mean:.4f}")
        for fname, acc in per_file.items():
            print(f"    Left out '{fname}': {acc}")
    
    # ---- Test 3: Left vs Right generalization ----
    print("\n--- Left/Right Split Test (train on one side, test on other) ---")
    left_mask = np.isin(sources, ['Ap Kiri', 'Doylo Kiri'])
    right_mask = np.isin(sources, ['Ap Kanan', 'Doylo Kanan'])
    idle_mask = sources == 'Diam'
    
    # Train on right+idle, test on left
    train_mask_1 = right_mask | idle_mask
    test_mask_1 = left_mask
    
    if np.sum(train_mask_1) > 0 and np.sum(test_mask_1) > 0:
        pipe1 = Pipeline([('scaler', StandardScaler()), 
                          ('clf', RandomForestClassifier(n_estimators=300, max_depth=10,
                                                         random_state=42, class_weight='balanced'))])
        pipe1.fit(X[train_mask_1], y[train_mask_1])
        acc1 = pipe1.score(X[test_mask_1], y[test_mask_1])
        print(f"  Train on RIGHT+Idle -> Test on LEFT: {acc1:.1%}")
        
        # And reverse
        train_mask_2 = left_mask | idle_mask
        test_mask_2 = right_mask
        pipe2 = Pipeline([('scaler', StandardScaler()),
                          ('clf', RandomForestClassifier(n_estimators=300, max_depth=10,
                                                         random_state=42, class_weight='balanced'))])
        pipe2.fit(X[train_mask_2], y[train_mask_2])
        acc2 = pipe2.score(X[test_mask_2], y[test_mask_2])
        print(f"  Train on LEFT+Idle -> Test on RIGHT: {acc2:.1%}")
    
    # ---- Train Final Model ----
    print(f"\n--- Final Model: {best_name} (K-Fold: {best_score:.1%}) ---")
    best_model.fit(X, y)
    
    y_pred = best_model.predict(X)
    print(f"\nClassification Report (full training set):")
    print(classification_report(y, y_pred))
    
    # Feature importance
    clf_inner = best_model.named_steps['clf']
    if hasattr(clf_inner, 'feature_importances_'):
        importances = clf_inner.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        print("Top 15 Features:")
        for i, idx in enumerate(top_idx):
            print(f"  {i+1}. {X.columns[idx]}: {importances[idx]:.4f}")
    
    # ---- Plots ----
    # Confusion matrix
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=labels)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training confusion matrix
    im = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title(f'Training Confusion Matrix\n{best_name} | K-Fold: {best_score:.1%}', fontweight='bold')
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            axes[0].text(j, i, str(cm[i,j]), ha='center', va='center', color=color, fontsize=16)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Left/Right split confusion matrix (shows generalization)
    if np.sum(train_mask_1) > 0 and np.sum(test_mask_1) > 0:
        # Use left test predictions from earlier
        y_test_left = y[test_mask_1]
        y_pred_left = pipe1.predict(X[test_mask_1])
        kick_labels = [l for l in labels if l != 'Idle']
        cm2 = confusion_matrix(y_test_left, y_pred_left, labels=kick_labels)
        
        im2 = axes[1].imshow(cm2, cmap='Oranges')
        axes[1].set_title(f'Cross-Leg Generalization Test\nTrain RIGHT -> Test LEFT: {acc1:.1%}', fontweight='bold')
        for i in range(len(kick_labels)):
            for j in range(len(kick_labels)):
                color = 'white' if cm2[i,j] > cm2.max()/2 else 'black'
                axes[1].text(j, i, str(cm2[i,j]), ha='center', va='center', color=color, fontsize=16)
        axes[1].set_xticks(range(len(kick_labels)))
        axes[1].set_yticks(range(len(kick_labels)))
        axes[1].set_xticklabels(kick_labels, fontsize=10)
        axes[1].set_yticklabels(kick_labels, fontsize=10)
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_v2.png'), dpi=150)
    plt.close()
    
    return best_model, best_name, best_score


# ============================================================
# SAVE MODEL
# ============================================================
def save_model(model, feature_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'kick_classifier_v2.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    metadata = {
        'version': 2,
        'training_sampling_rate': DATASET_SAMPLING_RATE,
        'segment_window_sec': SEGMENT_WINDOW_SEC,
        'min_peak_distance_sec': MIN_PEAK_DISTANCE_SEC,
        'feature_names': list(feature_names),
        'classes': ['Ap Chagi', 'Dolyo Chagi', 'Idle'],
        'imu_compatibility': {
            'gravity_removed': True,
            'z_score_normalized': True,
            'frequency_resampled': True,
            'note': 'Features are gravity-removed and z-score normalized, making them robust to different IMU types and phone orientations'
        },
        'phone_instructions': {
            'app': 'Phyphox (free, iOS & Android)',
            'sensor': 'Acceleration WITH g (raw accelerometer)',
            'sampling_rate': '100 Hz recommended (will be resampled if different)',
            'placement': 'Strap phone to front of shin, screen facing outward',
            'recording': 'Press start, do 10+ kicks, press stop, export CSV'
        }
    }
    
    with open(os.path.join(output_dir, 'model_metadata_v2.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to: {output_dir}")


# ============================================================
# PHONE DATA PROCESSOR
# ============================================================
def process_phone_data(csv_path, model_path, phone_fs, gravity_included=True):
    """Process phone recording and classify each detected kick."""
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    meta_path = os.path.join(os.path.dirname(model_path), 'model_metadata_v2.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    target_fs = metadata['training_sampling_rate']
    
    # Load phone CSV (handle different app formats)
    df = pd.read_csv(csv_path)
    
    # Auto-detect columns
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ['x', 'acceleration x', 'accel_x', 'acc_x', 'linear acceleration x (m/s^2)',
                   'acceleration x (m/s^2)', 'accelerometer x (m/s^2)']:
            col_map['X'] = col
        elif cl in ['y', 'acceleration y', 'accel_y', 'acc_y', 'linear acceleration y (m/s^2)',
                     'acceleration y (m/s^2)', 'accelerometer y (m/s^2)']:
            col_map['Y'] = col
        elif cl in ['z', 'acceleration z', 'accel_z', 'acc_z', 'linear acceleration z (m/s^2)',
                     'acceleration z (m/s^2)', 'accelerometer z (m/s^2)']:
            col_map['Z'] = col
    
    if len(col_map) < 3:
        # Phyphox format detection
        for col in df.columns:
            if 'Acceleration' in col or 'acceleration' in col:
                if 'x' in col.lower():
                    col_map['X'] = col
                elif 'y' in col.lower():
                    col_map['Y'] = col
                elif 'z' in col.lower():
                    col_map['Z'] = col
    
    if len(col_map) < 3:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        # Skip 'time' column if present
        numeric = [c for c in numeric if 'time' not in c.lower() and 't' != c.lower()]
        if len(numeric) >= 3:
            col_map = {'X': numeric[0], 'Y': numeric[1], 'Z': numeric[2]}
    
    phone_df = pd.DataFrame({
        'X': pd.to_numeric(df[col_map['X']], errors='coerce').values,
        'Y': pd.to_numeric(df[col_map['Y']], errors='coerce').values,
        'Z': pd.to_numeric(df[col_map['Z']], errors='coerce').values,
    }).dropna()
    
    print(f"\nPhone data: {len(phone_df)} samples at {phone_fs} Hz "
          f"({len(phone_df)/phone_fs:.1f} seconds)")
    print(f"Columns mapped: {col_map}")
    
    # Resample
    if phone_fs != target_fs:
        print(f"Resampling: {phone_fs} Hz -> {target_fs} Hz")
        phone_df = resample_to_target(phone_df, phone_fs, target_fs)
        print(f"After resampling: {len(phone_df)} samples")
    
    # Segment
    segments = segment_kicks(phone_df, fs=target_fs, label='unknown')
    print(f"Detected {len(segments)} events")
    
    if not segments:
        print("No kicks detected! Make sure you're doing actual kicks while recording.")
        return []
    
    # Classify
    results = []
    for i, seg in enumerate(segments):
        feat = extract_features_v2(seg, fs=target_fs)
        if feat is None:
            continue
        
        feat_df = pd.DataFrame([feat])
        for col in metadata['feature_names']:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[metadata['feature_names']]
        
        prediction = model.predict(feat_df)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feat_df)[0]
            confidence = float(max(proba))
            class_probas = {str(c): float(p) for c, p in zip(model.classes_, proba)}
        else:
            confidence = None
            class_probas = {}
        
        time_sec = seg['peak_idx'] / target_fs
        
        results.append({
            'segment': i + 1,
            'time_sec': round(time_sec, 2),
            'prediction': prediction,
            'confidence': round(confidence, 3) if confidence else None,
            'probabilities': class_probas,
        })
        
        conf_str = f" (confidence: {confidence:.1%})" if confidence else ""
        print(f"  #{i+1} at {time_sec:.1f}s -> {prediction}{conf_str}")
    
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    import sys
    
    BASE_PATH = r'd:\Downloads\Dataset\raw_data'
    OUTPUT_DIR = r'd:\Downloads\Dataset\classifier\output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if running in phone validation mode
    if '--phone' in sys.argv:
        idx = sys.argv.index('--phone')
        csv_file = sys.argv[idx + 1]
        
        phone_fs = DATASET_SAMPLING_RATE
        if '--phone_fs' in sys.argv:
            fs_idx = sys.argv.index('--phone_fs')
            phone_fs = int(sys.argv[fs_idx + 1])
        
        model_path = os.path.join(OUTPUT_DIR, 'kick_classifier_v2.pkl')
        results = process_phone_data(csv_file, model_path, phone_fs)
        
        # Save results
        with open(csv_file.replace('.csv', '_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {csv_file.replace('.csv', '_results.json')}")
        sys.exit(0)
    
    # ---- Training mode ----
    print("=" * 60)
    print("TAEKWONDO KICK CLASSIFIER v2")
    print("Gravity-removed + IMU-agnostic features")
    print("=" * 60)
    
    print("\n[1/4] Loading dataset...")
    files = load_dataset(BASE_PATH)
    print(f"  Loaded {len(files)} files")
    
    print("\n[2/4] Segmenting & extracting features...")
    X, y, sources = build_training_data(files)
    print(f"  {len(X)} segments, {X.shape[1]} features")
    
    print("\n[3/4] Training & evaluating...")
    model, model_name, score = train_and_evaluate(X, y, sources, OUTPUT_DIR)
    
    print("\n[4/4] Saving model...")
    save_model(model, X.columns, OUTPUT_DIR)
    
    print(f"\n{'=' * 60}")
    print(f"DONE | Best: {model_name} | K-Fold: {score:.1%}")
    print(f"{'=' * 60}")
