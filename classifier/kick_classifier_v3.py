"""
Kick Classifier v3 - Handles pauses between kicks
===================================================
Problem in v2: Phone data has clear pauses between kicks.
The fixed-window segmentation captured pauses, making kicks look like "idle".

Fix: Use BURST detection instead of peak detection.
- Find periods where dynamic acceleration is HIGH (actual kick motion)
- Extract features from ONLY the burst, not surrounding pauses
- This matches training data where kicks are continuous
"""

import pandas as pd
import numpy as np
import os, json, pickle, warnings, sys
warnings.filterwarnings('ignore')

from scipy.signal import find_peaks, resample
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_FS = 100
MIN_BURST_SAMPLES = 15  # minimum burst length (~0.15s at 100Hz)


def load_dataset(base_path):
    files = {}
    for name in ['Ap Kiri', 'Diam', 'Doylo Kiri']:
        p = os.path.join(base_path, f'{name}.csv')
        if os.path.exists(p):
            df = pd.read_csv(p); df.columns = ['X','Y','Z']; files[name] = df
    for name in ['Ap Kanan', 'Doylo Kanan']:
        p = os.path.join(base_path, f'{name}.xlsx')
        if os.path.exists(p):
            df = pd.read_excel(p, sheet_name=0); df.columns = ['X','Y','Z']; files[name] = df
    return files


def resample_to_target(data, src_fs, tgt_fs):
    if src_fs == tgt_fs: return data
    n = int(len(data) * tgt_fs / src_fs)
    if isinstance(data, pd.DataFrame):
        r = pd.DataFrame()
        for c in data.columns: r[c] = resample(data[c].values, n)
        return r
    return resample(data, n)


def segment_bursts(df, fs=DATASET_FS, label=None):
    """
    BURST-based segmentation: find contiguous regions of high dynamic acceleration.
    This naturally skips pauses between kicks.
    """
    x_dyn = df['X'].values - np.mean(df['X'].values)
    y_dyn = df['Y'].values - np.mean(df['Y'].values)
    z_dyn = df['Z'].values - np.mean(df['Z'].values)
    dyn_mag = np.sqrt(x_dyn**2 + y_dyn**2 + z_dyn**2)
    
    # Smooth to avoid tiny gaps breaking a burst
    smooth_w = max(3, int(0.03 * fs))
    dyn_smooth = pd.Series(dyn_mag).rolling(window=smooth_w, center=True).mean()
    dyn_smooth = dyn_smooth.fillna(pd.Series(dyn_mag)).values
    
    # Threshold: above this = "active kick motion"
    threshold = np.percentile(dyn_smooth, 60)  # top 40% of dynamic acceleration
    
    # Find contiguous regions above threshold
    above = dyn_smooth > threshold
    segments = []
    in_burst = False
    start = 0
    
    for i in range(len(above)):
        if above[i] and not in_burst:
            start = i
            in_burst = True
        elif not above[i] and in_burst:
            if (i - start) >= MIN_BURST_SAMPLES:
                segments.append({
                    'X': df['X'].values[start:i],
                    'Y': df['Y'].values[start:i],
                    'Z': df['Z'].values[start:i],
                    'mag': np.sqrt(df['X'].values[start:i]**2 + df['Y'].values[start:i]**2 + df['Z'].values[start:i]**2),
                    'peak_idx': start + np.argmax(dyn_mag[start:i]),
                    'label': label, 'fs': fs,
                })
            in_burst = False
    
    # Handle last burst
    if in_burst and (len(above) - start) >= MIN_BURST_SAMPLES:
        segments.append({
            'X': df['X'].values[start:],
            'Y': df['Y'].values[start:],
            'Z': df['Z'].values[start:],
            'mag': np.sqrt(df['X'].values[start:]**2 + df['Y'].values[start:]**2 + df['Z'].values[start:]**2),
            'peak_idx': start + np.argmax(dyn_mag[start:]),
            'label': label, 'fs': fs,
        })
    
    return segments


def extract_features(seg, fs=DATASET_FS):
    """Extract IMU-agnostic features from a burst segment."""
    x, y, z = seg['X'], seg['Y'], seg['Z']
    if len(x) < 8: return None
    
    # Remove gravity
    x_d = x - np.mean(x); y_d = y - np.mean(y); z_d = z - np.mean(z)
    mag_d = np.sqrt(x_d**2 + y_d**2 + z_d**2)
    
    features = {}
    for name, data in [('x', x_d), ('y', y_d), ('z', z_d), ('mag', mag_d)]:
        std = np.std(data)
        dn = (data - np.mean(data)) / std if std > 1e-6 else data - np.mean(data)
        
        features[f'{name}_std'] = float(std)
        features[f'{name}_rms'] = float(np.sqrt(np.mean(data**2)))
        features[f'{name}_peak'] = float(np.max(np.abs(data)))
        features[f'{name}_range'] = float(np.ptp(data))
        features[f'{name}_skew'] = float(skew(dn))
        features[f'{name}_kurtosis'] = float(kurtosis(dn))
        features[f'{name}_zcr'] = float(np.sum(np.diff(np.sign(dn)) != 0) / (len(data)/fs))
        
        if len(data) > 2:
            deriv = np.diff(data) * fs
            features[f'{name}_jerk_rms'] = float(np.sqrt(np.mean(deriv**2)))
            features[f'{name}_jerk_peak'] = float(np.max(np.abs(deriv)))
        
        n = len(dn)
        if n > 10:
            yf = np.abs(fft(dn))[:n//2]; xf = fftfreq(n, 1.0/fs)[:n//2]
            if len(yf) > 1:
                features[f'{name}_dom_freq'] = float(xf[np.argmax(yf[1:])+1])
                te = np.sum(yf**2) + 1e-10
                features[f'{name}_low_freq_ratio'] = float(np.sum(yf[xf<5]**2)/te)
                features[f'{name}_spectral_centroid'] = float(np.sum(xf*yf)/(np.sum(yf)+1e-10))
    
    # Ratios
    te = np.std(x_d)**2 + np.std(y_d)**2 + np.std(z_d)**2 + 1e-10
    features['x_energy_share'] = float(np.std(x_d)**2/te)
    features['y_energy_share'] = float(np.std(y_d)**2/te)
    features['z_energy_share'] = float(np.std(z_d)**2/te)
    
    try:
        features['corr_xy'] = float(np.corrcoef(x_d,y_d)[0,1])
        features['corr_xz'] = float(np.corrcoef(x_d,z_d)[0,1])
        features['corr_yz'] = float(np.corrcoef(y_d,z_d)[0,1])
    except:
        features['corr_xy']=features['corr_xz']=features['corr_yz']=0.0
    
    features['time_to_peak_ratio'] = float(np.argmax(mag_d)/max(len(mag_d)-1,1))
    mid = len(mag_d)//2
    fhe = np.sum(mag_d[:mid]**2)+1e-10; she = np.sum(mag_d[mid:]**2)+1e-10
    features['energy_symmetry'] = float(fhe/(fhe+she))
    features['burst_duration'] = float(len(x)/fs)  # how long this burst lasted (seconds)
    
    for k,v in features.items():
        if np.isnan(v) or np.isinf(v): features[k] = 0.0
    return features


def build_training_data(files, fs=DATASET_FS):
    label_map = {
        'Ap Kanan':'Ap Chagi', 'Ap Kiri':'Ap Chagi',
        'Doylo Kanan':'Dolyo Chagi', 'Doylo Kiri':'Dolyo Chagi',
        'Diam':'Idle',
    }
    X_list, y_list, src_list = [], [], []
    for name, df in files.items():
        label = label_map[name]
        for seg in segment_bursts(df, fs=fs, label=label):
            feat = extract_features(seg, fs=fs)
            if feat:
                X_list.append(feat); y_list.append(label); src_list.append(name)
    return pd.DataFrame(X_list), np.array(y_list), np.array(src_list)


def train_model(X, y, sources, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X = X.fillna(0)
    
    print(f"\nDataset: {len(X)} segments, {X.shape[1]} features")
    for c, n in zip(*np.unique(y, return_counts=True)): print(f"  {c}: {n}")
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, 
            min_samples_split=5, min_samples_leaf=3, random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42, 
                   class_weight='balanced', probability=True),
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model, best_score, best_name = None, 0, ""
    
    print("\n--- 5-Fold CV ---")
    for name, clf in classifiers.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
        print(f"  {name}: {scores.mean():.1%} (+/-{scores.std():.1%})")
        if scores.mean() > best_score:
            best_score = scores.mean(); best_model = pipe; best_name = name
    
    # Left/Right test
    print("\n--- Left/Right Generalization ---")
    left = np.isin(sources, ['Ap Kiri','Doylo Kiri'])
    right = np.isin(sources, ['Ap Kanan','Doylo Kanan'])
    idle = sources == 'Diam'
    
    p1 = Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, class_weight='balanced'))])
    p1.fit(X[right|idle], y[right|idle])
    print(f"  Train RIGHT+Idle -> Test LEFT: {p1.score(X[left], y[left]):.1%}")
    
    p2 = Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, class_weight='balanced'))])
    p2.fit(X[left|idle], y[left|idle])
    print(f"  Train LEFT+Idle -> Test RIGHT: {p2.score(X[right], y[right]):.1%}")
    
    # Train final
    print(f"\nFinal model: {best_name} ({best_score:.1%})")
    best_model.fit(X, y)
    
    with open(os.path.join(output_dir, 'kick_classifier_v3.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    meta = {'version':3, 'training_fs': DATASET_FS, 'feature_names': list(X.columns),
            'classes': list(np.unique(y))}
    with open(os.path.join(output_dir, 'model_metadata_v3.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    return best_model, best_name, best_score


def test_phone_data(recordings, model, metadata, phone_fs, output_dir):
    """Test phone recordings against the model trained on Indonesian data only."""
    print(f"\n{'='*60}")
    print("PHONE VALIDATION (test only — model never saw this data)")
    print(f"{'='*60}")
    print(f"Phone: Samsung SM-M346B2, ICM42632M IMU, {phone_fs} Hz")
    print(f"Model trained on: Indonesian dataset, MPU6050, ~{DATASET_FS} Hz")
    
    all_results = {}
    
    for rec_name, (csv_path, expected_label) in recordings.items():
        print(f"\n--- {rec_name} (expected: {expected_label}) ---")
        
        df = pd.read_csv(csv_path)
        phone_df = pd.DataFrame({
            'X': df['Acceleration x (m/s^2)'].values,
            'Y': df['Acceleration y (m/s^2)'].values,
            'Z': df['Acceleration z (m/s^2)'].values,
        })
        
        print(f"  Raw: {len(phone_df)} samples at {phone_fs} Hz ({len(phone_df)/phone_fs:.1f}s)")
        
        # Resample to training frequency
        if phone_fs != DATASET_FS:
            phone_df = resample_to_target(phone_df, phone_fs, DATASET_FS)
            print(f"  Resampled to {DATASET_FS} Hz: {len(phone_df)} samples")
        
        # Segment using burst detection (handles pauses)
        segments = segment_bursts(phone_df, fs=DATASET_FS, label='unknown')
        print(f"  Detected {len(segments)} activity bursts")
        
        results = []
        for i, seg in enumerate(segments):
            feat = extract_features(seg, fs=DATASET_FS)
            if not feat: continue
            
            feat_df = pd.DataFrame([feat])
            for col in metadata['feature_names']:
                if col not in feat_df.columns: feat_df[col] = 0.0
            feat_df = feat_df[metadata['feature_names']].fillna(0)
            
            pred = model.predict(feat_df)[0]
            proba = model.predict_proba(feat_df)[0] if hasattr(model, 'predict_proba') else None
            conf = float(max(proba)) if proba is not None else None
            
            t = seg['peak_idx'] / DATASET_FS
            dur = len(seg['X']) / DATASET_FS
            
            results.append({'segment': i+1, 'time': round(t,1), 'duration': round(dur,2),
                           'prediction': pred, 'confidence': round(conf,3) if conf else None})
            
            print(f"    #{i+1} at {t:.1f}s ({dur:.2f}s) -> {pred} ({conf:.0%})" if conf else 
                  f"    #{i+1} at {t:.1f}s ({dur:.2f}s) -> {pred}")
        
        if results:
            preds = [r['prediction'] for r in results]
            correct = sum(1 for p in preds if p == expected_label)
            print(f"  Result: {correct}/{len(preds)} correct ({correct/len(preds):.0%})")
            for cls in sorted(set(preds) | {expected_label}):
                n = preds.count(cls)
                tag = " <-- EXPECTED" if cls == expected_label else ""
                print(f"    {cls}: {n}/{len(preds)}{tag}")
        
        all_results[rec_name] = results
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    total_c, total_n = 0, 0
    for rec_name, (_, expected_label) in recordings.items():
        if all_results[rec_name]:
            preds = [r['prediction'] for r in all_results[rec_name]]
            c = sum(1 for p in preds if p == expected_label)
            total_c += c; total_n += len(preds)
            print(f"  {rec_name}: {c}/{len(preds)} ({c/len(preds):.0%})")
    if total_n:
        print(f"\n  OVERALL: {total_c}/{total_n} ({total_c/total_n:.1%})")
    
    return all_results


if __name__ == '__main__':
    BASE = r'd:\Downloads\Dataset\raw_data'
    OUT = r'd:\Downloads\Dataset\classifier\output'
    
    print("="*60)
    print("KICK CLASSIFIER v3 — Burst-based segmentation")
    print("="*60)
    
    # TRAIN on Indonesian data only
    print("\n[1/3] Loading Indonesian dataset...")
    files = load_dataset(BASE)
    
    print("\n[2/3] Training (Indonesian data only)...")
    X, y, sources = build_training_data(files)
    model, model_name, score = train_model(X, y, sources, OUT)
    
    # Load metadata
    with open(os.path.join(OUT, 'model_metadata_v3.json')) as f:
        meta = json.load(f)
    
    # TEST on phone data
    print("\n[3/3] Testing on YOUR phone recordings...")
    rec_base = r'd:\Downloads\Dataset\my_records'
    recordings = {
        'Front Kick (Ap Chagi)': (
            os.path.join(rec_base, 'frontkick(around 12-13 not perfect kicks)', 'Raw Data.csv'),
            'Ap Chagi'),
        'Roundhouse (Dolyo Chagi)': (
            os.path.join(rec_base, 'roundhouse(around 10 or 11 not perfect kicks)', 'Raw Data.csv'),
            'Dolyo Chagi'),
        'Standing/Walking (Idle)': (
            os.path.join(rec_base, 'standing_and_walking', 'Raw Data.csv'),
            'Idle'),
    }
    
    results = test_phone_data(recordings, model, meta, phone_fs=500, output_dir=OUT)
    
    with open(os.path.join(OUT, 'phone_validation_v3.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
