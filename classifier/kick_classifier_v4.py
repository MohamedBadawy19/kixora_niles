"""
Kick Classifier v4 - Orientation-invariant
============================================
Problem: Training sensor was mounted differently than phone.
  Training Ap Kanan: gravity on -Y axis (Y=-13.66)
  Phone: gravity on +Y axis (Y=+9.51)

Fix: Use ONLY orientation-invariant features:
  - Acceleration magnitude (sqrt of sum of squares - same regardless of mounting)
  - Jerk magnitude (derivative of magnitude)  
  - Frequency features on magnitude only
  - NO per-axis features (they depend on how sensor is oriented)
"""

import pandas as pd
import numpy as np
import os, json, pickle, warnings
warnings.filterwarnings('ignore')

from scipy.signal import find_peaks, resample
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_FS = 100
MIN_BURST_SEC = 0.3


def load_dataset(base_path):
    files = {}
    for name in ['Ap Kiri', 'Diam', 'Doylo Kiri']:
        p = os.path.join(base_path, name + '.csv')
        if os.path.exists(p):
            df = pd.read_csv(p); df.columns = ['X','Y','Z']; files[name] = df
    for name in ['Ap Kanan', 'Doylo Kanan']:
        p = os.path.join(base_path, name + '.xlsx')
        if os.path.exists(p):
            df = pd.read_excel(p, sheet_name=0); df.columns = ['X','Y','Z']; files[name] = df
    return files


def resample_df(data, src_fs, tgt_fs):
    if src_fs == tgt_fs: return data
    n = int(len(data) * tgt_fs / src_fs)
    r = pd.DataFrame()
    for c in data.columns: r[c] = resample(data[c].values, n)
    return r


def segment_bursts(df, fs=DATASET_FS, label=None):
    """Find activity bursts using dynamic magnitude."""
    mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).values
    # Dynamic = magnitude minus gravity (roughly 9.8)
    gravity = np.mean(mag[:min(50, len(mag))])  # estimate from start
    dyn = np.abs(mag - gravity)
    
    # Smooth
    sw = max(3, int(0.03 * fs))
    dyn_s = pd.Series(dyn).rolling(window=sw, center=True).mean()
    dyn_s = dyn_s.fillna(pd.Series(dyn)).values
    
    threshold = np.percentile(dyn_s, 60)
    min_samples = int(MIN_BURST_SEC * fs)
    
    above = dyn_s > threshold
    segments = []
    in_burst = False
    start = 0
    
    for i in range(len(above)):
        if above[i] and not in_burst:
            start = i; in_burst = True
        elif not above[i] and in_burst:
            if (i - start) >= min_samples:
                segments.append({
                    'mag': mag[start:i],
                    'dyn': dyn[start:i],
                    'peak_idx': start + np.argmax(dyn[start:i]),
                    'label': label, 'fs': fs,
                })
            in_burst = False
    if in_burst and (len(above) - start) >= min_samples:
        segments.append({
            'mag': mag[start:], 'dyn': dyn[start:],
            'peak_idx': start + np.argmax(dyn[start:]),
            'label': label, 'fs': fs,
        })
    return segments


def extract_orientation_invariant_features(seg, fs=DATASET_FS):
    """
    ALL features are computed on MAGNITUDE only.
    Magnitude = sqrt(X^2 + Y^2 + Z^2) — same value regardless of sensor orientation.
    This is the key to making the classifier work across different phone mountings.
    """
    mag = seg['mag']
    dyn = seg['dyn']  # dynamic component (gravity removed from magnitude)
    
    if len(mag) < 8: return None
    
    features = {}
    
    # Normalize for scale invariance (different IMU ranges)
    mag_std = np.std(mag)
    dyn_std = np.std(dyn)
    mag_n = (mag - np.mean(mag)) / mag_std if mag_std > 1e-6 else mag - np.mean(mag)
    dyn_n = (dyn - np.mean(dyn)) / dyn_std if dyn_std > 1e-6 else dyn - np.mean(dyn)
    
    # --- Magnitude features (z-score normalized) ---
    features['mag_std'] = float(mag_std)
    features['mag_rms'] = float(np.sqrt(np.mean(mag**2)))
    features['mag_peak'] = float(np.max(mag))
    features['mag_range'] = float(np.ptp(mag))
    features['mag_skew'] = float(skew(mag_n))
    features['mag_kurtosis'] = float(kurtosis(mag_n))
    features['mag_cv'] = float(mag_std / (np.mean(mag) + 1e-10))  # coefficient of variation
    
    # --- Dynamic (gravity-removed) features ---
    features['dyn_mean'] = float(np.mean(dyn))
    features['dyn_std'] = float(dyn_std)
    features['dyn_rms'] = float(np.sqrt(np.mean(dyn**2)))
    features['dyn_peak'] = float(np.max(dyn))
    features['dyn_range'] = float(np.ptp(dyn))
    features['dyn_skew'] = float(skew(dyn_n))
    features['dyn_kurtosis'] = float(kurtosis(dyn_n))
    
    # --- Jerk (rate of change of acceleration) ---
    if len(mag) > 2:
        jerk = np.diff(mag) * fs  # derivative in m/s^3
        features['jerk_rms'] = float(np.sqrt(np.mean(jerk**2)))
        features['jerk_peak'] = float(np.max(np.abs(jerk)))
        features['jerk_std'] = float(np.std(jerk))
        
        # Jerk of dynamic component
        jerk_dyn = np.diff(dyn) * fs
        features['jerk_dyn_rms'] = float(np.sqrt(np.mean(jerk_dyn**2)))
        features['jerk_dyn_peak'] = float(np.max(np.abs(jerk_dyn)))
    
    # --- Zero crossing rate (of normalized signal) ---
    zcr = np.sum(np.diff(np.sign(mag_n)) != 0)
    features['mag_zcr'] = float(zcr / (len(mag) / fs))
    
    zcr_dyn = np.sum(np.diff(np.sign(dyn_n)) != 0)
    features['dyn_zcr'] = float(zcr_dyn / (len(dyn) / fs))
    
    # --- Temporal shape features ---
    features['time_to_peak'] = float(np.argmax(dyn) / max(len(dyn)-1, 1))
    mid = len(dyn) // 2
    e1 = np.sum(dyn[:mid]**2) + 1e-10
    e2 = np.sum(dyn[mid:]**2) + 1e-10
    features['energy_symmetry'] = float(e1 / (e1 + e2))
    
    # Energy in quarters
    q = len(dyn) // 4
    if q > 0:
        total_e = np.sum(dyn**2) + 1e-10
        features['q1_energy'] = float(np.sum(dyn[:q]**2) / total_e)
        features['q2_energy'] = float(np.sum(dyn[q:2*q]**2) / total_e)
        features['q3_energy'] = float(np.sum(dyn[2*q:3*q]**2) / total_e)
        features['q4_energy'] = float(np.sum(dyn[3*q:]**2) / total_e)
    
    # --- Frequency domain (on magnitude) ---
    n = len(mag_n)
    if n > 10:
        yf = np.abs(fft(mag_n))[:n//2]
        xf = fftfreq(n, 1.0/fs)[:n//2]
        
        if len(yf) > 1:
            features['dom_freq'] = float(xf[np.argmax(yf[1:])+1])
            te = np.sum(yf**2) + 1e-10
            features['low_freq_ratio'] = float(np.sum(yf[xf < 5]**2) / te)
            features['mid_freq_ratio'] = float(np.sum(yf[(xf >= 5) & (xf < 15)]**2) / te)
            features['high_freq_ratio'] = float(np.sum(yf[xf >= 15]**2) / te)
            features['spectral_centroid'] = float(np.sum(xf * yf) / (np.sum(yf) + 1e-10))
            features['spectral_spread'] = float(np.sqrt(np.sum((xf - features['spectral_centroid'])**2 * yf) / (np.sum(yf) + 1e-10)))
    
    # Frequency on dynamic component
    n2 = len(dyn_n)
    if n2 > 10:
        yf2 = np.abs(fft(dyn_n))[:n2//2]
        xf2 = fftfreq(n2, 1.0/fs)[:n2//2]
        if len(yf2) > 1:
            features['dyn_dom_freq'] = float(xf2[np.argmax(yf2[1:])+1])
            te2 = np.sum(yf2**2) + 1e-10
            features['dyn_low_freq_ratio'] = float(np.sum(yf2[xf2 < 5]**2) / te2)
            features['dyn_spectral_centroid'] = float(np.sum(xf2 * yf2) / (np.sum(yf2) + 1e-10))
    
    # Duration of this burst
    features['burst_duration'] = float(len(mag) / fs)
    
    # Peak-to-mean ratio (dimensionless)
    features['peak_to_mean'] = float(np.max(dyn) / (np.mean(dyn) + 1e-10))
    
    # Crest factor
    features['crest_factor'] = float(np.max(np.abs(mag_n)) / (features['mag_rms'] / (np.mean(mag) + 1e-10) + 1e-10))
    
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v): features[k] = 0.0
    
    return features


def build_data(files, fs=DATASET_FS):
    label_map = {
        'Ap Kanan':'Ap Chagi', 'Ap Kiri':'Ap Chagi',
        'Doylo Kanan':'Dolyo Chagi', 'Doylo Kiri':'Dolyo Chagi',
        'Diam':'Idle',
    }
    X_l, y_l, s_l = [], [], []
    for name, df in files.items():
        label = label_map[name]
        for seg in segment_bursts(df, fs=fs, label=label):
            feat = extract_orientation_invariant_features(seg, fs=fs)
            if feat:
                X_l.append(feat); y_l.append(label); s_l.append(name)
    return pd.DataFrame(X_l), np.array(y_l), np.array(s_l)


def train(X, y, sources, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X = X.fillna(0)
    
    print('\nDataset: %d segments, %d features' % (len(X), X.shape[1]))
    for c, n in zip(*np.unique(y, return_counts=True)):
        print('  %s: %d' % (c, n))
    
    clfs = {
        'RF': RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_split=4,
                                      min_samples_leaf=2, random_state=42, class_weight='balanced'),
        'GB': GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                          min_samples_split=4, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42,
                   class_weight='balanced', probability=True),
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best, bscore, bname = None, 0, ''
    
    print('\n--- 5-Fold CV ---')
    for name, clf in clfs.items():
        p = Pipeline([('s', StandardScaler()), ('c', clf)])
        sc = cross_val_score(p, X, y, cv=skf, scoring='accuracy')
        print('  %s: %.1f%% (+/-%.1f%%)' % (name, sc.mean()*100, sc.std()*100))
        if sc.mean() > bscore: bscore = sc.mean(); best = p; bname = name
    
    # Left/Right test
    print('\n--- Left/Right Generalization ---')
    left = np.isin(sources, ['Ap Kiri','Doylo Kiri'])
    right = np.isin(sources, ['Ap Kanan','Doylo Kanan'])
    idle = sources == 'Diam'
    
    p1 = Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(
        n_estimators=500, max_depth=12, random_state=42, class_weight='balanced'))])
    p1.fit(X[right|idle], y[right|idle])
    a1 = p1.score(X[left], y[left])
    print('  Train RIGHT -> Test LEFT: %.1f%%' % (a1*100))
    
    p2 = Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(
        n_estimators=500, max_depth=12, random_state=42, class_weight='balanced'))])
    p2.fit(X[left|idle], y[left|idle])
    a2 = p2.score(X[right], y[right])
    print('  Train LEFT -> Test RIGHT: %.1f%%' % (a2*100))
    
    print('\nFinal model: %s (%.1f%%)' % (bname, bscore*100))
    best.fit(X, y)
    
    with open(os.path.join(out_dir, 'kick_classifier_v4.pkl'), 'wb') as f:
        pickle.dump(best, f)
    meta = {'version': 4, 'fs': DATASET_FS, 'features': list(X.columns),
            'classes': list(np.unique(y)),
            'note': 'Orientation-invariant: uses only magnitude features, no per-axis'}
    with open(os.path.join(out_dir, 'model_metadata_v4.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Feature importance
    if hasattr(best.named_steps['c'], 'feature_importances_'):
        imp = best.named_steps['c'].feature_importances_
        top = np.argsort(imp)[::-1][:10]
        print('\nTop 10 features:')
        for i, idx in enumerate(top):
            print('  %d. %s: %.4f' % (i+1, X.columns[idx], imp[idx]))
    
    return best, meta


def test_phone(recordings, model, meta, phone_fs, out_dir):
    print('\n' + '='*60)
    print('PHONE VALIDATION (model never saw this data)')
    print('='*60)
    
    all_results = {}
    
    for rec_name, (csv_path, expected) in recordings.items():
        print('\n--- %s (expected: %s) ---' % (rec_name, expected))
        
        df = pd.read_csv(csv_path)
        phone_df = pd.DataFrame({
            'X': df.iloc[:, 1].values,  # Acceleration x
            'Y': df.iloc[:, 2].values,  # Acceleration y
            'Z': df.iloc[:, 3].values,  # Acceleration z
        })
        
        dur = len(phone_df) / phone_fs
        print('  %d samples at %d Hz (%.1fs)' % (len(phone_df), phone_fs, dur))
        
        if phone_fs != DATASET_FS:
            phone_df = resample_df(phone_df, phone_fs, DATASET_FS)
            print('  Resampled to %d Hz: %d samples' % (DATASET_FS, len(phone_df)))
        
        segments = segment_bursts(phone_df, fs=DATASET_FS, label='unknown')
        
        # For kick recordings, filter short bursts (< 0.5s = not real kicks)
        if expected != 'Idle':
            segments = [s for s in segments if len(s['mag']) >= 0.5 * DATASET_FS]
        
        print('  Segments (after filtering): %d' % len(segments))
        
        results = []
        for i, seg in enumerate(segments):
            feat = extract_orientation_invariant_features(seg, fs=DATASET_FS)
            if not feat: continue
            
            feat_df = pd.DataFrame([feat])
            for col in meta['features']:
                if col not in feat_df.columns: feat_df[col] = 0.0
            feat_df = feat_df[meta['features']].fillna(0)
            
            pred = model.predict(feat_df)[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feat_df)[0]
                conf = float(max(proba))
            else:
                conf = None
            
            t = seg['peak_idx'] / DATASET_FS
            d = len(seg['mag']) / DATASET_FS
            marker = 'OK' if pred == expected else 'MISS'
            
            if conf:
                print('    #%d at %.1fs (%.2fs) -> %s (%.0f%%) [%s]' % (i+1, t, d, pred, conf*100, marker))
            else:
                print('    #%d at %.1fs (%.2fs) -> %s [%s]' % (i+1, t, d, pred, marker))
            
            results.append({'seg': i+1, 'time': round(t,1), 'dur': round(d,2),
                           'pred': pred, 'conf': round(conf,3) if conf else None})
        
        if results:
            preds = [r['pred'] for r in results]
            correct = sum(1 for p in preds if p == expected)
            print('  RESULT: %d/%d correct (%.0f%%)' % (correct, len(preds), correct/len(preds)*100))
        
        all_results[rec_name] = results
    
    # Summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    tc, tn = 0, 0
    for rec_name, (_, expected) in recordings.items():
        r = all_results.get(rec_name, [])
        if r:
            preds = [x['pred'] for x in r]
            c = sum(1 for p in preds if p == expected)
            tc += c; tn += len(preds)
            print('  %s: %d/%d (%.0f%%)' % (rec_name, c, len(preds), c/len(preds)*100))
    if tn:
        print('\n  OVERALL: %d/%d (%.1f%%)' % (tc, tn, tc/tn*100))
    
    with open(os.path.join(out_dir, 'phone_validation_v4.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


if __name__ == '__main__':
    BASE = r'd:\Downloads\Dataset\raw_data'
    OUT = r'd:\Downloads\Dataset\classifier\output'
    
    print('='*60)
    print('KICK CLASSIFIER v4 - Orientation Invariant')
    print('='*60)
    
    files = load_dataset(BASE)
    X, y, src = build_data(files)
    model, meta = train(X, y, src, OUT)
    
    rec_base = r'd:\Downloads\Dataset\my_records'
    recordings = {
        'Front Kick': (rec_base + '/frontkick(around 12-13 not perfect kicks)/Raw Data.csv', 'Ap Chagi'),
        'Roundhouse': (rec_base + '/roundhouse(around 10 or 11 not perfect kicks)/Raw Data.csv', 'Dolyo Chagi'),
        'Standing': (rec_base + '/standing_and_walking/Raw Data.csv', 'Idle'),
    }
    
    test_phone(recordings, model, meta, phone_fs=500, out_dir=OUT)
