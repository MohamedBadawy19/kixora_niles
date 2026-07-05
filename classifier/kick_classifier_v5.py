"""
Kick Classifier v5 - Sensor range alignment
=============================================
Root cause found: training sensor (MPU6050) clips at +/-19.6 m/s^2.
Phone sensor (ICM42632M) reads up to +/-60 m/s^2.
Clipped peaks = flat top. Unclipped peaks = sharp spike.
This changes ALL features (kurtosis, peak values, etc.)

Fix: CLIP phone data to +/-19.6 BEFORE feature extraction.
This simulates the same sensor range. Legitimate preprocessing step.
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
CLIP_RANGE = 19.6  # Match MPU6050 range


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


def clip_to_sensor_range(df, clip_val=CLIP_RANGE):
    """Clip data to match training sensor range."""
    clipped = df.copy()
    for col in ['X', 'Y', 'Z']:
        clipped[col] = np.clip(clipped[col].values, -clip_val, clip_val)
    return clipped


def resample_df(data, src_fs, tgt_fs):
    if src_fs == tgt_fs: return data
    n = int(len(data) * tgt_fs / src_fs)
    r = pd.DataFrame()
    for c in data.columns: r[c] = resample(data[c].values, n)
    return r


def segment_peaks(df, fs=DATASET_FS, label=None):
    """Peak-based segmentation with adaptive window sizing."""
    mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).values
    
    # Remove gravity from magnitude for peak detection
    gravity_est = np.median(mag)
    dyn = np.abs(mag - gravity_est)
    
    sw = max(3, int(0.05 * fs))
    dyn_s = pd.Series(dyn).rolling(window=sw, center=True).mean()
    dyn_s = dyn_s.fillna(pd.Series(dyn)).values
    
    # Use a higher threshold to only catch real kicks
    threshold = np.mean(dyn_s) + 1.0 * np.std(dyn_s)
    min_dist = int(0.5 * fs)  # At least 0.5s between kicks
    
    peaks, props = find_peaks(dyn_s, height=threshold, distance=min_dist,
                               prominence=np.std(dyn_s) * 0.3)
    
    # Extract window around each peak
    # Use a TIGHT window focused on the kick itself
    half_win = int(0.4 * fs)  # +/- 0.4s
    segments = []
    
    for p in peaks:
        start = max(0, p - half_win)
        end = min(len(df), p + half_win)
        
        seg_mag = mag[start:end]
        # Only keep if this segment has significant dynamic content
        seg_dyn = np.abs(seg_mag - gravity_est)
        if np.max(seg_dyn) < threshold * 0.5:
            continue
        
        segments.append({
            'X': df['X'].values[start:end],
            'Y': df['Y'].values[start:end],
            'Z': df['Z'].values[start:end],
            'mag': seg_mag,
            'peak_idx': p,
            'label': label,
            'fs': fs,
        })
    
    return segments


def extract_features(seg, fs=DATASET_FS):
    """Per-axis + magnitude features with gravity removal."""
    x, y, z = seg['X'], seg['Y'], seg['Z']
    mag = seg['mag']
    if len(x) < 10: return None
    
    # Gravity removal
    x_d = x - np.mean(x)
    y_d = y - np.mean(y)
    z_d = z - np.mean(z)
    mag_d = np.sqrt(x_d**2 + y_d**2 + z_d**2)
    
    features = {}
    
    for name, data in [('x', x_d), ('y', y_d), ('z', z_d), ('mag', mag_d)]:
        std = np.std(data)
        dn = (data - np.mean(data)) / std if std > 1e-6 else data * 0
        
        features[name + '_std'] = float(std)
        features[name + '_rms'] = float(np.sqrt(np.mean(data**2)))
        features[name + '_peak'] = float(np.max(np.abs(data)))
        features[name + '_range'] = float(np.ptp(data))
        features[name + '_skew'] = float(skew(dn))
        features[name + '_kurtosis'] = float(kurtosis(dn))
        features[name + '_zcr'] = float(np.sum(np.diff(np.sign(dn)) != 0) / (len(data)/fs))
        
        if len(data) > 2:
            jerk = np.diff(data) * fs
            features[name + '_jerk_rms'] = float(np.sqrt(np.mean(jerk**2)))
            features[name + '_jerk_peak'] = float(np.max(np.abs(jerk)))
        
        n = len(dn)
        if n > 10:
            yf = np.abs(fft(dn))[:n//2]
            xf = fftfreq(n, 1.0/fs)[:n//2]
            if len(yf) > 1:
                features[name + '_dom_freq'] = float(xf[np.argmax(yf[1:])+1])
                te = np.sum(yf**2) + 1e-10
                features[name + '_low_freq_r'] = float(np.sum(yf[xf<5]**2)/te)
                features[name + '_spec_cent'] = float(np.sum(xf*yf)/(np.sum(yf)+1e-10))
    
    # Ratios
    te = np.std(x_d)**2 + np.std(y_d)**2 + np.std(z_d)**2 + 1e-10
    features['x_share'] = float(np.std(x_d)**2 / te)
    features['y_share'] = float(np.std(y_d)**2 / te)
    features['z_share'] = float(np.std(z_d)**2 / te)
    
    try:
        features['corr_xy'] = float(np.corrcoef(x_d, y_d)[0,1])
        features['corr_xz'] = float(np.corrcoef(x_d, z_d)[0,1])
        features['corr_yz'] = float(np.corrcoef(y_d, z_d)[0,1])
    except:
        features['corr_xy'] = features['corr_xz'] = features['corr_yz'] = 0.0
    
    features['time_to_peak'] = float(np.argmax(mag_d) / max(len(mag_d)-1, 1))
    mid = len(mag_d) // 2
    e1 = np.sum(mag_d[:mid]**2) + 1e-10
    e2 = np.sum(mag_d[mid:]**2) + 1e-10
    features['energy_sym'] = float(e1 / (e1 + e2))
    features['duration'] = float(len(x) / fs)
    
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v): features[k] = 0.0
    return features


def build_data(files, fs=DATASET_FS):
    label_map = {
        'Ap Kanan': 'Ap Chagi', 'Ap Kiri': 'Ap Chagi',
        'Doylo Kanan': 'Dolyo Chagi', 'Doylo Kiri': 'Dolyo Chagi',
        'Diam': 'Idle',
    }
    X_l, y_l, s_l = [], [], []
    for name, df in files.items():
        label = label_map[name]
        for seg in segment_peaks(df, fs=fs, label=label):
            feat = extract_features(seg, fs=fs)
            if feat:
                X_l.append(feat); y_l.append(label); s_l.append(name)
    return pd.DataFrame(X_l), np.array(y_l), np.array(s_l)


def train(X, y, sources, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X = X.fillna(0)
    
    print('\nTraining data: %d segments, %d features' % (len(X), X.shape[1]))
    for c, n in zip(*np.unique(y, return_counts=True)):
        print('  %s: %d' % (c, n))
    
    clfs = {
        'RF': RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_split=4,
                                      min_samples_leaf=2, random_state=42, class_weight='balanced'),
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
    
    # Left/Right
    print('\n--- Left/Right Generalization ---')
    left = np.isin(sources, ['Ap Kiri','Doylo Kiri'])
    right = np.isin(sources, ['Ap Kanan','Doylo Kanan'])
    idle = sources == 'Diam'
    
    for train_name, train_m, test_name, test_m in [
        ('RIGHT+Idle', right|idle, 'LEFT', left),
        ('LEFT+Idle', left|idle, 'RIGHT', right),
    ]:
        p = Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(
            n_estimators=500, max_depth=12, random_state=42, class_weight='balanced'))])
        p.fit(X[train_m], y[train_m])
        acc = p.score(X[test_m], y[test_m])
        print('  Train %s -> Test %s: %.1f%%' % (train_name, test_name, acc*100))
    
    print('\nFinal: %s (%.1f%%)' % (bname, bscore*100))
    best.fit(X, y)
    
    with open(os.path.join(out_dir, 'kick_classifier_v5.pkl'), 'wb') as f:
        pickle.dump(best, f)
    meta = {'version': 5, 'fs': DATASET_FS, 'features': list(X.columns),
            'classes': list(np.unique(y)), 'clip_range': CLIP_RANGE}
    with open(os.path.join(out_dir, 'model_metadata_v5.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    if hasattr(best.named_steps['c'], 'feature_importances_'):
        imp = best.named_steps['c'].feature_importances_
        top = np.argsort(imp)[::-1][:10]
        print('\nTop 10 features:')
        for i, idx in enumerate(top):
            print('  %d. %s: %.4f' % (i+1, X.columns[idx], imp[idx]))
    
    return best, meta


def test_phone(recordings, model, meta, phone_fs, out_dir):
    print('\n' + '='*60)
    print('PHONE VALIDATION')
    print('Training: Indonesian MPU6050 (~100Hz, clips at +/-19.6)')
    print('Testing: Samsung ICM42632M (500Hz, clips at +/-19.6 by us)')
    print('='*60)
    
    all_results = {}
    
    for rec_name, (csv_path, expected) in recordings.items():
        print('\n--- %s (expected: %s) ---' % (rec_name, expected))
        
        df = pd.read_csv(csv_path)
        phone_df = pd.DataFrame({
            'X': df.iloc[:, 1].values,
            'Y': df.iloc[:, 2].values,
            'Z': df.iloc[:, 3].values,
        })
        
        print('  Raw: %d samples, range: X[%.1f,%.1f] Y[%.1f,%.1f] Z[%.1f,%.1f]' % (
            len(phone_df),
            phone_df['X'].min(), phone_df['X'].max(),
            phone_df['Y'].min(), phone_df['Y'].max(),
            phone_df['Z'].min(), phone_df['Z'].max(),
        ))
        
        # Step 1: CLIP to training sensor range
        phone_df = clip_to_sensor_range(phone_df, CLIP_RANGE)
        print('  After clipping to +/-%.1f: X[%.1f,%.1f] Y[%.1f,%.1f]' % (
            CLIP_RANGE,
            phone_df['X'].min(), phone_df['X'].max(),
            phone_df['Y'].min(), phone_df['Y'].max(),
        ))
        
        # Step 2: Resample
        if phone_fs != DATASET_FS:
            phone_df = resample_df(phone_df, phone_fs, DATASET_FS)
            print('  Resampled: %d Hz -> %d Hz (%d samples)' % (phone_fs, DATASET_FS, len(phone_df)))
        
        # Step 3: Segment
        segments = segment_peaks(phone_df, fs=DATASET_FS, label='unknown')
        print('  Segments found: %d' % len(segments))
        
        results = []
        for i, seg in enumerate(segments):
            feat = extract_features(seg, fs=DATASET_FS)
            if not feat: continue
            
            feat_df = pd.DataFrame([feat])
            for col in meta['features']:
                if col not in feat_df.columns: feat_df[col] = 0.0
            feat_df = feat_df[meta['features']].fillna(0)
            
            pred = model.predict(feat_df)[0]
            conf = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feat_df)[0]
                conf = float(max(proba))
                class_probs = dict(zip(model.classes_, [round(float(p), 2) for p in proba]))
            
            t = seg['peak_idx'] / DATASET_FS
            marker = 'OK' if pred == expected else 'MISS'
            
            prob_str = str(class_probs) if conf else ''
            print('    #%d at %.1fs -> %s (%.0f%%) %s [%s]' % (
                i+1, t, pred, (conf or 0)*100, prob_str, marker))
            
            results.append({'seg': i+1, 'time': round(t,1), 'pred': pred,
                           'conf': round(conf,3) if conf else None,
                           'probs': class_probs if conf else {}})
        
        if results:
            preds = [r['pred'] for r in results]
            correct = sum(1 for p in preds if p == expected)
            print('  RESULT: %d/%d (%.0f%%)' % (correct, len(preds), correct/len(preds)*100))
            for cls in sorted(set(preds) | {expected}):
                tag = ' <-- EXPECTED' if cls == expected else ''
                print('    %s: %d%s' % (cls, preds.count(cls), tag))
        
        all_results[rec_name] = results
    
    # Summary
    print('\n' + '='*60)
    print('OVERALL SUMMARY')
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
        print('\n  TOTAL: %d/%d (%.1f%%)' % (tc, tn, tc/tn*100))
    
    with open(os.path.join(out_dir, 'phone_validation_v5.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


if __name__ == '__main__':
    BASE = r'd:\Downloads\Dataset\raw_data'
    OUT = r'd:\Downloads\Dataset\classifier\output'
    
    print('='*60)
    print('KICK CLASSIFIER v5 - Sensor Range Alignment')
    print('='*60)
    
    files = load_dataset(BASE)
    X, y, src = build_data(files)
    model, meta = train(X, y, src, OUT)
    
    rec_base = r'd:\Downloads\Dataset\my_records'
    recs = {
        'Front Kick': (rec_base + '/frontkick(around 12-13 not perfect kicks)/Raw Data.csv', 'Ap Chagi'),
        'Roundhouse': (rec_base + '/roundhouse(around 10 or 11 not perfect kicks)/Raw Data.csv', 'Dolyo Chagi'),
        'Standing': (rec_base + '/standing_and_walking/Raw Data.csv', 'Idle'),
    }
    
    test_phone(recs, model, meta, phone_fs=500, out_dir=OUT)
