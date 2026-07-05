"""
Kick Classifier v6 - Sliding Window + Majority Vote
=====================================================
ROOT CAUSE: Training data = continuous rapid kicks (no pauses).
Every window in training data is ALL kick motion.
Phone data = kick, pause, kick, pause.
Peak-centered windows include calm portions -> model sees them as idle.

SOLUTION: Use SLIDING WINDOW across entire recording.
- Slide 0.8s window across the data
- Classify each window
- Windows during a kick will be classified as that kick type
- Windows during pauses will be classified as Idle
- Report: what % of active windows are each class
"""

import pandas as pd
import numpy as np
import os, json, pickle, warnings
warnings.filterwarnings('ignore')

from scipy.signal import resample
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_FS = 100
WINDOW_SEC = 0.8  # seconds per window
STEP_SEC = 0.2    # step size (overlap = window - step)
CLIP_RANGE = 19.6


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


def sliding_windows(df, fs, window_sec=WINDOW_SEC, step_sec=STEP_SEC, label=None):
    """Slide a fixed-size window across the data."""
    win_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    segments = []
    i = 0
    while i + win_samples <= len(df):
        seg = {
            'X': df['X'].values[i:i+win_samples],
            'Y': df['Y'].values[i:i+win_samples],
            'Z': df['Z'].values[i:i+win_samples],
            'start_idx': i,
            'label': label,
            'fs': fs,
        }
        seg['mag'] = np.sqrt(seg['X']**2 + seg['Y']**2 + seg['Z']**2)
        segments.append(seg)
        i += step_samples
    
    return segments


def extract_features(seg, fs=DATASET_FS):
    """Features on gravity-removed data, per-axis + magnitude."""
    x, y, z = seg['X'], seg['Y'], seg['Z']
    mag = seg['mag']
    if len(x) < 10: return None
    
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
        
        n = len(dn)
        if n > 10:
            yf = np.abs(fft(dn))[:n//2]
            xf = fftfreq(n, 1.0/fs)[:n//2]
            if len(yf) > 1:
                features[name + '_dom_freq'] = float(xf[np.argmax(yf[1:])+1])
                te = np.sum(yf**2) + 1e-10
                features[name + '_low_freq_r'] = float(np.sum(yf[xf<5]**2)/te)
                features[name + '_spec_cent'] = float(np.sum(xf*yf)/(np.sum(yf)+1e-10))
    
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
        for seg in sliding_windows(df, fs=fs, label=label):
            feat = extract_features(seg, fs=fs)
            if feat:
                X_l.append(feat); y_l.append(label); s_l.append(name)
    return pd.DataFrame(X_l), np.array(y_l), np.array(s_l)


def train(X, y, sources, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X = X.fillna(0)
    
    print('\nTraining: %d windows, %d features' % (len(X), X.shape[1]))
    for c, n in zip(*np.unique(y, return_counts=True)):
        print('  %s: %d windows' % (c, n))
    
    clfs = {
        'RF': RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=5,
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
    left = np.isin(sources, ['Ap Kiri','Doylo Kiri'])
    right = np.isin(sources, ['Ap Kanan','Doylo Kanan'])
    idle = sources == 'Diam'
    
    print('\n--- Left/Right ---')
    p1 = Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(
        n_estimators=500, max_depth=15, random_state=42, class_weight='balanced'))])
    p1.fit(X[right|idle], y[right|idle])
    print('  Train RIGHT -> Test LEFT: %.1f%%' % (p1.score(X[left], y[left])*100))
    
    p2 = Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(
        n_estimators=500, max_depth=15, random_state=42, class_weight='balanced'))])
    p2.fit(X[left|idle], y[left|idle])
    print('  Train LEFT -> Test RIGHT: %.1f%%' % (p2.score(X[right], y[right])*100))
    
    print('\nFinal: %s (%.1f%%)' % (bname, bscore*100))
    best.fit(X, y)
    
    with open(os.path.join(out_dir, 'kick_classifier_v6.pkl'), 'wb') as f:
        pickle.dump(best, f)
    meta = {'version': 6, 'fs': DATASET_FS, 'features': list(X.columns),
            'classes': list(np.unique(y)), 'window_sec': WINDOW_SEC, 'step_sec': STEP_SEC}
    with open(os.path.join(out_dir, 'model_metadata_v6.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    return best, meta


ACTIVITY_THRESHOLD_STD = 1.5  # std of dynamic mag to be considered "active"


def test_phone(recordings, model, meta, phone_fs, out_dir):
    print('\n' + '='*60)
    print('PHONE VALIDATION — Sliding Window + Active-Only Vote')
    print('='*60)
    
    fig, axes = plt.subplots(len(recordings), 1, figsize=(16, 5*len(recordings)))
    if len(recordings) == 1: axes = [axes]
    
    all_results = {}
    
    for ax_idx, (rec_name, (csv_path, expected)) in enumerate(recordings.items()):
        print('\n--- %s (expected: %s) ---' % (rec_name, expected))
        
        df = pd.read_csv(csv_path)
        phone_df = pd.DataFrame({
            'X': np.clip(df.iloc[:, 1].values, -CLIP_RANGE, CLIP_RANGE),
            'Y': np.clip(df.iloc[:, 2].values, -CLIP_RANGE, CLIP_RANGE),
            'Z': np.clip(df.iloc[:, 3].values, -CLIP_RANGE, CLIP_RANGE),
        })
        
        # Resample
        if phone_fs != DATASET_FS:
            n_new = int(len(phone_df) * DATASET_FS / phone_fs)
            resampled = pd.DataFrame()
            for c in phone_df.columns:
                resampled[c] = resample(phone_df[c].values, n_new)
            phone_df = resampled
        
        print('  Resampled: %d samples at %d Hz (%.1fs)' % (
            len(phone_df), DATASET_FS, len(phone_df)/DATASET_FS))
        
        # Sliding window classification
        windows = sliding_windows(phone_df, fs=DATASET_FS)
        print('  Windows: %d' % len(windows))
        
        times = []
        predictions = []
        confidences = []
        
        # Pre-compute dynamic activity level for each window for filtering
        global_mag = np.sqrt(phone_df['X']**2 + phone_df['Y']**2 + phone_df['Z']**2).values
        global_gravity = np.median(global_mag)
        global_dyn = np.abs(global_mag - global_gravity)
        global_dyn_std = np.std(global_dyn)
        active_threshold = np.mean(global_dyn) + ACTIVITY_THRESHOLD_STD * global_dyn_std
        
        for win in windows:
            feat = extract_features(win, fs=DATASET_FS)
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
                conf = 0.5
            
            # Mark window as "active" if it has significant motion
            start = win['start_idx']
            end = start + int(WINDOW_SEC * DATASET_FS)
            win_dyn = global_dyn[start:min(end, len(global_dyn))]
            is_active = np.max(win_dyn) > active_threshold
            
            t = win['start_idx'] / DATASET_FS
            times.append(t)
            predictions.append(pred)
            confidences.append(conf)
            win['is_active'] = is_active
            win['pred'] = pred
            win['conf'] = conf
            win['time'] = t
        
        # All-window distribution
        pred_counts = {}
        for p in predictions:
            pred_counts[p] = pred_counts.get(p, 0) + 1
        total = len(predictions)
        
        print('\n  All windows distribution:')
        for cls in sorted(pred_counts.keys()):
            n = pred_counts[cls]
            tag = ' <-- EXPECTED' if cls == expected else ''
            print('    %s: %d/%d (%.0f%%)%s' % (cls, n, total, n/total*100, tag))
        
        # Active-only vote (windows with significant motion)
        active_wins = [w for w in windows if w.get('is_active', False) and 'pred' in w]
        active_preds = [w['pred'] for w in active_wins]
        active_confs = [w['conf'] for w in active_wins]
        
        active_counts = {}
        for p in active_preds:
            active_counts[p] = active_counts.get(p, 0) + 1
        
        print('\n  ACTIVE windows only (%d/%d windows):' % (len(active_wins), total))
        for cls in sorted(active_counts.keys()):
            n = active_counts[cls]
            tag = ' <-- EXPECTED' if cls == expected else ''
            print('    %s: %d/%d (%.0f%%)%s' % (cls, n, len(active_wins), n/len(active_wins)*100 if active_wins else 0, tag))
        
        # Confidence-weighted vote on active windows
        class_scores = {}
        for pred, conf in zip(active_preds, active_confs):
            class_scores[pred] = class_scores.get(pred, 0.0) + conf
        
        majority_all = max(pred_counts, key=pred_counts.get) if pred_counts else 'Unknown'
        majority_active = max(active_counts, key=active_counts.get) if active_counts else 'Unknown'
        majority_weighted = max(class_scores, key=class_scores.get) if class_scores else 'Unknown'
        
        print('\n  VOTES:')
        print('    All windows majority:              %s  -> %s' % (majority_all, 'PASS' if majority_all==expected else 'FAIL'))
        print('    Active windows majority:           %s  -> %s' % (majority_active, 'PASS' if majority_active==expected else 'FAIL'))
        print('    Active confidence-weighted vote:   %s  -> %s' % (majority_weighted, 'PASS' if majority_weighted==expected else 'FAIL'))
        
        all_results[rec_name] = {
            'distribution': pred_counts,
            'active_distribution': active_counts,
            'majority_all': majority_all,
            'majority_active': majority_active,
            'majority_weighted': majority_weighted,
            'expected': expected,
            'match_all': majority_all == expected,
            'match_active': majority_active == expected,
            'match_weighted': majority_weighted == expected,
            'total_windows': total,
            'active_windows': len(active_wins),
        }
        
        # Plot timeline
        ax = axes[ax_idx]
        color_map = {'Ap Chagi': 'green', 'Dolyo Chagi': 'blue', 'Idle': 'gray'}
        colors = [color_map.get(p, 'red') for p in predictions]
        ax.scatter(times, [1]*len(times), c=colors, s=20, alpha=0.7)
        
        # Also plot magnitude for reference
        mag = np.sqrt(phone_df['X']**2 + phone_df['Y']**2 + phone_df['Z']**2).values
        t_sig = np.arange(len(mag)) / DATASET_FS
        ax2 = ax.twinx()
        ax2.plot(t_sig, mag, color='black', alpha=0.3, linewidth=0.5)
        ax2.set_ylabel('|a| m/s^2', fontsize=9, color='gray')
        
        ax.set_title('%s | Expected: %s | Active Vote: %s | Weighted Vote: %s' % (
            rec_name, expected, majority_active, majority_weighted),
            fontweight='bold', fontsize=11,
            color='green' if majority_active==expected else 'red')
        ax.set_yticks([])
        ax.set_xlabel('Time (s)')
        
        # Legend
        from matplotlib.patches import Patch
        legend_items = [Patch(color=c, label=l) for l, c in color_map.items()]
        ax.legend(handles=legend_items, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'phone_timeline_v6.png'), dpi=150)
    plt.close()
    
    # Summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    c_all = sum(1 for r in all_results.values() if r['match_all'])
    c_act = sum(1 for r in all_results.values() if r['match_active'])
    c_wgt = sum(1 for r in all_results.values() if r['match_weighted'])
    n = len(all_results)
    print('  All-window majority:     %d/%d correct' % (c_all, n))
    print('  Active-window majority:  %d/%d correct' % (c_act, n))
    print('  Confidence-weighted:     %d/%d correct' % (c_wgt, n))
    print()
    for rec_name, r in all_results.items():
        sym = 'PASS' if r['match_active'] else 'FAIL'
        print('  [%s] %s -> active=%s, weighted=%s (expected=%s)' % (
            sym, rec_name, r['majority_active'], r['majority_weighted'], r['expected']))
    
    with open(os.path.join(out_dir, 'phone_validation_v6.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


if __name__ == '__main__':
    BASE = r'd:\Downloads\Dataset\raw_data'
    OUT = r'd:\Downloads\Dataset\classifier\output'
    
    print('='*60)
    print('KICK CLASSIFIER v6 - Sliding Window + Majority Vote')
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
