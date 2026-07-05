"""
Kick Classifier v7 - Gravity-Aligned + Canonical Frame
=======================================================
Root cause: Training sensor gravity = (X:-2.4, Y:-13.7, Z:-2.4) → gravity on -Y
            Phone sensor gravity    = (X:-0.7, Y:+9.5,  Z:+2.1) → gravity on +Y

The phone is rotated ~180° around X-axis relative to training sensor.
Fix: detect gravity direction per-recording, flip axes to canonical frame
where gravity always points on -Y. Then features are comparable.

Phone is 500 Hz (confirmed from time column).
Training is ~100 Hz (estimated from kick intervals).
"""

import pandas as pd
import numpy as np
import os, json, pickle, warnings
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
PHONE_FS   = 500
CLIP_RANGE = 19.6  # MPU6050 +-2g range


# ============================================================
# GRAVITY ALIGNMENT
# ============================================================
def estimate_gravity_vector(df, fs, rest_sec=2.0):
    """Estimate gravity from the calmest part of the signal (lowest variance window)."""
    win = int(rest_sec * fs)
    x, y, z = df['X'].values, df['Y'].values, df['Z'].values
    
    best_var, best_start = np.inf, 0
    step = max(1, win // 4)
    for i in range(0, max(1, len(df) - win), step):
        v = np.var(x[i:i+win]) + np.var(y[i:i+win]) + np.var(z[i:i+win])
        if v < best_var:
            best_var = v
            best_start = i
    
    return {
        'X': float(np.mean(x[best_start:best_start+win])),
        'Y': float(np.mean(y[best_start:best_start+win])),
        'Z': float(np.mean(z[best_start:best_start+win])),
    }


def align_to_canonical(df, grav):
    """
    Flip axes so gravity always points toward -Y (matching training sensor orientation).
    This makes per-axis features comparable across different phone mountings.
    
    Strategy: find which axis has the largest gravity component, 
    then remap and flip to put that on -Y.
    For small rotations (same axis, different sign): just flip sign.
    """
    out = df.copy()
    
    # Target: gravity on -Y (like training data Ap Kanan: Y≈-13.6)
    # Current gravity vector
    gx, gy, gz = grav['X'], grav['Y'], grav['Z']
    
    # Find the dominant gravity axis
    abs_g = [abs(gx), abs(gy), abs(gz)]
    dominant = np.argmax(abs_g)  # 0=X, 1=Y, 2=Z
    dominant_val = [gx, gy, gz][dominant]
    
    # If gravity is on Y but positive (+Y instead of -Y): flip Y and Z
    if dominant == 1:
        if dominant_val > 0:
            # Rotate 180° around X: Y -> -Y, Z -> -Z
            out['Y'] = -df['Y']
            out['Z'] = -df['Z']
        # else: already on -Y, no change needed
    
    # If gravity is on Z: rotate to put it on Y
    elif dominant == 2:
        if dominant_val > 0:
            # Gravity on +Z: rotate so it goes to -Y
            out['Y'] = -df['Z']
            out['Z'] = df['Y']
        else:
            # Gravity on -Z: rotate so it goes to -Y
            out['Y'] = df['Z']
            out['Z'] = -df['Y']
    
    # If gravity is on X: rotate to put it on Y  
    elif dominant == 0:
        if dominant_val > 0:
            out['Y'] = -df['X']
            out['X'] = df['Y']
        else:
            out['Y'] = df['X']
            out['X'] = -df['Y']
    
    return out


# ============================================================
# DATA LOADING
# ============================================================
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


def resample_df(df, src_fs, tgt_fs):
    if src_fs == tgt_fs: return df.copy()
    n = int(len(df) * tgt_fs / src_fs)
    out = pd.DataFrame()
    for c in df.columns:
        out[c] = resample(df[c].values, n)
    return out


# ============================================================
# SEGMENTATION — peak-based, frequency-aware
# ============================================================
def segment_peaks(df, fs=DATASET_FS, label=None):
    mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).values
    gravity_est = np.median(mag)
    dyn = mag - gravity_est

    sw = max(3, int(0.04 * fs))
    dyn_s = pd.Series(np.abs(dyn)).rolling(window=sw, center=True).mean()
    dyn_s = dyn_s.fillna(pd.Series(np.abs(dyn))).values

    threshold = np.mean(dyn_s) + 0.8 * np.std(dyn_s)
    min_dist = int(0.5 * fs)

    peaks, _ = find_peaks(dyn_s, height=threshold, distance=min_dist,
                          prominence=np.std(dyn_s) * 0.3)

    half_win = int(0.4 * fs)
    segments = []
    for p in peaks:
        s = max(0, p - half_win)
        e = min(len(df), p + half_win)
        seg_dyn = dyn_s[s:e]
        if np.max(seg_dyn) < threshold * 0.6:
            continue
        segments.append({
            'X': df['X'].values[s:e], 'Y': df['Y'].values[s:e],
            'Z': df['Z'].values[s:e],
            'mag': mag[s:e], 'peak_idx': p, 'label': label, 'fs': fs,
        })
    return segments


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(seg, fs=DATASET_FS):
    x, y, z = seg['X'], seg['Y'], seg['Z']
    if len(x) < 10: return None

    # Gravity removal
    xd = x - np.mean(x); yd = y - np.mean(y); zd = z - np.mean(z)
    mag_d = np.sqrt(xd**2 + yd**2 + zd**2)

    feats = {}

    for nm, data in [('x', xd), ('y', yd), ('z', zd), ('mag', mag_d)]:
        std = np.std(data)
        dn = (data - np.mean(data)) / std if std > 1e-6 else data * 0.0

        feats[nm+'_std']      = float(std)
        feats[nm+'_rms']      = float(np.sqrt(np.mean(data**2)))
        feats[nm+'_peak']     = float(np.max(np.abs(data)))
        feats[nm+'_range']    = float(np.ptp(data))
        feats[nm+'_skew']     = float(skew(dn))
        feats[nm+'_kurtosis'] = float(kurtosis(dn))
        feats[nm+'_zcr']      = float(np.sum(np.diff(np.sign(dn)) != 0) / (len(data)/fs))

        if len(data) > 2:
            jk = np.diff(data) * fs
            feats[nm+'_jerk_rms']  = float(np.sqrt(np.mean(jk**2)))
            feats[nm+'_jerk_peak'] = float(np.max(np.abs(jk)))

        n = len(dn)
        if n > 10:
            yf = np.abs(fft(dn))[:n//2]
            xf = fftfreq(n, 1.0/fs)[:n//2]
            if len(yf) > 1:
                feats[nm+'_dom_freq']  = float(xf[np.argmax(yf[1:])+1])
                te = np.sum(yf**2) + 1e-10
                feats[nm+'_lfr']  = float(np.sum(yf[xf<5]**2)/te)
                feats[nm+'_scent']= float(np.sum(xf*yf)/(np.sum(yf)+1e-10))

    te = np.std(xd)**2 + np.std(yd)**2 + np.std(zd)**2 + 1e-10
    feats['x_share'] = float(np.std(xd)**2/te)
    feats['y_share'] = float(np.std(yd)**2/te)
    feats['z_share'] = float(np.std(zd)**2/te)

    try:
        feats['corr_xy'] = float(np.corrcoef(xd, yd)[0,1])
        feats['corr_xz'] = float(np.corrcoef(xd, zd)[0,1])
        feats['corr_yz'] = float(np.corrcoef(yd, zd)[0,1])
    except:
        feats['corr_xy'] = feats['corr_xz'] = feats['corr_yz'] = 0.0

    feats['time_to_peak'] = float(np.argmax(mag_d)/max(len(mag_d)-1, 1))
    mid = len(mag_d)//2
    e1 = np.sum(mag_d[:mid]**2)+1e-10; e2 = np.sum(mag_d[mid:]**2)+1e-10
    feats['energy_sym'] = float(e1/(e1+e2))

    for k, v in feats.items():
        if np.isnan(v) or np.isinf(v): feats[k] = 0.0
    return feats


# ============================================================
# BUILD TRAINING DATA
# ============================================================
def build_data(files, fs=DATASET_FS):
    label_map = {
        'Ap Kanan': 'Ap Chagi', 'Ap Kiri': 'Ap Chagi',
        'Doylo Kanan': 'Dolyo Chagi', 'Doylo Kiri': 'Dolyo Chagi',
        'Diam': 'Idle',
    }
    X_l, y_l, s_l = [], [], []
    for name, df in files.items():
        label = label_map[name]
        # Align training data to canonical frame too
        grav = estimate_gravity_vector(df, fs)
        df_aligned = align_to_canonical(df, grav)
        print('  %s: gravity=(%.1f,%.1f,%.1f) -> aligned gravity=(%.1f,%.1f,%.1f)' % (
            name, grav['X'], grav['Y'], grav['Z'],
            np.mean(df_aligned['X']), np.mean(df_aligned['Y']), np.mean(df_aligned['Z'])))
        for seg in segment_peaks(df_aligned, fs=fs, label=label):
            feat = extract_features(seg, fs=fs)
            if feat:
                X_l.append(feat); y_l.append(label); s_l.append(name)
    return pd.DataFrame(X_l), np.array(y_l), np.array(s_l)


# ============================================================
# TRAIN
# ============================================================
def train(X, y, sources, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X = X.fillna(0)

    print('\nTraining: %d segments, %d features' % (len(X), X.shape[1]))
    for c, n in zip(*np.unique(y, return_counts=True)):
        print('  %s: %d' % (c, n))

    clfs = {
        'RF':  RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_split=4,
                                       min_samples_leaf=2, random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42,
                   class_weight='balanced', probability=True),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best, bscore, bname = None, 0, ''

    print('\n--- 5-Fold CV ---')
    for nm, clf in clfs.items():
        p = Pipeline([('s', StandardScaler()), ('c', clf)])
        sc = cross_val_score(p, X, y, cv=skf, scoring='accuracy')
        print('  %s: %.1f%% (+/-%.1f%%)' % (nm, sc.mean()*100, sc.std()*100))
        if sc.mean() > bscore: bscore = sc.mean(); best = p; bname = nm

    # Left/Right generalization
    left  = np.isin(sources, ['Ap Kiri','Doylo Kiri'])
    right = np.isin(sources, ['Ap Kanan','Doylo Kanan'])
    idle  = sources == 'Diam'

    print('\n--- Left/Right Generalization ---')
    for tr_mask, tr_name, te_mask, te_name in [
        (right|idle, 'RIGHT+Idle', left,  'LEFT'),
        (left|idle,  'LEFT+Idle',  right, 'RIGHT'),
    ]:
        p = Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(
            n_estimators=500, max_depth=12, random_state=42, class_weight='balanced'))])
        p.fit(X[tr_mask], y[tr_mask])
        acc = p.score(X[te_mask], y[te_mask])
        print('  Train %s -> Test %s: %.1f%%' % (tr_name, te_name, acc*100))

    print('\nFinal: %s (%.1f%%)' % (bname, bscore*100))
    best.fit(X, y)

    with open(os.path.join(out_dir, 'kick_classifier_v7.pkl'), 'wb') as f:
        pickle.dump(best, f)
    meta = {'version': 7, 'fs': DATASET_FS, 'features': list(X.columns),
            'classes': list(np.unique(y)), 'clip_range': CLIP_RANGE}
    with open(os.path.join(out_dir, 'model_metadata_v7.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    if hasattr(best.named_steps['c'], 'feature_importances_'):
        imp = best.named_steps['c'].feature_importances_
        top = np.argsort(imp)[::-1][:10]
        print('\nTop 10 features:')
        for i, idx in enumerate(top):
            print('  %d. %s: %.4f' % (i+1, X.columns[idx], imp[idx]))

    return best, meta


# ============================================================
# TEST PHONE DATA
# ============================================================
def test_phone(recordings, model, meta, phone_fs, out_dir):
    print('\n' + '='*60)
    print('PHONE VALIDATION (gravity-aligned)')
    print('='*60)

    fig, axes = plt.subplots(len(recordings), 1, figsize=(16, 5*len(recordings)))
    if len(recordings) == 1: axes = [axes]

    all_results = {}

    for ax_idx, (rec_name, (csv_path, expected)) in enumerate(recordings.items()):
        print('\n--- %s (expected: %s) ---' % (rec_name, expected))

        df = pd.read_csv(csv_path)
        phone_df = pd.DataFrame({
            'X': np.clip(df.iloc[:,1].values, -CLIP_RANGE, CLIP_RANGE),
            'Y': np.clip(df.iloc[:,2].values, -CLIP_RANGE, CLIP_RANGE),
            'Z': np.clip(df.iloc[:,3].values, -CLIP_RANGE, CLIP_RANGE),
        })

        # Step 1: Clip to training sensor range
        # Step 2: Resample to training frequency
        phone_df = resample_df(phone_df, phone_fs, DATASET_FS)
        print('  Resampled: %d Hz -> %d Hz (%d samples = %.1fs)' % (
            phone_fs, DATASET_FS, len(phone_df), len(phone_df)/DATASET_FS))

        # Step 3: Estimate gravity and align to canonical frame
        grav = estimate_gravity_vector(phone_df, DATASET_FS)
        phone_df = align_to_canonical(phone_df, grav)
        print('  Gravity before: (%.1f, %.1f, %.1f)' % (grav['X'], grav['Y'], grav['Z']))
        print('  Gravity after alignment: (%.1f, %.1f, %.1f)' % (
            np.mean(phone_df['X']), np.mean(phone_df['Y']), np.mean(phone_df['Z'])))

        # Step 4: Segment
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
                class_probs = {str(c): round(float(p),2) for c, p in zip(model.classes_, proba)}

            t = seg['peak_idx'] / DATASET_FS
            marker = 'OK' if pred == expected else 'MISS'
            print('    #%d at %.1fs -> %s (%.0f%%) %s [%s]' % (
                i+1, t, pred, (conf or 0)*100, class_probs, marker))
            results.append({'seg': i+1, 'time': round(t,1), 'pred': pred,
                            'conf': round(conf,3) if conf else None})

        if results:
            preds = [r['pred'] for r in results]
            correct = sum(1 for p in preds if p == expected)
            print('\n  RESULT: %d/%d correct (%.0f%%)' % (
                correct, len(preds), correct/len(preds)*100))
            counts = {}
            for p in preds: counts[p] = counts.get(p, 0) + 1
            for cls in sorted(counts):
                tag = ' <-- EXPECTED' if cls == expected else ''
                print('    %s: %d%s' % (cls, counts[cls], tag))

        all_results[rec_name] = results

        # Plot
        ax = axes[ax_idx]
        color_map = {'Ap Chagi':'green', 'Dolyo Chagi':'royalblue', 'Idle':'gray'}
        mag_full = np.sqrt(phone_df['X']**2 + phone_df['Y']**2 + phone_df['Z']**2).values
        t_full = np.arange(len(mag_full)) / DATASET_FS
        ax.plot(t_full, mag_full, color='black', alpha=0.4, linewidth=0.6)
        for r in results:
            c = color_map.get(r['pred'], 'red')
            ax.axvline(x=r['time'], color=c, alpha=0.7, linewidth=2)

        summary_txt = ''
        if results:
            preds = [r['pred'] for r in results]
            correct = sum(1 for p in preds if p == expected)
            summary_txt = ' | %d/%d correct' % (correct, len(preds))
        ax.set_title('%s | Expected: %s%s' % (rec_name, expected, summary_txt),
                     fontweight='bold', fontsize=11,
                     color='green' if (results and correct==len(results)) else 'darkred')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('|a| m/s^2')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=l) for l,c in color_map.items()],
                  loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'phone_timeline_v7.png'), dpi=150)
    plt.close()

    # Summary
    print('\n' + '='*60 + '\nSUMMARY\n' + '='*60)
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

    with open(os.path.join(out_dir, 'phone_validation_v7.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results


if __name__ == '__main__':
    BASE = r'd:\Downloads\Dataset\raw_data'
    OUT  = r'd:\Downloads\Dataset\classifier\output'

    print('='*60)
    print('KICK CLASSIFIER v7 - Gravity-Aligned Canonical Frame')
    print('='*60)

    print('\n[1/3] Loading & aligning training data...')
    files = load_dataset(BASE)

    print('\n[2/3] Building features & training...')
    X, y, src = build_data(files)
    model, meta = train(X, y, src, OUT)

    print('\n[3/3] Testing on phone recordings (gravity-aligned)...')
    rec_base = r'd:\Downloads\Dataset\my_records'
    recs = {
        'Front Kick': (rec_base + '/frontkick(around 12-13 not perfect kicks)/Raw Data.csv', 'Ap Chagi'),
        'Roundhouse':  (rec_base + '/roundhouse(around 10 or 11 not perfect kicks)/Raw Data.csv', 'Dolyo Chagi'),
        'Standing':    (rec_base + '/standing_and_walking/Raw Data.csv', 'Idle'),
    }
    test_phone(recs, model, meta, phone_fs=PHONE_FS, out_dir=OUT)
