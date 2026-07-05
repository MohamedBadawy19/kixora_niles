"""
Kick Classifier v8 - Adaptive Window (extracts only active kick motion)
========================================================================
Core fix: use an ADAPTIVE window around each peak.
- Expand from peak outward only while dynamic acceleration > threshold
- Stop as soon as signal returns to baseline
- This extracts JUST the kick motion, not surrounding quiet

Training data: kicks every ~0.85s -> window always stays active (no quiet)
Phone data:    kicks every ~3s   -> adaptive window stops at baseline

Both now see the same thing: the pure kick motion signal.

Also handles: Hz (resample), sensor range (clip), orientation (gravity align).
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_FS  = 100     # Hz - training data
PHONE_FS    = 500     # Hz - your Samsung
CLIP_RANGE  = 19.6    # m/s^2 - match MPU6050 +-2g
MIN_WIN_SEC = 0.1     # seconds - minimum, tight around actual kick motion
MAX_WIN_SEC = 0.8     # seconds - maximum window (avoid merging adjacent kicks)
ACTIVE_MULT = 0.5     # dynamic acceleration must be > ACTIVE_MULT * peak_dyn to stay in window


def estimate_gravity(df, fs, rest_sec=2.0):
    """Find gravity from calmest portion of recording."""
    win = max(10, int(rest_sec * fs))
    x, y, z = df['X'].values, df['Y'].values, df['Z'].values
    best_var, best_start = np.inf, 0
    step = max(1, win // 4)
    for i in range(0, max(1, len(df) - win), step):
        v = np.var(x[i:i+win]) + np.var(y[i:i+win]) + np.var(z[i:i+win])
        if v < best_var:
            best_var = v; best_start = i
    return (float(np.mean(x[best_start:best_start+win])),
            float(np.mean(y[best_start:best_start+win])),
            float(np.mean(z[best_start:best_start+win])))


def align_gravity(df, grav):
    """Flip/remap axes so gravity always points on -Y (canonical frame)."""
    out = df.copy()
    gx, gy, gz = grav
    abs_g = [abs(gx), abs(gy), abs(gz)]
    dom = int(np.argmax(abs_g))
    dom_val = [gx, gy, gz][dom]

    if dom == 1 and dom_val > 0:       # gravity on +Y  -> flip Y,Z
        out['Y'] = -df['Y']; out['Z'] = -df['Z']
    elif dom == 2 and dom_val > 0:     # gravity on +Z  -> rotate
        out['Y'] = -df['Z']; out['Z'] =  df['Y']
    elif dom == 2 and dom_val < 0:
        out['Y'] =  df['Z']; out['Z'] = -df['Y']
    elif dom == 0 and dom_val > 0:
        out['Y'] = -df['X']; out['X'] =  df['Y']
    elif dom == 0 and dom_val < 0:
        out['Y'] =  df['X']; out['X'] = -df['Y']
    # dom==1 and dom_val<0: already canonical
    return out


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
    for c in df.columns: out[c] = resample(df[c].values, n)
    return out


def compute_dynamic(df):
    """Return per-sample dynamic (gravity-removed) magnitude."""
    mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).values
    gravity = np.median(mag)
    return np.abs(mag - gravity), gravity


def adaptive_window(dyn, peak_idx, fs,
                    min_sec=MIN_WIN_SEC, max_sec=MAX_WIN_SEC,
                    active_mult=ACTIVE_MULT):
    """
    Expand window from peak_idx outward while dynamic acceleration is
    above active_mult * peak_dyn. Stops at baseline or max window.
    Returns (start, end) sample indices.
    """
    peak_dyn = dyn[peak_idx]
    threshold = max(peak_dyn * active_mult, 0.5)   # at least 0.5 m/s^2
    min_s = int(min_sec * fs)
    max_s = int(max_sec * fs)

    # Expand left
    left = peak_idx
    while left > 0 and (peak_idx - left) < max_s:
        if dyn[left - 1] < threshold:
            break
        left -= 1

    # Expand right
    right = peak_idx
    while right < len(dyn) - 1 and (right - peak_idx) < max_s:
        if dyn[right + 1] < threshold:
            break
        right += 1

    # Ensure minimum length by padding symmetrically if needed
    current = right - left
    if current < min_s:
        pad = (min_s - current) // 2
        left  = max(0, left - pad)
        right = min(len(dyn) - 1, right + pad)

    return left, right + 1


def segment_adaptive(df, fs=DATASET_FS, label=None):
    """
    Find kicks using peak detection, then extract adaptive windows
    that contain only the active kick motion.
    """
    dyn, gravity = compute_dynamic(df)

    # Smooth for peak detection only
    sw = max(3, int(0.04 * fs))
    dyn_s = pd.Series(dyn).rolling(window=sw, center=True).mean()
    dyn_s = dyn_s.fillna(pd.Series(dyn)).values

    threshold = np.mean(dyn_s) + 0.6 * np.std(dyn_s)
    min_dist  = int(0.5 * fs)

    peaks, _ = find_peaks(dyn_s, height=threshold, distance=min_dist,
                          prominence=np.std(dyn_s) * 0.25)

    segments = []
    for p in peaks:
        s, e = adaptive_window(dyn_s, p, fs)
        if (e - s) < int(MIN_WIN_SEC * fs):
            continue
        seg_dyn = dyn_s[s:e]
        if np.max(seg_dyn) < threshold * 0.5:
            continue
        segments.append({
            'X': df['X'].values[s:e],
            'Y': df['Y'].values[s:e],
            'Z': df['Z'].values[s:e],
            'mag': np.sqrt(df['X'].values[s:e]**2 +
                           df['Y'].values[s:e]**2 +
                           df['Z'].values[s:e]**2),
            'peak_idx': p,
            'win_start': s,
            'win_end': e,
            'win_sec': (e - s) / fs,
            'label': label,
            'fs': fs,
        })
    return segments


def compute_baseline(df, fs, quiet_percentile=30):
    """
    Compute magnitude baseline statistics from the quiet portions of the recording.
    Returns mean and std of the idle-level magnitude.
    """
    mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).values
    # Use a rolling window to find low-activity regions
    sw = int(0.5 * fs)
    rolling_std = pd.Series(mag).rolling(window=sw, center=True).std()
    rolling_std = rolling_std.fillna(rolling_std.median()).values
    # Quiet regions = bottom percentile of rolling std
    thr = np.percentile(rolling_std, quiet_percentile)
    quiet_mag = mag[rolling_std <= thr]
    if len(quiet_mag) < 10:
        quiet_mag = mag
    return float(np.mean(quiet_mag)), float(np.std(quiet_mag))


def extract_features(seg, fs=DATASET_FS, baseline_mean=None, baseline_std=None):
    x, y, z = seg['X'], seg['Y'], seg['Z']
    if len(x) < 6: return None

    xd = x - np.mean(x); yd = y - np.mean(y); zd = z - np.mean(z)
    mag_d = np.sqrt(xd**2 + yd**2 + zd**2)
    # Raw magnitude
    mag = seg['mag']

    feats = {}

    # === Recording-relative features ===
    # These express the kick energy RELATIVE to the recording's own baseline.
    # This removes absolute sensor differences between MPU6050 and ICM42632M.
    if baseline_mean is not None and baseline_std is not None:
        bm, bs = baseline_mean, max(baseline_std, 0.1)
        feats['rel_mag_peak']  = float((np.max(mag) - bm) / bm)      # how many times above baseline
        feats['rel_mag_std']   = float(np.std(mag) / bm)              # relative variability
        feats['rel_mag_rms']   = float(np.sqrt(np.mean(mag**2)) / bm)
        feats['rel_dyn_peak']  = float(np.max(mag_d) / bm)
        feats['rel_dyn_std']   = float(np.std(mag_d) / bm)
        snr = np.std(mag) / bs if bs > 0 else 0
        feats['snr'] = float(snr)  # kick signal / baseline noise
    for nm, data in [('x', xd), ('y', yd), ('z', zd), ('mag', mag_d)]:
        std = np.std(data)
        dn  = (data - np.mean(data)) / std if std > 1e-6 else data * 0.0

        feats[nm+'_std']      = float(std)
        feats[nm+'_rms']      = float(np.sqrt(np.mean(data**2)))
        feats[nm+'_peak']     = float(np.max(np.abs(data)))
        feats[nm+'_range']    = float(np.ptp(data))
        feats[nm+'_skew']     = float(skew(dn))
        feats[nm+'_kurtosis'] = float(kurtosis(dn))
        feats[nm+'_zcr']      = float(np.sum(np.diff(np.sign(dn))!=0) / (len(data)/fs))

        if len(data) > 2:
            jk = np.diff(data) * fs
            feats[nm+'_jrms'] = float(np.sqrt(np.mean(jk**2)))
            feats[nm+'_jpk']  = float(np.max(np.abs(jk)))

        n = len(dn)
        if n > 8:
            yf = np.abs(fft(dn))[:n//2]
            xf = fftfreq(n, 1.0/fs)[:n//2]
            if len(yf) > 1:
                feats[nm+'_df']   = float(xf[np.argmax(yf[1:])+1])
                te = np.sum(yf**2) + 1e-10
                feats[nm+'_lfr']  = float(np.sum(yf[xf<5]**2)/te)
                feats[nm+'_sc']   = float(np.sum(xf*yf)/(np.sum(yf)+1e-10))

    te = np.std(xd)**2 + np.std(yd)**2 + np.std(zd)**2 + 1e-10
    feats['xs'] = float(np.std(xd)**2/te)
    feats['ys'] = float(np.std(yd)**2/te)
    feats['zs'] = float(np.std(zd)**2/te)

    try:
        feats['cxy'] = float(np.corrcoef(xd, yd)[0,1])
        feats['cxz'] = float(np.corrcoef(xd, zd)[0,1])
        feats['cyz'] = float(np.corrcoef(yd, zd)[0,1])
    except:
        feats['cxy'] = feats['cxz'] = feats['cyz'] = 0.0

    feats['ttp'] = float(np.argmax(mag_d) / max(len(mag_d)-1, 1))
    mid = len(mag_d)//2
    e1 = np.sum(mag_d[:mid]**2)+1e-10; e2 = np.sum(mag_d[mid:]**2)+1e-10
    feats['esym'] = float(e1/(e1+e2))
    feats['wdur'] = float(seg['win_sec'])   # window duration (kick speed proxy)

    for k, v in feats.items():
        if np.isnan(v) or np.isinf(v): feats[k] = 0.0
    return feats


def prepare_df(raw_df, src_fs, tgt_fs, clip_val=CLIP_RANGE):
    """Full preprocessing pipeline: clip -> resample -> gravity align."""
    df = pd.DataFrame({
        'X': np.clip(raw_df['X'].values, -clip_val, clip_val),
        'Y': np.clip(raw_df['Y'].values, -clip_val, clip_val),
        'Z': np.clip(raw_df['Z'].values, -clip_val, clip_val),
    })
    if src_fs != tgt_fs:
        df = resample_df(df, src_fs, tgt_fs)
    grav = estimate_gravity(df, tgt_fs)
    df   = align_gravity(df, grav)
    return df, grav


def build_training(files, fs=DATASET_FS):
    label_map = {
        'Ap Kanan':'Ap Chagi',   'Ap Kiri':'Ap Chagi',
        'Doylo Kanan':'Dolyo Chagi', 'Doylo Kiri':'Dolyo Chagi',
        'Diam':'Idle',
    }
    X_l, y_l, s_l = [], [], []
    for name, df in files.items():
        label = label_map[name]
        df_p, grav = prepare_df(df, fs, fs)
        bm, bs = compute_baseline(df_p, fs)
        segs = segment_adaptive(df_p, fs=fs, label=label)
        for seg in segs:
            feat = extract_features(seg, fs=fs, baseline_mean=bm, baseline_std=bs)
            if feat:
                X_l.append(feat); y_l.append(label); s_l.append(name)
    return pd.DataFrame(X_l), np.array(y_l), np.array(s_l)


def train_model(X, y, sources, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X = X.fillna(0)

    print('\nTraining: %d segments, %d features' % (len(X), X.shape[1]))
    for c, n in zip(*np.unique(y, return_counts=True)):
        print('  %-15s %d segments' % (c+':', n))

    clfs = {
        'RF':  RandomForestClassifier(n_estimators=500, max_depth=12,
                                       min_samples_split=4, min_samples_leaf=2,
                                       random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42,
                   class_weight='balanced', probability=True),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best, bscore, bname = None, 0, ''

    print('\n--- 5-Fold CV ---')
    for nm, clf in clfs.items():
        p  = Pipeline([('s', StandardScaler()), ('c', clf)])
        sc = cross_val_score(p, X, y, cv=skf, scoring='accuracy')
        print('  %-5s %.1f%% (+/-%.1f%%)' % (nm+':', sc.mean()*100, sc.std()*100))
        if sc.mean() > bscore: bscore = sc.mean(); best = p; bname = nm

    left  = np.isin(sources, ['Ap Kiri','Doylo Kiri'])
    right = np.isin(sources, ['Ap Kanan','Doylo Kanan'])
    idle  = sources == 'Diam'

    print('\n--- Left/Right Generalization ---')
    for tr_m, tr_n, te_m, te_n in [
        (right|idle,'RIGHT+Idle',left,'LEFT'),
        (left|idle, 'LEFT+Idle', right,'RIGHT'),
    ]:
        p = Pipeline([('s', StandardScaler()),
                      ('c', RandomForestClassifier(n_estimators=500, max_depth=12,
                                                    random_state=42, class_weight='balanced'))])
        p.fit(X[tr_m], y[tr_m])
        print('  Train %-12s -> Test %-6s: %.1f%%' % (tr_n, te_n, p.score(X[te_m],y[te_m])*100))

    print('\nFinal: %s (%.1f%%)' % (bname, bscore*100))
    best.fit(X, y)

    with open(os.path.join(out_dir, 'kick_classifier_v8.pkl'), 'wb') as f:
        pickle.dump(best, f)
    meta = {'version':8, 'fs':DATASET_FS, 'features':list(X.columns),
            'classes':list(np.unique(y)), 'clip':CLIP_RANGE}
    with open(os.path.join(out_dir, 'model_metadata_v8.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    if hasattr(best.named_steps['c'], 'feature_importances_'):
        imp = best.named_steps['c'].feature_importances_
        top = np.argsort(imp)[::-1][:12]
        print('\nTop 12 features:')
        for i, idx in enumerate(top):
            print('  %2d. %-20s %.4f' % (i+1, X.columns[idx], imp[idx]))

    return best, meta


def test_phone(recordings, model, meta, out_dir):
    print('\n' + '='*60)
    print('PHONE VALIDATION (adaptive window)')
    print('Compatible preprocessing: clip->resample->gravity-align->adaptive-window')
    print('='*60)

    fig, axes = plt.subplots(len(recordings), 1, figsize=(16, 5*len(recordings)))
    if len(recordings) == 1: axes = [axes]

    all_results = {}

    for ax_idx, (rec_name, (csv_path, expected)) in enumerate(recordings.items()):
        print('\n--- %s | expected: %s ---' % (rec_name, expected))

        raw = pd.read_csv(csv_path)
        raw_df = pd.DataFrame({
            'X': raw.iloc[:,1].values,
            'Y': raw.iloc[:,2].values,
            'Z': raw.iloc[:,3].values,
        })

        df_p, grav = prepare_df(raw_df, PHONE_FS, DATASET_FS)
        print('  Samples: %d @ %dHz -> %d @ %dHz (%.1fs)' % (
            len(raw_df), PHONE_FS, len(df_p), DATASET_FS, len(df_p)/DATASET_FS))
        print('  Gravity: (%.1f,%.1f,%.1f) -> canonical (%.1f,%.1f,%.1f)' % (
            grav[0], grav[1], grav[2],
            np.mean(df_p['X']), np.mean(df_p['Y']), np.mean(df_p['Z'])))

        # Compute baseline for relative features
        bm, bs = compute_baseline(df_p, DATASET_FS)
        print('  Baseline: mean=%.2f  std=%.2f' % (bm, bs))

        segs = segment_adaptive(df_p, fs=DATASET_FS, label='unknown')
        print('  Adaptive segments: %d  (avg window: %.2fs)' % (
            len(segs),
            np.mean([s['win_sec'] for s in segs]) if segs else 0))

        results = []
        for i, seg in enumerate(segs):
            feat = extract_features(seg, fs=DATASET_FS, baseline_mean=bm, baseline_std=bs)
            if not feat: continue

            fd = pd.DataFrame([feat])
            for col in meta['features']:
                if col not in fd.columns: fd[col] = 0.0
            fd = fd[meta['features']].fillna(0)

            pred = model.predict(fd)[0]
            if hasattr(model, 'predict_proba'):
                pr   = model.predict_proba(fd)[0]
                conf = float(max(pr))
                probs= {str(c): round(float(v),2) for c,v in zip(model.classes_, pr)}
            else:
                conf = None; probs = {}

            t  = seg['peak_idx'] / DATASET_FS
            wd = seg['win_sec']
            ok = pred == expected
            print('    #%2d at %5.1fs  win=%.2fs  -> %-13s (%.0f%%) %s %s' % (
                i+1, t, wd, pred, (conf or 0)*100,
                str(probs), 'OK' if ok else 'MISS'))

            results.append({'seg':i+1,'time':round(t,1),'win':round(wd,2),
                            'pred':pred,'conf':round(conf,3) if conf else None,
                            'probs':probs})

        if results:
            preds   = [r['pred'] for r in results]
            correct = sum(1 for p in preds if p == expected)
            counts  = {}
            for p in preds: counts[p] = counts.get(p,0)+1
            print('\n  RESULT: %d/%d (%.0f%%)' % (correct, len(preds), correct/len(preds)*100))
            for cls in sorted(counts):
                tag = ' <-- EXPECTED' if cls == expected else ''
                print('    %-15s %d%s' % (cls+':', counts[cls], tag))

        all_results[rec_name] = results

        # Plot
        ax = axes[ax_idx]
        cmap = {'Ap Chagi':'green','Dolyo Chagi':'royalblue','Idle':'gray'}
        mag_f = np.sqrt(df_p['X']**2+df_p['Y']**2+df_p['Z']**2).values
        tf    = np.arange(len(mag_f))/DATASET_FS
        ax.plot(tf, mag_f, color='black', alpha=0.35, linewidth=0.6, label='|a|')

        for r in results:
            color = cmap.get(r['pred'],'red')
            ax.axvline(x=r['time'], color=color, alpha=0.8, linewidth=2.0)

        preds   = [r['pred'] for r in results] if results else []
        correct = sum(1 for p in preds if p == expected)
        ax.set_title('%s | Expected: %s | Result: %d/%d (%.0f%%)' % (
            rec_name, expected, correct, len(preds),
            correct/len(preds)*100 if preds else 0),
            fontweight='bold', fontsize=11,
            color='green' if (preds and correct==len(preds)) else 'darkred')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('|a| m/s²')

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c,label=l) for l,c in cmap.items()],
                  loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'phone_timeline_v8.png'), dpi=150)
    plt.close()

    print('\n' + '='*60 + '\nSUMMARY\n' + '='*60)
    tc, tn = 0, 0
    for rn, (_, exp) in recordings.items():
        r = all_results.get(rn,[])
        if r:
            preds = [x['pred'] for x in r]
            c = sum(1 for p in preds if p==exp)
            tc += c; tn += len(preds)
            print('  %-30s %d/%d (%.0f%%)' % (rn+':', c, len(preds), c/len(preds)*100))
    if tn:
        print('\n  TOTAL: %d/%d (%.1f%%)' % (tc, tn, tc/tn*100))

    with open(os.path.join(out_dir,'phone_validation_v8.json'),'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results


if __name__ == '__main__':
    BASE = r'd:\Downloads\Dataset\raw_data'
    OUT  = r'd:\Downloads\Dataset\classifier\output'
    REC  = r'd:\Downloads\Dataset\my_records'

    print('='*60)
    print('KICK CLASSIFIER v8 - Adaptive Window')
    print('Preprocessing: clip -> resample -> gravity-align -> adaptive-window')
    print('='*60)

    print('\n[1/3] Loading training data...')
    files = load_dataset(BASE)

    print('\n[2/3] Building features & training...')
    X, y, src = build_training(files)
    model, meta = train_model(X, y, src, OUT)

    print('\n[3/3] Testing your phone data...')
    recs = {
        'Front Kick':  (REC+'/frontkick(around 12-13 not perfect kicks)/Raw Data.csv',  'Ap Chagi'),
        'Roundhouse':  (REC+'/roundhouse(around 10 or 11 not perfect kicks)/Raw Data.csv','Dolyo Chagi'),
        'Standing':    (REC+'/standing_and_walking/Raw Data.csv',                         'Idle'),
    }
    test_phone(recs, model, meta, OUT)
