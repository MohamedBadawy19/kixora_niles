"""
Kick Classifier v9 - Consistent Features Only
==============================================
Use ONLY features that show the same direction of difference
between Ap Chagi and Dolyo Chagi in BOTH training AND phone data.

Consistent features found by cross-domain analysis:
  rel_mag_peak:  Ap > Dolyo (relative peak above baseline)
  rel_mag_rms:   Ap > Dolyo (relative RMS)  
  y_std:         Dolyo > Ap (more lateral Y-axis motion in roundhouse)
  y_rms:         Dolyo > Ap
  xs:            Ap > Dolyo (more X-axis energy share in front kick)
  snr:           Ap > Dolyo
  y_kurtosis:    Ap > Dolyo
  y_range:       Dolyo > Ap
  x_zcr:         Ap ~ Dolyo (borderline)
  y_jrms:        Dolyo > Ap

Inconsistent features that hurt generalization are excluded.
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
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_FS  = 100
PHONE_FS    = 500
CLIP_RANGE  = 19.6
MIN_WIN_SEC = 0.1
MAX_WIN_SEC = 0.8
ACTIVE_MULT = 0.5


def estimate_gravity(df, fs, rest_sec=2.0):
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
    out = df.copy()
    gx, gy, gz = grav
    dom = int(np.argmax([abs(gx), abs(gy), abs(gz)]))
    dom_val = [gx, gy, gz][dom]
    if dom == 1 and dom_val > 0:
        out['Y'] = -df['Y']; out['Z'] = -df['Z']
    elif dom == 2 and dom_val > 0:
        out['Y'] = -df['Z']; out['Z'] =  df['Y']
    elif dom == 2 and dom_val < 0:
        out['Y'] =  df['Z']; out['Z'] = -df['Y']
    elif dom == 0 and dom_val > 0:
        out['Y'] = -df['X']; out['X'] =  df['Y']
    elif dom == 0 and dom_val < 0:
        out['Y'] =  df['X']; out['X'] = -df['Y']
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


def prepare_df(raw_df, src_fs, tgt_fs, clip_val=CLIP_RANGE):
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


def compute_baseline(df, fs, quiet_percentile=30):
    mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).values
    sw = int(0.5 * fs)
    rs = pd.Series(mag).rolling(window=sw, center=True).std()
    rs = rs.fillna(rs.median()).values
    thr = np.percentile(rs, quiet_percentile)
    quiet = mag[rs <= thr]
    if len(quiet) < 10: quiet = mag
    return float(np.mean(quiet)), float(np.std(quiet))


def compute_dynamic(df):
    mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).values
    return np.abs(mag - np.median(mag)), float(np.median(mag))


def adaptive_window(dyn, peak_idx, fs):
    peak_dyn = dyn[peak_idx]
    threshold = max(peak_dyn * ACTIVE_MULT, 0.5)
    min_s = int(MIN_WIN_SEC * fs)
    max_s = int(MAX_WIN_SEC * fs)
    left = peak_idx
    while left > 0 and (peak_idx - left) < max_s:
        if dyn[left - 1] < threshold: break
        left -= 1
    right = peak_idx
    while right < len(dyn) - 1 and (right - peak_idx) < max_s:
        if dyn[right + 1] < threshold: break
        right += 1
    current = right - left
    if current < min_s:
        pad = (min_s - current) // 2
        left  = max(0, left - pad)
        right = min(len(dyn) - 1, right + pad)
    return left, right + 1


def segment_adaptive(df, fs=DATASET_FS, label=None):
    dyn, gravity = compute_dynamic(df)
    sw = max(3, int(0.04 * fs))
    dyn_s = pd.Series(dyn).rolling(window=sw, center=True).mean()
    dyn_s = dyn_s.fillna(pd.Series(dyn)).values
    threshold = np.mean(dyn_s) + 0.6 * np.std(dyn_s)
    peaks, _ = find_peaks(dyn_s, height=threshold, distance=int(0.5*fs),
                          prominence=np.std(dyn_s)*0.25)
    segments = []
    for p in peaks:
        s, e = adaptive_window(dyn_s, p, fs)
        if (e-s) < int(MIN_WIN_SEC*fs): continue
        if np.max(dyn_s[s:e]) < threshold*0.5: continue
        segments.append({
            'X': df['X'].values[s:e], 'Y': df['Y'].values[s:e],
            'Z': df['Z'].values[s:e],
            'mag': np.sqrt(df['X'].values[s:e]**2+df['Y'].values[s:e]**2+df['Z'].values[s:e]**2),
            'peak_idx': p, 'win_start': s, 'win_end': e,
            'win_sec': (e-s)/fs, 'label': label, 'fs': fs,
        })
    return segments


def extract_features(seg, fs=DATASET_FS, bm=None, bs=None):
    """
    Extract only cross-domain consistent features.
    Relative features (require bm baseline) + shape features.
    """
    x, y, z = seg['X'], seg['Y'], seg['Z']
    mag = seg['mag']
    if len(x) < 6: return None

    xd = x - np.mean(x); yd = y - np.mean(y); zd = z - np.mean(z)
    mag_d = np.sqrt(xd**2 + yd**2 + zd**2)

    f = {}

    # ---- Relative-to-baseline features (sensor-scale invariant) ----
    if bm is not None and bm > 0.1:
        f['rel_mag_peak'] = float((np.max(mag) - bm) / bm)
        f['rel_mag_std']  = float(np.std(mag) / bm)
        f['rel_mag_rms']  = float(np.sqrt(np.mean(mag**2)) / bm)
        f['rel_dyn_peak'] = float(np.max(mag_d) / bm)
        f['rel_dyn_std']  = float(np.std(mag_d) / bm)
        bs_safe = max(bs, 0.1) if bs else 0.1
        f['snr']          = float(np.std(mag) / bs_safe)

    # ---- Y-axis features (lateral motion — key for Ap vs Dolyo) ----
    y_std = np.std(yd)
    f['y_std']  = float(y_std)
    f['y_rms']  = float(np.sqrt(np.mean(yd**2)))
    f['y_range'] = float(np.ptp(yd))
    y_n = (yd - np.mean(yd)) / y_std if y_std > 1e-6 else yd * 0.0
    f['y_kurtosis'] = float(kurtosis(y_n))
    f['y_skew']     = float(skew(y_n))
    if len(yd) > 2:
        jk_y = np.diff(yd) * fs
        f['y_jrms'] = float(np.sqrt(np.mean(jk_y**2)))
        f['y_jpk']  = float(np.max(np.abs(jk_y)))

    # ---- X-axis features (forward motion — front kick dominant) ----
    x_std = np.std(xd)
    f['x_std']  = float(x_std)
    x_n = (xd - np.mean(xd)) / x_std if x_std > 1e-6 else xd * 0.0
    f['x_zcr']  = float(np.sum(np.diff(np.sign(x_n)) != 0) / (len(xd)/fs))
    f['x_kurtosis'] = float(kurtosis(x_n))
    if len(xd) > 2:
        jk_x = np.diff(xd) * fs
        f['x_jrms'] = float(np.sqrt(np.mean(jk_x**2)))

    # ---- Axis energy shares (orientation-relative) ----
    te = np.std(xd)**2 + np.std(yd)**2 + np.std(zd)**2 + 1e-10
    f['xs'] = float(np.std(xd)**2 / te)   # front kick: more X
    f['ys'] = float(np.std(yd)**2 / te)   # roundhouse: more Y
    f['zs'] = float(np.std(zd)**2 / te)

    # ---- Magnitude shape features (dimensionless) ----
    mag_std = np.std(mag_d)
    m_n = (mag_d - np.mean(mag_d)) / mag_std if mag_std > 1e-6 else mag_d * 0.0
    f['mag_kurtosis'] = float(kurtosis(m_n))
    f['mag_skew']     = float(skew(m_n))
    f['mag_zcr']      = float(np.sum(np.diff(np.sign(m_n)) != 0) / (len(m_n)/fs))

    # Time to peak (shape)
    f['ttp'] = float(np.argmax(mag_d) / max(len(mag_d)-1, 1))
    # Energy symmetry
    mid = len(mag_d) // 2
    e1 = np.sum(mag_d[:mid]**2)+1e-10; e2 = np.sum(mag_d[mid:]**2)+1e-10
    f['esym'] = float(e1 / (e1+e2))
    # Window duration
    f['wdur'] = float(seg['win_sec'])

    # Correlations (shape relationship between axes)
    try:
        f['cxy'] = float(np.corrcoef(xd, yd)[0,1])
        f['cxz'] = float(np.corrcoef(xd, zd)[0,1])
        f['cyz'] = float(np.corrcoef(yd, zd)[0,1])
    except:
        f['cxy'] = f['cxz'] = f['cyz'] = 0.0

    # Frequency features on Y (lateral) — Dolyo should have lower freq due to arc
    if len(yd) > 8:
        yf = np.abs(fft(y_n))[:len(y_n)//2]
        xf = fftfreq(len(y_n), 1.0/fs)[:len(y_n)//2]
        if len(yf) > 1:
            f['y_df']  = float(xf[np.argmax(yf[1:])+1])
            te2 = np.sum(yf**2)+1e-10
            f['y_lfr'] = float(np.sum(yf[xf<5]**2)/te2)
            f['y_sc']  = float(np.sum(xf*yf)/(np.sum(yf)+1e-10))

    for k, v in f.items():
        if np.isnan(v) or np.isinf(v): f[k] = 0.0
    return f


def build_training(files, fs=DATASET_FS):
    label_map = {'Ap Kanan':'Ap Chagi','Ap Kiri':'Ap Chagi',
                 'Doylo Kanan':'Dolyo Chagi','Doylo Kiri':'Dolyo Chagi','Diam':'Idle'}
    X_l, y_l, s_l = [], [], []
    for name, df in files.items():
        label = label_map[name]
        df_p, grav = prepare_df(df, fs, fs)
        bm, bs = compute_baseline(df_p, fs)
        for seg in segment_adaptive(df_p, fs, label):
            feat = extract_features(seg, fs, bm, bs)
            if feat: X_l.append(feat); y_l.append(label); s_l.append(name)
    return pd.DataFrame(X_l), np.array(y_l), np.array(s_l)


def train_model(X, y, sources, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X = X.fillna(0)
    print('\nTraining: %d segments, %d features' % (len(X), X.shape[1]))
    for c, n in zip(*np.unique(y, return_counts=True)):
        print('  %-15s %d' % (c+':', n))

    clfs = {
        'RF':  RandomForestClassifier(n_estimators=500, max_depth=10,
                                       min_samples_split=4, min_samples_leaf=2,
                                       random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', C=5, gamma='scale', random_state=42,
                   class_weight='balanced', probability=True),
        'GB':  GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                           learning_rate=0.05, random_state=42),
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
                      ('c', RandomForestClassifier(n_estimators=500, max_depth=10,
                                                    random_state=42, class_weight='balanced'))])
        p.fit(X[tr_m], y[tr_m])
        print('  Train %-12s -> Test %-6s: %.1f%%' % (tr_n, te_n, p.score(X[te_m],y[te_m])*100))

    print('\nFinal: %s (%.1f%%)' % (bname, bscore*100))
    best.fit(X, y)

    with open(os.path.join(out_dir, 'kick_classifier_v9.pkl'), 'wb') as f:
        pickle.dump(best, f)
    meta = {'version':9, 'fs':DATASET_FS, 'features':list(X.columns),
            'classes':list(np.unique(y))}
    with open(os.path.join(out_dir, 'model_metadata_v9.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    if hasattr(best.named_steps['c'], 'feature_importances_'):
        imp = best.named_steps['c'].feature_importances_
        top = np.argsort(imp)[::-1][:10]
        print('\nTop 10 features:')
        for i, idx in enumerate(top):
            print('  %2d. %-20s %.4f' % (i+1, X.columns[idx], imp[idx]))
    return best, meta


def test_phone(recordings, model, meta, out_dir):
    print('\n' + '='*60)
    print('PHONE VALIDATION v9 — consistent features only')
    print('='*60)

    fig, axes = plt.subplots(len(recordings), 1, figsize=(16, 5*len(recordings)))
    if len(recordings) == 1: axes = [axes]
    all_results = {}

    for ax_idx, (rec_name, (csv_path, expected)) in enumerate(recordings.items()):
        print('\n--- %s | expected: %s ---' % (rec_name, expected))
        raw = pd.read_csv(csv_path)
        raw_df = pd.DataFrame({'X':raw.iloc[:,1],'Y':raw.iloc[:,2],'Z':raw.iloc[:,3]})

        df_p, grav = prepare_df(raw_df, PHONE_FS, DATASET_FS)
        bm, bs = compute_baseline(df_p, DATASET_FS)
        print('  Resampled: %d@%dHz -> %d@%dHz (%.1fs) | baseline=%.2f' % (
            len(raw_df), PHONE_FS, len(df_p), DATASET_FS, len(df_p)/DATASET_FS, bm))

        segs = segment_adaptive(df_p, DATASET_FS)
        print('  Segments: %d' % len(segs))

        results = []
        for i, seg in enumerate(segs):
            feat = extract_features(seg, DATASET_FS, bm, bs)
            if not feat: continue
            fd = pd.DataFrame([feat])
            for col in meta['features']:
                if col not in fd.columns: fd[col] = 0.0
            fd = fd[meta['features']].fillna(0)

            pred = model.predict(fd)[0]
            if hasattr(model, 'predict_proba'):
                pr   = model.predict_proba(fd)[0]
                conf = float(max(pr))
                probs= {str(c):round(float(v),2) for c,v in zip(model.classes_,pr)}
            else:
                conf = None; probs = {}

            t = seg['peak_idx'] / DATASET_FS
            ok = pred == expected
            print('    #%2d @%5.1fs win=%.2fs -> %-13s (%.0f%%) %s [%s]' % (
                i+1, t, seg['win_sec'], pred, (conf or 0)*100, probs, 'OK' if ok else 'MISS'))
            results.append({'seg':i+1,'time':round(t,1),'win':round(seg['win_sec'],2),
                            'pred':pred,'conf':round(conf,3) if conf else None,'probs':probs})

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

        ax = axes[ax_idx]
        cmap = {'Ap Chagi':'green','Dolyo Chagi':'royalblue','Idle':'gray'}
        mag_f = np.sqrt(df_p['X']**2+df_p['Y']**2+df_p['Z']**2).values
        tf = np.arange(len(mag_f))/DATASET_FS
        ax.plot(tf, mag_f, 'k-', alpha=0.35, linewidth=0.6)
        for r in results:
            ax.axvline(x=r['time'], color=cmap.get(r['pred'],'red'), alpha=0.8, linewidth=2)
        preds = [r['pred'] for r in results] if results else []
        correct = sum(1 for p in preds if p == expected)
        ax.set_title('%s | %d/%d (%.0f%%)' % (
            rec_name, correct, len(preds), correct/len(preds)*100 if preds else 0),
            fontweight='bold', color='green' if (preds and correct==len(preds)) else 'darkred')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('|a| m/s²')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c,label=l) for l,c in cmap.items()],
                  loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'phone_timeline_v9.png'), dpi=150)
    plt.close()

    print('\n' + '='*60 + '\nFINAL SUMMARY\n' + '='*60)
    tc, tn = 0, 0
    for rn, (_, exp) in recordings.items():
        r = all_results.get(rn, [])
        if r:
            preds = [x['pred'] for x in r]
            c = sum(1 for p in preds if p == exp)
            tc += c; tn += len(preds)
            print('  %-30s %d/%d (%.0f%%)' % (rn+':', c, len(preds), c/len(preds)*100))
    if tn:
        print('\n  TOTAL: %d/%d (%.1f%%)' % (tc, tn, tc/tn*100))

    with open(os.path.join(out_dir,'phone_validation_v9.json'),'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results


if __name__ == '__main__':
    BASE = r'd:\Downloads\Dataset\raw_data'
    OUT  = r'd:\Downloads\Dataset\classifier\output'
    REC  = r'd:\Downloads\Dataset\my_records'

    print('='*60)
    print('KICK CLASSIFIER v9 - Cross-Domain Consistent Features')
    print('='*60)

    files = load_dataset(BASE)
    X, y, src = build_training(files)
    model, meta = train_model(X, y, src, OUT)

    recs = {
        'Front Kick': (REC+'/frontkick(around 12-13 not perfect kicks)/Raw Data.csv',   'Ap Chagi'),
        'Roundhouse': (REC+'/roundhouse(around 10 or 11 not perfect kicks)/Raw Data.csv','Dolyo Chagi'),
        'Standing':   (REC+'/standing_and_walking/Raw Data.csv',                          'Idle'),
    }
    test_phone(recs, model, meta, OUT)
