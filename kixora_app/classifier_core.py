"""
classifier_core.py — Extracted v9 pipeline for real-time use.
Drop-in from kick_classifier_v9.py, adapted for streaming segments.
"""
import numpy as np
import pandas as pd
import pickle, json, os
from scipy.signal import find_peaks, resample
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

DATASET_FS  = 100
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


def segment_adaptive(df, fs=DATASET_FS):
    dyn, _ = compute_dynamic(df)
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
            'win_sec': (e-s)/fs,
        })
    return segments


def extract_features(seg, fs=DATASET_FS, bm=None, bs=None):
    x, y, z = seg['X'], seg['Y'], seg['Z']
    mag = seg['mag']
    if len(x) < 6: return None
    xd = x - np.mean(x); yd = y - np.mean(y); zd = z - np.mean(z)
    mag_d = np.sqrt(xd**2 + yd**2 + zd**2)
    f = {}
    if bm is not None and bm > 0.1:
        f['rel_mag_peak'] = float((np.max(mag) - bm) / bm)
        f['rel_mag_std']  = float(np.std(mag) / bm)
        f['rel_mag_rms']  = float(np.sqrt(np.mean(mag**2)) / bm)
        f['rel_dyn_peak'] = float(np.max(mag_d) / bm)
        f['rel_dyn_std']  = float(np.std(mag_d) / bm)
        bs_safe = max(bs, 0.1) if bs else 0.1
        f['snr']          = float(np.std(mag) / bs_safe)
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
    x_std = np.std(xd)
    f['x_std']  = float(x_std)
    x_n = (xd - np.mean(xd)) / x_std if x_std > 1e-6 else xd * 0.0
    f['x_zcr']  = float(np.sum(np.diff(np.sign(x_n)) != 0) / (len(xd)/fs))
    f['x_kurtosis'] = float(kurtosis(x_n))
    if len(xd) > 2:
        jk_x = np.diff(xd) * fs
        f['x_jrms'] = float(np.sqrt(np.mean(jk_x**2)))
    te = np.std(xd)**2 + np.std(yd)**2 + np.std(zd)**2 + 1e-10
    f['xs'] = float(np.std(xd)**2 / te)
    f['ys'] = float(np.std(yd)**2 / te)
    f['zs'] = float(np.std(zd)**2 / te)
    mag_std = np.std(mag_d)
    m_n = (mag_d - np.mean(mag_d)) / mag_std if mag_std > 1e-6 else mag_d * 0.0
    f['mag_kurtosis'] = float(kurtosis(m_n))
    f['mag_skew']     = float(skew(m_n))
    f['mag_zcr']      = float(np.sum(np.diff(np.sign(m_n)) != 0) / (len(m_n)/fs))
    f['ttp'] = float(np.argmax(mag_d) / max(len(mag_d)-1, 1))
    mid = len(mag_d) // 2
    e1 = np.sum(mag_d[:mid]**2)+1e-10; e2 = np.sum(mag_d[mid:]**2)+1e-10
    f['esym'] = float(e1 / (e1+e2))
    f['wdur'] = float(seg['win_sec'])
    try:
        f['cxy'] = float(np.corrcoef(xd, yd)[0,1])
        f['cxz'] = float(np.corrcoef(xd, zd)[0,1])
        f['cyz'] = float(np.corrcoef(yd, zd)[0,1])
    except:
        f['cxy'] = f['cxz'] = f['cyz'] = 0.0
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


def resample_to_100hz(df, src_fs):
    if src_fs == DATASET_FS: return df.copy()
    n = int(len(df) * DATASET_FS / src_fs)
    out = pd.DataFrame()
    for c in df.columns:
        out[c] = resample(df[c].values, n)
    return out


MIN_CONFIDENCE = 0.60   # only emit kicks above this confidence

class RealtimeClassifier:
    """
    Sliding-window real-time classifier.
    Call feed_imu(x, y, z, timestamp_ms) repeatedly.
    Call get_kicks() to retrieve newly classified kicks.
    """
    def __init__(self, model_path, meta_path, src_fs=500):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        self.src_fs = src_fs
        self.features = self.meta['features']
        self.classes   = self.meta['classes']
        # Ring buffer: 10 seconds at src_fs
        self.buf_len = src_fs * 10
        self.buf_x   = np.zeros(self.buf_len)
        self.buf_y   = np.zeros(self.buf_len)
        self.buf_z   = np.zeros(self.buf_len)
        self.buf_t   = np.zeros(self.buf_len)
        self.write_idx = 0
        self.n_samples = 0
        self.gravity   = None          # estimated once we have enough data
        self.baseline_mag = None
        self.baseline_std = None
        self.pending_kicks = []
        self.last_classified_end = 0   # sample index of last classified segment end
        # How many samples to accumulate before first classification
        self.warmup_samples = src_fs * 3   # 3 seconds

    def feed_imu(self, x, y, z, timestamp_ms):
        idx = self.write_idx % self.buf_len
        self.buf_x[idx] = np.clip(x, -CLIP_RANGE, CLIP_RANGE)
        self.buf_y[idx] = np.clip(y, -CLIP_RANGE, CLIP_RANGE)
        self.buf_z[idx] = np.clip(z, -CLIP_RANGE, CLIP_RANGE)
        self.buf_t[idx] = timestamp_ms
        self.write_idx += 1
        self.n_samples  = min(self.n_samples + 1, self.buf_len)
        if self.n_samples >= self.warmup_samples:
            self._try_classify()

    def _get_window(self):
        """Return ordered arrays from ring buffer."""
        if self.n_samples < self.buf_len:
            return (self.buf_x[:self.n_samples].copy(),
                    self.buf_y[:self.n_samples].copy(),
                    self.buf_z[:self.n_samples].copy(),
                    self.buf_t[:self.n_samples].copy())
        start = self.write_idx % self.buf_len
        return (np.roll(self.buf_x, -start)[:self.n_samples],
                np.roll(self.buf_y, -start)[:self.n_samples],
                np.roll(self.buf_z, -start)[:self.n_samples],
                np.roll(self.buf_t, -start)[:self.n_samples])

    def _try_classify(self):
        wx, wy, wz, wt = self._get_window()
        raw_df = pd.DataFrame({'X': wx, 'Y': wy, 'Z': wz})
        # Resample to 100 Hz
        df100 = resample_to_100hz(raw_df, self.src_fs)
        # Estimate gravity on first call
        if self.gravity is None:
            self.gravity = estimate_gravity(df100, DATASET_FS)
        df_al = align_gravity(df100, self.gravity)
        # Compute baseline once
        if self.baseline_mag is None:
            self.baseline_mag, self.baseline_std = compute_baseline(df_al, DATASET_FS)
        segments = segment_adaptive(df_al, DATASET_FS)
        for seg in segments:
            # Only process segments whose peak is new (not yet classified)
            peak_in_100hz = seg['peak_idx']
            # Convert to src_fs index to track position
            peak_src = int(peak_in_100hz * self.src_fs / DATASET_FS)
            abs_peak = (self.write_idx - self.n_samples) + peak_src
            if abs_peak <= self.last_classified_end:
                continue
            feat = extract_features(seg, DATASET_FS, self.baseline_mag, self.baseline_std)
            if not feat:
                continue
            fd = pd.DataFrame([feat])
            for col in self.features:
                if col not in fd.columns: fd[col] = 0.0
            fd = fd[self.features].fillna(0)
            pred = self.model.predict(fd)[0]
            probs = {}
            conf = 0.0
            if hasattr(self.model, 'predict_proba'):
                pr = self.model.predict_proba(fd)[0]
                conf = float(max(pr))
                probs = {str(c): round(float(v), 3) for c, v in zip(self.model.classes_, pr)}
            self.last_classified_end = abs_peak
            # Skip Idle and low-confidence predictions
            if pred != 'Idle' and conf >= MIN_CONFIDENCE:
                self.pending_kicks.append({
                    'label': pred,
                    'confidence': round(conf, 3),
                    'probs': probs,
                    'win_sec': round(seg['win_sec'], 3),
                    'peak_time_s': round(peak_in_100hz / DATASET_FS, 2),
                })

    def get_kicks(self):
        kicks = self.pending_kicks[:]
        self.pending_kicks = []
        return kicks

    def reset_gravity(self):
        """Call this if phone orientation changes between sessions."""
        self.gravity = None
        self.baseline_mag = None
        self.baseline_std = None
