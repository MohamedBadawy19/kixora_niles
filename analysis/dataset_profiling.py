"""
Dataset 1 Profiling & Segmentation Test
========================================
This script:
1. Loads all 5 real data files
2. Profiles each (stats, clipping analysis, column consistency)
3. Runs the acceleration-magnitude segmentation test (peak detection)
4. Extracts per-segment features (mean, std, RMS, peak, dominant freq via FFT)
5. Generates diagnostic plots saved as PNGs
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq
import os
import json

BASE = r'd:\Downloads\Dataset\raw_data'
OUT_DIR = r'd:\Downloads\Dataset\analysis\plots'
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Load all files
# ──────────────────────────────────────────────
def load_all():
    files = {}
    
    # CSV files (X, Y, Z)
    for name in ['Ap Kiri', 'Diam', 'Doylo Kiri']:
        df = pd.read_csv(os.path.join(BASE, f'{name}.csv'))
        # Normalize column names
        df.columns = ['X', 'Y', 'Z']
        files[name] = df
    
    # Excel files
    for name in ['Ap Kanan', 'Doylo Kanan']:
        df = pd.read_excel(os.path.join(BASE, f'{name}.xlsx'), sheet_name=0)
        # Normalize column names
        df.columns = ['X', 'Y', 'Z']
        files[name] = df
    
    return files

# ──────────────────────────────────────────────
# 2. Profile each file
# ──────────────────────────────────────────────
def profile_files(files):
    profiles = {}
    for name, df in files.items():
        mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
        
        # Clipping analysis
        clip_threshold = 19.5  # close to ±19.6
        clipped_samples = {}
        for axis in ['X', 'Y', 'Z']:
            n_clipped = ((df[axis].abs() >= clip_threshold)).sum()
            clipped_samples[axis] = int(n_clipped)
        
        profile = {
            'rows': len(df),
            'columns': list(df.columns),
            'stats': df.describe().to_dict(),
            'magnitude_stats': {
                'mean': float(mag.mean()),
                'std': float(mag.std()),
                'min': float(mag.min()),
                'max': float(mag.max()),
            },
            'clipped_samples': clipped_samples,
            'total_clipped_pct': round(
                sum(clipped_samples.values()) / (len(df) * 3) * 100, 2
            ),
        }
        profiles[name] = profile
        print(f"\n=== {name} ===")
        print(f"  Rows: {profile['rows']}")
        print(f"  Magnitude: mean={profile['magnitude_stats']['mean']:.2f}, "
              f"max={profile['magnitude_stats']['max']:.2f}")
        print(f"  Clipping: {profile['clipped_samples']} "
              f"({profile['total_clipped_pct']}% of all axis samples)")
    
    return profiles

# ──────────────────────────────────────────────
# 3. Segmentation test via peak detection
# ──────────────────────────────────────────────
def segment_signal(mag, name, assumed_fs=100):
    """
    Try to segment individual kick repetitions from the magnitude signal.
    Uses a simple approach: find peaks in the magnitude that are:
    - Above a threshold (mean + 1.5*std)
    - At least 0.3 seconds apart (assumes ~100 Hz sampling)
    """
    # Light smoothing to reduce noise
    window = min(5, len(mag) // 10)
    if window > 0:
        mag_series = pd.Series(mag)
        mag_smooth = mag_series.rolling(window=window, center=True).mean()
        mag_smooth = mag_smooth.fillna(mag_series)
    else:
        mag_smooth = mag
    
    # Adaptive threshold
    threshold = np.mean(mag_smooth) + 1.5 * np.std(mag_smooth)
    min_distance = int(0.3 * assumed_fs)  # 0.3 sec minimum between peaks
    
    peaks, properties = find_peaks(
        mag_smooth.values if hasattr(mag_smooth, 'values') else mag_smooth,
        height=threshold,
        distance=min_distance,
        prominence=np.std(mag_smooth) * 0.5
    )
    
    return peaks, threshold, mag_smooth

def run_segmentation_test(files, assumed_fs=100):
    results = {}
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=False)
    fig.suptitle('Acceleration Magnitude & Detected Kick Peaks', fontsize=14, fontweight='bold')
    
    for i, (name, df) in enumerate(files.items()):
        mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
        peaks, threshold, mag_smooth = segment_signal(mag.values, name, assumed_fs)
        
        time_axis = np.arange(len(mag)) / assumed_fs  # seconds (assumed)
        
        ax = axes[i]
        ax.plot(time_axis, mag, alpha=0.4, color='steelblue', linewidth=0.5, label='Raw |a|')
        ax.plot(time_axis, mag_smooth, color='navy', linewidth=0.8, label='Smoothed')
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                    label=f'Threshold ({threshold:.1f})')
        
        if len(peaks) > 0:
            ax.scatter(peaks / assumed_fs, mag_smooth.values[peaks] if hasattr(mag_smooth, 'values') else mag_smooth[peaks],
                       color='red', s=30, zorder=5, label=f'{len(peaks)} peaks detected')
        
        ax.set_title(f'{name} — {len(peaks)} peaks detected | {len(df)} samples', fontsize=11)
        ax.set_ylabel('|a| (m/s²)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Determine if peaks are "clean" (consistent spacing + similar heights)
        if len(peaks) > 2:
            intervals = np.diff(peaks) / assumed_fs
            peak_heights = mag.values[peaks]
            regularity = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
            height_consistency = np.std(peak_heights) / np.mean(peak_heights) if np.mean(peak_heights) > 0 else float('inf')
            
            quality = "CLEAN" if regularity < 0.5 and height_consistency < 0.3 else \
                      "MESSY" if regularity > 1.0 or height_consistency > 0.5 else "MODERATE"
        else:
            regularity = float('inf')
            height_consistency = float('inf')
            quality = "TOO FEW PEAKS"
        
        results[name] = {
            'n_peaks': int(len(peaks)),
            'threshold': float(threshold),
            'peak_regularity_cv': round(float(regularity), 3) if regularity != float('inf') else None,
            'peak_height_cv': round(float(height_consistency), 3) if height_consistency != float('inf') else None,
            'segmentation_quality': quality,
        }
        
        print(f"\n  {name}: {len(peaks)} peaks | Quality: {quality} | "
              f"Regularity CV: {regularity:.3f} | Height CV: {height_consistency:.3f}")
    
    axes[-1].set_xlabel('Time (seconds, assumed 100 Hz)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'segmentation_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  -> Saved: {os.path.join(OUT_DIR, 'segmentation_test.png')}")
    
    return results

# ──────────────────────────────────────────────
# 4. Per-axis raw signal plots
# ──────────────────────────────────────────────
def plot_raw_signals(files, assumed_fs=100):
    for name, df in files.items():
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f'{name} — Raw Tri-axial Accelerometer Signal', fontsize=13, fontweight='bold')
        
        time_axis = np.arange(len(df)) / assumed_fs
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        
        for j, axis in enumerate(['X', 'Y', 'Z']):
            axes[j].plot(time_axis, df[axis], color=colors[j], linewidth=0.6, alpha=0.8)
            axes[j].axhline(y=19.6, color='gray', linestyle=':', alpha=0.5)
            axes[j].axhline(y=-19.6, color='gray', linestyle=':', alpha=0.5)
            axes[j].set_ylabel(f'{axis} (m/s²)')
            axes[j].grid(True, alpha=0.3)
            axes[j].set_ylim(-25, 25)
        
        axes[-1].set_xlabel('Time (seconds, assumed 100 Hz)')
        plt.tight_layout()
        safe_name = name.replace(' ', '_')
        plt.savefig(os.path.join(OUT_DIR, f'raw_{safe_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

# ──────────────────────────────────────────────
# 5. Feature extraction per detected segment
# ──────────────────────────────────────────────
def extract_segment_features(files, assumed_fs=100):
    """Extract features from detected segments for comparability analysis."""
    all_features = {}
    
    for name, df in files.items():
        if name == 'Diam':
            continue  # Skip idle for kick feature comparison
        
        mag = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).values
        peaks, threshold, mag_smooth = segment_signal(mag, name, assumed_fs)
        
        if len(peaks) < 2:
            continue
        
        # Define segments: window around each peak
        half_window = int(0.25 * assumed_fs)  # ±0.25 sec around each peak
        segments = []
        
        for p in peaks:
            start = max(0, p - half_window)
            end = min(len(mag), p + half_window)
            seg = mag[start:end]
            
            if len(seg) < 10:
                continue
            
            # Time-domain features
            seg_mean = np.mean(seg)
            seg_std = np.std(seg)
            seg_rms = np.sqrt(np.mean(seg**2))
            seg_peak = np.max(seg)
            seg_range = np.max(seg) - np.min(seg)
            
            # Per-axis features from the segment
            x_seg = df['X'].values[start:end]
            y_seg = df['Y'].values[start:end]
            z_seg = df['Z'].values[start:end]
            
            # Frequency domain (dominant frequency via FFT)
            n = len(seg)
            yf = np.abs(fft(seg - np.mean(seg)))[:n//2]
            xf = fftfreq(n, 1/assumed_fs)[:n//2]
            if len(yf) > 1:
                dom_freq = float(xf[np.argmax(yf[1:]) + 1])  # skip DC
            else:
                dom_freq = 0.0
            
            segments.append({
                'mean_mag': float(seg_mean),
                'std_mag': float(seg_std),
                'rms_mag': float(seg_rms),
                'peak_mag': float(seg_peak),
                'range_mag': float(seg_range),
                'mean_x': float(np.mean(x_seg)),
                'mean_y': float(np.mean(y_seg)),
                'mean_z': float(np.mean(z_seg)),
                'dom_freq': float(dom_freq),
            })
        
        all_features[name] = segments
        print(f"  {name}: extracted {len(segments)} segment feature vectors")
    
    return all_features

# ──────────────────────────────────────────────
# 6. Feature distribution comparison plot
# ──────────────────────────────────────────────
def plot_feature_distributions(features):
    feature_names = ['mean_mag', 'std_mag', 'rms_mag', 'peak_mag', 'range_mag', 'dom_freq']
    kick_types = list(features.keys())
    
    if len(kick_types) == 0:
        print("  No kick segments to compare.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Feature Distributions Across Kick Types (Dataset 1)', fontsize=14, fontweight='bold')
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, feat in enumerate(feature_names):
        ax = axes[i // 3, i % 3]
        for j, kick in enumerate(kick_types):
            vals = [s[feat] for s in features[kick]]
            if len(vals) > 0:
                ax.hist(vals, bins=15, alpha=0.5, color=colors[j % len(colors)], 
                        label=kick, edgecolor='white', linewidth=0.5)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {os.path.join(OUT_DIR, 'feature_distributions.png')}")

# ──────────────────────────────────────────────
# 7. Clipping visualization
# ──────────────────────────────────────────────
def plot_clipping_analysis(files):
    fig, axes = plt.subplots(1, len(files), figsize=(18, 5))
    fig.suptitle('Clipping Analysis — Distribution of Values Near ±19.6 Boundary', 
                 fontsize=13, fontweight='bold')
    
    for i, (name, df) in enumerate(files.items()):
        ax = axes[i]
        for axis, color in zip(['X', 'Y', 'Z'], ['#e74c3c', '#2ecc71', '#3498db']):
            ax.hist(df[axis], bins=80, alpha=0.5, color=color, label=axis, edgecolor='white', linewidth=0.3)
        ax.axvline(x=19.6, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=-19.6, color='black', linestyle='--', alpha=0.5)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'clipping_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {os.path.join(OUT_DIR, 'clipping_analysis.png')}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("DATASET 1 — FULL PROFILING & SEGMENTATION TEST")
    print("=" * 60)
    
    print("\n[1/6] Loading files...")
    files = load_all()
    
    print("\n[2/6] Profiling files...")
    profiles = profile_files(files)
    
    print("\n[3/6] Running segmentation test (peak detection on |a|)...")
    seg_results = run_segmentation_test(files)
    
    print("\n[4/6] Plotting raw signals...")
    plot_raw_signals(files)
    print("  -> Saved raw signal plots")
    
    print("\n[5/6] Extracting per-segment features...")
    features = extract_segment_features(files)
    
    print("\n[6/6] Plotting feature distributions & clipping analysis...")
    plot_feature_distributions(features)
    plot_clipping_analysis(files)
    
    # Save results JSON
    results = {
        'profiles': {},
        'segmentation': seg_results,
    }
    for name, p in profiles.items():
        results['profiles'][name] = {
            'rows': p['rows'],
            'magnitude_stats': p['magnitude_stats'],
            'clipped_samples': p['clipped_samples'],
            'total_clipped_pct': p['total_clipped_pct'],
        }
    
    with open(os.path.join(OUT_DIR, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("DONE — All plots saved to: " + OUT_DIR)
    print(f"{'=' * 60}")
