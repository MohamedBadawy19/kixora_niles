"""
Diagnose: compare feature values between training segments and phone segments.
This shows EXACTLY why the classifier says Idle for kick windows.
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, r'd:\Downloads\Dataset\classifier')
from kick_classifier_v8 import (load_dataset, prepare_df, segment_adaptive,
                                 extract_features, DATASET_FS, PHONE_FS, CLIP_RANGE)

# --- Training data ---
BASE = r'd:\Downloads\Dataset\raw_data'
files = load_dataset(BASE)

label_map = {'Ap Kanan':'Ap Chagi','Ap Kiri':'Ap Chagi',
             'Doylo Kanan':'Dolyo Chagi','Doylo Kiri':'Dolyo Chagi','Diam':'Idle'}

train_feats = {'Ap Chagi':[], 'Dolyo Chagi':[], 'Idle':[]}
for name, df in files.items():
    label = label_map[name]
    df_p, grav = prepare_df(df, DATASET_FS, DATASET_FS)
    for seg in segment_adaptive(df_p, fs=DATASET_FS):
        f = extract_features(seg, DATASET_FS)
        if f:
            train_feats[label].append(f)

print('=== TRAINING SEGMENTS ===')
key_feats = ['mag_std','mag_rms','mag_peak','mag_range','wdur','mag_kurtosis','mag_zcr']
for label, flist in train_feats.items():
    if not flist: continue
    df = pd.DataFrame(flist)
    print('\n%s (%d segs):' % (label, len(flist)))
    for k in key_feats:
        if k in df.columns:
            print('  %-20s mean=%.3f  std=%.3f  min=%.3f  max=%.3f' % (
                k, df[k].mean(), df[k].std(), df[k].min(), df[k].max()))

# --- Phone data ---
REC = r'd:\Downloads\Dataset\my_records'
phone_recs = {
    'Front Kick (Ap Chagi)': REC+'/frontkick(around 12-13 not perfect kicks)/Raw Data.csv',
    'Roundhouse (Dolyo Chagi)': REC+'/roundhouse(around 10 or 11 not perfect kicks)/Raw Data.csv',
    'Standing (Idle)': REC+'/standing_and_walking/Raw Data.csv',
}

print('\n\n=== PHONE SEGMENTS ===')
for rec_name, csv_path in phone_recs.items():
    raw = pd.read_csv(csv_path)
    raw_df = pd.DataFrame({'X':raw.iloc[:,1].values,'Y':raw.iloc[:,2].values,'Z':raw.iloc[:,3].values})
    df_p, grav = prepare_df(raw_df, PHONE_FS, DATASET_FS)
    segs = segment_adaptive(df_p, fs=DATASET_FS)
    flist = [extract_features(s, DATASET_FS) for s in segs]
    flist = [f for f in flist if f]
    if not flist: continue
    df = pd.DataFrame(flist)
    print('\n%s (%d segs):' % (rec_name, len(flist)))
    for k in key_feats:
        if k in df.columns:
            print('  %-20s mean=%.3f  std=%.3f  min=%.3f  max=%.3f' % (
                k, df[k].mean(), df[k].std(), df[k].min(), df[k].max()))
