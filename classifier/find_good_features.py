"""
Final diagnosis: look at which specific features separate Ap from Dolyo
in training data, and check if those same features differ in phone data.
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, r'd:\Downloads\Dataset\classifier')
from kick_classifier_v8 import (load_dataset, prepare_df, segment_adaptive,
                                  extract_features, compute_baseline,
                                  DATASET_FS, PHONE_FS)

# Build training features
BASE = r'd:\Downloads\Dataset\raw_data'
files = load_dataset(BASE)
label_map = {'Ap Kanan':'Ap Chagi','Ap Kiri':'Ap Chagi',
             'Doylo Kanan':'Dolyo Chagi','Doylo Kiri':'Dolyo Chagi','Diam':'Idle'}

train = {'Ap Chagi':[], 'Dolyo Chagi':[]}
for name, df in files.items():
    label = label_map[name]
    if label == 'Idle': continue
    df_p, grav = prepare_df(df, DATASET_FS, DATASET_FS)
    bm, bs = compute_baseline(df_p, DATASET_FS)
    for seg in segment_adaptive(df_p, DATASET_FS):
        f = extract_features(seg, DATASET_FS, bm, bs)
        if f: train[label].append(f)

# Build phone features
REC = r'd:\Downloads\Dataset\my_records'
phone = {
    'Ap Chagi': REC+'/frontkick(around 12-13 not perfect kicks)/Raw Data.csv',
    'Dolyo Chagi': REC+'/roundhouse(around 10 or 11 not perfect kicks)/Raw Data.csv',
}
phone_feats = {}
for label, path in phone.items():
    raw = pd.read_csv(path)
    raw_df = pd.DataFrame({'X':raw.iloc[:,1],'Y':raw.iloc[:,2],'Z':raw.iloc[:,3]})
    df_p, grav = prepare_df(raw_df, PHONE_FS, DATASET_FS)
    bm, bs = compute_baseline(df_p, DATASET_FS)
    segs = segment_adaptive(df_p, DATASET_FS)
    phone_feats[label] = [extract_features(s, DATASET_FS, bm, bs) for s in segs]
    phone_feats[label] = [f for f in phone_feats[label] if f]

# Find features that best separate Ap vs Dolyo in TRAINING data
ap_df  = pd.DataFrame(train['Ap Chagi'])
do_df  = pd.DataFrame(train['Dolyo Chagi'])
common = list(set(ap_df.columns) & set(do_df.columns))

diffs = {}
for col in common:
    ap_mean = ap_df[col].mean(); do_mean = do_df[col].mean()
    ap_std  = ap_df[col].std();  do_std  = do_df[col].std()
    pool_std = (ap_std + do_std) / 2 + 1e-10
    diffs[col] = abs(ap_mean - do_mean) / pool_std  # Cohen's d

top_feats = sorted(diffs, key=diffs.get, reverse=True)[:15]

print('TOP 15 FEATURES SEPARATING Ap vs Dolyo (training data):')
print('='*70)
print('%-25s %-12s %-12s %-12s %-8s' % ('Feature', 'Ap(train)', 'Dolyo(train)', 'Cohen d', 'Consistent?'))
print('-'*70)

for feat in top_feats:
    ap_tr  = ap_df[feat].mean()  if feat in ap_df.columns else np.nan
    do_tr  = do_df[feat].mean()  if feat in do_df.columns else np.nan

    ap_ph_df = pd.DataFrame(phone_feats['Ap Chagi'])
    do_ph_df = pd.DataFrame(phone_feats['Dolyo Chagi'])
    ap_ph  = ap_ph_df[feat].mean() if feat in ap_ph_df.columns else np.nan
    do_ph  = do_ph_df[feat].mean() if feat in do_ph_df.columns else np.nan

    # Consistent = same direction of difference in phone data
    tr_dir = np.sign(ap_tr - do_tr)
    ph_dir = np.sign(ap_ph - do_ph) if not (np.isnan(ap_ph) or np.isnan(do_ph)) else 0
    consistent = 'YES' if tr_dir == ph_dir else 'NO'

    print('%-25s %10.3f  %10.3f  %8.3f  %s (phone: Ap=%.2f Do=%.2f)' % (
        feat, ap_tr, do_tr, diffs[feat], consistent, ap_ph, do_ph))
