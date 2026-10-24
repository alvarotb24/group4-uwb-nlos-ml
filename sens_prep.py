import pandas as pd
from features import load_raw, clean, compute_features
raw = load_raw('UWB-LOS-NLOS-Data-Set/dataset'); df, rep = clean(raw); del raw
f = compute_features(df); f['PREAM_LEN'] = df['PREAM_LEN'].to_numpy()
f.to_pickle('feats_sens.pkl'); print(rep, f.shape); print(f.PREAM_LEN.value_counts().to_dict())
print(pd.crosstab(f.PREAM_LEN, f.NLOS))
