import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from features import FEATURES
f = pd.read_pickle('feats_sens.pkl')
which = sys.argv[1]; models = sys.argv[2].split(',')
if which == 'no1536':
    d = f[f.PREAM_LEN != 1536].reset_index(drop=True)
elif which == 'matched':
    rng = np.random.default_rng(42)
    bins = np.floor(f.RANGE / 0.25).astype(int)
    keep = []
    for b, g in f.groupby(bins):
        los = g.index[g.NLOS == 0].to_numpy(); nlos = g.index[g.NLOS == 1].to_numpy()
        k = min(len(los), len(nlos))
        if k: keep += list(rng.choice(los, k, replace=False)) + list(rng.choice(nlos, k, replace=False))
    d = f.loc[sorted(keep)].reset_index(drop=True)
else:
    d = f
y = d.NLOS.to_numpy()
print(f"{which}: n={len(d)} LOS={int((y==0).sum())} NLOS={int((y==1).sum())}")
if which == 'matched':
    print('  RANGE median LOS %.2f NLOS %.2f' % (d.RANGE[y==0].median(), d.RANGE[y==1].median()))
for c in ['RANGE','RX_POW','FP_POW','RXPACC','PWR_DIFF','MEAN_DELAY']:
    a = roc_auc_score(y, d[c]); print(f"  single-feature AUC {c}: {max(a,1-a):.3f}")
mk = {'M0': (lambda: make_pipeline(StandardScaler(), LogisticRegression()), ['PWR_DIFF']),
      'M2': (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), FEATURES),
      'M4': (lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=2), FEATURES)}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for name in models:
    fac, cols = mk[name]; X = d[cols].to_numpy(); acc=[]; auc=[]; miss=[]
    for tr, te in skf.split(X, y):
        m = fac().fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:,1]; pr = (p>=.5).astype(int)
        acc.append(accuracy_score(y[te], pr)); auc.append(roc_auc_score(y[te], p)); miss.append(np.mean(pr[y[te]==1]==0))
    print(f"  {name}: acc {100*np.mean(acc):.1f} ± {100*np.std(acc,ddof=1):.1f} %  AUC {np.mean(auc):.3f} ± {np.std(auc,ddof=1):.3f}  missed NLOS {100*np.mean(miss):.1f} %", flush=True)
