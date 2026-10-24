# Opposition question 4.2: missed-NLOS vs false alarms for M0, M2, M4 (out-of-fold, stratified 5-fold, seed 42)
import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from features import FEATURES
f = pd.read_pickle('feats_sens.pkl')
which = sys.argv[1]
if which == 'matched':
    rng = np.random.default_rng(42); bins = np.floor(f.RANGE/0.25).astype(int); keep=[]
    for b,g in f.groupby(bins):
        los=g.index[g.NLOS==0].to_numpy(); nl=g.index[g.NLOS==1].to_numpy(); k=min(len(los),len(nl))
        if k: keep += list(rng.choice(los,k,replace=False))+list(rng.choice(nl,k,replace=False))
    d=f.loc[sorted(keep)].reset_index(drop=True)
else:
    d=f
y=d.NLOS.to_numpy()
mk={'M0':(lambda: make_pipeline(StandardScaler(),LogisticRegression()),['PWR_DIFF']),
    'M2':(lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),FEATURES),
    'M4':(lambda: RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=2),FEATURES)}
skf=StratifiedKFold(5,shuffle=True,random_state=42)
oof={}
for name,(fac,cols) in mk.items():
    X=d[cols].to_numpy(); p=np.zeros(len(y))
    for tr,te in skf.split(X,y):
        p[te]=fac().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    oof[name]=p
def rates(score,thr):
    pr=score>=thr
    return 100*np.mean(~pr[y==1]), 100*np.mean(pr[y==0])  # missed NLOS, false alarm
print(f"== {which}: n={len(y)}")
print("Default threshold 0.5:  missed NLOS / false alarm")
for n,p in oof.items():
    m,fa=rates(p,0.5); print(f"  {n}: missed {m:.1f} %  false alarm {fa:.1f} %  acc {100*np.mean((p>=.5)==y):.1f} %")
pdiff=d.PWR_DIFF.to_numpy()
for t in (6,10):
    m,fa=rates(pdiff,t); print(f"  vendor rule literal, NLOS if PWR_DIFF >= {t} dB: missed {m:.1f} %  false alarm {fa:.1f} %")
def fnr_at_fpr(p,target):
    los=np.sort(p[y==0])[::-1]; k=int(np.floor(target/100*len(los)))
    thr=los[k] if k<len(los) else -np.inf  # predict NLOS if score > thr  -> FPR <= target
    pr=p>thr
    return 100*np.mean(~pr[y==1]), 100*np.mean(pr[y==0])
print("Missed NLOS at an equal false-alarm rate (threshold set on out-of-fold scores):")
for target in (5,10,20,29):
    row=[]
    for n,p in oof.items():
        m,fa=fnr_at_fpr(p,target); row.append(f"{n} {m:.1f} % (FA {fa:.1f})")
    print(f"  FA ~{target} %: "+" | ".join(row))
print("Missed NLOS at the vendor rule's own false-alarm rates:")
for t in (6,10):
    _,fa=rates(pdiff,t); row=[]
    for n in ('M2','M4'):
        m,fa2=fnr_at_fpr(oof[n],fa); row.append(f"{n} {m:.1f} % (FA {fa2:.1f})")
    mr,_=rates(pdiff,t)
    print(f"  rule {t} dB: rule missed {mr:.1f} % at FA {fa:.1f} % | "+" | ".join(row))
