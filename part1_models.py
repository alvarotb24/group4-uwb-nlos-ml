"""Part 1 (RQ1): reproduce the model comparison of Table 2 in the report.

    python part1_models.py --data PATH/TO/UWB-LOS-NLOS-Data-Set/dataset

Cleans the eWINE data (41 977 samples), computes the features with features.py and
evaluates the five classifiers with the same stratified 5-fold cross-validation
(seed 42) as the report:

  M0  vendor rule: logistic regression on PWR_DIFF only (= best single threshold)
  M1  logistic regression on the first 10 principal components of the
      peak-normalised CIR window (no hand-crafted features)
  M2  standardised logistic regression on the ten features
  M3  M2 + RANGE
  M4  random forest, 300 trees, on the ten features

Prints accuracy and ROC AUC (mean +/- SD over folds), the missed-NLOS rate and the
false-alarm rate, and writes them to results_part1/table2.csv.
Expected (report, Table 2): M0 67.3 %, M1 70.7 %, M2 86.0 %, M3 86.3 %, M4 88.1 %.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F


def cir_window(df: pd.DataFrame) -> np.ndarray:
    """Peak-normalised CIR window, 5 samples before to 100 after the first path."""
    n_acc = df["RXPACC"].to_numpy(float)
    cir = df[[f"CIR{i}" for i in range(F.N_CIR)]].to_numpy(float) / n_acc[:, None]
    start = np.floor(df["FP_IDX"].to_numpy(float)).astype(int) - F.WIN_BEFORE
    idx = np.clip(start[:, None] + np.arange(F.WIN_BEFORE + F.WIN_AFTER + 1)[None, :], 0, F.N_CIR - 1)
    win = np.take_along_axis(cir, idx, axis=1)
    return win / win.max(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with uwb_dataset_part1..7.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_part1")
    args = ap.parse_args()

    raw = F.load_raw(args.data)
    df, cleaning = F.clean(raw)
    del raw
    print("Cleaning:", cleaning)
    feats = F.compute_features(df)
    y = feats["NLOS"].to_numpy()
    s = args.seed
    models = {
        "M0 vendor rule (PWR_DIFF)": (lambda: make_pipeline(StandardScaler(), LogisticRegression()),
                                      feats[["PWR_DIFF"]].to_numpy()),
        "M1 CIR shape (10 PCs)": (lambda: make_pipeline(PCA(n_components=10), LogisticRegression(max_iter=2000)),
                                  cir_window(df)),
        "M2 logistic, 10 features": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                     feats[F.FEATURES].to_numpy()),
        "M3 M2 + RANGE": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                          feats[F.FEATURES + ["RANGE"]].to_numpy()),
        "M4 random forest, 300 trees": (lambda: RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1),
                                        feats[F.FEATURES].to_numpy()),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
    rows = []
    for name, (factory, X) in models.items():
        acc, auc, miss, fa = [], [], [], []
        for tr, te in skf.split(X, y):
            p = factory().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            pred = p >= 0.5
            acc.append(accuracy_score(y[te], pred))
            auc.append(roc_auc_score(y[te], p))
            miss.append(np.mean(~pred[y[te] == 1]))
            fa.append(np.mean(pred[y[te] == 0]))
        row = {"model": name,
               "accuracy_%": 100 * np.mean(acc), "accuracy_sd": 100 * np.std(acc, ddof=1),
               "auc": np.mean(auc), "auc_sd": np.std(auc, ddof=1),
               "missed_nlos_%": 100 * np.mean(miss), "false_alarm_%": 100 * np.mean(fa)}
        rows.append(row)
        print(f"  {name:30s} acc {row['accuracy_%']:5.1f} ± {row['accuracy_sd']:.1f} %  "
              f"AUC {row['auc']:.3f} ± {row['auc_sd']:.3f}  missed NLOS {row['missed_nlos_%']:.1f} %  "
              f"false alarms {row['false_alarm_%']:.1f} %", flush=True)
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(args.out, "table2.csv"), index=False)
    print(f"Written to {args.out}/table2.csv")


if __name__ == "__main__":
    main()
