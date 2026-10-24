"""Step 1 - train the two models used in RQ2 and export everything the benchmark needs.

    python train_export.py --data PATH/TO/UWB-LOS-NLOS-Data-Set/dataset

Writes to ./models/:
  m2_logreg.joblib       M2: standardised logistic regression on the 10 features
  m4_rf.joblib           M4: random forest, 300 trees, on the 10 features
  payload_pool.json      feature vectors drawn from the dataset (request payloads)
  metadata.json          versions, seed, cleaning counts, CV check, file hashes, sizes
  requirements-lock.txt  exact library versions -> install these on the cloud VM
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
from importlib import metadata as importlib_metadata

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, load_features

MODEL_FILES = {"lr": "m2_logreg.joblib", "rf": "m4_rf.joblib"}


def make_models(seed: int) -> dict:
    return {
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        # n_jobs=1: single-threaded prediction, so timing is not affected by thread pools.
        "rf": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
    }


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version(pkg: str) -> str | None:
    try:
        return importlib_metadata.version(pkg)
    except importlib_metadata.PackageNotFoundError:
        return None


def cross_validate(feats, seed: int) -> dict:
    """Sanity check against Table 2 of the report (stratified 5-fold CV)."""
    y = feats["NLOS"].to_numpy()
    X10 = feats[FEATURES].to_numpy()
    Xrule = feats[["PWR_DIFF"]].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = {
        "M0 vendor rule (logistic on PWR_DIFF)": (lambda: make_pipeline(StandardScaler(), LogisticRegression()), Xrule),
        "M2 logistic regression, 10 features": (lambda: make_models(seed)["lr"], X10),
        "M4 random forest, 10 features": (lambda: make_models(seed)["rf"], X10),
    }
    out = {}
    for name, (factory, X) in candidates.items():
        acc, auc, miss = [], [], []
        for tr, te in skf.split(X, y):
            m = factory().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            pred = (p >= 0.5).astype(int)
            acc.append(accuracy_score(y[te], pred))
            auc.append(roc_auc_score(y[te], p))
            miss.append(np.mean(pred[y[te] == 1] == 0))
        out[name] = {
            "accuracy_mean": float(np.mean(acc)), "accuracy_sd": float(np.std(acc, ddof=1)),
            "auc_mean": float(np.mean(auc)), "auc_sd": float(np.std(auc, ddof=1)),
            "missed_nlos_rate": float(np.mean(miss)),
        }
        print(f"  {name:40s} acc {100*np.mean(acc):5.1f} ± {100*np.std(acc, ddof=1):.1f} %   "
              f"AUC {np.mean(auc):.3f} ± {np.std(auc, ddof=1):.3f}   missed NLOS {100*np.mean(miss):.1f} %")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with uwb_dataset_part1..7.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", type=int, default=4000, help="number of feature vectors in the payload pool")
    ap.add_argument("--skip-cv", action="store_true", help="skip the cross-validation sanity check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading and cleaning the eWINE dataset ...")
    feats, cleaning = load_features(args.data)
    print(f"  {cleaning}")
    if cleaning["n_clean"] != 41977:
        print("  WARNING: expected 41 977 clean samples (report, Section 2.2).")

    cv = None
    if not args.skip_cv:
        print("Cross-validation sanity check (report: M0 67.3 %, M2 86.0 %, M4 88.1 %) ...")
        cv = cross_validate(feats, args.seed)

    print("Training M2 and M4 on all clean samples ...")
    X = feats[FEATURES].to_numpy()
    y = feats["NLOS"].to_numpy()
    files = {}
    for key, model in make_models(args.seed).items():
        model.fit(X, y)
        path = os.path.join(args.out, MODEL_FILES[key])
        joblib.dump(model, path, compress=3)
        files[key] = {"file": MODEL_FILES[key], "sha256": sha256(path), "bytes": os.path.getsize(path)}
        print(f"  {key}: {path} ({files[key]['bytes'] / 1e6:.1f} MB)")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(feats), size=min(args.pool, len(feats)), replace=False)
    pool = np.round(X[pick], 6).tolist()
    with open(os.path.join(args.out, "payload_pool.json"), "w") as f:
        json.dump({"features": FEATURES, "rows": pool}, f)

    versions = {pkg: version(pkg) for pkg in ["numpy", "scipy", "scikit-learn", "joblib", "pandas"]}
    meta = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "seed": args.seed,
        "features": FEATURES,
        "cleaning": cleaning,
        "models": files,
        "model_params": {
            "lr": "StandardScaler + LogisticRegression(max_iter=2000), other settings default",
            "rf": "RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1), other settings default",
        },
        "payload_pool_size": len(pool),
        "cv_check": cv,
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Exact versions for the cloud VM: pickled scikit-learn models must be loaded
    # with the same scikit-learn (and compatible numpy) version they were saved with.
    lock = [f"{pkg}=={versions[pkg]}" for pkg in ["numpy", "scipy", "scikit-learn", "joblib"]]
    lock += [f"{pkg}=={version(pkg)}" if version(pkg) else pkg for pkg in ["fastapi", "uvicorn"]]
    with open(os.path.join(args.out, "requirements-lock.txt"), "w") as f:
        f.write("# Install on the cloud VM with the same Python minor version as this machine "
                f"(Python {platform.python_version()}).\n" + "\n".join(lock) + "\n")
    print(f"Done. Python {platform.python_version()}, scikit-learn {versions['scikit-learn']}. "
          f"Files are in ./{args.out}/")


if __name__ == "__main__":
    sys.exit(main())
