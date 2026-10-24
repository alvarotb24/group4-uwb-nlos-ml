"""NLOS classification REST service (conditions H and C of RQ2).

Run with:   uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
            (or ./run_server.sh)

Endpoints
  GET  /health          -> status, model SHA-256 hashes, library versions
  POST /predict/{model} -> model in {"lr", "rf"}; body {"features": [[10 floats], ...]}
                           returns {"p_nlos": [...]} and the header X-Compute-Ns with the
                           time spent in predict_proba (nanoseconds, measured on the server).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MAX_BATCH = 64

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    META = json.load(f)

trained_with = META["versions"]["scikit-learn"]
if sklearn.__version__ != trained_with and not os.environ.get("ALLOW_VERSION_MISMATCH"):
    raise SystemExit(
        f"scikit-learn {sklearn.__version__} is installed, but the models were saved with "
        f"{trained_with}. Install models/requirements-lock.txt (or set ALLOW_VERSION_MISMATCH=1)."
    )


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MODELS = {}
HASHES = {}
for key, info in META["models"].items():
    path = os.path.join(MODEL_DIR, info["file"])
    MODELS[key] = joblib.load(path)
    HASHES[key] = _sha256(path)

N_FEATURES = len(META["features"])

# One prediction per model at start-up so that lazy initialisation is not timed.
for _m in MODELS.values():
    _m.predict_proba(np.zeros((1, N_FEATURES)))


class PredictRequest(BaseModel):
    features: list[list[float]]


app = FastAPI(title="UWB NLOS classifier (II2202 Group 4)")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "models": HASHES,
        "features": META["features"],
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


@app.post("/predict/{model}")
def predict(model: str, req: PredictRequest, response: Response) -> dict:
    clf = MODELS.get(model)
    if clf is None:
        raise HTTPException(status_code=404, detail=f"unknown model {model!r}; use one of {sorted(MODELS)}")
    if not 1 <= len(req.features) <= MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"send between 1 and {MAX_BATCH} feature vectors")
    t0 = time.perf_counter_ns()
    X = np.asarray(req.features, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise HTTPException(status_code=422, detail=f"each feature vector needs {N_FEATURES} values")
    p = clf.predict_proba(X)[:, 1]
    compute_ns = time.perf_counter_ns() - t0
    response.headers["X-Compute-Ns"] = str(compute_ns)
    response.headers["Server-Timing"] = f"compute;dur={compute_ns / 1e6:.4f}"
    return {"p_nlos": p.tolist()}
