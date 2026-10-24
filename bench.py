"""Step 3 - run one benchmark session for RQ2 (run three sessions at different times of day).

    python bench.py --session 1 --cloud-url http://<VM-PUBLIC-IP>:8000

Conditions (report, Section 2.3):
  L  model called in this process (no HTTP)
  H  same service on this machine, http://127.0.0.1:8000 (start it first, see README)
  C  same service on the cloud VM

For every combination of condition x model (lr, rf) x batch size (1, 4) the client sends
--warmup untimed requests and then --n timed requests over one persistent connection.
The 12 combinations run in a random order that depends on the session number.
Before the runs, it records --tcp-samples TCP connection set-up times to each host as a
network reference.

Output: results/raw_session<k>_<timestamp>.csv and results/meta_session<k>_<timestamp>.json
Quick functional test: python bench.py --session 0 --quick --conditions L,H
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from urllib.parse import urlparse

import joblib
import numpy as np
import requests
import sklearn

MODEL_KEYS = ["lr", "rf"]
BATCHES = [1, 4]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tcp_connect_times(url: str, n: int) -> list[float]:
    """TCP connection set-up time (ms) to the service's host and port: one network RTT."""
    u = urlparse(url)
    port = u.port or (443 if u.scheme == "https" else 80)
    out = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        try:
            with socket.create_connection((u.hostname, port), timeout=5):
                out.append((time.perf_counter_ns() - t0) / 1e6)
        except OSError:
            out.append(float("nan"))
        time.sleep(0.05)
    return out


def check_service(url: str, local_hashes: dict) -> dict:
    with requests.Session() as s:
        s.trust_env = False
        r = s.get(url.rstrip("/") + "/health", timeout=10)
    r.raise_for_status()
    info = r.json()
    if info.get("models") != local_hashes:
        raise SystemExit(
            f"The models served at {url} differ from ./models (SHA-256 mismatch).\n"
            "Copy the same models/ folder to the server and restart it."
        )
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=int, required=True, help="session number (1, 2, 3)")
    ap.add_argument("--cloud-url", help="e.g. http://203.0.113.10:8000 (needed for condition C)")
    ap.add_argument("--local-url", default="http://127.0.0.1:8000")
    ap.add_argument("--conditions", default="L,H,C")
    ap.add_argument("--n", type=int, default=1000, help="timed requests per combination")
    ap.add_argument("--warmup", type=int, default=50, help="untimed warm-up requests per combination")
    ap.add_argument("--tcp-samples", type=int, default=100)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="functional test: n=50, warmup=5, tcp-samples=5")
    ap.add_argument("--part", default="1/1", help="run part k of N of the session's shuffled combinations (e.g. 2/4); "
                    "TCP samples are taken in part 1 only")
    args = ap.parse_args()
    if args.quick:
        args.n, args.warmup, args.tcp_samples = 50, 5, 5

    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    if "C" in conditions and not args.cloud_url:
        raise SystemExit("Condition C needs --cloud-url http://<VM-PUBLIC-IP>:8000")
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.models, "metadata.json")) as f:
        meta = json.load(f)
    if sklearn.__version__ != meta["versions"]["scikit-learn"]:
        print(f"WARNING: scikit-learn {sklearn.__version__} here, models saved with "
              f"{meta['versions']['scikit-learn']}.")
    with open(os.path.join(args.models, "payload_pool.json")) as f:
        pool = json.load(f)["rows"]
    local_models = {k: joblib.load(os.path.join(args.models, meta["models"][k]["file"])) for k in MODEL_KEYS}
    local_hashes = {k: sha256(os.path.join(args.models, meta["models"][k]["file"])) for k in MODEL_KEYS}
    for m in local_models.values():
        m.predict_proba(np.zeros((1, len(meta["features"]))))

    urls = {"H": args.local_url, "C": args.cloud_url}
    servers = {}
    for cond in conditions:
        if cond in urls:
            print(f"Checking {cond} service at {urls[cond]} ...")
            servers[cond] = check_service(urls[cond], local_hashes)

    tcp = {}
    for cond in (servers if args.part.split("/")[0] == "1" else []):
        print(f"Measuring {args.tcp_samples} TCP connection set-up times to {urls[cond]} ...")
        tcp[cond] = tcp_connect_times(urls[cond], args.tcp_samples)
        print(f"  median {np.nanmedian(tcp[cond]):.2f} ms")

    combos = [(c, m, b) for c in conditions for m in MODEL_KEYS for b in BATCHES]
    random.Random(1000 + args.session).shuffle(combos)
    k, n_parts = (int(x) for x in args.part.split("/"))
    if not 1 <= k <= n_parts:
        raise SystemExit("--part must be k/N with 1 <= k <= N")
    all_combos = list(combos)
    size = -(-len(combos) // n_parts)
    combos = combos[(k - 1) * size:k * size]
    print(f"Part {k}/{n_parts}: {len(combos)} of {len(all_combos)} combinations")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_p{k}of{n_parts}" if n_parts > 1 else ""
    raw_path = os.path.join(args.out, f"raw_session{args.session}{tag}_{stamp}.csv")
    meta_path = os.path.join(args.out, f"meta_session{args.session}{tag}_{stamp}.json")
    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    errors = 0
    offset = random.Random(args.session).randrange(len(pool))

    with open(raw_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["session", "condition", "model", "batch", "iter", "total_ns", "server_ns", "ok"])
        for k, (cond, key, b) in enumerate(combos, 1):
            print(f"[{k:2d}/{len(combos)}] condition {cond}, model {key}, batch {b} ...", flush=True)
            http = requests.Session() if cond in urls else None
            if http is not None:
                http.trust_env = False  # always connect directly, never through a system proxy
            url = (urls.get(cond) or "").rstrip("/") + f"/predict/{key}"
            model = local_models[key]
            for i in range(-args.warmup, args.n):
                start = (offset + (i + args.warmup) * b) % (len(pool) - b)
                rows = pool[start:start + b]
                server_ns = ""
                ok = 1
                if cond == "L":
                    t0 = time.perf_counter_ns()
                    model.predict_proba(np.asarray(rows, dtype=float))
                    total = time.perf_counter_ns() - t0
                else:
                    t0 = time.perf_counter_ns()
                    try:
                        r = http.post(url, json={"features": rows}, timeout=10)
                        r.raise_for_status()
                        r.json()
                        total = time.perf_counter_ns() - t0
                        server_ns = r.headers.get("X-Compute-Ns", "")
                    except requests.RequestException as e:
                        total, ok = "", 0
                        errors += 1
                        print(f"    request failed: {e}", file=sys.stderr)
                if i >= 0:
                    w.writerow([args.session, cond, key, b, i, total, server_ns, ok])
            if http is not None:
                http.close()

    session_meta = {
        "session": args.session,
        "started": started,
        "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "part": args.part,
        "order": [list(c) for c in combos],
        "full_order": [list(c) for c in all_combos],
        "n": args.n, "warmup": args.warmup,
        "errors": errors,
        "client": {
            "platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "requests": requests.__version__,
            "cpu_count": os.cpu_count(),
        },
        "urls": {c: urls[c] for c in servers},
        "servers": servers,
        "tcp_connect_ms": tcp,
    }
    with open(meta_path, "w") as f:
        json.dump(session_meta, f, indent=2)
    print(f"Done: {raw_path} ({errors} failed requests)")


if __name__ == "__main__":
    main()
