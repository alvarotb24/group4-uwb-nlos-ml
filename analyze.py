"""Step 4 - analyse all benchmark sessions for RQ2.

    python analyze.py                # reads results/raw_session*.csv

Writes to results/:
  summary.csv              one row per condition x model x batch (pooled over sessions)
  per_session.csv          medians per session (stability check)
  table_latency_b1.tex     tabular for Table 3 of the report (batch size 1)
  table_latency_b4.tex     the same for batch size 4 (appendix)
  fig_latency_ecdf.pdf/png ECDF figure for the report (batch size 1)
  summary.md               the key numbers in sentences, ready to adapt for the report

Definitions (report, Section 2.4):
  added latency   Delta = median(C) - median(L), with a 95 % percentile-bootstrap CI
  computation     median server-measured compute time in C
  HTTP overhead   median(total - server compute) in H   (HTTP + JSON, no network)
  network         median(total - server compute) in C  minus  the HTTP overhead
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

COND_NAMES = {"L": "L: in-process", "H": "H: REST, localhost", "C": "C: REST, cloud VM"}
MODEL_NAMES = {"lr": "M2 (LR)", "rf": "M4 (RF)"}
# Okabe-Ito blue / bluish green / vermilion (checked for colour-vision deficiency);
# line styles carry the same identity in grayscale.
STYLE = {"L": ("#0072B2", "-"), "H": ("#009E73", "--"), "C": ("#D55E00", "-.")}
BUDGET_MS = 100.0  # illustrative 10 Hz update rate


def bootstrap_median_diff(a: np.ndarray, b: np.ndarray, reps: int, rng) -> tuple[float, float]:
    """95 % percentile-bootstrap CI for median(a) - median(b), resampling each group."""
    diffs = np.empty(reps)
    chunk = 250
    for s in range(0, reps, chunk):
        k = min(chunk, reps - s)
        ma = np.median(a[rng.integers(0, len(a), (k, len(a)))], axis=1)
        mb = np.median(b[rng.integers(0, len(b), (k, len(b)))], axis=1)
        diffs[s:s + k] = ma - mb
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    if x < 1:
        return f"{x:.3f}"
    if x < 10:
        return f"{x:.2f}"
    return f"{x:.1f}"


def latex_table(summary: pd.DataFrame, batch: int) -> str:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Condition & Median & IQR & 95th pct. & 99th pct. & Server compute \\",
        r"\midrule",
    ]
    first = True
    for key in ["lr", "rf"]:
        if not first:
            lines.append(r"\addlinespace")
        first = False
        for j, cond in enumerate(["L", "H", "C"]):
            row = summary[(summary.model == key) & (summary.batch == batch) & (summary.condition == cond)]
            label = MODEL_NAMES[key] if j == 0 else ""
            if row.empty:
                lines.append(f"{label} & {COND_NAMES[cond]} & -- & -- & -- & -- & -- \\\\")
                continue
            r = row.iloc[0]
            server = "n/a" if cond == "L" else fmt(r.server_median_ms)
            lines.append(f"{label} & {COND_NAMES[cond]} & {fmt(r.median_ms)} & {fmt(r.iqr_ms)} & "
                         f"{fmt(r.p95_ms)} & {fmt(r.p99_ms)} & {server} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def ecdf_figure(df: pd.DataFrame, path_base: str, batch: int = 1) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#6b7280", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharex=True, sharey=True, constrained_layout=True)
    label_y = {"L": 0.75, "H": 0.5, "C": 0.25}  # staggered so direct labels never collide
    for ax, key in zip(axes, ["lr", "rf"]):
        for cond in ["L", "H", "C"]:
            x = np.sort(df[(df.model == key) & (df.batch == batch) & (df.condition == cond)].ms.to_numpy())
            if len(x) == 0:
                continue
            y = np.arange(1, len(x) + 1) / len(x)
            color, ls = STYLE[cond]
            ax.plot(x, y, color=color, ls=ls, lw=1.8, label=COND_NAMES[cond])
            yq = label_y[cond]
            ax.annotate(cond, xy=(np.quantile(x, yq), yq), xytext=(5, -3), textcoords="offset points",
                        fontsize=9, color="#1f2937", fontweight="bold")
        ax.axvline(BUDGET_MS, color="#9ca3af", lw=0.8, ls=":")
        ax.text(BUDGET_MS, 1.02, "100 ms", ha="center", va="bottom", fontsize=7, color="#4b5563")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{MODEL_NAMES[key]}, batch size {batch}", fontsize=9, pad=12)
        ax.set_xlabel("Per-request latency (ms, log scale)")
        ax.grid(True, which="major", color="#e5e7eb", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cumulative share of requests")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.savefig(path_base + ".pdf")
    fig.savefig(path_base + ".png", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--reps", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-test", action="store_true", help="also use session 0 (functional tests)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "raw_session*.csv")))
    if not files:
        raise SystemExit(f"No raw_session*.csv files in {args.results}/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if not args.keep_test:
        df = df[df.session > 0]
    if df.empty:
        raise SystemExit("Only test runs (session 0) found. Run bench.py with --session 1, 2, 3.")
    n_failed = int((df.ok == 0).sum())
    df = df[df.ok == 1].copy()
    df["ms"] = df.total_ns / 1e6
    df["server_ms"] = df.server_ns / 1e6
    df["overhead_ms"] = df.ms - df.server_ms
    sessions = sorted(int(v) for v in df.session.unique())
    print(f"{len(files)} file(s), sessions {sessions}, {len(df)} timed requests, {n_failed} failed")

    rows = []
    for (cond, key, b), g in df.groupby(["condition", "model", "batch"]):
        x = g.ms.to_numpy()
        q25, q50, q75, q95, q99 = np.percentile(x, [25, 50, 75, 95, 99])
        rows.append({
            "condition": cond, "model": key, "batch": int(b), "n": len(x),
            "median_ms": q50, "q25_ms": q25, "q75_ms": q75, "iqr_ms": q75 - q25,
            "p95_ms": q95, "p99_ms": q99, "mean_ms": float(x.mean()),
            "server_median_ms": float(np.nanmedian(g.server_ms)) if cond != "L" else np.nan,
            "overhead_median_ms": float(np.nanmedian(g.overhead_ms)) if cond != "L" else np.nan,
            "share_over_budget": float(np.mean(x > BUDGET_MS)),
        })
    summary = pd.DataFrame(rows).sort_values(["batch", "model", "condition"])
    summary.to_csv(os.path.join(args.results, "summary.csv"), index=False)

    per_session = (df.groupby(["session", "condition", "model", "batch"]).ms.median()
                   .unstack("session").round(3))
    per_session.to_csv(os.path.join(args.results, "per_session.csv"))

    for b in sorted(summary.batch.unique()):
        with open(os.path.join(args.results, f"table_latency_b{b}.tex"), "w") as f:
            f.write(latex_table(summary, b))
    if (df.batch == 1).any():
        ecdf_figure(df, os.path.join(args.results, "fig_latency_ecdf"), batch=1)

    rng = np.random.default_rng(args.seed)
    lines = ["# RQ2 latency summary", "",
             f"Sessions: {sessions}; timed requests: {len(df)}; failed requests: {n_failed}.", ""]
    tcp = {}
    for mf in sorted(glob.glob(os.path.join(args.results, "meta_session*.json"))):
        with open(mf) as f:
            m = json.load(f)
        if int(m["session"]) not in sessions:
            continue
        for cond, vals in m.get("tcp_connect_ms", {}).items():
            tcp.setdefault(cond, {})[int(m["session"])] = float(np.nanmedian(vals))
    for cond, d in tcp.items():
        lines.append(f"- TCP connection set-up time to {cond}, median per session: "
                     + ", ".join(f"session {s}: {v:.2f} ms" for s, v in sorted(d.items())))
    lines.append("")

    def med(cond, key, b, col="ms"):
        g = df[(df.condition == cond) & (df.model == key) & (df.batch == b)]
        return g, (float(np.nanmedian(g[col])) if len(g) else np.nan)

    for b in sorted(df.batch.unique()):
        for key in ["lr", "rf"]:
            gl, ml = med("L", key, b)
            gh, mh = med("H", key, b)
            gc, mc = med("C", key, b)
            lines.append(f"## {MODEL_NAMES[key]}, batch size {b}")
            lines.append(f"- Medians: L {fmt(ml)} ms, H {fmt(mh)} ms, C {fmt(mc)} ms.")
            if len(gl) and len(gc):
                lo, hi = bootstrap_median_diff(gc.ms.to_numpy(), gl.ms.to_numpy(), args.reps, rng)
                delta = mc - ml
                lines.append(f"- Added latency of the cloud service: Delta = {fmt(delta)} ms "
                             f"(95 % CI {fmt(lo)} to {fmt(hi)} ms).")
                comp_c = float(np.nanmedian(gc.server_ms))
                net = float(np.nanmedian(gc.overhead_ms))
                http_h = float(np.nanmedian(gh.overhead_ms)) if len(gh) else np.nan
                parts = [f"computation on the VM {fmt(comp_c)} ms (local in-process: {fmt(ml)} ms)"]
                if np.isfinite(http_h):
                    parts.append(f"HTTP/JSON overhead {fmt(http_h)} ms (from H)")
                    parts.append(f"network {fmt(net - http_h)} ms")
                    if delta > 0:
                        parts.append(f"network share of Delta about {100 * (net - http_h) / delta:.0f} %")
                else:
                    parts.append(f"HTTP + network {fmt(net)} ms (run H to separate them)")
                lines.append("- Split (medians, approximately additive): " + "; ".join(parts) + ".")
                p99 = float(np.percentile(gc.ms, 99))
                lines.append(f"- 99th percentile in C: {fmt(p99)} ms; "
                             f"{100 * np.mean(gc.ms > BUDGET_MS):.1f} % of C requests exceed {BUDGET_MS:.0f} ms.")
            if len(gl) and len(gh):
                lo, hi = bootstrap_median_diff(gh.ms.to_numpy(), gl.ms.to_numpy(), args.reps, rng)
                lines.append(f"- Local REST service vs in-process: {fmt(mh - ml)} ms (95 % CI {fmt(lo)} to {fmt(hi)} ms).")
            lines.append("")
    lines.append("Per-session medians (stability check) are in per_session.csv.")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.results, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    print(f"Wrote summary.csv, per_session.csv, table_latency_b*.tex, fig_latency_ecdf.pdf/png to {args.results}/")


if __name__ == "__main__":
    main()
