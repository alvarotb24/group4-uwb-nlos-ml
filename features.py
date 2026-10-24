"""Feature extraction for the eWINE UWB LOS/NLOS dataset (II2202 HT26, Group 4).

Reproduces the cleaning rules and the ten classifier features described in
Section 2 of the report (and in the quantitative exercise):

  FP_POW, RX_POW, FP_SNR, NOISE_STD, RXPACC,
  MEAN_DELAY, RMS_DS, PEAK_DELAY, FP_PEAK_RATIO, KURTOSIS

plus PWR_DIFF (vendor rule) and RANGE (potential confounder).

Checked against the eWINE data: the cleaning removes exactly 23 samples
(41 977 left), and the power, SNR, noise, RXPACC, PEAK_DELAY and FP_PEAK_RATIO
statistics match Table 1 of the report. MEAN_DELAY and KURTOSIS differ from the
original exercise script by a few tenths, which does not matter for RQ2.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

# DW1000 User Manual: constant A for a 64 MHz pulse repetition frequency (dBm).
A_PRF64 = 121.74
# CIR window around the detected first path, in CIR samples (~1 ns each).
WIN_BEFORE = 5
WIN_AFTER = 100
# Samples after the first-path index that still count as the first-path lobe.
# (FP_LOBE = 2 reproduces the FP_PEAK_RATIO medians/IQRs of the exercise exactly.)
FP_LOBE = 2
N_CIR = 1016

FEATURES = [
    "FP_POW", "RX_POW", "FP_SNR", "NOISE_STD", "RXPACC",
    "MEAN_DELAY", "RMS_DS", "PEAK_DELAY", "FP_PEAK_RATIO", "KURTOSIS",
]


def load_raw(dataset_dir: str) -> pd.DataFrame:
    """Read and concatenate uwb_dataset_part*.csv from the eWINE 'dataset' folder."""
    files = sorted(glob.glob(os.path.join(dataset_dir, "uwb_dataset_part*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No uwb_dataset_part*.csv files in {dataset_dir!r}. Point --data at the "
            "'dataset' folder of the eWINE repository."
        )
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove physically invalid samples (same rules as the report, Section 2.2)."""
    zero_range = df["RANGE"] <= 0
    zero_power = df["CIR_PWR"] <= 0
    late_fp = df["FP_IDX"] > 800
    drop = zero_range | zero_power | late_fp
    report = {
        "n_raw": int(len(df)),
        "removed_zero_range": int(zero_range.sum()),
        "removed_zero_cir_power": int((zero_power & ~zero_range).sum()),
        "removed_fp_idx_gt_800": int((late_fp & ~zero_range & ~zero_power).sum()),
        "removed_total": int(drop.sum()),
    }
    out = df.loc[~drop].reset_index(drop=True)
    report["n_clean"] = int(len(out))
    report["n_los"] = int((out["NLOS"] == 0).sum())
    report["n_nlos"] = int((out["NLOS"] == 1).sum())
    return out, report


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with NLOS, the ten features, PWR_DIFF and RANGE."""
    n_acc = df["RXPACC"].to_numpy(float)
    f1, f2, f3 = (df[c].to_numpy(float) for c in ("FP_AMP1", "FP_AMP2", "FP_AMP3"))
    cir_pwr = df["CIR_PWR"].to_numpy(float)

    fp_pow = 10 * np.log10((f1**2 + f2**2 + f3**2) / n_acc**2) - A_PRF64
    rx_pow = 10 * np.log10(cir_pwr * 2**17 / n_acc**2) - A_PRF64
    fp_snr = 20 * np.log10(f2 / df["STDEV_NOISE"].to_numpy(float))

    # CIR magnitude normalised to one preamble symbol (dataset documentation).
    cir_cols = [f"CIR{i}" for i in range(N_CIR)]
    cir = df[cir_cols].to_numpy(float) / n_acc[:, None]

    fp = df["FP_IDX"].to_numpy(float)
    start = np.floor(fp).astype(int) - WIN_BEFORE
    idx = start[:, None] + np.arange(WIN_BEFORE + WIN_AFTER + 1)[None, :]
    idx = np.clip(idx, 0, N_CIR - 1)
    win = np.take_along_axis(cir, idx, axis=1)

    power = win**2
    tau = idx - fp[:, None]  # delay relative to the detected first path (ns)
    p_sum = power.sum(axis=1)
    mean_delay = (tau * power).sum(axis=1) / p_sum
    rms_ds = np.sqrt(((tau - mean_delay[:, None]) ** 2 * power).sum(axis=1) / p_sum)

    k_peak = win.argmax(axis=1)
    peak_delay = (idx[np.arange(len(idx)), k_peak] - np.floor(fp)).astype(float)
    fp_amp = win[:, WIN_BEFORE:WIN_BEFORE + FP_LOBE + 1].max(axis=1)
    fp_peak_ratio = 20 * np.log10(win.max(axis=1) / fp_amp)
    kurt = kurtosis(win, axis=1, fisher=False)

    feats = pd.DataFrame({
        "NLOS": df["NLOS"].astype(int).to_numpy(),
        "FP_POW": fp_pow,
        "RX_POW": rx_pow,
        "FP_SNR": fp_snr,
        "NOISE_STD": df["STDEV_NOISE"].to_numpy(float),
        "RXPACC": n_acc,
        "MEAN_DELAY": mean_delay,
        "RMS_DS": rms_ds,
        "PEAK_DELAY": peak_delay,
        "FP_PEAK_RATIO": fp_peak_ratio,
        "KURTOSIS": kurt,
        "PWR_DIFF": rx_pow - fp_pow,
        "RANGE": df["RANGE"].to_numpy(float),
    })
    return feats


def load_features(dataset_dir: str) -> tuple[pd.DataFrame, dict]:
    raw = load_raw(dataset_dir)
    cleaned, report = clean(raw)
    return compute_features(cleaned), report
