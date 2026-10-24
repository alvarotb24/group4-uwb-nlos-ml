# Code and trained models: NLOS detection for UWB ranging (II2202 HT26, Group 4)

This package contains the code, the trained models and the raw latency measurements behind the report

> Á. Tévar Becerra and Y. P. Patil, *Machine-Learning NLOS Detection for UWB Ranging: Classification Accuracy and the Latency Cost of Cloud-Hosted Inference*, II2202 Research Methodology and Scientific Writing, KTH, 2026.

Everything needed to check the numbers in the report is here, except the dataset itself, which is public and is downloaded separately (see step 1).

## Contents

| File or folder | What it is | Report |
|---|---|---|
| `features.py` | Cleaning rules (42 000 → 41 977 samples) and the ten features | Sections 2.2–2.3 |
| `part1_models.py` | Cross-validation of the five classifiers M0–M4 | Table 2 |
| `sens_prep.py`, `sens_run.py` | Sensitivity analyses S1 (no 1 536-symbol preamble) and S2 (matched distance) | Table 3, Section 3.3 |
| `opp_q2.py` | Missed-NLOS rate at an equal false-alarm rate | Section 3.2 |
| `train_export.py` | Trains M2 and M4 on all clean samples and exports them to `models/` | Section 2.3, Part 2 |
| `models/` | **The trained models** used in the latency benchmark, with their SHA-256 hashes and exact library versions | Part 2 |
| `server.py`, `run_server.sh` | The REST service (FastAPI + Uvicorn) used in conditions H and C | Section 2.3 |
| `bench.py`, `run_part.sh` | The benchmark client (conditions L, H and C) | Section 2.3 |
| `analyze.py` | Latency statistics, Table 4 and Figure 4 | Section 3.4 |
| `results/` | Raw timings and metadata of the three measurement sessions (8 October 2026; session 0 is a functional test) | Section 3.4 |
| `results_part1/table2.csv` | Output of `part1_models.py` from a clean re-run (23 October 2026) | Table 2 |
| `results_part1/sensitivity_output.txt` | Output of `sens_run.py` for the full data, S1 and S2 | Table 3 |
| `results_part1/error_rates_equal_false_alarm.txt` | Output of `opp_q2.py` (full data and S2) | Section 3.2 |
| `data/ewine_features_clean.csv` | The 41 977 cleaned samples with the label, the ten features, PWR_DIFF, RANGE and the preamble length | Sections 2.2–2.3 |
| `models/m2_coefficients.csv` | Model M2 written out as numbers (scaling, weights and intercept) | Appendix A |
| `BENCHMARK_SETUP.md` | Step-by-step set-up of the cloud VM and the benchmark | Section 2.3 |

## 1. Environment and data

Use **Python 3.12**. The exact versions used for the models are in `models/requirements-lock.txt` (scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1). The pickled models must be loaded with the same scikit-learn version.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r models/requirements-lock.txt
git clone https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set.git
```

The eWINE UWB LOS and NLOS Data Set (K. Bregar, Jožef Stefan Institute) is published under CC-BY 4.0 and is not redistributed here. We used commit `9a7e755` (14 February 2017) of that repository.

## 2. Part 1 (RQ1): classification

```bash
python part1_models.py --data UWB-LOS-NLOS-Data-Set/dataset      # Table 2, about 2 min
python sens_prep.py                                               # writes feats_sens.pkl
python sens_run.py all     M0,M2,M4                               # full data
python sens_run.py no1536  M0,M2,M4                               # S1
python sens_run.py matched M0,M2,M4                               # S2 (Table 3)
python opp_q2.py all                                              # missed NLOS at equal false-alarm rate
```

All models use stratified 5-fold cross-validation with seed 42. A clean re-run of `part1_models.py` gives:

| Model | Accuracy | ROC AUC | Missed NLOS | False alarms |
|---|---|---|---|---|
| M0 vendor rule (logistic on PWR_DIFF) | 67.3 % | 0.727 | 41.2 % | 24.3 % |
| M1 CIR shape (10 principal components) | 70.7 % | 0.781 | 36.2 % | 22.4 % |
| M2 logistic regression, 10 features | 86.0 % | 0.923 | 17.6 % | 10.4 % |
| M3 M2 + RANGE | 86.3 % | 0.923 | 17.7 % | 9.7 % |
| M4 random forest, 300 trees | 88.1 % | 0.948 | 15.0 % | 8.9 % |

These match Table 2 of the report. The only differences are in the last digit: the random forest's AUC is 0.948 here and 0.949 in the report, and two fold-to-fold standard deviations differ by 0.1 because of rounding.

## The data we used

`data/ewine_features_clean.csv` holds one row per measurement after cleaning (41 977 rows): `NLOS` (1 = NLOS, 0 = LOS), the ten features used by the classifiers, `PWR_DIFF` (the vendor rule's indicator, RX_POW − FP_POW), `RANGE` (measured distance in metres) and `PREAM_LEN` (preamble length in symbols). It is derived from the eWINE UWB LOS and NLOS Data Set by K. Bregar, Jožef Stefan Institute (CC BY 4.0, https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set), with the cleaning rules and feature definitions in `features.py`. With this file alone, `sens_run.py` and `opp_q2.py` can be re-run without downloading the raw data.

## 3. The trained models

| File | Model | SHA-256 |
|---|---|---|
| `models/m2_logreg.joblib` | M2: `StandardScaler` + `LogisticRegression(max_iter=2000)` | `b18860bc89932526e9073ea1ed03ae2a578ba17d053388cd64e8f7cd6b18df55` |
| `models/m4_rf.joblib` | M4: `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)` | `35d815d97d9504eddddb37b94bc5ef98b0f3de623b82d5cc1a1d2b6c4bf36069` |

Both were trained on all 41 977 clean samples with the ten features in this order: FP_POW, RX_POW, FP_SNR, NOISE_STD, RXPACC, MEAN_DELAY, RMS_DS, PEAK_DELAY, FP_PEAK_RATIO, KURTOSIS. To use them:

```python
import joblib, json
m2 = joblib.load("models/m2_logreg.joblib")
rows = json.load(open("models/payload_pool.json"))["rows"][:4]   # four feature vectors from the dataset
print(m2.predict_proba(rows)[:, 1])                               # probability of NLOS
```

`python train_export.py --data UWB-LOS-NLOS-Data-Set/dataset` retrains both models and rewrites `models/`, including `metadata.json` with the cleaning counts, versions and hashes.

**M2 as plain numbers.** `models/m2_coefficients.csv` lists, for each feature, the mean and standard deviation used for scaling and the logistic-regression weight. The probability of NLOS is p = 1 / (1 + exp(−(b + Σ w_i · (x_i − mean_i) / std_i))), with b = 0.4296. The whole model is these 31 numbers.

**M4.** `models/m4_rf.joblib` is a 33 MB file with 300 trees (2.15 million nodes in total). Its SHA-256 is given above.


## 4. Part 2 (RQ2): latency benchmark

The service is started with `run_server.sh` (or `uvicorn server:app`), and `bench.py` times the three conditions L (in-process), H (service on localhost) and C (service on the cloud VM). `BENCHMARK_SETUP.md` describes the full procedure, including the Oracle Cloud VM. The raw timings of the three sessions are in `results/`, and `python analyze.py` recomputes the statistics, Table 4 and Figure 4 from them.

The measurements used an Oracle Cloud Always Free VM (VM.Standard.E2.1.Micro, 1/8 OCPU, 1 GB RAM, Stockholm region). The VM has been shut down, and its public IP address is replaced by `<cloud-VM-IP>` in the scripts and metadata.

## Use of AI tools

As stated in Section 2.5 of the report, the code was written with the assistance of Claude (Anthropic). The authors checked the outputs against the report, and all methodological decisions are theirs.
