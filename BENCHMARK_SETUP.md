# RQ2 latency benchmark (II2202 HT26, Group 4)

This folder measures how much latency a cloud-hosted NLOS classifier adds compared with running it locally. The procedure is the one described in Sections 2.3 and 2.4 of the report.

| Condition | What is timed |
|---|---|
| **L** | the model called directly in the client program (computation only) |
| **H** | the same REST service running on your laptop (`127.0.0.1`), which adds HTTP and JSON handling but no network |
| **C** | the same REST service on the Oracle Cloud VM, which adds the real network round trip |

There are 2 models (M2 logistic regression, M4 random forest) and 2 batch sizes (1 and 4 links), so each session runs 12 combinations. Each combination gets 50 warm-up requests and then 1 000 timed requests. You run three sessions at different times of day.

| File | Purpose |
|---|---|
| `features.py` | Cleaning rules and the ten features, the same as in the report. It reproduces the 41 977 clean samples. |
| `train_export.py` | Trains M2 and M4, checks the cross-validation numbers against the report, and writes `models/`. |
| `server.py`, `run_server.sh` | The REST service. It reports its own compute time in the `X-Compute-Ns` header. |
| `bench.py` | Runs one benchmark session and writes raw timings to `results/`. |
| `analyze.py` | Produces the summary, the LaTeX table, the ECDF figure and the numbers for the report. |

Total time: about 1–1.5 h of setup, then about 10 min per session.

---

## 1. Laptop setup (Windows, PowerShell)

Use **Python 3.12** on both the laptop and the VM. Ubuntu 24.04 ships with 3.12, and the pickled models must be loaded with the same scikit-learn version they were saved with.

```powershell
cd "...\II2202 HT26 Research Methodology and Scientific Writing\RQ2 Latency Benchmark"
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

Get the eWINE dataset with `git clone https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set.git`, or download the ZIP from that page and unpack it here.

## 2. Train and export the models (about 3 min)

```powershell
python train_export.py --data UWB-LOS-NLOS-Data-Set\dataset
```

Check the printout. It should show 41 977 clean samples and cross-validation accuracy close to the report: M0 about 67.3 %, M2 about 86.0 %, M4 about 88.1 %.

The script creates `models/` with the two models, `payload_pool.json`, `metadata.json` and `requirements-lock.txt`.

## 3. Create the Oracle Cloud VM (condition C)

1. In the OCI console, go to **Compute → Instances → Create instance**.
   - **Image:** Canonical Ubuntu 24.04.
   - **Shape:** an Always Free shape. Prefer **VM.Standard.A1.Flex (Ampere)** with 1 OCPU and 6 GB RAM. The random-forest service needs about 0.5 GB of RAM, and the 1 GB `VM.Standard.E2.1.Micro` may run out of memory.
   - Keep the public IPv4 address and download the SSH private key.
   - **Write down for the report:** shape, OCPUs/RAM, region, image.
2. **Open port 8000 only to your own IP.** Find your public IP with `curl.exe ifconfig.me` in PowerShell.
   - Go to the instance's **subnet → Security List → Add Ingress Rule**.
   - Source CIDR: `<your-ip>/32`, IP protocol: TCP, destination port: `8000`.
   - If you move between networks (home or eduroam), update this rule. Better still, run all sessions from the same network.
3. SSH in and open the OS firewall. Oracle's Ubuntu images block extra ports with iptables:
   ```bash
   ssh -i C:\path\to\key.key ubuntu@<VM-IP>
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
   sudo netfilter-persistent save
   sudo apt update && sudo apt install -y python3-venv tmux
   mkdir -p ~/nlos
   ```
4. Copy the service and models from the laptop (in PowerShell, in this folder):
   ```powershell
   scp -i C:\path\to\key.key server.py run_server.sh ubuntu@<VM-IP>:~/nlos/
   scp -i C:\path\to\key.key -r models ubuntu@<VM-IP>:~/nlos/
   ```
5. Install and start the service on the VM:
   ```bash
   cd ~/nlos && python3 -m venv .venv && source .venv/bin/activate
   pip install -r models/requirements-lock.txt
   chmod +x run_server.sh
   tmux new -s nlos       # keeps the server alive after you log out
   ./run_server.sh        # detach with Ctrl+B, then D
   ```
   If `./run_server.sh` fails with a `\r` error (Windows line endings), start the server directly with `uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log`.
   If `pip` cannot find a pinned version, your laptop's Python differs from the VM's. Use Python 3.12 on the laptop, rerun step 2 and copy `models/` again.
6. Test it from the laptop: `curl.exe http://<VM-IP>:8000/health` should return `"status":"ok"` and two model hashes.

## 4. Start the local service (condition H)

Open a **second** PowerShell window in this folder, keep it open during every session, and run:

```powershell
.venv\Scripts\Activate.ps1
uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log --log-level warning
```

## 5. Functional test (1 min)

```powershell
python bench.py --session 0 --quick --cloud-url http://<VM-IP>:8000
```

Session 0 is ignored by the analysis, so you can repeat this as often as needed.

## 6. The three measurement sessions

Run one session each in the morning, afternoon and evening. Use the same laptop and the same network each time, keep the laptop on mains power, close heavy apps, and turn any VPN off.

```powershell
python bench.py --session 1 --cloud-url http://<VM-IP>:8000
python bench.py --session 2 --cloud-url http://<VM-IP>:8000
python bench.py --session 3 --cloud-url http://<VM-IP>:8000
```

Each session takes about 5–10 min. The client checks that the laptop and the VM serve identical model files (SHA-256) before it starts. The client's details, the start and end times, the condition order and 100 TCP connection set-up times are stored in `results/meta_session*.json`.

## 7. Analyse

```powershell
python analyze.py
```

The results appear in `results/`:

- `summary.md`: the added latency Δ with its 95 % bootstrap CI, the split into computation, HTTP and network, the 99th percentile and the share of requests over 100 ms
- `table_latency_b1.tex`: Table 3 of the report
- `table_latency_b4.tex`: the same for batch size 4, for the appendix
- `fig_latency_ecdf.pdf`: Figure 4 of the report
- `summary.csv` and `per_session.csv`: all statistics, and per-session medians for the stability check

## 8. Put the results into the report

- **Table 3:** replace the `tabular` block of `tab:latency` with the contents of `table_latency_b1.tex`. Change the caption from "Placeholder" to the real description.
- **Figure 4:** copy `fig_latency_ecdf.pdf` to the report's `figures/` folder and replace the `\fbox{...}` placeholder with `\includegraphics[width=\linewidth]{figures/fig_latency_ecdf.pdf}`.
- **Section 3.3 and RQ2 in Section 4.1:** fill these in from `summary.md`. Is Δ dominated by the network (H2)? Is the random forest's computation visible? Does C stay within 100 ms at the 99th percentile? What does batch size 4 change?
- **Section 2.2:** fill in the VM shape, region and OS, and the laptop and network. The software versions are in `models/metadata.json`.
- **Section 2.3:** fill in the session dates and times from the `meta_session*.json` files.

## 9. Afterwards

Stop or terminate the VM and delete the port-8000 ingress rule. The service has no authentication, which is why it was only reachable from your IP.
