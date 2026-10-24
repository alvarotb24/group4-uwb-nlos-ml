#!/usr/bin/env bash
# usage: ./run_part.sh SESSION PART   (PART = k/12)
cd "$(dirname "$0")" && source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log --log-level warning > server_h.log 2>&1 &
SRV=$!
for i in $(seq 1 30); do sleep 1; curl -s --noproxy '*' -o /dev/null http://127.0.0.1:8000/health && break; done
SECONDS=0
python bench.py --session "$1" --part "$2" --cloud-url http://<cloud-VM-IP>:8000 --out results 2>&1 | grep -E "median|Part|condition|Done|failed|Error"
echo "part $2 took ${SECONDS}s"
kill $SRV
