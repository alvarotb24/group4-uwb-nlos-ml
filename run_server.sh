#!/usr/bin/env bash
# Start the NLOS service on the cloud VM (condition C). Run from the folder that holds
# server.py and models/, with the virtual environment activated.
#   ./run_server.sh            -> listens on 0.0.0.0:8000
#   PORT=8080 ./run_server.sh  -> another port
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --no-access-log --log-level warning
