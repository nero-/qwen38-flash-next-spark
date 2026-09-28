#!/usr/bin/env bash
# Single-Spark controller, run on the Spark: start|stop|status|wait|logs
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$HOME/$(python3 "$HERE/config.py" remote_root)"
API_PORT="$(python3 "$HERE/config.py" api_port)"
case "${1:-status}" in
  start)
    python3 "$ROOT/launch.py" start
    echo 'Started. Run wait or logs to follow startup.'
    ;;
  stop) python3 "$ROOT/launch.py" stop ;;
  status)
    result=0
    echo "Selected profile: $(cat "$ROOT/selected-profile.txt" 2>/dev/null || python3 "$HERE/config.py" default_profile)"
    python3 "$ROOT/launch.py" status || result=$?
    curl --max-time 5 -s -o /dev/null -w 'API HTTP %{http_code}\n' "http://127.0.0.1:$API_PORT/health" || result=$?
    exit "$result"
    ;;
  wait)
    deadline=$((SECONDS+1500))
    while (( SECONDS < deadline )); do
      if curl --max-time 3 -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
        echo 'API ready.'; exit 0
      fi
      [[ "$(docker inspect -f '{{.State.Running}}' qwen-tp1)" == true ]] || { echo 'Container stopped; check logs.' >&2; exit 1; }
      sleep 5
    done
    echo 'Readiness timeout; inspect logs.' >&2; exit 1
    ;;
  logs) docker logs --tail 60 -f qwen-tp1 ;;
  *) echo 'Usage: node.sh start|stop|status|wait|logs' >&2; exit 2 ;;
esac
