#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
readarray -t SETTINGS < <(python3 - "$HERE/cluster-config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
print(c['remote_root'])
print(c['ranks'][1]['peer_ssh'])
print(' '.join(c['ssh_options']))
print(c['api_port'])
PY
)
ROOT="$HOME/${SETTINGS[0]}"
WORKER="${SETTINGS[1]}"
read -r -a SSH_OPTIONS <<<"${SETTINGS[2]}"
worker() { ssh "${SSH_OPTIONS[@]}" "$WORKER" "$@"; }
REMOTE_ROOT="~/${SETTINGS[0]}"
API_PORT="${SETTINGS[3]}"
case "${1:-status}" in
  start)
    local_profile=$(cat "$ROOT/selected-profile.txt" 2>/dev/null || python3 "$HERE/config.py" default_profile)
    local_cables=$(cat "$ROOT/selected-cables.txt" 2>/dev/null || python3 "$HERE/config.py" default_cables)
    remote_profile=$(worker "cat ${REMOTE_ROOT}/selected-profile.txt 2>/dev/null || python3 ${REMOTE_ROOT}/config.py default_profile")
    remote_cables=$(worker "cat ${REMOTE_ROOT}/selected-cables.txt 2>/dev/null || python3 ${REMOTE_ROOT}/config.py default_cables")
    if [[ "$local_profile" != "$remote_profile" || "$local_cables" != "$remote_cables" ]]; then
      echo 'Rank profile/cable selections differ. Use the coordinated selector before starting.' >&2
      exit 1
    fi
    python3 "$ROOT/launch.py" check --rank 0 >/dev/null
    worker "python3 ${REMOTE_ROOT}/launch.py check --rank 1" >/dev/null
    local_running=$(docker inspect -f '{{.State.Running}}' qwen-tp2 2>/dev/null || true)
    remote_running=$(worker "docker inspect -f '{{.State.Running}}' qwen-tp2 2>/dev/null || true")
    if [[ "$local_running" == true && "$remote_running" != true ]] || [[ "$local_running" != true && "$remote_running" == true ]]; then
      echo 'Only one rank is running. Run stop, then start to restart the pair.' >&2
      exit 1
    fi
    worker "python3 ${REMOTE_ROOT}/launch.py start --rank 1"
    if ! python3 "$ROOT/launch.py" start --rank 0; then
      worker "python3 ${REMOTE_ROOT}/launch.py stop --rank 1" || true
      exit 1
    fi
    echo 'Both ranks started. Run wait or logs to follow startup.'
    ;;
  stop)
    result=0
    python3 "$ROOT/launch.py" stop --rank 0 || result=$?
    worker "python3 ${REMOTE_ROOT}/launch.py stop --rank 1" || result=$?
    exit "$result"
    ;;
  status)
    result=0
    echo "Selected profile: $(cat "$ROOT/selected-profile.txt" 2>/dev/null || echo hc-adaptive)"
    echo "Selected cables: $(cat "$ROOT/selected-cables.txt" 2>/dev/null || echo 2)"
    echo 'Rank 0:'
    python3 "$ROOT/launch.py" status --rank 0 || result=$?
    echo 'Rank 1:'
    worker "python3 ${REMOTE_ROOT}/launch.py status --rank 1" || result=$?
    curl --max-time 5 -s -o /dev/null -w 'API HTTP %{http_code}\n' "http://127.0.0.1:$API_PORT/health" || result=$?
    exit "$result"
    ;;
  wait)
    deadline=$((SECONDS+1200))
    while (( SECONDS < deadline )); do
      if curl --max-time 3 -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
        echo 'TP2 API ready.'; exit 0
      fi
      [[ "$(docker inspect -f '{{.State.Running}}' qwen-tp2)" == true ]] || { echo 'Rank 0 stopped; check logs.' >&2; exit 1; }
      sleep 5
    done
    echo 'Readiness timeout; inspect both ranks.' >&2; exit 1
    ;;
  logs) docker logs --tail 60 -f qwen-tp2 ;;
  logs-r1) worker 'docker logs --tail 60 -f qwen-tp2' ;;
  *) echo 'Usage: cluster.sh start|stop|status|wait|logs|logs-r1' >&2; exit 2 ;;
esac
