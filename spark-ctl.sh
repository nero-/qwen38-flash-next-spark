#!/usr/bin/env bash
# Mac controller for Qwen on both directly connected Sparks.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
cfg() { python3 "$HERE/tp2/config.py" "$1"; }
REMOTE_ROOT="$(cfg remote_root)"
API_PORT="$(cfg api_port)"
SPARK_HOST="${SPARK_HOST:-$(cfg ranks.0.ssh)}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
[[ "$SPARK_HOST" != -* && "$SPARK_HOST" =~ ^[a-zA-Z0-9_.@:-]+$ ]] || exit 2
[[ "$LOCAL_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && ((LOCAL_PORT <= 65535)) || exit 2
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
case "${1:-help}" in
  start|stop|status|wait|logs|logs-r1)
    exec ssh "${ssh_args[@]}" "$SPARK_HOST" "bash ~/$REMOTE_ROOT/cluster.sh $1"
    ;;
  profile)
    case "${2:-}" in
      balanced|adaptive) selected=hc-adaptive ;;
      original) selected=hc ;;
      coding) selected=hc-k20-mtp5 ;;
      baseline) selected=baseline ;;
      *) echo 'Usage: ./spark-ctl.sh profile balanced|original|coding|baseline' >&2; exit 2 ;;
    esac
    exec ssh "${ssh_args[@]}" "$SPARK_HOST" "python3 ~/$REMOTE_ROOT/select_profile.py $selected"
    ;;
  cables)
    case "${2:-}" in 1|2) selected="$2" ;; *) echo 'Usage: ./spark-ctl.sh cables 1|2' >&2; exit 2 ;; esac
    exec ssh "${ssh_args[@]}" "$SPARK_HOST" "python3 ~/$REMOTE_ROOT/select_cables.py $selected"
    ;;
  tunnel)
    echo "API: http://localhost:$LOCAL_PORT/v1 (server port $API_PORT) — keep this terminal open."
    exec ssh "${ssh_args[@]}" -o ExitOnForwardFailure=yes -N \
      -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$API_PORT" "$SPARK_HOST"
    ;;
  help|-h|--help)
    cat <<EOF
Usage: ./spark-ctl.sh start|stop|status|wait|logs|logs-r1|tunnel
       ./spark-ctl.sh profile balanced|original|coding|baseline
       ./spark-ctl.sh cables 1|2

start/stop control BOTH Sparks. Finish active requests before stopping.
wait allows up to 20 minutes for startup and does not stop the model on Ctrl+C.
logs follows rank 0; logs-r1 follows rank 1. Ctrl+C only stops following.
Model: qwen3.8-flash-next-4p89bpw
API through tunnel: http://localhost:8000/v1 (key: local if required)
Do not start the old TP1 container while TP2 is running.
profile restarts BOTH idle ranks and waits for readiness; active requests block it.
balanced: adaptive HC decode, owned-row prefill, MTP3, BF16 head.
original: original HC decode backup, owned-row prefill, MTP3, BF16 head.
coding: optional HC + top-20 draft + MTP5 for non-thinking speed workloads;
        99 tok/s in the earlier capped Tetris run. Target head remains BF16.
baseline: original TP2 model configuration, retaining MTU9000.
EOF
    ;;
  *) echo 'Unknown command; run help.' >&2; exit 2 ;;
esac
