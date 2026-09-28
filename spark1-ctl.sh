#!/usr/bin/env bash
# Mac controller for Qwen on ONE DGX Spark (tp1/node-config.json).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
cfg() { python3 "$HERE/tp1/config.py" "$1"; }
REMOTE_ROOT="$(cfg remote_root)"
API_PORT="$(cfg api_port)"
SPARK_HOST="${SPARK_HOST:-$(cfg ssh)}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
[[ "$SPARK_HOST" != -* && "$SPARK_HOST" =~ ^[a-zA-Z0-9_.@:-]+$ ]] || exit 2
[[ "$LOCAL_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && ((LOCAL_PORT <= 65535)) || exit 2
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
case "${1:-help}" in
  start|stop|status|wait|logs)
    exec ssh "${ssh_args[@]}" "$SPARK_HOST" "bash ~/$REMOTE_ROOT/node.sh $1"
    ;;
  profile)
    case "${2:-}" in
      default) selected="$(cfg default_profile)" ;;
      tp1|tp1+*) selected="$2" ;;
      *) echo 'Usage: ./spark1-ctl.sh profile default|tp1[+modifier...]' >&2; exit 2 ;;
    esac
    [[ "$selected" =~ ^[a-z0-9+]+$ ]] || exit 2
    exec ssh "${ssh_args[@]}" "$SPARK_HOST" "cd ~/$REMOTE_ROOT && python3 select_profile.py '$selected'"
    ;;
  tunnel)
    echo "API: http://localhost:$LOCAL_PORT/v1 (server port $API_PORT) — keep this terminal open."
    exec ssh "${ssh_args[@]}" -o ExitOnForwardFailure=yes -N \
      -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$API_PORT" "$SPARK_HOST"
    ;;
  help|-h|--help)
    cat <<EOF
Usage: ./spark1-ctl.sh start|stop|status|wait|logs|tunnel
       ./spark1-ctl.sh profile default|tp1[+modifier...]

Controls the single-Spark server on $SPARK_HOST (container qwen-tp1).
Finish active requests before stopping. wait allows up to 25 minutes for startup.
Model: qwen3.8-flash-next-4p89bpw
API through tunnel: http://localhost:$LOCAL_PORT/v1
profile refuses active requests, restarts the server and waits for readiness;
a failed start restores the previous profile. Modifiers are listed in tp1/README.md.
EOF
    ;;
  *) echo 'Unknown command; run help.' >&2; exit 2 ;;
esac
