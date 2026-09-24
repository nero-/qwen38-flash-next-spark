#!/usr/bin/env bash
# TP2 bootstrap entry point. Requires SSH/rsync/Docker on both configured hosts.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
exec bash "$HERE/tp2/bootstrap.sh" "$@"
