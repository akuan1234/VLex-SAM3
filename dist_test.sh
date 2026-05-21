#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash dist_test.sh <config> [extra eval args...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$1"
shift

WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/work_logs}"
GPUS="${GPUS:-4}"
PORT="${PORT:-29500}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" \
"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$GPUS" \
  --master_port="$PORT" \
  "$SCRIPT_DIR/eval.py" \
  "$CONFIG" \
  --launcher pytorch \
  --cfg-options work_dir="$WORK_DIR" \
  "$@"
