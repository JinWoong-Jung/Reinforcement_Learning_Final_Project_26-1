#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train.sh <algo> <level>

Arguments:
  <algo>   ppo | dqn
  <level>  low | mid | high

Example:
  bash scripts/train.sh ppo mid
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

ALGO="$1"
LEVEL="$2"

case "$ALGO" in
  ppo|dqn) ;;
  *)
    echo "[error] algo must be one of: ppo, dqn" >&2
    exit 1
    ;;
esac

case "$LEVEL" in
  low|mid|high) ;;
  *)
    echo "[error] level must be one of: low, mid, high" >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="configs/${ALGO}/train_${LEVEL}.yaml"
OUTPUT_ROOT="runs/${ALGO}/train__${LEVEL}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[error] config not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "[train] algo=${ALGO} level=${LEVEL}"
echo "[train] config=${CONFIG_PATH}"
echo "[train] output_root=${OUTPUT_ROOT}"

python main.py --mode train --config "$CONFIG_PATH" --output "$OUTPUT_ROOT"

