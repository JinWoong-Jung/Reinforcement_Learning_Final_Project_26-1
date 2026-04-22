#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train.sh <algo> <subject> <level>
  bash scripts/train.sh <subject> <level>        # defaults to ppo

Arguments:
  <algo>     ppo | dqn
  <subject>  calculus | geometry | prob_stat
  <level>    low | mid | high

Example:
  bash scripts/train.sh ppo calculus mid
  bash scripts/train.sh dqn calculus mid
EOF
}

if [[ $# -eq 2 ]]; then
  ALGO="ppo"
  SUBJECT="$1"
  LEVEL="$2"
elif [[ $# -eq 3 ]]; then
  ALGO="$1"
  SUBJECT="$2"
  LEVEL="$3"
else
  usage
  exit 1
fi

case "$ALGO" in
  ppo|dqn) ;;
  *)
    echo "[error] algo must be one of: ppo, dqn" >&2
    exit 1
    ;;
esac

case "$SUBJECT" in
  calculus|geometry|prob_stat) ;;
  *)
    echo "[error] subject must be one of: calculus, geometry, prob_stat" >&2
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

CONFIG_PATH="configs/${ALGO}/${SUBJECT}/${LEVEL}.yaml"
OUTPUT_ROOT="runs/${ALGO}/${SUBJECT}/${LEVEL}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[error] config not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "[train] algorithm=${ALGO} subject=${SUBJECT} level=${LEVEL}"
echo "[train] config=${CONFIG_PATH}"
echo "[train] output_root=${OUTPUT_ROOT}"

python main.py --mode train --config "$CONFIG_PATH" --output "$OUTPUT_ROOT"
