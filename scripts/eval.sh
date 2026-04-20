#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval.sh <algo> <level>

Arguments:
  <algo>   ppo | dqn
  <level>  low | mid | high

Behavior:
  - Finds the most recently modified run directory under runs/<algo>/train__<level>/
  - Evaluates that model on:
    * data/25_math_calculus.json
    * data/25_math_geometry.json
    * data/25_math_prob_stat.json

Example:
  bash scripts/eval.sh dqn high
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

RUN_ROOT="runs/${ALGO}/train__${LEVEL}"
LATEST_RUN="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | xargs -0 ls -td 2>/dev/null | head -n 1)"

if [[ -z "${LATEST_RUN:-}" ]]; then
  echo "[error] no saved runs found under ${RUN_ROOT}" >&2
  exit 1
fi

SNAPSHOT_PATH="${LATEST_RUN}/config_snapshot.yaml"
MODEL_PATH="${LATEST_RUN}/checkpoints/${ALGO}_final.zip"

if [[ ! -f "$SNAPSHOT_PATH" ]]; then
  echo "[error] config snapshot not found: $SNAPSHOT_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[error] model not found: $MODEL_PATH" >&2
  exit 1
fi

declare -A EXAM_MAP=(
  ["calculus"]="data/25_math_calculus.json"
  ["geometry"]="data/25_math_geometry.json"
  ["prob_stat"]="data/25_math_prob_stat.json"
)

echo "[eval] algo=${ALGO} level=${LEVEL}"
echo "[eval] latest_run=${LATEST_RUN}"

for SUBJECT in calculus geometry prob_stat; do
  EXAM_PATH="${EXAM_MAP[$SUBJECT]}"
  OUTPUT_DIR="results/${ALGO}/${LEVEL}/${SUBJECT}"

  echo
  echo "[eval] subject=${SUBJECT}"
  python main.py \
    --mode eval \
    --config "$SNAPSHOT_PATH" \
    --model-path "$MODEL_PATH" \
    --algorithm "$ALGO" \
    --exam-data "$EXAM_PATH" \
    --episodes 100 \
    --output "$OUTPUT_DIR"
done

