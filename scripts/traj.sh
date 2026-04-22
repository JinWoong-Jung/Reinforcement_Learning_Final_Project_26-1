#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/traj.sh <algo> <subject> <level>
  bash scripts/traj.sh <subject> <level>        # defaults to ppo

Arguments:
  <algo>     ppo | dqn
  <subject>  calculus | geometry | prob_stat
  <level>    low | mid | high

Behavior:
  - Finds the most recently modified run directory under runs/<algo>/<subject>/<level>/
  - Generates a trajectory report on the matching CSAT file

Example:
  bash scripts/traj.sh ppo geometry low
  bash scripts/traj.sh dqn geometry low
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

RUN_ROOT="runs/${ALGO}/${SUBJECT}/${LEVEL}"
LATEST_RUN="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | xargs -0 ls -td 2>/dev/null | head -n 1)"

if [[ -z "${LATEST_RUN:-}" ]]; then
  echo "[error] no saved runs found under ${RUN_ROOT}" >&2
  exit 1
fi

MODEL_PATH="${LATEST_RUN}/checkpoints/${ALGO}_final.zip"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[error] model not found: $MODEL_PATH" >&2
  exit 1
fi

declare -A EXAM_MAP=(
  ["calculus"]="data/25_math_calculus.json"
  ["geometry"]="data/25_math_geometry.json"
  ["prob_stat"]="data/25_math_prob_stat.json"
)

EXAM_PATH="${EXAM_MAP[$SUBJECT]}"
OUTPUT_PATH="results/${ALGO}/${SUBJECT}/${LEVEL}_trajectory.json"

echo "[traj] algorithm=${ALGO} subject=${SUBJECT} level=${LEVEL}"
echo "[traj] latest_run=${LATEST_RUN}"

python analysis/trajectory_report.py \
  --run-dir "$LATEST_RUN" \
  --model-path "$MODEL_PATH" \
  --algorithm "$ALGO" \
  --exam-data "$EXAM_PATH" \
  --episodes 10 \
  --max-logged-steps 80 \
  --output "$OUTPUT_PATH"
