#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-}"
REFRESH_SEC="${REFRESH_SEC:-5}"
TAIL_LINES="${TAIL_LINES:-30}"
MONITOR_VERBOSE="${MONITOR_VERBOSE:-0}"
MONITOR_CARD_COLS="${MONITOR_CARD_COLS:-4}"
PY_BIN="${PY_BIN:-/home/hj/dsl/modeling/venv/bin/python}"
HYUNJIN_ROOT="/home/hj/dsl/modeling/Hyunjin"
ARTIFACT_ROOT="$HYUNJIN_ROOT/artifacts/raycast_hunt"

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(ls -1dt "$ARTIFACT_ROOT"/*/ 2>/dev/null | head -n1 | xargs -r basename)"
fi

if [[ -z "$RUN_NAME" ]]; then
  echo "No run found under $ARTIFACT_ROOT"
  exit 1
fi

RUN_DIR="$ARTIFACT_ROOT/$RUN_NAME"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR"
  exit 1
fi

TRAIN_LOG="$RUN_DIR/train.log"
MONITOR_LOG="$RUN_DIR/monitor_terminal.log"

{
  echo "[$(date '+%F %T %Z')] monitor attach run=$RUN_NAME refresh=${REFRESH_SEC}s tail=${TAIL_LINES}"
  echo "run_dir=$RUN_DIR"
  echo "monitor_log=$MONITOR_LOG"
  echo "verbose=$MONITOR_VERBOSE"

  while true; do
    echo
    echo "===== $(date '+%F %T %Z') run=$RUN_NAME ====="
    "$PY_BIN" "$HYUNJIN_ROOT/monitor_raycast_training_speed.py" \
      --run-name "$RUN_NAME" \
      --once \
      --output-format table \
      --max-card-cols "$MONITOR_CARD_COLS" \
      --summary-only || true

    if [[ "$MONITOR_VERBOSE" == "1" ]]; then
      echo "--- process ---"
      if command -v rg >/dev/null 2>&1; then
        ps -ef | rg "run_next_training_select_best.py|train_craftground_raycast_curriculum.py.*$RUN_NAME" | rg -v rg || true
      else
        ps -ef | grep -E "run_next_training_select_best.py|train_craftground_raycast_curriculum.py.*$RUN_NAME" | grep -v grep || true
      fi

      echo "--- train.log tail (last $TAIL_LINES lines) ---"
      if [[ -f "$TRAIN_LOG" ]]; then
        tail -n "$TAIL_LINES" "$TRAIN_LOG" || true
      else
        echo "No train.log yet: $TRAIN_LOG"
      fi
    fi

    sleep "$REFRESH_SEC"
  done
} | tee -a "$MONITOR_LOG"
