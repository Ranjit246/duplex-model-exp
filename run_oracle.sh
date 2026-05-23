#!/usr/bin/env bash
# Runs oracle generation and auto-restarts if stuck (no new files for STALE_MINUTES).
# Usage: TEXT_DIR=... ORACLE_DIR=... LLM_BASE_URL=... LLM_MODEL=... bash run_oracle.sh
set -euo pipefail

TEXT_DIR="${TEXT_DIR:-data/my_dataset/text}"
ORACLE_DIR="${ORACLE_DIR:-data/my_dataset/oracle_raw}"
LLM_BASE_URL="${LLM_BASE_URL:-}"          # e.g. http://localhost:8001/v1
LLM_MODEL="${LLM_MODEL:-gpt-4.1-nano}"    # model name served by the LLM endpoint
STALE_MINUTES=5   # restart if no new files written in this many minutes
WORKERS="${WORKERS:-32}"

CMD=(
  uv run --extra oracle -m tools.generate_oracle_from_text
  --text_dir "$TEXT_DIR"
  --output_dir "$ORACLE_DIR"
  --target_channel 1 --time_interval 1.0
  --B_channel 1 --resume --fallback_to_hint_on_error
  --workers "$WORKERS"
  --model "$LLM_MODEL"
)
if [[ -n "$LLM_BASE_URL" ]]; then
  CMD+=(--base_url "$LLM_BASE_URL")
fi

total=$(find "$TEXT_DIR" -maxdepth 1 -name "*.json" | wc -l | tr -d ' ')

while true; do
  done=$(find "$ORACLE_DIR" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
  echo "[$(date '+%H:%M:%S')] Starting oracle generation — $done/$total done"

  # Launch in background
  "${CMD[@]}" &
  PID=$!

  # Watchdog: check every 60s if file count grew
  last_count=$done
  while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    current=$(find "$ORACLE_DIR" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$current" -gt "$last_count" ]]; then
      last_count=$current
      echo "[$(date '+%H:%M:%S')] Progress: $current/$total files"
    else
      # No progress — check stale threshold
      : $(( stale_checks = ${stale_checks:-0} + 1 ))
      if [[ "$stale_checks" -ge "$STALE_MINUTES" ]]; then
        echo "[$(date '+%H:%M:%S')] STALE — no new files in ${STALE_MINUTES}m. Killing PID $PID and restarting..."
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        stale_checks=0
        break
      fi
    fi
  done

  wait "$PID" 2>/dev/null || true

  # Check if fully done
  done=$(find "$ORACLE_DIR" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$done" -ge "$total" ]]; then
    echo "[$(date '+%H:%M:%S')] All $total files done!"
    break
  fi

  echo "[$(date '+%H:%M:%S')] Restarting in 5s..."
  sleep 5
done
