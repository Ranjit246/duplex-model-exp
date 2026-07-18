#!/usr/bin/env bash
# Launch the oracle-enabled Kame inference server on the step_12000 checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
source "$HOME/.local/bin/env"

CLEAN_DIR="${CLEAN_DIR:-output/moshiko-finetuned/step_12000_cleaned}"
TOKENIZER_PATH="${TOKENIZER_PATH:-init_models/tokenizer_spm_32k_3.model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

# --- Oracle LLM (self-hosted vLLM / OpenAI-compatible) ---
# Set these in your environment (do not commit secrets):
#   export OPENAI_BASE_URL=...   OPENAI_API_KEY=...   ORACLE_MODEL=...
export OPENAI_BASE_URL="${OPENAI_BASE_URL:?set OPENAI_BASE_URL to your OpenAI-compatible oracle endpoint}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY for the oracle LLM}"
export ORACLE_MODEL="${ORACLE_MODEL:-gpt-4.1}"

# --- ASR provider (deepgram | google) ---
# Deepgram: set DEEPGRAM_API_KEY; provider auto-selects when a key is present.
export ASR_PROVIDER="${ASR_PROVIDER:-auto}"
export DEEPGRAM_MODEL="${DEEPGRAM_MODEL:-nova-3}"
export DEEPGRAM_LANGUAGE="${DEEPGRAM_LANGUAGE:-multi}"   # code-switching Hindi+English
export DEEPGRAM_API_KEY="${DEEPGRAM_API_KEY:-}"          # set to enable Deepgram ASR
# Google fallback: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# --- File logging ---
LOG_DIR="${LOG_DIR:-logs/inference_sessions}"   # per-session conversation + token logs
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1   # stream server/[LLM] logs to the file live

uv run -m kame.server_oracle \
  --moshi-weight "$CLEAN_DIR/model.safetensors" \
  --config-path "$CLEAN_DIR/moshi_lm_kwargs.json" \
  --tokenizer "$TOKENIZER_PATH" \
  --log-dir "$LOG_DIR" \
  --host "$HOST" \
  --port "$PORT"
