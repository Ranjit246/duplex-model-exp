#!/usr/bin/env bash
# Watch output/moshiko-finetuned for finished step_<N> checkpoints, upload to HF, delete locally.
# Usage:
#   nohup bash run_ckpt_watcher.sh > logs/ckpt_watcher.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")"

# HF write token. Override at the command line if you prefer not to keep it in the file:
#   HF_TOKEN=hf_xxx bash run_ckpt_watcher.sh
export HF_TOKEN="${HF_TOKEN:-HF_TOKEN_HERE}"

export REPO_ID="${REPO_ID:-Ranjit/moshiko-kame-hinglish-ft-exp}"
export CKPT_ROOT="${CKPT_ROOT:-output/moshiko-finetuned}"
export POLL_SECONDS="${POLL_SECONDS:-60}"
export STABLE_SECONDS="${STABLE_SECONDS:-300}"

mkdir -p logs
exec uv run python -m tools.watch_upload_checkpoints
