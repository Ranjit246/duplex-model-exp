#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD="python3"

# Install system dependencies
apt-get update -qq
apt-get install -y -qq libopus-dev

echo "Using: $($PYTHON_CMD --version)"

# Create virtual environment and install dependencies
VENV_DIR="/workspace/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

echo "Installing moshi..."
"$VENV_DIR/bin/pip" install -e /workspace/personaplex/moshi
echo "Installation complete."

: "${HF_TOKEN:?HF_TOKEN must be set (export HF_TOKEN=<your-hf-token> before running)}"
export HF_TOKEN

# Setup logging
LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/moshi_$(date '+%Y%m%d_%H%M%S').log"

# Start moshi server
echo "Starting moshi-pytorch server..."
echo "Logging to: $LOG_FILE"
SSL_DIR=$(mktemp -d)
"$VENV_DIR/bin/python3" -m moshi.server --ssl "$SSL_DIR" 2>&1 | tee "$LOG_FILE"


# open at https://localhost:8998 not http - it will not work!