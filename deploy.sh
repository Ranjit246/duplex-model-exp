#!/usr/bin/env bash
set -euo pipefail

RECOMMENDED_PYTHON_VERSION="3.12"
PYTHON_FULL_VERSION="3.12.8"
PYTHON_CMD="python${RECOMMENDED_PYTHON_VERSION}"

install_python312() {
    echo "Installing Python ${PYTHON_FULL_VERSION} from source..."

    apt-get update -qq
    apt-get install -y -qq \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        wget \
        libbz2-dev \
        liblzma-dev

    local tmp_dir
    tmp_dir=$(mktemp -d)
    cd "$tmp_dir"

    wget -q "https://www.python.org/ftp/python/${PYTHON_FULL_VERSION}/Python-${PYTHON_FULL_VERSION}.tgz"
    tar -xzf "Python-${PYTHON_FULL_VERSION}.tgz"
    cd "Python-${PYTHON_FULL_VERSION}"

    ./configure --enable-optimizations --prefix=/usr/local 2>&1 | tail -1
    make -j"$(nproc)" 2>&1 | tail -1
    make altinstall 2>&1 | tail -1

    cd /
    rm -rf "$tmp_dir"

    echo "Python ${PYTHON_FULL_VERSION} installed successfully."
}

# Check for recommended Python version
if command -v "$PYTHON_CMD" &>/dev/null; then
    echo "Found $($PYTHON_CMD --version)"
else
    echo "Python ${RECOMMENDED_PYTHON_VERSION} not found."
    install_python312
fi

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
"$VENV_DIR/bin/pip" install -e /workspace/moshi/moshi
"$VENV_DIR/bin/pip" install gradio-webrtc>=0.0.18
echo "Installation complete."

# Setup logging
LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/moshi_$(date '+%Y%m%d_%H%M%S').log"

# Start moshi server
echo "Starting moshi-pytorch server..."
echo "Logging to: $LOG_FILE"
"$VENV_DIR/bin/python3" -m moshi.server --gradio-tunnel --hf-repo kyutai/moshika-pytorch-bf16 2>&1 | tee "$LOG_FILE"
