#!/bin/bash
# DigitalOcean Droplet Setup for Alpaca Trading Bot
# Run this as root on a fresh Ubuntu 24.04 droplet
#
# Usage: bash setup_droplet.sh

set -euo pipefail

echo "========================================"
echo "  Alpaca Trading Bot — Droplet Setup"
echo "========================================"

# 1. System updates
echo "[1/7] Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

# 2. Install Python build dependencies
echo "[2/7] Installing Python build dependencies..."
apt-get install -y -qq \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev git

# 3. Install Python 3.12 (stable, available in Ubuntu 24.04 repos)
# We use 3.12 instead of 3.14 for stability on the server
echo "[3/7] Installing Python 3.12..."
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev python3-pip

# 4. Install uv (fast Python package manager)
echo "[4/7] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 5. Clone the repo (user must set REPO_URL)
echo "[5/7] Cloning repository..."
INSTALL_DIR="/opt/alpaca-trading"
if [ -d "$INSTALL_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd "$INSTALL_DIR"
    git pull
else
    if [ -z "${REPO_URL:-}" ]; then
        echo "ERROR: Set REPO_URL environment variable first."
        echo "  export REPO_URL=https://github.com/YOUR_USER/alpaca-trading.git"
        exit 1
    fi
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 6. Create virtual environment and install dependencies
echo "[6/7] Setting up Python environment..."
cd "$INSTALL_DIR/strategy-engine"
python3.12 -m venv .venv
source .venv/bin/activate

# Install with uv (much faster than pip)
if command -v uv &>/dev/null; then
    uv pip install -e ".[finbert]"
else
    pip install -e ".[finbert]"
fi

# 7. Create log directory
echo "[7/7] Creating directories..."
mkdir -p "$INSTALL_DIR/logs"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Create .env file:  nano $INSTALL_DIR/.env"
echo "  2. Install systemd services:  bash $INSTALL_DIR/deploy/install_services.sh"
echo "  3. Check status:  systemctl status alpaca-*"
echo ""
