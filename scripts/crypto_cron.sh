#!/bin/bash
# Crypto strategy execution wrapper for launchd/cron
# Runs every 4 hours, 24/7 — crypto never sleeps
#
# Crontab example:
#   0 */4 * * * /Users/Tarek/alpaca-trading/scripts/crypto_cron.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENGINE_DIR="$PROJECT_DIR/strategy-engine"
LOG_DIR="$PROJECT_DIR/logs"
UV="/Users/Tarek/.local/bin/uv"

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/crypto_$(date +%Y%m%d_%H%M%S).log"

# Source environment
set -a
source "$PROJECT_DIR/.env"
set +a

# Wait for network connectivity (Mac may be waking from sleep)
MAX_WAIT=120
WAITED=0
while ! ping -c 1 -W 2 data.alpaca.markets >/dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "=== $(date) — FAILED: No network after ${MAX_WAIT}s ===" >> "$LOG_FILE"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -gt 0 ]; then
    echo "=== Waited ${WAITED}s for network ===" >> "$LOG_FILE"
fi

echo "=== Crypto Trading Run: $(date) ===" >> "$LOG_FILE"

# Run with retry (up to 3 attempts)
cd "$ENGINE_DIR"
ATTEMPT=0
MAX_ATTEMPTS=3
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    if $UV run python "$SCRIPT_DIR/run_crypto.py" --paper >> "$LOG_FILE" 2>&1; then
        break
    else
        echo "=== Attempt $ATTEMPT failed, retrying in 30s... ===" >> "$LOG_FILE"
        sleep 30
    fi
done

echo "=== Completed: $(date) ===" >> "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "crypto_*.log" -mtime +30 -delete 2>/dev/null || true
