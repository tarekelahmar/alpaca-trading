#!/bin/bash
# Daily trading execution wrapper for launchd/cron
# Runs at 9:35 AM ET (5 min after market open) on weekdays

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENGINE_DIR="$PROJECT_DIR/strategy-engine"
LOG_DIR="$PROJECT_DIR/logs"
UV="/Users/Tarek/.local/bin/uv"

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/daily_$(date +%Y%m%d_%H%M%S).log"

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

echo "=== Daily Trading Run: $(date) ===" >> "$LOG_FILE"

# Run with retry (up to 3 attempts)
cd "$ENGINE_DIR"
ATTEMPT=0
MAX_ATTEMPTS=3
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    if $UV run python "$SCRIPT_DIR/run_daily.py" --paper >> "$LOG_FILE" 2>&1; then
        break
    else
        echo "=== Attempt $ATTEMPT failed, retrying in 30s... ===" >> "$LOG_FILE"
        sleep 30
    fi
done

echo "=== Completed: $(date) ===" >> "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "daily_*.log" -mtime +30 -delete 2>/dev/null || true
