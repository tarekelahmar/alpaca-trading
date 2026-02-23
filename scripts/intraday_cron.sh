#!/bin/bash
# Intraday scanner wrapper for launchd
# Runs every 30 minutes during market hours (9:45 AM - 3:45 PM ET)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENGINE_DIR="$PROJECT_DIR/strategy-engine"
LOG_DIR="$PROJECT_DIR/logs"
UV="/Users/Tarek/.local/bin/uv"

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/intraday_$(date +%Y%m%d).log"

# Source environment
set -a
source "$PROJECT_DIR/.env"
set +a

# Wait for network (up to 60s)
MAX_WAIT=60
WAITED=0
while ! ping -c 1 -W 2 data.alpaca.markets >/dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "" >> "$LOG_FILE"
        echo "=== $(date) — Skipped: No network after ${MAX_WAIT}s ===" >> "$LOG_FILE"
        exit 0
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

echo "" >> "$LOG_FILE"
echo "=== Intraday Scan: $(date) ===" >> "$LOG_FILE"

cd "$ENGINE_DIR"
$UV run python "$SCRIPT_DIR/intraday_scanner.py" --paper >> "$LOG_FILE" 2>&1 || true

echo "=== Done: $(date) ===" >> "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "intraday_*.log" -mtime +30 -delete 2>/dev/null || true
