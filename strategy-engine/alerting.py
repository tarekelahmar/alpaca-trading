"""Lightweight alerting module for trading notifications.

Sends alerts via Slack webhook and/or Discord webhook.
Falls back silently to stderr-only if no webhook is configured.

Usage:
    from alerting import alert, AlertLevel
    alert("Kill switch triggered! Daily loss: $5,200", AlertLevel.CRITICAL)
    alert("Daily run complete: 5 orders executed", AlertLevel.INFO)
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from enum import IntEnum


class AlertLevel(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


_LEVEL_EMOJI = {
    AlertLevel.INFO: "INFO",
    AlertLevel.WARNING: "WARN",
    AlertLevel.ERROR: "ERROR",
    AlertLevel.CRITICAL: "CRITICAL",
}

_SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL", "")
_DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
_MIN_LEVEL = AlertLevel[os.environ.get("ALERT_MIN_LEVEL", "WARNING").upper()]
_MODE = "PAPER" if os.environ.get("PAPER_TRADING", "true").lower() == "true" else "LIVE"


def _send_slack(message: str, level: AlertLevel) -> bool:
    if not _SLACK_WEBHOOK:
        return False
    try:
        payload = {
            "text": f"[{_LEVEL_EMOJI[level]}] [{_MODE}] {message}",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _SLACK_WEBHOOK,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[Alert] Slack send failed: {e}", file=sys.stderr)
        return False


def _send_discord(message: str, level: AlertLevel) -> bool:
    if not _DISCORD_WEBHOOK:
        return False
    try:
        payload = {
            "content": f"[{_LEVEL_EMOJI[level]}] [{_MODE}] {message}",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _DISCORD_WEBHOOK,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as e:
        print(f"[Alert] Discord send failed: {e}", file=sys.stderr)
        return False


def alert(message: str, level: AlertLevel = AlertLevel.INFO) -> None:
    """Send an alert via configured channels.

    Always logs to stderr. Sends to Slack/Discord if webhook is
    configured and level >= ALERT_MIN_LEVEL.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log_line = f"[{_LEVEL_EMOJI[level]}] [{now}] {message}"
    print(log_line, file=sys.stderr)

    if level >= _MIN_LEVEL:
        _send_slack(message, level)
        _send_discord(message, level)
