#!/bin/bash
# Install and enable all systemd services for the trading bot.
# Run as root after setup_droplet.sh and creating .env

set -euo pipefail

DEPLOY_DIR="/opt/alpaca-trading/deploy"

echo "Installing systemd services..."

# Copy service files — equities
cp "$DEPLOY_DIR/alpaca-monitor.service" /etc/systemd/system/
cp "$DEPLOY_DIR/alpaca-daily.service" /etc/systemd/system/
cp "$DEPLOY_DIR/alpaca-daily.timer" /etc/systemd/system/
cp "$DEPLOY_DIR/alpaca-intraday.service" /etc/systemd/system/
cp "$DEPLOY_DIR/alpaca-intraday.timer" /etc/systemd/system/

# Copy service files — crypto (24/7)
cp "$DEPLOY_DIR/alpaca-crypto-monitor.service" /etc/systemd/system/
cp "$DEPLOY_DIR/alpaca-crypto.service" /etc/systemd/system/
cp "$DEPLOY_DIR/alpaca-crypto.timer" /etc/systemd/system/

# Install log rotation
cp "$DEPLOY_DIR/alpaca-logrotate.conf" /etc/logrotate.d/alpaca-trading

# Set timezone for correct timer firing
timedatectl set-timezone America/New_York
echo "Timezone set to: $(timedatectl show --property=Timezone --value)"

# Reload systemd
systemctl daemon-reload

# Enable and start services
echo ""
echo "Enabling services..."

# Price monitor — runs 24/7 (sleeps when market closed)
systemctl enable alpaca-monitor.service
systemctl start alpaca-monitor.service
echo "  ✓ Price monitor (30s loop) — started"

# Daily engine timer — fires at 9:35 AM ET weekdays
systemctl enable alpaca-daily.timer
systemctl start alpaca-daily.timer
echo "  ✓ Daily engine timer — 9:35 AM ET, Mon-Fri"

# Intraday scanner timer — fires every 15 min weekdays
systemctl enable alpaca-intraday.timer
systemctl start alpaca-intraday.timer
echo "  ✓ Intraday scanner timer — every 15 min, Mon-Fri"

# Crypto monitor — runs 24/7, every 60 seconds
systemctl enable alpaca-crypto-monitor.service
systemctl start alpaca-crypto-monitor.service
echo "  ✓ Crypto monitor (60s loop) — started"

# Crypto strategy timer — fires every 4 hours, every day
systemctl enable alpaca-crypto.timer
systemctl start alpaca-crypto.timer
echo "  ✓ Crypto strategy timer — every 4h, 24/7"

echo ""
echo "========================================"
echo "  All services installed and running!"
echo "========================================"
echo ""
echo "Useful commands:"
echo "  systemctl status alpaca-monitor            # Equity monitor status"
echo "  systemctl status alpaca-crypto-monitor     # Crypto monitor status"
echo "  systemctl status alpaca-daily.timer        # Equity daily timer"
echo "  systemctl status alpaca-crypto.timer       # Crypto strategy timer"
echo "  journalctl -u alpaca-monitor -f            # Live equity monitor logs"
echo "  journalctl -u alpaca-crypto-monitor -f     # Live crypto monitor logs"
echo "  journalctl -u alpaca-daily -n 50           # Last 50 daily log lines"
echo "  journalctl -u alpaca-crypto -n 50          # Last 50 crypto log lines"
echo "  systemctl list-timers                      # See all timer schedules"
echo "  tail -f /opt/alpaca-trading/logs/monitor.log        # Raw equity logs"
echo "  tail -f /opt/alpaca-trading/logs/crypto-monitor.log # Raw crypto logs"
echo ""
