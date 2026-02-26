# Alpaca Trading — Deploy Checklist

## Step 1: Create private GitHub repo and push

1. **Create the repo (you must do this in the browser)**  
   Go to: **https://github.com/new**  
   - Repository name: `alpaca-trading`  
   - Visibility: **Private**  
   - Do **not** add a README, .gitignore, or license (we already have them).  
   Click **Create repository**.

2. **Add remote and push (from your Mac)**  
   Replace `YOUR_USERNAME` with your GitHub username, then run:

   ```bash
   cd ~/alpaca-trading
   git remote add origin https://github.com/YOUR_USERNAME/alpaca-trading.git
   git push -u origin main
   ```

   If you use SSH:  
   `git remote add origin git@github.com:YOUR_USERNAME/alpaca-trading.git`

---

## Step 2: Deploy to your droplet

1. **SSH into the droplet**
   ```bash
   ssh root@YOUR_DROPLET_IP
   ```

2. **Run the setup script**

   **If the repo is public** (from the droplet):
   ```bash
   export REPO_URL=https://github.com/YOUR_USERNAME/alpaca-trading.git
   bash <(curl -sL https://raw.githubusercontent.com/YOUR_USERNAME/alpaca-trading/main/deploy/setup_droplet.sh)
   ```

   **If the repo is private**, after SSH’ing in:
   - Install a deploy key or use `gh repo clone` with `gh auth login`, or  
   - Clone from your machine with `git bundle` and copy, or  
   - Use a Personal Access Token in the URL:  
     `https://TOKEN@github.com/YOUR_USERNAME/alpaca-trading.git`  
   Then run the script locally on the droplet:
   ```bash
   cd /opt/alpaca-trading   # or wherever you cloned
   bash deploy/setup_droplet.sh
   ```
   (Set `REPO_URL` first if you cloned manually and the script expects it.)

---

## Step 3: Configure and start services (on the droplet)

1. **Create `.env`**
   ```bash
   nano /opt/alpaca-trading/.env
   ```
   Add your Alpaca keys and any other config (same as your local `.env.example`).

2. **Install and enable systemd services**
   ```bash
   bash /opt/alpaca-trading/deploy/install_services.sh
   ```

3. **Check status**
   ```bash
   # Equities
   systemctl status alpaca-monitor alpaca-daily alpaca-intraday
   # Crypto
   systemctl status alpaca-crypto-monitor alpaca-crypto.timer
   ```

---

## What runs on the droplet

### Equities (market hours only)

| Service          | Frequency              | What it does                                                                 |
|------------------|------------------------|-------------------------------------------------------------------------------|
| Price Monitor    | Every 30s              | Checks positions, triggers stop-loss / trailing stop exits instantly        |
| Daily Engine     | 9:35 AM ET, Mon-Fri    | Full 5-strategy run with FinBERT + earnings + confluence                     |
| Intraday Scanner | Every 15 min, Mon-Fri  | Dip-buy and breakout signals on 15-min bars                                  |

### Crypto (24/7)

| Service          | Frequency              | What it does                                                                 |
|------------------|------------------------|-------------------------------------------------------------------------------|
| Crypto Monitor   | Every 60s, 24/7        | Monitors crypto positions: hard stops, trailing stops, scale-out exits       |
| Crypto Engine    | Every 4h, 24/7         | 4 crypto strategies: momentum, trend, mean reversion, BTC dominance          |

All services auto-restart on crash, survive reboots, and use log rotation.

### Crypto-specific env vars (optional, add to `.env`)

```bash
# Crypto risk limits (defaults shown - only add if you want to override)
CRYPTO_MAX_DAILY_LOSS=3000
CRYPTO_DRAWDOWN_KILL_SWITCH=0.15
CRYPTO_METADATA_PATH=/opt/alpaca-trading/data/crypto_position_metadata.json
```

---

## Local development (macOS)

For testing on your Mac before deploying to the droplet:

```bash
# Run crypto strategy engine once (paper trading)
cd ~/alpaca-trading/strategy-engine
uv run python ../scripts/run_crypto.py --paper

# Run crypto monitor in a terminal (paper trading)
uv run python ../scripts/crypto_monitor.py --interval 60

# Or use the cron wrapper (for crontab/launchd)
# Runs every 4 hours:
0 */4 * * * /Users/Tarek/alpaca-trading/scripts/crypto_cron.sh
```
