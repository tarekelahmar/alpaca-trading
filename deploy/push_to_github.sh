#!/bin/bash
# Run from ~/alpaca-trading after creating the repo at github.com/new
# Usage: GITHUB_USER=YourUsername bash deploy/push_to_github.sh

set -e
GITHUB_USER="${GITHUB_USER:-}"
if [ -z "$GITHUB_USER" ]; then
  echo "Usage: GITHUB_USER=YourUsername bash deploy/push_to_github.sh"
  echo "Example: GITHUB_USER=tarek bash deploy/push_to_github.sh"
  exit 1
fi

cd "$(dirname "$0")/.."
if git remote get-url origin 2>/dev/null; then
  echo "Remote 'origin' already exists. Update with:"
  echo "  git remote set-url origin https://github.com/$GITHUB_USER/alpaca-trading.git"
  exit 1
fi
git remote add origin "https://github.com/$GITHUB_USER/alpaca-trading.git"
git push -u origin main
echo "Done. Repo: https://github.com/$GITHUB_USER/alpaca-trading"
