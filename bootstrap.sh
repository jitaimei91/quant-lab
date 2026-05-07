#!/usr/bin/env bash
# Quant Lab — fast-path bootstrap.
# Creates a public GitHub repo, sets the Discord webhook secret, enables Pages,
# and triggers the first morning run. Requires: gh CLI authenticated.
#
# Full path (Alpaca + Turso) is added in Phase 2.

set -euo pipefail

REPO_NAME="${REPO_NAME:-quant-lab}"
MODE="${1:-fast}"

err() { echo "error: $*" >&2; exit 1; }

command -v gh >/dev/null 2>&1 || err "gh CLI not found. Install: https://cli.github.com/"
command -v python3 >/dev/null 2>&1 || err "python3 not found."

gh auth status >/dev/null 2>&1 || err "gh not authenticated. Run: gh auth login"

echo
echo "Quant Lab — fast-path setup"
echo "==========================="
echo

if [ "$MODE" != "--fast" ] && [ "$MODE" != "fast" ]; then
  echo "Note: only --fast mode is supported in Phase 1." >&2
fi

# 1) Discord webhook
read -r -p "Paste your Discord webhook URL (or leave blank to skip): " DISCORD_WEBHOOK
echo

# 2) Create repo if it doesn't exist
GH_USER=$(gh api user --jq .login)
REPO_FULL="$GH_USER/$REPO_NAME"

if gh repo view "$REPO_FULL" >/dev/null 2>&1; then
  echo "Repo $REPO_FULL exists, skipping create."
else
  echo "Creating $REPO_FULL ..."
  gh repo create "$REPO_FULL" --public --source=. --remote=origin --push
fi

# 3) Set secret
if [ -n "$DISCORD_WEBHOOK" ]; then
  echo "Setting DISCORD_WEBHOOK secret ..."
  gh secret set DISCORD_WEBHOOK -b "$DISCORD_WEBHOOK" --repo "$REPO_FULL"
else
  echo "Skipping Discord secret (no webhook provided)."
fi

# 4) Enable GitHub Pages (Actions source)
echo "Enabling GitHub Pages ..."
gh api -X POST "/repos/$REPO_FULL/pages" \
  -f "build_type=workflow" >/dev/null 2>&1 || \
  gh api -X PUT "/repos/$REPO_FULL/pages" \
  -f "build_type=workflow" >/dev/null 2>&1 || true

DASHBOARD_URL="https://${GH_USER}.github.io/${REPO_NAME}/"
gh variable set DASHBOARD_URL -b "$DASHBOARD_URL" --repo "$REPO_FULL" || true

# 5) Trigger first run
echo "Triggering first morning run ..."
gh workflow run morning.yml --repo "$REPO_FULL" || true

echo
echo "Done."
echo "  Repo:       https://github.com/$REPO_FULL"
echo "  Dashboard:  $DASHBOARD_URL  (live after first run + Pages build, ~3-5 min)"
echo "  Watch run:  gh run watch --repo $REPO_FULL"
