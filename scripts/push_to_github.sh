#!/usr/bin/env bash
# One-shot: get vkGSplat onto github.com/chienpinglu/vkGSplat.
# Handles whatever half-state `gh repo create` may have left behind.
#
# Usage:
#   chmod +x scripts/push_to_github.sh
#   ./scripts/push_to_github.sh

set -e

OWNER="chienpinglu"
REPO="vkGSplat"
DESC="Compute-first Vulkan path for robotics synthetic data on AI accelerators"
URL="https://github.com/${OWNER}/${REPO}.git"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 0. Sanity: are we in a git repo with at least one commit?
git rev-parse --git-dir >/dev/null 2>&1 \
    || { echo "ERROR: not a git repo. Run \`git init\` first."; exit 1; }
git rev-parse HEAD >/dev/null 2>&1 \
    || { echo "ERROR: no commits yet. Run \`git add -A && git commit\` first."; exit 1; }

# 1. Make sure the local remote is right
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$URL"
else
    git remote add origin "$URL"
fi
echo "[1/3] origin -> $(git remote get-url origin)"

# 2. Make sure the GitHub repo exists
if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: gh CLI not installed. Install with: brew install gh && gh auth login"
    exit 1
fi

if gh repo view "${OWNER}/${REPO}" >/dev/null 2>&1; then
    echo "[2/3] github.com/${OWNER}/${REPO} already exists, skipping create"
else
    echo "[2/3] creating github.com/${OWNER}/${REPO}"
    gh repo create "${OWNER}/${REPO}" --public --description "$DESC"
fi

# 3. Push
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "[3/3] pushing branch '${BRANCH}'"
git push -u origin "$BRANCH"

echo
echo "Done. https://github.com/${OWNER}/${REPO}"
