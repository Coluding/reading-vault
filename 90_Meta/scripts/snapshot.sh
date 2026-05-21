#!/usr/bin/env bash
# Manual snapshot commit of the vault. The Worker commits per-webhook;
# this script is for committing local edits (deep notes you wrote by
# hand, MOC tweaks, etc.) in one batch.

set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -z "$(git status --porcelain)" ]]; then
  echo "Nothing to commit."
  exit 0
fi

git add -A
git commit -m "snapshot $(date -u +%Y-%m-%dT%H:%MZ)"
echo "Snapshot committed. Push manually with: git push"
