#!/usr/bin/env bash
# Prints a one-paragraph status of unprocessed raw inbox items.
# Invoked by the SessionStart hook so Claude sees the count at session start.
#
# Output goes to stdout; the hook captures it as session context.

set -euo pipefail
cd "$(dirname "$0")/../.."

RAW_DIR="00_Inbox/raw"

if [[ ! -d "$RAW_DIR" ]]; then
  echo "Inbox status: raw inbox directory missing."
  exit 0
fi

total=0
day_lines=()

# Iterate raw/*.md files (NOT the processed/ subdir — find with -maxdepth 1)
while IFS= read -r f; do
  count=$(grep -c "^- " "$f" 2>/dev/null || true)
  count=${count:-0}
  if (( count > 0 )); then
    base=$(basename "$f" .md)
    day_lines+=("  - $base: $count item(s)")
    total=$((total + count))
  fi
done < <(find "$RAW_DIR" -maxdepth 1 -type f -name "*.md" | sort)

if (( total == 0 )); then
  echo "Inbox status: empty (no unprocessed items in 00_Inbox/raw/)."
else
  echo "Inbox status: $total unprocessed item(s) in 00_Inbox/raw/:"
  printf '%s\n' "${day_lines[@]}"
  echo ""
  echo "Surface this to the user. They can run /process-inbox to triage."
fi
