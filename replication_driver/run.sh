#!/usr/bin/env bash
# Cron entry point for the AER replication driver.
# Processes ONE paper per invocation and exits, so each cron tick gets a
# fresh `claude -p` process (and therefore a fresh context window).
#
# Schedule (suggested): 0 */6 * * * /Users/davehedengren/code/replication-studies/replication_driver/run.sh

set -euo pipefail

REPO="/Users/davehedengren/code/replication-studies"
DRIVER="$REPO/replication_driver"
SSD_ZIPS="/Volumes/Extreme SSD/AER_replication_data"
SSD_PDFS="/Volumes/Extreme SSD/AER_replication_data_pdfs"

QUEUE="$DRIVER/queue.txt"
DONE="$DRIVER/done.txt"
RUN_LOG="$DRIVER/run.log"
LOGS_DIR="$DRIVER/logs"

# Quota gate: don't start a new paper if the current 5-hour block has already
# consumed more than this many tokens. ccusage doesn't expose plan limits, so
# this is an absolute ceiling, not a percentage. Tune after the dry run reveals
# how much one paper actually costs.
#
#   Max 5x  ≈ ~88M  tokens / 5h window  → set MAX_TOKENS_PER_BLOCK ≈ 60M
#   Max 20x ≈ ~440M tokens / 5h window  → set MAX_TOKENS_PER_BLOCK ≈ 300M
#
# Default is conservative (60M) to avoid blowing the smaller plan.
MAX_TOKENS_PER_BLOCK="${MAX_TOKENS_PER_BLOCK:-60000000}"

# Hard per-paper spend cap, enforced by `claude --max-budget-usd`. If a paper
# would cost more than this, claude exits early. This is the safety net the
# user asked for — even if quality drops slightly on hard papers, we never
# blow the bank on a single replication. Tune after the dry run.
MAX_BUDGET_PER_PAPER="${MAX_BUDGET_PER_PAPER:-10.00}"

# Model to use for each replication. Pin for determinism across runs.
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-6}"

# Pre-flight disk gate: skip the tick if the boot volume has fewer than this
# many GB available. macOS APFS reports "available" as the free space *after*
# accounting for purgeable blocks (local snapshots, cached content, Time
# Machine local snapshots), so this is the "real" number — if it's small,
# claude will get ENOSPC mid-run and either fail or, worse, try to free space
# by deleting user caches. Default 5 GB is enough for one paper's outputs
# (~30 MB) plus claude's session log (~10 MB) plus Bash tool output files,
# with headroom for pip-installing an extra package.
MIN_FREE_GB="${MIN_FREE_GB:-5}"

mkdir -p "$LOGS_DIR"
cd "$REPO"

log() {
  local line
  line="$(date '+%Y-%m-%d %H:%M:%S') $*"
  echo "$line" >> "$RUN_LOG"
  # If stderr is attached to a terminal, also echo there so manual runs show
  # lifecycle events live. Cron has no TTY, so this stays silent under cron
  # and doesn't fill the local mail queue.
  if [ -t 2 ]; then
    echo "$line" >&2
  fi
}

# Single-instance lock. mkdir is atomic on POSIX, so two concurrent invocations
# can never both succeed. If a cron tick fires while a prior run is still
# working (or while a manual run is in progress), the new instance exits
# immediately rather than racing on queue.txt / done.txt.
LOCK_DIR="$DRIVER/.run.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  # Check if the lock is stale (owning pid no longer exists).
  if [ -f "$LOCK_DIR/pid" ]; then
    LOCK_PID=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
    if [ -n "$LOCK_PID" ] && ! kill -0 "$LOCK_PID" 2>/dev/null; then
      log "stale lock from dead pid $LOCK_PID; removing and retrying"
      rm -rf "$LOCK_DIR"
      mkdir "$LOCK_DIR" || { log "SKIP: lock still held"; exit 0; }
    else
      log "SKIP: another run is in progress (pid $LOCK_PID)"
      exit 0
    fi
  else
    log "SKIP: another run is in progress"
    exit 0
  fi
fi
echo $$ > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

# Resolve a fresh PATH for cron (cron has a minimal env by default).
# These are the directories where node/npm/claude/ccusage typically live on macOS.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.npm-global/bin:$PATH"

# ----- 1. Pre-flight quota check ------------------------------------------------
# We shell out to ccusage and ask it for the current 5-hour block's remaining %.
# The exact JSON shape varies by ccusage version; install_ccusage.sh prints it.
# This parser tries a couple common shapes and falls back to "skip the check"
# if it can't find a number — better to over-run a window than to never run.

check_disk() {
  # `df -k /` returns 512-byte blocks on macOS; -k normalizes to 1024-byte blocks.
  # Portable parse: skip header, field 4 is Avail in KB.
  local avail_kb avail_gb
  avail_kb=$(df -k / | awk 'NR==2 {print $4}')
  if [ -z "$avail_kb" ] || ! [ "$avail_kb" -eq "$avail_kb" ] 2>/dev/null; then
    log "WARN: could not parse df output, skipping disk check"
    return 0
  fi
  avail_gb=$(( avail_kb / 1024 / 1024 ))
  log "disk check: ${avail_gb} GB free on / (threshold ${MIN_FREE_GB} GB)"
  if [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
    log "SKIP: disk too tight (${avail_gb} GB < ${MIN_FREE_GB} GB)"
    return 1
  fi
  return 0
}

check_quota() {
  if ! command -v ccusage >/dev/null 2>&1; then
    log "WARN: ccusage not on PATH, skipping quota check"
    return 0
  fi

  local json
  json=$(ccusage blocks --json 2>/dev/null || echo "")
  if [ -z "$json" ]; then
    log "WARN: ccusage produced no JSON, skipping quota check"
    return 0
  fi

  # Pull the active 5-hour block. ccusage shape:
  #   {"blocks": [{"isActive": bool, "totalTokens": int, "burnRate": {...},
  #                "projection": {"totalTokens": int, "remainingMinutes": int}, ...}]}
  # We use:
  #   - totalTokens (already used in this block)
  #   - projection.totalTokens (where the burn rate says we'll end up)
  # If the projection already exceeds MAX_TOKENS_PER_BLOCK, skip — it means the
  # rest of the block doesn't have room for another paper at the current rate.
  read -r used projected <<<"$(printf '%s' "$json" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    blocks = data.get("blocks", [])
    active = next((b for b in blocks if b.get("isActive")), None)
    if not active:
        print("0 0")
        sys.exit(0)
    used = int(active.get("totalTokens") or 0)
    proj = (active.get("projection") or {}).get("totalTokens") or used
    print(f"{used} {int(proj)}")
except Exception:
    print("0 0")
' 2>/dev/null)"
  used=${used:-0}
  projected=${projected:-0}

  log "quota check: active block used=${used} projected=${projected} cap=${MAX_TOKENS_PER_BLOCK}"

  if [ "$used" -ge "$MAX_TOKENS_PER_BLOCK" ]; then
    log "SKIP: active block already at ${used} tokens (>= cap ${MAX_TOKENS_PER_BLOCK})"
    return 1
  fi
  return 0
}

# ----- 2. Pick the next paper ---------------------------------------------------

next_paper() {
  comm -23 <(sort -u "$QUEUE") <(sort -u "$DONE") | head -1
}

# ----- 3. Main ------------------------------------------------------------------

if ! check_disk; then
  exit 0
fi

if ! check_quota; then
  exit 0
fi

NEXT=$(next_paper)
if [ -z "$NEXT" ]; then
  log "queue empty — nothing to do"
  exit 0
fi

log "starting paper $NEXT"

# Sanity check inputs
ZIP="$SSD_ZIPS/${NEXT}.zip"
PDF="$SSD_PDFS/${NEXT}.pdf"
if [ ! -f "$ZIP" ]; then
  log "FAILED $NEXT: zip not found at $ZIP"
  exit 1
fi
if [ ! -f "$PDF" ]; then
  log "FAILED $NEXT: pdf not found at $PDF"
  exit 1
fi

# Unzip the replication package next to the existing replication_* dirs.
# We use Python's zipfile module instead of `unzip(1)` so we can tolerate
# per-file errors — openICPSR packages occasionally contain filenames with
# non-UTF-8 bytes that macOS APFS cannot represent, and `unzip` aborts on
# those with a non-zero exit code even though the load-bearing content
# (Code/, Data/, etc.) extracts fine. The Python extractor skips the
# individual bad entries and logs them, then continues.
DEST="$REPO/${NEXT}-V1"
# Re-extract if the dest is missing OR exists but is empty. An empty dest can
# be left behind by a failed extract attempt (e.g., the Python extractor
# creates the dir before opening the zip; if the zip read raises, the empty
# dir persists). Without this guard, the next run sees the empty dir and
# silently hands a packageless directory to claude.
if [ ! -d "$DEST" ] || [ -z "$(ls -A "$DEST" 2>/dev/null)" ]; then
  rm -rf "$DEST"
  log "extracting $ZIP -> $DEST"
  set +e
  EXTRACT_OUT=$("$REPO/venv/bin/python" "$DRIVER/extract_zip.py" "$ZIP" "$DEST" "$LOGS_DIR/${NEXT}.extract.log" 2>&1)
  EXTRACT_RC=$?
  set -e
  log "extract: $EXTRACT_OUT"
  if [ $EXTRACT_RC -ne 0 ]; then
    log "FAILED $NEXT: extract error rc=$EXTRACT_RC (see ${NEXT}.extract.log)"
    exit 1
  fi
fi

# Build the per-paper prompt by substituting {{ID}} into the template.
PROMPT=$(sed "s/{{ID}}/$NEXT/g" "$DRIVER/prompt_template.md")

# Invoke Claude Code in headless mode. Each invocation is a fresh process,
# so context is guaranteed empty.
log "invoking claude -p for $NEXT"
PAPER_LOG="$LOGS_DIR/${NEXT}.log"
{
  echo "===== run started $(date) ====="
  echo "===== paper $NEXT ====="
} >> "$PAPER_LOG"

if claude -p "$PROMPT" \
    --dangerously-skip-permissions \
    --model "$CLAUDE_MODEL" \
    --max-budget-usd "$MAX_BUDGET_PER_PAPER" \
    --no-session-persistence \
    --output-format stream-json \
    --verbose \
    --include-partial-messages \
    >> "$PAPER_LOG" 2>&1; then
  CLAUDE_EXIT=0
else
  CLAUDE_EXIT=$?
fi
echo "===== claude exit $CLAUDE_EXIT $(date) =====" >> "$PAPER_LOG"

# ----- 4. Verify completion -----------------------------------------------------

WRITEUP="$REPO/replication_${NEXT}/writeup_${NEXT}.md"
INFEASIBLE="$REPO/replication_${NEXT}/INFEASIBLE.md"

if [ -f "$WRITEUP" ] || [ -f "$INFEASIBLE" ]; then
  echo "$NEXT" >> "$DONE"
  if [ -f "$WRITEUP" ]; then
    log "completed $NEXT (writeup produced)"
  else
    log "completed $NEXT (marked infeasible)"
  fi
else
  log "FAILED $NEXT (no writeup or INFEASIBLE.md produced; claude exit=$CLAUDE_EXIT)"
fi
