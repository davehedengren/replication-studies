#!/usr/bin/env bash
# Sequential back-to-back paper runner.
#
# Calls run.sh in a loop until the queue is drained. Between papers, there
# is no dead time — the next paper starts as soon as the previous one
# completes. Only one paper runs at a time, so peak memory / CPU / API
# load stays bounded to a single claude -p subprocess.
#
# All the safety gates (disk, quota, single-instance lock) live in run.sh;
# this script just invokes run.sh and reacts to whether a paper completed.
# If run.sh skipped a tick because a gate was tight, loop.sh waits
# LOOP_WAIT seconds and retries. If the queue is empty, loop.sh exits.
#
# Usage:
#   ./replication_driver/loop.sh                   # run in the foreground
#   nohup ./replication_driver/loop.sh &           # detach (survives logout)
#   LOOP_WAIT=120 ./replication_driver/loop.sh     # shorter retry delay
#
# Ctrl-C is safe: run.sh's own trap cleans up the lock directory, and this
# script catches SIGINT / SIGTERM to print a final summary before exiting.

set -euo pipefail

DRIVER="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DRIVER/.." && pwd)"
QUEUE="$DRIVER/queue.txt"
DONE="$DRIVER/done.txt"
RUN_LOG="$DRIVER/run.log"

# How long to wait after a no-op run.sh call before retrying. This only
# kicks in when a pre-flight gate rejected the tick (disk tight, quota
# tight, or the single-instance lock was held by another run) — successful
# paper completions do not wait, they proceed to the next paper immediately.
LOOP_WAIT="${LOOP_WAIT:-300}"  # 5 minutes

log() {
  local line
  line="$(date '+%Y-%m-%d %H:%M:%S') [loop] $*"
  echo "$line"
  echo "$line" >> "$RUN_LOG"
}

remaining_count() {
  comm -23 <(sort -u "$QUEUE") <(sort -u "$DONE" 2>/dev/null) | wc -l | tr -d ' '
}

done_count() {
  wc -l < "$DONE" | tr -d ' '
}

stop() {
  log "caught signal — stopping after the current paper finishes (or immediately if idle)"
  exit 0
}
trap stop INT TERM

cd "$REPO"

start_remaining=$(remaining_count)
log "starting: $start_remaining paper(s) remaining in queue"

processed=0
while true; do
  rem=$(remaining_count)
  if [ "$rem" -eq 0 ]; then
    log "queue drained — processed $processed paper(s) this session"
    exit 0
  fi

  before=$(done_count)
  "$DRIVER/run.sh"
  after=$(done_count)

  if [ "$after" -gt "$before" ]; then
    processed=$(( processed + (after - before) ))
    rem=$(remaining_count)
    log "paper completed; $rem remaining, $processed processed this session"
    # Immediately continue — no dead time between papers.
    continue
  fi

  # run.sh did not advance done.txt. Three possible reasons:
  #   - a gate rejected the tick (disk, quota, or the lock was held)
  #   - run.sh called claude but claude didn't produce a writeup (failed run)
  #   - queue became empty (handled at top of loop on next iteration)
  # For the first two, wait and retry. The run.log will have the reason.
  log "no progress on last tick — sleeping ${LOOP_WAIT}s before retrying"
  log "  (most recent run.log lines:)"
  tail -5 "$RUN_LOG" | sed 's/^/    /'
  sleep "$LOOP_WAIT"
done
