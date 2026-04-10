#!/usr/bin/env bash
# One-shot ccusage installer + JSON-shape probe.
# Run this ONCE before enabling the cron. It installs ccusage globally and
# prints a sample of its output so we can lock in the parser in run.sh.

set -euo pipefail

echo "==> Installing ccusage globally..."
npm install -g ccusage

echo
echo "==> ccusage version:"
ccusage --version || true

echo
echo "==> ccusage default output (human-readable):"
ccusage || true

echo
echo "==> ccusage --json output (use this to lock in the run.sh parser):"
ccusage --json 2>/dev/null || ccusage blocks --json 2>/dev/null || echo "(--json not supported on this version; check 'ccusage --help')"

echo
echo "==> ccusage blocks (current 5-hour window):"
ccusage blocks 2>/dev/null || true

echo
echo "Done. Inspect the JSON shape above and update parse_quota() in run.sh accordingly."
