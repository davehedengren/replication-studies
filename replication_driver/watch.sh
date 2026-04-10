#!/usr/bin/env bash
# Watch live progress of the replication driver.
#
# Usage:
#   ./replication_driver/watch.sh             # auto-follow whichever paper is currently active;
#                                              # switches automatically when a new paper starts
#   ./replication_driver/watch.sh 113192      # follow a specific paper log (no auto-switching)
#
# Stall detection: if you see no new events for 2-3 minutes, something is wrong.

set -euo pipefail

DRIVER="$(cd "$(dirname "$0")" && pwd)"
LOGS="$DRIVER/logs"

exec python3 -u - "$LOGS" "${1:-}" <<'PY'
import json, os, sys, time

LOGS, pinned = sys.argv[1], sys.argv[2] or None

def newest_log():
    try:
        files = [os.path.join(LOGS, f) for f in os.listdir(LOGS) if f.endswith(".log")]
    except FileNotFoundError:
        return None
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def pretty(obj):
    t = obj.get("type", "?")
    ts = time.strftime("%H:%M:%S")
    if t == "system":
        return f"[{ts}] system/{obj.get('subtype','')}"
    if t == "result":
        err = obj.get("is_error", False)
        cost = obj.get("total_cost_usd")
        u = obj.get("usage", {}) or {}
        toks = (u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0)
        marker = "ERROR" if err else "OK"
        cost_s = f" ${cost:.2f}" if cost is not None else ""
        return f"[{ts}] === RESULT {marker}{cost_s} ({toks:,} tokens) ==="
    if t == "assistant":
        out = []
        for c in obj.get("message", {}).get("content", []):
            ct = c.get("type")
            if ct == "text":
                txt = c.get("text", "").strip().replace("\n", " ")
                if txt:
                    out.append(f"text: {txt[:140]}")
            elif ct == "tool_use":
                name = c.get("name", "?")
                inp = c.get("input", {})
                if name == "Read":
                    d = inp.get("file_path", "")
                    if inp.get("pages"):
                        d += f" [pg={inp['pages']}]"
                elif name == "Bash":
                    d = (inp.get("command", "") or "").replace("\n", " ")[:120]
                elif name in ("Write", "Edit"):
                    d = inp.get("file_path", "")
                elif name == "Glob":
                    d = inp.get("pattern", "")
                elif name == "Grep":
                    d = inp.get("pattern", "")
                elif name == "Agent":
                    d = inp.get("description", "")
                elif name == "TodoWrite":
                    todos = inp.get("todos", []) or []
                    d = f"({len(todos)} items)"
                else:
                    d = json.dumps(inp)[:120]
                out.append(f"{name}({d})")
        return f"[{ts}] " + " | ".join(out) if out else None
    if t == "user":
        for c in obj.get("message", {}).get("content", []):
            if isinstance(c, dict) and c.get("type") == "tool_result":
                err_mark = " (error)" if c.get("is_error") else ""
                result = c.get("content", "")
                if isinstance(result, list):
                    result = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in result)
                snip = (result or "")[:90].replace("\n", " ")
                return f"[{ts}]   -> tool_result{err_mark}: {snip}"
        return None
    if t == "rate_limit_event":
        info = obj.get("rate_limit_info", {})
        return f"[{ts}] rate_limit: {info.get('status')} ({info.get('rateLimitType')})"
    return None

def stream_file(path):
    """Tail a file, yield parsed events. Return when a newer file appears
    (only when not pinned to a specific id)."""
    sys.stderr.write(f"\n=== watching {os.path.basename(path)} ===\n")
    sys.stderr.flush()
    size = 0
    idle = 0.0
    last_newest_check = 0.0
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                idle += 0.5
                # Auto-switch check (every 3 seconds) — only if not pinned
                if pinned is None and time.time() - last_newest_check > 3:
                    last_newest_check = time.time()
                    n = newest_log()
                    if n and n != path and os.path.getmtime(n) > os.path.getmtime(path) + 1:
                        return n
                # Stall warning
                if idle > 0 and int(idle) % 120 == 0 and idle >= 120:
                    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] (no events for {int(idle)}s — possible stall)\n")
                    sys.stderr.flush()
                    idle += 0.5  # avoid repeating immediately
                continue
            idle = 0.0
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                print(line, flush=True)
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            out = pretty(obj)
            if out:
                print(out, flush=True)

# ---- main ----
if pinned:
    path = os.path.join(LOGS, f"{pinned}.log")
    if not os.path.exists(path):
        sys.stderr.write(f"No log at {path}\n")
        sys.exit(1)
    try:
        stream_file(path)
    except KeyboardInterrupt:
        pass
else:
    # Wait for a log to appear, then auto-switch to newer ones as they show up.
    try:
        current = newest_log()
        while True:
            if current is None:
                sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] waiting for a paper log to appear in {LOGS}...\n")
                sys.stderr.flush()
                while current is None:
                    time.sleep(3)
                    current = newest_log()
            nxt = stream_file(current)
            current = nxt
    except KeyboardInterrupt:
        pass
PY
