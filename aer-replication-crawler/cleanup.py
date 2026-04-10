"""Cleanup utilities for Playwright temp artifacts.

Playwright creates `playwright-artifacts-*` directories in the system temp
folder on every CDP connection. These never get cleaned up automatically and
can consume tens of GB over a long crawl session, eventually filling the
boot volume. This module provides one cleanup helper and auto-registers it
with `atexit` on import, so any script that does `from cleanup import ...`
automatically gets cleanup on normal exit, `sys.exit()`, and most uncaught
exceptions.

Scripts that want extra coverage should also:
  1. Call `cleanup_playwright_artifacts()` at startup, to sweep stale
     artifacts left behind by a prior run that died abnormally.
  2. Call it periodically during long loops (every N downloads or so).
  3. Install SIGTERM and SIGINT handlers that invoke it before exiting.
"""

import atexit
import glob
import os
import shutil
import tempfile


def cleanup_playwright_artifacts() -> int:
    """Remove Playwright temp artifact dirs. Returns the number of dirs
    successfully removed. Silent on per-dir failures (the active dir for an
    in-use Playwright connection will raise and be skipped).
    """
    tmp = tempfile.gettempdir()
    removed = 0
    for d in glob.glob(os.path.join(tmp, "playwright-artifacts-*")):
        try:
            shutil.rmtree(d)
            removed += 1
        except Exception:
            pass
    return removed


# Belt-and-suspenders: register cleanup to run at interpreter exit. This
# covers normal termination, `sys.exit()`, and uncaught exceptions that
# unwind the whole interpreter. It does NOT cover SIGKILL or hard crashes —
# for those, the startup-time cleanup in main.py handles the leftover state.
atexit.register(cleanup_playwright_artifacts)
