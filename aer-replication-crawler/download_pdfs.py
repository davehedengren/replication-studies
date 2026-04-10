"""Download the PDFs we found. Tries each URL in order until one works."""

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

import config

BASE_DIR = config.BASE_DIR
PDFS_INDEX = config.PAPER_PDFS_INDEX
PDF_DIR = config.PDF_ARCHIVE
STATUS_JSON = BASE_DIR / "pdf_download_status.json"

DELAY = 1.5
TIMEOUT = 120
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def load_status():
    if STATUS_JSON.exists():
        with open(STATUS_JSON) as f:
            return json.load(f)
    return {}


def save_status(status):
    with open(STATUS_JSON, "w") as f:
        json.dump(status, f, indent=2)


def try_download(url, dest):
    """Try downloading. Returns (success, error_msg, bytes_downloaded)."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            ct = resp.headers.get("Content-Type", "")
            data = resp.read()

        if not data.startswith(b"%PDF"):
            return False, f"not_pdf (ct={ct[:30]})", 0

        if len(data) < 10_000:  # <10KB probably not a real paper
            return False, f"too_small ({len(data)}B)", 0

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return True, None, len(data)

    except urllib.error.HTTPError as e:
        return False, f"http_{e.code}", 0
    except urllib.error.URLError as e:
        return False, f"url_error: {str(e)[:60]}", 0
    except Exception as e:
        return False, f"err: {str(e)[:60]}", 0


def run():
    with open(PDFS_INDEX) as f:
        index = json.load(f)

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    status = load_status()

    # Gather candidates
    candidates = []
    for pid, info in index.items():
        pdfs = info.get("all_pdfs", [])
        if not pdfs:
            continue
        if pid in status and status[pid].get("success"):
            continue
        candidates.append((pid, pdfs))

    print(f"Attempting {len(candidates)} PDFs\n", flush=True)

    ok = 0
    fail = 0
    for i, (pid, pdfs) in enumerate(candidates):
        dest = PDF_DIR / f"{pid}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            status[pid] = {"success": True, "size": dest.stat().st_size, "url": "existing"}
            ok += 1
            continue

        print(f"[{i+1}/{len(candidates)}] {pid} ({len(pdfs)} URLs)", flush=True)

        # Try each URL in order
        attempts = []
        saved = False
        for p in pdfs:
            url = p["url"]
            print(f"    trying: {url[:80]}", flush=True)
            success, err, size = try_download(url, dest)
            attempts.append({"url": url, "ok": success, "err": err, "size": size})
            if success:
                print(f"    OK {size / 1024:.0f} KB", flush=True)
                status[pid] = {"success": True, "url": url, "size": size, "source": p.get("source")}
                ok += 1
                saved = True
                break
            else:
                print(f"    FAIL: {err}", flush=True)
            time.sleep(0.5)

        if not saved:
            status[pid] = {"success": False, "attempts": attempts}
            fail += 1

        if (i + 1) % 10 == 0:
            save_status(status)
        time.sleep(DELAY)

    save_status(status)
    print(f"\nDone: {ok} downloaded, {fail} failed", flush=True)
    print(f"PDFs in: {PDF_DIR}", flush=True)


if __name__ == "__main__":
    run()
