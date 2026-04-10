"""Re-download projects that were previously downloaded but lost from disk.

Uses CDP to set Chrome's download path directly to the external drive.
"""

import csv
import os
import glob
import shutil
import time
import random
from pathlib import Path
from playwright.sync_api import sync_playwright

import config

DEST_DIR = config.PACKAGE_ARCHIVE
TRACKER_CSV = config.TRACKER_CSV
CDP_URL = config.CDP_URL
TMPDIR = os.environ.get("TMPDIR", "/tmp")


def get_ids_to_redownload():
    """Get project IDs marked downloaded but not on disk."""
    on_disk = set()
    if DEST_DIR.exists():
        for f in os.listdir(DEST_DIR):
            pid = f.replace(".zip", "").split("-V")[0]
            on_disk.add(pid)

    need = []
    with open(TRACKER_CSV) as f:
        for row in csv.DictReader(f):
            if row["downloaded"] == "True" and row["project_id"] not in on_disk:
                need.append(row["project_id"])
    return need


def find_new_downloads(before_snapshot):
    """Find new files in playwright artifact dirs and ~/Downloads."""
    new_files = set()

    # Check playwright artifact dirs
    for d in glob.glob(os.path.join(TMPDIR, "playwright-artifacts-*")):
        for f in glob.glob(os.path.join(d, "*")):
            if os.path.isfile(f) and f not in before_snapshot and os.path.getsize(f) > 1000:
                new_files.add(f)

    # Also check ~/Downloads
    downloads = os.path.expanduser("~/Downloads")
    for f in glob.glob(os.path.join(downloads, "*")):
        if os.path.isfile(f) and f not in before_snapshot and os.path.getsize(f) > 1000:
            new_files.add(f)

    return new_files


def snapshot_downloads():
    """Snapshot all files in potential download locations."""
    files = set()
    for d in glob.glob(os.path.join(TMPDIR, "playwright-artifacts-*")):
        for f in glob.glob(os.path.join(d, "*")):
            files.add(f)
    downloads = os.path.expanduser("~/Downloads")
    for f in glob.glob(os.path.join(downloads, "*")):
        files.add(f)
    return files


def download_one(page, project_id):
    """Download a single project. Returns size in MB or 0 on failure."""
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = str(DEST_DIR / f"{project_id}.zip")

    if os.path.exists(dest):
        return round(os.path.getsize(dest) / (1024 * 1024), 2)

    try:
        before = snapshot_downloads()

        # Accept terms
        terms_url = (
            f"https://www.openicpsr.org/openicpsr/project/{project_id}/version/V1"
            f"/download/terms?path=/openicpsr/{project_id}/fcr:versions/V1&type=project"
        )
        page.goto(terms_url, wait_until="networkidle", timeout=60000)
        time.sleep(2)

        if "terms" in page.url.lower():
            page.evaluate("""() => {
                const btns = document.querySelectorAll('button, input[type=submit], a');
                for (const b of btns) {
                    const text = (b.textContent || b.value || '').trim();
                    if (text === 'I Agree' || text === 'I agree' || text === 'I AGREE') {
                        b.click(); return true;
                    }
                }
                return false;
            }""")
            page.wait_for_load_state("networkidle", timeout=30000)
            time.sleep(3)
            print(f"    Accepted terms", flush=True)

        # Go to project page and click download
        project_url = f"https://www.openicpsr.org/openicpsr/project/{project_id}/version/V1/view"
        page.goto(project_url, wait_until="networkidle", timeout=60000)
        time.sleep(3)

        btn = None
        for sel in [
            "a:has-text('DOWNLOAD THIS PROJECT')",
            "a:has-text('Download this project')",
            "a:has-text('Download This Project')",
            "button:has-text('DOWNLOAD THIS PROJECT')",
            "a:has-text('Download All')",
        ]:
            btn = page.query_selector(sel)
            if btn:
                break

        if not btn:
            print(f"    No download button found", flush=True)
            return 0

        btn.click()
        time.sleep(5)

        # Wait for download to appear and complete
        start = time.time()
        last_size = 0
        stall_count = 0

        while time.time() - start < 900:  # 15 min timeout for large files
            time.sleep(5)

            new_files = find_new_downloads(before)

            if not new_files:
                elapsed = int(time.time() - start)
                if elapsed > 120:
                    print(f"    No file appeared after {elapsed}s", flush=True)
                    return 0
                continue

            total_size = sum(os.path.getsize(f) for f in new_files if os.path.exists(f))
            elapsed = int(time.time() - start)

            if total_size > last_size:
                last_size = total_size
                stall_count = 0
                if elapsed % 30 < 6:
                    print(f"    Downloading... {total_size / (1024*1024):.0f} MB ({elapsed}s)", flush=True)
                continue

            stall_count += 1
            if stall_count < 3:
                continue

            # Download complete — find the largest new file
            src = max(new_files, key=lambda f: os.path.getsize(f) if os.path.exists(f) else 0)
            size = os.path.getsize(src)
            if size < 1000:
                print(f"    File too small ({size} bytes)", flush=True)
                return 0

            # Move to external drive
            shutil.move(src, dest)

            # Clean up any other new files
            for f in new_files:
                if f != src and os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

            size_mb = size / (1024 * 1024)
            print(f"    Saved {size_mb:.1f} MB -> {dest}", flush=True)
            return round(size_mb, 2)

        print(f"    Timed out after 15 min", flush=True)
        return 0

    except Exception as e:
        print(f"    Error: {e}", flush=True)
        return 0


def run():
    ids = get_ids_to_redownload()
    print(f"Re-downloading {len(ids)} projects to {DEST_DIR}", flush=True)
    print(f"Chrome must be running with --remote-debugging-port=9222\n", flush=True)

    if not ids:
        print("Nothing to re-download!")
        return

    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(CDP_URL)
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()

        done = 0
        failed = []

        for i, pid in enumerate(ids):
            print(f"\n[{i+1}/{len(ids)}] Project {pid}", flush=True)

            size = download_one(page, pid)
            if size > 0:
                done += 1
                print(f"    OK ({done} done, {len(ids)-i-1} remaining)", flush=True)
            else:
                failed.append(pid)
                print(f"    FAILED", flush=True)

            # Slow and reliable
            delay = random.randint(30, 60)
            print(f"    {delay}s pause", flush=True)
            time.sleep(delay)

        print(f"\n\nDone: {done}/{len(ids)} downloaded", flush=True)
        if failed:
            print(f"Failed ({len(failed)}): {failed}", flush=True)


if __name__ == "__main__":
    run()
