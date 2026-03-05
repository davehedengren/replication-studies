#!/usr/bin/env python3
"""
Slow, polite crawler to find and download AEA replication packages
with all publicly available data from openICPSR.

Connects to the already-running OpenClaw browser via CDP.
Logs EVERY checked project to a CSV so we never re-check.
"""

import csv
import json
import time
import random
import os
import re
import tempfile
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
import fitz  # PyMuPDF for PDF text extraction

# --- Config ---
CDP_URL = "http://127.0.0.1:18800"
DOWNLOAD_DIR = "/Users/davehedengren/code/replication_studies"
TRACKER_CSV = os.path.join(DOWNLOAD_DIR, "project_log.csv")
TRACKER_JSON = os.path.join(DOWNLOAD_DIR, "download_tracker.json")
TARGET_COUNT = 100
MIN_DELAY = 30
MAX_DELAY = 60
MAX_PAGES = 200
SEARCH_BASE = "https://www.openicpsr.org/openicpsr/search/aea/studies"
SEARCH_SORT = "DATEUPDATED%20desc"
TARGET_YEARS = {"2021", "2022"}  # filter to these paper years

# Patterns indicating all data is publicly available
PUBLIC_DATA_PATTERNS = [
    r"all\s+data\s+.*?publicly\s+available",
    r"all\s+data\s+.*?publicly\s+accessible",
    r"raw\s+data.*?publicly\s+available",
    r"no\s+proprietary\s+or\s+confidential\s+data",
    r"data\s+.*?open.access",
]

RESTRICTED_PATTERNS = [
    r"(?<!no )data\s+.*?not\s+publicly\s+available",
    r"restricted.use\s+data",
    r"(?:some|certain)\s+data.*?cannot\s+be\s+(shared|redistributed|provided)",
    r"some\s+data.*?not.*?publicly\s+available",
    r"data.*?available\s+under\s+restricted",
]

# Patterns for data that's freely downloadable from external sources
# (not included in package, but publicly obtainable with registration/download)
EXTERNAL_PUBLIC_PATTERNS = [
    r"freely\s+downloaded\s+from",
    r"can\s+be\s+(freely\s+)?downloaded\s+(from|at)",
    r"publicly\s+available.*?(download|https?://|www\.)",
    r"available\s+(for\s+download\s+)?(from|at)\s+https?://",
    r"available\s+.*?registration",
    r"download.*?from\s+https?://",
    r"provided\s*\|\s*no.*?download",  # table format: Provided | No ... download
]

CSV_FIELDS = [
    "project_id", "title", "checked_at", "has_readme", "readme_type",
    "data_public", "data_external_public", "downloaded", "download_size_mb",
    "error", "notes"
]


def load_checked_ids():
    """Load set of already-checked project IDs from CSV."""
    checked = set()
    if os.path.exists(TRACKER_CSV):
        with open(TRACKER_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                checked.add(row["project_id"])
    return checked


def append_csv_row(row_dict):
    """Append one row to the CSV log."""
    file_exists = os.path.exists(TRACKER_CSV)
    with open(TRACKER_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def load_json_tracker():
    try:
        with open(TRACKER_JSON) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"status": "starting", "downloaded_count": 0, "checked_count": 0}


def save_json_tracker(t):
    with open(TRACKER_JSON, "w") as f:
        json.dump(t, f, indent=2)


def polite_wait(label=""):
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    print(f"  ⏳ {delay:.0f}s {label}", flush=True)
    time.sleep(delay)


def check_session_alive(page):
    """Check if we're still logged in. If session expired, wait and retry."""
    max_waits = 60  # 60 x 5 min = 5 hours max wait
    for i in range(max_waits):
        url = page.url.lower()
        content = page.content().lower() if page.url else ""
        if "login" in url or "sign in" in url or "signin" in url or "log in with" in content:
            if i == 0:
                print(f"\n🔒 SESSION EXPIRED — waiting for re-login...", flush=True)
                print(f"   Please log back in at the browser window.", flush=True)
            # Wait 5 minutes and check again
            time.sleep(300)
            try:
                page.reload(wait_until="networkidle", timeout=30000)
            except:
                pass
        else:
            if i > 0:
                print(f"  ✅ Session restored! Resuming...", flush=True)
            return True
    print(f"  ❌ Session never restored after {max_waits * 5} minutes", flush=True)
    return False


def extract_project_id(url):
    m = re.search(r"/project/(\d+)", url)
    return m.group(1) if m else None


def classify_data_availability(text):
    """
    Classify data availability from README text.
    Returns one of:
      "included"  — all data included in package, publicly available
      "external"  — data freely downloadable from external sources (not in package)
      "restricted" — data has genuine access restrictions
      "unknown"   — can't determine
    """
    text_lower = text.lower()

    # Check for hard restricted patterns first
    for pat in RESTRICTED_PATTERNS:
        if re.search(pat, text_lower):
            # But check if this is actually "redistribution restricted but freely downloadable"
            has_external = any(re.search(p, text_lower) for p in EXTERNAL_PUBLIC_PATTERNS)
            if has_external:
                return "external"
            return "restricted"

    # Check for all-data-included patterns
    for pat in PUBLIC_DATA_PATTERNS:
        if re.search(pat, text_lower):
            return "included"

    # Check for external-but-public patterns
    for pat in EXTERNAL_PUBLIC_PATTERNS:
        if re.search(pat, text_lower):
            return "external"

    return "unknown"


def check_text_for_public_data(text):
    """Legacy wrapper — returns True for included data."""
    return classify_data_availability(text) == "included"


def _scan_links_for_readme(page):
    """Scan current page's links for README files. Returns list of (type, name, path)."""
    try:
        links = page.query_selector_all("a")
    except Exception:
        return []
    readmes = []
    for link in links:
        try:
            text = (link.inner_text() or "").strip()
            href = link.get_attribute("href") or ""
            tl = text.lower()
            if "readme" in tl and "type=file" in href:
                path_m = re.search(r"path=([^&]+)", href)
                if path_m:
                    fpath = path_m.group(1)
                    if tl.endswith(".md"):
                        readmes.append(("md", text, fpath))
                    elif tl.endswith(".txt"):
                        readmes.append(("txt", text, fpath))
                    elif tl.endswith(".pdf") or tl.endswith(".docx"):
                        readmes.append(("pdf", text, fpath))
                    else:
                        readmes.append(("txt", text, fpath))
        except:
            continue
    priority = {"md": 0, "txt": 1, "pdf": 3}
    readmes.sort(key=lambda x: priority.get(x[0], 2))
    return readmes


def _get_folder_links(page):
    """Get folder links from current page."""
    try:
        links = page.query_selector_all("a")
    except Exception:
        return []
    folders = []
    for link in links:
        try:
            text = (link.inner_text() or "").strip()
            href = link.get_attribute("href") or ""
            if "type=folder" in href:
                folders.append((text, href))
        except:
            continue
    return folders


def find_readme_info(page, project_id):
    """Find README filename and path.

    Search strategy:
    1) root folder
    2) recurse through subfolders (BFS) up to depth 2

    Returns (type, name, path) or None.
    """
    # Check root level first
    readmes = _scan_links_for_readme(page)
    if readmes:
        return readmes[0]

    original_url = page.url

    def _to_full_url(href):
        if href.startswith("?"):
            return f"https://www.openicpsr.org/openicpsr/project/{project_id}/version/V1/view{href}"
        if href.startswith("http"):
            return href
        return f"https://www.openicpsr.org{href}"

    # BFS over folders to catch README files in nested dirs like /data/... 
    # Limit breadth/depth to stay polite + avoid runaway traversals.
    max_depth = 2
    max_folders = 40
    visited = set()
    queue = []

    for folder_name, folder_href in _get_folder_links(page):
        queue.append((folder_name, folder_href, 1, folder_name))

    scanned = 0
    while queue and scanned < max_folders:
        folder_name, folder_href, depth, trail = queue.pop(0)
        folder_url = _to_full_url(folder_href)
        if folder_url in visited:
            continue
        visited.add(folder_url)
        scanned += 1

        try:
            page.goto(folder_url, wait_until="networkidle", timeout=45000)
            time.sleep(1)

            readmes = _scan_links_for_readme(page)
            if readmes:
                print(f"  📂 Found README in subfolder: {trail}/", flush=True)
                return readmes[0]

            if depth < max_depth:
                for child_name, child_href in _get_folder_links(page):
                    queue.append((child_name, child_href, depth + 1, f"{trail}/{child_name}"))
        except Exception:
            continue

    # Navigate back to project root
    try:
        page.goto(original_url, wait_until="networkidle", timeout=30000)
        time.sleep(1)
    except:
        pass

    return None


def fetch_readme_text(page, project_id, file_path, ftype):
    """Fetch README content via getBinary endpoint. Handles text and PDF."""
    if ftype == "pdf":
        return _fetch_pdf_readme(page, project_id, file_path)

    ctype = "text/x-web-markdown" if ftype == "md" else "text/plain"
    try:
        content = page.evaluate(
            """async ([pid, fpath, ct]) => {
                try {
                    const resp = await fetch(
                        `/openicpsr/project/${pid}/version/V1/getBinary?filePath=${fpath}&contentType=${ct}`,
                        {credentials: 'include'}
                    );
                    if (!resp.ok) return '';
                    return await resp.text();
                } catch(e) { return ''; }
            }""",
            [project_id, file_path, ctype],
        )
        return content or ""
    except Exception as e:
        print(f"  ⚠️  Fetch failed: {e}", flush=True)
        return ""


def _fetch_pdf_readme(page, project_id, file_path):
    """Download PDF README to temp file and extract text with PyMuPDF."""
    try:
        import base64
        pdf_b64 = page.evaluate(
            """async ([pid, fpath]) => {
                try {
                    const resp = await fetch(
                        `/openicpsr/project/${pid}/version/V1/getBinary?filePath=${fpath}&contentType=application/pdf`,
                        {credentials: 'include'}
                    );
                    if (!resp.ok) return '';
                    const buf = await resp.arrayBuffer();
                    const bytes = new Uint8Array(buf);
                    let binary = '';
                    for (let i = 0; i < bytes.length; i++) {
                        binary += String.fromCharCode(bytes[i]);
                    }
                    return btoa(binary);
                } catch(e) { return ''; }
            }""",
            [project_id, file_path],
        )
        if not pdf_b64:
            return ""

        pdf_bytes = base64.b64decode(pdf_b64)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            doc = fitz.open(tmp.name)
            text = ""
            for pg in doc:
                text += pg.get_text()
            doc.close()
            return text
    except Exception as e:
        print(f"  ⚠️  PDF extract failed: {e}", flush=True)
        return ""


def get_title(page):
    try:
        h1 = page.query_selector("h1")
        if h1:
            return h1.inner_text().strip()[:200]
    except:
        pass
    return "Unknown"


def in_target_years(page):
    """Heuristic year filter using visible project page text."""
    try:
        text = (page.inner_text("body") or "")[:20000]
    except Exception:
        return False

    for y in TARGET_YEARS:
        if y in text:
            return True
    return False


def _save_download(dl, project_id):
    """Save a Playwright download object. Returns size in MB or 0."""
    dest = os.path.join(DOWNLOAD_DIR, f"{project_id}.zip")
    dl.save_as(dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  ✅ Downloaded {size_mb:.1f} MB → {dest}", flush=True)
    return round(size_mb, 2)


def download_project(page, project_id):
    """
    Download project ZIP. Handles the Terms of Use interstitial.
    
    Strategy:
    1. Click Download → goes to Terms page
    2. Accept Terms → redirects back to project page (may or may not trigger download)
    3. If no download triggered, click Download again (terms now accepted → direct download)
    
    Returns file size in MB or 0 on failure.
    """
    project_url = f"https://www.openicpsr.org/openicpsr/project/{project_id}/version/V1/view"

    try:
        # Step 0: Ensure we're on the project page
        if f"/project/{project_id}/" not in page.url or "terms" in page.url:
            page.goto(project_url, wait_until="networkidle", timeout=60000)
            time.sleep(2)

        # Step 1: Click Download → navigate to Terms
        btn = page.query_selector(
            "a:has-text('DOWNLOAD THIS PROJECT'), "
            "a:has-text('Download this project'), "
            "button:has-text('Download this project')"
        )
        if not btn:
            print(f"  ❌ No download button", flush=True)
            return 0

        btn.click()
        page.wait_for_load_state("networkidle", timeout=30000)
        time.sleep(2)

        # Step 2: Accept Terms if present
        if "terms" in page.url.lower():
            print(f"  📋 Accepting Terms of Use...", flush=True)
            # First try: maybe I Agree triggers download directly
            try:
                with page.expect_download(timeout=30000) as dl_info:
                    page.evaluate("""() => {
                        const btns = document.querySelectorAll('button, input[type=submit], a');
                        for (const b of btns) {
                            if (b.textContent.trim() === 'I Agree' || b.value === 'I Agree') {
                                b.click(); return true;
                            }
                        }
                        return false;
                    }""")
                return _save_download(dl_info.value, project_id)
            except PWTimeout:
                # I Agree redirected back without triggering download
                print(f"  ℹ️  Terms accepted, retrying download...", flush=True)
                time.sleep(3)

        # Step 3: We're back on the project page (or never left).
        # Terms should be accepted now. Try download directly.
        if f"/project/{project_id}/" not in page.url:
            page.goto(project_url, wait_until="networkidle", timeout=60000)
            time.sleep(2)

        btn = page.query_selector(
            "a:has-text('DOWNLOAD THIS PROJECT'), "
            "a:has-text('Download this project'), "
            "button:has-text('Download this project')"
        )
        if not btn:
            print(f"  ❌ No download button after terms", flush=True)
            return 0

        try:
            with page.expect_download(timeout=300000) as dl_info:
                btn.click()
                # Might go to terms again — handle it
                time.sleep(3)
                if "terms" in page.url.lower():
                    page.evaluate("""() => {
                        const btns = document.querySelectorAll('button, input[type=submit], a');
                        for (const b of btns) {
                            if (b.textContent.trim() === 'I Agree' || b.value === 'I Agree') {
                                b.click(); return true;
                            }
                        }
                        return false;
                    }""")
            return _save_download(dl_info.value, project_id)
        except PWTimeout:
            print(f"  ❌ Download timed out after retry", flush=True)
            return 0

    except Exception as e:
        print(f"  ❌ Download error: {e}", flush=True)
        return 0


def get_search_projects(page, page_num):
    # openICPSR AEA endpoint paginates with `start`, not `paging.startRow`.
    # start=0,25,50,... for rows=25.
    start = (page_num - 1) * 25
    url = f"{SEARCH_BASE}?start={start}&ARCHIVE=aea&sort={SEARCH_SORT}&rows=25"
    page.goto(url, wait_until="domcontentloaded", timeout=90000)
    time.sleep(3)

    # Check for session expiry after navigation
    if not check_session_alive(page):
        return []

    # Re-navigate if we were bounced away from search page
    if "search/aea/studies" not in page.url:
        page.goto(url, wait_until="domcontentloaded", timeout=90000)
        time.sleep(3)

    links = page.query_selector_all("a[href*='/openicpsr/project/']")
    projects = []
    seen = set()
    for link in links:
        href = link.get_attribute("href") or ""
        pid = extract_project_id(href)
        if pid and pid not in seen:
            seen.add(pid)
            full = href if href.startswith("http") else f"https://www.openicpsr.org{href}"
            projects.append((pid, full))
    return projects


def run():
    checked_ids = load_checked_ids()
    # Also skip the ones already downloaded manually
    already_have = {"179162", "192297", "227802", "228101", "206261"}
    checked_ids.update(already_have)

    tracker = load_json_tracker()
    downloaded_count = tracker.get("downloaded_count", 0)

    # Count existing downloads from CSV
    if os.path.exists(TRACKER_CSV):
        with open(TRACKER_CSV) as f:
            for row in csv.DictReader(f):
                if row.get("downloaded") == "True":
                    downloaded_count = max(downloaded_count, 1)
        # Recount properly
        downloaded_count = 0
        with open(TRACKER_CSV) as f:
            for row in csv.DictReader(f):
                if row.get("downloaded") == "True":
                    downloaded_count += 1

    print(f"🚀 AEA Package Crawler", flush=True)
    print(f"   Target: {TARGET_COUNT} | Downloaded so far: {downloaded_count}", flush=True)
    print(f"   Already checked/skipped: {len(checked_ids)}", flush=True)
    print(f"   Delay: {MIN_DELAY}-{MAX_DELAY}s\n", flush=True)

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_URL)
        ctx = browser.contexts[0]
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.set_default_timeout(60000)

        for pg in range(1, MAX_PAGES + 1):
            if downloaded_count >= TARGET_COUNT:
                break

            print(f"\n{'='*60}", flush=True)
            print(f"📄 Search page {pg}", flush=True)

            try:
                projects = get_search_projects(page, pg)
            except Exception as e:
                print(f"  ❌ Search page failed: {e}", flush=True)
                polite_wait("after error")
                continue

            new = [(pid, u) for pid, u in projects if pid not in checked_ids]
            print(f"   {len(projects)} total, {len(new)} unchecked", flush=True)

            if not new:
                polite_wait("no new projects")
                continue

            for pid, url in new:
                if downloaded_count >= TARGET_COUNT:
                    break

                print(f"\n🔍 [{downloaded_count}/{TARGET_COUNT}] Project {pid}", flush=True)

                row = {
                    "project_id": pid,
                    "title": "",
                    "checked_at": datetime.now().isoformat(),
                    "has_readme": "",
                    "readme_type": "",
                    "data_public": "False",
                    "data_external_public": "False",
                    "downloaded": "False",
                    "download_size_mb": "",
                    "error": "",
                    "notes": "",
                }

                try:
                    page.goto(url, wait_until="networkidle", timeout=60000)
                    time.sleep(2)
                    # Session check
                    if not check_session_alive(page):
                        break
                    if f"/project/{pid}/" not in page.url:
                        page.goto(url, wait_until="networkidle", timeout=60000)
                        time.sleep(2)
                except Exception as e:
                    row["error"] = str(e)[:200]
                    append_csv_row(row)
                    checked_ids.add(pid)
                    print(f"  ❌ Page load failed: {e}", flush=True)
                    polite_wait("after error")
                    continue

                title = get_title(page)
                row["title"] = title
                print(f"   {title[:100]}", flush=True)

                # User requested focus on 2021/2022 papers only
                if not in_target_years(page):
                    row["notes"] = "out_of_target_years_2021_2022"
                    append_csv_row(row)
                    checked_ids.add(pid)
                    print(f"  ⏭️  Not a 2021/2022 paper — skip", flush=True)
                    polite_wait()
                    continue

                readme = find_readme_info(page, pid)
                if not readme:
                    row["has_readme"] = "False"
                    row["notes"] = "no_readme_found"
                    append_csv_row(row)
                    checked_ids.add(pid)
                    print(f"  ⏭️  No README — skip", flush=True)
                    polite_wait()
                    continue

                ftype, fname, fpath = readme
                row["has_readme"] = "True"
                row["readme_type"] = ftype
                print(f"  📄 {fname}", flush=True)

                content = fetch_readme_text(page, pid, fpath, ftype)
                if not content:
                    row["notes"] = "empty_readme"
                    append_csv_row(row)
                    checked_ids.add(pid)
                    print(f"  ⏭️  Empty README — skip", flush=True)
                    polite_wait()
                    continue

                classification = classify_data_availability(content)
                row["data_public"] = str(classification == "included")
                row["data_external_public"] = str(classification == "external")

                if classification == "restricted":
                    row["notes"] = "restricted_data"
                    append_csv_row(row)
                    checked_ids.add(pid)
                    print(f"  🔒 Restricted data — skip", flush=True)
                    polite_wait()
                    continue
                elif classification == "unknown":
                    row["notes"] = "no_public_data_statement"
                    append_csv_row(row)
                    checked_ids.add(pid)
                    print(f"  ⏭️  No public data statement — skip", flush=True)
                    polite_wait()
                    continue
                elif classification == "external":
                    # Data is freely available but needs separate download
                    # Still download the package (has code), tag it for later
                    print(f"  🌐 EXTERNAL PUBLIC DATA — downloading (code + instructions)...", flush=True)
                    polite_wait("before download")
                    size = download_project(page, pid)
                    if size > 0:
                        row["downloaded"] = "True"
                        row["download_size_mb"] = str(size)
                        row["notes"] = "external_data_needs_separate_download"
                        # Don't count toward the 100 target — these need extra work
                    else:
                        row["error"] = "download_failed"
                    append_csv_row(row)
                    checked_ids.add(pid)
                    polite_wait("before next")
                    continue

                # classification == "included" — all data in package!
                print(f"  ✅ ALL DATA INCLUDED — downloading...", flush=True)
                polite_wait("before download")

                size = download_project(page, pid)
                if size > 0:
                    row["downloaded"] = "True"
                    row["download_size_mb"] = str(size)
                    downloaded_count += 1
                    print(f"  📊 Progress: {downloaded_count}/{TARGET_COUNT}", flush=True)
                else:
                    row["error"] = "download_failed"

                append_csv_row(row)
                checked_ids.add(pid)

                # Update JSON tracker
                tracker["downloaded_count"] = downloaded_count
                tracker["checked_count"] = len(checked_ids)
                tracker["status"] = "running"
                save_json_tracker(tracker)

                polite_wait("before next")

        status = "completed" if downloaded_count >= TARGET_COUNT else f"stopped_at_{downloaded_count}"
        tracker["status"] = status
        tracker["downloaded_count"] = downloaded_count
        tracker["checked_count"] = len(checked_ids)
        save_json_tracker(tracker)

        print(f"\n{'='*60}", flush=True)
        print(f"🏁 Done! Status: {status}", flush=True)
        print(f"   Downloaded: {downloaded_count}/{TARGET_COUNT}", flush=True)
        print(f"   Total checked: {len(checked_ids)}", flush=True)
        print(f"   Log: {TRACKER_CSV}", flush=True)

        browser.close()


if __name__ == "__main__":
    run()
