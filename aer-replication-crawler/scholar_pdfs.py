"""Find and download PDFs via Google Scholar using real Chrome.

Uses Playwright CDP connection to search Google Scholar for each paper title,
find [PDF] links, and download them. Very slow and polite to avoid blocks.
"""

import csv
import json
import os
import re
import time
import random
import urllib.request
import urllib.error
from pathlib import Path
from playwright.sync_api import sync_playwright

import config

BASE_DIR = config.BASE_DIR
ICPSR_PUBS = config.ICPSR_PUBS_JSON
PDF_DIR = config.PDF_ARCHIVE
STATUS_JSON = BASE_DIR / "scholar_pdf_status.json"
CDP_URL = config.CDP_URL

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 Chrome/120.0.0.0",
    "Accept": "application/pdf,*/*",
}


def load_status():
    if STATUS_JSON.exists():
        with open(STATUS_JSON) as f:
            return json.load(f)
    return {}


def save_status(status):
    with open(STATUS_JSON, "w") as f:
        json.dump(status, f, indent=2)


def clean_title(raw):
    """Clean ICPSR title for search."""
    for prefix in [
        "Data and Code for: ", "Data and code for: ",
        "Data and Code for ", "Data and code for ",
        "Code for: ", "Code for ",
        "Data for: ", "Replication data for: ",
        "Replication data for ",
    ]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):]
            break
    return raw.strip().strip('"').strip('"').strip('"')


def get_papers_needing_pdfs():
    """Get list of (project_id, clean_title) for papers we still need."""
    have = set()
    if PDF_DIR.exists():
        have = set(f.replace(".pdf", "") for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))

    with open(ICPSR_PUBS) as f:
        pubs = json.load(f)

    # Use the publication citation for cleaner titles
    papers = []
    for pid, info in pubs.items():
        if pid in have:
            continue
        citations = info.get("related_pubs", [])
        if not citations:
            continue

        # Extract paper title from citation (usually in quotes)
        cite = citations[0]
        m = re.search(r'"([^"]+)"', cite)
        if m:
            title = m.group(1)
        else:
            # Fall back to ICPSR title cleaning
            title = cite.split(".")[1].strip() if "." in cite else cite

        if len(title) < 10:
            continue
        papers.append((pid, title))

    return papers


def search_scholar(page, title):
    """Search Google Scholar and return list of {title, url, pdf_url}."""
    query = f'"{title}"'
    search_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(query)}"

    try:
        page.goto(search_url, wait_until="networkidle", timeout=30000)
        time.sleep(2)

        # Check for CAPTCHA
        text = page.inner_text("body")
        if "unusual traffic" in text.lower() or "captcha" in text.lower():
            print("    CAPTCHA detected — pausing 5 min", flush=True)
            time.sleep(300)
            return "captcha"

        # Extract results
        results = page.evaluate("""() => {
            const items = document.querySelectorAll('.gs_r.gs_or');
            return Array.from(items).slice(0, 5).map(item => {
                const titleEl = item.querySelector('.gs_rt a');
                const pdfLink = item.querySelector('.gs_or_ggsm a, .gs_ggsd a');
                return {
                    title: titleEl ? titleEl.textContent : '',
                    url: titleEl ? titleEl.href : '',
                    pdf_url: pdfLink ? pdfLink.href : '',
                    pdf_text: pdfLink ? pdfLink.textContent : '',
                };
            });
        }""")

        return results or []

    except Exception as e:
        print(f"    Scholar error: {e}", flush=True)
        return []


def download_pdf(url, dest):
    """Download PDF via urllib, return (success, size)."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if data.startswith(b"%PDF") and len(data) > 10000:
            with open(dest, "wb") as f:
                f.write(data)
            return True, len(data)
        return False, 0
    except Exception:
        return False, 0


def download_pdf_via_chrome(ctx, url, dest):
    """Download PDF using Chrome for sites that need browser sessions (SSRN etc)."""
    import glob, shutil
    TMPDIR = os.environ.get("TMPDIR", "/tmp")

    page = ctx.new_page()

    # Snapshot before
    before = set()
    for d in glob.glob(os.path.join(TMPDIR, "playwright-artifacts-*")):
        before.update(glob.glob(os.path.join(d, "*")))
    before.update(glob.glob(os.path.join(os.path.expanduser("~/Downloads"), "*")))

    try:
        # For SSRN: go to abstract page first, then click download
        if "ssrn.com" in url:
            # Extract abstract ID from URL
            m = re.search(r"abstractid=(\d+)", url)
            if m:
                abstract_id = m.group(1)
                abstract_url = f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={abstract_id}"
                page.goto(abstract_url, wait_until="networkidle", timeout=30000)
                time.sleep(2)
                page.click("text=Download This Paper")
                time.sleep(5)
            else:
                page.goto(url, wait_until="networkidle", timeout=30000)
                time.sleep(3)
        else:
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(3)
            # Click any download/PDF button
            try:
                page.click("text=Download", timeout=5000)
                time.sleep(3)
            except Exception:
                pass

        # Wait for download to appear
        for _ in range(12):
            time.sleep(5)
            after = set()
            for d in glob.glob(os.path.join(TMPDIR, "playwright-artifacts-*")):
                after.update(glob.glob(os.path.join(d, "*")))
            after.update(glob.glob(os.path.join(os.path.expanduser("~/Downloads"), "*")))
            new = after - before
            new = {f for f in new if os.path.isfile(f) and os.path.getsize(f) > 10000}
            if new:
                src = max(new, key=os.path.getsize)
                with open(src, "rb") as f:
                    head = f.read(5)
                if head == b"%PDF-":
                    shutil.move(src, str(dest))
                    page.close()
                    return True, os.path.getsize(str(dest))
                page.close()
                return False, 0
        page.close()
        return False, 0
    except Exception as e:
        print(f"    Chrome download error: {e}", flush=True)
        try:
            page.close()
        except Exception:
            pass
        return False, 0


def run(limit=None):
    papers = get_papers_needing_pdfs()
    status = load_status()

    # Skip already attempted
    papers = [(pid, t) for pid, t in papers if pid not in status]
    if limit:
        papers = papers[:limit]

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Searching Google Scholar for {len(papers)} papers\n", flush=True)

    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(CDP_URL)
        ctx = browser.contexts[0]
        page = ctx.new_page()

        ok = 0
        fail = 0
        captcha_count = 0

        for i, (pid, title) in enumerate(papers):
            print(f"[{i+1}/{len(papers)}] {pid}: {title[:60]}", flush=True)

            results = search_scholar(page, title)

            if results == "captcha":
                captcha_count += 1
                if captcha_count >= 3:
                    print("Too many CAPTCHAs, stopping.", flush=True)
                    break
                continue

            # Look for PDF link in results
            found = False
            for r in results:
                pdf_url = r.get("pdf_url", "")
                if not pdf_url:
                    continue

                # Skip aeaweb (paywall)
                if "aeaweb.org" in pdf_url:
                    continue

                print(f"    PDF: {pdf_url[:80]}", flush=True)
                dest = PDF_DIR / f"{pid}.pdf"

                # Try direct download first
                success, size = download_pdf(pdf_url, dest)
                if success:
                    print(f"    OK {size // 1024} KB", flush=True)
                    status[pid] = {"success": True, "url": pdf_url, "size": size, "source": "scholar"}
                    ok += 1
                    found = True
                    break

                # If direct fails, try via Chrome (handles SSRN, etc)
                print(f"    direct failed, trying via Chrome...", flush=True)
                success, size = download_pdf_via_chrome(ctx, pdf_url, dest)
                if success:
                    print(f"    OK via Chrome {size // 1024} KB", flush=True)
                    status[pid] = {"success": True, "url": pdf_url, "size": size, "source": "scholar_chrome"}
                    ok += 1
                    found = True
                    break
                else:
                    print(f"    Chrome download also failed", flush=True)

            if not found:
                status[pid] = {"success": False, "results": len(results)}
                fail += 1
                if results:
                    print(f"    {len(results)} results, no downloadable PDF", flush=True)
                else:
                    print(f"    no results", flush=True)

            if (i + 1) % 10 == 0:
                save_status(status)

            # Random delay to be polite (15-45s)
            delay = random.randint(15, 45)
            print(f"    {delay}s pause", flush=True)
            time.sleep(delay)

        page.close()

    save_status(status)
    print(f"\nDone: {ok} downloaded, {fail} failed, {captcha_count} CAPTCHAs", flush=True)


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(limit=limit)
