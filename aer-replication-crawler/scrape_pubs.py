"""Scrape 'Related Publications' from ICPSR project pages.

For each project in project_log.csv, visit the ICPSR page and extract any
linked publication info (title, authors, journal, DOI/URL). Writes to
icpsr_publications.json (keyed by project_id) so we can resume.
"""

import csv
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

import config

BASE_DIR = config.BASE_DIR
TRACKER_CSV = config.TRACKER_CSV
OUTPUT_JSON = config.ICPSR_PUBS_JSON
CDP_URL = config.CDP_URL


def scrape_one(page, project_id):
    """Extract related publications from one ICPSR project page."""
    url = f"https://www.openicpsr.org/openicpsr/project/{project_id}/version/V1/view"
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(1)
    except Exception as e:
        return {"error": str(e)[:100]}

    return page.evaluate("""() => {
        const text = document.body.innerText;
        const result = {related_pubs: [], related_links: []};

        // Find "Related Publications" section
        const idx = text.indexOf('Related Publications');
        if (idx < 0) return result;

        // Extract section (up to next major section or download button)
        let section = text.substring(idx, idx + 1500);
        const endIdx = section.search(/\\s+DOWNLOAD THIS PROJECT|Usage Metrics/);
        if (endIdx > 0) section = section.substring(0, endIdx);

        if (section.includes('No related publications')) return result;

        // Remove header line
        section = section.replace(/Related Publications\\s*\\n?/, '')
                         .replace(/The following publications[^\\n]*\\n/, '')
                         .trim();

        // Split into individual citations (by blank line or period+newline)
        const citations = section.split(/\\n\\s*\\n/).map(s => s.trim()).filter(s => s.length > 20);
        result.related_pubs = citations;

        // Extract URLs embedded in citation text
        const urlRegex = /https?:\\/\\/[^\\s"'<>)]+/g;
        const allUrls = [];
        for (const cite of citations) {
            const matches = cite.match(urlRegex) || [];
            allUrls.push(...matches);
        }
        result.related_links = allUrls;

        return result;
    }""")


def load_cache():
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(cache, f, indent=2)


def run(limit=None, only_downloaded=True):
    """Scrape related pubs. Set only_downloaded=True to focus on projects we actually have."""
    with open(TRACKER_CSV) as f:
        rows = list(csv.DictReader(f))

    if only_downloaded:
        rows = [r for r in rows if r["downloaded"] == "True"]

    cache = load_cache()

    # Skip already-cached
    to_scrape = [r for r in rows if r["project_id"] not in cache]
    if limit:
        to_scrape = to_scrape[:limit]

    print(f"Scraping {len(to_scrape)} projects ({len(cache)} cached)", flush=True)

    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(CDP_URL)
        page = browser.contexts[0].pages[0]

        for i, row in enumerate(to_scrape):
            pid = row["project_id"]
            print(f"[{i+1}/{len(to_scrape)}] {pid}: {row['title'][:60]}", flush=True)

            result = scrape_one(page, pid)
            cache[pid] = result

            if result.get("related_pubs"):
                print(f"    {len(result['related_pubs'])} pub(s), {len(result.get('related_links', []))} link(s)", flush=True)
                for link in result.get("related_links", [])[:3]:
                    print(f"    -> {link[:80]}", flush=True)
            elif result.get("error"):
                print(f"    ERROR: {result['error']}", flush=True)
            else:
                print(f"    no related pubs", flush=True)

            time.sleep(1.5)  # polite

            if (i + 1) % 25 == 0:
                save_cache(cache)

    save_cache(cache)
    print(f"\nDone. Saved to {OUTPUT_JSON}", flush=True)

    # Summary
    with_pubs = sum(1 for k, v in cache.items() if v.get("related_pubs"))
    with_links = sum(1 for k, v in cache.items() if v.get("related_links"))
    print(f"  With publications: {with_pubs}", flush=True)
    print(f"  With direct links: {with_links}", flush=True)


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(limit=limit)
