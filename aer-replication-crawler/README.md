# aer-replication-crawler

Pipeline for collecting AER (American Economic Review) replication packages and
their associated paper PDFs from openICPSR. Drives a real Chrome browser to
avoid bot detection, uses Claude to classify which packages have all data
available, and falls back through several open-access sources to find free PDFs.

This is the data-collection layer of the parent
[replication-studies](..) repo. The actual replication / robustness analysis
lives one directory up.

## What the pipeline produces

| Output | Where it lives | What's in it |
|---|---|---|
| `project_log.csv` | parent repo root | One row per project ICPSR returned, with title, authors, citation, DOI, classification, download status |
| ZIP packages | `PACKAGE_ARCHIVE` (external drive) | Replication code + data, ~50 MB – 5 GB each |
| Paper PDFs | `PDF_ARCHIVE` (external drive) | Open-access versions of the papers themselves |
| `data/icpsr_publications.json` | repo, committed | Scraped "Related Publications" section per project (citations + DOI links) |
| `data/paper_pdfs_index.json` | repo, committed | Per-DOI PDF candidates from OpenAlex + Unpaywall |

## How it works (4 phases)

### Phase 1 — Crawl + classify (`main.py`)
1. Connects to a real Chrome instance over CDP (so Cloudflare doesn't block us)
2. Walks the openICPSR AEA studies search page-by-page
3. For each project: fetches the README, asks Claude whether all data is included
4. If `included` → downloads the ZIP
5. Logs every checked project to `project_log.csv` so the run is fully resumable

The crawler also pulls the project's authors, full citation, and ICPSR DOI
straight from the project page.

### Phase 2 — Scrape Related Publications (`scrape_pubs.py`)
For each downloaded project, visit the ICPSR page again and pull the "Related
Publications" section. About 88% of projects have this populated, and almost
all of those include a direct DOI link to the AER paper. Saves to
`data/icpsr_publications.json`.

### Phase 3 — Find OA PDFs (`find_pdfs.py`)
For every paper DOI, query [OpenAlex](https://openalex.org/) and
[Unpaywall](https://unpaywall.org/) to enumerate open-access PDF candidates.
Many papers exist as NBER working papers, arXiv preprints, or author-deposited
copies even when the published version is paywalled. Saves to
`data/paper_pdfs_index.json`.

### Phase 4 — Download PDFs (`download_pdfs.py`, `scholar_pdfs.py`)
- `download_pdfs.py` — fetches the candidate URLs found in Phase 3 directly via
  `urllib`. Handles ~75% of downloads.
- `scholar_pdfs.py` — last-resort fallback: uses real Chrome via Playwright to
  search Google Scholar, click `[PDF]` links, and capture downloads (handles
  SSRN and other JS-required sites).

## Setup

```bash
cd aer-replication-crawler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env  # then edit .env with your keys + paths
```

You'll need:
- An [Anthropic API key](https://console.anthropic.com/) for README classification
- An openICPSR account (the crawler will prompt for manual login on first run)
- Chrome already running with remote debugging:
  ```bash
  /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
      --remote-debugging-port=9222 \
      --no-first-run \
      --user-data-dir="$HOME/chrome-debug-profile"
  ```

## Running

```bash
# Phase 1: crawl + download (slow, hours/days)
python main.py

# Phase 2: scrape related publications
python scrape_pubs.py

# Phase 3: find OA PDF candidates
python find_pdfs.py

# Phase 4: download PDFs
python download_pdfs.py
python scholar_pdfs.py  # only for stragglers, slow but uses real Chrome

# After a session, move staged downloads to the package archive
./move_to_external.sh
```

All phases are resumable — re-running them only processes things that haven't
been done yet.

## Helper scripts

- `recheck.py` — re-process projects that errored out or were missed (CAPTCHAs,
  session expiry)
- `retry_failed.py` — retry just the projects classified as `included` /
  `external_public` that failed to download
- `redownload.py` — re-fetch projects that show as downloaded in the CSV but
  are missing from disk
- `enrich.py` — earlier OpenAlex enrichment attempt (kept for reference;
  Phase 2 + 3 do this more reliably via DOI)

## Coverage we got

From a single multi-day run targeting all years on openICPSR:

| | Count |
|---|---|
| Projects checked | ~1,068 |
| Replication packages downloaded | 261 |
| Related publications scraped | 246 |
| AER DOIs identified | 240 |
| PDFs successfully retrieved | 63 (24% of downloaded projects) |

The PDF coverage is the weak link — AER articles are paywalled and only ~30%
of recent papers have a freely available NBER/arXiv/SSRN version that
OpenAlex+Unpaywall+Scholar can find. Older papers fare much better.

## Known issues / lessons learned

- **Cloudflare blocks Playwright-launched browsers.** You must connect to a
  real Chrome via CDP, not `chromium.launch()`.
- **Playwright via CDP redirects Chrome's downloads** to its temp artifact
  directories (`/var/folders/.../playwright-artifacts-*/`), not `~/Downloads`.
  The downloader watches both locations.
- **`os.rename` fails across devices** (e.g. local disk → external drive).
  Always use `shutil.move` for cross-device file moves.
- **AEA returns HTML at `aeaweb.org/articles/pdf/doi/...`** unless you have
  institutional access. Skip those URLs early.
- **SSRN blocks direct downloads from `Delivery.cfm` URLs.** You have to go
  through the abstract page first and click "Download This Paper" — handled
  in `scholar_pdfs.py`.
