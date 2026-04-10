"""Enrich project_log.csv with metadata from OpenAlex API.

Searches for each paper by cleaned title, pulls authors, DOI, citation count,
OA status, PDF links, journal info, and more. Writes enriched data to
project_log_enriched.csv.

Rate limited to ~1 request/second to be polite.
"""

import csv
import json
import os
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path
from difflib import SequenceMatcher

import config

BASE_DIR = config.BASE_DIR
INPUT_CSV = config.TRACKER_CSV
OUTPUT_CSV = BASE_DIR / "project_log_enriched.csv"
CACHE_FILE = config.OPENALEX_CACHE

MAILTO = config.OPENALEX_EMAIL or "noreply@example.com"  # OpenAlex polite pool
DELAY = 1.0  # seconds between requests

# AER source IDs in OpenAlex
AER_SOURCE_IDS = {
    "https://openalex.org/S23254222",       # American Economic Review
    "https://openalex.org/S4210174288",     # AER: Insights
}

# Output CSV fields
ENRICHED_FIELDS = [
    # Original fields
    "project_id", "title", "checked_at", "has_readme", "readme_type",
    "classification", "rationale", "downloaded", "download_size_mb",
    "error", "notes",
    # New fields from OpenAlex
    "paper_title", "authors", "doi", "openalex_id",
    "publication_date", "journal", "volume", "issue",
    "first_page", "last_page", "cited_by_count",
    "is_oa", "oa_status", "oa_url", "pdf_url",
    "abstract_snippet", "match_confidence",
]


def clean_title(raw_title):
    """Strip 'Data and Code for:' prefix and quotes from ICPSR title."""
    t = raw_title.strip()
    for prefix in [
        "Data and Code for: ",
        "Data and code for: ",
        "Data and Code for ",
        "Data and code for ",
        "Code for: ",
        "Code for ",
        "Data for: ",
        "Data for ",
    ]:
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):]
            break
    # Strip surrounding quotes
    t = t.strip('"').strip('"').strip('"')
    return t.strip()


def title_similarity(a, b):
    """Score how similar two titles are (0-1)."""
    a = re.sub(r'[^\w\s]', '', a.lower())
    b = re.sub(r'[^\w\s]', '', b.lower())
    return SequenceMatcher(None, a, b).ratio()


def search_openalex(title):
    """Search OpenAlex for a paper by title. Returns best match or None."""
    encoded = urllib.parse.quote(title)
    url = (
        f"https://api.openalex.org/works?search={encoded}"
        f"&filter=type:article"
        f"&per_page=5"
        f"&mailto={MAILTO}"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clawbot/1.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
    except Exception as e:
        print(f"    API error: {e}", flush=True)
        return None

    if not data.get("results"):
        return None

    # Score each result by title similarity, prefer AER
    best = None
    best_score = 0

    for w in data["results"]:
        w_title = w.get("title", "")
        sim = title_similarity(title, w_title)

        # Boost AER papers
        source_id = ""
        loc = w.get("primary_location") or {}
        source = loc.get("source") or {}
        source_id = source.get("id", "")
        if source_id in AER_SOURCE_IDS:
            sim += 0.15

        if sim > best_score:
            best_score = sim
            best = w

    if best_score < 0.55:
        return None

    return best, round(best_score, 3)


def extract_fields(work, confidence):
    """Extract relevant fields from an OpenAlex work object."""
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    biblio = work.get("biblio") or {}
    oa = work.get("open_access") or {}

    authors = []
    for a in work.get("authorships", []):
        name = a.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)

    # Try to find PDF from any location
    pdf_url = loc.get("pdf_url")
    if not pdf_url:
        for alt in work.get("locations", []):
            if alt.get("pdf_url"):
                pdf_url = alt["pdf_url"]
                break

    # Get abstract if available (OpenAlex returns inverted index)
    abstract = ""
    inv_abstract = work.get("abstract_inverted_index")
    if inv_abstract:
        # Reconstruct first ~200 chars
        word_positions = []
        for word, positions in inv_abstract.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        abstract = " ".join(w for _, w in word_positions[:40])
        if len(word_positions) > 40:
            abstract += "..."

    return {
        "paper_title": work.get("title", ""),
        "authors": "; ".join(authors),
        "doi": work.get("doi", ""),
        "openalex_id": work.get("id", ""),
        "publication_date": work.get("publication_date", ""),
        "journal": source.get("display_name", ""),
        "volume": biblio.get("volume", ""),
        "issue": biblio.get("issue", ""),
        "first_page": biblio.get("first_page", ""),
        "last_page": biblio.get("last_page", ""),
        "cited_by_count": work.get("cited_by_count", ""),
        "is_oa": oa.get("is_oa", ""),
        "oa_status": oa.get("oa_status", ""),
        "oa_url": oa.get("oa_url", ""),
        "pdf_url": pdf_url or "",
        "abstract_snippet": abstract,
        "match_confidence": confidence,
    }


def load_cache():
    """Load cached results to avoid re-fetching."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Save cache to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def run():
    # Load existing data
    with open(INPUT_CSV) as f:
        rows = list(csv.DictReader(f))

    cache = load_cache()
    enriched = []
    found = 0
    not_found = 0

    print(f"Enriching {len(rows)} projects via OpenAlex API", flush=True)
    print(f"Rate limit: {DELAY}s between requests\n", flush=True)

    for i, row in enumerate(rows):
        pid = row["project_id"]
        raw_title = row["title"]
        clean = clean_title(raw_title)

        # Start with original fields
        out = {field: row.get(field, "") for field in ENRICHED_FIELDS if field in row}

        if pid in cache:
            # Use cached result
            out.update(cache[pid])
            if cache[pid].get("paper_title"):
                found += 1
            else:
                not_found += 1
        elif not clean or len(clean) < 10:
            print(f"[{i+1}/{len(rows)}] {pid}: title too short, skipping", flush=True)
            not_found += 1
            cache[pid] = {f: "" for f in ENRICHED_FIELDS if f not in row}
        else:
            print(f"[{i+1}/{len(rows)}] {pid}: {clean[:70]}", flush=True)

            result = search_openalex(clean)
            if result:
                work, confidence = result
                fields = extract_fields(work, confidence)
                out.update(fields)
                cache[pid] = fields
                found += 1
                print(f"    -> {fields['paper_title'][:60]} (conf={confidence})", flush=True)
                if fields["pdf_url"]:
                    print(f"    PDF: {fields['pdf_url'][:80]}", flush=True)
            else:
                not_found += 1
                cache[pid] = {f: "" for f in ENRICHED_FIELDS if f not in row}
                print(f"    -> no match", flush=True)

            time.sleep(DELAY)

            # Save cache every 50 requests
            if (i + 1) % 50 == 0:
                save_cache(cache)
                print(f"\n  Progress: {found} found, {not_found} not found\n", flush=True)

        enriched.append(out)

    # Write enriched CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ENRICHED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)

    save_cache(cache)

    print(f"\nDone!", flush=True)
    print(f"  Found: {found}/{len(rows)}", flush=True)
    print(f"  Not found: {not_found}/{len(rows)}", flush=True)
    print(f"  Output: {OUTPUT_CSV}", flush=True)

    # Summary of OA PDFs
    pdf_count = sum(1 for r in enriched if r.get("pdf_url"))
    oa_count = sum(1 for r in enriched if r.get("is_oa") == True or r.get("is_oa") == "True")
    print(f"  Open access: {oa_count}", flush=True)
    print(f"  PDFs available: {pdf_count}", flush=True)


if __name__ == "__main__":
    run()
