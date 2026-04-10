"""Find OA PDFs for papers using DOIs from ICPSR.

For each downloaded project with an AER DOI, query OpenAlex and Unpaywall
to find open-access PDF URLs. Saves results to paper_pdfs_index.json.
"""

import csv
import json
import re
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

import config

BASE_DIR = config.BASE_DIR
TRACKER_CSV = config.TRACKER_CSV
ICPSR_PUBS = config.ICPSR_PUBS_JSON
OUTPUT_JSON = config.PAPER_PDFS_INDEX

EMAIL = config.OPENALEX_EMAIL or "noreply@example.com"
DELAY = 0.5

DOI_RE = re.compile(r"doi\.org/(10\.\d{4,}/[^\s,;)]+)")


def extract_doi(text):
    """Extract DOI from a URL or citation text."""
    m = DOI_RE.search(text)
    if m:
        return m.group(1).rstrip(".,;)")
    return None


def get_dois_from_icpsr():
    """Map project_id -> DOI using ICPSR related publications."""
    with open(ICPSR_PUBS) as f:
        data = json.load(f)

    pid_to_doi = {}
    for pid, info in data.items():
        for link in info.get("related_links", []):
            doi = extract_doi(link)
            if doi and "10.1257" in doi:  # prefer AEA DOIs
                pid_to_doi[pid] = doi
                break
        if pid not in pid_to_doi:
            # fall back to any DOI
            for link in info.get("related_links", []):
                doi = extract_doi(link)
                if doi:
                    pid_to_doi[pid] = doi
                    break
    return pid_to_doi


def query_openalex(doi):
    """Returns dict with pdf_url, oa_status, title, authors, locations."""
    url = f"https://api.openalex.org/works/doi:{doi}?mailto={EMAIL}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clawbot/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            w = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": "not_found"}
        return {"error": f"http_{e.code}"}
    except Exception as e:
        return {"error": str(e)[:80]}

    oa = w.get("open_access") or {}

    # Collect all PDF URLs from all locations
    pdf_urls = []
    for loc in w.get("locations", []):
        pdf = loc.get("pdf_url")
        if pdf:
            src_name = ""
            if loc.get("source"):
                src_name = loc["source"].get("display_name", "")
            pdf_urls.append({
                "url": pdf,
                "source": src_name,
                "version": loc.get("version"),
                "is_oa": loc.get("is_oa"),
            })

    return {
        "title": w.get("title"),
        "authors": [a.get("author", {}).get("display_name") for a in w.get("authorships", [])],
        "oa_status": oa.get("oa_status"),
        "is_oa": oa.get("is_oa"),
        "pdf_urls": pdf_urls,
        "primary_location_url": (w.get("primary_location") or {}).get("landing_page_url"),
    }


def query_unpaywall(doi):
    """Returns dict with oa_locations, best PDF URL."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clawbot/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            d = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}"}
    except Exception as e:
        return {"error": str(e)[:80]}

    pdf_urls = []
    for loc in d.get("oa_locations", []):
        pdf = loc.get("url_for_pdf")
        if pdf:
            pdf_urls.append({
                "url": pdf,
                "source": loc.get("repository_institution") or loc.get("host_type"),
                "version": loc.get("version"),
                "host_type": loc.get("host_type"),
            })
    return {
        "is_oa": d.get("is_oa"),
        "oa_status": d.get("oa_status"),
        "title": d.get("title"),
        "pdf_urls": pdf_urls,
    }


def run():
    pid_to_doi = get_dois_from_icpsr()
    print(f"Have DOIs for {len(pid_to_doi)} projects\n", flush=True)

    # Load existing results
    results = {}
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON) as f:
            results = json.load(f)

    pids = [pid for pid in pid_to_doi if pid not in results]
    print(f"Querying {len(pids)} new projects ({len(results)} cached)\n", flush=True)

    for i, pid in enumerate(pids):
        doi = pid_to_doi[pid]
        print(f"[{i+1}/{len(pids)}] {pid} (DOI: {doi})", flush=True)

        oa = query_openalex(doi)
        time.sleep(DELAY)
        up = query_unpaywall(doi)

        # Combine PDFs from both sources
        all_pdfs = []
        seen = set()
        for src in ["openalex", "unpaywall"]:
            data = oa if src == "openalex" else up
            for p in data.get("pdf_urls", []):
                url = p["url"]
                if url not in seen:
                    seen.add(url)
                    p["_from"] = src
                    all_pdfs.append(p)

        results[pid] = {
            "doi": doi,
            "title": oa.get("title") or up.get("title"),
            "openalex": oa,
            "unpaywall": up,
            "all_pdfs": all_pdfs,
        }

        n_pdfs = len(all_pdfs)
        status = oa.get("oa_status") or up.get("oa_status") or "?"
        print(f"    {n_pdfs} PDF(s), oa_status={status}", flush=True)
        if all_pdfs:
            print(f"    -> {all_pdfs[0]['url'][:80]}", flush=True)

        if (i + 1) % 20 == 0:
            with open(OUTPUT_JSON, "w") as f:
                json.dump(results, f, indent=2)

        time.sleep(DELAY)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    with_pdf = sum(1 for v in results.values() if v.get("all_pdfs"))
    print(f"\nDone: {with_pdf}/{len(results)} have PDF URLs", flush=True)


if __name__ == "__main__":
    run()
