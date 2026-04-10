"""Progress tracking for resumable crawling."""

import csv
import json
from datetime import datetime
from config import TRACKER_CSV, TRACKER_JSON, CSV_FIELDS


def load_checked_ids():
    """Load set of already-checked project IDs from CSV."""
    checked = set()
    if TRACKER_CSV.exists():
        with open(TRACKER_CSV) as f:
            for row in csv.DictReader(f):
                checked.add(row["project_id"])
    return checked


def count_downloads():
    """Count how many projects have been downloaded."""
    if not TRACKER_CSV.exists():
        return 0
    count = 0
    with open(TRACKER_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("downloaded") == "True":
                count += 1
    return count


def append_row(row_dict):
    """Append one row to the CSV log.

    Backfills the legacy data_public / data_external_public columns from the
    richer `classification` field for compatibility with older tooling.
    """
    classification = row_dict.get("classification", "")
    if classification == "included":
        row_dict.setdefault("data_public", "True")
        row_dict.setdefault("data_external_public", "False")
    elif classification == "external_public":
        row_dict.setdefault("data_public", "False")
        row_dict.setdefault("data_external_public", "True")
    elif classification == "restricted":
        row_dict.setdefault("data_public", "False")
        row_dict.setdefault("data_external_public", "False")

    exists = TRACKER_CSV.exists()
    with open(TRACKER_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)


def make_row(project_id):
    """Create a blank row dict with defaults."""
    return {
        "project_id": project_id,
        "title": "",
        "authors": "",
        "citation": "",
        "icpsr_doi": "",
        "checked_at": datetime.now().isoformat(),
        "has_readme": "",
        "readme_type": "",
        "classification": "",
        "rationale": "",
        "data_public": "",
        "data_external_public": "",
        "downloaded": "False",
        "download_size_mb": "",
        "error": "",
        "notes": "",
    }


def load_state():
    """Load JSON state (last page, counts)."""
    try:
        with open(TRACKER_JSON) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"status": "starting", "last_page": 0, "downloaded_count": 0, "checked_count": 0}


def save_state(state):
    """Save JSON state."""
    with open(TRACKER_JSON, "w") as f:
        json.dump(state, f, indent=2)
