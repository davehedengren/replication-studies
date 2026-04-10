"""aer-replication-crawler configuration — all settings in one place.

Paths default to sensible locations but can be overridden via environment
variables in .env (see .env.example).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# --- API credentials (from .env) ---
EMAIL_LOGIN = os.getenv("EMAIL_LOGIN", "")
PASSWORD = os.getenv("PASSWORD", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")  # for polite OpenAlex pool

# --- Paths ---
# Tracker CSV lives at the parent repo root by default so it joins with the
# replication-studies project log. Override with TRACKER_CSV env var if running
# the crawler standalone.
TRACKER_CSV = Path(os.getenv("TRACKER_CSV", BASE_DIR.parent / "project_log.csv"))

# Downloaded replication ZIPs land in DOWNLOAD_DIR (staging) and are then moved
# to PACKAGE_ARCHIVE (typically an external drive) by move_to_external.sh.
DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", BASE_DIR / "download_staging"))
PACKAGE_ARCHIVE = Path(os.getenv("PACKAGE_ARCHIVE", BASE_DIR / "downloads"))

# Paper PDFs from enrichment phase
PDF_ARCHIVE = Path(os.getenv("PDF_ARCHIVE", BASE_DIR / "pdfs"))

# Cache + state files (gitignored)
SESSION_FILE = BASE_DIR / "session.json"
TRACKER_JSON = BASE_DIR / "download_tracker.json"
ICPSR_PUBS_JSON = BASE_DIR / "data" / "icpsr_publications.json"
PAPER_PDFS_INDEX = BASE_DIR / "data" / "paper_pdfs_index.json"
OPENALEX_CACHE = BASE_DIR / "openalex_cache.json"

# --- Search ---
SEARCH_URL = "https://www.openicpsr.org/openicpsr/search/aea/studies"
SEARCH_SORT = "DATEUPDATED%20desc"
ROWS_PER_PAGE = 25
MAX_PAGES = 500

# --- Targets ---
TARGET_COUNT = int(os.getenv("TARGET_COUNT", "500"))
TARGET_YEARS = None  # Set to e.g. {"2021", "2022"} to filter, or None for all

# --- Politeness ---
MIN_DELAY = int(os.getenv("MIN_DELAY", "30"))
MAX_DELAY = int(os.getenv("MAX_DELAY", "60"))

# --- Browser ---
CDP_URL = os.getenv("CDP_URL", "http://127.0.0.1:9222")

# --- LLM ---
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# --- CSV fields ---
# Schema is the union of clawbot's richer fields plus the older
# data_public / data_external_public columns from the original crawler, so the
# tracker CSV is backwards-compatible with existing replication-studies tooling.
CSV_FIELDS = [
    "project_id", "title", "authors", "citation", "icpsr_doi",
    "checked_at", "has_readme", "readme_type",
    "classification", "rationale",
    "data_public", "data_external_public",
    "downloaded", "download_size_mb",
    "error", "notes",
]
