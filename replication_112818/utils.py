"""Shared paths and helpers for replication of paper 112818."""
import os

PAPER_ID = "112818"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "112818-V1", "P2014_1109_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "figure1.dta")
