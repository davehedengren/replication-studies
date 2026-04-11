# Replication Study: 148001-V1 — STALLED, NEEDS RECOVERY

> **STATUS: PARTIAL / STALLED.** The headless `claude -p` session for this
> paper completed Phases 1–4 (orientation, translation, data audit,
> robustness) and identified a bug in the published code, but hung
> indefinitely before writing the full structured writeup. The stuck
> subprocess was killed at 21:31 on 2026-04-10 after ~3h 21min of
> zero-event silence (190 minutes since the last log line at 18:19:20).
> This stub is a placeholder that lets the driver advance to the next
> paper; the work on disk is real and a focused recovery run should
> finish the writeup using the existing scripts as input.
>
> Listed in `replication_driver/needs_recovery.txt` for follow-up.

---

## 0. TLDR (preserved from the partial session)

- **Replication status (partial):** Sample matches (N = 8,446, mean dep var
  = 0.482). Table 2 main result replicates: **−0.2025** (this replication)
  vs **−0.210** (paper). Tables 3, 4, 5 were attempted and produced
  output CSVs. The full structured writeup (results table, robustness
  table, file manifest) was not written before the stall.
- **Bug status (preliminary):** **Bug found.** The session text reads:
  *"Table 3 bench columns use only case controls — that's why my balance
  test differed. Found a Table 5 Panel B labeling bug. Moving on to data
  audit + robustness."* The exact line and impact need verification in
  the recovery run.
- **Why stalled:** Unknown root cause. The last successful tool call was
  `Read replication_226781/writeup_226781.md` (the gold-standard format
  reference), and the next expected action was `Write writeup_148001.md`.
  The process never emitted another stream event. Claude state was
  interruptible sleep at 0.1% CPU, suggesting a hung network read inside
  an API call rather than an infinite loop.

---

## 1. Paper context (best-effort from on-disk artifacts)

From `02_table2.py` docstring:

| Spec | Coefficient | SE | Notes |
|---|---|---|---|
| col 1 | Retirements_2010 × Post2010 = −0.237 | 0.0342 | no controls |
| col 2 | Retirements_2010 × Post2010 = −0.210 | 0.0383 | all controls (baseline) |
| col 3 | Retirements_2009 × Post2010 = +0.0677 | 0.0589 | placebo |
| col 4 | Retirements_2008 × Post2010 = +0.143 | 0.112 | placebo |
| col 5 | Retirements_2007 × Post2010 = +0.115 | 0.0968 | placebo |

`N = 8,446`, `mean dep var = 0.482`. The shape of the design (DiD on
"Retirements" × Post2010 with placebo years) suggests a paper about
judicial-selection reform and case outcomes ("State Wins" likely
referring to government wins in litigation).

The full paper title and authors were not captured in the partial
session — recovery should pull them from `148001-V1/README.pdf` or
`/Volumes/Extreme SSD/AER_replication_data_pdfs/148001.pdf`.

---

## 2. What's on disk

```
replication_148001/
  utils.py
  01_clean.py            # produces analytic sample
  02_table2.py           # main result, runs cleanly
  03_tables345.py        # Tables 3, 4, 5 attempt
  04_data_audit.py       # ran cleanly
  05_robustness.py       # ran cleanly
  output/
    table2.csv           # produced
    robustness.csv       # produced
```

The driver-extracted package at `148001-V1/` is intact; source zip
is at `/Volumes/Extreme SSD/AER_replication_data/148001.zip`.

---

## 3. Recovery instructions

1. Re-extract is not needed — `148001-V1/` is already on disk and the
   shared venv has all packages it needs.
2. Run a focused recovery `claude -p` session with a prompt that says
   roughly: *"Paper 148001's prior session stalled before writing the
   writeup. The five scripts and output CSVs in `replication_148001/`
   are valid and the main coefficient (−0.2025) replicates the paper's
   −0.210. There is also a Table 5 Panel B labeling bug noted in the
   prior session — please verify it in `148001-V1/REPLICATION/Code/`
   and document it. Read the existing scripts and outputs, read the
   paper PDF, and write a complete `writeup_148001.md` in the standard
   five-section format. Do not re-run anything that already produced
   output."*
3. Verify the bug independently by inspecting
   `148001-V1/REPLICATION/Code/Tables.do` (the prior session's last
   Read target) and the Table 5 Panel B section in particular.
4. Replace this stub with the full writeup.

---

## 4. File manifest (current)

| File | Status |
|---|---|
| `utils.py` | ✓ written by stalled session |
| `01_clean.py` | ✓ written and ran |
| `02_table2.py` | ✓ written and ran (main result matches) |
| `03_tables345.py` | ✓ written and ran |
| `04_data_audit.py` | ✓ written and ran |
| `05_robustness.py` | ✓ written and ran |
| `output/table2.csv` | ✓ produced |
| `output/robustness.csv` | ✓ produced |
| `writeup_148001.md` | **STUB ONLY** — needs full recovery writeup |
| `148001-V1/` | unzipped package, intact |
