You are running as part of an automated replication study driver. Process EXACTLY ONE paper end-to-end and then stop. Paper ID: {{ID}}.

# Hard rules (read first)

1. **Single-paper scope.** You are working on paper {{ID}} ONLY. Do not read, reference, or import data from any other `replication_*/` directory unless you need to cite a directly comparable published result, and even then, name the source explicitly in the writeup.
2. **No memory writes.** Do NOT write anything to the persistent memory system at `~/.claude/projects/-Users-davehedengren-code-replication-studies/memory/` during this run. Memory is for cross-session user/project facts. Replication-specific data must not pollute it.
3. **No git commits.** Dave reviews and commits each paper himself. Leave the working tree dirty.
4. **No `docs/` edits.** The web view is out of scope. Touch only the per-paper directory and `README.md`.
5. **Stop when done.** When the writeup is finalized and `README.md` is updated, end the session. Do not look for "what's next."

# Inputs

- **Paper PDF**: `/Volumes/Extreme SSD/AER_replication_data_pdfs/{{ID}}.pdf`
- **Replication package** (already unzipped by the driver): `./{{ID}}-V1/`
- **Per-paper recipe**: `./instructions.txt` — follow this verbatim, all five phases.
- **Gold-standard writeup example to imitate**: `./replication_226781/writeup_226781.md`
- **Aggregate writeup to update**: `./README.md`

# Python environment (required)

**Do NOT create a new venv.** Use the shared venv at `./venv/` that is already
set up with numpy, pandas, scipy, matplotlib, statsmodels, linearmodels,
scikit-learn, pyreadr, pyreadstat, openpyxl, xlrd, geopandas, pyarrow,
tabulate, and tqdm (see `./requirements.txt`).

Activate it for every script you run:

```bash
source venv/bin/activate && python replication_{{ID}}/01_clean.py
```

If a paper needs a package that is not in the shared venv, install it into
the shared venv with `pip install <package>` after activation. Prefer
maintained PyPI packages; avoid one-off forks. If you install something new,
append it to `./requirements.txt` so future papers inherit it.

# Workflow

Follow `instructions.txt` for Phases 1-5. Some clarifications specific to this automated context:

- **Phase 1 (Orientation)**: Read the PDF. Read the README inside `{{ID}}-V1/`. Read the source code. If the paper is infeasible (restricted data, 20+ GB raw with no intermediates, MATLAB-only structural model with no empirical component, etc.), create `replication_{{ID}}/INFEASIBLE.md` explaining exactly why, update README.md (Skipped Papers table + Summary Statistics), and stop.
- **Phase 2 (Translate & Reproduce)**: Create `replication_{{ID}}/`. Write `utils.py` and the numbered scripts. Run each script. Compare to published values.
- **Phase 3 (Data Audit)**: Write `04_data_audit.py`.
- **Phase 4 (Robustness)**: Write `05_robustness.py` with 8-12 checks tailored to the paper's method.
- **Phase 5 (Writeup)**: Write `replication_{{ID}}/writeup_{{ID}}.md` following the structure in `instructions.txt`. Match the level of detail in `replication_226781/writeup_226781.md`.

# README update (required after Phase 5)

After the writeup is finalized, update `./README.md`:

1. **Summary Statistics table** (top of file): increment "Papers assessed" by 1, and increment either "Completed replications" or "Skipped (data unavailable)" depending on outcome. If a bug was found, increment "Bugs found"; if it changed conclusions, also increment "Bugs affecting conclusions."
2. **Add a row to the appropriate table**:
   - Full or near-exact replication → "Full / Near-Exact Replications" table
   - Partial replication → "Partial Replications (data constraints)" table
   - Infeasible → "Skipped Papers" table
3. **Add a "Key Findings by Paper" entry** for the new paper. Match the tone and length of existing entries (3-6 sentences, mention the headline result, any concerns, and bug status if applicable).
4. **If a bug was found**, add a row to the "Bugs Found" table.

Do all README edits as a single coherent pass. Re-read the file after editing to make sure the tables still parse and counts are consistent.

# Quality checks before stopping

- Every script in `replication_{{ID}}/` runs without error under the shared venv (`source venv/bin/activate && python replication_{{ID}}/<script>`).
- The writeup contains side-by-side comparisons of your computed values vs the published values, to enough precision to demonstrate the match (or document the discrepancy).
- The README counts add up: assessed = completed + skipped.
- The new entry in the README is internally consistent with the writeup.

When all of the above is true, stop. The driver will detect completion by the existence of `replication_{{ID}}/writeup_{{ID}}.md` or `replication_{{ID}}/INFEASIBLE.md`.
