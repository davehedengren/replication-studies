#!/usr/bin/env python3
"""Post-loop integrity sweep for the replication driver.

Run this after the cron has chewed through the queue (or any time you want
a snapshot). It produces a markdown punch list you can hand to a fresh
Claude session for cleanup.
"""

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "replication_driver"
QUEUE = DRIVER / "queue.txt"
DONE = DRIVER / "done.txt"
RUN_LOG = DRIVER / "run.log"
README = REPO / "README.md"

# Required "slots" for a completed replication. Each slot is satisfied by
# any file matching any of its glob patterns. This accommodates papers that
# split a phase across multiple files (e.g., 02_table2.py + 02_table3.py
# instead of a single 02_tables.py).
REQUIRED_SLOTS = [
    ("utils",      ["utils.py"]),
    ("clean",      ["01_clean.py", "01_*.py"]),
    ("tables",     ["02_tables.py", "02_*.py", "03_table*.py"]),
    ("data_audit", ["04_data_audit.py", "04_*.py"]),
    ("robustness", ["05_robustness.py", "05_*.py"]),
]
OPTIONAL_SCRIPTS = [
    "03_figures.py",  # Not every paper has figures to replicate
]


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    queue = read_ids(QUEUE)
    done = set(read_ids(DONE))
    issues: list[tuple[str, str]] = []  # (severity, description)

    # 1. Queue vs done
    unprocessed = [pid for pid in queue if pid not in done]
    if unprocessed:
        issues.append(
            ("info", f"{len(unprocessed)} paper(s) still unprocessed: {', '.join(unprocessed)}")
        )

    # 2. Per-paper directory checks for everything in done
    for pid in sorted(done):
        rep_dir = REPO / f"replication_{pid}"
        writeup = rep_dir / f"writeup_{pid}.md"
        infeasible = rep_dir / "INFEASIBLE.md"

        if not rep_dir.exists():
            issues.append(("error", f"{pid}: marked done but {rep_dir.name}/ does not exist"))
            continue

        if not writeup.exists() and not infeasible.exists():
            issues.append(
                ("error", f"{pid}: no writeup_{pid}.md and no INFEASIBLE.md (mid-run crash?)")
            )
            continue

        # If feasible, check for the standard script slots
        if writeup.exists():
            checked_files: set[Path] = set()
            for slot_name, patterns in REQUIRED_SLOTS:
                matches: list[Path] = []
                for pat in patterns:
                    matches.extend(p for p in rep_dir.glob(pat) if p.is_file())
                # Dedupe while preserving order
                seen: set[Path] = set()
                unique = [p for p in matches if not (p in seen or seen.add(p))]
                if not unique:
                    issues.append(
                        ("warn", f"{pid}: no file matches the '{slot_name}' slot (expected one of {patterns})")
                    )
                    continue
                for p in unique:
                    checked_files.add(p)
                    try:
                        ast.parse(p.read_text())
                    except SyntaxError as e:
                        issues.append(
                            ("error", f"{pid}: {p.name} has syntax error at line {e.lineno}: {e.msg}")
                        )
            # Optional scripts: only syntax-check if present
            for script in OPTIONAL_SCRIPTS:
                p = rep_dir / script
                if p.exists() and p not in checked_files:
                    try:
                        ast.parse(p.read_text())
                    except SyntaxError as e:
                        issues.append(
                            ("error", f"{pid}: {script} has syntax error at line {e.lineno}: {e.msg}")
                        )

            # Cross-paper contamination check: does the writeup reference
            # any other paper's ID without justification?
            text = writeup.read_text()
            other_ids = set(re.findall(r"\breplication_(\d{6})\b", text)) | set(
                re.findall(r"\bwriteup_(\d{6})\b", text)
            )
            other_ids.discard(pid)
            if other_ids:
                issues.append(
                    (
                        "warn",
                        f"{pid}: writeup references other replication IDs ({', '.join(sorted(other_ids))}) — verify these are intentional cross-references, not contamination",
                    )
                )

    # 3. Stray replication_* directories not in done and not in original 53
    existing_dirs = {
        d.name.removeprefix("replication_")
        for d in REPO.iterdir()
        if d.is_dir() and d.name.startswith("replication_")
    }
    queue_set = set(queue)
    stray_from_queue = existing_dirs & queue_set - done
    if stray_from_queue:
        issues.append(
            (
                "error",
                f"replication_*/ directories exist for queue papers not yet marked done: {', '.join(sorted(stray_from_queue))} (likely mid-run crash)",
            )
        )

    # 4. README drift
    if README.exists():
        readme_text = README.read_text()
        for pid in sorted(done):
            if pid not in readme_text:
                issues.append(
                    ("warn", f"{pid}: marked done but ID does not appear anywhere in README.md")
                )

    # 5. Run log scan for FAILED entries. Suppress any FAILED line whose
    # paper ID later ended up in done.txt — those are historical transient
    # failures that were resolved on a retry, not current problems.
    if RUN_LOG.exists():
        for line in RUN_LOG.read_text().splitlines():
            if "FAILED" not in line:
                continue
            m = re.search(r"FAILED\s+(\d{6,7})", line)
            if m and m.group(1) in done:
                continue
            issues.append(("error", f"run.log: {line.strip()}"))

    # ----- output -----
    print("# Replication Driver Audit Report\n")
    print(f"- Queue size: **{len(queue)}**")
    print(f"- Marked done: **{len(done)}**")
    print(f"- Unprocessed: **{len(unprocessed)}**")
    print(f"- Issues found: **{len(issues)}**\n")

    if not issues:
        print("All clear.")
        return 0

    by_sev: dict[str, list[str]] = {"error": [], "warn": [], "info": []}
    for sev, desc in issues:
        by_sev[sev].append(desc)

    for sev in ("error", "warn", "info"):
        if by_sev[sev]:
            print(f"\n## {sev.upper()} ({len(by_sev[sev])})\n")
            for desc in by_sev[sev]:
                print(f"- {desc}")

    return 1 if by_sev["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
