#!/usr/bin/env python3
"""Tolerant zip extractor for openICPSR replication packages.

openICPSR packages occasionally contain filenames with non-UTF-8 bytes
(usually Cyrillic or Chinese) that macOS APFS cannot represent. The system
`unzip(1)` aborts on those with a non-zero exit, even though the
load-bearing content (Code/, Data/, etc.) would have extracted fine. This
extractor skips the individual bad entries, logs them, and continues.

Usage: extract_zip.py <zip> <dest> <extract_log>
Exits 0 if any entries extracted, 2 if zero extracted.
"""

import os
import sys
import zipfile


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: extract_zip.py <zip> <dest> <log>", file=sys.stderr)
        return 2

    src, dst, logpath = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(dst, exist_ok=True)

    skipped = 0
    extracted = 0
    with zipfile.ZipFile(src) as z, open(logpath, "w") as logf:
        logf.write(f"source: {src}\ndest: {dst}\n\n")
        for info in z.infolist():
            try:
                z.extract(info, dst)
                extracted += 1
            except (OSError, UnicodeError, zipfile.BadZipFile) as e:
                skipped += 1
                logf.write(f"SKIP {info.filename!r}: {type(e).__name__}: {e}\n")
        logf.write(f"\nTotal: extracted={extracted} skipped={skipped}\n")

    print(f"extracted={extracted} skipped={skipped}")
    return 0 if extracted > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
