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

    # Open the zip *before* creating the destination, so a failure to read
    # the source (PermissionError, BadZipFile, missing file) doesn't leave
    # behind an empty destination directory that the driver would later
    # mistake for a successful extraction.
    try:
        zf = zipfile.ZipFile(src)
    except (OSError, zipfile.BadZipFile) as e:
        print(f"open failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    os.makedirs(dst, exist_ok=True)
    skipped = 0
    extracted = 0
    try:
        with zf, open(logpath, "w") as logf:
            logf.write(f"source: {src}\ndest: {dst}\n\n")
            for info in zf.infolist():
                try:
                    zf.extract(info, dst)
                    extracted += 1
                except (OSError, UnicodeError, zipfile.BadZipFile) as e:
                    skipped += 1
                    logf.write(f"SKIP {info.filename!r}: {type(e).__name__}: {e}\n")
            logf.write(f"\nTotal: extracted={extracted} skipped={skipped}\n")
    finally:
        # If we extracted nothing, clean up the empty dest dir so the next
        # run treats this paper as un-extracted and retries from scratch.
        if extracted == 0:
            try:
                os.rmdir(dst)
            except OSError:
                pass

    print(f"extracted={extracted} skipped={skipped}")
    return 0 if extracted > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
