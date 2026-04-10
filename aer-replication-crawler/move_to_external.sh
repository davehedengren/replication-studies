#!/bin/bash
# Move completed downloads from staging to the package archive (e.g. external drive).
# Run after a crawler session, or via cron.
#
# Reads SRC and DST from env vars (sourced from .env if available), with sensible
# defaults. Override by setting DOWNLOAD_DIR / PACKAGE_ARCHIVE in your .env.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$SCRIPT_DIR/.env"
    set +a
fi

SRC="${DOWNLOAD_DIR:-$SCRIPT_DIR/download_staging}"
DST="${PACKAGE_ARCHIVE:-$SCRIPT_DIR/downloads}"

mkdir -p "$DST"
count=0
for f in "$SRC"/*.zip; do
    [ -f "$f" ] || continue
    mv "$f" "$DST/" && count=$((count + 1))
done
echo "Moved $count files from $SRC to $DST"
