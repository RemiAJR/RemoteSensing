#!/usr/bin/env bash
# setup_data.sh — Download and extract MUMUCD PRISMA data from Zenodo.
#
# Usage:
#   bash setup_data.sh [--dest data/mumucd] [--dry-run]
#
# Requires: zenodo-get (pip install zenodo-get)

set -euo pipefail

DEST="data/mumucd"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)    DEST="$2";    shift 2 ;;
    --dry-run) DRY_RUN=true; shift   ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

RECORD_ID="10674011"

echo "=== MUMUCD PRISMA download ==="
echo "  Zenodo record : ${RECORD_ID}"
echo "  Destination   : ${DEST}"
echo ""

if $DRY_RUN; then
  echo "[dry-run] Would create: ${DEST}"
  echo "[dry-run] Would run: zenodo_get ${RECORD_ID} --record-filter '*prisma*' -o ${DEST}"
  exit 0
fi

mkdir -p "${DEST}"

# Check for zenodo_get
if ! command -v zenodo_get &>/dev/null; then
  echo "zenodo_get not found. Installing..."
  pip install zenodo-get
fi

echo "Downloading PRISMA archives from Zenodo record ${RECORD_ID}..."
zenodo_get "${RECORD_ID}" --record-filter "*prisma*" -o "${DEST}"

# Extract any archives found
echo "Extracting archives..."
for f in "${DEST}"/*.tar "${DEST}"/*.tar.gz "${DEST}"/*.tgz; do
  [[ -f "$f" ]] || continue
  echo "  Extracting $f ..."
  tar -xf "$f" -C "${DEST}"
done

echo ""
echo "Done. PRISMA GeoTIFFs should now be in: ${DEST}/"
echo "Verify with: ls ${DEST}/*.tif | wc -l  (expect ~70 files)"
