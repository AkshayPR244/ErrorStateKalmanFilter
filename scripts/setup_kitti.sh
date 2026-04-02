#!/usr/bin/env bash
# =============================================================================
# KITTI Raw Data Setup Script
# Downloads, extracts, and verifies the 2011_09_26_drive_0001_sync sequence
# =============================================================================
set -e  # exit immediately on any error

DOWNLOADS_DIR="${1:-$HOME/Downloads}"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
DATE="2011_09_26"
SEQUENCE="2011_09_26_drive_0001_sync"

echo "============================================="
echo " KITTI Raw Data Setup"
echo "============================================="
echo " Looking for zips in: $DOWNLOADS_DIR"
echo " Extracting to:       $DATA_DIR"
echo "============================================="
echo ""

# ── Step 1: Check required zip files ─────────────────────────────────────────
SYNC_ZIP="$DOWNLOADS_DIR/${SEQUENCE}.zip"
CALIB_ZIP="$DOWNLOADS_DIR/${DATE}_calib.zip"

missing=0
for f in "$SYNC_ZIP" "$CALIB_ZIP"; do
    if [[ ! -f "$f" ]]; then
        echo "[MISSING] $f"
        missing=1
    else
        echo "[FOUND]   $f ($(du -sh "$f" | cut -f1))"
    fi
done

if [[ $missing -eq 1 ]]; then
    echo ""
    echo "============================================="
    echo " HOW TO GET THE MISSING FILES:"
    echo "============================================="
    echo ""
    echo " 1. Register (free) at:"
    echo "    https://www.cvlibs.net/datasets/kitti/raw_data.php"
    echo ""
    echo " 2. Log in, then find date: $DATE"
    echo "    Download these two files:"
    echo ""
    echo "    a) Sequence zip (sync+rectified, ~1.5 GB):"
    echo "       ${SEQUENCE}.zip"
    echo "       -- look under '$DATE' → drive 0001 → 'synced+rectified data'"
    echo ""
    echo "    b) Calibration zip (~4 KB):"
    echo "       ${DATE}_calib.zip"
    echo "       -- look under '$DATE' → 'calibration'"
    echo ""
    echo " 3. Save both to: $DOWNLOADS_DIR"
    echo ""
    echo " 4. Re-run this script:"
    echo "    bash scripts/setup_kitti.sh"
    echo ""
    echo " TIP: If your Downloads folder is elsewhere, pass the path:"
    echo "    bash scripts/setup_kitti.sh /path/to/downloads"
    echo ""
    exit 1
fi

# ── Step 2: Extract ───────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"

echo ""
echo "[1/2] Extracting sequence (~1.5 GB, this may take a minute)..."
unzip -q -o "$SYNC_ZIP" -d "$DATA_DIR"

echo "[2/2] Extracting calibration..."
unzip -q -o "$CALIB_ZIP" -d "$DATA_DIR"

# ── Step 3: Verify expected files ─────────────────────────────────────────────
echo ""
echo "Verifying extracted structure..."

EXPECTED_FILES=(
    "$DATA_DIR/$DATE/${SEQUENCE}/oxts/data/0000000000.txt"
    "$DATA_DIR/$DATE/${SEQUENCE}/oxts/timestamps.txt"
    "$DATA_DIR/$DATE/${SEQUENCE}/image_00/data/0000000000.png"
    "$DATA_DIR/$DATE/calib_cam_to_cam.txt"
    "$DATA_DIR/$DATE/calib_imu_to_velo.txt"
    "$DATA_DIR/$DATE/calib_velo_to_cam.txt"
)

all_ok=1
for f in "${EXPECTED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "  [OK] ${f#$DATA_DIR/}"
    else
        echo "  [MISSING] ${f#$DATA_DIR/}"
        all_ok=0
    fi
done

if [[ $all_ok -eq 0 ]]; then
    echo ""
    echo "ERROR: Some expected files are missing. The zip may be incomplete or"
    echo "       the wrong sequence was downloaded. Please check and re-download."
    exit 1
fi

# ── Step 4: Quick sanity check on OXTS file ───────────────────────────────────
OXTS_FILE="$DATA_DIR/$DATE/${SEQUENCE}/oxts/data/0000000000.txt"
FIELD_COUNT=$(awk '{print NF}' "$OXTS_FILE")

echo ""
if [[ "$FIELD_COUNT" -eq 30 ]]; then
    echo "  [OK] OXTS record has $FIELD_COUNT fields (expected 30)"
else
    echo "  [WARN] OXTS record has $FIELD_COUNT fields (expected 30) — check file format"
fi

# Count frames
FRAME_COUNT=$(ls "$DATA_DIR/$DATE/${SEQUENCE}/oxts/data/" | wc -l)
IMAGE_COUNT=$(ls "$DATA_DIR/$DATE/${SEQUENCE}/image_00/data/" | wc -l)
echo "  [OK] $FRAME_COUNT OXTS frames, $IMAGE_COUNT left-camera images"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo " Setup complete!"
echo " Data is at: data/$DATE/$SEQUENCE/"
echo "============================================="
echo ""
echo "OXTS file layout (first record, first 6 fields):"
echo "  lat lon alt roll pitch yaw ..."
head -c 80 "$OXTS_FILE" && echo " ..."
echo ""
