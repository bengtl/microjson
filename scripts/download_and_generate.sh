#!/usr/bin/env bash
# Download remaining MouseLight brains from S3 and generate 3D tile pyramids.
# Run from repo root. Processes one brain at a time to be progressive.
set -e

OBJ_BASE="data/mouselight"
TILES_BASE="data/mouselight/tiles"
S3_BUCKET="s3://janelia-mouselight-imagery/registration"
MAX_ZOOM=3

# All brains with HortaObj/ on S3 (excluding the 5 we already have)
REMAINING=(
  2016-10-31
  2017-01-15
  2017-02-22
  2017-04-19
  2017-05-04
  2017-06-10
  2017-06-28
  2017-08-10
  2017-08-28
  2017-09-11
  2017-09-19
  2017-09-25
  2017-10-31
  2017-11-17
  2017-12-19
  2018-01-30
  2018-03-09
  2018-03-26
  2018-04-03
  2018-05-23
  2018-06-14
  2018-07-02
  2018-08-01
  2018-10-01
  2018-12-01
  2019-04-17
  2019-05-27
  2019-08-08
  2019-09-06
  2019-10-04
  2020-01-13
  2020-01-23
  2020-04-15
)

TOTAL=${#REMAINING[@]}
echo "=== Downloading and generating $TOTAL brains ==="
echo ""

for i in "${!REMAINING[@]}"; do
  DATE="${REMAINING[$i]}"
  N=$((i + 1))
  OBJ_DIR="$OBJ_BASE/$DATE"

  echo "============================================="
  echo "[$N/$TOTAL] $DATE"
  echo "============================================="

  # Download if not already present
  if [ -d "$OBJ_DIR" ] && [ "$(ls "$OBJ_DIR"/*.obj 2>/dev/null | wc -l)" -ge 700 ]; then
    echo "  Already downloaded, skipping download."
  else
    echo "  Downloading from S3..."
    mkdir -p "$OBJ_DIR"
    aws s3 sync "$S3_BUCKET/$DATE/HortaObj/" "$OBJ_DIR/" --no-sign-request --quiet
    echo "  Downloaded $(ls "$OBJ_DIR"/*.obj | wc -l) OBJ files."
  fi

  # Generate pyramid
  TILES_DIR="$TILES_BASE/$DATE/3dtiles"
  if [ -f "$TILES_DIR/tileset.json" ] && [ -f "$TILES_DIR/features.json" ]; then
    echo "  Pyramid already exists, skipping generation."
  else
    echo "  Generating pyramid..."
    .venv/bin/python scripts/generate_all_pyramids.py \
      --obj-base "$OBJ_BASE" \
      --output-base "$TILES_BASE" \
      --max-zoom "$MAX_ZOOM" \
      --only "$DATE"
  fi

  echo ""
done

# Regenerate manifest with all brains
echo "=== Regenerating manifest ==="
.venv/bin/python scripts/generate_all_pyramids.py \
  --obj-base "$OBJ_BASE" \
  --output-base "$TILES_BASE" \
  --max-zoom "$MAX_ZOOM"

echo ""
echo "=== DONE: All $TOTAL brains processed ==="
