#!/bin/bash
# Extract MD5-verified rars in place (Raw Video/f_raw_b_1/*.mp4). Existing files are not overwritten (-o-).
# Uses the official RARLAB unrar binary (~/bin/unrar). bsdtar (libarchive 3.6) gives header CRC errors on these RAR5 files.
set -eu
ROOT=${1:-data/raw/FallVision}
UNRAR=${UNRAR:-$HOME/bin/unrar}
find "$ROOT" -path "*Raw Video*" -name "*.rar" -print0 | while IFS= read -r -d '' f; do
  echo "== $f"
  (cd "$(dirname "$f")" && "$UNRAR" x -o- -idq "$(basename "$f")")
done
echo "videos: $(find "$ROOT" -iname '*.mp4' | wc -l)"
