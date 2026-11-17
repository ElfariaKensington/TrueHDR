#!/bin/sh
# Side-by-side crops for visual review (after test_runtime.ts wrote the PNGs):
#   left->right: bicubic | tsr-live-fast | tsr-live-quality | tsr-anime-quality  (GPU f16 outputs)
# Usage: dev/sr/montage.sh <image> <crop w:h:x:y in output pixels>
set -eu
O="$(cd "$(dirname "$0")" && pwd)/out"
IMG="$1"; CROP="$2"
ffmpeg -v error -y -i "$O/${IMG}_bicubic.png" -i "$O/${IMG}_tsr-live-fast_f16.png" -i "$O/${IMG}_tsr-live-quality_f16.png" \
  -i "$O/${IMG}_tsr-anime-quality_f16.png" -filter_complex \
  "[0]crop=$CROP[a];[1]crop=$CROP[b];[2]crop=$CROP[c];[3]crop=$CROP[d];[a][b][c][d]hstack=4,scale=iw*2:ih*2:flags=neighbor" \
  "$O/montage_${IMG}.png"
echo "$O/montage_${IMG}.png"
