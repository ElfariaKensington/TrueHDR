#!/bin/sh
# Generates the SR test images into dev/sr/out/testdata (all local: ffmpeg lavfi + macOS sample photos).
set -eu
D="$(cd "$(dirname "$0")" && pwd)/out/testdata"
mkdir -p "$D"
ff() { ffmpeg -v error -y "$@"; }
ff -f lavfi -i testsrc2=size=320x180:rate=1 -frames:v 1 "$D/testsrc2_320x180.png"
ff -i "$D/testsrc2_320x180.png" -vf crop=203:117:37:21 "$D/testsrc2_203x117.png"
# Photo-like content: macOS sample photos (HEIC -> PNG via sips), HR versions for PSNR, LR = downscaled
TMP="$(mktemp -d)"
sips -s format png "/Library/User Pictures/Flowers/Dahlia.heic" --out "$TMP/dahlia.png" >/dev/null
sips -s format png "/System/Library/Desktop Pictures/Sonoma.heic" --out "$TMP/sonoma.png" >/dev/null
sips -s format png "/Library/User Pictures/Fun/Gingerbread Man.heic" --out "$TMP/ginger.png" >/dev/null
ff -i "$TMP/dahlia.png" -vf format=rgb24 "$D/dahlia_hr_512.png"
ff -i "$TMP/sonoma.png" -vf "crop=3840:2160:1088:1928,scale=1280:720:flags=area,format=rgb24" "$D/sonoma_hr_1280x720.png"
ff -i "$TMP/ginger.png" -vf format=rgb24 "$D/ginger_hr_512.png"
ff -i "$D/dahlia_hr_512.png" -vf scale=256:256:flags=area "$D/dahlia_256.png"
ff -i "$D/sonoma_hr_1280x720.png" -vf scale=640:360:flags=area "$D/sonoma_640x360.png"
ff -i "$D/ginger_hr_512.png" -vf scale=256:256:flags=area "$D/ginger_256.png"
rm -rf "$TMP"
ls -1 "$D"
