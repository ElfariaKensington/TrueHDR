#!/bin/zsh
# Builds the probe tools and HDR test assets into dev/focus-test/{.build,assets}.
set -euo pipefail
HERE=${0:A:h}
FT=${HERE:h}
B=$FT/.build
A=$FT/assets
mkdir -p $B $A
swiftc -O -o $B/hdrprobe $HERE/hdrprobe.swift
APP=$B/EDRApp.app
mkdir -p $APP/Contents/MacOS
swiftc -O -o $APP/Contents/MacOS/EDRApp $HERE/edrapp.swift
cat > $APP/Contents/Info.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>dev.truehdr.focustest.edrapp</string>
<key>CFBundleName</key><string>EDRApp</string>
<key>CFBundleExecutable</key><string>EDRApp</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>NSPrincipalClass</key><string>NSApplication</string>
<key>LSMinimumSystemVersion</key><string>26.0</string>
</dict></plist>
EOF
codesign -s - --force $APP >/dev/null 2>&1 || true

# HDR <img>: gain-map HEIC, patches at linear 1, 2, 4 (x SDR white)
[[ -f $A/patches-hdr.heic ]] || $B/hdrprobe make-hdr-image $A/patches-hdr.heic 1,2,4

# HDR <video>: HEVC Main10 PQ, patches at 203 / 406 / 812 nits (= 1x / 2x / 4x of BT.2408 ref white)
if [[ ! -f $A/patches-pq.mp4 ]]; then
  python3 -I - "$A/patches-pq.png" <<'EOF'
import sys, struct, zlib
import numpy as np
def pq(n):
    m1, m2, c1, c2, c3 = 2610/16384, 2523/4096*128, 3424/4096, 2413/4096*32, 2392/4096*32
    y = (n/10000.0)**m1
    return ((c1 + c2*y)/(1 + c3*y))**m2
w, h = 480, 160
img = np.zeros((h, w, 3), dtype=np.uint16)
for i, nits in enumerate([203, 406, 812]):
    img[:, i*160:(i+1)*160, :] = int(round(pq(nits)*65535))
raw = b''.join(b'\x00' + img[y].astype('>u2').tobytes() for y in range(h))
def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t+d) & 0xffffffff)
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 16, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')
open(sys.argv[1], 'wb').write(png)
EOF
  ffmpeg -hide_banner -loglevel error -y -loop 1 -framerate 30 -i $A/patches-pq.png -t 4 \
    -vf "scale=in_range=full:out_range=tv:out_color_matrix=bt2020nc,format=yuv420p10le" \
    -c:v libx265 -tag:v hvc1 -x265-params "log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited:hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400" \
    -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc -movflags +faststart \
    $A/patches-pq.mp4
  rm -f $A/patches-pq.png
fi
ls -la $B $A
