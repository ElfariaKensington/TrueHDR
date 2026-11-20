#!/bin/zsh
# Builds EDRUnsuppress.dylib and installs it as a FloatingSidebar loader plugin, so stock Safari
# keeps full EDR headroom while another app is frontmost (see EDRUnsuppress.m).
#   ./install.sh            build + install (also installs the Plugins-aware loader)
#   ./install.sh uninstall  remove the plugin and restore the previous loader
# Needs the FloatingSidebar loader setup (SIP off, DYLD_INSERT_LIBRARIES login agent).
# Relaunch Safari afterwards; check with: log show --last 5m --predicate 'eventMessage CONTAINS "EDRUnsuppress"'
set -e
cd "${0:A:h}"
FSB_SRC="${0:A:h}/../../../FloatingSidebar"
DEST="$HOME/Library/Application Support/FloatingSidebar"
PLUG="$DEST/Plugins"
BACKUP="$DEST/FloatingSidebarLoader.dylib.pre-plugins"

if [[ "${1:-}" == uninstall ]]; then
  rm -f "$PLUG/EDRUnsuppress.dylib"; rmdir "$PLUG" 2>/dev/null || true
  [[ -f "$BACKUP" ]] && mv -f "$BACKUP" "$DEST/FloatingSidebarLoader.dylib"
  echo "removed; relaunch Safari"; exit 0
fi

[[ -f "$DEST/FloatingSidebarLoader.dylib" ]] || { echo "FloatingSidebar loader not installed at $DEST"; exit 1; }
grep -q '"/Plugins"' "$FSB_SRC/loader.c" || { echo "$FSB_SRC/loader.c has no Plugins support"; exit 1; }

clang -arch arm64 -arch arm64e -arch x86_64 -dynamiclib -fobjc-arc -O2 -Wall -mmacosx-version-min=26.0 \
  -framework Foundation -install_name @rpath/EDRUnsuppress.dylib EDRUnsuppress.m -o EDRUnsuppress.dylib
clang -dynamiclib -O2 -Wall -arch arm64 -arch arm64e -arch x86_64 -mmacosx-version-min=26.0 \
  -install_name @rpath/FloatingSidebarLoader.dylib "$FSB_SRC/loader.c" -o "$FSB_SRC/FloatingSidebarLoader.dylib"
codesign -f -s - EDRUnsuppress.dylib "$FSB_SRC/FloatingSidebarLoader.dylib" 2>/dev/null

mkdir -p "$PLUG"
[[ -f "$BACKUP" ]] || cp -p "$DEST/FloatingSidebarLoader.dylib" "$BACKUP"
# Replace via rename: running processes keep the old mapping, new launches get the new file.
cp "$FSB_SRC/FloatingSidebarLoader.dylib" "$DEST/.loader.new" && mv -f "$DEST/.loader.new" "$DEST/FloatingSidebarLoader.dylib"
cp EDRUnsuppress.dylib "$PLUG/.edr.new" && mv -f "$PLUG/.edr.new" "$PLUG/EDRUnsuppress.dylib"
echo "installed: $PLUG/EDRUnsuppress.dylib (loader backup: $BACKUP); relaunch Safari"
