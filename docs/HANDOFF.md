# Handoff — AI upscaler + low-power HDR rework (updated 2026-11-17 03:50)

Baseline commit `05cd1c3`; this work is committed on top. Upscaler contract: `docs/SR_SPEC.md`.
HDR uses no ML (shader pipeline only). The neural net is used only for upscaling.

## Status: all tasks done (corners cut where noted)
| Area | Result |
|---|---|
| HDR power | HDR-only mode renders at min(source, display) resolution. Analysis runs at 1/8 res. 1080p→2880×1620: 1.30 → 0.51 ms GPU (2.6×). 720p: 4.2×. Measured with `dev/harness/bench.js` while training shared the GPU, so the numbers are noisy. |
| Birds/halos | `dev/harness/quality.js`: halo overshoot around small birds 38% → 0%. Pump and move variation ≤ 0.01%. Cut reset 17 → 0 frames. Caption peak 9.8× → 2.9× (text protection). |
| SR runtime | 17/17 tests vs PyTorch pass (f16 ≤ 1.7/255). Default tiling changed to 1 px/invocation, which is 1.7–2.7× faster in real WebKit (`dev/sr/webkit_run.ts bench`). |
| Custom models | `tsr-live-fast` (f12x5, 4.4k MAC/px) and `tsr-live-quality` (q20x5, 11.7k). Trained on codec-degraded DIV2K. Corner cut: stopped at 10k/30k steps, best.pt from step 7.5k. DIV2K-valid degraded PSNR: bicubic 28.38, fast 28.99, quality 29.06. Clean: 31.12 / 32.67 / 32.80. Merged into `extension/sr-models.js`; the FSRCNN live entries stay in the file but are overridden (last id wins). The FSRCNN comparison eval was not finished. |
| Focus dimming | Root cause (WebKit source): macOS 26 `NSApplicationShouldBeginSuppressingHighDynamicRangeContent` makes WebKit clamp page headroom to 1.6× (`WebProcessPool::applyEDRSuppressionIfNeeded`). Workaround, verified with `dev/focus-test` (`dual=poll`): a twin canvas is configured fresh when `document.hasFocus()` is false and shows about 4.9×. Switching back to the original canvas restores 8.8×. Implemented in `hdr-engine.js` (`updateFocusTwin`, 500 ms poll). |
| Build | `xcodebuild -project xcode/TrueHDR/TrueHDR.xcodeproj -scheme TrueHDR -configuration Release -derivedDataPath build build` |

## Known issues / next steps
- Resizing a configured canvas (`canvas.width` change) left the original canvas at 1.6× even while active in the focus test (`resizeAt`). The engine resizes on layout changes such as theater mode, so a resize can lose full headroom until reload. A possible fix is to recreate the canvas plus context on resize (untested).
- Anime quality (ArtCNN) is about 12 ms at 720p and 26 ms at 1080p in WebKit, so the budget guard will step it down at 60 fps.
- More training (resume: same `train.py --name final ...` command) would add roughly +0.1 dB.
- 2026-11-21: Stock Safari 27.3 (`dev/focus-test`, valid phases) clamps canvas, `<img>` and `<video>` to 1.6x while Safari is not the frontmost app. The twin canvas no longer escapes this. Another Safari window being key keeps full headroom. AppKit has no defaults opt-out. The patched WebKit was not in use: the running Safari was the system one. Fix: `tools/edr-unsuppress` (EDRUnsuppress.dylib, loaded through the FloatingSidebar loader's new `Plugins/` dir) drops `NSApplicationShouldBeginSuppressingHighDynamicRangeContentNotification` in Safari processes. Verified after a Safari relaunch: all 4 phases valid; canvas, img and video are identical active vs inactive (gpuMain@4 4.92x, the display's current headroom, img@4 4.0x), where they were 1.6x before.
- 2026-11-21: Text protection (flat-tone + line-run detector, monotone cap) is back in the Safari engine at the user's request. Measured on DIV2K photos: ≤2.8% of bright pixels dulled, but 30-75% of glyph pixels still leak. The first dense-edge variant dulled up to 62% of bright pixels (foliage with sky holes reads as text). TrueHDR-Game carries that first dense-edge variant, so it can dull real scenes. `quality.js` has caption/ui core-spread and rim metrics (frame `ui_1920x1080`).
- 2026-11-21: Upscaling targets 4K again: every below-4K source in a player at least 560x315 CSS px is rendered at 3840x2160. 80cf35d had removed this. The popup labels a non-upscaled source as `(native)`.
