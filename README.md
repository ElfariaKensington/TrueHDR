# TrueHDR for Safari

True hardware **Extended Dynamic Range (EDR / HDR)** inverse tone-mapping for YouTube and web videos in Safari on macOS.

Instead of applying SDR contrast/saturation CSS filters, **TrueHDR** intercepts video frames via WebGPU (`device.importExternalTexture`) and renders into an unclamped IEEE 754 **`rgba16float`** swapchain configured with **Display P3** color space and **`toneMapping: { mode: 'extended' }`**, driving macOS Core Animation EDR headroom past SDR white (`> 1.0`) on Liquid Retina XDR and HDR displays.

## How It Works
1. **True Float16 EDR Pipeline (`extension/hdr-engine.js`)**:
   - **Pass 1 (GPU Scene Analyzer)**: Downsamples each video frame to `32×18` and updates a `1×1` temporal EMA state texture entirely on the GPU (zero CPU readback), measuring Average Picture Level (APL) and checking whether the incoming video is already native HDR (PQ/HLG).
   - **Pass 2 (Inverse Tone Mapping + Display P3)**: Anchors skin tones and midtones near SDR reference, smoothly expands compressed highlights above the knee up to `8.0×` SDR white into hardware EDR headroom, isolates local specular highlights (sparks, streetlamps, reflections, neon), converts Rec.709 → Display P3 with controlled wide-gamut saturation, and applies triangular-PDF dithering to remove 8-bit YouTube banding.
   - **White-Screen Guard**: Automatically attenuates peak EDR boost on bright full-screen slides/pages so text and white backgrounds stay comfortable.
   - **Native HDR Auto-Bypass**: Detects native PQ/HLG video frames via `VideoFrame.colorSpace.transfer` (and pixel peaks `> 1.03`) and passes them through untouched.
2. **AI Super-Resolution (`extension/sr-runtime.js`, `extension/sr-models.js`)**:
   - A small CNN runs per frame in WebGPU compute shaders (Safari's WebGPU runs on Metal; f16 when `shader-f16` is available), upscaling 2× per pass up to your display's **native resolution**. MetalFX cannot reach Safari's video frames, so this follows the per-frame ML video SR approach.
   - Models are trained on YouTube-like compression damage (blocking, banding, softened detail) and reconstruct luma; colour is carried by a matched bicubic path.
   - Options: **Fast** / **Quality** model tier, **Live action** / **Animation** content, plus optional **Extra Sharpening** on top of the AI output.
3. **YouTube & Web Player Integration**:
   - Attaches seamlessly inside `.html5-video-container` (below player controls, subtitles, and menus) and automatically tracks Theater Mode, Fullscreen, Shorts, and `object-fit` letterboxing.
   - Only renders when new video frames arrive (`requestVideoFrameCallback`), using `0%` GPU when paused.

## Keyboard Shortcuts
- `⌥H` (**Option + H**): Toggle TrueHDR on/off for the current site
- `⌥S` (**Option + S**): Toggle Split-Screen comparison (`SDR | EDR`)

## Installing & Enabling in Safari
1. Launch the signed app at `build/Build/Products/Release/TrueHDR.app` once (signed with `Apple Development: elfametesar@gmail.com (N77M7JPZ84)`, Team `RTFTQ673C9`).
2. Open **Safari → Settings → Extensions**, enable **TrueHDR**, and grant website access (**Always Allow on Every Website** or per-site for `youtube.com`).
3. Play any SDR YouTube video on your XDR/HDR display and click the **TrueHDR** toolbar button to adjust presets, peak brightness, AI upscaling, and split-screen comparison!

*(Note: If you ever rebuild and Safari still doesn't list a locally dev-signed extension depending on Safari's unsigned/dev policy, enable **Safari → Settings → Advanced → Show features for web developers** and **Develop → Allow Unsigned Extensions**.)*
