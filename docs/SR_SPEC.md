# TrueHDR AI Super-Resolution — shared contract

Binding for: SR runtime (`extension/sr-runtime.js`), model files (`extension/sr-models.js`),
training (`training/`), and the engine (`extension/hdr-engine.js`). Change only via the orchestrator.

## Why this design
MetalFX / VideoToolbox super-resolution cannot reach video frames from a Safari Web Extension
(frames live in the WebContent process; the native handler only gets JSON messages). Safari's
WebGPU runs on Metal, so the upscaler is a CNN in WebGPU compute shaders (f16 when
`shader-f16` is available, f32 fallback). Video has no motion vectors/jitter, so this is
per-frame ML super-resolution (the RTX-VSR approach), trained on YouTube-like codec damage.

## Model family (all models, pretrained ports and custom-trained)
- Scale: 2x. Larger ratios chain two runtime instances (engine decides).
- Input: nonlinear luma `Y' = 0.2126 R' + 0.7152 G' + 0.0722 B'` of the decoded, gamma-encoded
  frame (values ~[0,1]), at source resolution. No linearisation.
- Layers: sequential `conv` layers, kernel k in {1,3,5}, stride 1, "same" size with
  **replicate (clamp-to-edge) padding** on every layer (PyTorch `padding_mode='replicate'`).
  Activations: `relu`, `prelu` (per-channel alpha), `crelu` (concat(relu(x), relu(-x)), doubles
  channels), `none`. Optional `skip`: index of an earlier layer whose (post-activation) output is
  added to this layer's pre-activation output (shapes must match). Layer -1 means the input luma.
- Last layer has `cout = 4` and is followed by depth-to-space (PyTorch `PixelShuffle(2)` ordering:
  channel c = 2*dy + dx).
- `base`: `"bicubic"` → result = bicubic2x(Y') + shuffled residual; `"none"` → result = shuffled output.
  bicubic2x must match `torch.nn.functional.interpolate(mode='bicubic', align_corners=False)`
  exactly (a = -0.75, clamped source indices).
- Channel counts used by custom models are multiples of 4 (vec4 packing).
- Cost budget at input resolution: `fast` ≤ 4.5k MAC/px, `quality` ≤ 12k MAC/px.

## Colour reconstruction (runtime, not learned)
`rgbUp = bicubic2x(R'G'B')` (same kernel as above), `Ysr` = model result, output
`out = max(rgbUp + (Ysr - dot(rgbUp, k709)), 0)` — luma delta applied equally to R'G'B'.
Output alpha = 1.

## Model file format (`extension/sr-models.js`)
Plain script (runs in the page MAIN world before `sr-runtime.js`), appends to
`globalThis.__TRUEHDR_SR_MODELS__ = (globalThis.__TRUEHDR_SR_MODELS__ || [])`:
```
{ id: 'tsr-live-fast', label: 'Live action · Fast', content: 'live'|'anime', tier: 'fast'|'quality',
  scale: 2, base: 'bicubic'|'none', macsPerPixel: <int>, license: '<spdx + source url>',
  layers: [ { k, cin, cout, act, skip?: <int>,
              w: '<base64 little-endian float16, layout [cout][cin][k][k]>',
              b: '<base64 f16 [cout]>', alpha?: '<base64 f16 [cout]>' } ] }
```
Total for all shipped models: < 250 KB (the file is injected into every frame).
Model ids the engine expects: `tsr-live-fast`, `tsr-live-quality`, `tsr-anime-fast`, `tsr-anime-quality`
(an id may alias another model's weights if only one exists for that content).

## Runtime API (`extension/sr-runtime.js`, MAIN world, loaded after sr-models.js, before hdr-engine.js)
```
globalThis.__TRUEHDR_SR__ = {
  version: 1,
  listModels() -> [{ id, label, content, tier, macsPerPixel }],
  async create(device, { modelId, useF16 }) -> SRUpscaler,   // throws on unknown model / compile error
}
SRUpscaler {
  modelId, scale: 2,
  // Records compute passes into `encoder`. inputView: rgba16float texture view (needs TEXTURE_BINDING),
  // nonlinear R'G'B' (alpha ignored), size w×h. Returns an rgba16float view, size 2w×2h, owned by the
  // upscaler and reused across frames (reallocated on size change). Output texture usage includes
  // TEXTURE_BINDING | STORAGE_BINDING (| RENDER_ATTACHMENT not required).
  encode(encoder, inputView, w, h) -> { view, width, height },
  destroy()
}
```
`device` is created by the engine; the engine requests `shader-f16` if `adapter.features` has it and
passes `useF16 = device.features.has('shader-f16')`. Runtime must not hold per-frame JS allocations
beyond bind groups for changed inputs (cache by input view identity).

## Verification standard
- WGSL output vs PyTorch reference of the same weights on real images: max abs error ≤ 2/255
  (f16), ≤ 1e-4 (f32).
- Benchmarks reported as GPU ms/frame at 1280×720 and 1920×1080 input on this Mac (M1 Pro).
