// Shared Deno helpers for the TrueHDR offscreen harness (validate / bench / quality).
// Run scripts with: deno run --unstable-webgpu -A dev/harness/<script>.js
// Sources are texture_2d<f32> substitutes of the shipped texture_external shaders.

const HERE = new URL('.', import.meta.url).pathname;
export const ROOT = new URL('../../', import.meta.url).pathname;
export const FRAMES_DIR = `${ROOT}dev/media/frames/`;

export async function getDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const requiredFeatures = ['timestamp-query', 'shader-f16'].filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredFeatures });
  device.addEventListener?.('uncapturederror', (e) => console.error('[uncaptured]', e.error?.message || e.error));
  return { adapter, device };
}

// --------------------------------------------------------------------------- engine loading
export async function loadNewCoreApi(path = `${ROOT}extension/hdr-engine.js`) {
  delete globalThis.__TRUEHDR_ENGINE_LOADED__;
  delete globalThis.__TRUEHDR_CORE__;
  const src = await Deno.readTextFile(path);
  (0, eval)(src);
  if (!globalThis.__TRUEHDR_CORE__) throw new Error('engine did not expose __TRUEHDR_CORE__');
  return globalThis.__TRUEHDR_CORE__;
}

export async function gitShow(rev, file) {
  const out = await new Deno.Command('git', { args: ['-C', ROOT, 'show', `${rev}:${file}`], stdout: 'piped', stderr: 'piped' }).output();
  if (!out.success) throw new Error(new TextDecoder().decode(out.stderr));
  return new TextDecoder().decode(out.stdout);
}

export function extractShaders(src) {
  const re = /const (\w+) = \/\* wgsl \*\/ `([\s\S]*?)`;/g;
  const out = {};
  let m;
  while ((m = re.exec(src))) out[m[1]] = m[2];
  return out;
}

// --------------------------------------------------------------------------- old (baseline) pipeline
// Faithful port of the baseline renderFrame(): copy+linearise @src, 16x12 reduce, [restore @src],
// final EASU/HDR @canvas. texture_external -> texture_2d<f32> as for the new core.
export class OldRenderer {
  static async create(device, rev = 'HEAD') {
    const sh = extractShaders(await gitShow(rev, 'extension/hdr-engine.js'));
    const r = new OldRenderer();
    r.device = device;
    const vs = device.createShaderModule({ code: sh.FULLSCREEN_VS });
    const mk = (code) => device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: vs, entryPoint: 'vsMain' },
      fragment: { module: device.createShaderModule({ code }), entryPoint: 'fsMain', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' }
    });
    [r.copy, r.restore, r.reduce, r.hdr] = await Promise.all([
      mk(sh.COPY_SRC_FS.split('texture_external').join('texture_2d<f32>')),
      mk(sh.RESTORE_FS), mk(sh.REDUCE_FS), mk(sh.HDR_RENDER_FS)
    ]);
    r.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    r.ubuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    r.u = new Float32Array(20);
    r.state = [0, 1].map(() => device.createTexture({ size: [1, 1], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }));
    r.stateV = r.state.map((t) => t.createView());
    r.stateIdx = 0; r.srcIdx = 0; r.w = 0; r.h = 0; r.hasPrev = false; r.seed = 0;
    return r;
  }

  ensure(w, h) {
    if (this.w === w && this.h === h) return;
    const d = this.device;
    const desc = { size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING };
    this.src = [d.createTexture(desc), d.createTexture(desc)];
    this.srcV = this.src.map((t) => t.createView());
    this.feat = d.createTexture(desc);
    this.featV = this.feat.createView();
    this.w = w; this.h = h; this.hasPrev = false;
    this.restoreBG = [0, 1].map((s) => d.createBindGroup({ layout: this.restore.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.srcV[s] }, { binding: 1, resource: this.srcV[1 - s] }, { binding: 2, resource: { buffer: this.ubuf } }] }));
    this.reduceBG = [0, 1].map((s) => [0, 1].map((st) => d.createBindGroup({ layout: this.reduce.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.srcV[s] }, { binding: 1, resource: this.stateV[st] }, { binding: 2, resource: { buffer: this.ubuf } }] })));
    this.hdrBG = [0, 1].map((s) => [0, 1].map((st) => d.createBindGroup({ layout: this.hdr.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.sampler }, { binding: 1, resource: this.srcV[s] }, { binding: 2, resource: this.featV },
      { binding: 3, resource: this.stateV[st] }, { binding: 4, resource: { buffer: this.ubuf } }] })));
  }

  // p: { srcW, srcH, rect, s (settings), upscale, reset, tsw: { querySet, begin, end } }
  encode(encoder, srcView, targetView, p) {
    const d = this.device;
    this.ensure(p.srcW, p.srcH);
    const s = p.s;
    const u = this.u;
    u.set(p.rect, 0);
    u[4] = s.peakBoost; u[5] = s.threshold; u[6] = s.localPunch; u[7] = s.gamutBoost;
    u[8] = s.shadowDepth; u[9] = s.aplGuard; u[10] = s.splitPos ?? -1; u[11] = (s.hdr === false ? 0 : 1) | 2;
    this.seed = (this.seed + 1) % 256;
    const reset = p.reset || !this.hasPrev;
    u[12] = 1 / p.srcW; u[13] = 1 / p.srcH; u[14] = this.seed; u[15] = reset ? 1 : 0;
    u[16] = p.srcW; u[17] = p.srcH; u[18] = s.upscaleSharpness ?? 0.55; u[19] = p.upscale ? 1 : 0;
    d.queue.writeBuffer(this.ubuf, 0, u);
    const prev = this.stateIdx; const next = 1 - prev; this.stateIdx = next;
    const cur = 1 - this.srcIdx; this.srcIdx = cur; this.hasPrev = true;
    const pass = (view, pipe, bg, tsw) => {
      const rp = encoder.beginRenderPass({ colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }], ...(tsw ? { timestampWrites: tsw } : {}) });
      rp.setPipeline(pipe); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
    };
    const copyBG = d.createBindGroup({ layout: this.copy.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: srcView }] });
    const t = p.tsw;
    pass(this.srcV[cur], this.copy, copyBG, t && t.begin !== undefined ? { querySet: t.querySet, beginningOfPassWriteIndex: t.begin } : null);
    pass(this.stateV[next], this.reduce, this.reduceBG[cur][prev]);
    if (p.upscale) pass(this.featV, this.restore, this.restoreBG[cur]);
    pass(targetView, this.hdr, this.hdrBG[cur][next], t && t.end !== undefined ? { querySet: t.querySet, endOfPassWriteIndex: t.end } : null);
  }
}

// --------------------------------------------------------------------------- new pipeline wrapper
export class NewPipeline {
  static async create(device, api) {
    const n = new NewPipeline();
    n.api = api || await loadNewCoreApi();
    n.core = await n.api.createCore(device, { texture2DSource: true });
    n.r = new n.api.FrameRenderer(n.core);
    n.device = device;
    n.seed = 0;
    return n;
  }

  // p: { srcW, srcH, outW, outH, rect, s, mode, sr, reset, dt, measureTs }
  encode(encoder, srcView, targetView, p) {
    const s = p.s;
    this.seed = (this.seed + 1) % 256;
    return this.r.encode(encoder, srcView, targetView, {
      srcW: p.srcW, srcH: p.srcH, rect: p.rect, outW: p.outW, outH: p.outH,
      tone: [s.peakBoost, s.threshold, s.localPunch, s.gamutBoost],
      shadowDepth: s.shadowDepth, aplGuard: s.aplGuard, splitPos: s.splitPos ?? -1,
      hdr: s.hdr !== false, bypassNativeHdr: true, sharpness: s.upscaleSharpness ?? 0.25,
      mode: p.mode || 'direct', sr: p.sr, reset: p.reset, dt: p.dt ?? 1 / 30, frameSeed: this.seed, measure: p.measure, tsw: p.tsw
    });
  }

  async readState() {
    const d = this.device;
    const buf = d.createBuffer({ size: this.api.STATE_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = d.createCommandEncoder();
    e.copyBufferToBuffer(this.r.state, 0, buf, 0, this.api.STATE_BYTES);
    d.queue.submit([e.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(buf.getMappedRange().slice(0));
    buf.unmap(); buf.destroy();
    return { apl: f[0], hdrFrac: f[1], effPeak: f[2], textCap: f[3], aplRaw: f[4], cut: f[5], histDist: f[6], frames: f[7], p50: f[8], p98: f[9] };
  }
}

// --------------------------------------------------------------------------- presets (popup.js)
export const PRESETS = {
  natural: { peakBoost: 4.0, threshold: 0.52, localPunch: 0.50, gamutBoost: 0.30, shadowDepth: 0.15, aplGuard: 0.60 },
  vivid: { peakBoost: 6.0, threshold: 0.46, localPunch: 0.70, gamutBoost: 0.45, shadowDepth: 0.14, aplGuard: 0.45 },
  cinema: { peakBoost: 10.0, threshold: 0.38, localPunch: 0.95, gamutBoost: 0.70, shadowDepth: 0.18, aplGuard: 0.30 },
  comfort: { peakBoost: 2.6, threshold: 0.58, localPunch: 0.35, gamutBoost: 0.20, shadowDepth: 0.08, aplGuard: 0.70 }
};

// --------------------------------------------------------------------------- frames & textures
let manifestCache = null;
export async function manifest() {
  if (!manifestCache) manifestCache = JSON.parse(await Deno.readTextFile(`${FRAMES_DIR}manifest.json`));
  return manifestCache;
}

export async function loadFrames(key) {
  const m = (await manifest())[key];
  if (!m) throw new Error(`no frame set ${key}; run python3 dev/harness/gen_media.py`);
  const bytes = await Deno.readFile(`${FRAMES_DIR}${m.file}`);
  const sz = m.w * m.h * 4;
  const frames = [];
  for (let i = 0; i < m.frames; i++) frames.push(bytes.subarray(i * sz, (i + 1) * sz));
  return { ...m, data: frames };
}

export function makeSourceTexture(device, w, h) {
  const tex = device.createTexture({ size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  return { tex, view: tex.createView(), w, h };
}

export function uploadFrame(device, src, rgba) {
  device.queue.writeTexture({ texture: src.tex }, rgba, { bytesPerRow: src.w * 4, rowsPerImage: src.h }, [src.w, src.h]);
}

export function makeTarget(device, w, h) {
  const tex = device.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING });
  return { tex, view: tex.createView(), w, h };
}

function f16(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export async function readTarget(device, target) {
  const { w, h } = target;
  const bpr = Math.ceil((w * 8) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: target.tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
  device.queue.submit([e.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const u16 = new Uint16Array(buf.getMappedRange());
  const out = new Float32Array(w * h * 4);
  const rowU16 = bpr / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w * 4; x++) out[y * w * 4 + x] = f16(u16[y * rowU16 + x]);
  }
  buf.unmap(); buf.destroy();
  return out;
}

// Output (Display-P3, sRGB-transfer, extended) -> linear luminance relative to SDR white
const eotf = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
export function outLuma(rgba, w, h) {
  const L = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    L[i] = 0.2289746 * eotf(Math.max(rgba[i * 4], 0)) + 0.6917385 * eotf(Math.max(rgba[i * 4 + 1], 0)) + 0.0792869 * eotf(Math.max(rgba[i * 4 + 2], 0));
  }
  return L;
}

// Input sRGB8 -> linear luminance
const LUT = new Float32Array(256).map((_, i) => eotf(i / 255));
export function inLuma(rgba8, w, h) {
  const L = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) L[i] = 0.2126 * LUT[rgba8[i * 4]] + 0.7152 * LUT[rgba8[i * 4 + 1]] + 0.0722 * LUT[rgba8[i * 4 + 2]];
  return L;
}

export async function submitAndWait(device, encoder) {
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

export function fmt(x, d = 3) {
  return typeof x === 'number' ? x.toFixed(d) : String(x);
}

export function table(rows, cols) {
  const widths = cols.map((c) => Math.max(c.length, ...rows.map((r) => String(r[c] ?? '').length)));
  const line = (vals) => vals.map((v, i) => String(v ?? '').padEnd(widths[i])).join('  ');
  console.log(line(cols));
  console.log(widths.map((w) => '-'.repeat(w)).join('  '));
  for (const r of rows) console.log(line(cols.map((c) => r[c])));
}

export { HERE };
