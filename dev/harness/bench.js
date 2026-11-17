// Offscreen GPU benchmark: baseline (git HEAD) vs current engine.
//   deno run --unstable-webgpu -A dev/harness/bench.js [--frames=300] [--rev=HEAD]
import { getDevice, loadFrames, makeSourceTexture, uploadFrame, makeTarget, NewPipeline, OldRenderer, PRESETS, table, fmt } from './lib.js';

const args = Object.fromEntries(Deno.args.map((a) => a.replace(/^--/, '').split('=')));
const N = Number(args.frames || 300);
const WARM = 30;
const rev = args.rev || 'HEAD';

const { device } = await getDevice();
if (!device.features.has('timestamp-query')) console.warn('timestamp-query unavailable: GPU ms column will be empty');
const oldR = await OldRenderer.create(device, rev);
const newP = await NewPipeline.create(device);
await import('./sr-stub.js');
const srStub = await globalThis.__TRUEHDR_SR__.create(device, { modelId: 'tsr-live-quality', useF16: device.features.has('shader-f16') });

const S = { ...PRESETS.vivid, upscaleSharpness: 0.25 };
// GPU ms/frame: K frames encoded into one command buffer, timestamps at the first pass begin and
// the last pass end, span / K. Configs are interleaved round-robin for R rounds and we report the
// minimum (least-contended) and the median, because other processes may share the GPU.
const K = 8;
const R = Number(args.rounds || 40);
const qs = device.createQuerySet({ type: 'timestamp', count: 2 });
const qResolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const qRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const sorted = (xs) => [...xs].sort((a, b) => a - b);

async function readSpan(e) {
  e.resolveQuerySet(qs, 0, 2, qResolve, 0);
  e.copyBufferToBuffer(qResolve, 0, qRead, 0, 16);
  device.queue.submit([e.finish()]);
  await qRead.mapAsync(GPUMapMode.READ);
  const a = new BigInt64Array(qRead.getMappedRange());
  const ms = Number(a[1] - a[0]) / 1e6;
  qRead.unmap();
  return ms;
}

function makeCase(kind, src, cfg) {
  const target = makeTarget(device, cfg.canvasW, cfg.canvasH);
  const encodeFrame = (e, tsw) => {
    if (kind === 'old') {
      oldR.encode(e, src.view, target.view, { srcW: src.w, srcH: src.h, rect: cfg.rect, s: S, upscale: cfg.upscale, tsw });
    } else {
      newP.encode(e, src.view, target.view, { srcW: src.w, srcH: src.h, rect: cfg.rect, outW: cfg.outW, outH: cfg.outH, s: S,
        mode: cfg.mode, sr: cfg.mode === 'sr' ? cfg.sr : null, tsw });
    }
  };
  return {
    samples: [],
    async round() {
      const e = device.createCommandEncoder();
      for (let i = 0; i < K; i++) {
        encodeFrame(e, i === 0 ? { querySet: qs, begin: 0 } : i === K - 1 ? { querySet: qs, end: 1 } : null);
      }
      this.samples.push((await readSpan(e)) / K);
    },
    done() { target.tex.destroy(); }
  };
}

const D = { w: 2880, h: 1620 };
const cases = [];
const sources = [];
for (const [srcKey, label] of [['pump_1920x1080', '1920x1080'], ['halo_with_1280x720', '1280x720']]) {
  const fr = await loadFrames(srcKey);
  const src = makeSourceTexture(device, fr.w, fr.h);
  uploadFrame(device, src, fr.data[0]);
  sources.push(src);
  const defs = [
    ['old HDR-only (canvas 2880x1620)', 'old', { canvasW: D.w, canvasH: D.h, rect: [0, 0, 1, 1], upscale: false }],
    [`new HDR-only (canvas ${fr.w}x${fr.h}, compositor scales)`, 'new', { canvasW: fr.w, canvasH: fr.h, rect: [0, 0, 1, 1], outW: fr.w, outH: fr.h, mode: 'direct' }],
    ['new HDR-only forced @2880x1620 (info)', 'new', { canvasW: D.w, canvasH: D.h, rect: [0, 0, 1, 1], outW: D.w, outH: D.h, mode: 'direct' }],
    ['old upscale restore+EASU @2880x1620', 'old', { canvasW: D.w, canvasH: D.h, rect: [0, 0, 1, 1], upscale: true }],
    ['new upscale spatial EASU @2880x1620', 'new', { canvasW: D.w, canvasH: D.h, rect: [0, 0, 1, 1], outW: D.w, outH: D.h, mode: 'spatial' }],
    ['new upscale SR-stub x1 + resample @2880x1620', 'new', { canvasW: D.w, canvasH: D.h, rect: [0, 0, 1, 1], outW: D.w, outH: D.h, mode: 'sr', sr: [srStub] }]
  ];
  for (const [name, kind, cfg] of defs) cases.push({ label, name, c: makeCase(kind, src, cfg) });
}
for (const cs of cases) { await cs.c.round(); cs.c.samples.length = 0; } // warm-up
for (let r = 0; r < R; r++) for (const cs of cases) await cs.c.round();

const rows = [];
const res = {};
for (const cs of cases) {
  const s = sorted(cs.c.samples);
  const r = { min: s[0], p25: s[Math.floor(s.length / 4)], med: s[Math.floor(s.length / 2)] };
  res[`${cs.label}|${cs.name}`] = r;
  rows.push({ source: cs.label, path: cs.name, 'GPU ms min': fmt(r.min), 'p25': fmt(r.p25), 'median': fmt(r.med) });
  cs.c.done();
}
table(rows, ['source', 'path', 'GPU ms min', 'p25', 'median']);
for (const label of ['1920x1080', '1280x720']) {
  const o = res[`${label}|old HDR-only (canvas 2880x1620)`];
  const n = res[Object.keys(res).find((k) => k.startsWith(`${label}|new HDR-only (canvas`))];
  console.log(`${label} -> 2880x1620 HDR-only: min ${fmt(o.min)} -> ${fmt(n.min)} ms (${fmt(o.min / n.min, 2)}x), p25 ${fmt(o.p25 / n.p25, 2)}x, median ${fmt(o.med / n.med, 2)}x`);
}
for (const src of sources) src.tex.destroy();
srStub.destroy();
