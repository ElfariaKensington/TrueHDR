// GPU benchmark for the SR runtime (Deno WebGPU -> wgpu -> Metal).
//   deno run --unstable-webgpu -A dev/sr/bench.ts [--models a,b] [--sizes 1280x720,1920x1080]
//        [--prec f16,f32] [--frames 60] [--config '{"px":2,"py":2}'] [--configs '[{..},{..}]'] [--sweep] [--passes]
// Reports per model/precision/size: create() time, GPU ms/frame from pass timestamps (first pass
// begin -> last pass end, median over frames) and wall-clock ms/frame for back-to-back submits.
import { getDevice, loadRuntime, uploadRgb } from './gpu_common.ts';

const args = new Map<string, string>();
for (let i = 0; i < Deno.args.length; i++) {
  const a = Deno.args[i];
  if (a.startsWith('--')) args.set(a.slice(2), Deno.args[i + 1] && !Deno.args[i + 1].startsWith('--') ? Deno.args[++i] : '1');
}
const SR = loadRuntime(args.get('models-file'));
const device = await getDevice();
const hasTs = device.features.has('timestamp-query');
const models = (args.get('models') ?? 'tsr-live-fast,tsr-live-quality,tsr-anime-quality').split(',');
const sizes = (args.get('sizes') ?? '1280x720,1920x1080').split(',').map((s) => s.split('x').map(Number));
const precs = (args.get('prec') ?? 'f16,f32').split(',').filter((p) => p === 'f32' || device.features.has('shader-f16'));
const FRAMES = Number(args.get('frames') ?? 60);
const baseCfg = args.has('config') ? JSON.parse(args.get('config')!) : {};
const showPasses = args.has('passes');

function pattern(w: number, h: number) {
  const rgb = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 3;
      rgb[i] = 0.5 + 0.5 * Math.sin(x * 0.07 + y * 0.013);
      rgb[i + 1] = ((x ^ y) & 31) / 31;
      rgb[i + 2] = (x * y % 255) / 255;
    }
  }
  return rgb;
}

const median = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1];
const min = (a: number[]) => Math.min(...a);

async function bench(id: string, f16: boolean, w: number, h: number, config: Record<string, unknown>) {
  const t0 = performance.now();
  const up = await SR.create(device, { modelId: id, useF16: f16, config });
  const createMs = performance.now() - t0;
  const { tex } = uploadRgb(device, w, h, pattern(w, h));
  const view = tex.createView();
  // warmup (also allocates intermediates)
  for (let i = 0; i < 5; i++) {
    const e = device.createCommandEncoder();
    up.encode(e, view, w, h);
    device.queue.submit([e.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  // wall clock, back-to-back
  const tw = performance.now();
  for (let i = 0; i < FRAMES; i++) {
    const e = device.createCommandEncoder();
    up.encode(e, view, w, h);
    device.queue.submit([e.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const wallMs = (performance.now() - tw) / FRAMES;
  // timestamps
  let gpuMs = NaN, gpuMinMs = NaN;
  const passMs: number[] = [];
  if (hasTs) {
    const n = 2 * up._passCount;
    const qs = device.createQuerySet({ type: 'timestamp', count: n });
    const resolve = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const totals: number[] = [];
    const per: number[][] = Array.from({ length: up._passCount }, () => []);
    for (let f = 0; f < FRAMES; f++) {
      const e = device.createCommandEncoder();
      up._encodeTimed(e, view, w, h, qs);
      e.resolveQuerySet(qs, 0, n, resolve, 0);
      e.copyBufferToBuffer(resolve, 0, read, 0, n * 8);
      device.queue.submit([e.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const t = new BigInt64Array(read.getMappedRange().slice(0));
      read.unmap();
      totals.push(Number(t[n - 1] - t[0]) / 1e6);
      for (let p = 0; p < up._passCount; p++) per[p].push(Number(t[2 * p + 1] - t[2 * p]) / 1e6);
    }
    gpuMs = median(totals);
    gpuMinMs = min(totals);
    for (const p of per) passMs.push(min(p));
    qs.destroy();
    resolve.destroy();
    read.destroy();
  }
  tex.destroy();
  const kernels = up._kernels.map((k: { tile: number[]; smemBytes: number }) => `${k.tile.join('x')}`).join(',');
  up.destroy();
  return { createMs, wallMs, gpuMs, gpuMinMs, passMs, kernels };
}

const rows: string[] = [];
const configs: Record<string, unknown>[] = [];
if (args.has('sweep')) {
  for (const [wx, wy] of [[8, 8], [16, 8], [16, 16], [32, 4], [8, 16]]) {
    for (const [px, py] of [[1, 1], [2, 1], [1, 2], [2, 2], [4, 1], [4, 2]]) {
      if (wx * wy > 256) continue;
      configs.push({ ...baseCfg, wx, wy, px, py });
    }
  }
} else if (args.has('configs')) {
  for (const c of JSON.parse(args.get('configs')!)) configs.push({ ...baseCfg, ...c });
} else configs.push(baseCfg);

console.log(`device features: ${[...device.features].filter((f) => f === 'shader-f16' || f === 'timestamp-query').join(',')}; frames=${FRAMES}`);
for (const id of models) {
  for (const prec of precs) {
    for (const [w, h] of sizes) {
      for (const cfg of configs) {
        try {
          const r = await bench(id, prec === 'f16', w, h, cfg);
          const line = `${id.padEnd(18)} ${prec} ${`${w}x${h}`.padEnd(9)} gpu min ${r.gpuMinMs.toFixed(2).padStart(6)} med ${r.gpuMs.toFixed(2).padStart(6)} ms  sum(min passes) ${r.passMs.reduce((a, b) => a + b, 0).toFixed(2)}  wall ${r.wallMs.toFixed(2).padStart(6)} ms  create ${r.createMs.toFixed(0).padStart(5)} ms  tiles ${r.kernels}  cfg ${JSON.stringify(cfg)}` +
            (showPasses ? `\n    min passes: ${r.passMs.map((x) => x.toFixed(2)).join(' ')}` : '');
          rows.push(line);
          console.log(line);
        } catch (e) {
          console.log(`${id} ${prec} ${w}x${h} ${JSON.stringify(cfg)} ERROR ${String(e).slice(0, 300)}`);
        }
      }
    }
  }
}
