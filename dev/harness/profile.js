// Per-pass GPU cost of the new pipeline (min over interleaved rounds).
//   deno run --unstable-webgpu -A dev/harness/profile.js
import { getDevice, loadFrames, makeSourceTexture, uploadFrame, makeTarget, NewPipeline, PRESETS, table, fmt } from './lib.js';

const { device } = await getDevice();
const newP = await NewPipeline.create(device);
const core = newP.core;
const r = newP.r;
const qs = device.createQuerySet({ type: 'timestamp', count: 2 });
const qResolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const qRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const K = 20;
async function span(fn) {
  const e = device.createCommandEncoder();
  fn(e);
  e.resolveQuerySet(qs, 0, 2, qResolve, 0);
  e.copyBufferToBuffer(qResolve, 0, qRead, 0, 16);
  device.queue.submit([e.finish()]);
  await qRead.mapAsync(GPUMapMode.READ);
  const a = new BigInt64Array(qRead.getMappedRange());
  const ms = Number(a[1] - a[0]) / 1e6;
  qRead.unmap();
  return ms / K;
}
const rows = [];
for (const key of ['pump_1920x1080', 'halo_with_1280x720']) {
  const fr = await loadFrames(key);
  const src = makeSourceTexture(device, fr.w, fr.h);
  uploadFrame(device, src, fr.data[0]);
  const target = makeTarget(device, fr.w, fr.h);
  const S = PRESETS.vivid;
  const e0 = device.createCommandEncoder();
  newP.encode(e0, src.view, target.view, { srcW: fr.w, srcH: fr.h, rect: [0, 0, 1, 1], outW: fr.w, outH: fr.h, s: S });
  device.queue.submit([e0.finish()]);
  const videoBG = device.createBindGroup({ layout: core.layouts.videoBGL, entries: [{ binding: 0, resource: src.view }] });
  const comp = (pipe, bgs, wg) => (e) => {
    const cp = e.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
    cp.setPipeline(pipe);
    bgs.forEach((b, i) => cp.setBindGroup(i, b));
    for (let i = 0; i < K; i++) cp.dispatchWorkgroups(...wg);
    cp.end();
  };
  const finalBG = r.finalBindGroup(core.dummyView, core.dummyView, 0);
  const tests = {
    analyze: comp(core.pipes.analyze, [videoBG, r.anaBG], [r.groupsW, r.groupsH]),
    stats: comp(core.pipes.stats, [r.statsBG], [1]),
    map: comp(core.pipes.map, [r.mapBGs[0]], [r.groupsW, r.groupsH]),
    final: (e) => {
      for (let i = 0; i < K; i++) {
        const tw = i === 0 ? { querySet: qs, beginningOfPassWriteIndex: 0 } : i === K - 1 ? { querySet: qs, endOfPassWriteIndex: 1 } : null;
        const rp = e.beginRenderPass({ colorAttachments: [{ view: target.view, loadOp: 'clear', storeOp: 'store' }], ...(tw ? { timestampWrites: tw } : {}) });
        rp.setPipeline(core.pipes.finalDirect); rp.setBindGroup(0, videoBG); rp.setBindGroup(1, finalBG); rp.draw(3); rp.end();
      }
    }
  };
  const samples = Object.fromEntries(Object.keys(tests).map((k) => [k, []]));
  for (let round = 0; round < 25; round++) for (const [k, fn] of Object.entries(tests)) samples[k].push(await span(fn));
  for (const [k, xs] of Object.entries(samples)) {
    xs.sort((a, b) => a - b);
    rows.push({ source: `${fr.w}x${fr.h}`, pass: k, 'min ms': fmt(xs[0]), 'median ms': fmt(xs[Math.floor(xs.length / 2)]) });
  }
}
table(rows, ['source', 'pass', 'min ms', 'median ms']);
