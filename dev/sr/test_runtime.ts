// WGSL runtime vs PyTorch reference (docs/SR_SPEC.md verification standard).
//   ./dev/sr/make_testdata.sh            # once: test images into dev/sr/out/testdata
//   deno test --unstable-webgpu -A dev/sr/test_runtime.ts
// Env: SR_MODELS=path/to/sr-models.js (default extension/sr-models.js), SR_PYTHON=python3 with torch,
//      SR_IMAGES=comma list of testdata names, SR_PNG=0 to skip writing PNGs.
// For every model id x image: f16 path (rgba16float out) must be within 2/255 of the reference,
// f32 path within 1e-4 (checked on a rgba32float output; the rgba16float output is also checked
// against the reference with 1e-4 + one f16 ulp). Writes bicubic vs model PNGs to dev/sr/out/.
import { assert, bicubic2x, getDevice, loadRuntime, OUT, PY, readPng, readTexture, ROOT, uploadRgb, writePng } from './gpu_common.ts';

const MODELS_PATH = Deno.env.get('SR_MODELS') ?? ROOT + 'extension/sr-models.js';
const SR = loadRuntime(MODELS_PATH);
const device = await getDevice();
const hasF16 = device.features.has('shader-f16');
const IMAGES = (Deno.env.get('SR_IMAGES') ?? 'testsrc2_320x180,testsrc2_203x117,dahlia_256,sonoma_640x360').split(',');
const WRITE_PNG = Deno.env.get('SR_PNG') !== '0';
const TOL_F16 = 2 / 255, TOL_F32 = 1e-4;
await Deno.mkdir(OUT + 'ref', { recursive: true });

// deno-lint-ignore no-explicit-any
const entries: Map<string, any> = new Map((globalThis as any).__TRUEHDR_SR_MODELS__.map((m: any) => [m.id, m]));

async function sha(parts: (string | Uint8Array)[]) {
  const enc = new TextEncoder();
  const bufs = parts.map((p) => (typeof p === 'string' ? enc.encode(p) : p));
  const all = new Uint8Array(bufs.reduce((a, b) => a + b.length, 0));
  let o = 0;
  for (const b of bufs) { all.set(b, o); o += b.length; }
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', all));
  return Array.from(d.slice(0, 12), (x) => x.toString(16).padStart(2, '0')).join('');
}

async function reference(id: string, w: number, h: number, exact: Float32Array): Promise<Float32Array> {
  const e = entries.get(id);
  const key = await sha([JSON.stringify({ base: e.base, layers: e.layers }), `${w}x${h}`, new Uint8Array(exact.buffer)]);
  const outPath = `${OUT}ref/${key}.out.f32`;
  try {
    return new Float32Array((await Deno.readFile(outPath)).buffer);
  } catch { /* not cached */ }
  const inPath = `${OUT}ref/${key}.in.f32`;
  await Deno.writeFile(inPath, new Uint8Array(exact.buffer));
  const r = await new Deno.Command(PY, {
    args: ['-I', ROOT + 'dev/sr/reference.py', '--models', MODELS_PATH, '--model', id, '--in-raw', inPath,
      '--w', String(w), '--h', String(h), '--out-raw', outPath],
  }).output();
  await Deno.remove(inPath);
  if (!r.success) throw new Error('reference failed: ' + new TextDecoder().decode(r.stderr));
  return new Float32Array((await Deno.readFile(outPath)).buffer);
}

async function runGpu(id: string, useF16: boolean, outputFormat: string, tex: GPUTexture, w: number, h: number) {
  const up = await SR.create(device, { modelId: id, useF16, outputFormat });
  const enc = device.createCommandEncoder();
  const r = up.encode(enc, tex.createView(), w, h);
  assert(r.width === 2 * w && r.height === 2 * h);
  device.queue.submit([enc.finish()]);
  // second frame through the cached bind groups must give the same result
  const enc2 = device.createCommandEncoder();
  up.encode(enc2, tex.createView(), w, h);
  device.queue.submit([enc2.finish()]);
  const out = await readTexture(device, up._outputTexture(), 2 * w, 2 * h, outputFormat);
  up.destroy();
  return out;
}

/** One float16 ulp at |r| (the rgba16float store may round either way). */
function f16Ulp(r: number) {
  const a = Math.abs(r);
  if (a < 2 ** -14) return 2 ** -24;
  return 2 ** (Math.floor(Math.log2(a)) - 10);
}

function compare(gpu: Float32Array, ref: Float32Array, n: number, f16Slack = false) {
  let max = 0, sum = 0, bad = 0, worst = -1;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const g = gpu[i * 4 + c], r = ref[i * 3 + c];
      let d = Math.abs(g - r);
      if (!Number.isFinite(g)) d = Infinity;
      if (f16Slack) d = Math.max(0, d - f16Ulp(r)); // remove output rounding (GPU may round toward zero)
      if (d > max) { max = d; worst = i; }
      sum += d;
    }
    if (gpu[i * 4 + 3] !== 1) bad++;
  }
  return { max, mean: sum / (n * 3), alphaBad: bad, worst };
}

const results: string[] = [];
const imgCache = new Map<string, Awaited<ReturnType<typeof readPng>>>();

for (const img of IMAGES) {
  for (const id of entries.keys()) {
    Deno.test({
      name: `${id} on ${img}`, sanitizeOps: false, sanitizeResources: false,
      fn: async () => {
        if (!imgCache.has(img)) imgCache.set(img, await readPng(`${OUT}testdata/${img}.png`));
        const { w, h, rgb } = imgCache.get(img)!;
        const { tex, exact } = uploadRgb(device, w, h, rgb);
        const ref = await reference(id, w, h, exact);
        const n = 4 * w * h;
        assert(ref.length === n * 3, 'reference size');
        const line: string[] = [`${id.padEnd(18)} ${img.padEnd(17)}`];
        let ok = true;
        if (hasF16) {
          const g16 = await runGpu(id, true, 'rgba16float', tex, w, h);
          const c = compare(g16, ref, n);
          line.push(`f16 max ${c.max.toExponential(2)} (${(c.max * 255).toFixed(2)}/255) mean ${c.mean.toExponential(1)}`);
          ok &&= c.max <= TOL_F16 && c.alphaBad === 0;
          if (WRITE_PNG) await writePng(`${OUT}${img}_${id}_f16.png`, 2 * w, 2 * h, g16, 4);
        }
        const g32 = await runGpu(id, false, 'rgba32float', tex, w, h);
        const c32 = compare(g32, ref, n);
        line.push(`f32 max ${c32.max.toExponential(2)} mean ${c32.mean.toExponential(1)}`);
        ok &&= c32.max <= TOL_F32 && c32.alphaBad === 0;
        const g32h = await runGpu(id, false, 'rgba16float', tex, w, h);
        const c32h = compare(g32h, ref, n, true);
        line.push(`f32->rgba16f max ${c32h.max.toExponential(2)}`);
        ok &&= c32h.max <= TOL_F32;
        if (WRITE_PNG) {
          await writePng(`${OUT}${img}_bicubic.png`, 2 * w, 2 * h, bicubic2x(w, h, exact));
        }
        tex.destroy();
        results.push(line.join(' | ') + (ok ? '' : '  <-- FAIL'));
        console.log(results[results.length - 1]);
        assert(ok, line.join(' | '));
      },
    });
  }
}

Deno.test({
  name: 'API surface', sanitizeOps: false, sanitizeResources: false,
  fn: async () => {
    const list = SR.listModels();
    for (const id of ['tsr-live-fast', 'tsr-live-quality', 'tsr-anime-fast', 'tsr-anime-quality']) {
      assert(list.some((m: { id: string }) => m.id === id), 'missing ' + id);
    }
    let threw = false;
    try { await SR.create(device, { modelId: 'nope', useF16: false }); } catch { threw = true; }
    assert(threw, 'unknown model must throw');
    // size change reallocates; output view stays usable
    const up = await SR.create(device, { modelId: 'tsr-live-fast', useF16: hasF16 });
    for (const [w, h] of [[64, 36], [65, 37], [64, 36]]) {
      const { tex } = uploadRgb(device, w, h, new Float32Array(w * h * 3).fill(0.5));
      const enc = device.createCommandEncoder();
      const r = up.encode(enc, tex.createView(), w, h);
      device.queue.submit([enc.finish()]);
      assert(r.width === 2 * w && r.height === 2 * h);
      const o = await readTexture(device, up._outputTexture(), 2 * w, 2 * h, 'rgba16float');
      // flat grey in -> flat grey out (model-specific DC gain is tiny)
      for (let i = 0; i < o.length; i += 4) assert(Math.abs(o[i] - 0.5) < 0.02, `flat field ${o[i]}`);
      tex.destroy();
    }
    up.destroy();
  },
});
