// Shared helpers for dev/sr tests and bench (Deno + wgpu on Metal).
// Loads extension/sr-models.js + extension/sr-runtime.js into the global scope like the page does.

export const ROOT = new URL('../../', import.meta.url).pathname;
export const OUT = ROOT + 'dev/sr/out/';
export const PY = Deno.env.get('SR_PYTHON') ?? '/Users/elfaria/.venv/bin/python3';

// deno-lint-ignore no-explicit-any
export type SR = any;

export function assert(cond: unknown, msg = 'assertion failed'): asserts cond {
  if (!cond) throw new Error(msg);
}

export function loadRuntime(modelsPath = ROOT + 'extension/sr-models.js'): SR {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  delete g.__TRUEHDR_SR_MODELS__;
  delete g.__TRUEHDR_SR__;
  (0, eval)(Deno.readTextFileSync(modelsPath));
  (0, eval)(Deno.readTextFileSync(ROOT + 'extension/sr-runtime.js'));
  return g.__TRUEHDR_SR__;
}

export async function getDevice(): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const want = (['shader-f16', 'timestamp-query'] as GPUFeatureName[]).filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredFeatures: want });
  device.addEventListener('uncapturederror', (e) => console.error('GPU error:', (e as GPUUncapturedErrorEvent).error.message));
  return device;
}

// ---- float16 helpers
const f32b = new Float32Array(1), u32b = new Uint32Array(f32b.buffer);
export function f32ToF16(v: number): number {
  f32b[0] = v;
  const x = u32b[0], sign = (x >>> 16) & 0x8000, e = (x >>> 23) & 0xff, m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  const ee = e - 127 + 15;
  if (ee >= 31) return sign | 0x7c00;
  if (ee <= 0) {
    if (ee < -10) return sign;
    const mm = m | 0x800000, shift = 14 - ee;
    let r = mm >>> shift;
    const rem = mm & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rem > half || (rem === half && (r & 1))) r++;
    return sign | r;
  }
  let r = (ee << 10) | (m >>> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (r & 1))) r++;
  return sign | r;
}
export function f16ToF32(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// ---- images (ffmpeg for PNG I/O)
export async function readPng(path: string): Promise<{ w: number; h: number; rgb: Float32Array }> {
  const probe = await new Deno.Command('ffprobe', {
    args: ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path],
    stdout: 'piped',
  }).output();
  const [w, h] = new TextDecoder().decode(probe.stdout).trim().split(',').map(Number);
  const out = await new Deno.Command('ffmpeg', {
    args: ['-v', 'error', '-i', path, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-'],
    stdout: 'piped',
  }).output();
  const u16 = new Uint16Array(out.stdout.buffer, out.stdout.byteOffset, w * h * 3);
  const rgb = new Float32Array(w * h * 3);
  for (let i = 0; i < rgb.length; i++) rgb[i] = u16[i] / 65535;
  return { w, h, rgb };
}

export async function writePng(path: string, w: number, h: number, rgb: Float32Array, stride = 3) {
  const b = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h; i++, j += stride) {
    for (let c = 0; c < 3; c++) b[i * 3 + c] = Math.max(0, Math.min(255, Math.round(rgb[j + c] * 255)));
  }
  const p = new Deno.Command('ffmpeg', {
    args: ['-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${w}x${h}`, '-i', '-', '-frames:v', '1', path],
    stdin: 'piped',
  }).spawn();
  const wr = p.stdin.getWriter();
  await wr.write(b);
  await wr.close();
  const st = await p.status;
  if (!st.success) throw new Error('ffmpeg write failed ' + path);
}

/** Upload RGB (rounded to f16) as an rgba16float texture; returns the texture and the exact f16 values. */
export function uploadRgb(device: GPUDevice, w: number, h: number, rgb: Float32Array) {
  const half = new Uint16Array(w * h * 4);
  const exact = new Float32Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) {
      const hb = f32ToF16(rgb[i * 3 + c]);
      half[i * 4 + c] = hb;
      exact[i * 3 + c] = f16ToF32(hb);
    }
    half[i * 4 + 3] = 0x3c00;
  }
  const tex = device.createTexture({
    size: { width: w, height: h }, format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture: tex }, half, { bytesPerRow: w * 8 }, { width: w, height: h });
  return { tex, exact };
}

/** Read an rgba16float / rgba32float texture back as Float32Array RGBA. */
export async function readTexture(device: GPUDevice, tex: GPUTexture, w: number, h: number, fmt: string) {
  const bpp = fmt === 'rgba32float' ? 16 : 8;
  const bytesPerRow = Math.ceil((w * bpp) / 256) * 256;
  const buf = device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, { width: w, height: h });
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buf.getMappedRange());
  const out = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const row = src.subarray(y * bytesPerRow, y * bytesPerRow + w * bpp);
    if (bpp === 16) out.set(new Float32Array(row.slice().buffer), y * w * 4);
    else {
      const u = new Uint16Array(row.slice().buffer);
      for (let i = 0; i < w * 4; i++) out[y * w * 4 + i] = f16ToF32(u[i]);
    }
  }
  buf.unmap();
  buf.destroy();
  return out;
}

/** Spec bicubic2x (PyTorch a=-0.75, align_corners=False, clamped) of RGB HWC, for visual comparisons. */
export function bicubic2x(w: number, h: number, rgb: Float32Array): Float32Array {
  const P = [[-0.03515625, 0.26171875, 0.87890625, -0.10546875], [-0.10546875, 0.87890625, 0.26171875, -0.03515625]];
  const W2 = 2 * w, H2 = 2 * h, tmp = new Float32Array(W2 * h * 3), out = new Float32Array(W2 * H2 * 3);
  const cl = (v: number, n: number) => Math.max(0, Math.min(n - 1, v));
  for (let y = 0; y < h; y++) {
    for (let ox = 0; ox < W2; ox++) {
      const x = ox >> 1, d = ox & 1, s0 = x + d - 2;
      for (let c = 0; c < 3; c++) {
        let a = 0;
        for (let t = 0; t < 4; t++) a += P[d][t] * rgb[(y * w + cl(s0 + t, w)) * 3 + c];
        tmp[(y * W2 + ox) * 3 + c] = a;
      }
    }
  }
  for (let oy = 0; oy < H2; oy++) {
    const y = oy >> 1, d = oy & 1, s0 = y + d - 2;
    for (let ox = 0; ox < W2; ox++) {
      for (let c = 0; c < 3; c++) {
        let a = 0;
        for (let t = 0; t < 4; t++) a += P[d][t] * tmp[(cl(s0 + t, h) * W2 + ox) * 3 + c];
        out[(oy * W2 + ox) * 3 + c] = a;
      }
    }
  }
  return out;
}
