// Runs the SR runtime inside real WebKit (WKWebView host, dev/sr/webkit/run.swift) -- the same WGSL
// compiler and WebGPU/Metal backend as Safari -- for correctness vs the PyTorch reference and bench.
//   deno run -A dev/sr/webkit_run.ts test  [--images a,b] [--configs '[{...}]']
//   deno run -A dev/sr/webkit_run.ts bench [--models a,b] [--sizes 1280x720,1920x1080] [--prec f16,f32] [--frames 10] [--rounds 4] [--configs '[..]']
// Bench variants (configs) are measured round-robin; reported numbers are per-pass minima (other GPU
// clients on the machine inflate medians).
// Test cases reuse the reference cache of test_runtime.ts (dev/sr/out/ref); missing ones are computed.
import { OUT, PY, readPng, ROOT, f16ToF32, f32ToF16 } from './gpu_common.ts';

const mode = Deno.args[0] ?? 'test';
const opt = new Map<string, string>();
for (let i = 1; i < Deno.args.length; i += 2) opt.set(Deno.args[i].replace(/^--/, ''), Deno.args[i + 1]);
const MODELS_PATH = ROOT + 'extension/sr-models.js';
const TOL_F16 = 2 / 255, TOL_F32 = 1e-4;

// ---- build the Swift host if needed
const bin = OUT + 'wkrun';
const src = ROOT + 'dev/sr/webkit/run.swift';
const needBuild = await Deno.stat(bin).then((b) => Deno.statSync(src).mtime! > b.mtime!).catch(() => true);
if (needBuild) {
  const r = await new Deno.Command('swiftc', { args: ['-O', src, '-o', bin] }).output();
  if (!r.success) throw new Error('swiftc failed: ' + new TextDecoder().decode(r.stderr));
}

// ---- test cases (input f16-exact RGB + reference output)
const cases: { id: string; image: string; w: number; h: number; input: string; ref: string }[] = [];
const files = new Map<string, Uint8Array<ArrayBuffer>>();
if (mode === 'test') {
  const node = await new Deno.Command('node', {
    args: ['-e', "const vm=require('vm'),fs=require('fs');const c=vm.createContext({});vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),c);process.stdout.write(JSON.stringify(c.__TRUEHDR_SR_MODELS__))", MODELS_PATH],
    stdout: 'piped',
  }).output();
  // deno-lint-ignore no-explicit-any
  const entries = new Map<string, any>(JSON.parse(new TextDecoder().decode(node.stdout)).map((m: any) => [m.id, m]));
  const images = (opt.get('images') ?? 'testsrc2_203x117,dahlia_256').split(',');
  for (const img of images) {
    const { w, h, rgb } = await readPng(`${OUT}testdata/${img}.png`);
    const exact = new Float32Array(rgb.length);
    for (let i = 0; i < rgb.length; i++) exact[i] = f16ToF32(f32ToF16(rgb[i]));
    for (const [id, e] of entries) {
      const enc = new TextEncoder();
      const parts = [enc.encode(JSON.stringify({ base: e.base, layers: e.layers })), enc.encode(`${w}x${h}`), new Uint8Array(exact.buffer)];
      const all = new Uint8Array(parts.reduce((a, b) => a + b.length, 0));
      let o = 0;
      for (const p of parts) { all.set(p, o); o += p.length; }
      const key = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', all)).slice(0, 12), (x) => x.toString(16).padStart(2, '0')).join('');
      const refPath = `${OUT}ref/${key}.out.f32`;
      if (!(await Deno.stat(refPath).catch(() => null))) {
        await Deno.mkdir(OUT + 'ref', { recursive: true });
        const inPath = `${OUT}ref/${key}.in.f32`;
        await Deno.writeFile(inPath, new Uint8Array(exact.buffer));
        const r = await new Deno.Command(PY, { args: ['-I', ROOT + 'dev/sr/reference.py', '--models', MODELS_PATH, '--model', id, '--in-raw', inPath, '--w', String(w), '--h', String(h), '--out-raw', refPath] }).output();
        await Deno.remove(inPath);
        if (!r.success) throw new Error('reference failed ' + new TextDecoder().decode(r.stderr));
      }
      files.set(`/case/${key}.in`, new Uint8Array(exact.buffer));
      cases.push({ id, image: img, w, h, input: `/case/${key}.in`, ref: `/ref/${key}.out.f32` });
    }
  }
}

// ---- server
let failures = 0;
let finish: () => void;
const finished = new Promise<void>((r) => (finish = r));
const mime: Record<string, string> = { js: 'text/javascript', html: 'text/html', json: 'application/json' };
const server = Deno.serve({ hostname: '127.0.0.1', port: 0, onListen: () => {} }, async (req) => {
  const url = new URL(req.url);
  const p = decodeURIComponent(url.pathname);
  if (req.method === 'POST' && p === '/report') {
    const r = await req.json();
    print(r);
    return new Response('ok');
  }
  if (req.method === 'POST' && p === '/done') { finish(); return new Response('ok'); }
  if (p === '/cases.json') return Response.json(cases);
  if (files.has(p)) return new Response(files.get(p));
  if (p.startsWith('/ref/')) return new Response(await Deno.readFile(OUT + 'ref/' + p.slice(5)));
  if (p.startsWith('/extension/') || p.startsWith('/dev/sr/webkit/')) {
    try {
      const body = await Deno.readFile(ROOT + p.slice(1));
      return new Response(body, { headers: { 'content-type': mime[p.split('.').pop()!] ?? 'application/octet-stream', 'cache-control': 'no-store' } });
    } catch { return new Response('not found', { status: 404 }); }
  }
  return new Response('not found', { status: 404 });
});

// deno-lint-ignore no-explicit-any
function print(r: any) {
  if (r.type === 'info') console.log('# ' + r.msg);
  else if (r.type === 'error') { failures++; console.log('ERROR ' + r.msg); }
  else if (r.type === 'test') {
    if (r.error) { failures++; console.log(`${r.id} ${r.image} ERROR ${r.error}`); return; }
    const ok = (!r.f16 || (r.f16.max <= TOL_F16 && !r.f16.alphaBad)) && r.f32.max <= TOL_F32 && !r.f32.alphaBad && r.f32h.max <= TOL_F32;
    if (!ok) failures++;
    console.log(`${r.id.padEnd(18)} ${r.image.padEnd(17)} | f16 max ${r.f16 ? r.f16.max.toExponential(2) + ` (${(r.f16.max * 255).toFixed(2)}/255)` : 'n/a'} | f32 max ${r.f32.max.toExponential(2)} | f32->rgba16f ${r.f32h.max.toExponential(2)} ${JSON.stringify(r.config)}${ok ? '' : '  <-- FAIL'}`);
  } else if (r.type === 'bench') {
    if (r.error) { console.log(`${r.id} ${r.prec} ${r.size} ${JSON.stringify(r.config)} ERROR ${r.error}`); return; }
    console.log(`${r.id.padEnd(18)} ${r.prec} ${r.size.padEnd(9)} gpu min ${Number(r.gpuMinMs).toFixed(2).padStart(6)} med ${Number(r.gpuMs).toFixed(2).padStart(6)} ms  sum(min passes) ${r.passMs.reduce((a: number, b: number) => a + b, 0).toFixed(2)}  wall ${r.wallMs.toFixed(2).padStart(6)} ms  create ${r.createMs.toFixed(0).padStart(5)} ms  cfg ${JSON.stringify(r.config)}\n    min passes: ${r.passMs.map((x: number) => x.toFixed(2)).join(' ')}`);
  }
}

const qs = new URLSearchParams({ mode });
for (const k of ['models', 'sizes', 'prec', 'frames', 'rounds', 'configs']) if (opt.has(k)) qs.set(k, opt.get(k)!);
const pageUrl = `http://127.0.0.1:${server.addr.port}/dev/sr/webkit/harness.html?${qs}`;
const child = new Deno.Command(bin, { args: [pageUrl], stdout: 'inherit', stderr: 'inherit' }).spawn();
const timeoutMs = Number(opt.get('timeout') ?? 900000);
const timer = setTimeout(() => { console.log('ERROR timeout'); failures++; finish(); }, timeoutMs);
await Promise.race([finished, child.status.then((s) => { if (!s.success) { console.log('ERROR host exited ' + s.code); failures++; } })]);
clearTimeout(timer);
try { child.kill('SIGTERM'); } catch { /* already exited */ }
await server.shutdown();
if (mode === 'test') console.log(failures ? `FAILED (${failures})` : 'PASSED');
Deno.exit(failures ? 1 : 0);
