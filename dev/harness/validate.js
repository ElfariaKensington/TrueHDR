// WGSL validation of every engine module (texture_2d substitute) + pipeline creation, and a
// compile attempt of the shipped texture_external sources (naga may not support it; reported).
//   deno run --unstable-webgpu -A dev/harness/validate.js
import { getDevice, loadNewCoreApi } from './lib.js';

const { device } = await getDevice();
const api = await loadNewCoreApi();
let failures = 0;

async function check(label, code) {
  device.pushErrorScope('validation');
  const mod = device.createShaderModule({ label, code });
  const info = await mod.getCompilationInfo();
  const err = await device.popErrorScope();
  const msgs = info.messages.filter((m) => m.type === 'error' || m.type === 'warning');
  const ok = !err && !msgs.some((m) => m.type === 'error');
  console.log(`${ok ? 'OK  ' : 'FAIL'} ${label}${msgs.length ? '\n     ' + msgs.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`).join('\n     ') : ''}${err ? '\n     ' + err.message : ''}`);
  return ok;
}

console.log('--- texture_2d<f32> substitute (what the harness runs)');
for (const [name, code] of Object.entries(api.SHADER_SOURCES)) {
  if (!(await check(`${name} (2d)`, api.substituteTexture2D(code)))) failures++;
}

console.log('--- pipelines (explicit layouts, async creation)');
device.pushErrorScope('validation');
try {
  const core = await api.createCore(device, { texture2DSource: true });
  console.log('OK   pipelines:', Object.keys(core.pipes).join(', '));
} catch (e) {
  failures++;
  console.log('FAIL pipelines:', e.message);
}
const perr = await device.popErrorScope();
if (perr) { failures++; console.log('FAIL pipeline scope:', perr.message); }

console.log('--- shipped texture_external sources (informational: naga support varies)');
for (const [name, code] of Object.entries(api.SHADER_SOURCES)) {
  if (!code.includes('texture_external')) continue;
  device.pushErrorScope('validation');
  const info = await device.createShaderModule({ code }).getCompilationInfo();
  await device.popErrorScope();
  const errs = info.messages.filter((m) => m.type === 'error').map((m) => m.message);
  const capOnly = errs.length > 0 && errs.every((m) => m.includes('TEXTURE_EXTERNAL'));
  console.log(errs.length === 0 ? `OK   ${name} (external)`
    : capOnly ? `INFO ${name} (external): parses/resolves; this naga build lacks the TEXTURE_EXTERNAL capability`
    : `FAIL ${name} (external): ${errs.join(' | ')}`);
  if (errs.length && !capOnly) failures++;
}

// Substitution must touch only the declared type and the single loadVid() call
for (const [name, code] of Object.entries(api.SHADER_SOURCES)) {
  const a = code.split('\n');
  const b = api.substituteTexture2D(code).split('\n');
  const diff = a.filter((l, i) => l !== b[i]).map((l) => l.trim());
  if (diff.length && !diff.every((l) => l.startsWith('@group(0) @binding(0) var vidTex') || l.startsWith('fn loadVid'))) {
    failures++;
    console.log(`FAIL substitution in ${name} touched:`, diff);
  }
}

console.log('--- dev SR stub');
await import('./sr-stub.js');
try {
  const inst = await globalThis.__TRUEHDR_SR__.create(device, { modelId: 'tsr-live-quality', useF16: false });
  inst.destroy();
  console.log('OK   sr-stub pipeline');
} catch (e) {
  failures++;
  console.log('FAIL sr-stub', e.message);
}

console.log(failures ? `\n${failures} FAILURE(S)` : '\nALL OK');
Deno.exit(failures ? 1 : 0);
