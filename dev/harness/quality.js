// Image-quality checks for the HDR path (baseline git rev vs current engine), HDR-only mode at
// source resolution. Pass criteria from the rework brief:
//   halo   : |gain on the sky ring around each small bird - gain without birds| <= 2 %
//   pump   : global gain variation on an untouched reference box while a flock crosses <= 1 %
//   move   : background gain variation around a slowly moving bird <= 1 %
// Informational: speck boost (small highlights must not be capped), caption peak vs text cap,
// white-page gain (APL guard), scene-cut adaptation.
//   deno run --unstable-webgpu -A dev/harness/quality.js [--rev=HEAD] [--preset=vivid]
import { getDevice, loadFrames, makeSourceTexture, uploadFrame, makeTarget, readTarget, outLuma, inLuma,
  NewPipeline, OldRenderer, PRESETS, table, fmt } from './lib.js';

const args = Object.fromEntries(Deno.args.map((a) => a.replace(/^--/, '').split('=')));
const rev = args.rev || 'HEAD';
const S = { ...PRESETS[args.preset || 'vivid'], upscaleSharpness: 0.25 };

const { device } = await getDevice();

// Caption box + lower-third metrics for a text still: peak, core-gain spread, rim inversion.
function textMetrics(r, a, tag) {
  const [x0, y0, x1, y1] = a.fr.meta.caption;
  let pk = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) pk = Math.max(pk, a.L[y * a.fr.w + x]);
  r[`${tag} peak (x SDR white)`] = pk;
  // Glyph rendering: every full-white stroke core should get the same gain (spread = patches /
  // boxes), and no antialiased edge pixel may come out brighter than the cores (rim inversion).
  const core = [];
  let rim = 0;
  let edgeN = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = y * a.fr.w + x;
      if (a.Lin[i] >= 0.95) core.push(a.L[i]);
    }
  }
  core.sort((p, q) => p - q);
  const coreMed = core[core.length >> 1] || 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = y * a.fr.w + x;
      if (a.Lin[i] > 0.25 && a.Lin[i] < 0.8) { edgeN++; if (a.L[i] > coreMed * 1.001) rim++; }
    }
  }
  r[`${tag} core gain spread`] = core.length ? (core[core.length - 1] - core[0]) / coreMed : 0;
  r[`${tag} rim inversion frac`] = edgeN ? rim / edgeN : 0;
  const [lx0, ly0, lx1, ly1] = a.fr.meta.lowerThird;
  let lpk = 0;
  for (let y = ly0; y < ly1; y++) for (let x = lx0; x < lx1; x++) lpk = Math.max(lpk, a.L[y * a.fr.w + x]);
  r[`${tag} lower-third peak (x SDR white)`] = lpk;
}

async function makeRunner(kind) {
  const p = kind === 'old' ? await OldRenderer.create(device, rev) : await NewPipeline.create(device);
  return p;
}

// Render a frame sequence through a fresh pipeline; returns per-frame output luma (+ input luma).
async function run(kind, key, { keep = null } = {}) {
  const fr = await loadFrames(key);
  const pipe = await makeRunner(kind);
  const src = makeSourceTexture(device, fr.w, fr.h);
  const target = makeTarget(device, fr.w, fr.h);
  const out = [];
  for (let i = 0; i < fr.frames; i++) {
    uploadFrame(device, src, fr.data[i]);
    const e = device.createCommandEncoder();
    const p = { srcW: fr.w, srcH: fr.h, rect: [0, 0, 1, 1], outW: fr.w, outH: fr.h, s: S, mode: 'direct', upscale: false, dt: 1 / 30 };
    pipe.encode(e, src.view, target.view, p);
    device.queue.submit([e.finish()]);
    if (!keep || keep(i)) {
      const rgba = await readTarget(device, target);
      out[i] = { L: outLuma(rgba, fr.w, fr.h), Lin: inLuma(fr.data[i], fr.w, fr.h) };
    }
  }
  const state = kind === 'new' ? await pipe.readState() : null;
  src.tex.destroy();
  target.tex.destroy();
  return { fr, out, state };
}

// Single still frame rendered repeatedly so temporal filters settle; returns the last frame.
async function still(kind, key, reps = 12) {
  const fr = await loadFrames(key);
  const pipe = await makeRunner(kind);
  const src = makeSourceTexture(device, fr.w, fr.h);
  const target = makeTarget(device, fr.w, fr.h);
  uploadFrame(device, src, fr.data[0]);
  for (let i = 0; i < reps; i++) {
    const e = device.createCommandEncoder();
    pipe.encode(e, src.view, target.view, { srcW: fr.w, srcH: fr.h, rect: [0, 0, 1, 1], outW: fr.w, outH: fr.h, s: S, mode: 'direct', upscale: false, dt: 1 / 30 });
    device.queue.submit([e.finish()]);
  }
  const rgba = await readTarget(device, target);
  const state = kind === 'new' ? await pipe.readState() : null;
  src.tex.destroy();
  target.tex.destroy();
  return { fr, L: outLuma(rgba, fr.w, fr.h), Lin: inLuma(fr.data[0], fr.w, fr.h), state };
}

function meanIn(L, w, pred, x0, y0, x1, y1) {
  let s = 0, n = 0;
  for (let y = Math.max(0, y0); y < y1; y++) for (let x = Math.max(0, x0); x < x1; x++) {
    if (pred(x, y)) { s += L[y * w + x]; n++; }
  }
  return n ? s / n : NaN;
}

const results = {};
for (const kind of ['old', 'new']) {
  const r = (results[kind] = {});

  // ---- halo: ring around each bird, with vs without birds
  for (const res of ['1920x1080', '1280x720']) {
    const a = await still(kind, `halo_with_${res}`);
    const b = await still(kind, `halo_without_${res}`);
    const w = a.fr.w;
    let worst = 0;
    let worstDark = 0;
    for (const bd of a.fr.meta.birds) {
      // bird radius = size/2 (+1.5 px anti-aliasing); halo ring out to 24 px. Same background in
      // both frames, so compare per pixel: worst |out_with / out_without - 1| on the ring.
      const r0 = bd.size / 2 + 1.5, r1 = bd.size / 2 + 24;
      const ring = (x, y) => { const d = Math.hypot(x + 0.5 - bd.x, y + 0.5 - bd.y); return d >= r0 && d <= r1; };
      const box = [Math.floor(bd.x - r1), Math.floor(bd.y - r1), Math.ceil(bd.x + r1), Math.ceil(bd.y + r1)];
      for (let y = box[1]; y < box[3]; y++) for (let x = box[0]; x < box[2]; x++) {
        if (ring(x, y)) worst = Math.max(worst, Math.abs(a.L[y * w + x] / b.L[y * w + x] - 1));
      }
      const go = meanIn(b.L, w, ring, ...box) / meanIn(b.Lin, w, ring, ...box);
      // the bird itself: output vs input luma ratio relative to the sky gain (should stay ~ <= 1)
      const core = (x, y) => Math.hypot(x + 0.5 - bd.x, y + 0.5 - bd.y) <= Math.max(1, bd.size * 0.4);
      const gb = meanIn(a.L, w, core, ...box) / meanIn(a.Lin, w, core, ...box);
      worstDark = Math.max(worstDark, gb / go);
    }
    r[`halo ring overshoot ${res}`] = worst;
    r[`bird/sky gain ratio ${res}`] = worstDark;
  }

  // ---- pump: reference box gain over the flock sequence
  {
    const { fr, out } = await run(kind, 'pump_1920x1080');
    const { x0, y0, x1, y1 } = fr.meta.ref;
    const gains = [];
    for (let i = 5; i < fr.frames; i++) {
      const all = () => true;
      gains.push(meanIn(out[i].L, fr.w, all, x0, y0, x1, y1) / meanIn(out[i].Lin, fr.w, all, x0, y0, x1, y1));
    }
    const mean = gains.reduce((s, g) => s + g, 0) / gains.length;
    r['pump gain variation'] = (Math.max(...gains) - Math.min(...gains)) / mean;
  }

  // ---- move: background around a slowly moving bird
  {
    const { fr, out } = await run(kind, 'move_1920x1080');
    const track = fr.meta.map((m) => m.bird4);
    const far = (x, y) => track.every(([bx, by]) => Math.hypot(x + 0.5 - bx, y + 0.5 - by) > 14);
    const gains = [];
    for (let i = 3; i < fr.frames; i++) {
      gains.push(meanIn(out[i].L, fr.w, far, 720, 225, 840, 320) / meanIn(out[i].Lin, fr.w, far, 720, 225, 840, 320));
    }
    const mean = gains.reduce((s, g) => s + g, 0) / gains.length;
    r['move bg gain variation'] = (Math.max(...gains) - Math.min(...gains)) / mean;
  }

  // ---- specks: peak gain of small bright specks relative to the background gain
  for (const key of ['specks_sky_1920x1080', 'specks_night_1920x1080']) {
    const a = await still(kind, key);
    const w = a.fr.w;
    const sp = Array.isArray(a.fr.meta.specks) ? a.fr.meta.specks : Object.values(a.fr.meta.specks);
    let sum = 0;
    for (const s of sp) {
      const cx = Math.floor(s.x), cy = Math.floor(s.y);
      let pk = 0;
      for (let y = cy - 2; y <= cy + 2; y++) for (let x = cx - 2; x <= cx + 2; x++) pk = Math.max(pk, a.L[y * w + x]);
      sum += pk;
    }
    r[`speck peak (x SDR white) ${key.split('_')[1]}`] = sum / sp.length;
  }

  // ---- text: caption peak; white page gain
  {
    for (const [key, tag] of [['text_1920x1080', 'caption'], ['ui_1920x1080', 'ui']]) {
      const a = await still(kind, key);
      textMetrics(r, a, tag);
    }
    const wp = await still(kind, 'white_page_1920x1080');
    const all = () => true;
    r['white page gain'] = meanIn(wp.L, wp.fr.w, all, 0, 0, wp.fr.w, wp.fr.h) / meanIn(wp.Lin, wp.fr.w, all, 0, 0, wp.fr.w, wp.fr.h);
  }

  // ---- cut: frames to settle after night -> white cut (mean gain within 2 % of its final value)
  {
    const { fr, out } = await run(kind, 'cut_1920x1080');
    const all = () => true;
    const g = (i) => meanIn(out[i].L, fr.w, all, 0, 0, fr.w, fr.h) / meanIn(out[i].Lin, fr.w, all, 0, 0, fr.w, fr.h);
    const final = g(fr.frames - 1);
    let settle = 0;
    for (let i = fr.frames - 1; i >= fr.meta.cutAt; i--) if (Math.abs(g(i) / final - 1) > 0.02) { settle = i - fr.meta.cutAt + 1; break; }
    r['frames to settle after cut'] = settle;
  }
}

const rows = Object.keys(results.new).map((k) => ({ metric: k, old: fmt(results.old[k], 4), new: fmt(results.new[k], 4) }));
table(rows, ['metric', 'old', 'new']);
const n = results.new;
const checks = [
  ['halo <= 2%', Math.max(n['halo ring overshoot 1920x1080'], n['halo ring overshoot 1280x720']) <= 0.02],
  ['pump <= 1%', n['pump gain variation'] <= 0.01],
  ['move <= 1%', n['move bg gain variation'] <= 0.01]
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'} ${name}`);
if (checks.some(([, ok]) => !ok)) Deno.exitCode = 1;
