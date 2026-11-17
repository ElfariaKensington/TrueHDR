// SR harness page logic (runs in WebKit via dev/sr/webkit_run.ts, or any browser opened on the URL).
// Query: ?mode=test|bench&models=a,b&sizes=1280x720&prec=f16,f32&frames=40&configs=<json list>
// Reports lines to the driver with POST /report, then POST /done.
(function () {
  'use strict';
  var q = new URLSearchParams(location.search);
  function report(obj) {
    return fetch('/report', { method: 'POST', body: JSON.stringify(obj) }).catch(function () {});
  }
  function done() { return fetch('/done', { method: 'POST', body: '{}' }); }
  window.addEventListener('error', function (e) { report({ type: 'error', msg: String(e.message || e) }).then(done); });
  window.addEventListener('unhandledrejection', function (e) { report({ type: 'error', msg: String(e.reason && (e.reason.stack || e.reason.message) || e.reason) }).then(done); });

  var f32b = new Float32Array(1), u32b = new Uint32Array(f32b.buffer);
  function f32ToF16(v) {
    f32b[0] = v;
    var x = u32b[0], sign = (x >>> 16) & 0x8000, e = (x >>> 23) & 0xff, m = x & 0x7fffff;
    if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
    var ee = e - 127 + 15;
    if (ee >= 31) return sign | 0x7c00;
    if (ee <= 0) {
      if (ee < -10) return sign;
      var mm = m | 0x800000, shift = 14 - ee, r0 = mm >>> shift, rem0 = mm & ((1 << shift) - 1), half = 1 << (shift - 1);
      if (rem0 > half || (rem0 === half && (r0 & 1))) r0++;
      return sign | r0;
    }
    var r = (ee << 10) | (m >>> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (r & 1))) r++;
    return sign | r;
  }
  function f16ToF32(h) {
    var s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * m * Math.pow(2, -24);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  function f16Ulp(r) { var a = Math.abs(r); return a < Math.pow(2, -14) ? Math.pow(2, -24) : Math.pow(2, Math.floor(Math.log2(a)) - 10); }

  function upload(device, w, h, rgb) {
    var half = new Uint16Array(w * h * 4);
    for (var i = 0; i < w * h; i++) {
      for (var c = 0; c < 3; c++) half[i * 4 + c] = f32ToF16(rgb[i * 3 + c]);
      half[i * 4 + 3] = 0x3c00;
    }
    var tex = device.createTexture({ size: { width: w, height: h }, format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: tex }, half, { bytesPerRow: w * 8 }, { width: w, height: h });
    return tex;
  }

  async function readTex(device, tex, w, h, fmt) {
    var bpp = fmt === 'rgba32float' ? 16 : 8, bpr = Math.ceil(w * bpp / 256) * 256;
    var buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    var enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, { width: w, height: h });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    var src = new Uint8Array(buf.getMappedRange()), out = new Float32Array(w * h * 4);
    for (var y = 0; y < h; y++) {
      var row = src.slice(y * bpr, y * bpr + w * bpp);
      if (bpp === 16) out.set(new Float32Array(row.buffer), y * w * 4);
      else { var u = new Uint16Array(row.buffer); for (var i = 0; i < w * 4; i++) out[y * w * 4 + i] = f16ToF32(u[i]); }
    }
    buf.unmap(); buf.destroy();
    return out;
  }

  function compare(gpu, ref, n, slack) {
    var max = 0, sum = 0, alphaBad = 0;
    for (var i = 0; i < n; i++) {
      for (var c = 0; c < 3; c++) {
        var d = Math.abs(gpu[i * 4 + c] - ref[i * 3 + c]);
        if (!isFinite(gpu[i * 4 + c])) d = Infinity;
        if (slack) d = Math.max(0, d - f16Ulp(ref[i * 3 + c]));
        if (d > max) max = d;
        sum += d;
      }
      if (gpu[i * 4 + 3] !== 1) alphaBad++;
    }
    return { max: max, mean: sum / (3 * n), alphaBad: alphaBad };
  }

  async function runOnce(SR, device, id, f16, fmt, tex, w, h, config) {
    var up = await SR.create(device, { modelId: id, useF16: f16, outputFormat: fmt, config: config });
    var enc = device.createCommandEncoder();
    up.encode(enc, tex.createView(), w, h);
    device.queue.submit([enc.finish()]);
    var out = await readTex(device, up._outputTexture(), 2 * w, 2 * h, fmt);
    up.destroy();
    return out;
  }

  function pattern(w, h) {
    var rgb = new Float32Array(w * h * 3);
    for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) {
      var i = (y * w + x) * 3;
      rgb[i] = 0.5 + 0.5 * Math.sin(x * 0.07 + y * 0.013);
      rgb[i + 1] = ((x ^ y) & 31) / 31;
      rgb[i + 2] = (x * y % 255) / 255;
    }
    return rgb;
  }
  function median(a) { var b = a.slice().sort(function (x, y) { return x - y; }); return b[b.length >> 1]; }
  function min(a) { return Math.min.apply(null, a); }

  // Interleaved benchmark: all variants are created first, then measured round-robin so that
  // time-varying GPU load from other processes affects them equally; per-pass minimum is reported.
  async function prepare(SR, device, id, f16, w, h, config) {
    var t0 = performance.now();
    var up = await SR.create(device, { modelId: id, useF16: f16, config: config });
    var createMs = performance.now() - t0;
    var tex = upload(device, w, h, pattern(w, h)), view = tex.createView();
    var e0 = device.createCommandEncoder(); up.encode(e0, view, w, h); device.queue.submit([e0.finish()]);
    await device.queue.onSubmittedWorkDone();
    var n = 2 * up._passCount;
    var v = { up: up, tex: tex, view: view, w: w, h: h, createMs: createMs, totals: [], per: [], wall: [] };
    for (var p = 0; p < up._passCount; p++) v.per.push([]);
    if (device.features.has('timestamp-query')) {
      v.qs = device.createQuerySet({ type: 'timestamp', count: n });
      v.res = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      v.rd = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    return v;
  }
  async function measure(device, v, frames) {
    var tw = performance.now();
    for (var f = 0; f < frames; f++) { var e1 = device.createCommandEncoder(); v.up.encode(e1, v.view, v.w, v.h); device.queue.submit([e1.finish()]); }
    await device.queue.onSubmittedWorkDone();
    v.wall.push((performance.now() - tw) / frames);
    if (!v.qs) return;
    var n = 2 * v.up._passCount;
    for (var k = 0; k < frames; k++) {
      var e2 = device.createCommandEncoder();
      v.up._encodeTimed(e2, v.view, v.w, v.h, v.qs);
      e2.resolveQuerySet(v.qs, 0, n, v.res, 0);
      e2.copyBufferToBuffer(v.res, 0, v.rd, 0, n * 8);
      device.queue.submit([e2.finish()]);
      await v.rd.mapAsync(GPUMapMode.READ);
      var t = new BigInt64Array(v.rd.getMappedRange().slice(0));
      v.rd.unmap();
      v.totals.push(Number(t[n - 1] - t[0]) / 1e6);
      for (var p = 0; p < v.up._passCount; p++) v.per[p].push(Number(t[2 * p + 1] - t[2 * p]) / 1e6);
    }
  }
  function summary(v) {
    var r = { createMs: v.createMs, wallMs: min(v.wall), gpuMs: v.totals.length ? median(v.totals) : NaN,
      gpuMinMs: v.totals.length ? min(v.totals) : NaN, passMs: v.per.map(function (a) { return a.length ? min(a) : NaN; }) };
    v.tex.destroy(); v.up.destroy();
    if (v.qs) { v.qs.destroy(); v.res.destroy(); v.rd.destroy(); }
    return r;
  }

  async function main() {
    var SR = globalThis.__TRUEHDR_SR__;
    if (!navigator.gpu) { await report({ type: 'error', msg: 'navigator.gpu missing (WebGPU disabled?) UA=' + navigator.userAgent }); return done(); }
    var adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    var want = ['shader-f16', 'timestamp-query'].filter(function (f) { return adapter.features.has(f); });
    var device = await adapter.requestDevice({ requiredFeatures: want });
    device.addEventListener('uncapturederror', function (e) { report({ type: 'error', msg: 'uncaptured: ' + e.error.message }); });
    await report({ type: 'info', msg: 'UA ' + navigator.userAgent + ' | features ' + want.join(',') + ' | maxWGStorage ' + device.limits.maxComputeWorkgroupStorageSize });
    var mode = q.get('mode') || 'test';
    var configs = JSON.parse(q.get('configs') || '[{}]');
    if (mode === 'test') {
      var cases = await (await fetch('/cases.json')).json();
      for (var ci = 0; ci < cases.length; ci++) {
        var c = cases[ci];
        var inp = new Float32Array(await (await fetch(c.input)).arrayBuffer());
        var ref = new Float32Array(await (await fetch(c.ref)).arrayBuffer());
        var tex = upload(device, c.w, c.h, inp), n = 4 * c.w * c.h;
        for (var k = 0; k < configs.length; k++) {
          var r = { type: 'test', id: c.id, image: c.image, config: configs[k] };
          try {
            if (device.features.has('shader-f16')) r.f16 = compare(await runOnce(SR, device, c.id, true, 'rgba16float', tex, c.w, c.h, configs[k]), ref, n, false);
            r.f32 = compare(await runOnce(SR, device, c.id, false, 'rgba32float', tex, c.w, c.h, configs[k]), ref, n, false);
            r.f32h = compare(await runOnce(SR, device, c.id, false, 'rgba16float', tex, c.w, c.h, configs[k]), ref, n, true);
          } catch (e) { r.error = String(e && e.message || e); }
          await report(r);
        }
        tex.destroy();
      }
    } else {
      var models = (q.get('models') || 'tsr-live-fast,tsr-live-quality,tsr-anime-quality').split(',');
      var sizes = (q.get('sizes') || '1280x720,1920x1080').split(',').map(function (s) { return s.split('x').map(Number); });
      var precs = (q.get('prec') || 'f16,f32').split(',').filter(function (p) { return p === 'f32' || device.features.has('shader-f16'); });
      var frames = Number(q.get('frames') || 10), rounds = Number(q.get('rounds') || 4);
      for (var mi = 0; mi < models.length; mi++) for (var pi = 0; pi < precs.length; pi++) for (var si = 0; si < sizes.length; si++) {
        var vs = [];
        for (var cj = 0; cj < configs.length; cj++) {
          var rb = { type: 'bench', id: models[mi], prec: precs[pi], size: sizes[si].join('x'), config: configs[cj] };
          try { vs.push({ rb: rb, v: await prepare(SR, device, models[mi], precs[pi] === 'f16', sizes[si][0], sizes[si][1], configs[cj]) }); }
          catch (e) { rb.error = String(e && e.message || e); await report(rb); }
        }
        for (var rr = 0; rr < rounds; rr++) for (var vi = 0; vi < vs.length; vi++) await measure(device, vs[vi].v, frames);
        for (var vj = 0; vj < vs.length; vj++) { Object.assign(vs[vj].rb, summary(vs[vj].v)); await report(vs[vj].rb); }
      }
    }
    device.destroy();
    await done();
  }
  main();
})();
