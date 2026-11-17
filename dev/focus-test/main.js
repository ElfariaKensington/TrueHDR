// EDR focus test. URL params:
//   reconfigure=1   re-run ctx.configure() + redraw all WebGPU canvases on blur/focus/visibilitychange
//   loop=raf|none   redraw WebGPU canvases every rAF (default raf) or only once
//   video=0 img=0   remove the HDR <video> / <img>
//   rootLimit=no-limit|constrained|standard   set dynamic-range-limit on :root
//   reconfigureEvery=<ms>   call ctx.configure() again + redraw periodically (works without page focus)
//   recreateEvery=<ms>      replace every WebGPU canvas with a fresh element + context periodically
//   fix=1|blur              the proposed hdr-engine.js workaround: re-configure 400 ms and 1500 ms after
//                           window blur (and, for fix=1, every 3 s while the document has no focus);
//                           fixDelays=<ms,ms> overrides the delays
(() => {
  'use strict';
  const P = new URLSearchParams(location.search);
  const OPT = {
    reconfigure: P.get('reconfigure') === '1',
    loop: P.get('loop') || 'raf',
    video: P.get('video') !== '0',
    img: P.get('img') !== '0',
    rootLimit: P.get('rootLimit') || '',
    reconfigureEvery: +(P.get('reconfigureEvery') || 0),
    recreateEvery: +(P.get('recreateEvery') || 0),
    fix: P.get('fix') || '',
  };
  const PATCH = 90; // CSS px per patch
  const t0 = performance.now();
  const pending = [];
  const logEl = document.getElementById('log');
  const statusEl = document.getElementById('status');
  const canPost = location.protocol.startsWith('http');

  function log(kind, data = {}) {
    const e = { t: +((performance.now() - t0) / 1000).toFixed(3), wall: Date.now(), kind, ...data };
    pending.push(e);
    if (kind !== 'state') {
      const line = `${e.t.toFixed(3)} ${kind} ${JSON.stringify(data)}`;
      console.log('[focus-test]', line);
      logEl.textContent = (line + '\n' + logEl.textContent).slice(0, 20000);
    }
  }
  function flush() {
    if (!canPost || !pending.length) return;
    const body = JSON.stringify(pending.splice(0));
    fetch('/focus-log', { method: 'POST', body, keepalive: true }).catch(() => {});
  }

  if (OPT.rootLimit) document.documentElement.style.setProperty('dynamic-range-limit', OPT.rootLimit);
  if (!OPT.video) document.getElementById('videoCell').remove();
  if (!OPT.img) document.getElementById('imgCell').remove();

  const mqDR = matchMedia('(dynamic-range: high)');
  const mqVDR = matchMedia('(video-dynamic-range: high)');
  const support = {
    webgpu: !!navigator.gpu,
    cssDynamicRangeLimit: CSS.supports('dynamic-range-limit', 'no-limit'),
    cssConstrained: CSS.supports('dynamic-range-limit', 'constrained'),
    rvfc: 'requestVideoFrameCallback' in HTMLVideoElement.prototype,
  };

  // ---------- WebGPU ----------
  const WGSL = /* wgsl */`
    struct U { n: f32, patchW: f32, pad0: f32, pad1: f32, v: array<vec4f, 8> };
    @group(0) @binding(0) var<uniform> u: U;
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
      var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
      return vec4f(p[i], 0.0, 1.0);
    }
    @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let idx = u32(clamp(floor(pos.x / u.patchW), 0.0, u.n - 1.0));
      let v = u.v[idx].x;
      return vec4f(v, v, v, 1.0);
    }`;
  const gpu = { device: null, pipeline: null, items: [] };

  function sizeCanvas(c, n) {
    const dpr = devicePixelRatio || 1;
    c.style.width = `${PATCH * n}px`;
    c.style.height = `${PATCH}px`;
    c.width = Math.round(PATCH * n * dpr);
    c.height = Math.round(PATCH * dpr);
  }
  function configure(it) {
    it.ctx.configure({ device: gpu.device, format: 'rgba16float', colorSpace: 'display-p3',
      toneMapping: { mode: 'extended' }, alphaMode: 'premultiplied' });
  }
  function drawGPU(it) {
    const tex = it.ctx.getCurrentTexture();
    const enc = gpu.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(gpu.pipeline);
    pass.setBindGroup(0, it.bind);
    pass.draw(3);
    pass.end();
    gpu.device.queue.submit([enc.finish()]);
  }
  async function initGPU() {
    if (!navigator.gpu) { log('webgpu', { ok: false }); return; }
    const adapter = await navigator.gpu.requestAdapter();
    gpu.device = await adapter.requestDevice();
    const module = gpu.device.createShaderModule({ code: WGSL });
    gpu.pipeline = gpu.device.createRenderPipeline({ layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' } });
    for (const c of document.querySelectorAll('canvas[id^="gpu"]')) {
      const values = c.dataset.values.split(',').map(Number);
      sizeCanvas(c, values.length);
      const ctx = c.getContext('webgpu');
      const buf = gpu.device.createBuffer({ size: 16 + 16 * 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const arr = new Float32Array(4 + 32);
      arr[0] = values.length; arr[1] = c.width / values.length;
      values.forEach((v, i) => { arr[4 + i * 4] = v; });
      gpu.device.queue.writeBuffer(buf, 0, arr);
      const bind = gpu.device.createBindGroup({ layout: gpu.pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
      const it = { c, ctx, bind, values };
      configure(it);
      drawGPU(it);
      gpu.items.push(it);
    }
    log('webgpu', { ok: true, canvases: gpu.items.length, configuration: gpu.items[0]?.ctx.getConfiguration?.() ? JSON.stringify(gpu.items[0].ctx.getConfiguration().toneMapping) : 'n/a' });
  }
  function recreateAll() {
    for (const it of gpu.items) {
      const nc = it.c.cloneNode(false);
      it.c.replaceWith(nc);
      it.c = nc;
      sizeCanvas(nc, it.values.length);
      it.ctx = nc.getContext('webgpu');
      configure(it);
      drawGPU(it);
    }
  }
  function reconfigureAll(reason) {
    for (const it of gpu.items) { try { configure(it); drawGPU(it); } catch (e) { log('reconfigure-error', { e: String(e) }); } }
    log('reconfigured', { reason });
  }

  // ---------- 2D canvas ----------
  function init2D() {
    const c = document.getElementById('c2d');
    const values = c.dataset.values.split(',').map(Number);
    sizeCanvas(c, values.length);
    let ctx = null, mode = '';
    for (const attrs of [
      { colorSpace: 'display-p3', colorType: 'float16', toneMapping: { mode: 'extended' } },
      { colorSpace: 'display-p3', colorType: 'float16' },
      { colorSpace: 'display-p3' },
    ]) {
      try { ctx = c.getContext('2d', attrs); if (ctx) { mode = JSON.stringify(attrs); break; } } catch (_) {}
    }
    const got = ctx.getContextAttributes ? ctx.getContextAttributes() : {};
    const pw = c.width / values.length;
    let how = 'fillStyle';
    try {
      const id = new ImageData(c.width, c.height, { colorSpace: 'display-p3', pixelFormat: 'rgba-float16' });
      if (id.data && id.data.constructor && id.data.constructor.name === 'Float16Array') {
        for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
          const v = values[Math.min(values.length - 1, Math.floor(x / pw))];
          const k = (y * c.width + x) * 4;
          id.data[k] = v; id.data[k + 1] = v; id.data[k + 2] = v; id.data[k + 3] = 1;
        }
        ctx.putImageData(id, 0, 0);
        how = 'putImageData(rgba-float16)';
      } else throw new Error('no float16 ImageData');
    } catch (_) {
      values.forEach((v, i) => { ctx.fillStyle = `color(display-p3 ${v} ${v} ${v})`; ctx.fillRect(i * pw, 0, pw, c.height); });
    }
    document.getElementById('c2dInfo').textContent = `2d ${got.colorType || '?'} ${how}`;
    log('2d', { requested: mode, got: JSON.stringify(got), how });
  }

  // ---------- rates ----------
  let rafCount = 0, rvfcCount = 0;
  function onRaf() {
    rafCount++;
    if (OPT.loop === 'raf') for (const it of gpu.items) drawGPU(it);
    requestAnimationFrame(onRaf);
  }
  const video = document.getElementById('hdrVideo');
  if (video && support.rvfc) {
    const onVF = () => { rvfcCount++; video.requestVideoFrameCallback(onVF); };
    video.requestVideoFrameCallback(onVF);
  }

  // ---------- geometry for external measurement ----------
  function probes() {
    const chromeTop = outerHeight - innerHeight; // Safari: toolbar + tab bar (no bottom bar)
    const ox = screenX, oy = screenY + chromeTop;
    const out = {};
    const add = (name, r) => { out[name] = [+(ox + r.left).toFixed(1), +(oy + r.top).toFixed(1), +r.width.toFixed(1), +r.height.toFixed(1)]; };
    const split = (el, prefix, values) => {
      const r = el.getBoundingClientRect(); const w = r.width / values.length;
      values.forEach((v, i) => add(`${prefix}@${v}`, { left: r.left + i * w, top: r.top, width: w, height: r.height }));
    };
    for (const it of gpu.items) split(it.c, it.c.id, it.values);
    const c2 = document.getElementById('c2d'); split(c2, 'c2d', c2.dataset.values.split(','));
    const img = document.getElementById('hdrImg'); if (img) split(img, 'img', [1, 2, 4]);
    if (video) split(video, 'video', ['203', '406', '812']);
    return out;
  }

  function state(reason) {
    return {
      reason,
      hasFocus: document.hasFocus(),
      visibility: document.visibilityState,
      dynamicRangeHigh: mqDR.matches,
      videoDynamicRangeHigh: mqVDR.matches,
      screen: { x: screenX, y: screenY, outerW: outerWidth, outerH: outerHeight, innerW: innerWidth, innerH: innerHeight, dpr: devicePixelRatio, availW: screen.availWidth, availH: screen.availHeight },
      videoPaused: video ? video.paused : null,
      opt: OPT,
    };
  }

  let lastSec = performance.now();
  setInterval(() => {
    const now = performance.now();
    const dt = (now - lastSec) / 1000; lastSec = now;
    const s = { ...state('tick'), rafPerSec: +(rafCount / dt).toFixed(1), rvfcPerSec: +(rvfcCount / dt).toFixed(1), probes: probes() };
    rafCount = 0; rvfcCount = 0;
    log('state', s);
    statusEl.textContent =
      `focus=${s.hasFocus} vis=${s.visibility} (dynamic-range:high)=${s.dynamicRangeHigh} (video-dynamic-range:high)=${s.videoDynamicRangeHigh}\n` +
      `rAF/s=${s.rafPerSec} rVFC/s=${s.rvfcPerSec} dpr=${s.screen.dpr} support=${JSON.stringify(support)} opt=${JSON.stringify(OPT)}`;
    flush();
  }, 1000);

  for (const [target, ev] of [[window, 'focus'], [window, 'blur'], [document, 'visibilitychange'], [window, 'pageshow'], [window, 'pagehide'], [window, 'resize']]) {
    target.addEventListener(ev, () => {
      log('event', { ev, ...state(ev) });
      if (OPT.reconfigure && (ev === 'focus' || ev === 'blur' || ev === 'visibilitychange')) reconfigureAll(ev);
      flush();
    });
  }
  mqDR.addEventListener('change', (e) => { log('mq-change', { query: 'dynamic-range: high', matches: e.matches }); flush(); });
  mqVDR.addEventListener('change', (e) => { log('mq-change', { query: 'video-dynamic-range: high', matches: e.matches }); flush(); });

  log('start', { ua: navigator.userAgent, support, opt: OPT, ...state('start') });
  init2D();
  // ---------- proposed engine workaround ----------
  // Safari clamps the WebGPU canvas content headroom to <= 1.6 while the app is inactive
  // (WebProcessPool::applyEDRSuppressionIfNeeded). configure() recreates the render buffers
  // without that headroom metadata, and WebKit only re-applies it on the next screen-properties
  // change (i.e. when Safari is re-activated), so a re-configure after the clamp lands undoes it.
  if (OPT.fix) {
    let timers = [];
    const refresh = (why) => { if (!document.hasFocus()) reconfigureAll(why); };
    window.addEventListener('blur', () => {
      timers.forEach(clearTimeout);
      timers = (P.get('fixDelays') || '400,1500').split(',').map(Number).map((ms) => setTimeout(() => refresh(`fix-blur+${ms}`), ms));
    });
    window.addEventListener('focus', () => { timers.forEach(clearTimeout); timers = []; });
    if (OPT.fix === '1') setInterval(() => refresh('fix-poll'), 3000);
  }

  initGPU().then(() => {
    requestAnimationFrame(onRaf);
    if (OPT.reconfigureEvery) setInterval(() => reconfigureAll('timer'), OPT.reconfigureEvery);
    if (OPT.recreateEvery) setInterval(() => { recreateAll(); log('recreated'); }, OPT.recreateEvery);
    // resizeAt=<ms>: change every WebGPU canvas backing size once (does a resize lose headroom?)
    const resizeAt = +(P.get('resizeAt') || 0);
    if (resizeAt) setTimeout(() => {
      for (const it of gpu.items) { it.c.width = it.c.width - 2; drawGPU(it); }
      log('resized');
    }, resizeAt);
    // dual=1|poll: proposed engine workaround. Canvas A is configured once and kept (full headroom
    // while active, WebKit clamps it to 1.6 while Safari is inactive). Twin canvas B sits on top
    // of A; when the page loses focus B is configured fresh (new swap-chain surfaces) and shown,
    // when focus returns A is shown again and B is unconfigured. dual=poll switches on
    // document.hasFocus() polling (no content focus needed), dual=1 on blur/focus events.
    if (P.get('dual')) {
      for (const it of gpu.items) {
        const t = it.c.cloneNode(false);
        t.id = it.c.id + 'Twin';
        it.c.parentElement.style.position = 'relative';
        Object.assign(t.style, { position: 'absolute', left: `${it.c.offsetLeft}px`, top: `${it.c.offsetTop}px`, opacity: '0', pointerEvents: 'none' });
        t.width = it.c.width; t.height = it.c.height;
        it.c.insertAdjacentElement('afterend', t);
        it.twin = { c: t, ctx: t.getContext('webgpu'), bind: it.bind, on: false };
      }
      const setMode = (twinOn, why) => {
        for (const it of gpu.items) {
          const tw = it.twin;
          if (twinOn === tw.on) continue;
          if (twinOn) { configure(tw); drawGPU(tw); tw.c.style.opacity = '1'; it.c.style.opacity = '0'; }
          else { it.c.style.opacity = '1'; drawGPU(it); tw.c.style.opacity = '0'; tw.ctx.unconfigure(); }
          tw.on = twinOn;
        }
        log('dual', { twinOn, why });
      };
      const origDraw = drawGPU;
      // draw the visible one(s) each rAF
      const loopTwin = () => { for (const it of gpu.items) if (it.twin.on) origDraw(it.twin); requestAnimationFrame(loopTwin); };
      requestAnimationFrame(loopTwin);
      if (P.get('dual') === 'poll') setInterval(() => setMode(!document.hasFocus(), 'poll'), 250);
      else {
        window.addEventListener('blur', () => setMode(true, 'blur'));
        window.addEventListener('focus', () => setMode(false, 'focus'));
      }
    }
  }).catch((e) => log('webgpu-error', { e: String(e) }));
  if (video) video.play().catch((e) => log('video-play-error', { e: String(e) }));
})();
