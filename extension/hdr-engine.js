(() => {
  const G = globalThis;
  if (G.__TRUEHDR_ENGINE_LOADED__) return;
  G.__TRUEHDR_ENGINE_LOADED__ = true;

  // ---------------------------------------------------------------------------
  // GPU core (DOM-free; also loaded by the dev/harness benchmarks under Deno)
  //
  // Per new video frame:
  //   compute: analyze  - one invocation per 8x8 cell: 16 bilinear taps (= exact 2x2 box means),
  //                       writes (mean linear luma, robust (median-of-medians) log2 brightness,
  //                       strong-edge density, native-HDR fraction) + per-workgroup partials
  //            stats    - 1 workgroup: area-averaged APL, 32-bin histogram, scene-cut test,
  //                       asymmetric temporal smoothing, effective peak -> 176 B state buffer,
  //                       and a 256-entry tone LUT over log2 brightness (global curve terms)
  //            map      - per cell: robust (median-of-9 + range-weighted) log2 background,
  //                       caption/text-region score, temporal smoothing with scene-cut reset
  //   render:  final    - one pass at the canvas resolution: source sample, edge-aware 2x2 slice
  //                       of the 1/8-res map, inverse tone mapping, Display-P3, dither
  //   upscale path adds: decode (external -> rgba16float R'G'B' at source res), SR 2x (x1/x2),
  //                       and the final pass resamples the SR result (or runs EASU+RCAS fallback).
  // ---------------------------------------------------------------------------

  const CELL = 8;
  const FRAME_U_FLOATS = 28; // 7 x vec4f
  const STATE_BYTES = 176;   // see struct State
  const PARTIAL_BYTES = 144; // see struct Partial

  const WGSL_COMMON = /* wgsl */ `
    fn srgbToLinear(c: vec3f) -> vec3f {
      let lo = c / 12.92;
      let hi = pow(max((c + 0.055) / 1.055, vec3f(0.0)), vec3f(2.4));
      return select(hi, lo, c <= vec3f(0.04045));
    }

    // Decoded R'G'B' -> linear light (continuous sRGB transfer, NaN/Inf sanitized).
    fn decodeLinear(raw: vec3f) -> vec3f {
      let safe = select(vec3f(0.0), raw, (raw >= vec3f(0.0)) & (raw <= vec3f(64.0)));
      return srgbToLinear(safe);
    }

    fn luma709(c: vec3f) -> f32 {
      return dot(max(c, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
    }

    struct FrameU {
      rect: vec4f,   // content rect in canvas uv (ox, oy, w, h)
      tone: vec4f,   // peakBoost, threshold, localPunch, gamutBoost
      extra: vec4f,  // shadowDepth, aplGuard, splitPos (<0 = off), flags (1 = HDR, 2 = bypass native HDR)
      info: vec4f,   // srcW, srcH, frameSeed, reset
      res: vec4f,    // resample texture W, H, sharpness, EDR headroom cap (0 = none)
      outp: vec4f,   // content output W, H (canvas px), map uv scale x, y
      stat: vec4f,   // APL alpha up, APL alpha down, map alpha (also text attack), text alpha release
    };

    // First vec4 is copied to the final pass each frame (SceneU)
    struct State {
      apl: f32, hdrFrac: f32, effPeak: f32, textCap: f32,
      aplRaw: f32, cut: f32, histDist: f32, frames: f32,
      p50: f32, p98: f32, pad0: f32, pad1: f32,
      hist: array<f32, 32>,
    };

    // Per analysis workgroup (8x8 cells): sum of cell mean luma, native-HDR fraction, cell count,
    // and a 32-bin histogram of cell mean luma (perceptual axis)
    struct Partial {
      sumY: f32, hdr: f32, n: f32, pad: f32,
      hist: array<u32, 32>,
    };
  `;

  // The only texture_external user code. Harness builds substitute texture_2d<f32> and add the
  // mip argument to the single textureLoad below; everything else is identical.
  const WGSL_VIDEO = /* wgsl */ `
    @group(0) @binding(0) var vidTex: texture_external;
    fn loadVid(p: vec2i) -> vec4f { return textureLoad(vidTex, p); }
  `;

  const FULLSCREEN_VS = /* wgsl */ `
    struct VSOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vsMain(@builtin(vertex_index) idx: u32) -> VSOut {
      var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
      );
      let p = positions[idx];
      var out: VSOut;
      out.pos = vec4f(p, 0.0, 1.0);
      out.uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
      return out;
    }
  `;

  // Analysis: one invocation per 8x8-pixel cell. Each of the 16 taps is a bilinear sample at the
  // shared corner of a 2x2 pixel quad, i.e. an exact 2x2 box average, so every source pixel is
  // covered once at 1/4 of the fetches. Area statistics therefore move continuously as small
  // objects move (no point-sample jumps). Workgroup partial sums feed the global stats pass.
  // ANALYZE_LOAD_CS is a textureLoad-only variant, used only if a WebGPU implementation rejects
  // textureSampleBaseClampToEdge on texture_external in compute.
  function analyzeSource(sampled) {
    const tap = sampled
      ? 'textureSampleBaseClampToEdge(vidTex, samp, (base + o) * inv).rgb'
      : '0.25 * (loadVid(min(pq, mxp)).rgb + loadVid(min(pq + vec2i(1, 0), mxp)).rgb + loadVid(min(pq + vec2i(0, 1), mxp)).rgb + loadVid(min(pq + vec2i(1, 1), mxp)).rgb)';
    return /* wgsl */ `
    ${WGSL_COMMON}
    ${WGSL_VIDEO}
    @group(1) @binding(0) var cellOut: texture_storage_2d<rgba16float, write>;
    @group(1) @binding(1) var samp: sampler;
    @group(1) @binding(2) var<storage, read_write> partials: array<Partial>;

    var<workgroup> wAcc: array<atomic<u32>, 3>;  // fixed point (x1024): sum cell luma, sum HDR fraction; cell count
    var<workgroup> wHist: array<atomic<u32>, 32>;

    fn med4(a: f32, b: f32, c: f32, d: f32) -> f32 {
      return 0.5 * (a + b + c + d - min(min(a, b), min(c, d)) - max(max(a, b), max(c, d)));
    }

    @compute @workgroup_size(8, 8, 1)
    fn csAnalyze(
      @builtin(global_invocation_id) gid: vec3u,
      @builtin(workgroup_id) wid: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(local_invocation_index) li: u32
    ) {
      let dims = vec2i(textureDimensions(vidTex));
      let cells = (dims + vec2i(7)) / 8;
      let valid = i32(gid.x) < cells.x && i32(gid.y) < cells.y;
      if (li < 32u) { atomicStore(&wHist[li], 0u); }
      if (li < 3u) { atomicStore(&wAcc[li], 0u); }
      workgroupBarrier();

      if (valid) {
        let inv = 1.0 / vec2f(dims);
        let base = vec2f(gid.xy) * 8.0 + vec2f(1.0);
        let mxp = dims - vec2i(1);
        var q: array<f32, 16>;
        var pv: array<f32, 16>;
        var sumY = 0.0;
        var hdr = 0.0;
        for (var i = 0; i < 16; i = i + 1) {
          let o = vec2f(f32(i % 4), f32(i / 4)) * 2.0;
          let pq = vec2i(gid.xy) * 8 + vec2i(o);
          let raw0 = ${tap};
          let raw = select(vec3f(0.0), raw0, (raw0 >= vec3f(0.0)) & (raw0 <= vec3f(64.0)));
          let mx = max(raw.r, max(raw.g, raw.b));
          let lin = decodeLinear(raw);
          let Y = luma709(lin);
          sumY = sumY + Y;
          hdr = hdr + select(0.0, 1.0, mx > 1.05);
          q[i] = log2(max(mix(Y, max(lin.r, max(lin.g, lin.b)), 0.46), 1.0 / 16384.0));
          pv[i] = mix(luma709(raw), mx, 0.46);
        }
        // Strong perceptual transitions between neighbouring quads (24 pairs)
        var edges = 0.0;
        var pmin = pv[0];
        var pmax = pv[0];
        for (var r = 0; r < 4; r = r + 1) {
          for (var c = 0; c < 3; c = c + 1) {
            edges = edges + step(0.22, abs(pv[r * 4 + c] - pv[r * 4 + c + 1]));
            edges = edges + step(0.22, abs(pv[c * 4 + r] - pv[c * 4 + r + 4]));
          }
        }
        for (var i = 1; i < 16; i = i + 1) {
          pmin = min(pmin, pv[i]);
          pmax = max(pmax, pv[i]);
        }
        // UI text / captions: bright glyph fill on a flat dark backdrop. Nearly every tap sits at
        // the cell's min or max, the bright tone is the minority and the backdrop is dark. Natural
        // texture (foliage, rock, water) is a continuum, and branches against sky are a dark
        // minority on a bright majority; neither may count, or their highlights get the text cap.
        var nHi = 0.0;
        var nLo = 0.0;
        for (var i = 0; i < 16; i = i + 1) {
          nHi = nHi + select(0.0, 1.0, pv[i] > pmax - 0.08);
          nLo = nLo + select(0.0, 1.0, pv[i] < pmin + 0.08);
        }
        let flatTones = smoothstep(0.45, 0.70, (nHi + nLo) / 16.0) * smoothstep(0.30, 0.50, pmax - pmin)
          * smoothstep(0.55, 0.75, pmax) * (1.0 - smoothstep(0.20, 0.35, pmin))
          * smoothstep(0.25, 0.40, nLo / 16.0) * (1.0 - smoothstep(0.55, 0.70, nHi / 16.0));
        // Robust cell level: median of the four 4x4-pixel group medians (median-of-4 = mean of the
        // middle two). Rejects small objects (birds, specks) whatever their contrast.
        let g0 = med4(q[0], q[1], q[4], q[5]);
        let g1 = med4(q[2], q[3], q[6], q[7]);
        let g2 = med4(q[8], q[9], q[12], q[13]);
        let g3 = med4(q[10], q[11], q[14], q[15]);
        let cellY = sumY / 16.0;
        textureStore(cellOut, vec2i(gid.xy), vec4f(cellY, med4(g0, g1, g2, g3), edges / 24.0 * flatTones, hdr / 16.0));
        atomicAdd(&wHist[min(u32(pow(cellY, 1.0 / 2.2) * 32.0), 31u)], 1u);
        atomicAdd(&wAcc[0], u32(round(min(cellY, 16.0) * 1024.0)));
        atomicAdd(&wAcc[1], u32(round(hdr * 64.0)));
        atomicAdd(&wAcc[2], 1u);
      }
      workgroupBarrier();
      let wi = wid.y * nwg.x + wid.x;
      if (li == 0u) {
        partials[wi].sumY = f32(atomicLoad(&wAcc[0])) / 1024.0;
        partials[wi].hdr = f32(atomicLoad(&wAcc[1])) / 1024.0;
        partials[wi].n = f32(atomicLoad(&wAcc[2]));
      }
      if (li < 32u) { partials[wi].hist[li] = atomicLoad(&wHist[li]); }
    }
  `;
  }
  const ANALYZE_CS = analyzeSource(true);
  const ANALYZE_LOAD_CS = analyzeSource(false);

  // Global statistics: one workgroup reduces the analysis partials, then writes the per-frame
  // 256-entry tone LUT indexed by log2(E): (global EDR gain, highlight-weight, saturation term).
  const LUT_L0 = -8.0;
  const LUT_L1 = 0.5;
  const STATS_CS = /* wgsl */ `
    ${WGSL_COMMON}
    @group(0) @binding(0) var<storage, read> partials: array<Partial>;
    @group(0) @binding(1) var<storage, read_write> st: State;
    @group(0) @binding(2) var<uniform> u: FrameU;
    @group(0) @binding(3) var lutOut: texture_storage_2d<rgba16float, write>;

    var<workgroup> aSum: array<atomic<u32>, 4>;  // fixed point: sum luma, sum HDR, cell count, hist distance
    var<workgroup> aHist: array<atomic<u32>, 32>;
    var<workgroup> sPeak: f32;

    @compute @workgroup_size(256, 1, 1)
    fn csStats(@builtin(local_invocation_index) li: u32) {
      if (li < 32u) { atomicStore(&aHist[li], 0u); }
      if (li < 4u) { atomicStore(&aSum[li], 0u); }
      workgroupBarrier();

      let n = arrayLength(&partials);
      // Each partial carries <= 64 cells x 16 luma; keep the u32 sum below 2^32
      let scale = clamp(exp2(floor(log2(4.0e6 / f32(max(n, 1u))))), 16.0, 1024.0);
      var sy = 0.0;
      var sh = 0.0;
      var sn = 0.0;
      for (var i = li; i < n; i = i + 256u) {
        sy = sy + partials[i].sumY;
        sh = sh + partials[i].hdr;
        sn = sn + partials[i].n;
      }
      var hc = 0u;
      for (var i = li / 32u; i < n; i = i + 8u) {
        hc = hc + partials[i].hist[li % 32u];
      }
      atomicAdd(&aSum[0], u32(round(sy * scale)));
      atomicAdd(&aSum[1], u32(round(sh * scale)));
      atomicAdd(&aSum[2], u32(round(sn)));
      atomicAdd(&aHist[li % 32u], hc);
      workgroupBarrier();

      let cells = max(f32(atomicLoad(&aSum[2])), 1.0);
      if (li < 32u) {
        let h = f32(atomicLoad(&aHist[li])) / cells;
        atomicAdd(&aSum[3], u32(round(abs(h - st.hist[li]) * 65536.0)));
        st.hist[li] = h;
      }
      workgroupBarrier();

      if (li == 0u) {
        let aplRaw = f32(atomicLoad(&aSum[0])) / (scale * cells);
        let hdrRaw = f32(atomicLoad(&aSum[1])) / (scale * cells);
        let dist = f32(atomicLoad(&aSum[3])) / 65536.0;
        let cut = st.frames < 0.5 || u.info.w > 0.5 || dist > 0.6 || abs(aplRaw - st.aplRaw) > 0.15;
        // Brightening is tracked fast (white-screen guard engages quickly), darkening slowly.
        let a = select(u.stat.y, u.stat.x, aplRaw > st.apl);
        let apl = select(mix(st.apl, aplRaw, a), aplRaw, cut);
        st.apl = apl;
        st.hdrFrac = select(mix(st.hdrFrac, hdrRaw, u.stat.x), hdrRaw, cut);
        // White-screen APL guard -> effective peak; text cap for protected strokes
        let aplAtten = 1.0 - u.extra.y * smoothstep(0.45, 0.88, apl);
        let effPeak = 1.0 + (u.tone.x - 1.0) * clamp(aplAtten, 0.45, 1.0);
        st.effPeak = effPeak;
        st.textCap = max(1.65, effPeak * 0.32);
        st.aplRaw = aplRaw;
        st.cut = select(0.0, 1.0, cut);
        st.histDist = dist;
        var cum = 0.0;
        var p50 = 1.0;
        var p98 = 1.0;
        var got50 = false;
        var got98 = false;
        for (var b = 0u; b < 32u; b = b + 1u) {
          cum = cum + f32(atomicLoad(&aHist[b])) / cells;
          if (!got50 && cum >= 0.5) { p50 = (f32(b) + 1.0) / 32.0; got50 = true; }
          if (!got98 && cum >= 0.98) { p98 = (f32(b) + 1.0) / 32.0; got98 = true; }
        }
        st.p50 = p50;
        st.p98 = p98;
        st.frames = min(st.frames + 1.0, 1000000.0);
        sPeak = effPeak;
      }
      workgroupBarrier();

      // Tone LUT entry li at L = log2(E)
      let L = ${LUT_L0.toFixed(1)} + ${(LUT_L1 - LUT_L0).toFixed(1)} * (f32(li) + 0.5) / 256.0;
      let E = exp2(L);
      let Ep = exp2(L * (1.0 / 2.2));
      let knee = clamp(u.tone.y, 0.15, 0.88);
      var gain = 1.0;
      if (Ep > knee) {
        // EDR highlight curve above the perceptual knee
        let t = clamp((Ep - knee) / max(1.0 - knee, 0.05), 0.0, 1.0);
        let smoothRamp = t * t * (3.0 - 2.0 * t);
        let fastRamp = pow(t, 0.85);
        gain = 1.0 + (sPeak - 1.0) * mix(smoothRamp, fastRamp, 0.55);
      }
      textureStore(lutOut, vec2i(i32(li), 0), vec4f(gain, smoothstep(0.10, 0.62, E), u.tone.w * 0.35 * smoothstep(0.06, 0.70, E), 1.0));
    }
  `;

  // Local adaptation map at cell resolution (8x8 cells per workgroup, 20x12 shared tile).
  const MAP_CS = /* wgsl */ `
    ${WGSL_COMMON}
    @group(0) @binding(0) var cellTex: texture_2d<f32>;
    @group(0) @binding(1) var prevMap: texture_2d<f32>;
    @group(0) @binding(2) var mapOut: texture_storage_2d<rgba16float, write>;
    @group(0) @binding(3) var<storage, read> st: State;
    @group(0) @binding(4) var<uniform> u: FrameU;

    var<workgroup> tM: array<f32, 240>;  // 20x12 tile: 6-cell halo left/right, 2 top/bottom
    var<workgroup> tD: array<f32, 240>;

    fn cs(a: ptr<function, f32>, b: ptr<function, f32>) {
      let lo = min(*a, *b);
      *b = max(*a, *b);
      *a = lo;
    }

    // 19-comparator median-of-9 network
    fn median9(a0: f32, a1: f32, a2: f32, a3: f32, a4: f32, a5: f32, a6: f32, a7: f32, a8: f32) -> f32 {
      var p0 = a0; var p1 = a1; var p2 = a2; var p3 = a3; var p4 = a4;
      var p5 = a5; var p6 = a6; var p7 = a7; var p8 = a8;
      cs(&p1, &p2); cs(&p4, &p5); cs(&p7, &p8);
      cs(&p0, &p1); cs(&p3, &p4); cs(&p6, &p7);
      cs(&p1, &p2); cs(&p4, &p5); cs(&p7, &p8);
      cs(&p0, &p3); cs(&p5, &p8); cs(&p4, &p7);
      cs(&p3, &p6); cs(&p1, &p4); cs(&p2, &p5);
      cs(&p4, &p7); cs(&p4, &p2); cs(&p6, &p4);
      cs(&p4, &p2);
      return p4;
    }

    fn rw(m: f32, med: f32) -> f32 {
      let d = (m - med) / 0.40;
      return exp(-d * d);
    }

    @compute @workgroup_size(8, 8, 1)
    fn csMap(
      @builtin(workgroup_id) wid: vec3u,
      @builtin(local_invocation_id) lid: vec3u,
      @builtin(local_invocation_index) li: u32
    ) {
      let dims = vec2i(textureDimensions(cellTex));
      let mxc = dims - vec2i(1);
      let origin = vec2i(wid.xy) * 8 - vec2i(6, 2);
      for (var k = li; k < 240u; k = k + 64u) {
        let t = textureLoad(cellTex, clamp(origin + vec2i(i32(k % 20u), i32(k / 20u)), vec2i(0), mxc), 0);
        tM[k] = t.y;
        tD[k] = t.z;
      }
      workgroupBarrier();
      let c = vec2i(wid.xy) * 8 + vec2i(lid.xy);
      if (c.x >= dims.x || c.y >= dims.y) { return; }
      let tc = i32(lid.y + 2u) * 20 + i32(lid.x + 6u);

      let m0 = tM[tc - 21]; let m1 = tM[tc - 20]; let m2 = tM[tc - 19];
      let m3 = tM[tc - 1];  let m4 = tM[tc];      let m5 = tM[tc + 1];
      let m6 = tM[tc + 19]; let m7 = tM[tc + 20]; let m8 = tM[tc + 21];
      // Median of the 3x3 cell levels rejects objects spanning up to ~2x2 cells; then a
      // range-weighted mean around it keeps the map smooth.
      let med = median9(m0, m1, m2, m3, m4, m5, m6, m7, m8);
      let wc = rw(m0, med) * 0.5 + rw(m2, med) * 0.5 + rw(m6, med) * 0.5 + rw(m8, med) * 0.5;
      let we = rw(m1, med) * 0.7 + rw(m3, med) * 0.7 + rw(m5, med) * 0.7 + rw(m7, med) * 0.7;
      let w4 = rw(m4, med);
      let bsum = (rw(m0, med) * m0 + rw(m2, med) * m2 + rw(m6, med) * m6 + rw(m8, med) * m8) * 0.5
               + (rw(m1, med) * m1 + rw(m3, med) * m3 + rw(m5, med) * m5 + rw(m7, med) * m7) * 0.7
               + w4 * m4;
      // Use the brighter of the range-weighted median and 4-neighbor max so dark objects/text
      // never drag down the background reference and create white halos on adjacent light pixels.
      let cardHi = max(max(m1, m3), max(m5, m7));
      let B = max(bsum / max(wc + we + w4, 1e-6), mix(med, cardHi, 0.75));
      // Caption / UI text: dense strong edges spread over a 56x24 px window (7x3 cells).
      // An isolated speck or bird lights one or two cells and stays far below threshold.
      // Max over the window centred one cell up/here/down = 1-cell vertical dilation, so
      // ascenders/descenders poking out of a line's band stay inside the mask.
      // Text also forms lines: within one cell row, a 7-cell run (ending at, centred on or
      // starting at this cell) is mostly glyph cells. Scattered look-alikes (sky through foliage)
      // rarely fill a row run.
      var rows: array<f32, 5>;
      var line = 0.0;
      for (var dy = -2; dy <= 2; dy = dy + 1) {
        var rsum = 0.0;
        for (var dx = -3; dx <= 3; dx = dx + 1) {
          rsum = rsum + tD[tc + dy * 20 + dx];
        }
        rows[dy + 2] = rsum;
        if (dy >= -1 && dy <= 1) {
          var hit: array<f32, 13>;
          for (var dx = -6; dx <= 6; dx = dx + 1) {
            hit[dx + 6] = smoothstep(0.015, 0.04, tD[tc + dy * 20 + dx]);
          }
          var runL = 0.0;
          var runC = 0.0;
          var runR = 0.0;
          for (var j = 0; j < 7; j = j + 1) {
            runL = runL + hit[j];
            runC = runC + hit[j + 3];
            runR = runR + hit[j + 6];
          }
          line = max(line, max(runC, max(runL, runR)) / 7.0);
        }
      }
      let dens = max(rows[1] + rows[2] + rows[3], max(rows[0] + rows[1] + rows[2], rows[2] + rows[3] + rows[4]));
      let text = smoothstep(0.03, 0.08, dens / 21.0) * smoothstep(0.55, 0.80, line);
      let prevT = textureLoad(prevMap, c, 0).y;
      let aT = select(u.stat.w, u.stat.z, text > prevT);
      let outT = select(clamp(mix(prevT, text, aT), 0.0, 1.0), text, st.cut > 0.5);

      // Instant spatial background (zero temporal ghosting behind moving objects); only the
      // text score is smoothed (fast attack, short release) so captions never flicker.
      // z = plain 3x3 median: the backdrop behind bold glyphs that fill their own cells (B
      // then sits at glyph level, so such text would no longer read as "above background").
      textureStore(mapOut, c, vec4f(B, outT, med, 1.0));
    }
  `;

  // Upscale path: decode external texture once to nonlinear R'G'B' at source resolution.
  const DECODE_FS = /* wgsl */ `
    ${WGSL_VIDEO}
    @fragment
    fn fsDecode(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let p = vec2i(pos.xy);
      return vec4f(max(loadVid(p).rgb, vec3f(0.0)), 1.0);
    }
  `;

  const FINAL_FS = /* wgsl */ `
    ${WGSL_COMMON}
    ${WGSL_VIDEO}
    struct SceneU { v: vec4f };  // apl (smoothed), native-HDR fraction, effective peak, text cap

    @group(1) @binding(0) var samp: sampler;
    @group(1) @binding(1) var srcTex: texture_2d<f32>;  // decoded source (upscale path)
    @group(1) @binding(2) var resTex: texture_2d<f32>;  // SR result (ML path)
    @group(1) @binding(3) var mapTex: texture_2d<f32>;  // (log2 background, text region)
    @group(1) @binding(4) var<uniform> u: FrameU;
    @group(1) @binding(5) var<uniform> scene: SceneU;
    @group(1) @binding(6) var lutTex: texture_2d<f32>;  // per-frame tone LUT over log2(E)

    fn linearToSrgbExt(c: vec3f) -> vec3f {
      let lo = c * 12.92;
      let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
      return select(hi, lo, c <= vec3f(0.0031308));
    }

    fn srgbToP3(c: vec3f) -> vec3f {
      return vec3f(
        0.8224621 * c.r + 0.1775380 * c.g,
        0.0331941 * c.r + 0.9668058 * c.g,
        0.0170827 * c.r + 0.0723974 * c.g + 0.9105199 * c.b
      );
    }

    fn lumaP3(c: vec3f) -> f32 {
      return dot(max(c, vec3f(0.0)), vec3f(0.2289746, 0.6917385, 0.0792869));
    }

    fn ign(p: vec2f) -> f32 {
      let magic = vec3f(0.06711056, 0.00583715, 52.9829189);
      return fract(magic.z * fract(dot(p, magic.xy)));
    }

    // Untouched SDR (or native HDR) passthrough, colour-managed into the Display-P3 canvas
    fn sdrOut(raw: vec3f) -> vec4f {
      return vec4f(linearToSrgbExt(srgbToP3(decodeLinear(raw))), 1.0);
    }

    // Edge-aware slice of the 1/8-res map: 2x2 joint-bilateral weights pick the cells whose
    // background level matches this pixel, so large edges never bleed across (halo-free).
    fn sliceMap(vuv: vec2f, L: f32) -> vec3f {
      let muv = vuv * u.outp.zw;
      let g = textureGather(0, mapTex, samp, muv);
      let f = fract(muv * vec2f(textureDimensions(mapTex)) - vec2f(0.5));
      let bw = vec4f((1.0 - f.x) * f.y, f.x * f.y, f.x * (1.0 - f.y), (1.0 - f.x) * (1.0 - f.y));
      // Range weight: reject darker cells strongly so small dark silhouettes (birds) never
      // pull down the local background estimate of surrounding sky pixels.
      let d = vec4f(L) - g;
      let k = select(vec4f(0.85), vec4f(1.35), d > vec4f(0.0));
      let x = max(vec4f(1.0) - k * d * d, vec4f(0.0));
      let w = bw * (x * x + vec4f(1e-4));
      let B = dot(w, g) / max(dot(w, vec4f(1.0)), 1e-6);
      // Text score: max of the 2x2 footprint (no bilinear fall-off across glyph tops at a
      // mask edge); the backdrop median is plain bilinear.
      let gt = textureGather(1, mapTex, samp, muv);
      let t = textureSampleLevel(mapTex, samp, muv, 0.0).z;
      return vec3f(B, max(max(gt.x, gt.y), max(gt.z, gt.w)), t);
    }

    fn toneMapHDR(c0: vec3f, vuv: vec2f, fragPos: vec4f) -> vec4f {
      let p3Linear = srgbToP3(c0);
      let maxIn = max(c0.r, max(c0.g, c0.b));
      let E = mix(luma709(c0), maxIn, 0.46);
      let L = log2(max(E, 1.0 / 16384.0));
      // (global EDR gain over the knee, highlight weight, saturation term) for this brightness
      let lut = textureSampleLevel(lutTex, samp, vec2f((L - (${LUT_L0.toFixed(1)})) * ${(1 / (LUT_L1 - LUT_L0)).toFixed(8)}, 0.5), 0.0);

      // Pixel vs. local background (same map at source scale for every output path)
      let loc = sliceMap(vuv, L);
      let bg = exp2(loc.x);
      let detail = max(0.0, E - bg * 1.15);
      // Text protection only for glyph pixels well above the local backdrop inside text-dense
      // regions (dark text on a bright page stays below its backdrop and is left alone).
      let strokeLift = max(detail, E - exp2(loc.z) * 1.15);
      let textProt = clamp(loc.y, 0.0, 1.0) * smoothstep(0.08, 0.28, strokeLift);

      // Deep-black-only shadow floor (< 0.015 linear), kept gentle to avoid dark silhouette crush
      var shadowRatio = 1.0;
      if (maxIn < 0.015) {
        let blackToe = pow(max(maxIn, 1e-5), u.extra.x * 0.25);
        let blackMix = 1.0 - smoothstep(0.002, 0.015, maxIn);
        shadowRatio = mix(1.0, clamp(blackToe, 0.86, 1.0), blackMix);
      }

      // EDR highlight curve above the knee (white-screen guarded peak) from the LUT
      var hdrGain = lut.x;

      // Text: scale the EDR lift so the curve tops out at the text cap instead of the peak.
      // Monotone in E, so antialiased edges keep their shape (a hard min() cap flattened stroke
      // cores against their edges and drew grey rims/boxes around glyphs).
      let capK = clamp((scene.v.w - 1.0) / max(scene.v.z - 1.0, 1e-3), 0.0, 1.0);
      hdrGain = mix(hdrGain, 1.0 + (hdrGain - 1.0) * capK, textProt);

      // Local specular punch: only genuine bright local highlights above background
      let localMask = smoothstep(0.08, 0.32, detail) * lut.y * (1.0 - textProt);
      hdrGain = hdrGain * (1.0 + u.tone.z * 0.50 * localMask);

      var hdrP3 = p3Linear * (hdrGain * shadowRatio);
      let p3Y = lumaP3(hdrP3);
      let satScale = 1.0 + lut.z * (1.0 - textProt * 0.6);
      hdrP3 = max(vec3f(0.0), mix(vec3f(p3Y), hdrP3, satScale));

      // Background Safari: stock WebKit caps page EDR at u.res.w. Keep the focused curve up to a
      // knee and roll everything above it softly into the cap (scaled by the max channel, so hue
      // holds) instead of letting the compositor hard-clip highlights flat.
      let cap = u.res.w;
      if (cap > 1.0) {
        let knee = 1.0 + (cap - 1.0) * 0.35;
        let m = max(hdrP3.r, max(hdrP3.g, hdrP3.b));
        if (m > knee) {
          let room = cap - knee;
          hdrP3 = hdrP3 * ((knee + room * (1.0 - exp((knee - m) / room))) / m);
        }
      }

      var outP3 = linearToSrgbExt(hdrP3);
      let d1 = ign(fragPos.xy + vec2f(u.info.z * 17.0, 0.0));
      let d2 = ign(fragPos.xy + vec2f(0.0, u.info.z * 31.0));
      let dither = (d1 + d2 - 1.0) * (0.5 / 1023.0);
      outP3 = max(vec3f(0.0), outP3 + vec3f(dither));
      return vec4f(outP3, 1.0);
    }

    fn shade(raw: vec3f, vuv: vec2f, fragPos: vec4f) -> vec4f {
      let flags = u32(u.extra.w + 0.5);
      if ((flags & 1u) == 0u) {
        return sdrOut(raw);
      }
      let hdrCol = toneMapHDR(decodeLinear(raw), vuv, fragPos);
      // Frame-level smooth bypass for native HDR streams; never cut per-pixel holes at 1.03!
      if ((flags & 2u) != 0u && scene.v.y > 0.002) {
        let nativeW = smoothstep(0.002, 0.015, scene.v.y);
        return mix(hdrCol, sdrOut(raw), nativeW);
      }
      return hdrCol;
    }

    fn contentUV(uv: vec2f) -> vec2f {
      return (uv - u.rect.xy) / u.rect.zw;
    }

    fn outside(vuv: vec2f) -> bool {
      return any(vuv < vec2f(0.0)) || any(vuv > vec2f(1.0));
    }

    fn onDivider(uv: vec2f) -> bool {
      return u.extra.z >= 0.0 && abs(uv.x - u.extra.z) < 0.0016;
    }

    fn leftOfSplit(uv: vec2f) -> bool {
      return u.extra.z >= 0.0 && uv.x < u.extra.z;
    }

    // ---- HDR-only path: native/minification + optional RCAS clarity for native 4K+ ------------
    fn rcasTapRes(ip: vec2i, mxc: vec2i) -> vec3f {
      return textureLoad(resTex, clamp(ip, vec2i(0), mxc), 0).rgb;
    }

    fn rcasContrast(c: vec3f, n: vec3f, s: vec3f, e: vec3f, w: vec3f, sharp: f32) -> vec3f {
      if (sharp <= 0.001) { return c; }
      let sh = clamp(sharp, 0.0, 1.0);
      let mn = min(min(n, s), min(e, w));
      let mx = max(max(n, s), max(e, w));
      let mn5 = min(mn, c);
      let mx5 = max(mx, c);
      let hitMin = mn5 / max(4.0 * mx + vec3f(1e-4), vec3f(1e-4));
      let hitMax = (vec3f(1.0) - mx5) / max(4.0 * (vec3f(1.0) - mn) + vec3f(1e-4), vec3f(1e-4));
      let lobeV = min(hitMin, hitMax);
      let lobe = clamp(min(lobeV.r, min(lobeV.g, lobeV.b)), 0.0, 0.22);
      let peak = mix(0.08, 0.20, sh);
      let w0 = -min(lobe, peak) * sh;
      let sharpCol = (c + (n + s + e + w) * w0) / max(1.0 + 4.0 * w0, 1e-4);
      let ringTol = max(mx5 - mn5, vec3f(0.0)) * (0.06 * sh);
      return clamp(sharpCol, mn5 - ringTol, mx5 + ringTol);
    }

    fn easuSRTap(
      cSum: ptr<function, vec3f>,
      wSum: ptr<function, f32>,
      off: vec2f,
      dir: vec2f,
      len: vec2f,
      lob: f32,
      col: vec3f
    ) {
      let v = vec2f(dot(off, dir), dot(off, vec2f(-dir.y, dir.x))) * len;
      let d2 = min(dot(v, v), 4.0);
      let wB = 0.4 * d2 - 1.0;
      let wA = lob * d2 - 1.0;
      let w = (1.5625 * wB * wB - 0.5625) * (wA * wA);
      *cSum = *cSum + col * w;
      *wSum = *wSum + w;
    }

    fn easuSR(vuv: vec2f, fdims: vec2f) -> vec3f {
      let mxc = vec2i(fdims) - vec2i(1);
      let p = vuv * fdims - vec2f(0.5);
      let fp = floor(p);
      let f = p - fp;
      let ip = vec2i(fp);

      let b  = rcasTapRes(ip + vec2i( 0, -1), mxc);
      let c  = rcasTapRes(ip + vec2i( 1, -1), mxc);
      let e  = rcasTapRes(ip + vec2i(-1,  0), mxc);
      let f0 = rcasTapRes(ip + vec2i( 0,  0), mxc);
      let f1 = rcasTapRes(ip + vec2i( 1,  0), mxc);
      let h  = rcasTapRes(ip + vec2i( 2,  0), mxc);
      let i  = rcasTapRes(ip + vec2i(-1,  1), mxc);
      let j  = rcasTapRes(ip + vec2i( 0,  1), mxc);
      let k  = rcasTapRes(ip + vec2i( 1,  1), mxc);
      let l  = rcasTapRes(ip + vec2i( 2,  1), mxc);
      let n  = rcasTapRes(ip + vec2i( 0,  2), mxc);
      let o  = rcasTapRes(ip + vec2i( 1,  2), mxc);

      let lb = luma709(b); let lc = luma709(c);
      let le = luma709(e); let lf0 = luma709(f0); let lf1 = luma709(f1); let lh = luma709(h);
      let li = luma709(i); let lj = luma709(j);   let lk = luma709(k);   let ll = luma709(l);
      let ln = luma709(n); let lo = luma709(o);

      let g0 = vec2f(lf1 - le, lj - lb);
      let g1 = vec2f(lh - lf0, lk - lc);
      let g2 = vec2f(lk - li, ln - lf0);
      let g3 = vec2f(ll - lj, lo - lf1);

      let w0 = (1.0 - f.x) * (1.0 - f.y);
      let w1 = f.x * (1.0 - f.y);
      let w2 = (1.0 - f.x) * f.y;
      let w3 = f.x * f.y;

      let grad = g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3;
      let gLenSq = dot(grad, grad);
      let hasEdge = gLenSq > 1e-5;
      let normG = select(vec2f(1.0, 0.0), grad * inverseSqrt(max(gLenSq, 1e-5)), hasEdge);
      let dir = vec2f(-normG.y, normG.x);

      let edgeStrength = clamp(sqrt(gLenSq) * 2.4, 0.0, 1.0);
      let stretch = 1.0 + edgeStrength * 0.85;
      let len = vec2f(1.0 / stretch, 1.0 + edgeStrength * 0.40);
      let lob = mix(0.24, 0.36, edgeStrength);

      var cSum = vec3f(0.0);
      var wSum: f32 = 0.0;
      easuSRTap(&cSum, &wSum, vec2f( 0.0, -1.0) - f, dir, len, lob, b);
      easuSRTap(&cSum, &wSum, vec2f( 1.0, -1.0) - f, dir, len, lob, c);
      easuSRTap(&cSum, &wSum, vec2f(-1.0,  0.0) - f, dir, len, lob, e);
      easuSRTap(&cSum, &wSum, vec2f( 0.0,  0.0) - f, dir, len, lob, f0);
      easuSRTap(&cSum, &wSum, vec2f( 1.0,  0.0) - f, dir, len, lob, f1);
      easuSRTap(&cSum, &wSum, vec2f( 2.0,  0.0) - f, dir, len, lob, h);
      easuSRTap(&cSum, &wSum, vec2f(-1.0,  1.0) - f, dir, len, lob, i);
      easuSRTap(&cSum, &wSum, vec2f( 0.0,  1.0) - f, dir, len, lob, j);
      easuSRTap(&cSum, &wSum, vec2f( 1.0,  1.0) - f, dir, len, lob, k);
      easuSRTap(&cSum, &wSum, vec2f( 2.0,  1.0) - f, dir, len, lob, l);
      easuSRTap(&cSum, &wSum, vec2f( 0.0,  2.0) - f, dir, len, lob, n);
      easuSRTap(&cSum, &wSum, vec2f( 1.0,  2.0) - f, dir, len, lob, o);

      let min4 = min(min(f0, f1), min(j, k));
      let max4 = max(max(f0, f1), max(j, k));
      return clamp(cSum / max(wSum, 1e-4), min4, max4);
    }

    @fragment
    fn fsDirect(@builtin(position) fragPos: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
      let vuv = contentUV(uv);
      if (outside(vuv)) { return vec4f(0.0, 0.0, 0.0, 1.0); }
      if (onDivider(uv)) { return vec4f(1.8, 1.8, 1.8, 1.0); }
      let ratio = u.info.xy / max(u.outp.xy, vec2f(1.0));
      var raw: vec3f;
      if (max(ratio.x, ratio.y) > 1.4) {
        // Display smaller than the source: 4 bilinear taps = box-filtered minification
        let o = 0.25 * ratio / u.info.xy;
        raw = 0.25 * (textureSampleBaseClampToEdge(vidTex, samp, vuv + vec2f(-o.x, -o.y)).rgb
                    + textureSampleBaseClampToEdge(vidTex, samp, vuv + vec2f( o.x, -o.y)).rgb
                    + textureSampleBaseClampToEdge(vidTex, samp, vuv + vec2f(-o.x,  o.y)).rgb
                    + textureSampleBaseClampToEdge(vidTex, samp, vuv + vec2f( o.x,  o.y)).rgb);
      } else {
        raw = textureSampleBaseClampToEdge(vidTex, samp, vuv).rgb;
        let flags = u32(u.extra.w + 0.5);
        let sharp = clamp(u.res.z, 0.0, 1.0);
        if ((flags & 4u) != 0u && sharp > 0.01) {
          let t = 1.0 / max(u.info.xy, vec2f(1.0));
          let rn = textureSampleBaseClampToEdge(vidTex, samp, vuv + vec2f(0.0, -t.y)).rgb;
          let rs = textureSampleBaseClampToEdge(vidTex, samp, vuv + vec2f(0.0,  t.y)).rgb;
          let re = textureSampleBaseClampToEdge(vidTex, samp, vuv + vec2f( t.x, 0.0)).rgb;
          let rw = textureSampleBaseClampToEdge(vidTex, samp, vuv + vec2f(-t.x, 0.0)).rgb;
          raw = rcasContrast(raw, rn, rs, re, rw, sharp * 0.65);
        }
      }
      if (leftOfSplit(uv)) { return sdrOut(raw); }
      return shade(raw, vuv, fragPos);
    }

    // ---- Image path (<img>, GIF frames): source is a pre-downscaled premultiplied-alpha copy
    // at ~display resolution. Transparent areas stay transparent (premultiplied canvas).
    @fragment
    fn fsImage(@builtin(position) fragPos: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
      let vuv = contentUV(uv);
      if (outside(vuv)) { return vec4f(0.0); }
      if (onDivider(uv)) { return vec4f(1.8, 1.8, 1.8, 1.0); }
      let s = textureSampleBaseClampToEdge(vidTex, samp, vuv);
      let a = clamp(s.a, 0.0, 1.0);
      if (a < 0.002) { return vec4f(0.0); }
      let raw = s.rgb / a;
      var c: vec4f;
      if (leftOfSplit(uv)) { c = sdrOut(raw); } else { c = shade(raw, vuv, fragPos); }
      return vec4f(c.rgb * a, a);
    }

    // ---- ML path: resample the 2x/4x SR result to the exact device-pixel content rect --------
    fn crW(f: f32) -> vec4f {
      return vec4f(
        f * (-0.5 + f * (1.0 - 0.5 * f)),
        1.0 + f * f * (-2.5 + 1.5 * f),
        f * (0.5 + f * (2.0 - 1.5 * f)),
        f * f * (-0.5 + 0.5 * f)
      );
    }

    fn resampleSR(vuv: vec2f) -> vec3f {
      let R = u.res.xy;
      let s = R / max(u.outp.xy, vec2f(1.0));
      let sharp = clamp(u.res.z, 0.0, 1.0);
      let invR = 1.0 / max(R, vec2f(1.0));
      let stepR = clamp(s, vec2f(0.5), vec2f(1.35)) * invR;
      if (max(s.x, s.y) < 0.85) {
        // Large post-SR magnification (e.g. 480p/720p -> 4K display):
        // directional 12-tap EASU contour reconstruction + RCAS 4K detail pop.
        let base = easuSR(vuv, R);
        let n = textureSampleLevel(resTex, samp, vuv + vec2f(0.0, -stepR.y), 0.0).rgb;
        let sb = textureSampleLevel(resTex, samp, vuv + vec2f(0.0,  stepR.y), 0.0).rgb;
        let e = textureSampleLevel(resTex, samp, vuv + vec2f( stepR.x, 0.0), 0.0).rgb;
        let w = textureSampleLevel(resTex, samp, vuv + vec2f(-stepR.x, 0.0), 0.0).rgb;
        return rcasContrast(base, n, sb, e, w, sharp);
      }
      // Near-1:1 or downsampled SR output: hardware bilinear + 4-neighbor RCAS (5 taps vs 20)
      let base = textureSampleLevel(resTex, samp, vuv, 0.0).rgb;
      let n = textureSampleLevel(resTex, samp, vuv + vec2f(0.0, -stepR.y), 0.0).rgb;
      let sb = textureSampleLevel(resTex, samp, vuv + vec2f(0.0,  stepR.y), 0.0).rgb;
      let e = textureSampleLevel(resTex, samp, vuv + vec2f( stepR.x, 0.0), 0.0).rgb;
      let w = textureSampleLevel(resTex, samp, vuv + vec2f(-stepR.x, 0.0), 0.0).rgb;
      return rcasContrast(base, n, sb, e, w, sharp);
    }

    @fragment
    fn fsSR(@builtin(position) fragPos: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
      let vuv = contentUV(uv);
      if (outside(vuv)) { return vec4f(0.0, 0.0, 0.0, 1.0); }
      if (onDivider(uv)) { return vec4f(1.8, 1.8, 1.8, 1.0); }
      if (leftOfSplit(uv)) { return sdrOut(textureSampleLevel(srcTex, samp, vuv, 0.0).rgb); }
      return shade(max(resampleSR(vuv), vec3f(0.0)), vuv, fragPos);
    }

    // ---- Spatial fallback: 12-tap EASU + RCAS-style clamp sharpening from the decoded source --
    fn easuTap(
      cSum: ptr<function, vec3f>,
      wSum: ptr<function, f32>,
      off: vec2f,
      dir: vec2f,
      len: vec2f,
      lob: f32,
      col: vec3f
    ) {
      let v = vec2f(dot(off, dir), dot(off, vec2f(-dir.y, dir.x))) * len;
      let d2 = min(dot(v, v), 4.0);
      let wB = 0.4 * d2 - 1.0;
      let wA = lob * d2 - 1.0;
      let w = (1.5625 * wB * wB - 0.5625) * (wA * wA);
      *cSum = *cSum + col * w;
      *wSum = *wSum + w;
    }

    fn ld(coord: vec2i, mxc: vec2i) -> vec3f {
      return textureLoad(srcTex, clamp(coord, vec2i(0), mxc), 0).rgb;
    }

    fn easuSpatial(vuv: vec2f) -> vec3f {
      let fdims = vec2f(textureDimensions(srcTex));
      let mxc = vec2i(fdims) - vec2i(1);
      let p = vuv * fdims - vec2f(0.5);
      let fp = floor(p);
      let f = p - fp;
      let ip = vec2i(fp);

      let b  = ld(ip + vec2i( 0, -1), mxc);
      let c  = ld(ip + vec2i( 1, -1), mxc);
      let e  = ld(ip + vec2i(-1,  0), mxc);
      let f0 = ld(ip + vec2i( 0,  0), mxc);
      let f1 = ld(ip + vec2i( 1,  0), mxc);
      let h  = ld(ip + vec2i( 2,  0), mxc);
      let i  = ld(ip + vec2i(-1,  1), mxc);
      let j  = ld(ip + vec2i( 0,  1), mxc);
      let k  = ld(ip + vec2i( 1,  1), mxc);
      let l  = ld(ip + vec2i( 2,  1), mxc);
      let n  = ld(ip + vec2i( 0,  2), mxc);
      let o  = ld(ip + vec2i( 1,  2), mxc);

      let lb = luma709(b); let lc = luma709(c);
      let le = luma709(e); let lf0 = luma709(f0); let lf1 = luma709(f1); let lh = luma709(h);
      let li = luma709(i); let lj = luma709(j);   let lk = luma709(k);   let ll = luma709(l);
      let ln = luma709(n); let lo = luma709(o);

      let g0 = vec2f(lf1 - le, lj - lb);
      let g1 = vec2f(lh - lf0, lk - lc);
      let g2 = vec2f(lk - li, ln - lf0);
      let g3 = vec2f(ll - lj, lo - lf1);

      let w0 = (1.0 - f.x) * (1.0 - f.y);
      let w1 = f.x * (1.0 - f.y);
      let w2 = (1.0 - f.x) * f.y;
      let w3 = f.x * f.y;

      let grad = g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3;
      let gLenSq = dot(grad, grad);
      let hasEdge = gLenSq > 1e-5;
      let normG = select(vec2f(1.0, 0.0), grad * inverseSqrt(max(gLenSq, 1e-5)), hasEdge);
      let dir = vec2f(-normG.y, normG.x);

      let edgeStrength = clamp(sqrt(gLenSq) * 2.2, 0.0, 1.0);
      let stretch = 1.0 + edgeStrength * 0.75;
      let len = vec2f(1.0 / stretch, 1.0 + edgeStrength * 0.35);
      let lob = mix(0.24, 0.36, edgeStrength);

      var cSum = vec3f(0.0);
      var wSum: f32 = 0.0;
      easuTap(&cSum, &wSum, vec2f( 0.0, -1.0) - f, dir, len, lob, b);
      easuTap(&cSum, &wSum, vec2f( 1.0, -1.0) - f, dir, len, lob, c);
      easuTap(&cSum, &wSum, vec2f(-1.0,  0.0) - f, dir, len, lob, e);
      easuTap(&cSum, &wSum, vec2f( 0.0,  0.0) - f, dir, len, lob, f0);
      easuTap(&cSum, &wSum, vec2f( 1.0,  0.0) - f, dir, len, lob, f1);
      easuTap(&cSum, &wSum, vec2f( 2.0,  0.0) - f, dir, len, lob, h);
      easuTap(&cSum, &wSum, vec2f(-1.0,  1.0) - f, dir, len, lob, i);
      easuTap(&cSum, &wSum, vec2f( 0.0,  1.0) - f, dir, len, lob, j);
      easuTap(&cSum, &wSum, vec2f( 1.0,  1.0) - f, dir, len, lob, k);
      easuTap(&cSum, &wSum, vec2f( 2.0,  1.0) - f, dir, len, lob, l);
      easuTap(&cSum, &wSum, vec2f( 0.0,  2.0) - f, dir, len, lob, n);
      easuTap(&cSum, &wSum, vec2f( 1.0,  2.0) - f, dir, len, lob, o);

      let min4 = min(min(f0, f1), min(j, k));
      let max4 = max(max(f0, f1), max(j, k));
      let recon = clamp(cSum / max(wSum, 1e-4), min4, max4);

      // RCAS-style detail boost on the gamma-encoded signal, clamped to [min4, max4]
      let bilin = f0 * w0 + f1 * w1 + j * w2 + k * w3;
      let contrast = max(max(max4.r - min4.r, max4.g - min4.g), max4.b - min4.b);
      let noiseGate = smoothstep(0.015, 0.080, contrast);
      let lobe = clamp(min(luma709(min4), 1.0 - luma709(max4)) / max(luma709(max4), 1e-4), 0.0, 1.0);
      let sharpAmt = clamp(u.res.z, 0.0, 1.0) * 1.05 * noiseGate * (0.35 + 0.65 * lobe);
      return clamp(recon + (recon - bilin) * sharpAmt, min4, max4);
    }

    @fragment
    fn fsSpatial(@builtin(position) fragPos: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
      let vuv = contentUV(uv);
      if (outside(vuv)) { return vec4f(0.0, 0.0, 0.0, 1.0); }
      if (onDivider(uv)) { return vec4f(1.8, 1.8, 1.8, 1.0); }
      if (leftOfSplit(uv)) { return sdrOut(textureSampleLevel(srcTex, samp, vuv, 0.0).rgb); }
      return shade(max(easuSpatial(vuv), vec3f(0.0)), vuv, fragPos);
    }
  `;

  const SHADER_SOURCES = {
    vs: FULLSCREEN_VS,
    analyze: ANALYZE_CS,
    analyzeLoad: ANALYZE_LOAD_CS,
    stats: STATS_CS,
    map: MAP_CS,
    decode: DECODE_FS,
    final: FINAL_FS
  };

  // For offscreen tests: texture_2d<f32> instead of texture_external (otherwise identical source)
  function substituteTexture2D(code) {
    return code
      .split('texture_external').join('texture_2d<f32>')
      .split('textureLoad(vidTex, p)').join('textureLoad(vidTex, p, 0)');
  }

  async function createCore(device, opts = {}) {
    const test2D = Boolean(opts.texture2DSource);
    const prep = (code) => (test2D ? substituteTexture2D(code) : code);
    const SS = GPUShaderStage;
    const C = SS.COMPUTE;
    const F = SS.FRAGMENT;
    const tex = (visibility, binding) => ({ binding, visibility, texture: { sampleType: 'float' } });
    const ubuf = (visibility, binding) => ({ binding, visibility, buffer: { type: 'uniform' } });

    const videoBGL = device.createBindGroupLayout({
      label: 'truehdr-video',
      entries: [test2D
        ? { binding: 0, visibility: C | F, texture: { sampleType: 'float' } }
        : { binding: 0, visibility: C | F, externalTexture: {} }]
    });
    const anaBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 1, visibility: C, sampler: { type: 'filtering' } },
        { binding: 2, visibility: C, buffer: { type: 'storage' } }
      ]
    });
    const statsBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: C, buffer: { type: 'storage' } },
        ubuf(C, 2),
        { binding: 3, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float' } }
      ]
    });
    const mapBGL = device.createBindGroupLayout({
      entries: [
        tex(C, 0),
        tex(C, 1),
        { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 3, visibility: C, buffer: { type: 'read-only-storage' } },
        ubuf(C, 4)
      ]
    });
    const finalBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: F, sampler: { type: 'filtering' } },
        tex(F, 1),
        tex(F, 2),
        tex(F, 3),
        ubuf(F, 4),
        ubuf(F, 5),
        tex(F, 6)
      ]
    });

    const sources = {};
    const modules = {};
    for (const [name, code] of Object.entries(SHADER_SOURCES)) {
      sources[name] = prep(code);
      modules[name] = device.createShaderModule({ label: `truehdr-${name}`, code: sources[name] });
    }

    const PL = (bgls) => device.createPipelineLayout({ bindGroupLayouts: bgls });
    const compute = (name, entryPoint, layout) =>
      device.createComputePipelineAsync({ label: `truehdr-${name}`, layout, compute: { module: modules[name], entryPoint } });
    const render = (name, entryPoint, layout) =>
      device.createRenderPipelineAsync({
        label: `truehdr-${name}-${entryPoint}`,
        layout,
        vertex: { module: modules.vs, entryPoint: 'vsMain' },
        fragment: { module: modules[name], entryPoint, targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list' }
      });

    const finalLayout = PL([videoBGL, finalBGL]);
    const anaLayout = PL([videoBGL, anaBGL]);
    // Prefer bilinear quad taps; fall back to textureLoad if the implementation rejects sampling
    // an external texture in a compute shader.
    const analyzePipe = compute('analyze', 'csAnalyze', anaLayout).then(
      (pipe) => ({ pipe, variant: 'sampled' }),
      () => compute('analyzeLoad', 'csAnalyze', anaLayout).then((pipe) => ({ pipe, variant: 'load' }))
    );
    const [ana, stats, map, decode, finalDirect, finalSR, finalSpatial, finalImage] = await Promise.all([
      analyzePipe,
      compute('stats', 'csStats', PL([statsBGL])),
      compute('map', 'csMap', PL([mapBGL])),
      render('decode', 'fsDecode', PL([videoBGL])),
      render('final', 'fsDirect', finalLayout),
      render('final', 'fsSR', finalLayout),
      render('final', 'fsSpatial', finalLayout),
      // Image overlays only exist on the texture_2d core; skip the extra compile on the video core
      test2D ? render('final', 'fsImage', finalLayout) : Promise.resolve(null)
    ]);

    const sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge'
    });
    const dummyTex = device.createTexture({
      size: [1, 1, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING
    });

    return {
      device,
      test2D,
      sources,
      modules,
      sampler,
      dummyView: dummyTex.createView(),
      analyzeVariant: ana.variant,
      layouts: { videoBGL, anaBGL, statsBGL, mapBGL, finalBGL },
      pipes: { analyze: ana.pipe, stats, map, decode, finalDirect, finalSR, finalSpatial, finalImage }
    };
  }

  // Per-video GPU resources and per-frame encoding. Allocations happen only on size changes;
  // per frame: one bind group for the external texture (+ the swapchain view owned by the caller).
  class FrameRenderer {
    constructor(core) {
      this.core = core;
      const d = core.device;
      this.frameU = d.createBuffer({ size: FRAME_U_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.sceneU = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.state = d.createBuffer({
        size: STATE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      });
      this.u = new Float32Array(FRAME_U_FLOATS);
      this.lutTex = d.createTexture({
        size: [256, 1, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
      });
      this.lutView = this.lutTex.createView();
      this.srcW = 0;
      this.srcH = 0;
      this.cellsW = 0;
      this.cellsH = 0;
      this.cellTex = null;
      this.partials = null;
      this.mapTex = [null, null];
      this.mapViews = [null, null];
      this.mapIdx = 0;
      this.anaBG = null;
      this.statsBG = null;
      this.mapBGs = [null, null];
      this.decTex = null;
      this.decView = null;
      this.finalBGs = new Map();
      this.needReset = true;
      this.ts = null;
    }

    releaseSized() {
      for (const t of [this.cellTex, this.mapTex[0], this.mapTex[1], this.decTex, this.partials]) {
        if (t) {
          try { t.destroy(); } catch (_) {}
        }
      }
      this.cellTex = null;
      this.partials = null;
      this.mapTex = [null, null];
      this.mapViews = [null, null];
      this.decTex = null;
      this.decView = null;
      this.anaBG = null;
      this.statsBG = null;
      this.mapBGs = [null, null];
      this.finalBGs.clear();
      this.srcW = 0;
      this.srcH = 0;
    }

    ensure(w, h) {
      if (this.cellTex && this.srcW === w && this.srcH === h) return true;
      this.releaseSized();
      const core = this.core;
      const d = core.device;
      const cw = Math.ceil(w / CELL);
      const ch = Math.ceil(h / CELL);
      const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
      this.cellTex = d.createTexture({ size: [cw, ch, 1], format: 'rgba16float', usage });
      const cellView = this.cellTex.createView();
      for (let i = 0; i < 2; i++) {
        this.mapTex[i] = d.createTexture({ size: [cw, ch, 1], format: 'rgba16float', usage });
        this.mapViews[i] = this.mapTex[i].createView();
      }
      this.groupsW = Math.ceil(cw / 8);
      this.groupsH = Math.ceil(ch / 8);
      this.partials = d.createBuffer({ size: this.groupsW * this.groupsH * PARTIAL_BYTES, usage: GPUBufferUsage.STORAGE });
      this.anaBG = d.createBindGroup({
        layout: core.layouts.anaBGL,
        entries: [
          { binding: 0, resource: cellView },
          { binding: 1, resource: core.sampler },
          { binding: 2, resource: { buffer: this.partials } }
        ]
      });
      this.statsBG = d.createBindGroup({
        layout: core.layouts.statsBGL,
        entries: [
          { binding: 0, resource: { buffer: this.partials } },
          { binding: 1, resource: { buffer: this.state } },
          { binding: 2, resource: { buffer: this.frameU } },
          { binding: 3, resource: this.lutView }
        ]
      });
      for (let i = 0; i < 2; i++) {
        this.mapBGs[i] = d.createBindGroup({
          layout: core.layouts.mapBGL,
          entries: [
            { binding: 0, resource: cellView },
            { binding: 1, resource: this.mapViews[1 - i] },
            { binding: 2, resource: this.mapViews[i] },
            { binding: 3, resource: { buffer: this.state } },
            { binding: 4, resource: { buffer: this.frameU } }
          ]
        });
      }
      this.srcW = w;
      this.srcH = h;
      this.cellsW = cw;
      this.cellsH = ch;
      this.needReset = true;
      return true;
    }

    ensureDecoded() {
      if (this.decTex) return;
      this.decTex = this.core.device.createTexture({
        size: [this.srcW, this.srcH, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
      });
      this.decView = this.decTex.createView();
    }

    finalBindGroup(srcView, resView, idx) {
      let byRes = this.finalBGs.get(srcView);
      if (!byRes) {
        if (this.finalBGs.size > 6) this.finalBGs.clear();
        byRes = new Map();
        this.finalBGs.set(srcView, byRes);
      }
      let pair = byRes.get(resView);
      if (!pair) {
        if (byRes.size > 6) byRes.clear();
        const core = this.core;
        pair = [0, 1].map((i) => core.device.createBindGroup({
          layout: core.layouts.finalBGL,
          entries: [
            { binding: 0, resource: core.sampler },
            { binding: 1, resource: srcView },
            { binding: 2, resource: resView },
            { binding: 3, resource: this.mapViews[i] },
            { binding: 4, resource: { buffer: this.frameU } },
            { binding: 5, resource: { buffer: this.sceneU } },
            { binding: 6, resource: this.lutView }
          ]
        }));
        byRes.set(resView, pair);
      }
      return pair[idx];
    }

    enableTimestamps() {
      const d = this.core.device;
      if (this.ts) return true;
      if (!d.features || !d.features.has('timestamp-query')) return false;
      try {
        this.ts = {
          qs: d.createQuerySet({ type: 'timestamp', count: 2 }),
          resolve: d.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
          read: d.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
          busy: false,
          armed: false
        };
      } catch (_) {
        this.ts = null;
      }
      return Boolean(this.ts);
    }

    // Call after queue.submit(); resolves to GPU ms of the last measured frame (or null)
    readTimestamps() {
      const ts = this.ts;
      if (!ts || !ts.armed) return null;
      ts.armed = false;
      ts.busy = true;
      return ts.read.mapAsync(GPUMapMode.READ).then(
        () => {
          const a = new BigInt64Array(ts.read.getMappedRange());
          const ms = Number(a[1] - a[0]) / 1e6;
          ts.read.unmap();
          ts.busy = false;
          return ms > 0 && ms < 1000 ? ms : null;
        },
        () => {
          ts.busy = false;
          return null;
        }
      );
    }

    /**
     * p: { srcW, srcH, rect:[ox,oy,w,h] (canvas uv), outW, outH (content px in canvas),
     *      tone:[peak,thr,punch,gamut], shadowDepth, aplGuard, splitPos, hdr, bypassNativeHdr,
     *      sharpness, mode:'direct'|'sr'|'spatial', sr:[SRUpscaler], reset, dt, frameSeed, measure }
     * Returns { srPasses, resW, resH }.
     */
    encode(encoder, source, targetView, p) {
      const core = this.core;
      const d = core.device;
      this.ensure(p.srcW, p.srcH);
      const cur = this.mapIdx = 1 - this.mapIdx;
      const reset = Boolean(p.reset) || this.needReset;
      this.needReset = false;

      const dt = Math.min(Math.max(p.dt || 1 / 30, 1 / 240), 0.25);
      const u = this.u;
      u[0] = p.rect[0]; u[1] = p.rect[1]; u[2] = p.rect[2]; u[3] = p.rect[3];
      u[4] = p.tone[0]; u[5] = p.tone[1]; u[6] = p.tone[2]; u[7] = p.tone[3];
      u[8] = p.shadowDepth;
      u[9] = p.aplGuard;
      u[10] = p.splitPos;
      u[11] = (p.hdr ? 1 : 0) | (p.bypassNativeHdr ? 2 : 0) | (p.sharpDirect ? 4 : 0);
      u[12] = p.srcW; u[13] = p.srcH; u[14] = p.frameSeed || 0; u[15] = reset ? 1 : 0;
      u[18] = p.sharpness; u[19] = p.headroomCap > 1 ? p.headroomCap : 0;
      u[20] = p.outW; u[21] = p.outH;
      u[22] = p.srcW / (CELL * this.cellsW);
      u[23] = p.srcH / (CELL * this.cellsH);
      u[24] = 1 - Math.exp(-dt / 0.12);  // APL rising (guard engages)
      u[25] = 1 - Math.exp(-dt / 0.60);  // APL falling (guard releases)
      u[26] = 1 - Math.exp(-dt / 0.06);  // local map (and text attack)
      u[27] = 1 - Math.exp(-dt / 0.12);  // text release

      // p.tsw = { querySet, begin, end } lets the offscreen harness supply its own query slots
      const tsw = p.tsw || null;
      const measure = !tsw && Boolean(p.measure) && this.ts && !this.ts.busy;
      const tsQS = tsw ? tsw.querySet : measure ? this.ts.qs : null;
      const videoBG = d.createBindGroup({ layout: core.layouts.videoBGL, entries: [{ binding: 0, resource: source }] });

      const tsBegin = tsw ? tsw.begin : 0;
      const tsEnd = tsw ? tsw.end : 1;
      const runHdrCompute = Boolean(p.hdr);
      if (runHdrCompute) {
        const cp = encoder.beginComputePass(tsQS && tsBegin !== undefined
          ? { timestampWrites: { querySet: tsQS, beginningOfPassWriteIndex: tsBegin } }
          : undefined);
        cp.setPipeline(core.pipes.analyze);
        cp.setBindGroup(0, videoBG);
        cp.setBindGroup(1, this.anaBG);
        cp.dispatchWorkgroups(this.groupsW, this.groupsH);
        cp.setPipeline(core.pipes.stats);
        cp.setBindGroup(0, this.statsBG);
        cp.dispatchWorkgroups(1);
        cp.setPipeline(core.pipes.map);
        cp.setBindGroup(0, this.mapBGs[cur]);
        cp.dispatchWorkgroups(this.groupsW, this.groupsH);
        cp.end();
        encoder.copyBufferToBuffer(this.state, 0, this.sceneU, 0, 16);
      }

      let pipe = p.mode === 'image' && core.pipes.finalImage ? core.pipes.finalImage : core.pipes.finalDirect;
      let srcView = core.dummyView;
      let resView = core.dummyView;
      let resW = p.srcW;
      let resH = p.srcH;
      let srPasses = 0;
      let wroteBegin = runHdrCompute;
      if (p.mode === 'sr' || p.mode === 'spatial') {
        this.ensureDecoded();
        const rp = encoder.beginRenderPass({
          colorAttachments: [{ view: this.decView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
          ...(!wroteBegin && tsQS && tsBegin !== undefined
            ? { timestampWrites: { querySet: tsQS, beginningOfPassWriteIndex: tsBegin } }
            : {})
        });
        wroteBegin = true;
        rp.setPipeline(core.pipes.decode);
        rp.setBindGroup(0, videoBG);
        rp.draw(3);
        rp.end();
        srcView = this.decView;
        if (p.mode === 'sr' && p.sr && p.sr.length) {
          let view = this.decView;
          let w = p.srcW;
          let h = p.srcH;
          for (const inst of p.sr) {
            const o = inst.encode(encoder, view, w, h);
            view = o.view;
            w = o.width;
            h = o.height;
            srPasses++;
          }
          resView = view;
          resW = w;
          resH = h;
          pipe = core.pipes.finalSR;
        } else {
          pipe = core.pipes.finalSpatial;
        }
      }
      u[16] = resW;
      u[17] = resH;
      d.queue.writeBuffer(this.frameU, 0, u);

      const fpTS = tsQS ? {
        querySet: tsQS,
        ...(!wroteBegin && tsBegin !== undefined ? { beginningOfPassWriteIndex: tsBegin } : {}),
        ...(tsEnd !== undefined ? { endOfPassWriteIndex: tsEnd } : {})
      } : null;
      const fp = encoder.beginRenderPass({
        colorAttachments: [{ view: targetView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        ...(fpTS && Object.keys(fpTS).length ? { timestampWrites: fpTS } : {})
      });
      fp.setPipeline(pipe);
      fp.setBindGroup(0, videoBG);
      fp.setBindGroup(1, this.finalBindGroup(srcView, resView, cur));
      fp.draw(3);
      fp.end();
      if (measure) {
        encoder.resolveQuerySet(this.ts.qs, 0, 2, this.ts.resolve, 0);
        encoder.copyBufferToBuffer(this.ts.resolve, 0, this.ts.read, 0, 16);
        this.ts.armed = true;
      }
      return { srPasses, resW, resH };
    }

    destroy() {
      this.releaseSized();
      for (const b of [this.frameU, this.sceneU, this.state, this.lutTex]) {
        try { b.destroy(); } catch (_) {}
      }
      if (this.ts) {
        try { this.ts.qs.destroy(); this.ts.resolve.destroy(); this.ts.read.destroy(); } catch (_) {}
        this.ts = null;
      }
    }
  }

  const CORE_API = { CELL, STATE_BYTES, SHADER_SOURCES, substituteTexture2D, createCore, FrameRenderer };

  // Offscreen (Deno harness) use: expose the core and stop before any DOM work.
  if (typeof document === 'undefined' || typeof window === 'undefined') {
    G.__TRUEHDR_CORE__ = CORE_API;
    return;
  }

  // ---------------------------------------------------------------------------
  // Page integration
  // ---------------------------------------------------------------------------

  const CHANNEL = '__TRUEHDR_BRIDGE_V1__';

  const DEFAULT_SETTINGS = {
    enabled: true,
    hdrEnabled: true,
    upscaleEnabled: true,
    preset: 'vivid',
    peakBoost: 6.0,
    threshold: 0.46,
    localPunch: 0.70,
    gamutBoost: 0.45,
    shadowDepth: 0.14,
    aplGuard: 0.45,
    upscaleQuality: 'quality', // 'fast' | 'quality'
    upscaleContent: 'live',    // 'live' | 'anime'
    upscaleSharpness: 0.50,    // 0.0 .. 1.0
    bypassNativeHdr: true,
    imagesEnabled: false,      // also HDR-ify <img> (photos, GIFs, animated WebP/PNG)
    splitEnabled: false,
    splitPos: 0.5,
    effectiveHdr: true,
    effectiveUpscale: true,
    effectiveEnabled: true,
    effectiveImages: false
  };

  const SR_TIERS = ['quality', 'fast', 'spatial'];
  const SR_BUDGET_FRACTION = 0.45;   // keep GPU well below thermal/battery knee
  const SR_BUDGET_STRIKES = 3;       // fast step-down before GPU queue backs up

  let settings = { ...DEFAULT_SETTINGS };
  let isEdrDisplay = window.matchMedia('(dynamic-range: high)').matches;
  let gpuState = {
    status: 'init', // 'init' | 'ready' | 'unsupported' | 'error'
    device: null,
    core: null,
    initPromise: null,
    errorMsg: null
  };

  const instances = new Map(); // HTMLVideoElement -> VideoHDRProcessor

  function postToBridge(kind, payload) {
    window.postMessage({ channel: CHANNEL, from: 'engine', kind, payload }, '*');
  }

  // Background-app EDR limit. While Safari is not the frontmost app, stock WebKit lowers the
  // screen headroom it reports to every web/GPU process to 1.6x (WebProcessPool::
  // applyEDRSuppressionIfNeeded, on NSApplicationShouldBeginSuppressingHighDynamicRange-
  // ContentNotification); canvas, <img> and <video> are all capped and nothing on the page can
  // exceed it (measured with dev/focus-test on Safari 27.3). A curve rendered for 6x then
  // hard-clips every highlight flat at 1.6x, so the final pass keeps the curve below a knee and
  // rolls the rest softly into the limit (headroomCap -> FrameU res.w).
  // Another Safari window being key keeps full headroom, so page focus alone is not enough: the
  // bridge asks the native app whether Safari is frontmost and whether the clamp is lifted
  // (tools/edr-unsuppress with SIP off, or a patched WebKit).
  const SUPPRESSED_HEADROOM = 1.6;
  let appInactive = false;


  // Listen for dynamic range changes (e.g., moving window between XDR and external SDR monitor)
  try {
    const mq = window.matchMedia('(dynamic-range: high)');
    mq.addEventListener('change', (e) => {
      isEdrDisplay = e.matches;
      for (const inst of instances.values()) {
        inst.reconfigure();
        inst.scheduleRenderOnce();
      }
      broadcastStatus();
    });
  } catch (_) {}

  async function ensureGPU() {
    if (gpuState.status === 'ready') return gpuState;
    if (gpuState.status === 'unsupported' || gpuState.status === 'error') return null;
    if (gpuState.initPromise) return gpuState.initPromise;

    gpuState.initPromise = (async () => {
      try {
        if (!navigator.gpu) {
          gpuState.status = 'unsupported';
          gpuState.errorMsg = 'WebGPU is not available in this browser.';
          return null;
        }
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
        if (!adapter) {
          gpuState.status = 'unsupported';
          gpuState.errorMsg = 'No WebGPU adapter found.';
          return null;
        }
        const requiredFeatures = ['shader-f16', 'timestamp-query'].filter((f) => adapter.features.has(f));
        const requiredLimits = { maxTextureDimension2D: adapter.limits.maxTextureDimension2D };
        let device;
        try {
          device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
        } catch (_) {
          device = await adapter.requestDevice();
        }
        device.addEventListener('uncapturederror', (e) => {
          console.error('[TrueHDR] WebGPU uncaptured error:', e.error?.message || e.error);
          gpuState.lastWebGpuError = e.error?.message || String(e.error);
        });
        device.lost.then((info) => {
          console.warn('[TrueHDR] WebGPU device lost:', info.message);
          gpuState.status = 'init';
          gpuState.device = null;
          gpuState.core = null;
          gpuState.initPromise = null;
          for (const inst of instances.values()) {
            inst.onDeviceLost();
          }
          imageCore = { device: null, promise: null };
          for (const p of images.values()) {
            p.deactivate();
            p.core = null;
            p.ctx = null; // a context configured for the lost device is not reusable
          }
          scheduleImagePump();
        });

        const core = await createCore(device);
        gpuState.device = device;
        gpuState.core = core;
        gpuState.status = 'ready';
        return gpuState;
      } catch (err) {
        console.error('[TrueHDR] GPU init failed:', err);
        gpuState.status = 'error';
        gpuState.errorMsg = String(err && err.message ? err.message : err);
        return null;
      }
    })();

    return gpuState.initPromise;
  }

  // video === null: page-level toast (pages without a playing video, e.g. image HDR toggle)
  function showHudToast(video, text) {
    const host = video ? video.parentElement : document.body || document.documentElement;
    if (!host) return;
    let toast = host.querySelector(':scope > [data-truehdr-toast="1"]');
    if (!toast) {
      toast = document.createElement('div');
      toast.setAttribute('data-truehdr-toast', '1');
      Object.assign(toast.style, {
        position: video ? 'absolute' : 'fixed',
        top: '14px',
        right: '14px',
        zIndex: '2147483640',
        padding: '6px 12px',
        borderRadius: '999px',
        background: 'rgba(12, 14, 22, 0.78)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        border: '1px solid rgba(255, 200, 110, 0.45)',
        color: '#fff6e5',
        fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif',
        fontSize: '12px',
        fontWeight: '600',
        letterSpacing: '0.02em',
        pointerEvents: 'none',
        opacity: '0',
        transition: 'opacity 180ms ease'
      });
      host.appendChild(toast);
    }
    toast.textContent = text;
    toast.style.opacity = '1';
    clearTimeout(toast.__timer);
    toast.__timer = setTimeout(() => {
      toast.style.opacity = '0';
    }, 1500);
  }

  class VideoHDRProcessor {
    constructor(video) {
      this.video = video;
      this.canvas = null;
      this.ctx = null;
      this.renderer = null;
      this.upscalingActive = false;
      this.resetStateNextFrame = true;

      this.cachedRect = [0, 0, 1, 1];   // content rect in the video box (normalized)
      this.rectUniform = [0, 0, 1, 1];  // content rect in canvas uv
      this.contentOutW = 0;
      this.contentOutH = 0;
      this.cachedBoxW = 0;
      this.cachedBoxH = 0;
      this.lastLayoutCheckMs = 0;

      this.isVisible = !document.hidden;
      this.isActive = false;
      this.nativeHdrDetected = false;
      this.blockedReason = null;
      this.rvfcId = null;
      this.frameSeed = 0;
      this.frameCounter = 0;
      this.lastRenderedTime = -1;
      this.dirtyRender = true;
      this.lastRectKey = '';
      this.configuredWidth = 0;
      this.configuredHeight = 0;
      this.lastReportedSig = '';
      this.lastRenderMs = 0;
      this.destroyed = false;

      // ML super-resolution state
      this.srInsts = [];
      this.srKey = '';
      this.srGen = 0;
      this.srPending = null;
      this.srError = null;
      this.srDemoted = null;     // 'fast' | 'spatial' after the GPU-budget guard steps down
      this.srBudgetNote = null;
      this.srModel = null;
      this.srPasses = 0;
      this.srFallback = null;
      this.srCfgKey = '';

      // GPU timing / budget guard & queue backpressure
      this.gpuMs = null;
      this.gpuPending = false;
      this.inFlightGpu = 0;
      this.overBudget = 0;
      this.frameIntervalMs = 1000 / 30;
      this.lastVfcNow = 0;

      this.boundOnFrame = this.onVideoFrame.bind(this);
      this.boundSchedule = this.scheduleLoop.bind(this);
      this.boundRenderOnce = this.scheduleRenderOnce.bind(this);
      this.boundCheckMeta = this.checkVideoMetadata.bind(this);
      this.boundTimeUpdate = this.onTimeUpdate.bind(this);

      this.setupObservers();
      this.attachEvents();
      this.initResources();
    }

    setupObservers() {
      this.resizeObserver = new ResizeObserver(() => {
        if (this.syncOverlayLayout(true)) {
          this.dirtyRender = true;
          this.scheduleRenderOnce();
        }
      });
      this.resizeObserver.observe(this.video);
      if (this.video.parentElement) {
        this.resizeObserver.observe(this.video.parentElement);
      }

      this.intersectionObserver = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            this.isVisible = entry.isIntersecting;
          }
          if (this.isVisible && !document.hidden) {
            this.dirtyRender = true;
            this.scheduleLoop();
            this.scheduleRenderOnce();
          } else {
            this.stopLoop();
          }
        },
        { threshold: 0.02 }
      );
      this.intersectionObserver.observe(this.video);
    }

    attachEvents() {
      this.video.addEventListener('play', this.boundSchedule);
      this.video.addEventListener('playing', this.boundSchedule);
      this.video.addEventListener('canplay', this.boundSchedule);
      this.video.addEventListener('pause', this.boundRenderOnce);
      this.video.addEventListener('seeked', this.boundRenderOnce);
      this.video.addEventListener('timeupdate', this.boundTimeUpdate);
      this.video.addEventListener('loadeddata', this.boundCheckMeta);
      this.video.addEventListener('loadedmetadata', this.boundCheckMeta);
      this.video.addEventListener('resize', this.boundCheckMeta);
      this.video.addEventListener('encrypted', this.boundCheckMeta);
    }

    onTimeUpdate() {
      // Watchdog: if rVFC stalled during a seek/tab switch, kick loop back on
      if (!this.destroyed && this.isVisible && settings.effectiveEnabled && !this.video.paused && !this.video.ended) {
        if (this.rvfcId === null) {
          this.scheduleLoop();
          this.scheduleRenderOnce();
        }
      }
    }

    async initResources() {
      const gpu = await ensureGPU();
      if (!gpu || this.destroyed) {
        if (gpuState.errorMsg) {
          this.blockedReason = gpuState.errorMsg;
          broadcastStatus();
        }
        return;
      }

      if (!this.renderer) {
        this.renderer = new FrameRenderer(gpu.core);
        this.renderer.enableTimestamps();
      }

      this.ensureCanvas();
      this.checkVideoMetadata();
      this.syncOverlayLayout(true);
      this.scheduleLoop();
    }

    onDeviceLost() {
      this.releaseGpu();
      // Keep the canvas/context; the next layout pass re-configures it with the new device
      this.configuredWidth = 0;
      this.configuredHeight = 0;
      this.lastRectKey = '';
      if (!this.destroyed) {
        this.initResources();
      }
    }

    releaseGpu() {
      this.dropSR();
      if (this.renderer) {
        try { this.renderer.destroy(); } catch (_) {}
      }
      this.renderer = null;
      this.gpuMs = null;
      this.gpuPending = false;
      this.inFlightGpu = 0;
      this.resetStateNextFrame = true;
    }

    dropSR() {
      this.srGen++;
      for (const inst of this.srInsts) {
        try { inst.destroy(); } catch (_) {}
      }
      this.srInsts = [];
      this.srPending = null;
      this.srKey = '';
      this.srError = null;
    }

    resetSRGuard() {
      this.srDemoted = null;
      this.srBudgetNote = null;
      this.overBudget = 0;
      this.gpuMs = null;
      this.frameCounter = 0;
    }

    ensureCanvas() {
      if (this.canvas && this.canvas.isConnected) return;
      const parent = this.video.parentElement;
      if (!parent) return;

      const canvas = document.createElement('canvas');
      canvas.setAttribute('data-truehdr', '1');
      canvas.setAttribute('aria-hidden', 'true');
      canvas.tabIndex = -1;
      canvas.inert = true;
      Object.assign(canvas.style, {
        position: 'absolute',
        margin: '0',
        padding: '0',
        border: '0',
        background: 'transparent',
        display: 'none'
      });
      canvas.style.setProperty('pointer-events', 'none', 'important');
      canvas.style.setProperty('touch-action', 'none', 'important');
      canvas.style.setProperty('user-select', 'none', 'important');
      canvas.style.setProperty('-webkit-user-select', 'none', 'important');
      this.video.insertAdjacentElement('afterend', canvas);
      this.canvas = canvas;
      this.ctx = canvas.getContext('webgpu');
      this.ensureSplitHandle();
      this.configuredWidth = 0;
      this.configuredHeight = 0;
      this.lastRectKey = '';
    }

    ensureSplitHandle() {
      if (this.splitHandle && this.splitHandle.isConnected) return;
      const anchor = this.canvas;
      if (!anchor || !anchor.parentElement) return;
      const h = document.createElement('div');
      h.setAttribute('data-truehdr-split', '1');
      Object.assign(h.style, {
        position: 'absolute',
        width: '28px',
        marginLeft: '-14px',
        pointerEvents: 'none',
        touchAction: 'none',
        userSelect: 'none',
        webkitUserSelect: 'none',
        background: 'transparent',
        display: 'none'
      });
      const pill = document.createElement('div');
      Object.assign(pill.style, {
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: '22px',
        height: '44px',
        borderRadius: '999px',
        background: 'rgba(14, 16, 24, 0.78)',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
        border: '1px solid rgba(255, 255, 255, 0.65)',
        boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#fff',
        fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif',
        fontSize: '11px',
        fontWeight: '700',
        letterSpacing: '1px',
        cursor: 'ew-resize',
        pointerEvents: 'auto',
        touchAction: 'none'
      });
      pill.textContent = '‹›';
      h.appendChild(pill);

      let dragging = false;
      const updateFromEvent = (e) => {
        if (!this.canvas) return;
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width <= 0) return;
        settings.splitPos = Math.min(0.95, Math.max(0.05, (e.clientX - rect.left) / rect.width));
        this.syncSplitHandle();
        this.renderFrame();
      };
      pill.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        dragging = true;
        try { pill.setPointerCapture(e.pointerId); } catch (_) {}
        updateFromEvent(e);
      });
      pill.addEventListener('pointermove', (e) => {
        if (!dragging) return;
        e.preventDefault();
        e.stopPropagation();
        updateFromEvent(e);
      });
      const stopDrag = (e) => {
        if (!dragging) return;
        dragging = false;
        e.stopPropagation();
        try { pill.releasePointerCapture(e.pointerId); } catch (_) {}
        postToBridge('SET_SPLIT_POS', { splitPos: settings.splitPos });
      };
      pill.addEventListener('pointerup', stopDrag);
      pill.addEventListener('pointercancel', stopDrag);
      for (const ev of ['mousedown', 'click', 'dblclick']) {
        pill.addEventListener(ev, (e) => e.stopPropagation());
      }

      anchor.insertAdjacentElement('afterend', h);
      this.splitHandle = h;
    }

    syncSplitHandle() {
      const h = this.splitHandle;
      if (!h || !this.canvas) return;
      if (!settings.splitEnabled || this.canvas.style.display === 'none' || !this.shouldRender()) {
        if (h.style.display !== 'none') h.style.display = 'none';
        return;
      }
      if (this.canvas.parentElement && h.previousSibling !== this.canvas) {
        this.canvas.insertAdjacentElement('afterend', h);
      }
      const left = parseFloat(this.canvas.style.left) || 0;
      const top = parseFloat(this.canvas.style.top) || 0;
      const w = parseFloat(this.canvas.style.width) || 0;
      const ch = parseFloat(this.canvas.style.height) || 0;
      if (w <= 0 || ch <= 0) {
        h.style.display = 'none';
        return;
      }
      const padTop = Math.min(20, ch * 0.06);
      const padBot = Math.min(64, ch * 0.16);
      const z = (parseInt(this.canvas.style.zIndex, 10) || 0) + 1;
      h.style.left = `${left + settings.splitPos * w}px`;
      h.style.top = `${top + padTop}px`;
      h.style.height = `${Math.max(40, ch - padTop - padBot)}px`;
      h.style.zIndex = String(z);
      if (h.style.display !== 'block') h.style.display = 'block';
    }

    reconfigure() {
      if (!this.ctx || gpuState.status !== 'ready') return;
      try {
        this.ctx.configure({
          device: gpuState.device,
          format: 'rgba16float',
          colorSpace: 'display-p3',
          toneMapping: { mode: 'extended' },
          alphaMode: 'premultiplied'
        });
      } catch (err) {
        console.warn('[TrueHDR] Context configure failed:', err);
      }
    }

    checkVideoMetadata() {
      if (this.video.mediaKeys) {
        this.blockedReason = 'DRM-protected video';
      } else if (this.blockedReason === 'DRM-protected video') {
        this.blockedReason = null;
      }

      this.nativeHdrDetected = false;
      if (typeof VideoFrame === 'function' && this.video.readyState >= 2 && this.video.videoWidth > 0) {
        let vf = null;
        try {
          vf = new VideoFrame(this.video);
          const transfer = vf.colorSpace && vf.colorSpace.transfer;
          if (transfer === 'pq' || transfer === 'hlg') {
            this.nativeHdrDetected = true;
          }
        } catch (_) {
          // Cross-origin or DRM videos throw here; handled during render
        } finally {
          if (vf) vf.close();
        }
      }

      this.resetStateNextFrame = true;
      this.dirtyRender = true;
      this.resetSRGuard();
      this.syncOverlayLayout(true);
      this.scheduleRenderOnce();
      broadcastStatus();
    }

    shouldRender() {
      if (this.destroyed || document.hidden) return false;
      const wantHdr = Boolean(settings.effectiveHdr) && !(settings.bypassNativeHdr && this.nativeHdrDetected);
      const wantUpscale = Boolean(settings.effectiveUpscale);
      if (!wantHdr && !wantUpscale) return false;
      if (this.video.readyState < 2 || this.video.videoWidth === 0 || this.video.videoHeight === 0) return false;
      const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
      if (fsEl === this.video) return false;
      if (this.blockedReason) return false;
      return true;
    }

    computeContentRect(boxW, boxH, fit) {
      const vw = this.video.videoWidth || 1;
      const vh = this.video.videoHeight || 1;

      if (fit === 'fill') {
        return [0, 0, 1, 1];
      }
      const videoAspect = vw / vh;
      const boxAspect = Math.max(boxW, 1) / Math.max(boxH, 1);

      if (fit === 'cover') {
        if (videoAspect > boxAspect) {
          const rw = videoAspect / boxAspect;
          return [(1 - rw) * 0.5, 0, rw, 1];
        } else {
          const rh = boxAspect / videoAspect;
          return [0, (1 - rh) * 0.5, 1, rh];
        }
      }

      if (videoAspect > boxAspect) {
        const rh = boxAspect / videoAspect;
        return [0, (1 - rh) * 0.5, 1, rh];
      } else {
        const rw = videoAspect / boxAspect;
        return [(1 - rw) * 0.5, 0, rw, 1];
      }
    }

    // Canvas backing size:
    //  - upscaling: every below-4K source (480p/720p/1080p/1440p) is upscaled to 4K content
    //    (3840x2160; one 2x AI pass, EASU completes the rest), even when the player box is smaller, so it gets real
    //    4K detail in windowed/theater Safari; above-4K boxes use the box's device pixels.
    //    Thumbnails / hover previews (content under 560x315 CSS px) are left alone.
    //  - otherwise: content at min(source, display) pixels (1:1 texel-aligned when the display is
    //    larger); the compositor scales the canvas up, so the GPU only shades source pixels.
    computeCanvasPlan(boxCssW, boxCssH, rect) {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const maxDim = gpuState.device ? gpuState.device.limits.maxTextureDimension2D : 8192;
      const vw = Math.max(1, this.video.videoWidth);
      const vh = Math.max(1, this.video.videoHeight);
      const [ox, oy, rw, rh] = rect;
      const boxW = boxCssW * dpr;
      const boxH = boxCssH * dpr;
      const upRatio = Math.max((boxW * rw) / vw, (boxH * rh) / vh);
      const below4k = vw < 3840 && vh < 2160;
      // Any real player (content >= 560 CSS px wide or 315 tall) gets 4K; thumbnails and hover
      // previews below that stay native so they don't burn GPU on 4K SR.
      const playerSized = boxCssW * rw >= 560 || boxCssH * rh >= 315;
      const upscale = Boolean(settings.effectiveUpscale) && (upRatio > 1.15 || (below4k && playerSized));

      let targetW = boxW;
      let targetH = boxH;
      if (upscale && below4k) {
        const capW = vw >= vh ? Math.min(3840, maxDim) : Math.min(2160, maxDim);
        const capH = vw >= vh ? Math.min(2160, maxDim) : Math.min(3840, maxDim);
        const curCW = boxW * Math.min(rw, 1.0);
        const curCH = boxH * Math.min(rh, 1.0);
        const desireW = Math.max(curCW, capW);
        const desireH = Math.max(curCH, capH);
        const boost = Math.max(desireW / Math.max(curCW, 1), desireH / Math.max(curCH, 1));
        if (boost > 1.0) {
          targetW *= boost;
          targetH *= boost;
        }
      } else if (!upscale && upRatio > 1.0) {
        let cw = Math.max(1, Math.round(vw / rw));
        let ch = Math.max(1, Math.round(vh / rh));
        if (cw <= maxDim && ch <= maxDim) {
          const offX = Math.round((cw - vw) / 2);
          const offY = Math.round((ch - vh) / 2);
          return { upscale, cw, ch, rectUniform: [offX / cw, offY / ch, vw / cw, vh / ch], outW: vw, outH: vh };
        }
      }
      const k = Math.min(1, maxDim / Math.max(targetW, 1), maxDim / Math.max(targetH, 1));
      const cw = Math.max(1, Math.round(targetW * k));
      const ch = Math.max(1, Math.round(targetH * k));
      return {
        upscale,
        cw,
        ch,
        rectUniform: [ox, oy, rw, rh],
        outW: Math.max(1, Math.round(cw * rw)),
        outH: Math.max(1, Math.round(ch * rh))
      };
    }

    syncOverlayLayout(force = false) {
      if (!this.video.isConnected) {
        this.destroy();
        return false;
      }
      const now = performance.now();
      if (!force && (now - this.lastLayoutCheckMs) < 1000 && this.canvas && this.canvas.parentElement === this.video.parentElement && this.configuredWidth > 0) {
        return false;
      }
      this.lastLayoutCheckMs = now;

      if (!this.canvas) {
        this.ensureCanvas();
      }
      if (!this.canvas) return false;

      if (this.video.parentElement && this.canvas.parentElement !== this.video.parentElement) {
        this.video.insertAdjacentElement('afterend', this.canvas);
      }

      if (!this.shouldRender()) {
        if (this.canvas.style.display !== 'none') {
          this.canvas.style.display = 'none';
        }
        this.syncSplitHandle();
        if (this.isActive) {
          this.isActive = false;
          broadcastStatus();
        }
        return false;
      }

      const vRect = this.video.getBoundingClientRect();
      if (vRect.width < 96 || vRect.height < 64) {
        if (this.canvas.style.display !== 'none') {
          this.canvas.style.display = 'none';
        }
        this.syncSplitHandle();
        if (this.isActive) {
          this.isActive = false;
          broadcastStatus();
        }
        return false;
      }

      const cb = this.canvas.offsetParent || document.documentElement;
      let left;
      let top;
      if (cb === document.documentElement || cb === document.body) {
        left = vRect.left + window.scrollX;
        top = vRect.top + window.scrollY;
      } else {
        const cbRect = cb.getBoundingClientRect();
        left = vRect.left - cbRect.left - cb.clientLeft + cb.scrollLeft;
        top = vRect.top - cbRect.top - cb.clientTop + cb.scrollTop;
      }

      const cs = getComputedStyle(this.video);
      const fit = cs.objectFit || 'contain';
      const rect = this.computeContentRect(vRect.width, vRect.height, fit);
      this.cachedRect = rect;
      this.cachedBoxW = vRect.width;
      this.cachedBoxH = vRect.height;

      const plan = this.computeCanvasPlan(vRect.width, vRect.height, rect);
      const cw = plan.cw;
      const ch = plan.ch;
      this.rectUniform = plan.rectUniform;
      this.contentOutW = plan.outW;
      this.contentOutH = plan.outH;

      const rectKey = `${left.toFixed(1)},${top.toFixed(1)},${vRect.width.toFixed(1)},${vRect.height.toFixed(1)},${cw},${ch},${cs.zIndex},${cs.borderRadius},${Boolean(this.video.controls)}`;
      const rectChanged = rectKey !== this.lastRectKey;
      if (rectChanged) {
        this.lastRectKey = rectKey;
        this.canvas.style.left = `${left}px`;
        this.canvas.style.top = `${top}px`;
        this.canvas.style.width = `${vRect.width}px`;
        this.canvas.style.height = `${vRect.height}px`;
        this.canvas.style.zIndex = cs.zIndex;
        this.canvas.style.borderRadius = cs.borderRadius;
        this.canvas.style.clipPath = this.video.controls ? 'inset(0 0 64px 0)' : '';
        this.canvas.style.setProperty('pointer-events', 'none', 'important');

        if (this.canvas.width !== cw || this.canvas.height !== ch || this.configuredWidth === 0) {
          this.canvas.width = cw;
          this.canvas.height = ch;
          this.configuredWidth = cw;
          this.configuredHeight = ch;
          this.reconfigure();
        }
      }

      const upscalingChanged = this.upscalingActive !== plan.upscale;
      this.upscalingActive = plan.upscale;

      let shownChanged = false;
      if (this.canvas.style.display !== 'block') {
        this.canvas.style.display = 'block';
        shownChanged = true;
      }
      this.syncSplitHandle();
      if (!this.isActive || upscalingChanged) {
        this.isActive = true;
        broadcastStatus();
      }
      return rectChanged || upscalingChanged || shownChanged;
    }

    // Decide ML SR vs spatial fallback for this frame; creates runtime instances lazily.
    planUpscale(vw, vh, outW, outH) {
      const content = settings.upscaleContent === 'anime' ? 'anime' : 'live';
      const wantTier = settings.upscaleQuality === 'fast' ? 'fast' : 'quality';
      const cfgKey = `${content}|${wantTier}`;
      if (cfgKey !== this.srCfgKey) {
        this.srCfgKey = cfgKey;
        this.resetSRGuard();
      }
      const tierIdx = Math.max(SR_TIERS.indexOf(wantTier), this.srDemoted ? SR_TIERS.indexOf(this.srDemoted) : 0);
      const tier = SR_TIERS[tierIdx];
      const spatial = (why) => ({ mode: 'spatial', sr: null, model: null, fallback: why });

      if (tier === 'spatial') return spatial(this.srBudgetNote || 'spatial');
      const rt = G.__TRUEHDR_SR__;
      if (!rt || typeof rt.create !== 'function') return spatial('ML runtime not loaded');
      const maxDim = gpuState.device.limits.maxTextureDimension2D;
      if (2 * vw > maxDim || 2 * vh > maxDim) return spatial('source too large for SR');

      // Single 2x AI pass + EASU/RCAS spatial completion in fsSR (avoids 5x compute energy of
      // chaining a second 2x SR pass on 4x pixels).
      const passes = 1;

      const modelId = `tsr-${content}-${tier}`;
      const insts = this.ensureSRInstances(rt, modelId, passes);
      if (!insts) return spatial(this.srError || `loading ${modelId}`);
      return { mode: 'sr', sr: insts, model: modelId, fallback: this.srBudgetNote };
    }

    ensureSRInstances(rt, modelId, count) {
      if (this.srKey !== modelId) {
        this.dropSR();
        this.srKey = modelId;
      }
      if (this.srError) return null;
      if (!this.srPending && this.srInsts.length < count) {
        let known = true;
        try {
          const list = typeof rt.listModels === 'function' ? rt.listModels() : null;
          if (Array.isArray(list)) known = list.some((m) => m && m.id === modelId);
        } catch (_) {}
        if (!known) {
          this.srError = `model ${modelId} not available`;
          return null;
        }
        const device = gpuState.device;
        const gen = this.srGen;
        this.srPending = Promise.resolve()
          .then(() => rt.create(device, { modelId, useF16: device.features.has('shader-f16') }))
          .then(
            (inst) => {
              if (gen !== this.srGen || this.destroyed || gpuState.device !== device) {
                try { inst.destroy(); } catch (_) {}
                return;
              }
              this.srPending = null;
              this.srInsts.push(inst);
              this.dirtyRender = true;
              this.scheduleRenderOnce();
              broadcastStatus();
            },
            (err) => {
              if (gen !== this.srGen) return;
              this.srPending = null;
              this.srError = `SR init failed: ${err && err.message ? err.message : err}`;
              console.warn('[TrueHDR]', this.srError);
              broadcastStatus();
            }
          );
      }
      if (this.srInsts.length === 0) return null;
      return this.srInsts.slice(0, Math.min(count, this.srInsts.length));
    }

    noteGpuTime(ms) {
      if (typeof ms !== 'number' || !(ms >= 0) || !isFinite(ms)) return;
      this.gpuMs = this.gpuMs == null ? ms : this.gpuMs * 0.5 + ms * 0.5;
      if (this.srPasses === 0) {
        this.overBudget = 0;
        return;
      }
      const budget = SR_BUDGET_FRACTION * this.frameIntervalMs;
      if (this.gpuMs > budget) {
        this.overBudget += ms > budget * 1.5 ? 2 : 1;
      } else {
        this.overBudget = Math.max(0, this.overBudget - 2);
      }
      if (this.overBudget >= SR_BUDGET_STRIKES) {
        const wantTier = settings.upscaleQuality === 'fast' ? 'fast' : 'quality';
        const from = SR_TIERS[Math.max(SR_TIERS.indexOf(wantTier), this.srDemoted ? SR_TIERS.indexOf(this.srDemoted) : 0)];
        const to = SR_TIERS[Math.min(SR_TIERS.indexOf(from) + 1, SR_TIERS.length - 1)];
        this.srDemoted = to;
        this.srBudgetNote = `GPU ${this.gpuMs.toFixed(1)} ms > ${budget.toFixed(1)} ms budget: ${from} -> ${to}`;
        console.warn('[TrueHDR]', this.srBudgetNote);
        this.overBudget = 0;
        this.gpuMs = null;
        this.dropSR();
        broadcastStatus();
      }
    }

    renderFrame() {
      if (this.destroyed || gpuState.status !== 'ready' || !this.renderer) return;
      const maxInFlight = this.upscalingActive ? 1 : 2;
      if (this.inFlightGpu >= maxInFlight) {
        this.dirtyRender = true;
        return;
      }
      this.syncOverlayLayout(false);
      if (!this.shouldRender() || !this.ctx || !this.canvas) return;
      if (this.cachedBoxW < 96 || this.cachedBoxH < 64) return;
      if (this.configuredWidth === 0) {
        this.syncOverlayLayout(true);
        if (this.configuredWidth === 0) return;
      }

      const device = gpuState.device;
      const vw = this.video.videoWidth;
      const vh = this.video.videoHeight;
      const maxDim = device.limits.maxTextureDimension2D;
      if (vw < 1 || vh < 1 || vw > maxDim || vh > maxDim) return;

      let externalTex;
      try {
        externalTex = device.importExternalTexture({ source: this.video });
      } catch (err) {
        const msg = String(err);
        if (msg.includes('SecurityError') || msg.toLowerCase().includes('origin') || msg.toLowerCase().includes('taint')) {
          this.blockedReason = this.video.mediaKeys ? 'DRM-protected video' : 'Cross-origin restricted (CORS)';
          this.syncOverlayLayout(true);
          broadcastStatus();
        }
        // Transient NotReadable/InvalidState frames during seek or ABR switch are skipped without locking the loop
        return;
      }

      let swapTex;
      try {
        swapTex = this.ctx.getCurrentTexture();
      } catch (_) {
        this.reconfigure();
        try {
          swapTex = this.ctx.getCurrentTexture();
        } catch (e2) {
          return;
        }
      }

      const plan = this.upscalingActive
        ? this.planUpscale(vw, vh, this.contentOutW, this.contentOutH)
        : { mode: 'direct', sr: null, model: null, fallback: null };

      const now = performance.now();
      const dt = this.lastRenderMs > 0 ? (now - this.lastRenderMs) / 1000 : 1 / 30;
      this.lastRenderMs = now;
      this.frameSeed = (this.frameSeed + 1) % 256;
      const reset = this.resetStateNextFrame;
      this.resetStateNextFrame = false;
      this.dirtyRender = false;
      const isSr = plan.mode === 'sr';
      const measure = isSr && (this.gpuMs === null || this.frameCounter < 16 || this.overBudget > 0 || (this.frameCounter & 63) === 0);
      this.frameCounter++;

      const encoder = device.createCommandEncoder();
      let result;
      try {
        result = this.renderer.encode(encoder, externalTex, swapTex.createView(), {
          srcW: vw,
          srcH: vh,
          rect: this.rectUniform,
          outW: this.contentOutW,
          outH: this.contentOutH,
          tone: [settings.peakBoost, settings.threshold, settings.localPunch, settings.gamutBoost],
          headroomCap: appInactive ? SUPPRESSED_HEADROOM : 0,
          shadowDepth: settings.shadowDepth,
          aplGuard: settings.aplGuard,
          splitPos: settings.splitEnabled ? settings.splitPos : -1.0,
          hdr: Boolean(settings.effectiveHdr),
          bypassNativeHdr: Boolean(settings.bypassNativeHdr),
          sharpDirect: Boolean(settings.effectiveUpscale),
          sharpness: typeof settings.upscaleSharpness === 'number' ? settings.upscaleSharpness : DEFAULT_SETTINGS.upscaleSharpness,
          mode: plan.mode,
          sr: plan.sr,
          reset,
          dt,
          frameSeed: this.frameSeed,
          measure
        });
      } catch (err) {
        console.warn('[TrueHDR] Frame encode failed:', err);
        if (plan.mode === 'sr') {
          this.dropSR();
          this.srKey = plan.model || '';
          this.srError = `SR encode failed: ${err && err.message ? err.message : err}`;
          broadcastStatus();
        }
        this.resetStateNextFrame = true;
        return;
      }
      device.queue.submit([encoder.finish()]);
      this.inFlightGpu++;
      const tsPromise = measure ? this.renderer.readTimestamps() : null;
      if (tsPromise) {
        tsPromise.then((ms) => this.noteGpuTime(ms));
        device.queue.onSubmittedWorkDone().then(
          () => { this.inFlightGpu = Math.max(0, this.inFlightGpu - 1); },
          () => { this.inFlightGpu = Math.max(0, this.inFlightGpu - 1); }
        );
      } else {
        const measureWall = measure && !this.gpuPending;
        if (measureWall) this.gpuPending = true;
        const startMs = measureWall ? performance.now() : 0;
        device.queue.onSubmittedWorkDone().then(
          () => {
            this.inFlightGpu = Math.max(0, this.inFlightGpu - 1);
            if (measureWall) {
              this.gpuPending = false;
              this.noteGpuTime(performance.now() - startMs);
            }
          },
          () => {
            this.inFlightGpu = Math.max(0, this.inFlightGpu - 1);
            if (measureWall) this.gpuPending = false;
          }
        );
      }

      const srModel = plan.mode === 'sr' ? plan.model : null;
      const srFallback = plan.mode === 'sr' ? plan.fallback : (this.upscalingActive ? plan.fallback : null);
      this.srPasses = result.srPasses;
      const sig = `${vw}x${vh}->${this.contentOutW}x${this.contentOutH}|${Boolean(settings.effectiveHdr)}|${this.upscalingActive}|${srModel}|${result.srPasses}|${srFallback}`;
      if (this.lastReportedSig !== sig || this.srModel !== srModel || this.srFallback !== srFallback) {
        this.srModel = srModel;
        this.srFallback = srFallback;
        this.lastReportedSig = sig;
        broadcastStatus();
      }
    }

    onVideoFrame(now) {
      this.rvfcId = null;
      if (this.destroyed || document.hidden || !this.isVisible) return;
      const ct = this.video.currentTime;
      // Avoid requestVideoFrameCallback in Safari/WebKit: startVideoFrameMetadataGathering
      // forces sync GPU IPC + CPU CVPixelBuffer copies on every frame even when using
      // zero-copy GPU importExternalTexture().
      const sameFrame = ct === this.lastRenderedTime && !this.dirtyRender;
      if (!sameFrame) {
        if (typeof now === 'number') {
          if (this.lastVfcNow > 0) {
            const d = now - this.lastVfcNow;
            if (d > 4 && d < 150) this.frameIntervalMs = this.frameIntervalMs * 0.9 + d * 0.1;
          }
          this.lastVfcNow = now;
        }
        this.lastRenderedTime = ct;
        this.renderFrame();
      }
      if (!this.video.paused && !this.video.ended && this.isVisible && !document.hidden && settings.effectiveEnabled) {
        this.scheduleLoop();
      }
    }

    scheduleLoop() {
      if (this.destroyed || this.rvfcId !== null) return;
      if (!this.isVisible || document.hidden || !settings.effectiveEnabled) return;
      this.rvfcId = requestAnimationFrame(this.boundOnFrame);
    }

    scheduleRenderOnce() {
      if (this.destroyed || this._oncePending || document.hidden || !this.isVisible) return;
      this._oncePending = true;
      requestAnimationFrame(() => {
        this._oncePending = false;
        this.renderFrame();
      });
    }

    stopLoop() {
      if (this.rvfcId !== null) {
        cancelAnimationFrame(this.rvfcId);
        this.rvfcId = null;
      }
    }

    retryWithCors() {
      if (!this.video.src || this.video.src.startsWith('blob:') || this.video.src.startsWith('data:')) {
        return;
      }
      if (this.video.crossOrigin === 'anonymous') return;
      const t = this.video.currentTime;
      const wasPlaying = !this.video.paused;
      const src = this.video.currentSrc || this.video.src;
      this.blockedReason = null;
      this.video.crossOrigin = 'anonymous';
      this.video.src = src;
      this.video.load();
      this.video.addEventListener(
        'loadedmetadata',
        () => {
          try { this.video.currentTime = t; } catch (_) {}
          if (wasPlaying) this.video.play().catch(() => {});
          this.checkVideoMetadata();
        },
        { once: true }
      );
    }

    destroy() {
      if (this.destroyed) return;
      this.destroyed = true;
      this.stopLoop();
      if (this.resizeObserver) this.resizeObserver.disconnect();
      if (this.intersectionObserver) this.intersectionObserver.disconnect();
      this.video.removeEventListener('play', this.boundSchedule);
      this.video.removeEventListener('playing', this.boundSchedule);
      this.video.removeEventListener('canplay', this.boundSchedule);
      this.video.removeEventListener('pause', this.boundRenderOnce);
      this.video.removeEventListener('seeked', this.boundRenderOnce);
      this.video.removeEventListener('timeupdate', this.boundTimeUpdate);
      this.video.removeEventListener('loadeddata', this.boundCheckMeta);
      this.video.removeEventListener('loadedmetadata', this.boundCheckMeta);
      this.video.removeEventListener('resize', this.boundCheckMeta);
      this.video.removeEventListener('encrypted', this.boundCheckMeta);
      if (this.ctx) {
        try { this.ctx.unconfigure(); } catch (_) {}
      }
      if (this.canvas && this.canvas.parentElement) {
        this.canvas.remove();
      }
      if (this.splitHandle && this.splitHandle.parentElement) {
        this.splitHandle.remove();
      }
      this.releaseGpu();
      instances.delete(this.video);
      broadcastStatus();
    }
  }

  // ---------------------------------------------------------------------------
  // Images & GIFs (opt-in: settings.imagesEnabled). Each eligible on-screen <img> gets an EDR
  // overlay canvas fed through the same analysis + tone pipeline as video (texture_2d core).
  // Static images render once per layout/settings change; animated GIF/APNG/WebP/AVIF are
  // re-sampled at up to 30 fps. GPU memory is bounded by a count + pixel budget.
  // ---------------------------------------------------------------------------

  const IMG_MIN_CSS_W = 96;
  const IMG_MIN_CSS_H = 64;
  const IMG_MIN_NATURAL = 48;
  const IMG_MAX_ACTIVE = 10;
  const IMG_MAX_ANIMATED = 4;
  const IMG_PIXEL_BUDGET = 16e6;        // sum of active overlay canvas pixels
  const IMG_MAX_CANVAS_PIXELS = 4.2e6;  // per overlay; larger displays are compositor-upscaled
  const IMG_MAX_SOURCE_PIXELS = 8.4e6;
  const IMG_ANIM_INTERVAL_MS = 1000 / 30;
  const IMG_SNIFF_BYTES = 256 * 1024;
  const IMG_STATIC_EXT = /\.(jpe?g|jfif|bmp|ico|svg|heic|heif)(?:[?#]|$)/i;
  const IMG_ANIM_HINT = /(\.(gif|webp|apng)(?:[?#]|$))|^data:image\/(gif|webp|apng)/i;

  const images = new Map();            // HTMLImageElement -> ImageHDRProcessor
  const imgSourceCache = new Map();    // url -> Promise<{ kind: 'cors' } | { kind: 'blob', blob } | null>
  const imgAnimCache = new Map();      // url -> Promise<boolean>
  const imgFetchPending = new Map();   // request id -> resolve
  let imgFetchSeq = 0;
  let imageCore = { device: null, promise: null };

  function getImageCore() {
    return ensureGPU().then((gpu) => {
      if (!gpu) return null;
      if (imageCore.device !== gpu.device || !imageCore.promise) {
        imageCore.device = gpu.device;
        imageCore.promise = createCore(gpu.device, { texture2DSource: true }).catch((err) => {
          console.warn('[TrueHDR] image pipeline init failed:', err);
          imageCore.promise = null;
          return null;
        });
      }
      return imageCore.promise;
    });
  }

  function lruSet(map, key, value, max) {
    map.set(key, value);
    if (map.size > max) map.delete(map.keys().next().value);
    return value;
  }

  // Animated container detection from the leading bytes (no full decode).
  function sniffAnimated(b) {
    const n = b.length;
    const str = (o, len) => (o + len <= n ? String.fromCharCode.apply(null, b.subarray(o, o + len)) : '');
    if (str(0, 4) === 'GIF8' && n >= 13) {
      let p = 13;
      if (b[10] & 0x80) p += 3 * (1 << ((b[10] & 7) + 1));
      let frames = 0;
      while (p < n) {
        const t = b[p];
        if (t === 0x2c) {
          if (++frames > 1) return true;
          if (p + 10 > n) break;
          const packed = b[p + 9];
          p += 10;
          if (packed & 0x80) p += 3 * (1 << ((packed & 7) + 1));
          p += 1; // LZW minimum code size
          while (p < n && b[p] !== 0) p += b[p] + 1;
          p += 1;
        } else if (t === 0x21) {
          if (b[p + 1] === 0xff && b[p + 2] === 11) {
            const app = str(p + 3, 11);
            if (app === 'NETSCAPE2.0' || app === 'ANIMEXTS1.0') return true;
          }
          p += 2;
          while (p < n && b[p] !== 0) p += b[p] + 1;
          p += 1;
        } else {
          break;
        }
      }
      return false;
    }
    if (b[0] === 0x89 && str(1, 3) === 'PNG') {
      let p = 8;
      while (p + 8 <= n) {
        const len = ((b[p] << 24) | (b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3]) >>> 0;
        const type = str(p + 4, 4);
        if (type === 'acTL') return true;
        if (type === 'IDAT' || type === 'IEND') return false;
        p += 12 + len;
      }
      return false;
    }
    if (str(0, 4) === 'RIFF' && str(8, 4) === 'WEBP') {
      return str(12, 4) === 'VP8X' && n > 20 && (b[20] & 0x02) !== 0;
    }
    if (str(4, 4) === 'ftyp') {
      const size = Math.min(n, ((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]) >>> 0);
      for (let p = 8; p + 4 <= size; p += 4) {
        if (str(p, 4) === 'avis') return true;
      }
    }
    return false;
  }

  async function readUrlPrefix(url, limit) {
    const resp = await fetch(url, { cache: 'force-cache' });
    if (!resp.ok || !resp.body) return null;
    const reader = resp.body.getReader();
    const chunks = [];
    let total = 0;
    while (total < limit) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      total += value.length;
    }
    try { reader.cancel(); } catch (_) {}
    const out = new Uint8Array(Math.min(total, limit));
    let o = 0;
    for (const c of chunks) {
      const take = Math.min(c.length, out.length - o);
      out.set(c.subarray(0, take), o);
      o += take;
      if (o >= out.length) break;
    }
    return out;
  }

  function detectAnimated(url, blob) {
    if (!url || IMG_STATIC_EXT.test(url)) return Promise.resolve(false);
    let p = imgAnimCache.get(url);
    if (!p) {
      p = (async () => {
        const bytes = blob
          ? new Uint8Array(await blob.slice(0, IMG_SNIFF_BYTES).arrayBuffer())
          : await readUrlPrefix(url, IMG_SNIFF_BYTES);
        return bytes ? sniffAnimated(bytes) : IMG_ANIM_HINT.test(url);
      })().catch(() => IMG_ANIM_HINT.test(url));
      lruSet(imgAnimCache, url, p, 256);
    }
    return p;
  }

  function loadImageElement(src, cors) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      if (cors) im.crossOrigin = 'anonymous';
      im.decoding = 'async';
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error('image load failed'));
      im.src = src;
    });
  }

  function canReadPixels(im) {
    try {
      const c = document.createElement('canvas');
      c.width = 1;
      c.height = 1;
      const g = c.getContext('2d');
      g.drawImage(im, 0, 0, 1, 1);
      g.getImageData(0, 0, 1, 1);
      return true;
    } catch (_) {
      return false;
    }
  }

  function fetchImageViaExtension(url) {
    return new Promise((resolve) => {
      const id = ++imgFetchSeq;
      const timer = setTimeout(() => {
        imgFetchPending.delete(id);
        resolve(null);
      }, 20000);
      imgFetchPending.set(id, (payload) => {
        clearTimeout(timer);
        resolve(payload && payload.ok && payload.blob instanceof Blob ? payload.blob : null);
      });
      postToBridge('FETCH_IMAGE', { id, url });
    });
  }

  // Pixel-readable source for a cross-origin <img>: first a CORS re-request (standard web
  // security, works on most CDNs), then the extension's cookie-less fetch.
  function resolveTaintedSource(url) {
    let p = imgSourceCache.get(url);
    if (!p) {
      p = (async () => {
        if (/^https?:/i.test(url)) {
          try {
            const im = await loadImageElement(url, true);
            if (canReadPixels(im)) return { kind: 'cors' };
          } catch (_) {}
          const blob = await fetchImageViaExtension(url);
          if (blob) return { kind: 'blob', blob };
        }
        return null;
      })();
      lruSet(imgSourceCache, url, p, 48);
    }
    return p;
  }

  function parsePosToken(tok, boxLen, contentLen) {
    const kw = { left: 0, top: 0, center: 0.5, right: 1, bottom: 1 };
    if (tok in kw) return (boxLen - contentLen) * kw[tok];
    if (tok.endsWith('%')) return (boxLen - contentLen) * (parseFloat(tok) / 100);
    if (tok.endsWith('px')) return parseFloat(tok);
    return (boxLen - contentLen) * 0.5;
  }

  // Content rect of the image inside its CSS box, normalized (ox, oy, w, h); w/h > 1 = cropped.
  function imageContentRect(boxW, boxH, nw, nh, fit, position) {
    if (!fit || fit === 'fill' || boxW <= 0 || boxH <= 0) return [0, 0, 1, 1];
    const ar = nw / nh;
    const containW = boxW / boxH > ar ? boxH * ar : boxW;
    const containH = containW / ar;
    let cw;
    let ch;
    if (fit === 'cover') {
      cw = boxW / boxH > ar ? boxW : boxH * ar;
      ch = cw / ar;
    } else if (fit === 'none') {
      cw = nw;
      ch = nh;
    } else if (fit === 'scale-down' && nw <= containW) {
      cw = nw;
      ch = nh;
    } else {
      cw = containW;
      ch = containH;
    }
    const toks = String(position || '50% 50%').trim().split(/\s+/);
    let px = toks[0] || 'center';
    let py = toks[1] || 'center';
    if (toks.length === 1 && (px === 'top' || px === 'bottom')) { py = px; px = 'center'; }
    if (px === 'top' || px === 'bottom' || py === 'left' || py === 'right') { const t = px; px = py; py = t; }
    const ox = toks.length <= 2 ? parsePosToken(px, boxW, cw) : (boxW - cw) * 0.5;
    const oy = toks.length <= 2 ? parsePosToken(py, boxH, ch) : (boxH - ch) * 0.5;
    return [ox / boxW, oy / boxH, cw / boxW, ch / boxH];
  }

  class ImageHDRProcessor {
    constructor(img) {
      this.img = img;
      this.canvas = null;
      this.ctx = null;
      this.renderer = null;
      this.core = null;
      this.srcTex = null;
      this.srcView = null;
      this.srcW = 0;
      this.srcH = 0;
      this.scratch = null;
      this.scratch2d = null;
      this.proxy = null;
      this.proxyUrl = null;
      this.sourceBlob = null;
      this.sourceKey = '';
      this.sourceState = 'none'; // 'none' | 'direct' | 'resolving' | 'proxy' | 'blocked'
      this.sourceToken = 0;
      this.forceProxy = false;
      this.animated = false;
      this.inView = false;
      this.active = false;
      this.plan = null;
      this.lastKey = '';
      this.configuredW = 0;
      this.configuredH = 0;
      this.lastRenderMs = 0;
      this.frameSeed = 0;
      this.needReset = true;
      this.inFlight = 0;
      this.renderQueued = false;
      this.gen = 0;
      this.destroyed = false;

      this.onLoad = () => {
        this.resetSource();
        if (this.active) this.prepare();
        else scheduleImagePump();
      };
      img.addEventListener('load', this.onLoad);
      this.io = new IntersectionObserver((entries) => {
        for (const e of entries) this.inView = e.isIntersecting;
        scheduleImagePump();
      }, { rootMargin: '200px 0px', threshold: 0 });
      this.io.observe(img);
      this.ro = new ResizeObserver(() => this.requestRender());
      this.ro.observe(img);
    }

    resetSource() {
      this.sourceToken++;
      this.dropProxy();
      this.sourceKey = '';
      this.sourceState = 'none';
      this.sourceBlob = null;
      this.forceProxy = false;
      this.animated = false;
      this.needReset = true;
      this.lastKey = '';
    }

    dropProxy() {
      if (this.proxy && this.proxy.parentElement) this.proxy.remove();
      this.proxy = null;
      if (this.proxyUrl) {
        try { URL.revokeObjectURL(this.proxyUrl); } catch (_) {}
        this.proxyUrl = null;
      }
    }

    // On-screen area when this image should get an overlay right now, else 0.
    eligibleArea() {
      const img = this.img;
      if (this.destroyed || !img.isConnected || this.sourceState === 'blocked') return 0;
      if (!img.complete || img.naturalWidth < IMG_MIN_NATURAL || img.naturalHeight < IMG_MIN_NATURAL) return 0;
      const r = img.getBoundingClientRect();
      if (r.width < IMG_MIN_CSS_W || r.height < IMG_MIN_CSS_H) return 0;
      const cs = getComputedStyle(img);
      // Fades, hidden carousel slides and hover zooms act on the <img> itself and would not
      // apply to the sibling overlay: show the original until it settles.
      if (cs.visibility !== 'visible' || parseFloat(cs.opacity) < 0.99 || cs.transform !== 'none' || cs.display === 'none') return 0;
      if (cs.filter && cs.filter !== 'none') return 0;
      return r.width * r.height;
    }

    plannedPixels() {
      const r = this.img.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      return Math.min(IMG_MAX_CANVAS_PIXELS, r.width * r.height * dpr * dpr);
    }

    ensureCanvas() {
      if (this.canvas) return;
      const canvas = document.createElement('canvas');
      canvas.setAttribute('data-truehdr', 'img');
      canvas.setAttribute('aria-hidden', 'true');
      canvas.tabIndex = -1;
      canvas.inert = true;
      Object.assign(canvas.style, { position: 'absolute', margin: '0', padding: '0', border: '0', background: 'transparent', display: 'none' });
      canvas.style.setProperty('pointer-events', 'none', 'important');
      canvas.style.setProperty('user-select', 'none', 'important');
      canvas.style.setProperty('-webkit-user-select', 'none', 'important');
      this.canvas = canvas;
      this.ctx = null;
      this.configuredW = 0;
      this.configuredH = 0;
    }

    anchor() {
      const parent = this.img.parentElement;
      return parent && parent.tagName === 'PICTURE' ? parent : this.img;
    }

    configure() {
      if (!this.ctx && this.canvas) this.ctx = this.canvas.getContext('webgpu');
      if (!this.ctx || !this.core) return;
      try {
        this.ctx.configure({
          device: this.core.device,
          format: 'rgba16float',
          colorSpace: 'display-p3',
          toneMapping: { mode: 'extended' },
          alphaMode: 'premultiplied'
        });
      } catch (err) {
        console.warn('[TrueHDR] Image context configure failed:', err);
      }
    }

    activate() {
      if (this.active || this.destroyed) return;
      this.active = true;
      this.gen++;
      this.ensureCanvas();
      this.needReset = true;
      this.prepare();
    }

    deactivate() {
      if (!this.active) return;
      this.active = false;
      this.gen++;
      if (this.canvas) this.canvas.style.display = 'none';
      if (this.ctx) {
        try { this.ctx.unconfigure(); } catch (_) {}
      }
      this.configuredW = 0;
      this.configuredH = 0;
      this.lastKey = '';
      if (this.proxy && this.proxy.parentElement) this.proxy.remove();
      this.releaseGpu();
      if (this.scratch) {
        this.scratch.width = 0;
        this.scratch.height = 0;
        this.scratch2d = null;
      }
    }

    releaseGpu() {
      if (this.renderer) {
        try { this.renderer.destroy(); } catch (_) {}
      }
      this.renderer = null;
      if (this.srcTex) {
        try { this.srcTex.destroy(); } catch (_) {}
      }
      this.srcTex = null;
      this.srcView = null;
      this.srcW = 0;
      this.srcH = 0;
      this.inFlight = 0;
    }

    async prepare() {
      const gen = this.gen;
      const core = await getImageCore();
      if (!core || gen !== this.gen || !this.active) return;
      if (this.core !== core) {
        this.releaseGpu();
        this.core = core;
        if (this.ctx) {
          try { this.ctx.unconfigure(); } catch (_) {}
        }
        this.configuredW = 0;
        this.configuredH = 0;
      }
      if (!this.renderer) this.renderer = new FrameRenderer(core);
      const ok = await this.ensureSource();
      if (!ok || gen !== this.gen || !this.active) return;
      this.lastKey = '';
      this.renderNow();
      const url = this.sourceKey;
      const animated = await detectAnimated(url, this.sourceBlob);
      if (gen !== this.gen || !this.active || url !== this.sourceKey) return;
      this.animated = animated;
      if (animated) {
        this.mountProxy();
        startImageAnimation();
      }
    }

    async ensureSource() {
      const img = this.img;
      const url = img.currentSrc || img.src || '';
      if (!url) return false;
      if (this.sourceKey === url) {
        if (this.sourceState === 'direct' || this.sourceState === 'proxy') return true;
        if (this.sourceState === 'blocked') return false;
      }
      const token = ++this.sourceToken;
      this.dropProxy();
      this.sourceKey = url;
      this.sourceBlob = null;
      if (!this.forceProxy && canReadPixels(img)) {
        this.sourceState = 'direct';
        return true;
      }
      this.sourceState = 'resolving';
      const res = await resolveTaintedSource(url);
      if (token !== this.sourceToken || this.destroyed) return false;
      if (!res) {
        this.sourceState = 'blocked';
        scheduleImagePump();
        return false;
      }
      let proxy = null;
      let proxyUrl = null;
      try {
        if (res.kind === 'cors') {
          proxy = await loadImageElement(url, true);
        } else {
          proxyUrl = URL.createObjectURL(res.blob);
          proxy = await loadImageElement(proxyUrl, false);
        }
      } catch (_) {
        proxy = null;
      }
      if (token !== this.sourceToken || this.destroyed) {
        if (proxyUrl) URL.revokeObjectURL(proxyUrl);
        return false;
      }
      if (!proxy) {
        if (proxyUrl) URL.revokeObjectURL(proxyUrl);
        this.sourceState = 'blocked';
        scheduleImagePump();
        return false;
      }
      this.proxy = proxy;
      this.proxyUrl = proxyUrl;
      this.sourceBlob = res.kind === 'blob' ? res.blob : null;
      this.sourceState = 'proxy';
      return true;
    }

    // WebKit only advances GIF/APNG animation for images in the render tree, so an animated
    // proxy sits (invisibly) under the overlay.
    mountProxy() {
      const px = this.proxy;
      if (!px || !this.canvas || !this.canvas.parentElement) return;
      if (px.parentElement !== this.canvas.parentElement || px.nextSibling !== this.canvas) {
        px.setAttribute('data-truehdr', 'proxy');
        px.setAttribute('aria-hidden', 'true');
        px.alt = '';
        Object.assign(px.style, { position: 'absolute', margin: '0', padding: '0', border: '0', opacity: '0.01' });
        px.style.setProperty('pointer-events', 'none', 'important');
        this.canvas.insertAdjacentElement('beforebegin', px);
      }
      for (const k of ['left', 'top', 'width', 'height', 'zIndex']) px.style[k] = this.canvas.style[k];
    }

    // Overlay geometry; returns false when the overlay must stay hidden.
    syncLayout() {
      const img = this.img;
      const canvas = this.canvas;
      const anchor = this.anchor();
      if (!anchor.parentElement) return false;
      if (canvas.parentElement !== anchor.parentElement) {
        anchor.insertAdjacentElement('afterend', canvas);
        if (this.animated) this.mountProxy();
      }
      const r = img.getBoundingClientRect();
      if (r.width < IMG_MIN_CSS_W || r.height < IMG_MIN_CSS_H) return false;
      const cb = canvas.offsetParent || document.documentElement;
      let left;
      let top;
      if (cb === document.documentElement || cb === document.body) {
        left = r.left + window.scrollX;
        top = r.top + window.scrollY;
      } else {
        const cbRect = cb.getBoundingClientRect();
        left = r.left - cbRect.left - cb.clientLeft + cb.scrollLeft;
        top = r.top - cbRect.top - cb.clientTop + cb.scrollTop;
      }
      // Ancestors between the image and the overlay's containing block may clip the image
      // (overflow) but would not clip an absolutely positioned overlay: replicate as clip-path.
      let cl = r.left;
      let ct = r.top;
      let cr = r.right;
      let cbt = r.bottom;
      for (let el = img.parentElement; el && el !== cb && el !== document.body && el !== document.documentElement; el = el.parentElement) {
        const s = getComputedStyle(el);
        if (s.overflowX !== 'visible' || s.overflowY !== 'visible') {
          const er = el.getBoundingClientRect();
          cl = Math.max(cl, er.left);
          ct = Math.max(ct, er.top);
          cr = Math.min(cr, er.right);
          cbt = Math.min(cbt, er.bottom);
        }
      }
      if (cr - cl < 4 || cbt - ct < 4) return false;
      const clip = [ct - r.top, r.right - cr, r.bottom - cbt, cl - r.left].map((v) => Math.max(0, v));
      const clipCss = clip.some((v) => v > 0.5) ? `inset(${clip.map((v) => `${v.toFixed(1)}px`).join(' ')})` : '';

      const cs = getComputedStyle(img);
      const nw = img.naturalWidth;
      const nh = img.naturalHeight;
      const rect = imageContentRect(r.width, r.height, nw, nh, cs.objectFit, cs.objectPosition);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const maxDim = this.core.device.limits.maxTextureDimension2D;
      let cw = r.width * dpr;
      let ch = r.height * dpr;
      const k = Math.min(1, Math.sqrt(IMG_MAX_CANVAS_PIXELS / Math.max(cw * ch, 1)), maxDim / cw, maxDim / ch);
      cw = Math.max(1, Math.round(cw * k));
      ch = Math.max(1, Math.round(ch * k));
      const outW = Math.max(1, Math.round(cw * rect[2]));
      const outH = Math.max(1, Math.round(ch * rect[3]));
      const ks = Math.min(1, nw / outW, nh / outH, Math.sqrt(IMG_MAX_SOURCE_PIXELS / (outW * outH)), maxDim / outW, maxDim / outH);
      const sw = Math.max(1, Math.round(outW * ks));
      const sh = Math.max(1, Math.round(outH * ks));
      this.plan = { rect, outW, outH, sw, sh };

      const key = `${left.toFixed(1)},${top.toFixed(1)},${r.width.toFixed(1)},${r.height.toFixed(1)},${cw},${ch},${cs.zIndex},${cs.borderRadius},${clipCss},${rect.join(',')}`;
      if (key !== this.lastKey) {
        this.lastKey = key;
        Object.assign(canvas.style, {
          left: `${left}px`,
          top: `${top}px`,
          width: `${r.width}px`,
          height: `${r.height}px`,
          zIndex: cs.zIndex,
          borderRadius: cs.borderRadius,
          clipPath: clipCss
        });
        if (canvas.width !== cw || canvas.height !== ch || this.configuredW === 0) {
          canvas.width = cw;
          canvas.height = ch;
          this.configuredW = cw;
          this.configuredH = ch;
          this.configure();
        }
        if (this.animated) this.mountProxy();
      }
      return true;
    }

    uploadSource(device) {
      const { sw, sh } = this.plan;
      const src = this.proxy || this.img;
      if (!this.scratch) this.scratch = document.createElement('canvas');
      if (this.scratch.width !== sw || this.scratch.height !== sh || !this.scratch2d) {
        this.scratch.width = sw;
        this.scratch.height = sh;
        this.scratch2d = this.scratch.getContext('2d');
      }
      const g = this.scratch2d;
      g.clearRect(0, 0, sw, sh);
      g.imageSmoothingEnabled = true;
      g.imageSmoothingQuality = 'high';
      g.drawImage(src, 0, 0, sw, sh);
      if (!this.srcTex || this.srcW !== sw || this.srcH !== sh) {
        if (this.srcTex) {
          try { this.srcTex.destroy(); } catch (_) {}
        }
        this.srcTex = device.createTexture({
          size: [sw, sh, 1],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });
        this.srcView = this.srcTex.createView();
        this.srcW = sw;
        this.srcH = sh;
        this.needReset = true;
      }
      device.queue.copyExternalImageToTexture({ source: this.scratch }, { texture: this.srcTex, premultipliedAlpha: true }, [sw, sh]);
    }

    requestRender() {
      if (!this.active) {
        scheduleImagePump();
        return;
      }
      if (this.renderQueued) return;
      this.renderQueued = true;
      requestAnimationFrame(() => {
        this.renderQueued = false;
        this.renderNow();
      });
    }

    renderNow() {
      if (!this.active || this.destroyed || !this.renderer || !this.core || !this.canvas) return;
      if (this.sourceState !== 'direct' && this.sourceState !== 'proxy') return;
      if (this.inFlight >= 2) return;
      if (this.eligibleArea() <= 0 || !this.syncLayout()) {
        this.canvas.style.display = 'none';
        return;
      }
      const device = this.core.device;
      try {
        this.uploadSource(device);
      } catch (err) {
        if (err && err.name === 'SecurityError' && !this.forceProxy) {
          this.forceProxy = true;
          this.sourceKey = '';
          this.sourceState = 'none';
          this.prepare();
        }
        return;
      }
      if (!this.ctx) this.configure();
      let swapTex;
      try {
        swapTex = this.ctx.getCurrentTexture();
      } catch (_) {
        this.configure();
        try { swapTex = this.ctx.getCurrentTexture(); } catch (e2) { return; }
      }
      const now = performance.now();
      const dt = this.lastRenderMs > 0 ? (now - this.lastRenderMs) / 1000 : 1 / 30;
      this.lastRenderMs = now;
      this.frameSeed = (this.frameSeed + 1) % 256;
      const p = this.plan;
      const encoder = device.createCommandEncoder();
      try {
        this.renderer.encode(encoder, this.srcView, swapTex.createView(), {
          srcW: this.srcW,
          srcH: this.srcH,
          rect: p.rect,
          outW: p.outW,
          outH: p.outH,
          tone: [settings.peakBoost, settings.threshold, settings.localPunch, settings.gamutBoost],
          headroomCap: appInactive ? SUPPRESSED_HEADROOM : 0,
          shadowDepth: settings.shadowDepth,
          aplGuard: settings.aplGuard,
          splitPos: settings.splitEnabled ? settings.splitPos : -1.0,
          hdr: true,
          bypassNativeHdr: false,
          sharpDirect: false,
          sharpness: 0,
          mode: 'image',
          // Static images: every render is a fresh scene. Animations keep temporal smoothing.
          reset: this.needReset || !this.animated,
          dt,
          frameSeed: this.frameSeed
        });
      } catch (err) {
        console.warn('[TrueHDR] Image encode failed:', err);
        return;
      }
      this.needReset = false;
      device.queue.submit([encoder.finish()]);
      this.inFlight++;
      const done = () => { this.inFlight = Math.max(0, this.inFlight - 1); };
      device.queue.onSubmittedWorkDone().then(done, done);
      if (this.canvas.style.display !== 'block') this.canvas.style.display = 'block';
    }

    destroy() {
      if (this.destroyed) return;
      this.deactivate();
      this.destroyed = true;
      this.io.disconnect();
      this.ro.disconnect();
      this.img.removeEventListener('load', this.onLoad);
      this.dropProxy();
      if (this.canvas && this.canvas.parentElement) this.canvas.remove();
      this.canvas = null;
      this.ctx = null;
      images.delete(this.img);
    }
  }

  let imgPumpQueued = false;
  function scheduleImagePump() {
    if (imgPumpQueued) return;
    imgPumpQueued = true;
    requestAnimationFrame(() => {
      imgPumpQueued = false;
      pumpImages();
    });
  }

  // Decide which images get overlays: largest visible first, within count + pixel budgets.
  function pumpImages() {
    const enabled = Boolean(settings.effectiveImages) && !document.hidden;
    const cands = [];
    for (const p of Array.from(images.values())) {
      if (!p.img.isConnected) {
        p.destroy();
        continue;
      }
      const area = enabled && p.inView ? p.eligibleArea() : 0;
      if (area > 0) cands.push([area, p]);
      else p.deactivate();
    }
    cands.sort((a, b) => b[0] - a[0]);
    let budget = IMG_PIXEL_BUDGET;
    let n = 0;
    for (const [, p] of cands) {
      const px = p.plannedPixels();
      if (n < IMG_MAX_ACTIVE && px <= budget) {
        budget -= px;
        n++;
        if (p.active) p.requestRender();
        else p.activate();
      } else {
        p.deactivate();
      }
    }
    updateImageTicker();
    startImageAnimation();
    broadcastStatus();
  }

  // Static overlays are re-checked once a second for layout shifts the observers miss.
  let imgTicker = null;
  function updateImageTicker() {
    let any = false;
    for (const p of images.values()) if (p.active) { any = true; break; }
    if (any && !imgTicker) {
      imgTicker = setInterval(() => {
        if (document.hidden) return;
        for (const p of images.values()) {
          if (!p.active) continue;
          if (!p.img.isConnected || p.eligibleArea() <= 0) {
            scheduleImagePump();
            continue;
          }
          const before = p.lastKey;
          if (p.canvas && p.syncLayout() && p.lastKey !== before && !p.animated) p.renderNow();
        }
      }, 1000);
    } else if (!any && imgTicker) {
      clearInterval(imgTicker);
      imgTicker = null;
    }
  }

  let imgAnimRaf = null;
  function startImageAnimation() {
    if (imgAnimRaf !== null || document.hidden) return;
    for (const p of images.values()) {
      if (p.active && p.animated) {
        imgAnimRaf = requestAnimationFrame(imageAnimationTick);
        return;
      }
    }
  }

  function imageAnimationTick(now) {
    imgAnimRaf = null;
    if (document.hidden) return;
    const anim = [];
    for (const p of images.values()) if (p.active && p.animated) anim.push(p);
    if (anim.length === 0) return;
    anim.sort((a, b) => b.plannedPixels() - a.plannedPixels());
    for (const p of anim.slice(0, IMG_MAX_ANIMATED)) {
      if (now - p.lastRenderMs >= IMG_ANIM_INTERVAL_MS - 2) p.renderNow();
    }
    imgAnimRaf = requestAnimationFrame(imageAnimationTick);
  }

  function attachImage(img) {
    if (!(img instanceof HTMLImageElement) || images.has(img) || img.hasAttribute('data-truehdr')) return;
    images.set(img, new ImageHDRProcessor(img));
  }

  function scanImages(root = document) {
    if (!settings.effectiveImages || !root || !root.querySelectorAll) return;
    for (const img of root.querySelectorAll('img')) attachImage(img);
    scheduleImagePump();
  }

  function destroyAllImages() {
    for (const p of Array.from(images.values())) p.destroy();
    updateImageTicker();
  }

  function attachVideo(video) {
    if (!video || !(video instanceof HTMLVideoElement)) return;
    if (instances.has(video)) return;
    instances.set(video, new VideoHDRProcessor(video));
  }

  function scanVideos(root = document) {
    if (!root || !root.querySelectorAll) return;
    const videos = root.querySelectorAll('video');
    for (const v of videos) {
      attachVideo(v);
    }
  }

  let statusTimer = null;
  function broadcastStatus() {
    clearTimeout(statusTimer);
    statusTimer = setTimeout(() => {
      let activeCount = 0;
      let bypassedNativeHdr = false;
      let blockedReason = null;
      let videoCount = 0;
      let sourceRes = null;
      let outputRes = null;
      let upscalingActive = false;
      let srModel = null;
      let srPasses = 0;
      let srFallback = null;
      let gpuMs = null;

      for (const inst of instances.values()) {
        if (!inst.video.isConnected) {
          inst.destroy();
          continue;
        }
        videoCount++;
        if (inst.isActive) {
          activeCount++;
          if (!sourceRes && inst.video.videoWidth > 0) {
            sourceRes = `${inst.video.videoWidth}×${inst.video.videoHeight}`;
            outputRes = `${inst.contentOutW}×${inst.contentOutH}`;
            upscalingActive = inst.upscalingActive;
            srModel = inst.upscalingActive ? inst.srModel : null;
            srPasses = inst.upscalingActive && inst.srModel ? inst.srPasses : 0;
            srFallback = inst.upscalingActive ? inst.srFallback : null;
            gpuMs = typeof inst.gpuMs === 'number' ? Math.round(inst.gpuMs * 100) / 100 : null;
          }
        }
        if (inst.nativeHdrDetected && settings.bypassNativeHdr) bypassedNativeHdr = true;
        if (!blockedReason && inst.blockedReason) blockedReason = inst.blockedReason;
      }
      let imageActiveCount = 0;
      let imageAnimatedCount = 0;
      for (const p of images.values()) {
        if (p.active) {
          imageActiveCount++;
          if (p.animated) imageAnimatedCount++;
        }
      }

      postToBridge('STATUS', {
        host: location.hostname || '',
        isTopFrame: window.top === window,
        isEdrDisplay,
        hasWebGPU: Boolean(navigator.gpu),
        gpuStatus: gpuState.status,
        videoCount,
        activeCount,
        effectiveHdr: Boolean(settings.effectiveHdr),
        effectiveUpscale: Boolean(settings.effectiveUpscale),
        bypassedNativeHdr,
        blockedReason,
        sourceRes,
        outputRes,
        upscalingActive,
        srModel,
        srPasses,
        srFallback,
        gpuMs,
        effectiveImages: Boolean(settings.effectiveImages),
        imageActiveCount,
        imageAnimatedCount
      });
    }, 80);
  }

  // Watch DOM for dynamically added/removed videos (YouTube SPA, Shorts, embeds)
  const domObserver = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node instanceof HTMLVideoElement) {
          attachVideo(node);
        } else if (node instanceof Element && (node.tagName === 'VIDEO' || node.querySelector('video'))) {
          scanVideos(node);
        }
        if (settings.effectiveImages) {
          if (node instanceof HTMLImageElement) {
            attachImage(node);
            scheduleImagePump();
          } else if (node instanceof Element && node.getAttribute('data-truehdr') === null && node.querySelector('img')) {
            scanImages(node);
          }
        }
      }
      if (m.removedNodes.length && images.size) scheduleImagePump();
      for (const node of m.removedNodes) {
        if (node instanceof HTMLVideoElement && instances.has(node)) {
          instances.get(node).destroy();
        } else if (node instanceof Element) {
          for (const v of instances.keys()) {
            if (!v.isConnected) {
              instances.get(v).destroy();
            }
          }
        }
      }
    }
  });

  function startObserving() {
    scanVideos(document);
    const target = document.documentElement || document;
    domObserver.observe(target, { childList: true, subtree: true });
  }

  if (document.documentElement) {
    startObserving();
  } else {
    window.addEventListener('DOMContentLoaded', startObserving, { once: true });
  }

  document.addEventListener(
    'playing',
    (e) => {
      if (e.target instanceof HTMLVideoElement) {
        attachVideo(e.target);
      }
    },
    true
  );

  window.addEventListener('yt-navigate-finish', () => {
    setTimeout(() => {
      scanVideos(document);
      for (const inst of instances.values()) {
        inst.checkVideoMetadata();
      }
    }, 150);
  });

  document.addEventListener('visibilitychange', () => {
    if (images.size) scheduleImagePump();
    for (const inst of instances.values()) {
      if (document.hidden) {
        inst.stopLoop();
      } else if (inst.isVisible && settings.effectiveEnabled) {
        inst.lastLayoutCheckMs = 0;
        inst.dirtyRender = true;
        inst.scheduleRenderOnce();
        if (!inst.video.paused && !inst.video.ended) {
          inst.scheduleLoop();
        }
      }
    }
  });

  for (const evt of ['fullscreenchange', 'webkitfullscreenchange', 'resize']) {
    document.addEventListener(evt, () => {
      if (images.size) scheduleImagePump();
      for (const inst of instances.values()) {
        if (inst.syncOverlayLayout(true)) {
          inst.dirtyRender = true;
          inst.scheduleRenderOnce();
        }
      }
    });
  }

  function setAppInactive(v) {
    if (v === appInactive) return;
    appInactive = v;
    for (const inst of instances.values()) {
      inst.dirtyRender = true;
      inst.scheduleRenderOnce();
    }
    for (const p of images.values()) p.requestRender();
  }

  // Focused page => Safari is active (no query). Otherwise ask the bridge; the background caches
  // the native answer, so many frames polling cost one native round trip per 250 ms at most.
  function refreshAppState() {
    if (!instances.size && !images.size) return;
    if (document.hasFocus()) {
      setAppInactive(false);
      return;
    }
    postToBridge('QUERY_APP_STATE', {});
  }
  window.addEventListener('blur', refreshAppState);
  window.addEventListener('focus', refreshAppState);
  setInterval(refreshAppState, 500);

  const SHORTCUTS = {
    KeyM: ['TOGGLE_MASTER', () => `TrueHDR: ${!settings.enabled ? 'ON' : 'OFF'}`],
    KeyH: ['TOGGLE_HDR', () => (!settings.hdrEnabled ? `HDR: ON (${settings.peakBoost.toFixed(1)}x EDR)` : 'HDR: OFF')],
    KeyU: ['TOGGLE_UPSCALE', () => `Upscaling: ${!settings.upscaleEnabled ? 'ON' : 'OFF'}`],
    KeyS: ['TOGGLE_SPLIT', () => `Split View: ${!settings.splitEnabled ? 'SDR | EDR' : 'OFF'}`],
    KeyI: ['TOGGLE_IMAGES', () => `Image HDR: ${!settings.imagesEnabled ? 'ON' : 'OFF'}`]
  };

  window.addEventListener(
    'keydown',
    (e) => {
      const tag = e.target && e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) return;
      if (!e.altKey || e.metaKey || e.ctrlKey || e.shiftKey) return;
      const entry = SHORTCUTS[e.code];
      if (!entry) return;
      e.preventDefault();
      postToBridge(entry[0], {});
      const msg = entry[1]();
      let shown = false;
      for (const inst of instances.values()) {
        if (inst.video.videoWidth > 0) {
          showHudToast(inst.video, msg);
          shown = true;
          break;
        }
      }
      if (!shown && window.top === window) showHudToast(null, msg);
    },
    true
  );

  window.addEventListener('message', (event) => {
    if (event.source !== window) return;
    const data = event.data;
    if (!data || data.channel !== CHANNEL || data.from !== 'bridge') return;

    if (data.kind === 'SETTINGS') {
      const prevEnabled = settings.effectiveEnabled;
      const prevHdr = settings.effectiveHdr;
      const prevUp = settings.effectiveUpscale;
      const prevSrCfg = `${settings.upscaleQuality}|${settings.upscaleContent}`;
      const incoming = { ...DEFAULT_SETTINGS, ...(data.payload || {}) };
      // Legacy stored setting: upscaleMode 'off' meant upscaling disabled (the mode itself is ignored)
      if (incoming.upscaleMode === 'off' && (!data.payload || typeof data.payload.upscaleEnabled !== 'boolean')) {
        incoming.upscaleEnabled = false;
      }
      if (incoming.upscaleQuality !== 'fast' && incoming.upscaleQuality !== 'quality') {
        incoming.upscaleQuality = DEFAULT_SETTINGS.upscaleQuality;
      }
      if (incoming.upscaleContent !== 'live' && incoming.upscaleContent !== 'anime') {
        incoming.upscaleContent = DEFAULT_SETTINGS.upscaleContent;
      }
      if (!data.payload || typeof data.payload.effectiveHdr !== 'boolean') {
        incoming.effectiveHdr = Boolean(incoming.enabled && incoming.hdrEnabled);
      }
      if (!data.payload || typeof data.payload.effectiveUpscale !== 'boolean') {
        incoming.effectiveUpscale = Boolean(incoming.enabled && incoming.upscaleEnabled);
      }
      if (!data.payload || typeof data.payload.effectiveEnabled !== 'boolean') {
        incoming.effectiveEnabled = Boolean(incoming.effectiveHdr || incoming.effectiveUpscale);
      }
      if (!data.payload || typeof data.payload.effectiveImages !== 'boolean') {
        incoming.effectiveImages = Boolean(incoming.effectiveHdr && incoming.imagesEnabled);
      }
      const prevImages = Boolean(settings.effectiveImages);
      settings = incoming;
      if (settings.effectiveImages && !prevImages) {
        scanImages(document);
      } else if (!settings.effectiveImages && prevImages) {
        destroyAllImages();
      } else if (settings.effectiveImages) {
        for (const p of images.values()) p.requestRender();
      }
      const srCfgChanged = prevSrCfg !== `${settings.upscaleQuality}|${settings.upscaleContent}`;
      const needsLayout = srCfgChanged || prevEnabled !== settings.effectiveEnabled || prevHdr !== settings.effectiveHdr || prevUp !== settings.effectiveUpscale;
      for (const inst of instances.values()) {
        if (srCfgChanged) inst.resetSRGuard();
        inst.dirtyRender = true;
        inst.syncOverlayLayout(needsLayout);
        inst.syncSplitHandle();
        if (settings.effectiveEnabled) {
          if (needsLayout) {
            inst.renderFrame();
          }
          inst.scheduleRenderOnce();
          if (!inst.video.paused && !inst.video.ended) {
            inst.scheduleLoop();
          }
        } else if (prevEnabled) {
          inst.stopLoop();
        }
      }
      broadcastStatus();
    } else if (data.kind === 'IMAGE_DATA') {
      const payload = data.payload || {};
      const resolve = imgFetchPending.get(payload.id);
      if (resolve) {
        imgFetchPending.delete(payload.id);
        resolve(payload);
      }
    } else if (data.kind === 'APP_STATE') {
      const payload = data.payload || {};
      setAppInactive(!document.hasFocus() && payload.inactive !== false);
    } else if (data.kind === 'RETRY_CORS') {
      for (const inst of instances.values()) {
        if (inst.blockedReason && inst.blockedReason.includes('CORS')) {
          inst.retryWithCors();
        }
      }
    }
  });

  postToBridge('READY', {});
})();
