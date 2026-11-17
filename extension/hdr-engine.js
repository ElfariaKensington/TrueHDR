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

    // Decoded R'G'B' -> linear light. Pixels already above SDR white (native PQ/HLG decoded to
    // extended range) are treated as linear.
    fn decodeLinear(raw: vec3f) -> vec3f {
      let r = max(raw, vec3f(0.0));
      return select(srgbToLinear(r), r, max(r.r, max(r.g, r.b)) > 1.02);
    }

    fn luma709(c: vec3f) -> f32 {
      return dot(max(c, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
    }

    struct FrameU {
      rect: vec4f,   // content rect in canvas uv (ox, oy, w, h)
      tone: vec4f,   // peakBoost, threshold, localPunch, gamutBoost
      extra: vec4f,  // shadowDepth, aplGuard, splitPos (<0 = off), flags (1 = HDR, 2 = bypass native HDR)
      info: vec4f,   // srcW, srcH, frameSeed, reset
      res: vec4f,    // resample texture W, H, sharpness, unused
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
          let raw = max(${tap}, vec3f(0.0));
          let mx = max(raw.r, max(raw.g, raw.b));
          let lin = decodeLinear(raw);
          let Y = luma709(lin);
          sumY = sumY + Y;
          hdr = hdr + select(0.0, 1.0, mx > 1.03);
          q[i] = log2(max(mix(Y, max(lin.r, max(lin.g, lin.b)), 0.46), 1.0 / 16384.0));
          pv[i] = mix(luma709(raw), mx, 0.46);
        }
        // Strong perceptual transitions between neighbouring quads (24 pairs)
        var edges = 0.0;
        for (var r = 0; r < 4; r = r + 1) {
          for (var c = 0; c < 3; c = c + 1) {
            edges = edges + step(0.22, abs(pv[r * 4 + c] - pv[r * 4 + c + 1]));
            edges = edges + step(0.22, abs(pv[c * 4 + r] - pv[c * 4 + r + 4]));
          }
        }
        // Robust cell level: median of the four 4x4-pixel group medians (median-of-4 = mean of the
        // middle two). Rejects small objects (birds, specks) whatever their contrast.
        let g0 = med4(q[0], q[1], q[4], q[5]);
        let g1 = med4(q[2], q[3], q[6], q[7]);
        let g2 = med4(q[8], q[9], q[12], q[13]);
        let g3 = med4(q[10], q[11], q[14], q[15]);
        let cellY = sumY / 16.0;
        textureStore(cellOut, vec2i(gid.xy), vec4f(cellY, med4(g0, g1, g2, g3), edges / 24.0, hdr / 16.0));
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

  // Local adaptation map at cell resolution (8x8 cells per workgroup, 14x10 shared tile).
  const MAP_CS = /* wgsl */ `
    ${WGSL_COMMON}
    @group(0) @binding(0) var cellTex: texture_2d<f32>;
    @group(0) @binding(1) var prevMap: texture_2d<f32>;
    @group(0) @binding(2) var mapOut: texture_storage_2d<rgba16float, write>;
    @group(0) @binding(3) var<storage, read> st: State;
    @group(0) @binding(4) var<uniform> u: FrameU;

    var<workgroup> tM: array<f32, 140>;
    var<workgroup> tD: array<f32, 140>;

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
      let d = (m - med) / 0.6;
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
      let origin = vec2i(wid.xy) * 8 - vec2i(3, 1);
      for (var k = li; k < 140u; k = k + 64u) {
        let t = textureLoad(cellTex, clamp(origin + vec2i(i32(k % 14u), i32(k / 14u)), vec2i(0), mxc), 0);
        tM[k] = t.y;
        tD[k] = t.z;
      }
      workgroupBarrier();
      let c = vec2i(wid.xy) * 8 + vec2i(lid.xy);
      if (c.x >= dims.x || c.y >= dims.y) { return; }
      let tc = i32(lid.y + 1u) * 14 + i32(lid.x + 3u);

      let m0 = tM[tc - 15]; let m1 = tM[tc - 14]; let m2 = tM[tc - 13];
      let m3 = tM[tc - 1];  let m4 = tM[tc];      let m5 = tM[tc + 1];
      let m6 = tM[tc + 13]; let m7 = tM[tc + 14]; let m8 = tM[tc + 15];
      // Median of the 3x3 cell levels rejects objects spanning up to ~2x2 cells; then a
      // range-weighted mean around it keeps the map smooth.
      let med = median9(m0, m1, m2, m3, m4, m5, m6, m7, m8);
      let wc = rw(m0, med) * 0.5 + rw(m2, med) * 0.5 + rw(m6, med) * 0.5 + rw(m8, med) * 0.5;
      let we = rw(m1, med) * 0.7 + rw(m3, med) * 0.7 + rw(m5, med) * 0.7 + rw(m7, med) * 0.7;
      let w4 = rw(m4, med);
      let bsum = (rw(m0, med) * m0 + rw(m2, med) * m2 + rw(m6, med) * m6 + rw(m8, med) * m8) * 0.5
               + (rw(m1, med) * m1 + rw(m3, med) * m3 + rw(m5, med) * m5 + rw(m7, med) * m7) * 0.7
               + w4 * m4;
      let B = bsum / max(wc + we + w4, 1e-6);

      // Caption / UI text: dense strong edges spread over a 56x24 px window (7x3 cells).
      // An isolated speck or bird only lights one or two cells and stays far below threshold.
      var dens = 0.0;
      for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -3; dx <= 3; dx = dx + 1) {
          dens = dens + tD[tc + dy * 14 + dx];
        }
      }
      let text = smoothstep(0.035, 0.09, dens / 21.0);

      let prev = textureLoad(prevMap, c, 0);
      let reset = st.cut > 0.5;
      let aT = select(u.stat.w, u.stat.z, text > prev.y);
      let outB = select(mix(prev.x, B, u.stat.z), B, reset);
      let outT = select(mix(prev.y, text, aT), text, reset);
      textureStore(mapOut, c, vec4f(outB, outT, 0.0, 1.0));
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
    fn sliceMap(vuv: vec2f, L: f32) -> vec2f {
      let muv = vuv * u.outp.zw;
      let g = textureGather(0, mapTex, samp, muv);
      let f = fract(muv * vec2f(textureDimensions(mapTex)) - vec2f(0.5));
      let bw = vec4f((1.0 - f.x) * f.y, f.x * f.y, f.x * (1.0 - f.y), (1.0 - f.x) * (1.0 - f.y));
      // Range weight: cells within ~1.8 stops of the pixel; pixels far from all four (highlights)
      // fall back to plain bilinear through the epsilon.
      let d = vec4f(L) - g;
      let x = max(vec4f(1.0) - 0.3 * d * d, vec4f(0.0));
      let w = bw * (x * x + vec4f(1e-4));
      let B = dot(w, g) / max(dot(w, vec4f(1.0)), 1e-6);
      let t = textureSampleLevel(mapTex, samp, muv, 0.0).y;
      return vec2f(B, t);
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
      // Text protection only for pixels well above their background inside text-dense regions
      let textProt = clamp(loc.y, 0.0, 1.0) * smoothstep(0.08, 0.30, detail);

      // Deep-black-only shadow floor (< 0.015 linear)
      var shadowRatio = 1.0;
      if (maxIn < 0.015) {
        let blackToe = pow(max(maxIn, 1e-5), u.extra.x * 0.35);
        let blackMix = 1.0 - smoothstep(0.002, 0.015, maxIn);
        shadowRatio = mix(1.0, clamp(blackToe, 0.78, 1.0), blackMix);
      }

      // EDR highlight curve above the knee (white-screen guarded peak) from the LUT
      var hdrGain = lut.x;

      // Local specular punch: brightness above the robust local background
      let localMask = smoothstep(0.02, 0.24, detail) * lut.y * (1.0 - textProt * 0.85);
      hdrGain = hdrGain * (1.0 + u.tone.z * 0.90 * localMask);

      hdrGain = mix(hdrGain, min(hdrGain, scene.v.w), textProt * 0.85);

      var hdrP3 = p3Linear * (hdrGain * shadowRatio);
      let p3Y = lumaP3(hdrP3);
      let satScale = 1.0 + lut.z * (1.0 - textProt * 0.6);
      hdrP3 = max(vec3f(0.0), mix(vec3f(p3Y), hdrP3, satScale));

      var outP3 = linearToSrgbExt(hdrP3);
      let d1 = ign(fragPos.xy + vec2f(u.info.z * 17.0, 0.0));
      let d2 = ign(fragPos.xy + vec2f(0.0, u.info.z * 31.0));
      let dither = (d1 + d2 - 1.0) * (0.5 / 1023.0);
      outP3 = max(vec3f(0.0), outP3 + vec3f(dither));
      return vec4f(outP3, 1.0);
    }

    fn shade(raw: vec3f, vuv: vec2f, fragPos: vec4f) -> vec4f {
      let flags = u32(u.extra.w + 0.5);
      let maxRaw = max(raw.r, max(raw.g, raw.b));
      if ((flags & 1u) == 0u || ((flags & 2u) != 0u && (scene.v.y > 0.02 || maxRaw > 1.03))) {
        return sdrOut(raw);
      }
      return toneMapHDR(decodeLinear(raw), vuv, fragPos);
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

    // ---- HDR-only path: canvas content at min(source, display) resolution -------------------
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
      }
      if (leftOfSplit(uv)) { return sdrOut(raw); }
      return shade(raw, vuv, fragPos);
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

    fn bsW(f: f32) -> vec4f {
      let g = 1.0 - f;
      return vec4f(g * g * g, 4.0 - 6.0 * f * f + 3.0 * f * f * f, 1.0 + 3.0 * f + 3.0 * f * f - 3.0 * f * f * f, f * f * f) / 6.0;
    }

    fn resampleSR(vuv: vec2f) -> vec3f {
      let R = u.res.xy;
      let s = R / max(u.outp.xy, vec2f(1.0));
      let sharp = clamp(u.res.z, 0.0, 1.0);
      if (max(s.x, s.y) <= 1.0) {
        // Magnify: 4x4 Catmull-Rom, unsharp vs. cubic B-spline from the same taps,
        // clamped to the nearest 2x2 texels (no ringing / halos)
        let p = vuv * R - vec2f(0.5);
        let fp = floor(p);
        let f = p - fp;
        let ip = vec2i(fp);
        let mxc = vec2i(R) - vec2i(1);
        let wcx = crW(f.x);
        let wcy = crW(f.y);
        let wbx = bsW(f.x);
        let wby = bsW(f.y);
        var cr = vec3f(0.0);
        var bs = vec3f(0.0);
        var mn = vec3f(1e9);
        var mxv = vec3f(-1e9);
        for (var j = 0; j < 4; j = j + 1) {
          for (var i = 0; i < 4; i = i + 1) {
            let t = textureLoad(resTex, clamp(ip + vec2i(i - 1, j - 1), vec2i(0), mxc), 0).rgb;
            cr = cr + t * (wcx[i] * wcy[j]);
            bs = bs + t * (wbx[i] * wby[j]);
            if (i >= 1 && i <= 2 && j >= 1 && j <= 2) {
              mn = min(mn, t);
              mxv = max(mxv, t);
            }
          }
        }
        return clamp(cr + (cr - bs) * (1.2 * sharp), mn, mxv);
      }
      // Minify (s in (1, 2]): 4x4 bilinear taps spaced s/2 texels (each tap is a 2x2 prefilter),
      // Lanczos-2 weights at +-s/4, +-3s/4; unsharp vs. the flat average of the same taps.
      let h = 0.5 * max(s, vec2f(1.0));
      let c = vuv * R;
      let lw = vec4f(0.105, 0.395, 0.395, 0.105);
      var acc = vec3f(0.0);
      var flat = vec3f(0.0);
      var mn = vec3f(1e9);
      var mxv = vec3f(-1e9);
      for (var j = 0; j < 4; j = j + 1) {
        for (var i = 0; i < 4; i = i + 1) {
          let pos = c + (vec2f(f32(i), f32(j)) - vec2f(1.5)) * h;
          let t = textureSampleLevel(resTex, samp, pos / R, 0.0).rgb;
          acc = acc + t * (lw[i] * lw[j]);
          flat = flat + t;
          if (i >= 1 && i <= 2 && j >= 1 && j <= 2) {
            mn = min(mn, t);
            mxv = max(mxv, t);
          }
        }
      }
      flat = flat * (1.0 / 16.0);
      return clamp(acc + (acc - flat) * (0.30 + 0.90 * sharp), mn, mxv);
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
    const [ana, stats, map, decode, finalDirect, finalSR, finalSpatial] = await Promise.all([
      analyzePipe,
      compute('stats', 'csStats', PL([statsBGL])),
      compute('map', 'csMap', PL([mapBGL])),
      render('decode', 'fsDecode', PL([videoBGL])),
      render('final', 'fsDirect', finalLayout),
      render('final', 'fsSR', finalLayout),
      render('final', 'fsSpatial', finalLayout)
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
      pipes: { analyze: ana.pipe, stats, map, decode, finalDirect, finalSR, finalSpatial }
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
      u[11] = (p.hdr ? 1 : 0) | (p.bypassNativeHdr ? 2 : 0);
      u[12] = p.srcW; u[13] = p.srcH; u[14] = p.frameSeed || 0; u[15] = reset ? 1 : 0;
      u[18] = p.sharpness; u[19] = 0;
      u[20] = p.outW; u[21] = p.outH;
      u[22] = p.srcW / (CELL * this.cellsW);
      u[23] = p.srcH / (CELL * this.cellsH);
      u[24] = 1 - Math.exp(-dt / 0.12);  // APL rising (guard engages)
      u[25] = 1 - Math.exp(-dt / 0.60);  // APL falling (guard releases)
      u[26] = 1 - Math.exp(-dt / 0.06);  // local map (and text attack)
      u[27] = 1 - Math.exp(-dt / 0.40);  // text release

      // p.tsw = { querySet, begin, end } lets the offscreen harness supply its own query slots
      const tsw = p.tsw || null;
      const measure = !tsw && Boolean(p.measure) && this.ts && !this.ts.busy;
      const tsQS = tsw ? tsw.querySet : measure ? this.ts.qs : null;
      const videoBG = d.createBindGroup({ layout: core.layouts.videoBGL, entries: [{ binding: 0, resource: source }] });

      const tsBegin = tsw ? tsw.begin : 0;
      const tsEnd = tsw ? tsw.end : 1;
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

      let pipe = core.pipes.finalDirect;
      let srcView = core.dummyView;
      let resView = core.dummyView;
      let resW = p.srcW;
      let resH = p.srcH;
      let srPasses = 0;
      if (p.mode === 'sr' || p.mode === 'spatial') {
        this.ensureDecoded();
        const rp = encoder.beginRenderPass({
          colorAttachments: [{ view: this.decView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }]
        });
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

      const fp = encoder.beginRenderPass({
        colorAttachments: [{ view: targetView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        ...(tsQS && tsEnd !== undefined ? { timestampWrites: { querySet: tsQS, endOfPassWriteIndex: tsEnd } } : {})
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
    upscaleSharpness: 0.25,    // 0.0 .. 1.0
    bypassNativeHdr: true,
    splitEnabled: false,
    splitPos: 0.5,
    effectiveHdr: true,
    effectiveUpscale: true,
    effectiveEnabled: true
  };

  const SR_TIERS = ['quality', 'fast', 'spatial'];
  const SR_BUDGET_FRACTION = 0.6;   // of the video frame interval
  const SR_BUDGET_STRIKES = 6;      // over-budget samples before stepping down

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
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
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

  function showHudToast(video, text) {
    const host = video.parentElement;
    if (!host) return;
    let toast = host.querySelector(':scope > [data-truehdr-toast="1"]');
    if (!toast) {
      toast = document.createElement('div');
      toast.setAttribute('data-truehdr-toast', '1');
      Object.assign(toast.style, {
        position: 'absolute',
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

      this.isVisible = true;
      this.isActive = false;
      this.nativeHdrDetected = false;
      this.blockedReason = null;
      this.rvfcId = null;
      this.frameSeed = 0;
      this.frameCounter = 0;
      this.lastRectKey = '';
      this.configuredWidth = 0;
      this.configuredHeight = 0;
      this.lastReportedSig = '';
      this.lastRenderMs = 0;
      this.lastStatusMs = 0;
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

      // GPU timing / budget guard
      this.gpuMs = null;
      this.gpuPending = false;
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
        this.syncOverlayLayout(true);
        this.scheduleRenderOnce();
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
          if (this.isVisible) {
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
    }

    ensureCanvas() {
      if (this.canvas && this.canvas.isConnected) return;
      const parent = this.video.parentElement;
      if (!parent) return;

      const canvas = document.createElement('canvas');
      canvas.setAttribute('data-truehdr', '1');
      Object.assign(canvas.style, {
        position: 'absolute',
        pointerEvents: 'none',
        margin: '0',
        padding: '0',
        border: '0',
        background: 'transparent',
        display: 'none'
      });
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
        cursor: 'ew-resize',
        pointerEvents: 'auto',
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
        width: '20px',
        height: '42px',
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
        pointerEvents: 'none'
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
      h.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        dragging = true;
        try { h.setPointerCapture(e.pointerId); } catch (_) {}
        updateFromEvent(e);
      });
      h.addEventListener('pointermove', (e) => {
        if (!dragging) return;
        e.preventDefault();
        e.stopPropagation();
        updateFromEvent(e);
      });
      const stopDrag = (e) => {
        if (!dragging) return;
        dragging = false;
        e.stopPropagation();
        try { h.releasePointerCapture(e.pointerId); } catch (_) {}
        postToBridge('SET_SPLIT_POS', { splitPos: settings.splitPos });
      };
      h.addEventListener('pointerup', stopDrag);
      h.addEventListener('pointercancel', stopDrag);
      for (const ev of ['mousedown', 'click', 'dblclick']) {
        h.addEventListener(ev, (e) => e.stopPropagation());
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
      this.resetSRGuard();
      this.syncOverlayLayout(true);
      this.scheduleRenderOnce();
      broadcastStatus();
    }

    shouldRender() {
      if (this.destroyed) return false;
      const wantHdr = Boolean(settings.effectiveHdr) && !(settings.bypassNativeHdr && this.nativeHdrDetected);
      const wantUpscale = Boolean(settings.effectiveUpscale);
      if (!wantHdr && !wantUpscale) return false;
      if (this.video.readyState < 2 || this.video.videoWidth === 0 || this.video.videoHeight === 0) return false;
      if (document.fullscreenElement === this.video) return false;
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
    //  - upscaling (display content > 1.1x source): exact device pixels of the video box
    //  - otherwise: content at min(source, display) pixels (1:1 texel-aligned when the display is
    //    larger); the compositor scales the canvas up, so the GPU only shades source pixels.
    computeCanvasPlan(boxCssW, boxCssH, rect) {
      const dpr = Math.min(window.devicePixelRatio || 1, 3);
      const maxDim = gpuState.device ? gpuState.device.limits.maxTextureDimension2D : 8192;
      const vw = Math.max(1, this.video.videoWidth);
      const vh = Math.max(1, this.video.videoHeight);
      const [ox, oy, rw, rh] = rect;
      const boxW = boxCssW * dpr;
      const boxH = boxCssH * dpr;
      const upRatio = Math.max((boxW * rw) / vw, (boxH * rh) / vh);
      const upscale = Boolean(settings.effectiveUpscale) && upRatio > 1.1;

      if (!upscale && upRatio > 1.0) {
        let cw = Math.max(1, Math.round(vw / rw));
        let ch = Math.max(1, Math.round(vh / rh));
        if (cw <= maxDim && ch <= maxDim) {
          const offX = Math.round((cw - vw) / 2);
          const offY = Math.round((ch - vh) / 2);
          return { upscale, cw, ch, rectUniform: [offX / cw, offY / ch, vw / cw, vh / ch], outW: vw, outH: vh };
        }
      }
      const k = Math.min(1, maxDim / Math.max(boxW, 1), maxDim / Math.max(boxH, 1));
      const cw = Math.max(1, Math.round(boxW * k));
      const ch = Math.max(1, Math.round(boxH * k));
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
        return;
      }
      const now = performance.now();
      if (!force && (now - this.lastLayoutCheckMs) < 250 && this.canvas && this.canvas.parentElement === this.video.parentElement && this.configuredWidth > 0) {
        return;
      }
      this.lastLayoutCheckMs = now;

      if (!this.canvas) {
        this.ensureCanvas();
      }
      if (!this.canvas) return;

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
        return;
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
        return;
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

      const rectKey = `${left.toFixed(1)},${top.toFixed(1)},${vRect.width.toFixed(1)},${vRect.height.toFixed(1)},${cw},${ch},${cs.zIndex},${cs.borderRadius}`;
      if (rectKey !== this.lastRectKey) {
        this.lastRectKey = rectKey;
        this.canvas.style.left = `${left}px`;
        this.canvas.style.top = `${top}px`;
        this.canvas.style.width = `${vRect.width}px`;
        this.canvas.style.height = `${vRect.height}px`;
        this.canvas.style.zIndex = cs.zIndex === 'auto' ? '0' : cs.zIndex;
        this.canvas.style.borderRadius = cs.borderRadius;

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

      if (this.canvas.style.display !== 'block') {
        this.canvas.style.display = 'block';
      }
      this.syncSplitHandle();
      if (!this.isActive || upscalingChanged) {
        this.isActive = true;
        broadcastStatus();
      }
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

      // A second 2x instance costs 4x the first (it runs on the 2x image); only chain it when a
      // plain resample would have to stretch > 1.6x more and the intermediate is <= 1440p.
      const ratio = Math.max(outW / vw, outH / vh);
      let passes = 1;
      if (ratio / 2 > 1.6 && 2 * vh <= 1440 && 4 * vw <= maxDim && 4 * vh <= maxDim) passes = 2;

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
      const measure = this.gpuMs === null || this.frameCounter < 24 || this.overBudget > 0 || (this.frameCounter & 3) === 0;
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
          shadowDepth: settings.shadowDepth,
          aplGuard: settings.aplGuard,
          splitPos: settings.splitEnabled ? settings.splitPos : -1.0,
          hdr: Boolean(settings.effectiveHdr),
          bypassNativeHdr: Boolean(settings.bypassNativeHdr),
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

      if (measure) {
        const tsPromise = this.renderer.readTimestamps();
        if (tsPromise) {
          tsPromise.then((ms) => this.noteGpuTime(ms));
        } else if (!this.gpuPending) {
          this.gpuPending = true;
          const t0 = performance.now();
          device.queue.onSubmittedWorkDone().then(() => {
            this.gpuPending = false;
            this.noteGpuTime(performance.now() - t0);
          }, () => {
            this.gpuPending = false;
          });
        }
      }

      const srModel = plan.mode === 'sr' ? plan.model : null;
      const srFallback = plan.mode === 'sr' ? plan.fallback : (this.upscalingActive ? plan.fallback : null);
      this.srPasses = result.srPasses;
      const sig = `${vw}x${vh}->${this.contentOutW}x${this.contentOutH}|${Boolean(settings.effectiveHdr)}|${this.upscalingActive}|${srModel}|${result.srPasses}|${srFallback}`;
      if (this.lastReportedSig !== sig || this.srModel !== srModel || this.srFallback !== srFallback) {
        this.srModel = srModel;
        this.srFallback = srFallback;
        this.lastReportedSig = sig;
        this.lastStatusMs = now;
        broadcastStatus();
      } else if (now - this.lastStatusMs > 2000) {
        this.lastStatusMs = now;
        broadcastStatus();
      }
    }

    onVideoFrame(now) {
      this.rvfcId = null;
      if (this.destroyed) return;
      if (typeof now === 'number') {
        if (this.lastVfcNow > 0) {
          const d = now - this.lastVfcNow;
          if (d > 4 && d < 150) this.frameIntervalMs = this.frameIntervalMs * 0.9 + d * 0.1;
        }
        this.lastVfcNow = now;
      }
      this.renderFrame();
      if (!this.video.paused && !this.video.ended && this.isVisible && settings.effectiveEnabled) {
        this.scheduleLoop();
      }
    }

    scheduleLoop() {
      if (this.destroyed || this.rvfcId !== null) return;
      if (!this.isVisible || !settings.effectiveEnabled) return;
      if (typeof this.video.requestVideoFrameCallback === 'function') {
        this.rvfcId = this.video.requestVideoFrameCallback(this.boundOnFrame);
      } else {
        this.rvfcId = requestAnimationFrame(this.boundOnFrame);
      }
    }

    scheduleRenderOnce() {
      if (this.destroyed || this._oncePending) return;
      this._oncePending = true;
      requestAnimationFrame(() => {
        this._oncePending = false;
        this.renderFrame();
      });
    }

    stopLoop() {
      if (this.rvfcId !== null) {
        if (typeof this.video.cancelVideoFrameCallback === 'function') {
          try { this.video.cancelVideoFrameCallback(this.rvfcId); } catch (_) {}
        } else {
          cancelAnimationFrame(this.rvfcId);
        }
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
        gpuMs
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
      }
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

  for (const evt of ['fullscreenchange', 'webkitfullscreenchange', 'resize']) {
    document.addEventListener(evt, () => {
      for (const inst of instances.values()) {
        inst.syncOverlayLayout(true);
        inst.scheduleRenderOnce();
      }
    });
  }

  const SHORTCUTS = {
    KeyM: ['TOGGLE_MASTER', () => `TrueHDR: ${!settings.enabled ? 'ON' : 'OFF'}`],
    KeyH: ['TOGGLE_HDR', () => (!settings.hdrEnabled ? `HDR: ON (${settings.peakBoost.toFixed(1)}x EDR)` : 'HDR: OFF')],
    KeyU: ['TOGGLE_UPSCALE', () => `Upscaling: ${!settings.upscaleEnabled ? 'ON' : 'OFF'}`],
    KeyS: ['TOGGLE_SPLIT', () => `Split View: ${!settings.splitEnabled ? 'SDR | EDR' : 'OFF'}`]
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
      for (const inst of instances.values()) {
        if (inst.video.videoWidth > 0) {
          showHudToast(inst.video, msg);
          break;
        }
      }
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
      settings = incoming;
      const srCfgChanged = prevSrCfg !== `${settings.upscaleQuality}|${settings.upscaleContent}`;
      const needsLayout = srCfgChanged || prevEnabled !== settings.effectiveEnabled || prevHdr !== settings.effectiveHdr || prevUp !== settings.effectiveUpscale;
      for (const inst of instances.values()) {
        if (srCfgChanged) inst.resetSRGuard();
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
