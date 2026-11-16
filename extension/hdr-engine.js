(() => {
  if (window.__TRUEHDR_ENGINE_LOADED__) return;
  window.__TRUEHDR_ENGINE_LOADED__ = true;

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
    upscaleMode: 'auto',      // 'auto' | '4k'
    upscaleSharpness: 0.55,   // 0.0 .. 1.0
    bypassNativeHdr: true,
    splitEnabled: false,
    splitPos: 0.5,
    effectiveHdr: true,
    effectiveUpscale: true,
    effectiveEnabled: true
  };

  let settings = { ...DEFAULT_SETTINGS };
  let isEdrDisplay = window.matchMedia('(dynamic-range: high)').matches;
  let gpuState = {
    status: 'init', // 'init' | 'ready' | 'unsupported' | 'error'
    device: null,
    sampler: null,
    copyPipeline: null,
    restorePipeline: null,
    reducePipeline: null,
    hdrPipeline: null,
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

  // WGSL Shaders
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

  // Pass 0: Single external texture YUV->RGB decode + linearize at native video resolution (1 tap per source pixel)
  const COPY_SRC_FS = /* wgsl */ `
    @group(0) @binding(0) var samp: sampler;
    @group(0) @binding(1) var vidTex: texture_external;

    fn srgbToLinear(c: vec3f) -> vec3f {
      let lo = c / 12.92;
      let hi = pow(max((c + 0.055) / 1.055, vec3f(0.0)), vec3f(2.4));
      return select(hi, lo, c <= vec3f(0.04045));
    }

    @fragment
    fn fsMain(@location(0) uv: vec2f) -> @location(0) vec4f {
      let raw = max(textureSampleBaseClampToEdge(vidTex, samp, uv).rgb, vec3f(0.0));
      let maxRaw = max(raw.r, max(raw.g, raw.b));
      let lin = select(srgbToLinear(raw), raw, maxRaw > 1.02);
      // Store maxRaw in alpha to detect native PQ/HLG frames with zero extra reads
      return vec4f(lin, maxRaw);
    }
  `;

  // Pass 1: 1x1 Temporal EMA reduction over a 16x12 sparse grid directly from linearized srcTex
  const REDUCE_FS = /* wgsl */ `
    struct Uniforms {
      rect: vec4f,
      tone: vec4f,
      extra: vec4f,
      videoInfo: vec4f,
      upscale: vec4f,
    };

    @group(0) @binding(0) var srcTex: texture_2d<f32>;
    @group(0) @binding(1) var prevStateTex: texture_2d<f32>;
    @group(0) @binding(2) var<uniform> u: Uniforms;

    @fragment
    fn fsMain() -> @location(0) vec4f {
      let dims = max(vec2i(textureDimensions(srcTex, 0)), vec2i(1));
      var sumLuma: f32 = 0.0;
      var sumHighlight: f32 = 0.0;
      var maxPeak: f32 = 0.0;
      var sumHdr: f32 = 0.0;

      for (var y: i32 = 0; y < 12; y = y + 1) {
        let sy = clamp((y * dims.y) / 12 + dims.y / 24, 0, dims.y - 1);
        for (var x: i32 = 0; x < 16; x = x + 1) {
          let sx = clamp((x * dims.x) / 16 + dims.x / 32, 0, dims.x - 1);
          let s = textureLoad(srcTex, vec2i(sx, sy), 0);
          let luma = dot(s.rgb, vec3f(0.2126, 0.7152, 0.0722));
          let maxC = max(s.r, max(s.g, s.b));
          sumLuma = sumLuma + luma;
          sumHighlight = sumHighlight + max(0.0, maxC - 0.72);
          maxPeak = max(maxPeak, maxC);
          sumHdr = sumHdr + select(0.0, 1.0, s.a > 1.03);
        }
      }

      let invN: f32 = 1.0 / 192.0;
      let avgLuma = sumLuma * invN;
      let hlFrac = sumHighlight * invN;
      let hdrFrac = sumHdr * invN;

      let prev = textureLoad(prevStateTex, vec2i(0, 0), 0);
      let reset = u.videoInfo.w > 0.5;

      return vec4f(
        select(mix(prev.r, avgLuma, 0.14), avgLuma, reset),
        select(mix(prev.g, hlFrac, 0.14), hlFrac, reset),
        select(mix(prev.b, maxPeak, 0.14), maxPeak, reset),
        select(mix(prev.a, hdrFrac, 0.30), hdrFrac, reset)
      );
    }
  `;

  // Pass 1b: Source-Resolution Super-Resolution Feature Restoration
  // (Variance-clipped temporal shimmer denoise + edge-preserving bilateral deblock
  //  + 1px-stroke-protected structure-tensor contour steepening + clamped micro-texture deconv)
  const RESTORE_FS = /* wgsl */ `
    struct Uniforms {
      rect: vec4f,
      tone: vec4f,
      extra: vec4f,
      videoInfo: vec4f,
      upscale: vec4f,
    };

    @group(0) @binding(0) var currSrcTex: texture_2d<f32>;
    @group(0) @binding(1) var prevSrcTex: texture_2d<f32>;
    @group(0) @binding(2) var<uniform> u: Uniforms;

    fn ldSrc(tex: texture_2d<f32>, ip: vec2i, maxC: vec2i) -> vec4f {
      return textureLoad(tex, clamp(ip, vec2i(0), maxC), 0);
    }

    fn luma(c: vec3f) -> f32 {
      return dot(max(c, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
    }

    @fragment
    fn fsMain(@location(0) uv: vec2f) -> @location(0) vec4f {
      let dims = max(vec2i(textureDimensions(currSrcTex, 0)), vec2i(1));
      let maxC = dims - vec2i(1);
      let ip = vec2i(clamp(uv * vec2f(dims), vec2f(0.0), vec2f(maxC)));

      let sC = ldSrc(currSrcTex, ip, maxC);
      let c  = max(sC.rgb, vec3f(0.0));
      let n  = max(ldSrc(currSrcTex, ip + vec2i( 0, -1), maxC).rgb, vec3f(0.0));
      let s  = max(ldSrc(currSrcTex, ip + vec2i( 0,  1), maxC).rgb, vec3f(0.0));
      let e  = max(ldSrc(currSrcTex, ip + vec2i( 1,  0), maxC).rgb, vec3f(0.0));
      let w  = max(ldSrc(currSrcTex, ip + vec2i(-1,  0), maxC).rgb, vec3f(0.0));
      let nw = max(ldSrc(currSrcTex, ip + vec2i(-1, -1), maxC).rgb, vec3f(0.0));
      let ne = max(ldSrc(currSrcTex, ip + vec2i( 1, -1), maxC).rgb, vec3f(0.0));
      let sw = max(ldSrc(currSrcTex, ip + vec2i(-1,  1), maxC).rgb, vec3f(0.0));
      let se = max(ldSrc(currSrcTex, ip + vec2i( 1,  1), maxC).rgb, vec3f(0.0));

      let pC  = sqrt(c);
      let pN  = sqrt(n);  let pS  = sqrt(s);  let pE  = sqrt(e);  let pW  = sqrt(w);
      let pNW = sqrt(nw); let pNE = sqrt(ne); let pSW = sqrt(sw); let pSE = sqrt(se);

      let lc = luma(pC);
      let ln = luma(pN); let ls = luma(pS); let le = luma(pE); let lw = luma(pW);

      // 1px stroke / thin typography ridge detector (protects tiny code/UI text from distortion)
      let ridgeV = abs(lc - 0.5 * (ln + ls));
      let ridgeH = abs(lc - 0.5 * (le + lw));
      let isStroke = smoothstep(0.025, 0.075, max(ridgeV, ridgeH));

      // Edge-preserving bilateral deblock (removes MPEG/VP9 gradient noise & blocks)
      let invSigma2: f32 = 1.0 / (0.024 * 0.024);
      var bSum = pC;
      var bwSum: f32 = 1.0;
      let nbrs = array<vec3f, 8>(pN, pS, pE, pW, pNW, pNE, pSW, pSE);
      let swts = array<f32, 8>(0.75, 0.75, 0.75, 0.75, 0.45, 0.45, 0.45, 0.45);
      for (var idx: i32 = 0; idx < 8; idx = idx + 1) {
        let dl = luma(nbrs[idx]) - lc;
        let wt = swts[idx] * exp(-(dl * dl) * invSigma2) * (1.0 - isStroke * 0.9);
        bSum = bSum + nbrs[idx] * wt;
        bwSum = bwSum + wt;
      }
      var pClean = bSum / bwSum;

      // 3x3 neighborhood bounds (strictly prevents any halo/overshoot)
      let pMinN = min(min(min(pN, pS), min(pE, pW)), min(min(pNW, pNE), min(pSW, pSE)));
      let pMaxN = max(max(max(pN, pS), max(pE, pW)), max(max(pNW, pNE), max(pSW, pSE)));
      let pMinBox = min(pC, pMinN);
      let pMaxBox = max(pC, pMaxN);

      // Variance-clipped temporal shimmer cleanup (zero ghosting: cuts to 0 on motion/cuts)
      let pPrev = sqrt(max(ldSrc(prevSrcTex, ip, maxC).rgb, vec3f(0.0)));
      let pPrevClamped = clamp(pPrev, pMinBox, pMaxBox);
      let maxDiff = max(abs(pPrev.r - pC.r), max(abs(pPrev.g - pC.g), abs(pPrev.b - pC.b)));
      let tReset = u.videoInfo.w > 0.5;
      let tWeight = select(0.20 * smoothstep(0.028, 0.004, maxDiff) * (1.0 - isStroke * 0.5), 0.0, tReset);
      pClean = mix(pClean, pPrevClamped, tWeight);
      let lClean = luma(pClean);

      // Structure Tensor directional coherence
      let lnw = luma(pNW); let lne = luma(pNE); let lsw = luma(pSW); let lse = luma(pSE);
      let gx = (lne + 2.0 * le + lse) - (lnw + 2.0 * lw + lsw);
      let gy = (lsw + 2.0 * ls + lse) - (lnw + 2.0 * ln + lne);
      let Jxx = gx * gx; let Jyy = gy * gy; let Jxy = gx * gy;
      let trace = Jxx + Jyy;
      let diff = Jxx - Jyy;
      let disc = sqrt(max(diff * diff + 4.0 * Jxy * Jxy, 0.0));
      let lam1 = 0.5 * (trace + disc);
      let lam2 = 0.5 * (trace - disc);
      let coherence = (lam1 - lam2) / max(lam1 + lam2 + 1e-4, 1e-4);

      let sharp = clamp(u.upscale.z, 0.0, 1.0);

      // Luma-only contour steepening (tightens soft 2-3px codec ramps without warping strokes)
      let lMin3 = luma(pMinN);
      let lMax3 = luma(pMaxN);
      let localSpan = lMax3 - lMin3;
      let lMid = 0.5 * (lMin3 + lMax3);
      let isRamp = select(0.0, 1.0, lc > lMin3 + 0.008 && lc < lMax3 - 0.008);
      let pushL = select(lMin3 - lClean, lMax3 - lClean, lClean > lMid);
      let edgeGate = smoothstep(0.025, 0.12, localSpan) * (coherence * coherence) * isRamp * (1.0 - isStroke);
      let steepAmt = mix(0.10, 0.36, sharp);
      let lSteep = lClean + pushL * (steepAmt * edgeGate);

      // Fine perceptual micro-texture deconvolution (strictly clamped in [lMin3, lMax3])
      let blur1 = (pC * 4.0 + (pN + pS + pE + pW) * 2.0 + (pNW + pNE + pSW + pSE)) / 16.0;
      let fineL = lClean - luma(blur1);
      let texGate = smoothstep(0.014, 0.070, localSpan) * (1.0 - isStroke * 0.75);
      let deconvAmt = mix(0.16, 0.62, sharp);
      let lOut = clamp(lSteep + fineL * (deconvAmt * texGate), lMin3, lMax3);
      let pRestored = clamp(pClean * (lOut / max(lClean, 1e-4)), pMinBox, pMaxBox);

      return vec4f(pRestored * pRestored, sC.a);
    }
  `;

  // Pass 2: Unified Single-Pass Super-Resolution (EASU 12-tap + Halo-Free RCAS from featTex) + True-EDR Display-P3
  const HDR_RENDER_FS = /* wgsl */ `
    struct Uniforms {
      rect: vec4f,       // (ox, oy, rw, rh) normalized content rect
      tone: vec4f,       // (peakBoost, threshold, localPunch, gamutBoost)
      extra: vec4f,      // (shadowDepth, aplGuard, splitPos, flags)
      videoInfo: vec4f,  // (texelW, texelH, frameSeed, resetState)
      upscale: vec4f,    // (srcW, srcH, sharpness, isUpscaled)
    };

    @group(0) @binding(0) var samp: sampler;
    @group(0) @binding(1) var srcTex: texture_2d<f32>;
    @group(0) @binding(2) var featTex: texture_2d<f32>;
    @group(0) @binding(3) var stateTex: texture_2d<f32>;
    @group(0) @binding(4) var<uniform> u: Uniforms;

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

    fn luma709(c: vec3f) -> f32 {
      return dot(max(c, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
    }

    fn lumaP3(c: vec3f) -> f32 {
      return dot(max(c, vec3f(0.0)), vec3f(0.2289746, 0.6917385, 0.0792869));
    }

    fn ign(p: vec2f) -> f32 {
      let magic = vec3f(0.06711056, 0.00583715, 52.9829189);
      return fract(magic.z * fract(dot(p, magic.xy)));
    }

    fn loadFeat(coord: vec2i, maxC: vec2i) -> vec3f {
      return textureLoad(featTex, clamp(coord, vec2i(0), maxC), 0).rgb;
    }

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

    fn applyTrueHDR(
      c0: vec3f,
      n1: vec3f,
      s1: vec3f,
      e1: vec3f,
      w1: vec3f,
      localY: f32,
      avgLuma: f32,
      fragPos: vec4f
    ) -> vec4f {
      let p3Linear = srgbToP3(c0);
      let Y = luma709(c0);
      let maxIn = max(c0.r, max(c0.g, c0.b));
      let E = mix(Y, maxIn, 0.46);
      let Ep = pow(max(E, 0.0), 1.0 / 2.2);

      let eN1 = mix(luma709(n1), max(n1.r, max(n1.g, n1.b)), 0.46);
      let eS1 = mix(luma709(s1), max(s1.r, max(s1.g, s1.b)), 0.46);
      let eE1 = mix(luma709(e1), max(e1.r, max(e1.g, e1.b)), 0.46);
      let eW1 = mix(luma709(w1), max(w1.r, max(w1.g, w1.b)), 0.46);
      let minE1 = min(min(eN1, eS1), min(eE1, eW1));
      let strokeJump = max(0.0, E - localY);

      // Protect thin 1px UI/code text strokes without capping genuine pure-white sparks/highlights
      let isThinStroke = smoothstep(0.18, 0.48, strokeJump)
                       * (1.0 - smoothstep(0.06, 0.20, localY))
                       * (1.0 - smoothstep(0.15, 0.40, minE1 / max(E, 0.04)))
                       * (1.0 - smoothstep(0.74, 0.96, maxIn));

      // Gentle White-Screen APL guard (only activates on very bright full-white slides/snow, >= 0.45 linear)
      let aplAtten = 1.0 - u.extra.y * smoothstep(0.45, 0.88, avgLuma);
      let effPeak = 1.0 + (u.tone.x - 1.0) * clamp(aplAtten, 0.45, 1.0);

      // Deep-black-only shadow floor (strictly < 0.015 linear so UI & text are untouched)
      let shadowStrength = u.extra.x;
      let blackToe = pow(max(maxIn, 1e-5), 1.0 + shadowStrength * 0.35) / max(maxIn, 1e-5);
      let blackMix = smoothstep(0.015, 0.002, maxIn);
      let shadowRatio = mix(1.0, clamp(blackToe, 0.78, 1.0), blackMix);

      // Responsive EDR highlight curve above perceptual knee
      let knee = clamp(u.tone.y, 0.15, 0.88);
      var hdrGain: f32 = 1.0;
      if (Ep > knee) {
        let t = clamp((Ep - knee) / max(1.0 - knee, 0.05), 0.0, 1.0);
        let smoothRamp = t * t * (3.0 - 2.0 * t);
        let fastRamp = pow(t, 0.85);
        let curve = mix(smoothRamp, fastRamp, 0.55);
        hdrGain = 1.0 + (effPeak - 1.0) * curve;
      }

      // Local specular punch on isolated bright highlights
      let localDetail = max(0.0, E - localY);
      let localMask = smoothstep(0.02, 0.24, localDetail) * smoothstep(0.10, 0.62, E) * (1.0 - isThinStroke * 0.85);
      hdrGain = hdrGain * (1.0 + u.tone.z * 0.90 * localMask);

      // Gently bound 1px UI/code text bloom while leaving bright sparks/speculars uncapped
      let textCap = max(1.65, effPeak * 0.32);
      hdrGain = mix(hdrGain, min(hdrGain, textCap), isThinStroke * 0.85);

      var hdrP3 = p3Linear * (hdrGain * shadowRatio);

      let p3Y = lumaP3(hdrP3);
      let satScale = 1.0 + u.tone.w * 0.35 * smoothstep(0.06, 0.70, E) * (1.0 - isThinStroke * 0.6);
      hdrP3 = max(vec3f(0.0), mix(vec3f(p3Y), hdrP3, satScale));

      var outP3 = linearToSrgbExt(hdrP3);
      let d1 = ign(fragPos.xy + vec2f(u.videoInfo.z * 17.0, 0.0));
      let d2 = ign(fragPos.xy + vec2f(0.0, u.videoInfo.z * 31.0));
      let dither = (d1 + d2 - 1.0) * (0.5 / 1023.0);
      outP3 = max(vec3f(0.0), outP3 + vec3f(dither));

      return vec4f(outP3, 1.0);
    }

    @fragment
    fn fsMain(@builtin(position) fragPos: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
      let vuv = (uv - u.rect.xy) / u.rect.zw;
      if (vuv.x < 0.0 || vuv.x > 1.0 || vuv.y < 0.0 || vuv.y > 1.0) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
      }

      // Split Compare: Left = original native bilinear SDR, Right = active Upscaling and/or HDR
      let splitPos = u.extra.z;
      if (splitPos >= 0.0) {
        if (abs(uv.x - splitPos) < 0.0016) {
          return vec4f(1.8, 1.8, 1.8, 1.0);
        }
        if (uv.x < splitPos) {
          let orig = textureSampleLevel(srcTex, samp, vuv, 0.0).rgb;
          return vec4f(linearToSrgbExt(orig), 1.0);
        }
      }

      let flags = u32(u.extra.w + 0.5);
      let isHdrEnabled = (flags & 1u) != 0u;
      let bypassNativeHdr = (flags & 2u) != 0u;
      let isUpscaled = u.upscale.w > 0.5;

      let scene = textureLoad(stateTex, vec2i(0, 0), 0);
      let avgLuma = scene.r;
      let nativeHdrFrac = scene.a;

      let dims = textureDimensions(srcTex, 0);
      let maxC = vec2i(dims) - vec2i(1);
      let srcSize = vec2f(dims);
      let px = u.videoInfo.xy;

      if (!isUpscaled) {
        let s0 = textureSampleLevel(srcTex, samp, vuv, 0.0);
        let c0 = s0.rgb;
        if (!isHdrEnabled || (bypassNativeHdr && (nativeHdrFrac > 0.02 || s0.a > 1.03))) {
          return vec4f(linearToSrgbExt(c0), 1.0);
        }

        let n1 = textureSampleLevel(srcTex, samp, clamp(vuv + vec2f(0.0, -px.y), vec2f(0.0), vec2f(1.0)), 0.0).rgb;
        let s1 = textureSampleLevel(srcTex, samp, clamp(vuv + vec2f(0.0,  px.y), vec2f(0.0), vec2f(1.0)), 0.0).rgb;
        let e1 = textureSampleLevel(srcTex, samp, clamp(vuv + vec2f( px.x, 0.0), vec2f(0.0), vec2f(1.0)), 0.0).rgb;
        let w1 = textureSampleLevel(srcTex, samp, clamp(vuv + vec2f(-px.x, 0.0), vec2f(0.0), vec2f(1.0)), 0.0).rgb;

        let stepUV = px * 3.5;
        let cN = textureSampleLevel(srcTex, samp, clamp(vuv + vec2f(0.0, -stepUV.y), vec2f(0.0), vec2f(1.0)), 0.0).rgb;
        let cS = textureSampleLevel(srcTex, samp, clamp(vuv + vec2f(0.0,  stepUV.y), vec2f(0.0), vec2f(1.0)), 0.0).rgb;
        let cE = textureSampleLevel(srcTex, samp, clamp(vuv + vec2f( stepUV.x, 0.0), vec2f(0.0), vec2f(1.0)), 0.0).rgb;
        let cW = textureSampleLevel(srcTex, samp, clamp(vuv + vec2f(-stepUV.x, 0.0), vec2f(0.0), vec2f(1.0)), 0.0).rgb;
        let localY = 0.25 * (luma709(cN) + luma709(cS) + luma709(cE) + luma709(cW));

        return applyTrueHDR(c0, n1, s1, e1, w1, localY, avgLuma, fragPos);
      }

      // 12-tap EASU reconstruction + halo-free perceptual RCAS from restored featTex
      let fdims = textureDimensions(featTex, 0);
      let fmaxC = vec2i(fdims) - vec2i(1);
      let fsrcSize = vec2f(fdims);
      let p = vuv * fsrcSize - vec2f(0.5);
      let fp = floor(p);
      let f = p - fp;
      let ip = vec2i(fp);

      let b  = loadFeat(ip + vec2i( 0, -1), fmaxC);
      let c  = loadFeat(ip + vec2i( 1, -1), fmaxC);
      let e  = loadFeat(ip + vec2i(-1,  0), fmaxC);
      let f0 = loadFeat(ip + vec2i( 0,  0), fmaxC);
      let f1 = loadFeat(ip + vec2i( 1,  0), fmaxC);
      let h  = loadFeat(ip + vec2i( 2,  0), fmaxC);
      let i  = loadFeat(ip + vec2i(-1,  1), fmaxC);
      let j  = loadFeat(ip + vec2i( 0,  1), fmaxC);
      let k  = loadFeat(ip + vec2i( 1,  1), fmaxC);
      let l  = loadFeat(ip + vec2i( 2,  1), fmaxC);
      let n  = loadFeat(ip + vec2i( 0,  2), fmaxC);
      let o  = loadFeat(ip + vec2i( 1,  2), fmaxC);

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

      // Phase-locked halo-free perceptual RCAS detail boost (strictly clamped to [pMin4, pMax4])
      let n1 = mix(b, c, f.x);
      let s1 = mix(n, o, f.x);
      let e1 = mix(h, l, f.y);
      let w1n = mix(e, i, f.y);

      let bilin = f0 * w0 + f1 * w1 + j * w2 + k * w3;
      let pRecon = sqrt(max(recon, vec3f(0.0)));
      let pBilin = sqrt(max(bilin, vec3f(0.0)));
      let pMin4 = sqrt(max(min4, vec3f(0.0)));
      let pMax4 = sqrt(max(max4, vec3f(0.0)));
      let contrast = max(max(pMax4.r - pMin4.r, pMax4.g - pMin4.g), pMax4.b - pMin4.b);
      let noiseGate = smoothstep(0.015, 0.080, contrast);
      let lobe = clamp(min(luma709(pMin4), 1.0 - luma709(pMax4)) / max(luma709(pMax4), 1e-4), 0.0, 1.0);
      let sharpAmt = clamp(u.upscale.z, 0.0, 1.0) * 1.05 * noiseGate * (0.35 + 0.65 * lobe);
      let pSharp = clamp(pRecon + (pRecon - pBilin) * sharpAmt, pMin4, pMax4);
      let c0 = pSharp * pSharp;

      let centerRaw = textureSampleLevel(srcTex, samp, vuv, 0.0);
      if (!isHdrEnabled || (bypassNativeHdr && (nativeHdrFrac > 0.02 || centerRaw.a > 1.03))) {
        return vec4f(linearToSrgbExt(c0), 1.0);
      }

      let localY = 0.25 * (luma709(n1) + luma709(s1) + luma709(e1) + luma709(w1n));
      return applyTrueHDR(c0, n1, s1, e1, w1n, localY, avgLuma, fragPos);
    }
  `;

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
        const device = await adapter.requestDevice();
        device.addEventListener('uncapturederror', (e) => {
          console.error('[TrueHDR] WebGPU uncaptured error:', e.error?.message || e.error);
          gpuState.lastWebGpuError = e.error?.message || String(e.error);
        });
        device.lost.then((info) => {
          console.warn('[TrueHDR] WebGPU device lost:', info.message);
          gpuState.status = 'init';
          gpuState.device = null;
          gpuState.initPromise = null;
          for (const inst of instances.values()) {
            inst.onDeviceLost();
          }
        });

        const vsModule = device.createShaderModule({ code: FULLSCREEN_VS });
        const copyModule = device.createShaderModule({ code: COPY_SRC_FS });
        const restoreModule = device.createShaderModule({ code: RESTORE_FS });
        const redModule = device.createShaderModule({ code: REDUCE_FS });
        const hdrModule = device.createShaderModule({ code: HDR_RENDER_FS });

        const sampler = device.createSampler({
          magFilter: 'linear',
          minFilter: 'linear',
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge'
        });

        const makePipe = (mod) =>
          device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: vsModule, entryPoint: 'vsMain' },
            fragment: {
              module: mod,
              entryPoint: 'fsMain',
              targets: [{ format: 'rgba16float' }]
            },
            primitive: { topology: 'triangle-list' }
          });

        gpuState.device = device;
        gpuState.sampler = sampler;
        gpuState.copyPipeline = makePipe(copyModule);
        gpuState.restorePipeline = makePipe(restoreModule);
        gpuState.reducePipeline = makePipe(redModule);
        gpuState.hdrPipeline = makePipe(hdrModule);
        gpuState.status = 'ready';
        return gpuState;
      } catch (err) {
        console.error('[TrueHDR] GPU init failed:', err);
        gpuState.status = 'error';
        gpuState.errorMsg = String(err);
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
      this.uniformBuffer = null;
      this.uniformData = new Float32Array(20);
      this.srcTex = [null, null];
      this.srcView = [null, null];
      this.featTex = null;
      this.featView = null;
      this.srcWidth = 0;
      this.srcHeight = 0;
      this.srcIndex = 0;
      this.hasPrevSrc = false;
      this.stateTex = [null, null];
      this.stateViews = [null, null];
      this.restoreBindGroups = [null, null];
      this.reduceBindGroups = [[null, null], [null, null]];
      this.hdrBindGroups = [[null, null], [null, null]];
      this.stateIndex = 0;
      this.upscalingActive = false;
      this.resetStateNextFrame = true;

      this.cachedRect = [0, 0, 1, 1];
      this.cachedBoxW = 0;
      this.cachedBoxH = 0;
      this.lastLayoutCheckMs = 0;

      this.isVisible = true;
      this.isActive = false;
      this.nativeHdrDetected = false;
      this.blockedReason = null;
      this.rvfcId = null;
      this.frameSeed = 0;
      this.lastRectKey = '';
      this.configuredWidth = 0;
      this.configuredHeight = 0;
      this.lastReportedSig = '';
      this.lastRenderTime = -1;
      this.destroyed = false;

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

      const device = gpu.device;
      if (!this.uniformBuffer) {
        this.uniformBuffer = device.createBuffer({
          size: 80,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
      }

      for (let i = 0; i < 2; i++) {
        if (!this.stateTex[i]) {
          this.stateTex[i] = device.createTexture({
            size: [1, 1, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
          });
          this.stateViews[i] = this.stateTex[i].createView();
        }
      }

      this.ensureCanvas();
      this.checkVideoMetadata();
      this.syncOverlayLayout(true);
      this.scheduleLoop();
    }

    onDeviceLost() {
      this.ctx = null;
      this.uniformBuffer = null;
      this.releaseSourceTexture();
      this.stateTex = [null, null];
      this.stateViews = [null, null];
      this.configuredWidth = 0;
      this.configuredHeight = 0;
      if (!this.destroyed) {
        this.initResources();
      }
    }

    releaseSourceTexture() {
      for (let i = 0; i < 2; i++) {
        if (this.srcTex[i]) {
          try { this.srcTex[i].destroy(); } catch (_) {}
        }
        this.srcTex[i] = null;
        this.srcView[i] = null;
      }
      if (this.featTex) {
        try { this.featTex.destroy(); } catch (_) {}
      }
      this.featTex = null;
      this.featView = null;
      this.srcWidth = 0;
      this.srcHeight = 0;
      this.srcIndex = 0;
      this.hasPrevSrc = false;
      this.restoreBindGroups = [null, null];
      this.reduceBindGroups = [[null, null], [null, null]];
      this.hdrBindGroups = [[null, null], [null, null]];
    }

    ensureSourceTexture(w, h) {
      if (this.srcTex[0] && this.srcWidth === w && this.srcHeight === h && this.reduceBindGroups[0][0]) {
        return true;
      }
      this.releaseSourceTexture();
      const device = gpuState.device;
      if (!device || !this.stateViews[0] || !this.uniformBuffer) return false;

      const texDesc = {
        size: [w, h, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
      };
      for (let i = 0; i < 2; i++) {
        this.srcTex[i] = device.createTexture(texDesc);
        this.srcView[i] = this.srcTex[i].createView();
      }
      this.featTex = device.createTexture(texDesc);
      this.featView = this.featTex.createView();
      this.srcWidth = w;
      this.srcHeight = h;
      this.srcIndex = 0;
      this.hasPrevSrc = false;

      for (let s = 0; s < 2; s++) {
        this.restoreBindGroups[s] = device.createBindGroup({
          layout: gpuState.restorePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.srcView[s] },
            { binding: 1, resource: this.srcView[1 - s] },
            { binding: 2, resource: { buffer: this.uniformBuffer } }
          ]
        });
        for (let st = 0; st < 2; st++) {
          this.reduceBindGroups[s][st] = device.createBindGroup({
            layout: gpuState.reducePipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: this.srcView[s] },
              { binding: 1, resource: this.stateViews[st] },
              { binding: 2, resource: { buffer: this.uniformBuffer } }
            ]
          });
          this.hdrBindGroups[s][st] = device.createBindGroup({
            layout: gpuState.hdrPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: gpuState.sampler },
              { binding: 1, resource: this.srcView[s] },
              { binding: 2, resource: this.featView },
              { binding: 3, resource: this.stateViews[st] },
              { binding: 4, resource: { buffer: this.uniformBuffer } }
            ]
          });
        }
      }
      return true;
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
      this.configuredWidth = 0;
      this.configuredHeight = 0;
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
      const [ox, oy, rw, rh] = this.computeContentRect(vRect.width, vRect.height, fit);
      this.cachedRect = [ox, oy, rw, rh];
      this.cachedBoxW = vRect.width;
      this.cachedBoxH = vRect.height;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const maxDim = Math.min(3840, gpuState.device ? gpuState.device.limits.maxTextureDimension2D : 3840);
      const vw = Math.max(1, this.video.videoWidth);
      const vh = Math.max(1, this.video.videoHeight);
      const mode = settings.upscaleMode || 'auto';

      let cw = Math.max(1, Math.min(maxDim, Math.round(vRect.width * dpr)));
      let ch = Math.max(1, Math.min(maxDim, Math.round(vRect.height * dpr)));

      if (settings.effectiveUpscale && mode === '4k') {
        const capW = vw >= vh ? Math.min(3840, maxDim) : Math.min(2160, maxDim);
        const capH = vw >= vh ? Math.min(2160, maxDim) : Math.min(3840, maxDim);
        const contentW = cw * Math.min(rw, 1.0);
        const contentH = ch * Math.min(rh, 1.0);
        const scaleUp = Math.min(
          capW / Math.max(contentW, 1),
          capH / Math.max(contentH, 1),
          maxDim / Math.max(cw, 1),
          maxDim / Math.max(ch, 1)
        );
        if (scaleUp > 1.0) {
          cw = Math.min(maxDim, Math.round(cw * scaleUp));
          ch = Math.min(maxDim, Math.round(ch * scaleUp));
        }
      }

      const rectKey = `${left.toFixed(1)},${top.toFixed(1)},${vRect.width.toFixed(1)},${vRect.height.toFixed(1)},${cw},${ch},${cs.zIndex},${cs.borderRadius}`;
      if (rectKey !== this.lastRectKey) {
        this.lastRectKey = rectKey;
        this.canvas.style.left = `${left}px`;
        this.canvas.style.top = `${top}px`;
        this.canvas.style.width = `${vRect.width}px`;
        this.canvas.style.height = `${vRect.height}px`;
        this.canvas.style.zIndex = cs.zIndex === 'auto' ? '0' : cs.zIndex;
        this.canvas.style.borderRadius = cs.borderRadius;

        if (this.canvas.width !== cw || this.canvas.height !== ch) {
          this.canvas.width = cw;
          this.canvas.height = ch;
          this.configuredWidth = cw;
          this.configuredHeight = ch;
          this.reconfigure();
        }
      }

      const contentOutW = Math.round(cw * Math.min(rw, 1.0));
      const contentOutH = Math.round(ch * Math.min(rh, 1.0));
      const nextUpscaling = Boolean(settings.effectiveUpscale) && (contentOutW > vw * 1.04 || contentOutH > vh * 1.04);
      const upscalingChanged = this.upscalingActive !== nextUpscaling;
      this.upscalingActive = nextUpscaling;

      if (this.canvas.style.display !== 'block') {
        this.canvas.style.display = 'block';
      }
      if (!this.isActive || upscalingChanged) {
        this.isActive = true;
        broadcastStatus();
      }
    }

    renderFrame() {
      if (this.destroyed || gpuState.status !== 'ready') return;
      this.syncOverlayLayout(false);
      if (!this.shouldRender() || !this.ctx || !this.canvas) return;
      if (this.cachedBoxW < 96 || this.cachedBoxH < 64) return;
      if (this.configuredWidth === 0) {
        this.syncOverlayLayout(true);
        if (this.configuredWidth === 0) return;
      }

      const device = gpuState.device;
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

      const vw = Math.max(1, Math.min(3840, this.video.videoWidth));
      const vh = Math.max(1, Math.min(3840, this.video.videoHeight));
      if (!this.ensureSourceTexture(vw, vh)) return;

      const [ox, oy, rw, rh] = this.cachedRect;
      const u = this.uniformData;
      u[0] = ox;
      u[1] = oy;
      u[2] = rw;
      u[3] = rh;

      u[4] = settings.peakBoost;
      u[5] = settings.threshold;
      u[6] = settings.localPunch;
      u[7] = settings.gamutBoost;

      u[8] = settings.shadowDepth;
      u[9] = settings.aplGuard;
      u[10] = settings.splitEnabled ? settings.splitPos : -1.0;
      let flags = 0;
      if (settings.effectiveHdr) flags |= 1;
      if (settings.bypassNativeHdr) flags |= 2;
      u[11] = flags;

      this.frameSeed = (this.frameSeed + 1) % 256;
      const resetAll = this.resetStateNextFrame || !this.hasPrevSrc;
      u[12] = 1.0 / vw;
      u[13] = 1.0 / vh;
      u[14] = this.frameSeed;
      u[15] = resetAll ? 1.0 : 0.0;
      this.resetStateNextFrame = false;

      const contentOutW = Math.round(this.configuredWidth * Math.min(rw, 1.0));
      const contentOutH = Math.round(this.configuredHeight * Math.min(rh, 1.0));
      const shouldUpscale = Boolean(settings.effectiveUpscale) && (contentOutW > vw * 1.04 || contentOutH > vh * 1.04);

      const sig = `${vw}x${vh}->${contentOutW}x${contentOutH}|${Boolean(settings.effectiveHdr)}|${shouldUpscale}`;
      if (this.upscalingActive !== shouldUpscale || this.lastReportedSig !== sig) {
        this.upscalingActive = shouldUpscale;
        this.lastReportedSig = sig;
        broadcastStatus();
      }

      u[16] = vw;
      u[17] = vh;
      u[18] = typeof settings.upscaleSharpness === 'number' ? settings.upscaleSharpness : 0.55;
      u[19] = shouldUpscale ? 1.0 : 0.0;

      device.queue.writeBuffer(this.uniformBuffer, 0, u);

      const prevIdx = this.stateIndex;
      const nextIdx = 1 - prevIdx;
      this.stateIndex = nextIdx;

      const currSrcIdx = 1 - this.srcIndex;
      this.srcIndex = currSrcIdx;
      this.hasPrevSrc = true;

      const encoder = device.createCommandEncoder();

      // Pass 0: Copy + linearize external texture once at native video resolution
      const copyBindGroup = device.createBindGroup({
        layout: gpuState.copyPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: gpuState.sampler },
          { binding: 1, resource: externalTex }
        ]
      });
      const copyPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.srcView[currSrcIdx],
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 }
        }]
      });
      copyPass.setPipeline(gpuState.copyPipeline);
      copyPass.setBindGroup(0, copyBindGroup);
      copyPass.draw(3);
      copyPass.end();

      // Pass 1: 1x1 Temporal EMA scene statistics directly from srcTex[currSrcIdx]
      const redPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.stateViews[nextIdx],
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 }
        }]
      });
      redPass.setPipeline(gpuState.reducePipeline);
      redPass.setBindGroup(0, this.reduceBindGroups[currSrcIdx][prevIdx]);
      redPass.draw(3);
      redPass.end();

      // Pass 1b: Source-Resolution Super-Resolution Feature Restoration (only when upscaling)
      if (shouldUpscale) {
        const restorePass = encoder.beginRenderPass({
          colorAttachments: [{
            view: this.featView,
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 }
          }]
        });
        restorePass.setPipeline(gpuState.restorePipeline);
        restorePass.setBindGroup(0, this.restoreBindGroups[currSrcIdx]);
        restorePass.draw(3);
        restorePass.end();
      }

      // Pass 2: Super-Resolution (EASU 12-tap + Halo-Free RCAS) + True-EDR Display-P3 output to float16 swapchain
      const hdrPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: swapTex.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 }
        }]
      });
      hdrPass.setPipeline(gpuState.hdrPipeline);
      hdrPass.setBindGroup(0, this.hdrBindGroups[currSrcIdx][nextIdx]);
      hdrPass.draw(3);
      hdrPass.end();

      device.queue.submit([encoder.finish()]);
    }

    onVideoFrame() {
      this.rvfcId = null;
      if (this.destroyed) return;
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
      if (this.canvas && this.canvas.parentElement) {
        this.canvas.remove();
      }
      this.releaseSourceTexture();
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
            const rw = Math.min(inst.cachedRect[2] || 1, 1);
            const rh = Math.min(inst.cachedRect[3] || 1, 1);
            const outW = Math.round(inst.configuredWidth * rw);
            const outH = Math.round(inst.configuredHeight * rh);
            outputRes = `${outW}×${outH}`;
            upscalingActive = inst.upscalingActive;
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
        upscalingActive
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

  window.addEventListener(
    'keydown',
    (e) => {
      const tag = e.target && e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) return;
      if (e.altKey && !e.metaKey && !e.ctrlKey && e.code === 'KeyH') {
        e.preventDefault();
        postToBridge('TOGGLE_SITE', {});
        const nextOn = !settings.effectiveEnabled;
        for (const inst of instances.values()) {
          if (inst.video.videoWidth > 0) {
            showHudToast(inst.video, nextOn ? `TrueHDR: ON (${settings.peakBoost.toFixed(1)}x EDR)` : 'TrueHDR: OFF');
            break;
          }
        }
      } else if (e.altKey && !e.metaKey && !e.ctrlKey && e.code === 'KeyS') {
        e.preventDefault();
        postToBridge('TOGGLE_SPLIT', {});
        for (const inst of instances.values()) {
          if (inst.video.videoWidth > 0) {
            showHudToast(inst.video, !settings.splitEnabled ? 'Split View: SDR | EDR' : 'Split View: OFF');
            break;
          }
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
      const prevMode = settings.upscaleMode;
      const incoming = { ...DEFAULT_SETTINGS, ...(data.payload || {}) };
      if (incoming.upscaleMode === 'off') {
        if (!data.payload || typeof data.payload.upscaleEnabled !== 'boolean') {
          incoming.upscaleEnabled = false;
        }
        incoming.upscaleMode = 'auto';
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
      const needsLayout = prevMode !== settings.upscaleMode || prevEnabled !== settings.effectiveEnabled || prevHdr !== settings.effectiveHdr || prevUp !== settings.effectiveUpscale;
      for (const inst of instances.values()) {
        inst.syncOverlayLayout(needsLayout);
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
