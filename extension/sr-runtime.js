// TrueHDR super-resolution runtime (spec: docs/SR_SPEC.md). Plain script, page MAIN world,
// loaded after sr-models.js and before hdr-engine.js. Defines globalThis.__TRUEHDR_SR__.
//
// Generic WebGPU implementation of the SR model family: per model, WGSL is generated at create()
// time with the weights baked in as constants (fastest on Apple GPUs, see dev/sr/bench.ts).
// Kernels: every k>1 conv starts a compute pass; following k=1 convs are fused into it as
// per-pixel epilogues; the last layer additionally does depth-to-space, the bicubic base and the
// luma-delta colour reconstruction and writes the rgba16float 2w x 2h output.
// Spatial convs stage their input tile (+ halo, clamp-to-edge = replicate padding) in workgroup
// memory and each invocation computes a PX x PY block of pixels for all output channels.
// Intermediates are rgba16float (f16 path) / rgba32float (f32 path) 2d-array textures, one array
// layer per 4 channels, pooled by liveness and reallocated only when the input size changes.
//
// WebKit-compatibility constraints kept on purpose: only `enable f16;`, no subgroups, no pointer
// parameters, no override constants, workgroup storage <= 16 KiB, <= 256 invocations, at most
// 4 storage textures per stage, uniform buffer per kernel only for the size.
(function () {
  'use strict';
  if (globalThis.__TRUEHDR_SR__ && globalThis.__TRUEHDR_SR__.version >= 1) return;

  var K709 = [0.2126, 0.7152, 0.0722];
  // PyTorch bicubic (a = -0.75, align_corners=False) taps for 2x: phase 0 (t = 0.75) uses source
  // offsets -2..+1, phase 1 (t = 0.25) uses -1..+2 (indices clamped to the edge).
  var BIC = [[-0.03515625, 0.26171875, 0.87890625, -0.10546875],
             [-0.10546875, 0.87890625, 0.26171875, -0.03515625]];

  // Defaults benchmarked in real WebKit (webkit_run.ts, M1 Pro, f16): direct texture loads on 16x8
  // workgroups (1 px/invocation) avoid shared-memory staging barriers (~1.3-1.6x faster in WebKit).
  var DEFAULT_CONFIG = { wx: 16, wy: 8, px: 1, py: 1, scalarPx: 1, scalarPy: 1, pointwisePx: 1, pointwisePy: 1, direct: true };

  // ---------------------------------------------------------------- model data
  function b64ToBytes(s) {
    var bin = atob(s);
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function f16ToF32(h) {
    var s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * m * 5.9604644775390625e-8; // 2^-24
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  function decodeF16(s, n, what) {
    var bytes = b64ToBytes(s);
    if (bytes.length !== n * 2) throw new Error('TrueHDR SR: ' + what + ' has ' + (bytes.length / 2) + ' values, expected ' + n);
    var out = new Float32Array(n);
    for (var i = 0; i < n; i++) {
      var v = f16ToF32(bytes[2 * i] | (bytes[2 * i + 1] << 8));
      if (!isFinite(v)) throw new Error('TrueHDR SR: non-finite value in ' + what);
      out[i] = v;
    }
    return out;
  }
  function decodeQ8(wqB64, wsB64, cout, fanIn, what) {
    var q = b64ToBytes(wqB64);
    var n = cout * fanIn;
    if (q.length !== n) throw new Error('TrueHDR SR: ' + what + ' has ' + q.length + ' int8 values, expected ' + n);
    var scales = decodeF16(wsB64, cout, what + ' scales');
    var out = new Float32Array(n);
    for (var co = 0; co < cout; co++) {
      var s = scales[co], off = co * fanIn;
      for (var j = 0; j < fanIn; j++) {
        var v = q[off + j];
        out[off + j] = (v > 127 ? v - 256 : v) * s;
      }
    }
    return out;
  }

  function f32ArrayToF16(arr) {
    var f = new Float32Array(1), u = new Uint32Array(f.buffer), out = new Uint16Array(arr.length);
    for (var i = 0; i < arr.length; i++) {
      f[0] = arr[i];
      var x = u[0], sign = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112, m = x & 0x7fffff, r;
      if (e >= 31) r = sign | 0x7c00;
      else if (e <= 0) r = e < -10 ? sign : sign | Math.round((m | 0x800000) / Math.pow(2, 14 - e));
      else r = sign | ((e << 10) + Math.round(m / 8192));
      out[i] = r;
    }
    return out;
  }

  function registry() {
    var list = globalThis.__TRUEHDR_SR_MODELS__;
    var byId = new Map();
    if (Array.isArray(list)) for (var i = 0; i < list.length; i++) if (list[i] && list[i].id) byId.set(list[i].id, list[i]);
    return byId; // last entry with a given id wins (lets a later script override a model)
  }

  function prepareModel(m) {
    var where = 'TrueHDR SR model ' + m.id;
    if (m.scale !== 2) throw new Error(where + ': only scale 2 is supported');
    if (m.base !== 'bicubic' && m.base !== 'none') throw new Error(where + ': unknown base ' + m.base);
    if (!Array.isArray(m.layers) || !m.layers.length) throw new Error(where + ': no layers');
    var layers = [], prevC = 1, outC = [];
    for (var i = 0; i < m.layers.length; i++) {
      var L = m.layers[i], k = L.k | 0, cin = L.cin | 0, cout = L.cout | 0;
      var lw = where + ' layer ' + i;
      if (k !== 1 && k !== 3 && k !== 5) throw new Error(lw + ': kernel ' + L.k + ' not in {1,3,5}');
      if (cin !== prevC) throw new Error(lw + ': cin ' + cin + ' != previous output ' + prevC);
      if (['relu', 'prelu', 'crelu', 'none'].indexOf(L.act) < 0) throw new Error(lw + ': unknown act ' + L.act);
      if (L.act === 'crelu' && cout % 4) throw new Error(lw + ': crelu needs cout % 4 == 0');
      var skip = (L.skip === undefined || L.skip === null) ? null : (L.skip | 0);
      if (skip !== null) {
        if (skip < -1 || skip >= i) throw new Error(lw + ': bad skip ' + L.skip);
        if ((skip === -1 ? 1 : outC[skip]) !== cout) throw new Error(lw + ': skip shape mismatch');
      }
      var layer = {
        k: k, cin: cin, cout: cout, act: L.act, skip: skip,
        w: L.wq ? decodeQ8(L.wq, L.ws, cout, cin * k * k, lw + ' wq') : decodeF16(L.w, cout * cin * k * k, lw + ' w'),
        b: decodeF16(L.b, cout, lw + ' b'),
        alpha: L.act === 'prelu' ? decodeF16(L.alpha, cout, lw + ' alpha') : null,
        outC: L.act === 'crelu' ? 2 * cout : cout
      };
      layers.push(layer);
      outC.push(layer.outC);
      prevC = layer.outC;
    }
    if (prevC !== 4 || layers[layers.length - 1].act === 'crelu') throw new Error(where + ': last layer must output 4 channels');
    return layers;
  }

  // ---------------------------------------------------------------- planning
  // Kernel = one spatial (or the first) layer + following k=1 layers. Tensors that cross kernels
  // (kernel inputs, skip sources used in later kernels) are materialized as textures.
  function plan(layers) {
    var kernels = [];
    for (var i = 0; i < layers.length; i++) {
      if (i === 0 || layers[i].k > 1) kernels.push({ layers: [i] });
      else kernels[kernels.length - 1].layers.push(i);
    }
    var kernelOf = [];
    kernels.forEach(function (K, ki) { K.layers.forEach(function (li) { kernelOf[li] = ki; }); });
    var mat = {}; // layer index -> { firstUse, lastUse }
    function need(li, ki) {
      if (li < 0) return;
      var r = mat[li] || (mat[li] = { prod: kernelOf[li], last: ki });
      r.last = Math.max(r.last, ki);
    }
    kernels.forEach(function (K, ki) {
      if (ki > 0) need(K.layers[0] - 1, ki);
      K.layers.forEach(function (li) {
        var s = layers[li].skip;
        if (s !== null && s >= 0 && kernelOf[s] !== ki) need(s, ki);
      });
    });
    // texture pool by liveness: a texture can be reused by a tensor produced in kernel p if its
    // previous tensor's last reader ran in a kernel < p (no read/write of one texture per pass).
    var pool = [], slot = {};
    Object.keys(mat).map(Number).sort(function (a, b) { return a - b; }).forEach(function (li) {
      var r = mat[li], groups = Math.ceil(layers[li].outC / 4), found = -1;
      for (var t = 0; t < pool.length; t++) if (pool[t].groups === groups && pool[t].busyUntil < r.prod) { found = t; break; }
      if (found < 0) { pool.push({ groups: groups, busyUntil: -1 }); found = pool.length - 1; }
      pool[found].busyUntil = r.last;
      slot[li] = found;
    });
    return { kernels: kernels, kernelOf: kernelOf, slot: slot, pool: pool };
  }

  // ---------------------------------------------------------------- WGSL generation
  function fmtNum(x, f16) {
    var s;
    if (x === 0) s = '0.0';
    else if (Number.isInteger(x)) s = x.toFixed(1);
    else s = String(x);
    return s + (f16 ? 'h' : '');
  }

  function genKernel(model, layers, P, ki, f16, cfg, outFmt) {
    var K = P.kernels[ki], first = layers[K.layers[0]], li0 = K.layers[0];
    var isFirst = ki === 0, isLast = ki === P.kernels.length - 1;
    var T = f16 ? 'f16' : 'f32', V = 'vec4<' + T + '>', M = 'mat4x4<' + T + '>';
    // accumulation type: f16 path can accumulate in f32 (cfg.acc32) while storing f16
    var A = (f16 && !cfg.acc32) ? 'f16' : 'f32', AV = 'vec4<' + A + '>', AM = 'mat4x4<' + A + '>', cvt = A !== T;
    var interFmt = f16 ? 'rgba16float' : 'rgba32float';
    var R = (first.k - 1) >> 1;
    var scalarIn = isFirst; // network input = luma (1 channel)
    var GI = scalarIn ? 1 : Math.ceil(first.cin / 4);
    var spatial = first.k > 1;
    var PX, PY;
    if (!spatial) { PX = cfg.pointwisePx; PY = cfg.pointwisePy; }
    else if (scalarIn) { PX = cfg.scalarPx; PY = cfg.scalarPy; }
    else { PX = cfg.px; PY = cfg.py; }
    var WX = cfg.wx, WY = cfg.wy;
    var TW = WX * PX, TH = WY * PY, TWH = TW + 2 * R, THH = TH + 2 * R, TN = TWH * THH;
    var elemBytes = (scalarIn ? 1 : 4) * (f16 ? 2 : 4);
    var direct = !!cfg.direct; // read the input window straight from the texture (no workgroup staging)
    var smemBytes = spatial && !direct ? GI * TN * elemBytes : 0;
    var nPix = PX * PY;
    var lit = function (x) { return fmtNum(x, A === 'f16'); };
    var consts = [], body = [], bindings = [], bi = 0, weightMats = null;
    var bind = function (decl, entry) { bindings.push('@group(0) @binding(' + bi + ') ' + decl.replace('$', 'b' + bi)); entry.binding = bi; bi++; return entry; };
    var layout = [];
    layout.push(bind('var<uniform> U : Params;', { kind: 'uniform' }));
    var needRgb = isFirst || isLast || K.layers.some(function (li) { return layers[li].skip === -1; });
    if (needRgb) layout.push(bind('var rgbIn : texture_2d<f32>;', { kind: 'rgb' }));
    if (!isFirst) layout.push(bind('var srcT : texture_2d_array<f32>;', { kind: 'tex', layer: li0 - 1 }));
    var skipVar = {};
    K.layers.forEach(function (li) {
      var s = layers[li].skip;
      if (s !== null && s >= 0 && P.kernelOf[s] !== ki && !(s === li0 - 1 && !isFirst) && !skipVar[s]) {
        var e = bind('var sk' + s + ' : texture_2d_array<f32>;', { kind: 'tex', layer: s });
        layout.push(e); skipVar[s] = 'sk' + s;
      }
    });
    if (!isFirst) K.layers.forEach(function (li) { if (layers[li].skip === li0 - 1) skipVar[li0 - 1] = 'srcT'; });
    var stores = []; // layers of this kernel that must be written out
    K.layers.forEach(function (li) { if (P.slot[li] !== undefined) stores.push(li); });
    var dstVar = {};
    stores.forEach(function (li) {
      layout.push(bind('var dst' + li + ' : texture_storage_2d_array<' + interFmt + ', write>;', { kind: 'store', layer: li }));
      dstVar[li] = 'dst' + li;
    });
    if (isLast) layout.push(bind('var outT : texture_storage_2d<' + outFmt + ', write>;', { kind: 'out' }));

    var head = [];
    if (f16) head.push('enable f16;');
    head.push('struct Params { size : vec2<i32>, pad : vec2<i32> };');
    head.push('const K709 = vec3<f32>(' + K709.join(', ') + ');');

    // weights as module-scope constants
    function wMat(li, go, gi, ty, tx) {
      var L = layers[li], k = L.k, cin = L.cin, e = [];
      for (var j = 0; j < 4; j++) for (var i = 0; i < 4; i++) {
        var co = 4 * go + i, cj = 4 * gi + j;
        e.push(co < L.cout && cj < cin ? L.w[((co * cin + cj) * k + ty) * k + tx] : 0);
      }
      return e;
    }
    function wVecScalar(li, go, ty, tx) { // cin == 1
      var L = layers[li], k = L.k, e = [];
      for (var i = 0; i < 4; i++) { var co = 4 * go + i; e.push(co < L.cout ? L.w[(co * k + ty) * k + tx] : 0); }
      return e;
    }
    function vec4Of(arr, go, n) { var e = []; for (var i = 0; i < 4; i++) e.push(4 * go + i < n ? arr[4 * go + i] : 0); return e; }
    var constIds = 0, constCache = new Map();
    function constOf(prefix, type, e, allowZero) {
      if (!allowZero && e.every(function (x) { return x === 0; })) return null;
      var text = type + '(' + e.map(lit).join(', ') + ')', name = constCache.get(text);
      if (!name) {
        name = prefix + (constIds++);
        constCache.set(text, name);
        consts.push('const ' + name + ' = ' + text + ';');
      }
      return name;
    }
    function constMat(e) { return constOf('W', AM, e, false); }
    function constVec(e, allowZero) { return constOf('C', AV, e, allowZero); }

    // ---- prologue: indices
    body.push('let size = U.size;');
    body.push('let tileOrg = vec2<i32>(wid.xy) * vec2<i32>(' + TW + ', ' + TH + ');');
    body.push('let pix0 = tileOrg + vec2<i32>(lid.xy) * vec2<i32>(' + PX + ', ' + PY + ');');
    var smem = [];
    function texWin(gi, xe, ye) { // direct window load (clamped coordinates = replicate padding)
      return scalarIn ? T + '(dot(textureLoad(rgbIn, vec2<i32>(' + xe + ', ' + ye + '), 0).rgb, K709))'
        : V + '(textureLoad(srcT, vec2<i32>(' + xe + ', ' + ye + '), ' + gi + ', 0))';
    }
    if (spatial && direct) {
      for (var dxi = 0; dxi < PX + 2 * R; dxi++) body.push('let xc' + dxi + ' = clamp(pix0.x + ' + (dxi - R) + ', 0, size.x - 1);');
      for (var dyi = 0; dyi < PY + 2 * R; dyi++) body.push('let yc' + dyi + ' = clamp(pix0.y + ' + (dyi - R) + ', 0, size.y - 1);');
    } else if (spatial) {
      var ST = scalarIn ? T : V;
      smem.push('var<workgroup> tile : array<' + ST + ', ' + (GI * TN) + '>;');
      body.push('for (var i = li; i < ' + TN + 'u; i = i + ' + (WX * WY) + 'u) {');
      body.push('  let q = clamp(tileOrg + vec2<i32>(i32(i % ' + TWH + 'u), i32(i / ' + TWH + 'u)) - vec2<i32>(' + R + '), vec2<i32>(0), size - vec2<i32>(1));');
      if (scalarIn) body.push('  tile[i] = ' + T + '(dot(textureLoad(rgbIn, q, 0).rgb, K709));');
      else for (var g = 0; g < GI; g++) body.push('  tile[' + (g * TN) + 'u + i] = ' + V + '(textureLoad(srcT, q, ' + g + ', 0));');
      body.push('}');
      body.push('workgroupBarrier();');
      body.push('let t0 = i32(lid.y) * ' + (PY * TWH) + ' + i32(lid.x) * ' + PX + ';');
    } else {
      // pointwise first layer: load the pixel(s) directly
      for (var p = 0; p < nPix; p++) {
        var ox = p % PX, oy = (p / PX) | 0;
        body.push('let q' + p + ' = min(pix0 + vec2<i32>(' + ox + ', ' + oy + '), size - vec2<i32>(1));');
      }
    }

    // ---- per-layer compute
    var cur = []; // cur[p][g] = expression name of current tensor values for pixel p, group g
    var layerVals = {}; // li -> values (for in-kernel skips)
    K.layers.forEach(function (li, idx) {
      var L = layers[li], GO = Math.ceil(L.cout / 4), acc = [];
      for (var p = 0; p < nPix; p++) {
        acc.push([]);
        for (var go = 0; go < GO; go++) {
          var bname = constVec(vec4Of(L.b, go, L.cout), true);
          var a = 'a' + li + '_' + p + '_' + go;
          body.push('var ' + a + ' = ' + bname + ';');
          acc[p].push(a);
        }
      }
      if (idx === 0 && spatial && cfg.weights === 'uniform') {
        // weights in a uniform buffer, loops over taps (small code, fast compile)
        var kq = L.k, mats = [];
        if (scalarIn) {
          var vecs = [];
          for (var tq = 0; tq < kq * kq; tq++) for (var gq = 0; gq < GO; gq++) vecs.push(wVecScalar(li, gq, (tq / kq) | 0, tq % kq));
          for (var vq = 0; vq < vecs.length; vq += 4) {
            var mq = [];
            for (var cq = 0; cq < 4; cq++) mq = mq.concat(vecs[vq + cq] || [0, 0, 0, 0]);
            mats.push(mq);
          }
        } else {
          for (var tq2 = 0; tq2 < kq * kq; tq2++) for (var gq2 = 0; gq2 < GI; gq2++) for (var oq = 0; oq < GO; oq++)
            mats.push(wMat(li, oq, gq2, (tq2 / kq) | 0, tq2 % kq));
        }
        weightMats = mats;
        layout.push(bind('var<uniform> Wt : array<' + M + ', ' + mats.length + '>;', { kind: 'weights' }));
        body.push('for (var ty = 0; ty < ' + kq + '; ty = ty + 1) {');
        body.push('  for (var tx = 0; tx < ' + kq + '; tx = tx + 1) {');
        if (!direct) body.push('    let tb = t0 + ty * ' + TWH + ' + tx;');
        body.push('    let wb = (ty * ' + kq + ' + tx) * ' + (scalarIn ? GO : GI * GO) + ';');
        for (var gu = 0; gu < GI; gu++) {
          for (var pu = 0; pu < nPix; pu++) {
            var pxu = pu % PX, pyu = (pu / PX) | 0;
            var tl = direct ? texWin(gu, 'clamp(pix0.x + tx + ' + (pxu - R) + ', 0, size.x - 1)', 'clamp(pix0.y + ty + ' + (pyu - R) + ', 0, size.y - 1)')
              : 'tile[' + (gu * TN) + ' + tb + ' + (pyu * TWH + pxu) + ']';
            body.push('    let s' + gu + '_' + pu + ' = ' + (cvt ? (scalarIn ? A : AV) + '(' + tl + ')' : tl) + ';');
          }
          for (var gou = 0; gou < GO; gou++) {
            var wexp = scalarIn ? 'Wt[(wb + ' + gou + ') >> 2u][(wb + ' + gou + ') & 3]' : 'Wt[wb + ' + (gu * GO + gou) + ']';
            if (cvt) wexp = (scalarIn ? AV : AM) + '(' + wexp + ')';
            body.push('    { let w = ' + wexp + ';');
            for (var pu2 = 0; pu2 < nPix; pu2++) {
              var au = acc[pu2][gou];
              body.push('      ' + au + ' = ' + (scalarIn ? 'fma(w, ' + AV + '(s' + gu + '_' + pu2 + '), ' + au + ')' : au + ' + w * s' + gu + '_' + pu2) + ';');
            }
            body.push('    }');
          }
        }
        body.push('  }');
        body.push('}');
      } else if (idx === 0 && spatial) {
        var kk = L.k;
        for (var gi = 0; gi < GI; gi++) {
          // window values
          var wn = function (wy, wx) { return 's' + gi + '_' + wy + '_' + wx; };
          for (var wy = 0; wy < PY + 2 * R; wy++) for (var wx = 0; wx < PX + 2 * R; wx++)
            body.push('let ' + wn(wy, wx) + ' = ' + (cvt ? (scalarIn ? A : AV) + '(' : '') + (direct ? texWin(gi, 'xc' + wx, 'yc' + wy) : 'tile[' + (gi * TN) + ' + t0 + ' + (wy * TWH + wx) + ']') + (cvt ? ')' : '') + ';');
          for (var ty = 0; ty < kk; ty++) for (var tx = 0; tx < kk; tx++) for (var go2 = 0; go2 < GO; go2++) {
            var name = scalarIn ? constVec(wVecScalar(li, go2, ty, tx)) : constMat(wMat(li, go2, gi, ty, tx));
            if (!name) continue;
            for (var p2 = 0; p2 < nPix; p2++) {
              var px = p2 % PX, py = (p2 / PX) | 0, a2 = acc[p2][go2];
              body.push(a2 + ' = ' + (scalarIn ? 'fma(' + name + ', ' + AV + '(' + wn(py + ty, px + tx) + '), ' + a2 + ')' : a2 + ' + ' + name + ' * ' + wn(py + ty, px + tx)) + ';');
            }
          }
        }
      } else {
        // pointwise layer: input = cur (or direct load for a pointwise first layer)
        var inG = idx === 0 ? GI : Math.ceil(L.cin / 4);
        for (var p3 = 0; p3 < nPix; p3++) {
          for (var gi3 = 0; gi3 < inG; gi3++) {
            var src;
            if (idx === 0) {
              src = scalarIn ? 'x' + li + '_' + p3 : 'x' + li + '_' + p3 + '_' + gi3;
              if (scalarIn) body.push('let ' + src + ' = ' + A + '(dot(textureLoad(rgbIn, q' + p3 + ', 0).rgb, K709));');
              else body.push('let ' + src + ' = ' + AV + '(textureLoad(srcT, q' + p3 + ', ' + gi3 + ', 0));');
            } else src = cur[p3][gi3];
            for (var go3 = 0; go3 < GO; go3++) {
              var nm = (idx === 0 && scalarIn) ? constVec(wVecScalar(li, go3, 0, 0)) : constMat(wMat(li, go3, gi3, 0, 0));
              if (!nm) continue;
              var a3 = acc[p3][go3];
              body.push(a3 + ' = ' + ((idx === 0 && scalarIn) ? 'fma(' + nm + ', ' + AV + '(' + src + '), ' + a3 + ')' : a3 + ' + ' + nm + ' * ' + src) + ';');
            }
            if (idx === 0 && scalarIn) break;
          }
        }
      }
      // skip (pre-activation)
      if (L.skip !== null) {
        for (var p4 = 0; p4 < nPix; p4++) {
          var ox4 = p4 % PX, oy4 = (p4 / PX) | 0;
          var qc = 'min(pix0 + vec2<i32>(' + ox4 + ', ' + oy4 + '), size - vec2<i32>(1))';
          for (var g4 = 0; g4 < GO; g4++) {
            var sv;
            if (L.skip === -1) sv = AV + '(' + A + '(dot(textureLoad(rgbIn, ' + qc + ', 0).rgb, K709)), 0.0, 0.0, 0.0)';
            else if (layerVals[L.skip]) sv = layerVals[L.skip][p4][g4];
            else sv = AV + '(textureLoad(' + skipVar[L.skip] + ', ' + qc + ', ' + g4 + ', 0))';
            body.push(acc[p4][g4] + ' = ' + acc[p4][g4] + ' + ' + sv + ';');
          }
        }
      }
      // activation
      var outVals = [];
      for (var p5 = 0; p5 < nPix; p5++) {
        outVals.push([]);
        for (var g5 = 0; g5 < GO; g5++) {
          var a5 = acc[p5][g5], v5 = 'v' + li + '_' + p5 + '_' + g5;
          if (L.act === 'relu' || L.act === 'crelu') body.push('let ' + v5 + ' = max(' + a5 + ', ' + AV + '(0.0));');
          else if (L.act === 'prelu') {
            var al = constVec(vec4Of(L.alpha, g5, L.cout), true);
            body.push('let ' + v5 + ' = select(' + a5 + ' * ' + al + ', ' + a5 + ', ' + a5 + ' > ' + AV + '(0.0));');
          } else body.push('let ' + v5 + ' = ' + a5 + ';');
          outVals[p5].push(v5);
        }
        if (L.act === 'crelu') for (var g6 = 0; g6 < GO; g6++) {
          var v6 = 'n' + li + '_' + p5 + '_' + g6;
          body.push('let ' + v6 + ' = max(-' + acc[p5][g6] + ', ' + AV + '(0.0));');
          outVals[p5].push(v6);
        }
      }
      layerVals[li] = outVals;
      cur = outVals;
      if (dstVar[li]) {
        for (var p7 = 0; p7 < nPix; p7++) {
          var ox7 = p7 % PX, oy7 = (p7 / PX) | 0;
          body.push('if (all(pix0 + vec2<i32>(' + ox7 + ', ' + oy7 + ') < size)) {');
          for (var g7 = 0; g7 < outVals[p7].length; g7++)
            body.push('  textureStore(' + dstVar[li] + ', pix0 + vec2<i32>(' + ox7 + ', ' + oy7 + '), ' + g7 + ', vec4<f32>(' + outVals[p7][g7] + '));');
          body.push('}');
        }
      }
    });

    // ---- final epilogue: depth-to-space + bicubic RGB + luma delta
    if (isLast) {
      var CW = PX + 4, CH = PY + 4;
      for (var cx = 0; cx < CW; cx++) body.push('let cx' + cx + ' = clamp(pix0.x + ' + (cx - 2) + ', 0, size.x - 1);');
      for (var cy = 0; cy < CH; cy++) body.push('let cy' + cy + ' = clamp(pix0.y + ' + (cy - 2) + ', 0, size.y - 1);');
      for (cy = 0; cy < CH; cy++) for (cx = 0; cx < CW; cx++)
        body.push('let r' + cy + '_' + cx + ' = textureLoad(rgbIn, vec2<i32>(cx' + cx + ', cy' + cy + '), 0).rgb;');
      // horizontal pass: h{row}_{outcol}
      for (cy = 0; cy < CH; cy++) for (var oc = 0; oc < 2 * PX; oc++) {
        var pxo = oc >> 1, dx = oc & 1, c0 = pxo + dx, w = BIC[dx];
        body.push('let h' + cy + '_' + oc + ' = ' + w.map(function (wt, t) { return 'r' + cy + '_' + (c0 + t) + ' * ' + fmtNum(wt, false); }).join(' + ') + ';');
      }
      for (var p8 = 0; p8 < nPix; p8++) {
        var ox8 = p8 % PX, oy8 = (p8 / PX) | 0;
        var y8 = 'y' + p8;
        body.push('let ' + y8 + ' = vec4<f32>(' + cur[p8][0] + ');');
        body.push('if (all(pix0 + vec2<i32>(' + ox8 + ', ' + oy8 + ') < size)) {');
        for (var dy = 0; dy < 2; dy++) for (var dx2 = 0; dx2 < 2; dx2++) {
          var oc2 = 2 * ox8 + dx2, r0 = oy8 + dy, wv = BIC[dy];
          var up = '(' + wv.map(function (wt, t) { return 'h' + (r0 + t) + '_' + oc2 + ' * ' + fmtNum(wt, false); }).join(' + ') + ')';
          var lane = y8 + '[' + (2 * dy + dx2) + ']';
          var rgb = model.base === 'bicubic' ? 'max(u + vec3<f32>(' + lane + '), vec3<f32>(0.0))'
            : 'max(u + vec3<f32>(' + lane + ' - dot(u, K709)), vec3<f32>(0.0))';
          body.push('  { let u = ' + up + '; textureStore(outT, 2 * (pix0 + vec2<i32>(' + ox8 + ', ' + oy8 + ')) + vec2<i32>(' + dx2 + ', ' + dy + '), vec4<f32>(' + rgb + ', 1.0)); }');
        }
        body.push('}');
      }
    }

    var code = head.concat(bindings, consts, smem,
      ['@compute @workgroup_size(' + WX + ', ' + WY + ', 1)',
        'fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>, @builtin(local_invocation_index) li : u32) {'],
      body.map(function (s) { return '  ' + s; }), ['}']).join('\n');
    return { code: code, layout: layout, tile: [TW, TH], smemBytes: smemBytes, weightMats: weightMats, label: 'truehdr-sr-' + model.id + '-k' + ki };
  }

  // Shrinks the tile (workgroup rows, then pixels per invocation) until the staged input fits.
  function genFitting(m, layers, P, ki, f16, cfg, outFmt, limit) {
    var c = Object.assign({}, cfg);
    for (;;) {
      var g = genKernel(m, layers, P, ki, f16, c, outFmt);
      if (g.smemBytes <= limit) return g;
      if (c.wy > 4) c.wy >>= 1;
      else if (c.py > 1) { c.py >>= 1; c.scalarPy = Math.min(c.scalarPy, c.py); }
      else if (c.px > 1) { c.px >>= 1; c.scalarPx = Math.min(c.scalarPx, c.px); }
      else if (c.wx > 4) c.wx >>= 1;
      else throw new Error('TrueHDR SR: kernel ' + ki + ' needs ' + g.smemBytes + ' B workgroup memory');
    }
  }

  // ---------------------------------------------------------------- GPU objects
  function bglFor(device, layout, f16, outFmt) {
    var interFmt = f16 ? 'rgba16float' : 'rgba32float';
    return device.createBindGroupLayout({
      entries: layout.map(function (e) {
        var v = GPUShaderStage.COMPUTE;
        if (e.kind === 'uniform' || e.kind === 'weights') return { binding: e.binding, visibility: v, buffer: { type: 'uniform' } };
        if (e.kind === 'rgb') return { binding: e.binding, visibility: v, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } };
        if (e.kind === 'tex') return { binding: e.binding, visibility: v, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } };
        if (e.kind === 'store') return { binding: e.binding, visibility: v, storageTexture: { access: 'write-only', format: interFmt, viewDimension: '2d-array' } };
        return { binding: e.binding, visibility: v, storageTexture: { access: 'write-only', format: outFmt, viewDimension: '2d' } };
      })
    });
  }

  var pipelineCache = typeof WeakMap === 'function' ? new WeakMap() : null;

  async function getBuilt(device, m, f16, cfg, outFmt) {
    var layers = prepareModel(m);
    var P = plan(layers);
    var maxSmem = (device.limits && device.limits.maxComputeWorkgroupStorageSize) || 16384;
    var maxInv = (device.limits && device.limits.maxComputeInvocationsPerWorkgroup) || 256;
    if (cfg.wx * cfg.wy > maxInv) throw new Error('TrueHDR SR: workgroup too large');

    var kernels = [];
    for (var ki = 0; ki < P.kernels.length; ki++) kernels.push(genFitting(m, layers, P, ki, f16, cfg, outFmt, Math.min(maxSmem, 16384)));
    device.pushErrorScope('validation');
    var built = await Promise.all(kernels.map(async function (g) {
      var module = device.createShaderModule({ label: g.label, code: g.code });
      if (module.getCompilationInfo) {
        var info = await module.getCompilationInfo();
        var errs = info.messages.filter(function (x) { return x.type === 'error'; });
        if (errs.length) throw new Error('TrueHDR SR: WGSL compile error in ' + g.label + ': ' + errs.map(function (x) { return x.lineNum + ':' + x.linePos + ' ' + x.message; }).join('; '));
      }
      var bgl = bglFor(device, g.layout, f16, outFmt);
      var wbuf = null;
      if (g.weightMats) {
        var flat = [];
        g.weightMats.forEach(function (mm) { for (var q = 0; q < 16; q++) flat.push(mm[q]); });
        var data = f16 ? f32ArrayToF16(flat) : new Float32Array(flat);
        wbuf = device.createBuffer({ label: g.label + '-weights', size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(wbuf, 0, data);
      }
      var pipeline = await device.createComputePipelineAsync({
        label: g.label, layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: module, entryPoint: 'main' }
      });
      return { gen: g, bgl: bgl, pipeline: pipeline, wbuf: wbuf };
    }));
    var err = await device.popErrorScope();
    if (err) throw new Error('TrueHDR SR: pipeline creation failed: ' + err.message);
    return { P: P, built: built };
  }

  async function create(device, opts) {
    opts = opts || {};
    var m = registry().get(opts.modelId);
    if (!m) throw new Error('TrueHDR SR: unknown model ' + opts.modelId);
    var f16 = !!opts.useF16;
    if (f16 && !device.features.has('shader-f16')) throw new Error('TrueHDR SR: useF16 requires the shader-f16 feature');
    var outFmt = opts.outputFormat || 'rgba16float'; // rgba32float only for verification
    if (outFmt !== 'rgba16float' && outFmt !== 'rgba32float') throw new Error('TrueHDR SR: bad outputFormat');
    var cfg = Object.assign({}, DEFAULT_CONFIG, opts.config || {});
    var compiled;
    if (pipelineCache && !opts.config) {
      var byKey = pipelineCache.get(device);
      if (!byKey) { byKey = new Map(); pipelineCache.set(device, byKey); }
      var ck = m.id + '|' + (f16 ? 1 : 0) + '|' + outFmt;
      var prom = byKey.get(ck);
      if (!prom) {
        prom = getBuilt(device, m, f16, cfg, outFmt).catch(function (e) { byKey.delete(ck); throw e; });
        byKey.set(ck, prom);
      }
      compiled = await prom;
    } else {
      compiled = await getBuilt(device, m, f16, cfg, outFmt);
    }
    var P = compiled.P, built = compiled.built;

    var uniform = device.createBuffer({ label: 'truehdr-sr-size', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    var state = { w: 0, h: 0, textures: [], views: [], out: null, outView: null, staticBG: [], inputBG: [], destroyed: false };
    var interFmt = f16 ? 'rgba16float' : 'rgba32float';

    function freeSized() {
      state.textures.forEach(function (t) { t.destroy(); });
      if (state.out) state.out.destroy();
      state.textures = []; state.views = []; state.out = null; state.outView = null;
      state.staticBG = []; state.inputBG = [];
    }

    function allocate(w, h) {
      freeSized();
      state.w = w; state.h = h;
      device.queue.writeBuffer(uniform, 0, new Int32Array([w, h, 0, 0]));
      P.pool.forEach(function (pt, i) {
        var t = device.createTexture({
          label: 'truehdr-sr-inter' + i, size: { width: w, height: h, depthOrArrayLayers: pt.groups },
          format: interFmt, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });
        state.textures.push(t);
        state.views.push(t.createView({ dimension: '2d-array' }));
      });
      state.out = device.createTexture({
        label: 'truehdr-sr-out', size: { width: 2 * w, height: 2 * h }, format: outFmt,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
      });
      state.outView = state.out.createView();
      built.forEach(function (b, i) {
        state.inputBG.push(new Map());
        state.staticBG.push(b.gen.layout.some(function (e) { return e.kind === 'rgb'; }) ? null : makeBG(i, null));
      });
    }

    function makeBG(i, inputView) {
      var b = built[i];
      return device.createBindGroup({
        layout: b.bgl,
        entries: b.gen.layout.map(function (e) {
          if (e.kind === 'uniform') return { binding: e.binding, resource: { buffer: uniform } };
          if (e.kind === 'weights') return { binding: e.binding, resource: { buffer: b.wbuf } };
          if (e.kind === 'rgb') return { binding: e.binding, resource: inputView };
          if (e.kind === 'tex' || e.kind === 'store') return { binding: e.binding, resource: state.views[P.slot[e.layer]] };
          return { binding: e.binding, resource: state.outView };
        })
      });
    }

    function bindGroup(i, inputView) {
      if (state.staticBG[i]) return state.staticBG[i];
      var cache = state.inputBG[i], bg = cache.get(inputView);
      if (!bg) {
        if (cache.size >= 4) cache.delete(cache.keys().next().value); // bounded: engine normally reuses one view
        bg = makeBG(i, inputView);
        cache.set(inputView, bg);
      }
      return bg;
    }

    function encodeImpl(encoder, inputView, w, h, querySet) {
      if (state.destroyed) throw new Error('TrueHDR SR: upscaler destroyed');
      w = w | 0; h = h | 0;
      if (w < 1 || h < 1) throw new Error('TrueHDR SR: bad size');
      if (w !== state.w || h !== state.h) allocate(w, h);
      if (querySet) {
        for (var i = 0; i < built.length; i++) {
          var b = built[i];
          var pass = encoder.beginComputePass({
            label: b.gen.label,
            timestampWrites: { querySet: querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 }
          });
          pass.setPipeline(b.pipeline);
          pass.setBindGroup(0, bindGroup(i, inputView));
          pass.dispatchWorkgroups(Math.ceil(w / b.gen.tile[0]), Math.ceil(h / b.gen.tile[1]), 1);
          pass.end();
        }
      } else {
        // Single compute pass for all SR kernels avoids 5-6 encoder transitions per frame.
        var pass = encoder.beginComputePass({ label: 'truehdr-sr-' + m.id });
        for (var j = 0; j < built.length; j++) {
          var b2 = built[j];
          pass.setPipeline(b2.pipeline);
          pass.setBindGroup(0, bindGroup(j, inputView));
          pass.dispatchWorkgroups(Math.ceil(w / b2.gen.tile[0]), Math.ceil(h / b2.gen.tile[1]), 1);
        }
        pass.end();
      }
      return { view: state.outView, width: 2 * w, height: 2 * h };
    }

    return {
      modelId: m.id,
      scale: 2,
      encode: function (encoder, inputView, w, h) { return encodeImpl(encoder, inputView, w, h, null); },
      destroy: function () { state.destroyed = true; freeSized(); uniform.destroy(); },
      // diagnostics (bench/tests): per-kernel timestamps, generated code, texture of the output
      _encodeTimed: encodeImpl,
      _passCount: built.length,
      _kernels: built.map(function (b) { return { label: b.gen.label, tile: b.gen.tile, smemBytes: b.gen.smemBytes, code: b.gen.code }; }),
      _outputTexture: function () { return state.out; }
    };
  }

  globalThis.__TRUEHDR_SR__ = {
    version: 1,
    listModels: function () {
      var out = [];
      registry().forEach(function (m) { out.push({ id: m.id, label: m.label, content: m.content, tier: m.tier, macsPerPixel: m.macsPerPixel }); });
      return out;
    },
    create: create,
    // exposed for dev tools (dev/sr): WGSL generation without a device
    _generate: function (modelId, useF16, config) {
      var m = registry().get(modelId);
      if (!m) throw new Error('unknown model ' + modelId);
      var layers = prepareModel(m), P = plan(layers), cfg = Object.assign({}, DEFAULT_CONFIG, config || {});
      return P.kernels.map(function (_, ki) { return genFitting(m, layers, P, ki, !!useF16, cfg, 'rgba16float', 16384); });
    }
  };
})();
