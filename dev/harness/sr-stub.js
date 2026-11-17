// Development stub of the SR runtime API in docs/SR_SPEC.md (NOT shipped).
// 2x upscale = PyTorch-exact bicubic (a = -0.75, align_corners=False, clamped indices) on R'G'B',
// i.e. the spec's model with an all-zero residual. Lets the engine's ML path run end to end.
(() => {
  if (globalThis.__TRUEHDR_SR__) return;

  const WGSL = /* wgsl */ `
    @group(0) @binding(0) var src: texture_2d<f32>;
    @group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

    fn cubic(x: f32) -> f32 {
      let a = -0.75;
      let ax = abs(x);
      if (ax <= 1.0) { return ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0; }
      if (ax < 2.0) { return ((a * ax - 5.0 * a) * ax + 8.0 * a) * ax - 4.0 * a; }
      return 0.0;
    }

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
      let od = textureDimensions(dst);
      if (gid.x >= od.x || gid.y >= od.y) { return; }
      let mx = vec2i(textureDimensions(src)) - vec2i(1);
      let sp = (vec2f(gid.xy) + vec2f(0.5)) * 0.5 - vec2f(0.5);
      let fl = floor(sp);
      let t = sp - fl;
      let i0 = vec2i(fl);
      var acc = vec3f(0.0);
      for (var j = -1; j <= 2; j = j + 1) {
        let wy = cubic(t.y - f32(j));
        for (var i = -1; i <= 2; i = i + 1) {
          let wx = cubic(t.x - f32(i));
          acc = acc + (wx * wy) * textureLoad(src, clamp(i0 + vec2i(i, j), vec2i(0), mx), 0).rgb;
        }
      }
      textureStore(dst, vec2i(gid.xy), vec4f(max(acc, vec3f(0.0)), 1.0));
    }
  `;

  const MODELS = ['live', 'anime'].flatMap((content) => ['fast', 'quality'].map((tier) => ({
    id: `tsr-${content}-${tier}`,
    label: `STUB bicubic (${content} · ${tier})`,
    content,
    tier,
    macsPerPixel: 0
  })));

  const pipelines = new WeakMap(); // device -> pipeline

  function pipelineFor(device) {
    let p = pipelines.get(device);
    if (!p) {
      p = device.createComputePipeline({
        label: 'sr-stub-bicubic2x',
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: WGSL }), entryPoint: 'main' }
      });
      pipelines.set(device, p);
    }
    return p;
  }

  globalThis.__TRUEHDR_SR__ = {
    version: 1,
    stub: true,
    listModels() {
      return MODELS.map((m) => ({ ...m }));
    },
    async create(device, { modelId, useF16 } = {}) {
      if (!MODELS.some((m) => m.id === modelId)) throw new Error(`unknown model ${modelId}`);
      const pipeline = pipelineFor(device);
      let out = null;
      let outView = null;
      let ow = 0;
      let oh = 0;
      let lastIn = null;
      let bg = null;
      return {
        modelId,
        scale: 2,
        useF16: Boolean(useF16),
        encode(encoder, inputView, w, h) {
          if (!out || ow !== 2 * w || oh !== 2 * h) {
            if (out) out.destroy();
            ow = 2 * w;
            oh = 2 * h;
            out = device.createTexture({
              size: [ow, oh, 1],
              format: 'rgba16float',
              usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
            });
            outView = out.createView();
            lastIn = null;
          }
          if (inputView !== lastIn) {
            bg = device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: [{ binding: 0, resource: inputView }, { binding: 1, resource: outView }]
            });
            lastIn = inputView;
          }
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(ow / 8), Math.ceil(oh / 8));
          pass.end();
          return { view: outView, width: ow, height: oh };
        },
        destroy() {
          if (out) out.destroy();
          out = null;
          outView = null;
          bg = null;
          lastIn = null;
        }
      };
    }
  };
})();
