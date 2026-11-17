"""SR model family exactly as defined in docs/SR_SPEC.md.

A model is a list of layer dicts {k, cin, cout, act, skip?} (the export schema) plus base
'bicubic'|'none'. Every conv uses replicate padding; the last layer has cout=4 and is followed
by PixelShuffle(2). skip=i adds layer i's post-activation output (i=-1: input luma) to this
layer's pre-activation output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

K709 = (0.2126, 0.7152, 0.0722)

# Candidate architectures (all channel counts multiples of 4; MAC/px at input resolution).
def _chain(widths, ks, skips=None, act='prelu'):
    """widths: [1, c1, ..., 4]; ks: kernel per layer; skips: {layer_index: skip_from}."""
    skips = skips or {}
    layers = []
    n = len(widths) - 1
    for i in range(n):
        L = dict(k=ks[i], cin=widths[i], cout=widths[i + 1], act='none' if i == n - 1 else act)
        if i in skips:
            L['skip'] = skips[i]
        layers.append(L)
    return layers

ARCHS = {
    # ---- fast tier (<= 4500 MAC/px) ----
    'f12x5':  _chain([1, 12, 12, 12, 12, 4], [3, 3, 3, 3, 3], {3: 1}),                 # 4428, RF 11
    'f16x3':  _chain([1, 16, 16, 4], [5, 3, 3]),                                        # 3280, RF 9
    'f8x8':   _chain([1, 8, 8, 8, 8, 8, 8, 8, 4], [5, 3, 3, 3, 3, 3, 3, 3], {3: 1, 5: 3}),  # 3944, RF 19
    # ---- quality tier (<= 12000 MAC/px) ----
    'q16x6':  _chain([1, 16, 16, 16, 16, 16, 4], [5, 3, 3, 3, 3, 3], {2: 0, 4: 2}),     # 10192, RF 15
    'q20x5':  _chain([1, 20, 20, 20, 20, 4], [3, 3, 3, 3, 3], {3: 1}),                  # 11700, RF 11
    'q12x10': _chain([1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 4], [3] * 10,
                     {2: 0, 4: 2, 6: 4, 8: 6}),                                           # 10908, RF 21
}


def macs_per_pixel(layers):
    return sum(L['k'] * L['k'] * L['cin'] * L['cout'] for L in layers)


def bicubic2x(y):
    return F.interpolate(y, scale_factor=2, mode='bicubic', align_corners=False)


class SRNet(nn.Module):
    def __init__(self, layers, base='bicubic'):
        super().__init__()
        self.spec = [dict(L) for L in layers]
        self.base = base
        assert layers[-1]['cout'] == 4 and layers[0]['cin'] == 1
        self.convs = nn.ModuleList()
        self.prelus = nn.ModuleDict()
        prev_out = 1
        for i, L in enumerate(layers):
            assert L['k'] in (1, 3, 5)
            assert L['cin'] == prev_out, f'layer {i}: cin {L["cin"]} != {prev_out}'
            c = nn.Conv2d(L['cin'], L['cout'], L['k'], padding=L['k'] // 2, padding_mode='replicate')
            self.convs.append(c)
            if L['act'] == 'prelu':
                self.prelus[str(i)] = nn.PReLU(L['cout'], init=0.1)
            prev_out = L['cout'] * (2 if L['act'] == 'crelu' else 1)
        # small init of the residual head so training starts at the bicubic base
        nn.init.normal_(self.convs[-1].weight, std=1e-3)
        nn.init.zeros_(self.convs[-1].bias)

    @property
    def macs(self):
        return macs_per_pixel(self.spec)

    def residual(self, y):
        outs = []
        x = y
        for i, (L, conv) in enumerate(zip(self.spec, self.convs)):
            h = conv(x)
            if 'skip' in L and L['skip'] is not None:
                h = h + (y if L['skip'] == -1 else outs[L['skip']])
            a = L['act']
            if a == 'relu':
                h = F.relu(h)
            elif a == 'prelu':
                h = self.prelus[str(i)](h)
            elif a == 'crelu':
                h = torch.cat([F.relu(h), F.relu(-h)], 1)
            outs.append(h)
            x = h
        return F.pixel_shuffle(x, 2)

    def forward(self, y):
        r = self.residual(y)
        if self.base == 'bicubic':
            r = r + bicubic2x(y)
        return r


def build(arch_name, base='bicubic'):
    return SRNet(ARCHS[arch_name], base)


# ---------------- export format helpers (SR_SPEC model file) ----------------
import base64, json, re  # noqa: E402
import numpy as np  # noqa: E402


def f16_b64(t):
    a = t.detach().float().cpu().numpy().astype('<f2')
    return base64.b64encode(a.tobytes()).decode('ascii')


def b64_f16(s, shape):
    return torch.from_numpy(np.frombuffer(base64.b64decode(s), dtype='<f2').astype(np.float32).reshape(shape))


def to_export(model, **fields):
    layers = []
    for i, (L, conv) in enumerate(zip(model.spec, model.convs)):
        e = dict(k=L['k'], cin=L['cin'], cout=L['cout'], act=L['act'])
        if L.get('skip') is not None:
            e['skip'] = L['skip']
        e['w'] = f16_b64(conv.weight)          # [cout][cin][k][k]
        e['b'] = f16_b64(conv.bias)
        if L['act'] == 'prelu':
            e['alpha'] = f16_b64(model.prelus[str(i)].weight)
        layers.append(e)
    d = dict(fields)
    d.update(scale=2, base=model.base, macsPerPixel=model.macs, layers=layers)
    return d


def from_export(d):
    spec = [{k: L[k] for k in ('k', 'cin', 'cout', 'act', 'skip') if k in L} for L in d['layers']]
    m = SRNet(spec, d['base'])
    with torch.no_grad():
        for i, (L, conv) in enumerate(zip(d['layers'], m.convs)):
            conv.weight.copy_(b64_f16(L['w'], conv.weight.shape))
            conv.bias.copy_(b64_f16(L['b'], conv.bias.shape))
            if L['act'] == 'prelu':
                m.prelus[str(i)].weight.copy_(b64_f16(L['alpha'], (L['cout'],)))
    return m.eval()


def load_js(path):
    """Parse a model file written by export.py (one JSON object per push line)."""
    out = []
    for line in open(path):
        mm = re.match(r'\s*globalThis\.__TRUEHDR_SR_MODELS__\.push\((\{.*\})\);\s*$', line)
        if mm:
            out.append(json.loads(mm.group(1)))
    return out


def load_ckpt(path):
    st = torch.load(path, map_location='cpu')
    m = SRNet(st['spec'], st.get('base', 'bicubic'))
    m.load_state_dict(st['model'])
    return m.eval(), st
