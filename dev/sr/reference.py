"""PyTorch reference for TrueHDR SR models (docs/SR_SPEC.md semantics), loading extension/sr-models.js.

Used to verify the WGSL runtime and any custom-trained model exported in the same format.

  # spec-exact upscale of an image (float64 on CPU), write PNG (and/or raw float32 RGB HWC)
  python3 -I dev/sr/reference.py --model tsr-anime-quality --in-png in.png --out-png out.png
  # raw I/O used by dev/sr/test_runtime.ts: float32 RGB HWC, w x h in, 2w x 2h out
  python3 -I dev/sr/reference.py --model ID --in-raw in.f32 --w W --h H --out-raw out.f32
  # quality check: downscale HR by 2 (bicubic, antialiased), upscale, PSNR vs HR for bicubic and model
  python3 -I dev/sr/reference.py --model ID --eval-hr hr.png [--out-png sr.png]
  # list models
  python3 -I dev/sr/reference.py --list
"""
import argparse
import base64
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEFAULT_MODELS = os.path.join(ROOT, 'extension', 'sr-models.js')
K709 = (0.2126, 0.7152, 0.0722)

_NODE_LOADER = r"""
const vm = require('vm'), fs = require('fs');
const ctx = vm.createContext({});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), ctx, { filename: process.argv[1] });
process.stdout.write(JSON.stringify(ctx.__TRUEHDR_SR_MODELS__ || []));
"""


def load_models(path=DEFAULT_MODELS):
    """Evaluate sr-models.js in a clean JS context (node) and return {id: entry}; last id wins."""
    out = subprocess.run(['node', '-e', _NODE_LOADER, path], check=True, capture_output=True, text=True).stdout
    models = {}
    for e in json.loads(out):
        models[e['id']] = e
    return models


def decode_f16(b64):
    return np.frombuffer(base64.b64decode(b64), dtype='<f2').astype(np.float64)


def decode_q8(wq_b64, ws_b64, cout, fan_in):
    q = np.frombuffer(base64.b64decode(wq_b64), dtype=np.int8).astype(np.float64).reshape(cout, fan_in)
    s = decode_f16(ws_b64).reshape(cout, 1)
    return (q * s).reshape(-1)


def decode_layers(entry, dtype=torch.float64):
    layers = []
    for l in entry['layers']:
        k, cin, cout = int(l['k']), int(l['cin']), int(l['cout'])
        w = decode_q8(l['wq'], l['ws'], cout, cin * k * k) if 'wq' in l else decode_f16(l['w'])
        assert w.size == cout * cin * k * k, ('bad weight size', entry['id'])
        b = decode_f16(l['b'])
        assert b.size == cout
        d = {'k': k, 'cin': cin, 'cout': cout, 'act': l['act'], 'skip': l.get('skip'),
             'w': torch.tensor(w.reshape(cout, cin, k, k), dtype=dtype),
             'b': torch.tensor(b, dtype=dtype)}
        if l['act'] == 'prelu':
            a = decode_f16(l['alpha'])
            assert a.size == cout
            d['alpha'] = torch.tensor(a, dtype=dtype)
        layers.append(d)
    return layers


def macs_per_pixel(entry):
    return sum(int(l['k']) ** 2 * int(l['cin']) * int(l['cout']) for l in entry['layers'])


def bicubic2x(x):
    return F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)


def run_luma(entry, y, layers=None):
    """y: [N,1,H,W] nonlinear luma -> model result Ysr [N,1,2H,2W] (before colour reconstruction)."""
    layers = layers or decode_layers(entry, y.dtype)
    outs = {-1: y}
    x = y
    for i, l in enumerate(layers):
        r = l['k'] // 2
        assert x.shape[1] == l['cin'], ('channel mismatch at layer', i, x.shape[1], l['cin'])
        xp = F.pad(x, (r, r, r, r), mode='replicate') if r else x
        z = F.conv2d(xp, l['w'], l['b'])
        if l['skip'] is not None:
            z = z + outs[l['skip']]
        a = l['act']
        if a == 'relu':
            z = F.relu(z)
        elif a == 'prelu':
            z = torch.where(z > 0, z, z * l['alpha'].view(1, -1, 1, 1))
        elif a == 'crelu':
            z = torch.cat([F.relu(z), F.relu(-z)], 1)
        elif a != 'none':
            raise ValueError('unknown activation ' + a)
        outs[i] = z
        x = z
    assert x.shape[1] == 4, 'last layer must output 4 channels'
    res = F.pixel_shuffle(x, 2)
    if entry['base'] == 'bicubic':
        return bicubic2x(y) + res
    if entry['base'] == 'none':
        return res
    raise ValueError('unknown base ' + entry['base'])


def upscale_rgb(entry, rgb, layers=None):
    """rgb: [N,3,H,W] nonlinear R'G'B' -> [N,3,2H,2W] per the spec's luma-delta colour reconstruction."""
    k = torch.tensor(K709, dtype=rgb.dtype).view(1, 3, 1, 1)
    y = (rgb * k).sum(1, keepdim=True)
    ysr = run_luma(entry, y, layers)
    up = bicubic2x(rgb)
    return torch.clamp(up + (ysr - (up * k).sum(1, keepdim=True)), min=0)


def _read_png(path):
    """8-bit/16-bit PNG -> float64 [1,3,H,W] in [0,1] via ffmpeg (no PIL dependency)."""
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=width,height', '-of', 'csv=p=0', path], check=True, capture_output=True,
                           text=True).stdout.strip().split(',')
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-frames:v', '1', '-f', 'rawvideo',
                          '-pix_fmt', 'rgb48le', '-'], check=True, capture_output=True).stdout
    a = np.frombuffer(raw, dtype='<u2').reshape(h, w, 3).astype(np.float64) / 65535.0
    return torch.tensor(a).permute(2, 0, 1).unsqueeze(0)


def _write_png(path, t):
    a = t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    h, w = a.shape[:2]
    data = np.round(a * 255.0).astype(np.uint8).tobytes()
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (w, h),
                    '-i', '-', '-frames:v', '1', path], input=data, check=True)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=DEFAULT_MODELS)
    ap.add_argument('--model')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--in-raw')
    ap.add_argument('--w', type=int)
    ap.add_argument('--h', type=int)
    ap.add_argument('--out-raw')
    ap.add_argument('--in-png')
    ap.add_argument('--out-png')
    ap.add_argument('--eval-hr')
    args = ap.parse_args()
    models = load_models(args.models)
    if args.list:
        for m in models.values():
            print('%-20s %-8s %-8s base=%-8s MAC/px=%-6d (declared %s) layers=%d  %s' % (
                m['id'], m['content'], m['tier'], m['base'], macs_per_pixel(m), m['macsPerPixel'],
                len(m['layers']), m['license']))
        return
    entry = models[args.model]
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    if args.in_raw:
        a = np.fromfile(args.in_raw, dtype='<f4').reshape(args.h, args.w, 3).astype(np.float64)
        rgb = torch.tensor(a).permute(2, 0, 1).unsqueeze(0)
        out = upscale_rgb(entry, rgb)
        out[0].permute(1, 2, 0).numpy().astype('<f4').tofile(args.out_raw)
        return
    if args.in_png:
        out = upscale_rgb(entry, _read_png(args.in_png))
        if args.out_png:
            _write_png(args.out_png, out)
        return
    if args.eval_hr:
        hr = _read_png(args.eval_hr)
        H, W = hr.shape[2] // 2 * 2, hr.shape[3] // 2 * 2
        hr = hr[:, :, :H, :W]
        lr = F.interpolate(hr, size=(H // 2, W // 2), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        sr = upscale_rgb(entry, lr).clamp(0, 1)
        bc = bicubic2x(lr).clamp(0, 1)
        k = torch.tensor(K709, dtype=hr.dtype).view(1, 3, 1, 1)
        yb = lambda t: (t * k).sum(1, keepdim=True)[:, :, 4:-4, 4:-4]  # noqa: E731
        print(json.dumps({'model': entry['id'], 'psnr_y_bicubic': round(psnr(yb(bc), yb(hr)), 3),
                          'psnr_y_model': round(psnr(yb(sr), yb(hr)), 3),
                          'psnr_rgb_bicubic': round(psnr(bc, hr), 3), 'psnr_rgb_model': round(psnr(sr, hr), 3)}))
        if args.out_png:
            _write_png(args.out_png, sr)
        return
    ap.error('nothing to do')


if __name__ == '__main__':
    main()
