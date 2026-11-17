#!/usr/bin/env python3
"""PSNR/SSIM (Y', 4px shave) on DIV2K-valid: bicubic vs models, clean and degraded LR.

  python eval.py --js export/tsr-live-models.js --out runs/eval
  python eval.py --model fast=runs/full/f12x5/latest.pt --model quality=runs/full/q16x6/latest.pt

Valid set (degrade.py --split valid, seed 1234): v0 = clean bicubic LR, v1/v2 = degraded
(same random pipeline as training, never clean). Writes <out>/eval.json and <out>/crops_*.png.
"""
import argparse, json, os, sys
from collections import defaultdict

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srmodel import bicubic2x, load_ckpt, load_js, from_export  # noqa: E402
from data_loader import load_eval_set  # noqa: E402
from metrics import psnr, ssim  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def crop_sheet(rows, names, path, zoom=2):
    """rows: list of lists of uint8 HxW tiles (same size). Writes a labelled PNG grid."""
    th, tw = rows[0][0].shape
    head = 28
    sheet = np.full((head + len(rows) * (th * zoom + 4), len(names) * (tw * zoom + 4)), 255, np.uint8)
    for j, n in enumerate(names):
        cv2.putText(sheet, n, (j * (tw * zoom + 4) + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    for i, r in enumerate(rows):
        for j, t in enumerate(r):
            z = cv2.resize(t, (tw * zoom, th * zoom), interpolation=cv2.INTER_NEAREST)
            y0, x0 = head + i * (th * zoom + 4), j * (tw * zoom + 4)
            sheet[y0:y0 + th * zoom, x0:x0 + tw * zoom] = z
    cv2.imwrite(path, sheet)


def busiest_window(lr, size):
    """Top-left of the size x size LR window with the most gradient energy (deterministic)."""
    g = np.abs(np.diff(lr.astype(np.float32), axis=1))[:-1] + np.abs(np.diff(lr.astype(np.float32), axis=0))[:, :-1]
    ii = cv2.integral(g)
    h, w = g.shape
    best, pos = -1, (0, 0)
    for y in range(8, h - size - 8, 16):
        for x in range(8, w - size - 8, 16):
            e = ii[y + size, x + size] - ii[y, x + size] - ii[y + size, x] + ii[y, x]
            if e > best:
                best, pos = e, (y, x)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', default=[], help='name=checkpoint.pt')
    ap.add_argument('--js', default=None, help='evaluate every model in an exported .js file (f16 weights)')
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'gen', 'valid'))
    ap.add_argument('--limit', type=int, default=0, help='number of valid images (0 = all 100)')
    ap.add_argument('--out', default=os.path.join(HERE, 'runs', 'eval'))
    ap.add_argument('--crops', type=int, default=8)
    a = ap.parse_args()

    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    models = {}
    for s in a.model:
        n, p = s.split('=', 1)
        models[n] = load_ckpt(p)[0].to(dev)
    if a.js:
        for d in load_js(a.js):
            models[d['id']] = from_export(d).to(dev)
    names = ['bicubic'] + list(models)
    data = load_eval_set(a.data, limit=a.limit)
    os.makedirs(a.out, exist_ok=True)

    acc = defaultdict(lambda: defaultdict(list))   # group -> name -> [(psnr, ssim)]
    crop_rows = defaultdict(list)
    crop_stems = sorted({s['stem'] for s in data})[::max(1, len(data) // 3 // max(a.crops, 1))][:a.crops]
    with torch.no_grad():
        for s in data:
            lr = torch.from_numpy(s['lr']).float().div(255)[None, None].to(dev)
            hr = torch.from_numpy(s['hr']).float().div(255)[None, None].to(dev)
            outs = {'bicubic': bicubic2x(lr).clamp(0, 1)}
            for n, m in models.items():
                outs[n] = m(lr).clamp(0, 1)
            groups = ['clean'] if s['k'] == 0 else ['degraded', 'degraded/' + s['codec']]
            for n, o in outs.items():
                pv, sv = psnr(o, hr).item(), ssim(o, hr).item()
                for g in groups:
                    acc[g][n].append((pv, sv))
            if s['stem'] in crop_stems and s['k'] in (0, 1):
                P = 96
                y, x = busiest_window(s['lr'], P)
                lr_up = cv2.resize(s['lr'][y:y + P, x:x + P], (2 * P, 2 * P), interpolation=cv2.INTER_NEAREST)
                tiles = [lr_up] + [np.round(outs[n][0, 0, 2 * y:2 * y + 2 * P, 2 * x:2 * x + 2 * P].cpu().numpy() * 255)
                                   .astype(np.uint8) for n in names] + [s['hr'][2 * y:2 * y + 2 * P, 2 * x:2 * x + 2 * P]]
                crop_rows['clean' if s['k'] == 0 else 'degraded'].append(tiles)

    res = {g: {n: dict(psnr=float(np.mean([v[0] for v in l])), ssim=float(np.mean([v[1] for v in l])), n=len(l))
               for n, l in d.items()} for g, d in acc.items()}
    json.dump(dict(models=names, results=res), open(os.path.join(a.out, 'eval.json'), 'w'), indent=1)
    order = ['clean', 'degraded'] + sorted(g for g in res if g.startswith('degraded/'))
    print('| set | n | ' + ' | '.join(f'{n} PSNR / SSIM' for n in names) + ' |')
    print('|---|---|' + '---|' * len(names))
    for g in order:
        if g in res:
            r = res[g]
            print(f'| {g} | {r["bicubic"]["n"]} | ' + ' | '.join(f'{r[n]["psnr"]:.2f} / {r[n]["ssim"]:.4f}' for n in names) + ' |')
    for g, rows in crop_rows.items():
        p = os.path.join(a.out, f'crops_{g}.png')
        crop_sheet(rows, ['LR (nearest)'] + names + ['HR'], p)
        print('crops:', p)


if __name__ == '__main__':
    main()
