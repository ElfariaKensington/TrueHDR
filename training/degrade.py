#!/usr/bin/env python3
"""Offline degradation pipeline: DIV2K HR (R'G'B' PNG) -> YouTube-like degraded LR luma.

Per HR image (read straight from the DIV2K zip, never extracted):
  * HR region = image cropped by MARGIN px on each side, size floored to a multiple of 4.
    Target = Y' = 0.2126R' + 0.7152G' + 0.0722B' of that region (uint8 PNG, round(Y'*255)).
  * Each LR variant (all the same size = HR region / 2) is built from R'G'B' float:
      1. optional Gaussian blur (prob BLUR_P, sigma_LR ~ U(BLUR_SIGMA); applied at HR as 2*sigma_LR)
      2. 2x downscale, kernel ~ KERNELS (PIL antialiased bicubic a=-0.5 / lanczos3, cv2 INTER_AREA)
      3. R'G'B' -> 8-bit BT.709 limited-range YUV 4:2:0 (chroma = 2x2 mean)
      4. codec ~ CODECS: 'clean' = no codec (8-bit 4:2:0 only), else ffmpeg encode + decode:
           mode 'intra' = single frame (I-frame), 'clip' = N in [3,6] frames of a pan
           (velocity v ~ U{-6..6}^2 HR px/frame, P(static)=0.2) ending on the HR region; the last
           decoded frame (inter-coded) is kept.
      5. decoded YUV -> R'G'B' (bilinear chroma upsample, clip [0,1]) -> Y' (uint8 PNG).
Everything is seeded per (seed, split, image index) so runs are reproducible.
"""
import argparse, json, os, shutil, subprocess, sys, time, zipfile
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

FFMPEG = '/opt/homebrew/bin/ffmpeg'
KR, KG, KB = 0.2126, 0.7152, 0.0722
MARGIN = 32                      # HR px; >= max pan displacement (6 px/frame * 5 frames)
KERNELS = (('bicubic', 0.4), ('area', 0.3), ('lanczos', 0.3))
BLUR_P, BLUR_SIGMA = 0.5, (0.1, 0.6)
CODECS = (('clean', 0.15), ('x264', 0.35), ('vp9', 0.25), ('av1', 0.25))
INTRA_P = 0.35
X264_CRF, X264_PRESETS = (18, 40), ('veryfast', 'faster', 'medium', 'slow')
VP9_CRF, VP9_CPU = (24, 56), (3, 4, 5)
AV1_CRF, AV1_PRESETS = (24, 58), (6, 8, 10)
PARAMS_DOC = dict(MARGIN=MARGIN, KERNELS=KERNELS, BLUR_P=BLUR_P, BLUR_SIGMA_LR=BLUR_SIGMA,
                  CODECS=CODECS, INTRA_P=INTRA_P, CLIP_FRAMES=(3, 6), PAN_V=(-6, 6), STATIC_P=0.2,
                  X264_CRF=X264_CRF, X264_PRESETS=X264_PRESETS, VP9_CRF=VP9_CRF, VP9_CPU=VP9_CPU,
                  AV1_CRF=AV1_CRF, AV1_PRESETS=AV1_PRESETS)


def luma(rgb):
    return rgb[..., 0] * KR + rgb[..., 1] * KG + rgb[..., 2] * KB


def to_u8(x):
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def rgb_to_yuv420(rgb):
    y = luma(rgb)
    pb = (rgb[..., 2] - y) / 1.8556
    pr = (rgb[..., 0] - y) / 1.5748
    h, w = y.shape
    pb = pb.reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    pr = pr.reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    q = lambda v: np.clip(np.round(v), 0, 255).astype(np.uint8)
    return q(16 + 219 * y), q(128 + 224 * pb), q(128 + 224 * pr)


def yuv420_to_luma(Y, U, V):
    h, w = Y.shape
    y = (Y.astype(np.float32) - 16) / 219
    pb = cv2.resize((U.astype(np.float32) - 128) / 224, (w, h), interpolation=cv2.INTER_LINEAR)
    pr = cv2.resize((V.astype(np.float32) - 128) / 224, (w, h), interpolation=cv2.INTER_LINEAR)
    r = np.clip(y + 1.5748 * pr, 0, 1)
    b = np.clip(y + 1.8556 * pb, 0, 1)
    g = np.clip((y - KR * (y + 1.5748 * pr) - KB * (y + 1.8556 * pb)) / KG, 0, 1)
    return KR * r + KG * g + KB * b


def downscale(rgb, kernel, sigma_lr):
    if sigma_lr > 0:
        rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=2 * sigma_lr, borderType=cv2.BORDER_REFLECT)
    h, w = rgb.shape[:2]
    size = (w // 2, h // 2)
    if kernel == 'area':
        lr = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    else:
        f = Image.BICUBIC if kernel == 'bicubic' else Image.LANCZOS
        lr = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c]), 'F').resize(size, f))
                       for c in range(3)], -1)
    return np.clip(lr, 0, 1)


def encode_decode(frames, codec, crf, preset, tmpdir):
    h, w = frames[0][0].shape
    raw = b''.join(p.tobytes() for f in frames for p in f)
    ext = {'x264': 'mkv', 'vp9': 'webm', 'av1': 'mkv'}[codec]
    out = os.path.join(tmpdir, 'v.' + ext)
    inp = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
           '-s', f'{w}x{h}', '-r', '30', '-i', '-']
    if codec == 'x264':
        enc = ['-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-threads', '1']
    elif codec == 'vp9':
        enc = ['-c:v', 'libvpx-vp9', '-crf', str(crf), '-b:v', '0', '-deadline', 'good',
               '-cpu-used', str(preset), '-threads', '1', '-row-mt', '0']
    else:
        enc = ['-c:v', 'libsvtav1', '-crf', str(crf), '-preset', str(preset), '-svtav1-params', 'lp=1']
    # setparams only tags the frames (no conversion); without it ffmpeg >= 7 inserts a
    # colourspace conversion from the untagged raw input to the bt709-tagged output.
    tags = ['-vf', 'setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709',
            '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
            '-color_trc', 'bt709', '-color_range', 'tv']
    subprocess.run(inp + enc + tags + [out], input=raw, check=True, capture_output=True)
    nbytes = os.path.getsize(out)
    dec = subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-i', out, '-fps_mode', 'passthrough',
                          '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], check=True, capture_output=True).stdout
    fs = w * h * 3 // 2
    n = len(dec) // fs
    if n != len(frames):
        raise RuntimeError(f'{codec}: decoded {n} frames, expected {len(frames)}')
    last = np.frombuffer(dec[(n - 1) * fs:n * fs], np.uint8)
    c = (w // 2) * (h // 2)
    Y = last[:w * h].reshape(h, w)
    U = last[w * h:w * h + c].reshape(h // 2, w // 2)
    V = last[w * h + c:w * h + 2 * c].reshape(h // 2, w // 2)
    return Y, U, V, nbytes * 8 / (w * h * len(frames))


def pick(rng, table):
    names, p = zip(*table)
    return names[rng.choice(len(names), p=np.array(p) / sum(p))]


def sample_variant(rng, force_codec=None, no_clean=False):
    v = dict(kernel=pick(rng, KERNELS))
    v['sigma_lr'] = float(rng.uniform(*BLUR_SIGMA)) if rng.random() < BLUR_P else 0.0
    codec = force_codec or pick(rng, CODECS)
    while no_clean and codec == 'clean':
        codec = pick(rng, CODECS)
    v['codec'] = codec
    if codec == 'clean':
        v['mode'] = 'intra'
        return v
    v['mode'] = 'intra' if rng.random() < INTRA_P else 'clip'
    if v['mode'] == 'clip':
        v['frames'] = int(rng.integers(3, 7))
        v['vel'] = [0, 0] if rng.random() < 0.2 else [int(x) for x in rng.integers(-6, 7, 2)]
    if codec == 'x264':
        v['crf'] = int(rng.integers(X264_CRF[0], X264_CRF[1] + 1)); v['preset'] = str(rng.choice(X264_PRESETS))
    elif codec == 'vp9':
        v['crf'] = int(rng.integers(VP9_CRF[0], VP9_CRF[1] + 1)); v['preset'] = int(rng.choice(VP9_CPU))
    else:
        v['crf'] = int(rng.integers(AV1_CRF[0], AV1_CRF[1] + 1)); v['preset'] = int(rng.choice(AV1_PRESETS))
    return v


def make_variant(rgb, region, v, tmpdir):
    """rgb: full HR float image; region: (oy, ox, ch, cw) = HR region of the (last) frame."""
    oy, ox, ch, cw = region
    if v['mode'] == 'clip':
        n, (vx, vy) = v['frames'], v['vel']
        offs = [(oy - (n - 1 - i) * vy, ox - (n - 1 - i) * vx) for i in range(n)]
    else:
        offs = [(oy, ox)]
    frames = [rgb_to_yuv420(downscale(rgb[y:y + ch, x:x + cw], v['kernel'], v['sigma_lr'])) for y, x in offs]
    clean = yuv420_to_luma(*frames[-1])
    if v['codec'] == 'clean':
        return clean, clean, 0.0
    Y, U, V, bpp = encode_decode(frames, v['codec'], v['crf'], v['preset'], tmpdir)
    return yuv420_to_luma(Y, U, V), clean, bpp


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a.astype(np.float64) - b) ** 2), 1e-12)))


_zip = None
def work(args):
    global _zip
    zpath, member, idx, outdir, split, seed, nvar, valid_mode = args
    stem = os.path.splitext(os.path.basename(member))[0]
    meta_path = os.path.join(outdir, 'meta', stem + '.json')
    if os.path.exists(meta_path):
        return json.load(open(meta_path))
    if _zip is None or _zip.filename != zpath:
        _zip = zipfile.ZipFile(zpath)
    buf = np.frombuffer(_zip.read(member), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    H, W = rgb.shape[:2]
    ch, cw = (H - 2 * MARGIN) // 4 * 4, (W - 2 * MARGIN) // 4 * 4
    region = (MARGIN, MARGIN, ch, cw)
    rng = np.random.default_rng([seed, 0 if split == 'train' else 1, idx])
    tmpdir = os.path.join(outdir, '..', 'tmp', str(os.getpid()))
    os.makedirs(tmpdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, 'hr', stem + '.png'), to_u8(luma(rgb[MARGIN:MARGIN + ch, MARGIN:MARGIN + cw])),
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if valid_mode:   # v0 = clean bicubic (no blur), v1..: degraded (never clean)
        variants = [dict(kernel='bicubic', sigma_lr=0.0, codec='clean', mode='intra')]
        variants += [sample_variant(rng, no_clean=True) for _ in range(nvar - 1)]
    else:
        variants = [sample_variant(rng) for _ in range(nvar)]
    for k, v in enumerate(variants):
        for attempt in range(3):
            try:
                lr, clean, bpp = make_variant(rgb, region, v, tmpdir)
                break
            except Exception as e:  # rare codec failure: resample the codec params
                print(f'{stem} v{k} {v} failed: {e}', file=sys.stderr, flush=True)
                v = sample_variant(rng, no_clean=valid_mode and k > 0)
        else:
            raise RuntimeError(f'{stem} v{k}: 3 failures')
        lr8 = to_u8(lr)
        v['bpp'] = round(bpp, 4)
        v['psnr_vs_clean_lr'] = round(psnr(lr8 / 255.0, clean), 2)
        cv2.imwrite(os.path.join(outdir, 'lr', f'{stem}_{k}.png'), lr8, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    meta = dict(stem=stem, hr_size=[ch, cw], lr_size=[ch // 2, cw // 2], variants=variants)
    json.dump(meta, open(meta_path + '.tmp', 'w'))
    os.replace(meta_path + '.tmp', meta_path)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', choices=['train', 'valid'], required=True)
    ap.add_argument('--variants', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    for d in ('hr', 'lr', 'meta'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    members = sorted(m for m in zipfile.ZipFile(a.zip).namelist() if m.endswith('.png'))
    if a.limit:
        members = members[:a.limit]
    jobs = [(a.zip, m, i, a.out, a.split, a.seed, a.variants, a.split == 'valid') for i, m in enumerate(members)]
    t0 = time.time()
    metas = []
    with Pool(a.workers, maxtasksperchild=50) as pool:
        for i, m in enumerate(pool.imap_unordered(work, jobs)):
            metas.append(m)
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f'{i + 1}/{len(jobs)} images, {time.time() - t0:.0f}s', flush=True)
    metas.sort(key=lambda m: m['stem'])
    with open(os.path.join(a.out, 'index.json'), 'w') as f:
        json.dump(dict(params=PARAMS_DOC, seed=a.seed, split=a.split, images=metas), f)
    shutil.rmtree(os.path.join(a.out, '..', 'tmp'), ignore_errors=True)
    ps = {}
    for m in metas:
        for v in m['variants']:
            ps.setdefault(v['codec'], []).append(v['psnr_vs_clean_lr'])
    for c, l in sorted(ps.items()):
        l = np.array(l)
        print(f'{c:6s} n={len(l):5d} PSNR(LR vs clean LR) p10/50/90 = '
              f'{np.percentile(l, 10):.1f}/{np.percentile(l, 50):.1f}/{np.percentile(l, 90):.1f} dB')


if __name__ == '__main__':
    main()
