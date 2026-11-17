#!/usr/bin/env python3
"""Synthetic TrueHDR test content (needs numpy + Pillow + /opt/homebrew/bin/ffmpeg).

  python3 dev/harness/gen_media.py            # frames for the Deno tests + 1080p/720p clips
  python3 dev/harness/gen_media.py --no-clips # frames only

Writes into dev/media/ (gitignored):
  frames/<name>_<W>x<H>.rgba   raw RGBA8 (sRGB) frames, sequences concatenated
  frames/manifest.json         {name: {w, h, frames, file, meta}}
  sky_1080p.mp4, sky_720p.mp4  12 s H.264 clips: sky + birds/specks + captions, night lights,
                               white web page, scene cuts (for dev/harness/index.html)
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MEDIA = os.path.join(ROOT, 'dev', 'media')
FRAMES = os.path.join(MEDIA, 'frames')
FFMPEG = '/opt/homebrew/bin/ffmpeg'
FONT = '/System/Library/Fonts/Helvetica.ttc'
SS = 4  # supersampling for anti-aliased small objects


def font(px):
    try:
        return ImageFont.truetype(FONT, px)
    except OSError:
        return ImageFont.load_default()


def gradient(w, h, top, bottom, wobble=0.0):
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    img = (1 - t) * np.array(top, np.float32) + t * np.array(bottom, np.float32)
    img = np.repeat(img, w, axis=1)
    if wobble:
        x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        img = img + wobble * np.sin(x * 6.28 * 1.5)
    return img  # float 0..255


def sky(w, h):
    return gradient(w, h, (96, 150, 222), (206, 222, 240), wobble=3.0)


def night(w, h):
    return gradient(w, h, (8, 10, 24), (22, 24, 42))


def blend_shape(img, cx, cy, rx, ry, color, alpha=1.0):
    """Anti-aliased filled ellipse (supersampled coverage) blended into img (float, sRGB 0..255)."""
    h, w, _ = img.shape
    x0 = int(np.floor(cx - rx - 1)); x1 = int(np.ceil(cx + rx + 1))
    y0 = int(np.floor(cy - ry - 1)); y1 = int(np.ceil(cy + ry + 1))
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, w), min(y1, h)
    if x1c <= x0c or y1c <= y0c:
        return
    xs = (np.arange(x0c * SS, x1c * SS) + 0.5) / SS
    ys = (np.arange(y0c * SS, y1c * SS) + 0.5) / SS
    X, Y = np.meshgrid(xs, ys)
    inside = (((X - cx) / max(rx, 1e-3)) ** 2 + ((Y - cy) / max(ry, 1e-3)) ** 2) <= 1.0
    cov = inside.reshape(y1c - y0c, SS, x1c - x0c, SS).mean(axis=(1, 3))[:, :, None] * alpha
    patch = img[y0c:y1c, x0c:x1c]
    img[y0c:y1c, x0c:x1c] = patch * (1 - cov) + np.array(color, np.float32) * cov


def bird(img, cx, cy, size):
    blend_shape(img, cx, cy, size / 2.0, size / 3.2, (28, 30, 36))


def speck(img, cx, cy, size, value=255):
    blend_shape(img, cx, cy, size / 2.0, size / 2.0, (value, value, value))


def caption(img, text, y_frac=0.86, px=None, color=(255, 255, 255)):
    h, w, _ = img.shape
    px = px or max(16, int(h * 0.04))
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, 'RGBA')
    f = font(px)
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = int(h * y_frac) - th // 2
    pad = px // 3
    d.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad * 2], fill=(0, 0, 0, 190))
    d.text((x, y - bbox[1]), text, font=f, fill=color + (255,))
    img[:] = np.asarray(pil, dtype=np.float32)
    return (x - pad, y - pad, x + tw + pad, y + th + pad * 2)


def lower_third(img, text, x, y, px, color):
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, 'RGBA')
    f = font(px)
    bbox = d.textbbox((0, 0), text, font=f)
    d.text((x, y - bbox[1]), text, font=f, fill=color + (255,))
    img[:] = np.asarray(pil, dtype=np.float32)
    return (x, y, x + bbox[2] - bbox[0], y + bbox[3] - bbox[1])


def white_page(w, h):
    img = np.full((h, w, 3), 246.0, np.float32)
    pil = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, w, int(h * 0.08)], fill=(32, 96, 200))
    f_title = font(int(h * 0.05))
    f_body = font(max(12, int(h * 0.022)))
    d.text((int(w * 0.08), int(h * 0.12)), 'Release notes - TrueHDR test page', font=f_title, fill=(20, 20, 24))
    lorem = ('The quick brown fox jumps over the lazy dog while the white page stays comfortable. '
             'Body copy, links and code samples must not bloom when EDR is active on bright slides.')
    y = int(h * 0.24)
    for i in range(14):
        d.text((int(w * 0.08), y), lorem[(i * 7) % 40:][:110], font=f_body, fill=(30, 30, 34))
        y += int(h * 0.04)
    return np.asarray(pil, dtype=np.float32)


def to_u8(img):
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def rgba_bytes(img):
    u8 = to_u8(img)
    a = np.full(u8.shape[:2] + (1,), 255, np.uint8)
    return np.concatenate([u8, a], axis=2).tobytes()


# ----------------------------------------------------------------------------- test frames
BIRDS = [(3, 0.13, 0.42), (4, 0.27, 0.55), (5, 0.41, 0.38), (6, 0.55, 0.62), (7, 0.69, 0.45), (8, 0.83, 0.58),
         (3, 0.20, 0.72), (4, 0.34, 0.80), (5, 0.48, 0.70), (6, 0.62, 0.84), (7, 0.76, 0.74), (8, 0.90, 0.30)]
SPECK_SIZES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]


def halo_frames(w, h):
    base = sky(w, h)
    withb = base.copy()
    birds = []
    for i, (s, fx, fy) in enumerate(BIRDS):
        cx, cy = fx * w + 0.37 * (i % 3), fy * h + 0.21 * (i % 2)
        bird(withb, cx, cy, s)
        birds.append({'x': cx, 'y': cy, 'size': s})
    return base, withb, birds


def speck_frame(w, h, bg='night', value=255):
    img = night(w, h) if bg == 'night' else sky(w, h)
    specks = []
    n = len(SPECK_SIZES)
    for i, s in enumerate(SPECK_SIZES):
        cx = (i + 0.5) / n * w + 0.0
        cy = 0.5 * h
        speck(img, cx, cy, s, value)
        specks.append({'x': cx, 'y': cy, 'size': s, 'value': value})
    return img, specks


def move_seq(w, h, n=48):
    """Night half: 3 px and 2 px white specks drifting sub-pixel. Sky half: 4 px bird drifting."""
    frames, meta = [], []
    for f in range(n):
        img = night(w, h)
        img[: h // 2] = sky(w, h)[: h // 2]
        s1 = (0.30 * w + 0.37 * f, 0.75 * h + 0.11 * f)
        s2 = (0.60 * w + 0.23 * f, 0.70 * h + 0.31 * f)
        b1 = (0.40 * w + 0.41 * f, 0.25 * h + 0.07 * f)
        speck(img, *s1, 3)
        speck(img, *s2, 2)
        bird(img, *b1, 4)
        frames.append(img)
        meta.append({'speck3': s1, 'speck2': s2, 'bird4': b1})
    return frames, meta


def pump_seq(w, h, n=90, seed=7):
    """Flock of 40 birds + 12 white specks crossing a bright sky; reference box top-left untouched."""
    rng = np.random.default_rng(seed)
    base = sky(w, h)
    objs = []
    for i in range(52):
        objs.append({
            'kind': 'bird' if i < 40 else 'speck',
            'size': float(rng.integers(3, 9)),
            'x': float(rng.uniform(0.25 * w, 0.95 * w)),
            'y': float(rng.uniform(0.35 * h, 0.95 * h)),
            'vx': float(rng.uniform(-6, 6)),
            'vy': float(rng.uniform(-2, 2)),
        })
    frames = []
    for f in range(n):
        img = base.copy()
        for o in objs:
            x = 0.25 * w + ((o['x'] + o['vx'] * f) - 0.25 * w) % (0.70 * w)
            y = 0.35 * h + ((o['y'] + o['vy'] * f) - 0.35 * h) % (0.60 * h)
            (bird if o['kind'] == 'bird' else speck)(img, x, y, o['size'])
        frames.append(img)
    ref = {'x0': int(0.04 * w), 'y0': int(0.05 * h), 'x1': int(0.18 * w), 'y1': int(0.25 * h)}
    return frames, ref


def text_frame(w, h):
    img = gradient(w, h, (40, 46, 62), (52, 56, 70))
    cap = caption(img, 'Captions and UI text must stay comfortable on an EDR display', 0.86)
    lt = lower_third(img, 'LIVE  ·  Lower-third 14:32', int(0.06 * w), int(0.10 * h), max(14, int(h * 0.035)), (205, 205, 210))
    specks = []
    for i, s in enumerate([4, 8, 6]):
        cx, cy = (0.35 + 0.15 * i) * w, 0.45 * h
        speck(img, cx, cy, s, 255)
        specks.append({'x': cx, 'y': cy, 'size': s})
    return img, {'caption': cap, 'lowerThird': lt, 'specks': specks}


def cut_seq(w, h, n_night=20, n_white=20):
    night_img = night(w, h)
    for i, s in enumerate([4, 8, 16]):
        speck(night_img, (0.3 + 0.2 * i) * w, 0.4 * h, s)
    page = white_page(w, h)
    return [night_img] * n_night + [page] * n_white


def write_frames(manifest, name, frames, w, h, meta=None):
    os.makedirs(FRAMES, exist_ok=True)
    fn = f'{name}_{w}x{h}.rgba'
    with open(os.path.join(FRAMES, fn), 'wb') as fh:
        for fr in frames:
            fh.write(rgba_bytes(fr))
    manifest[f'{name}_{w}x{h}'] = {'w': w, 'h': h, 'frames': len(frames), 'file': fn, 'meta': meta}


def build_frames():
    manifest = {}
    for (w, h) in [(1920, 1080), (1280, 720)]:
        base, withb, birds = halo_frames(w, h)
        write_frames(manifest, 'halo_without', [base], w, h)
        write_frames(manifest, 'halo_with', [withb], w, h, {'birds': birds})
        for bg in ['night', 'sky']:
            img, sp = speck_frame(w, h, bg)
            write_frames(manifest, f'specks_{bg}', [img], w, h, {'specks': sp})
        img, meta = text_frame(w, h)
        write_frames(manifest, 'text', [img], w, h, meta)
        write_frames(manifest, 'white_page', [white_page(w, h)], w, h)
    frames, meta = move_seq(1920, 1080)
    write_frames(manifest, 'move', frames, 1920, 1080, meta)
    frames, ref = pump_seq(1920, 1080)
    write_frames(manifest, 'pump', frames, 1920, 1080, {'ref': ref})
    write_frames(manifest, 'cut', cut_seq(1920, 1080), 1920, 1080, {'cutAt': 20})
    with open(os.path.join(FRAMES, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1)
    print('frames ->', FRAMES, len(manifest), 'sets')


# ----------------------------------------------------------------------------- clips
def clip_frame(w, h, f, fps, rng_state):
    t = f / fps
    if t < 4.0 or t >= 9.5:  # sky segment (+ after the last cut)
        img = sky(w, h)
        tt = t if t < 4.0 else t - 9.5
        for i, (s, fx, fy) in enumerate(BIRDS):
            bird(img, (fx * w + (18 + 4 * i) * tt) % w, fy * h * 0.8 + 6 * np.sin(tt * 2 + i), s)
        for i in range(8):
            speck(img, (0.1 * w + i * 0.11 * w - 22 * tt) % w, 0.25 * h + 0.05 * h * i, 3 + (i % 4), 255)
        caption(img, 'Small birds at a distance should stay clean: no halos, no pumping', 0.88)
        return img
    if t < 7.0:  # night city
        img = night(w, h)
        speck(img, 0.8 * w, 0.2 * h, int(0.08 * h), 235)  # moon (large)
        rs = np.random.default_rng(3)
        for i in range(60):
            x, y = rs.uniform(0, w), rs.uniform(0.45 * h, 0.95 * h)
            s = rs.integers(1, 7)
            v = rs.integers(150, 256)
            speck(img, x + 3 * np.sin(t * 3 + i), y, s, v)
        lower_third(img, 'NIGHT CITY  ·  lights of every size', int(0.05 * w), int(0.07 * h), max(14, int(h * 0.04)), (255, 255, 255))
        return img
    return white_page(w, h)


def build_clip(w, h, out, seconds=12, fps=30):
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p',
           '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
           '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-movflags', '+faststart', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(seconds * fps):
        p.stdin.write(to_u8(clip_frame(w, h, f, fps, None)).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit(f'ffmpeg failed for {out}')
    print('clip ->', out)


if __name__ == '__main__':
    os.makedirs(MEDIA, exist_ok=True)
    build_frames()
    if '--no-clips' not in sys.argv:
        build_clip(1920, 1080, os.path.join(MEDIA, 'sky_1080p.mp4'))
        build_clip(1280, 720, os.path.join(MEDIA, 'sky_720p.mp4'))
