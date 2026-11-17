"""Training / validation data from the pre-generated degradation set (degrade.py output).

Train batches are produced by DataLoader workers from a small per-worker pool of decoded
(HR, LR-variant) PNG pairs: aligned random crops + dihedral augmentation per crop. RAM stays small
(no in-memory dataset) and PNG decode cost is amortised.
"""
import json, os, random

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


def load_index(root):
    return json.load(open(os.path.join(root, 'index.json')))


def read_gray(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(path)
    return a


class CropStream(IterableDataset):
    """Each worker keeps a pool of decoded (HR, LR-variant) pairs, cuts every batch from random pool
    entries and replaces `refresh` entries per batch (PNG decode is the CPU bottleneck: ~25-100 ms
    per 2.6 MP HR frame). Each decoded pair yields ~batch*pool/(pool*refresh) crops on average."""

    def __init__(self, root, batch=16, pool=8, refresh=2, patch_sizes=(96, 104, 112, 120, 128), seed=0):
        self.root = root
        idx = load_index(root)
        self.pairs = [(m['stem'], k) for m in idx['images'] for k in range(len(m['variants']))]
        self.batch, self.pool, self.refresh, self.sizes, self.seed = batch, pool, refresh, patch_sizes, seed

    def _load(self, rng):
        stem, k = rng.choice(self.pairs)
        return (read_gray(os.path.join(self.root, 'hr', stem + '.png')),
                read_gray(os.path.join(self.root, 'lr', f'{stem}_{k}.png')))

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid)
        pool = [self._load(rng) for _ in range(self.pool)]
        while True:
            P = rng.choice(self.sizes)
            lrs, hrs = [], []
            for _ in range(self.batch):
                hr, lr = pool[rng.randrange(len(pool))]
                h, w = lr.shape
                y, x = rng.randrange(0, h - P + 1), rng.randrange(0, w - P + 1)
                a = lr[y:y + P, x:x + P]
                b = hr[2 * y:2 * y + 2 * P, 2 * x:2 * x + 2 * P]
                t = rng.randrange(8)
                if t & 1:
                    a, b = a[:, ::-1], b[:, ::-1]
                if t & 2:
                    a, b = a[::-1], b[::-1]
                if t & 4:
                    a, b = a.T, b.T
                lrs.append(np.ascontiguousarray(a))
                hrs.append(np.ascontiguousarray(b))
            yield torch.from_numpy(np.stack(lrs)[:, None]), torch.from_numpy(np.stack(hrs)[:, None])
            for _ in range(self.refresh):
                pool[rng.randrange(len(pool))] = self._load(rng)


def train_loader(root, batch=16, workers=4, seed=0, pool=8, refresh=2):
    return DataLoader(CropStream(root, batch, pool=pool, refresh=refresh, seed=seed), batch_size=None, num_workers=workers,
                      persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)


def load_eval_set(root, variants=None, limit=0):
    """Returns list of dict(stem, k, codec, lr uint8 HxW, hr uint8 2Hx2W)."""
    idx = load_index(root)
    out = []
    for m in idx['images'][:limit or None]:
        hr = read_gray(os.path.join(root, 'hr', m['stem'] + '.png'))
        for k, v in enumerate(m['variants']):
            if variants is not None and k not in variants:
                continue
            lr = read_gray(os.path.join(root, 'lr', f"{m['stem']}_{k}.png"))
            out.append(dict(stem=m['stem'], k=k, codec=v['codec'], meta=v, lr=lr, hr=hr))
    return out
