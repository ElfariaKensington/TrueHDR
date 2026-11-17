#!/usr/bin/env python3
"""Train one or more SR_SPEC models on the same batch stream (PyTorch, MPS).

  python train.py --name full --archs f12x5,q16x6 --steps 150000

Writes runs/<name>/<arch>/{latest.pt,best.pt,val.csv} and runs/<name>/train.log.
Re-running the same command resumes from latest.pt.
Loss: Charbonnier(eps=1e-3) + lambda_grad * L1(finite-difference gradients)
      + lambda_fft * L1(rfft2 ortho, complex difference).
"""
import argparse, csv, json, math, os, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srmodel import ARCHS, SRNet, bicubic2x  # noqa: E402
from data_loader import train_loader, load_eval_set  # noqa: E402
from metrics import psnr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def grad_loss(a, b):
    dxa, dxb = a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1]
    dya, dyb = a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :]
    return (dxa - dxb).abs().mean() + (dya - dyb).abs().mean()


def fft_loss(a, b):
    d = torch.fft.rfft2((a - b).float(), norm='ortho')
    return (d.real.abs() + d.imag.abs()).mean()


def lr_at(step, total, base, warm=1000, final_frac=0.01):
    if step < warm:
        return base * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return base * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


@torch.no_grad()
def validate(models, val, dev):
    """Mean PSNR over the val list (full images), per model + bicubic."""
    res = {n: [] for n in list(models) + ['bicubic']}
    for s in val:
        lr = torch.from_numpy(s['lr']).float().div(255)[None, None].to(dev)
        hr = torch.from_numpy(s['hr']).float().div(255)[None, None].to(dev)
        res['bicubic'].append(psnr(bicubic2x(lr).clamp(0, 1), hr).item())
        for n, m in models.items():
            m.eval()
            res[n].append(psnr(m(lr).clamp(0, 1), hr).item())
            m.train()
    return {n: sum(v) / len(v) for n, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--archs', required=True)
    ap.add_argument('--steps', type=int, default=150000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lambda-grad', type=float, default=0.1)
    ap.add_argument('--lambda-fft', type=float, default=0.02)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--pool', type=int, default=8, help='decoded image pairs per loader worker')
    ap.add_argument('--refresh', type=int, default=1, help='pool entries replaced per batch')
    ap.add_argument('--val-every', type=int, default=2500)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'gen'))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    run_dir = os.path.join(HERE, 'runs', a.name)
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, 'train.log'), 'a')

    def log(msg):
        line = time.strftime('%H:%M:%S ') + msg
        print(line, flush=True)
        logf.write(line + '\n'); logf.flush()

    names = a.archs.split(',')
    models, opts, start = {}, {}, 0
    for n in names:
        d = os.path.join(run_dir, n)
        os.makedirs(d, exist_ok=True)
        m = SRNet(ARCHS[n]).to(dev)
        o = torch.optim.Adam(m.parameters(), lr=a.lr, betas=(0.9, 0.99))
        ck = os.path.join(d, 'latest.pt')
        if os.path.exists(ck):
            st = torch.load(ck, map_location='cpu')
            m.load_state_dict(st['model']); o.load_state_dict(st['opt'])
            start = st['step']
            log(f'resumed {n} at step {start}')
        models[n], opts[n] = m, o
        log(f'{n}: macs/px={m.macs} params={sum(p.numel() for p in m.parameters())} layers={json.dumps(m.spec)}')
    json.dump(vars(a), open(os.path.join(run_dir, 'args.json'), 'w'), indent=1)

    val = load_eval_set(os.path.join(a.data, 'valid'), variants=[0, 1], limit=20)
    val_deg = [s for s in val if s['k'] == 1]
    val_clean = [s for s in val if s['k'] == 0]
    best = {n: -1.0 for n in names}
    for n in names:
        p = os.path.join(run_dir, n, 'best.json')
        if os.path.exists(p):
            best[n] = json.load(open(p))['val_deg']

    loader = iter(train_loader(os.path.join(a.data, 'train'), a.batch, a.workers, seed=a.seed * 7919 + start,
                               pool=a.pool, refresh=a.refresh))
    ema = {n: None for n in names}
    acc = {n: 0.0 for n in names}
    t0, tl = time.time(), time.time()
    for step in range(start, a.steps):
        lr_u8, hr_u8 = next(loader)
        lr_t = lr_u8.to(dev, non_blocking=True).float().div_(255)
        hr_t = hr_u8.to(dev, non_blocking=True).float().div_(255)
        cur_lr = lr_at(step, a.steps, a.lr, warm=a.warmup)
        for n, m in models.items():
            for g in opts[n].param_groups:
                g['lr'] = cur_lr
            out = m(lr_t)
            loss = charbonnier(out, hr_t)
            if a.lambda_grad:
                loss = loss + a.lambda_grad * grad_loss(out, hr_t)
            if a.lambda_fft:
                loss = loss + a.lambda_fft * fft_loss(out, hr_t)
            opts[n].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opts[n].step()
            acc[n] = acc[n] + loss.detach()
        if (step + 1) % a.log_every == 0:
            for n in names:
                ema[n] = acc[n].item() / a.log_every
                acc[n] = 0.0
            dt = time.time() - tl; tl = time.time()
            rate = a.log_every / dt
            eta = (a.steps - step - 1) / rate / 3600
            log(f'step {step + 1}/{a.steps} lr {cur_lr:.2e} ' +
                ' '.join(f'{n}:{ema[n]:.5f}' for n in names) + f' | {rate:.1f} it/s ETA {eta:.2f} h')
        if (step + 1) % a.val_every == 0 or step + 1 == a.steps:
            vd, vc = validate(models, val_deg, dev), validate(models, val_clean, dev)
            log(f'VAL step {step + 1} degraded: ' + ' '.join(f'{n}={v:.3f}' for n, v in vd.items()) +
                ' | clean: ' + ' '.join(f'{n}={v:.3f}' for n, v in vc.items()))
            for n, m in models.items():
                d = os.path.join(run_dir, n)
                st = dict(model=m.state_dict(), opt=opts[n].state_dict(), step=step + 1, spec=m.spec, base=m.base)
                torch.save(st, os.path.join(d, 'latest.pt.tmp'))
                os.replace(os.path.join(d, 'latest.pt.tmp'), os.path.join(d, 'latest.pt'))
                newf = not os.path.exists(os.path.join(d, 'val.csv'))
                with open(os.path.join(d, 'val.csv'), 'a') as f:
                    w = csv.writer(f)
                    if newf:
                        w.writerow(['step', 'val_deg', 'val_clean', 'bicubic_deg', 'bicubic_clean'])
                    w.writerow([step + 1, f'{vd[n]:.4f}', f'{vc[n]:.4f}', f'{vd["bicubic"]:.4f}', f'{vc["bicubic"]:.4f}'])
                if vd[n] > best[n]:
                    best[n] = vd[n]
                    torch.save(dict(model=m.state_dict(), step=step + 1, spec=m.spec, base=m.base),
                               os.path.join(d, 'best.pt'))
                    json.dump(dict(step=step + 1, val_deg=vd[n], val_clean=vc[n]), open(os.path.join(d, 'best.json'), 'w'))
    log(f'done in {(time.time() - t0) / 3600:.2f} h')
    os._exit(0)  # don't wait on persistent loader workers


if __name__ == '__main__':
    main()
