"""Y' metrics on [0,1] tensors (N,1,H,W). Border of SHAVE HR px excluded."""
import torch
import torch.nn.functional as F

SHAVE = 4


def _shave(x):
    return x[..., SHAVE:-SHAVE, SHAVE:-SHAVE]


def psnr(a, b):
    a, b = _shave(a).float(), _shave(b).float()
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10 * torch.log10(1.0 / mse.cpu().double())


def _gauss(win=11, sigma=1.5, device='cpu'):
    g = torch.arange(win, dtype=torch.float32, device=device) - win // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a, b):
    """Standard SSIM (Wang et al.): 11x11 Gaussian sigma=1.5, K1=0.01, K2=0.03, L=1, valid window."""
    a, b = _shave(a).float(), _shave(b).float()
    w = _gauss(device=a.device)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    saa = F.conv2d(a * a, w) - mu_a ** 2
    sbb = F.conv2d(b * b, w) - mu_b ** 2
    sab = F.conv2d(a * b, w) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return m.mean(dim=(1, 2, 3))
