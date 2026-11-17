# TrueHDR live-action SR training (`tsr-live-fast`, `tsr-live-quality`)

Models are exactly the `docs/SR_SPEC.md` family: Y' (BT.709 luma of R'G'B') in, replicate-padded
convs + PReLU, last conv → 4 ch → PixelShuffle(2), added to bicubic2x (a=-0.75, align_corners=False).
Python: `/Users/elfaria/.venv/bin/python3` (torch MPS, numpy, opencv, pillow). All commands from `training/`.

## 1. Data (DIV2K, ETH host)
```
mkdir -p data/div2k && cd data/div2k
curl -L -C - -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip   # 3.5 GB, 800 imgs
curl -L -C - -O https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip   # 0.45 GB, 100 imgs
cd ../..
PY=/Users/elfaria/.venv/bin/python3
$PY degrade.py --zip data/div2k/DIV2K_valid_HR.zip --out data/gen/valid --split valid --variants 3 --workers 7
$PY degrade.py --zip data/div2k/DIV2K_train_HR.zip --out data/gen/train --split train --variants 5 --workers 7
```
PNGs are read straight from the zips (no extraction). Output: `data/gen/<split>/hr/<id>.png` (HR Y',
uint8), `lr/<id>_<k>.png` (LR Y' variants), `index.json` (all per-variant parameters + PSNR of the
degraded LR vs the clean LR). Train ≈ 2.5 GB, 17 min on 7 workers; valid ≈ 0.26 GB, 1 min.

### Degradation parameters (`degrade.py`, seed 1234; per image and variant)
| step | parameters |
|---|---|
| HR region | image cropped 32 px per side, size floored to ×4; target Y' = 0.2126R'+0.7152G'+0.0722B' |
| blur | p=0.5, σ_LR ~ U(0.1, 0.6) (applied at HR as 2σ, Gaussian) |
| 2× downscale | bicubic 0.4 (PIL, antialiased, a=-0.5) / area 0.3 (cv2) / lanczos3 0.3 (PIL) |
| to video | 8-bit BT.709 limited-range YUV 4:2:0 (chroma 2×2 mean), tagged bt709/tv via `setparams` |
| codec | clean (no codec) 0.15 / libx264 0.35 / libvpx-vp9 0.25 / libsvtav1 0.25 |
| x264 | CRF U{18..40}, preset ∈ {veryfast, faster, medium, slow} |
| VP9 | CRF U{24..56}, `-b:v 0 -deadline good -cpu-used {3,4,5}` |
| AV1 | SVT-AV1 CRF U{24..58}, preset ∈ {6, 8, 10} |
| GOP | intra p=0.35 (single I-frame); clip p=0.65: 3–6 frames panning v~U{-6..6}² HR px/frame (static p=0.2) ending on the HR region; last (inter-coded) decoded frame kept |
| decode | ffmpeg → yuv420p → R'G'B' (bilinear chroma, clip [0,1]) → Y' |
| variants | train: 5 per image (4000 pairs); valid: v0 clean bicubic (no blur), v1–v2 degraded (never clean) |

Resulting degraded-vs-clean LR PSNR (train, p10/50/90): x264 25/31/37 dB, VP9 34/41/46 dB,
AV1 30/36/43 dB — harsh 360p-like to mild 1080p-like (DIV2K is denser than typical video, so the
same CRF gives lower PSNR here than on real YouTube frames).

## 2. Train
```
nohup $PY train.py --name <run> --archs <arch[,arch...]> --steps N --batch B > runs/<run>.out 2>&1 &
```
Several archs in one process share the batch stream (used for the architecture sweep). Re-running
the same command resumes from `runs/<run>/<arch>/latest.pt`. Loss: Charbonnier (ε=1e-3)
+ 0.1·L1(finite-difference gradients) + 0.02·L1(rfft2 ortho, complex diff); no GAN. Adam
(β=0.9/0.99), lr 1e-3, 1000-step linear warmup, cosine to 1e-5, grad-clip 1.0; random 96–128 px LR
patches (size per batch), 8-way dihedral augmentation. Validation every 2500 steps (20 valid images,
full frames, degraded v1 and clean v0) → `val.csv`, `best.pt` (by degraded PSNR), `train.log`.

Architectures live in `srmodel.py:ARCHS` (MAC/px at input resolution = Σ k²·cin·cout).

## 3. Evaluate / export
```
$PY export.py --model tsr-live-fast=fast:runs/<run>/<fast-arch>/latest.pt \
              --model tsr-live-quality=quality:runs/<run>/<quality-arch>/latest.pt \
              --out export/tsr-live-models.js
$PY eval.py --js export/tsr-live-models.js --out runs/<run>/eval
```
`export.py` writes the SR_SPEC file (base64 LE f16 `[cout][cin][k][k]`, `license: 'MIT (custom-trained,
TrueHDR)'`) and fails unless the weights decoded back from the file reproduce the f32 checkpoint within
2/255 on real frames (also prints f16-activation error, informational). `eval.py` reports PSNR/SSIM
(Y', 4 px shave) for bicubic vs each model on clean v0 and degraded v1/v2 (+ per codec) of all 100
DIV2K-valid images and writes `crops_clean.png` / `crops_degraded.png` side-by-sides.
