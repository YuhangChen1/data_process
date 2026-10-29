"""Full-reference fidelity of the model's renders at the 10 input steps against the input walkthrough frames, plus
no-reference NIQE / BRISQUE of those renders.

ctx.input_index: the 10 trajectory steps that carry ground truth; ctx.input_gt: [10,H,W,3] uint8 ground truth.
Renders whose size differs from the ground truth are resized to it (INTER_AREA) before scoring.

  fidelity.psnr / ssim         skimage, data_range 255, SSIM win 7 (same as R2M appearance)
  fidelity.ms_ssim             piq.multi_scale_ssim (RGB in [0,1], data_range 1)
  fidelity.lpips_alex / _vgg   lpips 0.1.4 on [-1,1] tensors at native resolution
  fidelity.dists               pyiqa 'dists'
  fidelity.dreamsim            DreamSim ensemble distance
  fidelity.dinov2              DINOv2 ViT-B/14 global cosine (R2M scene-identity preprocessing)
  fidelity.niqe / brisque      pyiqa no-reference scores of the renders (lower is better); GT scores in raw
"""
from __future__ import annotations

import numpy as np

from .base import Result, NotApplicable, operator, device
from . import r2m as R

SRC = 'evaluate1 (ours): full-reference image fidelity at input-frame poses'


def _pairs(ctx):
    import cv2
    idx = list(ctx.input_index or []); gt = ctx.input_gt
    if not idx or gt is None or len(gt) != len(idx): raise NotApplicable('input_index / input_gt missing or mismatched')
    out = []
    for k, step in enumerate(idx):
        if not 0 <= step < len(ctx.frames): continue
        p = R._to_uint8(ctx.frames[step]); g = R._to_uint8(gt[k])
        if p.shape != g.shape: p = cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
        out.append((step, p, g))
    if not out: raise NotApplicable('no input step inside the rendered frames')
    return out


def _fid(ctx, name, fn, **raw):
    rows = [{'step': s, 'value': float(fn(p, g))} for s, p, g in _pairs(ctx)]
    vals = [r['value'] for r in rows if np.isfinite(r['value'])]
    return Result(name, float(np.mean(vals)) if vals else None, per_item=rows, raw=raw)


def _t01(x):
    import torch
    return torch.from_numpy(R._to_uint8(x)).float().permute(2, 0, 1).unsqueeze(0).div(255.).to(device())


PYIQA_FILES = {'niqe': ['niqe_modelparameters.mat'], 'brisque': ['brisque_svm_weights.pth'],
               'dists': ['DISTS_weights-f5e65c96.pth']}


def _pyiqa(name):
    """pyiqa metric whose parameter files come from WEIGHTS/pyiqa (and torchvision backbones from WEIGHTS/torch_hub);
    pyiqa's module-level cache dir is swapped only while the metric is built."""
    def build():
        import pyiqa
        from pyiqa.utils import download_util as du
        if not R.ALLOW_DOWNLOAD:
            R._need(*[R.PYIQA_DIR / f for f in PYIQA_FILES.get(name, [])])
            if name == 'dists': R._need(R.TORCH_HUB / 'checkpoints' / 'vgg16-397923af.pth')
        prev = du.DEFAULT_CACHE_DIR; du.DEFAULT_CACHE_DIR = str(R.PYIQA_DIR); R.PYIQA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with R._hub():
                return pyiqa.create_metric(name, device=device())
        finally:
            du.DEFAULT_CACHE_DIR = prev
    return R._model('pyiqa_' + name, build)


@operator('fidelity.psnr', 'fidelity', inputs=('frames', 'input_index', 'input_gt'), unit='dB', source=SRC)
def fid_psnr(ctx):
    return _fid(ctx, 'fidelity.psnr', lambda p, g: R.compute_psnr_ssim(p, g)['psnr'], psnr_cap_db=100)


@operator('fidelity.ssim', 'fidelity', inputs=('frames', 'input_index', 'input_gt'), source=SRC)
def fid_ssim(ctx):
    return _fid(ctx, 'fidelity.ssim', lambda p, g: R.compute_psnr_ssim(p, g)['ssim'])


@operator('fidelity.ms_ssim', 'fidelity', inputs=('frames', 'input_index', 'input_gt'), source=SRC, requires=('piq',), gpu=True)
def fid_ms_ssim(ctx):
    import piq, torch
    def f(p, g):
        with torch.no_grad(): return piq.multi_scale_ssim(_t01(p), _t01(g), data_range=1.).item()
    return _fid(ctx, 'fidelity.ms_ssim', f)


@operator('fidelity.lpips_alex', 'fidelity', inputs=('frames', 'input_index', 'input_gt'), higher_is_better=False, source=SRC,
          requires=('lpips',), gpu=True)
def fid_lpips_alex(ctx):
    return _fid(ctx, 'fidelity.lpips_alex', R.get_lpips('alex'))


@operator('fidelity.lpips_vgg', 'fidelity', inputs=('frames', 'input_index', 'input_gt'), higher_is_better=False, source=SRC,
          requires=('lpips',), gpu=True)
def fid_lpips_vgg(ctx):
    return _fid(ctx, 'fidelity.lpips_vgg', R.get_lpips('vgg'))


@operator('fidelity.dists', 'fidelity', inputs=('frames', 'input_index', 'input_gt'), higher_is_better=False, source=SRC,
          requires=('pyiqa',), gpu=True)
def fid_dists(ctx):
    import torch
    m = _pyiqa('dists')
    def f(p, g):
        with torch.no_grad(): return m(_t01(p), _t01(g)).item()
    return _fid(ctx, 'fidelity.dists', f)


@operator('fidelity.dreamsim', 'fidelity', inputs=('frames', 'input_index', 'input_gt'), higher_is_better=False, source=SRC,
          requires=('dreamsim',), gpu=True)
def fid_dreamsim(ctx):
    return _fid(ctx, 'fidelity.dreamsim', R.get_dreamsim())


@operator('fidelity.dinov2', 'fidelity', inputs=('frames', 'input_index', 'input_gt'), source=SRC, requires=('dinov2',), gpu=True)
def fid_dinov2(ctx):
    return _fid(ctx, 'fidelity.dinov2', lambda p, g: R.get_dino().similarity(p, g))


def _noref(ctx, name, metric):
    import torch
    m = _pyiqa(metric); rows = []
    with torch.no_grad():
        for s, p, g in _pairs(ctx):
            rows.append({'step': s, 'value': float(m(_t01(p)).item()), 'gt': float(m(_t01(g)).item())})
    return Result(name, float(np.mean([r['value'] for r in rows])), per_item=rows,
                  raw={'gt_mean': float(np.mean([r['gt'] for r in rows])), 'metric': 'pyiqa ' + metric})


@operator('fidelity.niqe', 'fidelity', inputs=('frames', 'input_index', 'input_gt'), higher_is_better=False, source=SRC,
          requires=('pyiqa',), gpu=True)
def fid_niqe(ctx):
    return _noref(ctx, 'fidelity.niqe', 'niqe')


@operator('fidelity.brisque', 'fidelity', inputs=('frames', 'input_index', 'input_gt'), higher_is_better=False, source=SRC,
          requires=('pyiqa',), gpu=True)
def fid_brisque(ctx):
    return _noref(ctx, 'fidelity.brisque', 'brisque')
