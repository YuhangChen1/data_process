"""WBench video-quality metrics V1-V6 as evaluate1 operators (+ helpers shared by the other wbench_* modules).

Reference implementation: WBench (meituan-longcat/WBench, snapshot in /mnt/cfs/cyh/tmp/refs/WBench-main; pinned third-party
code copied from /mnt/cfs/cyh/WBench/third_party to /mnt/cfs/cyh/tmp/data_pipeline/third_party/wbench). Weights: the
meituan-longcat/WBench-weights layout under base.WEIGHTS/'wbench'.

Every operator returns WBench's own [0,1] score in raw['score01'] and value = 100 * score01 (the paper's 0-100 scale).
Frame sampling follows src/metrics/video_quality/evaluator.py FRAME_SAMPLING_CONFIG, with the case fps as video fps.
"""
from __future__ import annotations
from pathlib import Path
import contextlib, gc, io, json, os, sys, time
import numpy as np

from .base import operator, Result, NotApplicable, WEIGHTS, TMP, GPT_MODEL, device, cached

WB = WEIGHTS / 'wbench'                                  # == WBench WBENCH_WEIGHTS_DIR
WB_TP = TMP / 'third_party' / 'wbench'                   # == WBench third_party/ (pinned submodule commits)
REF = 'meituan-longcat/WBench @ /mnt/cfs/cyh/tmp/refs/WBench-main'
os.environ.setdefault('WBENCH_WEIGHTS_DIR', str(WB))

# src/metrics/video_quality/evaluator.py:211-218
FRAME_SAMPLING = {'aesthetic_quality': {'fps': 2.0}, 'imaging_quality': {'fps': 2.0}, 'temporal_flickering': {'fps': 10.0},
                  'dynamic_degree': {'fps': 8.0}, 'motion_smoothness': {'fps': 10.0}, 'hpsv3_quality': 20}


# ───────────────────────────── shared helpers ─────────────────────────────

def sample_indices(n, video_fps, cfg, start=0):
    """evaluator.py:264-286 / video_utils.py: {'fps': f} -> every round(video_fps/f)-th frame; int N -> linspace(N)."""
    if isinstance(cfg, dict):
        step = max(1, round(video_fps / cfg['fps']))
        return [start + i for i in range(0, n, step)]
    if cfg == -1 or cfg >= n: return [start + i for i in range(n)]
    return [start + int(i) for i in np.linspace(0, n - 1, cfg, dtype=int)]


def pil(frame):
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(frame))


def frames_at(ctx, idx):
    return [pil(ctx.frames[i]) for i in idx]


def wb_score(score01, raw=None, per_item=None, note=''):
    raw = dict(raw or {}); raw['score01'] = None if score01 is None else float(score01)
    return Result('', None if score01 is None else 100.0 * float(score01), raw=raw, per_item=per_item or [], note=note)


class _Failure:
    def __init__(self, exc): self.exc = exc


def cached_try(ctx, key, fn):
    """ctx.cache memo that also remembers failures, so an expensive estimator that crashed is not re-run by every
    operator that depends on it (C1, C2, I1 share poses; C4, C5, C6 share DA3)."""
    if key in ctx.cache:
        v = ctx.cache[key]
        if isinstance(v, _Failure): raise v.exc
        return v
    try: v = fn()
    except Exception as e:
        ctx.cache[key] = _Failure(e); raise
    ctx.cache[key] = v
    return v


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        pass


def hub_dir():
    """weight_utils.setup_torch_hub_dir: torch.hub cache = weights/torch_hub (DINOv2 code + checkpoints)."""
    import torch
    torch.hub.set_dir(str(WB / 'torch_hub'))
    return WB / 'torch_hub'


@contextlib.contextmanager
def syspath(*paths):
    added = [str(p) for p in paths if str(p) not in sys.path]
    for p in reversed(added): sys.path.insert(0, p)
    try: yield
    finally:
        for p in added:
            if p in sys.path: sys.path.remove(p)


@contextlib.contextmanager
def hidden_modules(*names):
    """Temporarily hide already-imported top-level modules (e.g. a foreign bare `utils`) so a third-party root can
    import its own same-named package; the hidden entries are restored afterwards."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k in names or any(k.startswith(n + '.') for n in names)}
    try: yield
    finally:
        for k in [k for k in list(sys.modules) if k in names or any(k.startswith(n + '.') for n in names)]: del sys.modules[k]
        sys.modules.update(saved)


def purge_modules(root):
    """Drop modules imported by bare name from a third-party root (RAFT/AMT both ship top-level `utils`)."""
    root = str(root) + os.sep
    for k in [k for k, v in list(sys.modules.items()) if root in (getattr(v, '__file__', '') or '')]:
        del sys.modules[k]


def turn_segments(ctx, kinds=None):
    """WBench 'interactions' -> (turn, start_step, end_step_exclusive). ctx.turns frames are inclusive [i0, i1]."""
    out = []
    for i, t in enumerate(ctx.turns or []):
        if kinds and t.get('kind', t.get('type')) not in kinds and t.get('type') not in kinds: continue
        s, e = int(t['frames'][0]), int(t['frames'][1]) + 1
        out.append((t, s, min(e, len(ctx.frames))))
    return out


def contact_sheet(images, cols=4, cell=384, label=True, start_label=0):
    """Several frames -> one grid image (temporal order, numbered). The gateway only accepts ~100 KiB requests, so
    multi-frame VLM prompts send numbered contact sheets instead of WBench's base64 video / per-frame images."""
    from PIL import Image, ImageDraw
    n = len(images); cols = min(cols, n); rows = (n + cols - 1) // cols
    w0, h0 = images[0].size; ch = int(cell * h0 / w0)
    sheet = Image.new('RGB', (cols * cell, rows * ch), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for k, im in enumerate(images):
        x, y = (k % cols) * cell, (k // cols) * ch
        sheet.paste(im.convert('RGB').resize((cell, ch)), (x, y))
        if label:
            d.rectangle([x, y, x + 34, y + 18], fill=(0, 0, 0)); d.text((x + 3, y + 3), str(start_label + k + 1), fill=(255, 255, 0))
    return sheet


def vlm_sheets(images, per_sheet=16, max_sheets=3, cols=4, cell=384):
    """Split frames into at most max_sheets numbered sheets (uniformly subsampling if there are more frames)."""
    if len(images) > per_sheet * max_sheets:
        keep = np.linspace(0, len(images) - 1, per_sheet * max_sheets).astype(int); images = [images[i] for i in keep]
    out = []
    for s in range(0, len(images), per_sheet):
        out.append(contact_sheet(images[s:s + per_sheet], cols=cols, cell=cell, start_label=s))
    return out


def vlm_ask(ctx, prompt, images, tag, system=None, max_tokens=1200):
    """One gpt-6.1-sol request through the data_pipeline gateway client (annotation_pipeline.vlm.VLM) with a disk cache.
    The client demands a single JSON object, so WBench's free-text formats are requested as JSON fields."""
    import hashlib
    from annotation_pipeline.vlm import VLM
    cache_dir = Path(ctx.case_dir) / 'vlm_cache' if ctx.case_dir else TMP / 'vlm_cache'
    run_tag = hashlib.sha1(str(ctx.extra.get('frames_dir', id(ctx.frames))).encode()).hexdigest()[:8]
    img_dir = TMP / 'vlm_images' / run_tag; img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, im in enumerate(images):
        p = img_dir / f'{tag}_{k}.jpg'; im.save(p, quality=90); paths.append(p)
    text = (f'[System]\n{system}\n\n' if system else '') + prompt
    key = hashlib.sha1((text + '|'.join(hashlib.sha1(p.read_bytes()).hexdigest() for p in paths)).encode()).hexdigest()[:16]
    model = VLM(model=GPT_MODEL, route=GPT_MODEL)
    return model.call(text, paths, cache_dir / f'wbench_{tag}_{key}.json', max_tokens=max_tokens)


# ───────────────────────────── V1 aesthetic quality ─────────────────────────────

def _aesthetic_models():
    import clip, torch, torch.nn as nn
    dev = device()
    clip_model, preprocess = clip.load('ViT-L/14', device=dev, download_root=str(WB / 'clip'))
    head = nn.Linear(768, 1)
    head.load_state_dict(torch.load(WB / 'aesthetic' / 'sa_0_4_vit_l_14_linear.pth', map_location='cpu'))
    return clip_model, preprocess, head.to(dev).eval()


@operator('wbench.V1_aesthetic_quality', 'wbench.video_quality', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/video_quality/aesthetic_quality.py:39-48 (LAION sa_0_4_vit_l_14_linear on CLIP ViT-L/14)',
          requires=('clip', 'weights/wbench/clip/ViT-L-14.pt', 'weights/wbench/aesthetic'))
def aesthetic_quality(ctx):
    """Mean over 2-fps frames of LAION-aesthetic(CLIP-L/14 L2-normalised feature)/10."""
    import torch, torch.nn.functional as F
    idx = sample_indices(len(ctx.frames), ctx.fps, FRAME_SAMPLING['aesthetic_quality'])
    clip_model, preprocess, head = _aesthetic_models(); dev = device(); scores = []
    with torch.no_grad():
        for i in idx:
            feats = clip_model.encode_image(preprocess(pil(ctx.frames[i])).unsqueeze(0).to(dev)).to(torch.float32)
            scores.append(head(F.normalize(feats, dim=-1, p=2)).item() / 10.0)
    del clip_model, head; free_gpu()
    return wb_score(np.mean(scores), {'num_frames': len(idx)}, [{'step': i, 'score01': s} for i, s in zip(idx, scores)])


# ───────────────────────────── V2 imaging quality ─────────────────────────────

@operator('wbench.V2_imaging_quality', 'wbench.video_quality', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/video_quality/imaging_quality.py:58-93 (pyiqa MUSIQ-KonIQ, VBench transform)',
          requires=('pyiqa', 'weights/wbench/pyiqa/musiq_koniq_ckpt-e95806b9.pth'))
def imaging_quality(ctx):
    """Mean MUSIQ/100 over 2-fps frames; long edge resized to 512 (antialias=False) as VBench does."""
    import torch, torchvision.transforms as T
    hub_dir()
    import pyiqa, pyiqa.utils.download_util as dl
    dl.DEFAULT_CACHE_DIR = str(WB / 'pyiqa')
    model = pyiqa.create_metric('musiq', device=device())
    idx = sample_indices(len(ctx.frames), ctx.fps, FRAME_SAMPLING['imaging_quality']); scores = []
    with torch.no_grad():
        for i in idx:
            t = torch.from_numpy(np.ascontiguousarray(ctx.frames[i])).permute(2, 0, 1)
            _, h, w = t.shape
            if max(h, w) > 512:
                s = 512 / max(h, w); t = T.Resize(size=(int(s * h), int(s * w)), antialias=False)(t)
            scores.append(model((t.float() / 255.0).unsqueeze(0).to(device())).item() / 100.0)
    del model; free_gpu()
    return wb_score(np.mean(scores), {'num_frames': len(idx)}, [{'step': i, 'score01': s} for i, s in zip(idx, scores)])


# ───────────────────────────── V3 temporal flickering ─────────────────────────────

@operator('wbench.V3_temporal_flickering', 'wbench.video_quality', ('frames',), range=(0, 100),
          source=REF + ' src/metrics/video_quality/temporal_flickering.py:13-24')
def temporal_flickering(ctx):
    """(255 - mean |I_t - I_t+1|) / 255 over 10-fps frames (higher = less flicker)."""
    idx = sample_indices(len(ctx.frames), ctx.fps, FRAME_SAMPLING['temporal_flickering'])
    if len(idx) < 2: return wb_score(1.0, note='fewer than 2 frames')
    diffs = [float(np.mean(np.abs(ctx.frames[a].astype(np.float32) - ctx.frames[b].astype(np.float32)))) for a, b in zip(idx, idx[1:])]
    mae = float(np.mean(diffs))
    return wb_score((255.0 - mae) / 255.0, {'avg_mae': mae, 'num_frames': len(idx)})


# ───────────────────────────── V4 dynamic degree ─────────────────────────────

def _raft():
    import torch
    root = WB_TP / 'RAFT' / 'core'
    with hidden_modules('utils', 'raft', 'update', 'extractor', 'corr'), syspath(root):
        from raft import RAFT                # RAFT's bare `utils` is dropped again on exit (would shadow AMT's)
        from utils.utils import InputPadder
        from easydict import EasyDict as edict
    w = WB / 'raft' / 'raft-things.pth'
    model = RAFT(edict({'model': str(w), 'small': False, 'mixed_precision': False, 'alternate_corr': False}))
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(w, map_location='cpu').items()})
    return model.to(device()).eval(), InputPadder


def _resize_min_edge(img, max_edge=512, resample=None):
    w, h = img.size
    if min(w, h) <= max_edge: return img
    nw, nh = (int(w * max_edge / h), max_edge) if h < w else (max_edge, int(h * max_edge / w))
    return img.resize((nw, nh)) if resample is None else img.resize((nw, nh), resample)


@operator('wbench.V4_dynamic_degree', 'wbench.video_quality', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/video_quality/dynamic_degree.py:157-234 (RAFT-things, VBench rule; Farneback fallback)',
          requires=('third_party/wbench/RAFT', 'weights/wbench/raft/raft-things.pth'))
def dynamic_degree(ctx):
    """1 if at least round(4*N/16) consecutive 8-fps frame pairs have mean top-5% RAFT flow > 6*min_side/256, else 0."""
    import torch, cv2
    idx = sample_indices(len(ctx.frames), ctx.fps, FRAME_SAMPLING['dynamic_degree'])
    if len(idx) < 2: return wb_score(0.0, note='fewer than 2 frames')
    imgs = [_resize_min_edge(pil(ctx.frames[i]), 512) for i in idx]
    try:
        model, Padder = _raft(); method = 'raft'
    except Exception as e:
        model, method, err = None, 'farneback', f'{type(e).__name__}: {e}'
    scale = min(np.array(imgs[0]).shape[:2]); thres = 6.0 * (scale / 256.0)
    count_num = max(1, round(4 * (len(imgs) / 16.0))); move = 0; mags = []
    if model is not None:
        ts = [torch.from_numpy(np.array(f).astype(np.uint8)).permute(2, 0, 1).float()[None].to(device()) for f in imgs]
        with torch.no_grad():
            for a, b in zip(ts, ts[1:]):
                p = Padder(a.shape); i1, i2 = p.pad(a, b)
                _, flow = model(i1, i2, iters=20, test_mode=True)
                f = flow[0].permute(1, 2, 0).cpu().numpy(); rad = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
                mags.append(float(np.mean(np.sort(rad.flatten())[-int(rad.size * .05):])))
                if mags[-1] > thres: move += 1
                if move >= count_num: break
        del model; free_gpu()
    else:
        prev = cv2.cvtColor(np.array(imgs[0]), cv2.COLOR_RGB2GRAY)
        for f in imgs[1:]:
            cur = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, cur, None, .5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1]); mags.append(float(np.mean(np.sort(mag.flatten())[-int(mag.size * .05):])))
            if mags[-1] > thres: move += 1
            if move >= count_num: break
            prev = cur
    raw = {'method': method, 'move_count': move, 'count_num': count_num, 'threshold_px': thres, 'num_frames': len(imgs),
           'top5pct_flow_px': mags}
    if method != 'raft': raw['raft_error'] = err
    return wb_score(1.0 if move >= count_num else 0.0, raw, note='' if method == 'raft' else 'RAFT unavailable, Farneback fallback')


# ───────────────────────────── V5 motion smoothness ─────────────────────────────

def _amt():
    import torch
    amt_root = WB_TP / 'amt'
    with hidden_modules('utils', 'networks', 'datasets'), syspath(amt_root):
        from omegaconf import OmegaConf
        with syspath(WB_TP):
            from amt.utils.build_utils import build_from_cfg
        model = build_from_cfg(OmegaConf.load(amt_root / 'cfgs' / 'AMT-S.yaml').network)
    pref = str(amt_root) + os.sep          # motion_smoothness.py:52-63: drop leaked bare modules, keep `amt.*`
    for k in [k for k, v in list(sys.modules.items()) if not (k == 'amt' or k.startswith('amt.')) and pref in (getattr(v, '__file__', '') or '')]:
        del sys.modules[k]
    ck = torch.load(WB / 'amt' / 'amt-s.pth', map_location='cpu'); sd = ck.get('state_dict', ck)
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    return model.to(device()).eval()


@operator('wbench.V5_motion_smoothness', 'wbench.video_quality', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/video_quality/motion_smoothness.py:69-115 (AMT-S frame interpolation, VBench)',
          requires=('third_party/wbench/amt', 'weights/wbench/amt/amt-s.pth'))
def motion_smoothness(ctx, batch_size=8):
    """(255 - mean |AMT(I_i, I_i+2) - I_i+1|) / 255 over 10-fps frames, min side resized to 512 (LANCZOS)."""
    import torch, cv2
    from PIL import Image
    idx = sample_indices(len(ctx.frames), ctx.fps, FRAME_SAMPLING['motion_smoothness'])
    if len(idx) < 3: raise NotApplicable('motion smoothness needs >= 3 frames')
    model = _amt()
    with syspath(WB_TP):
        from amt.utils.utils import img2tensor, tensor2img, InputPadder
    frames = [_resize_min_edge(pil(ctx.frames[i]), 512, Image.LANCZOS) for i in idx]
    inputs = [img2tensor(np.array(f)).to(device()) for f in frames]
    padder = InputPadder(inputs[0].shape); embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(device())
    pairs = [(i, i + 2, i + 1) for i in range(0, len(inputs) - 2, 2)]; diffs = []
    with torch.no_grad():
        for s in range(0, len(pairs), batch_size):
            bp = pairs[s:s + batch_size]
            i0, i1 = padder.pad(torch.cat([inputs[p[0]] for p in bp]), torch.cat([inputs[p[1]] for p in bp]))
            out = model(i0, i1, embt.expand(len(bp), -1, -1, -1), scale_factor=1.0, eval=True)
            pred = padder.unpad(out['imgt_pred'])
            for j, p in enumerate(bp): diffs.append(float(np.mean(cv2.absdiff(tensor2img(pred[j:j + 1]), np.array(frames[p[2]])))))
    del model; free_gpu()
    mae = float(np.mean(diffs))
    return wb_score((255.0 - mae) / 255.0, {'vfi_mae': mae, 'num_frames': len(idx), 'pairs': len(pairs)})


# ───────────────────────────── V6 HPSv3-Norm ─────────────────────────────

HPSV3_P1, HPSV3_P99 = -5.21, 8.61                       # hpsv3_quality.py:140-141 (fixed global winsorised scale)


@operator('wbench.V6_hpsv3_norm', 'wbench.video_quality', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/video_quality/hpsv3_quality.py:144-212 (HPSv3 on Qwen2-VL-7B, 20 frames, p1/p99 norm)',
          requires=('third_party/wbench/HPSv3', 'trl', 'weights/wbench/HPSv3', 'weights/wbench/Qwen2-VL-7B-Instruct'))
def hpsv3_norm(ctx):
    """clip((mean HPSv3 reward over 20 uniformly sampled frames, empty prompt) - p1) / (p99 - p1), 0, 1)."""
    import torch, yaml
    import transformers.image_utils as iu
    if not hasattr(iu, 'VideoInput'): iu.VideoInput = list
    ck, cfg = WB / 'HPSv3' / 'HPSv3.safetensors', WB / 'HPSv3' / 'HPSv3_7B_local.yaml'
    qwen = WB / 'Qwen2-VL-7B-Instruct'
    if not ck.exists(): raise FileNotFoundError(f'HPSv3 checkpoint missing: {ck}')
    import json as _json
    shards = set(_json.loads((qwen / 'model.safetensors.index.json').read_text())['weight_map'].values())
    base_ok = all((qwen / x).exists() for x in shards)
    c = yaml.safe_load(cfg.read_text()); c['model_name_or_path'] = str(qwen)
    resolved = TMP / 'wbench_hpsv3_resolved.yaml'; resolved.write_text(yaml.dump(c))
    os.environ['HF_HUB_OFFLINE'] = '1'
    for k in [k for k in sys.modules if k in ('datasets', 'utils') or k.startswith(('datasets.', 'utils.'))]:
        if 'amt' in (getattr(sys.modules[k], '__file__', '') or '') or 'RAFT' in (getattr(sys.modules[k], '__file__', '') or ''): del sys.modules[k]
    with syspath(WB_TP / 'HPSv3'):
        from hpsv3.inference import HPSv3RewardInferencer
        import hpsv3.train as ht
        own = 'from_pretrained' in ht.Qwen2VLRewardModelBT.__dict__; orig = ht.Qwen2VLRewardModelBT.__dict__.get('from_pretrained')
        if not base_ok:
            # Qwen2-VL-7B base shards absent: build the architecture from its config.json without loading them. The
            # inferencer then loads HPSv3.safetensors with strict=True, which sets every parameter, so the result is
            # identical to WBench's (base weights are fully overwritten).
            def from_config(name, *a, **kw):
                from transformers import AutoConfig
                from transformers.modeling_utils import no_init_weights
                cfg = AutoConfig.from_pretrained(name); cfg._attn_implementation = kw.get('attn_implementation', 'sdpa')
                cfg.use_cache = False; dt = kw.get('torch_dtype') or torch.bfloat16
                old_dt = torch.get_default_dtype(); torch.set_default_dtype(dt if isinstance(dt, torch.dtype) else torch.bfloat16)
                try:
                    with no_init_weights():
                        m = ht.Qwen2VLRewardModelBT(cfg, output_dim=kw['output_dim'], reward_token=kw['reward_token'],
                                                    special_token_ids=kw['special_token_ids'], rm_head_type=kw['rm_head_type'],
                                                    rm_head_kwargs=kw['rm_head_kwargs'])
                finally: torch.set_default_dtype(old_dt)
                return m
            ht.Qwen2VLRewardModelBT.from_pretrained = staticmethod(from_config)
        # The checkpoint uses pre-4.52 Qwen2-VL key names (visual.*, model.layers.*); WBench pins transformers<4.52.
        # Under 4.57 apply transformers' own _checkpoint_conversion_mapping before the inferencer's strict load.
        import re as _re, safetensors.torch as _st
        from transformers import Qwen2VLForConditionalGeneration as _Q
        mapping = getattr(_Q, '_checkpoint_conversion_mapping', {}) or {}; orig_load = _st.load_file

        def load_remapped(path, device='cpu'):
            out = {}
            for k, v in orig_load(path, device=device).items():
                for pat, rep in mapping.items():
                    nk, n = _re.subn(pat, rep, k)
                    if n: k = nk; break
                out[k] = v
            return out
        _st.load_file = load_remapped
        try:
            inf = HPSv3RewardInferencer(config_path=str(resolved), checkpoint_path=str(ck), device=device())
        finally:
            _st.load_file = orig_load
        rm = inf.model           # HPSv3 forward uses the <4.52 layout: self.model = text LM, self.visual = ViT
        if not hasattr(rm.model, 'embed_tokens') and hasattr(rm.model, 'language_model'):
            vis, lm = rm.model.visual, rm.model.language_model
            if not hasattr(vis, 'get_dtype'): vis.get_dtype = lambda: next(vis.parameters()).dtype
            rm.visual = vis; rm.model = lm
            if not base_ok:
                if own: ht.Qwen2VLRewardModelBT.from_pretrained = orig
                else: del ht.Qwen2VLRewardModelBT.from_pretrained
    idx = sample_indices(len(ctx.frames), ctx.fps, FRAME_SAMPLING['hpsv3_quality'])
    tmp = TMP / 'tmp' / f'hpsv3_{os.getpid()}'; tmp.mkdir(parents=True, exist_ok=True); paths = []
    for k, i in enumerate(idx):
        p = tmp / f'{k:03d}.png'; pil(ctx.frames[i]).save(p); paths.append(str(p))
    try:
        with torch.no_grad(): rewards = inf.reward([''] * len(paths), paths)
        raws = [float(rewards[k][0].item()) for k in range(len(paths))]
    finally:
        for p in paths: Path(p).unlink(missing_ok=True)
        del inf; free_gpu()
    raw_mean = float(np.mean(raws))
    s = max(0.0, min(1.0, (raw_mean - HPSV3_P1) / (HPSV3_P99 - HPSV3_P1)))
    return wb_score(s, {'raw_mean': raw_mean, 'raw_scores': raws, 'p1': HPSV3_P1, 'p99': HPSV3_P99, 'num_frames': len(idx),
                        'qwen2vl_base_weights_loaded': base_ok},
                    note='' if base_ok else 'Qwen2-VL-7B base shards absent: architecture from config, all weights from HPSv3.safetensors (strict)')
