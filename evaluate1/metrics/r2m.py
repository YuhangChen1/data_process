"""R2M-Bench metric families as evaluate1 operators (revisit vs gap-matched baseline vs short-range pairs -> MG / NMR).

Faithful to /mnt/cfs/cyh/tmp/refs/R2MBench-main:
  appearance        src/metrics/appearance_fidelity.py:18-63   PSNR (skimage, data_range 255), SSIM (win 7, channel_axis 2),
                                                               LPIPS AlexNet on [-1,1] full-resolution frames
  scene identity    src/metrics/scene_identity.py:40-233       DINOv2 ViT-B/14 (resize 518 bicubic + centre crop 518),
                                                               BoQ (DINOv2-B + BoQ 12288-d, 322x322), MutualVPR (518x518, 512-d);
                                                               cosine of L2-normalised global descriptors
  local geometry    src/metrics/geometric_correspondence.py    SuperPoint (1024 kp) + LightGlue; match ratio
                                                               |M|/max(min(Ka,Kb),1); RANSAC F (3 px, 0.99, >=8 matches) inlier ratio
  object identity   src/metrics/object_identity.py:206-507     GroundingDINO SwinT-OGC (box 0.25, text 0.20, NMS 0.60, <=10 anchors,
                                                               min area 0.002) + SAM2.1-L box masks; re-detect each anchor concept
                                                               (<=5 candidates); appearance = clip01((cos_DINO(masked crops)+1)/2);
                                                               semantic = 0.5 GDINO conf + 0.5 clip01((cos_CLIP(crop, concept)+1)/2);
                                                               match if 0.5 app + 0.5 sem >= 0.5 (eval_gpu_metrics.py:228-236 defaults);
                                                               persistence = mean over anchors with 0 for unmatched
  persistent state  src/metrics/persistent_state.py:28-130     same rubric and score parsing (1..5); judge gpt-6.1-sol instead of
                                                               gemini-3.1-pro-preview; <=10 pairs per type (run_parallel_eval.sh:228)
  MG / NMR          src/compute_nmr_summary.py (see pairs.py)

Every metric operator returns value = NMR of that metric for this case (higher is better; LPIPS differences reversed),
raw = {revisit, baseline, short: {mean,std,median,count}, mg, dr, nmr, nmr_valid}. 'r2m.summary.nmr' is the Overall NMR
(mean over families of mean metric NMR). The model loaders here are shared with revisit.py and fidelity.py.

Deviations (documented in docs/r2m_revisit_ops.md): PSNR of bit-identical pairs is capped at 100 dB (skimage returns inf);
VLM images go through the gateway client (JPEG <= 42 KB each, the gateway's request limit) instead of lossless PNG,
and the system prompt is sent inside the user prompt.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .base import TMP, WEIGHTS, GPT_MODEL, Result, NotApplicable, operator, device, run as run_op
from . import pairs as P

TP = TMP / 'third_party' / 'r2m'
DINOV2_REPO = WEIGHTS / 'dinov2' / 'facebookresearch_dinov2_main'
DINOV2_CKPT = WEIGHTS / 'dinov2' / 'dinov2_vitb14_pretrain.pth'
BOQ_CKPT = WEIGHTS / 'boq' / 'dinov2_12288.pth'
MVPR_CKPT = WEIGHTS / 'mutualvpr' / 'MutualVPR_model_512.pth'
CLIP_CKPT = WEIGHTS / 'clip' / 'ViT-B-32.pt'
GDINO_CFG = TP / 'GroundingDINO' / 'groundingdino' / 'config' / 'GroundingDINO_SwinT_OGC.py'
GDINO_CKPT = WEIGHTS / 'groundingdino' / 'groundingdino_swint_ogc.pth'
SAM2_CFG = TP / 'sam2' / 'sam2' / 'configs' / 'sam2.1' / 'sam2.1_hiera_l.yaml'
SAM2_CKPT = WEIGHTS / 'sam2' / 'sam2.1_hiera_large.pt'
DREAMSIM_DIR = WEIGHTS / 'wbench' / 'dreamsim'        # shared with wbench_consistency.py (one 1.2 GB download)
TORCH_HUB = WEIGHTS / 'torch_hub'                      # SuperPoint, LightGlue, AlexNet / VGG16 (LPIPS, DISTS)
PYIQA_DIR = WEIGHTS / 'pyiqa'                          # NIQE / BRISQUE / DISTS parameters
BERT_DIR = WEIGHTS / 'bert-base-uncased'               # config + tokenizer only (weights live in the GDINO checkpoint)
VLM_CACHE = TMP / 'vlm_cache' / 'r2m_state'
SRC = 'R2M-Bench (arXiv 2608.27328), github.com/AMAP-ML/R2MBench'

DEFAULT_TEXT_PROMPT = ('object . furniture . building . vehicle . plant . person . animal . sign . door . window . chair . '
                       'table . sofa . bed . cabinet . lamp . screen . picture . shelf . tree . road . wall . floor .')

_MODELS: dict = {}
_LOCK = threading.RLock()                                # loaders nest (GroundedSam -> DINOv2, CLIP)
os.environ.setdefault('TORCH_HOME', str(TMP / 'torch'))
# Runtime never downloads: every loader reads from WEIGHTS (prefetch_weights() fills it once, with retries). Set
# EVAL_ALLOW_DOWNLOAD=1 only for that prefetch.
ALLOW_DOWNLOAD = os.environ.get('EVAL_ALLOW_DOWNLOAD') == '1'


@contextlib.contextmanager
def _syspath(path):
    p = str(path); sys.path.insert(0, p)
    try: yield
    finally:
        if p in sys.path: sys.path.remove(p)


@contextlib.contextmanager
def _hub(path=TORCH_HUB):
    """Point torch.hub (load_state_dict_from_url, torchvision pretrained weights) at the weight store for one load."""
    import torch
    prev = torch.hub.get_dir(); Path(path, 'checkpoints').mkdir(parents=True, exist_ok=True); torch.hub.set_dir(str(path))
    try: yield
    finally: torch.hub.set_dir(prev)


def _need(*files):
    missing = [str(f) for f in files if not Path(f).exists()]
    if missing and not ALLOW_DOWNLOAD:
        raise FileNotFoundError('weights missing (run `python3 evaluate1/tests/smoke_r2m.py --prefetch` once): ' + ', '.join(missing))


def _model(name, build):
    with _LOCK:
        if name not in _MODELS: _MODELS[name] = build()
        return _MODELS[name]


def _to_uint8(img):
    img = np.asarray(img)
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------------------------------
# model loaders (shared)
# ----------------------------------------------------------------------------------------------------------------------

def _dinov2_backbone():
    import torch
    _need(DINOV2_REPO / 'hubconf.py', DINOV2_CKPT)
    m = torch.hub.load(str(DINOV2_REPO), 'dinov2_vitb14', source='local', pretrained=False)
    m.load_state_dict(torch.load(DINOV2_CKPT, map_location='cpu', weights_only=True), strict=True)
    return m


class DINOv2Extractor:
    """scene_identity.py:40-79."""
    def __init__(self, dev):
        import torch, torchvision.transforms as T
        self.torch = torch; self.dev = dev
        self.model = _dinov2_backbone().to(dev).eval()
        self.transform = T.Compose([T.Resize(518, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(518), T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def extract_feature(self, image):
        from PIL import Image
        with self.torch.no_grad():
            x = self.transform(Image.fromarray(_to_uint8(image))).unsqueeze(0).to(self.dev)
            return self.torch.nn.functional.normalize(self.model(x), dim=-1)

    def similarity(self, a, b):
        return float(self.torch.nn.functional.cosine_similarity(self.extract_feature(a), self.extract_feature(b), dim=-1).item())


def get_dino():
    return _model('dino', lambda: DINOv2Extractor(device()))


class _BoQ:
    """scene_identity.py:82-193 (DinoV2Backbone + BoQ aggregator, 322x322)."""
    def __init__(self, dev):
        import torch, torch.nn as nn, torchvision.transforms as T
        self.torch = torch; self.dev = dev
        _need(BOQ_CKPT)
        with _syspath(TP / 'Bag-of-Queries' / 'src'):
            from boq import BoQ

        class DinoV2Backbone(nn.Module):
            def __init__(self):
                super().__init__(); self.dino = _dinov2_backbone()
                for p in self.dino.parameters(): p.requires_grad = False
                self.out_channels = self.dino.embed_dim

            def forward(self, t):
                b, _, h, w = t.shape
                with torch.no_grad():
                    tok = self.dino.prepare_tokens_with_masks(t)
                    for blk in self.dino.blocks: tok = blk(tok)
                tok = tok[:, 1:]; ps = self.dino.patch_embed.patch_size[0]
                return tok.permute(0, 2, 1).view(b, tok.shape[-1], h // ps, w // ps)

        class Model(nn.Module):
            def __init__(self, backbone, aggregator):
                super().__init__(); self.backbone = backbone; self.aggregator = aggregator

            def forward(self, t):
                d, _ = self.aggregator(self.backbone(t)); return d

        bb = DinoV2Backbone()
        agg = BoQ(in_channels=bb.out_channels, proj_channels=384, num_queries=64, num_layers=2, row_dim=12288 // 384)
        m = Model(bb, agg); m.load_state_dict(torch.load(BOQ_CKPT, map_location='cpu', weights_only=True), strict=True)
        self.model = m.to(dev).eval()
        self.transform = T.Compose([T.Resize((322, 322), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def extract_feature(self, image):
        from PIL import Image
        with self.torch.no_grad():
            x = self.transform(Image.fromarray(_to_uint8(image))).unsqueeze(0).to(self.dev)
            return self.torch.nn.functional.normalize(self.model(x), dim=-1)


def get_boq():
    return _model('boq', lambda: _BoQ(device()))


class _MutualVPR:
    """scene_identity.py:196-233 (DINOv2 + GeM + Linear 512, 518x518)."""
    def __init__(self, dev):
        import torch, torchvision.transforms as T
        self.torch = torch; self.dev = dev
        _need(MVPR_CKPT)
        with _syspath(TP / 'MutualVPR'):
            from cosplace_model.cosplace_network import MutualVPR
        m = MutualVPR(pretrained_foundation=False, output_dim=512)
        sd = torch.load(MVPR_CKPT, map_location='cpu')
        m.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
        self.model = m.to(dev).eval()
        self.transform = T.Compose([T.Resize((518, 518), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def extract_feature(self, image):
        from PIL import Image
        with self.torch.no_grad():
            x = self.transform(Image.fromarray(_to_uint8(image))).unsqueeze(0).to(self.dev)
            return self.torch.nn.functional.normalize(self.model(x), dim=-1)


def get_mvpr():
    return _model('mvpr', lambda: _MutualVPR(device()))


class LPIPSComputer:
    """appearance_fidelity.py:28-63 (LPIPS on [-1,1] tensors at native resolution)."""
    def __init__(self, net, dev):
        import torch, lpips
        self.torch = torch; self.dev = dev
        _need(TORCH_HUB / 'checkpoints' / {'alex': 'alexnet-owt-7be5be79.pth', 'vgg': 'vgg16-397923af.pth'}[net])
        with _hub():
            self.fn = lpips.LPIPS(net=net, verbose=False).to(dev).eval()

    def __call__(self, a, b):
        t = lambda x: (self.torch.from_numpy(_to_uint8(x)).float().permute(2, 0, 1) / 255.0 * 2 - 1).unsqueeze(0).to(self.dev)
        with self.torch.no_grad():
            return float(self.fn(t(a), t(b)).item())


def get_lpips(net='alex'):
    return _model('lpips_' + net, lambda: LPIPSComputer(net, device()))


class KeypointMatcher:
    """geometric_correspondence.py:17-72."""
    def __init__(self, dev, max_keypoints=1024):
        import torch
        self.torch = torch; self.dev = dev
        with _syspath(TP / 'LightGlue'):
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import numpy_image_to_torch
        self.to_torch = numpy_image_to_torch
        _need(TORCH_HUB / 'checkpoints' / 'superpoint_v1.pth', TORCH_HUB / 'checkpoints' / 'superpoint_lightglue_v0-1_arxiv.pth')
        with _hub():
            self.superpoint = SuperPoint(max_num_keypoints=max_keypoints).eval().to(dev)
            self.lightglue = LightGlue(features='superpoint').eval().to(dev)

    def match(self, a, b):
        torch = self.torch
        with torch.no_grad():
            fa = self.superpoint.extract(self.to_torch(_to_uint8(a)).to(self.dev))
            fb = self.superpoint.extract(self.to_torch(_to_uint8(b)).to(self.dev))
            ka, kb = int(fa['keypoints'].shape[1]), int(fb['keypoints'].shape[1])
            m0 = self.lightglue({'image0': fa, 'image1': fb})['matches0'][0]
            valid = m0 > -1; n = int(valid.sum().item())
            pa = fa['keypoints'][0][torch.where(valid)[0]].cpu().numpy() if n else np.empty((0, 2))
            pb = fb['keypoints'][0][m0[valid].long()].cpu().numpy() if n else np.empty((0, 2))
        return {'num_keypoints_a': ka, 'num_keypoints_b': kb, 'num_matches': n, 'match_ratio': round(n / max(min(ka, kb), 1), 4),
                'matched_kp_a': pa, 'matched_kp_b': pb}


def get_matcher():
    return _model('matcher', lambda: KeypointMatcher(device()))


def compute_two_view_geometry(pa, pb):
    """geometric_correspondence.py:75-111."""
    import cv2
    zero = {'ransac_inlier_ratio': 0.0, 'num_ransac_inliers': 0}
    if pa is None or len(pa) < 8: return dict(zero)
    try:
        F, mask = cv2.findFundamentalMat(pa.astype(np.float64), pb.astype(np.float64), cv2.FM_RANSAC, 3.0, 0.99)
        if F is None or mask is None: return dict(zero)
        n = int(mask.ravel().astype(bool).sum())
        return {'ransac_inlier_ratio': round(n / len(pa), 6), 'num_ransac_inliers': n}
    except Exception:
        return dict(zero)


def compute_psnr_ssim(a, b, psnr_cap=100.0):
    """appearance_fidelity.py:18-25 (+ cap for identical frames)."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    a = _to_uint8(a); b = _to_uint8(b)
    if a.shape != b.shape:
        import cv2
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    psnr = float(peak_signal_noise_ratio(a, b, data_range=255)) if not np.array_equal(a, b) else float('inf')
    ssim = float(structural_similarity(a, b, data_range=255, channel_axis=2, win_size=7))
    return {'psnr': round(min(psnr, psnr_cap), 4), 'ssim': round(ssim, 6), 'psnr_capped': not math.isfinite(psnr) or psnr > psnr_cap}


class CLIPExtractor:
    """object_identity.py:186-203 (openai/CLIP ViT-B/32 via clip.load)."""
    def __init__(self, dev):
        import torch
        self.torch = torch; self.dev = dev
        _need(CLIP_CKPT)
        with _syspath(TP):
            import clip as clip_lib
        self.model, self.preprocess = clip_lib.load(str(CLIP_CKPT), device=dev)
        self.model.eval(); self.tokenize = clip_lib.tokenize
        self._text = {}

    def image(self, img):
        from PIL import Image
        with self.torch.no_grad():
            f = self.model.encode_image(self.preprocess(Image.fromarray(_to_uint8(img))).unsqueeze(0).to(self.dev)).float()
            return self.torch.nn.functional.normalize(f, dim=-1)

    def text(self, s):
        if s not in self._text:
            with self.torch.no_grad():
                f = self.model.encode_text(self.tokenize([s]).to(self.dev)).float()
                self._text[s] = self.torch.nn.functional.normalize(f, dim=-1)
        return self._text[s]


def get_clip():
    return _model('clip', lambda: CLIPExtractor(device()))


def _bert_path():
    """bert-base-uncased config + tokenizer (HF files, verified by git-blob hash). GroundingDINO's checkpoint already
    contains every BERT parameter (200 'bert.*' tensors, 109.5 M values), which overwrite the language model after
    construction exactly as in R2M (BertModel.from_pretrained, then load_state_dict(ckpt, strict=False)), so the
    440 MB pretrained BERT file is not needed: the model is built from its config and filled from the checkpoint."""
    _need(BERT_DIR / 'config.json', BERT_DIR / 'vocab.txt', BERT_DIR / 'tokenizer_config.json')
    return str(BERT_DIR)


def _gdino_ops():
    """Compile GroundingDINO's MultiScaleDeformableAttention CUDA op (setup.py ext 'groundingdino._C') with torch's JIT
    loader; if that fails, route the op to GroundingDINO's own pure-PyTorch implementation."""
    import torch
    csrc = TP / 'GroundingDINO' / 'groundingdino' / 'models' / 'GroundingDINO' / 'csrc'
    try:
        from torch.utils.cpp_extension import load
        build = Path(os.environ.get('TORCH_EXTENSIONS_DIR', TMP / 'torch_ext')) / 'groundingdino_C'
        build.mkdir(parents=True, exist_ok=True)
        mod = load(name='groundingdino_C', build_directory=str(build), verbose=False,
                   sources=[str(csrc / 'vision.cpp'), str(csrc / 'cuda_version.cu'),
                            str(csrc / 'MsDeformAttn' / 'ms_deform_attn_cpu.cpp'), str(csrc / 'MsDeformAttn' / 'ms_deform_attn_cuda.cu')],
                   extra_include_paths=[str(csrc)], extra_cflags=['-DWITH_CUDA', '-O2'],
                   extra_cuda_cflags=['-DWITH_CUDA', '-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__',
                                      '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__'])
        sys.modules['groundingdino._C'] = mod
        return 'cuda_op'
    except Exception as e:                                       # pragma: no cover - depends on toolchain
        return 'pytorch_fallback: ' + f'{type(e).__name__}: {str(e)[:200]}'


class GroundedSam:
    """object_identity.py:206-507 GroundedSamObjectEvaluator (R2M GPU-runner defaults)."""
    def __init__(self, dev, dino, clip, text_prompt=DEFAULT_TEXT_PROMPT, box_threshold=.25, text_threshold=.20, nms_iou=.60,
                 max_objects=10, min_area_ratio=.002, match_threshold=.50, max_candidates_per_concept=5,
                 appearance_weight=.5, semantic_weight=.5):
        import torch
        self.torch = torch; self.dev = dev; self.dino = dino; self.clip = clip
        self.text_prompt = text_prompt; self.box_threshold = box_threshold; self.text_threshold = text_threshold
        self.nms_iou = nms_iou; self.max_objects = max_objects; self.min_area_ratio = min_area_ratio
        self.match_threshold = match_threshold; self.max_candidates_per_concept = max_candidates_per_concept
        self.appearance_weight = appearance_weight; self.semantic_weight = semantic_weight
        sys.path.insert(0, str(TP / 'GroundingDINO')) if str(TP / 'GroundingDINO') not in sys.path else None
        sys.path.insert(0, str(TP / 'sam2')) if str(TP / 'sam2') not in sys.path else None
        self.ops = _gdino_ops()
        import groundingdino
        if 'groundingdino._C' in sys.modules: groundingdino._C = sys.modules['groundingdino._C']
        from groundingdino.models.GroundingDINO import ms_deform_attn as msda
        if not self.ops.startswith('cuda'):
            msda.MultiScaleDeformableAttnFunction.apply = staticmethod(
                lambda v, s, lsi, loc, w, step: msda.multi_scale_deformable_attn_pytorch(v, s, loc, w))
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict
        _need(GDINO_CKPT)
        from groundingdino.util import get_tokenlizer as gtk
        from transformers import BertConfig, BertModel
        orig = gtk.get_pretrained_language_model
        gtk.get_pretrained_language_model = lambda t, p: (BertModel(BertConfig.from_pretrained(p))
                                                          if t == 'bert-base-uncased' and p else orig(t, p))
        try:
            args = SLConfig.fromfile(str(GDINO_CFG)); args.device = dev; args.bert_base_uncased_path = _bert_path()
            gm = build_model(args)
        finally:
            gtk.get_pretrained_language_model = orig
        sd = clean_state_dict(torch.load(GDINO_CKPT, map_location='cpu')['model'])
        missing = [k for k, _ in gm.named_parameters() if k not in sd]
        if missing: raise RuntimeError(f'GroundingDINO checkpoint lacks {len(missing)} parameters, e.g. {missing[:3]}')
        gm.load_state_dict(sd, strict=False)
        self.gdino = gm.eval().to(dev)
        self.sam = load_sam2(dev)

    # object_identity.py:76-95
    @staticmethod
    def _gd_transform(image_np):
        import groundingdino.datasets.transforms as T
        from PIL import Image
        tf = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return tf(Image.fromarray(image_np).convert('RGB'), None)[0]

    def detect_boxes(self, image_np, prompt):
        """GroundingDINO boxes (xyxy pixels), logits, phrases after NMS (object_identity.py:271-310)."""
        torch = self.torch
        from groundingdino.util.inference import predict
        h, w = image_np.shape[:2]
        boxes, logits, phrases = predict(model=self.gdino, image=self._gd_transform(image_np), caption=prompt,
                                         box_threshold=self.box_threshold, text_threshold=self.text_threshold, device=self.dev)
        if boxes is None or len(boxes) == 0: return torch.zeros((0, 4)), torch.zeros((0,)), []
        boxes = boxes.to(self.dev); logits = logits.to(self.dev)
        cx, cy, bw, bh = boxes.unbind(-1)
        xyxy = torch.stack([(cx - .5 * bw) * w, (cy - .5 * bh) * h, (cx + .5 * bw) * w, (cy + .5 * bh) * h], -1)
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w - 1); xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h - 1)
        try:
            from torchvision.ops import nms
            keep = nms(xyxy, logits, self.nms_iou)
        except Exception:
            keep = torch.argsort(logits, descending=True)
        return xyxy[keep], logits[keep], [phrases[int(i)] for i in keep.cpu().tolist()]

    def sam_masks(self, image_np, boxes_xyxy, key=None):
        """object_identity.py:121-148 (one SAM2 mask per box, multimask_output=False). The image embedding is reused
        while consecutive calls refer to the same frame `key` (R2M re-encodes it every call; results are identical)."""
        if key is None or key != getattr(self, '_sam_key', None):
            self.sam.set_image(image_np); self._sam_key = key
        masks = []; scores = []
        for box in np.asarray(boxes_xyxy, np.float32):
            m, s, _ = self.sam.predict(point_coords=None, point_labels=None, box=box, multimask_output=False)
            masks.append((m[0] if m.ndim == 3 else m).astype(bool)); scores.append(float(np.ravel(s)[0]))
        if not masks: return np.zeros((0,) + image_np.shape[:2], bool), np.zeros((0,), np.float32)
        return np.stack(masks), np.asarray(scores, np.float32)

    def detect_objects(self, image_np, prompt, max_objects, cache=None, key=None):
        """object_identity.py:254-361."""
        if cache is not None and key is not None and (key, prompt, max_objects) in cache: return cache[(key, prompt, max_objects)]
        image_np = _to_uint8(image_np); h, w = image_np.shape[:2]
        xyxy, logits, phrases = self.detect_boxes(image_np, prompt)
        if max_objects > 0: xyxy, logits, phrases = xyxy[:max_objects], logits[:max_objects], phrases[:max_objects]
        objects = []
        if len(xyxy):
            masks, mscores = self.sam_masks(image_np, xyxy.detach().cpu().numpy(), key=(id(cache), key) if key is not None else None)
            for i, mask in enumerate(masks):
                area = float(mask.sum() / max(w * h, 1))
                if area < self.min_area_ratio: continue
                x0, y0, x1, y1 = [int(round(v)) for v in xyxy[i].detach().cpu().tolist()]
                x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(w - 1, x1), min(h - 1, y1)
                if x1 <= x0 or y1 <= y0: continue
                crop = image_np[y0:y1 + 1, x0:x1 + 1].copy(); cm = mask[y0:y1 + 1, x0:x1 + 1]
                if cm.any(): crop[~cm] = 255
                objects.append({'bbox_xyxy': [float(x0), float(y0), float(x1), float(y1)], 'phrase': str(phrases[i]),
                                'concept': clean_concept(str(phrases[i])), 'box_score': float(logits[i].item()),
                                'mask_score': float(mscores[i]), 'area_ratio': area, 'mask': mask,
                                'feature': self.dino.extract_feature(crop).detach().cpu(),
                                'clip_feature': self.clip.image(crop).detach().cpu() if self.clip is not None else None})
        objects.sort(key=lambda o: o['box_score'] * max(o['area_ratio'], 1e-6), reverse=True)
        if max_objects > 0: objects = objects[:max_objects]
        if cache is not None and key is not None: cache[(key, prompt, max_objects)] = objects
        return objects

    def score_pair(self, a, b, fa, fb, cache=None):
        """object_identity.py:363-507."""
        F = self.torch.nn.functional
        anchors = self.detect_objects(a, self.text_prompt, self.max_objects, cache, fa)
        res = {'num_anchor_objects': len(anchors), 'num_detected_candidates': 0, 'num_object_matches': 0,
               'object_existence_rate': 0.0, 'object_appearance_similarity': 0.0, 'object_semantic_similarity': 0.0,
               'object_appearance_persistence': 0.0, 'object_semantic_persistence': 0.0, 'object_recall': 0.0,
               'object_identity_similarity': 0.0, 'object_consistency': 0.0}
        if not anchors: return res
        wsum = max(self.appearance_weight + self.semantic_weight, 1e-8); matches = {}; total = 0
        for ai, anc in enumerate(anchors):
            concept = anc['concept'] or clean_concept(anc['phrase'])
            cands = self.detect_objects(b, f'{concept} .', self.max_candidates_per_concept, cache, fb); total += len(cands)
            best = None; best_score = -float('inf')
            for c in cands:
                app = _clip01((float(F.cosine_similarity(anc['feature'], c['feature'], dim=-1).item()) + 1.0) * .5)
                gc = _clip01(c['box_score'])
                if self.clip is not None and c['clip_feature'] is not None:
                    tf = self.clip.text(concept)
                    sem = .5 * gc + .5 * _clip01((float(F.cosine_similarity(c['clip_feature'].to(tf.device), tf, dim=-1).item()) + 1.0) * .5)
                else:
                    sem = gc
                s = (self.appearance_weight * app + self.semantic_weight * sem) / wsum
                if s > best_score: best_score = s; best = (app, sem, s)
            if best is not None and best[2] >= self.match_threshold: matches[ai] = best
        n = len(anchors); m = len(matches)
        app_p = [matches[i][0] if i in matches else 0.0 for i in range(n)]
        sem_p = [matches[i][1] if i in matches else 0.0 for i in range(n)]
        appearance = float(np.mean([v[0] for v in matches.values()])) if matches else 0.0
        semantic = float(np.mean([v[1] for v in matches.values()])) if matches else 0.0
        res.update(num_detected_candidates=total, num_object_matches=m, object_existence_rate=round(m / max(n, 1), 6),
                   object_appearance_similarity=round(appearance, 6), object_semantic_similarity=round(semantic, 6),
                   object_appearance_persistence=round(float(np.mean(app_p)), 6),
                   object_semantic_persistence=round(float(np.mean(sem_p)), 6), object_recall=round(m / max(n, 1), 6),
                   object_identity_similarity=round(appearance, 6), object_consistency=round(float(np.mean(app_p)), 6))
        return res


def clean_concept(text):
    """object_identity.py:168-174."""
    text = text.lower(); text = re.sub(r'\([^)]*\)', ' ', text); text = re.sub(r'[^a-z0-9 /_-]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip(' .') or 'object'


def _clip01(v): return float(max(0.0, min(1.0, v)))


def load_sam2(dev):
    """object_identity.py:98-118 (config loaded directly, R2M's mask-decoder overrides) -> SAM2ImagePredictor."""
    def build():
        if str(TP / 'sam2') not in sys.path: sys.path.insert(0, str(TP / 'sam2'))
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import _load_checkpoint
        _need(SAM2_CKPT)
        cfg = OmegaConf.load(str(SAM2_CFG))
        OmegaConf.update(cfg, 'model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability', True, merge=True)
        OmegaConf.update(cfg, 'model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta', 0.05, merge=True)
        OmegaConf.update(cfg, 'model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh', 0.98, merge=True)
        OmegaConf.resolve(cfg)
        m = instantiate(cfg.model, _recursive_=True); _load_checkpoint(m, str(SAM2_CKPT))
        return SAM2ImagePredictor(m.to(dev).eval())
    return _model('sam2', build)


def get_grounded_sam(**kw):
    key = 'gsam_' + hashlib.md5(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:8]
    return _model(key, lambda: GroundedSam(device(), get_dino(), get_clip(), **kw))


class DreamSim:
    def __init__(self, dev):
        import torch
        from dreamsim import dreamsim
        DREAMSIM_DIR.mkdir(parents=True, exist_ok=True)
        if not ALLOW_DOWNLOAD: _need(DREAMSIM_DIR / 'ensemble_lora' / 'adapter_config.json')
        busy = [p.name for p in DREAMSIM_DIR.rglob('*.part')]
        if busy: raise RuntimeError(f'DreamSim weights are being downloaded by another process ({busy[:3]}); retry later')
        self.torch = torch; self.dev = dev
        hub = torch.hub.get_dir()                              # dreamsim calls torch.hub.set_dir(cache_dir) globally
        try:
            self.model, self.preprocess = dreamsim(pretrained=True, device=dev, cache_dir=str(DREAMSIM_DIR))
        finally:
            torch.hub.set_dir(hub)

    def __call__(self, a, b):
        from PIL import Image
        with self.torch.no_grad():
            return float(self.model(self.preprocess(Image.fromarray(_to_uint8(a))).to(self.dev),
                                    self.preprocess(Image.fromarray(_to_uint8(b))).to(self.dev)).item())


def get_dreamsim():
    return _model('dreamsim', lambda: DreamSim(device()))


# ----------------------------------------------------------------------------------------------------------------------
# per-pair engine with caching
# ----------------------------------------------------------------------------------------------------------------------

def _frame_feature(ctx, kind, idx):
    store = ctx.cache.setdefault('r2m_feat', {})
    if (kind, idx) not in store:
        ex = {'dino': get_dino, 'boq': get_boq, 'mvpr': get_mvpr}[kind]()
        store[(kind, idx)] = ex.extract_feature(ctx.frames[idx]).detach().cpu()
    return store[(kind, idx)]


def _pair_values(ctx, group, a, b):
    """Compute (and cache) one metric group for frame pair (a, b); returns dict of R2M result keys."""
    store = ctx.cache.setdefault('r2m_pair', {})
    key = (group, int(a), int(b))
    if key in store: return store[key]
    fa, fb = ctx.frames[a], ctx.frames[b]
    if group == 'appearance':
        v = compute_psnr_ssim(fa, fb); v['lpips'] = round(get_lpips('alex')(fa, fb), 6)
    elif group in ('dino', 'boq', 'mvpr'):
        import torch
        sim = float(torch.nn.functional.cosine_similarity(_frame_feature(ctx, group, a), _frame_feature(ctx, group, b), dim=-1).item())
        v = {{'dino': 'dino_similarity', 'boq': 'boq_similarity', 'mvpr': 'mvpr_similarity'}[group]: round(sim, 6)}
    elif group == 'geometric':
        kp = get_matcher().match(fa, fb)
        v = {k: kp[k] for k in ('num_keypoints_a', 'num_keypoints_b', 'num_matches', 'match_ratio')}
        v.update(compute_two_view_geometry(kp['matched_kp_a'], kp['matched_kp_b']))
    elif group == 'object':
        v = get_grounded_sam().score_pair(fa, fb, int(a), int(b), cache=ctx.cache.setdefault('r2m_obj_frames', {}))
    else:
        raise KeyError(group)
    store[key] = v
    return v


REVISIT_SETS = ('keypoint_revisit', 'revisit')     # headline first: our 6 keypoint revisits, then R2M-mined revisits


def _pair_sets(ctx, max_pairs_per_type):
    """revisit (R2M-mined), keypoint_revisit (our protocol: visit/return of each keypoint), baseline, short."""
    if not ctx.pairs: raise NotApplicable('case has no R2M pairs')
    T = len(ctx.frames)
    rev, base, short = P.select_for_video(ctx.pairs, T, max_pairs_per_type)
    kp = P.select_for_video({'revisit': ctx.pairs.get('keypoint_revisit', [])}, T, max_pairs_per_type)[0]
    if not kp and ctx.keypoints:
        kp = [(int(k['visit']), int(k['return'])) for k in ctx.keypoints if int(k['return']) < T]
    if not rev and not kp: raise NotApplicable('no revisit pairs inside the rendered frame range')
    return {'keypoint_revisit': kp, 'revisit': rev, 'baseline': base, 'short': short}


def metric_sets(ctx, group, metric, max_pairs_per_type=100):
    sets = _pair_sets(ctx, max_pairs_per_type); out = {}; items = []
    for name, prs in sets.items():
        vals = []
        for a, b in prs:
            v = _pair_values(ctx, group, a, b)
            if metric in v: vals.append(float(v[metric])); items.append({'set': name, 'pair': [a, b], metric: v[metric]})
        out[name] = vals
    return out, items


def memory_result(op_name, vals, metric, items, **extra):
    """MG / NMR for both revisit sets against the shared baseline / short sets. value = keypoint-revisit NMR (our
    headline) when defined, else the R2M-mined NMR. raw[<set>] holds each block."""
    blocks = {k: P.memory_block(vals, metric, rev_key=k) for k in REVISIT_SETS}
    head = next((k for k in REVISIT_SETS if blocks[k].get('nmr_valid')), None)
    raw = {'headline_set': head, **blocks, **extra}
    for k in REVISIT_SETS:                                   # flat copies for quick reading
        raw[f'nmr_{k}'] = blocks[k].get('nmr'); raw[f'mg_{k}'] = blocks[k].get('mg')
    raw['dr'] = next((blocks[k].get('dr') for k in REVISIT_SETS if blocks[k].get('dr') is not None), None)
    if head is None:
        return Result(op_name, None, status='n/a', unit='NMR', raw=raw, per_item=items,
                      note='NMR undefined for both revisit sets: dynamic range (short - baseline) <= eps or a pair set is empty')
    return Result(op_name, blocks[head]['nmr'], unit='NMR', raw=raw, per_item=items,
                  note='' if head == 'keypoint_revisit' else 'keypoint-revisit NMR undefined; value is the R2M-mined NMR')


def _metric_result(op_name, ctx, group, metric, max_pairs_per_type):
    vals, items = metric_sets(ctx, group, metric, max_pairs_per_type)
    return memory_result(op_name, vals, metric, items)


_R2M_METRICS = [
    # operator name,                            family,                  group,       R2M key
    ('r2m.appearance.psnr',                     'r2m.appearance',        'appearance', 'psnr'),
    ('r2m.appearance.ssim',                     'r2m.appearance',        'appearance', 'ssim'),
    ('r2m.appearance.lpips',                    'r2m.appearance',        'appearance', 'lpips'),
    ('r2m.scene_identity.dinov2',               'r2m.scene_identity',    'dino',       'dino_similarity'),
    ('r2m.scene_identity.boq',                  'r2m.scene_identity',    'boq',        'boq_similarity'),
    ('r2m.scene_identity.mutualvpr',            'r2m.scene_identity',    'mvpr',       'mvpr_similarity'),
    ('r2m.geometry.match_ratio',                'r2m.geometry',          'geometric',  'match_ratio'),
    ('r2m.geometry.inlier_ratio',               'r2m.geometry',          'geometric',  'ransac_inlier_ratio'),
    ('r2m.object_identity.appearance',          'r2m.object_identity',   'object',     'object_appearance_persistence'),
    ('r2m.object_identity.semantic',            'r2m.object_identity',   'object',     'object_semantic_persistence'),
    ('r2m.object_identity.existence',           'r2m.object_identity',   'object',     'object_existence_rate'),
    ('r2m.object_identity.appearance_similarity', 'r2m.object_identity', 'object',     'object_appearance_similarity'),
    ('r2m.object_identity.semantic_similarity', 'r2m.object_identity',   'object',     'object_semantic_similarity'),
]
_REQ = {'appearance': ('skimage', 'lpips'), 'dino': ('dinov2',), 'boq': ('dinov2', 'boq'), 'mvpr': ('MutualVPR',),
        'geometric': ('lightglue', 'cv2'), 'object': ('groundingdino', 'sam2', 'dinov2', 'clip')}


def _register(op_name, family, group, metric):
    @operator(op_name, family, inputs=('frames', 'pairs'), higher_is_better=True, unit='NMR', source=SRC, requires=_REQ[group], gpu=True)
    def _op(ctx, max_pairs_per_type=100):
        return _metric_result(op_name, ctx, group, metric, max_pairs_per_type)
    _op.__name__ = op_name.replace('.', '_'); _op.__doc__ = f'R2M {metric} on revisit/baseline/short pairs -> NMR.'
    return _op


for _n, _f, _g, _m in _R2M_METRICS:
    globals()[_n.replace('.', '_')] = _register(_n, _f, _g, _m)


# ----------------------------------------------------------------------------------------------------------------------
# persistent state (VLM rubric)
# ----------------------------------------------------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = """You are an expert evaluator of Video World Models.

Task: Assign a pairwise persistent-state consistency score to two images sampled from the same generated video.

* Image A is earlier and Image B is later.
* Do NOT assume that the images depict the same place.
* This is not raw pixel similarity: viewpoint, occlusion, and mild rendering changes may differ without implying a state inconsistency.
* Score whether the images provide evidence of the same local scene and, when they do, whether the shared scene content remains persistent.

Procedure:

1. Classify the scene relation as same, partial, different, or uncertain.
2. If shared scene content exists, estimate the viewpoint change and identify the co-visible region.
3. Judge identity, geometry, and persistence only where content can be compared, discounting differences explained by viewpoint or occlusion.
4. If there is no recognizable shared scene content, assign low consistency; absence of comparable content is not evidence of perfect consistency.

Viewpoint exemptions for same or partially overlapping scenes — do NOT penalize:
* Any difference plausibly caused by camera motion, perspective, scale, or occlusion.
* Objects entering or leaving the frame due to the viewpoint shift.
* Minor lighting, exposure, or rendering-noise differences.
* When uncertain whether a difference is from viewpoint or a true change, assume viewpoint.

Inconsistency types (report only confident findings):
* scene: images depict different local scenes or share no recognizable place cues.
* identity: same object changed color, shape, material, texture, or category.
* geometry: impossible spatial layout or structural contradictions.
* persistence: object disappeared or appeared well inside the shared region, or count of repeated elements changed without occlusion explanation.

Scoring anchors:
* 5.0 = strong evidence of the same local scene; shared state is consistent.
* 4.0 = same local scene with only minor confident inconsistencies.
* 3.0 = partial or uncertain scene overlap, or moderate inconsistencies.
* 2.0 = weak shared-scene evidence or major state inconsistencies.
* 1.0 = clearly different local scenes or fundamentally incompatible states.
Intermediate values are allowed. Do not give a high score solely because the images are individually plausible or because no content is comparable.

Output constraints — keep SHORT:
* scene_relation is one of: same, partial, different, uncertain
* viewpoint_difference ≤ 15 words
* at most 3 inconsistencies, each description ≤ 12 words
* reason ≤ 15 words
* Each inconsistency has ONLY keys: type, severity, description.

Output ONLY this JSON (no markdown):
{"scene_relation":"same|partial|different|uncertain","viewpoint_difference":"...","inconsistencies":[{"type":"scene|identity|geometry|persistence","severity":"minor|moderate|major|catastrophic","description":"..."}],"score":0.0,"reason":"..."}
"""
EVAL_USER_PROMPT = 'Evaluate the pairwise persistent-state consistency of Image A and Image B. Do not assume their pair type. Output strict JSON only.'


def parse_state_score(parsed):
    """persistent_state.py:65-108 clamp to [1,5]; -1 when unparseable."""
    try: return max(1.0, min(5.0, float(parsed.get('score'))))
    except Exception: return -1.0


def vlm_pair_score(img_a, img_b, tag=''):
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])) if str(Path(__file__).resolve().parents[2]) not in sys.path else None
    from annotation_pipeline.vlm import VLM
    VLM_CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(_to_uint8(img_a).tobytes() + b'|' + _to_uint8(img_b).tobytes()).hexdigest()[:24]
    with tempfile.TemporaryDirectory(dir=str(TMP / 'tmp')) as td:
        pa, pb = Path(td) / f'A_{h}.png', Path(td) / f'B_{h}.png'
        cv2.imwrite(str(pa), _to_uint8(img_a)[:, :, ::-1]); cv2.imwrite(str(pb), _to_uint8(img_b)[:, :, ::-1])
        prompt = EVAL_SYSTEM_PROMPT + '\n\n' + EVAL_USER_PROMPT + '\nThe first attached image is Image A, the second is Image B.'
        parsed = VLM(model=GPT_MODEL, route=GPT_MODEL).call(prompt, [pa, pb], VLM_CACHE / f'{h}.json', max_tokens=800)
    return {'score': parse_state_score(parsed), 'parsed': parsed, 'cache': str(VLM_CACHE / f'{h}.json')}


@operator('r2m.persistent_state.vlm', 'r2m.persistent_state', inputs=('frames', 'pairs'), higher_is_better=True, unit='NMR',
          source=SRC + ' src/metrics/persistent_state.py', requires=('gateway',), vlm=True)
def persistent_state_vlm(ctx, max_pairs_per_type=10, workers=3):
    """R2M persistent-state rubric (1..5) on revisit/baseline/short pairs, judged by gpt-6.1-sol -> NMR of state_score."""
    sets = _pair_sets(ctx, max_pairs_per_type)
    store = ctx.cache.setdefault('r2m_state', {})
    todo = sorted({(a, b) for prs in sets.values() for a, b in prs if (a, b) not in store})
    (TMP / 'tmp').mkdir(parents=True, exist_ok=True)

    def job(p):
        for _ in range(4):                                   # persistent_state.py:153 retries unparseable responses
            try:
                r = vlm_pair_score(ctx.frames[p[0]], ctx.frames[p[1]])
                if r['score'] > 0: return p, r
            except Exception as e:
                r = {'score': -1, 'error': f'{type(e).__name__}: {e}'[:300]}
        return p, r
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for p, r in ex.map(job, todo): store[p] = r
    vals = {k: [store[p]['score'] for p in prs if store[p]['score'] > 0] for k, prs in sets.items()}
    items = [{'set': k, 'pair': list(p), 'score': store[p]['score'], 'relation': (store[p].get('parsed') or {}).get('scene_relation')}
             for k, prs in sets.items() for p in prs]
    return memory_result('r2m.persistent_state.vlm', vals, 'state_score', items, judge=GPT_MODEL)


@operator('r2m.summary.nmr', 'r2m.summary', inputs=('frames', 'pairs'), higher_is_better=True, unit='NMR',
          source=SRC + ' src/compute_nmr_summary.py:270', gpu=True, vlm=True)
def summary_nmr(ctx, max_pairs_per_type=100, vlm_pairs_per_type=10, include_vlm=True):
    """Overall NMR = mean over the 5 R2M families of the mean metric NMR (unavailable families are skipped, as in R2M),
    computed separately for the keypoint-revisit set (headline value) and the R2M-mined revisit set."""
    key_of = {m: n for n, _, _, m in _R2M_METRICS}
    metric_nmr = {k: {} for k in REVISIT_SETS}; detail = {}
    for metric in P.METRICS:
        if metric == 'state_score':
            if not include_vlm: continue
            r = run_op('r2m.persistent_state.vlm', ctx, max_pairs_per_type=vlm_pairs_per_type)
        else:
            r = run_op(key_of[metric], ctx, max_pairs_per_type=max_pairs_per_type)
        detail[metric] = {'status': r.status, 'note': r.note, 'dr': r.raw.get('dr'),
                          **{f'{x}_{k}': r.raw.get(f'{x}_{k}') for k in REVISIT_SETS for x in ('nmr', 'mg')}}
        for k in REVISIT_SETS:
            v = r.raw.get(f'nmr_{k}') if r.status in ('ok', 'n/a') else None
            if v is not None and r.raw.get(k, {}).get('nmr_valid'): metric_nmr[k][metric] = v
    overall = {k: P.overall_nmr(metric_nmr[k]) for k in REVISIT_SETS}
    head = next((k for k in REVISIT_SETS if not math.isnan(overall[k][0])), None)
    if head is None: raise NotApplicable('no metric produced a valid NMR')
    raw = {'headline_set': head, 'metrics': detail}
    for k in REVISIT_SETS:
        raw[f'overall_{k}'] = None if math.isnan(overall[k][0]) else overall[k][0]
        raw[f'families_{k}'] = overall[k][1]
        raw[f'missing_families_{k}'] = [f for f in P.FAMILIES if f not in overall[k][1]]
    return Result('r2m.summary.nmr', overall[head][0], unit='NMR', raw=raw,
                  note='' if head == 'keypoint_revisit' else 'keypoint-revisit Overall NMR undefined; value is R2M-mined')
