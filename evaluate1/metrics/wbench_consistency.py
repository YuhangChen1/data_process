"""WBench consistency metrics C1-C8 as evaluate1 operators.

Mapping of WBench's case structure onto an evaluate1 case:
  * loop / round-trip (spatial_consistency.py): WBench compares frame 0 of a symmetric action sequence with the best
    return frame inside the last turn. evaluate1 has 6 keypoints, each = visit step, 3 actions, 3 inverse actions, so
    every keypoint is one loop: first frame = keypoint['visit'], last turn = its last inverse turn. Scores are averaged
    over keypoints (WBench averages over loop cases).
  * subject masks (SAM2 tracking from an initial mask, tools/run_sam2_track.py) exist only when the case names a
    subject: ctx.extra['subject_mask'] = path/array (HxW, bool) at ctx.extra.get('subject_mask_step', 0).
  * DA3 depth/pose cache (tools/run_da3_depth.py, 3 fps, DA3-GIANT-1.1) is shared through ctx.cache.
"""
from __future__ import annotations
from pathlib import Path
import os, sys, tempfile
import numpy as np

from .base import operator, Result, NotApplicable, TMP, device, cached
from .wbench_quality import (cached_try, WB, WB_TP, REF, sample_indices, pil, wb_score, free_gpu, hub_dir, syspath)

TAU = 0.15                     # spatial_consistency.py:25
N_SAMPLE_FRAMES = 10           # spatial_consistency.py:26


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


# ───────────────────────────── DreamSim / C1 / C2 ─────────────────────────────

def _dreamsim(ctx):
    def load():
        hub_dir()
        from dreamsim import dreamsim
        m, pre = dreamsim(pretrained=True, cache_dir=str(WB / 'dreamsim'), device=device())
        return m.to(device()).eval(), pre
    return cached(ctx, 'wbench_dreamsim', load)


def _loop_results(ctx):
    """spatial_consistency.py:77-171 evaluated once per keypoint loop; cached for C1 and C2."""
    def compute():
        import torch
        if not ctx.keypoints: raise NotApplicable('case has no revisit keypoints (WBench loop cases)')
        from .wbench_interaction import estimated_poses
        poses, src, stride = estimated_poses(ctx, allow_commanded=True)
        model, pre = _dreamsim(ctx); out = []
        ten = lambda i: pre(pil(ctx.frames[i])).to(device())
        for k in ctx.keypoints:
            visit = int(k['visit']); turns = k.get('turns') or []
            last = turns[-1]['steps'] if turns else [visit, int(k['return'])]
            p0 = poses[visit // stride]; best_i, best_rot = int(last[0]), 999.0
            for pi in range(int(last[0]) // stride, min(int(last[1]) // stride + 1, len(poses))):
                r = rot_angle_deg(p0[:3, :3].T @ poses[pi][:3, :3])
                if r < best_rot: best_rot, best_i = r, pi * stride
            ret = min(best_i, len(ctx.frames) - 1)
            t0 = ten(visit); sims = []
            with torch.no_grad():
                for fi in np.linspace(visit, ret, N_SAMPLE_FRAMES, dtype=int):
                    sims.append(1.0 / (1.0 + float(model(t0, ten(int(fi))).cpu())))
                ret_dist = float(model(t0, ten(ret)).cpu())
            ret_sim = 1.0 / (1.0 + ret_dist); min_sim = min(sims)
            gate = min(1.0, (1.0 - min_sim) / TAU)
            out.append({'keypoint': k['id'], 'visit': visit, 'return_frame': ret, 'commanded_return': int(k['return']),
                        'ret_rot_deg': round(best_rot, 2), 'ret_sim': ret_sim, 'min_sim': min_sim, 'gate': gate,
                        'score': ret_sim * gate})
        return {'per_keypoint': out, 'pose_source': src}
    return cached_try(ctx, 'wbench_loops', compute)


@operator('wbench.C1_spatial_consistency', 'wbench.consistency', ('frames', 'keypoints'), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/consistency/spatial_consistency.py:77-171 + main.py:909-913 (ungated ret_sim)',
          requires=('dreamsim', 'weights/wbench/dreamsim', 'pose estimator (see wbench_interaction.estimated_poses)'))
def spatial_consistency(ctx):
    """DreamSim similarity 1/(1+d) between the keypoint visit frame and the return frame (closest estimated rotation to
    the visit pose inside the last inverse turn); mean over keypoints."""
    r = _loop_results(ctx)
    v = float(np.mean([x['ret_sim'] for x in r['per_keypoint']]))
    return wb_score(v, {'pose_source': r['pose_source']}, r['per_keypoint'],
                    note='' if r['pose_source'].startswith('megasam') else f"return frame chosen with {r['pose_source']} poses")


@operator('wbench.C2_gated_spatial_consistency', 'wbench.consistency', ('frames', 'keypoints'), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/consistency/spatial_consistency.py:67-74,136-164 (score = ret_sim * min(1,(1-min_sim)/0.15))',
          requires=('dreamsim',))
def gated_spatial_consistency(ctx):
    """ret_sim * gate, gate = min(1, (1 - min over 10 visit->return samples of DreamSim sim) / tau=0.15); mean over keypoints."""
    r = _loop_results(ctx)
    v = float(np.mean([x['score'] for x in r['per_keypoint']]))
    return wb_score(v, {'pose_source': r['pose_source'], 'tau': TAU}, r['per_keypoint'])


# ───────────────────────────── C3 segment continuity ─────────────────────────────

def _transnet():
    import torch
    root = WB_TP / 'transnetv2_pytorch'
    with syspath(root):
        from transnetv2_pytorch.model import TransNetV2
    w = WB / 'transnetv2' / 'transnetv2-pytorch-weights.pth'
    if not w.exists(): w = root / 'transnetv2_pytorch' / 'transnetv2-pytorch-weights.pth'
    m = TransNetV2(); m.load_state_dict(torch.load(str(w), map_location='cpu'))
    return m.eval().to(device()), str(w)


@operator('wbench.C3_segment_continuity', 'wbench.consistency', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/consistency/segment_continuity.py:237-326 (TransNetV2, thr 0.5, min scene len 10)',
          requires=('third_party/wbench/transnetv2_pytorch',))
def segment_continuity(ctx, threshold=0.5, min_scene_len=10):
    """1 if TransNetV2 finds no hard cut (p > 0.5, cuts closer than 10 frames merged) on 48x27 frames, else 0."""
    import torch, cv2
    model, wpath = _transnet()
    frames = np.stack([cv2.resize(f, (48, 27)) for f in ctx.frames]).astype(np.uint8)
    total = len(frames)
    padded = np.concatenate((np.repeat(frames[:1], 25, 0), frames, np.repeat(frames[-1:], 75 - total % 50, 0)), 0)
    preds = []
    with torch.no_grad():
        for s in range(0, len(padded) - 99, 50):
            logits = model(torch.from_numpy(padded[s:s + 100]).unsqueeze(0).to(device()))
            if isinstance(logits, tuple): logits = logits[0]
            preds.append(torch.sigmoid(logits[0, 25:75, 0]).cpu().numpy())
    p = np.concatenate(preds)[:total]; raw_cuts = np.where(p > threshold)[0].tolist(); cuts = raw_cuts[:1]
    for c in raw_cuts[1:]:
        if c - cuts[-1] >= min_scene_len: cuts.append(c)
    del model; free_gpu()
    return wb_score(0.0 if cuts else 1.0, {'n_cuts': len(cuts), 'cut_frames': cuts, 'max_prob': float(p.max()),
                                          'threshold': threshold, 'weights': wpath})


# ───────────────────────────── SAM2 subject tracking (C4, C7) ─────────────────────────────

def _subject_masks(ctx, target_fps=5.0):
    """tools/run_sam2_track.py: SAM2.1-hiera-b+ video predictor at 5 fps, seeded with the case's initial subject mask."""
    def compute():
        import torch, cv2
        from PIL import Image
        m0 = ctx.extra.get('subject_mask')
        if m0 is None: raise NotApplicable('static scene: no subject mask in the case (ctx.extra["subject_mask"])')
        m0 = (np.array(Image.open(m0).convert('L')) > 127) if isinstance(m0, (str, Path)) else np.asarray(m0) > 0
        start = int(ctx.extra.get('subject_mask_step', 0))
        n = len(ctx.frames) - start; step = ctx.fps / target_fps if 0 < target_fps < ctx.fps else 1
        idx = sorted({start + int(round(i * step)) for i in range(int(n / step)) if int(round(i * step)) < n})
        with syspath(WB_TP / 'sam2'):
            from sam2.build_sam import build_sam2_video_predictor
            pred = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml',
                                              str(WB / 'sam2.1-hiera-base-plus' / 'sam2.1_hiera_base_plus.pt'), device=device())
        H, W = ctx.frames.shape[1:3]
        if m0.shape != (H, W): m0 = cv2.resize(m0.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        masks = {}
        with tempfile.TemporaryDirectory(prefix='sam2_frames_', dir=str(TMP / 'tmp')) as d:
            for k, i in enumerate(idx): pil(ctx.frames[i]).save(os.path.join(d, f'{k:05d}.jpg'))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                state = pred.init_state(video_path=d)
                pred.add_new_mask(inference_state=state, frame_idx=0, obj_id=1, mask=m0.astype(np.uint8))
                for fi, _, logits in pred.propagate_in_video(state):
                    masks[idx[fi]] = (logits[0] > 0.0).float().cpu().numpy().squeeze().astype(bool)
                pred.reset_state(state)
        del pred; free_gpu()
        return {i: masks.get(i, np.zeros((H, W), bool)) for i in idx}
    return cached_try(ctx, 'wbench_subject_masks', compute)


@operator('wbench.C4_perspective_consistency', 'wbench.consistency', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/consistency/perspective_consistency.py:30-106 (SAM2 centroid stability; DA3 depth reported only)',
          requires=('third_party/wbench/sam2', 'weights/wbench/sam2.1-hiera-base-plus'))
def perspective_consistency(ctx):
    """max(0, 1 - std(normalised mask centroid)/0.3) * presence over SAM2-tracked subject masks (area >= 10 px)."""
    masks = _subject_masks(ctx); cx, cy, ds = [], [], []
    depth = None
    try: depth = _da3(ctx)
    except Exception: pass
    for fid, b in sorted(masks.items()):
        if b.sum() < 10: continue
        ys, xs = np.where(b); cx.append(xs.mean() / b.shape[1]); cy.append(ys.mean() / b.shape[0])
        if depth is not None:
            di = min(int(np.searchsorted(depth['index'], fid)), len(depth['index']) - 1); d = depth['depth'][di]
            import cv2
            br = cv2.resize(b.astype(np.uint8), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            if br.any(): ds.append(float(d[br].mean()))
    total = len(masks); valid = len(cx); presence = valid / max(total, 1)
    if valid < 2: return wb_score(0.0, {'presence': presence, 'valid_frames': valid, 'total_frames': total})
    c_std = float(np.sqrt(np.std(cx) ** 2 + np.std(cy) ** 2)); cs = max(0.0, 1.0 - c_std / 0.3) * presence
    raw = {'presence': presence, 'valid_frames': valid, 'total_frames': total, 'centroid_std': c_std, 'centroid_stability': cs}
    if len(ds) > 1:
        d_cv = float(np.std(ds)) / (float(np.mean(ds)) + 1e-6); raw.update(depth_cv=d_cv, depth_stability=max(0.0, 1.0 - d_cv / 0.5) * presence)
    return wb_score(cs, raw)


def _dinov2_b14():
    import torch
    hub_dir()
    local = WB / 'torch_hub' / 'facebookresearch_dinov2_main'
    if local.exists():
        return torch.hub.load(str(local), 'dinov2_vitb14', source='local', pretrained=True).to(device()).eval()
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True).to(device()).eval()


def _hf_clip_b16():
    from transformers import CLIPModel, CLIPProcessor
    local = WB / 'clip-vit-base-patch16'
    has = (local / 'model.safetensors').exists() or (local / 'pytorch_model.bin').exists()
    name = str(local) if has else 'openai/clip-vit-base-patch16'
    return CLIPModel.from_pretrained(name).to(device()).eval(), CLIPProcessor.from_pretrained(name)


def masked_crop(frame, mask, padding=10):
    """subject_consistency.py:187-201: gray background outside the mask, crop to the mask box + 10 px."""
    from PIL import Image
    f = np.array(frame)
    if mask.shape != f.shape[:2]:
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((f.shape[1], f.shape[0]), Image.NEAREST)) > 127
    out = np.full_like(f, 128); out[mask] = f[mask]; ys, xs = np.where(mask)
    if len(ys) == 0: return frame
    y1, y2 = max(0, ys.min() - padding), min(f.shape[0], ys.max() + padding)
    x1, x2 = max(0, xs.min() - padding), min(f.shape[1], xs.max() + padding)
    return Image.fromarray(out[y1:y2, x1:x2])


@operator('wbench.C7_subject_consistency', 'wbench.consistency', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/consistency/subject_consistency.py:133-237 + evaluate.py:138-182',
          requires=('third_party/wbench/sam2', 'weights/wbench/torch_hub (DINOv2 ViT-B/14)', 'weights/wbench/clip-vit-base-patch16'))
def subject_consistency(ctx, batch_size=16):
    """mean_i (max(0,cos DINOv2(crop_i-1, crop_i)) + max(0,cos CLIP(crop_0, crop_i)))/2 over SAM2 subject crops (5 fps)."""
    import torch, torch.nn.functional as F, torchvision.transforms as T
    masks = _subject_masks(ctx); ids = sorted(masks)
    if len(ids) < 2: return wb_score(1.0, {'skipped': True})
    crops = [masked_crop(pil(ctx.frames[i]), masks[i]) if masks[i].sum() >= 10 else pil(ctx.frames[i]) for i in ids]
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dino = _dinov2_b14(); clipm, proc = _hf_clip_b16(); dfe, cfe = [], []
    with torch.no_grad():
        for s in range(0, len(crops), batch_size):
            b = crops[s:s + batch_size]
            dfe.append(F.normalize(dino(torch.stack([tf(c) for c in b]).to(device())), dim=-1))
            cfe.append(F.normalize(clipm.get_image_features(**proc(images=b, return_tensors='pt').to(device())), dim=-1))
    d, c = torch.cat(dfe), torch.cat(cfe)
    adj = [max(0.0, F.cosine_similarity(d[i - 1:i], d[i:i + 1]).item()) for i in range(1, len(d))]
    fst = [max(0.0, F.cosine_similarity(c[0:1], c[i:i + 1]).item()) for i in range(1, len(c))]
    del dino, clipm; free_gpu()
    return wb_score(float(np.mean([(a + b) / 2 for a, b in zip(adj, fst)])),
                    {'dinov2_adj_mean': float(np.mean(adj)), 'clip_first_mean': float(np.mean(fst)), 'frames': len(ids)})


# ───────────────────────────── DA3 reconstruction (C5, C6) ─────────────────────────────

def _da3_stubs():
    """src/compat.py: stub DA3's optional heavy deps that are not needed for depth/pose inference."""
    import types, importlib.machinery
    def stub(name):
        m = types.ModuleType(name); m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None); sys.modules[name] = m; return m
    for name in ['moviepy', 'moviepy.editor']:
        if name not in sys.modules:
            try: __import__(name)
            except Exception: stub(name)


DA3_SMALL_LOCAL = Path('/mnt/cfs/cyh/any2videobench/.runtime/weights/DA3-SMALL')


def da3_model_dir():
    """WBench uses DA3-GIANT-1.1. If that 5.4 GB checkpoint is not (yet) complete, fall back to the local DA3-SMALL
    copy so C4/C5/C6/I1 still run; the substitution is reported in raw['da3_model'] and the operator note."""
    g = WB / 'DA3-GIANT-1.1'
    if (g / 'model.safetensors').exists(): return g, 'DA3-GIANT-1.1'
    if (DA3_SMALL_LOCAL / 'model.safetensors').exists(): return DA3_SMALL_LOCAL, 'DA3-SMALL (substitute: GIANT weights missing)'
    raise FileNotFoundError(f'DA3 weights missing: {g}')


def _da3(ctx, fps=3.0):
    """tools/run_da3_depth.py process_single: DA3-GIANT-1.1 on every round(video_fps/3)-th frame (all views jointly)."""
    def compute():
        import torch
        idx = list(range(0, len(ctx.frames), 1 if fps >= ctx.fps else max(1, int(round(ctx.fps / fps)))))
        if len(idx) < 3: raise NotApplicable('DA3 needs >= 3 frames')
        mdir, mname = da3_model_dir()
        _da3_stubs()
        with syspath(WB_TP / 'depth-anything-3' / 'src'):
            from depth_anything_3.api import DepthAnything3
            model = DepthAnything3.from_pretrained(str(mdir)).to(device=device())
            with tempfile.TemporaryDirectory(prefix='da3_', dir=str(TMP / 'tmp')) as d:
                paths = []
                for k, i in enumerate(idx):
                    p = os.path.join(d, f'{k:04d}.png'); pil(ctx.frames[i]).save(p); paths.append(p)
                pred = model.inference(paths)
        out = {'index': np.array(idx), 'depth': np.asarray(pred.depth, np.float32), 'extrinsics': np.asarray(pred.extrinsics),
               'intrinsics': np.asarray(pred.intrinsics), 'fps': fps, 'model': mname}
        if getattr(pred, 'conf', None) is not None: out['conf'] = np.asarray(pred.conf, np.float32)
        del model, pred; free_gpu()
        return out
    return cached_try(ctx, f'wbench_da3_{fps}', compute)


def _reconstruction(ctx, max_pts_per_frame=5000):
    """reconstruction_consistency.py:130-269 (seeded randperm for determinism)."""
    def compute():
        import torch, cv2
        da = _da3(ctx); depth, ext, intr = da['depth'], da['extrinsics'], da['intrinsics']
        N, H, W = depth.shape
        frames = np.stack([cv2.resize(ctx.frames[i], (W, H)) for i in da['index']]).astype(np.uint8)
        dev = device(); g = torch.Generator(device=dev).manual_seed(0)
        pts, cols, sids = [], [], []
        for i in range(N):
            d = torch.from_numpy(depth[i]).float().to(dev); K = torch.from_numpy(intr[i]).double().to(dev)
            E = torch.from_numpy(ext[i]).double().to(dev); R, t = E[:3, :3], E[:3, 3]
            mask = d > 1e-6
            if mask.sum() == 0: continue
            ys, xs = torch.where(mask); dsv = d[mask].double()
            if len(dsv) > max_pts_per_frame:
                k = torch.randperm(len(dsv), device=dev, generator=g)[:max_pts_per_frame]; xs, ys, dsv = xs[k], ys[k], dsv[k]
            ph = torch.stack([xs.double(), ys.double(), torch.ones_like(xs, dtype=torch.float64)], 0)
            world = R.T @ (dsv.unsqueeze(0) * (torch.linalg.inv(K) @ ph) - t.unsqueeze(1))
            pts.append(world.T.float()); cols.append(torch.from_numpy(frames[i]).float().to(dev)[ys.long(), xs.long()])
            sids.append(torch.full((world.shape[1],), i, dtype=torch.long, device=dev))
        pts, cols, sids = torch.cat(pts).double(), torch.cat(cols), torch.cat(sids)
        depth_t = torch.from_numpy(depth).float().to(dev); frames_t = torch.from_numpy(frames).float().to(dev)
        geo, psnr = [], []
        for i in range(N):
            other = sids != i; po, co = pts[other], cols[other]
            E = torch.from_numpy(ext[i]).double().to(dev); K = torch.from_numpy(intr[i]).double().to(dev)
            cam = E[:3, :3] @ po.T + E[:3, 3].unsqueeze(1); front = cam[2] > 1e-6; cam, co = cam[:, front], co[front]
            if cam.shape[1] == 0: geo.append(float('inf')); psnr.append(0.0); continue
            pr = K @ cam; u = (pr[0] / pr[2]).float(); v = (pr[1] / pr[2]).float(); z = cam[2].float()
            ok = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1); u, v, z, cv = u[ok], v[ok], z[ok], co[ok]
            if u.shape[0] == 0: geo.append(float('inf')); psnr.append(0.0); continue
            ui, vi = u.round().long().clamp(0, W - 1), v.round().long().clamp(0, H - 1)
            dp = depth_t[i, vi, ui]; dv = dp > 1e-6
            geo.append(float(((z[dv] - dp[dv]).abs() / dp[dv]).median()) if dv.sum() > 0 else float('inf'))
            order = z.argsort(descending=True); rend = torch.zeros(H, W, 3, device=dev); zb = torch.full((H, W), float('inf'), device=dev)
            rend[vi[order], ui[order]] = cv[order].float(); zb[vi[order], ui[order]] = z[order]
            has = zb < float('inf')
            if has.sum() < 100: psnr.append(0.0); continue
            mse = float((((rend[has] - frames_t[i][has]) / 255.0) ** 2).mean()); psnr.append(-10 * np.log10(mse + 1e-10))
        g_arr = np.array(geo); vg = g_arr[np.isfinite(g_arr)]; p_arr = np.array(psnr); vp = p_arr[p_arr > 0]
        del pts, cols, depth_t, frames_t; free_gpu()
        return {'geometric_consistency': 1.0 / (1.0 + float(np.mean(vg))) if len(vg) else 0.0,
                'geo_mean_rel_error': float(np.mean(vg)) if len(vg) else None, 'geo_valid_frames': int(len(vg)),
                'photo_mean_psnr': float(np.mean(vp)) if len(vp) else 0.0, 'photo_valid_frames': int(len(vp)),
                'num_frames': int(N), 'per_frame_rel_err': [None if not np.isfinite(x) else float(x) for x in geo],
                'per_frame_psnr': [float(x) for x in psnr], 'steps': da['index'].tolist(), 'da3_model': da['model']}
    return cached_try(ctx, 'wbench_reconstruction', compute)


@operator('wbench.C5_geometric_consistency', 'wbench.consistency', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/consistency/reconstruction_consistency.py:176-260 + tools/run_da3_depth.py (DA3-GIANT-1.1, 3 fps)',
          requires=('third_party/wbench/depth-anything-3', 'weights/wbench/DA3-GIANT-1.1'))
def geometric_consistency(ctx):
    """1/(1+mean_i median |z_reproj - d_i|/d_i): every frame's DA3 points reprojected into every other 3-fps frame."""
    r = _reconstruction(ctx)
    return wb_score(r['geometric_consistency'], {k: v for k, v in r.items() if k != 'per_frame_psnr'},
                    note='' if r['da3_model'].startswith('DA3-GIANT') else r['da3_model'])


@operator('wbench.C6_photometric_consistency', 'wbench.consistency', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' reconstruction_consistency.py:234-267 + main.py:899-901 (score = 1 - 10^(-PSNR/20))',
          requires=('third_party/wbench/depth-anything-3', 'weights/wbench/DA3-GIANT-1.1'))
def photometric_consistency(ctx):
    """1 - 10^(-PSNR/20), PSNR of z-buffered colour reprojection of all other frames' DA3 points (mean over frames)."""
    r = _reconstruction(ctx); p = r['photo_mean_psnr']
    return wb_score(1.0 - 10 ** (-p / 20.0), {'photometric_psnr': p, 'psnr_score_100_alt': min(100.0, max(0.0, (p - 10) / 15 * 100)),
                                              'valid_frames': r['photo_valid_frames'], 'per_frame_psnr': r['per_frame_psnr'],
                                              'da3_model': r['da3_model']}, note='' if r['da3_model'].startswith('DA3-GIANT') else r['da3_model'])


# ───────────────────────────── C8 background consistency ─────────────────────────────

@operator('wbench.C8_background_consistency', 'wbench.consistency', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/consistency/background_consistency.py:249-298 + evaluate.py:56-81 (CLIP ViT-B/32, 2 fps)',
          requires=('clip', 'weights/wbench/clip/ViT-B-32.pt'))
def background_consistency(ctx, batch_size=32):
    """VBench: mean_i (max(0,cos(f_i-1,f_i)) + max(0,cos(f_0,f_i)))/2 on CLIP ViT-B/32 features of 2-fps frames."""
    import torch, torch.nn.functional as F, torchvision.transforms as T, clip
    idx = sample_indices(len(ctx.frames), ctx.fps, {'fps': 2.0})
    if len(idx) < 2: return wb_score(1.0)
    model, _ = clip.load('ViT-B/32', device=device(), download_root=str(WB / 'clip')); model.eval()
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    feats = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = torch.stack([tf(pil(ctx.frames[i])) for i in idx[s:s + batch_size]]).to(device())
            feats.append(F.normalize(model.encode_image(b).float(), dim=-1, p=2))
    f = torch.cat(feats); sims = []
    for i in range(1, len(f)):
        sims.append((max(0.0, F.cosine_similarity(f[i - 1:i], f[i:i + 1]).item()) + max(0.0, F.cosine_similarity(f[0:1], f[i:i + 1]).item())) / 2)
    del model; free_gpu()
    return wb_score(float(np.mean(sims)), {'num_frames': len(idx)}, [{'step': i, 'sim': s} for i, s in zip(idx[1:], sims)])
