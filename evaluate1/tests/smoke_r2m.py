"""Smoke test of every R2M / revisit / fidelity operator on real frames of the pilot scene.

  ssh -p <ssh_port> root@<host> 'CUDA_VISIBLE_DEVICES=0 unshare -m bash .../tools/remote_exec.sh python3 evaluate1/tests/smoke_r2m.py'

Synthetic rollout from the hesai walkthrough: frames 0..59 forward, then back to frame 10 (R2M-style leave-and-return,
so revisit / baseline / short pairs exist), plus one appended "return" frame for a keypoint at step 30 that is frame 30
shifted by 6 px with noise. Model output = JPEG-75 copies with a slight brightness drift on the way back; input GT =
the clean frames. Objects = GroundingDINO+SAM2 masks on the visit frame. Depth = mesh depth from the annotation clip
when available. Estimated poses = commanded poses at scale 2 with noise.
"""
import json, math, os, shutil, sys, time
from pathlib import Path

import numpy as np
import cv2

DP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DP))
if '--prefetch' in sys.argv: os.environ['EVAL_ALLOW_DOWNLOAD'] = '1'
from evaluate1.metrics import base, pairs as P, r2m, revisit, fidelity  # noqa: E402


def prefetch(tries=6):
    """One-time, network-allowed fill of the weight store used by the R2M / revisit / fidelity operators (CPU only).
    Large checkpoints (DINOv2, BoQ, MutualVPR, CLIP, GroundingDINO, SAM2.1-L) are fetched by the download script and
    only verified here."""
    def retry(what, fn):
        for i in range(tries):
            try: fn(); print('ok  ', what, flush=True); return True
            except Exception as e: print(f'try {i + 1}/{tries} {what}: {type(e).__name__}: {str(e)[:160]}', flush=True); time.sleep(5)
        print('FAIL', what, flush=True); return False
    ck = r2m.TORCH_HUB / 'checkpoints'; ck.mkdir(parents=True, exist_ok=True)
    for f in ('superpoint_v1.pth', 'superpoint_lightglue_v0-1_arxiv.pth', 'alexnet-owt-7be5be79.pth', 'vgg16-397923af.pth'):
        src = base.TMP / 'torch' / 'hub' / 'checkpoints' / f
        if not (ck / f).exists() and src.exists(): shutil.copy2(src, ck / f)
    import torch
    retry('superpoint+lightglue', lambda: r2m.KeypointMatcher('cpu'))
    retry('lpips alex', lambda: r2m.LPIPSComputer('alex', 'cpu')); retry('lpips vgg', lambda: r2m.LPIPSComputer('vgg', 'cpu'))
    retry('bert-base-uncased config/tokenizer', r2m._bert_path)
    for n in ('niqe', 'brisque', 'dists'):
        retry('pyiqa ' + n, lambda n=n: fidelity._pyiqa(n))
    retry('dreamsim', lambda: r2m.DreamSim('cpu'))
    for f in (r2m.DINOV2_CKPT, r2m.DINOV2_REPO / 'hubconf.py', r2m.BOQ_CKPT, r2m.MVPR_CKPT, r2m.CLIP_CKPT, r2m.GDINO_CKPT, r2m.SAM2_CKPT):
        print('have' if Path(f).exists() else 'MISSING', f, flush=True)
    del torch

SCENE = DP / 'hesai_dataset/20261019/办公或教育空间/教育科研/KS-IN-OE-S-F1_01_202609121414_图书阅览室/manyou_videos/main_walkthrough'
DEPTH = DP / 'outputs/KS-IN-OE-S-F1_01_202609121414/hesai_walkthrough/main/mesh_depth_mm'
OUT = base.TMP / 'smoke_r2m'; OUT.mkdir(parents=True, exist_ok=True)


class LazyDepth:
    def __init__(self, src_index): self.src_index = src_index
    def __len__(self): return len(self.src_index)
    def __getitem__(self, i):
        p = DEPTH / f'{self.src_index[i]:05d}.png'
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32) / 1000.0


def main():
    t0 = time.time()
    poses = json.loads((SCENE / 'poses.json').read_text())['poses']
    seq = list(range(0, 60)) + list(range(58, 9, -1))
    visit = 30; T = len(seq) + 1
    gt_frames = {i: cv2.imread(str(SCENE / 'frames_hd' / f'{i:06d}.png'))[:, :, ::-1].copy() for i in sorted(set(seq))}
    frames = []
    for k, i in enumerate(seq):
        img = gt_frames[i]
        if k >= 60: img = np.clip(img.astype(np.float32) * (1 - 0.0015 * (k - 59)), 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode('.jpg', img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
        frames.append(cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1])
    ret = np.roll(frames[visit], 6, axis=1).astype(np.int16) + np.random.default_rng(0).integers(-4, 5, frames[visit].shape)
    frames.append(np.clip(ret, 0, 255).astype(np.uint8))
    frames = np.stack(frames)
    c2w = np.array([poses[i]['c2w'] for i in seq] + [poses[visit]['c2w']], float)
    H, W = frames.shape[1:3]; fy = (H / 2) / math.tan(math.radians(poses[0]['fov_y_deg']) / 2)
    K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1.]])
    mined = P.mine_pairs(c2w, up=(0., -1., 0.))           # scan frame: -Y is up
    print('pairs', {k: len(v) for k, v in mined.items() if k != 'meta'}, 'tau', mined['meta']['angle_tolerance_deg'],
          mined['meta']['translation_tolerance_m'], 'method', mined['meta']['detection']['method'], flush=True)
    rng = np.random.default_rng(1)
    est = c2w.copy(); est[:, :3, 3] = est[:, :3, 3] * 2.0 + rng.normal(0, 0.01, (T, 3))
    input_index = [int(x) for x in np.linspace(0, 59, 10).round()]
    depth = LazyDepth(seq + [visit]) if (DEPTH / f'{visit:05d}.png').exists() else None

    # fake annotated objects: GroundingDINO + SAM2 on the visit frame
    gs = r2m.get_grounded_sam()
    xyxy, logits, phrases = gs.detect_boxes(frames[visit], 'chair . bookshelf . table .')
    masks, _ = gs.sam_masks(frames[visit], xyxy[:3].cpu().numpy())
    objects = []
    for j, m in enumerate(masks):
        p = OUT / f'obj{j}.png'; cv2.imwrite(str(p), m.astype(np.uint8) * 255)
        label = r2m.clean_concept(phrases[j]).split(' ')[0] or 'object'
        objects.append({'global_id': j + 1, 'label': label, 'visible_at': {visit: str(p), T - 1: str(p)}})
    print('objects', [(o['label'], int(cv2.imread(o['visible_at'][visit], 0).astype(bool).sum())) for o in objects], flush=True)

    ctx = base.EvalContext(frames=frames, fps=12., K=K, c2w_cmd=c2w, c2w_est=est, depth=depth,
                           keypoints=[{'id': 'K1', 'visit': visit, 'return': T - 1, 'excursion': [visit + 1, T - 2]}],
                           pairs={k: [tuple(x) for x in mined[k]] for k in ('revisit', 'baseline', 'short')},
                           input_index=input_index, input_gt=np.stack([gt_frames[seq[i]] for i in input_index]), objects=objects)
    names = [n for n in base.REGISTRY if n.split('.')[0] in ('r2m', 'revisit', 'fidelity')]
    params = {n: {'max_pairs_per_type': 4} for n in names if n.startswith('r2m.')}
    params['r2m.persistent_state.vlm'] = {'max_pairs_per_type': 2}
    params['r2m.summary.nmr'] = {'max_pairs_per_type': 4, 'vlm_pairs_per_type': 2}
    results = []
    for n in names:
        r = base.run(n, ctx, **params.get(n, {}))
        results.append(r.as_dict())
        v = 'None' if r.value is None else f'{r.value:.4f}'
        print(f'{n:48s} {r.status:5s} {v:>10s} {r.seconds:8.2f}s  {r.note[:110]}', flush=True)
        if r.status == 'error': print('   ', r.raw.get('traceback', '')[-1200:].replace('\n', '\n    '), flush=True)
    (OUT / 'results.json').write_text(json.dumps({'results': results, 'pairs_meta': mined['meta'],
                                                  'gdino_ops': gs.ops}, indent=1, default=str))
    st = {s: sum(r['status'] == s for r in results) for s in ('ok', 'n/a', 'error')}
    print('SUMMARY', st, 'operators', len(results), f'{time.time() - t0:.0f}s', 'gdino ops:', gs.ops, flush=True)


if __name__ == '__main__':
    prefetch() if '--prefetch' in sys.argv else main()
