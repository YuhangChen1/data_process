"""Render a (4D) Gaussian model along an evaluate1 case trajectory.

  python -m evaluate1.render --case evaluate1/cases/<scene> --model oracle            # the scan's own 3DGS, static
  python -m evaluate1.render --case ... --model ply:/path/to/gaussians.ply --frame source|canonical

Output  evaluate1/runs/<scene>/<model>/frames/<step:05d>.png  (+ depth/<step:05d>.npy at input/visit/return steps,
expected camera-Z depth from the Gaussians) and render.json.

A real 4DGS method renders its own frames: for every record of trajectory.json use camera_to_source (or
camera_to_canonical, whichever frame the model lives in), K, width/height and time t, and write frames/<step:05d>.png.
The runner only needs that folder. The built-in `oracle` renders the scan's own static 3DGS: an upper-bound reference
(revisits are perfectly consistent); `ply:` renders any static Gaussian ply as a baseline.
"""
from pathlib import Path
import argparse, json, math, time
import numpy as np

DP = Path(__file__).resolve().parents[1]


def load_ply(path):
    from plyfile import PlyData
    v = PlyData.read(str(path), mmap=True)['vertex']; names = v.data.dtype.names
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)
    scales = np.exp(np.clip(np.stack([v[f'scale_{i}'] for i in range(3)], 1), -30, 20)).astype(np.float32)
    opacity = (1 / (1 + np.exp(-np.clip(v['opacity'], -40, 40)))).astype(np.float32)
    quats = np.stack([v[f'rot_{i}'] for i in range(4)], 1).astype(np.float32)
    dc = np.stack([v[f'f_dc_{i}'] for i in range(3)], 1).astype(np.float32)
    rest = sorted([n for n in names if n.startswith('f_rest_')], key=lambda n: int(n.split('_')[-1]))
    sh = None
    if rest:   # (N, 3*(K-1)) stored channel-major -> (N, K, 3)
        r = np.stack([v[n] for n in rest], 1).astype(np.float32).reshape(len(xyz), 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([dc[:, None], r], 1)
    keep = np.isfinite(xyz).all(1) & np.isfinite(scales).all(1) & (opacity > .005) & (scales.max(1) < 2.)
    lo, hi = np.quantile(xyz[keep], [.001, .999], axis=0); keep &= ((xyz >= lo - .75) & (xyz <= hi + .75)).all(1)
    g = {'means': xyz[keep], 'scales': scales[keep], 'opacities': opacity[keep], 'quats': quats[keep] / np.linalg.norm(quats[keep], axis=1, keepdims=True)}
    if sh is None: g['colors'] = np.clip(.5 + .28209479177387814 * dc[keep], 0, 1); g['sh_degree'] = None
    else: g['colors'] = sh[keep]; g['sh_degree'] = int(round(math.sqrt(sh.shape[1]))) - 1
    return g


class GaussianModel:
    """Static Gaussians; subclasses override at(t, step, case) to make them time dependent (4D)."""
    def __init__(self, g):
        import torch
        self.sh_degree = g.pop('sh_degree'); self.g = {k: torch.as_tensor(v, device='cuda') for k, v in g.items()}
    def at(self, t, step, case): return self.g


def render(model, c2w, K, w, h, t, step, case, want_depth=False):
    import torch, gsplat
    g = model.at(t, step, case)
    view = torch.as_tensor(np.linalg.inv(c2w)[None].astype(np.float32), device='cuda')
    Ks = torch.as_tensor(np.asarray(K, np.float32)[None], device='cuda')
    kw = dict(means=g['means'], quats=g['quats'], scales=g['scales'], opacities=g['opacities'], colors=g['colors'])
    if model.sh_degree is not None: kw['sh_degree'] = model.sh_degree
    img, alpha, _ = gsplat.rasterization(**kw, viewmats=view, Ks=Ks, width=w, height=h, near_plane=.03, far_plane=100.,
                                         packed=True, render_mode='RGB+ED' if want_depth else 'RGB', radius_clip=.4)
    img = img[0]; a = alpha[0, :, :, 0]
    rgb = torch.clamp((img[..., :3] + 1 - a[..., None]) * 255, 0, 255).byte().cpu().numpy()
    return rgb, (img[..., 3].float().cpu().numpy() if want_depth else None)


def main():
    import cv2
    ap = argparse.ArgumentParser(); ap.add_argument('--case', required=True); ap.add_argument('--model', default='oracle')
    ap.add_argument('--name'); ap.add_argument('--frame', default='source', choices=['source', 'canonical'])
    ap.add_argument('--depth', default='key', choices=['key', 'all', 'none'])
    a = ap.parse_args(); case_dir = Path(a.case).resolve(); case = json.loads((case_dir / 'case.json').read_text())
    traj = json.loads((case_dir / 'trajectory.json').read_text()); cams = np.load(case_dir / 'cameras.npz')
    traj['fps'] = case['fps']
    if a.model == 'oracle':
        model = GaussianModel(load_ply(DP / 'sources' / case['scene_id'] / 'gaussian.ply')); frame = 'source'; name = a.name or 'oracle'
    elif a.model.startswith('ply:'):
        model = GaussianModel(load_ply(a.model[4:])); frame = a.frame; name = a.name or Path(a.model[4:]).stem
    else: raise SystemExit('unknown model')
    out = DP / 'evaluate1' / 'runs' / case['scene_id'] / name; (out / 'frames').mkdir(parents=True, exist_ok=True); (out / 'depth').mkdir(exist_ok=True)
    c2w = cams['c2w_source'] if frame == 'source' else cams['c2w_canonical']; K = cams['K']; w, h = int(cams['width']), int(cams['height'])
    key_steps = set(case['input_steps']) | {k['visit'] for k in case['keypoints']} | {k['return'] for k in case['keypoints']}
    t0 = time.time()
    for i, rec in enumerate(traj['steps']):
        want = a.depth == 'all' or (a.depth == 'key' and i in key_steps)
        rgb, depth = render(model, c2w[i], K, w, h, rec['t'], i, traj, want)
        cv2.imwrite(str(out / 'frames' / f'{i:05d}.png'), rgb[:, :, ::-1])
        if depth is not None: np.save(out / 'depth' / f'{i:05d}.npy', depth.astype(np.float16))
        if i % 100 == 0: print('rendered', i, len(traj['steps']), f'{time.time() - t0:.0f}s', flush=True)
    (out / 'render.json').write_text(json.dumps({'case': str(case_dir), 'model': a.model, 'frame': frame, 'steps': len(traj['steps']), 'depth_steps': sorted(key_steps) if a.depth == 'key' else a.depth,
        'renderer': 'gsplat RGB(+ED), white background, SH as stored', 'seconds': round(time.time() - t0, 1)}, indent=2))
    print('done', out, flush=True)


if __name__ == '__main__':
    main()
