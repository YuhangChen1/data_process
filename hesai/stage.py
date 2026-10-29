"""Hesai scan -> data_pipeline workspace adapter.

  python -m hesai.stage --scene KS-IN-OE-S-F1_01_202609121414 --step source|geometry|walkthrough|render|all

source       sources/<key>/{gaussian.ply,mesh_post1.obj} symlinks to the raw scan (scene_pipeline reads mesh_post1.obj)
geometry     scene_pipeline.geometry.analyze -> outputs/<key>/ (leveled Z-up canonical frame, floors, nav maps, regions);
             shops/restaurants without a scanned ceiling get a synthetic ceiling 2.8 m above each floor peak
walkthrough  trajectory rule: if the scan ships its forward walkthrough (manyou_videos/main_walkthrough with poses.json +
             frames_hd) it becomes the annotation/evaluation clip outputs/<key>/hesai_walkthrough/main/ (canonical
             cameras, K, same-camera mesh depth, 3DGS re-render PSNR check); if it does not, this step is skipped and
             the render step plans A* walkthroughs on the navigation map instead
render       scene_pipeline render stage (top-down, panoramas, depth validation); A* region walkthrough videos only when
             the scan has no trajectory of its own
"""
from pathlib import Path
import argparse, json, math, os, time
import numpy as np

DP = Path(__file__).resolve().parents[1]
RAW = DP / 'hesai_dataset' / '20261019'


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); tmp.replace(path)


def read(path): return json.loads(Path(path).read_text())


def scene_dirs():
    """<RAW>/<scene>/ or <RAW>/<category>/[<sub>/]<scene>/ holding gaussian.ply + mesh_post.obj (bounded walk;
    rglob would descend into the frames_hd trees)."""
    found = []
    def walk(d, depth):
        if (d / 'mesh_post.obj').exists() and (d / 'gaussian.ply').exists():
            found.append(d); return
        if depth >= 3: return
        for c in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith('.')):
            walk(c, depth + 1)
    walk(RAW, 0)
    return found


def key_of(d):
    parts = d.name.split('_')
    return '_'.join(parts[:3])          # KS-IN-OE-S-F1_01_202609121414 (label dropped, ASCII only)


def find(key):
    for d in scene_dirs():
        if key_of(d) == key: return d
    raise SystemExit(f'unknown scene {key}')


def source(key):
    raw = find(key); st = DP / 'sources' / key; st.mkdir(parents=True, exist_ok=True)
    for name, target in (('mesh_post1.obj', raw / 'mesh_post.obj'), ('gaussian.ply', raw / 'gaussian.ply')):
        link = st / name
        if link.is_symlink() or link.exists(): link.unlink()
        link.symlink_to(target)
    rel = raw.relative_to(RAW)
    dump(st / 'source.json', {'scene_id': key, 'raw_dir': str(raw), 'relative': str(rel), 'label_zh': raw.name.split('_', 3)[-1],
                              'category_path': [p for p in rel.parts[:-1]], 'files': {n: {'bytes': (raw / n).stat().st_size,
                              'mtime': (raw / n).stat().st_mtime} for n in ['gaussian.ply', 'mesh_post.obj', 'data.json']}})
    return st


def geometry(key, force=False):
    import scene_pipeline.geometry as G
    st = source(key); out = DP / 'outputs' / key
    if (out / 'analysis.json').exists() and not force: return read(out / 'analysis.json')
    try:
        G.analyze(st, out)
    except RuntimeError as exc:
        if 'No credible floor/ceiling pair' not in str(exc): raise
        orig = G.peak_table
        def peak_table(c, n, a):
            centers, hist, tables = orig(c, n, a)
            for fl in tables[0]:
                if not any(1.8 < r['height'] - fl['height'] < 4.2 for r in tables[1]):
                    tables[1].append(dict(height=fl['height'] + 2.8, area=1e9, peak=0.0, synthetic=True))
            return centers, hist, tables
        G.peak_table = peak_table
        try: G.analyze(st, out)
        finally: G.peak_table = orig
        a = read(out / 'analysis.json'); a['ceiling_synthetic_m'] = 2.8; dump(out / 'analysis.json', a)
    walkthrough_floor(key)
    return read(out / 'analysis.json')


def walkthrough_floor(key, tolerance=.08):
    """The largest upward plane is not always the floor the scan was walked on (e.g. the library sits on a platform
    16 cm above an adjoining hall / outdoor ground). Measure the floor under the walkthrough cameras (downward mesh
    rays) and, when it differs from the chosen floor by more than `tolerance`, rebuild the floor maps at that height.
    The leveling (world_to_canonical) is computed before floor selection and is asserted unchanged."""
    import scene_pipeline.geometry as G, open3d as o3d
    raw = find(key); st = DP / 'sources' / key; out = DP / 'outputs' / key
    a = read(out / 'analysis.json'); W2C = np.array(a['world_to_canonical'])
    if 'walkthrough_floor_check' in a: return a['walkthrough_floor_check']
    if not has_walkthrough(key):
        a['walkthrough_floor_check'] = {'skipped': 'scan has no walkthrough trajectory; floor from plane analysis'}
        dump(out / 'analysis.json', a); return a['walkthrough_floor_check']
    poses = read(raw / 'manyou_videos' / 'main_walkthrough' / 'poses.json')['poses']
    P = (W2C[None] @ np.array([p['c2w'] for p in poses]))[:, :3, 3]
    m = np.load(out / 'mesh_cache.npz'); v = (m['vertices'] @ W2C[:3, :3].T).astype(np.float32)
    ray = G.build_rayscene(v, m['faces'])
    t = ray.cast_rays(o3d.core.Tensor(np.concatenate([P, np.tile([0, 0, -1.], (len(P), 1))], 1).astype(np.float32)))['t_hit'].numpy()
    ok = np.isfinite(t) & (t > .6) & (t < 2.2)
    z = float(np.median(P[ok, 2] - t[ok])) if ok.any() else None
    floors = a['floors']; chosen = min(floors, key=lambda f: abs(f['height'] - z)) if z is not None else None
    check = {'camera_floor_z': z, 'cameras_with_floor_hit': int(ok.sum()), 'cameras': len(P),
             'analysis_floor_z': chosen['height'] if chosen else None, 'tolerance_m': tolerance, 'rebuilt': False}
    if z is not None and abs(chosen['height'] - z) > tolerance:
        orig = G.peak_table; calls = {'n': 0}
        def peak_table(c, n, aa):
            centers, hist, tables = orig(c, n, aa); calls['n'] += 1
            if calls['n'] == 2:      # second call = after leveling: force the walked floor as the main candidate
                main = tables[0][0]
                tables[0].insert(0, dict(height=z, area=main['area'], peak=main.get('peak', 0.), forced_by='walkthrough_cameras',
                                         replaced_peak_height=chosen['height']))
            return centers, hist, tables
        G.peak_table = peak_table
        try: G.analyze(st, out)
        finally: G.peak_table = orig
        b = read(out / 'analysis.json')
        if not np.allclose(np.array(b['world_to_canonical']), W2C, atol=1e-9): raise RuntimeError('leveling changed while forcing floor')
        check['rebuilt'] = True; check['new_floor_z'] = [f['height'] for f in b['floors']]
        if 'ceiling_synthetic_m' in a: b['ceiling_synthetic_m'] = a['ceiling_synthetic_m']
        a = b
    a['walkthrough_floor_check'] = check; dump(out / 'analysis.json', a)
    print('walkthrough floor check', check, flush=True)
    return check


def has_walkthrough(key):
    w = find(key) / 'manyou_videos' / 'main_walkthrough'
    return (w / 'poses.json').exists() and (w / 'frames_hd').is_dir() and any((w / 'frames_hd').glob('*.png'))


def walkthrough_K(meta, width, height):
    fy = height / 2 / math.tan(math.radians(meta['fov_y_deg']) / 2)
    return np.array([[fy, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], np.float64)


def walkthrough(key, force=False, check_every=1):
    """Hesai main walkthrough -> annotation clip with canonical cameras + mesh depth + render check."""
    import cv2, torch
    from scene_pipeline.render import Renderer
    raw = find(key); st = source(key); out = DP / 'outputs' / key
    src = raw / 'manyou_videos' / 'main_walkthrough'; clip = out / 'hesai_walkthrough' / 'main'
    if (clip / 'metadata.json').exists() and not force: return read(clip / 'metadata.json')
    if not has_walkthrough(key):
        print('no walkthrough trajectory in the scan: A* walkthroughs will be rendered by the render step', flush=True); return None
    a = read(out / 'analysis.json'); W2C = np.array(a['world_to_canonical'], np.float64)
    doc = read(src / 'poses.json'); poses = doc['poses']
    frames = sorted((src / 'frames_hd').glob('*.png'))
    if len(frames) != len(poses): raise RuntimeError(f'frames {len(frames)} != poses {len(poses)}')
    h, w = cv2.imread(str(frames[0])).shape[:2]
    if len({p['fov_y_deg'] for p in poses}) != 1: raise RuntimeError('per-frame fov changes are not supported')
    K = walkthrough_K(poses[0], w, h)
    c2s = np.array([p['c2w'] for p in poses], np.float64)          # camera -> scan (source) frame, OpenCV axes
    c2c = W2C[None] @ c2s                                          # camera -> canonical Z-up
    renderer = Renderer(st, out)
    (clip / 'mesh_depth_mm').mkdir(parents=True, exist_ok=True); (clip / 'check').mkdir(exist_ok=True)
    floors = a['floors']; psnr = []; fl_ids = []
    for i, f in enumerate(frames):
        pose = c2c[i].astype(np.float32)
        depth = renderer.mesh_depth(pose, w, h, K.astype(np.float32))
        cv2.imwrite(str(clip / 'mesh_depth_mm' / f'{i:05d}.png'), np.uint16(np.clip(depth * 1000, 0, 65535)))
        z = c2c[i][2, 3]; below = [x for x in floors if x['height'] <= z] or floors
        fl_ids.append(max(below, key=lambda x: x['height'])['floor_id'])
        if i % check_every == 0:
            rgb = render_K(renderer, pose, K, w, h)
            gt = cv2.imread(str(f))[:, :, ::-1].astype(np.float64)
            mse = np.mean((rgb.astype(np.float64) - gt) ** 2); psnr.append(float(10 * np.log10(255 ** 2 / max(mse, 1e-9))))
            if i in (0, len(frames) // 2, len(frames) - 1):
                cv2.imwrite(str(clip / 'check' / f'{i:05d}.jpg'), np.concatenate([gt[:, :, ::-1], rgb[:, :, ::-1]], 1)[::2, ::2])
    if len(set(fl_ids)) != 1: raise RuntimeError(f'walkthrough spans floors {sorted(set(fl_ids))}; split it first')
    fl = next(x for x in floors if x['floor_id'] == fl_ids[0])
    np.savez(clip / 'cameras.npz', camera_to_canonical=c2c, camera_to_source=c2s, K=K, timestamps=np.arange(len(poses)) / 12.0)
    meta = {'schema': 'data_pipeline.hesai_walkthrough', 'scene_id': key, 'source': str(src), 'frames_dir': str(src / 'frames_hd'),
            'frames': len(frames), 'fps': 12, 'width': w, 'height': h, 'fov_y_deg': poses[0]['fov_y_deg'], 'K': K.tolist(),
            'floor': f'f{fl["floor_id"]}', 'camera_height_above_floor': float(np.median(c2c[:, 2, 3] - fl['height'])),
            'trajectory_collision_checked': False, 'region_order': [], 'planner': doc.get('planner'),
            'pose_convention': 'poses.json c2w = camera->scan frame, OpenCV camera (+X right,+Y down,+Z forward); '
                               'camera_to_canonical = world_to_canonical @ c2w',
            'intrinsics': 'pinhole, fy = H/2/tan(fov_y/2), fx = fy, principal point at image centre',
            'render_check': {'renderer': 'scene_pipeline Renderer (gsplat, SH0)', 'frames_checked': len(psnr),
                             'psnr_db_median': float(np.median(psnr)), 'psnr_db_min': float(np.min(psnr)),
                             'psnr_db_p10': float(np.percentile(psnr, 10)), 'per_frame_psnr_db': [round(x, 3) for x in psnr]},
            'purpose': 'annotation clip == evaluate1 forward roaming trajectory'}
    dump(clip / 'metadata.json', meta)
    del renderer; torch.cuda.empty_cache()
    return meta


def render_K(renderer, pose, K, w, h):
    """Renderer.rgbd with explicit pinhole K (Renderer.rgbd only takes a horizontal fov)."""
    import torch, gsplat
    view = np.linalg.inv(pose) @ renderer.world_to_canonical
    img, alpha, _ = gsplat.rasterization(**renderer.active, viewmats=torch.as_tensor(view[None].astype(np.float32), device='cuda'),
                                         Ks=torch.as_tensor(K[None].astype(np.float32), device='cuda'), width=w, height=h,
                                         near_plane=.03, far_plane=100., packed=True, render_mode='RGB', radius_clip=.4)
    img = img[0].cpu().numpy(); alpha = alpha[0, :, :, 0].cpu().numpy()
    return np.uint8(np.clip((img + 1 - alpha[:, :, None]) * 255, 0, 255))


def render(key, force=False):
    """Top-down / panoramas / depth validation; plus A* walkthroughs only when the scan ships no trajectory."""
    from scene_pipeline.render import render_scene
    out = DP / 'outputs' / key
    if (out / 'render_complete.json').exists() and not force: return
    render_scene(source(key), out, walkthroughs=False)
    if not has_walkthrough(key): astar(key)


def astar(key):
    """A* region walkthrough videos on the navigation map (fallback trajectory): per connected component, the
    annotation route through region centres is rendered as 3DGS RGB + mesh depth + cameras."""
    import torch
    from scene_pipeline.render import Renderer, video
    out = DP / 'outputs' / key; renderer = Renderer(source(key), out); clips = []
    for fl in renderer.report['floors']:
        folder = out / f'floor_{fl["floor_id"]}'; maps = np.load(folder / 'maps.npz')
        clips += [video(renderer, folder, fl, maps, route) for route in read(folder / 'annotation_routes.json')]
    del renderer; torch.cuda.empty_cache()
    return clips


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scene', required=True)
    ap.add_argument('--step', default='all', choices=['source', 'geometry', 'walkthrough', 'render', 'all'])
    ap.add_argument('--force', action='store_true'); args = ap.parse_args()
    t = time.time()
    if args.step in ('source', 'all'): print(source(args.scene))
    if args.step in ('geometry', 'all'): a = geometry(args.scene, args.force); print('floors', [f['floor_id'] for f in a['floors']])
    if args.step in ('walkthrough', 'all'):
        m = walkthrough(args.scene, args.force)
        if m: print('render check', {k: v for k, v in m['render_check'].items() if k != 'per_frame_psnr_db'})
    if args.step in ('render', 'all'): render(args.scene, args.force)
    print('done', args.step, f'{time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
