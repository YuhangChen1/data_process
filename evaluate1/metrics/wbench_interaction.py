"""WBench setting adherence (S1, S2) and interaction adherence (I1-I4) as evaluate1 operators, plus camera-pose
estimation shared with C1/C2.

I1 navigation: WBench runs MegaSaM on the whole case video, splits it into turns and scores each navigation case with
nATE accuracy + cross-turn consistency (navigation_trajectory.py). In evaluate1 every revisit keypoint is one
navigation case: its 3 actions + 3 inverse actions are 6 contiguous turns with exact step ranges (ctx.turns /
ctx.keypoints[i]['turns']); walk segments ('FOLLOW') are not WBench navigation actions and are excluded exactly like
WBench's nav_mask. Accuracy / consistency are averaged over keypoints, NavScore = (Acc + Cons) / 2 (main.py:964-975).

Poses: ctx.c2w_est if a caller supplied it, else MegaSaM (WBench's estimator, run in-process because box GPUs are in
Exclusive_Process mode: a subprocess cannot open the GPU this process already holds), else DA3-GIANT per keypoint clip,
else (C1/C2 only) commanded poses. The source is always recorded in raw['pose_source'].

VLM judges (S1, S2, I2-I4): WBench prompts verbatim, judged by gpt-6.1-sol. Frames go as numbered contact sheets and the
answer is requested as JSON (gateway client constraint) instead of ARK base64 video / free-text 'Answer: Yes'.
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import hashlib, os, runpy, sys, tempfile, time
import numpy as np

from .base import operator, Result, NotApplicable, TMP, device, cached
from .wbench_quality import (cached_try, WB, WB_TP, REF, sample_indices, pil, wb_score, free_gpu, syspath, vlm_ask, vlm_sheets,
                             turn_segments)

MS = WB_TP / 'mega-sam'
NAV_TOKENS = {'W': 'W', 'S': 'S', 'A': 'A', 'D': 'D', 'LEFT': 'left', 'RIGHT': 'right', 'UP': 'up', 'DOWN': 'down',
              'left': 'left', 'right': 'right', 'up': 'up', 'down': 'down', 'FORWARD': 'W', 'BACKWARD': 'S'}


# ───────────────────────────── navigation_trajectory.py (verbatim port) ─────────────────────────────
ACTION_TO_MOTION = {
    "W": {"translation": (0, 0, 1), "rotation": None}, "S": {"translation": (0, 0, -1), "rotation": None},
    "A": {"translation": (-1, 0, 0), "rotation": None}, "D": {"translation": (1, 0, 0), "rotation": None},
    "left": {"translation": None, "rotation": (-1, 0, 0)}, "right": {"translation": None, "rotation": (1, 0, 0)},
    "up": {"translation": None, "rotation": (0, 0, 1)}, "down": {"translation": None, "rotation": (0, 0, -1)},
    "W+A": {"translation": (-0.707, 0, 0.707), "rotation": None}, "W+D": {"translation": (0.707, 0, 0.707), "rotation": None},
    "S+A": {"translation": (-0.707, 0, -0.707), "rotation": None}, "S+D": {"translation": (0.707, 0, -0.707), "rotation": None},
    "W+left": {"translation": (0, 0, 1), "rotation": (-1, 0, 0)}, "W+right": {"translation": (0, 0, 1), "rotation": (1, 0, 0)},
    "W+up": {"translation": (0, 0, 1), "rotation": (0, 0, 1)}, "W+down": {"translation": (0, 0, 1), "rotation": (0, 0, -1)},
    "S+left": {"translation": (0, 0, -1), "rotation": (-1, 0, 0)}, "S+right": {"translation": (0, 0, -1), "rotation": (1, 0, 0)},
    "A+left": {"translation": (-1, 0, 0), "rotation": (-1, 0, 0)}, "A+right": {"translation": (-1, 0, 0), "rotation": (1, 0, 0)},
    "D+left": {"translation": (1, 0, 0), "rotation": (-1, 0, 0)}, "D+right": {"translation": (1, 0, 0), "rotation": (1, 0, 0)},
}
ACTION_ALIASES = {"forward": "W", "backward": "S", "cam_left": "left", "cam_right": "right", "cam_up": "up", "cam_down": "down",
                  "look_left": "left", "look_right": "right", "look_up": "up", "look_down": "down", "pitch_up": "up",
                  "pitch_down": "down", "yaw_left": "left", "yaw_right": "right"}
MIN_DISP_NORM, MIN_ROT_NORM, MIN_DISP_THRESHOLD, MIN_ROT_THRESHOLD = 0.5, 10.0, 0.1, 3.0
ARC_SAMPLES_PER_TURN, FALLBACK_DISP, FALLBACK_ROT_DEG, FALLBACK_DEPTH = 20, 1.0, 30.0, 1.0
_SYMMETRIC_GROUPS = {"W": "forward_back", "S": "forward_back", "A": "lateral", "D": "lateral",
                     "left": "yaw", "right": "yaw", "up": "pitch", "down": "pitch"}
_MIRROR_PAIRS = {frozenset({"right", "left"}): "yaw", frozenset({"up", "down"}): "pitch",
                 frozenset({"W", "S"}): "forward", frozenset({"A", "D"}): "lateral"}
_MIRROR_AXES = {"yaw": 0, "pitch": 1, "forward": 2, "lateral": 0}


def normalize_action(action):
    parts = [p.strip() for p in str(action).strip().replace(",", "+").split("+") if p.strip()]
    out = []
    for p in parts:
        a = ACTION_ALIASES.get(p, p)
        if a in {"w", "a", "s", "d"}: a = a.upper()
        out.append(a)
    return "+".join(sorted(out)) if len(out) > 1 else out[0] if out else action


ACTION_TO_MOTION = {normalize_action(k): v for k, v in ACTION_TO_MOTION.items()}


def _rot_angle_deg(R): return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
def _path_length(p): return 0.0 if len(p) < 2 else float(np.sum(np.linalg.norm(np.diff(p[:, :3, 3], axis=0), axis=1)))
def _total_rotation_deg(p): return sum(_rot_angle_deg(p[i, :3, :3].T @ p[i + 1, :3, :3]) for i in range(len(p) - 1))


def _compute_ate(gt, pred):
    n = min(len(gt), len(pred))
    t = np.linalg.norm(gt[:n, :3, 3] - pred[:n, :3, 3], axis=1)
    r = np.array([_rot_angle_deg(gt[i, :3, :3].T @ pred[i, :3, :3]) for i in range(n)])
    return float(t.mean()), float(r.mean())


def _is_pure_translation(a):
    m = ACTION_TO_MOTION.get(normalize_action(a)); return m is not None and m["rotation"] is None


def _get_symmetry_key(a):
    a = normalize_action(a)
    if "+" in a: return "+".join(sorted(set(_SYMMETRIC_GROUPS.get(p, p) for p in a.split("+"))))
    return _SYMMETRIC_GROUPS.get(a, a)


def _resample_by_arc(poses, n_samples):
    from scipy.spatial.transform import Rotation as R, Slerp
    n = len(poses)
    if n <= 2 or n_samples <= 2: return poses[np.linspace(0, max(n - 1, 0), min(n, n_samples)).astype(int)]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]); total = arc[-1]
    if total < 1e-8: return poses[np.linspace(0, n - 1, n_samples).astype(int)]
    target = np.linspace(0, total, n_samples)
    pos = np.column_stack([np.interp(target, arc, poses[:, :3, 3][:, d]) for d in range(3)])
    an = arc / total
    for i in range(1, len(an)):
        if an[i] <= an[i - 1]: an[i] = an[i - 1] + 1e-12
    rots = Slerp(an, R.from_matrix(poses[:, :3, :3]))(np.clip(target / total, an[0], an[-1]))
    out = np.zeros((n_samples, 4, 4)); out[:, :3, :3] = rots.as_matrix(); out[:, :3, 3] = pos; out[:, 3, 3] = 1.0
    return out


def _build_gt_trajectory(action, n_frames, perspective="first_person", rotation_angle_deg=60.0, step_size=2.0, subject_depth=2.0):
    from scipy.spatial.transform import Rotation as R
    exp = ACTION_TO_MOTION.get(normalize_action(action))
    if exp is None: return None
    tR, tT = np.eye(3), np.zeros(3)
    if exp["translation"] is not None: tT = np.array(exp["translation"], float) * step_size
    if exp["rotation"] is not None:
        rd = np.array(exp["rotation"], float); ang = np.deg2rad(rotation_angle_deg)
        if rd[0] != 0: tR = R.from_euler("y", rd[0] * ang).as_matrix()
        elif rd[2] != 0: tR = R.from_euler("x", rd[2] * ang).as_matrix()
    rv = R.from_matrix(tR).as_rotvec(); ht, hr = exp["translation"] is not None, exp["rotation"] is not None
    n = max(n_frames - 1, 1); poses = []

    def orbit(t, sp):
        Tt = sp + R.from_rotvec(t * rv).as_matrix() @ np.array([0, 0, -subject_depth])
        fwd = sp - Tt; fwd /= np.linalg.norm(fwd) + 1e-8; up = np.array([0.0, -1.0, 0.0])
        right = np.cross(fwd, up); right /= np.linalg.norm(right) + 1e-8; up = np.cross(right, fwd)
        P = np.eye(4); P[:3, :3] = np.stack([right, -up, fwd], 1); P[:3, 3] = Tt; return P

    if ht and hr:
        if perspective == "first_person":
            Rc, pc = np.eye(3), np.zeros(3)
            for i in range(n_frames):
                P = np.eye(4); P[:3, :3] = Rc; P[:3, 3] = pc; poses.append(P)
                if i < n_frames - 1: Rc = Rc @ R.from_rotvec(rv / n).as_matrix(); pc = pc + Rc @ (tT / n)
        else:
            for i in range(n_frames): poses.append(orbit(i / n, np.array([0, 0, subject_depth]) + i / n * tT))
    elif hr and perspective != "first_person":
        for i in range(n_frames): poses.append(orbit(i / n, np.array([0, 0, subject_depth], float)))
    elif hr:
        for i in range(n_frames):
            P = np.eye(4); P[:3, :3] = R.from_rotvec(i / n * rv).as_matrix(); poses.append(P)
    else:
        for i in range(n_frames):
            P = np.eye(4); P[:3, 3] = i / n * tT; poses.append(P)
    return np.array(poses, float)


def _estimate_orbit_radius(p):
    chord = np.linalg.norm(p[-1, :3, 3] - p[0, :3, 3]); th = np.deg2rad(_total_rotation_deg(p))
    if th < np.deg2rad(5) or np.sin(th / 2) < 1e-6: return None
    return float(chord / (2 * np.sin(th / 2)))


def _build_gt_translation(pt, action, Rw):
    nf = len(pt); d = np.array(ACTION_TO_MOTION[normalize_action(action)]["translation"], float); d /= np.linalg.norm(d) + 1e-12
    wd = Rw @ d; disp = np.linalg.norm(pt[-1, :3, 3] - pt[0, :3, 3]); g = disp if disp > MIN_DISP_THRESHOLD else FALLBACK_DISP
    seg = np.zeros((nf, 4, 4))
    for i in range(nf):
        seg[i, :3, :3] = Rw; seg[i, :3, 3] = pt[0, :3, 3] + i / max(nf - 1, 1) * g * wd; seg[i, 3, 3] = 1
    return seg


def _build_gt_rotation(pt, action, perspective):
    nf = len(pt); pr = _total_rotation_deg(pt); eR = _estimate_orbit_radius(pt)
    if pr < MIN_ROT_THRESHOLD: kw = {"rotation_angle_deg": FALLBACK_ROT_DEG, "subject_depth": FALLBACK_DEPTH}
    elif eR is not None:
        kw = {"rotation_angle_deg": max(pr, FALLBACK_ROT_DEG), "subject_depth": eR if perspective == "first_person" else max(eR, FALLBACK_DEPTH)}
    else: kw = {"rotation_angle_deg": FALLBACK_ROT_DEG, "subject_depth": FALLBACK_DEPTH}
    g = _build_gt_trajectory(action, nf, perspective, **kw)
    if g is None: return np.tile(np.eye(4), (nf, 1, 1))
    Ra = pt[0, :3, :3] @ g[0, :3, :3].T
    gp, pp = g[:, :3, 3] - g[0, :3, 3], pt[:, :3, 3] - pt[0, :3, 3]
    gr, prr = np.linalg.norm(gp, axis=1).mean() + 1e-12, np.linalg.norm(pp, axis=1).mean() + 1e-12
    sc = 1.0 if (pr < MIN_ROT_THRESHOLD or prr < MIN_DISP_THRESHOLD) else prr / gr
    gpr = (Ra @ gp.T).T; seg = np.zeros_like(g)
    for i in range(len(g)):
        seg[i, :3, :3] = Ra @ g[i, :3, :3]; seg[i, :3, 3] = sc * gpr[i] + pt[0, :3, 3]; seg[i, 3, 3] = 1
    return seg


def _need_mirror(a1, a2):
    a1, a2 = normalize_action(a1), normalize_action(a2)
    if a1 == a2: return []
    g1 = {_SYMMETRIC_GROUPS.get(p, p): p for p in a1.split("+")}; g2 = {_SYMMETRIC_GROUPS.get(p, p): p for p in a2.split("+")}
    if set(g1) != set(g2): return None
    out = []
    for grp, c1 in g1.items():
        c2 = g2[grp]
        if c1 == c2: continue
        m = _MIRROR_PAIRS.get(frozenset({c1, c2}))
        if m is None: return None
        out.append(m)
    return out


def _normalize_turn(p):
    R0i, t0 = p[0, :3, :3].T, p[0, :3, 3]; out = np.zeros_like(p)
    for i in range(len(p)):
        out[i, :3, :3] = R0i @ p[i, :3, :3]; out[i, :3, 3] = R0i @ (p[i, :3, 3] - t0); out[i, 3, 3] = 1
    return out


def _mirror_turn(p, types):
    M = np.eye(3)
    for ax in {_MIRROR_AXES[t] for t in types}: M[ax, ax] = -1.0
    out = np.zeros_like(p)
    for i in range(len(p)):
        out[i, :3, :3] = M @ p[i, :3, :3] @ M; out[i, :3, 3] = M @ p[i, :3, 3]; out[i, 3, 3] = 1
    return out


def _turn_pair_error(p1, p2, mt=()):
    n1, n2 = _normalize_turn(p1), _normalize_turn(p2)
    if mt: n2 = _mirror_turn(n2, mt)
    nc = min(len(n1), len(n2)); n1 = n1[np.linspace(0, len(n1) - 1, nc).astype(int)]; n2 = n2[np.linspace(0, len(n2) - 1, nc).astype(int)]
    at = float(np.linalg.norm(n1[:, :3, 3] - n2[:, :3, 3], axis=1).mean())
    ar = float(np.mean([_rot_angle_deg(n1[i, :3, :3].T @ n2[i, :3, :3]) for i in range(nc)]))
    path = (_path_length(n1) + _path_length(n2)) / 2; rot = (_total_rotation_deg(n1) + _total_rotation_deg(n2)) / 2
    return {"nATE_t": min(at / max(path, MIN_DISP_NORM), 1.0), "nATE_r": min(ar / max(rot, MIN_ROT_NORM), 1.0)}


def evaluate_navigation(poses, turn_bounds, actions, perspective="first_person"):
    """navigation_trajectory.py:540-631 verbatim (poses: c2w OpenCV [N,4,4]; bounds: [(s, e_exclusive)])."""
    Rw = poses[turn_bounds[0][0], :3, :3].copy(); cur = poses[turn_bounds[0][0], :3, 3].copy(); gts = []
    for (s, e), a in zip(turn_bounds, actions):
        pt = poses[s:e]
        g = _build_gt_translation(pt, a, Rw) if _is_pure_translation(a) else _build_gt_rotation(pt, a, perspective)
        gg = g.copy(); gg[:, :3, 3] += cur - g[0, :3, 3]; cur = gg[-1, :3, 3].copy()
        if not _is_pure_translation(a): Rw = gg[-1, :3, :3].copy()
        gts.append(gg)
    gs, ps = [], []
    for (s, e), g in zip(turn_bounds, gts):
        pt = poses[s:e]; nt = min(len(g), len(pt))
        gs.append(_resample_by_arc(g[:nt], ARC_SAMPLES_PER_TURN)); ps.append(_resample_by_arc(pt[:nt], ARC_SAMPLES_PER_TURN))
    ate_t, ate_r = _compute_ate(np.concatenate(gs), np.concatenate(ps))
    full = poses[turn_bounds[0][0]:turn_bounds[-1][1]]; tp, tr = _path_length(full), _total_rotation_deg(full)
    nt, nr = min(ate_t / max(tp, MIN_DISP_NORM), 1.0), min(ate_r / max(tr, MIN_ROT_NORM), 1.0)
    data = [(poses[s:e], a, _get_symmetry_key(a)) for (s, e), a in zip(turn_bounds, actions)]; errs = []
    for i, j in combinations(range(len(data)), 2):
        if data[i][2] != data[j][2]: continue
        mt = _need_mirror(data[i][1], data[j][1])
        if mt is None: continue
        errs.append(_turn_pair_error(data[i][0], data[j][0], mt))
    acc = 1.0 - (nt + nr) / 2.0
    cons = 1.0 - (np.mean([e["nATE_t"] for e in errs]) + np.mean([e["nATE_r"] for e in errs])) / 2.0 if errs else None
    return {"NavScore": (acc + cons) / 2.0 if cons is not None else None, "accuracy": float(acc),
            "consistency": None if cons is None else float(cons), "nATE_t": float(nt), "nATE_r": float(nr),
            "cnATE_t": float(np.mean([e["nATE_t"] for e in errs])) if errs else None,
            "cnATE_r": float(np.mean([e["nATE_r"] for e in errs])) if errs else None, "consistency_pairs": len(errs),
            "ATE_t": ate_t, "ATE_r": ate_r, "total_path_length": tp, "total_rotation_deg": tr}


# ───────────────────────────── pose estimation ─────────────────────────────

def _fingerprint(ctx):
    f = ctx.frames; idx = np.linspace(0, len(f) - 1, 8).astype(int)
    h = hashlib.sha1(str(f.shape).encode())
    for i in idx: h.update(np.ascontiguousarray(f[i, ::16, ::16]).tobytes())
    return h.hexdigest()[:16]


def _run_script(path, argv, paths, cwd, env):
    """runpy a MegaSaM CLI script in this process (exclusive-process GPUs), then drop its modules again."""
    old = (list(sys.argv), list(sys.path), os.getcwd(), {k: os.environ.get(k) for k in env})
    before = set(sys.modules)
    try:
        os.chdir(cwd); os.environ.update(env); sys.argv = [str(path)] + [str(a) for a in argv]
        sys.path[:0] = [str(p) for p in paths]
        runpy.run_path(str(path), run_name='__main__')
    finally:
        sys.argv, sys.path[:] = old[0], old[1]; os.chdir(old[2])
        for k, v in old[3].items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        for k in [k for k in set(sys.modules) - before if str(MS) in (getattr(sys.modules[k], '__file__', '') or '')]:
            del sys.modules[k]
        free_gpu()


def megasam_poses(ctx):
    """tools/run_megasam.py run_single (target 15 fps -> stride max(1, int(fps/15))): Depth-Anything mono depth,
    UniDepth-V2 metric depth (eval mode, tools/megasam_unidepth_eval.py), DROID-based camera tracking -> cam_c2w."""
    def compute():
        import torch, cv2
        fp = _fingerprint(ctx); cache = TMP / 'wbench_cache' / f'megasam_{fp}.npz'
        if cache.exists():
            d = np.load(cache); return d['cam_c2w'], int(d['stride'])
        need = [WB / 'megasam' / 'depth_anything_vitl14.pth', WB / 'megasam' / 'megasam_final.pth']
        miss = [str(p) for p in need if not p.exists()]
        if miss: raise FileNotFoundError('MegaSaM weights missing: ' + ', '.join(miss))
        stride = max(1, int(ctx.fps / 15)); scene = f'wb_{fp}'
        th = TMP / 'megasam_torch_home' / 'hub'; th.mkdir(parents=True, exist_ok=True)
        for name, tgt in (('torchhub', WB / 'torch_hub'), ('checkpoints', WB / 'megasam' / 'torch_hub_checkpoints')):
            if not (th / name).exists(): (th / name).symlink_to(tgt)
        if not (MS / 'torchhub').exists(): (MS / 'torchhub').symlink_to(WB / 'torch_hub')
        hf = Path(os.environ.get('HF_HOME', TMP / 'hf')) / 'hub'; hf.mkdir(parents=True, exist_ok=True)
        src = WB / 'megasam' / 'huggingface' / 'hub' / 'models--lpiccinelli--unidepth-v2-vitl14'
        if src.exists() and not (hf / src.name).exists(): (hf / src.name).symlink_to(src)
        prev_hub = torch.hub.get_dir(); torch.hub.set_dir(str(th))
        env = {'TORCH_HOME': str(th.parent), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        base = MS / 'base'; lie = base / 'thirdparty' / 'lietorch'
        try:
            with tempfile.TemporaryDirectory(prefix='megasam_', dir=str(TMP / 'tmp')) as td:
                td = Path(td); fdir = td / 'frames' / scene; fdir.mkdir(parents=True)
                for k, i in enumerate(range(0, len(ctx.frames), stride)):
                    cv2.imwrite(str(fdir / f'{k:05d}.jpg'), ctx.frames[i][:, :, ::-1])
                _run_script(MS / 'Depth-Anything' / 'run_videos.py', ['--encoder', 'vitl', '--load-from', need[0], '--img-path', fdir,
                            '--outdir', td / 'mono' / scene, '--localhub'], [MS / 'Depth-Anything'], MS, env)
                with syspath(MS / 'UniDepth'):
                    from unidepth.models import UniDepthV2
                    from_pre = UniDepthV2.from_pretrained           # tools/megasam_unidepth_eval.py: force eval()
                    own = 'from_pretrained' in UniDepthV2.__dict__; orig = UniDepthV2.__dict__.get('from_pretrained')
                    UniDepthV2.from_pretrained = lambda *a, **k: from_pre(*a, **k).eval()
                    try:
                        _run_script(MS / 'UniDepth' / 'scripts' / 'demo_mega-sam.py', ['--scene-name', scene, '--img-path', fdir,
                                    '--outdir', td / 'metric'], [MS / 'UniDepth'], MS, env)
                    finally:
                        if own: UniDepthV2.from_pretrained = orig
                        else: del UniDepthV2.from_pretrained
                _run_script(MS / 'camera_tracking_scripts' / 'test_demo.py', ['--datapath', fdir, '--weights', need[1], '--scene_name', scene,
                            '--mono_depth_path', td / 'mono', '--metric_depth_path', td / 'metric', '--disable_vis'],
                            # venv torch_scatter is ABI-broken for torch 2.5.1: pt25cu121 wheel in pylib_megasam first
                            [TMP / 'pylib_megasam', base, lie, base / 'droid_slam', MS / 'camera_tracking_scripts'], MS, env)
                out = MS / 'outputs' / f'{scene}_droid.npz'; d = np.load(out); c2w = d['cam_c2w'].astype(np.float64)
                out.unlink(missing_ok=True)
                import shutil; shutil.rmtree(MS / 'reconstructions' / scene, ignore_errors=True)
        finally:
            torch.hub.set_dir(prev_hub)
        cache.parent.mkdir(parents=True, exist_ok=True); np.savez(cache, cam_c2w=c2w, stride=stride)
        return c2w, stride
    return cached_try(ctx, 'wbench_megasam', compute)


def _da3_clip_poses(ctx, s, e):
    """DA3-GIANT poses (world-to-camera extrinsics -> c2w) for every frame of steps [s, e)."""
    def compute():
        from .wbench_consistency import _da3_stubs, da3_model_dir
        mdir, _ = da3_model_dir()
        _da3_stubs()
        with syspath(WB_TP / 'depth-anything-3' / 'src'):
            from depth_anything_3.api import DepthAnything3
            m = cached(ctx, 'wbench_da3_model', lambda: DepthAnything3.from_pretrained(str(mdir)).to(device=device()))
            pred = m.inference([np.ascontiguousarray(f) for f in ctx.frames[s:e]])
        ext = np.asarray(pred.extrinsics, np.float64)
        E = np.zeros((len(ext), 4, 4)); E[:, :3, :4] = ext[:, :3, :4]; E[:, 3, 3] = 1
        return np.linalg.inv(E)
    return cached_try(ctx, f'wbench_da3_clip_{s}_{e}', compute)


def keypoint_poses(ctx, k, allow_commanded=False):
    """Poses for steps [visit, end of last turn] of keypoint k -> (poses [n,4,4], first_step, source)."""
    s = int(k['visit']); turns = k.get('turns') or []
    e = (int(turns[-1]['steps'][1]) if turns else int(k['return'])) + 1
    if ctx.c2w_est is not None: return np.asarray(ctx.c2w_est)[s:e], s, 'ctx.c2w_est'
    errs = []
    try:
        c2w, stride = megasam_poses(ctx)
        if stride == 1: return c2w[s:e], s, 'megasam'
        errs.append(f'megasam stride {stride} != 1')
    except Exception as ex:
        errs.append(f'megasam: {type(ex).__name__}: {str(ex)[:200]}')
    try:
        return _da3_clip_poses(ctx, s, e), s, 'da3_clip(' + '; '.join(errs) + ')'
    except Exception as ex:
        errs.append(f'da3: {type(ex).__name__}: {str(ex)[:200]}')
    if allow_commanded and ctx.c2w_cmd is not None:
        return np.asarray(ctx.c2w_cmd)[s:e], s, 'commanded(' + '; '.join(errs) + ')'
    raise RuntimeError('no pose estimator available: ' + '; '.join(errs))


def estimated_poses(ctx, allow_commanded=False):
    """Whole-video poses for C1/C2 when available: (poses, source, stride). Falls back to per-keypoint estimation
    stitched into a commanded-pose array (relative rotations inside each keypoint clip are what C1/C2 use)."""
    if ctx.c2w_est is not None: return np.asarray(ctx.c2w_est), 'ctx.c2w_est', 1
    try:
        c2w, stride = megasam_poses(ctx); return c2w, 'megasam', stride
    except Exception as ex:
        err = f'{type(ex).__name__}: {str(ex)[:200]}'
    base_ = np.array(ctx.c2w_cmd if ctx.c2w_cmd is not None else np.tile(np.eye(4), (len(ctx.frames), 1, 1)), float); srcs = set()
    for k in ctx.keypoints or []:
        p, s, src = keypoint_poses(ctx, k, allow_commanded); base_[s:s + len(p)] = p; srcs.add(src.split('(')[0])
    return base_, '+'.join(sorted(srcs)) + f' (megasam unavailable: {err})', 1


# ───────────────────────────── I1 navigation ─────────────────────────────

@operator('wbench.I1_navigation_score', 'wbench.interaction_adherence', ('frames', 'keypoints'), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/interaction/navigation_trajectory.py:540-631 + main.py:964-975 + tools/run_megasam.py',
          requires=('third_party/wbench/mega-sam (+ built droid_backends, lietorch)', 'weights/wbench/megasam', 'or DA3 fallback'))
def navigation_score(ctx):
    """NavScore = (Accuracy + Consistency)/2, Accuracy = 1-(nATE_t+nATE_r)/2 vs adaptive GT per action turn,
    Consistency = 1-(cnATE_t+cnATE_r)/2 over same-group (mirrored) turn pairs; each keypoint excursion = one case."""
    per, accs, conss, srcs = [], [], [], set()
    for k in ctx.keypoints:
        turns = [t for t in (k.get('turns') or []) if NAV_TOKENS.get(t.get('token', t.get('action'))) in ACTION_TO_MOTION]
        if not turns: continue
        poses, s0, src = keypoint_poses(ctx, k); srcs.add(src)
        bounds = [(int(t['steps'][0]) - s0, int(t['steps'][1]) + 1 - s0) for t in turns]
        acts = [NAV_TOKENS[t.get('token', t.get('action'))] for t in turns]
        r = evaluate_navigation(poses, bounds, acts, 'first_person'); r['keypoint'] = k['id']; r['actions'] = acts
        per.append(r); accs.append(r['accuracy'])
        if r['consistency'] is not None: conss.append(r['consistency'])
    if not per: raise NotApplicable('no navigation turns in the keypoint programs')
    acc = float(np.mean(accs)); cons = float(np.mean(conss)) if conss else None
    nav = (acc + cons) / 2 if cons is not None else None
    return wb_score(nav, {'accuracy': acc, 'consistency': cons, 'pose_source': sorted(srcs), 'cases': len(per)}, per,
                    note='' if all(s == 'megasam' for s in srcs) else 'pose source: ' + ', '.join(sorted(srcs))[:200])


# ───────────────────────────── S1 / S2 setting adherence ─────────────────────────────

SCENE_PROMPT = """You are a video analysis expert. You will see video frames from Turn 1 of a generated video (sampled at 2fps).

Evaluate the video on two aspects:

**1. Environment Maintenance (1-5)**
How well does the video maintain these scene elements that were visible in the initial frame? Also consider whether the visual style remains consistent.
Scene description: {visible_part}
Visual style: {style}

Scoring:
- 5: All elements perfectly maintained, style fully consistent
- 4: Most elements maintained, style mostly consistent, minor issues
- 3: Some elements maintained or style partially drifts
- 2: Few elements maintained, significant changes or style mismatch
- 1: Scene barely matches the description or style completely wrong

**2. Offscreen Content Appearance (0 or 1)**
The following elements were described as existing OFF-SCREEN (not visible in the initial frame). Did ANY of them appear in the video as the camera moved?
Description: {offscreen_part}

- 1: At least one offscreen element appeared
- 0: None of the offscreen elements appeared

Answer in JSON format ONLY. Each reason must be within 20 words.
{{"maintenance": <1-5>, "maintenance_reason": "...", "offscreen": <0 or 1>, "offscreen_reason": "..."}}"""

SUBJECT_PROMPT = """You are a video analysis expert. You will see video frames from Turn 1 of a generated video (sampled at 2fps).

Evaluate the video on two aspects:

**1. Subject Appearance Maintenance (1-5)**
How well does the video maintain the subject's visual appearance (shape, color, clothing, equipment, category)?
Subject appearance: {appearance_part}

Scoring:
- 5: Subject appearance perfectly maintained throughout
- 4: Mostly maintained, minor visual changes
- 3: Some features maintained, noticeable changes
- 2: Few features maintained, significant appearance drift
- 1: Subject barely recognizable or completely changed

**2. Subject Action (0 or 1)**
Does the subject exhibit the described movement/action pattern in the video?
Expected action: {action_part}

- 1: The described action/movement pattern is clearly visible
- 0: The action/movement pattern is not visible or completely different

Answer in JSON format ONLY. Each reason must be within 20 words.
{{"maintenance": <1-5>, "maintenance_reason": "...", "action": <0 or 1>, "action_reason": "..."}}"""

SHEET_NOTE = ('\n\nThe frames are given as numbered contact sheet image(s) in temporal order '
              '(number in the top-left corner of each cell).')


def _turn1_frames(ctx, fps=2.0, scope='turn1'):
    if scope == 'turn1' and ctx.turns:
        s, e = int(ctx.turns[0]['frames'][0]), int(ctx.turns[0]['frames'][1]) + 1
    else: s, e = 0, len(ctx.frames)
    return [pil(ctx.frames[i]) for i in sample_indices(e - s, ctx.fps, {'fps': fps}, start=s)], (s, e)


@operator('wbench.S1_scene_adherence', 'wbench.setting_adherence', ('frames', 'turns'), range=(0, 100), vlm=True,
          source=REF + ' src/metrics/setting_adherence/scene_adherence.py:30-117 (judge: gpt-6.1-sol instead of doubao-seed-2.0-lite)')
def scene_adherence(ctx, scope='turn1'):
    """(maintenance/5 + offscreen)/2 judged on Turn-1 frames at 2 fps with the case's visible/offscreen scene text."""
    st = ctx.extra.get('setting') or {}
    visible = st.get('visible_part') or ctx.scene_text
    if not visible: raise NotApplicable('no scene description (case setting.visible_part / scene_text)')
    imgs, (s, e) = _turn1_frames(ctx, scope=scope)
    ans = vlm_ask(ctx, SCENE_PROMPT.format(visible_part=visible, offscreen_part=st.get('offscreen_part', ''),
                  style=st.get('style', 'realistic')) + SHEET_NOTE, vlm_sheets(imgs), 'S1')
    m, o = int(ans['maintenance']), int(ans['offscreen'])
    if m not in (1, 2, 3, 4, 5) or o not in (0, 1): raise ValueError(f'invalid judge answer {ans}')
    return wb_score((m / 5.0 + o) / 2.0, {'maintenance': m, 'offscreen': o, 'answer': ans, 'steps': [s, e - 1], 'frames': len(imgs),
                                          'offscreen_part_given': bool(st.get('offscreen_part'))})


@operator('wbench.S2_subject_adherence', 'wbench.setting_adherence', ('frames', 'turns'), range=(0, 100), vlm=True,
          source=REF + ' src/metrics/setting_adherence/subject_adherence.py:30-113')
def subject_adherence(ctx):
    """(appearance maintenance/5 + action)/2 on Turn-1 frames at 2 fps; needs a subject (ctx.subject_text / extra['subject'])."""
    sub = ctx.extra.get('subject') or {}
    appearance = sub.get('appearance_part') or ctx.subject_text
    if not appearance: raise NotApplicable('static scene: no subject description')
    imgs, (s, e) = _turn1_frames(ctx)
    ans = vlm_ask(ctx, SUBJECT_PROMPT.format(appearance_part=appearance, action_part=sub.get('action_part', 'stays in view')) + SHEET_NOTE,
                  vlm_sheets(imgs), 'S2')
    m, a = int(ans['maintenance']), int(ans['action'])
    return wb_score((m / 5.0 + a) / 2.0, {'maintenance': m, 'action': a, 'answer': ans, 'steps': [s, e - 1]})


# ───────────────────────────── I2 / I3 / I4 VLM interaction ─────────────────────────────

EVAL_SYSTEM_PROMPT = (
    "You are evaluating AI-generated video segments from interactive world models. Each video segment corresponds to one "
    "interaction turn where a specific event was instructed to happen.\n\nYour task: answer each question with ONLY 'Yes' or 'No' "
    "based on what you observe in the provided frames. Follow these guidelines:\n- These are AI-generated videos, not real footage. "
    "Minor visual artifacts, slight inconsistencies, or imperfect rendering should NOT cause a 'No' if the overall intent is "
    "recognizable.\n- Focus on whether the CORE semantic meaning of the described event is visually present, not whether every "
    "literal detail matches perfectly.\n- A long or detailed event description may only be partially depicted. Judge based on "
    "whether the main action or change is recognizable.\n- When in doubt between Yes and No, lean toward the interpretation that "
    "better matches what you actually see, not what you expect to see.")
REASON_JSON = ('Return JSON only: {"reason": "<max 50 words; only describe specific issues found; if the answer is not the '
               'favourable one you MUST name the exact problem>", "answer": "Yes" or "No"}')


def event_edit_questions(a):
    return [
        ("Q1", f"Does the scene remain largely unchanged, with no clear indication of '{a}' occurring?\n(Criterion: answer No if there is ANY recognizable change related to this event. Only answer Yes if the scene is completely static or changes are entirely unrelated.)", "no"),
        ("Q2", f"Does the video show something resembling or consistent with the described event: '{a}'?\n(Criterion: answer Yes if the core action or change is visually recognizable, even if not every detail matches perfectly. Partial depiction counts as Yes.)", "yes"),
        ("Q3", f"By the end of this segment, has the described event '{a}' reached a clear conclusion or outcome?\n(Criterion: answer Yes only if the event has a visible end state or result. If the event is still ongoing, just beginning, or only partially shown, answer No.)", "yes"),
        ("Q4", f"Are the key details of '{a}' accurately depicted, such as the correct objects, agents, directions, and quantities involved?\n(Criterion: focus on the key entities and actions mentioned in the description. Ignore minor background differences or irrelevant details.)", "yes"),
        ("Q5", f"Does any unexpected object, entity, or visual anomaly appear in this segment that is clearly unrelated to '{a}'?\n(Criterion: only answer Yes if something obviously out-of-place appears. Natural consequences of the event do not count. Minor AI rendering artifacts like texture flickering do not count.)", "no")]


def subject_action_questions(a):
    return [
        ("Q1", f"Does the subject remain idle or stationary, with no attempt to perform '{a}'?\n(Criterion: answer No if the subject shows ANY movement or gesture related to this action. Only answer Yes if the subject is completely still or doing something unrelated.)", "no"),
        ("Q2", f"Does the video show the subject performing something resembling or consistent with: '{a}'?\n(Criterion: answer Yes if the core action is visually recognizable, even if not every detail matches perfectly. Partial execution counts as Yes.)", "yes"),
        ("Q3", f"By the end of this segment, has the subject's action '{a}' reached a clear conclusion or outcome?\n(Criterion: answer Yes only if there is a visible end state or result of the action. If the action is still ongoing, just beginning, or only partially shown, answer No.)", "yes"),
        ("Q4", f"Are the key details of '{a}' accurately depicted, such as the correct subject, objects being interacted with, and manner of the action?\n(Criterion: focus on the key entities and the manner of action mentioned in the description. Ignore minor background differences or irrelevant details.)", "yes"),
        ("Q5", f"Does the subject exhibit any unnatural body movement, impossible pose, or physically implausible interaction while performing '{a}'?\n(Criterion: only answer Yes for clearly unnatural movements, impossible poses, or physically impossible interactions. Minor AI rendering artifacts like slight jitter or imperfect hand shape do not count.)", "no")]


def _yes(ans):
    return str(ans.get('answer', '')).strip().lower().startswith('yes')


def _binary_turns(ctx, kind, qgen, tag):
    turns = [(t, s, e) for t, s, e in turn_segments(ctx) if t.get('type', t.get('kind')) == kind]
    if not turns: raise NotApplicable(f'no {kind} turns in the case')
    details, scores = [], []
    for t, s, e in turns:
        imgs = [pil(ctx.frames[i]) for i in sample_indices(e - s, ctx.fps, {'fps': 3.0}, start=s)]
        sheets = vlm_sheets(imgs); act = t.get('text') or t.get('action'); qs = []
        for sub, q, exp in qgen(act):
            try:
                ans = vlm_ask(ctx, f'{q}\n\n{REASON_JSON}{SHEET_NOTE}', sheets, f'{tag}_{t.get("id")}_{sub}', system=EVAL_SYSTEM_PROMPT)
                qs.append({'sub_id': sub, 'correct': _yes(ans) == (exp == 'yes'), 'answer': ans.get('answer'), 'expected': exp, 'reason': ans.get('reason')})
            except Exception as ex:
                qs.append({'sub_id': sub, 'correct': None, 'answer': 'error', 'expected': exp, 'reason': str(ex)[:200]})
        sc = sum(q['correct'] is True for q in qs) / len(qs); scores.append(sc)
        details.append({'turn': t.get('id'), 'action': act, 'score': sc, 'questions': qs})
    return wb_score(float(np.mean(scores)), {'num_turns': len(turns)}, details)


@operator('wbench.I2_event_editing_adherence', 'wbench.interaction_adherence', ('frames', 'turns'), range=(0, 100), vlm=True,
          source=REF + ' src/metrics/interaction/vlm_interaction.py:54-115,417-439,610-687')
def event_editing_adherence(ctx):
    """5 progressive binary checks per event_edit turn (3-fps turn frames); turn score = correct/5; mean over turns."""
    return _binary_turns(ctx, 'event_edit', event_edit_questions, 'I2')


@operator('wbench.I3_subject_action_adherence', 'wbench.interaction_adherence', ('frames', 'turns'), range=(0, 100), vlm=True,
          source=REF + ' src/metrics/interaction/vlm_interaction.py:118-180,442-464,610-687')
def subject_action_adherence(ctx):
    """5 progressive binary checks per subject_action turn; turn score = correct/5; mean over turns."""
    return _binary_turns(ctx, 'subject_action', subject_action_questions, 'I3')


PS_SYSTEM_PROMPT = (
    "You are evaluating perspective (camera viewpoint) switches in AI-generated video segments from interactive world models. "
    "Each segment corresponds to one interaction turn where a specific perspective change was instructed.\n\nKey definitions:\n"
    "- First-person (FP): The camera is the character's eyes. No part of the character's body is visible except possibly hands/weapon.\n"
    "- Third-person (TP): The camera follows the character from behind or above. The character's full or partial body is visible in the frame.\n"
    "- Scope/ADS: The view is through a weapon scope or binoculars, typically showing a circular vignette, crosshair/reticle, and magnified view.\n\n"
    "Your task: answer each question based on what you observe in the provided frames. Always give a one-sentence reason, then answer Yes or No.\n"
    "- These are AI-generated videos. Minor visual artifacts should NOT cause a 'No' if the overall intent is recognizable.\n"
    "- Focus on whether the PERSPECTIVE TYPE is correct and the transition is coherent, not whether every detail is perfect.")
PT_DESC = {"fp": "first-person — camera through the character's eyes, looking outward at the world; no part of the character's own body is visible except possibly hands or a held weapon/tool",
           "tp": "third-person — an external camera showing a character's body (back, side, or full figure) from behind, above, or at an angle",
           "scope": "scoped / ADS — a magnified view through a weapon scope, binoculars, or similar optic, typically with a circular vignette, crosshair, reticle, or other scope overlay"}
PT_SHORT = {"fp": "first-person", "tp": "third-person", "scope": "scoped/magnified"}


def _parse_ps(a):
    a = a.lower().strip().split(':')[0].strip()
    for k in ('fp_to_tp', 'tp_to_fp', 'tp_to_tp', 'fp_to_fp', 'fp_to_scope', 'scope_to_fp'):
        if k in a: return tuple(k.split('_to_'))
    p = a.split('_to_'); return (p[0].strip(), p[1].strip()) if len(p) == 2 else ('fp', 'tp')


def perspective_questions(action):
    """vlm_interaction.py:229-331 verbatim."""
    b, a = _parse_ps(action); bs, as_, ad = PT_SHORT.get(b, b), PT_SHORT.get(a, a), PT_DESC.get(a, a)
    if b == a:
        q1 = (f"You are shown 6 frames: the first 3 are from the BEGINNING and the last 3 from the END of this segment. Does the camera viewpoint change noticeably? Even though both before and after are {bs}, the camera position, the followed subject, or the viewing angle should shift significantly.\n(Answer No only if the viewpoint stays essentially the same throughout — same position, same subject, same angle.)")
    else:
        q1 = (f"You are shown 6 frames: the first 3 are from the BEGINNING and the last 3 from the END of this segment. Does the perspective clearly change from {bs} to {as_}?\n(Answer No only if the perspective type remains {bs} throughout with no visible transition.)")
    q3 = (f"You are shown 3 frames from the very END of this segment (AFTER the perspective switch). Is the perspective at this point {as_}?\nDefinition: {ad}.\n(Answer Yes if these frames match this perspective type.)")
    if a == 'tp':
        q4 = ("You are shown 3 frames from the END of this segment where the view should be third-person. Evaluate whether this is a STANDARD third-person camera setup. ALL of these must be true:\n  1. The character is roughly CENTERED in the frame\n  2. The camera is positioned BEHIND the character (over-the-shoulder or directly behind, NOT from the side or front)\n  3. The camera is at an appropriate DISTANCE — the character's upper body or full body is visible, not an extreme close-up or a far-away wide shot\n(Answer Yes only if all three criteria are met.)")
    elif a == 'fp':
        q4 = ("You are shown 3 frames from the END of this segment where the view should be first-person. Check whether the view is a VALID first-person perspective with NO impossible self-view artifacts.\nA valid first-person view must NOT show any of:\n  - The character's own FACE (front of head, facial features)\n  - The character's own BACK, rear of head, or shoulders from behind\n  - The character's own full body seen from outside\nAcceptable: hands, held weapon/tool, feet when looking down — these are normal in first-person.\n(Answer Yes if the view is a clean first-person perspective with none of the above violations. Answer No if the character's face, back, or full body is visible.)")
    else:
        q4 = ("You are shown 3 frames from the END of this segment where the view should be through a scope or optic. Check whether the scoped view is VALID.\nA valid scope view should show at least one of:\n  - Circular vignette or scope ring border\n  - Crosshair, reticle, or aiming markers\n  - Visible magnification (objects appear closer than normal)\n  - HUD overlay, distance readout, or targeting display\n(Answer Yes if any scope/optic visual element is present. Answer No if the view looks like a normal unscoped perspective.)")
    qc = ("You are shown frames sampled evenly across the ENTIRE segment. A perspective switch should be achieved by CAMERA movement (pulls back, pushes in, pans, or cuts to a new angle) or by a CINEMATIC CUT.\nIt should NOT be achieved by CHARACTER movement alone.\nIs the viewpoint change achieved through camera motion or a cinematic cut (rather than just character movement)?\n(Answer Yes if the camera clearly moves, zooms, or cuts.)")
    return [("Q1", q1, "yes", "all"), ("Q3", q3, "yes", "late"), ("Q4", q4, "yes", "late"), ("Q_cam", qc, "yes", "full")]


@operator('wbench.I4_perspective_switching_adherence', 'wbench.interaction_adherence', ('frames', 'turns'), range=(0, 100), vlm=True,
          source=REF + ' src/metrics/interaction/vlm_interaction.py:229-331,364-410,496-607 (strict Q1∧Q3∧Q4)')
def perspective_switching_adherence(ctx):
    """Per perspective_switch turn: success iff Q1 (6 early+late frames), Q3 and Q4 (3 late frames) all pass; % of turns."""
    turns = [(t, s, e) for t, s, e in turn_segments(ctx) if t.get('type', t.get('kind')) == 'perspective_switch']
    if not turns: raise NotApplicable('no perspective_switch turns in the case')
    scores, details = [], []
    for t, s, e in turns:
        n = e - s; margin = min(max(0.6, n / ctx.fps * 0.18), n / ctx.fps * 0.40); mf = int(margin * ctx.fps)
        early = [pil(ctx.frames[i]) for i in np.linspace(s, min(s + mf, e - 1), 3, dtype=int)]
        late = [pil(ctx.frames[i]) for i in np.linspace(max(e - mf, s), e - 1, 3, dtype=int)]
        full = [pil(ctx.frames[i]) for i in sample_indices(n, ctx.fps, {'fps': 3.0}, start=s)]
        sets = {'all': early + late, 'late': late, 'full': full}; q = {}
        for sub, text, exp, fs in perspective_questions(t.get('action', '')):
            try:
                ans = vlm_ask(ctx, f'{text}\n\n{REASON_JSON}{SHEET_NOTE}', [x for x in vlm_sheets(sets[fs], per_sheet=16)],
                              f'I4_{t.get("id")}_{sub}', system=PS_SYSTEM_PROMPT)
                q[sub] = {'pass': _yes(ans) == (exp == 'yes'), 'reason': ans.get('reason')}
            except Exception as ex:
                q[sub] = {'pass': None, 'reason': str(ex)[:200]}
        ok = all(q.get(k, {}).get('pass') is True for k in ('Q1', 'Q3', 'Q4')); scores.append(1.0 if ok else 0.0)
        details.append({'turn': t.get('id'), 'action': t.get('action'), 'score': scores[-1], 'q': q})
    return wb_score(float(np.mean(scores)), {'num_turns': len(turns)}, details)
