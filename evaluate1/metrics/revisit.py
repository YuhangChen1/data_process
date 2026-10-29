"""Keypoint revisit operators (ours): at each of the 6 keypoints the camera pauses, performs 3 actions and their 3
inverse actions, and returns to the keypoint pose. Frames `visit` (first visit) and `return` (after the excursion) must
show the same view, so everything measured here has an ideal value (0 error / similarity 1 / IoU 1).

ctx.keypoints: [{'id', 'visit': step, 'return': step, 'excursion': [...]}]   (excursion is informational)
ctx.objects:   [{'global_id', 'label' (English, used as detector prompt), 'label_zh', 'visible_at': {step: mask}}]
               mask = path to a PNG/NPY (non-zero = object) or {'path': ..., 'id': global_id} for an instance-ID map.

Families
  revisit.pose       relative camera pose visit->return estimated from the images (SuperPoint+LightGlue; essential matrix,
                     homography for near-zero baselines, PnP with ctx.depth for metric translation) compared with the
                     commanded relative pose; commanded-pose sanity; drift of an estimated trajectory (ctx.c2w_est)
  revisit.image      visit-vs-return PSNR / SSIM / LPIPS / DreamSim / DINOv2
  revisit.object     画面物体重合: per annotated object, label-prompted GroundingDINO + SAM2 re-segmentation in both frames
                     -> mask IoU, centroid shift, area consistency, presence recall / precision; label-agnostic SAM2
                     automatic masks matched by Hungarian IoU
  revisit.excursion  mirrored poses inside the excursion (pose reached by action k and again by its inverse) -> similarity
"""
from __future__ import annotations

import math

import numpy as np

from .base import Result, NotApplicable, operator, cached
from . import r2m as R

SRC = 'evaluate1 (ours); matcher / detector stack from R2M-Bench'


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------

def _keypoints(ctx):
    kps = [k for k in (ctx.keypoints or []) if 0 <= int(k['visit']) < len(ctx.frames) and 0 <= int(k['return']) < len(ctx.frames)]
    if not kps: raise NotApplicable('no keypoint with both visit and return inside the rendered frames')
    return kps


def _K(ctx, step):
    K = np.asarray(ctx.K, float)
    return K[step] if K.ndim == 3 else K


def rot_angle_deg(R_):
    return float(np.degrees(np.arccos(np.clip((np.trace(R_) - 1) / 2, -1, 1))))


def cmd_relative(ctx, a, b):
    """Commanded T_b<-a = inv(c2w_b) @ c2w_a (maps camera-a coordinates into camera-b coordinates)."""
    if ctx.c2w_cmd is None: return np.eye(4)
    return np.linalg.inv(np.asarray(ctx.c2w_cmd[b], float)) @ np.asarray(ctx.c2w_cmd[a], float)


def _matches(ctx, a, b):
    store = ctx.cache.setdefault('revisit_matches', {})
    if (a, b) not in store: store[(a, b)] = R.get_matcher().match(ctx.frames[a], ctx.frames[b])
    return store[(a, b)]


def estimate_relative_pose(ctx, a, b, h_ratio=0.8):
    """Relative pose b<-a from matches. Near-zero baseline (the ideal revisit) makes E degenerate (its decomposition can
    flip by 180 deg), so — as in ORB-SLAM / Torr's model selection — the rotation-only homography K R K^-1 is used when
    it explains at least `h_ratio` as many matches (2 px) as the essential matrix (1 px); h_ratio=0 forces E."""
    import cv2
    m = _matches(ctx, a, b); pa, pb = m['matched_kp_a'], m['matched_kp_b']
    out = {'num_matches': int(m['num_matches']), 'median_disp_px': float(np.median(np.linalg.norm(pb - pa, axis=1))) if len(pa) else None}
    if len(pa) < 8: out.update(ok=False, reason='fewer than 8 matches'); return out
    Ka, Kb = _K(ctx, a), _K(ctx, b)
    # rotation-only model
    H, hm = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
    Rh = None; h_res = None; h_inl = int(hm.sum()) if hm is not None else 0
    if H is not None:
        Rh = np.linalg.inv(Kb) @ H @ Ka; U, _, Vt = np.linalg.svd(Rh); Rh = U @ Vt
        if np.linalg.det(Rh) < 0: Rh = -Rh
        proj = (Kb @ Rh @ np.linalg.inv(Ka) @ np.c_[pa, np.ones(len(pa))].T).T; proj = proj[:, :2] / proj[:, 2:3]
        h_res = float(np.median(np.linalg.norm(proj - pb, axis=1)))
    # general model (normalised coordinates so different K per frame is allowed)
    na = cv2.undistortPoints(pa.reshape(-1, 1, 2).astype(np.float64), Ka, None).reshape(-1, 2)
    nb = cv2.undistortPoints(pb.reshape(-1, 1, 2).astype(np.float64), Kb, None).reshape(-1, 2)
    E, em = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.0 / Ka[0, 0])
    Re = te = None; e_inl = 0
    if E is not None and E.shape == (3, 3):
        e_inl = int(em.sum()) if em is not None else 0
        _, Re, te, _ = cv2.recoverPose(E, na, nb, np.eye(3), mask=em.copy() if em is not None else None)
        te = te.ravel()
    use_h = Rh is not None and (Re is None or (h_ratio > 0 and h_inl >= h_ratio * max(e_inl, 1)))
    if Rh is None and Re is None: out.update(ok=False, reason='no homography / essential model'); return out
    out.update(ok=True, model='homography_rotation' if use_h else 'essential', R=(Rh if use_h else Re), t_dir=None if use_h else te,
               homography_residual_px=h_res, homography_inliers=h_inl, essential_inliers=e_inl)
    return out


def pnp_relative_pose(ctx, a, b):
    """Metric T_b<-a from ctx.depth[a] (back-projected keypoints of the visit frame) + PnP-RANSAC on the return frame."""
    import cv2
    if ctx.depth is None: return None
    try:
        D = ctx.depth[a]                                       # ndarray or dict-like store keyed by step
    except (KeyError, IndexError):
        D = None
    if D is None: return {'ok': False, 'reason': f'no depth at visit step {a}'}
    m = _matches(ctx, a, b); pa, pb = m['matched_kp_a'], m['matched_kp_b']
    D = np.asarray(D, float); Ka, Kb = _K(ctx, a), _K(ctx, b)
    u = np.clip(np.round(pa[:, 0]).astype(int), 0, D.shape[1] - 1); v = np.clip(np.round(pa[:, 1]).astype(int), 0, D.shape[0] - 1)
    z = D[v, u]; good = np.isfinite(z) & (z > 1e-3)
    if good.sum() < 6: return {'ok': False, 'reason': 'fewer than 6 matches with valid depth'}
    X = np.c_[(pa[good, 0] - Ka[0, 2]) / Ka[0, 0] * z[good], (pa[good, 1] - Ka[1, 2]) / Ka[1, 1] * z[good], z[good]]
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X.astype(np.float64), pb[good].astype(np.float64), Kb, None,
                                             reprojectionError=2.0, iterationsCount=1000, confidence=0.999)
    if not ok: return {'ok': False, 'reason': 'PnP failed'}
    return {'ok': True, 'R': cv2.Rodrigues(rvec)[0], 't': tvec.ravel(), 'inliers': 0 if inl is None else int(len(inl)), 'points': int(good.sum())}


def _per_kp(ctx, fn):
    rows = []
    for kp in _keypoints(ctx):
        a, b = int(kp['visit']), int(kp['return'])
        r = fn(a, b); r = r if isinstance(r, dict) else {'value': r}
        rows.append({'keypoint': kp.get('id'), 'visit': a, 'return': b, **r})
    vals = [r['value'] for r in rows if r.get('value') is not None and np.isfinite(r['value'])]
    if not vals: raise NotApplicable('no keypoint produced a value: ' + '; '.join(str(r.get('reason')) for r in rows)[:300])
    return float(np.mean(vals)), rows


# ----------------------------------------------------------------------------------------------------------------------
# revisit.pose
# ----------------------------------------------------------------------------------------------------------------------

@operator('revisit.pose.rot_err_deg', 'revisit.pose', inputs=('frames', 'keypoints', 'K'), higher_is_better=False, unit='deg',
          source=SRC, requires=('lightglue', 'cv2'), gpu=True)
def pose_rot_err(ctx):
    """Angle between the image-estimated rotation visit->return and the commanded one (identity for an exact revisit)."""
    def f(a, b):
        e = estimate_relative_pose(ctx, a, b)
        if not e.get('ok') or e.get('R') is None: return {'value': None, 'reason': e.get('reason', 'no model')}
        Rc = cmd_relative(ctx, a, b)[:3, :3]
        return {'value': rot_angle_deg(Rc.T @ e['R']), 'model': e['model'], 'num_matches': e['num_matches'],
                'homography_residual_px': e['homography_residual_px'], 'homography_inliers': e['homography_inliers'], 'essential_inliers': e['essential_inliers']}
    v, rows = _per_kp(ctx, f)
    return Result('revisit.pose.rot_err_deg', v, per_item=rows)


@operator('revisit.pose.pixel_disp_px', 'revisit.pose', inputs=('frames', 'keypoints'), higher_is_better=False, unit='px',
          source=SRC, requires=('lightglue',), gpu=True)
def pose_pixel_disp(ctx):
    """Median displacement of matched keypoints between the visit and return frames (0 for a pixel-exact revisit)."""
    def f(a, b):
        m = _matches(ctx, a, b)
        if m['num_matches'] < 8: return {'value': None, 'reason': 'fewer than 8 matches', 'num_matches': m['num_matches']}
        d = np.linalg.norm(m['matched_kp_b'] - m['matched_kp_a'], axis=1)
        return {'value': float(np.median(d)), 'p90_px': float(np.percentile(d, 90)), 'num_matches': m['num_matches'],
                'diag_px': float(np.hypot(*ctx.frames[a].shape[:2]))}
    v, rows = _per_kp(ctx, f)
    return Result('revisit.pose.pixel_disp_px', v, per_item=rows,
                  raw={'normalised_by_diagonal': float(np.mean([r['value'] / r['diag_px'] for r in rows if r.get('value') is not None]))})


@operator('revisit.pose.match_ratio', 'revisit.pose', inputs=('frames', 'keypoints'), higher_is_better=True, unit='ratio',
          source=SRC, requires=('lightglue',), gpu=True)
def pose_match_ratio(ctx):
    """LightGlue match ratio |M|/min(Ka,Kb) between visit and return frames (R2M local-geometry measure on the keypoint pair)."""
    v, rows = _per_kp(ctx, lambda a, b: {'value': float(_matches(ctx, a, b)['match_ratio'])})
    return Result('revisit.pose.match_ratio', v, per_item=rows)


@operator('revisit.pose.trans_dir_err_deg', 'revisit.pose', inputs=('frames', 'keypoints', 'K', 'c2w_cmd'), higher_is_better=False,
          unit='deg', source=SRC, requires=('lightglue', 'cv2'), gpu=True)
def pose_trans_dir_err(ctx, min_cmd_baseline_m=0.05):
    """Translation-direction error (essential matrix) — only defined where the commanded visit/return baseline is
    >= min_cmd_baseline_m; an exact revisit has zero baseline, so this is normally n/a (use trans_err_m with depth)."""
    def f(a, b):
        tc = cmd_relative(ctx, a, b)[:3, 3]
        if np.linalg.norm(tc) < min_cmd_baseline_m: return {'value': None, 'reason': 'commanded baseline ~0 (exact revisit)'}
        e = estimate_relative_pose(ctx, a, b, h_ratio=0.0)
        if not e.get('ok') or e.get('t_dir') is None: return {'value': None, 'reason': e.get('reason', 'no essential model')}
        c = np.clip(abs(float(tc @ e['t_dir'])) / np.linalg.norm(tc) / max(np.linalg.norm(e['t_dir']), 1e-12), -1, 1)
        return {'value': float(np.degrees(np.arccos(c)))}
    v, rows = _per_kp(ctx, f)
    return Result('revisit.pose.trans_dir_err_deg', v, per_item=rows)


@operator('revisit.pose.trans_err_m', 'revisit.pose', inputs=('frames', 'keypoints', 'K', 'depth'), higher_is_better=False,
          unit='m', source=SRC, requires=('lightglue', 'cv2'), gpu=True)
def pose_trans_err(ctx):
    """Metric translation error of the return camera: PnP of visit-frame depth points into the return frame vs commanded."""
    def f(a, b):
        p = pnp_relative_pose(ctx, a, b)
        if not p or not p.get('ok'): return {'value': None, 'reason': (p or {}).get('reason')}
        Tc = cmd_relative(ctx, a, b)
        return {'value': float(np.linalg.norm(p['t'] - Tc[:3, 3])), 'pnp_rot_err_deg': rot_angle_deg(Tc[:3, :3].T @ p['R']),
                'inliers': p['inliers'], 'points': p['points']}
    v, rows = _per_kp(ctx, f)
    return Result('revisit.pose.trans_err_m', v, per_item=rows,
                  raw={'pnp_rot_err_deg': float(np.mean([r['pnp_rot_err_deg'] for r in rows if r.get('value') is not None]))})


@operator('revisit.pose.cmd_rot_err_deg', 'revisit.pose', inputs=('keypoints', 'c2w_cmd'), higher_is_better=False, unit='deg', source=SRC)
def pose_cmd_rot(ctx):
    """Sanity: commanded rotation difference between visit and return (0 when the action program closes the loop)."""
    v, rows = _per_kp(ctx, lambda a, b: rot_angle_deg(cmd_relative(ctx, a, b)[:3, :3]))
    return Result('revisit.pose.cmd_rot_err_deg', v, per_item=rows)


@operator('revisit.pose.cmd_trans_err_m', 'revisit.pose', inputs=('keypoints', 'c2w_cmd'), higher_is_better=False, unit='m', source=SRC)
def pose_cmd_trans(ctx):
    """Sanity: commanded camera-centre distance between visit and return."""
    v, rows = _per_kp(ctx, lambda a, b: float(np.linalg.norm(np.asarray(ctx.c2w_cmd[a])[:3, 3] - np.asarray(ctx.c2w_cmd[b])[:3, 3])))
    return Result('revisit.pose.cmd_trans_err_m', v, per_item=rows)


def umeyama(src, dst, with_scale=True):
    """Similarity transform minimising ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0); xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src)); D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: D[2, 2] = -1
    R_ = U @ D @ Vt; s = (S * np.diag(D)).sum() / max(xs.var(0).sum(), 1e-12) if with_scale else 1.0
    return s, R_, mu_d - s * R_ @ mu_s


def _est_aligned(ctx):
    def build():
        est = np.asarray(ctx.c2w_est, float); cmd = np.asarray(ctx.c2w_cmd, float); n = min(len(est), len(cmd))
        s, R_, t = umeyama(est[:n, :3, 3], cmd[:n, :3, 3])
        al = est[:n].copy(); al[:, :3, :3] = R_ @ est[:n, :3, :3]; al[:, :3, 3] = (s * (R_ @ est[:n, :3, 3].T)).T + t
        ate = float(np.sqrt(np.mean(np.sum((al[:, :3, 3] - cmd[:n, :3, 3]) ** 2, 1))))
        return {'aligned': al, 'scale': float(s), 'ate_rmse_m': ate}
    return cached(ctx, 'revisit_est_aligned', build)


@operator('revisit.pose.est_drift_rot_deg', 'revisit.pose', inputs=('keypoints', 'c2w_est', 'c2w_cmd'), higher_is_better=False,
          unit='deg', source=SRC)
def pose_est_drift_rot(ctx):
    """Rotation drift between visit and return in an image-estimated trajectory (e.g. from a pose operator), relative to
    the commanded relative rotation."""
    al = _est_aligned(ctx)['aligned']
    def f(a, b):
        if b >= len(al): return {'value': None, 'reason': 'estimate shorter than trajectory'}
        Re = (np.linalg.inv(al[b]) @ al[a])[:3, :3]
        return rot_angle_deg(cmd_relative(ctx, a, b)[:3, :3].T @ Re)
    v, rows = _per_kp(ctx, f)
    return Result('revisit.pose.est_drift_rot_deg', v, per_item=rows, raw={k: v_ for k, v_ in _est_aligned(ctx).items() if k != 'aligned'})


@operator('revisit.pose.est_drift_trans_m', 'revisit.pose', inputs=('keypoints', 'c2w_est', 'c2w_cmd'), higher_is_better=False,
          unit='m', source=SRC)
def pose_est_drift_trans(ctx):
    """Camera-centre drift visit->return in the Sim(3)-aligned estimated trajectory minus the commanded displacement."""
    al = _est_aligned(ctx)['aligned']; cmd = np.asarray(ctx.c2w_cmd, float)
    def f(a, b):
        if b >= len(al): return {'value': None, 'reason': 'estimate shorter than trajectory'}
        return float(np.linalg.norm((al[b, :3, 3] - al[a, :3, 3]) - (cmd[b, :3, 3] - cmd[a, :3, 3])))
    v, rows = _per_kp(ctx, f)
    return Result('revisit.pose.est_drift_trans_m', v, per_item=rows, raw={k: v_ for k, v_ in _est_aligned(ctx).items() if k != 'aligned'})


# ----------------------------------------------------------------------------------------------------------------------
# revisit.image
# ----------------------------------------------------------------------------------------------------------------------

@operator('revisit.image.psnr', 'revisit.image', inputs=('frames', 'keypoints'), unit='dB', source=SRC)
def image_psnr(ctx):
    v, rows = _per_kp(ctx, lambda a, b: R.compute_psnr_ssim(ctx.frames[a], ctx.frames[b])['psnr'])
    return Result('revisit.image.psnr', v, per_item=rows, note='identical frames capped at 100 dB')


@operator('revisit.image.ssim', 'revisit.image', inputs=('frames', 'keypoints'), unit='', source=SRC)
def image_ssim(ctx):
    v, rows = _per_kp(ctx, lambda a, b: R.compute_psnr_ssim(ctx.frames[a], ctx.frames[b])['ssim'])
    return Result('revisit.image.ssim', v, per_item=rows)


@operator('revisit.image.lpips', 'revisit.image', inputs=('frames', 'keypoints'), higher_is_better=False, source=SRC,
          requires=('lpips',), gpu=True)
def image_lpips(ctx):
    v, rows = _per_kp(ctx, lambda a, b: R.get_lpips('alex')(ctx.frames[a], ctx.frames[b]))
    return Result('revisit.image.lpips', v, per_item=rows)


@operator('revisit.image.dreamsim', 'revisit.image', inputs=('frames', 'keypoints'), higher_is_better=False, source=SRC,
          requires=('dreamsim',), gpu=True)
def image_dreamsim(ctx):
    v, rows = _per_kp(ctx, lambda a, b: R.get_dreamsim()(ctx.frames[a], ctx.frames[b]))
    return Result('revisit.image.dreamsim', v, per_item=rows, note='DreamSim distance (ensemble), 0 = identical')


@operator('revisit.image.dinov2', 'revisit.image', inputs=('frames', 'keypoints'), source=SRC, requires=('dinov2',), gpu=True)
def image_dinov2(ctx):
    v, rows = _per_kp(ctx, lambda a, b: R.get_dino().similarity(ctx.frames[a], ctx.frames[b]))
    return Result('revisit.image.dinov2', v, per_item=rows)


# ----------------------------------------------------------------------------------------------------------------------
# revisit.object  (画面物体重合)
# ----------------------------------------------------------------------------------------------------------------------

def load_mask(spec, shape):
    import cv2
    if isinstance(spec, dict):
        path, gid = spec.get('path'), spec.get('id')
    else:
        path, gid = spec, None
    if isinstance(path, np.ndarray):
        m = path
    elif str(path).endswith('.npy'):
        m = np.load(path)
    elif str(path).endswith('.npz'):
        with np.load(path) as z: m = z[z.files[0]] if gid is None or str(gid) not in z.files else z[str(gid)]
    else:
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None: raise FileNotFoundError(path)
        if m.ndim == 3: m = m[..., 0]
    m = (m == gid) if gid is not None else (m > 0)
    if m.shape != tuple(shape):
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m.astype(bool)


def mask_iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _centroid(m):
    yy, xx = np.nonzero(m)
    return np.array([xx.mean(), yy.mean()]) if len(xx) else None


def _visible(obj, step):
    va = obj.get('visible_at') or {}
    return va.get(step, va.get(str(step)))


def _label_candidates(ctx, step, label, max_candidates=5):
    """GroundingDINO ('<label> .', box 0.25 / text 0.20) + SAM2 box masks on frame `step` (cached)."""
    store = ctx.cache.setdefault('revisit_label_dets', {})
    key = (step, label)
    if key not in store:
        gs = R.get_grounded_sam(); img = R._to_uint8(ctx.frames[step])
        xyxy, logits, _ = gs.detect_boxes(img, f'{label} .')
        xyxy, logits = xyxy[:max_candidates], logits[:max_candidates]
        masks, _ = gs.sam_masks(img, xyxy.detach().cpu().numpy(), key=(id(ctx), 'step', step)) if len(xyxy) else (np.zeros((0,) + img.shape[:2], bool), None)
        store[key] = [{'mask': masks[i], 'score': float(logits[i].item())} for i in range(len(masks))]
    return store[key]


def _object_rows(ctx, iou_threshold=0.5):
    def build():
        if not ctx.objects: raise NotApplicable('case has no annotated objects')
        rows = []; precision_rows = []
        for kp in _keypoints(ctx):
            a, b = int(kp['visit']), int(kp['return']); shape = ctx.frames[a].shape[:2]
            objs = [(o, _visible(o, a)) for o in ctx.objects]; objs = [(o, s) for o, s in objs if s is not None]
            gts = {}
            for o, spec in objs:
                gt = load_mask(spec, shape)
                if gt.sum() < 50: continue
                gts[o['global_id']] = (o, gt)
                ca = _label_candidates(ctx, a, o['label']); cb = _label_candidates(ctx, b, o['label'])
                ia = [mask_iou(gt, c['mask']) for c in ca]; ib = [mask_iou(gt, c['mask']) for c in cb]
                va = ca[int(np.argmax(ia))]['mask'] if ca and max(ia) > 0 else None
                vb = cb[int(np.argmax(ib))]['mask'] if cb and max(ib) > 0 else None
                found_v = bool(ia and max(ia) >= iou_threshold)          # detector re-finds the annotated object at visit
                row = {'keypoint': kp.get('id'), 'visit': a, 'return': b, 'global_id': o['global_id'], 'label': o['label'],
                       'gt_area_px': int(gt.sum()), 'iou_gt_visit': max(ia) if ia else 0.0, 'iou_gt_return': max(ib) if ib else 0.0,
                       'found_at_visit': found_v, 'present_vs_gt': bool(ib and max(ib) >= iou_threshold)}
                if found_v:                                              # consistency is judged where the detector works
                    row['pair_iou'] = mask_iou(va, vb) if vb is not None else 0.0
                    row['kept'] = row['pair_iou'] >= iou_threshold
                if vb is not None and va is not None:
                    c0, c1 = _centroid(va), _centroid(vb)
                    row['centroid_shift_px'] = float(np.linalg.norm(c1 - c0))
                    row['area_consistency'] = float(min(va.sum(), vb.sum()) / max(va.sum(), vb.sum(), 1))
                rows.append(row)
            # precision: every same-label detection in the return frame should already exist in the visit frame
            # (relative, ideal 1); 'matched_gt' additionally checks it against the annotation (annotation misses lower it)
            for label in sorted({o['label'] for o, _ in gts.values()}):
                same = [gt for o, gt in gts.values() if o['label'] == label]
                vis = [c['mask'] for c in _label_candidates(ctx, a, label)]
                for c in _label_candidates(ctx, b, label):
                    bv = max([mask_iou(c['mask'], m) for m in vis] or [0.0]); bg = max(mask_iou(c['mask'], g) for g in same)
                    precision_rows.append({'keypoint': kp.get('id'), 'label': label, 'best_iou_visit': bv, 'matched': bv >= iou_threshold,
                                           'best_iou_gt': bg, 'matched_gt': bg >= iou_threshold})
        if not rows: raise NotApplicable('no annotated object visible at any keypoint visit frame')
        return rows, precision_rows
    return cached(ctx, f'revisit_object_rows_{iou_threshold}', build)


def _obj_metric(ctx, name, key, reduce=np.mean):
    rows, _ = _object_rows(ctx)
    vals = [r[key] for r in rows if key in r]
    if not vals: raise NotApplicable(f'no object has {key} (detector found none of the annotated objects at visit)')
    return Result(name, float(reduce(vals)), per_item=[{k: v for k, v in r.items()} for r in rows],
                  raw={'objects': len(rows), 'objects_scored': len(vals), 'keypoints': len({r['keypoint'] for r in rows}),
                       'oracle_ceiling': float(np.mean([r['iou_gt_visit'] for r in rows]))})


@operator('revisit.object.mask_iou', 'revisit.object', inputs=('frames', 'keypoints', 'objects'), unit='IoU', source=SRC,
          requires=('groundingdino', 'sam2'), gpu=True)
def object_mask_iou(ctx):
    """IoU between each annotated object's GT mask at the visit frame and its label-prompted re-segmentation in the
    return frame (0 when not re-detected); mean over objects x keypoints. Absolute: bounded by detector / annotation
    agreement (raw 'oracle_ceiling' = the same IoU measured on the visit frame itself)."""
    return _obj_metric(ctx, 'revisit.object.mask_iou', 'iou_gt_return')


@operator('revisit.object.pair_iou', 'revisit.object', inputs=('frames', 'keypoints', 'objects'), unit='IoU', source=SRC,
          requires=('groundingdino', 'sam2'), gpu=True)
def object_pair_iou(ctx):
    """IoU between the same object's predicted masks in the visit and return frames (same detector on both frames, so
    segmentation bias cancels; GT only selects which instance). Objects the detector cannot re-find even in the visit
    frame are skipped; a visit detection with no return detection counts 0."""
    return _obj_metric(ctx, 'revisit.object.pair_iou', 'pair_iou')


@operator('revisit.object.centroid_shift_px', 'revisit.object', inputs=('frames', 'keypoints', 'objects'), higher_is_better=False,
          unit='px', source=SRC, requires=('groundingdino', 'sam2'), gpu=True)
def object_centroid_shift(ctx):
    return _obj_metric(ctx, 'revisit.object.centroid_shift_px', 'centroid_shift_px')


@operator('revisit.object.area_consistency', 'revisit.object', inputs=('frames', 'keypoints', 'objects'), unit='ratio',
          source=SRC, requires=('groundingdino', 'sam2'), gpu=True)
def object_area_consistency(ctx):
    """min(area_visit, area_return) / max(...) of the matched object masks."""
    return _obj_metric(ctx, 'revisit.object.area_consistency', 'area_consistency')


@operator('revisit.object.recall', 'revisit.object', inputs=('frames', 'keypoints', 'objects'), unit='ratio', source=SRC,
          requires=('groundingdino', 'sam2'), gpu=True)
def object_recall(ctx):
    """Of the annotated objects the detector finds in the visit frame, the fraction still present at the same place in
    the return frame (pair IoU >= 0.5). raw: recall against the annotation masks directly."""
    r = _obj_metric(ctx, 'revisit.object.recall', 'kept')
    rows, _ = _object_rows(ctx); r.raw['recall_vs_gt'] = float(np.mean([x['present_vs_gt'] for x in rows])); return r


@operator('revisit.object.precision', 'revisit.object', inputs=('frames', 'keypoints', 'objects'), unit='ratio', source=SRC,
          requires=('groundingdino', 'sam2'), gpu=True)
def object_precision(ctx):
    """Fraction of return-frame detections (labels of the visible annotated objects) that coincide (IoU >= 0.5) with a
    same-label detection of the visit frame — objects that appear / duplicate after the excursion lower it.
    raw: the same against the annotation masks (annotation misses also lower that one)."""
    _, prec = _object_rows(ctx)
    if not prec: raise NotApplicable('no detections in return frames')
    return Result('revisit.object.precision', float(np.mean([p['matched'] for p in prec])), per_item=prec,
                  raw={'precision_vs_gt': float(np.mean([p['matched_gt'] for p in prec]))})


def _amg_masks(ctx, step, points_per_side=32, min_area_ratio=0.002):
    store = ctx.cache.setdefault('revisit_amg', {})
    if step not in store:
        predictor = R.load_sam2(R.device())                    # also puts the vendored sam2 on sys.path
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        gen = R._model('sam2_amg', lambda: SAM2AutomaticMaskGenerator(predictor.model, points_per_side=points_per_side))
        img = R._to_uint8(ctx.frames[step]); area = img.shape[0] * img.shape[1]
        store[step] = [m['segmentation'] for m in gen.generate(img) if m['area'] >= min_area_ratio * area]
    return store[step]


@operator('revisit.object.agnostic_iou', 'revisit.object', inputs=('frames', 'keypoints'), unit='IoU', source=SRC,
          requires=('sam2',), gpu=True)
def object_agnostic_iou(ctx, iou_threshold=0.5):
    """Label-free: SAM2 automatic masks of the visit and return frames, Hungarian matching on IoU; area-weighted mean
    matched IoU over all masks of both frames (unmatched masks count 0). raw: recall/precision at IoU 0.5."""
    from scipy.optimize import linear_sum_assignment
    def f(a, b):
        ma, mb = _amg_masks(ctx, a), _amg_masks(ctx, b)
        if not ma or not mb: return {'value': 0.0 if (ma or mb) else None, 'reason': 'no masks'}
        iou = np.array([[mask_iou(x, y) for y in mb] for x in ma]); r_, c_ = linear_sum_assignment(-iou)
        ia = np.zeros(len(ma)); ib = np.zeros(len(mb)); ia[r_] = iou[r_, c_]; ib[c_] = iou[r_, c_]
        wa = np.array([m.sum() for m in ma], float); wb = np.array([m.sum() for m in mb], float)
        return {'value': float((ia @ wa + ib @ wb) / (wa.sum() + wb.sum())), 'masks_visit': len(ma), 'masks_return': len(mb),
                'recall_at_0.5': float((ia >= iou_threshold).mean()), 'precision_at_0.5': float((ib >= iou_threshold).mean())}
    v, rows = _per_kp(ctx, f)
    return Result('revisit.object.agnostic_iou', v, per_item=rows)


# ----------------------------------------------------------------------------------------------------------------------
# revisit.excursion
# ----------------------------------------------------------------------------------------------------------------------

def mirror_pairs(ctx, rot_tol_deg=0.1, trans_tol_m=1e-3):
    """Inside each keypoint window [visit, return], poses reached during action k and again (mirrored) during its
    inverse. Frames are grouped into identical-pose classes; a class split into >= 2 contiguous runs yields the pair
    (last frame of first run, first frame of last run). The class of the keypoint pose itself is excluded (that is the
    main revisit)."""
    def build():
        c2w = np.asarray(ctx.c2w_cmd, float); out = []
        for kp in _keypoints(ctx):
            a, b = int(kp['visit']), int(kp['return']); steps = list(range(a, b + 1)); classes = []
            for s in steps:
                for cl in classes:
                    r = cl['rep']
                    if np.linalg.norm(c2w[s, :3, 3] - c2w[r, :3, 3]) <= trans_tol_m and rot_angle_deg(c2w[r, :3, :3].T @ c2w[s, :3, :3]) <= rot_tol_deg:
                        cl['steps'].append(s); break
                else:
                    classes.append({'rep': s, 'steps': [s]})
            for cl in classes:
                if cl['rep'] == a: continue
                runs = np.split(np.array(cl['steps']), np.where(np.diff(cl['steps']) > 1)[0] + 1)
                if len(runs) >= 2:
                    out.append({'keypoint': kp.get('id'), 'pair': (int(runs[0][-1]), int(runs[-1][0])), 'gap': int(runs[-1][0] - runs[0][-1])})
        return out
    return cached(ctx, 'revisit_mirror_pairs', build)


def _excursion(ctx, name, fn, **kw):
    prs = mirror_pairs(ctx)
    prs = [p for p in prs if p['pair'][1] < len(ctx.frames)]
    if not prs: raise NotApplicable('no mirrored action/inverse poses inside the keypoint windows')
    rows = [{**p, 'value': float(fn(*p['pair']))} for p in prs]
    by_kp = {}
    for r in rows: by_kp.setdefault(r['keypoint'], []).append(r['value'])
    return Result(name, float(np.mean([r['value'] for r in rows])), per_item=rows,
                  raw={'pairs': len(rows), 'per_keypoint_mean': {str(k): float(np.mean(v)) for k, v in by_kp.items()}})


@operator('revisit.excursion.psnr', 'revisit.excursion', inputs=('frames', 'keypoints', 'c2w_cmd'), unit='dB', source=SRC)
def excursion_psnr(ctx):
    return _excursion(ctx, 'revisit.excursion.psnr', lambda a, b: R.compute_psnr_ssim(ctx.frames[a], ctx.frames[b])['psnr'])


@operator('revisit.excursion.lpips', 'revisit.excursion', inputs=('frames', 'keypoints', 'c2w_cmd'), higher_is_better=False,
          source=SRC, requires=('lpips',), gpu=True)
def excursion_lpips(ctx):
    return _excursion(ctx, 'revisit.excursion.lpips', lambda a, b: R.get_lpips('alex')(ctx.frames[a], ctx.frames[b]))


@operator('revisit.excursion.dinov2', 'revisit.excursion', inputs=('frames', 'keypoints', 'c2w_cmd'), source=SRC,
          requires=('dinov2',), gpu=True)
def excursion_dinov2(ctx):
    return _excursion(ctx, 'revisit.excursion.dinov2', lambda a, b: R.get_dino().similarity(ctx.frames[a], ctx.frames[b]))


@operator('revisit.excursion.pixel_disp_px', 'revisit.excursion', inputs=('frames', 'keypoints', 'c2w_cmd'), higher_is_better=False,
          unit='px', source=SRC, requires=('lightglue',), gpu=True)
def excursion_pixel_disp(ctx):
    def f(a, b):
        m = _matches(ctx, a, b)
        return float(np.median(np.linalg.norm(m['matched_kp_b'] - m['matched_kp_a'], axis=1))) if m['num_matches'] >= 8 else float('nan')
    r = _excursion(ctx, 'revisit.excursion.pixel_disp_px', f)
    vals = [x['value'] for x in r.per_item if np.isfinite(x['value'])]
    r.value = float(np.mean(vals)) if vals else None
    if r.value is None: r.status, r.note = 'n/a', 'fewer than 8 matches on every mirrored pair'
    return r
