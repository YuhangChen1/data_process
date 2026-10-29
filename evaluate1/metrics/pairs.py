"""R2M-Bench pair mining (revisit / gap-matched baseline / short-range) and MemoryGain / NMR aggregation.

Faithful port of R2M-Bench (refs: /mnt/cfs/cyh/tmp/refs/R2MBench-main):
  - revisit detection        src/eval_revisit.py:67-432   (yaw + 3D translation, motion segments, adaptive P95 tolerances,
                                                          3D fallback, static-tail trim)
  - pair sampling            src/sample_pairs.py:90-210   (min_gap_ratio 0.2, max_eval_pairs 100, seed 42,
                                                          exclusion_radius max(10, T//20), 10 temporal bins, gap tol 0.3)
  - baseline / short pairs   src/pair_sampling.py:89-520  (verbatim logic; debug prints dropped)
  - MG / NMR                 src/compute_nmr_summary.py:210-283 (video-balanced; eps 1e-8; DR<=eps excluded;
                                                          Overall = mean over families of mean metric NMR)
                             src/eval_gpu_metrics.py:82-102 (per-video NMR convention, MIN_DYNAMIC_RANGE 1e-5) kept as
                             `per_video_nmr` for parity with R2M per-video result files.

Only adaptation: R2M reads `extrinsic` (OpenCV camera-to-world) in a y-down world and takes yaw = atan2(R[0,2], R[0,0]).
Our cameras live in a Z-up canonical world, so `r2m_pose_data` first maps the world so that its up vector becomes -y
(rotation A, det +1) — after that every formula runs unchanged.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# world adapter
# ----------------------------------------------------------------------------------------------------------------------

def _world_to_r2m(up=(0., 0., 1.)) -> np.ndarray:
    """Rotation A with A @ up = (0,-1,0) (R2M / OpenCV y-down world). For up=+Z: A=[[1,0,0],[0,0,-1],[0,1,0]]."""
    u = np.asarray(up, float); u /= np.linalg.norm(u)
    target = np.array([0., -1., 0.])
    if np.allclose(u, [0, 0, 1]): return np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    if np.allclose(u, target): return np.eye(3)
    v = np.cross(u, target); s = np.linalg.norm(v); c = float(u @ target)
    if s < 1e-9: return np.diag([1., -1., -1.])          # u = +y: rotate 180 deg about x
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def r2m_pose_data(c2w: np.ndarray, up=(0., 0., 1.)) -> Dict[str, dict]:
    """[T,4,4] OpenCV camera-to-world -> R2M trajectory dict {"0": {"extrinsic": 4x4}, ...} in a y-down world."""
    A = np.eye(4); A[:3, :3] = _world_to_r2m(up)
    return {str(i): {'extrinsic': (A @ np.asarray(m, float)).tolist()} for i, m in enumerate(c2w)}


# ----------------------------------------------------------------------------------------------------------------------
# revisit detection: src/eval_revisit.py
# ----------------------------------------------------------------------------------------------------------------------

def _circular_yaw_distance_deg(a: float, b: float) -> float:              # eval_revisit.py:67
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _extract_frame_poses(pose_data, sorted_keys):                          # eval_revisit.py:84
    frame_poses = {}
    for key in sorted_keys:
        idx = int(key); ext = pose_data[key]['extrinsic']
        yaw_deg = math.atan2(ext[0][2], ext[0][0]) * 180.0 / math.pi
        frame_poses[idx] = {'yaw': yaw_deg, 'tx': ext[0][3], 'ty': ext[1][3], 'tz': ext[2][3]}
    return frame_poses


def _detect_motion_segments(frame_poses, sorted_keys):                     # eval_revisit.py:96
    keys = [int(k) for k in sorted_keys]
    motion_phases = []; current_phase = None; phase_start = 0
    for i in range(1, len(keys)):
        idx = keys[i]; prev_idx = keys[i - 1]; cur = frame_poses[idx]; prev = frame_poses[prev_idx]
        dtx = cur['tx'] - prev['tx']; dty = cur['ty'] - prev['ty']; dtz = cur['tz'] - prev['tz']
        dyaw = cur['yaw'] - prev['yaw']
        translation_dist = math.sqrt(dtx * dtx + dty * dty + dtz * dtz)
        if translation_dist > 0.01 and abs(dyaw) < 0.5:
            abs_vals = {'tx': abs(dtx), 'ty': abs(dty), 'tz': abs(dtz)}
            primary_axis = max(abs_vals, key=abs_vals.get)
            primary_val = {'tx': dtx, 'ty': dty, 'tz': dtz}[primary_axis]
            motion = 'W' if primary_val > 0 else 'S'
        elif abs(dyaw) > 0.5:
            motion = 'R' if dyaw > 0 else 'L'
        else:
            motion = '.'
        if motion != current_phase and motion != '.':
            if current_phase is not None:
                motion_phases.append((phase_start, keys[i - 1], current_phase))
            current_phase = motion; phase_start = idx
    if current_phase is not None:
        motion_phases.append((phase_start, keys[-1], current_phase))
    return motion_phases


def _find_revisit_pairs(frame_poses, segments, angle_tol, trans_tol, min_frame_gap=10):   # eval_revisit.py:137
    revisit_groups = []
    for later_idx in range(1, len(segments)):
        later_start, later_end, later_type = segments[later_idx]
        for earlier_idx in range(later_idx):
            earlier_start, earlier_end, earlier_type = segments[earlier_idx]
            pairs = []
            for later_frame in range(later_start, later_end + 1):
                lp = frame_poses[later_frame]; best_match = None; best_diff = float('inf')
                for earlier_frame in range(earlier_start, earlier_end + 1):
                    if abs(later_frame - earlier_frame) < min_frame_gap: continue
                    ep = frame_poses[earlier_frame]
                    yaw_diff = _circular_yaw_distance_deg(lp['yaw'], ep['yaw'])
                    pos_diff = math.sqrt((lp['tx'] - ep['tx']) ** 2 + (lp['ty'] - ep['ty']) ** 2 + (lp['tz'] - ep['tz']) ** 2)
                    total_diff = yaw_diff + pos_diff * 10
                    if yaw_diff <= angle_tol and pos_diff <= trans_tol and total_diff < best_diff:
                        best_diff = total_diff; best_match = earlier_frame
                if best_match is not None: pairs.append((later_frame, best_match))
            if pairs:
                revisit_groups.append({'name': f'Seg{later_idx}({later_type})<->Seg{earlier_idx}({earlier_type})', 'pairs': pairs})
    return revisit_groups


def _rotation_distance_deg(rot_a, rot_b):                                  # eval_revisit.py:197
    trace = sum(rot_a[i][j] * rot_b[i][j] for i in range(3) for j in range(3))
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))) * 180.0 / math.pi


def _find_revisit_pairs_3d(poses, sorted_keys, position_threshold, rotation_threshold_deg, min_frame_gap=10):  # :205
    keys = [int(k) for k in sorted_keys]; pairs = []; seen = set()
    for i in range(len(keys)):
        for j in range(i + min_frame_gap, len(keys)):
            fi, fj = keys[i], keys[j]; pi, pj = poses[fi], poses[fj]
            dist = math.sqrt((pi['tx'] - pj['tx']) ** 2 + (pi['ty'] - pj['ty']) ** 2 + (pi['tz'] - pj['tz']) ** 2)
            if dist > position_threshold: continue
            if _circular_yaw_distance_deg(pi['yaw'], pj['yaw']) > rotation_threshold_deg: continue
            if (fi, fj) not in seen:
                pairs.append((fj, fi)); seen.add((fi, fj))
    return [{'name': '3D_position_match', 'pairs': pairs}] if pairs else []


def _trim_static_tail(pose_data, sorted_keys, move_threshold=0.01, rot_threshold_deg=0.5):   # eval_revisit.py:247
    if len(sorted_keys) < 3: return sorted_keys, 0
    keys = list(sorted_keys); last_moving_idx = len(keys) - 1
    for i in range(len(keys) - 1, 0, -1):
        a = pose_data[keys[i]]['extrinsic']; b = pose_data[keys[i - 1]]['extrinsic']
        dist = math.sqrt((a[0][3] - b[0][3]) ** 2 + (a[1][3] - b[1][3]) ** 2 + (a[2][3] - b[2][3]) ** 2)
        rot = _rotation_distance_deg([a[r][:3] for r in range(3)], [b[r][:3] for r in range(3)])
        if dist > move_threshold or rot > rot_threshold_deg:
            last_moving_idx = i; break
    trimmed = len(keys) - 1 - last_moving_idx
    return (keys[:last_moving_idx + 1] if trimmed > 0 else keys), trimmed


def _compute_adaptive_thresholds(frame_poses, sorted_keys):                # eval_revisit.py:288
    keys = [int(k) for k in sorted_keys]; trans_deltas = []; yaw_deltas = []
    for i in range(1, len(keys)):
        cur = frame_poses[keys[i]]; prev = frame_poses[keys[i - 1]]
        trans_deltas.append(math.sqrt((cur['tx'] - prev['tx']) ** 2 + (cur['ty'] - prev['ty']) ** 2 + (cur['tz'] - prev['tz']) ** 2))
        yaw_deltas.append(_circular_yaw_distance_deg(cur['yaw'], prev['yaw']))
    if not trans_deltas: return 2.0, 0.1
    p95_trans = float(np.percentile(trans_deltas, 95)); p95_yaw = float(np.percentile(yaw_deltas, 95))
    return max(1.0, min(2.0 * p95_yaw, 5.0)), max(0.1, min(2.0 * p95_trans, 1.0))


def extract_revisit_pairs(pose_data, angle_tolerance=None, translation_tolerance=None):     # eval_revisit.py:328
    """pose_data: R2M trajectory dict (see r2m_pose_data). Returns (groups, n_pose_after_trim, threshold_info)."""
    sorted_keys = sorted(pose_data.keys(), key=lambda x: int(x))
    angle_is_adaptive = angle_tolerance is None; translation_is_adaptive = translation_tolerance is None
    trimmed_keys, trimmed_count = _trim_static_tail(pose_data, sorted_keys)
    frame_poses = _extract_frame_poses(pose_data, trimmed_keys)
    if angle_tolerance is None or translation_tolerance is None:
        auto_angle, auto_trans = _compute_adaptive_thresholds(frame_poses, trimmed_keys)
        if angle_tolerance is None: angle_tolerance = auto_angle
        if translation_tolerance is None: translation_tolerance = auto_trans
    segments = _detect_motion_segments(frame_poses, trimmed_keys)
    seg_groups = _find_revisit_pairs(frame_poses, segments, angle_tolerance, translation_tolerance)
    seg_count = sum(len(g['pairs']) for g in seg_groups) if seg_groups else 0
    method = 'segments'
    if seg_count < 3:
        pos3d = _find_revisit_pairs_3d(frame_poses, trimmed_keys, position_threshold=translation_tolerance,
                                       rotation_threshold_deg=angle_tolerance)
        pos3d_count = sum(len(g['pairs']) for g in pos3d) if pos3d else 0
        groups = pos3d if pos3d_count > seg_count else seg_groups
        if pos3d_count > seg_count: method = '3d_fallback'
    else:
        groups = seg_groups
    mode = 'adaptive' if angle_is_adaptive and translation_is_adaptive else ('mixed' if angle_is_adaptive or translation_is_adaptive else 'fixed')
    info = {'mode': mode, 'angle_tolerance_deg': float(angle_tolerance), 'translation_tolerance_m': float(translation_tolerance),
            'trimmed_static_tail': trimmed_count, 'segments': len(segments), 'method': method}
    return groups, len(trimmed_keys), info


def flatten_revisit_groups(revisit_groups):                                 # eval_revisit.py:418
    flat = []; seen = set()
    for group in revisit_groups:
        for later_frame, earlier_frame in group['pairs']:
            if (earlier_frame, later_frame) not in seen:
                flat.append((earlier_frame, later_frame, 0.0, 0.0)); seen.add((earlier_frame, later_frame))
    flat.sort(key=lambda x: (x[1], x[0]))
    return flat


# ----------------------------------------------------------------------------------------------------------------------
# baseline / short sampling: src/pair_sampling.py (verbatim logic)
# ----------------------------------------------------------------------------------------------------------------------

def _angle_diff_deg(a, b): return abs((a - b + 180.0) % 360.0 - 180.0)


def _translation_distance(p1, p2):
    return float(np.linalg.norm(np.array([p1['tx'], p1['ty'], p1['tz']]) - np.array([p2['tx'], p2['ty'], p2['tz']])))


def _make_temporal_bins(frames, n_video, num_temporal_bins):
    num_temporal_bins = max(1, int(num_temporal_bins)); bin_size = max(1, math.ceil(n_video / num_temporal_bins))
    bins = [[] for _ in range(num_temporal_bins)]
    for f in frames: bins[min(f // bin_size, num_temporal_bins - 1)].append(f)
    return bins


def _get_bin_index(frame_idx, n_video, num_temporal_bins):
    num_temporal_bins = max(1, int(num_temporal_bins)); bin_size = max(1, math.ceil(n_video / num_temporal_bins))
    return min(frame_idx // bin_size, num_temporal_bins - 1)


def _candidate_frames_from_expanded_bins(temporal_bins, target_bin, max_expand=None):
    num_bins = len(temporal_bins); max_expand = num_bins if max_expand is None else max_expand
    for radius in range(max_expand + 1):
        lo = max(0, target_bin - radius); hi = min(num_bins, target_bin + radius + 1)
        candidates = [f for b in range(lo, hi) for f in temporal_bins[b]]
        if len(candidates) >= 2: return candidates
    return [f for b in temporal_bins for f in b]


def sample_gap_matched_temporal_baselines(revisit_video_pairs, frame_poses, n_video, num_temporal_bins=10,
                                          exclusion_radius=10, gap_tolerance_ratio=0.3, min_gap_tolerance=3,
                                          max_attempts_per_pair=500, allow_duplicate=False, seed=None):
    """pair_sampling.py:89-422. Baseline (u, u+gap): same temporal bin as the revisit anchor, |gap - rev_gap| within
    max(10, 0.3 rev_gap), NotNearReturn, forward-only; relaxed (0.5 rev_gap) fallback until 20 baseline pairs."""
    rng = random.Random(seed)
    if not revisit_video_pairs or n_video <= 1:
        return [], {'num_revisit': len(revisit_video_pairs), 'num_baseline': 0, 'matched_rate': 0.0, 'reason': 'empty_input'}
    valid_revisit = []
    for a, b in revisit_video_pairs:
        if a == b or not (0 <= a < n_video and 0 <= b < n_video) or a not in frame_poses or b not in frame_poses: continue
        valid_revisit.append((a, b) if a < b else (b, a))
    if not valid_revisit:
        return [], {'num_revisit': len(revisit_video_pairs), 'num_valid_revisit': 0, 'num_baseline': 0, 'matched_rate': 0.0,
                    'reason': 'no_valid_revisit'}
    revisit_frame_set: Set[int] = set()
    for anchor, ret in valid_revisit: revisit_frame_set.update((anchor, ret))
    available_frames = [f for f in range(n_video) if f in frame_poses and f not in revisit_frame_set]
    if len(available_frames) < 2:
        return [], {'num_revisit': len(revisit_video_pairs), 'num_valid_revisit': len(valid_revisit), 'num_baseline': 0,
                    'matched_rate': 0.0, 'reason': 'too_few_available_frames'}
    temporal_bins = _make_temporal_bins(available_frames, n_video, num_temporal_bins)
    used_pairs: Set[Tuple[int, int]] = set(); baseline_pairs: List[Tuple[int, int]] = []
    num_no_candidate = num_attempt_failed = 0; gap_errors = []; trans_dists = []; yaw_diffs = []

    def ok(u, v):
        if u in revisit_frame_set or v in revisit_frame_set: return False
        return not any(abs(u - a) <= exclusion_radius and abs(v - r) <= exclusion_radius for a, r in valid_revisit)

    for anchor, ret in valid_revisit:
        rev_gap = ret - anchor
        if rev_gap <= 0: continue
        delta_t = max(10, int(round(gap_tolerance_ratio * rev_gap)))
        gap_low = max(1, rev_gap - delta_t); gap_high = min(n_video - 1, rev_gap + delta_t)
        target_bin = _get_bin_index(anchor, n_video, num_temporal_bins)
        cands = [f for f in _candidate_frames_from_expanded_bins(temporal_bins, target_bin) if f + gap_high < n_video]
        if not cands:
            cands = [f for f in _candidate_frames_from_expanded_bins(temporal_bins, target_bin) if f + gap_low < n_video]
        if not cands:
            num_no_candidate += 1; continue
        best_pair = None; best_score = float('inf')
        for _ in range(max_attempts_per_pair):
            u = rng.choice(cands); gap = rng.randint(gap_low, gap_high); v = u + gap
            if v >= n_video or v not in frame_poses or not ok(u, v): continue
            if not allow_duplicate and (u, v) in used_pairs: continue
            score = abs(gap - rev_gap)
            if score < best_score:
                best_score = score; best_pair = (u, v)
                if score == 0: break
        if best_pair is None:
            num_attempt_failed += 1; continue
        baseline_pairs.append(best_pair)
        if not allow_duplicate: used_pairs.add(best_pair)
        gap_errors.append(abs((best_pair[1] - best_pair[0]) - rev_gap))
        trans_dists.append(_translation_distance(frame_poses[best_pair[0]], frame_poses[best_pair[1]]))
        yu = frame_poses[best_pair[0]].get('yaw'); yv = frame_poses[best_pair[1]].get('yaw')
        if yu is not None and yv is not None: yaw_diffs.append(_angle_diff_deg(float(yu), float(yv)))

    MIN_BASELINE_PAIRS = 20; num_fallback = 0
    if len(baseline_pairs) < MIN_BASELINE_PAIRS and len(available_frames) >= 2:
        needed = MIN_BASELINE_PAIRS - len(baseline_pairs)
        fb_bins = _make_temporal_bins(available_frames, n_video, num_temporal_bins)
        for anchor, ret in valid_revisit:
            if num_fallback >= needed: break
            rev_gap = ret - anchor
            if rev_gap <= 0: continue
            relaxed = max(min_gap_tolerance, int(round(0.5 * rev_gap)))
            lo = max(1, rev_gap - relaxed); hi = min(n_video - 1, rev_gap + relaxed)
            tb = _get_bin_index(anchor, n_video, num_temporal_bins)
            cands = [f for f in _candidate_frames_from_expanded_bins(fb_bins, tb) if f + hi < n_video]
            if not cands: cands = [f for f in _candidate_frames_from_expanded_bins(fb_bins, tb) if f + lo < n_video]
            if not cands: continue
            for _ in range(max_attempts_per_pair):
                if num_fallback >= needed: break
                u = rng.choice(cands); gap = rng.randint(lo, hi); v = u + gap
                if v >= n_video or v not in frame_poses or (u, v) in used_pairs or not ok(u, v): continue
                baseline_pairs.append((u, v)); used_pairs.add((u, v)); num_fallback += 1
                gap_errors.append(abs(gap - rev_gap)); trans_dists.append(_translation_distance(frame_poses[u], frame_poses[v]))
                yu = frame_poses[u].get('yaw'); yv = frame_poses[v].get('yaw')
                if yu is not None and yv is not None: yaw_diffs.append(_angle_diff_deg(float(yu), float(yv)))
    stats = {'num_revisit': len(revisit_video_pairs), 'num_valid_revisit': len(valid_revisit), 'num_baseline': len(baseline_pairs),
             'num_fallback_baseline': num_fallback,
             'matched_rate': float(len(baseline_pairs) / len(valid_revisit)) if valid_revisit else 0.0,
             'exclusion_radius': int(exclusion_radius), 'num_temporal_bins': int(num_temporal_bins),
             'num_no_candidate': int(num_no_candidate), 'num_attempt_failed': int(num_attempt_failed),
             'mean_gap_error': float(np.mean(gap_errors)) if gap_errors else None,
             'mean_negative_translation': float(np.mean(trans_dists)) if trans_dists else None,
             'mean_negative_yaw': float(np.mean(yaw_diffs)) if yaw_diffs else None}
    return baseline_pairs, stats


def sample_short_temporal_pairs(frame_poses, n_video, count=20, min_gap=2, max_gap=None, seed=None):   # pair_sampling.py:425
    rng = random.Random(seed)
    empty = {'num_short': 0, 'target_count': count, 'matched_rate': 0.0, 'min_gap': min_gap, 'max_gap': max_gap or 0}
    if n_video <= 1: return [], empty
    if max_gap is None: max_gap = max(6, int(0.02 * n_video))
    min_gap = max(1, int(min_gap)); max_gap = max(min_gap, int(max_gap))
    valid_frames = [f for f in range(n_video) if f in frame_poses]; valid_set = set(valid_frames)
    if len(valid_frames) < 2: return [], empty
    anchors = [f for f in valid_frames if f + min_gap < n_video]
    if not anchors: return [], empty
    pairs = []; used = set()
    anchor_list = [anchors[int(round(x))] for x in np.linspace(0, len(anchors) - 1, num=min(count, len(anchors)))]
    attempts = max(10, 5000 // max(1, len(anchor_list)))
    for anchor in anchor_list:
        gap_high = min(max_gap, n_video - 1 - anchor)
        if gap_high < min_gap: continue
        for _ in range(attempts):
            b = anchor + rng.randint(min_gap, gap_high)
            if b not in valid_set or (anchor, b) in used: continue
            pairs.append((anchor, b)); used.add((anchor, b)); break
    return pairs, {'num_short': len(pairs), 'target_count': count, 'matched_rate': len(pairs) / max(1, count),
                   'min_gap': min_gap, 'max_gap': max_gap}


# ----------------------------------------------------------------------------------------------------------------------
# full sampler: src/sample_pairs.py:90 (video length passed in instead of read from a representative mp4)
# ----------------------------------------------------------------------------------------------------------------------

def mine_pairs(c2w: np.ndarray, n_video: Optional[int] = None, up=(0., 0., 1.), max_eval_pairs: int = 100, seed: int = 42,
               min_gap_ratio: float = 0.2, angle_tolerance=None, translation_tolerance=None, short_multiplier: float = 1.0):
    """Commanded camera-to-world poses [T,4,4] (OpenCV, world up = `up`) -> {'revisit','baseline','short','meta'}.

    Pairs are (earlier, later) step indices into the case trajectory (== model output frame index)."""
    rng = random.Random(seed); np.random.seed(seed)
    pose_data = r2m_pose_data(c2w, up)
    groups, n_pose, info = extract_revisit_pairs(pose_data, angle_tolerance, translation_tolerance)
    flat = flatten_revisit_groups(groups)
    frame_poses = _extract_frame_poses(pose_data, sorted(pose_data.keys(), key=int))
    n_video = len(c2w) if n_video is None else int(n_video)
    effective = min(n_video, n_pose) if n_pose else n_video
    min_gap = int(effective * min_gap_ratio)
    filtered = [p for p in flat if p[0] < effective and p[1] < effective and abs(p[1] - p[0]) >= min_gap]
    meta = {'n_pose': n_pose, 'n_video': n_video, 'effective_frames': effective, 'seed': seed, 'max_eval_pairs': max_eval_pairs,
            'min_gap_ratio': min_gap_ratio, 'min_gap': min_gap, 'raw_revisit_pairs': len(flat), 'revisit_threshold_mode': info['mode'],
            'revisit_sampling_version': 'adaptive-p95-v1', 'angle_tolerance_deg': info['angle_tolerance_deg'],
            'translation_tolerance_m': info['translation_tolerance_m'], 'detection': info,
            'source': 'R2M-Bench src/sample_pairs.py:90-210'}
    if not filtered:
        return {'revisit': [], 'baseline': [], 'short': [], 'meta': {**meta, 'error': 'no valid revisit pairs'}}
    sampled = rng.sample(filtered, max_eval_pairs) if max_eval_pairs and len(filtered) > max_eval_pairs else filtered
    revisit = [[int(a), int(b)] for a, b, _, _ in sampled]
    exclusion_radius = max(10, effective // 20)
    base_raw, base_stats = sample_gap_matched_temporal_baselines([(a, b) for a, b in revisit], frame_poses, effective,
                                                                 num_temporal_bins=10, exclusion_radius=exclusion_radius,
                                                                 gap_tolerance_ratio=0.3, seed=seed)
    short_count = min(100, max(20, int(len(revisit) * short_multiplier)))
    short_raw, short_stats = sample_short_temporal_pairs(frame_poses, effective, count=short_count, min_gap=2, seed=seed)
    meta.update(exclusion_radius=exclusion_radius, baseline_stats=base_stats, short_stats=short_stats)
    return {'revisit': revisit, 'baseline': [[int(a), int(b)] for a, b in base_raw],
            'short': [[int(a), int(b)] for a, b in short_raw], 'meta': meta}


def select_for_video(pairs, n_video, max_per_type=0):                       # presampled_pairs.py:28
    def prep(seq):
        valid = [(int(a), int(b)) for a, b in seq if 0 <= a < n_video and 0 <= b < n_video]
        return valid[:max_per_type] if max_per_type and len(valid) > max_per_type else valid
    return prep(pairs.get('revisit', [])), prep(pairs.get('baseline', [])), prep(pairs.get('short', []))


# ----------------------------------------------------------------------------------------------------------------------
# MemoryGain / NMR
# ----------------------------------------------------------------------------------------------------------------------

LOWER_IS_BETTER = {'lpips'}
EPS = 1e-8
MIN_DYNAMIC_RANGE = 1e-5          # eval_gpu_metrics.py:62 (per-video result files)

# compute_nmr_summary.py:30-73 (metric keys follow R2M's result schema)
METRICS = ['psnr', 'ssim', 'lpips', 'dino_similarity', 'boq_similarity', 'mvpr_similarity', 'match_ratio',
           'ransac_inlier_ratio', 'state_score', 'object_appearance_persistence', 'object_semantic_persistence']
FAMILIES = {'Appearance': ['psnr', 'ssim', 'lpips'],
            'Scene ID': ['dino_similarity', 'boq_similarity', 'mvpr_similarity'],
            'Local Geo.': ['match_ratio', 'ransac_inlier_ratio'],
            'State': ['state_score'],
            'Object': ['object_appearance_persistence', 'object_semantic_persistence']}


def set_stats(values):
    """eval_gpu_metrics.py:68 _aggregate for one pair set."""
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not v: return None
    return {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'median': float(np.median(v)), 'count': len(v)}


def video_mg_dr(rev: float, base: float, short: float, metric: str):
    """compute_nmr_summary.py:210 video_mg_dr."""
    return (base - rev, base - short) if metric in LOWER_IS_BETTER else (rev - base, short - base)


def per_video_nmr(rev, base, short, metric):
    """eval_gpu_metrics.py:82 _compute_nmr (per-video file convention: NMR 0 when DR <= 1e-5)."""
    num, dyn = video_mg_dr(rev, base, short, metric)[0], video_mg_dr(rev, base, short, metric)[1]
    return 0.0 if dyn <= MIN_DYNAMIC_RANGE else float(num / dyn)


def aggregate_metric(videos: List[Tuple[float, float, float]], metric: str, eps: float = EPS):
    """compute_nmr_summary.py:222 aggregate_metric over videos [(rev, base, short)]: DR<=eps videos excluded,
    NMR = mean(MG)/(mean(DR)+eps)."""
    pairs = [video_mg_dr(r, b, s, metric) for r, b, s in videos]
    valid = [(mg, dr) for mg, dr in pairs if dr > eps]
    if not valid: return {'n': 0, 'mg': math.nan, 'dr': math.nan, 'nmr': math.nan}
    mg = float(np.mean([m for m, _ in valid])); dr = float(np.mean([d for _, d in valid]))
    return {'n': len(valid), 'mg': mg, 'dr': dr, 'nmr': mg / (dr + eps)}


def overall_nmr(metric_nmr: Dict[str, float]):
    """compute_nmr_summary.py:270 overall_value: mean over available families of the mean metric NMR."""
    fam = {}
    for name, metrics in FAMILIES.items():
        vals = [metric_nmr[m] for m in metrics if m in metric_nmr and metric_nmr[m] is not None and not math.isnan(metric_nmr[m])]
        if vals: fam[name] = float(np.mean(vals))
    return (float(np.mean(list(fam.values()))) if fam else math.nan), fam


def memory_block(per_set: Dict[str, List[float]], metric: str, eps: float = EPS, rev_key: str = 'revisit'):
    """Per-set stats + MG / DR / NMR for one metric of one video (case). `rev_key` selects which revisit set is
    compared with the shared baseline / short sets ('revisit' = R2M-mined, 'keypoint_revisit' = our keypoint protocol)."""
    st = {'revisit': set_stats(per_set.get(rev_key, [])), 'baseline': set_stats(per_set.get('baseline', [])),
          'short': set_stats(per_set.get('short', []))}
    out = {'revisit_set': rev_key, 'revisit': st['revisit'], 'baseline': st['baseline'], 'short': st['short'], 'metric': metric,
           'lower_is_better': metric in LOWER_IS_BETTER}
    if all(st[k] is not None for k in st):
        r, b, s = st['revisit']['mean'], st['baseline']['mean'], st['short']['mean']
        mg, dr = video_mg_dr(r, b, s, metric)
        agg = aggregate_metric([(r, b, s)], metric, eps)
        out.update(mg=mg, dr=dr, nmr=None if math.isnan(agg['nmr']) else agg['nmr'], nmr_valid=agg['n'] == 1,
                   nmr_per_video_file=per_video_nmr(r, b, s, metric))
    else:
        out.update(mg=None, dr=None, nmr=None, nmr_valid=False)
    return out
