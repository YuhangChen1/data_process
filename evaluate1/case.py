"""Build an evaluate1 case from one annotated scene.

  python -m evaluate1.case --scene KS-IN-OE-S-F1_01_202609121414

Protocol (evaluate1 v1)
  * Trajectory = the scan's own forward roaming video when it ships one (hesai manyou_videos/main_walkthrough, the same clip the annotation
    pipeline labels), one trajectory step per walkthrough frame at 12 fps.
  * Input = 10 walkthrough frames evenly spaced over the clip: their camera poses + images are the ground truth the
    model's renders at those steps are scored against (fidelity / image quality).
  * Revisit = 6 keypoints spread along the walkthrough. At each keypoint the camera pauses, executes 3 actions from the
    WBench navigation vocabulary (W/S/A/D translate, ←/→ yaw, ↑/↓ pitch), then the 3 inverse actions in reverse order
    (LIFO) so it lands exactly on the keypoint pose again, and then continues along the walkthrough. The keypoint frame
    before the excursion (visit) and the frame after it (return) form a revisit pair.

Outputs  evaluate1/cases/<scene>/
  case.json        protocol parameters, input steps, keypoints + action programs, WBench-style turns, scene text,
                   annotated objects at the keypoints, R2M pairs
  trajectory.json  one record per step: kind (walk|pause|action|inverse|return), walkthrough frame, keypoint, action
                   token, t, camera_to_canonical, camera_to_source, relative pose to the previous step
  cameras.npz      c2w_canonical [T,4,4], c2w_source [T,4,4], K [3,3], t [T]  (OpenCV cameras)
  input/           the 10 ground-truth input frames (symlinks) + input.json
  objects/         binary masks of annotated objects at keypoint visit frames
A model is evaluated by rendering its 4DGS at every step of cameras.npz (pose c2w_source or c2w_canonical, time t)
and writing frames/<step:05d>.png (+ optional depth/<step:05d>.npy); see evaluate1/README.md.
"""
from pathlib import Path
import argparse, json, math, os, shutil
import numpy as np

DP = Path(__file__).resolve().parents[1]
VERSION = 'evaluate1.case.v1'

# Action vocabulary (WBench first-person navigation): translations relative to the current heading, rotations about
# world up (yaw) or the camera's right axis (pitch).
INVERSE = {'W': 'S', 'S': 'W', 'A': 'D', 'D': 'A', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'UP': 'DOWN', 'DOWN': 'UP'}
TEXT = {'W': 'move forward', 'S': 'move backward', 'A': 'move left', 'D': 'move right', 'LEFT': 'turn left',
        'RIGHT': 'turn right', 'UP': 'look up', 'DOWN': 'look down'}
TRANSLATIONS = ('W', 'S', 'A', 'D')
# One program per keypoint; together they use every token, translations and rotations interleaved.
TEMPLATES = [('LEFT', 'W', 'UP'), ('RIGHT', 'S', 'DOWN'), ('A', 'LEFT', 'W'),
             ('D', 'RIGHT', 'S'), ('UP', 'LEFT', 'D'), ('DOWN', 'RIGHT', 'A')]


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); tmp.replace(path)


def read(path): return json.loads(Path(path).read_text())


def rot_axis(axis, angle):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis); x, y, z = axis; c, s = math.cos(angle), math.sin(angle)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


class Camera:
    """Keypoint-relative camera state: position, yaw about canonical +Z, pitch about the camera's right axis."""
    def __init__(self, c2w):
        self.R0 = c2w[:3, :3].copy(); self.p = c2w[:3, 3].copy(); self.yaw = 0.; self.pitch = 0.

    def pose(self):
        R = rot_axis([0, 0, 1], self.yaw) @ self.R0 @ rot_axis([1, 0, 0], self.pitch)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = self.p; return T

    def heading(self):
        f = rot_axis([0, 0, 1], self.yaw) @ self.R0[:, 2]; f = np.array([f[0], f[1], 0.]); f /= np.linalg.norm(f)
        left = np.array([-f[1], f[0], 0.])          # +90 deg about +Z (camera left = -X_cam for a level OpenCV camera)
        return f, left

    def apply(self, token, amount):
        """amount: metres for translations, radians for rotations (sign handled by the token)."""
        f, left = self.heading()
        if token == 'W': self.p = self.p + amount * f
        elif token == 'S': self.p = self.p - amount * f
        elif token == 'A': self.p = self.p + amount * left
        elif token == 'D': self.p = self.p - amount * left
        elif token == 'LEFT': self.yaw += amount
        elif token == 'RIGHT': self.yaw -= amount
        elif token == 'UP': self.pitch += amount       # +angle about camera +X tilts forward towards -Y_cam = up
        elif token == 'DOWN': self.pitch -= amount
        else: raise ValueError(token)


class Scene:
    """Trajectory rule: the scan's own forward walkthrough when it ships one (outputs/<key>/hesai_walkthrough/main),
    otherwise the longest A* region walkthrough planned on the navigation map (floor_*/annotation_walkthrough)."""
    def __init__(self, key, trajectory='auto'):
        self.key = key; self.out = DP / 'outputs' / key
        self.analysis = read(self.out / 'analysis.json'); self.W2C = np.array(self.analysis['world_to_canonical'])
        clip = self.out / 'hesai_walkthrough' / 'main'
        if trajectory == 'provided' or (trajectory == 'auto' and (clip / 'metadata.json').exists()):
            self.meta = read(clip / 'metadata.json'); self.trajectory_kind = 'provided_walkthrough'
            cams = np.load(clip / 'cameras.npz'); self.c2w = cams['camera_to_canonical']; self.c2s = cams['camera_to_source']; self.K = cams['K']
            self.frames_dir = Path(self.meta['frames_dir']); self.clip_id = f'{self.meta["floor"].replace("f", "floor_")}_hesai_main'
        else:
            comps = sorted(self.out.glob('floor_*/annotation_walkthrough/component_*/metadata.json'), key=lambda m: -read(m)['frames'])
            if not comps: raise SystemExit('no trajectory: the scan has no walkthrough and no A* walkthrough was rendered (hesai.stage --step render)')
            clip = comps[0].parent; m = read(comps[0]); self.trajectory_kind = 'astar_walkthrough'
            cams = np.load(clip / 'cameras.npz'); self.c2w = cams['camera_to_canonical']; self.c2s = cams['camera_to_original']; self.K = cams['K']
            floor = clip.parent.parent.name; self.frames_dir = clip / 'frames'; self.clip_id = f'{floor}_{clip.name}'
            if len(list(self.frames_dir.glob('*.png'))) != m['frames']:      # decode the A* RGB video once into PNG frames
                import cv2
                self.frames_dir.mkdir(exist_ok=True); cap = cv2.VideoCapture(str(clip / 'rgb.mp4')); i = 0
                while True:
                    ok, im = cap.read()
                    if not ok: break
                    cv2.imwrite(str(self.frames_dir / f'{i:06d}.png'), im); i += 1
                cap.release()
            self.meta = {**m, 'floor': floor.replace('floor_', 'f'), 'source': str(clip), 'frames_dir': str(self.frames_dir)}
        self.depth_dir = clip / 'mesh_depth_mm'
        self.floor = next(f for f in self.analysis['floors'] if f'f{f["floor_id"]}' == self.meta['floor'])
        maps = np.load(self.out / f'floor_{self.floor["floor_id"]}' / 'maps.npz')
        self.nav = maps['navigable'].astype(bool); self.origin = maps['origin_xy']; self.res = float(maps['resolution'])
        self._ray = None

    @property
    def ray(self):
        if self._ray is None:
            from scene_pipeline.geometry import build_rayscene
            m = np.load(self.out / 'mesh_cache.npz')
            v = m['vertices'] @ self.W2C[:3, :3].T + self.W2C[:3, 3]      # canonical coordinates
            self._ray = build_rayscene(v.astype(np.float32), m['faces'])
        return self._ray

    def free_xy(self, p):
        rc = np.floor((np.asarray(p[:2]) - self.origin)[::-1] / self.res).astype(int)
        return 0 <= rc[0] < self.nav.shape[0] and 0 <= rc[1] < self.nav.shape[1] and bool(self.nav[tuple(rc)])

    def clearance(self, p, d):
        import open3d as o3d
        rays = np.array([[*p, *d]], np.float32)
        t = self.ray.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()[0]
        return float(t) if np.isfinite(t) else 1e9

    def translation_ok(self, cam, token, amount, margin=.3):
        """Camera may translate when the straight segment keeps `margin` from the mesh at camera height and 0.5 m
        lower (furniture tops), and the end point stays over navigable floor (no camera through walls/out of scan)."""
        f, left = cam.heading()
        d = {'W': f, 'S': -f, 'A': left, 'D': -left}[token]
        for dz in (0., -.5):
            if self.clearance(cam.p + [0, 0, dz], d) < amount + margin: return False
        return self.free_xy(cam.p + amount * d)

    def depth_score(self, i):
        import cv2
        d = cv2.imread(str(self.depth_dir / f'{i:05d}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        valid = d > 0; med = float(np.median(d[valid])) if valid.any() else 0.
        return valid.mean() * (1. if 1.0 <= med <= 8. else .3), med


def plan_program(scene, c2w, template, move, yaw, pitch):
    """Instantiate a template at one keypoint; infeasible translations fall back to their opposite, then to any other
    translation, then to a shorter move (0.25 m), then to an unused rotation. Returns (program, substitutions)."""
    cam = Camera(c2w); program = []; subs = []; used = set(template)
    for token in template:
        chosen = None
        if token in TRANSLATIONS:
            options = [(token, move), (INVERSE[token], move)] + [(t, move) for t in TRANSLATIONS if t not in (token, INVERSE[token])] \
                      + [(t, .25) for t in (token, INVERSE[token]) + tuple(x for x in TRANSLATIONS if x not in (token, INVERSE[token]))]
            for t, m in options:
                if scene.translation_ok(cam, t, m): chosen = (t, m); break
            if chosen is None:
                rot = next(t for t in ('LEFT', 'RIGHT', 'UP', 'DOWN') if t not in used); chosen = (rot, yaw if rot in ('LEFT', 'RIGHT') else pitch)
        else:
            chosen = (token, yaw if token in ('LEFT', 'RIGHT') else pitch)
        if chosen[0] != token or (chosen[0] in TRANSLATIONS and chosen[1] != move):
            subs.append({'template': token, 'used': chosen[0], 'magnitude': chosen[1]})
        used.add(chosen[0]); program.append({'token': chosen[0], 'magnitude': float(chosen[1]),
                                            'unit': 'm' if chosen[0] in TRANSLATIONS else 'rad'})
        cam.apply(*chosen)
    return program, subs


def choose_keypoints(scene, n_key, frame_objects):
    N = len(scene.c2w); lo, hi = int(round(.06 * N)), int(round(.94 * N))
    edges = np.linspace(lo, hi, n_key + 1).astype(int); keys = []
    max_obj = max([len(v) for v in frame_objects.values()] + [1])
    for b in range(n_key):
        best = None
        for i in range(edges[b] + 2, edges[b + 1] - 2):
            content, med = scene.depth_score(i)
            cam = Camera(scene.c2w[i]); feas = sum(scene.translation_ok(cam, t, .4) for t in TRANSLATIONS) / 4
            objs = len(frame_objects.get(i, [])) / max_obj if frame_objects else 0.
            centre = 1 - 2 * abs(i - (edges[b] + edges[b + 1]) / 2) / max(edges[b + 1] - edges[b], 1)   # spread keypoints
            s = 2 * feas + content + objs + .5 * centre
            if best is None or s > best[0]:
                best = (s, i, {'translation_feasible_fraction': feas, 'depth_valid_x_range': round(float(content), 3),
                               'median_depth_m': round(med, 3), 'annotated_objects_visible': len(frame_objects.get(i, []))})
        keys.append({'walk_frame': best[1], 'selection': {'score': round(float(best[0]), 3), **best[2]}})
    return keys


def annotations(scene):
    """Walkthrough frame -> visible annotated thing objects (global_pipeline output), plus a scene-level label list."""
    gi = DP / 'auto' / scene.key / 'global_instances'
    if not (gi / 'frame_assignments.json').exists(): return {}, {}, None
    frames = read(gi / 'frames.json')['frames']; assign = read(gi / 'frame_assignments.json')['frames']
    inst = {o['global_id']: o for o in read(gi / 'global_instances.json')['instances']}
    by_index = {}; meta = {}
    for f in frames:
        if f['clip_id'] != scene.clip_id: continue
        objs = []
        for a in assign.get(f['id'], []):
            o = inst.get(a['global_id'])
            if not a.get('accepted') or o is None or o.get('semantic_label') in (None, 'unknown') or o.get('category_type') != 'thing': continue
            if a.get('support_points', 0) < 50 or a.get('purity', 0) < .8: continue
            objs.append({'global_id': a['global_id'], 'label': o['semantic_label'], 'support_points': a['support_points'], 'purity': a['purity']})
        by_index[f['source_frame_index']] = objs; meta[f['source_frame_index']] = f
    return by_index, meta, gi


SETTING_PROMPT = """You write the world-setting text for a navigation-video benchmark case (WBench-style setting adherence).
The first image is a contact sheet of the INPUT walkthrough of one real indoor scan: tile 0 (top-left, larger) is the
initial frame, tiles 1-9 are later frames in time order. Automatic annotation (noisy) lists objects visible in the initial
frame: {first}; and objects that only appear later in the walkthrough: {later}.
Return JSON only:
{{"visible_part": "one or two English sentences describing the scene elements visible in the initial frame",
  "offscreen_part": "one English sentence naming elements NOT visible in the initial frame that appear as the camera moves",
  "style": "short English visual-style phrase",
  "scene_text": "2-3 English sentences describing the whole scene (layout, main objects, materials, lighting)",
  "scene_text_zh": "同一场景的中文描述，2-3 句"}}
Describe only what is visible; no invented rooms, people, text content or measurements."""


def contact_sheet(paths, out, w=384):
    """4-column sheet: tile 0 (initial frame) doubled at top-left, tiles 1-4 top-right, 5-8 third row, 9 last row."""
    import cv2
    ims = [cv2.imread(str(p)) for p in paths]; h = int(ims[0].shape[0] * w / ims[0].shape[1])
    rows = 3 + (len(ims) > 9); sheet = np.full((rows * h, 4 * w, 3), 255, np.uint8)
    cells = [(0, 0, 2)] + [(r, c, 1) for r in (0, 1) for c in (2, 3)] + [(2, c, 1) for c in range(4)] + [(3, 0, 1)]
    for k, (im, (r, c, n)) in enumerate(zip(ims, cells)):
        sheet[r * h:(r + n) * h, c * w:(c + n) * w] = cv2.resize(im, (n * w, n * h))
        cv2.putText(sheet, str(k), (c * w + 8, r * h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    Path(out).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]); return out


def setting_text(out, inputs, frame_objects, model='gpt-6.1-sol'):
    from annotation_pipeline.vlm import VLM
    first = sorted({o['label'] for o in frame_objects.get(inputs[0]['walk_frame'], [])})
    later = sorted({o['label'] for f in inputs[1:] for o in frame_objects.get(f['walk_frame'], [])} - set(first))
    sheet = contact_sheet([inputs[n]['source_image'] for n in range(len(inputs))], out / 'input' / 'contact_sheet.jpg')
    prompt = SETTING_PROMPT.format(first=', '.join(first) or 'not available', later=', '.join(later) or 'not available')
    r = VLM(model=model).call(prompt, [sheet], out / 'vlm' / 'setting.json', max_tokens=1500)
    return {'visible_part': r.get('visible_part', ''), 'offscreen_part': r.get('offscreen_part', ''), 'style': r.get('style', 'realistic'),
            'annotation_labels_initial': first, 'annotation_labels_later': later, 'source': f'gpt:{model} from input contact sheet + annotation labels',
            'evidence': 'vlm/setting.json'}, r.get('scene_text', ''), r.get('scene_text_zh', '')


def build(key, n_input=10, n_key=6, pause=6, action_steps=12, move=.4, yaw_deg=30., pitch_deg=15., force=False, keep_keypoints=False,
          trajectory='auto', out=None):
    scene = Scene(key, trajectory); out = Path(out) if out else DP / 'evaluate1' / 'cases' / key
    keep = None
    if keep_keypoints and (out / 'case.json').exists():   # refresh annotations/text without moving the trajectory
        keep = [k['walk_frame'] for k in read(out / 'case.json')['keypoints']]; old_cams = np.load(out / 'cameras.npz')['c2w_canonical']
    if out.exists() and force: shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    frame_objects, frame_meta, gi = annotations(scene)
    N = len(scene.c2w); fps = float(scene.meta['fps'])
    input_frames = [int(round(x)) for x in np.linspace(0, N - 1, n_input)]
    if keep:
        keys = [{'walk_frame': i, 'selection': {'kept_from_previous_case': True, 'annotated_objects_visible': len(frame_objects.get(i, []))}} for i in keep]
    else:
        keys = choose_keypoints(scene, n_key, frame_objects)
    for k, kp in enumerate(keys):
        kp['id'] = f'K{k + 1}'; kp['template'] = list(TEMPLATES[k % len(TEMPLATES)])
        kp['program'], kp['substitutions'] = plan_program(scene, scene.c2w[kp['walk_frame']], kp['template'], move, yaw, pitch)
    by_frame = {kp['walk_frame']: kp for kp in keys}

    steps = []; C2S = np.linalg.inv(scene.W2C)
    def emit(c2w, kind, walk_frame=None, keypoint=None, token=None, action_index=None):
        steps.append({'step': len(steps), 'kind': kind, 'walk_frame': walk_frame, 'keypoint': keypoint, 'token': token,
                      'action_index': action_index, 'c2w': np.asarray(c2w, float)})
    for i in range(N):
        emit(scene.c2w[i], 'walk', walk_frame=i)
        kp = by_frame.get(i)
        if kp is None: continue
        kp['visit'] = len(steps) - 1
        for _ in range(pause): emit(scene.c2w[i], 'pause', keypoint=kp['id'])
        cam = Camera(scene.c2w[i]); kp['turns'] = []
        sequence = [(a['token'], a['magnitude'], 'action', j) for j, a in enumerate(kp['program'])] + \
                   [(INVERSE[a['token']], a['magnitude'], 'inverse', j) for j, a in reversed(list(enumerate(kp['program'])))]
        for token, mag, kind, j in sequence:
            s0 = len(steps)
            for _ in range(action_steps):
                cam.apply(token, mag / action_steps); emit(cam.pose(), kind, keypoint=kp['id'], token=token, action_index=j)
            kp['turns'].append({'kind': kind, 'token': token, 'magnitude': mag, 'action_index': j, 'steps': [s0, len(steps) - 1]})
        err = np.abs(cam.pose() - scene.c2w[i]).max()
        if err > 1e-6: raise RuntimeError(f'{kp["id"]} excursion does not close: {err}')
        emit(scene.c2w[i], 'return', walk_frame=i, keypoint=kp['id']); kp['return'] = len(steps) - 1
    T = len(steps)
    c2w = np.stack([s['c2w'] for s in steps]); c2s = C2S[None] @ c2w; t = np.arange(T) / fps
    if keep is not None and (old_cams.shape != c2w.shape or not np.allclose(old_cams, c2w)):
        raise RuntimeError('--keep-keypoints changed the trajectory')
    np.savez(out / 'cameras.npz', c2w_canonical=c2w, c2w_source=c2s, K=scene.K, t=t, width=scene.meta['width'], height=scene.meta['height'])

    records = []
    for s in steps:
        i = s['step']; rel = np.linalg.inv(c2w[i - 1]) @ c2w[i] if i else np.eye(4)
        records.append({k: s[k] for k in ('step', 'kind', 'walk_frame', 'keypoint', 'token', 'action_index')} |
                       {'t': round(float(t[i]), 6), 'camera_to_canonical': c2w[i].round(8).tolist(), 'camera_to_source': c2s[i].round(8).tolist(),
                        'relative_to_previous': rel.round(8).tolist()})
    dump(out / 'trajectory.json', {'schema': VERSION + '.trajectory', 'scene_id': key, 'fps': fps, 'width': scene.meta['width'],
                                   'height': scene.meta['height'], 'K': scene.K.tolist(), 'steps': records})

    # WBench-style turns: walkthrough segments between excursions + every action / inverse as its own turn
    turns = []; seg_start = 0
    for kp in keys:
        turns.append({'id': f'walk_{len(turns)}', 'frames': [seg_start, kp['visit']], 'action': 'FOLLOW', 'text': 'continue along the walkthrough'})
        for tr in kp['turns']:
            turns.append({'id': f'{kp["id"]}_{tr["kind"]}{tr["action_index"]}_{tr["token"]}', 'frames': [tr['steps'][0] - 1, tr['steps'][1]],
                          'action': tr['token'], 'magnitude': tr['magnitude'], 'keypoint': kp['id'], 'kind': tr['kind'],
                          'text': f'{TEXT[tr["token"]]} {tr["magnitude"]:.2f} ' + ('m' if tr['token'] in TRANSLATIONS else 'rad')})
        seg_start = kp['return']
    turns.append({'id': f'walk_{len(turns)}', 'frames': [seg_start, T - 1], 'action': 'FOLLOW', 'text': 'continue along the walkthrough'})

    # input frames: walkthrough frame index -> trajectory step (walk record)
    walk_step = {s['walk_frame']: s['step'] for s in steps if s['kind'] == 'walk'}
    (out / 'input').mkdir(exist_ok=True); inputs = []
    for n, i in enumerate(input_frames):
        src = scene.frames_dir / f'{i:06d}.png'; dst = out / 'input' / f'{n:02d}_{i:06d}.png'
        if dst.is_symlink() or dst.exists(): dst.unlink()
        dst.symlink_to(src)
        inputs.append({'n': n, 'walk_frame': i, 'step': walk_step[i], 'image': str(dst.relative_to(out)), 'source_image': str(src),
                       'camera_to_canonical': scene.c2w[i].tolist(), 'camera_to_source': scene.c2s[i].tolist()})
    dump(out / 'input' / 'input.json', {'K': scene.K.tolist(), 'width': scene.meta['width'], 'height': scene.meta['height'], 'frames': inputs})

    # annotated objects at keypoints: binary masks from the global instance PNG of the visit frame
    objects = []
    if gi is not None:
        import cv2
        (out / 'objects').mkdir(exist_ok=True)
        for kp in keys:
            f = frame_meta.get(kp['walk_frame'])
            if f is None: continue
            ids = cv2.imread(str(gi / 'rendered' / f['clip_id'] / 'instance' / (f['id'] + '.png')), cv2.IMREAD_UNCHANGED)
            for o in frame_objects.get(kp['walk_frame'], []):
                m = ids == o['global_id']
                if m.sum() < 400: continue
                path = out / 'objects' / f'{kp["id"]}_g{o["global_id"]:05d}.png'; cv2.imwrite(str(path), m.astype(np.uint8) * 255)
                objects.append({**o, 'keypoint': kp['id'], 'visible_at': {str(kp['visit']): str(path.relative_to(out)),
                                str(kp['return']): str(path.relative_to(out))}, 'mask_pixels': int(m.sum()),
                                'annotation_frame_id': f['id']})

    pairs = {'keypoint_revisit': [[kp['visit'], kp['return']] for kp in keys]}
    try:
        from evaluate1.metrics import pairs as P
        mined = P.mine_pairs(c2w)
        pairs.update({k: [list(map(int, x)) for x in v] for k, v in mined.items()})
    except Exception as e:                    # pairs.py is optional at build time; the runner mines them again
        pairs['note'] = f'R2M pair mining deferred to the runner ({type(e).__name__}: {e})'

    try:
        setting, scene_text, scene_text_zh = setting_text(out, inputs, frame_objects)
    except Exception as e:
        setting, scene_text, scene_text_zh = {'error': f'{type(e).__name__}: {e}'}, None, None
    case = {'schema': VERSION, 'scene_id': key, 'source': read(DP / 'sources' / key / 'source.json'),
            'setting': setting, 'scene_text': scene_text, 'scene_text_zh': scene_text_zh, 'subject_text': None,
            'trajectory_source': str(scene.meta['source']), 'trajectory_kind': scene.trajectory_kind, 'fps': fps, 'steps': T, 'walk_frames': N,
            'width': scene.meta['width'], 'height': scene.meta['height'], 'K': scene.K.tolist(),
            'parameters': {'n_input': n_input, 'n_keypoints': n_key, 'pause_steps': pause, 'steps_per_action': action_steps,
                           'move_m': move, 'yaw_deg': yaw_deg, 'pitch_deg': pitch_deg, 'translation_margin_m': .3,
                           'units': 'inferred_meter', 'templates': [list(t) for t in TEMPLATES]},
            'input_steps': [x['step'] for x in inputs], 'input_walk_frames': input_frames,
            'keypoints': [{k: v for k, v in kp.items()} for kp in keys], 'turns': turns, 'pairs': pairs,
            'objects': objects, 'annotation': str(gi) if gi else None,
            'conventions': {'camera': 'OpenCV (+X right, +Y down, +Z forward), camera-to-world',
                            'canonical': 'scene_pipeline leveled Z-up frame (outputs/<scene>/analysis.json world_to_canonical)',
                            'source': 'raw scan frame of gaussian.ply / mesh_post.obj (-Y up)', 'time': 't = step / fps seconds'}}
    dump(out / 'case.json', case)
    return case


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scene', required=True); ap.add_argument('--force', action='store_true')
    ap.add_argument('--keep-keypoints', action='store_true', help='reuse the existing keypoints (trajectory unchanged)')
    ap.add_argument('--trajectory', default='auto', choices=['auto', 'provided', 'astar'], help='auto: scan walkthrough if shipped, else A*')
    ap.add_argument('--out', help='case directory (default evaluate1/cases/<scene>)')
    ap.add_argument('--n-input', type=int, default=10); ap.add_argument('--n-key', type=int, default=6)
    ap.add_argument('--pause', type=int, default=6); ap.add_argument('--action-steps', type=int, default=12)
    ap.add_argument('--move', type=float, default=.4); ap.add_argument('--yaw', type=float, default=30.); ap.add_argument('--pitch', type=float, default=15.)
    a = ap.parse_args()
    c = build(a.scene, a.n_input, a.n_key, a.pause, a.action_steps, a.move, a.yaw, a.pitch, a.force, a.keep_keypoints, a.trajectory, a.out)
    print('steps', c['steps'], 'input', c['input_steps'])
    for kp in c['keypoints']:
        print(kp['id'], 'walk', kp['walk_frame'], 'visit', kp['visit'], 'return', kp['return'],
              [(p['token'], round(p['magnitude'], 3)) for p in kp['program']], 'subs', kp['substitutions'], kp['selection'])
    print('objects', len(c['objects']), 'pairs', {k: len(v) for k, v in c['pairs'].items() if isinstance(v, list)})


if __name__ == '__main__':
    main()
