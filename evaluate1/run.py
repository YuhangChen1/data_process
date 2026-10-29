"""Score one model output on an evaluate1 case with every registered operator.

  python -m evaluate1.run --case evaluate1/cases/<scene> --frames evaluate1/runs/<scene>/<model>/frames
  python -m evaluate1.run --case evaluate1/cases/<scene> --report        # compare every scored model of the case

frames/ holds <step:05d>.png|jpg for every trajectory step (see evaluate1/render.py and README.md); an optional sibling
depth/<step:05d>.npy gives metric camera-Z depth at any subset of steps. Results: <run>/scores.json (every operator,
including n/a and errors with the reason) and <run>/scores.md; --report writes <case runs dir>/report.md.
"""
from pathlib import Path
import argparse, importlib, json, pkgutil, sys, time
import numpy as np

DP = Path(__file__).resolve().parents[1]
if str(DP) not in sys.path: sys.path.insert(0, str(DP))
from evaluate1.metrics import base  # noqa: E402


def load_operators():
    import evaluate1.metrics as M
    loaded, failed = [], {}
    for m in sorted(pkgutil.iter_modules(M.__path__), key=lambda x: x.name):
        if m.name in ('base', '__init__'): continue
        try: importlib.import_module(f'evaluate1.metrics.{m.name}'); loaded.append(m.name)
        except Exception as e: failed[m.name] = f'{type(e).__name__}: {e}'
    return loaded, failed


class StepStore:
    """dict-like lazy per-step arrays (depth): ctx.depth[step] -> float32 [H,W] or KeyError when not rendered."""
    def __init__(self, folder):
        self.files = {int(p.stem): p for p in Path(folder).glob('*.npy')}
    def __contains__(self, i): return int(i) in self.files
    def __getitem__(self, i): return np.load(self.files[int(i)]).astype(np.float32)
    def __len__(self): return len(self.files)
    def keys(self): return sorted(self.files)


def read_frames(folder, steps):
    import cv2
    files = {int(p.stem): p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in ('.png', '.jpg', '.jpeg')}
    missing = [i for i in range(steps) if i not in files]
    if missing: raise SystemExit(f'{len(missing)} trajectory steps have no frame, first missing {missing[:5]}')
    first = cv2.imread(str(files[0])); out = np.empty((steps, *first.shape), np.uint8)
    for i in range(steps): out[i] = cv2.imread(str(files[i]))[:, :, ::-1]
    return out


def context(case_dir, frames_dir, with_frames=True):
    import cv2
    case_dir = Path(case_dir).resolve(); case = json.loads((case_dir / 'case.json').read_text())
    traj = json.loads((case_dir / 'trajectory.json').read_text()); cams = np.load(case_dir / 'cameras.npz')
    T = case['steps']; frames = read_frames(frames_dir, T) if with_frames else None
    K = np.array(case['K'], float)
    if frames is not None and (frames.shape[2], frames.shape[1]) != (case['width'], case['height']):
        sx, sy = frames.shape[2] / case['width'], frames.shape[1] / case['height']; K = np.diag([sx, sy, 1.]) @ K
    inp = json.loads((case_dir / 'input' / 'input.json').read_text())['frames']
    gt = np.stack([cv2.imread(str(case_dir / f['image']))[:, :, ::-1] for f in inp])
    if frames is not None and gt.shape[1:3] != frames.shape[1:3]:
        gt = np.stack([cv2.resize(g, (frames.shape[2], frames.shape[1]), interpolation=cv2.INTER_AREA) for g in gt])
    pairs = {k: [tuple(map(int, x)) for x in v] for k, v in case['pairs'].items() if isinstance(v, list)}
    try:
        from evaluate1.metrics import pairs as P
        mined = P.mine_pairs(cams['c2w_canonical']); pairs_meta = mined.get('meta')
        pairs.update({k: [tuple(map(int, x)) for x in v] for k, v in mined.items() if isinstance(v, list) and k not in pairs})
    except Exception as e:
        pairs_meta = {'error': f'{type(e).__name__}: {e}'}; print('pair mining unavailable:', type(e).__name__, e)
    objects = []
    for o in case.get('objects', []):
        objects.append({**o, 'visible_at': {int(k): str(case_dir / v) for k, v in o['visible_at'].items()}})
    keypoints = [{**k, 'excursion': [k['visit'] + 1, k['return'] - 1]} for k in case['keypoints']]
    actions = [{'keypoint': k['id'], **{kk: tr[kk] for kk in ('kind', 'token', 'magnitude', 'action_index')},
                'steps': list(range(tr['steps'][0], tr['steps'][1] + 1))} for k in case['keypoints'] for tr in k['turns']]
    depth_dir = Path(frames_dir).parent / 'depth'
    ctx = base.EvalContext(case_dir=case_dir, frames=frames, fps=case['fps'], K=K, c2w_cmd=cams['c2w_canonical'],
                           depth=StepStore(depth_dir) if depth_dir.exists() and any(depth_dir.glob('*.npy')) else None,
                           segments=traj['steps'], actions=actions, turns=case['turns'], keypoints=keypoints, pairs=pairs,
                           input_index=case['input_steps'], input_gt=gt, scene_text=case.get('scene_text'),
                           subject_text=case.get('subject_text'), objects=objects or None,
                           extra={'case': case, 'c2w_source': cams['c2w_source'], 'scene_text_zh': case.get('scene_text_zh'),
                                  'setting': case.get('setting'), 'pairs_meta': pairs_meta})
    return ctx


def score(case_dir, frames_dir, name=None, only=None, skip_vlm=False, vlm_only=False, merge=False):
    import torch
    loaded, failed = load_operators(); ctx = context(case_dir, frames_dir)
    run_dir = Path(frames_dir).resolve().parent; name = name or run_dir.name
    results = {}; t0 = time.time()
    # WBench first: its navigation/consistency operators run MegaSaM, whose camera trajectory then feeds the
    # revisit.pose.est_drift_* operators as ctx.c2w_est (only when tracked at stride 1 = one pose per step).
    specs = sorted(base.REGISTRY.values(), key=lambda s: (not s.family.startswith('wbench'), s.family, s.name))
    est_set = False
    for spec in specs:
        if not est_set and not spec.family.startswith('wbench'):
            est_set = True; m = ctx.cache.get('wbench_megasam')
            if ctx.c2w_est is None and isinstance(m, tuple) and m[1] == 1 and len(m[0]) == len(ctx.frames):
                ctx.c2w_est = np.asarray(m[0], float); ctx.extra['c2w_est_source'] = 'MegaSaM (wbench_interaction.megasam_poses)'
        if only and not any(spec.name.startswith(o) or spec.family.startswith(o) for o in only): continue
        if vlm_only and not spec.vlm: continue
        if skip_vlm and spec.vlm:
            results[spec.name] = base.Result(spec.name, None, 'n/a', note='skipped (--skip-vlm)').as_dict(); continue
        r = base.run(spec.name, ctx); results[spec.name] = {**r.as_dict(), 'family': spec.family, 'source': spec.source}
        print(f'{r.status:5s} {spec.name:55s} {r.value!s:>22.22s} {r.seconds:7.1f}s {r.note[:80]}', flush=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if merge and (run_dir / 'scores.json').exists():   # keep earlier results of operators not re-run now
        prev = json.loads((run_dir / 'scores.json').read_text())['results']; results = {**prev, **results}
    doc = {'schema': 'evaluate1.scores.v1', 'case': str(Path(case_dir).resolve()), 'frames': str(Path(frames_dir).resolve()),
           'model': name, 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': round(time.time() - t0, 1),
           'modules_loaded': loaded, 'modules_failed': failed, 'operators': len(results), 'c2w_est_source': ctx.extra.get('c2w_est_source'),
           'status_counts': {s: sum(r['status'] == s for r in results.values()) for s in ('ok', 'n/a', 'error')}, 'results': results}
    (run_dir / 'scores.json').write_text(json.dumps(doc, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    (run_dir / 'scores.md').write_text(markdown([doc]))
    print('wrote', run_dir / 'scores.json', doc['status_counts'])
    return doc


def fmt(v):
    if v is None: return '—'
    return f'{v:.4g}' if isinstance(v, (int, float)) else str(v)


def markdown(docs):
    names = sorted({k for d in docs for k in d['results']}, key=lambda k: (docs[0]['results'].get(k, {}).get('family', ''), k))
    lines = ['| family | operator | ' + ' | '.join(d['model'] for d in docs) + ' | better |', '|---|---|' + '---|' * len(docs) + '---|']
    for k in names:
        r0 = next(d['results'][k] for d in docs if k in d['results'])
        cells = []
        for d in docs:
            r = d['results'].get(k)
            cells.append('' if r is None else (fmt(r['value']) if r['status'] == 'ok' else r['status']))
        lines.append(f"| {r0.get('family', '')} | {k} | " + ' | '.join(cells) + f" | {'↑' if r0.get('higher_is_better', True) else '↓'} |")
    return '\n'.join(lines) + '\n'


def report(case_dir):
    case = json.loads((Path(case_dir) / 'case.json').read_text())
    runs = DP / 'evaluate1' / 'runs' / case['scene_id']
    docs = [json.loads(p.read_text()) for p in sorted(runs.glob('*/scores.json'))]
    if not docs: raise SystemExit('no scored runs')
    (runs / 'report.md').write_text(f"# evaluate1 — {case['scene_id']}\n\nmodels: {', '.join(d['model'] for d in docs)}\n\n" + markdown(docs))
    print('wrote', runs / 'report.md')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--case', required=True); ap.add_argument('--frames'); ap.add_argument('--name')
    ap.add_argument('--only', nargs='*'); ap.add_argument('--skip-vlm', action='store_true'); ap.add_argument('--report', action='store_true')
    ap.add_argument('--vlm-only', action='store_true', help='run only GPT-judge operators'); ap.add_argument('--merge', action='store_true', help='merge into existing scores.json')
    a = ap.parse_args()
    if a.report: report(a.case)
    else: score(a.case, a.frames, a.name, a.only, a.skip_vlm, a.vlm_only, a.merge)


if __name__ == '__main__':
    main()
