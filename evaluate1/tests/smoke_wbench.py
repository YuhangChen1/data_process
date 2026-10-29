"""Smoke test: every WBench operator (22) on real rendered frames of the pilot case.

  python3 -m evaluate1.tests.smoke_wbench [--frames <frames dir>] [--only V1 C4 ...] [--no-vlm]

The static library case has no subject and no event/subject/perspective turns, so the test additionally runs the
subject- and turn-type operators (C4, C7, S2, I2, I3, I4) on an augmented copy of the context: a rectangle subject mask
(the foreground bookcase) and one fake event_edit / subject_action / perspective_switch turn. Their values are
meaningless; the point is that the code paths execute. Results: <TMP>/wbench_test/<run>/smoke.json.
"""
from pathlib import Path
import argparse, copy, json, sys, time
import numpy as np

DP = Path(__file__).resolve().parents[2]
if str(DP) not in sys.path: sys.path.insert(0, str(DP))
from evaluate1 import run as R                                       # noqa: E402
from evaluate1.metrics import base                                   # noqa: E402

CASE = DP / 'evaluate1' / 'cases' / 'KS-IN-OE-S-F1_01_202609121414'
FRAMES = DP / 'evaluate1' / 'runs' / 'KS-IN-OE-S-F1_01_202609121414' / 'oracle' / 'frames'
AUGMENTED = ('C4', 'C7', 'S2', 'I2', 'I3', 'I4')


def test_dir(frames):
    """Private run dir with per-file symlinks so nothing is written next to the real model outputs."""
    src = Path(frames).resolve(); out = base.TMP / 'wbench_test' / src.parent.name; (out / 'frames').mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        q = out / 'frames' / p.name
        if not q.exists(): q.symlink_to(p)
    if (src.parent / 'depth').exists() and not (out / 'depth').exists(): (out / 'depth').symlink_to(src.parent / 'depth')
    return out


def augment(ctx):
    a = copy.copy(ctx); H, W = ctx.frames.shape[1:3]
    m = np.zeros((H, W), bool); m[int(.45 * H):int(.95 * H), int(.30 * W):int(.62 * W)] = True
    a.extra = dict(ctx.extra, subject_mask=m, subject_mask_step=0,
                   subject={'appearance_part': 'a short freestanding pale-wood bookcase filled with books', 'action_part': 'stays in place'})
    a.subject_text = 'a short freestanding pale-wood bookcase filled with books'
    w = ctx.turns[0]['frames']
    a.turns = list(ctx.turns) + [
        {'id': 'fake_event', 'frames': w, 'type': 'event_edit', 'action': 'the ceiling lights switch off'},
        {'id': 'fake_subject', 'frames': w, 'type': 'subject_action', 'action': 'a person walks to the bookcase'},
        {'id': 'fake_ps', 'frames': w, 'type': 'perspective_switch', 'action': 'fp_to_tp'}]
    a.cache = {}                     # own cache: the plain case memoises 'no subject' as n/a
    return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--frames', default=str(FRAMES)); ap.add_argument('--only', nargs='*')
    ap.add_argument('--no-vlm', action='store_true'); args = ap.parse_args()
    R.load_operators(); out = test_dir(args.frames)
    ctx = R.context(CASE, out / 'frames'); actx = augment(ctx); rows = []
    for name in sorted(k for k in base.REGISTRY if k.startswith('wbench.')):
        code = name.split('.')[1].split('_')[0]
        if args.only and code not in args.only: continue
        spec = base.REGISTRY[name]
        if args.no_vlm and spec.vlm: continue
        for tag, c in (('case', ctx),) + ((('augmented', actx),) if code in AUGMENTED else ()):
            r = base.run(name, c); rows.append({'operator': name, 'input': tag, **r.as_dict()})
            print(f'{r.status:5s} {name:42s} {tag:9s} {r.value!s:>20.20s} {r.seconds:7.1f}s {r.note[:90]}', flush=True)
            try:
                import torch; torch.cuda.empty_cache()
            except Exception: pass
    (out / 'smoke.json').write_text(json.dumps({'created': time.strftime('%F %T'), 'frames': args.frames, 'rows': rows}, indent=1, default=str))
    print('wrote', out / 'smoke.json', {s: sum(r['status'] == s for r in rows) for s in ('ok', 'n/a', 'error')})


if __name__ == '__main__':
    main()
