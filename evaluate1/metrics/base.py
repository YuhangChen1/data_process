"""Operator contract shared by every evaluate1 metric.

An operator is a plain function `fn(ctx: EvalContext, **params) -> Result` registered with @operator. It reads only the
fields of EvalContext it declares in `inputs`; the runner skips it (status 'n/a') when one of them is missing, so every
metric can exist as an operator even when a case does not exercise it (e.g. subject metrics on a static scene).

Frames are RGB uint8 numpy arrays [T,H,W,3]. Cameras are OpenCV camera-to-world 4x4 (+X right, +Y down, +Z forward) in
the scene's canonical Z-up frame (inferred metres). Index i of `frames` is trajectory step i of the case.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import os, sys, time, traceback

import numpy as np

TMP = Path('/mnt/cfs/cyh/tmp/data_pipeline')
WEIGHTS = TMP / 'weights'            # every downloaded checkpoint lives here (never under /root)
PYLIBS = [TMP / 'pylib_wbench', TMP / 'pylib_r2m', TMP / 'pylib_common']   # pip --target dirs; shared envs untouched
for _p in PYLIBS:
    if _p.exists() and str(_p) not in sys.path: sys.path.append(str(_p))
for _k, _v in {'HF_HOME': TMP / 'hf', 'TORCH_HOME': TMP / 'torch', 'XDG_CACHE_HOME': TMP / 'cache'}.items():
    os.environ.setdefault(_k, str(_v))

GPT_MODEL = os.environ.get('EVAL_VLM_MODEL', 'gpt-6.1-sol')     # judge model (GPT gateway, see annotation_pipeline.vlm)


@dataclass
class EvalContext:
    case_dir: Path | None = None
    frames: np.ndarray | None = None                 # [T,H,W,3] uint8, model output along the case trajectory
    fps: float = 12.0
    K: np.ndarray | None = None                      # [3,3] or [T,3,3] intrinsics of `frames`
    c2w_cmd: np.ndarray | None = None                # [T,4,4] commanded camera poses (canonical, OpenCV)
    c2w_est: np.ndarray | None = None                # [T,4,4] poses estimated from `frames` (filled by a pose operator)
    depth: np.ndarray | None = None                  # [T,H,W] float32 metres if the model can render depth (optional)
    segments: list[dict] | None = None               # trajectory tags per step: {'kind': walk|pause|action|inverse, ...}
    actions: list[dict] | None = None                # action program: [{'token': 'W', 'magnitude': .3, 'steps': [i..j]}, ...]
    turns: list[dict] | None = None                  # WBench-style turns: [{'id', 'frames': [i0, i1], 'action', 'text'}]
    keypoints: list[dict] | None = None              # revisit keypoints: {'id', 'visit': i, 'return': j, 'excursion': [..]}
    pairs: dict[str, list[tuple[int, int]]] | None = None   # R2M pairs: {'revisit': [...], 'baseline': [...], 'short': [...]}
    input_index: list[int] | None = None             # the 10 input frames: trajectory steps carrying ground truth
    input_gt: np.ndarray | None = None               # [10,H,W,3] ground-truth images at those steps (from the input video)
    scene_text: str | None = None                    # scene description from the annotation pipeline (setting adherence)
    subject_text: str | None = None                  # main subject description, None for static scenes
    objects: list[dict] | None = None                # annotated objects: {'global_id','label','label_zh','visible_at': {step: mask_path}}
    extra: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)   # operators share expensive intermediates (features, poses, masks)


@dataclass
class Result:
    name: str
    value: float | None
    status: str = 'ok'                               # ok | n/a | error
    unit: str = ''
    higher_is_better: bool = True
    range: tuple[float, float] | None = None
    raw: dict = field(default_factory=dict)          # un-normalised intermediate numbers
    per_item: list = field(default_factory=list)     # per frame / pair / turn values
    note: str = ''
    seconds: float = 0.0

    def as_dict(self):
        return {k: getattr(self, k) for k in ('name', 'value', 'status', 'unit', 'higher_is_better', 'range', 'raw',
                                             'per_item', 'note', 'seconds')}


@dataclass
class Spec:
    name: str                                        # e.g. 'wbench.V1_aesthetic_quality', 'r2m.appearance.lpips'
    family: str                                      # e.g. 'wbench.video_quality', 'r2m.scene_identity', 'revisit.pose'
    fn: Callable
    inputs: tuple[str, ...]
    higher_is_better: bool = True
    range: tuple[float, float] | None = None
    unit: str = ''
    source: str = ''                                 # paper / repo reference this follows
    requires: tuple[str, ...] = ()                   # python modules / weights the operator needs
    gpu: bool = False
    vlm: bool = False


REGISTRY: dict[str, Spec] = {}


def operator(name, family, inputs, higher_is_better=True, range=None, unit='', source='', requires=(), gpu=False, vlm=False):
    def wrap(fn):
        if name in REGISTRY: raise ValueError(f'duplicate operator {name}')
        REGISTRY[name] = Spec(name, family, fn, tuple(inputs), higher_is_better, range, unit, source, tuple(requires), gpu, vlm)
        fn.spec = REGISTRY[name]
        return fn
    return wrap


def missing_inputs(spec: Spec, ctx: EvalContext):
    return [k for k in spec.inputs if getattr(ctx, k, None) is None and k not in ctx.extra]


def run(name: str, ctx: EvalContext, **params) -> Result:
    spec = REGISTRY[name]; t = time.time()
    miss = missing_inputs(spec, ctx)
    if miss:
        return Result(name, None, 'n/a', spec.unit, spec.higher_is_better, spec.range, note='missing inputs: ' + ', '.join(miss))
    try:
        r = spec.fn(ctx, **params)
        if not isinstance(r, Result): r = Result(name, float(r))
        r.name = name; r.unit = r.unit or spec.unit; r.higher_is_better = spec.higher_is_better; r.range = r.range or spec.range
    except NotApplicable as e:
        r = Result(name, None, 'n/a', spec.unit, spec.higher_is_better, spec.range, note=str(e))
    except Exception as e:
        r = Result(name, None, 'error', spec.unit, spec.higher_is_better, spec.range, note=f'{type(e).__name__}: {e}',
                   raw={'traceback': traceback.format_exc()[-4000:]})
    r.seconds = round(time.time() - t, 3)
    return r


class NotApplicable(Exception):
    """Raise inside an operator when the case cannot exercise it (e.g. no subject, no repeated action)."""


def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def cached(ctx: EvalContext, key: str, fn: Callable):
    if key not in ctx.cache: ctx.cache[key] = fn()
    return ctx.cache[key]
