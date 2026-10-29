"""WBench physics metrics P1 (causal fidelity, VLM dual-track) and P2 (visual plausibility, PAVRM) as operators.

P1: WBench prompts verbatim (causal_fidelity.py), judge gpt-6.1-sol; frames are the 3-fps video frames sent as numbered
contact sheets (gateway size limit) and answers are requested as JSON {"reason", "score"}.
P2: the official PAVRM reward model (weights/wbench/qwen3vl-a3b-visual-plausibility, Qwen3-VL-30B-A3B, bf16 ~62 GB) when
it is downloaded; otherwise the same rating question is asked to gpt-6.1-sol and the result is flagged as a proxy.
"""
from __future__ import annotations
from pathlib import Path
import os, re, tempfile
import numpy as np

from .base import operator, Result, NotApplicable, TMP, device
from .wbench_quality import WB, REF, sample_indices, pil, wb_score, free_gpu, vlm_ask, vlm_sheets
from .wbench_interaction import SHEET_NOTE

PHYSICS_DIMENSIONS = {
    "fluid_dynamics": ("Fluid & Smoke", "Rate fluid, smoke, and related effects (0-3).\n3 = Realistic flow, splash, smoke rise, underwater drift\n2 = Minor artifacts but behavior is recognizable\n1 = Notable issues: liquid defies gravity, smoke static, no bubbles\n0 = Fluid/smoke behavior completely unrealistic or absent"),
    "collision": ("Collision & Clipping", "Rate collision and solid body integrity (0-3).\nONLY evaluate collisions/contacts that actually occur.\n3 = All contacts are natural, no clipping or pass-through\n2 = Minor clipping but mostly plausible contacts\n1 = Obvious clipping: limbs pass through objects or surfaces\n0 = Severe clipping, objects completely ignore each other"),
    "surface_interaction": ("Surface Tracks & Imprints", "Rate surface interaction and imprint accuracy (0-3).\n3 = Clear, appropriate marks visible (footprints in snow, tracks in sand)\n2 = Some marks visible but incomplete or inconsistent\n1 = No marks visible on a surface that should clearly show them\n0 = Surface shows no response at all to contact"),
    "deformation": ("Deformation & Destruction", "Rate deformation and destruction effects (0-3).\n3 = Realistic shattering, crumbling, bending, debris scatter\n2 = Minor issues but deformation is recognizable\n1 = Deformation clearly unnatural or inconsistent\n0 = No deformation when expected, or completely unrealistic"),
    "wind_weather": ("Wind & Environmental Forces", "Rate wind and environmental force effects (0-3).\n3 = Natural swaying, drifting, fluttering of affected objects\n2 = Minor issues but environmental effects are recognizable\n1 = Effects are stiff, static, or inconsistent with the scene\n0 = No environmental effects visible when expected"),
    "reflection": ("Reflection & Lighting", "Rate the accuracy of SPECIAL reflection and lighting effects (0-3).\nEvaluate ONLY optical effects that are actually VISIBLE in the video. Do NOT penalize for effects that are absent from the scene.\nCheck for:\n- Water/mirror reflections: do they match the shape and motion of the source?\n- Point light sources: is the illumination range and falloff plausible?\n- Shadows: do they point in the correct direction relative to the light?\n- Metallic/glass surfaces: do they show plausible environment reflections?\n\n3 = All visible optical effects are accurate and physically consistent\n2 = Minor errors in one effect (e.g. reflection slightly misaligned)\n1 = Clear errors: reflections show wrong content, shadows point wrong way, or lighting does not respond to scene changes\n0 = Optical effects are completely wrong or absent when clearly expected"),
    "human_physics": ("Human Motion & Expression", "Rate human motion naturalness and facial expression quality (0-3).\nFocus on MOVEMENT ACROSS FRAMES, not just single-frame anatomy:\n- Does the character move fluidly between poses, or jerk/teleport?\n- Are body movements biomechanically plausible (natural gait, realistic reach, believable weight shifts)?\n- Do limbs move in coordinated, purposeful ways, or flail randomly?\n- Is the face coherent (no melting, distortion, or uncanny shifts)?\nMinor structural issues (slight hand distortion, brief extra finger) should NOT heavily impact the score if overall motion is natural.\n3 = Smooth, natural human motion throughout; face is coherent\n2 = Mostly natural motion with minor stiffness or brief glitches\n1 = Clearly unnatural: jerky limbs, puppet-like movement, frozen poses, or obvious face distortion\n0 = Severely broken: limbs flail impossibly, body contorts unnaturally, face melts or deforms grotesquely"),
}
TRACK1_SYSTEM_PROMPT = (
    "You are evaluating the RENDERING-LEVEL physical plausibility and causal consistency of an AI-generated video. You will see "
    "frames sampled from the ENTIRE video in temporal order.\n\nIMPORTANT RULES:\n- Do NOT judge whether depicted actions are possible "
    "in the real world. Fantasy/sci-fi elements are acceptable if the setting calls for them.\n- Do NOT penalize camera behavior "
    "(tilting, shaking, panning, zooming). Camera motion is NOT a physics issue.\n- ONLY evaluate the physics of OBJECTS and "
    "CHARACTERS.\n- ANY violation, no matter how brief or small in screen area, counts. A single clear clipping event or causal "
    "error is enough to deduct.\n\nScan ALL frames for:\nA. RENDERING PHYSICS\n  1. Solid body integrity — no clipping\n  2. "
    "Consistent gravity\n  3. Motion continuity — no teleporting\n  4. Object permanence — no random appear/vanish\n  5. Natural "
    "deformation\n  6. Character physics — natural movement\n\nB. CAUSAL CONSISTENCY\n  NOTE: Instructed actions and their "
    "consequences are EXPECTED — do NOT penalize them. Only penalize UNINTENDED changes.\n  7. No effect without cause (for "
    "non-instructed changes)\n  8. No cause without effect\n  9. No something from nothing\n\nScoring (0-3):\n3 = Good: physics and "
    "causality natural and consistent throughout\n2 = Fair: 1-2 noticeable problems anywhere in the video\n1 = Poor: multiple "
    "violations or one severe breakdown\n0 = Failure: physics/causality mostly broken")
TRACK2_SYSTEM_PROMPT = (
    "You are evaluating a specific physics phenomenon across an ENTIRE AI-generated video. You will see frames from the full video.\n"
    "Score based ONLY on the specified physics dimension.\nANY violation of this dimension anywhere in the video, no matter how brief "
    "or localized, should lower the score.\nMinor AI rendering artifacts are acceptable. Focus on whether the specific physical "
    "behavior is plausible throughout.")
SCORE_JSON = ('Return JSON only: {"reason": "<max 50 words; only describe specific issues found; if score is not full marks you '
              'MUST name the exact problem; do not describe what looks good>", "score": <a single integer from 0 to 3>}')


def _score(ans, max_score=3):
    v = str(ans.get('score', '')).strip()
    if re.fullmatch(r'\d', v): return min(int(v), max_score)
    raise ValueError(f'no explicit score in judge answer {ans}')


@operator('wbench.P1_causal_fidelity', 'wbench.physics', ('frames',), range=(0, 100), vlm=True,
          source=REF + ' src/metrics/physical/causal_fidelity.py:218-341 + evaluate.py:213-245 (3 fps, judge gpt-6.1-sol)')
def causal_fidelity(ctx, max_sheets=3):
    """(Track1 0-3 + mean Track2 0-3 over annotated dims)/2 / 3; Track1 alone when the case annotates no physics dims."""
    idx = sample_indices(len(ctx.frames), ctx.fps, {'fps': 3.0}); sheets = vlm_sheets([pil(ctx.frames[i]) for i in idx], max_sheets=max_sheets)
    setting = ctx.extra.get('setting') or {}; cf = ctx.extra.get('causal_fidelity') or (ctx.extra.get('case') or {}).get('causal_fidelity') or {}
    acts = [t.get('text') or t.get('action') for t in (ctx.turns or []) if t.get('type') != 'perspective_switch' and (t.get('text') or t.get('action'))]
    astr = ' -> '.join(acts) if acts else 'navigation'
    q1 = (f"Scene context: {setting.get('environment', 'indoor')} / {setting.get('name') or ctx.scene_text or ''}\n" if (setting or ctx.scene_text) else '')
    q1 += f"Actions in this video: {astr}\n\nRate the overall rendering physics AND causal consistency of this ENTIRE video from 0 to 3. Any physics violation at any point should be noted and penalized."
    a1 = vlm_ask(ctx, f'{q1}\n\n{SCORE_JSON}{SHEET_NOTE}', sheets, 'P1_track1', system=TRACK1_SYSTEM_PROMPT); t1 = _score(a1)
    dims = [d for d in cf.get('dims', []) if d in PHYSICS_DIMENSIONS]; t2 = []
    for d in dims:
        name, prompt = PHYSICS_DIMENSIONS[d]
        a = vlm_ask(ctx, f'Actions in this video: {astr}\n\nEvaluating: {name}\n{prompt}\n\n{SCORE_JSON}{SHEET_NOTE}', sheets, f'P1_{d}', system=TRACK2_SYSTEM_PROMPT)
        t2.append({'dim': d, 'score': _score(a), 'reason': a.get('reason')})
    final = (t1 + np.mean([x['score'] for x in t2])) / 2.0 if t2 else float(t1)
    return wb_score(final / 3.0, {'raw_score': final, 'track1': {'score': t1, 'reason': a1.get('reason')}, 'track2': t2,
                                  'num_frames_3fps': len(idx), 'frames_sent': min(len(idx), 16 * max_sheets)},
                    note='' if dims else 'no physics dims annotated in the case: Track1 only (as WBench)')


VIDEO_QUALITY_PROMPT = (
    "Suppose you are an expert in judging and evaluating the quality of AI-generated videos, please watch the above provided video "
    "and give scores for the video's truthfulness and rationality. i.e., whether the video's overall appreance and motion are "
    "consistent with our common-sense, physical principles.\nYour rating should be chosen from the following five catefories: "
    "Perfect, Good, Fair, Poor, and Bad. Now please rate this video:")
ANCHOR_TOKEN_IDS = [51041, 15216, 60795, 84103, 17082]       # visual_plausibility.py:32-33 (Perfect..Bad)
ANCHOR_WEIGHTS = [5.0, 4.0, 3.0, 2.0, 1.0]
PAVRM = WB / 'qwen3vl-a3b-visual-plausibility'


def _pavrm_ready():
    idx = PAVRM / 'model.safetensors.index.json'
    if not idx.exists(): return False
    import json
    shards = set(json.loads(idx.read_text())['weight_map'].values())
    return all((PAVRM / s).exists() for s in shards)


def _pavrm_local(ctx, fps=2.0):
    """visual_plausibility.py:36-110: one generated token, softmax over the 5 rating anchors, expectation 1..5."""
    import torch, cv2
    from transformers import AutoModelForImageTextToText, AutoProcessor
    for k in [k for k, v in ctx.cache.items() if hasattr(v, 'parameters') or (isinstance(v, tuple) and any(hasattr(x, 'parameters') for x in v))]:
        del ctx.cache[k]                                         # free every cached torch model first (~62 GB needed)
    free_gpu()
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(PAVRM), dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map='auto')
    except (ImportError, ValueError):
        model = AutoModelForImageTextToText.from_pretrained(str(PAVRM), dtype=torch.bfloat16, device_map='auto')
    proc = AutoProcessor.from_pretrained(str(PAVRM))
    with tempfile.TemporaryDirectory(prefix='pavrm_', dir=str(TMP / 'tmp')) as d:
        vp = os.path.join(d, 'video.mp4'); h, w = ctx.frames.shape[1:3]
        wr = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*'mp4v'), ctx.fps, (w, h))
        for f in ctx.frames: wr.write(f[:, :, ::-1])
        wr.release()
        msgs = [{'role': 'user', 'content': [{'type': 'video', 'video': vp, 'fps': fps, 'max_pixels': 602112},
                                             {'type': 'text', 'text': VIDEO_QUALITY_PROMPT}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False, output_logits=True, return_dict_in_generate=True)
    lg = out.logits[0][0]; ids = torch.tensor(ANCHOR_TOKEN_IDS, device=lg.device)
    probs = torch.softmax(torch.gather(lg, -1, ids).float(), -1)
    raw = float(torch.sum(probs * torch.tensor(ANCHOR_WEIGHTS, device=lg.device)))
    del model, out, inputs; free_gpu()
    return raw, probs.cpu().tolist()


@operator('wbench.P2_visual_plausibility', 'wbench.physics', ('frames',), range=(0, 100), gpu=True,
          source=REF + ' src/metrics/physical/visual_plausibility.py:36-177 + tools/run_visual_plausibility.py (PAVRM, 2 fps)',
          requires=('transformers>=4.57 (Qwen3-VL-MoE)', 'weights/wbench/qwen3vl-a3b-visual-plausibility (62 GB)'))
def visual_plausibility(ctx, allow_proxy=True):
    """PAVRM expected rating (Perfect=5..Bad=1) / 5 at 2 fps; if the 62 GB checkpoint is absent, gpt-6.1-sol proxy."""
    if _pavrm_ready():
        raw, probs = _pavrm_local(ctx)
        return wb_score(raw / 5.0, {'raw_score': raw, 'anchor_probs_perfect_to_bad': probs, 'method': 'pavrm_qwen3vl_a3b'})
    if not allow_proxy: raise FileNotFoundError(f'PAVRM weights not complete under {PAVRM}')
    idx = sample_indices(len(ctx.frames), ctx.fps, {'fps': 2.0})
    ans = vlm_ask(ctx, VIDEO_QUALITY_PROMPT.replace('the above provided video', 'the provided video frames') +
                  '\nReturn JSON only: {"rating": "Perfect|Good|Fair|Poor|Bad", "reason": "<max 30 words>"}' + SHEET_NOTE,
                  vlm_sheets([pil(ctx.frames[i]) for i in idx]), 'P2_proxy')
    cats = {'perfect': 5.0, 'good': 4.0, 'fair': 3.0, 'poor': 2.0, 'bad': 1.0}
    raw = cats[str(ans.get('rating', '')).strip().lower()]
    return wb_score(raw / 5.0, {'raw_score': raw, 'answer': ans, 'method': 'gpt-6.1-sol proxy (PAVRM weights not available)'},
                    note='PROXY: gpt-6.1-sol rating, not the PAVRM reward model')
