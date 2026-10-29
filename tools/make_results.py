"""Assemble the delivery view of one scene: results/<scene>/ and the operator catalogue OPERATORS.md.

  python tools/make_results.py --scene KS-IN-OE-S-F1_01_202609121414 [--model oracle]

results/<scene>/annotation.md + manifest.json   scene manifest (hesai/package.py)
results/<scene>/case.md                          evaluate1 case summary (inputs, keypoints, action programs, objects)
results/<scene>/scores_<model>.json|.md          every operator's result for that model run
OPERATORS.md                                     72-operator catalogue with the pilot values of <model>
"""
from pathlib import Path
import argparse, collections, json, shutil, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEXT = {'W': '前进', 'S': '后退', 'A': '左移', 'D': '右移', 'LEFT': '左转', 'RIGHT': '右转', 'UP': '抬头', 'DOWN': '低头'}


def case_md(case):
    import math
    L = [f"# evaluate1 case — {case['scene_id']}", '',
         f"- 轨迹：{case['walk_frames']} 帧前进漫游 + 6 个关键点往返，共 {case['steps']} 步，{case['fps']:g} fps，{case['width']}×{case['height']}",
         f"- 输入 10 帧（漫游帧号）：{case['input_walk_frames']}；对应轨迹步：{case['input_steps']}",
         f"- 动作参数：平移 {case['parameters']['move_m']} m，偏航 {case['parameters']['yaw_deg']}°，俯仰 {case['parameters']['pitch_deg']}°，"
         f"每个动作 {case['parameters']['steps_per_action']} 步，关键点暂停 {case['parameters']['pause_steps']} 步",
         f"- 关键点处标注物体 mask：{len(case['objects'])} 个；场景描述（gpt-6.1-sol）：{case.get('scene_text_zh') or case.get('scene_text')}", '',
         '| 关键点 | 漫游帧 | 出发步 | 返回步 | 3 个动作（之后按相反顺序执行反向动作） | 可见标注物体 |', '|---|---|---|---|---|---|']
    objs = collections.Counter(o['keypoint'] for o in case['objects'])
    for k in case['keypoints']:
        prog = ' → '.join(f"{TEXT[p['token']]} {p['magnitude']:.2f} m" if p['unit'] == 'm' else f"{TEXT[p['token']]} {math.degrees(p['magnitude']):.0f}°"
                          for p in k['program'])
        L.append(f"| {k['id']} | {k['walk_frame']} | {k['visit']} | {k['return']} | {prog} | {objs.get(k['id'], 0)} |")
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scene', required=True); ap.add_argument('--model', default='oracle'); a = ap.parse_args()
    import operators as O
    out = ROOT / 'results' / a.scene; out.mkdir(parents=True, exist_ok=True)
    for src, dst in [('README.md', 'annotation.md'), ('manifest.json', 'manifest.json')]:
        shutil.copy(ROOT / 'scenes' / a.scene / src, out / dst)
    case = json.loads((ROOT / 'evaluate1' / 'cases' / a.scene / 'case.json').read_text()); (out / 'case.md').write_text(case_md(case))
    run = ROOT / 'evaluate1' / 'runs' / a.scene / a.model
    scores = json.loads((run / 'scores.json').read_text())
    shutil.copy(run / 'scores.json', out / f'scores_{a.model}.json')
    counts = collections.Counter(r['status'] for r in scores['results'].values())
    head = [f"# {a.model} — {a.scene}", '', f"状态：{dict(counts)}。n/a / error 的原因见 scores_{a.model}.json 的 note 字段。", '']
    (out / f'scores_{a.model}.md').write_text('\n'.join(head) + O.catalogue(scores['results']))
    intro = ['# 最终算子（72 个）', '',
             '入口 `operators.py`：`list_operators()`、`load_context(case, frames)`、`run_operator(name, ctx)`、`run_all(ctx)`；',
             '命令行 `python operators.py list | doc | run`。每个算子是 `evaluate1/metrics/*.py` 中按 `metrics/base.py` 约定注册的函数，',
             '返回 `ok` / `n/a`（case 不具备所需输入，附原因）/ `error`（附 traceback）。',
             '方向 ↑ 表示越高越好。WBench 类按 WBench 的 0–100 归一；R2M 类的值是关键点重访对的 NMR（raw 中另有 MG、R2M 挖掘配对的结果）。',
             f"最后一列是试点场景 {a.scene} 上 `{a.model}`（扫描原始 3DGS，作为上界参考）的值。", '',
             '公式、模型权重、与原 benchmark 的差异：`evaluate1/docs/wbench_ops.md`、`evaluate1/docs/r2m_revisit_ops.md`。']
    (ROOT / 'OPERATORS.md').write_text('\n'.join(intro) + '\n' + O.catalogue(scores['results']))
    print('wrote', out, dict(counts))


if __name__ == '__main__':
    main()
