"""Final evaluation operators of anno_eval — the single public entry point.

    python operators.py list                                   # all operators (group, name, direction, inputs)
    python operators.py doc > OPERATORS.md                     # regenerate the operator catalogue
    python operators.py run --case evaluate1/cases/<scene> --frames evaluate1/runs/<scene>/<model>/frames [--only revisit wbench.C1]

    from operators import list_operators, load_context, run_operator, run_all
    ctx = load_context('evaluate1/cases/<scene>', 'evaluate1/runs/<scene>/<model>/frames')
    run_operator('revisit.object.pair_iou', ctx)               # -> dict(name, value, status, unit, raw, per_item, ...)

Every operator is a function registered in evaluate1/metrics/*.py with the contract in evaluate1/metrics/base.py.
Operators whose inputs the case cannot provide return status 'n/a' with the reason; failures return 'error' with the
traceback in raw — a run never stops on one operator.
"""
from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

# (family prefix, group code, group title) — order of the catalogue
GROUPS = [
    ('wbench.video_quality', 'A', '画面质量（WBench V）'),
    ('fidelity', 'B', '输入帧保真（10 帧真值）'),
    ('wbench.setting_adherence', 'C', '设定遵循（WBench S）'),
    ('wbench.interaction_adherence', 'D', '交互遵循（WBench I）'),
    ('wbench.consistency', 'E', '时空一致性（WBench C）'),
    ('wbench.physics', 'F', '物理合理性（WBench P）'),
    ('revisit', 'G', '关键点重访（本项目：位姿 / 画面 / 物体 / 往返镜像）'),
    ('r2m', 'H', '重访记忆 R2M（revisit vs baseline/short → MG / NMR）'),
]

ZH = {
    # A — WBench video quality
    'wbench.V1_aesthetic_quality': '美学质量：LAION aesthetic（CLIP ViT-L/14），2 fps 取帧平均，/10',
    'wbench.V2_imaging_quality': '成像质量：MUSIQ，长边缩到 512，2 fps 平均，/100',
    'wbench.V3_temporal_flickering': '时间闪烁：10 fps 相邻帧平均绝对差，(255−d)/255，越高越不闪',
    'wbench.V4_dynamic_degree': '动态程度：RAFT 光流 top-5% 幅值超过阈值的连续帧对是否足够（0/1）',
    'wbench.V5_motion_smoothness': '运动平滑：AMT 用 I(t−1)、I(t+1) 插出中间帧，与真实 I(t) 的误差',
    'wbench.V6_hpsv3_norm': '人类偏好：HPSv3 奖励（20 帧均匀采样、空提示词），按 p1–p99 百分位归一',
    # B — fidelity at the 10 input frames
    'fidelity.psnr': '10 个输入位姿上，模型渲染 vs 输入真值帧的 PSNR',
    'fidelity.ssim': '10 个输入位姿上，渲染 vs 输入真值帧的 SSIM',
    'fidelity.ms_ssim': '10 个输入位姿上，渲染 vs 真值的多尺度 SSIM',
    'fidelity.lpips_alex': '10 个输入位姿上，渲染 vs 真值的 LPIPS（AlexNet）感知距离',
    'fidelity.lpips_vgg': '10 个输入位姿上，渲染 vs 真值的 LPIPS（VGG）感知距离',
    'fidelity.dists': '10 个输入位姿上，渲染 vs 真值的 DISTS 结构/纹理感知距离',
    'fidelity.dreamsim': '10 个输入位姿上，渲染 vs 真值的 DreamSim 感知距离',
    'fidelity.dinov2': '10 个输入位姿上，渲染 vs 真值的 DINOv2 全局特征余弦相似度',
    'fidelity.niqe': '10 个输入位姿上渲染帧的无参考画质 NIQE（越低越好）',
    'fidelity.brisque': '10 个输入位姿上渲染帧的无参考画质 BRISQUE（越低越好）',
    # C — setting adherence
    'wbench.S1_scene_adherence': '场景遵循：GPT 看第 1 段帧，评可见场景元素是否保持(1–5)、画外元素是否出现(0/1)，用 case 的 visible/offscreen 描述',
    'wbench.S2_subject_adherence': '主体遵循：主体外观保持 + 动作符合描述（需要主体，静态场景 n/a）',
    # D — interaction adherence
    'wbench.I1_navigation_score': '导航分：MegaSaM 从画面估计相机位姿，与每个动作的指令轨迹比 nATE（对齐+弧长重采样），再加重复动作的跨轮一致性',
    'wbench.I2_event_editing_adherence': '事件编辑遵循：每个事件 turn 5 项递进判定（需要事件 turn，静态场景 n/a）',
    'wbench.I3_subject_action_adherence': '主体动作遵循：每个主体动作 turn 5 项递进判定（需要主体，n/a）',
    'wbench.I4_perspective_switching_adherence': '视角切换遵循：转场可见、目标类型一致、新视角结构合规三项全过的比例（需要视角切换 turn，n/a）',
    # E — consistency
    'wbench.C1_spatial_consistency': '空间一致：每个关键点出发帧 vs 返回帧（MegaSaM 选旋转最接近的帧）的 DreamSim 相似度',
    'wbench.C2_gated_spatial_consistency': '门控空间一致：C1 × 门控（往返途中画面确实变化过，防止原地不动拿高分）',
    'wbench.C3_segment_continuity': '片段连续：TransNetV2 检测不到硬切为 1',
    'wbench.C4_perspective_consistency': '视角一致：SAM2 跟踪主体，质心在画面中的稳定度（需要主体，n/a）',
    'wbench.C5_geometric_consistency': '几何一致：DA3 每帧深度点重投影到其它帧的相对深度误差',
    'wbench.C6_photometric_consistency': '光度一致：DA3 点云颜色重投影到其它帧后的 PSNR，1−10^(−PSNR/20)',
    'wbench.C7_subject_consistency': '主体一致：主体裁剪的 DINOv2 相邻帧 + CLIP 首帧锚定相似度（需要主体，n/a）',
    'wbench.C8_background_consistency': '背景一致：CLIP ViT-B/32 相邻帧与首帧相似度（VBench 定义）',
    # F — physics
    'wbench.P1_causal_fidelity': '因果合理性：GPT 两阶段物理/因果评分（静态场景无物理维度时只用阶段 1）',
    'wbench.P2_visual_plausibility': '视觉合理性：PAVRM（Qwen3-VL-30B 微调）1–5 分；权重未就绪时用 gpt-6.1-sol 代评并标记 PROXY',
    # G — keypoint revisit (ours)
    'revisit.pose.rot_err_deg': '位姿：从出发帧/返回帧画面估计的相对旋转，与指令（单位阵）的角度误差',
    'revisit.pose.trans_err_m': '位姿：用出发帧深度做 PnP 求返回相机，与指令位置的平移误差（米，需要 depth）',
    'revisit.pose.trans_dir_err_deg': '位姿：本质矩阵平移方向误差；只在指令基线非零时有定义（精确重访为 n/a）',
    'revisit.pose.pixel_disp_px': '位姿：出发帧与返回帧匹配点的中位像素位移（精确重访为 0）',
    'revisit.pose.match_ratio': '位姿：SuperPoint+LightGlue 匹配率 |M|/min(Ka,Kb)',
    'revisit.pose.cmd_rot_err_deg': '自检：指令轨迹中出发/返回的旋转差（动作程序闭合时为 0）',
    'revisit.pose.cmd_trans_err_m': '自检：指令轨迹中出发/返回的相机中心距离（应为 0）',
    'revisit.pose.est_drift_rot_deg': '估计轨迹漂移：MegaSaM 轨迹 Sim(3) 对齐到指令后，出发→返回的旋转漂移',
    'revisit.pose.est_drift_trans_m': '估计轨迹漂移：MegaSaM 轨迹 Sim(3) 对齐后，出发→返回的相机中心漂移（米）',
    'revisit.image.psnr': '画面：出发帧 vs 返回帧 PSNR（完全一致封顶 100）',
    'revisit.image.ssim': '画面：出发帧 vs 返回帧 SSIM',
    'revisit.image.lpips': '画面：出发帧 vs 返回帧 LPIPS',
    'revisit.image.dreamsim': '画面：出发帧 vs 返回帧 DreamSim 距离',
    'revisit.image.dinov2': '画面：出发帧 vs 返回帧 DINOv2 余弦相似度',
    'revisit.object.mask_iou': '物体：标注物体在返回帧按类别重分割（GroundingDINO+SAM2）与标注 mask 的 IoU',
    'revisit.object.pair_iou': '物体：同一检测器在出发帧/返回帧对同一物体的 mask IoU（理想 1）',
    'revisit.object.recall': '物体：出发帧能检出的标注物体，返回帧仍在原位（IoU≥0.5）的比例',
    'revisit.object.precision': '物体：返回帧检出的同类物体中，与出发帧位置吻合的比例',
    'revisit.object.centroid_shift_px': '物体：匹配物体 mask 质心位移（像素）',
    'revisit.object.area_consistency': '物体：匹配物体 mask 面积比 min/max',
    'revisit.object.agnostic_iou': '物体：不依赖标签，SAM2 自动 mask 匈牙利匹配后的面积加权 IoU',
    'revisit.excursion.psnr': '往返镜像：动作 k 之后与其反向动作之前的同位姿帧 PSNR',
    'revisit.excursion.lpips': '往返镜像：动作 k 之后与其反向动作之前的同位姿帧 LPIPS',
    'revisit.excursion.dinov2': '往返镜像：动作 k 之后与其反向动作之前的同位姿帧 DINOv2 相似度',
    'revisit.excursion.pixel_disp_px': '往返镜像：动作 k 之后与其反向动作之前的同位姿帧匹配点中位像素位移',
    # H — R2M revisit memory
    'r2m.appearance.psnr': '外观保真：PSNR 在 revisit / gap-matched baseline / short 三组配对上的均值 → NMR',
    'r2m.appearance.ssim': '外观保真：SSIM（7×7）→ NMR',
    'r2m.appearance.lpips': '外观保真：LPIPS（Alex）→ NMR',
    'r2m.scene_identity.dinov2': '场景身份：DINOv2-B/14 全局特征相似度 → NMR',
    'r2m.scene_identity.boq': '场景身份：BoQ 视觉位置识别描述子相似度 → NMR',
    'r2m.scene_identity.mutualvpr': '场景身份：MutualVPR 描述子相似度 → NMR',
    'r2m.object_identity.appearance': '物体身份：GroundingDINO+SAM2 物体裁剪的 DINOv2 外观持续性（消失记 0）→ NMR',
    'r2m.object_identity.appearance_similarity': '物体身份：仅对仍存在物体的外观相似度 → NMR',
    'r2m.object_identity.existence': '物体身份：首帧物体在配对帧中的存在率 → NMR',
    'r2m.object_identity.semantic': '物体身份：GDINO 置信度 × CLIP 图文相似度的语义持续性 → NMR',
    'r2m.object_identity.semantic_similarity': '物体身份：仅对仍存在物体的语义相似度 → NMR',
    'r2m.geometry.match_ratio': '局部几何：SuperPoint+LightGlue 匹配率 → NMR',
    'r2m.geometry.inlier_ratio': '局部几何：RANSAC 基础矩阵内点率 → NMR',
    'r2m.persistent_state.vlm': '状态持续：GPT 按 R2M 量表判场景关系/状态是否持续（1–5）→ NMR',
    'r2m.summary.nmr': 'R2M 总分：五族各自取 NMR 均值后等权平均（不可用的族跳过）',
}


def _load():
    from evaluate1.run import load_operators
    from evaluate1.metrics import base
    loaded, failed = load_operators()
    if failed: print('operator modules failed to import:', failed, file=sys.stderr)
    return base


def group_of(family):
    for prefix, code, title in GROUPS:
        if family == prefix or family.startswith(prefix + '.') or family.startswith(prefix): return code, title
    return 'Z', family


def list_operators(group=None):
    base = _load(); rows = []
    for name, s in base.REGISTRY.items():
        code, title = group_of(s.family)
        if group and code != group: continue
        rows.append({'group': code, 'group_title': title, 'name': name, 'family': s.family, 'zh': ZH.get(name, ''),
                     'inputs': list(s.inputs), 'higher_is_better': s.higher_is_better, 'unit': s.unit, 'range': s.range,
                     'vlm': s.vlm, 'gpu': s.gpu, 'source': s.source, 'doc': (s.fn.__doc__ or '').strip()})
    return sorted(rows, key=lambda r: (r['group'], r['family'], r['name']))


def load_context(case_dir, frames_dir):
    from evaluate1.run import context
    _load(); return context(case_dir, frames_dir)


def run_operator(name, ctx, **params):
    base = _load(); return base.run(name, ctx, **params).as_dict()


def run_all(ctx, only=None, skip_vlm=False):
    """In-memory run of every operator (WBench first so MegaSaM poses can feed the revisit drift operators)."""
    base = _load(); out = {}
    specs = sorted(base.REGISTRY.values(), key=lambda s: (not s.family.startswith('wbench'), s.family, s.name)); est = False
    for s in specs:
        if only and not any(s.name.startswith(o) or s.family.startswith(o) for o in only): continue
        if skip_vlm and s.vlm: continue
        if not est and not s.family.startswith('wbench'):
            est = True; m = ctx.cache.get('wbench_megasam')
            if ctx.c2w_est is None and isinstance(m, tuple) and m[1] == 1 and ctx.frames is not None and len(m[0]) == len(ctx.frames):
                import numpy as np; ctx.c2w_est = np.asarray(m[0], float)
        out[s.name] = base.run(s.name, ctx).as_dict()
    return out


def catalogue(scores=None):
    """Markdown operator table; `scores` = {name: result dict} adds the pilot oracle column."""
    rows = list_operators(); lines = []; cur = None
    for r in rows:
        if r['group'] != cur:
            cur = r['group']; lines += ['', f"### {r['group']}. {r['group_title']}", '',
                                       '| 算子 | 含义 | 方向 | 单位 | 输入 | 试点 oracle |', '|---|---|---|---|---|---|']
        v = ''
        if scores and r['name'] in scores:
            s = scores[r['name']]; v = (f"{s['value']:.4g}" if isinstance(s['value'], (int, float)) else str(s['value'])) if s['status'] == 'ok' else s['status']
        text = (r['zh'] or r['doc'][:80]).replace('|', '\\|')
        lines.append(f"| `{r['name']}` | {text} | {'↑' if r['higher_is_better'] else '↓'} | {r['unit'] or '—'} | "
                     f"{', '.join(r['inputs'])} | {v} |")
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list'); d = sub.add_parser('doc'); d.add_argument('--scores')
    r = sub.add_parser('run'); r.add_argument('--case', required=True); r.add_argument('--frames', required=True)
    r.add_argument('--only', nargs='*'); r.add_argument('--skip-vlm', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'list':
        for x in list_operators(): print(f"{x['group']} {x['name']:48s} {'↑' if x['higher_is_better'] else '↓'} {x['zh']}")
        print(len(list_operators()), 'operators')
    elif a.cmd == 'doc':
        scores = json.loads(Path(a.scores).read_text())['results'] if a.scores else None
        print(catalogue(scores))
    else:
        from evaluate1.run import score
        score(a.case, a.frames, only=a.only, skip_vlm=a.skip_vlm)


if __name__ == '__main__':
    main()
