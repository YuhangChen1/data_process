"""Pipeline operators of anno_eval — every 3D processing step packaged behind one interface.

    python ops3d.py list                                     # stages, operators, inputs -> outputs, device
    python ops3d.py run geom.analyze --scene <key>           # run one operator (GPU ones belong on a GPU box)
    python ops3d.py describe > docs/ops3d.json               # machine-readable interface (feeds docs/pipeline_flow.html)

    from ops3d import OPS, run
    run('traj.align_walkthrough', 'KS-IN-OE-S-F1_01_202609121414')

An operator is `fn(scene, **params) -> dict` registered with @op(name, stage, title, inputs, outputs, device, when).
inputs / outputs are path templates relative to anno_eval/ ({key} = scene key, {raw} = the scan folder under
hesai_dataset/); `when` states the condition under which the pipeline runs it. Upstream blocks (the scan and Dongting's
walkthrough renderer) are registered as external operators: they document their interface but are not run here.
The 72 evaluation metrics are the separate operator set in operators.py; eval.score runs them all.
"""
from pathlib import Path
import argparse, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
OPS = {}
STAGES = ['upstream', '3d_geometry', 'trajectory', '3d_annotation', 'evaluation']
STAGE_ZH = {'upstream': '上游：海塞扫描 + 董婷漫游输出', '3d_geometry': '3D 几何', 'trajectory': '轨迹',
            '3d_annotation': '3D 标注', 'evaluation': '评测 evaluate1'}


def op(name, stage, title, inputs, outputs, device='cpu', when='always', call=None, external=False, owner='anno_eval'):
    def wrap(fn):
        OPS[name] = dict(name=name, stage=stage, title=title, inputs=inputs, outputs=outputs, device=device, when=when,
                         call=call or f'ops3d.run({name!r}, scene)', external=external, owner=owner,
                         doc=(fn.__doc__ or '').strip(), fn=fn)
        return fn
    return wrap


def _module(mod, *args, env=None):
    e = dict(os.environ); e.update(env or {})
    subprocess.run([sys.executable, '-m', mod, *args], cwd=ROOT, env=e, check=True)


def _global(stage, scene):
    _module('global_pipeline.run', '--scene', scene, '--stage', stage, env={'GLOBAL_VLM_MODEL': 'gpt-6.1-sol'})
    return {'stage': stage}


# ---------------------------------------------------------------- upstream (external: documented, not run here)

@op('upstream.scan', 'upstream', '海塞扫描交付', [],
    ['{raw}/gaussian.ply', '{raw}/mesh_post.obj', '{raw}/mesh_post.ply', '{raw}/data.json', '{raw}/cover_image.png'],
    external=True, owner='海塞', call='—')
def upstream_scan(scene):
    """Per scene: 3DGS (SH0, ~8M Gaussians, scan frame −Y up), coloured mesh, the scanner's own camera trajectory
    (data.json: cam2world + timestamps, no images) and a cover image."""


@op('upstream.manyou_probe', 'upstream', '董婷 manyou：自由空间探针 + 低清试拍评分', ['{raw}/gaussian.ply'],
    ['{raw}/manyou_videos/main_walkthrough/probes.json', '.../adaptive_parameters.json', '.../lowres/*.png', '.../metrics.json'],
    device='gpu', external=True, owner='董婷', call='manyou src/main_walkthrough2/generate_main.py')
def upstream_probe(scene):
    """Adaptive clearance/voxel from the PLY's 5–95% box; safe stand points between opaque Gaussian surfaces, several
    headings each; low-res renders scored by valid depth, contrast, gradient and foreground fill."""


@op('upstream.manyou_walkthrough', 'upstream', '董婷 manyou：主漫游路线 + 高清渲染',
    ['.../probes.json', '.../metrics.json', '{raw}/gaussian.ply'],
    ['{raw}/manyou_videos/main_walkthrough/poses.json', '.../frames_hd/*.png (1536×1024)', '.../walkthrough.mp4 (12 fps)',
     '.../preview_quality/', '.../quality_gate.json'],
    device='gpu', external=True, owner='董婷', call='manyou src/main_walkthrough/generate.py + render_v1 + video_v1')
def upstream_walkthrough(scene):
    """Best-scoring safe headings become anchors, connected with clearance checks and interpolated into one continuous
    route (c2w in the scan frame, OpenCV, fov_y 70°), then rendered in HD and encoded."""


@op('upstream.capture_aware', 'upstream', '采集感知漫游 videos/', ['{raw}/data.json', '{raw}/gaussian.ply'],
    ['{raw}/videos/poses.json (~2300 poses)', '{raw}/videos/walkthrough.mp4', '{raw}/videos/quality_repair.json'],
    device='gpu', external=True, owner='董婷', call='manyou capture-aware planner')
def upstream_capture(scene):
    """Follows the scanner's real trajectory (data.json) interpolated to video rate; blurred / badly exposed frames
    are dropped (quality_repair.json)."""


# ---------------------------------------------------------------- 3D geometry

@op('scan.source', '3d_geometry', '登记扫描', ['{raw}/gaussian.ply', '{raw}/mesh_post.obj'],
    ['sources/{key}/gaussian.ply', 'sources/{key}/mesh_post1.obj', 'sources/{key}/source.json'])
def scan_source(scene):
    """Links the raw scan under a stable ASCII key (category path and Chinese label kept in source.json)."""
    from hesai.stage import source
    return {'source': str(source(scene))}


@op('geom.analyze', '3d_geometry', '调平 · 楼层 · 可通行图 · 区域',
    ['sources/{key}/mesh_post1.obj', 'sources/{key}/gaussian.ply'],
    ['outputs/{key}/analysis.json (world_to_canonical, floors)', 'outputs/{key}/mesh_cache.npz', 'outputs/{key}/alignment.json',
     'outputs/{key}/floor_k/maps.npz (0.05 m, r=0.25 m)', 'outputs/{key}/floor_k/regions.geojson', 'outputs/{key}/floor_k/annotation_routes.json'])
def geom_analyze(scene, force=False):
    """−Y-up scan → leveled Z-up canonical frame (dominant floor normal); floor/ceiling peaks; navigation grid from
    the 0.10–1.80 m obstacle slab; watershed region candidates; A* routes through region centres (planned only)."""
    from hesai.stage import geometry
    a = geometry(scene, force)
    return {'floors': [f['floor_id'] for f in a['floors']]}


@op('geom.floor_check', '3d_geometry', '楼层实测（漫游相机正下方）',
    ['{raw}/manyou_videos/main_walkthrough/poses.json', 'outputs/{key}/mesh_cache.npz', 'outputs/{key}/analysis.json'],
    ['outputs/{key}/analysis.json:walkthrough_floor_check', 'outputs/{key}/floor_k/maps.npz (rebuilt if > 8 cm off)'],
    when='扫描自带漫游时')
def geom_floor_check(scene):
    """Down-rays from the walkthrough cameras measure the walked floor; when it differs from the chosen plane by more
    than 8 cm the floor maps are rebuilt at that height (leveling unchanged)."""
    from hesai.stage import walkthrough_floor
    return walkthrough_floor(scene)


@op('geom.render_views', '3d_geometry', '俯视图 · 全景 · 深度一致性', ['outputs/{key}/analysis.json', 'outputs/{key}/floor_k/maps.npz',
    'outputs/{key}/floor_k/regions.json', 'sources/{key}/gaussian.ply'],
    ['outputs/{key}/floor_k/topdown_3dgs.png', 'outputs/{key}/floor_k/panoramas/*.jpg (2048×1024)',
     'outputs/{key}/floor_k/alignment_views/', 'outputs/{key}/floor_k/depth_alignment.json'], device='gpu')
def geom_render_views(scene):
    """3DGS top-down (ceiling clipped), 360° panoramas at region centres, 3DGS-vs-mesh depth agreement views."""
    from scene_pipeline.render import render_scene
    from hesai.stage import source, DP
    render_scene(source(scene), DP / 'outputs' / scene, walkthroughs=False)
    return {}


# ---------------------------------------------------------------- trajectory

@op('traj.align_walkthrough', 'trajectory', '对齐自带漫游',
    ['{raw}/manyou_videos/main_walkthrough/poses.json', '{raw}/manyou_videos/main_walkthrough/frames_hd/*.png',
     'outputs/{key}/analysis.json', 'outputs/{key}/mesh_cache.npz', 'sources/{key}/gaussian.ply'],
    ['outputs/{key}/hesai_walkthrough/main/cameras.npz (camera_to_canonical, camera_to_source, K)',
     'outputs/{key}/hesai_walkthrough/main/mesh_depth_mm/*.png', 'outputs/{key}/hesai_walkthrough/main/metadata.json (PSNR check)'],
    device='gpu', when='扫描自带漫游时（86/90）')
def traj_align(scene, force=False):
    """poses.json c2w (scan frame) → canonical; K from fov_y; same-camera mesh depth; every frame re-rendered from the
    3DGS and compared to frames_hd (pilot median PSNR 26.4 dB) to prove pose and intrinsics."""
    from hesai.stage import walkthrough
    m = walkthrough(scene, force)
    return {'frames': m['frames'], 'psnr_median': m['render_check']['psnr_db_median']} if m else {'skipped': 'no walkthrough'}


@op('traj.astar_walkthrough', 'trajectory', 'A* 规划漫游（兜底）',
    ['outputs/{key}/floor_k/maps.npz', 'outputs/{key}/floor_k/annotation_routes.json', 'sources/{key}/gaussian.ply', 'outputs/{key}/mesh_cache.npz'],
    ['outputs/{key}/floor_k/annotation_walkthrough/component_n/{rgb.mp4, mesh_depth_mm/, cameras.npz, metadata.json}'],
    device='gpu', when='扫描没有自带漫游时（4/90）')
def traj_astar(scene):
    """Collision-checked A* through region centres per connected component, smoothed, rendered at 1.5 m camera height
    (960×540, 8 fps) with mesh depth."""
    from hesai.stage import astar
    return {'clips': len(astar(scene))}


# ---------------------------------------------------------------- 3D annotation (global_pipeline)

@op('anno.frames', '3d_annotation', '标注帧索引', ['outputs/{key}/hesai_walkthrough/main/ 或 annotation_walkthrough/'],
    ['auto/{key}/global_instances/frames.json (rgb, depth, K, camera_to_canonical)', 'auto/{key}/global_instances/frames/<clip>/*.jpg'])
def anno_frames(scene):
    """One record per trajectory frame: image, aligned mesh depth, intrinsics, canonical pose, timestamp."""
    return _global('frames', scene)


@op('anno.inventory', '3d_annotation', 'GPT 逐帧类别清单', ['auto/{key}/global_instances/frames/'],
    ['auto/{key}/global_instances/concepts.json (类别 + SAM3 提示词)', 'auto/{key}/global_instances/evidence/gpt_inventory/'], device='gpt')
def anno_inventory(scene):
    """gpt-6.1-sol lists every visible segmentable type per frame with 1–3 SAM3 prompts (thing / stuff)."""
    return _global('discover', scene)


@op('anno.sam3_segment', '3d_annotation', 'SAM3 分割', ['auto/{key}/global_instances/frames/', 'auto/{key}/global_instances/concepts.json'],
    ['auto/{key}/global_instances/masks/<clip>/*.npz|json (互斥实例 mask)'], device='gpu',
    call='hesai/annotate.sh 启停 SAM3 服务后 ops3d.run("anno.sam3_segment", scene)')
def anno_segment(scene):
    """SAM3 text-prompted instances per frame; score ≥ 0.40, area ≥ 80 px, NMS IoU 0.80, exclusive pixel ownership.
    Needs SAM3 workers (GLOBAL_SAM_URLS); hesai/annotate.sh starts them on the GPU only for this step."""
    return _global('segment', scene)


@op('anno.superpoints', '3d_annotation', '超点 + z-buffer 投影', ['outputs/{key}/mesh_cache.npz', 'outputs/{key}/analysis.json',
    'auto/{key}/global_instances/frames.json'],
    ['auto/{key}/global_instances/superpoints.npz', 'auto/{key}/global_instances/projections/<clip>/*.npz (每帧可见点 + 像素)'])
def anno_superpoints(scene):
    """Mesh vertices voxelised at 4 cm; FH segmentation on a coloured kNN graph (≤ 300 points, ≤ 0.65 m); per frame
    the depth-consistent visible points and their pixels."""
    return _global('geometry', scene)


@op('anno.instances_3d', '3d_annotation', '3D 实例融合', ['masks/', 'superpoints.npz', 'projections/'],
    ['auto/{key}/global_instances/proposals.npz', 'coarse_instances.npz', 'fine_instances.npz (points, point_fine_ids)'])
def anno_instances(scene):
    """Mask → superpoint proposals; merge by shared superpoints with cannot-link sets; fine boundary migration with
    ≥ 2 independent views. fine_instances.npz:point_fine_ids is the source of every global ID."""
    return _global('instances', scene)


@op('anno.export', '3d_annotation', '多视角投票 → GPT 补 unknown → 剔除碎片 → 逐帧 PNG',
    ['fine_instances.npz', 'masks/', 'projections/', 'frames/'],
    ['auto/{key}/global_instances/global_instances.json (类别, fine_label, AABB, dropped_instances)', 'frame_assignments.json',
     'relabel_unknown.json', 'rendered/<clip>/{instance,semantic}/*.png (uint16)', 'labeled_points.ply', 'videos/'], device='gpt')
def anno_export(scene):
    """Re-project fine IDs into every mask; vote canonicalised labels across views (margin 0.15, ≥ 2 views); unknowns
    → one gpt-6.1-sol look at outlined views (accept: visible, single object, conf ≥ 0.6); instances no frame shows are
    dropped (IDs → 0); numeric instance/semantic PNGs, videos and labelled point cloud."""
    return _global('export', scene)


@op('anno.text', '3d_annotation', '文字层（可选）', ['global_instances.json', 'frame_assignments.json', 'frames/'],
    ['auto/{key}/global_instances/descriptions.json', 'auto/{key}/global_instances/frame_qa/ (五档问答)'], device='gpt', when='WITH_TEXT=1')
def anno_text(scene):
    """Per-frame and adjacent-frame descriptions; single-frame five-level QA grounded on measured facts."""
    _global('describe', scene)
    env = {'FRAME_QA_GPT_MODEL': 'gpt-6.1-sol'}
    _module('global_pipeline.frame_qa', '--scene', scene, env=env); _module('global_pipeline.frame_qa_grounded', '--scene', scene, env=env)
    return {}


@op('anno.verify', '3d_annotation', '校验 + 场景清单', ['auto/{key}/global_instances/*'],
    ['auto/{key}/global_instances/validation.json', 'auto/{key}/global_instances/qa/issues.json', 'scenes/{key}/{manifest.json, README.md}'])
def anno_verify(scene):
    """Uint16 labels, mask boundaries, ID registry ↔ point cloud, topology, video frame counts, GPT evidence; then the
    per-scene delivery manifest."""
    r = _global('verify', scene); _module('hesai.package', '--scene', scene); return r


# ---------------------------------------------------------------- evaluation

@op('eval.case', 'evaluation', '评测 case：10 输入 · 6 关键点 · 3+3 动作',
    ['轨迹 clip', 'global_instances.json + rendered/instance', 'floor_k/maps.npz', 'mesh_cache.npz'],
    ['evaluate1/cases/{key}/case.json', 'evaluate1/cases/{key}/trajectory.json', 'evaluate1/cases/{key}/cameras.npz (c2w, K, t)',
     'evaluate1/cases/{key}/input/ (10 帧真值)', 'evaluate1/cases/{key}/objects/ (关键点物体 mask)'], device='gpt')
def eval_case(scene):
    """Walk the trajectory; at 6 keypoints pause, 3 WBench actions (W/S/A/D 0.4 m, LEFT/RIGHT 30°, UP/DOWN 15°) and
    their inverses in reverse order back to the exact pose; translations checked against mesh and nav map."""
    from evaluate1.case import build
    c = build(scene); return {'steps': c['steps'], 'objects': len(c['objects'])}


@op('eval.render_model', 'evaluation', '被测模型渲染', ['evaluate1/cases/{key}/cameras.npz', '被测 4DGS'],
    ['evaluate1/runs/{key}/<model>/frames/<step>.png', 'evaluate1/runs/{key}/<model>/depth/<step>.npy（可选）'], device='gpu',
    call='模型自己渲染；参考：python -m evaluate1.render --case ... --model oracle|ply:<path>')
def eval_render(scene, model='oracle'):
    """The model renders its 4DGS at every step's pose and time t."""
    _module('evaluate1.render', '--case', f'evaluate1/cases/{scene}', '--model', model); return {}


@op('eval.score', 'evaluation', '72 个评测算子', ['evaluate1/cases/{key}/*', 'evaluate1/runs/{key}/<model>/frames/'],
    ['evaluate1/runs/{key}/<model>/scores.json (每个算子: value, status, raw, per_item)', 'scores.md'], device='gpu',
    call='python operators.py run --case ... --frames ...')
def eval_score(scene, model='oracle'):
    """A 画面质量 · B 输入帧保真 · C 设定遵循 · D 交互遵循 · E 时空一致性 · F 物理合理性 · G 关键点重访 · H R2M 重访记忆."""
    from evaluate1.run import score
    d = score(ROOT / 'evaluate1' / 'cases' / scene, ROOT / 'evaluate1' / 'runs' / scene / model / 'frames')
    return d['status_counts']


@op('eval.results', 'evaluation', '交付视图', ['scenes/{key}/', 'evaluate1/cases/{key}/case.json', 'scores.json'],
    ['results/{key}/{annotation.md, case.md, scores_<model>.json|md}', 'OPERATORS.md'])
def eval_results(scene, model='oracle'):
    """Collects the scene manifest, case summary and scores into results/ and regenerates the operator catalogue."""
    subprocess.run([sys.executable, 'tools/make_results.py', '--scene', scene, '--model', model], cwd=ROOT, check=True); return {}


# ----------------------------------------------------------------

def run(name, scene, **params):
    o = OPS[name]
    if o['external']: raise RuntimeError(f'{name} is produced upstream ({o["owner"]}), not by anno_eval')
    return o['fn'](scene, **params)


def describe():
    return [{k: v for k, v in o.items() if k != 'fn'} for o in OPS.values()]


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list'); sub.add_parser('describe')
    r = sub.add_parser('run'); r.add_argument('name'); r.add_argument('--scene', required=True)
    a = ap.parse_args()
    if a.cmd == 'list':
        for st in STAGES:
            print(f'== {STAGE_ZH[st]}')
            for o in (o for o in OPS.values() if o['stage'] == st):
                print(f"  {o['name']:26s} [{o['device']}] {o['title']}  ({o['when']})")
                print(f"      in : {'; '.join(o['inputs']) or '—'}\n      out: {'; '.join(o['outputs'])}")
    elif a.cmd == 'describe':
        print(json.dumps({'stages': STAGES, 'stage_zh': STAGE_ZH, 'ops': describe()}, ensure_ascii=False, indent=1))
    else:
        print(run(a.name, a.scene))


if __name__ == '__main__':
    main()
