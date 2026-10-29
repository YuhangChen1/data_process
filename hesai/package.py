"""Per-scene delivery manifest: one entry point for downstream users of the annotated data.

  python -m hesai.package --scene KS-IN-OE-S-F1_01_202609121414

Writes anno_eval/scenes/<key>/manifest.json + README.md. Nothing is copied: the manifest points at the files in
outputs/, auto/ and evaluate1/cases/, with their formats, counts and conventions, and records validation status.
"""
from pathlib import Path
import argparse, collections, json, time
import numpy as np

DP = Path(__file__).resolve().parents[1]


def read(p): return json.loads(Path(p).read_text())


def entry(path, what, fmt, **extra):
    p = DP / path; ok = p.exists()
    size = (sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else p.stat().st_size) if ok else 0
    n = sum(1 for f in p.rglob('*') if f.is_file()) if ok and p.is_dir() else None
    return {'path': path, 'exists': ok, 'bytes': size, **({'files': n} if n is not None else {}), 'content': what, 'format': fmt, **extra}


def build(key):
    src = read(DP / 'sources' / key / 'source.json'); out = f'outputs/{key}'; gi = f'auto/{key}/global_instances'
    a = read(DP / out / 'analysis.json'); fl = a['floors']
    inst = read(DP / gi / 'global_instances.json')['instances'] if (DP / gi / 'global_instances.json').exists() else []
    val = read(DP / gi / 'validation.json') if (DP / gi / 'validation.json').exists() else {}
    desc = read(DP / gi / 'descriptions.json') if (DP / gi / 'descriptions.json').exists() else {'level1': [], 'level2': []}
    fq = DP / gi / 'frame_qa' / 'manifest.json'; fqm = read(fq) if fq.exists() else None
    # trajectory rule: the scan's own walkthrough when shipped, otherwise A* walkthroughs on the navigation map
    wpath = DP / out / 'hesai_walkthrough' / 'main' / 'metadata.json'
    if wpath.exists():
        walk = read(wpath)
        traj = [entry(f'{out}/hesai_walkthrough/main', 'trajectory = the scan\'s own forward walkthrough, aligned: cameras.npz (camera_to_canonical, camera_to_source, K), mesh_depth_mm/, 3DGS re-render check',
                      'NPZ/PNG/JSON', frames=walk['frames'], render_check_psnr_median_db=round(walk['render_check']['psnr_db_median'], 2))]
    else:
        traj = [entry(str(m.parent.relative_to(DP)), 'trajectory = A* region walkthrough planned on the navigation map (the scan ships none): 3DGS RGB, mesh depth mm, cameras.npz', 'MP4/PNG/NPZ')
                for m in sorted((DP / out).glob('floor_*/annotation_walkthrough/component_*/metadata.json'))]
    labels = collections.Counter(i['semantic_label'] for i in inst)
    layers = {
        '0_source': [
            entry(f'sources/{key}', 'symlinks to the raw scan: gaussian.ply (3DGS, SH0, ~7.9M Gaussians) + mesh_post1.obj (= mesh_post.obj, vertex colours)', 'PLY / OBJ', raw_dir=src['raw_dir']),
        ],
        '1_scene_geometry': [
            entry(f'{out}/analysis.json', 'world_to_canonical (scan -Y-up -> leveled Z-up), floors, ceiling gaps, unit status (inferred metres)', 'JSON'),
            entry(f'{out}/floor_0/maps.npz', 'navigation grid 0.05 m: observed/inferred floor, obstacles (0.10-1.80 m slab), clearance, navigable (r=0.25 m)', 'NPZ [row,col] -> origin+([col,row]+.5)*res'),
            entry(f'{out}/floor_0/occupancy.pgm', 'ROS-style occupancy (+ .yaml)', 'PGM/YAML'),
            entry(f'{out}/floor_0/regions.geojson', 'automatic region candidates (watershed on free space; not semantic rooms)', 'GeoJSON canonical XY'),
            entry(f'{out}/floor_0/topdown_3dgs.png', 'top-down 3DGS render (ceiling clipped) + mesh height map', 'PNG'),
            entry(f'{out}/floor_0/panoramas', '360° equirectangular 3DGS panoramas at region centres + metadata', 'JPG 2048x1024 + JSON'),
        ] + traj,
        '2_annotations': [
            entry(f'{gi}/frames.json', 'frame index of the annotated clip: image, mesh depth, K, camera_to_canonical, timestamp', 'JSON'),
            entry(f'{gi}/frames', 'annotated RGB frames (JPEG q92 of the walkthrough PNGs)', 'JPG'),
            entry(f'{gi}/global_instances.json', '3D object instances: global_id, semantic_label (+fine_label, vote details), category thing/stuff, observed-surface AABB, best view', 'JSON canonical metres'),
            entry(f'{gi}/fine_instances.npz', 'point cloud (4 cm voxels of the mesh) with per-point global ID / superpoint ID — the source of truth for IDs', 'NPZ'),
            entry(f'{gi}/labeled_points.ply', 'same point cloud with instance colours, global_id, semantic_id, superpoint_id', 'PLY'),
            entry(f'{gi}/frame_assignments.json', 'per frame: which global IDs are visible, mask support, purity, accepted flag', 'JSON'),
            entry(f'{gi}/rendered', 'per frame numeric instance / semantic masks (uint16 PNG; 0 = unknown) + colour/overlay visualisations', 'PNG'),
            entry(f'{gi}/semantic_registry.json', 'semantic_id <-> label vocabulary', 'JSON'),
            entry(f'{gi}/descriptions.json', 'GPT (gpt-6.1-sol) text: level1 per-frame description with visible IDs, level2 adjacent-frame change description', 'JSON',
                  level1=len(desc['level1']), level2=len(desc['level2'])),
            entry(f'{gi}/frame_qa', 'single-frame five-level QA (recognition, attributes, relative position, world coordinates, camera distance) grounded on measured facts', 'JSON',
                  **({'questions': fqm.get('questions_total'), 'answer_status': fqm.get('answer_status_counts')} if fqm else {'status': 'not generated'})),
            entry(f'{gi}/videos', 'per-clip rgb / overlay / instance / semantic videos', 'MP4'),
            entry(f'{gi}/qa', 'automatic QA evidence: conflicts, reprojection metrics, issue list for human review', 'CSV/JSON'),
            entry(f'{gi}/evidence', 'every GPT request/response with image hashes and actual model (no keys)', 'JSON'),
        ],
        '3_evaluation': [
            entry(f'evaluate1/cases/{key}', 'evaluate1 case: 10 input frames, 6 keypoint revisits (3 actions + 3 inverses), trajectory/cameras for rendering a 4DGS, object masks, scene text', 'JSON/NPZ/PNG'),
        ],
    }
    summary = {'scene_id': key, 'label_zh': src['label_zh'], 'category_path': src['category_path'],
               'floors': [{'floor_id': f['floor_id'], 'height': round(f['height'], 3), 'ceiling_gap': round(f['ceiling_gap'], 3),
                           'navigable_area_m2': round(f['navigable_area'], 1)} for f in fl],
               'walkthrough_floor_check': a.get('walkthrough_floor_check'), 'trajectory': 'provided_walkthrough' if wpath.exists() else 'astar_walkthrough',
               'annotated_frames': len(json.loads((DP / gi / 'frames.json').read_text())['frames']) if (DP / gi / 'frames.json').exists() else 0,
               'global_instances': len(inst), 'labelled_instances': sum(i['semantic_label'] != 'unknown' for i in inst),
               'labels': dict(labels.most_common()), 'validation_passed': val.get('passed'), 'review_status': 'unreviewed (automatic)',
               'labelling_model': 'gpt-6.1-sol (GPT gateway) + SAM3 (model/sam3.pt)'}
    conventions = {'canonical': 'leveled Z-up, units = inferred metres (not independently verified); world_to_canonical in analysis.json maps scan coordinates',
                   'camera': 'OpenCV camera (+X right, +Y down, +Z forward), camera-to-world matrices; pixel centres at (u+.5, v+.5)',
                   'depth': 'mesh first-hit camera-Z depth, uint16 millimetres, 0 = invalid',
                   'ids': 'global IDs are per scene integers from fine_instances.npz:point_fine_ids; 0 = unknown',
                   'review': 'everything is automatic and marked unreviewed; qa/issues.json lists review leads'}
    return {'schema': 'data_pipeline.scene_manifest.v1', 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'summary': summary,
            'conventions': conventions, 'layers': layers}


def readme(m):
    s = m['summary']; L = [f"# {s['scene_id']} — {s['label_zh']}", '',
        f"类别：{' / '.join(s['category_path']) or '—'}。楼层 {len(s['floors'])}，可通行 {s['floors'][0]['navigable_area_m2']} m²（推断米制）。",
        f"标注：{s['annotated_frames']} 帧，{s['global_instances']} 个 3D 实例（{s['labelled_instances']} 个有类别），校验 {'通过' if s['validation_passed'] else '未通过'}，全部 unreviewed。", '']
    for layer, items in m['layers'].items():
        L += [f'## {layer}', '', '| 路径 | 内容 | 格式 |', '|---|---|---|']
        for e in items:
            flag = '' if e['exists'] else ' **(缺失)**'
            L.append(f"| `{e['path']}`{flag} | {e['content']} | {e['format']} |")
        L.append('')
    L += ['## 约定', ''] + [f'- **{k}**：{v}' for k, v in m['conventions'].items()] + ['']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scene', required=True); a = ap.parse_args()
    m = build(a.scene); d = DP / 'scenes' / a.scene; d.mkdir(parents=True, exist_ok=True)
    (d / 'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=2)); (d / 'README.md').write_text(readme(m))
    print(json.dumps(m['summary'], ensure_ascii=False)[:600]); print('missing', [e['path'] for l in m['layers'].values() for e in l if not e['exists']])


if __name__ == '__main__':
    main()
