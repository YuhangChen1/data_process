"""Measured per-frame facts; no IDs or numeric answers are inferred by GPT."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import cv2
import numpy as np
from .common import read, sha, fingerprint

COLOR_ZH = {'black': '黑色', 'white': '白色', 'gray': '灰色', 'red': '红色',
            'orange': '橙色', 'yellow': '黄色', 'green': '绿色', 'cyan': '青色',
            'blue': '蓝色', 'purple': '紫色', 'pink': '粉色', 'brown': '棕色',
            'beige': '米色', 'unknown': '不确定'}
PALETTE = list(COLOR_ZH)[:-1]
COORDINATE_CONVENTION = (
    '位置采用本场景 canonical 世界坐标系，Z 轴向上，X/Y 是重建水平轴，'
    '不代表已标定的东/北；数值单位为 inferred_meter（推断米制）。'
    '坐标指观测表面三维框中心，范围只覆盖重建中已观测部分。'
)


def rounded(value, digits=3):
    return np.asarray(value).round(digits).tolist()


def color_evidence(rgb):
    """Describe mask RGB, not material/true illumination-independent color."""
    if not len(rgb):
        return {'label': 'unknown', 'name_zh': '不确定', 'score': 0.,
                'source': 'current_rgb+exclusive_sam3_mask', 'status': 'unknown'}
    # Deterministic bounded sampling, without changing the original mask.
    sample = np.asarray(rgb)[np.linspace(0, len(rgb) - 1, min(4096, len(rgb)), dtype=int)]
    hsv = cv2.cvtColor(sample.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, s, v = hsv.T.astype(float)
    labels = np.full(len(sample), PALETTE.index('gray'), dtype=np.int16)
    for name, selected in [
        ('red', (h < 10) | (h >= 172)), ('orange', (h >= 10) & (h < 23)),
        ('yellow', (h >= 23) & (h < 38)), ('green', (h >= 38) & (h < 83)),
        ('cyan', (h >= 83) & (h < 99)), ('blue', (h >= 99) & (h < 132)),
        ('purple', (h >= 132) & (h < 158)), ('pink', (h >= 158) & (h < 172)),
        ('brown', (h < 30) & (s >= 45) & (v < 175)),
        ('beige', (h >= 10) & (h < 38) & (s >= 20) & (s < 100) & (v >= 175)),
        ('gray', s < 35), ('white', (s < 35) & (v > 200)), ('black', v < 52),
    ]:
        labels[selected] = PALETTE.index(name)
    counts = np.bincount(labels, minlength=len(PALETTE))
    order = np.argsort(-counts, kind='stable')
    shares = counts / len(sample)
    best = int(order[0]); label = PALETTE[best]
    name = COLOR_ZH[label]
    if shares[best] < .42:
        label = 'unknown'; name = '多种颜色，主色不确定'
    elif shares[order[1]] >= .24:
        name += '与' + COLOR_ZH[PALETTE[order[1]]] + '混合'
    return {'label': label, 'name_zh': name, 'score': round(float(shares[best]), 3),
            'dominant_colors': [{'label': PALETTE[int(i)], 'fraction': round(float(shares[i]), 3)}
                                for i in order[:3] if counts[i]],
            'median_rgb': rounded(np.median(sample, axis=0), 0),
            'sample_pixels': len(sample), 'source': 'current_rgb+exclusive_sam3_mask:HSV_bins',
            'status': 'estimated' if label != 'unknown' else 'unknown',
            'caveat': 'Rendered apparent color; affected by lighting, reconstruction and mask leakage.'}


def image_relation(a, b, width, height):
    delta = np.array(a['image']['center_px']) - np.array(b['image']['center_px'])
    horizontal = 'left_of' if delta[0] < -.04 * width else 'right_of' if delta[0] > .04 * width else 'aligned_x'
    vertical = 'above' if delta[1] < -.04 * height else 'below' if delta[1] > .04 * height else 'aligned_y'
    ca, cb = np.array(a['bbox']['center']), np.array(b['bbox']['center'])
    sa, sb = np.array(a['bbox']['size']), np.array(b['bbox']['size'])
    gap = np.maximum(np.abs(ca - cb) - (sa + sb) / 2, 0)
    return {'subject_global_id': a['global_id'], 'reference_global_id': b['global_id'],
            'image_horizontal': horizontal, 'image_vertical': vertical,
            'image_delta_px': rounded(delta, 2), 'reference_frame': 'image_x_right_y_down',
            'canonical_center_delta_xyz': rounded(ca - cb),
            'canonical_center_distance': round(float(np.linalg.norm(ca - cb)), 3),
            'observed_aabb_gap': round(float(np.linalg.norm(gap)), 3),
            'source': 'exclusive_mask_centers+observed_3d_aabbs',
            'score': min(a['score'], b['score']),
            'not_inferred': ['physical_contact', 'support_on', 'compass_direction']}


def camera_comparison(objects):
    eligible = [o for o in objects if o['category_type'] == 'thing' and o['camera'].get('support_points', 0) >= 20]
    candidates = []
    for i, a in enumerate(eligible):
        for b in eligible[i + 1:]:
            da, db = a['camera']['visible_surface_distance_median'], b['camera']['visible_surface_distance_median']
            margin = max(.2, .08 * min(da, db))
            # Prefer distinct, clearly separated foreground/background objects.
            priority = (abs(da - db) > margin, a['semantic_label'] != b['semantic_label'],
                        min(a['image']['mask_pixels'], b['image']['mask_pixels']), abs(da - db))
            candidates.append((priority, a, b, da, db, margin))
    if not candidates:
        return {'status': 'insufficient_evidence', 'global_ids': [o['global_id'] for o in eligible[:1]],
                'reason': 'Fewer than two recognized thing instances with 20 depth-consistent visible points.',
                'source': 'camera_pose+depth_consistent_fine_points', 'score': 0.}
    _, a, b, da, db, margin = max(candidates, key=lambda row: row[0])
    distinct = abs(da - db) > margin
    near, far = (a, b) if da < db else (b, a)
    return {'status': 'answerable' if distinct else 'ambiguous',
            'global_ids': [a['global_id'], b['global_id']],
            'nearer_global_id': near['global_id'] if distinct else None,
            'farther_global_id': far['global_id'] if distinct else None,
            'distances': {str(a['global_id']): da, str(b['global_id']): db},
            'difference': round(abs(da - db), 3), 'minimum_distinct_margin': round(margin, 3),
            'distance_definition': 'median Euclidean camera-to-depth-consistent-visible-surface-point distance',
            'reference_frame': 'current_frame_camera_origin',
            'units': 'inferred_meter', 'source': 'camera_pose+depth_consistent_fine_points',
            'score': min(a['score'], b['score']) if distinct else 0.,
            'caveat': 'A comparison of visible surface distance statistics, not entire object centers or navigation distance.'}


def choose_focus(objects, limit=3):
    ordered = sorted(objects, key=lambda o: (o['category_type'] != 'thing', -o['image']['mask_pixels'], o['global_id']))
    focus = []; labels = set()
    for obj in ordered:
        if obj['semantic_label'] in labels:
            continue
        focus.append(obj); labels.add(obj['semantic_label'])
        if len(focus) == limit:
            break
    return focus


def build_facts(out, frame, instances, assignments, points, point_ids, source_hashes):
    mask_path = out / 'masks' / frame['clip_id'] / (frame['id'] + '.npz')
    projection_path = out / 'projections' / frame['clip_id'] / (frame['id'] + '.npz')
    with np.load(mask_path) as data:
        local = data['local_instances']
    with np.load(projection_path) as data:
        indices, pixels = data['point_indices'], data['pixels']
    image_path = out / frame['rgb']
    im = cv2.imread(str(image_path))
    if im is None or im.shape[:2] != local.shape or local.shape != (frame['height'], frame['width']):
        raise ValueError('RGB/mask/frame dimensions disagree: ' + frame['id'])
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pose = np.array(frame['camera_to_canonical'], dtype=float)
    origin = pose[:3, 3]
    groups = defaultdict(list)
    for a in assignments:
        if a['accepted'] and a['global_id'] > 0 and a['global_id'] in instances:
            groups[a['global_id']].append(a)
    objects = []; unknown = []
    for gid, rows in sorted(groups.items()):
        obj = instances[gid]
        if obj['semantic_label'] == 'unknown':
            unknown.append(gid)
            continue
        local_ids = [r['local_id'] for r in rows]
        inside = np.isin(local, local_ids)
        yy, xx = np.where(inside)
        if not len(xx):
            continue
        supports = (point_ids[indices] == gid) & np.isin(local.ravel()[pixels], local_ids)
        xyz = points[indices[supports]]
        camera_xyz = (xyz - origin) @ pose[:3, :3]
        distance = np.linalg.norm(xyz - origin, axis=1)
        extent = [int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)]
        ratio = extent[0] / extent[1]
        orientation = '横向延展' if ratio > 1.5 else '竖向延展' if ratio < 1 / 1.5 else '宽高较接近'
        color = color_evidence(rgb[inside])
        record = {'global_id': gid, 'object_id': out.parent.name + '::g' + str(gid),
                  'semantic_label': obj['semantic_label'], 'category_type': obj['category_type'],
                  'semantic_source': obj['semantic_pred']['source'], 'local_ids': local_ids,
                  'score': round(min(obj['score'], max(r['purity'] for r in rows), max(r['score'] for r in rows)), 3),
                  'bbox': obj['bbox'], 'floor_id': frame['floor'], 'frame': 'canonical', 'units': 'inferred_meter',
                  'image': {'bbox_xyxy': [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
                            'center_px': [round(float(xx.mean()), 2), round(float(yy.mean()), 2)],
                            'mask_pixels': len(xx), 'area_fraction': round(len(xx) / local.size, 4),
                            'extent_px': extent, 'extent_orientation': orientation,
                            'geometry_scope': 'visible_mask_projection_not_amodal_shape'},
                  'attributes': {'color': color, 'material': {'label': 'unknown', 'source': 'not_annotated', 'score': 0.}},
                  'camera': {'support_points': len(xyz), 'source': 'camera_pose+zbuffer+mesh_depth+fine_global_id',
                             'camera_origin_canonical': rounded(origin), 'camera_axes': 'OpenCV +X right/+Y down/+Z forward'},
                  'review': {'status': 'unreviewed'}}
        if len(xyz):
            record['camera'].update(
                visible_surface_distance_median=round(float(np.median(distance)), 3),
                visible_surface_distance_q10_q90=rounded(np.quantile(distance, [.1, .9])),
                visible_surface_camera_xyz_median=rounded(np.median(camera_xyz, axis=0)),
                visible_surface_depth_z_median=round(float(np.median(camera_xyz[:, 2])), 3),
                bbox_center_euclidean_distance=round(float(np.linalg.norm(np.array(obj['bbox']['center']) - origin)), 3))
        objects.append(record)
    focus = choose_focus(objects)
    relations = [image_relation(focus[0], other, frame['width'], frame['height']) for other in focus[1:]] if focus else []
    comparison = camera_comparison(objects)
    mask_digest = hashlib.sha256(local.tobytes()).hexdigest()
    actual_rgb_sha256 = sha(image_path)
    manifest_rgb_sha256 = frame.get('rgb_sha256')
    return {'schema': 'kosmoworld.frame_qa_facts', 'schema_version': '1.0',
            'scene_id': out.parent.name, 'frame_id': frame['id'], 'clip_id': frame['clip_id'],
            'floor_id': frame['floor'], 'timestamp': frame['timestamp'],
            'frame': 'canonical', 'units': 'inferred_meter',
            'camera_to_canonical': frame['camera_to_canonical'], 'K': frame['K'],
            'image_size': [frame['width'], frame['height']], 'rgb': frame['rgb'],
            'objects': objects, 'unclassified_visible_global_ids': unknown,
            'focus_global_ids': [o['global_id'] for o in focus], 'relative_relations': relations,
            'camera_comparison': comparison, 'source': 'existing_automatic_annotations+measured_frame_geometry',
            'annotation_status': 'automatic_unreviewed_not_human_ground_truth',
            'source_hashes': source_hashes,
            'source_integrity': {'manifest_rgb_sha256': manifest_rgb_sha256,
                                 'actual_rgb_sha256': actual_rgb_sha256,
                                 'manifest_hash_matches': bool(manifest_rgb_sha256 == actual_rgb_sha256),
                                 'warning': None if manifest_rgb_sha256 == actual_rgb_sha256 else
                                            'The frame manifest hash is stale; current RGB bytes are retained and measured.'},
            'evidence': {'rgb_sha256': sha(image_path), 'local_instances_sha256': mask_digest,
                         'masks': str(mask_path.relative_to(out)), 'projection': str(projection_path.relative_to(out)),
                         'projection_sha256': sha(projection_path)},
            'limitations': ['Visibility inventory covers accepted masks, not proof of exhaustive scene coverage.',
                           'Fine instances can be fragments; IDs are scene-local and are not object counts.',
                           'Color is an automatic statistic of rendered masked RGB, not a human-verified material attribute.',
                           '3D boxes cover observed surfaces, not necessarily whole objects.'],
            'review': {'status': 'unreviewed'}}


def object_names(facts, glossary):
    groups = defaultdict(list)
    for obj in facts['objects']:
        groups[obj['semantic_label']].append(obj)
    names = {}
    for label, objects in groups.items():
        ordered = sorted(objects, key=lambda o: (*o['image']['center_px'], o['global_id']))
        for i, obj in enumerate(ordered):
            name = glossary.get(label, label)
            if len(ordered) > 1:
                name = f'已标注{ name }中从左到右第{i + 1}个'
            names[obj['global_id']] = name
    return names


def phrase_slots(facts, glossary):
    names = object_names(facts, glossary)
    byid = {o['global_id']: o for o in facts['objects']}
    focus = [byid[g] for g in facts['focus_global_ids']]
    pair = [byid[g] for g in facts['camera_comparison']['global_ids']]
    inventory = list(dict.fromkeys(glossary.get(o['semantic_label'], o['semantic_label']) for o in facts['objects'] if o['category_type'] == 'thing'))
    surfaces = list(dict.fromkeys(glossary.get(o['semantic_label'], o['semantic_label']) for o in facts['objects'] if o['category_type'] == 'stuff'))

    def attributes(objects):
        if not objects:
            return '该帧没有可可靠绑定类别的物体，无法确认颜色和轮廓。'
        parts = []
        for o in objects:
            im = o['image']; width, height = facts['image_size']
            parts.append(f'{names[o["global_id"]]}的可见区域主要呈{o["attributes"]["color"]["name_zh"]}，'
                         f'投影范围{im["extent_orientation"]}，覆盖画面宽约{im["extent_px"][0] / width:.0%}、'
                         f'高约{im["extent_px"][1] / height:.0%}')
        return '；'.join(parts) + '。这里的轮廓范围不代表完整物体的真实形状。'

    relation_texts = []
    for r in facts['relative_relations']:
        parts = []
        if r['image_horizontal'] != 'aligned_x':
            parts.append('左侧' if r['image_horizontal'] == 'left_of' else '右侧')
        if r['image_vertical'] != 'aligned_y':
            parts.append('上方' if r['image_vertical'] == 'above' else '下方')
        placement = '、'.join(parts) if parts else '中心大致对齐的位置'
        relation_texts.append(f'在当前图像中，{names[r["subject_global_id"]]}位于{names[r["reference_global_id"]]}的{placement}；'
                              f'两者观测框中心的三维直线距离约{r["canonical_center_distance"]:.2f}个推断米制单位')
    relative = '。'.join(relation_texts) + '。' if relation_texts else '缺少两个类别可辨且可绑定的对象，不能可靠说明物体间相对位置。'
    global_texts = []
    for o in focus:
        center, size = o['bbox']['center'], o['bbox']['size']
        xyz = ', '.join(f'{v:.2f}' for v in center)
        extent = ' × '.join(f'{v:.2f}' for v in size)
        global_texts.append(f'{names[o["global_id"]]}在{o["floor_id"]}层，观测表面框中心为({xyz})，'
                            f'沿X/Y/Z的观测范围为{extent}')
    global_text = '；'.join(global_texts) + '。' if global_texts else '该帧没有可绑定物体的三维位置。'
    comparison = facts['camera_comparison']
    if comparison['status'] == 'answerable':
        near, far = comparison['nearer_global_id'], comparison['farther_global_id']
        camera = (f'以当前拍摄相机为起点，到可见表面点的三维欧氏距离中位数为准，'
                  f'{names[near]}较近（约{comparison["distances"][str(near)]:.2f}），'
                  f'{names[far]}较远（约{comparison["distances"][str(far)]:.2f}），'
                  f'相差约{comparison["difference"]:.2f}个推断米制单位。'
                  '这是可见表面的距离统计，不是物体完整中心距离，也不是沿地面可行走的距离。')
    elif comparison['status'] == 'ambiguous':
        ds = '、'.join(f'{names[o["global_id"]]}约{comparison["distances"][str(o["global_id"])]:.2f}' for o in pair)
        camera = f'相机到可见表面的距离中位数分别为{ds}个推断米制单位，差异不足设定阈值，不能可靠判定谁更近。'
    else:
        camera = '缺少至少两个带充分三维可见点支持的独立物体，不能可靠比较与当前拍摄相机的远近。'
    note = '这里只列出当前帧可靠绑定的已标注类别，不把分割片段数当作完整实物数量。'
    if facts['unclassified_visible_global_ids']:
        note += '另有可见分割区域尚未确认类别。'
    return {'inventory': '、'.join(inventory) or '暂无可可靠识别的独立物体类别',
            'surfaces': '、'.join(surfaces) or '暂无可可靠绑定的连续表面类别',
            'inventory_note': note, 'focus_names': '、'.join(names[o['global_id']] for o in focus) or '当前可见的已标注对象',
            'attribute_details': attributes(focus), 'relative_details': relative,
            'global_details': global_text, 'coordinate_convention': COORDINATE_CONVENTION,
            'pair_names': '与'.join(names[o['global_id']] for o in pair) or '当前可见的独立物体',
            'pair_attributes': attributes(pair), 'camera_details': camera}
