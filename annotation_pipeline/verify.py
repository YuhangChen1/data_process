"""Validate actual graphs and their camera, mask, point, and API provenance."""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from jsonschema import Draft202012Validator

from .common import ROOT, read, dump, sha, utc, VOCAB, REGION_TYPES


def verify_scene(out, schema):
    errors = []
    warnings = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    def exists(value):
        path = (out / value).resolve()
        check(path.is_relative_to(ROOT) and path.is_file(), 'Missing/invalid evidence: ' + str(path))
        return path

    graph = read(out / 'scene_graph.json')
    info = read(out / 'run_info.json')
    errors.extend('Schema: ' + str(e.json_path) + ': ' + e.message
                  for e in Draft202012Validator(schema).iter_errors(graph))
    check(graph['scene_id'] == out.name, 'Scene ID differs from directory')
    check(graph['generation']['human_review_applied'] is False, 'Automatic graph includes human review')
    sections = ['floors', 'regions', 'connections', 'objects', 'texts']
    records = [graph['scene']] + sum([graph[k] for k in sections], [])
    ids = [r['id'] for r in records]
    check(len(ids) == len(set(ids)), 'Duplicate entity IDs')
    known = set(ids)
    floors = {x['id']: x for x in graph['floors']}
    regions = {x['id']: x for x in graph['regions']}
    objects = {x['id']: x for x in graph['objects']}
    connections = {x['id']: x for x in graph['connections']}

    def walk(value, location='graph'):
        if isinstance(value, dict):
            if 'score' in value:
                check(isinstance(value['score'], (float, int)) and 0 <= value['score'] <= 1,
                      location + ': invalid score')
                check(bool(value.get('source')), location + ': score without source')
            if 'review' in value:
                check(value['review'] == {'status': 'unreviewed'}, location + ': non-automatic review')
            for k, v in value.items():
                walk(v, location + '.' + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, location + '[' + str(i) + ']')
        elif isinstance(value, float):
            check(np.isfinite(value), location + ': non-finite number')

    walk(graph)
    for record in records + graph['issues']:
        check(record.get('review') == {'status': 'unreviewed'}, 'Missing review: ' + record.get('id', 'issue'))
        check('score' in record and bool(record.get('source')), 'Missing score/source: ' + record.get('id', 'issue'))
    for floor in floors.values():
        check(set(floor['region_ids']) == {r['id'] for r in regions.values() if r['floor'] == floor['id']},
              floor['id'] + ': incorrect region_ids')
        exists(floor['map'])
    for region in regions.values():
        check(region['floor'] in floors, region['id'] + ': unknown floor')
        check(set(region['object_ids']) == {o['id'] for o in objects.values() if o['region'] == region['id']},
              region['id'] + ': incorrect object_ids')
        check(region['type_pred']['label'] in REGION_TYPES, region['id'] + ': unknown region type')
        if region['type_pred']['source'] != 'vlm:failed':
            check(len(region['type_pred']['top3']) == 3, region['id'] + ': missing top3')
        exists(region['panorama'])
    for connection in connections.values():
        check(len(connection['regions']) == 2 and all(r in regions for r in connection['regions']),
              connection['id'] + ': invalid region endpoints')
        check(connection['type_pred']['label'] in ['door', 'opening', 'stairs'], 'Unknown connection type')
    for text in graph['texts']:
        check(text['ref'] in known, text['id'] + ': unknown reference')
        check(text['visibility_default'] == ('public' if text['level'] == 'scene' else 'private'),
              text['id'] + ': incorrect default visibility')
        check(not any(t in text['text'] for t in ['左侧', '右侧', '左边', '右边', '偏左', '偏右', '左上', '右上', '左下', '右下']),
              text['id'] + ': view-dependent left/right text')
        for item in text['grounded_on']:
            if item:
                exists(item)
    for issue in graph['issues']:
        refs = issue['ref'] if isinstance(issue['ref'], list) else [issue['ref']]
        check(all(ref in known for ref in refs), 'Issue with unknown reference: ' + str(refs))
    analysis = exists(graph['geometry_source']['analysis_path'])
    check(sha(analysis) == graph['geometry_source']['analysis_sha256'], 'Analysis hash mismatch')
    analysis_data = read(analysis)
    check(np.allclose(graph['geometry_source']['world_to_canonical'], analysis_data['world_to_canonical']),
          'Coordinate transform differs from geometry analysis')
    frame_doc = read(out / 'evidence/keyframes.json')
    frames = {f['id']: f for f in frame_doc['keyframes']}
    check(len(frames) == len(frame_doc['keyframes']), 'Duplicate keyframe IDs')
    mask_counts = {}
    metadata = {}
    for key, frame in frames.items():
        image_path = exists(frame['image_path'])
        depth_path = exists(frame['depth_path'])
        image = cv2.imread(str(image_path))
        depth = (cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
                 if frame['depth_encoding'] == 'uint16_millimeter' else np.load(depth_path))
        check(image is not None and image.shape[:2] == depth.shape, key + ': RGB/depth dimensions differ')
        check(np.isfinite(depth).all(), key + ': invalid depth')
        pose = np.array(frame['camera_to_canonical'])
        intrinsic = np.array(frame['K'])
        check(pose.shape == (4, 4) and np.allclose(pose[3], [0, 0, 0, 1]), key + ': invalid camera pose')
        check(np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
              and np.linalg.det(pose[:3, :3]) > .999, key + ': non-rigid/mirrored pose')
        check(intrinsic.shape == (3, 3) and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0, key + ': invalid K')
        if 'source_npz' in frame:
            source = np.load(exists(frame['source_npz']))
            expected_k = source['K'].copy()
            expected_k[1, 2] -= frame['crop_original_xyxy'][1]
        else:
            source_video = exists(frame['source_video'])
            source = np.load(source_video.parent / 'cameras.npz')
            expected_k = source['K']
        expected_pose = source['camera_to_canonical'] if 'source_npz' in frame else source['camera_to_canonical'][frame['frame_index']]
        check(np.allclose(expected_k, intrinsic) and np.allclose(expected_pose, pose), key + ': source camera mismatch')
        mask_path = out / 'evidence/masks' / (key + '.npz')
        meta = read(mask_path.with_suffix('.json'))
        masks = np.load(mask_path)['masks']
        check(masks.ndim == 3 and masks.shape[1:] == depth.shape and masks.dtype == np.bool_, key + ': invalid mask tensor')
        check(meta.get('image_sha256') == sha(image_path), key + ': mask image hash missing/mismatch')
        check(len(meta['detections']) == len(masks), key + ': detection/mask count differs')
        mask_counts[key] = len(masks)
        metadata[key] = meta
        del masks
    pointsets = np.load(out / 'evidence/object_points.npz')
    check(set(pointsets.files) == set(objects), 'Object point keys differ from object IDs')
    for obj in objects.values():
        oid = obj['id']
        check(obj['category_pred']['label'] in VOCAB, oid + ': unknown object category')
        check(obj['region'] in regions and obj['floor'] == regions[obj['region']]['floor'], oid + ': invalid region/floor')
        views = obj['view_evidence']
        check(obj['n_views'] == len({v['keyframe_id'] for v in views}), oid + ': incorrect n_views')
        check(obj['observations_count'] == len(views), oid + ': incorrect observation count')
        for view in views:
            key = view['keyframe_id']
            check(key in frames and 0 <= view['mask_index'] < mask_counts.get(key, 0), oid + ': invalid view/mask reference')
        points = pointsets[oid]
        check(points.ndim == 2 and points.shape[1] == 3 and len(points) >= 25 and np.isfinite(points).all(), oid + ': invalid points')
        low, high = np.quantile(points, [.02, .98], axis=0)
        box = obj['bbox']
        check(np.allclose(box['center'], (low + high)/2, atol=1e-5)
              and np.allclose(box['size'], np.maximum(high-low, .01), atol=1e-5), oid + ': point/bbox mismatch')
        check(all(s > 0 for s in box['size']) and box['yaw_deg'] == 0, oid + ': invalid observed AABB')
        check(obj['geometry_pred']['extent_is_complete'] is False, oid + ': unverified amodal extent')
        for relation in obj['relations_auto']:
            check(relation['type'] in ['near', 'on', 'above', 'below', 'near_connection'], oid + ': unsupported relation')
            check(relation['target'] in (connections if relation['type'] == 'near_connection' else objects), oid + ': missing relation target')
        for value in [obj['best_view']] + [obj['best_view_metadata'][k] for k in ['mask_path', 'context_path', 'full_view_path']]:
            exists(value)
        exists(obj['attributes_pred']['evidence'])
    vlm_counts = Counter()
    for path in (out / 'evidence/vlm').glob('*.json'):
        response = read(path)
        vlm_counts[response['status']] += 1
        if response['status'] == 'ok':
            check(bool(response.get('actual_model')) and bool(response.get('response_id')) and isinstance(response.get('parsed'), dict), str(path) + ': missing real API provenance')
            for image in response['images']:
                check(sha(Path(image['path'])) == image['sha256'], str(path) + ': VLM source image changed')
        else:
            warnings.append('Failed VLM evidence: ' + path.name)
    counts = {k: len(graph[k]) for k in sections + ['issues']}
    check(info['counts'] == counts, 'run_info counts mismatch')
    failures = [x for x in graph['issues'] if x['type'] in ['vlm_failure', 'sam3_prompt_failure', 'missing_vlm_object_result']]
    check(len(info['failure_issues']) == len(failures), 'run_info failure count mismatch')
    check(info['status'] == ('complete_with_stage_failures' if failures else 'complete'), 'Incorrect completion status')
    return {'scene_id': out.name, 'passed': not errors, 'errors': errors, 'warnings': warnings,
            'counts': counts, 'keyframes': len(frames), 'vlm_cache_records': dict(vlm_counts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    schema = read(ROOT / 'annotation_pipeline/scene_graph.schema.json')
    expected = {p.parent.name for p in (ROOT / 'outputs').glob('*/analysis.json')
                if not args.scene or p.parent.name == args.scene}
    actual = {p.parent.name for p in (ROOT / 'auto').glob('*/scene_graph.json')
              if not args.scene or p.parent.name == args.scene}
    results = []
    for scene in sorted(actual):
        try:
            result = verify_scene(ROOT / 'auto' / scene, schema)
        except Exception as exc:
            result = {'scene_id': scene, 'passed': False, 'errors': [type(exc).__name__ + ': ' + str(exc)]}
        results.append(result)
        print('PASS' if result['passed'] else 'FAIL', scene, result.get('errors', [])[:10], flush=True)
    report = {'created_at': utc(), 'passed': bool(results) and all(r['passed'] for r in results)
              and (args.allow_partial or expected == actual), 'expected_scenes': sorted(expected),
              'missing_scenes': sorted(expected-actual), 'unexpected_scenes': sorted(actual-expected),
              'scenes': results}
    dump(ROOT / 'auto' / ('validation_' + args.scene + '.json' if args.scene else 'validation.json'), report)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
