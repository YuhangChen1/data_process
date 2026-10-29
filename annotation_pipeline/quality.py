"""Shared confidence audit for generated graphs, including older first-run outputs."""
from collections import Counter

from .common import base, aabb_iou


def audit_confidence(graph, threshold=.6):
    # This audit is deterministic and can be repeated without duplicating issues.
    generated = {'low_confidence_predictions', 'low_confidence_relations',
                 'possible_duplicate_instance', 'missing_vlm_object_result'}
    graph['issues'] = [x for x in graph['issues'] if x['type'] not in generated]
    for obj in graph['objects']:
        obj['bbox'].update(score=obj['geometry_pred']['score'],
                           source=obj['geometry_pred']['source'])
    records = [graph['scene']] + sum(
        [graph[key] for key in ['floors', 'regions', 'connections', 'objects', 'texts']], [])
    for record in records:
        low = []
        for field, prediction in record.items():
            if isinstance(prediction, dict) and 'score' in prediction and 'source' in prediction:
                if prediction['score'] < threshold:
                    low.append({'field': field, 'score': prediction['score'],
                                'source': prediction['source']})
        # The record itself is also a scored prediction.
        if record.get('score', 1) < threshold:
            low.append({'field': 'score', 'score': record['score'], 'source': record['source']})
        if low:
            minimum = min(x['score'] for x in low)
            graph['issues'].append({
                'type': 'low_confidence_predictions', 'ref': record['id'],
                'detail': '以下自动预测分数低于 %.2f：%s。' %
                          (threshold, '、'.join('%s=%.2f' % (x['field'], x['score']) for x in low)),
                'fields': low, 'threshold': threshold, 'priority': .8,
                **base('rule:confidence_audit', 1-minimum)})
    for obj in graph['objects']:
        low = [r for r in obj['relations_auto'] if r['score'] < threshold]
        if low:
            graph['issues'].append({
                'type': 'low_confidence_relations', 'ref': obj['id'],
                'detail': '%d 条几何关系分数低于 %.2f，关系仅为候选。' % (len(low), threshold),
                'relations': [{'type': r['type'], 'target': r['target'], 'score': r['score'],
                               'source': r['source']} for r in low],
                'priority': .5, **base('rule:confidence_audit', 1-min(r['score'] for r in low))})
        if obj.get('attributes_pred', {}).get('source') == 'vlm:failed':
            obj['category_vlm_check']['source'] = 'vlm:failed'
            graph['issues'].append({
                'type': 'missing_vlm_object_result', 'ref': obj['id'],
                'detail': 'VLM 未返回此物体的有效属性记录；unknown 并非已确认属性。',
                'priority': 1., **base('pipeline:vlm_result_audit', 1.)})
    objects = graph['objects']
    for i, a in enumerate(objects):
        for b in objects[i+1:]:
            if a['floor'] != b['floor'] or a['category_pred']['label'] != b['category_pred']['label']:
                continue
            overlap = aabb_iou(a['bbox'], b['bbox'])
            if overlap > .35:
                graph['issues'].append({
                    'type': 'possible_duplicate_instance', 'ref': [a['id'], b['id']],
                    'detail': '同类观测框 IoU=%.2f，可能是同一物体的分裂候选，需核对多视图证据。' % overlap,
                    'aabb_iou': overlap, 'priority': .85,
                    **base('rule:same_category_aabb_overlap', overlap)})
    graph['issues'].sort(key=lambda x: -x.get('priority', 0))
    return dict(Counter(x['type'] for x in graph['issues']))
