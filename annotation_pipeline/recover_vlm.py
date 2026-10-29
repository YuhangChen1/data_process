"""Retry only missing semantic stages; preserve all geometry, SAM3 evidence and IDs."""
import argparse
import copy

from .common import ROOT, read, dump, utc
from .finalize import build_texts
from .quality import audit_confidence
from .semantics import annotate_attributes, annotate_scene
from .vlm import VLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene')
    args = parser.parse_args()
    vlm = VLM()
    for path in sorted((ROOT / 'auto').glob('*/scene_graph.json')):
        out = path.parent
        if args.scene and args.scene != out.name:
            continue
        info = read(out / 'run_info.json')
        if info['status'] not in ['complete', 'complete_with_stage_failures']:
            continue
        graph = read(path)
        missing = [o for o in graph['objects'] if o.get('attributes_pred', {}).get('source') == 'vlm:failed']
        scene_missing = graph['scene']['type_pred']['source'] == 'vlm:failed'
        if not missing and not scene_missing:
            continue
        previous = copy.deepcopy(graph)
        history = out / 'history' / (utc().replace(':', '').replace('-', '') + '_vlm_recovery')
        dump(history / 'scene_graph.json', previous)
        dump(history / 'run_info.json', info)
        print('Recovering VLM', out.name, 'missing object attributes', len(missing), flush=True)
        annotate_attributes(graph, out, vlm, only_missing=True)
        if scene_missing:
            annotate_scene(graph, out, vlm)
        registry_path = ROOT / 'auto/_registry' / (out.name + '.json')
        registry = read(registry_path)
        build_texts(graph, previous, registry)
        counts = audit_confidence(graph)
        failures = [x for x in graph['issues'] if x['type'] in
                    ['vlm_failure', 'sam3_prompt_failure', 'missing_vlm_object_result']]
        graph['generation']['vlm_recovered_at'] = utc()
        dump(path, graph)
        dump(registry_path, registry)
        info['counts'] = {k: len(graph[k]) for k in ['floors', 'regions', 'connections', 'objects', 'texts', 'issues']}
        info['issue_counts'] = counts
        info['failure_issues'] = failures
        info['status'] = 'complete_with_stage_failures' if failures else 'complete'
        info['stages']['attributes'] = 'partial' if any(o['attributes_pred']['source'] == 'vlm:failed' for o in graph['objects']) else 'ok'
        info['stages']['scene_text'] = 'partial' if graph['scene']['type_pred']['source'] == 'vlm:failed' else 'ok'
        info.setdefault('recovery_runs', []).append({'at': utc(), 'objects_retried': len(missing),
                                                    'remaining_failure_issues': len(failures),
                                                    'requested_model':vlm.model,
                                                    'geometry_and_ids_unchanged': True})
        dump(out / 'run_info.json', info)
        print('Recovered', out.name, 'remaining failures', len(failures), flush=True)


if __name__ == '__main__':
    main()
