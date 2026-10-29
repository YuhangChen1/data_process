"""Apply current deterministic quality rules without repeating SAM3 or VLM inference."""
import argparse
import copy
from pathlib import Path

from .common import ROOT, read, dump, sha, utc
from .finalize import build_texts
from .quality import audit_confidence


def process(out):
    graph = read(out / 'scene_graph.json')
    previous = copy.deepcopy(graph)
    registry_path = ROOT / 'auto/_registry' / (out.name + '.json')
    registry = read(registry_path)
    matching = build_texts(graph, previous, registry)
    counts_by_issue = audit_confidence(graph)
    if graph != previous:
        history = out / 'history' / (utc().replace(':', '').replace('-', '') + '_quality')
        dump(history / 'scene_graph.json', previous)
        graph.setdefault('generation', {})['quality_rules_updated_at'] = utc()
        dump(out / 'scene_graph.json', graph)
    dump(registry_path, registry)
    # Initial batch did not record image hashes. Its original evidence is retained;
    # explicitly record when the hash was added instead of claiming inference-time capture.
    model_manifest = read(ROOT / 'services/model_manifest.json')
    backfilled = 0
    for path in (out / 'evidence/masks').glob('*.json'):
        meta = read(path)
        if 'image_sha256' not in meta:
            meta['image_sha256'] = sha(Path(meta['image_path']))
            meta['hash_provenance'] = 'backfilled_from_retained_initial_run_evidence'
            meta['hash_recorded_at'] = utc()
            backfilled += 1
        meta.setdefault('checkpoint_sha256', model_manifest['checkpoint_sha256'])
        meta.setdefault('sam3_code_sha256', model_manifest['sam3_code_sha256'])
        dump(path, meta)
    info = read(out / 'run_info.json')
    info['counts'] = {k: len(graph[k]) for k in ['floors', 'regions', 'connections', 'objects', 'texts', 'issues']}
    info['issue_counts'] = counts_by_issue
    info['failure_issues'] = [x for x in graph['issues'] if x['type'] in
                            ['vlm_failure', 'sam3_prompt_failure', 'missing_vlm_object_result']]
    info['status'] = 'complete_with_stage_failures' if info['failure_issues'] else 'complete'
    actual_models={read(path).get('actual_model') for path in (out/'evidence/vlm').glob('*.json') if read(path).get('status')=='ok'}
    info.setdefault('models',{}).setdefault('vlm',{})['actual_models_used']=sorted(model for model in actual_models if model)
    info['stages']['attributes']='partial' if any(o['attributes_pred']['source']=='vlm:failed' for o in graph['objects']) else 'ok'
    info['stages']['scene_text']='partial' if graph['scene']['type_pred']['source']=='vlm:failed' else 'ok'
    info['stages']['region_vlm']='partial' if any(r['type_pred']['source']=='vlm:failed' for r in graph['regions']) else 'ok'
    info['quality_postprocessing'] = {
        'updated_at': utc(), 'model_inference_repeated': False,
        'text_id_matching': matching,
        'code_sha256': {p.name: sha(p) for p in [ROOT / 'annotation_pipeline' / name for name in
                                               ['finalize.py', 'quality.py', 'postprocess.py']]},
        'initial_image_hashes_added_this_pass': backfilled}
    dump(out / 'run_info.json', info)
    print('Quality audited', out.name, info['counts'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene')
    args = parser.parse_args()
    for path in sorted((ROOT / 'auto').glob('*/scene_graph.json')):
        if not args.scene or path.parent.name == args.scene:
            info = read(path.parent / 'run_info.json')
            if info['status'] in ['complete', 'complete_with_stage_failures']:
                process(path.parent)


if __name__ == '__main__':
    main()
