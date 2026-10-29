"""Reproducible GPT-language, measured-fact QA for all five frame difficulties.

python -m global_pipeline.frame_qa
python -m global_pipeline.frame_qa --scene household_1 --stage verify

Original descriptions, global IDs, human reviews and benchmark tasks are inputs,
not writable outputs of this additional annotation layer.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter
import argparse
import fcntl
import os
import time
import traceback
from .common import ROOT, read, dump, sha, utc, fingerprint
from .frame_qa_facts import build_facts, phrase_slots, object_names
from .frame_qa_prompts import VERSION, LEVELS, bank_prompt, validate_bank

LANGUAGE = ROOT / 'auto/frame_qa_language/v1'
INPUTS = ('frames.json', 'global_instances.json', 'frame_assignments.json', 'fine_instances.npz')


def input_hashes(out):
    return {name: sha(out / name) for name in INPUTS}


def prepare_scene(out, frames, workers=2):
    target = out / 'frame_qa'
    sources = input_hashes(out)
    instances = {o['global_id']: o for o in read(out / 'global_instances.json')['instances']}
    assignments = read(out / 'frame_assignments.json')['frames']
    with np_load(out / 'fine_instances.npz') as data:
        points, ids = data['points'], data['point_fine_ids']
    signature = fingerprint({'version': VERSION, 'sources': sources,
                             'implementation': sha(Path(__file__).with_name('frame_qa_facts.py'))})
    info = {'schema': 'kosmoworld.frame_qa_run', 'pipeline': VERSION, 'scene_id': out.parent.name,
            'status': 'preparing', 'created_at': utc(), 'source_hashes': sources,
            'source_signature': signature, 'requested_frames': len(frames), 'prepared_frames': 0}
    dump(target / 'run_info.json', info)

    def one(frame):
        path = target / 'facts' / (frame['id'] + '.json')
        # Stats for per-frame immutable evidence invalidate cached attributes
        # even when the global instance export has not been rerun.
        inputs = [out / frame['rgb'], out / 'masks' / frame['clip_id'] / (frame['id'] + '.npz'),
                  out / 'projections' / frame['clip_id'] / (frame['id'] + '.npz')]
        stamps = {str(p.relative_to(out)): [p.stat().st_mtime_ns, p.stat().st_size] for p in inputs}
        prior = read(path) if path.exists() else {}
        if prior.get('source_signature') == signature and prior.get('evidence_stamps') == stamps:
            return frame['id']
        facts = build_facts(out, frame, instances, assignments[frame['id']], points, ids, sources)
        facts.update(source_signature=signature, evidence_stamps=stamps, created_at=utc())
        dump(path, facts)
        return frame['id']

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(one, frame) for frame in frames]
        for f in as_completed(pending):
            f.result(); info['prepared_frames'] += 1
            if info['prepared_frames'] % 40 == 0 or info['prepared_frames'] == len(frames):
                info['updated_at'] = utc(); dump(target / 'run_info.json', info)
                print('FACTS', out.parent.name, info['prepared_frames'], '/', len(frames), flush=True)
    info.update(status='facts_ready', updated_at=utc())
    dump(target / 'run_info.json', info)
    return sources


def np_load(path):
    import numpy as np
    return np.load(path)


def generate_language(examples, labels):
    """Real GPT requests receive annotated objects and numerical sample facts."""
    from annotation_pipeline.vlm import VLM
    model = VLM(model=os.environ.get('FRAME_QA_GPT_MODEL', 'gpt-6-sol'))
    if not model.model.startswith('gpt-'):
        raise ValueError('GPT model required')
    LANGUAGE.mkdir(parents=True, exist_ok=True)
    outputs = {}

    def bank():
        prompt = bank_prompt(examples)
        for attempt in range(3):
            cache = LANGUAGE / 'evidence' / f'question_bank_{attempt}.json'
            result = model.call(prompt, [], cache, max_tokens=4200)
            try:
                rows = validate_bank(result)
                record = read(cache)
                if not str(record.get('actual_model', '')).startswith('gpt-'):
                    raise ValueError('Missing actual GPT model in gateway response')
                value = {'schema': 'kosmoworld.frame_qa_language_bank', 'schema_version': '1.0',
                         'pipeline': VERSION, 'levels': [rows[k] for k in sorted(rows)],
                         'prompts': [{'difficulty_level': k, 'task': v[0], 'name_zh': v[1], 'prompt': v[2]}
                                     for k, v in LEVELS.items()],
                         'source': 'gpt:' + record['actual_model'], 'evidence': str(cache.relative_to(LANGUAGE)),
                         'evidence_sha256': sha(cache), 'created_at': utc(),
                         'generation_scope': 'GPT question/answer templates, instantiated with measured facts per frame',
                         'review': {'status': 'unreviewed'}}
                dump(LANGUAGE / 'question_bank.json', value)
                return value
            except ValueError as exc:
                prompt += '\n上一版 schema 校验失败：' + str(exc) + '。请严格修正并返回完整五档 JSON。'
        raise ValueError('GPT did not produce a valid five-level language bank')

    def glossary(chunk, index):
        import json
        prompt = ('''Translate these existing indoor object/surface semantic labels into concise, stable Chinese category names.
Do not add attributes, colors, counts, positions or objects. These names will be filled into grounded questions and answers.
Return JSON only: {"names":{"exact input English label":"中文类别名"}}. Include EVERY supplied key, unchanged; no extra keys.
Existing annotation labels: ''' + json.dumps(chunk, ensure_ascii=False))
        cache = LANGUAGE / 'evidence' / f'glossary_{index:02d}.json'
        result = model.call(prompt, [], cache, max_tokens=3200)
        names = result.get('names', {})
        if set(names) != set(chunk) or any(not isinstance(v, str) or not v.strip() for v in names.values()):
            raise ValueError('GPT glossary must contain all and only supplied labels')
        record = read(cache)
        if not str(record.get('actual_model', '')).startswith('gpt-'):
            raise ValueError('Glossary response has no actual GPT model')
        return names, {'source': 'gpt:' + record['actual_model'], 'evidence': str(cache.relative_to(LANGUAGE)),
                       'sha256': sha(cache)}

    labels = sorted(set(labels) - {'unknown'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(bank): 'bank'}
        for i, start in enumerate(range(0, len(labels), 65)):
            futures[pool.submit(glossary, labels[start:start + 65], i)] = i
        names = {}; evidence = {}
        for future in as_completed(futures):
            key = futures[future]
            result = future.result()
            if key == 'bank': outputs['bank'] = result
            else:
                chunk, meta = result; names.update(chunk); evidence[str(key)] = meta
            print('GPT LANGUAGE READY', key, flush=True)
    outputs['glossary'] = {'names': names, 'evidence': evidence, 'source': 'gpt:semantic_label_translation',
                           'created_at': utc(), 'review': {'status': 'unreviewed'}}
    dump(LANGUAGE / 'glossary.json', outputs['glossary'])
    return outputs


def render_questions(facts, bank, glossary, facts_path, facts_hash, language_hash):
    slots = phrase_slots(facts, glossary)
    names = object_names(facts, glossary)
    objects = {o['global_id']: o for o in facts['objects']}
    levels = validate_bank(bank)
    questions = []
    for level in range(1, 6):
        ids = (list(objects) if level == 1 else facts['camera_comparison']['global_ids'] if level == 5
               else facts['focus_global_ids'])
        answerable = bool(ids)
        if level == 3: answerable = bool(facts['relative_relations'])
        if level == 5: answerable = facts['camera_comparison']['status'] == 'answerable'
        spatial = ({'relations': facts['relative_relations']} if level == 3 else
                   {'bbox_centers_and_extents': [{'global_id': gid, 'bbox': objects[gid]['bbox']} for gid in ids]} if level == 4 else
                   {'camera_comparison': facts['camera_comparison'], 'camera_to_canonical': facts['camera_to_canonical']} if level == 5 else {})
        score = min([objects[gid]['score'] for gid in ids] or [0.]) if answerable else 0.
        questions.append({
            'id': f'l1qa_{facts["frame_id"]}_d{level}', 'annotation_level': 1,
            'temporal_scope': 'single_frame', 'difficulty_level': level, 'task': LEVELS[level][0],
            'question': levels[level]['question_template'].format(**slots),
            'answer': levels[level]['answer_template'].format(**slots),
            'answer_status': 'answerable' if answerable else 'insufficient_or_ambiguous_evidence',
            'answer_annotations': {
                'global_ids': ids, 'object_ids': [objects[gid]['object_id'] for gid in ids],
                'objects': [{'global_id': gid, 'name_zh': names[gid], 'semantic_label': objects[gid]['semantic_label'],
                             'category_type': objects[gid]['category_type'], 'score': objects[gid]['score'],
                             'attributes': objects[gid]['attributes'], 'image': objects[gid]['image'],
                             **({'bbox': objects[gid]['bbox'], 'floor_id': facts['floor_id']} if level >= 3 else {}),
                             **({'camera': objects[gid]['camera']} if level == 5 else {})}
                            for gid in ids],
                'spatial': spatial, 'facts': facts_path, 'facts_sha256': facts_hash,
                'frame': 'canonical', 'units': 'inferred_meter',
                'id_namespace': facts['scene_id'], 'global_id_source': 'fine_instances.npz:point_fine_ids'},
            'input_requirements': ['RGB', 'canonical_reference+3d_annotations'] if level == 4 else
                                  ['RGB', 'calibrated_camera+depth_or_3d_annotations'] if level == 5 else ['RGB'],
            'score': score, 'source': 'gpt_language_bank+measured_annotation_facts',
            'language_source': bank['source'], 'language_bank_sha256': language_hash,
            'generation_mode': 'gpt_template_instantiation_not_independent_image_inference',
            'grounded_on': [facts['rgb']], 'review': {'status': 'unreviewed'},
            'visibility_default': 'private'})
    return {'schema': 'kosmoworld.frame_qa', 'schema_version': '1.0', 'scene_id': facts['scene_id'],
            'frame_id': facts['frame_id'], 'clip_id': facts['clip_id'], 'timestamp': facts['timestamp'],
            'annotation_level': 1, 'pipeline': VERSION, 'questions': questions,
            'source_hashes': facts['source_hashes'], 'facts_sha256': facts_hash,
            'language_bank_sha256': language_hash,
            'generation_fingerprint': fingerprint([VERSION, facts_hash, language_hash]),
            'created_at': utc(), 'annotation_status': facts['annotation_status']}


def render_scene(out, frames, language):
    target = out / 'frame_qa'
    bank, glossary = language['bank'], language['glossary']['names']
    dump(target / 'prompts.json', bank)
    dump(target / 'glossary.json', language['glossary'])
    language_hash = fingerprint([bank, language['glossary']])
    # Each scene is independently downloadable, including the real GPT evidence.
    for relative in [bank['evidence']] + [v['evidence'] for v in language['glossary']['evidence'].values()]:
        dump(target / relative, read(LANGUAGE / relative))
    rows = []; statuses = Counter(); levels = Counter(); issues = Counter(); direct_gpt = 0
    for frame in frames:
        path = target / 'facts' / (frame['id'] + '.json'); facts = read(path)
        facts_hash = sha(path)
        value = render_questions(facts, bank, glossary, 'facts/' + path.name, facts_hash, language_hash)
        result = target / 'frames' / path.name
        previous = read(result) if result.exists() else {}
        if previous.get('generation_fingerprint') != value['generation_fingerprint']:
            dump(result, value)
        elif previous.get('generation_mode') == 'gpt_answer_from_supplied_frame_ground_truth':
            direct_gpt += 1
        for q in value['questions']:
            statuses[q['answer_status']] += 1; levels[str(q['difficulty_level'])] += 1
        if facts['unclassified_visible_global_ids']: issues['unclassified_visible_instances'] += 1
        if not facts['objects']: issues['no_recognized_objects'] += 1
        if facts['camera_comparison']['status'] != 'answerable': issues['camera_comparison_unanswerable'] += 1
        rows.append({'frame_id': frame['id'], 'clip_id': frame['clip_id'], 'timestamp': frame['timestamp'],
                     'path': 'frames/' + path.name, 'facts': 'facts/' + path.name,
                     'n_questions': 5, 'recognized_global_ids': [o['global_id'] for o in facts['objects']]})
    total = len(read(out / 'frames.json')['frames'])
    manifest = {'schema': 'kosmoworld.frame_qa_index', 'schema_version': '1.0', 'pipeline': VERSION,
                'scene_id': out.parent.name, 'frames': rows, 'frames_total': total,
                'annotated_frames': len(rows), 'questions_total': len(rows) * 5,
                'coverage': 'full' if len(rows) == total else 'subset', 'status': 'complete',
                'source_hashes': read(target / 'run_info.json')['source_hashes'],
                'levels': dict(levels), 'answer_status_counts': dict(statuses), 'issue_frame_counts': dict(issues),
                'language_bank_sha256': language_hash, 'generation_mode': 'gpt_template_instantiation',
                'prompts': 'prompts.json', 'glossary': 'glossary.json', 'updated_at': utc(),
                'review': {'status': 'unreviewed'}}
    if direct_gpt:
        manifest['gpt_answers'] = {'status':'complete' if direct_gpt == len(rows) else 'running',
                                   'completed_frames':direct_gpt,'total_frames':len(rows),'updated_at':utc()}
        manifest['generation_mode'] = ('gpt_answer_from_supplied_frame_ground_truth' if direct_gpt == len(rows)
                                       else 'mixed_gpt_answers_and_references')
    dump(target / 'manifest.json', manifest)
    info = read(target / 'run_info.json')
    info.update(status='complete', annotated_frames=len(rows), questions_total=len(rows) * 5,
                updated_at=utc(), language_source=bank['source'], coverage=manifest['coverage'])
    dump(target / 'run_info.json', info)
    return manifest


def verify_scene(out):
    target = out / 'frame_qa'; manifest = read(target / 'manifest.json')
    sources = input_hashes(out)
    frames = {f['id']: f for f in read(out / 'frames.json')['frames']}
    errors = []; count = 0; direct_gpt_frames = 0; evidence_cache = {}
    if manifest['source_hashes'] != sources: errors.append('Source files changed after annotation')
    bank = read(target / 'prompts.json'); validate_bank(bank)
    glossary_doc = read(target / 'glossary.json'); glossary = glossary_doc['names']
    language_hash = fingerprint([bank, glossary_doc])
    for row in manifest['frames']:
        fid = row['frame_id']; facts_path = target / row['facts']; facts = read(facts_path)
        result = read(target / row['path'])
        if fid not in frames or frames[fid]['clip_id'] != result['clip_id']:
            errors.append(fid + ': invalid frame/clip')
        facts_hash = sha(facts_path)
        expected = render_questions(facts, bank, glossary, row['facts'], facts_hash, language_hash)
        if result.get('generation_mode') == 'gpt_answer_from_supplied_frame_ground_truth':
            from .frame_qa_grounded import truth_payload, validate_answers
            direct_gpt_frames += 1
            stable_path = target/'stable_questions.json'; stable = read(stable_path)
            truth = truth_payload(facts, expected, glossary)
            evidence_path = target/result['answer_evidence']
            if not evidence_path.resolve().is_relative_to(target.resolve()):
                errors.append(fid + ': invalid GPT evidence path')
                continue
            if str(evidence_path) not in evidence_cache:
                evidence_cache[str(evidence_path)] = (read(evidence_path), sha(evidence_path))
            evidence, evidence_hash = evidence_cache[str(evidence_path)]
            if evidence.get('status') != 'ok' or not str(evidence.get('actual_model','')).startswith('gpt-'):
                errors.append(fid + ': missing successful GPT response')
            parsed = {'frames':[row for row in evidence.get('parsed',{}).get('frames',[]) if row.get('frame_id') == fid]}
            try: answers = validate_answers(parsed, [truth])[fid]
            except ValueError as exc:
                errors.append(fid + ': invalid grounded GPT answer: ' + str(exc)); answers = {}
            for q, measured in zip(result['questions'],expected['questions']):
                level = q['difficulty_level']; answer = answers.get(level,{})
                if (q.get('answer_reference') != measured['answer'] or q.get('answer_annotations') != measured['answer_annotations']
                    or q.get('answer') != answer.get('answer') or q.get('evidence_sha256') != evidence_hash
                    or q.get('question') != stable['questions'].get(str(level),stable['questions'].get(level))
                    or q.get('answer_grounding',{}).get('request_truth_sha256') != fingerprint(truth)):
                    errors.append(fid + ': grounded GPT answer does not match supplied facts or evidence')
        elif result['questions'] != expected['questions']:
            errors.append(fid + ': questions differ from reproducible measured-fact rendering')
        if facts['source_hashes'] != sources or result['source_hashes'] != sources:
            errors.append(fid + ': stale annotation sources')
        integrity = facts.get('source_integrity', {})
        if integrity.get('actual_rgb_sha256') != facts['evidence']['rgb_sha256']:
            errors.append(fid + ': RGB evidence digest mismatch')
        for relative, stamp in facts['evidence_stamps'].items():
            p = out / relative
            if not p.exists() or stamp != [p.stat().st_mtime_ns, p.stat().st_size]:
                errors.append(fid + ': frame evidence changed: ' + relative)
        if result['facts_sha256'] != facts_hash: errors.append(fid + ': facts digest mismatch')
        if len(result['questions']) != 5: errors.append(fid + ': missing difficulty')
        for q in result['questions']:
            if not q['question'].strip() or not q['answer'].strip() or q['review']['status'] != 'unreviewed':
                errors.append(fid + ': empty QA or invalid review')
            if not set(q['answer_annotations']['global_ids']) <= {o['global_id'] for o in facts['objects']}:
                errors.append(fid + ': invented object ID')
            count += 1
    warnings = [row['frame_id'] + ': stale RGB hash in frames.json'
                for row in manifest['frames']
                if not read(target / row['facts']).get('source_integrity', {}).get('manifest_hash_matches', True)]
    record = {'passed': not errors, 'scene_id': out.parent.name, 'frames': len(manifest['frames']),
              'questions': count, 'coverage': manifest['coverage'], 'errors': errors,
              'direct_gpt_answer_frames': direct_gpt_frames,
              'template_answer_frames': len(manifest['frames'])-direct_gpt_frames,
              'warnings': warnings,
              'checks': ['source hashes', 'current RGB/mask/projection evidence', 'five difficulty levels',
                         'measured-fact references and actual GPT response provenance', 'existing scene-local global IDs',
                         'explicit coordinate/metric scope', 'original annotation files retained'],
              'checked_at': utc(), 'not_checked': ['human semantic correctness', 'metric scale calibration']}
    dump(target / 'validation.json', record)
    if errors: raise ValueError(str(errors[:5]))
    print('VERIFIED FRAME QA', out.parent.name, len(manifest['frames']), count, flush=True)
    return record


def report():
    rows = []
    for p in sorted((ROOT / 'auto').glob('*/global_instances/frame_qa/manifest.json')):
        d = read(p); v = read(p.with_name('validation.json')) if p.with_name('validation.json').exists() else {}
        rows.append({'scene_id': d['scene_id'], 'frames': d['annotated_frames'], 'questions': d['questions_total'],
                     'coverage': d['coverage'], 'answer_status': d['answer_status_counts'], 'passed': v.get('passed', False),
                     'direct_gpt_answer_frames': d.get('gpt_answers',{}).get('completed_frames',0),
                     'verified_gpt_answer_frames':v.get('direct_gpt_answer_frames',0) if v.get('passed') else 0,
                     'gpt_status': d.get('gpt_answers',{}).get('status','not_generated')})
    complete = rows and all(r['direct_gpt_answer_frames']==r['frames']==r['verified_gpt_answer_frames'] and r['passed'] for r in rows)
    status = ('complete' if complete else 'blocked' if any(r['gpt_status']=='blocked' for r in rows)
              else 'failed' if any(r['gpt_status']=='failed' for r in rows) else 'running')
    dump(ROOT / 'auto/FRAME_QA_SUMMARY.json', {'schema': 'kosmoworld.frame_qa_summary', 'pipeline': VERSION,
         'scenes': rows, 'frames': sum(r['frames'] for r in rows), 'questions': sum(r['questions'] for r in rows),
         'direct_gpt_answer_frames':sum(r['direct_gpt_answer_frames'] for r in rows),
         'direct_gpt_answer_questions':5*sum(r['direct_gpt_answer_frames'] for r in rows),
         'pending_gpt_answer_frames':sum(r['frames']-r['direct_gpt_answer_frames'] for r in rows),
         'status':status,
         'updated_at': utc(), 'review': {'status': 'unreviewed'}})
    text = ['# Household 帧内五档问答', '', '浏览：`/global`，选择场景、轨迹和当前帧。', '',
            '| 场景 | 帧 | 问答 | GPT 答案帧 | 已验收 GPT 帧 | 生成状态 |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        text.append(f'| {row["scene_id"]} | {row["frames"]} | {row["questions"]} | {row["direct_gpt_answer_frames"]} | {row["verified_gpt_answer_frames"]} | {row["gpt_status"]} |')
    text += ['', '五档均属于单帧 Level 1：识别、属性、相对空间、全局空间、相机远近。', '',
             '先由代码整理每帧已标注对象、颜色统计、几何坐标和距离，生成可核验事实参考答案；'
             '再将每帧物体、ID、空间标注及五档固定问题直接发送 GPT，生成逐帧自然语言答案。'
             '尚未经过后一步的条目明确标记为模板参考答案，不冒充 GPT 直接标注。'
             '真实 GPT 请求及响应位于每场景 frame_qa/evidence/gpt_answers/。', '',
             '绝对坐标题需要 canonical 参考系与三维数据，远近数值题需要标定与深度/三维标注。'
             '现有语义仍为自动未审核结果，颜色是渲染图像 mask 的统计量；观测框不是完整实物框。'
             '证据不足的题保留拒答和原因，不强填答案。未更改原 descriptions.json、scene_graph.json、人工审核或 benchmark。', '']
    (ROOT / 'auto/FRAME_QA_REPORT.md').write_text('\n'.join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', action='append'); parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--stage', choices=['all', 'prepare', 'language', 'render', 'verify'], default='all')
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1): parser.error('Positive workers/limit required')
    scenes = sorted((ROOT / 'auto').glob('*/global_instances/frames.json'))
    if args.scene:
        scenes = [p for p in scenes if p.parent.parent.name in args.scene]
        if len(scenes) != len(set(args.scene)): parser.error('Unknown or unavailable scene')
    lock_path = ROOT / 'services/frame_qa.lock'; lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: parser.error('Frame QA annotation is already running')
        selected = [(p.parent, read(p)['frames'][:args.limit]) for p in scenes]
        if args.stage in ('all', 'prepare'):
            for out, frames in selected:
                try: prepare_scene(out, frames, args.workers)
                except Exception as exc:
                    dump(out / 'frame_qa/run_info.json', {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc(), 'updated_at': utc()})
                    raise
        if args.stage == 'prepare': return
        if args.stage in ('all', 'language'):
            examples = []; labels = set()
            for out, frames in selected:
                labels.update(o['semantic_label'] for o in read(out / 'global_instances.json')['instances'])
                sample = max((read(out / 'frame_qa/facts' / (f['id'] + '.json')) for f in frames[:12]),
                             key=lambda d: len(d['focus_global_ids']), default=None)
                if sample:
                    # Show IDs and real spatial fields to GPT, without requiring
                    # it to discover or estimate the already annotated objects.
                    focus = set(sample['focus_global_ids']) | set(sample['camera_comparison']['global_ids'])
                    examples.append({'scene_id': sample['scene_id'], 'frame_id': sample['frame_id'],
                                     'objects': [{'global_id': o['global_id'], 'semantic_label': o['semantic_label'],
                                                  'color': o['attributes']['color']['name_zh'], 'bbox': o['bbox'], 'camera': o['camera']}
                                                 for o in sample['objects'] if o['global_id'] in focus],
                                     'relations': sample['relative_relations'], 'camera_comparison': sample['camera_comparison']})
            existing_bank = LANGUAGE / 'question_bank.json'
            existing_glossary = LANGUAGE / 'glossary.json'
            if existing_bank.is_file() and existing_glossary.is_file():
                language = {'bank': read(existing_bank), 'glossary': read(existing_glossary)}
                # Validate the cached GPT result before reusing it. This keeps
                # a rerun deterministic and avoids charging the gateway again.
                from .frame_qa_prompts import validate_bank
                validate_bank(language['bank']); language['glossary']['names']
                print('REUSE GPT FRAME QA LANGUAGE BANK', flush=True)
            else:
                language = generate_language(examples, labels)
        else:
            language = {'bank': read(LANGUAGE / 'question_bank.json'), 'glossary': read(LANGUAGE / 'glossary.json')}
        if args.stage == 'language': return
        for out, frames in selected:
            if args.stage in ('all', 'render'): render_scene(out, frames, language)
            verify_scene(out)
        report()


if __name__ == '__main__': main()
