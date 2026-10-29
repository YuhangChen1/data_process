"""GPT answers fixed questions using each frame's supplied annotation truth.

Run after ``python -m global_pipeline.frame_qa``. Each requested frame gets five
actual GPT answers. The earlier deterministic answers are retained as references;
they are never represented as direct GPT responses. Successful batches resume.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from pathlib import Path
import argparse
import copy
import fcntl
import json
import os
import re
import time
import traceback
from .common import ROOT, read, dump, sha, utc, fingerprint
from .frame_qa import LANGUAGE, report, input_hashes
from .frame_qa_prompts import LEVELS
from .frame_qa_facts import phrase_slots, object_names, COORDINATE_CONVENTION
from annotation_pipeline.vlm import VLM, GatewayRequestError

VERSION = 'frame_qa_grounded/1.0'
PROMPT = '''你是住宅场景问答数据的中文标注员。对每个 frame 回答给定的五个固定问题。
本次不让你猜测或重新发现物体。每帧已直接提供 objects、focus_global_ids、相机位置、
相对关系、三维位置和 reference_answers，全部来自已有画面标注及几何测量。
只允许使用该帧提供的事实，以自然、稳定的问题回答语气组织成完整答案；不能把不同帧混用。
reference_answers 是计算得到的事实底稿，请重新组织语句，而非只说“是/否”或泛泛描述。
回答要求：
1：完整归纳已知的可见物体类别与连续表面；不要将分割片段数解释为真实物体数量。
2：识别 focus_global_ids 的主要物体，再逐一说明已提供的可见颜色和轮廓/观测范围。
3：先概述这些主要物体和颜色轮廓，再以明确对象为参照解释相对位置。当前图像的左右上下
与 canonical 三维坐标必须区分；不要将“下面/相邻”说成“支撑在上面/正在接触”。
4：先识别主要物体、说明可见特征，再给出每个物体的 canonical 观测框中心坐标，
可概括观测范围；说明 Z 向上、单位是推断米制、不是完整实物尺寸。不要编造东南西北。
5：先识别 camera_comparison.global_ids 的对象并说明可见特征，再依据相机到可见表面
三维欧氏距离中位数判断近远，给出两个距离。差异不足/证据不足必须明确拒绝排序。
每档应有实际信息、对象名称和连贯解释，不要只给关键词；建议第1档40-100字，
第2档80-160字，第3档100-200字，第4档120-220字，第5档90-180字。
数字仅可从给定参考答案复制，最多保留两位小数，不增加新的估计、实物数量、材质或功能。
颜色是渲染图像中 mask 的统计，不等于已确认材质；unknown 不得自行补齐。
object ID 只写入 global_ids，不必在自然语言答案中反复写编号。
answer_status 已由代码确定，不要改变。证据不够的档位也返回完整解释答案，不跳过该题。
不讨论模型、提示词、输入 JSON 或内部处理过程。这里是静态场景，不虚构实物运动。
每条返回它实际依据的全部 global_ids，必须和该帧该题 allowed_global_ids 完全一致。
返回 JSON，不要其他文字。用 compact 数组避免重复字段：
{"frames":[{"frame_id":"g000000","answers":[[1,"中文完整答案",[1,2]],[2,"中文完整答案",[1,2]],[3,"中文完整答案",[1,2]],[4,"中文完整答案",[1,2]],[5,"中文完整答案",[1,2]]]}]}
每个答案行是 [difficulty_level, answer_text, global_ids]；每帧必须恰好五档，不要省略。
固定问题及逐帧真值如下：
'''


def stable_questions(model):
    path = LANGUAGE / 'stable_questions.json'
    prompt = '''为室内场景（住宅、办公、教育、商业空间）图像理解数据写五个固定中文问题，严格按五档递进。
1 当前帧看见哪些物体和连续表面；2 先识别三件主要对象，再说明它们的颜色和可见几何轮廓；
3 先识别并描述主要物体特征，再说明物体之间的相对位置，指出参照对象和图像/三维参考系；
4 先识别主要对象并描述特征，再在已给定 canonical 世界参考系中报告绝对位置；
5 先识别两件可见独立物体并描述特征，再比较谁更接近当前拍摄相机、谁更远并解释依据。
五个问题会用于所有帧：不用具体物体名称，不出现坐标/距离答案，不含实例 ID，不含模板变量。
语气自然，提问具体完整，每题约35-80汉字。对象不足或不确定时允许回答说明证据不足。
第4档评测另有 canonical 坐标参考/三维数据，第5档另有深度/标定，不能暗示单张 RGB 能确定米制。
不要写很长的实施规范，不出现 mask、真值、标注员、JSON、算法或 GPT 等内部术语。
只返回 JSON：{"levels":[{"difficulty_level":1,"question":"中文问题"}, ...恰好五项]}。
'''
    result = model.call(prompt, [], path, max_tokens=1500)
    levels = result.get('levels', [])
    if len(levels) != 5 or {r.get('difficulty_level') for r in levels} != set(LEVELS):
        raise ValueError('GPT must return exactly five stable questions')
    if any(not isinstance(r.get('question'), str) or len(r['question']) < 15 or '{' in r['question'] for r in levels):
        raise ValueError('Invalid stable question text')
    record = read(path)
    if not str(record.get('actual_model', '')).startswith('gpt-'):
        raise ValueError('No actual GPT model in stable question evidence')
    return {r['difficulty_level']: r['question'].strip() for r in levels}, path, record


def truth_payload(facts, reference, glossary):
    names = object_names(facts, glossary)
    focus = set(facts['focus_global_ids']) | set(facts['camera_comparison']['global_ids'])
    objects = []
    for obj in facts['objects']:
        entry = {'global_id': obj['global_id'], 'label': obj['semantic_label'],
                 'name_zh': names[obj['global_id']], 'category_type': obj['category_type']}
        if obj['global_id'] in focus:
            entry.update(color=obj['attributes']['color']['name_zh'],
                         silhouette=obj['image']['extent_orientation'], bbox=obj['bbox'],
                         camera=obj['camera'])
        objects.append(entry)
    return {'frame_id': facts['frame_id'], 'objects': objects,
            'focus_global_ids': facts['focus_global_ids'], 'floor_id': facts['floor_id'],
            'relative_relations': facts['relative_relations'], 'camera_comparison': facts['camera_comparison'],
            'coordinate_convention': COORDINATE_CONVENTION,
            'unclassified_visible_global_ids': facts['unclassified_visible_global_ids'],
            'reference_answers': [{'difficulty_level': q['difficulty_level'],
                                   'text': q.get('answer_reference', q['answer']),
                                   'answer_status': q['answer_status'],
                                   'allowed_global_ids': q['answer_annotations']['global_ids']}
                                  for q in reference['questions']]}


def validate_answers(parsed, requests):
    rows = parsed.get('frames')
    if not isinstance(rows, list) or len(rows) != len(requests):
        raise ValueError('GPT omitted frame entries')
    byid = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('frame_id') in byid:
            raise ValueError('Invalid/duplicate frame')
        byid[row.get('frame_id')] = row
    expected = {r['frame_id']: r for r in requests}
    if set(byid) != set(expected): raise ValueError('GPT returned foreign frame IDs')
    results = {}
    for fid, request in expected.items():
        answers = byid[fid].get('answers', [])
        if len(answers) != 5: raise ValueError('Missing difficulty answers: ' + fid)
        levels = {}
        allowed = {q['difficulty_level']: q for q in request['reference_answers']}
        for answer in answers:
            if not isinstance(answer, list) or len(answer) != 3:
                raise ValueError('Invalid answer row')
            level, text, ids = answer
            if type(level) is not int or level not in allowed or level in levels:
                raise ValueError('Invalid/duplicate difficulty')
            if not isinstance(text, str) or len(text.strip()) < 20:
                raise ValueError('Answer is empty or too short')
            if not isinstance(ids, list) or any(type(gid) is not int for gid in ids):
                raise ValueError('Invalid answer object IDs')
            if len(ids) != len(set(ids)) or set(ids) != set(allowed[level]['allowed_global_ids']):
                raise ValueError(f'{fid}/D{level}: Answer IDs differ from supplied truth; '
                                 f'expected {allowed[level]["allowed_global_ids"]}, got {ids}')
            # New numeric measurements are forbidden. Tolerate presentation
            # differences such as 1.20 vs 1.2, but not invented distances.
            permitted = {round(float(n), 2) for n in re.findall(r'-?\d+\.\d+', allowed[level]['text'])}
            decimals = {round(float(n), 2) for n in re.findall(r'-?\d+\.\d+', text)}
            if not decimals <= permitted:
                raise ValueError(f'{fid}/D{level}: invented numeric measurement {decimals-permitted}')
            if level == 4 and allowed[level]['answer_status'] == 'answerable':
                # Full 3D center triples must survive natural-language rewriting.
                triples = [tuple(float(n) for n in row) for row in re.findall(
                    r'[（(]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[)）]', text)]
                for obj in request['objects']:
                    if obj['global_id'] not in request['focus_global_ids']: continue
                    if not any(all(abs(round(v,2)-n)<.0051 for v,n in zip(obj['bbox']['center'],triple)) for triple in triples):
                        raise ValueError(fid + ': GPT dropped required world center coordinates')
            if level == 5 and request['camera_comparison']['status'] == 'answerable':
                numbers = {float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', text)}
                for distance in request['camera_comparison']['distances'].values():
                    if not any(abs(round(distance, 2)-n) < .0051 for n in numbers):
                        raise ValueError(fid + ': GPT dropped required camera distances')
            levels[level] = {'answer': text.strip(), 'global_ids': ids}
        results[fid] = levels
    return results


def partition_answers(parsed, requests):
    """Keep independently valid frames when another frame in a batch is invalid."""
    rows = parsed.get('frames', [])
    if not isinstance(rows, list):
        raise ValueError('Invalid frame list')
    expected = {r['frame_id']: r for r in requests}; returned = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('frame_id') not in expected or row['frame_id'] in returned:
            raise ValueError('Foreign or duplicate frame in response')
        returned[row['frame_id']] = row
    accepted = {}; errors = {}
    for fid, request in expected.items():
        try:
            if fid not in returned: raise ValueError(fid + ': frame omitted')
            accepted[fid] = validate_answers({'frames': [returned[fid]]}, [request])[fid]
        except (ValueError, TypeError, KeyError) as exc:
            errors[fid] = str(exc)
    return accepted, errors


def persist_answer(target, path, annotation, request, answers, questions, question_hash, cache, record):
    evidence = str(cache.relative_to(target)); evidence_hash = sha(cache)
    for q in annotation['questions']:
        level = q['difficulty_level']; value = answers[level]
        q.setdefault('answer_reference', q['answer'])
        q.setdefault('question_template_reference', q['question'])
        q.update(question=questions[level], answer=value['answer'], source='gpt:' + record['actual_model'],
                 answer_source='gpt:' + record['actual_model'], question_source='gpt:stable_question_bank',
                 generation_mode='gpt_answer_from_supplied_frame_ground_truth',
                 evidence=evidence, evidence_sha256=evidence_hash,
                 answer_grounding={'supplied_global_ids': value['global_ids'],
                                   'request_truth_sha256': fingerprint(request),
                                   'facts_sha256': annotation['facts_sha256']},
                 stable_question_bank_sha256=question_hash)
    annotation.update(answer_pipeline=VERSION, answer_evidence=evidence,
                      generation_mode='gpt_answer_from_supplied_frame_ground_truth', updated_at=utc())
    dump(path, annotation)


def recover_cached_answers(out, frames, questions, question_hash, glossary):
    """Reuse paid GPT results only when their supplied truth still matches exactly."""
    target = out / 'frame_qa'; wanted = {}; recovered = set()
    for frame in frames:
        path = target / 'frames' / (frame['id'] + '.json'); annotation = read(path)
        facts_path = target / 'facts' / (frame['id'] + '.json')
        if sha(facts_path) != annotation['facts_sha256']:
            raise ValueError('Changed facts require re-preparation: ' + frame['id'])
        wanted[frame['id']] = (path, annotation, truth_payload(read(facts_path), annotation, glossary))
    if not wanted: return recovered
    for cache in sorted((target / 'evidence/gpt_answers').glob('*.json')):
        if not wanted: break
        try:
            record = read(cache)
            if record.get('status') != 'ok' or not str(record.get('actual_model', '')).startswith('gpt-'):
                continue
            returned = record.get('parsed', {}).get('frames', [])
            relevant = [r for r in returned if r.get('frame_id') in wanted]
            if not relevant: continue
            raw = record['output_text']; raw_parsed = json.JSONDecoder().raw_decode(raw[raw.index('{'):])[0]
            if raw_parsed != record['parsed']: continue
            payload_text = record['prompt'].split('固定问题及逐帧真值如下：\n', 1)[1]
            supplied = json.JSONDecoder().raw_decode(payload_text)[0]
            if supplied.get('questions') != {str(k): v for k, v in questions.items()}: continue
            truths = {r['frame_id']: r for r in supplied['frames']}
            if len(truths) != len(supplied['frames']): continue
            if len({r['frame_id'] for r in returned}) != len(returned): continue
            if not {r['frame_id'] for r in returned} <= set(truths): continue
            for row in relevant:
                fid = row['frame_id']; path, annotation, truth = wanted[fid]
                if truths.get(fid) != truth: continue
                try: answers = validate_answers({'frames': [row]}, [truth])[fid]
                except (ValueError, KeyError, TypeError): continue
                persist_answer(target, path, annotation, truth, answers, questions, question_hash, cache, record)
                recovered.add(fid); del wanted[fid]
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            continue
    return recovered


def annotate_batch(model, out, frames, questions, question_hash, glossary, batch_size):
    target = out / 'frame_qa'
    prepared = []
    for f in frames:
        path = target / 'frames' / (f['id'] + '.json')
        annotation = read(path)
        facts = read(target / 'facts' / (f['id'] + '.json'))
        prepared.append((path, annotation, truth_payload(facts, annotation, glossary)))
    remaining = {row[2]['frame_id']: row for row in prepared}; completed = []; errors = {}
    try:
        for attempt in range(3):
            request = [row[2] for row in remaining.values()]
            request_hash = fingerprint([VERSION, questions, question_hash, request])
            cache = target / 'evidence/gpt_answers' / (request[0]['frame_id'] + f'_n{len(request)}_{request_hash[:12]}' + (f'_retry{attempt}' if attempt else '') + '.json')
            prompt = PROMPT + json.dumps({'questions': questions, 'frames': request}, ensure_ascii=False, separators=(',', ':'))
            if errors:
                prompt += '\n逐帧校验失败原因：' + json.dumps(errors, ensure_ascii=False) + '。仅重新回答本次列出的帧，精确保留每题给定的 ID、坐标和距离。'
            parsed = model.call(prompt, [], cache, max_tokens=max(5000, len(request) * 2500))
            record = read(cache)
            if not str(record.get('actual_model', '')).startswith('gpt-'):
                raise ValueError('Actual GPT model missing')
            try: results, errors = partition_answers(parsed, request)
            except ValueError as exc:
                results = {}; errors = {fid: str(exc) for fid in remaining}
            for fid, answers in results.items():
                path, annotation, truth = remaining.pop(fid)
                persist_answer(target, path, annotation, truth, answers, questions, question_hash, cache, record)
                completed.append(fid)
            if not remaining: return completed
        raise ValueError(json.dumps(errors, ensure_ascii=False))
    except Exception as exc:
        exc.completed_frame_ids = completed
        raise


def update_status(out, completed, failed, state='running'):
    target = out / 'frame_qa'; path = target / 'manifest.json'
    manifest = read(path)
    if len(completed) == manifest['annotated_frames']:
        state = 'complete'; failed = []
    elif state == 'complete':
        state = 'subset_complete'
    manifest['gpt_answers'] = {'pipeline': VERSION, 'status': state,
                               'completed_frames': len(completed), 'total_frames': manifest['annotated_frames'],
                               'failed_batches': failed, 'updated_at': utc()}
    if state == 'blocked' and failed:
        manifest['gpt_answers']['blocking_reason'] = failed[-1]['error']
    if len(completed) == manifest['annotated_frames']:
        manifest['generation_mode'] = 'gpt_answer_from_supplied_frame_ground_truth'
    elif completed:
        manifest['generation_mode'] = 'mixed_gpt_answers_and_references'
    dump(path, manifest)
    dump(target / 'gpt_run_info.json', {**manifest['gpt_answers'], 'scene_id': out.parent.name,
                                      'question_answer_count': len(completed) * 5})


def valid_completed_frame(target, frame_id, questions, question_hash, glossary, evidence_cache):
    """Resume only answers tied to the current facts and unmodified GPT evidence."""
    try:
        annotation = read(target / 'frames' / (frame_id + '.json'))
        if annotation.get('answer_pipeline') != VERSION or len(annotation.get('questions', [])) != 5:
            return False
        facts_path = target / 'facts' / (frame_id + '.json')
        facts_hash = sha(facts_path)
        if annotation.get('facts_sha256') != facts_hash:
            return False
        evidence_path = (target / annotation['answer_evidence']).resolve()
        if not evidence_path.is_relative_to(target.resolve()):
            return False
        if evidence_path not in evidence_cache:
            evidence_cache[evidence_path] = (read(evidence_path), sha(evidence_path))
        evidence, evidence_hash = evidence_cache[evidence_path]
        if evidence.get('status') != 'ok' or not str(evidence.get('actual_model', '')).startswith('gpt-'):
            return False
        truth = truth_payload(read(facts_path), annotation, glossary)
        truth_hash = fingerprint(truth)
        parsed = {'frames': [row for row in evidence['parsed']['frames'] if row.get('frame_id') == frame_id]}
        answers = validate_answers(parsed, [truth])[frame_id]
        return all(q.get('stable_question_bank_sha256') == question_hash
                   and q.get('question') == questions[q['difficulty_level']]
                   and q.get('answer') == answers[q['difficulty_level']]['answer']
                   and q.get('evidence_sha256') == evidence_hash
                   and q.get('answer_grounding', {}).get('facts_sha256') == facts_hash
                   and q.get('answer_grounding', {}).get('request_truth_sha256') == truth_hash
                   for q in annotation['questions'])
    except (OSError, KeyError, ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', action='append'); parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=4); parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--recover-cache-only', action='store_true', help='Recover valid existing GPT responses without new API calls')
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 8 or not 1 <= args.workers <= 16:
        parser.error('batch-size 1..8; workers 1..16')
    lock_path = ROOT / 'services/frame_qa_grounded.lock'
    with lock_path.open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: parser.error('Grounded GPT annotation is already running')
        model = VLM(model=os.environ.get('FRAME_QA_GPT_MODEL', 'gpt-6-sol')) if not args.recover_cache_only else None
        if args.recover_cache_only:
            question_path = LANGUAGE / 'stable_questions.json'; question_record = read(question_path)
            questions = {r['difficulty_level']: r['question'] for r in question_record['parsed']['levels']}
        else: questions, question_path, question_record = stable_questions(model)
        question_hash = sha(question_path)
        glossary = read(LANGUAGE / 'glossary.json')['names']
        scenes = sorted((ROOT / 'auto').glob('*/global_instances/frame_qa/manifest.json'))
        if args.scene:
            scenes = [p for p in scenes if p.parents[2].name in args.scene]
            if len(scenes) != len(set(args.scene)): parser.error('Unknown scene')
        pending = []; states = {}
        for path in scenes:
            out = path.parents[1]; manifest = read(path)
            if manifest['source_hashes'] != input_hashes(out): raise ValueError('Stale source ' + out.parent.name)
            frames = read(out / 'frames.json')['frames']
            dump(path.parent / 'evidence/stable_questions.json', read(question_path))
            dump(path.parent / 'stable_questions.json', {'questions': questions,
                 'source': 'gpt:' + question_record['actual_model'], 'evidence': 'evidence/stable_questions.json',
                 'sha256': question_hash, 'created_at': utc()})
            completed = set(); waiting = []; evidence_cache = {}
            for frame in frames:
                valid = valid_completed_frame(path.parent, frame['id'], questions, question_hash, glossary, evidence_cache)
                if valid: completed.add(frame['id'])
                elif args.limit is None or frame['index'] < args.limit: waiting.append(frame)
            recovered = recover_cached_answers(out, waiting, questions, question_hash, glossary)
            completed.update(recovered); waiting = [f for f in waiting if f['id'] not in recovered]
            if recovered: print('RECOVERED GPT CACHE', out.parent.name, len(recovered), 'frames', flush=True)
            states[str(out)] = {'completed': completed, 'failures': []}
            previous = manifest.get('gpt_answers', {})
            update_status(out, completed, previous.get('failed_batches', []) if args.recover_cache_only else [],
                          previous.get('status', 'not_generated') if args.recover_cache_only else 'running')
            for start in range(0, len(waiting), args.batch_size):
                pending.append((out, waiting[start:start + args.batch_size]))
        if args.recover_cache_only:
            report(); return
        # Interleave scenes so every household receives real GPT answers early.
        pending.sort(key=lambda row: (row[1][0]['index'], row[0].parent.name))
        blocked = None
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            queue = iter(pending); futures = {}
            def submit_next():
                item = next(queue, None)
                if item is not None:
                    out, frames = item
                    future = pool.submit(annotate_batch, model, out, frames, questions, question_hash, glossary, args.batch_size)
                    futures[future] = item
            for _ in range(args.workers): submit_next()
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in ready:
                    out, frames = futures.pop(future); state = states[str(out)]
                    try:
                        ids = future.result(); state['completed'].update(ids)
                        print('GROUNDED GPT', out.parent.name, len(state['completed']), 'frames', ids[0], flush=True)
                    except Exception as exc:
                        state['completed'].update(getattr(exc, 'completed_frame_ids', []))
                        failure = {'frame_ids': [f['id'] for f in frames if f['id'] not in state['completed']], 'error': str(exc)[:500]}
                        if isinstance(exc, GatewayRequestError) and not exc.retryable:
                            blocked = failure
                        state['failures'].append(failure); print('GPT FAILED', out.parent.name, failure, flush=True)
                    update_status(out, state['completed'], state['failures'])
                    if len(state['completed']) == read(out/'frame_qa/manifest.json')['annotated_frames']:
                        from .frame_qa import verify_scene
                        verify_scene(out)
                report()
                if blocked is None:
                    for _ in ready: submit_next()
        for out_string, state in states.items():
            failures = list(state['failures'])
            if blocked and (not failures or failures[-1]['error'] != blocked['error']):
                failures.append({'frame_ids': [], 'error': blocked['error']})
            update_status(Path(out_string), state['completed'], failures,
                          'blocked' if blocked else 'failed' if failures else 'complete')
        report()
        if blocked: raise RuntimeError('Gateway rejected the request; stopped without submitting remaining batches: '+blocked['error'])
        if any(state['failures'] for state in states.values()): raise RuntimeError('Some GPT batches failed; rerun to resume')


if __name__ == '__main__': main()
