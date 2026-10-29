import copy
import json
import tempfile
import unittest
from unittest.mock import Mock
from pathlib import Path
import cv2
import numpy as np
from .common import dump, sha, fingerprint
from .frame_qa_facts import camera_comparison, color_evidence, image_relation, build_facts
from .frame_qa_grounded import (validate_answers, valid_completed_frame, truth_payload, VERSION,
                               partition_answers, recover_cached_answers, PROMPT)
from .frame_qa_prompts import template_fields, validate_bank, SLOTS
from annotation_pipeline.vlm import check_response, GatewayRequestError
from services.resume_frame_qa import recoverable_gateway_reason, quota_refill


def obj(gid, distance=2., category='thing'):
    return {'global_id': gid, 'semantic_label': 'chair' if gid == 1 else 'table',
            'category_type': category, 'score': .9,
            'image': {'mask_pixels': 1000, 'center_px': [20*gid, 30*gid]},
            'bbox': {'center': [gid, 0, 1], 'size': [.5, .5, 1.]},
            'camera': {'support_points': 30, 'visible_surface_distance_median': distance}}


class FrameQaChecks(unittest.TestCase):
    def test_supervisor_prioritizes_shared_quota_over_frame_parse_failure(self):
        quota = 'Quota exceeded; next refill at 2026/10/19 06:00:00 (Asia/Shanghai)'
        blocked = [(None, {'blocking_reason':'Invalid JSON in earlier frame',
                          'failed_batches':[{'error':'Invalid JSON'}, {'error':quota}]}),
                   (None, {'blocking_reason':quota})]
        self.assertEqual(recoverable_gateway_reason(blocked), quota)
        self.assertIsNotNone(quota_refill(quota))
        self.assertIsNone(recoverable_gateway_reason([(None, {'blocking_reason':'Invalid JSON'})]))

    def test_partial_batch_preserves_valid_frame_and_identifies_wrong_ids(self):
        requests = [{'frame_id':fid, 'objects':[], 'focus_global_ids':[],
                     'camera_comparison':{'status':'insufficient_evidence'},
                     'reference_answers':[{'difficulty_level':k, 'allowed_global_ids':[],
                         'answer_status':'insufficient_or_ambiguous_evidence','text':'没有有效的物体证据。'} for k in range(1,6)]}
                    for fid in ['g000000','g000001']]
        parsed = {'frames':[{'frame_id':r['frame_id'], 'answers':[
            [k,'当前画面缺少可确认的物体，不能可靠回答颜色与空间关系。',[]] for k in range(1,6)]} for r in requests]}
        parsed['frames'][1]['answers'][4][2] = [999]
        accepted, errors = partition_answers(parsed, requests)
        self.assertEqual(set(accepted), {'g000000'})
        self.assertIn('g000001/D5', errors['g000001'])
        self.assertIn('expected []', errors['g000001'])
        parsed['frames'].append(parsed['frames'][0])
        with self.assertRaisesRegex(ValueError,'duplicate'): partition_answers(parsed, requests)

    def test_recover_cache_requires_the_exact_supplied_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); target = out/'frame_qa'; questions = {k:f'问题{k}' for k in range(1,6)}
            frames = [{'id':'g000000'}, {'id':'g000001'}]; requests = []; rows = []
            answer = '当前画面中没有可可靠确认的物体类别，也没有支持空间判断的充分证据。'
            for frame in frames:
                fid = frame['id']
                facts = {'frame_id':fid,'floor_id':'f0','objects':[],'focus_global_ids':[],
                         'camera_comparison':{'global_ids':[],'status':'insufficient_evidence'},
                         'relative_relations':[],'unclassified_visible_global_ids':[]}
                dump(target/'facts'/(fid+'.json'),facts)
                annotation = {'frame_id':fid,'facts_sha256':sha(target/'facts'/(fid+'.json')),
                              'questions':[{'difficulty_level':k,'question':questions[k],'answer':answer,
                                  'answer_status':'insufficient_or_ambiguous_evidence',
                                  'answer_annotations':{'global_ids':[]}} for k in range(1,6)]}
                dump(target/'frames'/(fid+'.json'),annotation)
                requests.append(truth_payload(facts,annotation,{}))
                rows.append({'frame_id':fid,'answers':[[k,answer,[]] for k in range(1,6)]})
            # The second response used different facts; it must not be adopted.
            requests[1]['floor_id'] = 'other_floor'
            parsed = {'frames':rows}
            record = {'status':'ok','actual_model':'gpt-test','parsed':parsed,
                      'output_text':json.dumps(parsed,ensure_ascii=False),
                      'prompt':PROMPT+json.dumps({'questions':questions,'frames':requests},ensure_ascii=False)}
            dump(target/'evidence/gpt_answers/batch.json',record)
            self.assertEqual(recover_cached_answers(out,frames,questions,'bank_hash',{}),{'g000000'})
            self.assertTrue(valid_completed_frame(target,'g000000',questions,'bank_hash',{},{}))
            self.assertFalse(valid_completed_frame(target,'g000001',questions,'bank_hash',{},{}))

    def test_gateway_rejection_is_redacted_and_permanent_errors_do_not_retry(self):
        response = Mock(status_code=400, reason='Bad Request')
        response.json.return_value = {'error': {'code': 'invalid_request', 'message': 'rejected fake-secret'}}
        with self.assertRaises(GatewayRequestError) as caught:
            check_response(response, 'fake-secret')
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn('fake-secret', str(caught.exception))
        response.close.assert_called_once()
        response.status_code = 429
        with self.assertRaises(GatewayRequestError) as caught:
            check_response(response, 'fake-secret')
        self.assertTrue(caught.exception.retryable)

    def test_resume_rejects_changed_facts_or_gpt_response(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            facts = {'frame_id':'g000000', 'floor_id':'f0', 'objects':[], 'focus_global_ids':[],
                     'camera_comparison':{'global_ids':[], 'status':'insufficient_evidence'},
                     'relative_relations':[], 'unclassified_visible_global_ids':[]}
            dump(target/'facts/g000000.json', facts)
            digest = sha(target/'facts/g000000.json')
            questions = {k:f'这是第{k}档问题' for k in range(1,6)}
            answer = '当前画面缺少可以可靠确认的对象，无法判断所要求的特征或空间关系。'
            annotation = {'frame_id':'g000000', 'answer_pipeline':VERSION, 'facts_sha256':digest,
                          'answer_evidence':'evidence/batch.json', 'questions':[
                              {'difficulty_level':k, 'question':questions[k], 'answer':answer,
                               'answer_reference':answer, 'answer_status':'insufficient_or_ambiguous_evidence',
                               'stable_question_bank_sha256':'bank_hash',
                               'answer_annotations':{'global_ids':[]}} for k in range(1,6)]}
            evidence = {'status':'ok', 'actual_model':'gpt-test', 'parsed':{
                'frames':[{'frame_id':'g000000', 'answers':[[k,answer,[]] for k in range(1,6)]}]}}
            dump(target/'evidence/batch.json', evidence)
            for q in annotation['questions']:
                q['evidence_sha256'] = sha(target/'evidence/batch.json')
                q['answer_grounding'] = {'facts_sha256':digest,
                    'request_truth_sha256':fingerprint(truth_payload(facts,annotation,{}))}
            dump(target/'frames/g000000.json', annotation)
            self.assertTrue(valid_completed_frame(target,'g000000',questions,'bank_hash',{},{}))
            facts['floor_id'] = 'changed'; dump(target/'facts/g000000.json',facts)
            self.assertFalse(valid_completed_frame(target,'g000000',questions,'bank_hash',{},{}))
            facts['floor_id'] = 'f0'; dump(target/'facts/g000000.json',facts)
            evidence['parsed']['frames'][0]['answers'][0][1] = '另一条答案'
            dump(target/'evidence/batch.json',evidence)
            self.assertFalse(valid_completed_frame(target,'g000000',questions,'bank_hash',{},{}))

    def test_distance_is_camera_relative_and_refuses_ties(self):
        result = camera_comparison([obj(1, 4), obj(2, 2)])
        self.assertEqual(result['nearer_global_id'], 2)
        self.assertEqual(result['farther_global_id'], 1)
        self.assertEqual(result['reference_frame'], 'current_frame_camera_origin')
        self.assertEqual(camera_comparison([obj(1, 2), obj(2, 2.1)])['status'], 'ambiguous')
        # A continuous wall must not become a second independent object.
        self.assertEqual(camera_comparison([obj(1), obj(2, category='stuff')])['status'], 'insufficient_evidence')

    def test_image_y_down_is_distinct_from_world_z_up(self):
        a, b = obj(1), obj(2)
        a['bbox']['center'][2] = 5.
        a['image']['center_px'] = [80, 80]
        b['image']['center_px'] = [20, 20]
        result = image_relation(a, b, 100, 100)
        self.assertEqual(result['image_horizontal'], 'right_of')
        self.assertEqual(result['image_vertical'], 'below')
        self.assertEqual(result['canonical_center_delta_xyz'][2], 4.)
        self.assertIn('support_on', result['not_inferred'])

    def test_color_uses_mask_and_unknown_remains_unknown(self):
        blue = color_evidence(np.tile([0, 0, 255], (100, 1)).astype(np.uint8))
        self.assertEqual(blue['label'], 'blue')
        empty = color_evidence(np.empty((0, 3), np.uint8))
        self.assertEqual(empty['label'], 'unknown')
        self.assertEqual(empty['score'], 0.)

    def test_visible_surface_distance_uses_points_not_bbox_center_or_z(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'household_test/global_instances'
            (out/'masks/c0').mkdir(parents=True)
            (out/'projections/c0').mkdir(parents=True)
            local = np.ones((10,10), np.uint16)
            np.savez_compressed(out/'masks/c0/g000000.npz', local_instances=local)
            np.savez_compressed(out/'projections/c0/g000000.npz', point_indices=np.arange(100), pixels=np.arange(100))
            cv2.imwrite(str(out/'rgb.jpg'), np.full((10,10,3), 90, np.uint8))
            frame = {'id':'g000000','clip_id':'c0','floor':'f0','timestamp':0.,'rgb':'rgb.jpg',
                     'width':10,'height':10,'K':np.eye(3).tolist(),'camera_to_canonical':np.eye(4).tolist(),
                     'rgb_sha256':'stale_hash'}
            points = np.tile([3.,0.,4.], (100,1))
            objects = {1:{'global_id':1,'semantic_label':'chair','category_type':'thing','score':.9,
                          'semantic_pred':{'source':'test'},'bbox':{'center':[50.,0.,0.],'size':[1.,1.,1.]}}}
            assignments = [{'accepted':True,'global_id':1,'local_id':1,'purity':1.,'score':.9}]
            value = build_facts(out, frame, objects, assignments, points, np.ones(100,np.int32), {})
            camera = value['objects'][0]['camera']
            self.assertEqual(camera['visible_surface_distance_median'], 5.)
            self.assertEqual(camera['visible_surface_depth_z_median'], 4.)
            self.assertEqual(camera['bbox_center_euclidean_distance'], 50.)
            self.assertFalse(value['source_integrity']['manifest_hash_matches'])
            self.assertEqual(value['objects'][0]['object_id'], 'household_test::g1')

    def test_unaccepted_and_unknown_instances_are_not_claimed_as_known_objects(self):
        # This is covered separately from raw model labels: unknown IDs remain
        # visible evidence, but must not acquire a guessed object category.
        self.assertEqual(camera_comparison([])['global_ids'], [])
        weak = obj(1); weak['camera']['support_points'] = 19
        self.assertEqual(camera_comparison([weak, obj(2)])['status'], 'insufficient_evidence')

    def test_question_templates_cannot_leak_answers_or_execute_expressions(self):
        with self.assertRaises(ValueError): template_fields('{obj.__class__}')
        with self.assertRaises(ValueError): template_fields('{value!r}')
        rows = [{'difficulty_level': k,
                 'question_template': '{focus_names}在哪？' if k in (2,3,4) else '{pair_names}谁近？' if k == 5 else '看到了什么？',
                 'answer_template': ''.join('{'+s+'}' for s in sorted(SLOTS[k]) if s not in ('focus_names','pair_names'))}
                for k in range(1,6)]
        validate_bank({'levels':rows})
        rows[0]['question_template'] = '答案是不是{inventory}？'
        with self.assertRaises(ValueError): validate_bank({'levels':rows})

    def test_actual_gpt_answers_require_exact_ids_and_supplied_numbers(self):
        truth = {'frame_id':'g000000','objects':[], 'focus_global_ids':[],
                 'camera_comparison':{'status':'insufficient_evidence'},
                 'reference_answers':[{'difficulty_level':level,'allowed_global_ids':[1],
                                       'answer_status':'answerable','text':'物体距离为2.50，颜色为黑色。'} for level in range(1,6)]}
        parsed = {'frames':[{'frame_id':'g000000','answers':[[k,'当前画面中的已知物体颜色为黑色，可见表面距离约2.50。',[1]] for k in range(1,6)]}]}
        self.assertEqual(len(validate_answers(parsed,[truth])['g000000']),5)
        wrong = copy.deepcopy(parsed); wrong['frames'][0]['answers'][0][2] = [999]
        with self.assertRaises(ValueError): validate_answers(wrong,[truth])
        wrong = copy.deepcopy(parsed); wrong['frames'][0]['answers'][2][1] += '相隔7.25。'
        with self.assertRaises(ValueError): validate_answers(wrong,[truth])
        wrong = copy.deepcopy(parsed); wrong['frames'][0]['frame_id'] = 'g000001'
        with self.assertRaises(ValueError): validate_answers(wrong,[truth])

    def test_global_coordinates_and_camera_distances_cannot_be_omitted(self):
        truth = {'frame_id':'g000000','objects':[{'global_id':1,'bbox':{'center':[1.25,-2.3,0.5]}}],
                 'focus_global_ids':[1], 'camera_comparison':{'status':'answerable','distances':{'1':3.25,'2':4.5}},
                 'reference_answers':[{'difficulty_level':level,'allowed_global_ids':[1],
                                       'answer_status':'answerable','text':'中心(1.25,-2.30,0.50)，距离3.25和4.50。'} for level in range(1,6)]}
        parsed = {'frames':[{'frame_id':'g000000','answers':[[k,'当前画面中的物体可以辨认，但是这里故意没有给出任何坐标或距离。',[1]] for k in range(1,6)]}]}
        with self.assertRaisesRegex(ValueError,'world center coordinates'): validate_answers(parsed,[truth])
        parsed['frames'][0]['answers'][3][1] = '物体的观测中心坐标为(0.50,-2.30,1.25)，这里的坐标顺序故意错误。'
        with self.assertRaisesRegex(ValueError,'world center coordinates'): validate_answers(parsed,[truth])
        parsed['frames'][0]['answers'][3][1] = '物体的观测中心坐标为(1.25,-2.30,0.50)，只表示已经观测的表面。'
        with self.assertRaisesRegex(ValueError,'camera distances'): validate_answers(parsed,[truth])
        truth['camera_comparison']['distances'] = {'1':3.0,'2':4.5}
        truth['reference_answers'][4]['text'] += '也可以有距离3.00。'
        parsed['frames'][0]['answers'][4][1] = '当前画面中的物体有可靠证据，两个对象到相机的距离分别为3和4.50。'
        self.assertEqual(len(validate_answers(parsed,[truth])['g000000']),5)


if __name__ == '__main__': unittest.main()
