"""Check numeric IDs, exclusive mask boundaries, temporal references and QA."""
from .common import ROOT,read,dump,frame_mask,utc
import cv2,numpy as np

def verify(out,doc,config,include_descriptions=True):
    from .qa import review_issues
    review_issues(out,config)
    errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    data=np.load(out/'fine_instances.npz');coarse=np.load(out/'coarse_instances.npz')
    ids=data['point_fine_ids'];point_coarse=coarse['point_coarse_ids'];sp=data['point_superpoint_ids']
    inventory=read(out/'concepts.json')
    check(inventory.get('all_frames_direct_gpt_audited') is True,'Not every RGB frame was directly audited by GPT')
    check(len(ids)==len(data['points'])==len(sp),'Point array correspondence')
    check(np.isfinite(data['points']).all(),'Non-finite points')
    check(np.array_equal(ids,data['superpoint_fine_ids'][sp]),'point_fine_ids do not derive from fine superpoints')
    check(set(ids[ids>0])==set(point_coarse[point_coarse>0]),'Fine topology changed')
    summary=read(out/'coarse_summary.json');check(summary['hard_conflicts_inside_clusters']==0,'Hard cannot-link violated')
    global_doc=read(out/'global_instances.json');known={o['global_id'] for o in global_doc['instances']};semantics=global_doc['semantic_registry']
    check(known==set(ids[ids>0]),'Global instance registry mismatch')
    assignments=read(out/'frame_assignments.json')['frames'];rendered={f['frame_id']:f for f in read(out/'rendered_frames.json')['frames']}
    frame_ids={f['id'] for f in doc['frames']}
    check(frame_ids==set(assignments)==set(rendered),'Missing per-frame assignments/images')
    for frame in doc['frames']:
        fid=frame['id'];record=rendered[fid]
        instance=cv2.imread(str(out/record['instance']),cv2.IMREAD_UNCHANGED);semantic=cv2.imread(str(out/record['semantic']),cv2.IMREAD_UNCHANGED)
        check(instance is not None and instance.dtype==np.uint16 and instance.shape==(frame['height'],frame['width']),fid+': invalid instance PNG')
        check(semantic is not None and semantic.dtype==np.uint16 and semantic.shape==instance.shape,fid+': invalid semantic PNG')
        check(set(np.unique(instance))<=(known|{0}),fid+': foreign global ID')
        check(set(np.unique(semantic))<=set(semantics.values()),fid+': foreign semantic ID')
        with np.load(frame_mask(out,frame)) as mask:local=mask['local_instances']
        for a in assignments[fid]:
            area=local==a['local_id'];expected=a['global_id']
            check(np.all(instance[area]==expected),fid+': modified or overlapping mask boundary')
            if a['accepted']:check(a['support_points']>=20 and a['purity']>=config['global_purity'],fid+': unreliable ID accepted')
        check(np.all(instance[local==0]==0),fid+': IDs invented outside SAM masks')
        for key in ['overlay','instance_color','semantic_color','sam2d']:
            check((out/record[key]).is_file(),fid+': missing rendered '+key)
    desc={'level1':[],'level2':[]}
    if include_descriptions:
        desc=read(out/'descriptions.json');check(not desc['failures'],'Failed GPT description batches')
        check({x['frame_id'] for x in desc['level1']}==frame_ids,'Level1 frame coverage')
        expected_pairs={(a,b) for clip in doc['clips'] for a,b in zip(clip['frames'][:-1],clip['frames'][1:])}
        check({(x['from_frame'],x['to_frame']) for x in desc['level2']}==expected_pairs,'Level2 transition coverage/cut leakage')
        for text in desc['level1']+desc['level2']:
            check(text['source'].startswith('gpt:') and 'qwen' not in text['source'].lower(),'Non-GPT description')
            check(text['review']=={'status':'unreviewed'},'Human review contaminated')
            check(bool(text['text']),text['id']+': empty description')
            evidence=read(out/text['evidence']);check(evidence.get('status')=='ok' and str(evidence.get('actual_model','')).startswith('gpt-'),'Missing real GPT evidence')
    for clip in doc['clips']:
        for name in ['rgb','overlay','instance','semantic']:
            cap=cv2.VideoCapture(str(out/'videos'/clip['id']/(name+'.mp4')))
            check(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(clip['frames']),clip['id']+': invalid '+name+' video count');cap.release()
    result={'passed':not errors,'created_at':utc(),'errors':errors,'frames':len(frame_ids),'clips':len(doc['clips']),
            'global_instances':len(known),'level1':len(desc['level1']),'level2':len(desc['level2']),
            'every_frame_gpt_inventory':inventory.get('all_frames_direct_gpt_audited',False),
            'global_id_source':'fine_instances.npz:point_fine_ids','all_reviews_unreviewed':True,
            'all_numeric_pngs_checked':True,'all_mask_boundaries_checked':True,
            'scope':'full_outputs' if include_descriptions else 'local_geometry_masks_videos_only',
            'descriptions_checked':include_descriptions}
    dump(out/('validation.json' if include_descriptions else 'local_validation.json'),result)
    print('VERIFY GLOBAL',out.parent.name,result,flush=True)
    if errors:raise RuntimeError('Global output validation failed: '+str(errors[:10]))
