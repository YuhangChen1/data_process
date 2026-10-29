"""Review priorities from automatic evidence, without changing instance topology."""
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .common import read,dump,utc
import numpy as np

def review_issues(out,config):
    doc=read(out/'global_instances.json');assignments=read(out/'frame_assignments.json')['frames']
    with np.load(out/'superpoints.npz') as cloud:
        edges=cloud['adjacency'];sizes=cloud['superpoint_sizes']
    with np.load(out/'fine_instances.npz') as fine:labels=fine['superpoint_fine_ids']
    issues=[]
    def add(kind,ref,detail,score,priority='medium',evidence=None):
        issues.append({'type':kind,'ref':ref,'detail':detail,'score':float(score),
            'source':'automatic_evidence_qa','priority':priority,'evidence':evidence,
            'review':{'status':'unreviewed'}})
    for item in doc['instances']:
        gid=item['global_id'];pred=item['semantic_pred']
        if item['semantic_label']=='unknown':
            add('semantic_unknown',gid,pred['unknown_reason'],0.,'high','global_instances.json')
        elif pred['score']<.60:
            add('low_semantic_vote_confidence',gid,'Winning semantic vote share below 0.60.',pred['score'],'high','global_instances.json')
        if item['n_frames']<3:
            add('few_supported_frames',gid,f"Only {item['n_frames']} frames have accepted 3D-supported masks.",min(1.,item['n_frames']/3),'medium','frame_assignments.json')
        # Separate observed components are a review signal; occlusion and missing
        # mesh may also disconnect a genuine whole object, so do not auto-split.
        nodes=np.flatnonzero(labels==gid)
        if len(nodes)<2:continue
        lookup=np.full(len(labels),-1,np.int32);lookup[nodes]=np.arange(len(nodes))
        selected=edges[(labels[edges[:,0]]==gid)&(labels[edges[:,1]]==gid)]
        local=lookup[selected]
        graph=csr_matrix((np.ones(len(local)*2),(np.r_[local[:,0],local[:,1]],np.r_[local[:,1],local[:,0]])),shape=(len(nodes),len(nodes)))
        count,components=connected_components(graph,directed=False)
        mass=np.bincount(components,weights=sizes[nodes]);fraction=float(mass.max()/max(mass.sum(),1))
        large=int(np.count_nonzero(mass>=max(20,.05*mass.sum())))
        if item['category_type']=='thing' and large>1:
            add('disconnected_observed_geometry',gid,f'{large} substantial sampled surface components; largest contains {fraction:.1%} of instance points. Occlusion or mesh gaps can also cause this.',fraction,'medium','fine_instances.npz')
    reasons=Counter(a['reason'] for records in assignments.values() for a in records if not a['accepted'])
    for fid,records in assignments.items():
        if not records:
            add('no_usable_2d_masks',fid,'No usable SAM3 masks after the full prompt bank; inspect RGB reconstruction quality and concepts. No labels were invented.',0.,'high','frames.json')
        elif not any(a['accepted'] for a in records):
            add('no_reliable_global_mask',fid,'2D masks exist but no mask has reliable 3D support and purity.',0.,'medium','frame_assignments.json')
    for reason,count in sorted(reasons.items()):
        add('rejected_mask_assignments',None,f'{count} frame masks: {reason}. Rejected labels remain 0.',0.,'medium','frame_assignments.json')
    descriptions=read(out/'descriptions.json') if (out/'descriptions.json').exists() else {}
    for text in descriptions.get('level1',[])+descriptions.get('level2',[]):
        value=str(text.get('text',''));confidence=float(text.get('score') or 0)
        if confidence<.60:
            add('low_description_confidence',text['id'],'GPT description score below 0.60; inspect the source frames.',confidence,'high',text.get('evidence'))
        if text.get('level')==1 and any(term in value for term in ['上一帧','前一帧','上帧','上一张','前一张']):
            add('frame_description_context_dependency',text['id'],'Level 1 text refers to an earlier frame; review as a standalone image description.',confidence,'medium',text.get('evidence'))
        if any(term in value for term in ['左侧','右侧','左边','右边','左方','右方','左前方','右前方']):
            add('view_relative_description',text['id'],'Text contains a view-relative left/right phrase; not a world-coordinate relation. Review before using in route text.',confidence,'medium',text.get('evidence'))
    order={'high':0,'medium':1,'low':2};issues.sort(key=lambda x:(order[x['priority']],x['score'],str(x['ref'])))
    dump(out/'qa/issues.json',{'created_at':utc(),'issues':issues,'counts':dict(Counter(i['type'] for i in issues)),
        'scope':'Review priorities from automatic evidence only; no human accuracy claims or automatic object splitting.',
        'quality_measure':'SAM3 text grounding has no predicted mask IoU; threshold stability is an explicit proxy.',
        'bbox_limit':'Bounding boxes describe observed surfaces and may include segmentation leakage or miss occluded extent.'})
    return issues

if __name__=='__main__':
    import argparse
    from .common import destination,DEFAULTS
    parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);args=parser.parse_args()
    issues=review_issues(destination(args.scene),DEFAULTS);print(args.scene,len(issues),'automatic review priorities')
