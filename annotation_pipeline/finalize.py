from .common import read,dump,base,score,stable_ids,OBJECT_ZH
import numpy as np
import re
from collections import Counter

def relations_and_scale(graph):
    obs=graph['objects'];fixed={'sofa','bed','cabinet','wardrobe','desk','bookshelf','television','door','window','refrigerator','sink','toilet','bathtub','shower','washing machine','air conditioner','stove','oven','railing','stairs'}
    graph['issues']=[x for x in graph['issues'] if x['type']!='scale_unverified']
    for obj in obs:obj['relations_auto']=[]
    counts=Counter((o['region'],o['category_pred']['label']) for o in obs)
    for a in obs:
        center=np.array(a['bbox']['center']);size=np.array(a['bbox']['size']);lo=center-size/2;hi=center+size/2
        for b in obs:
            if a['id']==b['id'] or a['floor']!=b['floor']:continue
            bc=np.array(b['bbox']['center']);bs=np.array(b['bbox']['size']);bl=bc-bs/2;bh=bc+bs/2
            gaps=np.maximum(0,np.maximum(lo-bh,bl-hi));dist=float(np.linalg.norm(gaps));xyinter=float(np.maximum(0,np.minimum(hi[:2],bh[:2])-np.maximum(lo[:2],bl[:2])).prod());xyfrac=xyinter/max(min(float(size[:2].prod()),float(bs[:2].prod())),1e-6)
            confidence=min(a['geometry_pred']['score'],b['geometry_pred']['score'])*.75
            if dist<1.0:a['relations_auto'].append({'type':'near','target':b['id'],'dist':dist,'distance_definition':'minimum AABB surface distance',**base('geometry:aabb_distance',confidence)})
            if xyfrac>.25:
                if bc[2]-center[2]>.25:a['relations_auto'].append({'type':'below','target':b['id'],**base('geometry:vertical_order+xy_overlap',confidence)})
                if center[2]-bc[2]>.25:a['relations_auto'].append({'type':'above','target':b['id'],**base('geometry:vertical_order+xy_overlap',confidence)})
                if abs(lo[2]-bh[2])<.12 and a['category_pred']['label'] not in ['door','window','curtain','railing'] and b['category_pred']['label'] in ['table','desk','cabinet','bed','sofa','refrigerator','stove']:
                    a['relations_auto'].append({'type':'on','target':b['id'],'vertical_gap':float(lo[2]-bh[2]),**base('geometry:support_contact_candidate',confidence*.8)})
        for c in graph['connections']:
            if a['floor'] not in c.get('floors',[c.get('floor')]):continue
            dist=float(np.linalg.norm(center[:2]-np.array(c['position'][:2])))
            if dist<1.5:a['relations_auto'].append({'type':'near_connection','target':c['id'],'dist':dist,**base('geometry:center_to_portal',.7*a['geometry_pred']['score'])})
        label=a['category_pred']['label'];unique=1/counts[(a['region'],label)];volume=float(np.prod(size))
        a['key_score']=float(.35*(label in fixed)+.2*min(volume/2,1)+.25*min(a['n_views']/5,1)+.2*unique)
        a['key_score_source']='rule:fixed_furniture+observed_volume+views+within_region_uniqueness'
    candidates=[]
    for o in obs:
        if o['category_pred']['label']!='door':continue
        fl=next(x for x in graph['floors'] if x['id']==o['floor']);z=o['bbox']['center'][2];h=o['bbox']['size'][2];bottom=z-h/2
        if o['n_views']>=2 and o['geometry_pred']['score']>=.6 and abs(bottom-fl['z_floor'])<.3 and 1.7<h<2.6:
            candidates.append({'id':o['id'],'height':h,'score':o['geometry_pred']['score'],'source':'sam3_door_mask+mesh_depth_observed_extent'})
    heights=[x['height'] for x in candidates];median=float(np.median(heights)) if heights else None
    status='insufficient_evidence' if len(heights)<2 else ('consistent_with_meter' if 2.0<=median<=2.1 else 'needs_review')
    graph['scene']['scale_check']={'method':'door_height','n':len(heights),'median':median,'status':status,'measurements':candidates,'score':.6 if status=='consistent_with_meter' else .3 if heights else 0.,'source':'sam3+mesh_depth','caveat':'Observed door surface extent may be partial. This check is heuristic and does not establish an independent metric scale.'}
    if status!='consistent_with_meter':graph['issues'].append({'type':'scale_unverified','ref':'s0','detail':f'门高核查状态 {status}，有效门 {len(heights)} 个，中位数 {median}；单位继续保留 inferred_meter。','priority':.85,**base('door_height_check',.9)})


def build_texts(graph,old,registry):
    texts=[]
    # View-dependent left/right text is out of scope, even when VLM described it.
    for record in [graph['scene']]+graph['regions']:
        desc=record.get('description_pred',{})
        normalized=False
        for key in ['short','long']:
            text=desc.get(key,'')
            cleaned=re.sub(r'(?:中央|中间|中部)偏[左右]','中央附近',text)
            cleaned=re.sub(r'(?:画面)?(?:左上|右上|左下|右下)(?:方|角|侧)?','画面边缘',cleaned)
            cleaned=re.sub(r'左右两侧','两侧',cleaned)
            cleaned=re.sub(r'(?:画面)?(?:左侧|右侧|左边|右边)','一侧',cleaned)
            if cleaned!=text:desc[key]=cleaned;normalized=True
        if normalized:
            desc['source']='vlm+rule:remove_view_dependent_left_right'
            desc['raw_text_preserved_in_vlm_evidence']=True
    scene=graph['scene'];desc=scene.get('description_pred',{})
    for length in ['short','long']:
        if desc.get(length):texts.append({'level':'scene','ref':'s0','length':length,'text':desc[length],'visibility_default':'public','grounded_on':[scene['type_pred'].get('evidence','')],**base(desc.get('source','vlm'),desc.get('score',0))})
    for r in graph['regions']:
        desc=r.get('description_pred',{})
        for length in ['short','long']:
            if desc.get(length):texts.append({'level':'region','ref':r['id'],'length':length,'text':desc[length],'grounded_on':[r['panorama']], 'visibility_default':'private',**base(desc.get('source','vlm'),desc.get('score',0))})
    for o in graph['objects']:
        label=OBJECT_ZH.get(o['category_pred']['label'],o['category_pred']['label']);a=o.get('attributes_pred',{})
        parts=[str(a[k]) for k in ['color','material'] if a.get(k) and a[k]!='unknown'];name='、'.join(parts)+'的'+label if parts else label
        text=f'候选物体 {o["id"]}：{name}；位于候选区域 {o["region"]}，有 {o["n_views"]} 个关键帧支持。'
        near=[r for r in o['relations_auto'] if r['type']=='near_connection']
        if near:
            closest=min(near,key=lambda x:x['dist']);text+=f' 中心距候选连接 {closest["target"]} 约 {closest["dist"]:.1f} 个推断米制单位。'
        texts.append({'level':'object','ref':o['id'],'length':'short','text':text,'grounded_on':[o['best_view']],'visibility_default':'private',**base('template:object_fields',min(o['score'],a.get('score',o['score'])))})
    matching=stable_ids(texts,old.get('texts',[]),'t',lambda a,b:float((a['level'],a['ref'],a['length'])==(b['level'],b['ref'],b.get('length','short'))),.5,registry)
    graph['texts']=texts
    graph['issues'].sort(key=lambda x:-x.get('priority',0))
    return matching
