from concurrent.futures import ThreadPoolExecutor,as_completed
from .common import ROOT,read,dump,base,score,VOCAB,REGION_TYPES,NAMES,OBJECT_ZH,rel
import json,cv2,numpy as np,os,re

def annotate_regions(graph,out,vlm):
    def one(region):
        types=json.dumps(REGION_TYPES);vocab=json.dumps(VOCAB)
        prompt=f'''Classify the functional area at the camera center in this residential panorama. A watershed patch may cover mixed functions, and visible neighboring rooms are not necessarily this area. Allowed region types: {types}. Allowed object categories: {vocab}.
Return JSON exactly shaped as {{"type_pred":{{"label":"one allowed type","score":0.0,"top3":[["type1",0.0],["type2",0.0],["type3",0.0]]}},"name_auto":"简短中文名称","description_short":"一句中文，可见证据","description_long":"2到4句中文，仅描述可见布局与物体","description_score":0.0,"visible_object_categories":["allowed category"],"uncertainties":["中文说明"]}}. Do not mention left/right relationships, compass directions, unobserved doors/rooms, identities or precise dimensions. Prefer unknown/mixed for ambiguous views. Scores in [0,1]. Region area is {region['area']:.2f} inferred square meters; this is navigable patch area, not full room area.'''
        cache=out/'evidence/vlm'/f'region_{region["id"]}.json'
        try:
            answer=vlm.call(prompt,[out/region['panorama']],cache)
            pred=answer.get('type_pred',{});label=pred.get('label','unknown');label=label if label in REGION_TYPES else 'unknown'
            top=pred.get('top3',[]);top=[[x[0],score(x[1])] for x in top if isinstance(x,list) and len(x)==2 and x[0] in REGION_TYPES]
            region['type_pred']={'label':label,'score':score(pred.get('score')),'top3':top[:3],'source':'vlm:'+vlm.model,'evidence':rel(cache,out)}
            region['name_auto']=str(answer.get('name_auto') or NAMES[label])[:80]
            region['visible_object_categories']=[c for c in answer.get('visible_object_categories',[]) if c in VOCAB]
            region['semantic_uncertainties']=answer.get('uncertainties',[])
            region['description_pred']={'short':str(answer.get('description_short','')),'long':str(answer.get('description_long','')),'score':score(answer.get('description_score',pred.get('score'))),'source':'vlm:'+vlm.model}
        except Exception as exc:
            region['type_pred']={'label':'unknown','score':0.,'top3':[],'source':'vlm:failed','evidence':rel(cache,out)};region['name_auto']=NAMES['unknown'];region['visible_object_categories']=[]
            region['description_pred']={'short':'','long':'','score':0.,'source':'vlm:failed'}
            graph['issues'].append({'type':'vlm_failure','ref':region['id'],'detail':str(exc),'priority':1.,**base('pipeline',1.)})
        if region['type_pred']['score']<.6:
            graph['issues'].append({'type':'low_confidence','ref':region['id'],'detail':f'区域类型置信度 {region["type_pred"]["score"]:.2f}，需人工复核。','priority':.9,**base('vlm',1-region['type_pred']['score'])})
        return region['id'],region['type_pred']['label']
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,r) for r in graph['regions']]
        for f in as_completed(futures):print('Region VLM',graph['scene_id'],*f.result(),flush=True)


def annotate_attributes(graph,out,vlm,only_missing=False):
    objects=graph['objects'];groups=[objects[i:i+5] for i in range(0,len(objects),5)]
    if only_missing:groups=[group for group in groups if any(o.get('attributes_pred',{}).get('source')=='vlm:failed' for o in group)]
    selected={o['id'] for group in groups for o in group}
    graph['issues']=[issue for issue in graph['issues'] if not (
        issue['type'] in ['category_disagreement','vlm_failure','missing_vlm_object_result'] and
        set(issue['ref'] if isinstance(issue['ref'],list) else [issue['ref']]) <= selected)]
    def batch(group):
        tiles=[]
        for ob in group:
            im=cv2.imread(str(out/ob['best_view']));h,w=im.shape[:2];scale=min(350/w,280/h);im=cv2.resize(im,(max(1,int(w*scale)),max(1,int(h*scale))))
            tile=np.full((320,370,3),245,np.uint8);tile[35:35+im.shape[0],10:10+im.shape[1]]=im
            cv2.putText(tile,f'{ob["id"]} | {ob["category_pred"]["label"]}',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1,cv2.LINE_AA);tiles.append(tile)
        collage=np.concatenate(tiles,axis=0 if len(tiles)<=2 else 1)
        evidence=out/'evidence/object_views'/('attributes_'+group[0]['id']+'.jpg');cv2.imwrite(str(evidence),collage)
        prompt='''Inspect each labeled object crop in this contact sheet. The predicted category is a proposal, not ground truth. Describe ONLY the highlighted/central target object; adjacent crop context is not part of it. Return JSON {"objects":[{"id":"o...","category_confirmed":true,"category_alternative":"allowed category or unknown","category_score":0.0,"color":"中文或unknown","material":"中文或unknown","attributes_score":0.0,"note":"中文可见证据/不确定性"}]}. Do not infer material from category alone; uncertain material = unknown. No compass directions or physical dimensions. Allowed categories: '''+json.dumps(VOCAB)
        cache=out/'evidence/vlm'/('attributes_'+group[0]['id']+'.json')
        try:
            result=vlm.call(prompt,[evidence],cache,max_tokens=3500)
            # Some valid responses omit the zero padding printed in crop labels.
            # Normalize only the numeric representation, never infer a different object.
            def canonical_id(value):
                value=str(value or '').strip();match=re.fullmatch(r'[oO](\d+)',value)
                return 'o'+str(int(match[1])).zfill(3) if match else value
            byid={canonical_id(x.get('id')):x for x in result.get('objects',[])}
        except Exception as exc:
            byid={};graph['issues'].append({'type':'vlm_failure','ref':[x['id'] for x in group],'detail':str(exc),'priority':1.,**base('pipeline',1.)})
        for ob in group:
            a=byid.get(ob['id'],{});ob['attributes_pred']={'color':str(a.get('color','unknown')),'material':str(a.get('material','unknown')),'score':score(a.get('attributes_score',0)),'source':'vlm:'+vlm.model if a else 'vlm:failed','evidence':rel(cache,out)}
            ob['category_vlm_check']={'confirmed':a.get('category_confirmed'), 'alternative':a.get('category_alternative','unknown'),'score':score(a.get('category_score',0)),'source':'vlm:'+vlm.model if a else 'vlm:failed','note':str(a.get('note',''))}
            if a.get('category_confirmed') is False:
                graph['issues'].append({'type':'category_disagreement','ref':ob['id'],'detail':f'SAM3 类别 {ob["category_pred"]["label"]}；VLM 建议 {a.get("category_alternative","unknown")}。','priority':.95,**base('sam3+vlm',score(a.get('category_score',0)))})
        print('Attributes VLM',graph['scene_id'],group[0]['id'],len(group),flush=True)
    with ThreadPoolExecutor(max_workers=int(os.environ.get('VLM_WORKERS','4'))) as pool:
        for future in as_completed([pool.submit(batch,g) for g in groups]):future.result()


def annotate_scene(graph,out,vlm):
    graph['issues']=[x for x in graph['issues'] if not (x['type']=='vlm_failure' and x['ref']=='s0')]
    summaries=[{'id':r['id'],'type':r['type_pred'],'description':r.get('description_pred',{}).get('short','')} for r in graph['regions']]
    prompt='''Summarize this reconstructed residence from region observations plus the first floor overhead image. Region patches may split a single room, so do NOT count patches as bedrooms/rooms. No left/right relationships or compass directions; reconstructed +X/+Y axes are not cardinal directions. Do not invent unseen floor connections. Return JSON {"type_pred":{"label":"apartment|house|duplex|unknown","score":0.0},"description_short":"一句中文","description_long":"2到4句中文","description_score":0.0,"uncertainties":["中文"]}. Physical units are inferred meters, not verified. Candidate floor count: '''+str(len(graph['floors']))+'. Region observations: '+json.dumps(summaries,ensure_ascii=False)
    path=ROOT/'outputs'/graph['scene_id']/'floor_0/topdown_3dgs.png';cache=out/'evidence/vlm/scene.json'
    try:
        a=vlm.call(prompt,[path],cache);pred=a.get('type_pred',{});graph['scene']['type_pred']={'label':pred.get('label','unknown'),'score':score(pred.get('score')),'source':'vlm:'+vlm.model,'evidence':rel(cache,out)}
        graph['scene']['description_pred']={'short':str(a.get('description_short','')),'long':str(a.get('description_long','')),'score':score(a.get('description_score')),'source':'vlm:'+vlm.model}
        graph['scene']['uncertainties']=a.get('uncertainties',[]);graph['scene']['score']=graph['scene']['type_pred']['score']
    except Exception as exc:
        graph['scene']['type_pred']={'label':'unknown','score':0.,'source':'vlm:failed'};graph['scene']['description_pred']={'short':'','long':'','score':0.,'source':'vlm:failed'}
        graph['issues'].append({'type':'vlm_failure','ref':'s0','detail':str(exc),'priority':1.,**base('pipeline',1.)})
