"""GPT-only frame observations (level 1) and within-trajectory changes (level 2)."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from .common import ROOT,read,dump,progress,utc,sha
from annotation_pipeline.vlm import VLM
import numpy as np,cv2,os

PROMPT='''You are annotating indoor navigation imagery, not writing an evaluation route. The contact sheet has four consecutive timestamped RGB frames, each clearly labeled with its frame ID. Return JSON only:
{"frames":[{"frame_id":"g000000","description":"中文：仅描述该帧可见的物体、连续表面与空间布置，1到3句","score":0.0,"uncertainties":["中文"]}],"changes":[{"from_frame":"g000000","to_frame":"g000001","description":"中文：对比两帧中实际可见的变化，1到2句","change_type":"camera_view_change|appearance_change|no_clear_change","score":0.0,"uncertainties":["中文"]}]}
Return one frame entry for every supplied frame ID, and one change entry for every adjacent pair within this sheet. These are rendered views of a STATIC reconstructed scene: camera motion, occlusion or visibility changes do not prove that an object moved. Never claim an object physically moved, disappeared from the world, was added/removed, or a door opened/closed without visual temporal evidence. Describe entering/leaving the field of view, exposure of surfaces, viewpoint/scale changes or no clear change. Do not invent materials, identities, compass directions, unseen rooms, exact physical sizes or left/right relational labels. Continuous surfaces such as walls, floor and ceiling may be described. Global ID hints are geometric candidates; report visible evidence, do not assert uncertain hints as truth. Scores are heuristic in [0,1]. Do not create evaluation questions, route instructions or decisions about public/private access. Use exact zero-padded frame IDs from the supplied list.
Frame/global ID hints: '''

COMPACT='''Annotate these consecutive frames of a STATIC indoor reconstruction, reading the contact sheet in row-major order. Return JSON only with compact array rows:
{"frames":[["g000000","中文单帧描述",0.8]],"changes":[["g000000","g000001","中文相邻帧变化","camera_view_change",0.8]]}
Each frame row is [exact_frame_id,text,score]; each change row is [exact_from_id,exact_to_id,text,change_type,score]. Include EVERY supplied frame and EVERY consecutive pair, preserving zero-padded IDs. One short specific Chinese sentence per row: frame text 25-45 Chinese characters about visible objects/surfaces/layout; change text 12-25 Chinese characters about actual visible differences. Do not repeat generic filler. If almost unchanged, say so with change_type no_clear_change. Other change types: camera_view_change,appearance_change. Scores are heuristic [0,1].
These are static rendered views: changes in camera position, scale, visibility, occlusion or mask support do not prove real object motion, world disappearance, additions/removals or door opening/closing. Do not invent identities, exact physical sizes, materials, compass directions, unseen rooms, left/right relations, evaluation questions, route instructions or public/private decisions. Acknowledge uncertainty within text when needed. Geometric global ID hints are candidates, not truth. Small details can remain unknown.
Frame/global ID hints: '''

def describe(out,doc,config):
    progress(out,'gpt_descriptions',described_frames=0,description_changes=0,description_failures=0,
             frames_total=len(doc['frames']))
    vlm=VLM(model=os.environ.get('GLOBAL_VLM_MODEL','gpt-6-sol'))
    if not vlm.model.startswith('gpt-'):raise ValueError('GPT model required')
    assignments=read(out/'frame_assignments.json')['frames'];instances=read(out/'global_instances.json')['instances']
    names={o['global_id']:o['semantic_label'] for o in instances};byid={f['id']:f for f in doc['frames']}
    batches=[]
    # Reuse already successful four-frame evidence, then cover pending adjacent
    # pairs with larger sheets. This reduces gateway calls during recovery.
    for clip in doc['clips']:
        ids=clip['frames'];covered=set();covered_frames=set();start=0
        while start<len(ids):
            group=ids[start:start+4]
            if len(group)==1 and start:break
            cache=out/'evidence/gpt_descriptions'/('sheet_'+group[0]+'.json')
            if cache.exists():
                prior=read(cache);answer=prior.get('parsed',{})
                available={x.get('frame_id') for x in answer.get('frames',[])}
                pairs={(x.get('from_frame'),x.get('to_frame')) for x in answer.get('changes',[])}
                if prior.get('status')=='ok' and set(group)<=available and set(zip(group[:-1],group[1:]))<=pairs:
                    batches.append((clip['id'],group));covered_frames.update(group);covered.update(zip(group[:-1],group[1:]))
            if start+4>=len(ids):break
            start+=3
        start=0
        while start<len(ids)-1:
            if (ids[start],ids[start+1]) in covered:start+=1;continue
            group=ids[start:start+config['description_batch']];batches.append((clip['id'],group))
            covered_frames.update(group);covered.update(zip(group[:-1],group[1:]));start+=len(group)-1
        if len(ids)==1 and ids[0] not in covered_frames:batches.append((clip['id'],ids))
    observations={};changes={};failures=[]
    def one(clip,ids):
        tiles=[];hints=[]
        for fid in ids:
            frame=byid[fid];image=cv2.imread(str(out/frame['rgb']));image=cv2.resize(image,(640,360))
            tile=np.full((390,640,3),245,np.uint8);tile[30:]=image
            cv2.putText(tile,f'{fid}  t={frame["timestamp"]:.3f}s',(10,21),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA);tiles.append(tile)
            hints.append({'frame_id':fid,'timestamp':frame['timestamp'],
                'observed_global_instances':[{'global_id':a['global_id'],'label':names.get(a['global_id'],'unknown'),'purity':round(a['purity'],2)} for a in assignments[fid] if a['accepted']]})
        columns=2 if len(ids)<=4 else 3;rows=max(2,int(np.ceil(len(tiles)/columns)))
        while len(tiles)<columns*rows:tiles.append(np.full((390,640,3),245,np.uint8))
        collage=np.concatenate([np.concatenate(tiles[row*columns:(row+1)*columns],axis=1) for row in range(rows)],axis=0)
        suffix='' if len(ids)<=4 else '_n'+str(len(ids))
        image_path=out/'evidence/gpt_descriptions'/('sheet_'+ids[0]+suffix+'.jpg');image_path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(image_path),collage)
        cache=image_path.with_suffix('.json');prompt=(PROMPT if len(ids)<=4 else COMPACT)+__import__('json').dumps(hints,ensure_ascii=False)
        prior=read(cache) if cache.exists() else {}
        if len(ids)>4 and prior.get('status')=='ok' and prior.get('images',[{}])[0].get('sha256')==sha(image_path):
            answer=prior['parsed']
        else:answer=vlm.call(prompt,[image_path],cache,max_tokens=5000 if len(ids)>4 else 5500)
        if answer.get('frames') and isinstance(answer['frames'][0],list):
            answer={'frames':[{'frame_id':x[0],'description':x[1],'score':x[2]} for x in answer['frames']],
                    'changes':[{'from_frame':x[0],'to_frame':x[1],'description':x[2],'change_type':x[3],'score':x[4]} for x in answer.get('changes',[])]}
        frames={x.get('frame_id'):x for x in answer.get('frames',[])}
        pairs={(x.get('from_frame'),x.get('to_frame')):x for x in answer.get('changes',[])}
        # Normalize zero-padding only, retaining exact numerical frame identity.
        import re
        normalize=lambda x:'g'+str(int(str(x)[1:])).zfill(6) if re.fullmatch(r'g\d+',str(x)) else str(x)
        frames={normalize(k):v for k,v in frames.items()};pairs={(normalize(a),normalize(b)):v for (a,b),v in pairs.items()}
        if (any(fid not in frames or not str(frames[fid].get('description','')).strip() for fid in ids)
                or any(pair not in pairs or not str(pairs[pair].get('description','')).strip() for pair in zip(ids[:-1],ids[1:]))):
            invalid=read(cache);invalid.update(status='failed',schema_validation_error='Missing frame/change IDs or empty descriptions');dump(cache,invalid)
            raise RuntimeError('GPT omitted requested frame/change IDs: '+ids[0])
        evidence=str(cache.relative_to(out));actual_model=read(cache).get('actual_model',vlm.model);result_frames=[];result_changes=[]
        for fid in ids:
            item=frames[fid];text=str(item.get('description','')).strip()
            if not text:raise RuntimeError('Empty frame description '+fid)
            result_frames.append({'id':'l1_'+fid,'level':1,'frame_id':fid,'clip_id':clip,
                'timestamp':byid[fid]['timestamp'],'text':text,
                'visible_global_ids':sorted({a['global_id'] for a in assignments[fid] if a['accepted']}),
                'score':float(np.clip(float(item.get('score') or 0),0,1)),'source':'gpt:'+actual_model,'evidence':evidence,
                'grounded_on':[byid[fid]['rgb']],'uncertainties':item.get('uncertainties',[]),
                'review':{'status':'unreviewed'},'visibility_default':'private'})
        for a,b in zip(ids[:-1],ids[1:]):
            item=pairs[(a,b)];before={x['global_id'] for x in assignments[a] if x['accepted']};after={x['global_id'] for x in assignments[b] if x['accepted']}
            result_changes.append({'id':'l2_'+a+'_'+b,'level':2,'from_frame':a,'to_frame':b,'clip_id':clip,
                'text':str(item.get('description','')),'change_type':item.get('change_type','camera_view_change'),
                'global_ids_entering_mask_support':sorted(after-before),'global_ids_leaving_mask_support':sorted(before-after),
                'global_ids_retained':sorted(before&after),'support_change_caveat':'Detection support changes are not evidence that a physical object moved.',
                'score':float(np.clip(float(item.get('score') or 0),0,1)),'source':'gpt:'+actual_model,'evidence':evidence,
                'grounded_on':[byid[a]['rgb'],byid[b]['rgb']],'uncertainties':item.get('uncertainties',[]),
                'review':{'status':'unreviewed'},'visibility_default':'private'})
        return result_frames,result_changes
    runtime=read(ROOT/'services/global_settings.json') if (ROOT/'services/global_settings.json').exists() else {}
    with ThreadPoolExecutor(max_workers=int(runtime.get('description_workers',os.environ.get('GLOBAL_DESCRIPTION_WORKERS','6')))) as pool:
        pending={pool.submit(one,clip,ids):(clip,ids) for clip,ids in batches}
        for future in as_completed(pending):
            try:
                frames,change=future.result()
                for f in frames:observations.setdefault(f['frame_id'],f)
                for c in change:changes[(c['from_frame'],c['to_frame'])]=c
            except Exception as exc:failures.append({'clip_id':pending[future][0],'frame_ids':pending[future][1],'error':str(exc)})
            progress(out,'gpt_descriptions',described_frames=len(observations),description_changes=len(changes),description_failures=len(failures))
    dump(out/'descriptions.json',{'schema':'kosmoworld.frame_descriptions','schema_version':'0.2','scene_id':out.parent.name,
        'level1':[observations[k] for k in sorted(observations)],'level2':[changes[k] for k in sorted(changes)],
        'model':vlm.model,'failures':failures,'temporal_scope':'Only adjacent frames within a continuous trajectory clip. No descriptions across cuts or floors.'})
    if failures:raise RuntimeError(f'{len(failures)} GPT description batches failed; rerun describe to retry')
    print('GPT descriptions',out.parent.name,len(observations),'frames',len(changes),'transitions',flush=True)
