"""GPT semantic inventories, ordered calibrated prompt banks and per-frame QA."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from .common import ROOT,read,dump,sha,progress,fingerprint,frame_mask
from annotation_pipeline.vlm import VLM
import os,re,requests,time,cv2,numpy as np

AUDIT='''Inspect this image as a semantic audit for a general-purpose segmentation system. Return a complete inventory of visible, physically segmentable object TYPES and continuous surfaces. Scan the whole image from foreground to background and from large structures to small details. Include types that are repeated, densely arranged, partly occluded, or small but recognizable; SAM3 will find all instances of each returned type later. Return one entry per semantic type, not an instance count.
Use stable, useful semantic names that distinguish the real object's role or appearance when needed: prefer context-specific names such as product bottle, product carton, liquor jar, display shelf, plastic storage container, service counter, or metal utility cart over overly generic names such as bottle, box, jar, shelf, container, or table when the image gives clear context. Use whole physical objects, never body/lid/handle/hinge parts or color/material variants. Exclude text, logos, labels, stickers, posters, signs, decorations, shadows, reflections, and graphic patterns as object categories. Do not merge independent objects into relational names such as "A with B". Do not invent categories without visible evidence.
For each type provide a prompt bank of 1-3 short English SAM3 prompts. The first is a broad base noun phrase. Later prompts may add visible appearance or coarse context; add exclusion wording only when it helps distinguish a similar object. Do not output boxes, coordinates, counts, instance IDs, definitions, explanations, or prose.
Return JSON only: {"concepts":[{"semantic_label":"bottle","category_type":"thing","sam3_prompts":["bottle","clear bottle on display shelf","bottle not box"],"visible":true,"confidence":"high"}]}'''

ALIASES={'flooring':'floor','floor surface':'floor','walls':'wall','ceiling surface':'ceiling',
 'couch':'sofa','tv':'television','tv screen':'television','potted plant':'plant','houseplant':'plant',
 'curtains':'curtain','drapes':'curtain','dining chairs':'dining chair',
 'coffee table':'coffee table','armchair':'armchair',
 'washbasin':'sink','wash basin':'sink','bathroom sink':'sink','kitchen sink':'sink','wash basin cabinet':'vanity cabinet',
 'fridge':'refrigerator','washing_machine':'washing machine','wardrobe cabinet':'wardrobe',
 'bedside table':'nightstand','bedside cabinet':'nightstand','night stand':'nightstand',
 'light fixture':'lamp','ceiling light':'ceiling lamp','ceiling light fixture':'ceiling lamp',
 'wine bottle':'product bottle','liquor bottle':'product bottle','boxed liquor':'product carton',
 'wooden door':'door','wood door':'door','wooden cabinet':'cabinet','wood cabinet':'cabinet',
 'wooden table':'table','wooden chair':'chair','glass window':'window','glass door':'door',
 'glass sliding door':'sliding door','sliding glass door':'sliding door','glass panel door':'door',
 'wooden floor':'floor','wood floor':'floor','tiled floor':'floor','tile floor':'floor',
 'white wall':'wall','painted wall':'wall','white ceiling':'ceiling','fabric curtain':'curtain',
 'fabric sofa':'sofa','leather sofa':'sofa','metal railing':'railing',
 'interior wall':'wall','interior door':'door','frosted glass door':'door',
 'frosted glass window':'window','frosted window':'window','framed window':'window',
 'wooden tv cabinet':'tv cabinet','television cabinet':'tv cabinet','media cabinet':'tv cabinet','tv console':'tv cabinet',
 'white cabinet':'cabinet','wood-topped cabinet':'cabinet','overhead cabinet':'wall cabinet','wall-mounted cabinet':'wall cabinet',
 'wall-mounted air conditioner':'air conditioner','front-loading washing machine':'washing machine',
 'bathroom vanity cabinet':'vanity cabinet','sink cabinet':'vanity cabinet','utility sink':'sink',
 'wall switch':'light switch','sink faucet':'faucet','curtain rod':'curtain rail','curtain track':'curtain rail',
 'doorframe':'door frame','doorway frame':'door frame','wooden doorway frame':'door frame',
 'framed artwork':'artwork','framed picture':'picture','picture frame':'picture',
 'bed headboard':'headboard'}
SURFACES={'wall','floor','ceiling','ground','lawn','road','pavement','grass'}
EXCLUDED={'text','logo','label','sticker','poster','sign','decoration','shadow','reflection','pattern','handle','hinge','lid','door handle','picture','artwork','wall art','painting','door frame','window frame','cabinet door','cabinet handle','cabinet drawer','chair leg','table leg','headboard','bed frame','backrest','armrest'}
PART_SUFFIX=re.compile(r'(?:^| )(?:door(?:way)? frame|window frame|cabinet door|cabinet drawer|chair leg|table leg|headboard|armrest|backrest|handle|hinge|lid)s?$')
# Only these prompts are explicitly identified as calibrated by the reference.
CALIBRATED={'product bottle':['bottle'],'product carton':['boxed liquor packaging'],
            'liquor jar':['red cloth covered jar']}

def clean_concepts(answer):
    found={}
    for c in answer.get('concepts',[]):
        label=re.sub(r'\s+',' ',str(c.get('semantic_label','')).strip().lower().replace('_',' '))
        label=ALIASES.get(label,label)
        if not label or label in EXCLUDED or PART_SUFFIX.search(label) or c.get('visible') is not True:continue
        prompts=[str(x).strip() for x in c.get('sam3_prompts',[]) if isinstance(x,str) and 0<len(x.strip())<=90]
        entry={'semantic_label':label,'category_type':'stuff' if label in SURFACES or c.get('category_type')=='stuff' else 'thing',
               'sam3_prompts':list(dict.fromkeys(CALIBRATED.get(label,[])+prompts))[:5],
               'visible':True,'confidence':c.get('confidence','medium'),'source':'gpt_visual_inventory'}
        if not entry['sam3_prompts']:entry['sam3_prompts']=[label]
        if label not in found:found[label]=entry
    return list(found.values())

def discover(out,doc,config):
    target=out/'concepts.json'
    progress(out,'gpt_inventory',audited_frames=0,audit_total=len(doc['frames']),audit_failures=0)
    vlm=VLM(model=os.environ.get('GLOBAL_VLM_MODEL','gpt-6-sol'))
    if not vlm.model.startswith('gpt-'):raise ValueError('This pipeline requires a GPT vision model')
    selected=[]
    for clip in doc['clips']:
        ids=clip['frames'];selected.extend(ids[::config['audit_stride']])
        if ids[-1] not in selected:selected.append(ids[-1])
    byid={f['id']:f for f in doc['frames']};audits={};failures=[]
    def one(fid):
        frame=byid[fid];cache=out/'evidence/gpt_inventory'/(fid+'.json')
        prior=read(cache) if cache.exists() else {}
        if prior.get('status')=='ok' and prior.get('derived_from_audit_sheet') and prior.get('original_rgb_sha256')==frame['rgb_sha256']:
            answer=prior['parsed']
        else:answer=vlm.call(AUDIT,[out/frame['rgb']],cache,max_tokens=4500)
        concepts=clean_concepts(answer)
        if not concepts:
            data=read(cache);data.update(status='failed',schema_validation_error='Empty usable semantic inventory');dump(cache,data)
            raise RuntimeError('GPT returned an empty usable semantic inventory')
        return fid,concepts
    def many(ids):
        if len(ids)==1:return [one(ids[0])]
        tiles=[]
        for fid in ids:
            rgb=cv2.resize(cv2.imread(str(out/byid[fid]['rgb'])),(640,360));tile=np.full((390,640,3),245,np.uint8);tile[30:]=rgb
            cv2.putText(tile,fid,(10,21),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA);tiles.append(tile)
        while len(tiles)<4:tiles.append(np.full((390,640,3),245,np.uint8))
        sheet=np.concatenate([np.concatenate(tiles[:2],axis=1),np.concatenate(tiles[2:],axis=1)],axis=0)
        image_path=out/'evidence/gpt_inventory'/('audit_sheet_'+ids[0]+'.jpg');image_path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(image_path),sheet)
        prompt=AUDIT+'\nThere are '+str(len(ids))+' independent images in this contact sheet, in row-major order. Audit EACH frame separately, never combine inventories. Exact frame IDs: '+str(ids)+'. Replace the outer response schema with {"frames":[{"frame_id":"g000000","concepts":[...]}]}. All concept and prompt requirements above apply independently to each frame. No counts or boxes.'
        cache=image_path.with_suffix('.json');answer=vlm.call(prompt,[image_path],cache,max_tokens=12000)
        frames={x.get('frame_id'):x for x in answer.get('frames',[])}
        if set(frames)!=set(ids) or any(not clean_concepts(frames[fid]) for fid in ids):
            invalid=read(cache);invalid.update(status='failed',schema_validation_error='Missing frame inventories');dump(cache,invalid)
            raise RuntimeError('Incomplete batched frame inventories')
        evidence=read(cache);result=[]
        for fid in ids:
            record={**evidence,'parsed':{'concepts':frames[fid]['concepts']},'derived_from_audit_sheet':str(cache.relative_to(out)),
                    'original_rgb':byid[fid]['rgb'],'original_rgb_sha256':byid[fid]['rgb_sha256'],'frame_id':fid}
            dump(out/'evidence/gpt_inventory'/(fid+'.json'),record);result.append((fid,clean_concepts(frames[fid])))
        return result
    runtime=read(ROOT/'services/global_settings.json') if (ROOT/'services/global_settings.json').exists() else {}
    with ThreadPoolExecutor(max_workers=int(runtime.get('audit_workers',os.environ.get('GLOBAL_AUDIT_WORKERS','6')))) as pool:
        groups=[]
        for clip in doc['clips']:
            pending=[]
            for fid in clip['frames']:
                if fid not in selected:continue
                cache=out/'evidence/gpt_inventory'/(fid+'.json')
                if cache.exists() and read(cache).get('status')=='ok':groups.append([fid])
                else:pending.append(fid)
            groups.extend(pending[i:i+4] for i in range(0,len(pending),4))
        futures={pool.submit(many,ids):ids for ids in groups}
        for future in as_completed(futures):
            try:
                for fid,concepts in future.result():audits[fid]=concepts
            except Exception as exc:failures.append({'frame_ids':futures[future],'error':str(exc)})
            progress(out,'gpt_inventory',audited_frames=len(audits),audit_total=len(selected),audit_failures=len(failures))
            if len(audits)%10==0:print('GPT inventory',out.parent.name,len(audits),'/',len(selected),flush=True)
    if failures:
        dump(out/'inventory_failures.json',failures);raise RuntimeError(f'{len(failures)} inventory requests failed; resume to retry cached stages')
    # Production default audits every frame. A sparse audit is an explicit option
    # for diagnostics only, and inheritance is recorded instead of called visibility.
    banks={};mapping={}
    for clip in doc['clips']:
        sequence=clip['frames'];references=[f for f in sequence if f in audits]
        for fid in sequence:
            index=byid[fid]['source_frame_index'];nearest=sorted(references,key=lambda r:abs(byid[r]['source_frame_index']-index))[:3]
            # Clip union preserves rare categories through brief occlusion and avoids
            # inventing a visible label; SAM3 must independently find each frame mask.
            union={}
            for ref in nearest+(references if config['audit_stride']>1 else []):
                for c in audits[ref]:
                    label=c['semantic_label']
                    if label not in union:union[label]=dict(c)
                    else:union[label]['sam3_prompts']=list(dict.fromkeys(union[label]['sam3_prompts']+c['sam3_prompts']))[:5]
            concepts=sorted(union.values(),key=lambda c:c['semantic_label']);key=fingerprint(concepts)[:20]
            banks[key]=concepts;mapping[fid]={'bank':key,'audit_refs':nearest,'direct_gpt_audit':fid in audits,
                                         'visibility':'must_be_confirmed_by_per_frame_sam3'}
    result={'model':vlm.model,'inventory_stride':config['audit_stride'],'audit_frames':audits,
            'frame_prompt_banks':mapping,'prompt_banks':banks,'calibrated_reference_prompts':CALIBRATED,
            'all_frames_direct_gpt_audited':len(audits)==len(doc['frames']),
            'note':'Default: every frame is directly audited by GPT, with neighboring audit prompt variants as recall backups. Explicit sparse diagnostic runs inherit clip inventories without claiming visibility.'}
    dump(target,result);return result

def segment_one(out,frame,concepts,config,url,checkpoint):
    target=frame_mask(out,frame);meta=target.with_suffix('.json')
    signature=fingerprint({'image':frame['rgb_sha256'],'concepts':concepts,'checkpoint':checkpoint,
                           'confidence':config['sam_confidence'],'nms':config['nms_iou'],'area':config['min_mask_area']})
    if target.exists() and meta.exists() and read(meta).get('request_fingerprint')==signature:return read(meta)
    s=requests.Session();s.trust_env=False
    for attempt in range(3):
        try:
            r=s.post(url+'/segment_concepts',json={'image_path':str((out/frame['rgb']).resolve()),'output_path':str(target.resolve()),
                        'concepts':concepts,'confidence':config['sam_confidence'],'nms_iou':config['nms_iou'],'min_area':config['min_mask_area']},timeout=240)
            r.raise_for_status();result=r.json()
            if result['errors']:raise RuntimeError(str(result['errors'])[:400])
            result['request_fingerprint']=signature;dump(meta,result);return result
        except Exception:
            if attempt==2:raise
            time.sleep(2)

def segment_frames(out,doc,inventories,config):
    progress(out,'sam3',segmented_frames=0,frames_total=len(doc['frames']),raw_masks=0,exclusive_masks=0,empty_frames=0)
    urls=os.environ.get('GLOBAL_SAM_URLS','http://127.0.0.1:18763').split(',');s=requests.Session();s.trust_env=False
    runtime=read(ROOT/'services/global_settings.json') if (ROOT/'services/global_settings.json').exists() else {}
    urls=runtime.get('sam_worker_groups',{}).get(urls[0],urls)
    health=[s.get(url+'/health',timeout=10).json() for url in urls]
    if not all(h.get('ready') for h in health):raise RuntimeError('SAM3 worker not ready')
    from threading import Lock
    mutex=Lock();done=0;stats={'raw_masks':0,'exclusive_masks':0,'empty_frames':0}
    def worker(index):
        nonlocal done
        for frame in doc['frames'][index::len(urls)]:
            bank=inventories['frame_prompt_banks'][frame['id']]['bank']
            result=segment_one(out,frame,inventories['prompt_banks'][bank],config,urls[index],health[index]['checkpoint_sha256'])
            with mutex:
                done+=1;stats['raw_masks']+=len(result['raw_detections']);stats['exclusive_masks']+=len(result['detections']);stats['empty_frames']+=not bool(result['detections'])
                if done%10==0 or done==len(doc['frames']):
                    progress(out,'sam3',segmented_frames=done,frames_total=len(doc['frames']),**stats)
                    print('SAM3 dense',out.parent.name,done,'/',len(doc['frames']),flush=True)
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        for future in as_completed([pool.submit(worker,i) for i in range(len(urls))]):future.result()
    dump(out/'segmentation_summary.json',{'frames':len(doc['frames']),**stats,'workers':health,'all_frames_segmented':True})
