"""Loopback-only persistent SAM3 image segmentation service."""
from pathlib import Path
import os,sys,time,json,threading,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.deps/sam3'),str(ROOT/'.deps/python')]
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

app=FastAPI(title='Local SAM3 annotation service');lock=threading.Lock();model=None;processor=None;started=time.time();requests_done=0;checkpoint_sha256=None;code_sha256=None
def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
class Segment(BaseModel):
    image_path:str
    prompts:list[str]
    output_path:str
    confidence:float=.45
    max_per_prompt:int=6

class ConceptSegment(BaseModel):
    image_path:str
    output_path:str
    concepts:list[dict]
    confidence:float=.40
    nms_iou:float=.80
    min_area:int=80

@app.on_event('startup')
def load():
    global model,processor,checkpoint_sha256,code_sha256
    torch.set_num_threads(4)
    checkpoint_sha256=file_sha(ROOT/'model/sam3.pt');code_sha256=file_sha(ROOT/'.deps/sam3/sam3/model_builder.py')
    model=build_sam3_image_model(checkpoint_path=str(ROOT/'model/sam3.pt'),load_from_HF=False,device='cuda',enable_inst_interactivity=False)
    processor=Sam3Processor(model,device='cuda',confidence_threshold=.45)
    print('SAM3 resident ready',os.getpid(),flush=True)

@app.get('/health')
def health():
    return {'service':'sam3_annotation','ready':model is not None,'pid':os.getpid(),'checkpoint':str(ROOT/'model/sam3.pt'),'checkpoint_sha256':checkpoint_sha256,'sam3_code_sha256':code_sha256,'device':torch.cuda.get_device_name(0),'requests_done':requests_done,'uptime_seconds':time.time()-started}

def inside(path):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT):raise HTTPException(400,'Path outside workspace')
    return p

@app.post('/segment')
def segment(req:Segment):
    global requests_done
    imgpath=inside(req.image_path);out=inside(req.output_path);out.parent.mkdir(parents=True,exist_ok=True)
    started_req=time.time();detections=[];all_masks=[];errors=[]
    with lock,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        processor.set_confidence_threshold(req.confidence)
        image=Image.open(imgpath).convert('RGB');state=processor.set_image(image)
        for prompt in req.prompts:
            try:
                processor.reset_all_prompts(state);result=processor.set_text_prompt(prompt,state)
                scores=result['scores'].float().cpu().numpy();order=np.argsort(-scores)[:req.max_per_prompt]
                masks=result['masks'].cpu().numpy().reshape(-1,image.height,image.width)
                boxes=result['boxes'].float().cpu().numpy()
                for index in order:
                    mask=masks[index].astype(bool);area=int(mask.sum())
                    if area<100:continue
                    detections.append({'label':prompt,'score':float(scores[index]),'source':'sam3:text_prompt','mask_index':len(all_masks),'bbox_xyxy':boxes[index].tolist(),'mask_area':area})
                    all_masks.append(mask)
            except Exception as exc:errors.append({'prompt':prompt,'error':type(exc).__name__+': '+str(exc)[:200]})
        del state;requests_done+=1
    masks=np.stack(all_masks) if all_masks else np.empty((0,image.height,image.width),bool)
    np.savez_compressed(out,masks=masks)
    data={'image_path':str(imgpath),'image_sha256':file_sha(imgpath),'mask_path':str(out),'prompts':req.prompts,'detections':detections,'errors':errors,'seconds':time.time()-started_req,'confidence':req.confidence,'checkpoint_sha256':checkpoint_sha256,'sam3_code_sha256':code_sha256}
    out.with_suffix('.json').write_text(json.dumps(data,indent=2)+'\n')
    return data

@app.post('/segment_concepts')
def segment_concepts(req:ConceptSegment):
    """One embedding, ordered fallback prompts, same-type NMS, exclusive pixels.

    Grounding SAM3 exposes detection scores, not a predicted mask IoU. Stability
    is measured from probabilities at .45/.55 and is never called an IoU score.
    """
    global requests_done
    from annotation_pipeline.common import dump
    imgpath=inside(req.image_path);out=inside(req.output_path);out.parent.mkdir(parents=True,exist_ok=True)
    start=time.time();raw=[];records=[];attempts=[];errors=[]
    with lock,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        processor.set_confidence_threshold(req.confidence)
        image=Image.open(imgpath).convert('RGB');state=processor.set_image(image)
        for concept in req.concepts:
            name=concept['semantic_label'];chosen=[]
            for prompt in dict.fromkeys(concept.get('sam3_prompts',[])[:5]):
                try:
                    processor.reset_all_prompts(state);result=processor.set_text_prompt(prompt,state)
                    scores=result['scores'].float().cpu().numpy()
                    masks=result['masks'].cpu().numpy().reshape(-1,image.height,image.width)
                    probabilities=result['masks_logits']
                    stability=((probabilities>.55).flatten(1).sum(1)/(probabilities>.45).flatten(1).sum(1).clamp(min=1)).float().cpu().numpy()
                    good=[int(i) for i in np.argsort(-scores) if int(masks[i].sum())>=req.min_area]
                    attempts.append({'semantic_label':name,'prompt':prompt,'masks':len(good)})
                    if good:
                        for i in good:
                            mask=masks[i].astype(bool);index=len(raw);raw.append(mask)
                            record={'semantic_label':name,'category_type':concept.get('category_type','thing'),
                                    'prompt':prompt,'score':float(scores[i]),'stability_score':float(stability[i]),
                                    'sam_iou':None,'sam_iou_available':False,'raw_mask_index':index,
                                    'raw_area':int(mask.sum()),'source':'sam3:text_grounding'}
                            duplicate=any(np.count_nonzero(mask&raw[r['raw_mask_index']])/max(np.count_nonzero(mask|raw[r['raw_mask_index']]),1)>=req.nms_iou for r in chosen)
                            record['kept_after_same_category_nms']=not duplicate;records.append(record)
                            if not duplicate:chosen.append(record)
                        break
                except Exception as exc:errors.append({'semantic_label':name,'prompt':prompt,'error':type(exc).__name__+': '+str(exc)[:160]})
        del state;requests_done+=1
    owner=np.zeros((image.height,image.width),np.uint16);kept=[]
    for record in sorted((r for r in records if r['kept_after_same_category_nms']),key=lambda r:-r['score']):
        mask=raw[record['raw_mask_index']]&(owner==0);area=int(mask.sum())
        if area<req.min_area:record['exclusive_local_id']=0;continue
        local_id=len(kept)+1;owner[mask]=local_id;record['exclusive_local_id']=local_id
        record['exclusive_area']=area;kept.append(record.copy())
    stack=np.stack(raw) if raw else np.empty((0,image.height,image.width),bool)
    np.savez_compressed(out,raw_masks=stack,local_instances=owner)
    data={'image_path':str(imgpath),'image_sha256':file_sha(imgpath),'mask_path':str(out),
          'checkpoint_sha256':checkpoint_sha256,'sam3_code_sha256':code_sha256,
          'concepts':req.concepts,'detections':kept,'raw_detections':records,'prompt_attempts':attempts,
          'errors':errors,'confidence':req.confidence,'nms_iou':req.nms_iou,'min_area':req.min_area,
          'seconds':time.time()-start,'quality_note':'No predicted mask IoU exposed by text-grounding SAM3. stability_score = area(p>.55)/area(p>.45); it is not a predicted IoU.'}
    dump(out.with_suffix('.json'),data)
    return data

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=int(os.environ.get('SAM3_PORT','18763')),access_log=False)
