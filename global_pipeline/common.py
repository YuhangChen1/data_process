"""Versioned global-instance outputs, independent of the v0.1 object candidates."""
from annotation_pipeline.common import ROOT,read,dump,sha,utc,base,stable_ids,aabb_iou
from pathlib import Path
import hashlib,json,os
import numpy as np
import cv2

VERSION='global_instances/0.2'
DEFAULTS={'voxel_size':.04,'knn':8,'edge_radius':.13,'spatial_weight':1.,'color_weight':.35,
          'fh_k':.14,'superpoint_min_size':6,'superpoint_max_size':300,'superpoint_max_extent':.65,
          'sam_confidence':.40,'nms_iou':.80,'min_mask_area':80,
          'depth_abs_tolerance':.06,'depth_relative_tolerance':.02,'proposal_min_points':20,
          'proposal_superpoint_fraction':.20,'merge_iou':.25,'merge_iomin':.65,
          'hard_quality':.90,'hard_score':.80,'fine_unary':.70,'fine_neighbor':.30,
          'fine_min_score':.50,'fine_gain':.08,'fine_min_views':2,
          'global_min_points':20,'global_purity':.65,'semantic_margin':.15,
          'semantic_min_views':2,'audit_stride':1,'description_batch':12,'frame_stride':1}

def destination(scene):return ROOT/'auto'/scene/'global_instances'
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def colors(ids):
    ids=np.asarray(ids,dtype=np.uint32);v=ids*2654435761
    out=np.stack([60+(v&255)*195//255,60+((v>>8)&255)*195//255,60+((v>>16)&255)*195//255],-1).astype(np.uint8)
    out[ids==0]=0;return out

def write_csv(path,rows,fields):
    import csv
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def progress(out,stage,**values):
    path=out/'run_info.json';info=read(path) if path.exists() else {}
    if info.get('error'):
        info.setdefault('recovered_failures',[]).append({key:info.pop(key) for key in ['error','traceback','failed_at'] if key in info})
    info.update(pipeline=VERSION,scene_id=out.parent.name,status='running',stage=stage,updated_at=utc())
    info.setdefault('progress',{}).update(values);dump(path,info)

def frame_mask(out,frame):return out/'masks'/frame['clip_id']/(frame['id']+'.npz')
def frame_projection(out,frame):return out/'projections'/frame['clip_id']/(frame['id']+'.npz')
