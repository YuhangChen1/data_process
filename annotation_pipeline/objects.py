from pathlib import Path
from .common import ROOT,read,dump,rel,base,score,VOCAB,aabb_iou,stable_ids,sha
import numpy as np,cv2,requests,json,math,time
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
import open3d as o3d
from functools import lru_cache

SAM_URL='http://127.0.0.1:18763'

@lru_cache(maxsize=2)
def checkpoint_hash(modified_ns,size):
    return sha(ROOT/'model/sam3.pt')

def model_signature():
    stat=(ROOT/'model/sam3.pt').stat()
    return {'checkpoint_sha256':checkpoint_hash(stat.st_mtime_ns,stat.st_size),
            'sam3_code_sha256':sha(ROOT/'.deps/sam3/sam3/model_builder.py')}

def extract_keyframes(graph,out):
    evidence=out/'evidence/keyframes';evidence.mkdir(parents=True,exist_ok=True);records=[]
    for fl in graph['floors']:
        folder=ROOT/'outputs'/graph['scene_id']/f'floor_{int(fl["id"][1:])}'
        for clip in sorted((folder/'annotation_walkthrough').glob('component_*')):
            meta=read(clip/'metadata.json');data=np.load(clip/'cameras.npz');poses=data['camera_to_canonical'];K=data['K'];N=len(poses)
            select=set([0,N-1]+list(range(0,N,16)))
            # Force a representative frame near each region center on this component.
            for r in graph['regions']:
                if r['floor']==fl['id'] and r['component']==meta['component']:select.add(int(np.argmin(np.linalg.norm(poses[:,:2,3]-r['center'],axis=1))))
            if np.linalg.norm(np.ptp(poses[:,:2,3],axis=0))<.2:select.update([N//4,N//2,N*3//4])
            cap=cv2.VideoCapture(str(clip/'rgb.mp4'));i=0
            while True:
                okay,image=cap.read()
                if not okay:break
                if i in select:
                    kid=f'k_{fl["id"]}_{clip.name}_{i:05d}';path=evidence/(kid+'.jpg')
                    cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,91])
                    depth=clip/'mesh_depth_mm'/f'{i:05d}.png'
                    records.append({'id':kid,'floor':fl['id'],'component':meta['component'],'image_path':rel(path,out),'depth_path':rel(depth,out),'depth_encoding':'uint16_millimeter','camera_to_canonical':poses[i].tolist(),'K':K.tolist(),'frame_index':i,'timestamp':float(data['timestamps'][i]),'source_video':rel(clip/'rgb.mp4',out),'source':'annotation_walkthrough','purpose':'annotation_only_not_evaluation_Q','score':1.})
                i+=1
            cap.release()
            if i!=N:raise RuntimeError(f'Video decoded {i}/{N} frames: {clip}')
        # Already-rendered orthogonal observations improve angular coverage.
        for npz in sorted((folder/'alignment_views').glob('view_*.npz')):
            dat=np.load(npz);panel=cv2.imread(str(npz.with_suffix('.jpg')));height,width=dat['mesh_depth'].shape
            kid=f'k_{fl["id"]}_aux_{npz.stem}';path=evidence/(kid+'.jpg');depth=evidence/(kid+'_depth.npy')
            cv2.imwrite(str(path),panel[28:height,:width],[cv2.IMWRITE_JPEG_QUALITY,91])
            np.save(depth,dat['mesh_depth'][28:].astype(np.float32))
            intrinsic=dat['K'].copy();intrinsic[1,2]-=28
            records.append({'id':kid,'floor':fl['id'],'image_path':rel(path,out),'depth_path':rel(depth,out),'depth_encoding':'float32_inferred_meter','camera_to_canonical':dat['camera_to_canonical'].tolist(),'K':intrinsic.tolist(),'source':'stage2_same_camera_validation_supplement','source_npz':rel(npz,out),'crop_original_xyxy':[0,28,width,height],'purpose':'annotation_only_not_evaluation_Q','score':1.})
    dump(out/'evidence/keyframes.json',{'frame':'canonical','units':'inferred_meter','pixel_convention':'OpenCV +X right +Y down +Z forward; rays at pixel centers (u+0.5,v+0.5)','keyframes':records})
    return records


def segment_frame(frame,graph,out):
    # Scene-level vocabulary proposed by panorama VLM, plus core fixed landmarks.
    proposed={x for r in graph['regions'] for x in r.get('visible_object_categories',[])}
    prompts=[x for x in VOCAB if x in proposed or x in ['door','window','sofa','bed','table','chair','cabinet','sink','toilet']]
    target=out/'evidence/masks'/(frame['id']+'.npz');meta=target.with_suffix('.json')
    image_sha=sha(out/frame['image_path'])
    signature=model_signature()
    if meta.exists() and target.exists():
        result=read(meta)
        if result.get('prompts')==prompts and result.get('confidence')==.45 and result.get('image_sha256')==image_sha and all(result.get(k)==v for k,v in signature.items()):return result
    session=requests.Session();session.trust_env=False
    response=session.post(SAM_URL+'/segment',json={'image_path':str((out/frame['image_path']).resolve()),'output_path':str(target.resolve()),'prompts':prompts,'confidence':.45,'max_per_prompt':6},timeout=180)
    response.raise_for_status();result=response.json();result['image_sha256']=image_sha;dump(meta,result);return result


def project_detection(frame,detection,masks,out):
    mask=masks[detection['mask_index']].astype(bool)
    # Remove thin boundary pixels where ray-depth discontinuities cause background leakage.
    mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    depth_path=out/frame['depth_path'];depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 if frame['depth_encoding']=='uint16_millimeter' else np.load(depth_path)
    original=int(mask.sum());valid=mask&np.isfinite(depth)&(depth>.12)&(depth<15)
    frac=float(valid.sum()/max(original,1))
    if valid.sum()<50 or frac<.25:return None,'insufficient_mesh_depth'
    gradx=np.abs(np.diff(depth,axis=1,prepend=depth[:,:1]));grady=np.abs(np.diff(depth,axis=0,prepend=depth[:1]));valid&=(gradx<.10+depth*.03)&(grady<.10+depth*.03)
    y,x=np.where(valid)
    if len(x)<50:return None,'insufficient_depth_interior'
    if len(x)>3500:
        idx=np.linspace(0,len(x)-1,3500).astype(int);x=x[idx];y=y[idx]
    z=depth[y,x];K=np.array(frame['K']);camera=np.c_[(x+.5-K[0,2])/K[0,0]*z,(y+.5-K[1,2])/K[1,1]*z,z]
    pose=np.array(frame['camera_to_canonical']);points=camera@pose[:3,:3].T+pose[:3,3]
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points));cloud=cloud.voxel_down_sample(.035);points=np.asarray(cloud.points)
    if len(points)<25:return None,'insufficient_3d_points'
    # Keep the dominant spatial component instead of bridging leaked background.
    labels=np.asarray(cloud.cluster_dbscan(eps=.14,min_points=5,print_progress=False))
    if (labels>=0).any():
        counts=np.bincount(labels[labels>=0]);label=int(np.argmax(counts));retained=points[labels==label]
        component_fraction=len(retained)/len(points)
        if len(retained)<25 or component_fraction<.2:return None,'fragmented_3d_mask'
        points=retained
    else:return None,'fragmented_3d_mask'
    lo,hi=np.quantile(points,[.02,.98],axis=0);size=hi-lo
    if np.any(size>7) or np.prod(size)>70:return None,'implausible_extent'
    if max(size)<.08:return None,'tiny_extent'
    return {'points':points.astype(np.float32),'bbox':{'center':((lo+hi)/2).tolist(),'size':np.maximum(size,.01).tolist(),'yaw_deg':0.,'type':'axis_aligned_observed_surface_bounds'},'label':detection['label'],'score':detection['score'],'frame_id':frame['id'],'bbox_xyxy':detection['bbox_xyxy'],'mask_index':detection['mask_index'],'mask_area':detection['mask_area'],'valid_depth_fraction':frac,'component_fraction':component_fraction,'floor':frame['floor']},None


def duplicate_2d(a,b):
    # Synonym prompts frequently detect the same table/cabinet/wardrobe.
    pairs=[{'table','desk'},{'cabinet','wardrobe'},{'cabinet','bookshelf'}]
    return any({a['label'],b['label']}<=p for p in pairs)


def merge_instances(observations):
    clusters=[]
    for obs in sorted(observations,key=lambda d:-d['score']):
        match=None;best=0.
        for c in clusters:
            if c['floor']!=obs['floor'] or (c['label']!=obs['label'] and not duplicate_2d(c,obs)):continue
            iou=aabb_iou(c['bbox'],obs['bbox'])
            a=np.array(c['bbox']['center']);b=np.array(obs['bbox']['center']);distance=np.linalg.norm(a-b)
            # Surface proximity handles partial views of elongated objects without merging nearby chairs.
            if distance>max(1.2,np.linalg.norm(c['bbox']['size'])*.7):continue
            d=cKDTree(c['points']).query(obs['points'][::max(1,len(obs['points'])//300)])[0]
            overlap=float(np.mean(d<.12))
            sameframe=any(x['frame_id']==obs['frame_id'] for x in c['observations'])
            if sameframe and not duplicate_2d(c,obs):continue
            good=(iou>.12 and overlap>.08) or overlap>.35
            quality=max(iou,overlap)
            if good and quality>best:match=c;best=quality
        if match is None:
            clusters.append({'label':obs['label'],'floor':obs['floor'],'points':obs['points'].copy(),'bbox':obs['bbox'],'observations':[obs]})
        else:
            match['observations'].append(obs);p=np.concatenate([match['points'],obs['points']]);p=np.unique(np.rint(p/.025).astype(np.int32),axis=0)*.025
            if len(p)>10000:p=p[np.linspace(0,len(p)-1,10000).astype(int)]
            match['points']=p.astype(np.float32);lo,hi=np.quantile(p,[.02,.98],axis=0);match['bbox']={'center':((lo+hi)/2).tolist(),'size':np.maximum(hi-lo,.01).tolist(),'yaw_deg':0.,'type':'axis_aligned_observed_surface_bounds'}
    return clusters


def label_region(graph,floor,xy):
    from shapely.geometry import Point,shape
    p=Point(xy);regions=[r for r in graph['regions'] if r['floor']==floor]
    distances=[shape(r['geometry']).distance(p) for r in regions];i=int(np.argmin(distances))
    return regions[i]['id'],float(distances[i])


def build_objects(graph,out,old,registry):
    session=requests.Session();session.trust_env=False
    health=session.get(SAM_URL+'/health',timeout=10).json();signature=model_signature()
    if not health.get('ready') or any(health.get(k)!=v for k,v in signature.items()):
        raise RuntimeError('Resident SAM3 does not match the current checkpoint/code. Restart the local SAM3 service first.')
    frames=extract_keyframes(graph,out);observations=[];rejected=[];segerrors=[]
    for i,frame in enumerate(frames):
        result=segment_frame(frame,graph,out)
        masks=np.load(result['mask_path'])['masks']
        for detection in result['detections']:
            obs,reason=project_detection(frame,detection,masks,out)
            if obs is not None:observations.append(obs)
            else:rejected.append({'frame_id':frame['id'],'label':detection['label'],'score':detection['score'],'reason':reason})
        segerrors += [{'frame_id':frame['id'],**e} for e in result.get('errors',[])]
        if i%10==0:print('SAM3 frames',graph['scene_id'],i+1,'/',len(frames),'observations',len(observations),flush=True)
    clusters=merge_instances(observations);objects=[];pointsets={};byframe={f['id']:f for f in frames}
    for cluster in clusters:
        ob=cluster['observations'];region,distance=label_region(graph,cluster['floor'],cluster['bbox']['center'][:2]);n=len(set(x['frame_id'] for x in ob));confidence=float(max(x['score'] for x in ob))
        best=max(ob,key=lambda x:x['score']*np.sqrt(x['mask_area'])*x['valid_depth_fraction'])
        obj={'floor':cluster['floor'],'region':region,'category_pred':{'label':cluster['label'],'score':confidence,'source':'sam3:text_prompt+3d_fusion'},'bbox':cluster['bbox'],'n_views':n,'observations_count':len(ob),'assignment_distance_to_region':distance,
             'geometry_pred':{'score':float(min(1,np.mean([x['valid_depth_fraction'] for x in ob])*(.65+.35*min(n/3,1)))),'source':'sam3_mask+mesh_depth:robust_observed_surface_aabb','extent_is_complete':False,'caveat':'Bounds cover observed reconstructed surfaces, not amodal full object geometry.'},
             'view_evidence':[{'keyframe_id':x['frame_id'],'mask_index':x['mask_index'],'score':x['score'],'source':'sam3','valid_depth_fraction':x['valid_depth_fraction']} for x in ob],
             'relations_auto':[],**base('sam3+mesh_depth',confidence),'_points':cluster['points'],'_best':best}
        objects.append(obj)
    def metric(a,b):
        if a['floor']!=b.get('floor'):return 0.
        ca=a['category_pred']['label'];cb=b['category_pred']['label']
        if ca!=cb and not duplicate_2d({'label':ca},{'label':cb}):return 0.
        return aabb_iou(a['bbox'],b['bbox'])
    matching=stable_ids(objects,old.get('objects',[]),'o',metric,.2,registry)
    graph['objects']=objects
    for obj in objects:
        best=obj.pop('_best');points=obj.pop('_points');pointsets[obj['id']]=points
        directory=out/'evidence/object_views'/obj['id'];directory.mkdir(parents=True,exist_ok=True)
        frame=byframe[best['frame_id']];image=cv2.imread(str(out/frame['image_path']));maskpath=out/'evidence/masks'/(frame['id']+'.npz');mask=np.load(maskpath)['masks'][best['mask_index']]
        y,x=np.where(mask);x0=max(0,int(x.min())-15);x1=min(image.shape[1],int(x.max())+16);y0=max(0,int(y.min())-15);y1=min(image.shape[0],int(y.max())+16)
        # Dim context while preserving target appearance for attribute VLM.
        crop=image[y0:y1,x0:x1].copy();m=mask[y0:y1,x0:x1];crop[~m]=np.uint8(crop[~m]*.35+165)
        cv2.imwrite(str(directory/'best.jpg'),crop,[cv2.IMWRITE_JPEG_QUALITY,93]);cv2.imwrite(str(directory/'context.jpg'),image[y0:y1,x0:x1]);cv2.imwrite(str(directory/'mask.png'),m.astype(np.uint8)*255)
        overlay=image.copy();contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(0,220,40),2);cv2.putText(overlay,obj['id']+' '+obj['category_pred']['label'],(15,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,255),2);cv2.imwrite(str(directory/'full_view.jpg'),overlay)
        obj['best_view']=rel(directory/'best.jpg',out);obj['points_ref']={'path':'evidence/object_points.npz','key':obj['id']};obj['best_view_metadata']={'keyframe_id':frame['id'],'crop_xyxy':[x0,y0,x1,y1],'mask_path':rel(directory/'mask.png',out),'context_path':rel(directory/'context.jpg',out),'full_view_path':rel(directory/'full_view.jpg',out)}
        next(r for r in graph['regions'] if r['id']==obj['region'])['object_ids'].append(obj['id'])
        if obj['n_views']<2:graph['issues'].append({'type':'single_view_object','ref':obj['id'],'detail':'物体仅有一个有效关键帧支持，实例融合和尺寸需复核。','priority':.65,**base('multi_view_fusion',.8)})
        if obj['score']<.6:graph['issues'].append({'type':'low_confidence','ref':obj['id'],'detail':f'SAM3 类别置信度 {obj["score"]:.2f}。','priority':.9,**base('sam3',1-obj['score'])})
        if obj['assignment_distance_to_region']>.8:graph['issues'].append({'type':'object_region_assignment_uncertain','ref':obj['id'],'detail':f'物体中心距最近候选可通行区域 {obj["assignment_distance_to_region"]:.2f} m；房间多边形尚非完整房间。','priority':.75,**base('geometry',.8)})
    np.savez_compressed(out/'evidence/object_points.npz',**pointsets)
    dump(out/'evidence/detection_rejections.json',{'rejected':rejected,'sam3_prompt_errors':segerrors,'notes':'Rejected mask-depth lifts are retained as audit records, not silently promoted to 3D objects.'})
    if segerrors:graph['issues'].append({'type':'sam3_prompt_failure','ref':'s0','detail':f'{len(segerrors)} 个关键帧文本提示执行失败，见 evidence/detection_rejections.json。','priority':1.,**base('pipeline',1.)})
    dump(out/'evidence/segmentation_summary.json',{'keyframes':len(frames),'valid_3d_observations':len(observations),'rejected_depth_lifts':len(rejected),'fused_instances':len(objects),'sam3_prompt_errors':len(segerrors),'confidence_threshold':.45,'stride_frames':16,'merge':'same/synonym category, same floor, robust AABB IoU and 3D observed point proximity','id_matching':matching})
    return matching
