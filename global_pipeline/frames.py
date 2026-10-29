"""Export every existing 3DGS trajectory frame with aligned mesh depth and pose."""
from .common import ROOT,read,dump,sha,progress,fingerprint
import cv2,numpy as np
from pathlib import Path

def prepare_frames(scene,out,config):
    path=out/'frames.json'
    # data_pipeline (hesai): the scan's own forward walkthrough is the annotation clip when present, so every label is
    # aligned with the evaluate1 trajectory; otherwise fall back to scene_pipeline's A* region walkthroughs.
    sources=sorted((ROOT/'outputs'/scene).glob('hesai_walkthrough/*/metadata.json'))
    if not sources:sources=sorted((ROOT/'outputs'/scene).glob('floor_*/annotation_walkthrough/component_*/metadata.json'))
    sources+=sorted((out/'supplemental_trajectories').glob('floor_*/sweep_*/metadata.json'))
    frame_registry_path=out/'frame_id_registry.json'
    registry=read(frame_registry_path) if frame_registry_path.exists() else {'maximum_id':-1,'sources':{}}
    if path.exists():
        for f in read(path).get('frames',[]):
            key=f['source_video']+':'+str(f['source_frame_index']);registry['sources'][key]=f['id'];registry['maximum_id']=max(registry['maximum_id'],int(f['id'][1:]))
    signature=fingerprint({'stride':config['frame_stride'],'sources':[(str(p),sha(p),sha(p.parent/'cameras.npz')) for p in sources]})
    if path.exists():
        previous=read(path)
        if previous.get('fingerprint')==signature and all((out/f['rgb']).exists() for f in previous['frames']):return previous
    frames=[];clips=[];index=0
    for source in sources:
        folder=source.parent;meta=read(source);floor='floor_'+meta['floor'][1:] if 'floor' in meta else folder.parent.parent.name
        hesai='frames_dir' in meta;clip_id=f'{floor}_hesai_{folder.name}' if hesai else f'{floor}_{folder.name}'
        target=out/'frames'/clip_id;target.mkdir(parents=True,exist_ok=True);cameras=np.load(folder/'cameras.npz');i=0;selected=[]
        if hesai:
            pngs=sorted(Path(meta['frames_dir']).glob('*.png'));cap=None
            video_path=Path(meta['frames_dir'])
        else:cap=cv2.VideoCapture(str(folder/'rgb.mp4'));video_path=folder/'rgb.mp4'
        while True:
            if hesai:
                okay=i<len(pngs);image=cv2.imread(str(pngs[i])) if okay else None
            else:okay,image=cap.read()
            if not okay:break
            if i%config['frame_stride']==0 or i==meta['frames']-1:
                key=str(video_path.relative_to(ROOT))+':'+str(i)
                if key not in registry['sources']:
                    registry['maximum_id']+=1;registry['sources'][key]=f'g{registry["maximum_id"]:06d}'
                fid=registry['sources'][key];rgb=target/(fid+'.jpg');cv2.imwrite(str(rgb),image,[cv2.IMWRITE_JPEG_QUALITY,92])
                frames.append({'id':fid,'index':index,'clip_id':clip_id,'floor':floor.replace('floor_','f'),
                    'source_frame_index':i,'timestamp':i/meta['fps'],'rgb':str(rgb.relative_to(out)),
                    'depth':str((folder/'mesh_depth_mm'/f'{i:05d}.png').relative_to(ROOT)),
                    'camera_to_canonical':cameras['camera_to_canonical'][i].tolist(),'K':cameras['K'].tolist(),
                    'width':image.shape[1],'height':image.shape[0],'rgb_sha256':sha(rgb),'source_video':str(video_path.relative_to(ROOT)),
                    'purpose':'annotation_trajectory_only_not_evaluation_Q'})
                selected.append(fid);index+=1
            i+=1
        if cap is not None:cap.release()
        if i!=meta['frames']:raise RuntimeError(f'RGB/camera frame mismatch: {folder}: {i}/{meta["frames"]}')
        clips.append({'id':clip_id,'floor':floor.replace('floor_','f'),'source_metadata':str(source.relative_to(ROOT)),
                      'source_fps':meta['fps'],'fps':meta['fps']/config['frame_stride'],'frames':selected,
                      'camera_height':meta['camera_height_above_floor'],'trajectory_collision_checked':meta['trajectory_collision_checked'],
                      'region_order':meta['region_order'],'video':'videos/'+clip_id+'/overlay.mp4'})
        progress(out,'frames',frames_exported=len(frames),clips_exported=len(clips))
    doc={'schema':'kosmoworld.aligned_frames','version':'0.2','frame':'canonical','units':'inferred_meter',
         'pixel_convention':'OpenCV; pixel centers u+.5,v+.5; mesh camera-Z uint16 mm with 0 invalid',
         'fingerprint':signature,'frames':frames,'clips':clips,'source_rendering':'Actual anisotropic 3DGS RGB and same-camera mesh depth; decoded losslessly in time from existing planned trajectory videos. JPEG image compression only.'}
    dump(path,doc);dump(frame_registry_path,registry);return doc
