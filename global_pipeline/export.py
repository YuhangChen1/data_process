"""Global-ID mask assignment, multi-view semantics, per-frame PNGs and videos."""
from .common import ROOT,read,dump,utc,progress,frame_mask,frame_projection,colors,write_csv
from .instances import quality
from .labels import canonical
from collections import defaultdict,Counter
import numpy as np,cv2,subprocess,shutil,os

def mask_assignments(out,doc,fine,config):
    point_ids=fine['point_fine_ids'];all_assignments={};votes=defaultdict(Counter);fine_votes=defaultdict(Counter);view_support=defaultdict(set);type_votes=defaultdict(Counter)
    frame_stats=[];metrics=[];best=defaultdict(list);sameframe=[]
    coarse=np.load(out/'coarse_instances.npz')['point_coarse_ids']
    for fi,frame in enumerate(doc['frames']):
        p=np.load(frame_projection(out,frame));idx=p['point_indices'];pixels=p['pixels'];visible=point_ids[idx]
        fine_visible=Counter(visible.tolist());coarse_visible=coarse[idx];coarse_counts=Counter(coarse_visible.tolist())
        with np.load(frame_mask(out,frame)) as mask:local=mask['local_instances'].ravel()[pixels]
        detections=read(frame_mask(out,frame).with_suffix('.json'))['detections'];assigned=[];seen=defaultdict(list)
        pose=np.array(frame['camera_to_canonical']);fwd=pose[:3,2];view_group=tuple(np.floor(pose[:3,3]/.25).astype(int).tolist()+[int(np.floor(np.arctan2(fwd[1],fwd[0])/(np.pi/6)))])
        candidates=[]
        for d in detections:
            inside=local==d['exclusive_local_id'];labels,n=np.unique(visible[inside],return_counts=True);keep=labels>0
            labels=labels[keep];n=n[keep];support=int(n.max()) if len(n) else 0
            total=int(inside.sum());gid=int(labels[np.argmax(n)]) if len(n) else 0
            purity=support/max(total,1);accepted=support>=config['global_min_points'] and purity>=config['global_purity']
            record={'local_id':d['exclusive_local_id'],'global_id':gid if accepted else 0,'candidate_global_id':gid,
                    'semantic_label_2d':d['semantic_label'],'category_type':d['category_type'],
                    'support_points':support,'all_mask_support_points':total,'purity':purity,
                    'score':d['score'],'stability_score':d['stability_score'],'sam_iou':d.get('sam_iou'),
                    'accepted':accepted,'reason':'accepted' if accepted else 'insufficient_support' if support<config['global_min_points'] else 'mixed_global_ids',
                    'source':'fine_point_ids+zbuffer+mesh_depth'}
            record['assignment_quality']=quality(d)
            candidates.append((d,record))
        # Keep the coarse/fine topology fixed. If two high-quality thing masks
        # claim the same global ID in one view, retain the strongest assignment
        # and mark the other unknown; log the conflict instead of hiding it.
        claimed={}
        for d,record in sorted(candidates,key=lambda pair:-(pair[1]['assignment_quality']*pair[1]['support_points']*pair[1]['purity'])):
            gid=record['global_id']
            high=d['score']>=config['hard_score'] and d['stability_score']>=config['hard_quality']
            if record['accepted'] and high and d['category_type']=='thing':
                if gid in claimed:
                    sameframe.append({'frame_id':frame['id'],'global_id':gid,'local_ids':str(claimed[gid])+','+str(record['local_id'])})
                    record.update(global_id=0,accepted=False,reason='same_frame_high_quality_global_id_conflict')
                else:claimed[gid]=record['local_id']
        for d,record in candidates:
            inside=local==d['exclusive_local_id'];support=record['support_points'];total=record['all_mask_support_points'];purity=record['purity'];gid=record['global_id']
            if record['accepted']:
                cover=support/max(fine_visible[gid],1);weight=quality(d)*cover*np.log1p(support)
                record['instance_cover']=cover;record['semantic_vote_weight']=float(weight)
                cls=canonical(d['semantic_label']);fine_votes[gid][d['semantic_label']]+=float(weight)
                votes[gid][cls]+=float(weight);view_support[(gid,cls)].add(view_group)
                type_votes[(gid,cls)][d['category_type']]+=float(weight)
                best[gid].append({'frame_id':frame['id'],'clip_id':frame['clip_id'],'local_id':d['exclusive_local_id'],
                                  'weight':float(weight),'support':support,'score':d['score']})
                seen[gid].append(d['exclusive_local_id'])
            assigned.append(record)
            # Two evaluation domains are explicit: sampled visible points and
            # all image pixels (the latter has low recall on sparse point clouds).
            c,cnt=np.unique(coarse_visible[inside],return_counts=True);good=c>0;c=c[good];cnt=cnt[good]
            cg=int(c[np.argmax(cnt)]) if len(c) else 0;tp=int(cnt.max()) if len(cnt) else 0
            pred=int(coarse_counts[cg]) if cg else 0;actual=total
            precision=tp/max(pred,1);recall=tp/max(actual,1)
            metrics.append({'frame_id':frame['id'],'local_id':d['exclusive_local_id'],'coarse_id':cg,
                'visible_point_iou':tp/max(pred+actual-tp,1),'visible_point_precision':precision,'visible_point_recall':recall,
                'visible_point_f1':2*precision*recall/max(precision+recall,1e-6),
                'dense_pixel_iou':tp/max(pred+d['exclusive_area']-tp,1),'dense_pixel_recall':tp/max(d['exclusive_area'],1),
                'support_points':total,'fine_global_id':record['global_id'],'fine_mask_purity':purity})
        for gid,local_ids in seen.items():
            if len(local_ids)>1:sameframe.append({'frame_id':frame['id'],'global_id':gid,'local_ids':','.join(map(str,local_ids))})
        all_assignments[frame['id']]=assigned
        frame_stats.append({'frame_id':frame['id'],'clip_id':frame['clip_id'],'masks':len(detections),'assigned_masks':sum(a['accepted'] for a in assigned),'visible_points':len(idx)})
        if fi%50==0:progress(out,'global_reprojection',assigned_frames=fi+1,frames_total=len(doc['frames']))
    glossary_path=out/'semantic_registry.json';registry=read(glossary_path) if glossary_path.exists() else {'unknown':0}
    vocabulary=sorted({label for tally in votes.values() for label in tally})
    for label in vocabulary:
        if label not in registry:registry[label]=max(registry.values())+1
    semantics={};instances=[];points=fine['points']
    gids=np.unique(point_ids[point_ids>0]);sp=fine['point_superpoint_ids']
    for gid in gids:
        tally=votes[int(gid)];ordered=tally.most_common();total=sum(tally.values())
        label,winner=ordered[0] if ordered else ('unknown',0);runner=ordered[1][1] if len(ordered)>1 else 0.
        margin=(winner-runner)/max(total,1e-6);nviews=len(view_support[(int(gid),label)])
        reliable=bool(ordered) and margin>=config['semantic_margin'] and nviews>=config['semantic_min_views']
        final=label if reliable else 'unknown';semantics[int(gid)]=registry[final]
        xyz=points[point_ids==gid];lo,hi=np.quantile(xyz,[.02,.98],axis=0)
        observations=sorted(best[int(gid)],key=lambda x:-x['weight']);best_view=observations[0] if observations else None
        instances.append({'global_id':int(gid),'semantic_id':registry[final],
            'semantic_label':final,'fine_label':(max(((n,v) for n,v in fine_votes[int(gid)].items() if canonical(n)==final),key=lambda x:x[1])[0] if final!='unknown' and fine_votes[int(gid)] else final),
            'label_variants':dict(fine_votes[int(gid)]),'category_type':type_votes[(int(gid),final)].most_common(1)[0][0] if type_votes[(int(gid),final)] else 'thing',
            'semantic_pred':{'label':final,'score':float(winner/max(total,1e-6)) if reliable else 0.,
                'source':'gpt_inventory+sam3+multi_view_weighted_vote','winner_label':label,'winner_margin':margin,
                'independent_views':nviews,'votes':dict(tally),'unknown_reason':None if reliable else 'no_votes' if not ordered else 'insufficient_independent_views' if nviews<config['semantic_min_views'] else 'semantic_vote_margin'},
            'bbox':{'center':((lo+hi)/2).tolist(),'size':np.maximum(hi-lo,.01).tolist(),'yaw_deg':0.,'observed_surface_extent':True},
            'points':len(xyz),'superpoints':len(np.unique(sp[point_ids==gid])),
            'n_frames':len({o['frame_id'] for o in observations}),'best_view':best_view,
            'review':{'status':'unreviewed'},'source':'fine_instances.npz:point_fine_ids',
            'score':float(winner/max(total,1e-6)) if reliable else 0.})
    dump(glossary_path,registry);dump(out/'global_instances.json',{'schema':'kosmoworld.global_instances','schema_version':'0.2',
        'scene_id':out.parent.name,'frame':'canonical','units':'inferred_meter','instances':instances,'semantic_registry':registry,
        'global_id_source':'fine_instances.npz:point_fine_ids','unknown_id':0})
    dump(out/'frame_assignments.json',{'frames':all_assignments,'acceptance':{'minimum_support_points':config['global_min_points'],'minimum_purity':config['global_purity']},
        'mask_boundary':'Kept exactly as exclusive per-frame SAM3 mask; rejected masks stay zero.'})
    write_csv(out/'qa/reprojection_metrics.csv',metrics,list(metrics[0]) if metrics else ['frame_id'])
    write_csv(out/'qa/frame_stats.csv',frame_stats,list(frame_stats[0]))
    write_csv(out/'qa/same_frame_global_id_collisions.csv',sameframe,['frame_id','global_id','local_ids'])
    metric_summary={key:float(np.mean([r[key] for r in metrics])) for key in ['visible_point_iou','visible_point_precision','visible_point_recall','visible_point_f1','dense_pixel_iou','dense_pixel_recall']} if metrics else {}
    dump(out/'qa/reprojection_summary.json',{'mask_evaluations':len(metrics),'metrics':metric_summary,
        'semantic_unknown_instances':sum(i['semantic_label']=='unknown' for i in instances),'global_instances':len(instances),
        'accepted_mask_assignments':sum(x['assigned_masks'] for x in frame_stats),'total_masks':sum(x['masks'] for x in frame_stats),
        'same_frame_global_id_collision_rows':len(sameframe),
        'metric_caveat':'visible_point metrics evaluate only z-buffered depth-consistent sampled points. dense_pixel metrics compare sparse point raster to full SAM masks, so sparse coverage lowers recall. Neither is independent human ground truth.'})
    return all_assignments,semantics,instances

def writer(path,width,height,fps):
    path.parent.mkdir(parents=True,exist_ok=True)
    return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
        '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)

def render_frames(out,doc,assignments,semantics,instances):
    names={i['global_id']:i['semantic_label'] for i in instances};lookup=np.zeros(max(semantics,default=0)+1,np.uint16)
    for gid,sid in semantics.items():lookup[gid]=sid
    byid={f['id']:f for f in doc['frames']};manifest=[];preview=None
    for clip in doc['clips']:
        first=byid[clip['frames'][0]];w,h=first['width'],first['height'];video=out/'videos'/clip['id'];video.mkdir(parents=True,exist_ok=True)
        outputs={name:writer(video/(name+'.mp4'),w,h,clip['fps']) for name in ['overlay','instance','semantic']}
        try:
            for fid in clip['frames']:
                frame=byid[fid];rgb=cv2.imread(str(out/frame['rgb']));target=out/'rendered'/clip['id'];target.mkdir(parents=True,exist_ok=True)
                with np.load(frame_mask(out,frame)) as mask:local=mask['local_instances']
                mapping=np.zeros(int(local.max())+1,np.uint16)
                for a in assignments[fid]:mapping[a['local_id']]=a['global_id']
                instance=mapping[local];semantic=lookup[instance]
                instance_color=colors(instance)[:,:,::-1].copy();semantic_color=colors(semantic)[:,:,::-1].copy();local_color=colors(local)[:,:,::-1].copy()
                overlay=rgb.copy();valid=instance>0;overlay[valid]=np.uint8(rgb[valid]*.55+instance_color[valid]*.45)
                for gid in np.unique(instance[valid]):
                    mask=(instance==gid).astype(np.uint8);contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,tuple(map(int,colors(np.array([gid]))[0][::-1])),1)
                    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
                    if count>1:
                        best=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);x,y=centroids[best].astype(int)
                        if stats[best,cv2.CC_STAT_AREA]>250:
                            text=f'{names.get(int(gid),"unknown")} #{gid}';cv2.putText(overlay,text,(max(2,x-20),max(15,y)),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),2,cv2.LINE_AA);cv2.putText(overlay,text,(max(2,x-20),max(15,y)),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA)
                paths={}
                for name,img,ext in [('instance',instance,'png'),('semantic',semantic,'png'),('instance_color',instance_color,'png'),('semantic_color',semantic_color,'png'),('sam2d',local_color,'jpg'),('overlay',overlay,'jpg')]:
                    folder=target/name;folder.mkdir(exist_ok=True);path=folder/(fid+'.'+ext);cv2.imwrite(str(path),img);paths[name]=str(path.relative_to(out))
                for name,img in [('overlay',overlay),('instance',instance_color),('semantic',semantic_color)]:outputs[name].stdin.write(img.tobytes())
                manifest.append({'frame_id':fid,**paths})
                if preview is None:
                    tiles=[rgb,overlay,instance_color]
                    preview=np.concatenate([cv2.resize(t,(640,360)) for t in tiles],axis=1)
        finally:
            for process in outputs.values():
                process.stdin.close()
                if process.wait()!=0:raise RuntimeError('Video encoder failed')
        # The RGB source is the actual Gaussian rendering on this same trajectory.
        source=ROOT/first['source_video']
        if source.is_dir():source=source.parent/'walkthrough.mp4'   # hesai frames_hd/ -> the scan's own encoded walkthrough
        shutil.copyfile(source,video/'rgb.mp4')
        print('Rendered global video',out.parent.name,clip['id'],len(clip['frames']),flush=True)
        progress(out,'video_export',rendered_frames=len(manifest),frames_total=len(doc['frames']))
    if preview is not None:cv2.imwrite(str(out/'preview.jpg'),preview)
    dump(out/'rendered_frames.json',{'frames':manifest,'encoding':{'instance':'uint16 PNG, 0 unknown, numeric fine global ID','semantic':'uint16 PNG, 0 unknown; semantic_registry.json maps IDs','colors':'deterministic visualization only, not numeric label encoding'},
        'videos':[{**clip,'videos':{name:f'videos/{clip["id"]}/{name}.mp4' for name in ['rgb','overlay','instance','semantic']}} for clip in doc['clips']]})
    # Per-component clips are authoritative; the scene preview uses explicit cuts.
    concat=out/'videos/concat.txt';concat.write_text(''.join("file '"+str((out/'videos'/clip['id']/'overlay.mp4').resolve())+"'\n" for clip in doc['clips']))
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(out/'projection_video.mp4')],check=True)
    dump(out/'trajectory.json',{'purpose':'annotation_only_not_evaluation_Q','scene_preview':'projection_video.mp4','clips':doc['clips'],
        'clip_transitions':'Explicit hard cuts between disconnected components and floors; no invented traversals.',
        'source':'Previously planned A* smoothed collision-checked free-map routes; camera 1.5 inferred meters above floor; OpenCV poses in frames.json.'})

def export(out,doc,config):
    fine=np.load(out/'fine_instances.npz');progress(out,'global_reprojection')
    assignments,semantics,instances=mask_assignments(out,doc,fine,config)
    if os.environ.get('GLOBAL_RELABEL_UNKNOWN','1')=='1':   # data_pipeline: GPT labels for vote-unknown instances
        from .relabel import relabel_unknown
        registry=read(out/'semantic_registry.json');scene=read(ROOT/'sources'/out.parent.name/'source.json').get('label_zh','') if (ROOT/'sources'/out.parent.name/'source.json').exists() else ''
        relabel_unknown(out,doc,fine,assignments,semantics,instances,registry,scene_text=scene)
        fine=np.load(out/'fine_instances.npz')   # unseen fragments may have been dropped (IDs -> 0)
    from plyfile import PlyData,PlyElement
    points=fine['points'];ids=fine['point_fine_ids'];rgb=colors(ids)
    vertices=np.empty(len(points),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1'),('global_id','i4'),('semantic_id','i4'),('superpoint_id','i4')])
    for axis,name in enumerate(['x','y','z']):vertices[name]=points[:,axis]
    for axis,name in enumerate(['red','green','blue']):vertices[name]=rgb[:,axis]
    vertices['global_id']=ids;vertices['semantic_id']=np.array([semantics.get(int(i),0) for i in ids],np.int32);vertices['superpoint_id']=fine['point_superpoint_ids']
    PlyData([PlyElement.describe(vertices,'vertex')],text=False).write(str(out/'labeled_points.ply'))
    render_frames(out,doc,assignments,semantics,instances)
