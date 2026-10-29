"""2D-to-superpoint proposals, constrained coarse topology, boundary-only refinement."""
from .common import ROOT,read,dump,progress,frame_mask,frame_projection,write_csv,utc
from collections import defaultdict,Counter
from itertools import combinations
from scipy.sparse import csr_matrix,coo_matrix
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import numpy as np

def quality(record):
    # A true predicted IoU is preferred if supplied by a future processor.
    # Current SAM3 text grounding exposes stability instead; provenance is explicit.
    measure=record.get('sam_iou')
    if measure is None:measure=record.get('stability_score',record['score'])
    return float(np.sqrt(max(0,record['score']*measure)))

def make_proposals(out,doc,cloud,config):
    sp=cloud['point_superpoint_ids'];points=cloud['points'];records=[];indices=[];counts=[];offsets=[0]
    for f,frame in enumerate(doc['frames']):
        projection=np.load(frame_projection(out,frame));pointidx=projection['point_indices'];pixels=projection['pixels']
        with np.load(frame_mask(out,frame)) as data:owner=data['local_instances'].ravel()[pixels]
        detections=read(frame_mask(out,frame).with_suffix('.json'))['detections']
        vis_ids,vis_counts=np.unique(sp[pointidx],return_counts=True);visible=dict(zip(vis_ids,vis_counts))
        for detection in detections:
            selected=pointidx[owner==detection['exclusive_local_id']]
            if len(selected)<config['proposal_min_points']:continue
            ids,n=np.unique(sp[selected],return_counts=True)
            good=np.array([count>=2 and count>=config['proposal_superpoint_fraction']*visible[sid] for sid,count in zip(ids,n)])
            ids=ids[good];n=n[good]
            if not len(ids):continue
            bbox=np.quantile(points[selected],[.02,.98],axis=0)
            q=quality(detection);metric=detection.get('sam_iou') if detection.get('sam_iou') is not None else detection['stability_score']
            hard=metric>=config['hard_quality'] and detection['score']>=config['hard_score'] and len(ids)>=2 and len(selected)>=20
            pose=np.array(frame['camera_to_canonical']);forward=pose[:3,2]
            view_group=tuple(np.floor(pose[:3,3]/.25).astype(int).tolist()+[int(np.floor(np.arctan2(forward[1],forward[0])/(np.pi/6)))])
            records.append({'proposal_id':len(records),'frame_id':frame['id'],'frame_index':f,'clip_id':frame['clip_id'],
                'local_id':detection['exclusive_local_id'],'semantic_label':detection['semantic_label'],
                'category_type':detection['category_type'],'score':detection['score'],'stability_score':detection['stability_score'],
                'sam_iou':detection.get('sam_iou'),'quality':q,'support_points':len(selected),'superpoints':len(ids),
                'hard_quality':bool(hard),'hard_quality_measure':'predicted_sam_iou' if detection.get('sam_iou') is not None else 'mask_threshold_stability_proxy',
                'bbox_low':bbox[0].tolist(),'bbox_high':bbox[1].tolist(),'view_group':view_group})
            indices.extend(ids.tolist());counts.extend(n.tolist());offsets.append(len(indices))
        if f%50==0:progress(out,'proposals',proposal_frames=f+1,proposals=len(records))
    matrix=csr_matrix((np.array(counts,np.float32),np.array(indices,np.int32),np.array(offsets,np.int64)),shape=(len(records),len(cloud['centroids'])))
    np.savez_compressed(out/'proposals.npz',superpoint_indices=matrix.indices,support_counts=matrix.data,offsets=matrix.indptr)
    dump(out/'proposals.json',{'proposals':records,'notes':'Proposals use exclusive 2D masks, per-pixel z-buffer and same-camera mesh depth checks.'})
    return records,matrix

def constrained_merge(records,support,out,config):
    n=len(records);binary=support.copy();binary.data=np.ones(len(binary.data),np.float32)
    overlap=(binary@binary.T).tocoo();size=np.diff(binary.indptr)
    positive=[]
    for a,b,intersection in zip(overlap.row,overlap.col,overlap.data):
        if a>=b or records[a]['frame_id']==records[b]['frame_id']:continue
        iou=intersection/max(size[a]+size[b]-intersection,1);iomin=intersection/max(min(size[a],size[b]),1)
        if iou>=config['merge_iou'] or iomin>=config['merge_iomin']:
            positive.append((max(iou,iomin),int(a),int(b),float(iou),float(iomin)))
    cannot=[set() for _ in records];hard=[];perframe=defaultdict(list)
    for r in records:perframe[r['frame_id']].append(r['proposal_id'])
    for frame,ids in perframe.items():
        for a,b in combinations(ids,2):
            ra,rb=records[a],records[b]
            gap=np.maximum(0,np.maximum(np.array(ra['bbox_low'])-rb['bbox_high'],np.array(rb['bbox_low'])-ra['bbox_high']))
            disjoint=np.linalg.norm(gap)>.12
            if (ra['hard_quality'] and rb['hard_quality']) or disjoint:
                reason='same_frame_high_quality_mask_stability_proxy' if ra['hard_quality'] and rb['hard_quality'] else 'same_frame_spatially_disjoint'
                cannot[a].add(b);cannot[b].add(a);hard.append({'frame_id':frame,'proposal_a':a,'proposal_b':b,'reason':reason})
    parent=np.arange(n);members=[{i} for i in range(n)];forbidden=[set(x) for x in cannot]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    conflicts=[];accepted=0
    for strength,a,b,iou,iomin in sorted(positive,reverse=True):
        x,y=root(a),root(b)
        if x==y:continue
        collision=forbidden[x].intersection(members[y]) or forbidden[y].intersection(members[x])
        if collision:
            conflicts.append({'proposal_a':a,'proposal_b':b,'iou3d':iou,'iomin3d':iomin,'reason':'transitive_cluster_cannot_link','witness':min(collision)})
            continue
        if len(members[x])<len(members[y]):x,y=y,x
        parent[y]=x;members[x].update(members[y]);forbidden[x].update(forbidden[y]);members[y]=set();forbidden[y]=set();accepted+=1
    roots=np.array([root(i) for i in range(n)],np.int32);_,cluster=np.unique(roots,return_inverse=True)
    weight=np.array([r['quality']*np.log1p(r['support_points']) for r in records])
    coo=support.tocoo();vote=coo_matrix((coo.data*weight[coo.row],(cluster[coo.row],coo.col)),shape=(int(cluster.max())+1 if n else 0,support.shape[1])).tocsc()
    coarse=np.zeros(support.shape[1],np.int32);confidence=np.zeros(support.shape[1],np.float32)
    for s in range(support.shape[1]):
        start,end=vote.indptr[s:s+2]
        if end>start:
            v=vote.data[start:end];best=int(np.argmax(v));coarse[s]=int(vote.indices[start+best])+1;confidence[s]=float(v[best]/max(v.sum(),1e-6))
    active=np.unique(coarse[coarse>0]);compact={int(label):i+1 for i,label in enumerate(active)}
    coarse=np.array([compact.get(int(label),0) for label in coarse],np.int32)
    write_csv(out/'qa/hard_mask_conflicts.csv',hard,['frame_id','proposal_a','proposal_b','reason'])
    write_csv(out/'qa/merge_conflicts.csv',conflicts,['proposal_a','proposal_b','iou3d','iomin3d','reason','witness'])
    dump(out/'coarse_summary.json',{'proposals':n,'positive_edges':len(positive),'accepted_merges':accepted,
        'cannot_link_pairs':len(hard),'blocked_transitive_merges':len(conflicts),'coarse_instances':len(active),
        'hard_conflicts_inside_clusters':int(sum(roots[x['proposal_a']]==roots[x['proposal_b']] for x in hard)),
        'hard_quality_caveat':'SAM3 text grounding has no predicted IoU; mask probability threshold stability >=.90 and detector score >=.80 replace that unavailable test.',
        'cluster_rule':'Every candidate union checks full cluster cannot-link sets; same-frame spatial conflicts are retained.'})
    return coarse,confidence,records

def refine(coarse,records,support,cloud,out,config):
    n=len(coarse);current=coarse.copy();unary_rows=[];unary_cols=[];unary_weights=[];views=defaultdict(set)
    for i,record in enumerate(records):
        start,end=support.indptr[i:i+2];ids=support.indices[start:end];counts=support.data[start:end]
        labels,positions=np.unique(coarse[ids],return_inverse=True);weights=np.bincount(positions,weights=counts)
        valid=labels>0;labels=labels[valid];weights=weights[valid]
        if not len(labels):continue
        mixture=weights/weights.sum()
        for label,fraction in zip(labels,mixture):
            if fraction<.05:continue
            unary_rows.extend(ids.tolist());unary_cols.extend([int(label)]*len(ids));unary_weights.extend((counts*record['quality']*fraction).tolist())
            for sid in ids:views[(int(sid),int(label))].add(tuple(record['view_group']))
    unary=coo_matrix((unary_weights,(unary_rows,unary_cols)),shape=(n,int(coarse.max())+1)).tocsr()
    totals=np.asarray(unary.sum(1)).ravel();unary=unary.multiply(1/np.maximum(totals[:,None],1e-6)).tocsr()
    centers=cloud['centroids'];adjacency=cloud['adjacency'];distance,neighbor=cKDTree(centers).query(centers,k=min(9,len(centers)))
    rows=np.repeat(np.arange(n),neighbor.shape[1]-1);cols=neighbor[:,1:].ravel();dist=distance[:,1:].ravel();good=dist<=.25
    pairs=np.r_[adjacency,np.stack([rows[good],cols[good]],1)];pairs=np.unique(np.sort(pairs,axis=1),axis=0)
    graph=csr_matrix((np.ones(2*len(pairs)),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(n,n))
    migrated=[]
    for iteration in range(3):
        counts=Counter(current.tolist());plans=[]
        for sid in range(n):
            neighbors=graph.indices[graph.indptr[sid]:graph.indptr[sid+1]]
            labels=current[neighbors];targets=set(labels.tolist())-{0,int(current[sid])}
            if not targets:continue
            if current[sid]>0 and np.all(labels==current[sid]):continue
            weights=np.exp(-np.linalg.norm(centers[neighbors]-centers[sid],axis=1)/.15)
            row=unary.getrow(sid);evidence=dict(zip(row.indices,row.data))
            def score(label):return config['fine_unary']*evidence.get(label,0)+config['fine_neighbor']*float(weights[labels==label].sum()/max(weights.sum(),1e-6))
            old=int(current[sid]);previous=score(old);target=max(targets,key=score);new=score(target)
            if new>=config['fine_min_score'] and new-previous>=config['fine_gain'] and len(views[(sid,target)])>=config['fine_min_views']:
                plans.append((new-previous,sid,old,int(target),previous,new,len(views[(sid,target)])))
        changed=0
        for gain,sid,old,new,old_score,new_score,nviews in sorted(plans,reverse=True):
            if old and counts[old]<=1:continue
            current[sid]=new;counts[old]-=1;counts[new]+=1;changed+=1
            migrated.append({'iteration':iteration,'superpoint':sid,'source':old,'target':new,'old_score':old_score,'new_score':new_score,'independent_views':nviews})
        if not changed:break
    assert set(current[current>0])==set(coarse[coarse>0]),'Fine refinement changed instance topology'
    write_csv(out/'qa/fine_boundary_migrations.csv',migrated,['iteration','superpoint','source','target','old_score','new_score','independent_views'])
    dump(out/'fine_summary.json',{'migrations':len(migrated),'moved_superpoints':int(np.count_nonzero(current!=coarse)),
        'topology_preserved':True,'new_instances_created':0,'coarse_regions_emptied':0,
        'independent_view_definition':'camera position cells .25 units and heading bins 30 degrees',
        'unary_weight':config['fine_unary'],'neighbor_weight':config['fine_neighbor'],'minimum_independent_views':config['fine_min_views']})
    return current

def stabilize(out,cloud,coarse,fine):
    previous=out/'fine_instances.npz';registry_path=out/'global_id_registry.json';registry=read(registry_path) if registry_path.exists() else {'maximum_id':0}
    labels=np.unique(fine[fine>0]);mapping={};matched=0
    if previous.exists():
        old=np.load(previous)
        if old['points'].shape==cloud['points'].shape and np.allclose(old['points'],cloud['points'],atol=1e-6):
            old_sp=old['superpoint_fine_ids'];old_labels=np.unique(old_sp[old_sp>0]);registry['maximum_id']=max(registry['maximum_id'],int(old_labels.max()) if len(old_labels) else 0)
            sizes=cloud['superpoint_sizes'];sim=np.zeros((len(labels),len(old_labels)))
            if len(labels) and len(old_labels):
                valid=(fine>0)&(old_sp>0)
                row=np.searchsorted(labels,fine[valid]);col=np.searchsorted(old_labels,old_sp[valid])
                intersection=np.bincount(row*len(old_labels)+col,weights=sizes[valid],minlength=sim.size).reshape(sim.shape)
                new_size=np.bincount(np.searchsorted(labels,fine[fine>0]),weights=sizes[fine>0],minlength=len(labels))
                old_size=np.bincount(np.searchsorted(old_labels,old_sp[old_sp>0]),weights=sizes[old_sp>0],minlength=len(old_labels))
                sim=intersection/np.maximum(new_size[:,None]+old_size[None,:]-intersection,1)
            if sim.size:
                rows,cols=linear_sum_assignment(-sim)
                for i,j in zip(rows,cols):
                    if sim[i,j]>=.25:mapping[int(labels[i])]=int(old_labels[j]);matched+=1
    for label in labels:
        if int(label) not in mapping:registry['maximum_id']+=1;mapping[int(label)]=registry['maximum_id']
    if registry['maximum_id']>65535:raise RuntimeError('Global IDs exceed uint16 PNG range; change export encoding explicitly')
    lookup=np.zeros(int(max(labels,default=0))+1,np.int32)
    for old,new in mapping.items():lookup[old]=new
    coarse=lookup[coarse];fine=lookup[fine];sp=cloud['point_superpoint_ids']
    np.savez_compressed(out/'coarse_instances.npz',point_coarse_ids=coarse[sp],superpoint_coarse_ids=coarse)
    if previous.exists():
        history=out/'history';history.mkdir(exist_ok=True)
        previous.replace(history/(utc().replace(':','').replace('-','')+'_fine_instances.npz'))
    np.savez_compressed(previous,points=cloud['points'],colors=cloud['colors'],point_superpoint_ids=sp,
                        point_fine_ids=fine[sp],superpoint_fine_ids=fine)
    dump(registry_path,registry);dump(out/'global_id_matching.json',{'matched':matched,'new':len(labels)-matched,'maximum_id':registry['maximum_id'],'threshold':.25,'method':'point support IoU on identical deterministic voxel cloud; IDs not reused'})
    return coarse,fine

def build_instances(out,doc,config):
    cloud=np.load(out/'superpoints.npz');progress(out,'proposals')
    records,support=make_proposals(out,doc,cloud,config)
    if not records:raise RuntimeError('No 3D-supported mask proposals')
    coarse,confidence,_=constrained_merge(records,support,out,config)
    progress(out,'fine',coarse_instances=int(coarse.max()))
    fine=refine(coarse,records,support,cloud,out,config);coarse,fine=stabilize(out,cloud,coarse,fine)
    sp=cloud['point_superpoint_ids'];sizes=cloud['superpoint_sizes']
    observed=np.zeros(len(sp),bool)
    for frame in doc['frames']:
        projection=np.load(frame_projection(out,frame));observed[projection['point_indices']]=True
    dump(out/'coverage.json',{'superpoint_coverage':float(np.mean(fine>0)),'point_coverage':float(np.mean(fine[sp]>0)),
       'visible_point_coverage':float(np.mean(fine[sp][observed]>0)) if observed.any() else 0.,
       'observed_points':int(observed.sum()),'unobserved_points':int((~observed).sum()),
       'assigned_superpoints':int(np.count_nonzero(fine)),'total_superpoints':len(fine),
       'assigned_points':int(np.count_nonzero(fine[sp])),'total_points':len(sp),
       'fine_instances':len(np.unique(fine[fine>0])),'global_id_source':'fine_instances.npz:point_fine_ids',
       'unknown_id':0,'units':'inferred_meter','frame':'canonical'})
    print('Instances',out.parent.name,read(out/'coverage.json'),flush=True)
