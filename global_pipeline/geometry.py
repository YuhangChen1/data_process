"""Colored kNN FH superpoints and camera/depth-consistent point projections."""
from .common import ROOT,read,dump,sha,progress,fingerprint,frame_projection
import numpy as np,cv2
from concurrent.futures import ThreadPoolExecutor,as_completed
from scipy.spatial import cKDTree
from numba import njit

@njit(cache=True)
def find(parent,i):
    while parent[i]!=i:
        parent[i]=parent[parent[i]];i=parent[i]
    return i

@njit(cache=True)
def fh(n,edges,weights,positions,k,min_size,max_size,max_extent):
    parent=np.arange(n);size=np.ones(n,np.int32);internal=np.zeros(n,np.float32)
    lower=positions.copy();upper=positions.copy()
    for sweep in range(2):
        for e in range(len(edges)):
            a=find(parent,edges[e,0]);b=find(parent,edges[e,1]);w=weights[e]
            if a==b or size[a]+size[b]>max_size:continue
            if max(max(upper[a,0],upper[b,0])-min(lower[a,0],lower[b,0]),max(upper[a,1],upper[b,1])-min(lower[a,1],lower[b,1]),max(upper[a,2],upper[b,2])-min(lower[a,2],lower[b,2]))>max_extent:continue
            acceptable=w<=min(internal[a]+k/size[a],internal[b]+k/size[b]) if sweep==0 else min(size[a],size[b])<min_size
            if not acceptable:continue
            if size[a]<size[b]:a,b=b,a
            parent[b]=a;size[a]+=size[b];internal[a]=max(w,internal[a],internal[b])
            for j in range(3):lower[a,j]=min(lower[a,j],lower[b,j]);upper[a,j]=max(upper[a,j],upper[b,j])
    for i in range(n):parent[i]=find(parent,i)
    return parent

def build_superpoints(scene,out,config):
    target=out/'superpoints.npz';meta=out/'superpoints.json';source=ROOT/'outputs'/scene
    signature=fingerprint({'mesh':sha(source/'mesh_cache.npz'),'analysis':sha(source/'analysis.json'),
        'parameters':{k:config[k] for k in ['voxel_size','knn','edge_radius','spatial_weight','color_weight','fh_k','superpoint_min_size','superpoint_max_size','superpoint_max_extent']}})
    if target.exists() and meta.exists() and read(meta).get('fingerprint')==signature:return np.load(target)
    cache=np.load(source/'mesh_cache.npz');transform=np.array(read(source/'analysis.json')['world_to_canonical'])
    vertices=(cache['vertices']@transform[:3,:3].T+transform[:3,3]).astype(np.float32)
    color=cache['colors'].astype(np.float32)
    cells=np.floor(vertices/config['voxel_size']).astype(np.int32)
    unique,inverse,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
    positions=np.stack([np.bincount(inverse,weights=vertices[:,a])/counts for a in range(3)],1).astype(np.float32)
    colors=np.stack([np.bincount(inverse,weights=color[:,a])/counts for a in range(3)],1).astype(np.float32)
    tree=cKDTree(positions);dist,neighbor=tree.query(positions,k=config['knn']+1,workers=4)
    a=np.repeat(np.arange(len(positions)),config['knn']);b=neighbor[:,1:].ravel();distance=dist[:,1:].ravel()
    keep=(a<b)&(distance<=config['edge_radius']);a=a[keep];b=b[keep];distance=distance[keep]
    weight=(config['spatial_weight']*distance+config['color_weight']*np.linalg.norm(colors[a]-colors[b],axis=1)).astype(np.float32)
    order=np.argsort(weight);edges=np.stack([a,b],1)[order].astype(np.int32);weight=weight[order]
    parent=fh(len(positions),edges,weight,positions,config['fh_k'],config['superpoint_min_size'],config['superpoint_max_size'],config['superpoint_max_extent'])
    _,sp=np.unique(parent,return_inverse=True);sp=sp.astype(np.int32);sizes=np.bincount(sp)
    centers=np.stack([np.bincount(sp,weights=positions[:,j])/sizes for j in range(3)],1).astype(np.float32)
    edge_sp=sp[edges];edge_sp=np.unique(np.sort(edge_sp[edge_sp[:,0]!=edge_sp[:,1]],axis=1),axis=0)
    np.savez_compressed(target,points=positions,colors=colors,point_superpoint_ids=sp,
                        centroids=centers,superpoint_sizes=sizes.astype(np.int32),adjacency=edge_sp.astype(np.int32))
    dump(meta,{'fingerprint':signature,'method':'Felzenszwalb-Huttenlocher colored 3D kNN graph; merge size and extent caps',
        'points':len(positions),'superpoints':len(sizes),'edges':len(edges),'adjacency_edges':len(edge_sp),
        'size_quantiles':np.quantile(sizes,[0,.5,.9,.99,1]).tolist(),'max_size':int(sizes.max()),
        'max_extent_cap':config['superpoint_max_extent'],'frame':'canonical','units':'inferred_meter','parameters':config,
        'source_mesh_vertices':len(vertices),'superpoints_are_instances':False})
    progress(out,'superpoints',points=len(positions),superpoints=len(sizes))
    print('Superpoints',scene,len(positions),'points',len(sizes),'regions',flush=True)
    return np.load(target)

def project_points(points,frame,depth,config):
    pose=np.array(frame['camera_to_canonical']);K=np.array(frame['K']);h,w=depth.shape
    camera=(points-pose[:3,3])@pose[:3,:3];z=camera[:,2]
    valid=(z>.1)&(z<40)
    candidate=np.flatnonzero(valid);local=camera[valid];zz=z[valid]
    u=np.floor(K[0,0]*local[:,0]/zz+K[0,2]).astype(np.int32);v=np.floor(K[1,1]*local[:,1]/zz+K[1,2]).astype(np.int32)
    inside=(u>=0)&(u<w)&(v>=0)&(v<h);candidate=candidate[inside];zz=zz[inside];u=u[inside];v=v[inside]
    if not len(candidate):return np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32)
    pixel=v*w+u
    order=np.lexsort((zz,pixel));pixel_sorted=pixel[order]
    first=np.r_[True,np.diff(pixel_sorted)!=0];take=order[first]
    candidate=candidate[take];pixel=pixel[take];zz=zz[take];reference=depth.ravel()[pixel]
    tolerance=config['depth_abs_tolerance']+config['depth_relative_tolerance']*reference
    good=(reference>.1)&np.isfinite(reference)&(np.abs(zz-reference)<=tolerance)
    return candidate[good].astype(np.int32),pixel[good].astype(np.int32),zz[good].astype(np.float32)

def projections(out,doc,cloud,config):
    signature=fingerprint({'cloud':read(out/'superpoints.json')['fingerprint'],'tolerances':[config['depth_abs_tolerance'],config['depth_relative_tolerance']]})
    total=0;done=0;points=cloud['points']
    def one(frame):
        path=frame_projection(out,frame);path.parent.mkdir(parents=True,exist_ok=True)
        frame_signature=fingerprint({'geometry':signature,'pose':frame['camera_to_canonical'],
            'intrinsics':frame['K'],'depth_sha256':sha(ROOT/frame['depth'])})
        if path.exists():
            data=np.load(path)
            if str(data['fingerprint'])==frame_signature:return len(data['point_indices'])
        depth=cv2.imread(str(ROOT/frame['depth']),cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
        indices,pixels,z=project_points(points,frame,depth,config)
        np.savez_compressed(path,point_indices=indices,pixels=pixels,camera_z=z,fingerprint=frame_signature)
        return len(indices)
    # Frames have independent immutable inputs and independent output files.
    # Keep the pool bounded to limit temporary full-cloud camera arrays.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(one,frame) for frame in doc['frames']]):
            total+=future.result();done+=1
            if done%40==0 or done==len(doc['frames']):progress(out,'projection',projected_frames=done,frames_total=len(doc['frames']))
    dump(out/'projection_summary.json',{'frames':len(doc['frames']),'depth_consistent_point_observations':total,
        'mean_visible_points':total/max(1,len(doc['frames'])),'zbuffer':'nearest sampled 3D point per pixel',
        'depth_check':'abs(camera_z - mesh_z) <= absolute + relative * mesh_z','fingerprint':signature})
