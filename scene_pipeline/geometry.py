"""Geometry inspection, conservative occupancy and annotation-only routes."""
from pathlib import Path
import json, heapq, math
import numpy as np
import open3d as o3d
import cv2
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from skimage.segmentation import watershed
from skimage.morphology import h_maxima, remove_small_objects, remove_small_holes
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plyfile import PlyData

RES=.05
RADIUS=.25

def save_json(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def load_mesh(path, cache):
    if cache.exists():
        d=np.load(cache); return d['vertices'],d['faces'],d['colors']
    mesh=o3d.io.read_triangle_mesh(str(path))
    v=np.asarray(mesh.vertices).astype(np.float32);f=np.asarray(mesh.triangles).astype(np.int32)
    color=np.asarray(mesh.vertex_colors).astype(np.float32)
    np.savez(cache,vertices=v,faces=f,colors=color)
    return v,f,color

def triangles(v,f):
    t=v[f];cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);area=np.linalg.norm(cross,axis=1)*.5
    return t.mean(1),cross/np.maximum(2*area[:,None],1e-12),area

def peak_table(c,n,a):
    bins=np.arange(c[:,2].min()-.05,c[:,2].max()+.1,.025)
    centers=(bins[:-1]+bins[1:])*.5
    hist=[];tables=[]
    for sign in [1,-1]:
        mask=n[:,2]*sign>.94
        h,_=np.histogram(c[mask,2],bins,weights=a[mask]);h=ndi.gaussian_filter1d(h,1.5)
        peaks,_=find_peaks(h,distance=8,prominence=max(.1,h.max()*.025))
        rows=[]
        for i in peaks:
            support=(abs(c[:,2]-centers[i])<.12)&mask
            rows.append({'height':float(centers[i]),'area':float(a[support].sum()),'peak':float(h[i])})
        tables.append(sorted(rows,key=lambda x:-x['area']))
        hist.append(h)
    return centers,hist,tables

def build_rayscene(v,f):
    scene=o3d.t.geometry.RaycastingScene(nthreads=6)
    mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v.astype(np.float32)),o3d.core.Tensor(f.astype(np.uint32)))
    scene.add_triangles(mesh)
    return scene

def alignment(v,f,ply_path,out):
    data=PlyData.read(str(ply_path),mmap=True)['vertex']
    rng=np.random.default_rng(20261014)
    indices=rng.choice(len(data),min(500000,len(data)),replace=False)
    pts=np.stack([data[k][indices] for k in ('x','y','z')],axis=1)
    op=1/(1+np.exp(-np.clip(data['opacity'][indices],-40,40)))
    scale=np.exp(np.clip(np.stack([data[k][indices] for k in ('scale_0','scale_1','scale_2')],1),-30,20)).max(1)
    lo,hi=np.quantile(v,[.002,.998],axis=0)
    in_box=((pts>=lo-.3)&(pts<=hi+.3)).all(1)
    keep=in_box&(op>.5)&(scale<.25)&np.isfinite(pts).all(1)
    pts=pts[keep];pts=pts[rng.choice(len(pts),min(len(pts),40000),replace=False)]
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f))
    o3d.utility.random.seed(42)
    target=mesh.sample_points_uniformly(120000)
    tp=np.asarray(target.points);tree=cKDTree(tp)
    moved=pts.copy().astype(float);T=np.eye(4)
    for _ in range(25):
        d,i=tree.query(moved,workers=4);mask=d<min(.3,float(np.quantile(d,.75)))
        if mask.sum()<100:break
        x=moved[mask];y=tp[i[mask]];xm=x.mean(0);ym=y.mean(0)
        u,s,vt=np.linalg.svd((x-xm).T@(y-ym));rot=vt.T@u.T
        if np.linalg.det(rot)<0:vt[-1]*=-1;rot=vt.T@u.T
        trans=ym-rot@xm;delta=np.eye(4);delta[:3,:3]=rot;delta[:3,3]=trans
        moved=moved@rot.T+trans;T=delta@T
        if np.linalg.norm(trans)<1e-5 and np.linalg.norm(rot-np.eye(3))<1e-5:break
    scene=build_rayscene(v,f)
    before=scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()
    after=scene.compute_distance(o3d.core.Tensor(moved.astype(np.float32))).numpy()
    d,i=tree.query(moved,workers=4);mask=d<np.quantile(d,.75)
    x=pts[mask].astype(float);y=tp[i[mask]];xx=x-x.mean(0);yy=y-y.mean(0)
    u,s,vt=np.linalg.svd(xx.T@yy);sign=np.ones(3);sign[-1]=np.linalg.det(vt.T@u.T)
    sim_scale=float(np.sum(s*sign)/np.sum(xx*xx))
    def stats(x):return {'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'p95':float(np.quantile(x,.95)),'rmse':float(np.sqrt(np.mean(x*x))),'fraction_under_5cm':float(np.mean(x<.05))}
    result={'gaussian_count':len(data),'sample_count':len(pts),'sample_inside_mesh_bbox_fraction':float(in_box.mean()),'sampling':'opacity > 0.5; largest Gaussian sigma < 0.25; mesh robust bbox +/- 0.3; deterministic random sample',
            'before_point_to_surface':stats(before),'after_icp_point_to_surface':stats(after),'icp_transform_original_world':T.tolist(),
            'icp_translation':float(np.linalg.norm(T[:3,3])),'icp_rotation_degrees':float(np.degrees(Rotation.from_matrix(T[:3,:3]).magnitude())),
            'local_similarity_scale':sim_scale,'transform_applied':False,
            'interpretation':'Local ICP is a consistency diagnostic, not independent proof of physical scale. Gaussian centers need not lie exactly on surfaces.'}
    result['same_frame_supported']=bool(result['before_point_to_surface']['median']<.05 and result['before_point_to_surface']['p90']<.15 and abs(sim_scale-1)<.03)
    save_json(out/'alignment.json',result)
    np.savez_compressed(out/'alignment_samples.npz',original=pts,surface_distance_before=before,surface_distance_after=after)
    return result

def quality(v,f,colors,out):
    c,n,a=triangles(v,f)
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    boundary=unique[counts==1]
    length=np.linalg.norm(v[boundary[:,0]]-v[boundary[:,1]],axis=1)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(boundary):
        ids,inv=np.unique(boundary,return_inverse=True);e=inv.reshape(-1,2)
        graph=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(ids),len(ids)))
        num,lab=connected_components(graph,directed=False)
        perimeters=np.bincount(lab[e[:,0]],weights=length,minlength=num)
        top=np.argsort(perimeters)[-30:][::-1]
        holes=[]
        for label in top:
            pts=v[ids[lab==label]]
            holes.append({'perimeter':float(perimeters[label]),'bbox_min':pts.min(0).tolist(),'bbox_max':pts.max(0).tolist(),'span':float(np.linalg.norm(pts.max(0)-pts.min(0)))})
    else:num=0;holes=[];perimeters=np.empty(0)
    result={'vertices':len(v),'triangles':len(f),'vertex_colors':len(colors)==len(v),'color_range':[float(colors.min()),float(colors.max())] if len(colors) else None,
            'surface_area':float(a.sum()),'degenerate_faces':int((a<1e-12).sum()),'boundary_edges':len(boundary),'boundary_length':float(length.sum()),'boundary_components':int(num),
            'boundary_components_perimeter_over_1m':int((perimeters>1).sum()),'nonmanifold_edges':int((counts>2).sum()),'watertight_edges':bool(np.all(counts==2)),
            'largest_open_boundaries':holes,'hole_caveat':'Open boundary components can be scan holes, door/window openings, or incomplete exterior. A large perimeter is not by itself a traversable opening.'}
    save_json(out/'mesh_quality.json',result)
    np.savez_compressed(out/'boundary_edges.npz',segments=v[boundary])
    return result

@njit(cache=True)
def raster_slab(vertices,faces,lo,hi,origin,res,ny,nx):
    """Project triangles clipped to height slab; include half-cell edge margin."""
    mask=np.zeros((ny,nx),np.uint8);height=np.full((ny,nx),-1e9,np.float32)
    margin=res*.71
    for face in faces:
        poly=np.zeros((8,3),np.float64);num=3
        for k in range(3):poly[k]=vertices[face[k]]
        if max(poly[0,2],poly[1,2],poly[2,2])<lo or min(poly[0,2],poly[1,2],poly[2,2])>hi:continue
        for plane in range(2):
            bound=lo if plane==0 else hi;buf=np.zeros((8,3),np.float64);count=0
            for k in range(num):
                p=poly[k];q=poly[(k+1)%num]
                pin=p[2]>=bound if plane==0 else p[2]<=bound
                qin=q[2]>=bound if plane==0 else q[2]<=bound
                if pin:buf[count]=p;count+=1
                if pin!=qin:
                    u=(bound-p[2])/(q[2]-p[2]);buf[count]=p+u*(q-p);count+=1
            poly=buf;num=count
            if num==0:break
        if num==0:continue
        minx=poly[:num,0].min();maxx=poly[:num,0].max();miny=poly[:num,1].min();maxy=poly[:num,1].max()
        x0=max(0,int(math.floor((minx-origin[0]-margin)/res)));x1=min(nx-1,int(math.floor((maxx-origin[0]+margin)/res)))
        y0=max(0,int(math.floor((miny-origin[1]-margin)/res)));y1=min(ny-1,int(math.floor((maxy-origin[1]+margin)/res)))
        for row in range(y0,y1+1):
            y=origin[1]+(row+.5)*res
            for col in range(x0,x1+1):
                x=origin[0]+(col+.5)*res;inside=False;near=False
                j=num-1
                for k in range(num):
                    ax=poly[j,0];ay=poly[j,1];bx=poly[k,0];by=poly[k,1]
                    if (ay>y)!=(by>y) and x<(bx-ax)*(y-ay)/(by-ay)+ax:inside=not inside
                    dx=bx-ax;dy=by-ay;den=dx*dx+dy*dy
                    u=max(0.,min(1.,((x-ax)*dx+(y-ay)*dy)/max(den,1e-20)))
                    if (x-ax-u*dx)**2+(y-ay-u*dy)**2<=margin*margin:near=True
                    j=k
                if inside or near:
                    mask[row,col]=1
                    z=poly[:num,2].mean()
                    if z>height[row,col]:height[row,col]=z
    return mask,height

def astar(mask,start,goal,clearance):
    if start==goal:return [start]
    queue=[(0.,start)];cost={start:0.};parent={};ny,nx=mask.shape
    steps=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while queue:
        _,p=heapq.heappop(queue)
        if p==goal:
            path=[p]
            while p!=start:p=parent[p];path.append(p)
            return path[::-1]
        for dy,dx in steps:
            q=(p[0]+dy,p[1]+dx)
            if not(0<=q[0]<ny and 0<=q[1]<nx and mask[q]):continue
            if dy and dx and (not mask[p[0]+dy,p[1]] or not mask[p[0],p[1]+dx]):continue
            new=cost[p]+math.hypot(dy,dx)*(1+.05/max(float(clearance[q]),.05))
            if new<cost.get(q,float('inf')):
                cost[q]=new;parent[q]=p;heapq.heappush(queue,(new+math.hypot(q[0]-goal[0],q[1]-goal[1]),q))
    raise RuntimeError('No A* path within same component')

def segment_valid(a,b,mask):
    # Dense sampling plus diagonal supercover: no shortcut through touching corners.
    n=max(2,int(np.linalg.norm(b-a)*5)+1);p=np.linspace(a,b,n)
    ij=np.rint(p).astype(int)
    if not mask[ij[:,0],ij[:,1]].all():return False
    for prev,cur in zip(ij[:-1],ij[1:]):
        if np.any(prev!=cur) and prev[0]!=cur[0] and prev[1]!=cur[1]:
            if not(mask[prev[0],cur[1]] and mask[cur[0],prev[1]]):return False
    return True

def smooth_route(path,mask):
    p=np.asarray(path,float)
    if len(p)<3:return p
    # Elastic smoothing retains collision checks for each segment.
    for _ in range(12):
        candidate=p.copy();candidate[1:-1]=.5*p[1:-1]+.25*(p[:-2]+p[2:])
        for i in range(1,len(p)-1):
            if segment_valid(p[i-1],candidate[i],mask) and segment_valid(candidate[i],p[i+1],mask):p[i]=candidate[i]
    return p

def plan_regions(nav,origin,res,out):
    clearance=ndi.distance_transform_edt(nav)*res
    comps,ncomp=ndi.label(nav)  # Four-connected components prohibit corner cutting.
    peaks=h_maxima(clearance,.22)&nav
    markers,n=ndi.label(peaks,structure=np.ones((3,3)))
    for comp in range(1,ncomp+1):
        mask=comps==comp
        if not (markers[mask]>0).any():
            ij=np.unravel_index(np.argmax(clearance*mask),nav.shape);n+=1;markers[ij]=n
    regions=watershed(-clearance,markers,mask=nav,connectivity=1)
    # Merge small watershed fragments only across shared free-space boundaries.
    for _ in range(3):
        counts=np.bincount(regions.ravel());changed=False
        for label in np.argsort(counts[1:])+1:
            if label>=len(counts) or counts[label]==0 or counts[label]*res*res>=.8:continue
            mask=regions==label
            neighbors=regions[ndi.binary_dilation(mask)&~mask]
            neighbors=neighbors[neighbors>0]
            if len(neighbors):
                adjacent=np.bincount(neighbors,minlength=len(counts));adjacent[label]=0
                destination=int(np.argmax(adjacent))
                regions[mask]=destination;counts[destination]+=counts[label];counts[label]=0;changed=True
        if not changed:break
    labels=np.unique(regions);labels=labels[labels>0]
    relabeled=np.zeros_like(regions)
    for index,label in enumerate(labels,1):relabeled[regions==label]=index
    regions=relabeled
    records=[];features=[]
    for label in np.unique(regions):
        if label==0:continue
        mask=regions==label;ij=np.unravel_index(np.argmax(clearance*mask),nav.shape)
        point=origin+(np.array(ij[::-1])+.5)*res
        contours,hierarchy=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        polygons=[]
        if hierarchy is not None:
            for k,contour in enumerate(contours):
                if hierarchy[0,k,3]!=-1:continue
                rings=[]
                indices=[k];child=hierarchy[0,k,2]
                while child!=-1:indices.append(child);child=hierarchy[0,child,0]
                for idx in indices:
                    coords=origin+(contours[idx][:,0,:]+.5)*res
                    if len(coords)<3:continue
                    rings.append(np.concatenate([coords,coords[:1]]).tolist())
                if rings:polygons.append(rings)
        record={'region_id':f'candidate_{int(label):03d}','label':int(label),'semantic_label':None,'status':'automatic_candidate_needs_annotation','area_m2_assuming_meters':float(mask.sum()*res*res),'component':int(comps[ij]),'center_xy':point.tolist(),'center_rc':list(map(int,ij)),'clearance_to_inflated_boundary_m':float(clearance[ij])}
        records.append(record);features.append({'type':'Feature','properties':record,'geometry':{'type':'MultiPolygon','coordinates':polygons}})
    save_json(out/'regions.geojson',{'type':'FeatureCollection','coordinate_system':'canonical local XY, not geographic longitude/latitude','features':features})
    save_json(out/'regions.json',records)
    np.save(out/'regions.npy',regions.astype(np.int32))
    routes=[]
    for comp in range(1,ncomp+1):
        candidates=[r for r in records if r['component']==comp]
        if not candidates:continue
        pending=candidates.copy();current=pending.pop(0);order=[current['region_id']];path=[tuple(current['center_rc'])]
        while pending:
            nxt=min(pending,key=lambda r:np.linalg.norm(np.array(r['center_xy'])-current['center_xy']))
            segment=astar(comps==comp,tuple(current['center_rc']),tuple(nxt['center_rc']),clearance)
            path+=segment[1:];order.append(nxt['region_id']);pending.remove(nxt);current=nxt
        smoothed=smooth_route(path,comps==comp)
        assert all(segment_valid(a,b,comps==comp) for a,b in zip(smoothed[:-1],smoothed[1:]))
        xy=origin+(smoothed[:,::-1]+.5)*res
        routes.append({'component':comp,'region_order':order,'xy':xy.tolist(),'length_m_assuming_meters':float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),'purpose':'annotation_only_not_evaluation_Q','collision_checked':True})
    save_json(out/'annotation_routes.json',routes)
    return records,regions,routes

def analyze(scene_dir,out):
    out.mkdir(parents=True,exist_ok=True)
    v,f,colors=load_mesh(scene_dir/'mesh_post1.obj',out/'mesh_cache.npz')
    print('Mesh loaded',scene_dir,flush=True)
    q=quality(v,f,colors,out)
    align=alignment(v,f,scene_dir/'gaussian.ply',out)
    # The signed floor/ceiling normals consistently establish -Y as up in these scans.
    R=np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
    canonical=v@R.T;c,n,a=triangles(canonical,f)
    centers,hist,tables=peak_table(c,n,a)
    strongest=tables[0][0];mask=(n[:,2]>.94)&(abs(c[:,2]-strongest['height'])<.12)
    ground_normal=np.sum(n[mask]*a[mask,None],axis=0);ground_normal/=np.linalg.norm(ground_normal)
    tilt=math.degrees(math.acos(np.clip(ground_normal[2],-1,1)))
    correction,_=Rotation.align_vectors([[0,0,1]],[ground_normal]);R=correction.as_matrix()@R
    canonical=(v@R.T).astype(np.float32);c,n,a=triangles(canonical,f)
    centers,hist,tables=peak_table(c,n,a)
    main=tables[0][0];floors=[]
    for candidate in tables[0]:
        if candidate['area']<max(5.,main['area']*.18):continue
        if any(abs(candidate['height']-old['height'])<1.9 for old in floors):continue
        ceilings=[row for row in tables[1] if 1.8<row['height']-candidate['height']<4.2 and row['area']>candidate['area']*.12]
        if not ceilings:continue
        ceiling=max(ceilings,key=lambda row:row['area'])
        floors.append({**candidate,'ceiling_height':ceiling['height'],'ceiling_gap':ceiling['height']-candidate['height']})
    floors.sort(key=lambda x:x['height'])
    if not floors:raise RuntimeError('No credible floor/ceiling pair')
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(centers,hist[0],label='upward-facing surfaces');ax.plot(centers,hist[1],label='downward-facing surfaces')
    for fl in floors:ax.axvline(fl['height'],color='green',ls='--');ax.axvline(fl['ceiling_height'],color='gray',ls=':')
    ax.set(xlabel='Canonical height (assumed meters)',ylabel='Surface area per 2.5 cm bin',title=scene_dir.name);ax.legend();fig.tight_layout();fig.savefig(out/'height_histogram.png',dpi=150);plt.close(fig)
    gap=[x['ceiling_gap'] for x in floors]
    report={'scene':str(scene_dir),'up_direction_original':(-R[2]).tolist(),'up_vector_original':R[2].tolist(),'dominant_up_axis_original':'-Y','leveling_correction_degrees':tilt,
            'world_to_canonical':np.block([[R,np.zeros((3,1))],[np.zeros((1,3)),np.ones((1,1))]]).tolist(),
            'unit_status':'meters_inferred_not_independently_verified','coordinate_scale_applied':1.,'ceiling_gaps':gap,'typical_residential_height_consistent':all(2.4<=g<=3.1 for g in gap),
            'floor_count':len(floors),'upward_plane_candidates':tables[0],'downward_plane_candidates':tables[1],
            'alignment':align,'mesh_quality':q,'floors':[],
            'limitations':['No metric reference object or camera calibration was provided; all meter thresholds assume 1 unit = 1 meter.','Plane peaks alone cannot distinguish a loft from an elevated platform. Candidate levels require visual review.','Unknown/unobserved floor is not considered traversable.','2D occupancy does not certify stairs, handrails, slopes, or physical safety.']}
    report.pop('up_direction_original')
    lo=np.quantile(canonical[:,:2],.001,axis=0)-.2;hi=np.quantile(canonical[:,:2],.999,axis=0)+.2
    origin=np.floor(lo/RES)*RES;nx,ny=np.ceil((hi-origin)/RES).astype(int)
    for index,fl in enumerate(floors):
        folder=out/f'floor_{index}';folder.mkdir(exist_ok=True);z=fl['height']
        support_idx=(n[:,2]>.85)&(abs(c[:,2]-z)<.20)
        observed,height=raster_slab(canonical,f[support_idx],z-.22,z+.22,origin,RES,ny,nx)
        obstacles,_=raster_slab(canonical,f,z+.10,z+1.80,origin,RES,ny,nx)
        observed=observed.astype(bool);obstacles=obstacles.astype(bool)
        # Only repair sub-cell cracks/tiny holes; record inferred support separately.
        support=ndi.binary_closing(observed,iterations=1)
        support=remove_small_holes(support,area_threshold=16)
        support=remove_small_objects(support,min_size=40)
        clearance=ndi.distance_transform_edt(support&~obstacles)*RES
        nav=(clearance>=RADIUS)
        nav=remove_small_objects(nav,min_size=int(.4/RES**2))
        raw_occupancy=np.full((ny,nx),-1,np.int8);raw_occupancy[support]=0;raw_occupancy[obstacles]=100
        occupancy=np.full((ny,nx),-1,np.int8);occupancy[support]=100;occupancy[obstacles]=100;occupancy[nav]=0
        height[~observed]=np.nan
        np.savez_compressed(folder/'maps.npz',observed_floor=observed,inferred_floor=support&~observed,support=support,obstacles=obstacles,navigable=nav,clearance=clearance,floor_height=height,occupancy=occupancy,raw_occupancy=raw_occupancy,origin_xy=origin,resolution=RES,floor_z=z)
        np.save(folder/'occupancy.npy',occupancy)
        # Image rows grow down; flip canonical +Y-up grids for PNG/PGM.
        pgm=np.full((ny,nx),205,np.uint8);pgm[occupancy==100]=0;pgm[occupancy==0]=254
        cv2.imwrite(str(folder/'occupancy.pgm'),np.flipud(pgm))
        (folder/'occupancy.yaml').write_text(f'image: occupancy.pgm\nresolution: {RES}\norigin: [{origin[0]}, {origin[1]}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n# floor z: {z}; meters are inferred; obstacles already inflated by {RADIUS} m\n')
        regions,labels,routes=plan_regions(nav,origin,RES,folder)
        image=np.full((ny,nx,3),225,np.uint8);image[support]=[200,215,205];image[obstacles]=[55,55,55];image[nav]=[220,245,220]
        palette=np.random.default_rng(42).integers(70,230,(max(1,len(regions))+100,3),dtype=np.uint8)
        for region in regions:image[labels==region['label']]=palette[region['label']%len(palette)]
        for route in routes:
            pts=np.rint((np.asarray(route['xy'])-origin)/RES-.5).astype(np.int32)
            if len(pts)>1:cv2.polylines(image,[pts],False,(255,255,255),1,cv2.LINE_AA)
        image=np.flipud(image).copy()
        for region in regions:
            y,x=region['center_rc'];cv2.putText(image,str(region['label']),(x,ny-1-y),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)
        cv2.imwrite(str(folder/'regions_and_routes.png'),image[:,:,::-1])
        flreport={**fl,'floor_id':index,'origin_xy':origin.tolist(),'resolution':RES,'grid_shape':[int(ny),int(nx)],'body_radius_m':RADIUS,'obstacle_slab_m':[.1,1.8],'camera_height_m':1.5,
                  'observed_floor_area':float(observed.sum()*RES**2),'inferred_floor_area':float((support&~observed).sum()*RES**2),'navigable_area':float(nav.sum()*RES**2),'regions':len(regions),'connected_components':len(routes),
                  'headroom_review_required':fl['ceiling_gap']<2.4,'coordinate_convention':'NPZ [row,col] maps to canonical XY = origin + ([col,row]+0.5)*resolution; PNG is vertically flipped.'}
        save_json(folder/'floor.json',flreport);report['floors'].append(flreport)
        print('Floor',index,'z',round(z,3),'nav',round(flreport['navigable_area'],2),'regions',len(regions),'components',len(routes),flush=True)
    save_json(out/'analysis.json',report)
    return report
