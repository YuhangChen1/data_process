from .common import ROOT,read,dump,sha,base,score,stable_ids,poly_iou,rel
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import open3d as o3d
from scene_pipeline.geometry import build_rayscene


def build_geometry(scene_id,out,old,registry):
    source=ROOT/'outputs'/scene_id;analysis=read(source/'analysis.json');floors=[];regions=[];mapping={};grids={}
    for fl in analysis['floors']:
        fid=f'f{fl["floor_id"]}';folder=source/f'floor_{fl["floor_id"]}';maps=np.load(folder/'maps.npz');labels=np.load(folder/'regions.npy');grids[fid]=(maps,labels)
        floor={'id':fid,'z_floor':fl['height'],'z_ceiling':fl['ceiling_height'],'navigable_area':fl['navigable_area'],'n_components':fl['connected_components'],'region_ids':[],**base('mesh:horizontal_plane_histogram+occupancy',.85),'map':rel(folder/'maps.npz',out),'body_radius':fl['body_radius_m']}
        floors.append(floor)
        features=read(folder/'regions.geojson')['features']
        for feature in features:
            p=feature['properties'];geom=feature['geometry'];largest=max(geom['coordinates'],key=lambda rings:len(rings[0]))
            region={'floor':fid,'polygon':largest[0],'geometry':geom,'area':p['area_m2_assuming_meters'],'center':p['center_xy'],'from':f'watershed:{p["region_id"]}',
                    'candidate_id':p['region_id'],'map_label':p['label'],'component':p['component'],'panorama':rel(folder/'panoramas'/(p['region_id']+'.jpg'),out),'object_ids':[],**base('mesh:distance_transform_watershed',.75)}
            regions.append(region)
    match=stable_ids(regions,old.get('regions',[]),'r',lambda a,b:poly_iou(a['geometry'],b['geometry']) if a['floor']==b['floor'] else 0,.5,registry)
    for r in regions:
        mapping[(r['floor'],r['map_label'])]=r
        next(f for f in floors if f['id']==r['floor'])['region_ids'].append(r['id'])
    cache=np.load(source/'mesh_cache.npz');R=np.array(analysis['world_to_canonical'])[:3,:3];v=(cache['vertices']@R.T).astype(np.float32);f=cache['faces']
    ray=build_rayscene(v,f)
    connections=[];issues=[]
    for fl in floors:
        fid=fl['id'];maps,labels=grids[fid];res=float(maps['resolution']);origin=maps['origin_xy'];contact={}
        raw_clear=ndi.distance_transform_edt(maps['support']&~maps['obstacles'])*res
        for dy,dx in [(0,1),(1,0)]:
            a=labels[:labels.shape[0]-dy or None,:labels.shape[1]-dx or None]
            b=labels[dy:,dx:];valid=(a>0)&(b>0)&(a!=b);yy,xx=np.where(valid)
            for y,x in zip(yy,xx):
                pair=tuple(sorted((int(a[y,x]),int(b[y,x]))));contact.setdefault(pair,[]).append([y+dy*.5,x+dx*.5])
        for pair,cells in contact.items():
            if len(cells)<2:continue
            cells=np.array(cells);middle=np.median(cells,axis=0);idx=np.argmin(np.linalg.norm(cells-middle,axis=1));rc=cells[idx]
            xy=origin+(rc[::-1]+.5)*res;grid=np.clip(np.rint(rc).astype(int),0,np.array(labels.shape)-1)
            width=max(.1,float(raw_clear[tuple(grid)])*2)
            positions=np.c_[origin+(cells[:,::-1]+.5)*res,np.full(len(cells),fl['z_floor']+.15)]
            if len(positions)>80:positions=positions[np.linspace(0,len(positions)-1,80).astype(int)]
            rays=np.c_[positions,np.tile([0,0,1],(len(positions),1))].astype(np.float32)
            depths=ray.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy();valid=depths[np.isfinite(depths)&(depths>.5)&(depths<6)]
            height=float(np.median(valid)+.15) if len(valid) else None
            lintel=height is not None and 1.8<=height<=2.35 and fl['z_ceiling']-fl['z_floor']-height>.15 and width<1.8
            label='door' if lintel else 'opening';conf=.65 if lintel else .55
            refs=sorted(mapping[(fid,int(k))]['id'] for k in pair)
            connections.append({'regions':refs,'floor':fid,'type_pred':{'label':label,'score':conf,'source':'mesh:shared_free_boundary+overhead_ray'},'position':[*xy,float(fl['z_floor'])],'width':width,'connected_in_map':True,'method':'shared_boundary','headroom_estimate':height,
                                'evidence':{'shared_boundary_grid_edges':len(cells),'grid_resolution':res,'width_method':'2x distance to raw obstacle/unknown at interface','door_test':'overhead mesh first hit 1.8..2.35 m; below ceiling by >0.15 m; width <1.8 m','caveat':'Watershed boundary may separate one open room; lintel test alone does not confirm a semantic doorway.'},**base('geometry',conf)})
        # Adjacent disconnected areas remain issues, not invented passable portals.
        candidates=[r for r in regions if r['floor']==fid]
        for i,a in enumerate(candidates):
            ac=np.argwhere(labels==a['map_label'])
            if not len(ac):continue
            tree=cKDTree(ac)
            for b in candidates[i+1:]:
                if a['component']==b['component']:continue
                bc=np.argwhere(labels==b['map_label'])
                if not len(bc):continue
                distance=float(tree.query(bc)[0].min()*res)
                if distance<=1.2:
                    issues.append({'type':'map_disconnected','ref':[a['id'],b['id']],'detail':f'候选区域边界相距约 {distance:.2f} m，但保守可通行图不连通；可能是墙、窄门、家具或缺失地面。','priority':.8,**base('occupancy:component_distance',.9)})
    # Sloping mesh clusters are only candidates; do not infer a stair connection from floor height alone.
    if len(floors)>1:
        from scene_pipeline.geometry import triangles
        c,n,area=triangles(v,f);slope=(n[:,2]>.25)&(n[:,2]<.985)&(area>1e-5)
        for lower,upper in zip(floors[:-1],floors[1:]):
            mask=slope&(c[:,2]>lower['z_floor']+.15)&(c[:,2]<upper['z_floor']-.15)
            pts=c[mask]
            if len(pts)>60000:pts=pts[np.linspace(0,len(pts)-1,60000).astype(int)]
            candidates=[]
            if len(pts)>20:
                cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts));cloud=cloud.voxel_down_sample(.08)
                p=np.asarray(cloud.points);labs=np.array(cloud.cluster_dbscan(eps=.22,min_points=8,print_progress=False))
                for lab in np.unique(labs[labs>=0]):
                    part=p[labs==lab];span=np.ptp(part,axis=0)
                    if span[2]<max(.8,(upper['z_floor']-lower['z_floor'])*.45) or np.linalg.norm(span[:2])<.7:continue
                    endpoint=[];distances=[]
                    for floor,z in [(lower,np.quantile(part[:,2],.05)),(upper,np.quantile(part[:,2],.95))]:
                        near=part[abs(part[:,2]-z)<.25,:2].mean(0)
                        target=min([r for r in regions if r['floor']==floor['id']],key=lambda r:np.linalg.norm(np.array(r['center'])-near))
                        maps,labels=grids[floor['id']];cells=np.argwhere(labels==target['map_label'])[:,::-1];xy=maps['origin_xy']+(cells+.5)*float(maps['resolution']);distance=float(cKDTree(xy).query(near)[0])
                        endpoint.append(target['id']);distances.append(distance)
                    candidates.append({'center':np.median(part,axis=0).tolist(),'span':span.tolist(),'region_endpoints':endpoint,'endpoint_distance':distances,'points':len(part)})
                    if max(distances)<1.2:
                        connections.append({'regions':sorted(endpoint),'floors':[lower['id'],upper['id']],'type_pred':{'label':'stairs','score':.45,'source':'mesh:ascending_sloped_cluster'},'position':np.median(part,axis=0).tolist(),'width':float(min(span[:2])),'connected_in_map':False,'method':'slope_geometry_candidate','evidence':candidates[-1],**base('mesh:slope_geometry',.45)})
            issues.append({'type':'cross_floor_connection_unverified','ref':[lower['id'],upper['id']],'detail':'楼层间几何仅做楼梯候选检测；尚无跨层可通行轨迹，不能保证连通。','slope_candidates':candidates,'priority':.85,**base('mesh:slope_geometry',.7)})
    connmatch=stable_ids(connections,old.get('connections',[]),'c',lambda a,b:1. if a['regions']==b['regions'] and np.linalg.norm(np.array(a['position'])-b['position'])<.8 else 0,.5,registry)
    for fl in floors:
        gap=fl['z_ceiling']-fl['z_floor']
        if gap<2.4 or gap>3.1:issues.append({'type':'scale_suspicious','ref':fl['id'],'detail':f'平面峰估计层内高度 {gap:.3f} 个单位，需复核物理尺度、夹层或天花板估计。','priority':.8,**base('mesh:ceiling_gap',.85)})
    return {'schema':'kosmoworld.scene_graph','schema_version':'0.1','scene_id':scene_id,'frame':'canonical','units':'inferred_meter',
            'geometry_source':{'analysis_sha256':sha(source/'analysis.json'),'analysis_path':rel(source/'analysis.json',out),'world_to_canonical':analysis['world_to_canonical'],'z_origin':'preserved canonical origin, floors are not individually rebased'},
            'scene':{'id':'s0',**base('geometry+vlm',0.)},'floors':floors,'regions':regions,'connections':connections,'objects':[],'texts':[],'issues':issues}, {'regions':match,'connections':connmatch}
