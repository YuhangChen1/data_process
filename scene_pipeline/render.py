"""Real Gaussian splat RGB/expected depth and mesh raycast annotation assets."""
from pathlib import Path
import json, math, subprocess, time
import numpy as np
import torch
import gsplat
import open3d as o3d
import cv2
from plyfile import PlyData
from scipy import ndimage as ndi
from .geometry import save_json,load_mesh,build_rayscene,segment_valid


def camera(center,forward,up=np.array([0.,0.,1.])):
    forward=np.asarray(forward,float);forward/=np.linalg.norm(forward)
    right=np.cross(forward,up);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    pose=np.eye(4);pose[:3,:3]=np.stack([right,down,forward],axis=1);pose[:3,3]=center
    return pose

class Renderer:
    def __init__(self,source,out):
        self.out=out;self.report=json.loads((out/'analysis.json').read_text())
        self.world_to_canonical=np.array(self.report['world_to_canonical'],np.float32)
        self.v,self.f,_=load_mesh(source/'mesh_post1.obj',out/'mesh_cache.npz')
        self.mesh=build_rayscene(self.v,self.f)
        p=PlyData.read(str(source/'gaussian.ply'),mmap=True)['vertex']
        xyz=np.stack([p[k] for k in ['x','y','z']],1)
        scales=np.exp(np.clip(np.stack([p[k] for k in ['scale_0','scale_1','scale_2']],1),-30,20))
        opacity=1/(1+np.exp(-np.clip(p['opacity'],-40,40)))
        lo,hi=np.quantile(self.v,[.001,.999],axis=0)
        keep=np.isfinite(xyz).all(1)&np.isfinite(scales).all(1)&((xyz>=lo-.75)&(xyz<=hi+.75)).all(1)&(opacity>.005)&(scales.max(1)<2.)
        self.canonical=xyz[keep]@self.world_to_canonical[:3,:3].T
        self.args={'means':torch.as_tensor(xyz[keep].copy(),device='cuda'),'scales':torch.as_tensor(scales[keep].copy(),device='cuda'),'opacities':torch.as_tensor(opacity[keep].copy(),device='cuda'),
                   'quats':torch.as_tensor(np.stack([p[k][keep] for k in ['rot_0','rot_1','rot_2','rot_3']],1),device='cuda'),
                   'colors':torch.as_tensor(np.clip(.5+.28209479177387814*np.stack([p[k][keep] for k in ['f_dc_0','f_dc_1','f_dc_2']],1),0,1),device='cuda')}
        self.args['quats']=torch.nn.functional.normalize(self.args['quats'],dim=-1)
        self.info={'renderer':'gsplat '+gsplat.__version__,'source_gaussians':len(p),'rendered_gaussians':int(keep.sum()),'spherical_harmonics_degree':0,
                   'filter':'finite values, opacity > 0.005, max sigma < 2 world units, robust mesh bbox expanded by 0.75; original files unchanged',
                   'depth_semantics':'3DGS: alpha-normalized expected camera-Z depth; mesh: first-hit camera-Z depth. These are different estimators, especially at translucent surfaces and discontinuities.',
                   'camera_convention':'OpenCV camera: right +X, down +Y, forward +Z. Poses are camera-to-world in canonical leveled Z-up coordinates.'}
        save_json(out/'render_info.json',self.info)
        self.active=self.args

    def clip_top(self,z,cut):
        mask=(self.canonical[:,2]>=z-.30)&(self.canonical[:,2]<=cut)
        t=torch.as_tensor(mask,device='cuda');self.active={k:v[t] for k,v in self.args.items()}

    def unclip(self):self.active=self.args

    @torch.inference_mode()
    def rgbd(self,pose,width,height,fov=75.,ortho_extent=None):
        view=np.linalg.inv(pose)@self.world_to_canonical
        if ortho_extent is None:
            focal=width/(2*np.tan(np.deg2rad(fov/2)));fx=fy=focal;model='pinhole'
        else:fx=width/ortho_extent[0];fy=height/ortho_extent[1];model='ortho'
        K=np.array([[fx,0,width/2],[0,fy,height/2],[0,0,1]],np.float32)
        img,alpha,_=gsplat.rasterization(**self.active,viewmats=torch.as_tensor(view[None].astype(np.float32),device='cuda'),Ks=torch.as_tensor(K[None],device='cuda'),width=width,height=height,
                                       near_plane=.03,far_plane=100.,packed=True,render_mode='RGB+ED',camera_model=model,
                                       radius_clip=.4)
        img=img[0].cpu().numpy();alpha=alpha[0,:,:,0].cpu().numpy()
        return np.uint8(np.clip((img[:,:,:3]+1-alpha[:,:,None])*255,0,255)),img[:,:,3],alpha,K

    def mesh_depth(self,pose,width,height,K):
        yy,xx=np.mgrid[:height,:width];dirs=np.stack([(xx+.5-K[0,2])/K[0,0],(yy+.5-K[1,2])/K[1,1],np.ones_like(xx)],-1)
        original_pose=np.linalg.inv(self.world_to_canonical)@pose
        directions=dirs@original_pose[:3,:3].T
        origins=np.broadcast_to(original_pose[:3,3],directions.shape)
        rays=np.concatenate([origins,directions],-1).astype(np.float32)
        depth=self.mesh.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy();depth[~np.isfinite(depth)]=0
        return depth

    def top_mesh_height(self,origin,extent,zcut,width,height):
        yy,xx=np.mgrid[:height,:width]
        origins=np.stack([origin[0]+(xx+.5)*extent[0]/width,origin[1]+extent[1]-(yy+.5)*extent[1]/height,np.full_like(xx,zcut,dtype=float)],-1)
        inv=np.linalg.inv(self.world_to_canonical)
        origins=origins@inv[:3,:3].T;directions=np.broadcast_to(inv[:3,:3]@np.array([0.,0.,-1.]),origins.shape)
        result=self.mesh.cast_rays(o3d.core.Tensor(np.concatenate([origins,directions],-1).astype(np.float32)))['t_hit'].numpy()
        return np.where(np.isfinite(result),zcut-result,np.nan)


def depth_image(d,valid=None,max_depth=8.):
    valid=(d>0)&np.isfinite(d) if valid is None else valid
    img=cv2.applyColorMap(np.uint8(np.clip(d/max_depth,0,1)*255),cv2.COLORMAP_TURBO)[:,:,::-1]
    img[~valid]=255
    return np.ascontiguousarray(img)


def write_rgb(path,image):cv2.imwrite(str(path),image[:,:,::-1])


def topdown(renderer,folder,fl,maps):
    origin=np.asarray(fl['origin_xy']);ny,nx=fl['grid_shape'];res=fl['resolution'];extent=np.array([nx,ny])*res
    w=max(128,int(round(1200*nx/max(nx,ny))));h=max(128,int(round(1200*ny/max(nx,ny))))
    z=fl['height'];cut=min(z+1.8,fl['ceiling_height']-.20)
    pose=camera(np.r_[origin+extent/2,cut+2.],[0,0,-1],np.array([0.,1.,0.]))
    renderer.clip_top(z,cut);rgb,gsdepth,alpha,K=renderer.rgbd(pose,w,h,ortho_extent=extent);renderer.unclip()
    meshheight=renderer.top_mesh_height(origin,extent,cut,w,h)
    valid=np.isfinite(meshheight)&(meshheight>=z-.35)
    height_rgb=depth_image(np.nan_to_num(meshheight-z),valid,max_depth=1.8)
    overlay=rgb.copy();overlay[valid]=np.uint8(.72*rgb[valid]+.28*height_rgb[valid])
    regions=json.loads((folder/'regions.json').read_text())
    for r in regions:
        x,y=np.array(r['center_xy']);px=int((x-origin[0])/extent[0]*w);py=int((1-(y-origin[1])/extent[1])*h)
        cv2.circle(overlay,(px,py),10,(255,255,255),-1);cv2.putText(overlay,str(r['label']),(px+12,py+5),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2,cv2.LINE_AA)
    write_rgb(folder/'topdown_3dgs.png',rgb);write_rgb(folder/'mesh_height.png',height_rgb);write_rgb(folder/'annotation_overlay.png',overlay)
    np.savez_compressed(folder/'topdown.npz',mesh_height=meshheight,gs_expected_depth=gsdepth,gs_alpha=alpha,camera_to_canonical=pose,K=K,origin_xy=origin,extent_xy=extent,ceiling_clip_z=cut)
    save_json(folder/'topdown_camera.json',{'camera_model':'orthographic','width':w,'height':h,'K':K.tolist(),'camera_to_canonical':pose.tolist(),'origin_xy':origin.tolist(),'extent_xy':extent.tolist(),'ceiling_clip_z':cut,
        'pixel_to_canonical':'x=origin_x+(u+0.5)*extent_x/width; y=origin_y+(height-v-0.5)*extent_y/height','gaussian_clip':'center-based height clipping; large ellipsoids may cross clip plane'})


def validate_depth(renderer,folder,fl,regions):
    out=folder/'alignment_views';out.mkdir(exist_ok=True);stats=[];previews=[]
    for i,r in enumerate(sorted(regions,key=lambda row:-row['area_m2_assuming_meters'])[:min(3,len(regions))]):
        center=np.r_[r['center_xy'],fl['height']+1.5]
        for j,yaw in enumerate([0,math.pi/2,math.pi,3*math.pi/2]):
            pose=camera(center,[math.cos(yaw),math.sin(yaw),0])
            rgb,gd,alpha,K=renderer.rgbd(pose,480,320)
            md=renderer.mesh_depth(pose,480,320,K)
            valid=(alpha>.8)&(md>.05)&(gd>.05)&(md<20)&(gd<20)
            delta=np.abs(md-gd);vals=delta[valid]
            row={'region_id':r['region_id'],'yaw_degrees':j*90,'valid_fraction':float(valid.mean()),'valid_pixels':int(valid.sum()),'median_abs_depth_difference':float(np.median(vals)) if len(vals) else None,'p90_abs_depth_difference':float(np.quantile(vals,.9)) if len(vals) else None,'fraction_under_10cm':float(np.mean(vals<.1)) if len(vals) else None}
            stats.append(row)
            error=depth_image(delta,valid,max_depth=.3)
            tiles=[rgb,depth_image(md),depth_image(gd,(alpha>.8)&(gd>0)),error]
            for tile,title in zip(tiles,['3DGS RGB','Mesh Z-depth','3DGS expected Z-depth','Abs difference (0 - 0.3 m)']):
                cv2.rectangle(tile,(0,0),(480,27),(255,255,255),-1);cv2.putText(tile,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1,cv2.LINE_AA)
            panel=np.concatenate(tiles,axis=1);write_rgb(out/f'view_{i}_{j}.jpg',panel)
            np.savez_compressed(out/f'view_{i}_{j}.npz',mesh_depth=md,gs_expected_depth=gd,alpha=alpha,camera_to_canonical=pose,K=K)
            if j==0:previews.append(panel)
    if previews:write_rgb(out/'contact_sheet.jpg',np.concatenate(previews,axis=0))
    save_json(folder/'depth_alignment.json',{'views':stats,'warning':'Expected Gaussian depth and first-hit mesh depth differ at transparent windows, holes and boundaries. Assess jointly with point-to-surface and ICP diagnostics.'})
    return stats


def panorama(renderer,center,path,width=2048,height=1024,face_size=512):
    # Longitude 0 faces canonical +X; positive longitude rotates toward +Y.
    dirs=[([1,0,0],[0,0,1]),([-1,0,0],[0,0,1]),([0,1,0],[0,0,1]),([0,-1,0],[0,0,1]),([0,0,1],[0,1,0]),([0,0,-1],[0,1,0])]
    yy,xx=np.mgrid[:height,:width];yaw=((xx+.5)/width*2-1)*np.pi;pitch=(.5-(yy+.5)/height)*np.pi
    rays=np.stack([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)],-1).astype(np.float32)
    axes=np.array([a for a,b in dirs],np.float32);assign=np.argmax(rays@axes.T,axis=-1)
    result=np.zeros((height,width,3),np.uint8);faceposes=[]
    for k,(direction,up) in enumerate(dirs):
        pose=camera(center,direction,np.array(up,float));rgb,_,_,K=renderer.rgbd(pose,face_size,face_size,fov=90)
        local=rays@pose[:3,:3];den=np.maximum(local[:,:,2],1e-8)
        u=(local[:,:,0]/den*K[0,0]+face_size/2-.5).astype(np.float32);v=(local[:,:,1]/den*K[1,1]+face_size/2-.5).astype(np.float32)
        # Bound unused face coordinates before remap to avoid float-to-int overflows.
        u=np.clip(u,-1,face_size);v=np.clip(v,-1,face_size)
        sampled=cv2.remap(rgb,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        result[assign==k]=sampled[assign==k];faceposes.append(pose.tolist())
    write_rgb(path,result)
    return faceposes


def path_frames(route,floor_z,nav,origin,res,fps=8,speed=.8):
    xy=np.asarray(route['xy'],float)
    if len(xy)<2 or np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()<.15:
        yaw=np.linspace(0,2*np.pi,4*fps,endpoint=False)
        return [camera(np.r_[xy[0],floor_z+1.5],[np.cos(t),np.sin(t),0]) for t in yaw]
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    unique=np.r_[True,np.diff(distance)>1e-8];distance=distance[unique];xy=xy[unique]
    positions=np.linspace(0,distance[-1],max(2,int(np.ceil(distance[-1]/speed*fps))+1))
    coords=np.stack([np.interp(positions,distance,xy[:,a]) for a in range(2)],1)
    # Preserve corridor bends if frame interpolation chords would leave the mask.
    for a,b in zip(coords[:-1],coords[1:]):
        ar=((a-origin)/res-.5)[::-1];br=((b-origin)/res-.5)[::-1]
        if not segment_valid(ar,br,nav):
            coords=xy;break
    ahead=np.minimum(positions+.5,distance[-1]);behind=np.maximum(positions-.15,0)
    if len(coords)!=len(positions):
        direction=np.gradient(coords,axis=0)
    else:direction=np.stack([np.interp(ahead,distance,xy[:,a])-np.interp(behind,distance,xy[:,a]) for a in range(2)],1)
    yaw=np.unwrap(np.arctan2(direction[:,1],direction[:,0]));yaw=ndi.gaussian_filter1d(yaw,2)
    return [camera(np.r_[point,floor_z+1.5],[np.cos(t),np.sin(t),0]) for point,t in zip(coords,yaw)]


def video(renderer,folder,fl,maps,route):
    name=f'component_{route["component"]:03d}';out=folder/'annotation_walkthrough'/name;out.mkdir(parents=True,exist_ok=True)
    poses=path_frames(route,fl['height'],maps['navigable'],maps['origin_xy'],fl['resolution']);width,height=960,540;fps=8
    args=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p']
    rgbwriter=subprocess.Popen(args+[str(out/'rgb.mp4')],stdin=subprocess.PIPE)
    depthwriter=subprocess.Popen(args+[str(out/'depth_preview.mp4')],stdin=subprocess.PIPE)
    depthdir=out/'mesh_depth_mm';depthdir.mkdir(exist_ok=True)
    originals=[]
    for old_depth in depthdir.glob('*.png'):old_depth.unlink()
    try:
        for i,pose in enumerate(poses):
            rgb,gd,alpha,K=renderer.rgbd(pose,width,height);depth=renderer.mesh_depth(pose,width,height,K)
            millimeters=np.uint16(np.clip(np.rint(depth*1000),0,65535));millimeters[depth>65.535]=0
            cv2.imwrite(str(depthdir/f'{i:05d}.png'),millimeters)
            rgbwriter.stdin.write(rgb.tobytes());depthwriter.stdin.write(depth_image(depth).tobytes())
            if i in [0,len(poses)//2,len(poses)-1]:write_rgb(out/f'frame_{i:05d}.jpg',rgb)
            originals.append(np.linalg.inv(renderer.world_to_canonical)@pose)
            if i%80==0:print(name,'frame',i,'/',len(poses),flush=True)
    finally:
        rgbwriter.stdin.close();depthwriter.stdin.close()
        r1=rgbwriter.wait();r2=depthwriter.wait()
    if r1 or r2:raise RuntimeError('FFmpeg failed')
    np.savez_compressed(out/'cameras.npz',camera_to_canonical=np.asarray(poses),camera_to_original=np.asarray(originals),K=K,timestamps=np.arange(len(poses))/fps)
    info={'purpose':'annotation_only_not_evaluation_Q','frames':len(poses),'fps':fps,'width':width,'height':height,'camera_height_above_floor':1.5,'camera_model':'pinhole','horizontal_fov_degrees':75,
          'region_order':route['region_order'],'component':route['component'],'trajectory_collision_checked':True,'depth':'16-bit PNG millimeters, 0=invalid; first-hit camera-Z mesh depth; meter assumption inherited from analysis',
          'rgb':'Actual anisotropic Gaussian splatting; SH degree 0 available in source','note':'Each connected component is a separate clip; there are no teleports through obstacles and no cross-floor stair traversal.'}
    save_json(out/'metadata.json',info)
    return info


def render_scene(source,out,preview=False,walkthroughs=True):
    renderer=Renderer(source,out);result=[]
    for fl in renderer.report['floors']:
        folder=out/f'floor_{fl["floor_id"]}';maps=np.load(folder/'maps.npz')
        regions=json.loads((folder/'regions.json').read_text());routes=json.loads((folder/'annotation_routes.json').read_text())
        topdown(renderer,folder,fl,maps);print('Topdown complete',folder,flush=True)
        stats=validate_depth(renderer,folder,fl,regions)
        if preview:return
        panodir=folder/'panoramas';panodir.mkdir(exist_ok=True);panometa=[]
        for region in regions:
            center=np.r_[region['center_xy'],fl['height']+1.5]
            faceposes=panorama(renderer,center,panodir/(region['region_id']+'.jpg'))
            panometa.append({'region_id':region['region_id'],'center_canonical':center.tolist(),'width':2048,'height':1024,'face_poses_camera_to_canonical':faceposes,'semantic_description':None,'description_status':'ready_for_VLM_or_human_annotation'})
            print('Panorama',folder,region['region_id'],flush=True)
        save_json(panodir/'metadata.json',{'projection':'equirectangular','yaw_at_center':'+X','yaw_positive':'toward +Y','pitch_positive':'+Z','regions':panometa})
        clips=[video(renderer,folder,fl,maps,route) for route in routes] if walkthroughs else []   # A* only when the scan has no trajectory
        result.append({'floor_id':fl['floor_id'],'panoramas':len(panometa),'walkthroughs':clips,'depth_comparison_views':len(stats)})
    save_json(out/'render_complete.json',{'floors':result})
    del renderer;torch.cuda.empty_cache()
