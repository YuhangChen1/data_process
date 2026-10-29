"""Actual 3DGS sweeps at every navigable region center to close view coverage gaps."""
from .common import ROOT,read,dump,progress
import argparse
import numpy as np,cv2,subprocess,torch
from scene_pipeline.render import Renderer,camera,write_rgb

def render_sweeps(scene):
    out=ROOT/'auto'/scene/'global_instances';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'for_Inspatio_0922'/scene.replace('__','/')
    analysis=read(ROOT/'outputs'/scene/'analysis.json');graph=read(ROOT/'auto'/scene/'scene_graph.json')
    pending=[]
    for region in graph['regions']:
        folder=out/'supplemental_trajectories'/('floor_'+region['floor'][1:])/('sweep_'+region['id'])
        if (folder/'metadata.json').exists():continue
        pending.append((region,folder))
    if not pending:return
    renderer=Renderer(source,ROOT/'outputs'/scene);width,height=960,540;fps=8;count=32
    for region,folder in pending:
        folder.mkdir(parents=True,exist_ok=True);depthdir=folder/'mesh_depth_mm';depthdir.mkdir(exist_ok=True)
        floor=next(f for f in graph['floors'] if f['id']==region['floor']);center=np.r_[region['center'],floor['z_floor']+1.5]
        # Navigable region centers are clearance maxima; avoid teleport paths by
        # exporting stationary sweep clips with explicit clip cuts.
        poses=[camera(center,[np.cos(t),np.sin(t),-.08]) for t in np.linspace(0,2*np.pi,count,endpoint=False)]
        encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
            '-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'rgb.mp4')],stdin=subprocess.PIPE)
        try:
            for i,pose in enumerate(poses):
                rgb,_,_,K=renderer.rgbd(pose,width,height);depth=renderer.mesh_depth(pose,width,height,K)
                encoded=np.uint16(np.clip(np.rint(depth*1000),0,65535));encoded[depth>65.535]=0
                cv2.imwrite(str(depthdir/f'{i:05d}.png'),encoded);encoder.stdin.write(rgb.tobytes())
        finally:
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Sweep video encoder failed')
        np.savez_compressed(folder/'cameras.npz',camera_to_canonical=np.array(poses),K=K,timestamps=np.arange(count)/fps)
        dump(folder/'metadata.json',{'purpose':'annotation_only_not_evaluation_Q','frames':count,'fps':fps,'width':width,'height':height,
             'camera_height_above_floor':1.5,'region_order':[region['id']],'component':'sweep_'+region['id'],
             'trajectory_collision_checked':True,'trajectory_type':'stationary_360_sweep_at_navigable_region_clearance_center',
             'depth':'mesh camera-Z uint16 mm, zero invalid','rgb':'actual gsplat anisotropic 3D Gaussian rendering',
             'not_connected_to_other_clips':True,'floor':region['floor']})
        print('Rendered coverage sweep',scene,region['id'],count,'frames',flush=True)
    del renderer;torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene');args=parser.parse_args()
    for path in sorted((ROOT/'outputs').glob('*/analysis.json')):
        if not args.scene or args.scene==path.parent.name:render_sweeps(path.parent.name)

if __name__=='__main__':main()
