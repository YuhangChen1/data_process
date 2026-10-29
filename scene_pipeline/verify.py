"""Check delivered maps, collision-free cameras, panorama and video assets."""
from pathlib import Path
import json,subprocess
import numpy as np
import cv2
from scipy import ndimage as ndi
from .geometry import segment_valid,save_json

def verify_scene(path):
    report=json.loads((path/'analysis.json').read_text());transform=np.asarray(report['world_to_canonical']);rot=transform[:3,:3]
    assert np.allclose(rot@rot.T,np.eye(3),atol=1e-6)
    assert np.isclose(np.linalg.det(rot),1,atol=1e-6)
    assert (path/'render_complete.json').exists(),f'Incomplete render: {path}'
    result={'scene':path.name,'floors':len(report['floors']),'videos':0,'frames':0,'panoramas':0}
    for fl in report['floors']:
        folder=path/f'floor_{fl["floor_id"]}';maps=np.load(folder/'maps.npz');nav=maps['navigable'];res=fl['resolution'];origin=maps['origin_xy']
        assert not (nav&maps['obstacles']).any()
        assert not (nav&~maps['support']).any()
        assert np.all(maps['clearance'][nav]>=fl['body_radius_m']-1e-7)
        assert np.array_equal(maps['occupancy']==0,nav)
        pgm=cv2.imread(str(folder/'occupancy.pgm'),cv2.IMREAD_UNCHANGED)
        assert np.array_equal(np.flipud(pgm)==254,nav)
        assert np.array_equal(np.flipud(pgm)==205,maps['occupancy']==-1)
        routes=json.loads((folder/'annotation_routes.json').read_text())
        regions=json.loads((folder/'regions.json').read_text())
        assert sum(len(r['region_order']) for r in routes)==len(regions)
        labels=maps['occupancy']
        for region in regions:
            assert nav[tuple(region['center_rc'])]
            panorama=folder/'panoramas'/(region['region_id']+'.jpg')
            img=cv2.imread(str(panorama));assert img is not None and img.shape==(1024,2048,3)
            result['panoramas']+=1
        for route in routes:
            assert route['purpose']=='annotation_only_not_evaluation_Q'
            rc=((np.asarray(route['xy'])-origin)/res-.5)[:,::-1]
            assert all(segment_valid(a,b,nav) for a,b in zip(rc[:-1],rc[1:]))
            clip=folder/'annotation_walkthrough'/f'component_{route["component"]:03d}'
            meta=json.loads((clip/'metadata.json').read_text());cameras=np.load(clip/'cameras.npz');poses=cameras['camera_to_canonical'];N=meta['frames']
            assert meta['purpose']=='annotation_only_not_evaluation_Q' and len(poses)==N
            assert np.allclose(poses[:,2,3],fl['height']+1.5)
            assert np.allclose(poses,np.einsum('ij,njk->nik',transform,cameras['camera_to_original']),atol=1e-5)
            assert np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-5)
            rc=((poses[:,:2,3]-origin)/res-.5)[:,::-1]
            assert all(nav[tuple(np.rint(p).astype(int))] for p in rc)
            assert all(segment_valid(a,b,nav) for a,b in zip(rc[:-1],rc[1:]))
            assert len(list((clip/'mesh_depth_mm').glob('*.png')))==N
            # Read every depth frame, checking the exported metric encoding and shape.
            for p in sorted((clip/'mesh_depth_mm').glob('*.png')):
                img=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
                assert img is not None and img.dtype==np.uint16 and img.shape==(meta['height'],meta['width'])
            for name in ['rgb.mp4','depth_preview.mp4']:
                info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(clip/name)]))['streams'][0]
                assert (info['width'],info['height'],int(info['nb_frames']))==(meta['width'],meta['height'],N)
                subprocess.run(['ffmpeg','-v','error','-i',str(clip/name),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            result['videos']+=1;result['frames']+=N
        for image in ['topdown_3dgs.png','mesh_height.png','annotation_overlay.png','regions_and_routes.png']:
            assert cv2.imread(str(folder/image)) is not None
    print('Verified',result,flush=True)
    return result


def main():
    root=Path(__file__).resolve().parents[1]/'outputs';results=[]
    paths=sorted(root.glob('*/analysis.json'));assert len(paths)==9
    for path in paths:results.append(verify_scene(path.parent))
    save_json(root/'verification.json',{'status':'passed','scenes':results,'total_panoramas':sum(x['panoramas'] for x in results),'total_clips':sum(x['videos'] for x in results),'total_frames':sum(x['frames'] for x in results),
        'checks':['Rigid coordinate transforms and original/canonical camera correspondence','Occupancy grid conventions and PNG/PGM orientation','Traversability support, obstacles and body radius','All route segments and rendered camera segments stay within inflated free space','All panorama dimensions','All 16-bit mesh-depth frame dimensions/types','Video frame counts, dimensions and full decode']})

if __name__=='__main__':main()
