"""Rebuild region candidates after removing discrete watershed fragments."""
from pathlib import Path
import json,shutil
import numpy as np,cv2
from .geometry import plan_regions,save_json

def main():
    for p in sorted(Path('outputs').glob('*/analysis.json')):
        analysis=json.loads(p.read_text());scene_changed=False
        for fl in analysis['floors']:
            folder=p.parent/f'floor_{fl["floor_id"]}';maps=np.load(folder/'maps.npz')
            old=(folder/'regions.json').read_text()
            regions,labels,routes=plan_regions(maps['navigable'],maps['origin_xy'],fl['resolution'],folder)
            changed=old!=(folder/'regions.json').read_text()
            if not changed:continue
            scene_changed=True
            # These are generated assets; regenerate them to match new region IDs.
            for name in ['panoramas','annotation_walkthrough','alignment_views']:
                if (folder/name).exists():shutil.rmtree(folder/name)
            fl['regions']=len(regions);fl['connected_components']=len(routes)
            save_json(folder/'floor.json',fl)
            ny,nx=maps['navigable'].shape;origin=maps['origin_xy'];res=fl['resolution']
            image=np.full((ny,nx,3),225,np.uint8);image[maps['support']]=[200,215,205];image[maps['obstacles']]=[55,55,55]
            palette=np.random.default_rng(42).integers(70,230,(len(regions)+100,3),dtype=np.uint8)
            for region in regions:image[labels==region['label']]=palette[region['label']]
            for route in routes:
                pts=np.rint((np.array(route['xy'])-origin)/res-.5).astype(np.int32)
                if len(pts)>1:cv2.polylines(image,[pts],False,(255,255,255),1,cv2.LINE_AA)
            image=np.flipud(image).copy()
            for region in regions:
                y,x=region['center_rc'];cv2.putText(image,str(region['label']),(x,ny-1-y),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)
            cv2.imwrite(str(folder/'regions_and_routes.png'),image[:,:,::-1])
            print(folder,'regions',len(regions),flush=True)
        if scene_changed:
            save_json(p,analysis);(p.parent/'render_complete.json').unlink(missing_ok=True)

if __name__=='__main__':main()
