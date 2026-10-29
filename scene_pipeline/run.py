from pathlib import Path
import argparse,os
from .geometry import analyze

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--stage',choices=['geometry','render','all'],default='all');p.add_argument('--preview',action='store_true');p.add_argument('--force',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parents[1];source=root/'for_Inspatio_0922';output=root/'outputs'
    scenes=sorted(x.parent for x in source.rglob('mesh_post1.obj'))
    for scene in scenes:
        name=str(scene.relative_to(source)).replace('/','__')
        if args.scene and args.scene!=name:continue
        out=output/name
        if args.stage in ['geometry','all'] and (args.force or not (out/'analysis.json').exists()):analyze(scene,out)
        if args.stage in ['render','all'] and (args.force or not (out/'render_complete.json').exists()):
            from .render import render_scene
            render_scene(scene,out,args.preview)
        print('Finished',name,args.stage,flush=True)

if __name__=='__main__':main()
