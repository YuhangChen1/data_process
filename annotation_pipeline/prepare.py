from .common import ROOT,read,dump,utc
from .geometry import build_geometry
from .semantics import annotate_regions
from .vlm import VLM
import argparse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene');args=parser.parse_args();vlm=VLM()
    for p in sorted((ROOT/'outputs').glob('*/analysis.json')):
        name=p.parent.name
        if args.scene and name!=args.scene:continue
        out=ROOT/'auto'/name;out.mkdir(parents=True,exist_ok=True);draft=out/'geometry_semantics.json'
        old=read(out/'scene_graph.json') if (out/'scene_graph.json').exists() else read(draft) if draft.exists() else {}
        registry_path=ROOT/'auto/_registry'/(name+'.json');registry=read(registry_path) if registry_path.exists() else {}
        graph,matching=build_geometry(name,out,old,registry)
        annotate_regions(graph,out,vlm)
        dump(draft,graph);dump(out/'id_matching.json',matching);dump(registry_path,registry)
        dump(out/'run_info.json',{'status':'geometry_and_region_semantics_complete','updated_at':utc(),'stages':{'geometry':'ok','region_vlm':'ok' if not any(x['type']=='vlm_failure' for x in graph['issues']) else 'partial','objects':'pending'},'model':vlm.model,'endpoint':vlm.endpoint,'id_matching':matching})
        print('Prepared',name,'regions',len(graph['regions']),'connections',len(graph['connections']),flush=True)

if __name__=='__main__':main()
