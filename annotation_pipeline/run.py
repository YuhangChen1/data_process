"""Re-runnable automatic annotation; never writes human/ or final/."""
from pathlib import Path
import argparse,time,traceback,shutil,json,os,hashlib
from .common import ROOT,read,dump,sha,utc
from .vlm import VLM
from .objects import build_objects,model_signature
from .semantics import annotate_attributes,annotate_scene
from .finalize import relations_and_scale,build_texts
from .quality import audit_confidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene');parser.add_argument('--force',action='store_true');args=parser.parse_args();vlm=VLM()
    for source in sorted((ROOT/'outputs').glob('*/analysis.json')):
        name=source.parent.name
        if args.scene and args.scene!=name:continue
        out=ROOT/'auto'/name
        if (out/'scene_graph.json').exists() and not args.force:continue
        if not (out/'geometry_semantics.json').exists():
            print('Awaiting prepared geometry',name,flush=True)
            continue
        started=time.time();graph=read(out/'geometry_semantics.json');old=read(out/'scene_graph.json') if (out/'scene_graph.json').exists() else {}
        registry_path=ROOT/'auto/_registry'/(name+'.json');registry=read(registry_path) if registry_path.exists() else {}
        info=read(out/'run_info.json');info.update(status='running',started_at=utc(),stages={'geometry':'ok','region_vlm':'ok','objects':'running','attributes':'pending','scene_text':'pending'});dump(out/'run_info.json',info)
        # Reset derived fields in a loaded draft.
        for r in graph['regions']:r['object_ids']=[]
        try:
            matching=build_objects(graph,out,old,registry);info['stages']['objects']='ok';dump(out/'objects_draft.json',graph)
            annotate_attributes(graph,out,vlm);info['stages']['attributes']='ok'
            annotate_scene(graph,out,vlm);info['stages']['scene_text']='ok'
            relations_and_scale(graph);textmatch=build_texts(graph,old,registry);issue_counts=audit_confidence(graph)
            if old:
                history=out/'history'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());history.mkdir(parents=True,exist_ok=True);dump(history/'scene_graph.json',old)
            graph['generation']={'pipeline':'annotation_pipeline/v0.1','generated_at':utc(),'purpose':'automatic_annotation_only','score_semantics':'model/rule confidence scores are heuristic, not calibrated probabilities','human_review_applied':False}
            dump(out/'scene_graph.json',graph);dump(registry_path,registry)
            matching_info=read(out/'id_matching.json');matching_info.update(objects=matching,texts=textmatch);dump(out/'id_matching.json',matching_info)
            manifest=ROOT/'services/model_manifest.json'
            if not manifest.exists():dump(manifest,{'checkpoint_path':str(ROOT/'model/sam3.pt'),'checkpoint_sha256':sha(ROOT/'model/sam3.pt'),'sam3_repository':'https://github.com/facebookresearch/sam3','sam3_code_sha256':sha(ROOT/'.deps/sam3/sam3/model_builder.py')})
            current_manifest=read(manifest);current_manifest.update(model_signature());dump(manifest,current_manifest)
            info.update(status='complete',finished_at=utc(),seconds=time.time()-started,models={'sam3':read(manifest),'vlm':{'requested_model':vlm.model,'endpoint':vlm.endpoint,'auth_path':'/mnt/cfs/cyh/auth.json','secret_logged':False}},
                        parameters={'segmentation_confidence':.45,'keyframe_stride':16,'point_voxel_m':.035,'mask_dbscan_eps_m':.14,'region_iou_match_threshold':.5,'object_aabb_iou_match_threshold':.2,'objects_are_observed_extents':True,'independent_scale_verified':False},
                        counts={key:len(graph[key]) for key in ['floors','regions','connections','objects','texts','issues']},issue_counts=issue_counts,failure_issues=[x for x in graph['issues'] if x['type'] in ['vlm_failure','sam3_prompt_failure','missing_vlm_object_result']],id_matching=matching_info)
            if info['failure_issues']:info['status']='complete_with_stage_failures'
            dump(out/'run_info.json',info);print('COMPLETE',name,info['counts'],'seconds',round(info['seconds'],1),flush=True)
        except Exception as exc:
            info.update(status='failed',finished_at=utc(),error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());dump(out/'run_info.json',info);print('FAILED',name,info['error'],flush=True)
            raise

if __name__=='__main__':main()
