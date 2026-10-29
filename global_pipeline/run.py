"""Resumeable dense global-instance pipeline. All writes are automatic evidence."""
import argparse,os,time,traceback,fcntl
from .common import ROOT,DEFAULTS,destination,read,dump,utc,progress,sha
from annotation_pipeline.vlm import gateway_endpoint

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene');parser.add_argument('--stage',default='all',choices=['all','frames','discover','segment','geometry','instances','describe','export','verify'])
    parser.add_argument('--frame-stride',type=int,default=1);parser.add_argument('--audit-stride',type=int,default=1)
    parser.add_argument('--force',action='store_true',help='Rebuild derived results while retaining valid model response caches')
    args=parser.parse_args();config=dict(DEFAULTS);config.update(frame_stride=args.frame_stride,audit_stride=args.audit_stride)
    scenes=sorted(p.parent.name for p in (ROOT/'outputs').glob('*/analysis.json'))
    if args.scene:
        if args.scene not in scenes:parser.error('Unknown scene')
        scenes=[args.scene]
    from .frames import prepare_frames
    for scene in scenes:
        out=destination(scene);out.mkdir(parents=True,exist_ok=True);started=time.time()
        lock=(out/'.pipeline.lock').open('a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Waiting for existing scene writer',scene,flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            info=read(out/'run_info.json') if (out/'run_info.json').exists() else {}
            doc=prepare_frames(scene,out,config)
            validation=read(out/'validation.json') if (out/'validation.json').exists() else {}
            if not args.force and args.stage=='all' and info.get('status')=='complete' and info.get('progress',{}).get('parameters')==config and validation.get('passed') and validation.get('frames')==len(doc['frames']):print('Already complete',scene,flush=True);continue
            progress(out,args.stage,parameters=config)
            info=read(out/'run_info.json')
            if info.get('implementation'):
                info.setdefault('implementation_history',[]).append(info['implementation'])
            info['implementation']={'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'global_pipeline').glob('*.py'))},
                'sam3_service_sha256':sha(ROOT/'annotation_pipeline/sam_service.py'),
                'vlm_client_sha256':sha(ROOT/'annotation_pipeline/vlm.py'),'captured_at':utc(),'capture_stage':args.stage,
                'requested_gpt_model':os.environ.get('GLOBAL_VLM_MODEL','gpt-6-sol'),
                'frames_fingerprint':doc['fingerprint'],'api_endpoint':gateway_endpoint(required=False),
                'auth_path':'/mnt/cfs/cyh/auth.json','secrets_saved':False}
            dump(out/'run_info.json',info)
            if args.stage=='frames':continue
            if args.stage in ['all','discover']:
                from .discovery import discover
                inventories=discover(out,doc,config)
                if args.stage=='discover':continue
            else:inventories=read(out/'concepts.json') if (out/'concepts.json').exists() else None
            if args.stage in ['all','segment']:
                from .discovery import segment_frames
                segment_frames(out,doc,inventories,config)
                if args.stage=='segment':continue
            if args.stage in ['all','geometry']:
                from .geometry import build_superpoints,projections
                cloud=build_superpoints(scene,out,config);projections(out,doc,cloud,config)
                if args.stage=='geometry':continue
            if args.stage in ['all','instances']:
                from .instances import build_instances
                build_instances(out,doc,config)
                if args.stage=='instances':continue
            if args.stage in ['all','export']:
                from .export import export
                export(out,doc,config)
                if args.stage=='export':continue
            if args.stage in ['all','describe']:
                from .describe import describe
                describe(out,doc,config)
                if args.stage=='describe':continue
            if args.stage in ['all','verify']:
                from .verify import verify
                # descriptions are an optional layer: verify them only when they were generated
                verify(out,doc,config,include_descriptions=(out/'descriptions.json').exists())
                validation=read(out/'validation.json')
            info=read(out/'run_info.json');info.update(status='complete',stage='complete',finished_at=utc(),updated_at=utc(),last_pass_seconds=time.time()-started)
            info['progress'].update(described_frames=validation.get('level1',len(doc['frames'])),description_changes=sum(max(0,len(c['frames'])-1) for c in doc['clips']))
            dump(out/'run_info.json',info)
            print('COMPLETE GLOBAL',scene,flush=True)
        except Exception as exc:
            info=read(out/'run_info.json');info.update(status='failed',failed_at=utc(),error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());dump(out/'run_info.json',info)
            print('FAILED GLOBAL',scene,info['error'],flush=True);raise
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
