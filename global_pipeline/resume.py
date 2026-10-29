"""Finish cached scenes without repeating completed rendering on GPT retries."""
import argparse,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from .common import ROOT,read,dump,utc

def export_ready(folder):
    if not all((folder/name).exists() for name in ['fine_instances.npz','global_instances.json','frame_assignments.json','rendered_frames.json']):return False
    doc=read(folder/'frames.json');rendered=read(folder/'rendered_frames.json')
    return {f['id'] for f in doc.get('frames',[])}=={f['frame_id'] for f in rendered.get('frames',[])} and bool(doc.get('frames'))

def finish(scene):
    folder=ROOT/'auto'/scene/'global_instances';info=read(folder/'run_info.json') if (folder/'run_info.json').exists() else {}
    validation=read(folder/'validation.json') if (folder/'validation.json').exists() else {}
    if info.get('status')=='complete' and validation.get('passed') and validation.get('frames')==len(read(folder/'frames.json').get('frames',[])):
        print('CACHED COMPLETE',scene,flush=True);return {'scene_id':scene,'status':'complete','cached':True}
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',GLOBAL_VLM_MODEL='gpt-6-sol',
        GLOBAL_SAM_URLS='http://127.0.0.1:18763,http://127.0.0.1:18764',VLM_ATTEMPTS='5')
    for attempt in range(8):
        stages=['describe','verify'] if export_ready(folder) else ['all']
        print('RESUME',scene,'attempt',attempt+1,'stages',stages,flush=True)
        for stage in stages:
            with (ROOT/'services'/('global_'+scene+'_completion.log')).open('ab') as log:
                result=subprocess.run([sys.executable,'-u','-m','global_pipeline.run','--scene',scene,'--stage',stage],
                    env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:break
        else:
            print('COMPLETE',scene,flush=True);return {'scene_id':scene,'status':'complete','attempts':attempt+1}
        print('RETRY',scene,'retaining successful model evidence',flush=True);time.sleep(5)
    return {'scene_id':scene,'status':'failed','attempts':8}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    scenes=['household_1','household_2','household_5','household_6','household_7','household_3','household_4_two_floors','household_3__0929_update','household_8']
    # Run the outstanding inventory/local stages independently while cached
    # scenes finish their GPT descriptions; shared API slots bound all traffic.
    scenes=['household_8']+[s for s in scenes if s!='household_8']
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(finish,scene) for scene in scenes]):
            results.append(future.result());dump(ROOT/'services/global_completion_state.json',{'updated_at':utc(),'results':results})
    from .report import main as report
    report()
    if any(r['status']!='complete' for r in results):raise RuntimeError('Some scenes need additional recovery')
    print('ALL NINE GLOBAL SCENES COMPLETE',flush=True)

if __name__=='__main__':main()
