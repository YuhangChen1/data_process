"""Two scene workers with separate resident SAM3 ports; retry recoverable stages."""
import os,sys,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from .common import ROOT,read,dump,utc

GROUPS=[['household_1','household_3','household_4_two_floors','household_6','household_8'],
        ['household_2','household_3__0929_update','household_5','household_7']]
PORTS=['http://127.0.0.1:18763,http://127.0.0.1:18764','http://127.0.0.1:18766,http://127.0.0.1:18767']

def worker(index):
    env=os.environ.copy();env.update(GLOBAL_SAM_URLS=PORTS[index],GLOBAL_VLM_MODEL='gpt-6-sol',
         GLOBAL_AUDIT_WORKERS='8',GLOBAL_DESCRIPTION_WORKERS='6',VLM_ATTEMPTS='5',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    for scene in GROUPS[index]:
        out=ROOT/'auto'/scene/'global_instances';expected=len(read(ROOT/'auto'/scene/'scene_graph.json')['regions'])
        while len(list((out/'supplemental_trajectories').glob('floor_*/sweep_*/metadata.json')))<expected:
            print('Waiting for rendered coverage sweeps',scene,flush=True);time.sleep(15)
        for attempt in range(5):
            print('SCENE START',scene,'attempt',attempt+1,flush=True)
            with (ROOT/'services'/('global_'+scene+'.log')).open('ab') as log:
                result=subprocess.run([sys.executable,'-u','-m','global_pipeline.run','--scene',scene],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0:print('SCENE COMPLETE',scene,flush=True);break
            print('SCENE RETRY',scene,'see per-scene log',flush=True);time.sleep(5)
        else:raise RuntimeError('Retries exhausted '+scene)

def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(worker,i) for i in range(2)]:future.result()
    print('ALL NINE GLOBAL SCENES COMPLETE',flush=True)

if __name__=='__main__':main()
