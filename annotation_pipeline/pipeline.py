"""One command to resume preparation, inference, quality audit, reporting and validation."""
import argparse
import os
import subprocess
import sys
import time

import requests

from .common import ROOT, read


def module(name, *args):
    subprocess.run([sys.executable, '-u', '-m', 'annotation_pipeline.' + name, *args],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', help='Scene ID; default: all nine scene versions')
    parser.add_argument('--force', action='store_true', help='Rebuild complete scenes, reuse matching IDs and valid model caches')
    args = parser.parse_args()
    scenes = sorted(p.parent.name for p in (ROOT / 'outputs').glob('*/analysis.json'))
    if args.scene and args.scene not in scenes:
        parser.error('Unknown scene: ' + args.scene)
    scenes = [args.scene] if args.scene else scenes
    subprocess.run(['bash', str(ROOT / 'services/start_sam3.sh')], check=True, cwd=ROOT)
    session = requests.Session()
    session.trust_env = False
    for attempt in range(120):
        try:
            if session.get('http://127.0.0.1:18763/health', timeout=2).json().get('ready'):
                break
        except (requests.RequestException, ValueError):
            pass
        if attempt % 15 == 0:
            print('Waiting for resident SAM3 to load...', flush=True)
        time.sleep(1)
    else:
        raise RuntimeError('SAM3 did not become ready; see services/sam3.log')
    for scene in scenes:
        out = ROOT / 'auto' / scene
        info = read(out / 'run_info.json') if (out / 'run_info.json').exists() else {}
        if not args.force and (out / 'scene_graph.json').exists() and info.get('status') == 'complete':
            print('Already complete:', scene, flush=True)
        else:
            if args.force or not (out / 'geometry_semantics.json').exists():
                module('prepare', '--scene', scene)
            module('run', '--scene', scene, '--force')
    options = ['--scene', args.scene] if args.scene else []
    module('recover_vlm', *options)
    module('postprocess', *options)
    module('report')
    module('verify', *options)


if __name__ == '__main__':
    main()
