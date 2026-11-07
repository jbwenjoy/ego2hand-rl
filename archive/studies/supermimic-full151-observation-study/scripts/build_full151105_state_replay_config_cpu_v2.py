"""Declare or explicitly authorize one new native GPU B1 Reader replay config.

Default never writes a config. --authorize writes one fresh config only: this
program never starts a GPU child, opens NPZ, converts states, or scores data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocols/full151105-matched-native-GPU-reader-replay-prereg-v1.json'
PROTOCOL_SHA = 'fddcf970d7a8e44732fe93e6c83d1918fe017d4d5df159ae5c851a77e15fca6c'
READY = ROOT / 'outputs/full151105_state_replay_preparation_CPU_v2/READY.json'
OWN = {'module': ROOT / 'src/full151_gradient/state_replay105_v2.py',
       'entry': ROOT / 'scripts/execute_full151_state_replay_v2.py',
       'supervisor': ROOT / 'scripts/run_full151_state_replay_v2.py',
       'checker': ROOT / 'scripts/check_full151_state_replay_cpu_v2.py'}
CONFIG = ROOT / 'configs/full151105-matched-native-GPU-reader-v2.json'
OUTPUT = ROOT / 'outputs/full151105_matched_native_GPU_reader_v2'
BUILD_REVIEW = ROOT / 'outputs/full151105_state_replay_builder_source_only_v2/READY_source_only.json'
DRIVER = Path('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py')
DRIVER_SHA = 'f4161cb4be7ac2e5e0fc598820a944f4d9a0e7240ee79b4eb68054732f246475'
LIFECYCLE = Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')
LIFECYCLE_SHA = '40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa'


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    p = Path(path)
    need(p.is_file(), 'Missing file: '+str(p))
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest()}


def checked(value):
    need(type(value) is dict and set(value)=={'path','sha256'} and type(value['path']) is str and Path(value['path']).is_absolute(), 'Exact absolute path/SHA pin required')
    need(type(value['sha256']) is str and len(value['sha256'])==64 and pin(value['path'])['sha256']==value['sha256'], 'Pinned bytes changed: '+value['path'])
    return Path(value['path'])


def same(a,b):
    return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def allchecks(value):
    checks=value.get('checks')
    need(type(checks) is dict and checks and all(type(x) is bool and x for x in checks.values()), 'Strict positive actual checks required')
    need(type(value.get('passed')) is int and type(value.get('total')) is int and value['passed']==value['total']==len(checks), 'Actual check counts differ')


def add(target,prefix,values):
    need(type(values) is dict and values, 'Nonempty flat source map required')
    for key,value in values.items():
        need(type(key) is str and prefix+key not in target, 'Source alias collision')
        checked(value)
        target[prefix+key]=value  # Retain recorded spelling; use samefile for identity.


def build(ready_path,authorize=False):
    need(type(authorize) is bool, 'Explicit authorization bool required')
    need(not any(x in sys.modules for x in ('torch','warp','mujoco','mujoco_warp')), 'Fresh metadata process required')
    need(Path(ready_path).samefile(READY), 'Use the new v2 actual CPU READY, not v1')
    ready=read(ready_path)
    need(ready.get('schema')=='full151105_state_replay_CPU_READY:1', 'Wrong actual CPU READY schema')
    need(ready.get('GPU_run_authorized') is False and ready.get('native_replay_qualified') is False and ready.get('proxy_scoring_authorized') is False, 'Preparation is not GPU/data qualification')
    protocol=read(PROTOCOL)
    need(pin(PROTOCOL)['sha256']==PROTOCOL_SHA and same(ready['protocol'],pin(PROTOCOL)), 'Wrong frozen single-condition preregistration')
    need(protocol.get('schema')=='full151105_matched_native_GPU_reader_replay_preregistration:1' and protocol['fixed_mode']=='NATIVE_FP32_NO_AUTOCAST_CUDA_BATCH1', 'Wrong native B1 mode')
    need(all(protocol[k] is False for k in ('GPU_run_authorized','CPU_conversion_authorized','proxy_scoring_authorized','training_authorized')), 'Protocol is not runtime authorization')
    need(protocol['lifecycle']['soft_seconds']==90 and protocol['lifecycle']['owned_watchdog_seconds']==120, 'Fixed90/120 lifecycle changed')
    proof=read(checked(ready['cpu_report']));allchecks(proof)
    need(proof.get('schema')=='full151105_state_replay_CPU_contract:1' and proof.get('status')=='PASS_PURE_CPU_CONTRACT_NO_GPU', 'Wrong current-source CPU proof')
    need(ready['cpu_checks']=={'passed':proof['passed'],'total':proof['total']} and same(proof['protocol'],pin(PROTOCOL)), 'CPU proof/READY/protocol mismatch')
    for key in ('actual_GPU_calls','native_state_calls','NPZ_input_reads','MJB_loads','model_loads'):
        need(type(proof.get(key)) is int and proof[key]==0, 'CPU fixture scope elevated: '+key)
    need(set(OWN)<=set(ready['source_pins']) and set(OWN)<=set(proof['source_pins']), 'Four actual v2 execution sources required')
    for key,path in OWN.items():
        need(same(ready['source_pins'][key],pin(path)) and same(proof['source_pins'][key],pin(path)), 'Executed CPU source differs: '+key)
    for source,digest in ((DRIVER,DRIVER_SHA),(LIFECYCLE,LIFECYCLE_SHA)):
        need(pin(source)['sha256']==digest, 'Qualified lifecycle dependency changed')

    sources={}
    forensic_config=read(checked(protocol['inputs']['forensic_config']))
    add(sources,'forensic_actual__',forensic_config['source_pins'])
    add(sources,'replay_prereg_source__',protocol['source_pins'])
    add(sources,'replay_input__',protocol['inputs'])
    add(sources,'replay_CPUtested__',ready['source_pins'])
    sources['replay_protocol']=pin(PROTOCOL)
    sources['replay_prereg_writer']=protocol['writer'];checked(protocol['writer'])
    sources['replay_CPU_ready']=pin(ready_path)
    sources['replay_CPU_contract_report']=ready['cpu_report']
    sources['reused_identity_driver']=pin(DRIVER)
    sources['reused_lifecycle']=pin(LIFECYCLE)
    sources['atomic_runtime_io']=pin(ROOT/'vendor/runtime_io.py')

    review=read(BUILD_REVIEW)
    need(review.get('schema')=='full151105_state_replay_builder_source_only:2' and review.get('AST_parse_only') is True and review.get('builder_executed') is False, 'Builder AST-only receipt required')
    need(same(review['source_pins']['builder'],pin(__file__)), 'Builder changed since its AST-only review')
    sources['replay_authorizer_builder']=pin(__file__)
    sources['replay_authorizer_AST_review']=pin(BUILD_REVIEW)
    need(not CONFIG.exists() and not OUTPUT.exists() and not OUTPUT.with_name(OUTPUT.name+'_watchdog').exists(), 'Fresh config/output/watchdog only; no overwrite or retry')
    c={'schema':'full151105_matched_native_GPU_reader_replay_run:1','role':'full151105_native_GPU_state_replay',
       'protocol':pin(PROTOCOL),'output':str(OUTPUT),'source_pins':sources,'CPU_ready':pin(ready_path),'CPU_contract_report':ready['cpu_report'],
       'GPU_run_authorized':authorize,'training_authorized':False,'Env_or_physics_authorized':False,'automatic_retry':False,'compile_fallback':False,
       'soft_worker_seconds':90,'hard_process_seconds':120,'evaluation_seed':0,'device':'cuda:0','allow_tf32':True}
    # Current v2 source is imported only after its actual frozen CPU proof joins.
    # Both methods below are metadata-only. Never call run_replay/Reader/state.
    sys.path.insert(0,str(ROOT/'src'))
    core=importlib.import_module('full151_gradient.state_replay105_v2')
    need(same(pin(core.__file__),pin(OWN['module'])), 'Unexpected imported v2 core')
    need(set(c)==core.KEYS and c['schema']==core.SCHEMA and c['role']==core.ROLE, 'Actual consumer ABI differs')
    core.validate_config(c)
    need(not any(x in sys.modules for x in ('torch','warp','mujoco','mujoco_warp')), 'Metadata path imported backend')
    for value in sources.values():checked(value)
    if authorize:
        with CONFIG.open('x',encoding='utf-8',newline='\n') as f:
            json.dump(c,f,indent=2);f.write('\n')
    return {'schema':'full151105_state_replay_builder_declaration:2',
      'status':'CREATED_EXPLICITLY_AUTHORIZED_GPU_REPLAY_CONFIG_ONLY' if authorize else 'DECLARED_NO_CONFIG_WRITTEN',
      'config':pin(CONFIG) if authorize else {'path':str(CONFIG),'written':False},'output':str(OUTPUT),
      'protocol':pin(PROTOCOL),'CPU_ready':pin(ready_path),'CPU_contract_report':ready['cpu_report'],'source_pin_count':len(sources),
      'GPU_run_authorized':authorize,'Torch_imported':False,'Warp_imported':False,'NPZ_opened':0,'native_state_calls':0,
      'MJB_models_objectives_Env_physics_GPU':0,'native_replay_qualified':False,'proxy_scoring_authorized':False,'old_failure_changed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready',required=True,type=Path)
    parser.add_argument('--authorize',action='store_true',help='Create one fresh GPU-authorized config only; does not run it.')
    args=parser.parse_args()
    print(json.dumps(build(args.ready,args.authorize),indent=2))


if __name__=='__main__':
    main()
