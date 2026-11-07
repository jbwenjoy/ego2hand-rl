"""Declare or create one fixed-WM all452 state-error CPU config; never run it.

Default writes nothing. --authorize creates a fresh config only. Dependencies
are archived once per actual file identity plus SHA, not per legacy alias.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'protocols/full151105_state_prediction452_CPU_prereg_v2.json'
PROTOCOL_SHA='37652718ed150421ed84d25bcafc14b27a1e7390a7dee7e729a860b3d91ef411'
OWN={'consumer':ROOT/'src/full151_gradient/state_prediction452105_v1.py',
     'CLI':ROOT/'scripts/diagnose_full151_state_prediction452_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_state_prediction452_cpu_v1.py'}
CONFIG=ROOT/'configs/full151105-state-prediction452-CPU-v1.json'
OUTPUT=ROOT/'outputs/full151105_state_prediction452_CPU_v1'
REVIEW=ROOT/'outputs/full151105_state_prediction452_builder_source_only_v1/READY_source_only.json'
SOURCE_REVIEW=ROOT/'outputs/full151105_state_prediction452_source_feasibility_v1/report.json'
SOURCE_REVIEW_SHA='ae63d8e8dcf86eac7e4ad29f3bb785cf2305c2144837c125846e3906a561d4ce'
COST_REVIEW=ROOT/'outputs/full151105_passive_MJB_callgraph_source_review_v1/report.json'
COST_REVIEW_SHA='98c3d3917acac0942a3d031f4a8a6d294da090dc33abb7068349a3c30198ee6e'
META_REVIEW=ROOT/'outputs/full151105_state_prediction452_metadata_preflight_v1/report.json'
META_REVIEW_SHA='a74c930e58957d8d775c42ab168c2ded899dd55e9c29ba938bc2bb5543780088'
READY_SCHEMA='full151105_state_prediction452_CPU_READY:1'
PROOF_SCHEMA='full151105_state_prediction452_CPU_contract:1'
STATUS='PASS_SYNTHETIC_GROUP_ERROR_AND_SINGLE_FORWARD_CONTRACT'


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class FilePins:
    """One physical identity is hashed once; different copies never merge.

    samestat/dev+ino is the filesystem identity used by samefile. Every merge
    is also explicitly checked with Path.samefile. A changed file or identity
    collision fails closed; SHA equality alone is never a deduplication key.
    """
    def __init__(self):
        self.cache={};self.source_map={};self.members={}

    def pin(self,path):
        p=Path(path);need(p.is_absolute() and p.is_file(),'Absolute existing file required: '+str(p))
        resolved=p.resolve(strict=True);st=resolved.stat()
        need(st.st_ino!=0,'Stable file identity unavailable: '+str(p))
        identity=(st.st_dev,st.st_ino);signature=(st.st_size,st.st_mtime_ns)
        if identity in self.cache:
            old=self.cache[identity]
            need(resolved.samefile(old['resolved']) and signature==old['signature'],'File changed or identity collision: '+str(p))
            sha=old['sha256']
        else:
            h=hashlib.sha256()
            with resolved.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            end=resolved.stat();need((end.st_dev,end.st_ino,end.st_size,end.st_mtime_ns)==(st.st_dev,st.st_ino,*signature),'File changed during hash')
            sha=h.hexdigest();self.cache[identity]={'resolved':resolved,'signature':signature,'sha256':sha}
        return {'path':str(p),'sha256':sha},identity

    def checked(self,value):
        need(type(value) is dict and set(value)=={'path','sha256'} and type(value['path']) is str and type(value['sha256']) is str and len(value['sha256'])==64,'Exact absolute path/SHA pin required')
        current,identity=self.pin(value['path']);need(current['sha256']==value['sha256'],'Pinned bytes changed: '+value['path'])
        return Path(value['path']),identity

    def same(self,left,right):
        a,ia=self.checked(left);b,ib=self.checked(right)
        return ia==ib and left['sha256']==right['sha256'] and a.samefile(b)

    def add(self,value):
        p,identity=self.checked(value);key=(*identity,value['sha256'])
        if key in self.members:
            need(p.samefile(Path(self.members[key]['path'])),'Never merge different physical copies')
            return
        self.members[key]=value
        self.source_map['file_%04d'%len(self.source_map)]=value  # First recorded spelling retained.

    def add_map(self,values):
        need(type(values) is dict and values,'Nonempty flat source map required')
        for value in values.values():self.add(value)


def positive(report):
    c=report.get('checks');need(type(c) is dict and c and all(type(x) is bool and x for x in c.values()),'Actual alltrue CPU proof required')
    need(type(report.get('passed')) is int and type(report.get('total')) is int and report['passed']==report['total']==len(c),'Actual CPU count mismatch')


def build(ready_path,authorize=False):
    need(type(authorize) is bool,'Explicit CPU authorization bool required')
    need(not any(x in sys.modules for x in ('torch','warp','mujoco','mujoco_warp')),'Fresh metadata process required')
    need(not any(x.startswith('PENDING_') for x in (READY_SCHEMA,PROOF_SCHEMA,STATUS)),'Wait for final actual CPU READY')
    pins=FilePins();ready_path=Path(ready_path).resolve(strict=True)
    ready=read(ready_path);pins.checked(ready['cpu_report']);proof=read(ready['cpu_report']['path']);positive(proof)
    need(ready.get('schema')==READY_SCHEMA and proof.get('schema')==PROOF_SCHEMA and ready.get('status')==proof.get('status')==STATUS,'Wrong current-source state452 CPU proof')
    need(ready['cpu_checks']=={'passed':proof['passed'],'total':proof['total']},'READY/proof count mismatch')
    need(ready.get('CPU_scientific_run_authorized') is False and ready.get('GPU_run_authorized') is False,'CPU preparation is not scientific authorization')
    for key in ('actual_saved_input_assembly','actual_WM_loads','actual_WM_forwards','actual_model_or_MJB_or_Env_or_GPU','new_AD'):
        need(type(proof['scope'].get(key)) is int and proof['scope'][key]==0,'Synthetic proof executed real science: '+key)
    protocol_pin,_=pins.pin(PROTOCOL);need(protocol_pin['sha256']==PROTOCOL_SHA and pins.same(ready['protocol'],protocol_pin) and pins.same(proof['protocol'],protocol_pin),'Frozen state-only452 protocol differs')
    need(set(ready['source_pins'])==set(proof['source_pins'])==set(OWN),'Exact current three tested sources required')
    for key,path in OWN.items():
        actual,_=pins.pin(path)
        need(pins.same(ready['source_pins'][key],actual) and pins.same(proof['source_pins'][key],actual),'CPU-tested source changed: '+key)
    need(set(ready['dependency_pins'])==set(proof['dependency_pins']),'CPU dependency source keys differ')
    for key,value in ready['dependency_pins'].items():need(pins.same(value,proof['dependency_pins'][key]),'CPU dependency changed: '+key)
    protocol=read(PROTOCOL)
    need(protocol['schema']=='full151105_state_prediction452_CPU_preregistration:1' and protocol['cpu_threads']==4,'Wrong state-only scope/threads')
    need(all(protocol[k] is False for k in ('CPU_run_authorized','GPU_run_authorized','training_authorized')),'Source-only protocol is not execution authority')
    for value in protocol['inputs'].values():pins.checked(value)
    review_pin,_=pins.pin(SOURCE_REVIEW);need(review_pin['sha256']==SOURCE_REVIEW_SHA,'Source-feasibility record changed')
    cost_pin,_=pins.pin(COST_REVIEW);need(cost_pin['sha256']==COST_REVIEW_SHA,'Source-accounted MJB cost supplement changed')
    meta_pin,_=pins.pin(META_REVIEW);need(meta_pin['sha256']==META_REVIEW_SHA,'Actual metadata join proof changed')
    meta=read(META_REVIEW)
    need(meta['schema']=='full151105_state_prediction452_actual_metadata_preflight:1' and meta['status']=='PASS_ACTUAL_REPLAY_AND_FIXED_MODEL_METADATA_ONLY','Actual model/replay metadata-only join required')
    own_pin,_=pins.pin(OWN['consumer']);need(pins.same(meta['consumer'],own_pin) and pins.same(meta['protocol'],protocol_pin),'Actual metadata source/protocol mismatch')
    for key in ('NPZ_decode_attempted','actual_inputs_assembled','actual_strict_loader_calls','actual_WM_forwards','MJB_Env_GPU'):
        need(type(meta[key]) is int and meta[key]==0,'Metadata prerequisite elevated scientific scope')
    need(meta['Torch_imported'] is False,'Metadata prerequisite imported Torch')
    for field,alias in [('checkpoint','fixed_A2_checkpoint'),('health','fixed_A2_final_health'),('health_auditor','fixed_A2_health_auditor')]:
        need(pins.same(meta['model'][field],protocol['inputs'][alias]),'Fixed-model semantic input identity mismatch: '+field)
    for field,alias in [('actual_config','GPU_replay_config'),('producer_report','GPU_replay_report'),('qualified_prestate','qualified_GPU_prestate1808')]:
        need(pins.same(meta['replay'][field],protocol['inputs'][alias]),'Replay semantic input identity mismatch: '+field)

    # Only the proved consumer's JSON/hash API is imported. No WM loader call.
    sys.path.insert(0,str(ROOT/'src'))
    core=importlib.import_module('full151_gradient.state_prediction452105_v1')
    actual,_=pins.pin(core.__file__);expected,_=pins.pin(OWN['consumer']);need(pins.same(actual,expected),'Unexpected consumer import')
    pins.add_map(core.required_sources(protocol))
    pins.add_map(protocol['source_pins']);pins.add_map(protocol['inputs'])
    pins.add_map(ready['source_pins']);pins.add_map(ready['dependency_pins'])
    for path in (ready_path,PROTOCOL,SOURCE_REVIEW,COST_REVIEW,META_REVIEW):pins.add(pins.pin(path)[0])
    pins.add(ready['cpu_report'])
    review=read(REVIEW)
    need(review['schema']=='full151105_state_prediction452_builder_source_only:1' and review['AST_parse_only'] is True and review['builder_executed'] is False,'Builder AST-only source record required')
    selfpin,_=pins.pin(Path(__file__).resolve());need(pins.same(review['source_pins']['builder'],selfpin),'Builder source changed after freeze')
    pins.add(selfpin);pins.add(pins.pin(REVIEW)[0])
    need(not CONFIG.exists() and not OUTPUT.exists(),'Fresh config/output required; no overwrite or retry')
    c={'schema':core.SCHEMA,'role':core.ROLE,'protocol':protocol_pin,'output':str(OUTPUT),'CPU_run_authorized':authorize,'cpu_threads':4,'source_pins':pins.source_map,'cpu_ready':pins.pin(ready_path)[0]}
    need(set(c)==core.KEYS and c['schema']==protocol['runplan_schema'] and c['role']==protocol['role'],'Actual state-only452 eight-key ABI differs')
    need(ready['api']['schema']==core.SCHEMA and ready['api']['role']==core.ROLE and ready['api']['keys']==sorted(core.KEYS),'Actual READY API keyset differs')
    api_module,_=pins.pin(ready['api']['module']);api_entry,_=pins.pin(ready['api']['entry']);own_entry,_=pins.pin(OWN['CLI'])
    need(pins.same(api_module,own_pin) and pins.same(api_entry,own_entry),'Actual READY module/CLI route differs')
    core.validate_header(c)
    need(not any(x in sys.modules for x in ('torch','warp','mujoco','mujoco_warp')),'Metadata imported a backend')
    for value in pins.source_map.values():pins.checked(value)
    if authorize:
        with CONFIG.open('x',encoding='utf-8',newline='\n') as f:json.dump(c,f,indent=2);f.write('\n')
    return {'schema':'full151105_state_prediction452_builder_declaration:1','status':'CREATED_EXPLICITLY_AUTHORIZED_CPU_CONFIG_ONLY' if authorize else 'DECLARED_NO_CONFIG_WRITTEN',
       'config':pins.pin(CONFIG)[0] if authorize else {'path':str(CONFIG),'written':False},'output':str(OUTPUT),'protocol':protocol_pin,'cpu_ready':pins.pin(ready_path)[0],
       'unique_source_file_count':len(pins.source_map),'source_dedup':'same physical file identity+SHA; never SHA-only; semantic input aliases remain in protocol',
       'CPU_run_authorized':authorize,'Torch_imported':False,'Warp_imported':False,'NPZ_opened':0,'models_or_state_calls':0,'WM_Actor_objective_FW_AD_Env_physics_GPU':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ready',required=True,type=Path);p.add_argument('--authorize',action='store_true',help='Create one fresh CPU-authorized config only; never execute it.')
    a=p.parse_args();print(json.dumps(build(a.ready,a.authorize),indent=2))


if __name__=='__main__':main()
