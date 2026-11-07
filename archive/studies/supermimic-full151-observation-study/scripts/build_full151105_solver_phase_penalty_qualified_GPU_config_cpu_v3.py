"""Declare or explicitly create a fresh CPU dense-penalty analysis config.

No conversion, MJB/objective/model, NPZ transformation, scoring or execution is
performed here. Default writes nothing. --authorize creates a config only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'protocols/full151105_solver_phase_penalty_qualified_GPU_CPU_prereg_v4.json'
PROTOCOL_SHA='c9a746f929d5943a03067c67d5b28f23e09b093084c2cde07cfae05f5db60028'
OWN={'consumer':ROOT/'src/full151_gradient/penalty_observability105_v3.py',
     'CLI':ROOT/'scripts/diagnose_solver_phase_penalty_cpu_v3.py',
     'checker':ROOT/'scripts/check_solver_phase_penalty_cpu_v3.py'}
CONFIG=ROOT/'configs/full151105-solver-phase-penalty-qualified-GPU-CPU-v1.json'
OUTPUT=ROOT/'outputs/full151105_solver_phase_penalty_GPU_qualified_CPU_v1'
REVIEW=ROOT/'outputs/full151105_solver_phase_penalty_qualified_GPU_builder_source_only_v3/READY_source_only.json'
DELTA=ROOT/'proposals/full151105_solver_phase_penalty_qualified_GPU_input_v3/revision4_metadata_shape_delta.json'
DELTA_SHA='d72b383539457943754d617ffaa5382c11e2dfda3f6662b1033d6793d4383f05'
# Finalized against the actual new immutable CPU proof before source freeze.
READY_SCHEMA='full151105_qualified_dense_penalty_CPU_contract_READY:1'
PROOF_SCHEMA='full151105_qualified_dense_penalty_CPU_contract_proof:1'
READY_STATUS='PASS_SYNTHETIC_QUALIFIED_PACKET_ALIGNMENT_AND_HOOK_ONLY'


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    p=Path(path);need(p.is_file(),'Missing file: '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest()}


def checked(v):
    need(type(v) is dict and set(v)=={'path','sha256'} and type(v['path']) is str and Path(v['path']).is_absolute(),'Exact absolute pin required')
    need(type(v['sha256']) is str and len(v['sha256'])==64 and pin(v['path'])['sha256']==v['sha256'],'Pinned bytes changed: '+v['path'])
    return Path(v['path'])


def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def allchecks(v):
    c=v.get('checks');need(type(c) is dict and c and all(type(x) is bool and x for x in c.values()),'Strict alltrue actual checks required')
    need(type(v.get('passed')) is int and type(v.get('total')) is int and v['passed']==v['total']==len(c),'Actual proof counts differ')


def add(target,prefix,values):
    need(type(values) is dict and values,'Nonempty flat source map required')
    for k,v in values.items():
        need(type(k) is str and prefix+k not in target,'Source alias collision');checked(v)
        target[prefix+k]=v  # Preserve recorded paths; identity uses samefile+SHA.


def build(ready_path,authorize=False):
    need(type(authorize) is bool,'Explicit authorization bool required')
    need(not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')),'Fresh metadata process only')
    need(not any(v.startswith('PENDING_') for v in (READY_SCHEMA,PROOF_SCHEMA,READY_STATUS)),'Wait for final current-source actual CPU READY')
    ready=read(ready_path);proof=read(checked(ready['cpu_report']));allchecks(proof)
    need(ready.get('schema')==READY_SCHEMA and ready.get('status')==READY_STATUS and proof.get('schema')==PROOF_SCHEMA and proof.get('status')==READY_STATUS,'Exact new qualified-packet CPU proof required')
    need(ready['cpu_checks']=={'passed':proof['passed'],'total':proof['total']},'READY/proof counts differ')
    need(ready.get('CPU_scientific_run_authorized') is False and ready.get('GPU_run_authorized') is False,'Preparation is not scientific execution authorization')
    scope=proof['scope']
    for key in ('actual_physical_state_conversions','actual_MJB_loads','actual_objective_constructs','actual_full05_data_reads','actual_GPU_packet_array_reads','actual_WM_Actor_weights_Env_physics_GPU'):
        need(type(scope.get(key)) is int and scope[key]==0,'CPU fixture scope elevated: '+key)
    need(scope.get('actual_dense_qualification') is False,'Synthetic proof cannot qualify actual dense data')
    need(scope.get('runtime_progress_fields_used_as_final_counts') is False,'Runtime progress cannot replace final actual counts')
    need(pin(PROTOCOL)['sha256']==PROTOCOL_SHA and same(ready['protocol'],pin(PROTOCOL)),'Frozen rev4 input protocol differs')
    need(same(proof['protocol'],pin(PROTOCOL)),'CPU proof belongs to another protocol')
    need(set(ready['source_pins'])==set(proof['source_pins'])==set(OWN),'Exact three CPU-tested source keys required')
    for key,path in OWN.items():
        need(same(ready['source_pins'][key],pin(path)) and same(proof['source_pins'][key],pin(path)),'Actual CPU-tested source differs: '+key)
    need(set(ready['dependency_pins'])==set(proof['dependency_pins']),'CPU dependency keysets differ')
    for key,value in ready['dependency_pins'].items():
        need(same(value,proof['dependency_pins'][key]),'CPU dependency proof/source differs: '+key)
    actual_meta=read(checked(ready['actual_metadata_preflight']))
    need(actual_meta['schema']=='full151105_qualified_dense_actual_metadata_preflight:1' and actual_meta['status']=='PASS_ACTUAL_REPLAY_METADATA_ONLY','Actual replay metadata-only evidence required')
    need(same(actual_meta['consumer'],pin(OWN['consumer'])) and same(actual_meta['protocol'],pin(PROTOCOL)),'Metadata proof/source/protocol differs')
    need(type(actual_meta['NPZ_decode_attempted']) is int and actual_meta['NPZ_decode_attempted']==0 and all(actual_meta[k] is False for k in ('Torch_imported','Warp_imported','MuJoCo_imported','actual_alignment_or_scoring')),'Metadata prerequisite elevated scientific scope')
    protocol=read(PROTOCOL)
    need(protocol['schema']=='full151105_penalty_solver_phase_observability_CPU_preregistration:1' and type(protocol['revision']) is int and protocol['revision']==4,'Wrong source-only protocol namespace')
    need(all(protocol[k] is False for k in ('CPU_run_authorized','GPU_run_authorized','training_authorized')),'Protocol is not execution authorization')
    for v in protocol['inputs'].values():checked(v)
    independent=read(checked(protocol['inputs']['GPU_replay_saved_independent']));allchecks(independent)
    need(independent['schema']=='full151105_matched_native_GPU_reader_saved_independent:1' and independent['status']=='PASS_SAVED_NATIVE_REPLAY_REVIEW' and independent['passed']==independent['total']==29,'Actual29 saved matched-replay review required')
    need(all(independent[k] is True for k in ('engineering_qualified','state_replay_qualified','future_scoring_inputs_qualified')) and independent['dense_scoring_eligibility'] is False,'Input qualification is not scoring authorization')
    for field,alias in [('config','GPU_replay_config'),('producer_report','GPU_replay_report'),('qualified_prestate','qualified_GPU_prestate1808')]:
        need(same(independent[field],protocol['inputs'][alias]),'Independent review points to another run: '+field)
    replay_config=read(checked(protocol['inputs']['GPU_replay_config']))
    need(Path(independent['run_dir']).samefile(Path(protocol['inputs']['GPU_replay_report']['path']).parent),'Independent run directory differs')
    need(pin(DELTA)['sha256']==DELTA_SHA,'Protocol source-delta record changed')

    # Import only the immutable consumer proved by the new actual CPU checker.
    # This API is JSON/hash-only. Never call run_analysis, factory or converter.
    sys.path.insert(0,str(ROOT/'src'))
    core=importlib.import_module('full151_gradient.penalty_observability105_v3')
    need(same(pin(core.__file__),pin(OWN['consumer'])),'Unexpected imported consumer')
    sources={};add(sources,'required__',core.required_sources(protocol))
    add(sources,'actual_replay__',replay_config['source_pins'])
    add(sources,'protocol_source__',protocol['source_pins'])
    add(sources,'protocol_input__',protocol['inputs'])
    add(sources,'CPUtested__',ready['source_pins'])
    add(sources,'CPUdependency__',ready['dependency_pins'])
    sources['qualified_dense_CPU_ready']=pin(ready_path);sources['qualified_dense_CPU_report']=ready['cpu_report']
    sources['qualified_dense_actual_metadata_preflight']=ready['actual_metadata_preflight']
    sources['qualified_dense_protocol']=pin(PROTOCOL);sources['qualified_dense_protocol_delta']=pin(DELTA)
    sources['matched_replay_saved_auditor']=independent['auditor'];checked(independent['auditor'])
    sources['matched_replay_saved_math']=independent['math'];checked(independent['math'])
    review=read(REVIEW)
    need(review['schema']=='full151105_qualified_GPU_dense_builder_source_only:1' and review['AST_parse_only'] is True and review['builder_executed'] is False,'Current builder AST-only receipt required')
    need(same(review['source_pins']['builder'],pin(__file__)),'Builder changed since AST source receipt')
    sources['qualified_dense_builder']=pin(__file__);sources['qualified_dense_builder_AST_review']=pin(REVIEW)
    need(not CONFIG.exists() and not OUTPUT.exists(),'Fresh config/output only; no overwrite/resume/retry')
    c={'schema':core.SCHEMA,'role':core.ROLE,'protocol':pin(PROTOCOL),'output':str(OUTPUT),'CPU_run_authorized':authorize,'cpu_threads':1,'source_pins':sources,'cpu_ready':pin(ready_path)}
    need(c['schema']==protocol['runplan_schema'] and c['role']==protocol['role'] and set(c)==core.KEYS,'Actual consumer eight-key ABI differs')
    need(ready['api']['schema']==core.SCHEMA and ready['api']['role']==core.ROLE and ready['api']['keys']==sorted(core.KEYS),'Executed READY API differs')
    need(same(pin(ready['api']['module']),pin(OWN['consumer'])) and same(pin(ready['api']['entry']),pin(OWN['CLI'])),'READY module/CLI route differs')
    core.validate_header(c)
    need(not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')),'Default metadata reached a backend')
    for v in sources.values():checked(v)
    if authorize:
        with CONFIG.open('x',encoding='utf-8',newline='\n') as f:json.dump(c,f,indent=2);f.write('\n')
    return {'schema':'full151105_qualified_GPU_dense_builder_declaration:1','status':'CREATED_EXPLICITLY_AUTHORIZED_CPU_CONFIG_ONLY' if authorize else 'DECLARED_NO_CONFIG_WRITTEN',
      'config':pin(CONFIG) if authorize else {'path':str(CONFIG),'written':False},'output':str(OUTPUT),'protocol':pin(PROTOCOL),'cpu_ready':pin(ready_path),'cpu_report':ready['cpu_report'],
      'source_pin_count':len(sources),'CPU_run_authorized':authorize,'Torch_imported':False,'Warp_imported':False,'NPZ_opened':0,'native_state_calls':0,'MJB_loads':0,'factory_calls':0,'proxy_WM_Actor_forward':0,'Env_physics_GPU':0,'old_failure_changed':False,'new_qualification_granted':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ready',required=True,type=Path);p.add_argument('--authorize',action='store_true',help='Create a fresh CPU-authorized config only; never run it.')
    a=p.parse_args();print(json.dumps(build(a.ready,a.authorize),indent=2))


if __name__=='__main__':main()
