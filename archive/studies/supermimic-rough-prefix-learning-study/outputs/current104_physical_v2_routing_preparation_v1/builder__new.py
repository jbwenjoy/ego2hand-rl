"""Build the fixed current104 physical-FD config on CPU, always GPU=false.

No Env, Actor, WorldModel, MuJoCo model, physics or direction generator is used.
Root authorization is a separate immutable config copy after source review.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,pin,checked,canonical,loadnp

PROTOCOL=ROOT/'protocols/current104-physical-fd-execution-v3.json'
PROPOSAL=ROOT/'proposals/current104_samewarp_physical_fd_v2/protocol.json'
PROPOSAL_SHA='ce9a9575b6aa7dc9b5b535bfe1cf7ff5be4db6065dc122ba4860561d40096e09'
DESTINATION=ROOT/'configs/current104-physical-FD-v2.json'
OUTPUT=ROOT/'outputs/current104_physical_FD_v2'
STAGE='GPU_physics_then_separateCPU'
ENTRY=ROOT/'scripts/probe_current104_physical_fd_v2.py'
SUPERVISOR=ROOT/'scripts/run_current104_physical_fd_v2.py'
WORKER=ROOT/'src/rough_prefix/physical_fd_v2.py'
TRANSACTION=ROOT/'src/rough_prefix/probe_transaction_v3.py'
WATCHDOG=Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')
IDENTITY=Path('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py')
NATIVE=Path('D:/workspace/dex-supermimic-research/src/dex_supermimic')
SITE=Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages')
KEYS={'schema','role','stage','evaluation_seed','input_bundle','output','directions','datasets','models',
      'source_pins','protocol','parent_proposal','GPU_run_authorized','worlds','hard_process_seconds',
      'soft_worker_seconds','automatic_retry','compile_fallback'}


def need(value,message):
    if not value:raise ValueError(message)


def pins_in(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:yield value
        else:
            for v in value.values():yield from pins_in(v)
    elif isinstance(value,list):
        for v in value:yield from pins_in(v)


def validate_protocol(protocol):
    """Only metadata/NPZ invariants; no model output or task-quality selection."""
    p=read(protocol)
    need(p.get('schema')=='rough_current104_physical_FD_execution_protocol:1'
         and type(p.get('protocol_revision')) is int and p['protocol_revision']==3,'Wrong explicit revision3 physical-FD protocol')
    need(p.get('parent_proposal')==pin(PROPOSAL) and p['parent_proposal']['sha256']==PROPOSAL_SHA,'Only integrated V2 proposal is admissible')
    q=read(checked(p['parent_proposal']))
    need(q.get('schema')=='current104_same_Warp_physical_FD_proposal:2'
         and q.get('GPU_run_authorized') is False and q.get('training_authorized') is False,'Unqualified or implicitly authorized parent proposal')
    expected={'stage':STAGE,'worlds':1,'evaluation_seed':0,'hard_process_seconds':240,'soft_worker_seconds':220,
              'automatic_retry':False,'compile_fallback':False,'GPU_run_authorized_by_protocol':False}
    for key,value in expected.items():
        need(type(p.get(key)) is type(value) and p[key]==value,'Protocol alias/type mismatch: '+key)
    need(p.get('input_bundle')==q['inputs']['input_bundle'],'Protocol/bundle alias differs')
    dp=q['inputs']['direction_preparation']
    wanted={'config':dp['plan'],'report':dp['report'],'arrays':dp['targets_and_vectors']}
    need(type(p.get('directions')) is dict and p['directions']==wanted,'Protocol/direction artifact aliases differ')
    result=p.get('producer_result_schema',{})
    for group,name,shape in [('branch_arrays','branch_object_terms_k',[560,2]),
                             ('branch_arrays','branch_object_terms_kplus1',[560,2]),
                             ('snapshot_arrays','donor_object_terms_k',[16,2]),
                             ('snapshot_arrays','donor_object_terms_kplus1',[16,2])]:
        need(result.get(group,{}).get(name)==shape,'Revision3 native FP32 reward term ABI missing: '+name)
    for value in pins_in(q):checked(value)
    for value in pins_in(p):checked(value)
    prep=read(checked(wanted['config']));report=read(checked(wanted['report']))
    need(prep.get('schema')=='rough_physical_probe_direction_CPU_preparation:1'
         and prep.get('GPU_authorized') is False and prep.get('physical_branches_authorized') is False
         and prep.get('automatic_retry') is False,'Directions are not CPU-only immutable preparation')
    need(prep['path_order']==['A0','A1L','P_ONLY','A2'] and prep['phase_order']==[0,4,8,12], 'Direction state selection changed')
    for key,value in {'seed':2026110401,'direction_count':8,'guard_ulps':8}.items():
        need(type(prep.get(key)) is int and prep[key]==value,'Direction integer definition changed: '+key)
    need(canonical(prep['epsilons'])==canonical([.002,.001]),'Direction epsilon changed')
    need(report.get('schema')=='rough_physical_probe_direction_CPU_result:1'
         and report.get('status')=='PREPARED_NOT_PHYSICALLY_QUALIFIED'
         and report.get('plan')==wanted['config'] and report.get('dataset')==wanted['arrays'],'Prepared direction report is not artifact-bound')
    for key in ('selection_uses_model_outputs','selection_uses_physical_perturbation_outputs','targets_clipped_or_resampled',
                'env_created','GPU_used','Torch_or_Warp_imported'):
        need(report.get(key) is False,'Direction preparation scope failed: '+key)
    need(report.get('source_and_input_pins_unchanged') is True and report.get('global_NumPy_RNG_unchanged') is True,'Prepared source/RNG preservation failed')
    need(type(report.get('physics_steps')) is int and report['physics_steps']==0
         and type(report.get('WM_forwards')) is int and report['WM_forwards']==0,'Directions used physical/model outcomes')
    a=loadnp(checked(wanted['arrays']))
    shapes={'centers':((16,20),np.float32),'lower':((20,),np.float32),'upper':((20,),np.float32),
            'free_mask':((16,20),np.bool_),'vectors':((16,8,20),np.float64),
            'plus':((16,8,2,20),np.float32),'minus':((16,8,2,20),np.float32),
            'valid_pair':((16,8,2),np.bool_),'effective_vectors':((16,8,2,20),np.float64)}
    for key,(shape,dtype) in shapes.items():
        need(a[key].shape==shape and a[key].dtype==dtype and np.isfinite(a[key]).all(),'Direction ABI mismatch: '+key)
    need(np.array_equal(a['epsilons'],np.asarray([.002,.001],np.float64)),'Prepared target epsilon table changed')
    need(np.all(a['lower']<a['upper']) and np.all((a['centers']>=a['lower'])&(a['centers']<=a['upper'])),'Prepared baseline targets outside inset domain')
    need(np.all((a['plus']>=a['lower'])&(a['plus']<=a['upper']))
         and np.all((a['minus']>=a['lower'])&(a['minus']<=a['upper'])),'Prepared perturbations outside inset domain')
    need(len(report['rows'])==16,'Direction report omitted states')
    for i,(name,k) in enumerate((name,k) for name in prep['path_order'] for k in prep['phase_order']):
        row=report['rows'][i]
        need(row['path']==name and type(row['phase']) is int and row['phase']==k,'Direction row identity/order differs')
        need(np.array_equal(np.asarray(row['nominal_u'],np.float32),a['centers'][i]),'Direction center/report differs')
    return p,q


def build_config(protocol=PROTOCOL,evidence=()):
    protocol=Path(protocol).resolve(strict=True);p,q=validate_protocol(protocol)
    datasets=copy.deepcopy(q['inputs']['saved_trajectories']);models=copy.deepcopy(q['inputs']['models'])
    need([d['name'] for d in datasets]==['A0','A1L','P_ONLY','A2'],'Wrong four-path data matrix')
    need([m['method'] for m in models]==['A1L','P_ONLY','A2'],'Wrong fixed final-model matrix')
    sources={};seen={}
    def add(label,value):
        checked(value);identity=str(Path(value['path']).resolve()).casefold()
        if identity in seen:
            need(seen[identity]['sha256']==value['sha256'],'Conflicting source SHA for one logical path')
            return
        seen[identity]=value;sources[label]=copy.deepcopy(value)
    named={'entry':ENTRY,'supervisor':SUPERVISOR,'worker':WORKER,'builder':Path(__file__),
           'protocol':protocol,'parent_proposal':PROPOSAL,'input_bundle':checked(p['input_bundle']),
           'transaction_helper':TRANSACTION,'identity_driver':IDENTITY,'watchdog':WATCHDOG,
           'runtime_IO':ROOT/'vendor/runtime_io.py'}
    for label,path in named.items():add(label,pin(path))
    for model in models:
        train=read(checked(model['training_config']));health=read(checked(model['final_health']))
        need(type(model['seed']) is int and model['seed']==0 and type(model['iteration']) is int and model['iteration']==2000,'Only declared final2000/seed0 models')
        need(train['method']==model['method'] and type(train['seed']) is int and train['seed']==0
             and train['role']=='primary_matched' and train['input_bundle']==p['input_bundle'],'Training model/data identity differs')
        need(type(train['training']['max_iterations']) is int and train['training']['max_iterations']==2000,'Training final-iteration declaration differs')
        need(train['controller']=={'kind':'ROUGH_PREFIX_BASELINE_EMA_C2','format_version':104,'margin_rad':.020,'resume':'from_scratch_only'},'Wrong104 controller provenance')
        need(Path(model['checkpoint']['path']).resolve()==Path(train['output']).resolve()/'train/model_final.pt','Not the fixed actual final checkpoint')
        need(health.get('schema')=='rough104_checkpoint_CPU_health:1' and health.get('status')=='PASS_SAVED_ENGINEERING_HEALTH' and health.get('checkpoint')==model['checkpoint']
             and health.get('method')==model['method'] and type(health.get('iteration')) is int and health['iteration']==2000
             and health.get('final_iteration_scope') is True and health['config']['sha256']==model['training_config']['sha256']
             and type(health.get('checks')) is dict and health['checks'] and all(v is True for v in health['checks'].values()),'Actual final health/provenance mismatch')
        need(type(health.get('passed')) is int and type(health.get('total')) is int
             and health['passed']==health['total']==len(health['checks'])
             and type(health.get('seed')) is int and health['seed']==0,'Malformed final health counter/seed types')
        checked(health['config']);checked(health['auditor'])
        need(health['auditor']['sha256']=='7a6a4e21d24f5acad0eccc6908bef03fd7e1c4acaec7196edf296cc75772850f','Require unchanged final104 v3 health producer')
        for value in train['source_pins'].values():add('closure__'+str(len(sources)).zfill(4),value)
    extra=[ROOT/'src/rough_prefix'/name for name in ('io.py','metadata.py','bundle.py','bundle_v2.py','native_v2.py',
            'runtime_v2.py','controller.py','capacity.py','evaluation_v2.py','policy_probe.py','probe.py')]
    extra += [ROOT/'vendor/measured_contacts.py',NATIVE/'rl/dex_supermimic/snapshot.py']
    extra += [SITE/'mjlab'/name for name in ('sim/sim.py','scene/scene.py','sensor/sensor.py','sensor/contact_sensor.py',
              'sensor/sensor_context.py','managers/reward_manager.py','managers/command_manager.py',
              'managers/event_manager.py')]
    extra += [SITE/'mujoco_warp'/name for name in ('__init__.py','_src/__init__.py','_src/io.py','_src/support.py','_src/types.py',
              '_src/forward.py','_src/solver.py','_src/constraint.py','_src/sensor.py','_src/smooth.py',
              '_src/passive.py','_src/history.py','_src/math.py','_src/warp_util.py','_src/collision_driver.py',
              '_src/collision_core.py','_src/collision_primitive.py','_src/collision_primitive_core.py',
              '_src/collision_convex.py','_src/collision_gjk.py','_src/collision_sdf.py','_src/collision_flex.py',
              '_src/bvh.py','_src/island.py','_src/sleep.py','_src/block_cholesky.py','_src/derivative.py',
              '_src/inverse.py','_src/ray.py','_src/render.py','_src/render_util.py',
              '_src/util_misc.py','_src/util_pkg.py','_src/cli.py')]
    for path in extra:add('closure__'+str(len(sources)).zfill(4),pin(path))
    for value in list(pins_in(q))+list(pins_in(p)):
        add('bound_data_and_evidence__'+str(len(sources)).zfill(4),value)
    for path in evidence:add('independent_evidence__'+str(len(sources)).zfill(4),pin(path))
    cfg={'schema':'rough_current104_physical_FD_run:1','role':'current104_same_Warp_physical_FD',
         'stage':STAGE,'evaluation_seed':0,'input_bundle':copy.deepcopy(p['input_bundle']),'output':str(OUTPUT),
         'directions':copy.deepcopy(p['directions']),'datasets':datasets,'models':models,'source_pins':sources,
         'protocol':pin(protocol),'parent_proposal':pin(PROPOSAL),'GPU_run_authorized':False,'worlds':1,
         'hard_process_seconds':240,'soft_worker_seconds':220,'automatic_retry':False,'compile_fallback':False}
    need(set(cfg)==KEYS,'Builder produced an unexpected header')
    need(not OUTPUT.exists(),'Never overwrite or retry existing physical probe output')
    need(not OUTPUT.with_name(OUTPUT.name+'_watchdog').exists(),'Never overwrite an existing owned watchdog attempt')
    need(not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')),'CPU builder imported simulation/framework package')
    return cfg


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,default=PROTOCOL)
    ap.add_argument('--evidence',type=Path,action='append',default=[]);args=ap.parse_args()
    cfg=build_config(args.protocol,args.evidence)
    DESTINATION.parent.mkdir(parents=True,exist_ok=True)
    with DESTINATION.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(cfg,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(canonical(pin(DESTINATION)))


if __name__=='__main__':main()
