"""New105 policy-evaluation CPU contracts; no actual Env, checkpoint or physics."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/full151_policy_probe_CPU_v1'
if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':raise ValueError('Explicit CPU mask -1 required')
if OUT.exists():raise FileExistsError('Preserve every CPU checker attempt')
OUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(ROOT/'src'))
from full151_native import policy_probe as pp, evaluation as ev
from full151_native.io import pin,canonical
sources={p.name:pin(p) for p in [Path(__file__),ROOT/'src/full151_native/policy_probe.py',ROOT/'src/full151_native/evaluation.py',ROOT/'src/full151_native/probe.py',ROOT/'src/full151_native/native.py',ROOT/'src/full151_native/checkpoint.py',ROOT/'src/full151_native/controller.py',ROOT/'src/full151_native/runtime.py',pp.PURE_PATH]}
snap=OUT/'source_before';snap.mkdir()
for name,value in sources.items():(snap/name).write_bytes(Path(value['path']).read_bytes())
checks={};details={}
def check(name,condition):
    try:
        assert bool(condition() if callable(condition) else condition)
        checks[name]=True
    except BaseException:
        checks[name]=False;details[name]=traceback.format_exc()
def rejects(fn):
    try:fn()
    except (ValueError,KeyError,TypeError):return True
    return False
check('default_new_modules_import_noTorchWarpMujoco',not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')))

clock=pp.fp32_clock_trace();lo=-np.ones(20);hi=np.ones(20)
events=[{'actualu':np.full((1,20),.1)} for _ in range(150)]
events[0]['actualu'][:]=.4
rate=pp.policy_rate(np.zeros(20),events)
check('startup_to_learnedU0_included_all150differences',rate['max_rate_rad_s']==20. and rate['startup_rate_rad_s']==20. and np.isclose(rate['subsequent_updates_max_rate_rad_s'],15.) and rate['rate_difference_count']==150)
check('partial149_controls_not_fullrate_qualification',rejects(lambda:pp.policy_rate(np.zeros(20),events[:-1])))
check('no_terminalU150_padding_accepted',rejects(lambda:pp.policy_rate(np.zeros(20),events+events[-1:])))
check('startup_wrongshape_rejected',rejects(lambda:pp.policy_rate(np.zeros((2,10)),events)))
bad=copy.deepcopy(events);bad[77]['actualu'][0,0]=np.nan
check('nonfinite_applied_control_rejected',rejects(lambda:pp.policy_rate(np.zeros(20),bad)))
states=[{'object_pose_w':np.array([0,0,.5,1,0,0,0.]),'joint_rad':np.zeros(20),'ctrl_rad':np.zeros(20) if k==0 else events[(k-1)//4]['actualu'][0].copy(),'raw_time_s':float(clock[k]),'external_force_max_abs':0.,'ctrl_exact':True} for k in range(601)]
solver=[{'actuator_force_solver_native':np.zeros(20),'penetration_m':0.,'object_environment_contacts':0} for _ in range(600)]
ref={'object_COM_world_m':np.tile([0.,0.,.5],(601,1)),'object_rotation_w':np.tile(np.eye(3),(601,1,1))}
args=(states,solver,{'status':'OK'},ref,lo,hi,-np.ones(20),np.ones(20),.5)
old=pp.frozen_A0_grade(*args);new=pp.grade_policy(*args,events)
check('old149rate_misses_first_learned_jump_new150rate_fails',old['max_actual_target_rate_rad_s']==5 and old['gates']['actual_target_rate'] is True and new['max_actual_target_rate_rad_s']==20 and new['gates']['actual_target_rate'] is False)
check('other12_gates_clock13criteria_unchanged',{k:v for k,v in old['gates'].items() if k!='actual_target_rate'}=={k:v for k,v in new['gates'].items() if k!='actual_target_rate'} and new['criteria']==old['criteria']==pp.GATES)
check('quality_not_A0_or_training_pass','native_A0_pass' not in new and new['native_policy_pass'] is False and new['status']=='COMPLETED_NATIVE_POLICY_QUALITY_RECORDED')
check('incomplete_nativegrade_UNKNOWN',pp.grade_policy(states[:-1],solver[:-1],{'status':'UNKNOWN'},ref,lo,hi,-np.ones(20),np.ones(20),.5,events[:-1])['native_policy_pass'] is None)

from full151_native.bundle import load_bundle
bundle=load_bundle(ROOT/'outputs/full151_native_input_v1/bundle.json',expected_sha256=pin(ROOT/'outputs/full151_native_input_v1/bundle.json')['sha256'])
fixture=OUT/'metadata_fixture';fixture.mkdir();run=fixture/'training_run';run.mkdir()
ckpt=run/'model_final.pt';ckpt.write_bytes(b'NON_CHECKPOINT_METADATA_FIXTURE_ONLY')
auditor=fixture/'synthetic_auditor.py';auditor.write_text('# Metadata fixture only. Never executed.\n')
training={'role':'engineering_smoke','method':'A1L','seed':0,'input_bundle':bundle['descriptor']['input_bundle'],'output':str(run),'training':{'max_iterations':40},'source_pins':{'controller':pin(ROOT/'src/full151_native/controller.py')}}
trainpath=fixture/'training_config.json';trainpath.write_text(json.dumps(training))
health={'schema':'full151105_checkpoint_CPU_health:1','status':'PASS_SAVED_ENGINEERING_HEALTH','checks':{'synthetic_metadata_only':True},'passed':1,'total':1,'method':'A1L','seed':0,'iteration':40,'role':'engineering_smoke','final_iteration_scope':True,'checkpoint':pin(ckpt),'config':pin(trainpath),'run_dir':str(run),'auditor':pin(auditor),'mode':bundle['descriptor']['mode'],'retarget_variant_id':bundle['descriptor']['retarget_variant_id'],'input_bundle':bundle['descriptor']['input_bundle'],'source_pins':training['source_pins']}
healthpath=fixture/'health.json';healthpath.write_text(json.dumps(health))
config={'role':'native_policy_eval','method':'A1L','worlds':1,'seed':0,'checkpoint_iteration':40,'checkpoint':pin(ckpt),'training_config':pin(trainpath),'health':pin(healthpath),'expected_health_auditor':pin(auditor),'GPU_run_authorized':False}
check('synthetic_metadata_105health_scope_accepts_noActorload',lambda:ev.validate_evaluation_inputs(config,bundle)['health']==health)
check('GPUfalse_rejects_before_native_Actor_Env_entry',rejects(lambda:ev.run_evaluation(config,bundle,OUT,lambda **kw:None)))
for name,change in [('wrongrole',{'role':'engineering_smoke'}),('missing_full151_target_arm',{'method':'A2'}),('boolworld',{'worlds':True}),('boolseed',{'seed':True}),('changed_iteration',{'checkpoint_iteration':41})]:
    check('metadata_rejects_'+name,rejects(lambda change=change:ev.validate_evaluation_inputs({**config,**change},bundle)))
variants={'oldhealthschema':{'schema':'rough_prefix_checkpoint_CPU_health:1'},'falsecheck':{'checks':{'bad':False}},'boolpassed':{'passed':True},'badsource':{'source_pins':{}},'wrongmethod':{'method':'P_ONLY'},'wrongseed':{'seed':1},'wrongrole':{'role':'primary_matched'},'not_final':{'final_iteration_scope':False},'wrongbundle':{'input_bundle':pin(trainpath)},'wrongauditor':{'auditor':pin(trainpath)},'wrongconfig':{'config':pin(healthpath)}}
for name,change in variants.items():
    path=fixture/(name+'.json');path.write_text(json.dumps({**health,**change}))
    check('health_rejects_'+name,rejects(lambda path=path:ev.validate_evaluation_inputs({**config,'health':pin(path)},bundle)))
check('metadata_checks_still_noTorchWarp_import',not any(k in sys.modules for k in ('torch','warp','mujoco_warp')))

import torch
torch.set_num_threads(2)
pure_spec=importlib.util.spec_from_file_location('_full151_policy_CPU_C2',pp.PURE_PATH);pure=importlib.util.module_from_spec(pure_spec);pure_spec.loader.exec_module(pure)
check('exact_frozen_pure_C2_hash',pin(pp.PURE_PATH)['sha256']==pp.PURE_SHA256)
lower=np.full(20,-1,np.float32);upper=np.full(20,1,np.float32);bias=np.zeros((1,20),np.float32)
action=np.linspace(-1,1,20,dtype=np.float32)[None];previous=np.linspace(-.2,.2,20,dtype=np.float32)[None]
norm_rng=torch.get_rng_state().clone()
pred=pp.expected_control(action,previous,np.full(20,.1,np.float32),lower,upper,bias,pure.target_inset)
check('EMA_control_prediction_FP32_and_no_RNG',all(v.dtype==np.float32 for v in pred.values()) and torch.equal(norm_rng,torch.get_rng_state()))
sat=pp.expected_control(np.ones((1,20),np.float32),np.full((1,20),10,np.float32),np.zeros(20,np.float32),lower,upper,bias,pure.target_inset)
check('EMA_history_unmodified_by_C2_saturation',np.all(sat['residual']>1) and np.array_equal(sat['hard_bounded'],np.ones((1,20),np.float32)) and np.array_equal(sat['actual'],np.full((1,20),.98,np.float32)))
check('bias0_required',rejects(lambda:pp.expected_control(action,previous,np.zeros(20),lower,upper,np.ones((1,20)),pure.target_inset)))
check('nonfinite_action_required',rejects(lambda:pp.expected_control(np.full((1,20),np.nan),previous,np.zeros(20),lower,upper,bias,pure.target_inset)))
check('C2_empty_domain_rejected',rejects(lambda:pp.expected_control(action,previous,np.zeros(20),np.zeros(20),np.full(20,.03),bias,pure.target_inset)))

# Reuse only the old actual ActionTerm constructor/method fixture, never its
# test suite or optional Warp path. Static MJB reading is permitted.
import mujoco
denied={}
def deny(name):
    def call(*a,**kw):denied[name]+=1;raise AssertionError('Forbidden native call '+name)
    return call
patches=[]
for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_kinematics','mj_comPos','mj_collision','mj_jac','mj_jacBody','mj_contactForce','mj_geomDistance'):
    denied[name]=0;p=patch.object(mujoco,name,deny(name));p.start();patches.append(p)
try:
    source=Path('D:/workspace/supermimic-chord-study/scripts/check_pen_target_limits.py')
    spec=importlib.util.spec_from_file_location('_full151_policy_native_action_fixture',source);native_fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(native_fixture)
    _,entity,term,_,_=native_fixture.fixture(Path(bundle['model']['path']),torch.zeros(1,20))
    term._processed_actions[:]=torch.from_numpy(previous);term.process_actions(torch.from_numpy(action))
    check('oracle_residual_equals_actualNativeActionTerm',np.array_equal(term.processed_action.numpy(),pred['residual']))
    from dex_supermimic.rl.dex_supermimic.networks import Actor
    with torch.random.fork_rng(devices=[]):actor=Actor(155).eval().requires_grad_(False)
    before={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()};h=pp.state_digest(before)
    with torch.no_grad():actual_action=actor(pp._policy_obs({'policy':torch.zeros(1,155)}))
    after={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    check('actualNativeActor155_20_CPU_forward_noNorm_or_weights_change',tuple(actual_action.shape)==(1,20) and torch.isfinite(actual_action).all() and h==pp.state_digest(after))
    check('bad155obs_shape_and_dtype_rejected',rejects(lambda:pp._policy_obs({'policy':torch.zeros(1,154)})) and rejects(lambda:pp._policy_obs({'policy':torch.zeros(1,155,dtype=torch.float64)})))
    check('nonfinite155obs_rejected',rejects(lambda:pp._policy_obs({'policy':torch.full((1,155),float('nan'))})))
finally:
    for p in reversed(patches):p.stop()

newtree=ast.parse((ROOT/'src/full151_native/policy_probe.py').read_text());oldtree=ast.parse((ROOT/'src/full151_native/probe.py').read_text())
newrun=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name=='run_policy_probe')
oldrun=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='run_probe')
extract=lambda node,name:next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
check('new_step_and_forward_wrappers_exact_frozenA0_AST',all(ast.dump(extract(newrun,n),include_attributes=False)==ast.dump(extract(oldrun,n),include_attributes=False) for n in ('step','forward')))
calls=[n for n in ast.walk(newrun) if isinstance(n,ast.Call)]
check('one_envstep_call_and_fixed150loop',sum(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='env' and n.func.attr=='step' for n in calls)==1 and any(isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and len(n.iter.args)==1 and isinstance(n.iter.args[0],ast.Constant) and n.iter.args[0].value==150 for n in ast.walk(newrun)))
check('no_probe_reset_compute_normupdate_or_optimizer',not any(isinstance(n.func,ast.Attribute) and (n.func.attr in ('reset','compute','backward','zero_grad') or (n.func.attr=='update' and isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='norm')) for n in calls))
probe_text=(ROOT/'src/full151_native/policy_probe.py').read_text();eval_text=(ROOT/'src/full151_native/evaluation.py').read_text()
check('actor_state_devicecopies_precede_final_capacity_read',probe_text.index('actor_after =')<probe_text.index('observer.final_read()'))
check('eval_CPU_loader_precedes_CUDA_and_Manager',eval_text.index('actor,load_receipt=load_actor_checkpoint')<eval_text.index("wp.set_device('cuda:0')")<eval_text.index('env=ManagerBasedRlEnv'))
check('eval_health_is_before_any_Torch_import',eval_text.index('evidence=validate_evaluation_inputs')<eval_text.index('    import torch'))
check('eval_never_constructs_Runner_or_optimizer',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('DexSuperMimicRunner','Actor','WorldModel','Adam','AdamW','RecordedRunner') for n in ast.walk(ast.parse(eval_text))))
check('no_denied_nativeAPI_executed',not any(denied.values()))
check('CUDA_false_uninitialized_zero_devices',not torch.cuda.is_initialized() and not torch.cuda.is_available() and torch.cuda.device_count()==0)
check('frozen_sources_unchanged',all(pin(v['path'])==v for v in sources.values()))
report={'schema':'full151_105_policy_evaluation_CPU_contract:1','status':'PASS_CPU_CONTRACT_ACTUAL105LOAD_ENV_AND_PHYSICS_PENDING' if all(checks.values()) else 'FAIL_CPU_CONTRACT_PRESERVED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'details':details,'source_pins':sources,'actual_bundle':pin(ROOT/'outputs/full151_native_input_v1/bundle.json'),'actual_ActionTerm_fixture_source':pin(source),'denied_native_API_calls':denied,'CUDA_initialized':torch.cuda.is_initialized(),'Warp_transitively_imported':('warp' in sys.modules),'scope':{'synthetic_health_metadata_not_actual_health':True,'synthetic_checkpoint_bytes_not_deserialized':True,'actual_native_Actor_random_CPU_fixture_only':True,'actual_ActionTerm_CPU_method_fixture':True,'actual_native_Env_constructed':False,'actual_105_trained_checkpoint_loaded':False,'run_policy_probe_executed':False,'physics_FK_collision_WM_GPU_training':False}}
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'report':pin(OUT/'report.json'),'passed':report['passed'],'total':report['total'],'status':report['status']}))
if not all(checks.values()):raise SystemExit(2)
