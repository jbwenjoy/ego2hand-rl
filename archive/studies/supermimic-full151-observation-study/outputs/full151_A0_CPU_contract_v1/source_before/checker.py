"""CPU A0 helper contracts; native Env/physics/FK/query/model forwards forbidden."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import ast
import copy
import hashlib
import json
import os
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/full151_A0_CPU_contract_v1'
if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':raise RuntimeError('Use explicit CPU-only CUDA_VISIBLE_DEVICES=-1')
if (OUT/'report.json').exists():raise FileExistsError('Never overwrite an executed CPU proof')
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from full151_native import native,probe
from full151_native.io import pin,canonical

sources={k:pin(p) for k,p in {
    'native':ROOT/'src/full151_native/native.py','probe':ROOT/'src/full151_native/probe.py',
    'io':ROOT/'src/full151_native/io.py','metadata':ROOT/'src/full151_native/metadata.py',
    'bundle':ROOT/'src/full151_native/bundle.py','checker':Path(__file__),
}.items()}
snap=OUT/'source_before';snap.mkdir(exist_ok=False)
for k,v in sources.items():(snap/(k+'.py')).write_bytes(Path(v['path']).read_bytes())
checks={};details={};denied={}
initial_imports={k:k in sys.modules for k in ('torch','warp','mujoco')}
def check(name,fn):
    try:
        value=fn()
        if not value:raise AssertionError('False check result')
        checks[name]=True
    except BaseException:
        checks[name]=False;details[name]=traceback.format_exc()
def reject(fn,typ=ValueError):
    try:fn()
    except typ:return True
    return False
def deny(name):
    def call(*a,**kw):denied[name]=denied.get(name,0)+1;raise AssertionError('CPU check forbids '+name)
    return call

check('native_probe_default_import_no_GPU_framework',lambda:not any(initial_imports.values()))
clock=probe.fp32_clock_trace()
check('clock600_FP32_endpoint_expected',lambda:clock.dtype==np.float32 and len(clock)==601 and float(clock[-1])==3.00002121925354)
ca=probe.clock_audit(clock.astype(float))
check('new_clock_bitexact_AND50us_pass_legacy2us_diagnosticfail',lambda:ca['pass'] and ca['all601_expected_FP32_bits_equal'] and ca['max_abs_nominal_drift_s']==.000021219253540039062 and ca['legacy_2us_diagnostic_pass'] is False)
badclock=clock.copy();badclock[299]=np.nextafter(badclock[299],np.float32(np.inf))
check('clock_single_bit_corruption_rejected',lambda:probe.clock_audit(badclock)['pass'] is False)
check('clock_partial_unknown',lambda:probe.clock_audit(clock[:600])['pass'] is False)
check('state4_uses_u0_state5_u1_state600_u149',lambda:[probe.interval_used_by_poststep(k) for k in (0,1,4,5,596,597,600)]==[0,0,0,1,148,149,149])
check('terminal_extra_step_and_bool_index_rejected',lambda:reject(lambda:probe.interval_used_by_poststep(601)) and reject(lambda:probe.interval_used_by_poststep(True)))

# Execute only the two wrappers extracted from the candidate source against
# no-device callables. Distinguish an API exception from a later sample failure.
probe_ast=ast.parse((ROOT/'src/full151_native/probe.py').read_text())
run_node=next(n for n in probe_ast.body if isinstance(n,ast.FunctionDef) and n.name=='run_probe')
wrapper_nodes=[copy.deepcopy(n) for n in run_node.body if isinstance(n,ast.FunctionDef) and n.name in ('step','forward')]
def wrapper_fixture(api_error=None,sample_error=None):
    counts={'step_attempted':0,'step_returned':0,'forward_attempted':0,'forward_returned':0}
    calls={'step':0,'forward':0,'sample':0}
    def api(name):
        def call():
            calls[name]+=1
            if api_error==name:raise RuntimeError('synthetic '+name+' API error')
            return 'native_return'
        return call
    def sample(index):
        calls['sample']+=1
        if sample_error:raise RuntimeError('synthetic sample error after returned step')
    scope={'states':[0],'deadline':float('inf'),'time':NS(perf_counter=lambda:0.),
           'api_counts':counts,'original_step':api('step'),'original_forward':api('forward'),'sample':sample}
    exec(compile(ast.Module(body=wrapper_nodes,type_ignores=[]),'<candidate-wrapper-fixture>','exec'),scope)
    return scope,counts,calls
s,c,calls=wrapper_fixture();s['step']();s['forward']()
check('source_wrapper_success_attempts_returns_sample_separate',lambda:c=={'step_attempted':1,'step_returned':1,'forward_attempted':1,'forward_returned':1} and calls=={'step':1,'forward':1,'sample':1})
s,c,calls=wrapper_fixture(api_error='step')
check('source_wrapper_API_throw_retains_attempt_without_return_or_sample',lambda:reject(s['step'],RuntimeError) and c['step_attempted']==1 and c['step_returned']==0 and calls['sample']==0)
s,c,calls=wrapper_fixture(sample_error=True)
check('source_wrapper_sample_failure_preserves_returned_physics_API_count',lambda:reject(s['step'],RuntimeError) and c['step_returned']==1 and calls['sample']==1)
s,c,calls=wrapper_fixture(api_error='forward')
check('source_wrapper_forward_throw_not_misreported_as_completed',lambda:reject(s['forward'],RuntimeError) and c['forward_attempted']==1 and c['forward_returned']==0)

states=[{'object_pose_w':np.array([0,0,.5,1,0,0,0.]),'joint_rad':np.zeros(20),'ctrl_rad':np.zeros(20),
         'raw_time_s':float(clock[k]),'external_force_max_abs':0.,'ctrl_exact':True} for k in range(601)]
solver=[{'actuator_force_solver_native':np.zeros(20),'penetration_m':0.,'object_environment_contacts':0} for _ in range(600)]
reference={'object_COM_world_m':np.tile([0.,0.,.5],(601,1)),'object_rotation_w':np.tile(np.eye(3),(601,1,1))}
lo=-np.ones(20);hi=np.ones(20)
def grade(st=states,sv=solver,cap=None):return probe.grade(st,sv,{'status':'OK'} if cap is None else cap,reference,lo,hi,-np.ones(20),np.ones(20),.5)
perfect=grade()
check('synthetic_full601_grade_exact13_gates',lambda:len(perfect['gates'])==13 and perfect['native_A0_pass'] is True)
partial=grade(states[:600])
check('missing_terminal_state_cannot_pass',lambda:partial['native_A0_pass'] is None)
check('capacity_UNKNOWN_cannot_pass',lambda:grade(cap={'status':'UNKNOWN'})['native_A0_pass'] is None)
between=copy.deepcopy(states);between[1]['object_pose_w'][0]=.03
check('151knot_primary_vs601diagnostic_not_silently_tightened',lambda:grade(between)['gates']['object_position_151'] and grade(between)['max_object_position_error_601_m']==.03)
pos=copy.deepcopy(states);pos[400]['object_pose_w'][0]=.03
check('primary_position_failure_recorded',lambda:grade(pos)['gates']['object_position_151'] is False and grade(pos)['status']=='COMPLETED_NATIVE_A0_QUALITY_RECORDED')
axis=copy.deepcopy(states);a=np.deg2rad(21)/2;axis[400]['object_pose_w'][3:]=[np.cos(a),0,0,np.sin(a)]
check('shaft_metric_localx_21deg_fails',lambda:grade(axis)['gates']['directed_shaft_151'] is False)
q=copy.deepcopy(states);q[400]['joint_rad'][0]=1.0015
check('actual_joint_1mrad_original_margin_retained',lambda:grade(q)['gates']['actual_q_all601'] is False)
deep=copy.deepcopy(solver);deep[399]['penetration_m']=.0031
check('original3mm_penetration_gate_retained',lambda:grade(sv=deep)['gates']['penetration_solver600'] is False)
fast=copy.deepcopy(states)
for k in range(5,9):fast[k]['ctrl_rad'][0]=.31
check('149updates_includes_state5_first_new_target',lambda:grade(fast)['gates']['actual_target_rate'] is False and grade(fast)['max_actual_target_rate_rad_s']==15.5)
nan=copy.deepcopy(states);nan[500]['joint_rad'][1]=np.nan
check('nonfinite_completed_arrays_invalid_not_falsequality',lambda:reject(lambda:grade(nan)))

# Actual MJB static binding and the actual factory config are permitted; every
# physical/kinematic API and actual Env constructor is denied before their use.
import mujoco
for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_kinematics','mj_comPos','mj_collision','mj_jac','mj_jacBody','mj_jacSite','mj_geomDistance','mj_contactForce'):
    denied[name]=0
patches=[patch.object(mujoco,name,deny(name)) for name in denied]
for ctx in patches:ctx.start()
try:
    from full151_native.bundle import load_bundle
    bundle_path=ROOT/'outputs/full151_native_input_v1/bundle.json'
    b=load_bundle(bundle_path,expected_sha256=pin(bundle_path)['sha256'])
    cfg={'role':'native_a0','method':'A0','training':{'num_envs':1},'seed':0,'GPU_run_authorized':False}
    descriptor_before=canonical(b['descriptor'])
    check('actual151_bundle_passes_A0_metadata_input_shape_contract',lambda:native.validate_a0_inputs(cfg,b) is None)
    check('A0_device_entry_rejects_no_authorization_before_framework_entry',lambda:reject(lambda:native.run_native(cfg,b,OUT,lambda **kw:None)))
    for role in ('engineering_smoke','primary_matched','learned_evaluation'):
        check('A0_rejects_role_'+role,lambda role=role:reject(lambda:native.validate_a0_inputs({**cfg,'role':role},b)))
    check('A0_rejects_old15_baseline',lambda:reject(lambda:native.validate_a0_inputs(cfg,{**b,'baseline_u150':b['baseline_u150'][:15]})))
    check('A0_rejects151_target_padding',lambda:reject(lambda:native.validate_a0_inputs(cfg,{**b,'baseline_u150':np.r_[b['baseline_u150'],b['baseline_u150'][-1:]]})))
    check('A0_rejects_FP64_baseline',lambda:reject(lambda:native.validate_a0_inputs(cfg,{**b,'baseline_u150':b['baseline_u150'].astype(float)})))
    check('A0_rejects_oldmode_even_with151_shapes',lambda:reject(lambda:native.validate_a0_inputs(cfg,{**b,'descriptor':{**b['descriptor'],'mode':'rough_observation_reference_prefix16_v1'}})))
    changed={**b['descriptor'],'clock_contract':{**b['descriptor']['clock_contract'],'nominal_drift_gate_s':2e-6}}
    check('A0_clock_authority_mismatch_rejected',lambda:reject(lambda:native.validate_a0_inputs(cfg,{**b,'descriptor':changed})))
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    denied['actualEnv.__init__']=0
    with patch.object(ManagerBasedRlEnv,'__init__',deny('actualEnv.__init__')):
        actual_cfg=native.configure_a0_env(make_env_cfg,b,0)
    check('actual_factory_play_terminations_match_reviewed3_after_position_pop',lambda:set(actual_cfg.terminations)=={'time_out','unstable','reference_end'})
    check('actual_factory_no_autoreset_or_noise_and_four5ms',lambda:actual_cfg.auto_reset is False and actual_cfg.decimation==4 and actual_cfg.sim.mujoco.timestep==.005 and not any(g.enable_corruption for g in actual_cfg.observations.values()))
    check('actual_factory_newrecipe151_path_bound',lambda:Path(actual_cfg.commands['motion'].recipe_path).samefile(b['native_reference']['recipe_path']))
    def fake_cfg():
        return NS(terminations={k:object() for k in ('time_out','unstable','reference_end','object_position_error')},
          decimation=4,sim=NS(mujoco=NS(timestep=.005)),commands={'motion':NS()},actions={'joint_pos':NS()},
          observations={'policy':NS(enable_corruption=False)},events={'reset_to_reference_state':NS(),
          'pen_dimensions':NS(params={'diameter_range_m':(.02,.02),'total_length_range_m':(.2,.2)})})
    wrong=fake_cfg();wrong.terminations['object_orientation_error']=object()
    check('factory_term_ABI_extra_orientation_rejected',lambda:reject(lambda:native.configure_a0_env(lambda *a,**kw:wrong,b,0)))
    wrong2=fake_cfg();wrong2.sim.mujoco.timestep=.002
    check('factory_wrong_timestep_rejected',lambda:reject(lambda:native.configure_a0_env(lambda *a,**kw:wrong2,b,0)))
    # Existing capacity implementation, new wrapper's before-yield retention.
    class Flags:
        def __init__(self,value):self.value=np.asarray(value);self.reads=0
        def numpy(self):self.reads+=1;return self.value.copy()
    def cap_env(flag):
        env=NS(sim=NS(wp_data=NS(overflow=Flags(flag))))
        env.reset_calls=0
        def reset(**kw):env.reset_calls+=1
        env.reset=reset
        return env
    ce=cap_env(np.array([0],np.int32))
    with native.observed_capacity(ce) as observer:ce.reset()
    cap=observer.snapshot()
    check('new_capacity_wrapper3reads_constructor_prereset_final',lambda:cap['record_count']==3 and cap['before_reset_read_count']==1 and cap['final_read_count']==1 and ce.reset_calls==1 and cap['status']=='OK')
    ce_bad=cap_env(np.array([1],np.int32))
    try:
        with native.observed_capacity(ce_bad):raise AssertionError('must never yield')
    except RuntimeError:pass
    badobs=ce_bad._full151_capacity_last_observer
    check('before_yield_failure_retains_original_rawflags_finalized_host_handle',lambda:ce_bad.reset_calls==0 and badobs.finalized and badobs.snapshot()['cumulative_OR_bitmask']==1 and badobs.snapshot()['record_count']==2)
    check('capacity_wrapper_restores_reset_scope_no_extra_finalread',lambda:not hasattr(ce,'_full151_capacity_observer') and not hasattr(ce,'_rough_capacity_observer') and ce.sim.wp_data.overflow.reads==3)
    check('actual_descriptor_unchanged_after_negativefixtures',lambda:canonical(b['descriptor'])==descriptor_before)
    details['actual_bundle']=pin(bundle_path)
    import inspect
    details['actual_factory_source']=pin(inspect.getsourcefile(make_env_cfg))
    details['capacity_helper_source']=pin(native.CAPACITY_SOURCE)
    details['contact_helper_source']=pin(probe.CONTACT_SOURCE)
    details['actual_factory_scope']='Actual make_env_cfg with151 recipe; no actualEnv/reader/FK/physics; constructor denied.'
finally:
    for ctx in reversed(patches):ctx.stop()

native_ast=ast.parse((ROOT/'src/full151_native/native.py').read_text())
check('A0_source_has_no_Actor_Runner_or_optimizer_constructor',lambda:not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('Actor','WorldModel','DexSuperMimicCfg','DexSuperMimicRunner','RecordedRunner','Adam','AdamW') for n in ast.walk(native_ast)))
check('no_real_native_API_called',lambda:not any(denied.values()))
import torch
check('CUDA_mask_minus1_uninitialized_no_devices',lambda:not torch.cuda.is_initialized() and not torch.cuda.is_available() and torch.cuda.device_count()==0)
check('source_snapshots_current_unchanged',lambda:all(pin(v['path'])==v for v in sources.values()))
report={'schema':'full151_native_A0_CPU_contract:1','status':'PASS_CPU_HELPERS_NATIVE_ENV_AND_PHYSICS_STILL_PENDING' if all(checks.values()) else 'CPU_CONTRACT_FAILURE_PRESERVED',
 'passed':sum(checks.values()),'total':len(checks),'checks':checks,'details':details,'source_pins':sources,'denied_API_calls':denied,
 'clock_expected':ca,'source_import_before_native_dependencies':initial_imports,
 'actualfactory_loaded_GPU_framework_modules':{k:k in sys.modules for k in ('torch','warp','mujoco')},
 'scope':{'real_factory_config_only':True,'synthetic_grade_and_capacity_fixtures':True,'actual_Env_constructed':False,'native_reader_reexecuted':False,'MJB_static_model_read_by_bundle':True,'FK_collision_physics_GPU_WM_training':False,'no_dynamic_probe_execution':True}}
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'report':pin(OUT/'report.json'),'passed':report['passed'],'total':report['total'],'status':report['status']}))
if not all(checks.values()):raise SystemExit(2)
