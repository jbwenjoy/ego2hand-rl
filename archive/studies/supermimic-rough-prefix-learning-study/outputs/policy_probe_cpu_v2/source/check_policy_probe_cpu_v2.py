"""Policy-probe math/phase contracts only; no Env, Warp or physics run."""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.util,json,os,sys,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);sources=out/'source';sources.mkdir()
    import numpy as np,torch
    torch.set_num_threads(2)
    from rough_prefix.policy_probe import expected_control,policy_rate,grade_policy,classify_capacity,_policy_obs,GATES
    from rough_prefix.probe import grade as old_grade
    from rough_prefix.controller import PURE_PATH,PURE_SHA256
    from dex_supermimic.rl.dex_supermimic.networks import Actor
    module_path=ROOT/'src/rough_prefix/policy_probe.py';pins={}
    for p in (Path(__file__),module_path,ROOT/'src/rough_prefix/probe.py',PURE_PATH):
        (sources/p.name).write_bytes(p.read_bytes());pins[p.name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    spec=importlib.util.spec_from_file_location('_policy_check_frozen_pure',PURE_PATH);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
    checks=[]
    def check(name,value):checks.append({'name':name,'passed':bool(value)})
    def reject(fn):
        try:fn()
        except (ValueError,KeyError):return True
        return False
    lower=np.full(20,-1,np.float32);upper=np.full(20,1,np.float32);baseline=np.full(20,.1,np.float32);action=np.linspace(-1,1,20,dtype=np.float32)[None];previous=np.linspace(-.2,.2,20,dtype=np.float32)[None];bias=np.zeros((1,20),np.float32)
    rng=torch.get_rng_state().clone();prediction=expected_control(action,previous,baseline,lower,upper,bias,pure.target_inset);prediction_rng_unchanged=torch.equal(rng,torch.get_rng_state())
    # Actual installed action method, used solely as a control-math fixture.
    fixture_file=Path('D:/workspace/supermimic-chord-study/scripts/check_pen_target_limits.py');fixture_spec=importlib.util.spec_from_file_location('_policy_native_action_fixture',fixture_file);fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)
    _,entity,term,_,_=fixture.fixture(Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb'),torch.zeros(2,20))
    term._processed_actions[:]=torch.from_numpy(previous).expand(2,-1);term.process_actions(torch.from_numpy(action).expand(2,-1))
    check('independent_EMA_matches_actual_native_process_method',np.array_equal(term.processed_action[0].numpy(),prediction['residual'][0]))
    check('actual_frozen_C2_source_bound',pins['target_inset.py']['sha256']==PURE_SHA256)
    check('prediction_C2_inside_inner_domain',np.all(prediction['actual']>=lower+np.float32(.02)) and np.all(prediction['actual']<=upper-np.float32(.02)))
    sat=expected_control(np.ones((1,20),np.float32),np.full((1,20),10,np.float32),np.zeros(20,np.float32),lower,upper,bias,pure.target_inset)
    check('saturation_hard_then_C2_not_residual_history_rewrite',np.all(sat['residual']>1) and np.array_equal(sat['hard_bounded'],np.ones((1,20),np.float32)) and np.array_equal(sat['actual'],np.full((1,20),.98,np.float32)))
    check('control_prediction_no_RNG',prediction_rng_unchanged)
    check('nonzero_bias_rejected',reject(lambda:expected_control(action,previous,baseline,lower,upper,np.ones((1,20),np.float32),pure.target_inset)))
    check('nonfinite_action_rejected',reject(lambda:expected_control(np.full((1,20),np.nan,np.float32),previous,baseline,lower,upper,bias,pure.target_inset)))
    events=[{'actualu':np.full((1,20),.1)} for _ in range(15)];events[0]['actualu'][:]=.4
    rate=policy_rate(np.zeros(20),events)
    check('startup_jump_included',rate['max_rate_rad_s']==20 and rate['startup_jump_max_rad']==.4 and rate['rate_difference_count']==15)
    check('partial_rate_not_qualified',reject(lambda:policy_rate(np.zeros(20),events[:-1])))
    states=[]
    for k in range(61):
        actual=np.zeros(20) if k==0 else events[(k-1)//4]['actualu'][0]
        states.append({'object_pose_w':np.array([0,0,0,1,0,0,0.]),'joint_rad':np.zeros(20),'raw_time_s':k*.005,'expected_time_s':k*.005,'ctrl_rad':actual,'external_force_max_abs':0.,'ctrl_exact':True})
    solver=[{'actuator_force_solver_native':np.zeros(20),'penetration_m':0.,'object_environment_contacts':0} for _ in range(60)]
    ref={'object_COM_world_m':np.zeros((61,3)),'object_rotation_w':np.tile(np.eye(3),(61,1,1))}
    args=(states,solver,{'status':'OK'},ref,lower,upper,-np.ones(20),np.ones(20),0.)
    old=old_grade(*args);new=grade_policy(*args,events)
    check('startup_case_old_grade_misses_peak_new_grade_detects',old['max_actual_target_rate_rad_s']==5 and new['max_actual_target_rate_rad_s']==20 and not new['gates']['actual_target_rate'])
    check('other_quality_criteria_unchanged',{k:v for k,v in old['gates'].items() if k!='actual_target_rate'}=={k:v for k,v in new['gates'].items() if k!='actual_target_rate'} and new['criteria']==GATES)
    check('policy_labels_do_not_claim_A0_or_training_success','native_A0_pass' not in new and new['native_policy_pass'] is False)
    partial=grade_policy(states[:-1],solver[:-1],{'status':'UNKNOWN'},ref,lower,upper,-np.ones(20),np.ones(20),0.,events[:-1])
    check('partial_quality_unknown_not_FAIL',partial['native_policy_pass'] is None and partial['status']=='INCOMPLETE_UNKNOWN')
    check('capacity_integer0_OK',classify_capacity(np.array([0],np.int32),1,2,192,4096)=='OK')
    check('capacity_nonzero_OVERFLOW',classify_capacity(np.array([8],np.int32),1,2,192,4096)=='OVERFLOW')
    check('capacity_bad_dtype_UNKNOWN',classify_capacity(np.array([0.]),1,2,192,4096)=='UNKNOWN')
    check('capacity_at_allocated_cap_UNKNOWN',classify_capacity(np.array([0],np.int32),192,2,192,4096)=='UNKNOWN')
    check('capacity_wrong_shape_UNKNOWN',classify_capacity(np.array([],np.int32),0,0,192,4096)=='UNKNOWN')
    with torch.random.fork_rng(devices=[]):actor=Actor(155).eval()
    before={k:v.clone() for k,v in actor.state_dict().items()};obs={'policy':torch.zeros(1,155)}
    with torch.no_grad():result=actor(_policy_obs(obs)).clamp(-1,1)
    check('native_Actor_eval_forward_finite_no_norm_update',tuple(result.shape)==(1,20) and bool(torch.isfinite(result).all()) and all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items()))
    check('wrong_policy_observation_shape_rejected',reject(lambda:_policy_obs({'policy':torch.zeros(1,154)})))
    check('nonfinite_policy_observation_rejected',reject(lambda:_policy_obs({'policy':torch.full((1,155),float('nan'))})))
    tree=ast.parse(module_path.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_policy_probe');calls=[n for n in ast.walk(run) if isinstance(n,ast.Call)]
    check('source_no_reset_observation_compute_norm_update_or_optimizer_step',not any(isinstance(n.func,ast.Attribute) and (n.func.attr in ('reset','compute','backward','zero_grad') or (n.func.attr=='update' and isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='norm')) for n in calls))
    check('source_one_env_step_call_in_fifteen_interval_loop',len([n for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='env' and n.func.attr=='step'])==1 and any(isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and len(n.iter.args)==1 and isinstance(n.iter.args[0],ast.Constant) and n.iter.args[0].value==15 for n in ast.walk(run)))
    check('CPU_no_CUDA_initialization_Env_or_physics',not torch.cuda.is_initialized())
    report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'source_pins':pins,'native_action_method_fixture':True,'actual_native_Actor_used':True,'actual_Env_or_Runner_constructed':False,'actual_policy_probe_executed':False,'physics_steps':0,'optimizer_steps':0,'CUDA_initialized':torch.cuda.is_initialized(),'Warp_imported_transitively':('warp' in sys.modules),'Warp_simulation_API_called_by_checker':False,'scope':'Math, labels, source cadence and actual native action/Actor CPU methods. Installed fixture may transitively import Warp; no Env, Warp simulation API, physics or GPU operation is executed. No native evaluation or physical quality qualification.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'checks':report['check_count'],'passed':report['passed'],'failed':[c for c in checks if not c['passed']]},indent=2));raise SystemExit(report['status']!='PASS')


if __name__=='__main__':main()
