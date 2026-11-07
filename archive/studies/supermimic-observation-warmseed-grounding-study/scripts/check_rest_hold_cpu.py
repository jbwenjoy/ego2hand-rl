"""CPU contract/negative checks. Never imports Warp or runs any physics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,ast,copy,importlib.util,json,sys,tempfile
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import rest_hold_warp as worker
import run_rest_hold as supervisor

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'outputs/CPU_contract_v1');args=parser.parse_args()
    planpath=ROOT/'configs/warmseed-rest-qref-review-v1.json';plan=worker.read(planpath);out=args.output;out.mkdir(parents=True,exist_ok=False)
    forbidden={k:0 for k in ['mj_step','mj_forward','mj_step1','mj_step2','mj_inverse','mj_contactForce','mj_geomDistance','mj_collision']}
    def deny(name):
        def f(*a,**k):forbidden[name]+=1;raise AssertionError('CPUdefault called '+name)
        return f
    for name in forbidden:setattr(mujoco,name,deny(name))
    p,protocol,model,host,actual,idx,cpu=worker.load_inputs(planpath)
    checks={'CPU_frame0_FK_cast_projection_passed':cpu['status']=='passed_CPU_inputs_FK_FP32_C2_contract','CUDA_not_initialized_Warp_not_imported':not cpu['torch_CUDA_initialized'] and not cpu['Warp_imported'],
        'eight_physical_fields_exact':list(actual)==list(worker.FIELDS),'host_initial_q_not_projected':np.array_equal(host.qpos[idx['qa']].astype(np.float32),idx['q32']),
        'host_initial_ctrl_is_once_projected_constant':np.array_equal(host.ctrl[idx['aids']].astype(np.float32),idx['target']),
        'explicit_rest_zero_vel_external':not host.qvel.any() and not host.xfrc_applied.any() and not host.qfrc_applied.any(),
        'CPU_no_dynamics_force_calls':not any(forbidden.values())}
    with tempfile.TemporaryDirectory() as temp:
        for label,mutate in [('61steps',lambda d:d.update(steps=61)),('changed_gate',lambda d:d['gates'].update(object_position_max_m=.021)),('unbound_model_alias',lambda d:d['model'].update(sha256='0'*64)),('changed_actual_state_limits',lambda d:d['joint_upper_rad'].__setitem__(0,d['joint_upper_rad'][0]+.01))]:
            bad=copy.deepcopy(plan);mutate(bad);file=Path(temp)/(label+'.json');worker.write(file,bad)
            try:worker.load_inputs(file)
            except ValueError:checks[label+'_rejected']=True
            else:checks[label+'_rejected']=False
        attempt=Path(temp)/'must_not_start'
        try:worker.run(planpath,attempt)
        except ValueError:checks['unauthorized_GPU_run_rejected_before_output_or_Warp']=not attempt.exists() and 'warp' not in sys.modules
        else:checks['unauthorized_GPU_run_rejected_before_output_or_Warp']=False
        for label,mutate in [('unknown_control_condition',lambda d:d.update(control_condition='UNKNOWN')),
                             ('DRAFT_preload_no_pin',lambda d:d.update(control_condition='STATIC_PD_PRELOAD_C2'))]:
            bad=copy.deepcopy(plan);mutate(bad);file=Path(temp)/(label+'.json');worker.write(file,bad)
            try:worker.load_inputs(file)
            except ValueError:checks[label+'_rejected']=True
            else:checks[label+'_rejected']=False
        # Synthetic valid/negative preload objects test ONLY loader mechanics.
        # They are not the Windows designed u0 or evidence of static feasibility.
        proposal=(idx['lower'].astype(np.float32)+idx['upper'].astype(np.float32))/2
        with np.load(plan['candidate']['path'],allow_pickle=False) as z:
            q0=z['joint_pos_rad'][0].copy();pose0=z['mapped_object_pose_w'][0].copy()
        design={'schema':'seeded_static_PD_preload:1','status':'SYNTHETIC_HARNESS_NOT_REAL_DESIGN',
                'feasible':True,'canonical_candidate':plan['candidate'],'q0_rad':q0.tolist(),'q0_FP32_rad':q0.astype(np.float32).tolist(),
                'object_pose_w':pose0.tolist(),'object_pose_w_FP32':pose0.astype(np.float32).tolist(),'u0_rad':proposal.tolist(),'C2_margin_rad':.020,
                'target_lower_rad':idx['lower'].tolist(),'target_upper_rad':idx['upper'].tolist(),
                'inset_lower_rad':(idx['lower'].astype(np.float32)+np.float32(.020)).tolist(),
                'inset_upper_rad':(idx['upper'].astype(np.float32)-np.float32(.020)).tolist(),
                'no_dynamic_qualification':True,'source_pins':{key:plan['source_pins'][key] for key in ('candidate','model','context','pure_projection')}}
        def preload_case(label,change=None):
            d=copy.deepcopy(design)
            if change is not None:change(d)
            dfile=Path(temp)/(label+'_design.json');worker.write(dfile,d)
            fake=copy.deepcopy(plan);fake['control_condition']='STATIC_PD_PRELOAD_C2'
            fake['preload']={'path':str(dfile),'sha256':worker.sha(dfile)};fake['source_pins']['preload']=fake['preload']
            f=Path(temp)/(label+'_plan.json');worker.write(f,fake);return f
        valid=preload_case('synthetic_identity')
        *_,bidx,binfo=worker.load_inputs(valid)
        checks['synthetic_preload_C2_bitexact_identity_loader_only']=binfo['preload_C2_exact_identity'] is True and np.array_equal(bidx['target'],proposal)
        checks['synthetic_preload_same_q0_pose_mocap']=np.array_equal(idx['q32'],bidx['q32']) and np.array_equal(idx['pose32'],bidx['pose32'])
        for label,mutate in [('infeasible',lambda d:d.update(feasible=False,u0_rad=None)),
                             ('wrong_candidate',lambda d:d['canonical_candidate'].update(sha256='0'*64)),
                             ('wrong_q0',lambda d:d['q0_rad'].__setitem__(0,d['q0_rad'][0]+1e-4)),
                             ('wrong_FP32_q0',lambda d:d['q0_FP32_rad'].__setitem__(0,d['q0_FP32_rad'][0]+1e-4)),
                             ('u0_not_exact_FP32',lambda d:d['u0_rad'].__setitem__(0,d['u0_rad'][0]+1e-12)),
                             ('wrong_pose',lambda d:d['object_pose_w'].__setitem__(0,d['object_pose_w'][0]+1e-4)),
                             ('u_would_clip',lambda d:d['u0_rad'].__setitem__(0,float(idx['upper'][0]))),
                             ('unbound_design_sources',lambda d:d.update(source_pins={})),
                             ('no_dynamic_scope_false',lambda d:d.update(no_dynamic_qualification=False))]:
            f=preload_case(label,mutate)
            try:worker.load_inputs(f)
            except ValueError:checks['preload_'+label+'_rejected']=True
            else:checks['preload_'+label+'_rejected']=False
    # Measurement-vendor source functions are exact AST copies, not rederived
    # force/normal formulas. This verifies provenance without running fixtures.
    vendor=ast.parse((ROOT/'vendor/measured_contacts.py').read_text());vendorfn={n.name:n for n in vendor.body if isinstance(n,ast.FunctionDef)}
    for name,source in worker.read(ROOT/'vendor/provenance.json')['functions'].items():
        checks['verbatim_measurement_AST_'+name]=worker.hashlib.sha256(ast.dump(vendorfn[name],include_attributes=False).encode()).hexdigest()==source['function_ast_sha256']
    tree=ast.parse((ROOT/'scripts/rest_hold_warp.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    warpforward=[n for n in calls if isinstance(n.func.value,ast.Name) and n.func.value.id=='mjwarp' and n.func.attr=='forward']
    warpstep=[n for n in calls if isinstance(n.func.value,ast.Name) and n.func.value.id=='mjwarp' and n.func.attr=='step']
    overflow=[n for n in calls if n.func.attr=='numpy' and isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='overflow']
    checks['source_one_startup_forward_callsite']=len(warpforward)==1
    checks['source_one_step_callsite_in_fixed60loop']=len(warpstep)==1 and any(isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and [x.value for x in n.iter.args if isinstance(x,ast.Constant)]==[1,61] and warpstep[0] in list(ast.walk(n)) for n in ast.walk(tree))
    checks['source_one_final_sticky_overflow_read_callsite']=len(overflow)==1 and overflow[0].lineno>warpstep[0].lineno
    constcalls=[n for n in calls if isinstance(n.func.value,ast.Name) and n.func.value.id=='mjwarp' and n.func.attr=='set_const_0']
    checks['source_explicit_set_const_restore_True']=len(constcalls)==1 and any(k.arg=='restore' and isinstance(k.value,ast.Constant) and k.value.value is True for k in constcalls[0].keywords)
    checks['actual_state_MJB_limits_and_object_FK_bound']=cpu['actual_state_gate_limits_match_MJB'] is True and cpu['CPU_object_FK_position_error_m']<=1e-12 and cpu['integrator']=='IMPLICITFAST'
    forbiddenwrites=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AugAssign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                for a in ast.walk(target):
                    if isinstance(a,ast.Attribute) and isinstance(a.value,ast.Name) and a.value.id=='data' and a.attr in {'qpos','qvel','ctrl','xfrc_applied','qfrc_applied','mocap_pos','mocap_quat'}:forbiddenwrites.append(a.attr)
    checks['source_no_postinit_Warp_state_or_ctrl_writes']=not forbiddenwrites
    states=[{'joint_rad':idx['q32'].astype(float).tolist(),'object_pose_w':[0.,0.,1.,1.,0.,0.,0.],'external_force_max_abs':0.,'ctrl_exact':True,'raw_time_s':i*.005,'expected_time_s':i*.005} for i in range(61)]
    solved=[{'penetration_m':.0005,'object_environment_contacts':0} for i in range(60)]
    checks['synthetic_gate_reference_passes']=worker.gates(states,solved,{'status':'OK'},p,True)['rest_hold_pass'] is True
    for label,mutate in [('position_failure',lambda s,c:s[-1]['object_pose_w'].__setitem__(0,.021)),('actualq_failure',lambda s,c:s[-1]['joint_rad'].__setitem__(0,p['joint_upper_rad'][0]+.00101)),('penetration_failure',lambda s,c:c[-1].update(penetration_m=.00301)),('nonhand_support_failure',lambda s,c:c[-1].update(object_environment_contacts=1)),('exogenous_failure',lambda s,c:s[-1].update(external_force_max_abs=1e-9)),('actualu_failure',lambda s,c:s[-1].update(ctrl_exact=False)),('clock_failure',lambda s,c:s[-1].update(raw_time_s=.300003))]:
        s,c=copy.deepcopy(states),copy.deepcopy(solved);mutate(s,c);result=worker.gates(s,c,{'status':'OK'},p,True);checks[label+'_preserves_full_quality_failure']=result['rest_hold_pass'] is False and len(s)==61 and len(c)==60
    unknown=worker.gates(states,solved,{'status':'UNKNOWN'},p,True);overflowed=worker.gates(states,solved,{'status':'OVERFLOW'},p,True)
    checks['unknown_capacity_verdict_is_UNKNOWN']=unknown['rest_hold_pass'] is None and unknown['physics_inference_qualified'] is False
    checks['overflow_capacity_verdict_is_UNKNOWN']=overflowed['rest_hold_pass'] is None and overflowed['physics_inference_qualified'] is False
    checks['partial_is_pending_not_physical_fail']=worker.gates(states[:11],solved[:10],{'status':'UNKNOWN'},p,False)['rest_hold_pass'] is None
    checks['reviewed45s_lifecycle_source']=supervisor.sha(supervisor.LIFECYCLE)==supervisor.LIFECYCLE_SHA and plan['hard_process_seconds']==45 and plan['soft_worker_seconds']==40
    checks['reviewed159c_bounded_IO_used_in_worker_and_lifecycle']=worker.sha(ROOT/'vendor/runtime_io.py')=='159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1' and supervisor.lifecycle().write is worker.write_json
    old=Path('D:/workspace/supermimic-observation-grounding-study/scripts/rest_hold_warp.py')
    oldtree=ast.parse(old.read_text())
    def named(t,name):return next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('gates','verify_context','sample'):
        checks['old_worker_'+name+'_AST_identical']=ast.dump(named(oldtree,name),include_attributes=False)==ast.dump(named(tree,name),include_attributes=False)
    def warp_calls(t):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id in ('wp','mjwarp')]
    checks['old_worker_all_Warp_calls_AST_identical']=warp_calls(oldtree)==warp_calls(tree)
    oldstep=next(n for n in ast.walk(oldtree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and ast.unparse(n.iter)=='range(1, 61)')
    newstep=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and ast.unparse(n.iter)=='range(1, 61)')
    checks['old_worker_full_60step_loop_AST_identical']=ast.dump(oldstep,include_attributes=False)==ast.dump(newstep,include_attributes=False)
    checks['old_worker_frozen8a269_unchanged']=worker.sha(old)=='8a269b22bc171b8b214ae45ebf7b961fcff93bc3f53e8717dd66e2a2655565b4'
    checks['canonical_shape1_not_151_trajectory']=np.load(plan['candidate']['path'],allow_pickle=False)['joint_pos_rad'].shape==(1,20)
    checks['canonical_baseline_seed_only']=plan['canonical_seed_id']=='B_baseline_composite'
    checks['real_preload_plan_still_DRAFT']=worker.read(ROOT/'configs/warmseed-rest-preload-DRAFT-v1.json')['status']=='DRAFT_PENDING_REAL_STATIC_PRELOAD_INPUT'
    checks['default_CPU_preflight_subprocess']=supervisor.preflight(planpath,Path(sys.executable))['status']=='passed_CPU_preflight_no_GPU'
    checks['no_Warp_import_and_no_physics']=not ('warp' in sys.modules or 'mujoco_warp' in sys.modules) and not any(forbidden.values())
    report={'status':'passed_CPU_contracts_not_GPU_runtime_qualification' if all(checks.values()) else 'failed_CPU_contracts','checks':checks,'input_contract':cpu,'plan_sha256':worker.sha(planpath),'worker_sha256':worker.sha(ROOT/'scripts/rest_hold_warp.py'),'supervisor_sha256':worker.sha(ROOT/'scripts/run_rest_hold.py'),'checker_sha256':worker.sha(__file__),'forbidden_calls':forbidden,'GPU_started':False,'physics_steps':0,'known_limits':'Warp copier/readback/actualforce/capacity are source-reviewed only until the separately authorized singleGPU attempt; CPU preflight is not a physical result.'}
    worker.write(out/'report.json',report);print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'plan_sha256':report['plan_sha256']},indent=2));raise SystemExit(not all(checks.values()))
if __name__=='__main__':main()
