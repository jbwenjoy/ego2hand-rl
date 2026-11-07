"""One frozen, bounded full151 contact-geometry-coupled CPU experiment."""
from pathlib import Path
import argparse,datetime,importlib.util,json,os,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151.io import checked,pin,read,write,all_pins,load_selected


def load_module(p,name):
    spec=importlib.util.spec_from_file_location(name,checked(p));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def save_grid(out,grid,summary):
    out.mkdir(exist_ok=False)
    np.savez_compressed(out/'native_grid.npz',**{k:v for k,v in grid.items() if isinstance(v,np.ndarray)})
    write(out/'witness_ids.json',grid['witness_ids']);write(out/'summary.json',summary)


def sparse_fields(A,prefix):
    A=A.tocsr();return {prefix+'_data':A.data,prefix+'_indices':A.indices,prefix+'_indptr':A.indptr,prefix+'_shape':np.asarray(A.shape)}


def main(protocol_path):
    begin=time.perf_counter();p=read(protocol_path)
    expected={'max_major':8,'initial_trust_rad':.15,'reject_trust_factor':.5,'accept_ratio_min':.1,'row_scale':1e4,'eta_penalty_rad_per_m':1e4,'geometry_planning_depth_m':.002999,'geometry_gate_m':.003,'eta_final_gate_m':1e-9,'raw_tolerance':1e-12,'per_LP_limit_s':10.,'soft_stop_before_next_LP_s':220.,'no_linesearch':True,'no_best_history':True}
    if p['schema']!='full151_contact_coupled_CPU_execution:1' or p['execution']!=expected:raise ValueError('Wrong frozen finite execution contract')
    for s in all_pins(p):checked(s)
    if checked(p['sources']['producer'])!=Path(__file__).resolve():raise ValueError('Wrong entry source')
    probe=read(checked(p['inputs']['local_probe_receipt']));ready=read(checked(p['inputs']['local_probe_READY']))
    if probe['status']!='PASS_LOCAL_WORST_BRANCHES_TIES_DIAGNOSTIC_ONLY' or not all(probe['checks'].values()) or ready['global_or_physical_qualified'] is not False:raise ValueError('Required limited local probe missing; do not replace by global qualification')
    out=Path(p['output']);out.mkdir(parents=True,exist_ok=False)
    report={'schema':'full151_contact_coupled_CPU_receipt:1','status':'STARTED','protocol':pin(protocol_path),'started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'checks':{},'majors':[],'cost':{'LP_attempted':0,'LP_returned':0,'solver_wall_s':0.,'grid_audits':0},
      'physical_qualified':False,'contact_intent_target_ready':False,'training_authorized':False,'limitations':p['limitations']}
    write(out/'execution_started.json',report);oracle=None
    try:
        from scipy import sparse
        from scipy.optimize import linprog
        from scipy.spatial.transform import Rotation
        from full151 import continuous_control as c
        from full151 import sparse_global_scp as s
        from full151.contact_row_oracle_v2 import geometry_rows
        from full151.native_witness import FrozenNativeOracle,geometry_summary
        old=load_module(p['sources']['frozen_control'],'_full151_coupled_frozen760')
        inputs=p['inputs'];checks=report['checks']
        ref=load_selected(inputs['parent_reference'],p['fields']['parent_reference'])
        init=load_selected(inputs['initialization'],p['fields']['initialization'])
        control=load_selected(inputs['parent_control'],['tau_ff_static0_FP32_Nm'])
        geo=load_selected(inputs['derived_geometry'],['time_s','joint_pos_rad','joint_lower_rad','joint_upper_rad'])
        params,pdmeta=old.load_actual_pd(checked(inputs['model']),checked(inputs['actual_context']),model_sha256=inputs['model']['sha256'],context_sha256=inputs['actual_context']['sha256'])
        P=ref['control_points_rad'].copy();P0=P[0].copy();P1=P[1].copy();tau=control['tau_ff_static0_FP32_Nm'];u0=init['u_planned0_FP32_rad']
        qlo,qhi=geo['joint_lower_rad'],geo['joint_upper_rad'];target=geo['joint_pos_rad'];t601=np.arange(601)*.005
        checks['source151_clock']=exact(geo['time_s'],c.BREAKS) and exact(ref['time_s'],c.BREAKS)
        checks['fixed_initial_q0']=exact(P0.astype(np.float32),init['q_init_requested_FP32_rad'])
        checks['fixed_initial_qd0']=exact(c.eval_controls(P,[0.],1)[0].astype(np.float32),init['qd_init_requested_FP32_rad_s'])
        checks['source601_q_curve']=np.max(np.abs(c.eval_controls(P,t601)-ref['planned_q601_rad']))<=1e-12
        checks['source_u0']=exact(old.c2_numpy(old.proposal_numpy(P0,c.eval_controls(P,[0.],1)[0],tau,params),params),u0)
        xyzw=Rotation.from_matrix(ref['object_rotation_w']).as_quat();poses=np.c_[ref['object_COM_world_m'],xyzw[:,[3,0,1,2]]];poses[::4]=ref['mapped_object_pose_w']
        checks['fixed_object_curve_rotation']=np.max(np.abs(Rotation.from_quat(poses[:,[4,5,6,3]]).as_matrix()-ref['object_rotation_w']))<=1e-12
        if not all(checks.values()):raise ValueError('Input preflight failed before optimizer')
        (out/'source_before').mkdir()
        for name,v in p['sources'].items():(out/'source_before'/(name+Path(v['path']).suffix)).write_bytes(checked(v).read_bytes())
        joints=[c.build_joint_lp(j,target,P0,P1,u0,tau,params,qlo,qhi) for j in range(20)]
        def fit_cost(curve):
            q=c.eval_controls(curve,c.BREAKS)
            return float(np.mean(np.abs(q-target),axis=0).sum()+4*np.mean(np.abs(np.diff(curve,n=2,axis=0)),axis=0).sum())
        def audit_grid(curve,folder):
            with oracle:grid=oracle.grid(c.eval_controls(curve,t601),poses,with_jacobian=True)
            summary=geometry_summary(grid,oracle.query.pair_kind);save_grid(folder,grid,summary);report['cost']['grid_audits']+=1
            return grid,summary
        oracle=FrozenNativeOracle(inputs['model'],inputs['actual_context'],p['sources']['native_geometry'],p['sources']['kinematics'])
        grid,gsummary=audit_grid(P,out/'base_geometry')
        checks['actual_model_qbounds']=exact(oracle.query.robot.lower,qlo) and exact(oracle.query.robot.upper,qhi)
        if not checks['actual_model_qbounds']:raise ValueError('Model hard limits drifted')
        write(out/'query_metadata.json',oracle.query.metadata)
        np.savez_compressed(out/'catalog.npz',pairs=oracle.query.pairs,pair_kind=oracle.query.pair_kind,geom_names=np.asarray(oracle.query.names))
        report['base_summary']=gsummary;report['base_fit_cost']=fit_cost(P)
        current_phi=fit_cost(P)+1e4*gsummary['planning_max_violation_m'];radius=.15;accepted_major=None;accepted_eta=None;last_linear=None
        stop='MAJOR_LIMIT';Bfree=c.basis().evaluate(t601)[:,2:]
        for major in range(1,9):
            if time.perf_counter()-begin>220.:stop='SOFT_DEADLINE_BEFORE_NEXT_LP';break
            path=out/f'major{major:02d}';path.mkdir(exist_ok=False)
            rows=geometry_rows(Bfree,grid['sample_index'],grid['contactdist_m'],grid['distance_jacobian_q_m_per_rad'],witness_ids=grid['witness_ids'],source_fields={'grid':pin((out/'base_geometry' if accepted_major is None else out/f'major{accepted_major:02d}'/'geometry')/'native_grid.npz')})
            built=s.build_scp_lp(joints,P,rows,trust_radius_rad=radius);kw,contract=built['linprog_kwargs'],built['contract']
            lo=np.array([-np.inf if v[0] is None else v[0] for v in kw['bounds']]);hi=np.array([np.inf if v[1] is None else v[1] for v in kw['bounds']])
            np.savez_compressed(path/'subproblem.npz',**sparse_fields(contract['A_raw'],'A_raw'),b_raw=contract['b_raw'],c=kw['c'],lower=lo,upper=hi,
              **sparse_fields(contract['A_original_absolute'],'A_original'),b_original=contract['b_original_absolute'],absolute_shift=contract['absolute_shift'],center_vector=contract['center_vector'],center_P=P)
            write(path/'contract.json',{k:v for k,v in contract.items() if not isinstance(v,np.ndarray) and not sparse.issparse(v)})
            row={'major':major,'radius_rad':radius,'center_accepted_major':accepted_major,'old_actual_phi':current_phi,'witness_rows':len(rows['rhs_m'])}
            if oracle.original:raise ValueError('LP must execute outside the geometry guard context')
            write(out/'live_state.json',{'stage':'LP_RUNNING','major':major,'cost':report['cost']})
            report['cost']['LP_attempted']+=1
            tick=time.perf_counter();result=linprog(**kw);elapsed=time.perf_counter()-tick
            report['cost']['LP_returned']+=1;report['cost']['solver_wall_s']+=elapsed
            row.update(solver_success=bool(result.success),solver_status=int(result.status),solver_message=str(result.message),solver_iterations=int(result.nit),solver_wall_s=elapsed)
            z=np.empty(0) if result.x is None else np.asarray(result.x)
            np.savez_compressed(path/'solver_result.npz',x=z,fun=np.asarray(np.nan if result.fun is None else result.fun))
            if not result.success or not z.size:
                stop='LP_TERMINAL_FAILURE_NO_RETRY';row['accepted']=False;report['majors'].append(row);write(path/'receipt.json',row);break
            linear=s.audit_subproblem_solution(z,contract,kw['bounds'])
            np.savez_compressed(path/'linear_audit_arrays.npz',**{k:v for k,v in linear.items() if isinstance(v,np.ndarray)})
            row['linear_audit']={k:v for k,v in linear.items() if not isinstance(v,np.ndarray)}
            trial,eta=s.unpack_trial(z,contract)
            np.savez_compressed(path/'trial_control_points.npz',P=trial,eta_m=np.asarray(eta))
            if not linear['raw_primal_pass']:
                stop='RAW_LINEAR_CERTIFICATE_FAIL_NO_REPAIR';row['accepted']=False;report['majors'].append(row);write(path/'receipt.json',row);break
            post=c.audit_post_C2(trial,params,tau,u0,qlo,qhi,old)
            row['postC2_gates']=post['gates']
            if not all(post['gates'].values()):
                stop='FP32_REFERENCE_CONTROL_GATE_FAIL';row['accepted']=False;report['majors'].append(row);write(path/'receipt.json',row);break
            trial_grid,trial_summary=audit_grid(trial,path/'geometry')
            delta=(trial[2:]-P[2:]).T.reshape(3020)
            predicted_violation=float(max(0.,float(np.max(rows['A_step']@delta-rows['rhs_m'])))) if len(rows['rhs_m']) else 0.
            direct_fit=fit_cost(trial);predicted_phi=direct_fit+1e4*predicted_violation
            new_phi=direct_fit+1e4*trial_summary['planning_max_violation_m']
            predicted_gain=current_phi-predicted_phi;actual_gain=current_phi-new_phi
            ratio=None if predicted_gain<=0 else actual_gain/predicted_gain
            merit_finite=bool(np.isfinite([current_phi,predicted_phi,new_phi,predicted_gain,actual_gain]).all() and (ratio is None or np.isfinite(ratio)))
            np.savez_compressed(path/'merit_raw.npz',old_phi=current_phi,predicted_phi=predicted_phi,new_phi=new_phi,predicted_gain=predicted_gain,actual_gain=actual_gain,ratio=np.nan if ratio is None else ratio)
            accept=bool(merit_finite and predicted_gain>0 and actual_gain>0 and ratio is not None and ratio>=.1)
            def finite_number(value):return float(value) if value is not None and np.isfinite(value) else None
            row.update(eta_m=eta,eta_final_gate_pass=linear['eta_gate_pass'],geometry_summary=trial_summary,direct_fit_cost=direct_fit,
              predicted_phi=finite_number(predicted_phi),new_actual_phi=finite_number(new_phi),predicted_gain=finite_number(predicted_gain),actual_gain=finite_number(actual_gain),acceptance_ratio=finite_number(ratio),merit_finite=merit_finite,accepted=accept)
            if not merit_finite:
                stop='NONFINITE_MERIT_STOP';report['majors'].append(row);write(path/'receipt.json',row);break
            if accept:
                P=trial;grid=trial_grid;gsummary=trial_summary;current_phi=new_phi;accepted_major=major;accepted_eta=eta;last_linear=linear
            else:radius*=.5
            row['next_radius_rad']=radius
            report['majors'].append(row);write(path/'receipt.json',row)
            print(json.dumps({'major':major,'accepted':accept,'eta_m':eta,'max_depth_m':trial_summary['maximum_generated_depth_m'],'actual_phi':new_phi,'ratio':ratio,'next_radius_rad':radius}),flush=True)
            if accept and linear['eta_gate_pass'] and trial_summary['sampled_geometry_gate']:
                stop='ACCEPTED_CANDIDATE_REQUIRES_FINAL_AUDIT';break
        # This is the optimizer's last accepted state, never a best-history pick.
        final_grid,final_summary=audit_grid(P,out/'final_geometry')
        post=c.audit_post_C2(P,params,tau,u0,qlo,qhi,old)
        checks['execution_without_fatal_numerical_stop']=stop not in {'LP_TERMINAL_FAILURE_NO_RETRY','RAW_LINEAR_CERTIFICATE_FAIL_NO_REPAIR','FP32_REFERENCE_CONTROL_GATE_FAIL','NONFINITE_MERIT_STOP'}
        checks['at_least_one_accepted_certified_step']=accepted_major is not None
        checks['last_accepted_linear_certificate']=last_linear is not None and last_linear['raw_primal_pass']
        checks['last_accepted_eta_final_gate']=accepted_eta is not None and 0<=accepted_eta<=1e-9
        checks['final_fresh_sampled_geometry']=final_summary['sampled_geometry_gate']
        checks.update({'final_'+k:v for k,v in post['gates'].items()})
        checks['final_P0_P1_exact']=exact(P[:2],ref['control_points_rad'][:2])
        checks['final_initial_qd0_exact']=exact(post['analytic_qd151_rad_s'][0].astype(np.float32),init['qd_init_requested_FP32_rad_s'])
        import torch
        tp=old.torch_parameters(params,device='cpu')
        pre=old.proposal_torch(torch.tensor(post['q_goal_rad'][:150].copy()),torch.tensor(post['analytic_qd151_rad_s'][:150].copy()),torch.tensor(tau.copy()),tp)
        original=load_module(p['sources']['original_C2'],'_coupled_original_C2')
        oracle_clamp=original.target_inset(pre,torch.tensor(params.target_lower.copy()),torch.tensor(params.target_upper.copy()),.020)
        checks['final_CPU_Torch_proposal_bitexact']=exact(pre.numpy(),post['u_pre150_FP32_rad'])
        checks['final_original_C2_bitexact']=exact(oracle_clamp.numpy(),post['u_postC2_150_FP32_rad'])
        checks['query_bound_and_one_constructor']=report['cost']['grid_audits']<=10 and oracle.allowed['mj_collision']==601*report['cost']['grid_audits'] and oracle.allowed['mj_kinematics']==1+oracle.allowed['mj_collision'] and oracle.allowed['mj_comPos']==oracle.allowed['mj_kinematics']
        checks['no_dynamic_force_or_optimization_in_geometry_guard']=not any(oracle.forbidden.values())
        checks['LP_budget_and_terminal_count']=report['cost']['LP_attempted']<=8 and report['cost']['LP_attempted']==report['cost']['LP_returned']
        checks['all_source_input_pins_unchanged']=all(checked(v) for v in all_pins(p))
        report.update(algorithm_stop=stop,last_accepted_major=accepted_major,last_accepted_eta_m=accepted_eta,final_geometry=final_summary,final_fit_cost=fit_cost(P),final_actual_phi=fit_cost(P)+1e4*final_summary['planning_max_violation_m'],PD_metadata=pdmeta)
        reference_keys=set(c.sample_controls(P))
        refout={k:post[k] for k in reference_keys}
        refout.update(control_points_rad=P,Bspline_knots_s=c.basis().knots,mapped_object_pose_w=ref['mapped_object_pose_w'])
        refout.update({k:v for k,v in ref.items() if k.startswith('object_')})
        np.savez_compressed(out/'reference_candidate151.npz',**refout);np.savez_compressed(out/'init_requested.npz',**init)
        np.savez_compressed(out/'control_candidate150.npz',**{k:v for k,v in post.items() if isinstance(v,np.ndarray)},tau_ff_static0_FP32_Nm=tau)
        report['status']='CPU_COUPLED_SAMPLE_GATES_PASS_PHYSICS_PENDING' if all(checks.values()) else 'CPU_COUPLED_CANDIDATE_FAIL_RETAINED'
    except Exception as e:
        report['status']='SCP_EXCEPTION_PRESERVED';report['exception']=repr(e);report['traceback']=traceback.format_exc()
    report['actual_native_calls']=None if oracle is None else {'allowed':oracle.allowed,'forbidden':oracle.forbidden}
    report['wall_s']=time.perf_counter()-begin;report['artifacts']=[pin(f) for f in sorted(out.rglob('*')) if f.is_file()]
    write(out/'receipt.json',report);write(out/'live_state.json',{'stage':'TERMINAL','status':report['status'],'cost':report['cost']})
    if report['status']=='CPU_COUPLED_SAMPLE_GATES_PASS_PHYSICS_PENDING':write(out/'READY_CPU_SAMPLED_ONLY.json',{'receipt':pin(out/'receipt.json'),'continuous_geometry_qualified':False,'physics_qualified':False,'contact_intent_target_ready':False,'training_authorized':False})
    print(json.dumps({'status':report['status'],'cost':report['cost'],'algorithm_stop':report.get('algorithm_stop'),'last_accepted_major':report.get('last_accepted_major'),'final_geometry':report.get('final_geometry'),'receipt':pin(out/'receipt.json')}),flush=True)
    return 0 if report['status']=='CPU_COUPLED_SAMPLE_GATES_PASS_PHYSICS_PENDING' else 1


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--protocol',required=True);args=a.parse_args();raise SystemExit(main(Path(args.protocol)))
