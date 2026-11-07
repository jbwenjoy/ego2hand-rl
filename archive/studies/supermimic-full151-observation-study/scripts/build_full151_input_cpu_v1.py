"""One declared CPU-only full151 fit. Preserve failures; never run physics."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,platform,sys,time,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from full151.io import checked,pin,read,write,all_pins,load_selected


def module(spec,name):
    path=checked(spec);s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
    return m


def exact(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def main(protocol_path):
    start=time.perf_counter();protocol=read(protocol_path)
    if protocol['schema']!='full151_input_cpu_protocol:1':raise ValueError('Unsupported protocol')
    if protocol['execution']!={'joint_LP_calls':20,'per_joint_time_limit_s':5.0,'retries':0,'lambda_coefficient':4.0,'GPU':False,'physics':False,'training':False}:raise ValueError('Execution contract changed')
    for p in all_pins(protocol):checked(p)
    if checked(protocol['sources']['producer'])!=Path(__file__).resolve():raise ValueError('Wrong producer identity')
    out=Path(protocol['output']);out.mkdir(parents=True,exist_ok=False)
    report={'schema':'full151_input_cpu_receipt:1','status':'STARTED','protocol':pin(protocol_path),
      'input_pins_checked_before':len(list(all_pins(protocol))),'started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'cost':{'LP_calls':0,'physics_steps':0,'geometry_queries':0,'WM_forwards':0,'training_updates':0,'GPU_used':False},
      'limitations':protocol['limitations'],'checks':{},'joint_results':[]}
    write(out/'execution_started.json',report)
    try:
        import scipy
        from scipy.optimize import linprog
        import mujoco
        report['runtime']={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'mujoco':mujoco.__version__}
        # All sources were pinned before import. Only generic frozen object math is imported by the object wrapper.
        from full151 import continuous_control as c
        from full151.object_curves import ObjectCurves151,primitive_math
        old=module(protocol['sources']['frozen_control'],'_full151_control760')
        inputs=protocol['inputs'];checks=report['checks']
        obs=load_selected(inputs['observations'],protocol['selected_fields']['observations'])
        geo=load_selected(inputs['derived_geometry'],protocol['selected_fields']['derived_geometry'])
        raw=load_selected(inputs['designed_FD'],protocol['selected_fields']['designed_FD'])
        anchor=load_selected(inputs['anchor_reference'],protocol['selected_fields']['anchor_reference'])
        init=load_selected(inputs['anchor_init'],protocol['selected_fields']['anchor_init'])
        prior=load_selected(inputs['anchor_control'],protocol['selected_fields']['anchor_control'])
        manifest=read(checked(inputs['observations_manifest']));cue=read(checked(inputs['observation_cues_READY']))
        checks['observation_identity_allowlist']=manifest['original_source_frames']==[200,351] and manifest['observation_sha256']==inputs['observations']['sha256'] and not manifest['is_real_ego'] and not manifest['contains_robot_joint_labels']
        checks['observation_unknown_confidence']=not any(np.any(obs[k]) for k in ('landmark_confidence_known','object_confidence_known','wrist_confidence_known'))
        checks['intent_cues_not_objective']=cue['status']=='READY_OBSERVATION_CUES_ONLY' and cue['target_ready'] is False and cue['objective_loading_permitted'] is False
        t=c.BREAKS;checks['all151_exact_clock']=all(exact(x,t) for x in (obs['time_s'],geo['time_s'],raw['time_s']))
        checks['object_mapping_exact']=exact(geo['mapped_object_pose_w'],raw['object_pose_mapped_w']) and exact(geo['mapped_object_pose_w'][:16],anchor['mapped_object_pose_w'])
        P0=anchor['control_points_rad'][0].copy();P1=anchor['control_points_rad'][1].copy()
        checks['anchor_q0_preserved']=exact(P0,anchor['q_goal_rad'][0]) and exact(P0.astype(np.float32),init['q_init_requested_FP32_rad'])
        checks['anchor_qd0_preserved']=exact((150*(P1-P0)).astype(np.float32),init['qd_init_requested_FP32_rad_s'])
        tau=prior['control__fp32_old_export__tau_ff_static0_FP32_Nm'];u0=init['u_planned0_FP32_rad']
        checks['first_target_saved_exact']=exact(u0,prior['control__fp32_old_export__tracking_u_exec_15_FP32_rad'][0])
        params,pdmeta=old.load_actual_pd(checked(inputs['model']),checked(inputs['actual_context']),model_sha256=inputs['model']['sha256'],context_sha256=inputs['actual_context']['sha256'])
        model=mujoco.MjModel.from_binary_path(str(checked(inputs['model'])))
        jids=np.asarray(pdmeta['joint_ids']);body=model.body('object/cube').id
        qlo=model.jnt_range[jids,0].copy();qhi=model.jnt_range[jids,1].copy()
        checks['hard_joint_bounds_from_actual_model']=exact(qlo,geo['joint_lower_rad']) and exact(qhi,geo['joint_upper_rad'])
        checks['zero_COM_offset']=not np.any(model.body_ipos[body])
        pdu=read(checked(inputs['static_preload']))
        checks['static_prior_saved_exact']=exact(old.static_prior_numpy(P0,np.asarray(pdu['u0_rad']),params),tau)
        checks['initial_proposal_saved_exact']=exact(old.c2_numpy(old.proposal_numpy(P0,anchor['analytic_qd16_rad_s'][0],tau,params),params),u0)
        # No quaternion roundtrip is allowed to replace the original initialization.
        om=primitive_math();pose=geo['mapped_object_pose_w'];r32=om.quaternion_matrices(init['object_pose_init_requested_FP32_w'][3:].astype(float))
        checks['object_initial_pose_exact']=exact(pose[0].astype(np.float32),init['object_pose_init_requested_FP32_w'])
        checks['object_initial_velocity_exact']=exact(raw['raw_v_world_m_s'][0].astype(np.float32),init['object_bodyorigin_v_requested_FP32_world_m_s'])
        checks['object_initial_world_omega_exact']=exact(raw['raw_omega_world_rad_s'][0].astype(np.float32),init['object_angular_requested_FP32_world_rad_s'])
        checks['object_initial_body_omega_exact']=exact((r32.T@raw['raw_omega_world_rad_s'][0]).astype(np.float32),init['object_angular_requested_FP32_body_rad_s'])
        if not all(checks.values()):raise ValueError('Pre-solve input gate failure; no LP permitted')
        obj=ObjectCurves151(t,pose,raw['raw_v_world_m_s'],raw['raw_a_world_m_s2'],raw['raw_omega_world_rad_s'],raw['raw_alpha_world_rad_s2'])
        knots=obj.sample(t);dense=obj.sample(np.arange(601)*.005)
        expected={'object_COM_world_m':pose[:,:3],'object_rotation_w':om.quaternion_matrices(pose[:,3:]),
          'object_v_world_m_s':raw['raw_v_world_m_s'],'object_a_world_m_s2':raw['raw_a_world_m_s2'],
          'object_omega_world_rad_s':raw['raw_omega_world_rad_s'],'object_alpha_world_rad_s2':raw['raw_alpha_world_rad_s2']}
        tol=protocol['object_numeric_tolerances'];objerr={k:float(np.max(np.abs(knots[k]-v))) for k,v in expected.items()}
        checks['object_knots_analytic_endpoints']=all(objerr[k]<=tol[k] for k in objerr)
        prefixerr={k:float(np.max(np.abs(dense[k][:61]-anchor[k]))) for k in expected}
        checks['object_first16_unchanged_law']=all(prefixerr[k]<=tol[k] for k in prefixerr)
        report['object_curve_errors']={'all151_knots':objerr,'first61_samples_vs_old_prefix':prefixerr,
          'initial_R32_vs_source_R64_max_abs':float(np.max(np.abs(r32-knots['object_rotation_w'][0])))}
        if not all(checks.values()):raise ValueError('Object reference gate failure; no LP permitted')
        snap=out/'source_before';snap.mkdir()
        for name,p in protocol['sources'].items():(snap/(name+Path(p['path']).suffix)).write_bytes(checked(p).read_bytes())
        target=geo['joint_pos_rad'];free=np.full((151,20),np.nan);total_solver=0.
        (out/'joint_LPs').mkdir()
        for j in range(20):
            built=c.build_joint_lp(j,target,P0,P1,u0,tau,params,qlo,qhi,lambda_coeff=4.)
            kw=built['linprog_kwargs'];A=kw['A_ub'];contract=built['contract'];base=out/'joint_LPs'/f'joint{j:02d}'
            np.savez_compressed(str(base)+'_input.npz',c=kw['c'],A_data=A.data,A_indices=A.indices,A_indptr=A.indptr,A_shape=np.asarray(A.shape),b=kw['b_ub'],bound_lower=np.array([v[0] for v in kw['bounds']]),bound_upper=np.array([np.inf if v[1] is None else v[1] for v in kw['bounds']]),**{k:v for k,v in contract.items() if isinstance(v,np.ndarray)})
            write(str(base)+'_contract.json',{k:v for k,v in contract.items() if not isinstance(v,np.ndarray)})
            begin=time.perf_counter();result=linprog(**kw);elapsed=time.perf_counter()-begin
            total_solver+=elapsed;report['cost']['LP_calls']+=1
            row={'joint':j,'success':bool(result.success),'status':int(result.status),'message':str(result.message),'iterations':int(result.nit),'wall_s':elapsed,'objective':None if result.fun is None else float(result.fun)}
            x=np.empty(0) if result.x is None else np.asarray(result.x)
            residual=np.empty(0) if not x.size else A@x-kw['b_ub']
            np.savez_compressed(str(base)+'_result.npz',x=x,raw_Ax_minus_b=residual)
            if x.size:
                lower=np.array([v[0] for v in kw['bounds']]);upper=np.array([np.inf if v[1] is None else v[1] for v in kw['bounds']])
                row['maximum_raw_Ax_minus_b']=float(residual.max());row['max_bound_excess']=float(np.maximum(np.maximum(lower-x,x-upper),0).max())
                row['arithmetic_1e12_gate']=bool(residual.max()<=1e-12 and row['max_bound_excess']<=1e-12)
                row['row_group_max_raw_Ax_minus_b']={g['name']:float(residual[g['start']:g['stop']].max()) for g in contract['row_groups']}
                if result.success and np.isfinite(x).all():free[:,j]=x[:151]
            else:row['arithmetic_1e12_gate']=False
            report['joint_results'].append(row);write(str(base)+'_receipt.json',row)
            print(json.dumps({'joint':j,'success':row['success'],'status':row['status'],'raw_gate':row['arithmetic_1e12_gate'],'wall_s':elapsed}),flush=True)
        report['cost']['solver_wall_s_sum']=total_solver
        checks['all20_solver_success']=all(r['success'] for r in report['joint_results'])
        checks['all20_raw_arithmetic_1e12']=all(r['arithmetic_1e12_gate'] for r in report['joint_results'])
        np.savez_compressed(out/'fit_solutions.npz',free_control_points_rad=free,P0=P0,P1=P1,fit_target151_rad=target)
        if np.isfinite(free).all():
            P=c.unpack_joint_results(free,P0,P1);export=c.audit_post_C2(P,params,tau,u0,qlo,qhi,old)
            checks.update(export['gates'])
            checks['analytic_init_qd_FP32_exact']=exact(export['analytic_qd151_rad_s'][0].astype(np.float32),init['qd_init_requested_FP32_rad_s'])
            checks['same_P0_P1_source64']=exact(P[:2],anchor['control_points_rad'][:2])
            reference={k:export[k] for k in ('time_s','q_goal_rad','analytic_qd151_rad_s','analytic_qdd151_rad_s2','native_forward_FD64_rad_s','native_forward_FD32_rad_s','planned_time601_s','planned_q601_rad','planned_qd601_rad_s','planned_qdd601_rad_s2')}
            reference.update(control_points_rad=P,Bspline_knots_s=c.basis().knots,mapped_object_pose_w=pose,object_p_coefficients=obj.p_coefficients,object_phi_coefficients=obj.phi_coefficients,**dense)
            np.savez_compressed(out/'reference_bspline151.npz',**reference)
            # Byte-identical values are copied only after independent boundary checks above.
            np.savez_compressed(out/'init_requested.npz',**init)
            control={k:v for k,v in export.items() if isinstance(v,np.ndarray) and k not in reference}
            control.update(tau_ff_static0_FP32_Nm=tau,force_bounds_Nm=np.stack((np.maximum(params.actuator_force_bounds[:,0],params.joint_force_bounds[:,0]),np.minimum(params.actuator_force_bounds[:,1],params.joint_force_bounds[:,1])),axis=-1))
            np.savez_compressed(out/'planned_control150.npz',**control)
            # Actual CPU proposal and original clamp path, without loading any model/WM/actor.
            import torch
            tp=old.torch_parameters(params,device='cpu');q=torch.tensor(export['q_goal_rad'][:150].copy(),device='cpu');qd=torch.tensor(export['analytic_qd151_rad_s'][:150].copy(),device='cpu')
            pre_t=old.proposal_torch(q,qd,torch.tensor(tau.copy(),device='cpu'),tp);exec_t=old.c2_torch(pre_t,tp)
            original=module(protocol['sources']['original_C2'],'_full151_original_C2')
            oracle=original.target_inset(pre_t,torch.tensor(params.target_lower.copy()),torch.tensor(params.target_upper.copy()),.020)
            checks['CPU_Torch_proposal150_bitexact']=exact(pre_t.numpy(),export['u_pre150_FP32_rad'])
            checks['CPU_Torch_C2_150_bitexact']=exact(exec_t.numpy(),export['u_postC2_150_FP32_rad'])
            checks['original_C2_150_bitexact']=exact(oracle.numpy(),export['u_postC2_150_FP32_rad'])
            delta=export['q_goal_rad']-target
            report['fit']={'mean_abs_q_error_rad':float(np.mean(np.abs(delta))),'max_abs_q_error_rad':float(np.max(np.abs(delta))),
              'max_abs_error_after_initial_rad':float(np.max(np.abs(delta[1:]))),'max_analytic_qd_rad_s':float(np.max(np.abs(export['analytic_qd151_rad_s']))),
              'mean_abs_coefficient_second_difference_rad':float(np.mean(np.abs(np.diff(P,n=2,axis=0)))),
              **{k:v for k,v in export.items() if not isinstance(v,np.ndarray) and k!='gates'}}
            report['PD_metadata']=pdmeta
            np.savez_compressed(out/'actual_PD_parameters.npz',**{k:getattr(params,k) for k in ('kp','bias','target_lower','target_upper','actuator_force_bounds','joint_force_bounds')},q_lower_rad=qlo,q_upper_rad=qhi)
        checks['all_input_source_pins_unchanged']=all(checked(p) for p in all_pins(protocol))
        report['status']='CPU_INPUT_GATES_PASS_PHYSICS_PENDING' if all(checks.values()) else 'CPU_CANDIDATE_GATES_FAIL_NO_QUALIFICATION'
    except Exception as e:
        report['status']='CPU_INPUT_EXCEPTION_PRESERVED';report['exception']=repr(e);report['traceback']=traceback.format_exc()
    report['wall_s']=time.perf_counter()-start
    report['artifacts']=[pin(p) for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'receipt.json',report)
    if report['status']=='CPU_INPUT_GATES_PASS_PHYSICS_PENDING':
        write(out/'READY.json',{'schema':'full151_CPU_input_ready:1','status':report['status'],'receipt':pin(out/'receipt.json'),
          'reference':pin(out/'reference_bspline151.npz'),'init':pin(out/'init_requested.npz'),'control':pin(out/'planned_control150.npz'),
          'physics_qualified':False,'geometry_qualified':False,'contact_target_ready':False,'training_authorized_by_this_receipt':False})
    print(json.dumps({'status':report['status'],'checks_passed':sum(bool(v) for v in report['checks'].values()),'checks_total':len(report['checks']),'LP_calls':report['cost']['LP_calls'],'wall_s':report['wall_s'],'receipt':pin(out/'receipt.json')}),flush=True)
    return 0 if report['status']=='CPU_INPUT_GATES_PASS_PHYSICS_PENDING' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    args=parser.parse_args();raise SystemExit(main(Path(args.protocol)))
