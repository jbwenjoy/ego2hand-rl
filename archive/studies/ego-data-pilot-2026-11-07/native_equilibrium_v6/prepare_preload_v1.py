"""One bounded linear two-pad preload per saved v5 seed; no time stepping."""
from pathlib import Path
import hashlib
import json
import time
import traceback

ROOT=Path(__file__).resolve().parent
P=ROOT.parent


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def plain(v):
    import numpy as np
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(plain(v),f,indent=2,allow_nan=False)


def main():
    inputs={f'seed{i}':pin(P/f'retarget_gravity_v3/support_refinement_v5_run1/seed{i}_result.json') for i in range(2)}
    inputs['v5_report']=pin(P/'retarget_gravity_v3/support_refinement_v5_run1/report.json')
    sources={'runner':pin(__file__),'native_evaluator':pin(ROOT/'evaluator_native_v6.py'),
             'frozen_geometry':pin(P/'retarget_gravity_v3/evaluator_v3.py')}
    protocol={'schema':'two_seed_linear_contact_preload_initialization:6','inputs':inputs,'source_pins':sources,
        'CPU_static_authorized':True,'integration_training_GPU_authorized':False,
        'target_pad_gap_m':[-.000025,-.000025],'max_abs_delta_q_rad':.02,
        'linear_operator':'G_i=n_hand_exit_i^T J_hand_i[:,:20], from saved v5 current unique positive-gap pair',
        'residual_scaling_m':.001,'regularizer_delta_q_scale_rad':.01,'regularizer_weight':.001,
        'linear_solver':{'method':'trf','lsq_solver':'exact','tol':1e-10,'max_iter':100},
        'pad_window_hypothesis_m':[-.0001,.0002],'nonpad_min_m':0.,'self_min_m':0.,
        'seed0_ctrl_rule':'Preserve saved proposed_tau20, invert actual unchanged fixed-gain affine PD at qnew; no clamping. Unreachable -> BLOCK native forward.',
        'seed1_ctrl_rule':'ctrl=qnew, cold-start position command, not holding proposal',
        'phonepose_wrist_scale_materials_unchanged':True,'max_native_forwards':2,'max_full_22_189_geometry_calls':2,
        'nonlinear_optimization_or_LP_calls':0,'automatic_retry':False,
        'scope':'Engineering soft-contact initialization; linearization approximate; no old geometry PASS or equilibrium/holding qualification inherited.'}
    pp=ROOT/'preload_protocol_v1.json';write(pp,protocol)
    out=ROOT/'preload_v1';out.mkdir();start=time.perf_counter();e=None;results=[];phase='construct'
    try:
        import numpy as np
        from scipy.optimize import lsq_linear
        from evaluator_native_v6 import NativeEquilibriumEvaluator
        for rec in [*inputs.values(),*sources.values()]:assert pin(rec['path'])==rec
        e=NativeEquilibriumEvaluator.from_existing_pilot(digest_check_period=1)
        for seed in range(2):
            phase=f'seed{seed}_bounded_linear_preload'
            saved=json.loads(Path(inputs[f'seed{seed}']['path']).read_text(encoding='utf8'))
            s=saved['verification_state'];x=np.asarray(saved['selected_evaluation']['x26'],dtype=np.float64)
            oldq=x[:20].copy();contacts={v['geom_id']:v for v in s['contacts']}
            padrows={v['geom_id']:v for v in s['phone_geometry_rows'] if v['is_pad']}
            ns=[];js=[];gaps=[]
            for gid in e.pads:
                c=contacts[gid]
                if c['pairing']['status']!='UNIQUE_CLOSEST_PAIR' or not padrows[gid]['exact_gap_m']>0:
                    raise ValueError('Saved original positive-gap unique pair required')
                ns.append(-np.asarray(c['force_on_object_normal_world'],dtype=np.float64))
                js.append(np.asarray(c['actual_hand_point_J3x26'],dtype=np.float64)[:,:20])
                gaps.append(padrows[gid]['exact_gap_m'])
            ns,js,gaps=np.asarray(ns),np.asarray(js),np.asarray(gaps)
            G=np.einsum('bi,bij->bj',ns,js)
            lo=np.maximum(-.02,e.m.jnt_range[:20,0]-oldq)
            hi=np.minimum(.02,e.m.jnt_range[:20,1]-oldq)
            target=np.asarray(protocol['target_pad_gap_m'])
            A=np.vstack([G/.001,np.eye(20)*(.001/.01)])
            b=np.r_[(target-gaps)/.001,np.zeros(20)]
            fit=lsq_linear(A,b,bounds=(lo,hi),**protocol['linear_solver'])
            delta=np.asarray(fit.x);newq=oldq+delta
            phase=f'seed{seed}_full_geometry'
            geometry=e.final_geometry(newq,x[20:23],x[23:],pad_gap_window_m=(-.0001,.0002),
                                      nonpad_buffer_m=0.,self_buffer_m=0.)
            newstate=geometry['state']
            actualgaps=np.asarray([next(v['exact_gap_m'] for v in newstate['phone_geometry_rows'] if v['geom_id']==gid) for gid in e.pads])
            gain=e.m.actuator_gainprm[:,0].copy()
            offset=e.m.actuator_biasprm[:,0]+e.m.actuator_biasprm[:,1]*newq
            tau=np.asarray(saved['proposed_tau20_Nm']) if seed==0 else None
            ctrl=(tau-offset)/gain if seed==0 else newq.copy()
            mask=e.m.actuator_ctrllimited.astype(bool)
            reachable=bool(np.isfinite(ctrl).all() and np.all(ctrl[mask]>=e.m.actuator_ctrlrange[mask,0])
                and np.all(ctrl[mask]<=e.m.actuator_ctrlrange[mask,1]))
            if seed==0:
                amask=e.m.actuator_forcelimited.astype(bool);jmask=e.m.jnt_actfrclimited[:20].astype(bool)
                reachable=bool(reachable and np.all(tau[amask]>=e.m.actuator_forcerange[amask,0])
                    and np.all(tau[amask]<=e.m.actuator_forcerange[amask,1])
                    and np.all(tau[jmask]>=e.m.jnt_actfrcrange[:20][jmask,0])
                    and np.all(tau[jmask]<=e.m.jnt_actfrcrange[:20][jmask,1]))
            native=None
            if reachable:
                phase=f'seed{seed}_one_native_forward'
                native=e.evaluate(newq,x[20:23],x[23:],ctrl)
                np.savez_compressed(out/f'seed{seed}_native_raw.npz',**native['raw'])
                write(out/f'seed{seed}_native.json',native)
            write(out/f'seed{seed}_geometry.json',geometry)
            np.savez_compressed(out/f'seed{seed}_preload_state.npz',qpos27=newstate['qpos27'],qvel26=np.zeros(26),ctrl20=ctrl,
                x26=np.r_[newq,x[20:23],x[23:]],old_q20=oldq,delta_q20=delta,delta_lower20=lo,delta_upper20=hi,
                hand_exit_normals2x3=ns,hand_J2x3x20=js,G2x20=G,old_pad_gap2=gaps,target_pad_gap2=target,
                linear_predicted_pad_gap2=gaps+G@delta,actual_independent_pad_gap2=actualgaps,
                pad_gap_linearization_error2=actualgaps-(gaps+G@delta),PD_gain20=gain,PD_offset_at_new_q20=offset)
            nonpads=[v for v in newstate['phone_geometry_rows'] if not v['is_pad']]
            worstnon=min(nonpads,key=lambda v:v['lower_bound_m'])
            worstself=min(newstate['self_geometry_rows'],key=lambda v:v['lower_bound_m'])
            result={'seed':seed,'linear_solver':{'status':int(fit.status),'success':bool(fit.success),'message':fit.message,
                'nit':int(fit.nit),'cost':float(fit.cost),'optimality':float(fit.optimality)},
                'delta_q20':delta,'old_pad_gap_m':gaps,'target_pad_gap_m':target,
                'linear_predicted_pad_gap_m':gaps+G@delta,'actual_pad_gap_m':actualgaps,
                'linearization_error_m':actualgaps-(gaps+G@delta),'geometry_gate_pass':geometry['geometry_gate_pass'],
                'complete22_189':geometry['complete22_189'],'worst_nonpad':worstnon,'worst_self':worstself,
                'ctrl_reachable_without_clamp':reachable,'native_forward_status':'RETURNED' if native else 'BLOCKED_CONTROL_UNREACHABLE_NO_FORWARD',
                'native_contact_count':None if native is None else native['native_contact_count'],
                'native_pad_aggregates':None if native is None else {k:native['native_contact_aggregates'][k] for k in ('thumb_pad','index_pad')},
                'a_COM_world_m_s2':None if native is None else native['raw']['a_COM_world_m_s2'],
                'alpha_world_rad_s2':None if native is None else native['raw']['alpha_world_rad_s2'],
                'joint_qacc20':None if native is None else native['raw']['qacc26'][:20],
                'state':pin(out/f'seed{seed}_preload_state.npz'),'equilibrium_or_holding_qualified':False}
            results.append(result);write(out/f'seed{seed}_result.json',result)
        assert e.counts['native_evaluations_returned']<=2 and e.counts['final_geometry_evaluations']==2
        audit=e.audit_model()
        report={'schema':'two_seed_linear_preload_result:6','status':'COMPLETED_LINEAR_INITIALIZATION_NOT_EQUILIBRIUM',
            'protocol':pin(pp),'results':results,'counts':dict(e.counts),'model_audit':audit,
            'geometry_counts':dict(e.geometry.counts),'elapsed_seconds':time.perf_counter()-start,
            'integration_training_GPU':0,'nonlinear_optimization_LP_calls':0,
            'outputs':{p.name:pin(p) for p in out.iterdir() if p.is_file()}}
        write(out/'report.json',report)
        print(json.dumps({'report':pin(out/'report.json'),'counts':e.counts,
            'summary':[{'seed':r['seed'],'gaps':r['actual_pad_gap_m'].tolist(),'geometry':r['geometry_gate_pass'],
                        'contacts':r['native_contact_count'],'control':r['ctrl_reachable_without_clamp']} for r in results]}))
    except BaseException as error:
        write(out/'failure.json',{'phase':phase,'error':repr(error),'traceback':traceback.format_exc(),
            'native_counts':None if e is None else dict(e.counts),'completed_candidates':len(results),
            'integration_steps':0,'automatic_retry':False})
        raise


if __name__=='__main__':main()
