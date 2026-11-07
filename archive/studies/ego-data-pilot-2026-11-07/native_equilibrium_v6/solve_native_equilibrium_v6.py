"""Zero-speed native equilibrium search, not learning or physical integration.

Actual native soft contacts and all original actuator/passive/constraint forces
determine accelerations. Independent geometry and free rollouts remain gates.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;P=ROOT.parent


def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def plain(v):
    import numpy as np
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(plain(v),f,indent=2,allow_nan=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args();cfg=read(a.protocol)
    if cfg['schema']!='native_soft_contact_equilibrium_protocol:6' or cfg['training_GPU_integration_authorized'] is not False:raise ValueError('Native static CPU only')
    for r in [*cfg['inputs'].values(),*cfg['source_pins'].values()]:
        if pin(r['path'])!=r:raise ValueError('Changed source/input: '+r['path'])
    if cfg['source_pins']['runner']!=pin(__file__):raise ValueError('Exact current source')
    out=Path(cfg['output'])
    if out.parent!=ROOT or out.exists():raise ValueError('Fresh native search directory')
    if not a.run:print(json.dumps({'status':'METADATA_VALID_NO_MODEL_FORWARD_OR_SEARCH','protocol':pin(a.protocol)}));return
    if cfg['CPU_static_run_authorized'] is not True:raise ValueError('Explicit static search scope')
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from evaluator_native_v6 import NativeEquilibriumEvaluator
    from importlib import import_module
    sys.path.insert(0,str(P/'retarget_gravity_v3'));gm=import_module('evaluator_v3')
    out.mkdir();write(out/'protocol.json',cfg);(out/'execution_source.py').write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter();global_deadline=start+cfg['budgets']['total_seconds']
    E=NativeEquilibriumEvaluator.from_existing_pilot(max_heavy_queries=cfg['budgets']['final_geometry_heavy_total'],deadline_monotonic=global_deadline,digest_check_period=cfg['model_digest_period'])
    G=E.geometry;m=E.m;mass=float(m.body_mass[E.oid]);gmag=float(np.linalg.norm(m.opt.gravity));Fscale=mass*gmag;Mscale=Fscale*cfg['characteristic_length_m']
    if np.any(m.dof_frictionloss[20:]!=0) or np.any(m.dof_damping[20:]!=0) or m.jnt_stiffness[int(m.body_jntadr[E.oid])]!=0:
        raise ValueError('No implicit world-referenced friction/damping/spring support on the free phone')
    qlo=m.jnt_range[:20,0].copy();qhi=m.jnt_range[:20,1].copy();ulo=m.actuator_ctrlrange[:,0].copy();uhi=m.actuator_ctrlrange[:,1].copy();qref=G.initial['qpos'][:20]
    all_rows=[];registrations=[];jacobian_rows=[];seed_results=[];counts={'cache_hits':0,'native_base_evaluations':0,'native_FD_evaluations':0,'base_registrations':0,'FD_to_base_promotions':0,'complete_Jacobians':0,'optimizer_calls':0,'integration_training_GPU':0}
    def preserve_unknown_native(s,seed,phase):
        path=out/f'unknown_native_seed{seed}_probe{s["probe_index"]}_{phase}.json'
        with path.open('x',encoding='utf8') as stream:
            json.dump(plain({'scope':'UNQUALIFIED_RAW_DIAGNOSTIC; nonfinite literals permitted only here','native':s}),stream,allow_nan=True)
        return pin(path)
    def measure(s):
        r=s['raw'];aCOM=r['a_COM_world_m_s2'];alpha=r['alpha_world_rad_s2'];netF=mass*aCOM;netM=r['phone_I_world_3x3']@alpha
        pad_normal=np.array([s['native_contact_aggregates'][name]['normal_force_signed_N'] for name in ('thumb_pad','index_pad')])
        pad_pen=np.zeros(2);off_pen=0.;off_force=0.
        for row,detail in zip(r['contact_rows24'],s['native_contact_details']):
            cat=detail['category'];pen=max(0.,-float(row[5]));fn=max(0.,float(row[18]))
            if cat in ('thumb_pad','index_pad'):pad_pen[0 if cat=='thumb_pad' else 1]=max(pad_pen[0 if cat=='thumb_pad' else 1],pen)
            else:off_pen=max(off_pen,pen);off_force+=fn
        v={'COM_acceleration_infinity_m_s2':float(np.max(abs(aCOM))),'angular_acceleration_infinity_rad_s2':float(np.max(abs(alpha))),
           'joint_acceleration_infinity_rad_s2':float(np.max(abs(r['qacc26'][:20]))),'net_force_world_N':netF,'net_moment_world_Nm':netM,
           'pad_normal_N':pad_normal,'pad_penetration_m':pad_pen,'native_other_penetration_m':off_pen,'native_other_normal_force_N':off_force,
           'requested_ctrl_unclamped':bool(np.array_equal(s['PD']['requested_ctrl20'],s['PD']['effective_ctrl20']))}
        gates={k:v[k]<=cfg['entry_gates'][k] for k in ('COM_acceleration_infinity_m_s2','angular_acceleration_infinity_rad_s2','joint_acceleration_infinity_rad_s2','native_other_penetration_m','native_other_normal_force_N')}
        gates.update(native_finite_no_warnings=not s['unknown'],native_two_pads_loaded=bool(np.all(pad_normal>=cfg['entry_gates']['minimum_pad_normal_N'])),native_pad_budget=bool(np.all(pad_normal<=cfg['entry_gates']['maximum_pad_normal_N']+1e-9)),native_pad_compression=bool(np.all(pad_pen<=cfg['pad_compression_limit_m'])),requested_ctrl_unclamped=v['requested_ctrl_unclamped'])
        return v,gates
    for seed_index,alias in enumerate(cfg['start_input_aliases']):
        seed_start=time.perf_counter();seed_end=min(global_deadline,seed_start+cfg['budgets']['per_seed_seconds']);search_end=seed_end-cfg['budgets']['final_reserve_seconds']
        with np.load(cfg['inputs'][alias]['path'],allow_pickle=False) as z:qpos=z['qpos27'].copy();u=z['ctrl20'].copy()
        R=Rotation.from_quat(qpos[23:27][[1,2,3,0]]).as_matrix();t=E.Rp.T@(qpos[20:23]-E.pp);rv=Rotation.from_matrix(E.Rp.T@R).as_rotvec();x0=np.r_[qpos[:20],t,rv,u]
        lo=np.r_[qlo,t-cfg['search_trust_translation_m'],np.maximum(-np.pi,rv-cfg['search_trust_rotation_rad']),ulo]
        hi=np.r_[qhi,t+cfg['search_trust_translation_m'],np.minimum(np.pi,rv+cfg['search_trust_rotation_rad']),uhi]
        if np.any(x0<lo) or np.any(x0>hi):raise ValueError('Initializer outside actual q/ctrl bounds')
        steps=np.r_[np.full(20,cfg['finite_difference']['q_rad']),np.full(3,cfg['finite_difference']['translation_m']),np.full(3,cfg['finite_difference']['rotation_rad']),np.full(20,cfg['finite_difference']['ctrl_rad'])]
        scale=np.r_[np.full(20,.2),np.full(3,.02),np.full(3,.2),np.full(20,.2)]
        cache={};jac_cache={};base_candidates=[];attempt=None;last=None;seed_native_start=E.counts['native_evaluations_attempted']
        def budget():
            if time.perf_counter()>=search_end:raise gm.StaticBudgetExceeded('Native search wall deadline')
            if E.counts['native_evaluations_attempted']-seed_native_start>=cfg['budgets']['native_forward_per_seed']:raise gm.StaticBudgetExceeded('Native forward budget')
        def register(ctx,reason):
            if ctx['registered']:return
            ctx['registered']=True;base_candidates.append(ctx);counts['base_registrations']+=1
            promoted=ctx['row']['original_purpose']=='finite_difference'
            if promoted:counts['FD_to_base_promotions']+=1
            registrations.append({'seed':seed_index,'native_probe_index':ctx['row']['native_probe_index'],'registration_reason':reason,'promoted_from_FD':promoted,'objective':ctx['objective'],'x46':ctx['x'].copy(),'no_new_forward_for_registration':True})
        def evaluate(x,purpose='base'):
            nonlocal attempt,last
            budget();x=np.asarray(x,np.float64);key=x.tobytes()
            if key in cache:
                counts['cache_hits']+=1;ctx=cache[key]
                if purpose=='base':register(ctx,'cached_function_base')
                return ctx
            attempt={'x46':x.copy(),'purpose':purpose,'elapsed_seed_s':time.perf_counter()-seed_start}
            s=E.evaluate(x[:20],x[20:23],x[23:26],x[26:]);d=E.last_data;r=s['raw']
            if s['unknown']:
                attempt['unknown_native_event']=preserve_unknown_native(s,seed_index,'search')
                raise gm.UnknownGeometry('Native forward warning/nonfinite; raw event retained')
            metrics,gates=measure(s)
            anchors=np.array([G.anchors[i]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid] for i,gid in enumerate(E.pads)])
            tips=d.site_xpos[G.tips].copy();Rt=r['phone_body_R_world'];pt=r['phone_root_world_m'];targets=G.anchor_targets@Rt.T+pt;human=G.human_tips@Rt.T+pt
            w=cfg['weights'];blocks={
              'native_object_net_wrench':w['object_wrench']*np.r_[metrics['net_force_world_N']/Fscale,metrics['net_moment_world_Nm']/Mscale],
              'native_joint_acceleration':r['qacc26'][:20]/cfg['joint_acceleration_scale_rad_s2'],
              'native_other_penetration':np.array([metrics['native_other_penetration_m']/cfg['penalty_length_m']*w['nonpad_penetration']]),
              'pad_compression_excess':np.maximum(metrics['pad_penetration_m']-cfg['pad_compression_penalty_start_m'],0)/cfg['penalty_length_m']*w['pad_compression'],
              'pad_normal_budget_excess':np.maximum(metrics['pad_normal_N']-cfg['pad_normal_penalty_start_N'],0)/3.*w['pad_normal_budget'],
              'missing_pad_load':np.maximum(cfg['entry_gates']['minimum_pad_normal_N']-metrics['pad_normal_N'],0)/cfg['entry_gates']['minimum_pad_normal_N']*w['missing_pad_load'],
              'contact_anchor_fidelity':w['anchor_fidelity']*((anchors-targets)/.005).ravel(),
              'candidate_tip_style':(.15*w['tip_style']*(tips[:2]-human[:2])/.01).ravel(),
              'noncontact_tip_style':(.05*w['tip_style']*(tips[2:]-human[2:])/.02).ravel(),
              'source_q_regularizer':.01*(x[:20]-qref)/(qhi-qlo),
              'pose_translation_regularizer':.1*(x[20:23]-t)/cfg['search_trust_translation_m'],
              'pose_rotation_regularizer':.1*(x[23:26]-rv)/cfg['search_trust_rotation_rad'],
              'ctrl_regularizer':.01*(x[26:]-x[:20])/(uhi-ulo),
            }
            residual=np.concatenate([v.ravel() for v in blocks.values()]);objective=float(residual@residual)
            row={'seed':seed_index,'native_probe_index':s['probe_index'],'original_purpose':purpose,'x46':x.copy(),'objective':objective,'blocks':{k:float(v@v) for k,v in blocks.items()},'metrics':metrics,'native_entry_gates':gates,'native_entry_pass':all(gates.values()),'anchor_errors_m':np.linalg.norm(anchors-targets,axis=1),'tip_errors_m':np.linalg.norm(tips-human,axis=1),'contact_count':s['native_contact_count'],'elapsed_seed_s':time.perf_counter()-seed_start,'counts':dict(E.counts)}
            ctx={'x':x.copy(),'qpos27':r['qpos27'].copy(),'residual':residual,'objective':objective,'row':row,'registered':False};cache[key]=ctx;all_rows.append(row);last=ctx
            counts['native_FD_evaluations' if purpose=='finite_difference' else 'native_base_evaluations']+=1
            if purpose=='base':register(ctx,'fresh_function_base')
            return ctx
        def jacobian(x):
            budget();key=np.asarray(x,np.float64).tobytes();b=evaluate(x)
            if key in jac_cache:return jac_cache[key]
            J=np.empty((len(b['residual']),46))
            for k,h in enumerate(steps):
                signed=h if x[k]+h<=hi[k] else -h;probe=np.array(x).copy();probe[k]+=signed;r=evaluate(probe,'finite_difference');J[:,k]=(r['residual']-b['residual'])/signed
            jac_cache[key]=J;counts['complete_Jacobians']+=1;jacobian_rows.append({'seed':seed_index,'jacobian_build':counts['complete_Jacobians'],'native_probe_index':b['row']['native_probe_index'],'objective':b['objective'],'entry_pass':b['row']['native_entry_pass'],'elapsed_seed_s':time.perf_counter()-seed_start});return J
        status='NOT_STARTED';error=None;solver=None
        try:
            evaluate(x0);counts['optimizer_calls']+=1
            fit=least_squares(lambda x:evaluate(x)['residual'],x0,jac=jacobian,bounds=(lo,hi),method='trf',x_scale=scale,max_nfev=cfg['solver']['max_nfev'],ftol=cfg['solver']['ftol'],xtol=cfg['solver']['xtol'],gtol=cfg['solver']['gtol'])
            evaluate(fit.x);status='SOLVER_RETURNED';solver={'success':bool(fit.success),'status':int(fit.status),'message':str(fit.message),'nfev':int(fit.nfev),'njev':None if fit.njev is None else int(fit.njev),'cost':float(fit.cost),'optimality':float(fit.optimality)}
        except (gm.StaticBudgetExceeded,gm.UnknownGeometry,ValueError,ArithmeticError) as ex:status='BOUNDED_STOP_OR_UNKNOWN_PRESERVED';error=repr(ex)
        ranked=sorted(base_candidates,key=lambda b:(not b['row']['native_entry_pass'],b['objective']))
        # Keep a bounded diverse ordered shortlist; the rule is frozen before
        # search, not chosen after observing free-rollout success.
        shortlist=[]
        for ctx in ranked:
            if all(np.max(abs((ctx['x']-prev['x'])/scale))>cfg['final_shortlist_min_scaled_difference'] for prev in shortlist):shortlist.append(ctx)
            if len(shortlist)>=cfg['final_shortlist_count']:break
        final=[]
        E.geometry.set_budget(deadline_monotonic=seed_end,max_additional_heavy_queries=cfg['budgets']['final_geometry_heavy_per_seed'])
        for rank,ctx in enumerate(shortlist):
            record={'rank':rank,'original_native_probe_index':ctx['row']['native_probe_index'],'x46':ctx['x'],'source_row':ctx['row'],'candidate_pass':False,'qualification_provisional_until_global_model_audit':True,'unknown':[]}
            x=ctx['x']
            try:
                if time.perf_counter()>=seed_end:raise gm.StaticBudgetExceeded('Final shortlist deadline')
                s=E.evaluate(x[:20],x[20:23],x[23:26],x[26:])
                if s['unknown']:
                    record['unknown_native_event']=preserve_unknown_native(s,seed_index,f'rank{rank}')
                    raise gm.UnknownGeometry('Final native warning/nonfinite; raw event retained')
                record['native']=s;metrics,gates=measure(s)
                ge=E.final_geometry(x[:20],x[20:23],x[23:26],pad_gap_window_m=(-cfg['pad_compression_limit_m'],.0002),nonpad_buffer_m=0.,self_buffer_m=0.)
                record.update(native=s,metrics=metrics,native_entry_gates=gates,geometry=ge,candidate_pass=bool(all(gates.values()) and ge['geometry_gate_pass']))
                prefix=f'seed{seed_index}_rank{rank}'
                np.savez_compressed(out/(prefix+'_state.npz'),qpos27=s['raw']['qpos27'],qvel26=np.zeros(26),ctrl20=x[26:],x46=x,geom_xpos=E.last_data.geom_xpos.copy(),geom_xmat=E.last_data.geom_xmat.copy(),site_xpos=E.last_data.site_xpos.copy(),body_xpos=E.last_data.xpos.copy(),body_xquat=E.last_data.xquat.copy())
                record['state']=pin(out/(prefix+'_state.npz'))
                # Fresh final repeat is separate from integration. Bit equality
                # is checked only for physical fields, not counters/IDs.
                if time.perf_counter()>=seed_end:raise gm.StaticBudgetExceeded('Final fresh-repeat deadline')
                repeat=E.evaluate(x[:20],x[20:23],x[23:26],x[26:]);keys=('qpos27','ctrl20','qacc26','qfrc_actuator26','qfrc_constraint26','a_COM_world_m_s2','alpha_world_rad_s2','contact_rows24')
                if repeat['unknown']:
                    record['repeat_unknown_event']=preserve_unknown_native(repeat,seed_index,f'rank{rank}_repeat')
                    raise gm.UnknownGeometry('Final repeat warning/nonfinite; raw event retained')
                record['fresh_repeat_native']=repeat
                _,repeat_gates=measure(repeat);record['fresh_repeat_entry_gates']=repeat_gates
                record['fresh_repeat_bit_equal']={k:bool(np.array_equal(s['raw'][k],repeat['raw'][k])) for k in keys}
                # Contact-row first column is a diagnostic probe ID, so it is
                # deliberately excluded from physical-contact bit comparison.
                record['fresh_repeat_bit_equal']['contact_rows24']=bool(np.array_equal(s['raw']['contact_rows24'][:,1:],repeat['raw']['contact_rows24'][:,1:]))
                record['candidate_pass']=bool(record['candidate_pass'] and not repeat['unknown'] and all(repeat_gates.values()) and all(record['fresh_repeat_bit_equal'].values()))
            except (gm.StaticBudgetExceeded,gm.UnknownGeometry,ValueError,ArithmeticError) as ex:
                record['unknown'].append(repr(ex));record['candidate_pass']=False
            final.append(record);write(out/f'seed{seed_index}_rank{rank}_verification.json',record)
        if last is not None:np.savez_compressed(out/f'seed{seed_index}_last_unqualified_state.npz',qpos27=last['qpos27'],qvel26=np.zeros(26),ctrl20=last['x'][26:],x46=last['x'])
        result={'seed':seed_index,'start':cfg['inputs'][alias],'status':status,'error':error,'solver':solver,'last_complete':None if last is None else last['row'],'failed_attempt':attempt if error else None,'native_base_candidate_count':len(base_candidates),'shortlist':final,'qualified_candidate_count':sum(v['candidate_pass'] for v in final),'elapsed_s':time.perf_counter()-seed_start,'holding_qualified':False}
        seed_results.append(result);write(out/f'seed{seed_index}_result.json',result)
        print(json.dumps(plain({'seed':seed_index,'status':status,'native_evals':len(cache),'base_candidates':len(base_candidates),'Jacobian_builds':len(jac_cache),'best_objective':ranked[0]['objective'] if ranked else None,'qualified_candidates':result['qualified_candidate_count'],'elapsed_s':result['elapsed_s']})),flush=True)
    for name,data in [('all_native_evaluations.jsonl',all_rows),('base_registrations.jsonl',registrations),('jacobian_constructions.jsonl',jacobian_rows)]:
        with (out/name).open('x',encoding='utf8') as f:
            for r in data:f.write(json.dumps(plain(r),allow_nan=False)+'\n')
    try:
        audit=E.audit_model();audit['passed']=True
    except Exception as ex:
        audit={'passed':False,'error':repr(ex),'counts':dict(E.counts),'all_qualifications_invalidated':True}
        write(out/'model_audit_failure.json',audit)
        for result in seed_results:
            result['qualified_candidate_count']=0
            for candidate in result['shortlist']:
                candidate['candidate_pass']=False;candidate['unknown'].append('GLOBAL_MODEL_AUDIT_FAILED')
    report={'schema':'native_soft_contact_equilibrium_result:6','status':'COMPLETED_BOUNDED_NATIVE_STATIC_SEARCH','protocol':pin(a.protocol),'seed_results':seed_results,'counts':counts,'native_counts':E.counts,'geometry_counts':E.geometry.counts,'model_audit':audit,'force_scale_N':Fscale,'moment_scale_Nm':Mscale,'elapsed_s':time.perf_counter()-start,'qualified_candidate_count':sum(v['qualified_candidate_count'] for v in seed_results),'holding_or_policy_qualified':False,'integration_training_GPU':0,'scope':'Each probe fresh native zero-speed MjData/forward; static pose/control optimization only. New designatedpad compression hypothesis, actual model native forces and frictionloss/limits. Missing nativecontacts are not clearance proof; final all22/189 independent geometry. Entry acceleration gates and repeats are proposals; unchanged2s free-object gate decides holding. Original model and wrist unchanged. No learned weights or object aid.'}
    write(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json'),'qualified_count':report['qualified_candidate_count'],'seconds':report['elapsed_s']}),flush=True)


if __name__=='__main__':main()
