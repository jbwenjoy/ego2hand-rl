"""UNFROZEN bounded fixed-q native-wrench control preparation; default metadata only."""
from pathlib import Path
import argparse,importlib.util,json,math,sys,time
H=Path(__file__).resolve().parent
SCHEMA='functional_tripod_native_preload_protocol:1'
RESULT='functional_tripod_native_preload_result:1'
BUDGET=dict(max_candidates=2,native_forward_per_candidate=40,total_native_forward=80,search_iterations=2,FD_control_axes=list(range(12)),final_repeat_forward=2)
RULES=dict(FD_step_rad=.001,joint_control_step_trust_rad=.02,damping=.001,line_search_fractions=[1.,.5,.25],
    force_reference='mass_times_gravity_norm_N',moment_lever_reference_m=.077,pad_loading_target_N=.100001,pad_deficit_scale_N=1.,
    routes=['ctrl_equal_q','actual_contact_normal_1N','actual_contact_normal_3N'],dt_s=.00015625,
    rank='legal_native_loaded_entry_first_then_squared_dimensionless_residual; all raw probes retained')
def load_core():
    p=H/'native_contact_preload_v1.py';s=importlib.util.spec_from_file_location('tripod_native_preload_core',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def metadata_preflight(path):
    core=load_core();p=core.read(path) if hasattr(core,'read') else json.loads(Path(path).read_text(encoding='utf8'))
    if p['schema']!=SCHEMA or type(p['native_forward_authorized']) is not bool or p['integration_training_GPU_authorized'] is not False or p['budget']!=BUDGET or p['rules']!=RULES:raise ValueError('Fixed bounded native-only source contract')
    for x in [*p['inputs'].values(),*p['source_pins'].values()]:core.checked(x)
    if set(p['inputs'])!={'model_mjb','model_binding','candidate_report'} or p['source_pins']['runner']!=core.pin(__file__) or p['source_pins']['core']!=core.pin(H/'native_contact_preload_v1.py'):raise ValueError('Exact pins/inputs')
    out=Path(p['output'])
    if not out.is_absolute() or out.parent.resolve()!=H.resolve() or out.exists():raise ValueError('Fresh own output')
    r=json.loads(Path(p['inputs']['candidate_report']['path']).read_text());entries=r['candidates']
    if r['schema']!='functional_tripod_geometric_reference_result:1' or r['model_mjb']!=p['inputs']['model_mjb'] or len(entries)>2:raise ValueError('Same compressed geometry family')
    selected=[];screen=[]
    for entry in entries:
        c=json.loads(core.checked(entry['candidate']).read_text());core.checked(entry['state_npz']);g=c['geometry'];reasons=[]
        if c['schema']!='functional_tripod_geometric_reference_candidate:1' or c['designated_geom_ids']!=[15,28,41]:raise ValueError('Exact candidate identity')
        if not (g.get('passed') is True and g.get('phone_pair_count')==22 and g.get('self_pair_count')==189 and g.get('unknown')==[] and g.get('table_complete') is True):reasons.append('Failed/incomplete original214 geometry')
        if c.get('geometry_and_compression_target_pass') is not True or c.get('compression_target_pass') is not True:reasons.append('Compression proposal target not qualified')
        screen.append(dict(candidate_id=c['candidate_id'],eligible=not reasons,reasons=reasons,candidate=entry['candidate'],state_npz=entry['state_npz'],
            source_geometry_pass=g.get('passed'),compression_target_pass=c.get('compression_target_pass'),
            geometry_and_compression_target_pass=c.get('geometry_and_compression_target_pass'),unknown=g.get('unknown'),violations=g.get('violations')))
        if not reasons:selected.append(dict(candidate_data=c,**entry))
    return core,p,out,selected,screen

def wrench_residual(probe,*,target_wrench_world6=None):
    import numpy as np
    target=np.asarray(probe['target_wrench_world6'] if target_wrench_world6 is None else target_wrench_world6,dtype=float)
    actual=np.asarray(probe['pad_only_wrench_world6']);scale=float(probe['mass_kg'])*float(np.linalg.norm(probe['gravity_world_m_s2']))
    force=(actual[:3]-target[:3])/scale;moment=(actual[3:]-target[3:])/(scale*RULES['moment_lever_reference_m'])
    deficit=np.maximum(0.,RULES['pad_loading_target_N']-np.asarray(probe['pad_normal_N3']))/RULES['pad_deficit_scale_N']
    return np.r_[force,moment,deficit]
def rank(probe):
    residual=wrench_residual(probe)
    return (not probe['measurements']['original_no_unwanted_penetration_control_bounds'],not probe['measurements']['native_loaded_entry'],float(residual@residual))

def refine_control(evaluator,qpos27,start,*,candidate_id,target_wrench_world6=None):
    """At most two explicit12-column finite-action GN steps; no AD/hidden FW.

    Reserve two FINAL repeat probes. Finite control differences are proposals,
    not contact-force derivatives or independent physical-gradient validation.
    """
    import numpy as np
    from scipy.optimize import lsq_linear
    current=start;log=[]
    if not start['measurements']['all3_generated']:return current,[dict(status='MISSING_GENERATED_PAD_FIXED_Q_CANNOT_CREATE_CONTACT')]
    lo,hi=evaluator.control_box(qpos27)
    for iteration in range(BUDGET['search_iterations']):
        u=np.asarray(current['raw']['ctrl20']).copy();r=wrench_residual(current,target_wrench_world6=target_wrench_world6);J=np.zeros((9,12));FD=[]
        for j in BUDGET['FD_control_axes']:
            if evaluator.per_candidate[candidate_id]>=BUDGET['native_forward_per_candidate']-2:break
            h=min(RULES['FD_step_rad'],float(hi[j]-u[j]))
            if h<1e-6:h=-min(RULES['FD_step_rad'],float(u[j]-lo[j]))
            changed=u.copy();changed[j]=float(u[j]+h);du=float(changed[j]-u[j])
            if abs(du)<1e-6:FD.append(dict(axis=j,status='NO_USABLE_CONTROL_MARGIN',actual_du=du));continue
            side=evaluator.evaluate(qpos27,changed,candidate_id=candidate_id,tag=f'FD_iter{iteration}_axis{j}',target_wrench_world6=target_wrench_world6)
            J[:,j]=(wrench_residual(side,target_wrench_world6=target_wrench_world6)-r)/du
            FD.append(dict(axis=j,status='FINITE_CONTROL_PROPOSAL_COLUMN',actual_du=du,probe=side.get('saved_probe')))
        if len([v for v in FD if v['status']=='FINITE_CONTROL_PROPOSAL_COLUMN'])<12:
            log.append(dict(iteration=iteration,status='INCOMPLETE12_COLUMN_BUDGET_OR_BOUND_STOP',FD=FD));break
        A=np.vstack([J,math.sqrt(RULES['damping'])*np.eye(12)]);b=np.r_[-r,np.zeros(12)]
        lower=np.maximum(lo[:12]-u[:12],-RULES['joint_control_step_trust_rad']);upper=np.minimum(hi[:12]-u[:12],RULES['joint_control_step_trust_rad'])
        solve=lsq_linear(A,b,bounds=(lower,upper),method='trf',tol=1e-10,max_iter=100)
        note=dict(iteration=iteration,FD=FD,numerical_bound_solve_success=bool(solve.success),dq_ctrl12=solve.x.tolist(),line_search=[],accepted=False)
        for fraction in RULES['line_search_fractions']:
            if evaluator.per_candidate[candidate_id]>=BUDGET['native_forward_per_candidate']-2:break
            trial=u.copy();trial[:12]+=fraction*solve.x
            value=evaluator.evaluate(qpos27,trial,candidate_id=candidate_id,tag=f'line_iter{iteration}_fraction{fraction}',target_wrench_world6=target_wrench_world6)
            good=value['measurements']['original_no_unwanted_penetration_control_bounds'] and rank(value)<rank(current)
            note['line_search'].append(dict(fraction=fraction,probe=value.get('saved_probe'),rank=rank(value),accepted=bool(good)))
            if good:current=value;note['accepted']=True;break
        log.append(note)
        if not note['accepted']:break
    return current,log

def main():
    a=argparse.ArgumentParser();a.add_argument('--protocol',type=Path,required=True);a.add_argument('--run',action='store_true');args=a.parse_args()
    core,p,out,selected,screen=metadata_preflight(args.protocol)
    if not args.run:
        print(json.dumps(dict(status='METADATA_ONLY_NO_NATIVE_OR_ARRAY_VALUES',screening=screen,maximum_native_forward=40*len(selected),integration_steps=0)));return
    if p['native_forward_authorized'] is not True:raise ValueError('Root native-forward authorization required')
    out.mkdir();core.write(out/'protocol.json',p);phase='construction';e=None;results=[];start=time.perf_counter()
    try:
        if selected:
            import numpy as np
            e=core.NativeTripodPreloadEvaluator(p['inputs']['model_mjb'],p['inputs']['model_binding'],output_dir=out)
            for item in selected:
                c=item['candidate_data'];cid=c['candidate_id'];phase=cid;q=np.asarray(c['qpos27_F64'],dtype=np.float64)
                with np.load(core.checked(item['state_npz']),allow_pickle=False) as z:
                    if z['qpos27_F64'].dtype!=np.float64 or z['qpos27_F64'].tobytes()!=q.tobytes() or np.any(z['qvel26_F64']):raise ValueError('Actual raw compressed state binding')
                entry=dict(candidate_id=cid,candidate=item['candidate'],state_npz=item['state_npz'],routes=[],search_log=[])
                lo,hi=e.control_box(q)
                if np.any(q[:20]<lo) or np.any(q[:20]>hi):
                    entry.update(status='BASELINE_CTRL_Q_UNAVAILABLE_NO_NATIVE_CONTACT_PROPOSAL',unavailable_reason='No silent command clamp or alternate initializer');results.append(entry);continue
                baseline=e.evaluate(q,q[:20],candidate_id=cid,tag='ctrl_equal_q');values=[baseline]
                entry['routes'].append(dict(route='ctrl_equal_q',available=True,probe=baseline['saved_probe']))
                for load in [1.,3.]:
                    proposal=core.normal_contact_proposal(baseline,load);entry['routes'].append(dict(route=f'actual_contact_normal_{load:g}N',proposal=proposal,available=proposal['available']))
                    if proposal['available']:
                        value=e.evaluate(q,proposal['ctrl20'],candidate_id=cid,tag=f'actual_contact_normal_{load:g}N');values.append(value);entry['routes'][-1]['probe']=value['saved_probe']
                chosen=min(values,key=rank);best,log=refine_control(e,q,chosen,candidate_id=cid);entry['search_log']=log
                final=[]
                for i in range(2):final.append(e.evaluate(q,best['raw']['ctrl20'],candidate_id=cid,tag=f'FINAL_fresh_repeat{i}'))
                raw_keys=['qpos27','qvel26','qacc26_descriptor','ctrl20','qfrc_actuator26','actuator_force20','contact_rows24']
                repeat={k:np.asarray(final[0]['raw'][k]).tobytes()==np.asarray(final[1]['raw'][k]).tobytes() for k in raw_keys if k!='contact_rows24'}
                # Column0 is the explicit probe id, not physical contact evidence.
                repeat['contact_rows24_excluding_probe_id']=np.asarray(final[0]['raw']['contact_rows24'])[:,1:].tobytes()==np.asarray(final[1]['raw']['contact_rows24'])[:,1:].tobytes()
                entry.update(status='COMPLETED_BOUNDED_NATIVE_LOAD_PREPARATION',selected_search_probe=best['saved_probe'],final_repeat_probes=[v['saved_probe'] for v in final],fresh_repeat_bit_equal=repeat,
                    native_loaded_entry=bool(all(repeat.values()) and all(v['measurements']['native_loaded_entry'] for v in final)),
                    final_pad_normal_N3=final[0]['pad_normal_N3'],final_actual_pad_wrench_world6=final[0]['pad_only_wrench_world6'],final_target_wrench_world6=final[0]['target_wrench_world6'],
                    final_actual_wrench_error_world6=final[0]['pad_wrench_error_world6'],
                    final_force_error_Linf_N=float(np.max(abs(np.asarray(final[0]['pad_wrench_error_world6'])[:3]))),
                    final_moment_error_Linf_Nm=float(np.max(abs(np.asarray(final[0]['pad_wrench_error_world6'])[3:]))),
                    native_wrench_equilibrium_qualified=False,holding_qualified=False)
                results.append(entry);core.write(out/(cid+'_result.json'),entry)
            audit=e.audit_model()
        else:audit=dict(model_loaded=False)
        core.write(out/'report.json',dict(schema=RESULT,status='COMPLETED_BOUNDED_ACTUAL_NATIVE_PRELOAD_NOT_HOLDING',protocol=core.pin(args.protocol),screening=screen,candidates=results,
            model_audit=audit,counts={} if e is None else e.counts,elapsed_s=time.perf_counter()-start,
            native_loaded_entries=sum(v.get('native_loaded_entry',False) for v in results),holding_qualified=False,
            scope='Fixed compressed geometry; same-forward actual native contact points/frames/J. Normal proposals and measured control-FD search remain separate from actual entry loading and target-wrench error. No qacc-zero, ideal LP, NN, integration, object aid or calibrated contact claim.'))
    except BaseException as error:
        core.write(out/'failure.json',dict(phase=phase,error=repr(error),completed=results,counts={} if e is None else e.counts,automatic_retry=False));raise
if __name__=='__main__':main()
