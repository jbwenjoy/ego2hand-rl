"""Sequential geometry polish: phone pose fixed, original20 joints optimized.

This closes the physically incomplete v3 search poses; it is not a replacement
for gravity support, true torque limits or free-object dynamical validation.
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


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(plain(v),f,indent=2,allow_nan=False)


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args();c=read(a.protocol)
    if c['schema']!='fixed_phone_joint_closure_protocol:4' or c['training_GPU_physics_authorized'] is not False:raise ValueError('Staticgeometryonly')
    for rec in [*c['inputs'].values(),*c['source_pins'].values()]:
        if pin(rec['path'])!=rec:raise ValueError('Changedsource/input')
    if c['source_pins']['runner']!=pin(__file__):raise ValueError('Currentrunner')
    out=Path(c['output'])
    if out.exists() or out.parent!=ROOT:raise ValueError('Freshoutput')
    if not a.run:print(json.dumps({'status':'METADATA_ONLY_VALID_NO_OPTIMIZATION','protocol':pin(a.protocol)}));return
    if c['CPU_static_run_authorized'] is not True:raise ValueError('Staticjobscope')
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    sys.path.insert(0,str(P/'retarget_gravity_v3'));from evaluator_v3 import StaticRetargetEvaluator,StaticBudgetExceeded,UnknownGeometry
    sys.path.insert(0,str(P/'gravity_support_design_v3'));from support_residual_lp_v3 import support_residual_lp
    out.mkdir();write(out/'protocol.json',c);(out/'execution_source.py').write_bytes(Path(__file__).read_bytes())
    started=time.perf_counter();deadline=started+c['budgets']['total_seconds'];E=StaticRetargetEvaluator.from_existing_pilot(max_heavy_queries=c['budgets']['heavy_total'],deadline_monotonic=deadline)
    qref=E.initial['qpos'][:20];qrange=E.qhi-E.qlo;rows=[];callbacks=[];results=[];LP_calls=0;FD_calls=0;base_calls=0
    for index,alias in enumerate(c['start_input_aliases']):
        seed_start=time.perf_counter();seed_end=min(deadline,seed_start+c['budgets']['per_seed_seconds'])
        with np.load(c['inputs'][alias]['path'],allow_pickle=False) as z:qpos=z['qpos27'].copy()
        R=Rotation.from_quat(qpos[23:27][[1,2,3,0]]).as_matrix();t=E.Rp.T@(qpos[20:23]-E.pp);rv=Rotation.from_matrix(E.Rp.T@R).as_rotvec()
        E.set_budget(deadline_monotonic=seed_end-c['budgets']['final_reserve_seconds'],max_additional_heavy_queries=min(c['budgets']['heavy_per_seed']-c['budgets']['heavy_final_reserve'],c['budgets']['heavy_total']-E.counts['heavy_queries_attempted']-c['budgets']['heavy_final_reserve']))
        cache={};jac_cache={};best_search=None;best_physical=None;best_violation=None;last=None;attempt=None;iteration=0
        def evaluate(q,purpose='base'):
            nonlocal best_search,best_physical,best_violation,last,attempt,base_calls,FD_calls
            key=np.asarray(q,np.float64).tobytes()
            if key in cache:return cache[key]
            attempt={'q20':np.asarray(q).copy(),'purpose':purpose};s=E.evaluate(q,t,rv)
            if not np.array_equal(s['phone_R_world'],R) and np.max(abs(s['phone_R_world']-R))>1e-15:raise ValueError('Phonefixedorientationchanged')
            if np.max(abs(s['phone_p_world_m']-qpos[20:23]))>1e-15:raise ValueError('Phonefixedpositionchanged')
            search=E.constraint_vector(s,pad_gap_window_m=c['geometry']['search_pad_window_m'],nonpad_buffer_m=c['geometry']['nonpad_buffer_m'],self_buffer_m=c['geometry']['self_buffer_m'])
            physical=E.constraint_vector(s,pad_gap_window_m=(0.,.0002),nonpad_buffer_m=0.,self_buffer_m=0.)
            b={'contact_anchor':((s['anchors_world_m']-s['anchor_targets_world_m'])/.005).ravel(),'candidate_tip':(.15*(s['tips_world_m'][:2]-s['human_tip_targets_world_m'][:2])/.01).ravel(),'noncontact_tip':(.05*(s['tips_world_m'][2:]-s['human_tip_targets_world_m'][2:])/.02).ravel(),'joint_regularizer':.01*(q-qref)/qrange}
            objective=float(sum(v@v for v in b.values()));constraints=search['values_m']/.001;violation=float(np.maximum(-constraints,0).max())
            row={'seed':index,'purpose':purpose,'q20':np.asarray(q).copy(),'objective':objective,'blocks':{k:float(v@v) for k,v in b.items()},'search_geometry_pass':search['all_satisfied'],'physical_geometry_pass':physical['all_satisfied'],'max_scaled_constraint_violation':violation,'fixed_material_gravity_axis_Nm':s['fixed_material_line_gravity_torque_Nm'],'pad_gaps_m':[v['exact_gap_m'] for v in s['phone_geometry_rows'] if v['is_pad']],'tip_errors_m':np.linalg.norm(s['tips_world_m']-s['human_tip_targets_world_m'],axis=1),'anchor_errors_m':np.linalg.norm(s['anchors_world_m']-s['anchor_targets_world_m'],axis=1),'elapsed_seed_s':time.perf_counter()-seed_start,'counts':dict(E.counts)}
            ctx={'q':np.asarray(q).copy(),'qpos27':s['qpos27'].copy(),'row':row,'objective':objective,'constraints':constraints};cache[key]=ctx;rows.append(row);last=ctx
            if purpose=='finite_difference':FD_calls+=1
            else:
                base_calls+=1
                if search['all_satisfied'] and (best_search is None or objective<best_search['objective']):best_search=ctx
                if physical['all_satisfied'] and (best_physical is None or objective<best_physical['objective']):best_physical=ctx
                if best_violation is None or (violation,objective)<(best_violation['row']['max_scaled_constraint_violation'],best_violation['objective']):best_violation=ctx
            return ctx
        def derivative(q):
            key=np.asarray(q,np.float64).tobytes()
            if key in jac_cache:return jac_cache[key]
            b=evaluate(q);grad=np.empty(20);jac=np.empty((len(b['constraints']),20))
            for k in range(20):
                h=c['finite_difference_q_rad'];h=h if q[k]+h<=E.qhi[k] else -h;probe=np.array(q).copy();probe[k]+=h;r=evaluate(probe,'finite_difference');grad[k]=(r['objective']-b['objective'])/h;jac[:,k]=(r['constraints']-b['constraints'])/h
            jac_cache[key]=(grad,jac);return grad,jac
        def callback(q):
            nonlocal iteration
            iteration+=1;callbacks.append({'iteration':iteration,**evaluate(q)['row']})
        status='NOT_STARTED';error=None;solver=None
        try:
            evaluate(qpos[:20]);fit=minimize(lambda q:evaluate(q)['objective'],qpos[:20],method='SLSQP',jac=lambda q:derivative(q)[0],bounds=list(zip(E.qlo,E.qhi)),constraints=[{'type':'ineq','fun':lambda q:evaluate(q)['constraints'],'jac':lambda q:derivative(q)[1]}],callback=callback,options={'maxiter':c['solver']['maxiter'],'ftol':c['solver']['ftol'],'disp':False});evaluate(fit.x)
            status='SOLVER_RETURNED';solver={'status':int(fit.status),'success':bool(fit.success),'message':str(fit.message),'nit':int(fit.nit),'nfev':int(fit.nfev),'njev':int(fit.njev)}
        except (StaticBudgetExceeded,UnknownGeometry,ValueError,ArithmeticError) as ex:status='BOUNDED_STOP_OR_UNKNOWN_PRESERVED';error=repr(ex)
        selected=best_search if best_search is not None else best_physical if best_physical is not None else best_violation
        result={'seed':index,'source_start':c['inputs'][alias],'status':status,'error':error,'solver':solver,'iterations':iteration,'complete_evaluations':len(cache),'selection':'BEST_SEARCH_GEOMETRY' if best_search else 'BEST_PHYSICAL_GEOMETRY' if best_physical else 'BEST_VIOLATION','last_complete':None if last is None else last['row'],'failed_attempt':attempt if error else None,'unknown':[],'physical_geometry_pass':False,'strict_support_proposal_pass':False,'eligible_for_bounded_dynamic_diagnostic':False}
        if selected is not None:
            result['selected_evaluation']=selected['row'];E.set_budget(deadline_monotonic=seed_end,max_additional_heavy_queries=min(c['budgets']['heavy_final_reserve'],max(0,c['budgets']['heavy_total']-E.counts['heavy_queries_attempted'])))
            try:
                s=E.evaluate(selected['q'],t,rv)
                for row in s['phone_geometry_rows']:
                    if row['status']=='DISJOINT_AABB_CERTIFICATE':
                        row.update(E._describe_pair({'kind':'convex','v':E.phone_vertices},E._shape(row['geom_id'],s['phone_R_world'],s['phone_p_world_m']),phone=True,force_exact=True));row.pop('not_exact_gap',None)
                s['counts']=dict(E.counts);result['verification_state']=s
                phys=E.constraint_vector(s,pad_gap_window_m=(0.,.0002),nonpad_buffer_m=0.,self_buffer_m=0.);result['physical_geometry_pass']=phys['all_satisfied']
                np.savez_compressed(out/f'seed{index}_selected_state.npz',qpos27=s['qpos27'],qvel26=np.zeros(26),ctrl20=np.zeros(20),x26=np.r_[selected['q'],t,rv],geom_xpos=E.d.geom_xpos.copy(),geom_xmat=E.d.geom_xmat.copy(),site_xpos=E.d.site_xpos.copy(),body_xpos=E.d.xpos.copy(),body_xquat=E.d.xquat.copy(),phone_R_world=s['phone_R_world'],phone_p_world=s['phone_p_world_m'])
                result['selected_state']=pin(out/f'seed{index}_selected_state.npz')
                if phys['all_satisfied'] and s['support_operands']['available'] and np.all(s['qfrc_passive26_Nm'][:20]==0):
                    lp=support_residual_lp(**E.support_lp_kwargs(s),normal_cap_N=3.,tangent_reference_axis_world=(0.,1.,0.));LP_calls+=lp['LP_calls'];result['support_LP']=lp
                    if lp.get('feasible_force_balance',False):
                        pts=s['support_operands']['contact_points_world'];axis=pts[1]-pts[0];length=float(np.linalg.norm(axis));below=None
                        if length>1e-9:
                            axis/=length;arm=s['support_operands']['object_COM']-pts[0];g=s['support_operands']['gravity_world'];below=float((arm-axis*(axis@arm))@(g-axis*(axis@g)))
                        tau=np.asarray(lp['tau_act_required_Nm']);ctrl=(tau-s['PD_offset20'])/s['PD_gain20'];reachable=bool(np.isfinite(ctrl).all() and np.all(ctrl>=E.m.actuator_ctrlrange[:,0]) and np.all(ctrl<=E.m.actuator_ctrlrange[:,1]) and np.max(abs(s['PD_gain20']*ctrl+s['PD_offset20']-tau))<=1e-12)
                        result.update(actual_contact_line_length_m=length,actual_COM_below_dot_m2_s2=below,proposed_ctrl20=ctrl,proposed_tau20_Nm=tau,proposed_ctrl_reachable_without_clamping=reachable,strict_support_proposal_pass=bool(lp['proposal_gate'] and below is not None and below>=0 and reachable),eligible_for_bounded_dynamic_diagnostic=bool(below is not None and below>=0 and reachable))
            except (StaticBudgetExceeded,UnknownGeometry,ValueError,ArithmeticError) as ex:
                result['unknown'].append(repr(ex));result['eligible_for_bounded_dynamic_diagnostic']=False
                np.savez_compressed(out/f'seed{index}_unverified_state.npz',qpos27=selected['qpos27'],qvel26=np.zeros(26),ctrl20=np.zeros(20));result['unverified_raw_state']=pin(out/f'seed{index}_unverified_state.npz')
        result['elapsed_s']=time.perf_counter()-seed_start;results.append(result);write(out/f'seed{index}_result.json',result)
        print(json.dumps(plain({'seed':index,'status':status,'iterations':iteration,'geometry_pass':result['physical_geometry_pass'],'strict_support_proposal':result['strict_support_proposal_pass'],'dynamic_diagnostic_eligible':result['eligible_for_bounded_dynamic_diagnostic'],'selected_gap_m':result.get('selected_evaluation',{}).get('pad_gaps_m'),'moment_infinity_Nm':result.get('support_LP',{}).get('moment_infinity_Nm')})),flush=True)
    for name,data in [('all_evaluations.jsonl',rows),('actual_iterations.jsonl',callbacks)]:
        with (out/name).open('x',encoding='utf8') as f:
            for row in data:f.write(json.dumps(plain(row),allow_nan=False)+'\n')
    report={'schema':'fixed_phone_joint_closure_result:4','status':'COMPLETED_STATIC_PHONE_FIXED_GEOMETRY_POLISH','protocol':pin(a.protocol),'seed_results':results,'counts':E.counts,'LP_calls':LP_calls,'base_evaluations':base_calls,'FD_evaluations':FD_calls,'elapsed_s':time.perf_counter()-started,'geometry_pass_count':sum(r['physical_geometry_pass'] for r in results),'strict_support_proposal_count':sum(r['strict_support_proposal_pass'] for r in results),'bounded_dynamic_diagnostic_eligible_count':sum(r['eligible_for_bounded_dynamic_diagnostic'] for r in results),'holding_or_ego_policy_qualified':False,'integration_training_GPU':0,'model_parameter_digest_unchanged':E.parameter_digest,'scope':'Sequential engineering exploration; selects from previous failed search prefixes. Phone/world wrist/scale/material/robot parameters fixed. Allstarts/results kept. Finalforcehard LP is separate from geometry objective. Dynamicdiagnostic enrollment does not require small moment and never upgrades the unchanged1e-4Nm strict proposal gate; only future unassisted rollout can test holding.'}
    write(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json'),'geometry_count':report['geometry_pass_count'],'diagnostic_count':report['bounded_dynamic_diagnostic_eligible_count']}),flush=True)


if __name__=='__main__':main()
