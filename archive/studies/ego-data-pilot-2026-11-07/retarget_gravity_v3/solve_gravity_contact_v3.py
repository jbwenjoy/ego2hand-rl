"""Two-stage original-fixed-wrist contact/gravity pose proposals, no physics.

Stage A uses a smooth fixed-material-line necessary condition. Stage B uses
current enrolled geometry and genuine unique closest pairs for conditional LP.
No small moment residual is described as exact statics or successful holding.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
P=ROOT.parent


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def plain(x):
    import numpy as np
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(plain(x),f,indent=2,allow_nan=False)


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    cfg=read(args.protocol)
    if (cfg['schema']!='fixed_wrist_gravity_contact_pose_protocol:3' or cfg['training_GPU_physics_authorized'] is not False
            or cfg['support']['proposal_moment_limit_Nm']!=1e-4):
        raise ValueError('Static CPU proposal protocol')
    for record in [*cfg['inputs'].values(),*cfg['source_pins'].values()]:
        if pin(record['path'])!=record:raise ValueError('Changed pinned input/source: '+record['path'])
    if cfg['source_pins']['runner']!=pin(__file__):raise ValueError('Executed source')
    out=Path(cfg['output'])
    if out.parent!=ROOT or out.exists():raise ValueError('Fresh result directory')
    if not args.run:
        print(json.dumps({'status':'METADATA_VALID_NO_MODELS_OR_OPTIMIZATION','protocol':pin(args.protocol),'budget':cfg['budgets']}));return
    if cfg['CPU_static_run_authorized'] is not True:raise ValueError('Explicit static job scope')
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    from evaluator_v3 import StaticRetargetEvaluator,StaticBudgetExceeded,UnknownGeometry
    sys.path.insert(0,str(P/'gravity_support_design_v3'))
    from support_residual_lp_v3 import support_residual_lp
    out.mkdir();write(out/'protocol.json',cfg)
    (out/'execution_source.py').write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter();deadline=start+cfg['budgets']['total_seconds']
    E=StaticRetargetEvaluator.from_existing_pilot(max_heavy_queries=cfg['budgets']['heavy_total'],deadline_monotonic=deadline)
    counts={'LP_calls':0,'primary_LP_attempts':0,'LP_unknown':0,'base_evaluations':0,'FD_evaluations':0,'optimizer_calls':0,'integration_training_GPU':0}
    q0=E.initial['qpos'][:20].copy();qrange=E.qhi-E.qlo
    parent=read(cfg['inputs']['old_geometry_protocol']['path'])
    pbox=np.asarray(parent['bounds']['phone_translation_palm_m'])
    scale=np.r_[np.ones(20),np.full(3,.1),np.ones(3)]
    lo=np.r_[E.qlo,pbox[0],[-np.pi]*3]/scale;hi=np.r_[E.qhi,pbox[1],[np.pi]*3]/scale
    steps=np.r_[np.full(20,cfg['finite_differences']['q_rad']),np.full(3,cfg['finite_differences']['translation_m']),np.full(3,cfg['finite_differences']['rotation_rad'])]/scale
    results=[];rows=[];iterations=[];pending_error=None
    def enroll(s):
        return E.constraint_vector(s,pad_gap_window_m=(0.,.0002),nonpad_buffer_m=0.,self_buffer_m=0.)
    def actual_support(s,geometry_ok):
        if not geometry_ok:return {'status':'GEOMETRY_NOT_ENROLLED','available':False,'LP_calls':0}
        if not s['support_operands']['available']:return {'status':'CURRENT_CLOSEST_PAIR_UNAVAILABLE','available':False,'LP_calls':0}
        if np.any(s['qfrc_passive26_Nm'][:20]!=0):return {'status':'NONZERO_PASSIVE_REQUIRES_NEW_STATIC_CONTRACT','available':False,'LP_calls':0}
        try:
            counts['primary_LP_attempts']+=1
            lp=support_residual_lp(**E.support_lp_kwargs(s),normal_cap_N=cfg['support']['normal_cap_N'],tangent_reference_axis_world=cfg['support']['tangent_reference_axis_world'])
            counts['LP_calls']+=lp['LP_calls'];lp['available']=True
            return lp
        except (ValueError,ArithmeticError) as ex:
            counts['LP_unknown']+=1
            return {'status':'SUPPORT_UNKNOWN','available':False,'error':str(ex),'LP_calls':0}
    for seed_index,seed_input in enumerate(cfg['start_input_aliases']):
        seed_start=time.perf_counter();seed_end=min(deadline,seed_start+cfg['budgets']['per_seed_seconds'])
        with np.load(cfg['inputs'][seed_input]['path'],allow_pickle=False) as z:q=z['qpos27'].copy()
        R=Rotation.from_quat(q[23:27][[1,2,3,0]]).as_matrix();xseed=np.r_[q[:20],E.Rp.T@(q[20:23]-E.pp),Rotation.from_matrix(E.Rp.T@R).as_rotvec()]
        seed_heavy=E.counts['heavy_queries_attempted'];selected=None;best_geometry=None;best_B_start=None;best_proposal=None;best_violation=None;stage_reports=[]
        for stage in ('A','B'):
            phase_start=time.perf_counter()
            if stage=='B':
                if best_B_start is None:
                    stage_reports.append({'stage':stage,'status':'NOT_RUN_NO_ENROLLED_EVALUABLE_SUPPORT_START','optimizer_calls':0});continue
                xseed=best_B_start['x'].copy()
            phase_seconds=cfg['budgets']['stage_A_seconds'] if stage=='A' else cfg['budgets']['stage_B_seconds']
            E.set_budget(deadline_monotonic=min(seed_end-cfg['budgets']['final_reserve_seconds'],phase_start+phase_seconds),max_additional_heavy_queries=max(0,min(cfg['budgets']['heavy_per_seed']-(E.counts['heavy_queries_attempted']-seed_heavy)-cfg['budgets']['heavy_final_reserve'],cfg['budgets']['heavy_total']-E.counts['heavy_queries_attempted']-cfg['budgets']['heavy_final_reserve'])))
            cache={};jac_cache={};ncallback=0;phase_best=None;last=None;last_attempted=None
            def evaluate(z,purpose='base'):
                nonlocal best_geometry,best_B_start,best_proposal,best_violation,phase_best,last,last_attempted
                key=np.asarray(z,np.float64).tobytes()
                if key in cache:return cache[key]
                x=np.asarray(z)*scale;last_attempted={'x26':x.copy(),'purpose':purpose,'elapsed_seed_s':time.perf_counter()-seed_start}
                s=E.evaluate(x[:20],x[20:23],x[23:])
                physical=enroll(s);g=E.constraint_vector(s,pad_gap_window_m=cfg['geometry']['search_pad_window_m'],nonpad_buffer_m=cfg['geometry']['nonpad_buffer_m'],self_buffer_m=cfg['geometry']['self_buffer_m'])
                lp=actual_support(s,physical['all_satisfied'])
                force_ok=bool(lp.get('feasible_force_balance',False));moment=lp.get('moment_infinity_Nm')
                recovery=lp.get('force_recovery') or {};force_recovery=recovery.get('force_infinity_N')
                arm=s['support_operands']['object_COM']-s['fixed_material_points_world_m'][0];axis=s['fixed_material_line_axis_world'];gravity=s['support_operands']['gravity_world']
                arm_perp=arm-axis*(axis@arm);gperp=gravity-axis*(axis@gravity);fixed_below=float(arm_perp@gperp)
                actual_below=None;actual_line_status='CURRENT_PAIR_UNAVAILABLE';actual_line_length=None
                if s['support_operands']['available']:
                    pts=s['support_operands']['contact_points_world'];actual_axis=pts[1]-pts[0];actual_line_length=float(np.linalg.norm(actual_axis))
                    if np.isfinite(actual_line_length) and actual_line_length>1e-9:
                        actual_axis/=actual_line_length;actual_arm=s['support_operands']['object_COM']-pts[0]
                        actual_below=float((actual_arm-actual_axis*(actual_axis@actual_arm))@(gravity-actual_axis*(actual_axis@gravity)));actual_line_status='DEFINED_CURRENT_PAIR_LINE'
                    else:actual_line_status='DEGENERATE_CURRENT_PAIR_LINE_NOT_QUALIFIABLE'
                below=actual_below if stage=='B' and actual_below is not None else fixed_below
                blocks={
                    'fixed_contact_anchor':((s['anchors_world_m']-s['anchor_targets_world_m'])/.005).ravel(),
                    'candidate_tip_style':(.15*(s['tips_world_m'][:2]-s['human_tip_targets_world_m'][:2])/.01).ravel(),
                    'noncontact_tip_style':(.05*(s['tips_world_m'][2:]-s['human_tip_targets_world_m'][2:])/.02).ravel(),
                    'joint_regularizer':.01*(x[:20]-q0)/qrange,
                    'fixed_material_gravity_axis':np.array([s['fixed_material_line_gravity_torque_Nm']/cfg['objective']['gravity_moment_scale_Nm']]),
                    'unstable_COM_side':np.array([min(0.,below)/cfg['objective']['below_dot_scale_m2_s2']]),
                }
                if stage=='B':
                    blocks['current_support_moment']=np.array([moment/cfg['objective']['gravity_moment_scale_Nm'] if moment is not None else 1.])
                    blocks['current_force_recovery']=np.array([0. if force_ok else (force_recovery if force_recovery is not None else 1.)/cfg['objective']['force_recovery_scale_N']])
                objective=float(sum(v@v for v in blocks.values()));c=np.r_[g['values_m']/.001,below/cfg['objective']['below_dot_scale_m2_s2']]
                if stage=='B':
                    # The primary moment constraint is undefined until the
                    # force-hard LP exists. Relax only that search row during
                    # explicit force recovery; final qualification never does.
                    if force_ok:force_c=1.
                    elif force_recovery is not None:force_c=-max(float(force_recovery),1e-6)
                    elif not physical['all_satisfied']:force_c=min(-1e-6,float(np.min(g['values_m']/.001)))
                    else:force_c=-1.  # genuine unknown, fail closed
                    moment_c=(cfg['support']['proposal_moment_limit_Nm']-moment)/cfg['objective']['gravity_moment_scale_Nm'] if moment is not None else 1.
                    c=np.r_[c,force_c,moment_c]
                proposal=bool(physical['all_satisfied'] and force_ok and lp.get('proposal_gate',False) and actual_below is not None and actual_below>=0)
                row={'seed':seed_index,'stage':stage,'purpose':purpose,'x26':x,'objective':objective,'blocks':{k:float(v@v) for k,v in blocks.items()},'max_scaled_constraint_violation':float(np.maximum(-c,0).max()),'search_constraints_pass':bool(np.all(c>=0)),'moment_search_constraint_active':stage=='B' and moment is not None,'physical_geometry_pass':physical['all_satisfied'],'below_line_dot_m2_s2':below,'fixed_material_below_line_dot_m2_s2':fixed_below,'actual_below_line_dot_m2_s2':actual_below,'fixed_material_axis_torque_Nm':s['fixed_material_line_gravity_torque_Nm'],'current_support_status':lp['status'],'force_balance_feasible':force_ok,'moment_infinity_Nm':moment,'force_recovery_infinity_N':force_recovery,'support_proposal_pass':proposal,'pad_gaps_m':[v['exact_gap_m'] for v in s['phone_geometry_rows'] if v['is_pad']],'tip_errors_m':np.linalg.norm(s['tips_world_m']-s['human_tip_targets_world_m'],axis=1),'anchor_errors_m':np.linalg.norm(s['anchors_world_m']-s['anchor_targets_world_m'],axis=1),'elapsed_seed_s':time.perf_counter()-seed_start,'counts':dict(E.counts)}
                row.update(actual_contact_line_status=actual_line_status,actual_contact_line_length_m=actual_line_length)
                context={'x':x.copy(),'qpos27':s['qpos27'].copy(),'row':row,'constraints':c.copy(),'objective':objective,'proposal':proposal,'support':lp}
                cache[key]=context;rows.append(row);last=context
                counts['FD_evaluations' if purpose=='finite_difference' else 'base_evaluations']+=1
                if purpose=='base':
                    if physical['all_satisfied'] and (best_geometry is None or (abs(row['fixed_material_axis_torque_Nm']),objective)<(abs(best_geometry['row']['fixed_material_axis_torque_Nm']),best_geometry['objective'])):best_geometry=context
                    finite_support=physical['all_satisfied'] and (moment is not None or force_recovery is not None)
                    if finite_support:
                        score=(not force_ok,actual_below is None or actual_below<0,moment if moment is not None else force_recovery*cfg['objective']['gravity_moment_scale_Nm'],abs(row['fixed_material_axis_torque_Nm']),objective)
                        context['B_start_score']=score
                        if best_B_start is None or score<best_B_start['B_start_score']:best_B_start=context
                    if proposal and (best_proposal is None or objective<best_proposal['objective']):best_proposal=context
                    if best_violation is None or (row['max_scaled_constraint_violation'],objective)<(best_violation['row']['max_scaled_constraint_violation'],best_violation['objective']):best_violation=context
                    if np.all(c>=0) and (phase_best is None or objective<phase_best['objective']):phase_best=context
                return context
            def derivatives(z):
                key=np.asarray(z,np.float64).tobytes()
                if key in jac_cache:return jac_cache[key]
                base=evaluate(z);grad=np.empty(26);J=np.empty((len(base['constraints']),26))
                for k,h in enumerate(steps):
                    signed=h if z[k]+h<=hi[k] else -h;zz=np.array(z).copy();zz[k]+=signed;probe=evaluate(zz,'finite_difference')
                    grad[k]=(probe['objective']-base['objective'])/signed;J[:,k]=(probe['constraints']-base['constraints'])/signed
                jac_cache[key]=(grad,J);return grad,J
            def callback(z):
                nonlocal ncallback
                ncallback+=1;r=evaluate(z)['row'];iterations.append({'seed':seed_index,'stage':stage,'iteration':ncallback,**r})
            solver=None;status='NOT_STARTED';error=None
            try:
                zseed=xseed/scale;evaluate(zseed);counts['optimizer_calls']+=1
                fit=minimize(lambda z:evaluate(z)['objective'],zseed,method='SLSQP',jac=lambda z:derivatives(z)[0],bounds=list(zip(lo,hi)),constraints=[{'type':'ineq','fun':lambda z:evaluate(z)['constraints'],'jac':lambda z:derivatives(z)[1]}],callback=callback,options={'maxiter':cfg['solver']['maxiter'],'ftol':cfg['solver']['ftol'],'disp':False})
                status='SOLVER_RETURNED';solver={'status':int(fit.status),'success':bool(fit.success),'message':str(fit.message),'nit':int(fit.nit),'nfev':int(fit.nfev),'njev':int(fit.njev)};evaluate(fit.x)
            except (StaticBudgetExceeded,UnknownGeometry,ValueError,ArithmeticError) as ex:
                status='BOUNDED_STOP_OR_UNKNOWN_PRESERVED';error=repr(ex)
            stage_reports.append({'stage':stage,'status':status,'error':error,'solver':solver,'callback_iterations':ncallback,'complete_evaluations':len(cache),'last_complete':None if last is None else last['row'],'failed_attempted_state_parameters':last_attempted if error is not None else None,'failed_attempt_not_a_valid_pose_capture':error is not None,'elapsed_s':time.perf_counter()-phase_start})
            print(json.dumps(plain({'seed':seed_index,'stage':stage,'status':status,'iterations':ncallback,'best_geometry_axis_Nm':None if best_geometry is None else best_geometry['row']['fixed_material_axis_torque_Nm'],'proposal':best_proposal is not None,'counts':E.counts})),flush=True)
        selected=best_proposal if best_proposal is not None else best_B_start if best_B_start is not None else best_geometry if best_geometry is not None else best_violation
        final={'seed':seed_index,'start_input':cfg['inputs'][seed_input],'stages':stage_reports,'selection':'BEST_SUPPORT_PROPOSAL' if best_proposal else 'BEST_EVALUABLE_SUPPORT' if best_B_start else 'BEST_GEOMETRY_BY_MATERIAL_AXIS_TORQUE' if best_geometry else 'BEST_VIOLATION','candidate_qualified_for_bounded_holding_diagnostic':False,'unknown':[]}
        if selected is not None:
            E.set_budget(deadline_monotonic=seed_end,max_additional_heavy_queries=min(cfg['budgets']['heavy_final_reserve'],max(0,cfg['budgets']['heavy_total']-E.counts['heavy_queries_attempted'])))
            x=selected['x'];final['selected_evaluation']=selected['row']
            try:
                s=E.evaluate(x[:20],x[20:23],x[23:],exact_all=False)
                # All phone pairs exact. Far self-pairs retain valid separating
                # AABB certificates instead of forcing needless huge hulls.
                for row in s['phone_geometry_rows']:
                    if row['status']=='DISJOINT_AABB_CERTIFICATE':
                        row.update(E._describe_pair({'kind':'convex','v':E.phone_vertices},E._shape(row['geom_id'],s['phone_R_world'],s['phone_p_world_m']),phone=True,force_exact=True))
                        row.pop('not_exact_gap',None)
                physical=enroll(s);lp=actual_support(s,physical['all_satisfied'])
                s['counts']=dict(E.counts)
                final['verification_state']=s;final['support_LP']=lp;final['physical_geometry_pass']=physical['all_satisfied']
                np.savez_compressed(out/f'seed{seed_index}_selected_state.npz',qpos27=s['qpos27'],qvel26=np.zeros(26),ctrl20=np.zeros(20),x26=x,geom_xpos=E.d.geom_xpos.copy(),geom_xmat=E.d.geom_xmat.copy(),site_xpos=E.d.site_xpos.copy(),body_xpos=E.d.xpos.copy(),body_xquat=E.d.xquat.copy(),phone_R_world=s['phone_R_world'],phone_p_world=s['phone_p_world_m'])
                final['selected_state']=pin(out/f'seed{seed_index}_selected_state.npz')
                if lp.get('tau_act_required_Nm') is not None:
                    tau=np.asarray(lp['tau_act_required_Nm']);ctrl=(tau-s['PD_offset20'])/s['PD_gain20'];final['proposed_ctrl20']=ctrl;final['proposed_tau20_Nm']=tau
                    reachable=bool(np.isfinite(ctrl).all() and np.all(ctrl>=E.m.actuator_ctrlrange[:,0]) and np.all(ctrl<=E.m.actuator_ctrlrange[:,1]) and np.max(np.abs(s['PD_gain20']*ctrl+s['PD_offset20']-tau))<=1e-12)
                    final['proposed_ctrl_reachable_without_clamping']=reachable
                    final['candidate_qualified_for_bounded_holding_diagnostic']=bool(physical['all_satisfied'] and lp.get('proposal_gate',False) and selected['row']['actual_below_line_dot_m2_s2'] is not None and selected['row']['actual_below_line_dot_m2_s2']>=0 and reachable and not final['unknown'])
            except (StaticBudgetExceeded,UnknownGeometry,ValueError,ArithmeticError) as ex:
                final['unknown'].append(repr(ex))
                raw_path=out/f'seed{seed_index}_unverified_selection.npz'
                np.savez_compressed(raw_path,qpos27=selected['qpos27'],qvel26=np.zeros(26),ctrl20=np.zeros(20),x26=x)
                final['unverified_selected_raw_state']=pin(raw_path)
        final['elapsed_s']=time.perf_counter()-seed_start;results.append(final);write(out/f'seed{seed_index}_result.json',final)
    for name,data in [('all_evaluations.jsonl',rows),('actual_iterations.jsonl',iterations)]:
        with (out/name).open('x',encoding='utf8') as f:
            for row in data:f.write(json.dumps(plain(row),allow_nan=False)+'\n')
    report={'schema':'fixed_wrist_gravity_contact_pose_result:3','status':'COMPLETED_BOUNDED_STATIC_SUPPORT_POSE_EXPLORATION','protocol':pin(args.protocol),'seed_results':results,'counts':counts,'static_mechanics_geometry_counts':E.counts,'unchanged_model_digest':E.parameter_digest,'elapsed_s':time.perf_counter()-start,'qualified_support_proposal_count':sum(c['candidate_qualified_for_bounded_holding_diagnostic'] for c in results),'holding_or_exact_statics_or_ego_policy_qualified':False,'scope':'Original fixed wrist/scale/robot/material; same two original starts; all failed phases and hard-bound unknowns retained. Small-moment support proposal only. No neural model, training, physical integration or GPU.'}
    write(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json'),'proposal_count':report['qualified_support_proposal_count'],'seconds':report['elapsed_s']}),flush=True)


if __name__=='__main__':main()
