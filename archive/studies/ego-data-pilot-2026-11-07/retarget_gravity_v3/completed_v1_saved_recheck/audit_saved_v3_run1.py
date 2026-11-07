"""Saved-only run1 protocol, counters, LP arithmetic and proposal-gate review."""
import collections,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent;P=ROOT.parent;RUN=ROOT/'completed_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def checked(x):
    if pin(x['path'])!=x:raise ValueError('Changed pin '+x['path'])
    return Path(x['path'])
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))
def cross(a,b):return np.array([math.fsum([a[1]*b[2],-a[2]*b[1]]),math.fsum([a[2]*b[0],-a[0]*b[2]]),math.fsum([a[0]*b[1],-a[1]*b[0]])])
def close(a,b,tol=1e-10):return bool(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)
def need(ok,label):
    if not ok:raise AssertionError(label)
def below(COM,points,g):
    x=np.asarray(points);u=x[1]-x[0];u/=np.linalg.norm(u);r=np.asarray(COM)-x[0];g=np.asarray(g)
    return dot(r-u*dot(u,r),g-u*dot(u,g))


def main():
    need(not (OUT/'report.json').exists(),'fresh output')
    producer=RUN/'report.json';need(pin(producer)['sha256']=='2a44331dc6aadc0a24f24583126dcb05e212affac9cab9bf34ad92e740dd70dc','expected actual report')
    R=read(producer);C=read(checked(R['protocol']))
    for record in [*C['inputs'].values(),*C['source_pins'].values()]:checked(record)
    need(read(RUN/'protocol.json')==C,'copied protocol semantic identity')
    need((RUN/'execution_source.py').read_bytes()==checked(C['source_pins']['runner']).read_bytes(),'executed source bytes')
    T=read(ROOT/'run1_process_terminal.json');need(T['returncode']==0,'terminal exit0');checked(T['stdout']);checked(T['stderr']);need(T['stderr']['bytes']==0,'empty process stderr')
    rows=[json.loads(x) for x in (RUN/'all_evaluations.jsonl').read_text().splitlines()]
    iters=[json.loads(x) for x in (RUN/'actual_iterations.jsonl').read_text().splitlines()]
    checks={'protocol_input_source_current_pins':True,'executed_source_protocol_terminal':True}
    need(sum(x['purpose']=='base' for x in rows)==R['counts']['base_evaluations'],'base count')
    need(sum(x['purpose']=='finite_difference' for x in rows)==R['counts']['FD_evaluations'],'FD count')
    numerical=[x for x in rows if x['current_support_status']=='PRIMARY_NUMERICAL_HARD_BOUND_FAILURE']
    status_counts=dict(collections.Counter(x['current_support_status'] for x in rows))
    need(all(np.isfinite(x['x26']).all() and len(x['x26'])==26 for x in rows),'raw finite26 parameters')
    for x in rows:
        moment=x['moment_infinity_Nm'];fok=x['force_balance_feasible'];rec=x['force_recovery_infinity_N']
        expected=x['physical_geometry_pass'] and fok and moment is not None and moment<=1e-4 and x['actual_below_line_dot_m2_s2'] is not None and x['actual_below_line_dot_m2_s2']>=0
        need(x['support_proposal_pass']==expected,'row proposal definition')
        need(x['search_constraints_pass']==(x['max_scaled_constraint_violation']==0),'search flag vs violation')
        need(x['moment_search_constraint_active']==(x['stage']=='B' and moment is not None),'moment inactive label')
        if x['stage']=='B' and not fok:need(not x['search_constraints_pass'],'recovery never search pass')
        if x['current_support_status']=='PRIMARY_NUMERICAL_HARD_BOUND_FAILURE':need(not x['support_proposal_pass'],'numerical hard-bound failure not proposal')
    checks['all_saved_row_proposal_and_recovery_logic']=True
    support_parent=read(checked(C['inputs']['support_parent']))
    old=checked(support_parent['outputs']['support_operands_and_controls.npz'])
    with np.load(old,allow_pickle=False) as z:O={k:z[k].copy() for k in z.files}
    results=[];full_LP_arithmetic=[];stage_summary=[];finals=0
    for S in R['seed_results']:
        seed=S['seed'];selected=S.get('selected_evaluation');state=S.get('verification_state')
        for st in S['stages']:
            group=[x for x in rows if x['seed']==seed and x['stage']==st['stage']]
            callbacks=[x for x in iters if x['seed']==seed and x['stage']==st['stage']]
            need(len(group)==st.get('complete_evaluations',0),'per-stage complete count')
            need(len(callbacks)==st.get('callback_iterations',0),'callback count')
            need([x['iteration'] for x in callbacks]==list(range(1,len(callbacks)+1)),'iteration order')
            if st.get('error'):
                need(st['failed_attempt_not_a_valid_pose_capture'] and st['failed_attempted_state_parameters'] is not None,'failed raw attempt preservation')
            stage_summary.append({'seed':seed,'stage':st['stage'],'status':st['status'],'iterations':len(callbacks),'complete_rows':len(group),'error':st.get('error')})
        if selected:need(selected in rows and selected['purpose']=='base','selected actual base row retained')
        need(not S['candidate_qualified_for_bounded_holding_diagnostic'],'known zero candidate run')
        item={'seed':seed,'selection':S['selection'],'selected_stage':selected['stage'] if selected else None,'selected_status':selected['current_support_status'] if selected else None,
              'selected_numerical_hard_bound_failure':bool(selected and selected['current_support_status']=='PRIMARY_NUMERICAL_HARD_BOUND_FAILURE'),'candidate_qualified':False,'unknown':S['unknown']}
        if state:
            finals+=1;checked(S['selected_state'])
            with np.load(S['selected_state']['path'],allow_pickle=False) as z:need(np.array_equal(z['qpos27'],np.array(state['qpos27'])),'saved finalqpos')
            need(len(state['phone_geometry_rows'])==22 and len(state['self_geometry_rows'])==189,'complete geometry rows')
            physical=not state['unknown']
            for g in state['phone_geometry_rows']:
                need(g['status']=='EXACT_NUMERICAL_IDEAL_CONVEX','final all phone exact')
                need(not g.get('not_exact_gap',False),'no stale AABB label')
                need(close(g['lower_bound_m'],g['exact_gap_m']-g['guard_m']),'query lower bound arithmetic')
                physical=physical and (g['exact_gap_m']-g['guard_m'])>=0
                if g['is_pad']:physical=physical and (g['exact_gap_m']+g['guard_m'])<=.0002
            for g in state['self_geometry_rows']:
                need(g['status'] in ['EXACT_NUMERICAL_IDEAL_CONVEX','DISJOINT_AABB_CERTIFICATE'],'self status')
                physical=physical and g['lower_bound_m']>=0
            need(bool(physical)==S['physical_geometry_pass'],'final saved geometry gate arithmetic')
            item['physical_geometry_pass']=bool(physical)
            LP=S['support_LP'];item['LP_status']=LP['status']
            if LP.get('lambda_normal_N') is not None:
                op=state['support_operands'];n=np.array(op['normals_force_on_object']);n/=np.linalg.norm(n,axis=1)[:,None]
                points=np.array(op['contact_points_world']);COM=np.array(op['object_COM']);J=np.array(op['J_robot_contact_3x20']);bias=np.array(op['robot_gravity_bias_tau']);g=np.array(op['gravity_world'])
                need(close(points,LP['contact_points_world_m']) and close(n,LP['normals_force_on_object']),'LP input binding')
                need(close(LP['mu_actual'],[.7,.7]) and close(LP['mu_design'],[.7/math.sqrt(2)]*2) and LP['normal_cap_N']==[3.,3.],'fixed friction/cap budget')
                for i,ct in enumerate(state['contacts']):
                    pair=ct['pairing']['closest_pair'];feat=ct['features'];Ps=np.array(feat['object_support_points_world_m']);Hs=np.array(feat['hand_support_points_world_m'])
                    pw=np.array(pair['object_weights']);hw=np.array(pair['hand_weights'])
                    pp=np.array([dot(pw,Ps[:,j]) for j in range(3)]);hh=np.array([dot(hw,Hs[:,j]) for j in range(3)])
                    need(pair['rank']==pair['unknown_count'] and ct['pairing']['status']=='UNIQUE_CLOSEST_PAIR','unique actual witnesses')
                    need(close(pp,points[i]) and close(hh,pair['hand_point_world_m']),'support barycentric points')
                    need(close(hh-pp,np.array(feat['normal_world'])*feat['signed_gap_m']),'full3 witness pairing')
                    need(close(n[i],-np.array(feat['normal_world'])) and close(J[i],np.array(ct['actual_hand_point_J3x26'])[:,:20]),'force normal/J paired-hand binding')
                axis=np.array([0.,1,0]);t1=np.array([axis-nn*dot(nn,axis) for nn in n]);t1/=np.linalg.norm(t1,axis=1)[:,None];t2=np.array([cross(nn,tt) for nn,tt in zip(n,t1)])
                rays=np.array([[nn+.7/math.sqrt(2)*(math.cos(k*math.pi/4)*a+math.sin(k*math.pi/4)*b) for k in range(8)] for nn,a,b in zip(n,t1,t2)])
                need(close(rays,LP['rays_force_world_per_normal_N']),'independent ray reconstruction')
                lam=np.array(LP['lambda_normal_N']);forces=np.array([[math.fsum(lam[i,k]*rays[i,k,c] for k in range(8)) for c in range(3)] for i in range(2)])
                force_res=np.array([math.fsum(forces[:,c])+op['mass']*g[c] for c in range(3)])
                moments=np.array([cross(p-COM,f) for p,f in zip(points,forces)]);moment=np.array([math.fsum(moments[:,c]) for c in range(3)])
                tau=np.array([bias[j]+math.fsum(J[i,c,j]*forces[i,c] for i in range(2) for c in range(3)) for j in range(20)])
                q=np.array(state['q20_rad']);offset=O['biasprm'][:,0]+O['biasprm'][:,1]*q
                lo=np.maximum.reduce([O['gain20']*O['ctrlrange'][:,0]+offset,O['actuator_forcerange'][:,0],O['joint_actfrcrange'][:,0]])
                hi=np.minimum.reduce([O['gain20']*O['ctrlrange'][:,1]+offset,O['actuator_forcerange'][:,1],O['joint_actfrcrange'][:,1]])
                fn=np.array([math.fsum(l) for l in lam]);defect=max(abs(moment));fvalid=max(abs(force_res))<=1e-8 and lam.min()>=-1e-9 and fn.max()<=3+1e-9 and min(tau-lo)>=-1e-9 and min(hi-tau)>=-1e-9
                for a,b,label in [(forces,LP['force_world_N'],'forces'),(force_res,LP['force_balance_residual_world_N'],'force residual'),(moment,LP['moment_residual_world_Nm'],'moment'),(tau,LP['tau_act_required_Nm'],'tau'),(fn,LP['normal_force_N'],'normal'),(lo,op['torque_lower'],'actual lower'),(hi,op['torque_upper'],'actual upper')]:need(close(a,b),label)
                need(close(defect,LP['moment_infinity_Nm']) and bool(fvalid)==LP['feasible_force_balance'],'LP hard gates')
                need(LP['solver_slack_objective_t_Nm']>=-1e-9 and defect<=LP['solver_slack_objective_t_Nm']+1e-9,'LP moment slack feasibility')
                need(LP['proposal_gate']==bool(fvalid and defect<=1e-4),'LP proposal gate')
                act_below=below(COM,points,g);need(close(act_below,selected['actual_below_line_dot_m2_s2']),'actual COMside recheck')
                ctrl=(tau-offset)/O['gain20'];need(close(ctrl,S['proposed_ctrl20']),'true ctrl inverse')
                reachable=bool(np.isfinite(ctrl).all() and np.all(ctrl>=O['ctrlrange'][:,0]) and np.all(ctrl<=O['ctrlrange'][:,1]))
                need(reachable==S['proposed_ctrl_reachable_without_clamping'],'ctrl reachability')
                qualified=bool(physical and LP['proposal_gate'] and act_below>=0 and reachable and not S['unknown'])
                need(qualified==S['candidate_qualified_for_bounded_holding_diagnostic'],'final full gate')
                rec={'seed':seed,'force_balance_feasible':bool(fvalid),'force_residual_max_N':float(max(abs(force_res))),'moment_world_Nm':moment.tolist(),'moment_infinity_Nm':float(defect),'normal_force_N':fn.tolist(),
                    'minimum_torque_slack_Nm':float(min(min(tau-lo),min(hi-tau))),'actual_below_dot_m2_s2':act_below,'proposal_gate':LP['proposal_gate'],'lambda_checked':True,'optimality_independently_certified':False}
                full_LP_arithmetic.append(rec);item.update(rec)
        results.append(item)
    checks['stage_counts_base_selection_and_raw_failures']=True
    checks['selected_saved_LP_rays_forces_moment_torque_controls']=True
    checks['all_final_geometry_and_proposal_gates_recomputed']=True
    need(R['qualified_support_proposal_count']==0 and not R['holding_or_exact_statics_or_ego_policy_qualified'],'zero proposals/no physics qualification')
    D=R['static_mechanics_geometry_counts'];N=D['evaluations']
    need(all(D[k]==N for k in ['mj_kinematics','mj_comPos','mj_comVel','mj_rne','mj_passive','mj_transmission']),'static pipeline counters')
    need(D['anchor_mj_jac']==2*N and D['actual_pair_mj_jac']==D['closest_pair_unique'],'Jacobian counter identities')
    need(D['heavy_queries_attempted']==D['heavy_queries_returned'] and D['unknown_queries']==0,'query completion')
    nonfinal_extra=N-len(rows)-finals
    stopped=sum(bool(st.get('error')) for s in R['seed_results'] for st in s['stages'])
    need(nonfinal_extra==stopped,'partial native states consistent with stopped phases')
    need(sum(s.get('optimizer_calls',1) if 'callback_iterations' in s else 0 for t in R['seed_results'] for s in t['stages'])==R['counts']['optimizer_calls'],'optimizer call counts')
    checks['native_counter_algebra_and_partial_stops']=True
    def candidate(x):
        gaps=x['pad_gaps_m'];side=x['actual_below_line_dot_m2_s2']
        return side is not None and side>=0 and all(g is not None and g>=0 for g in gaps)
    stable=[x for x in rows if candidate(x)]
    base=[x for x in stable if x['purpose']=='base'];callback_candidates=[x for x in iters if candidate(x)]
    def shortlist(xs):
        return sorted(xs,key=lambda x:(max(max(g-.0002,0) for g in x['pad_gaps_m']),not x['physical_geometry_pass'],x['moment_infinity_Nm'] if x['moment_infinity_Nm'] is not None else 1e99,x['max_scaled_constraint_violation']))[:6]
    polish={'scope':'No stable-holding claim. Below-side is only a necessary static descriptor. Positive pad gaps near200um alone do not prove the remaining physical geometry; only already saved rows, no new evaluation.',
      'all_below_side_nonnegative_pad_rows':len(stable),'base_below_side_nonnegative_pad_rows':len(base),
      'base_original_window_and_full_geometry_below_rows':sum(x['physical_geometry_pass'] and max(x['pad_gaps_m'])<=.0002 for x in base),
      'best_existing_base_rows_by_upper_window_overshoot':shortlist(base),'best_callback_rows_by_upper_window_overshoot':shortlist(callback_candidates)}
    save(OUT/'polish_candidates_saved_only.json',polish)
    with (OUT/'base_candidates_summary.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','stage','actual_below_dot','thumb_gap_m','index_gap_m','upper_200um_overshoot_m','full_physical_geometry_pass','force_feasible','moment_inf_Nm','support_proposal'])
        for x in base:w.writerow([x['seed'],x['stage'],x['actual_below_line_dot_m2_s2'],*x['pad_gaps_m'],max(max(g-.0002,0) for g in x['pad_gaps_m']),x['physical_geometry_pass'],x['force_balance_feasible'],x['moment_infinity_Nm'],x['support_proposal_pass']])
    checks['no_model_FK_LP_solve_or_NN_import']=not any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp','scipy.optimize'])
    report={'schema':'gravity_contact_v3_run1_saved_independent_review:1','status':'PASS_SAVED_ARITHMETIC_ZERO_SUPPORT_PROPOSALS',
      'source':pin(__file__),'producer_report':pin(producer),'protocol':R['protocol'],'terminal':pin(ROOT/'run1_process_terminal.json'),
      'checks':checks,'passed':sum(checks.values()),'total':len(checks),'stage_summary':stage_summary,'complete_rows':len(rows),'iterations':len(iters),
      'row_support_status_counts':status_counts,'PRIMARY_NUMERICAL_HARD_BOUND_FAILURE':{'count':len(numerical),'selected_count':sum(x['selected_numerical_hard_bound_failure'] for x in results),'rows':numerical},
      'selected_results':results,'full_saved_LP_arithmetic_count':len(full_LP_arithmetic),'counters':R['counts'],'static_counts':D,'partial_native_states_not_complete_rows':nonfinal_extra,
      'qualified_proposal_count':0,'holding_qualified':False,
      'scope':'Saved arithmetic/source pins only, no LP re-solve/FK/native model. Intermediate LP lambda/dual solutions are not retained in all_evaluations, so only the final complete saved LP is independently recomputed; intermediate statuses/gates/counters are consistency checks. Convex-query geometry distances and actual Jacobians are source/prior-contract evidence; this job recomputes their saved inequality arithmetic, not a new geometric/native qualification. LP optimum is not independently certified without duals.',
      'outputs':{f.name:pin(f) for f in OUT.iterdir() if f.is_file() and f.name!='report.json'}}
    save(OUT/'report.json',report);print(json.dumps({'report':pin(OUT/'report.json'),'checks':f"{report['passed']}/{report['total']}",'numerical_failure_count':len(numerical),'polish_base_rows':len(base)}))


if __name__=='__main__':main()
