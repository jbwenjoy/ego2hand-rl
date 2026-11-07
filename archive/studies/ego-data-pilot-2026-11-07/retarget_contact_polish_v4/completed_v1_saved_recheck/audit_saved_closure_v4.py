"""Pure saved-data arithmetic, no model/FK/LP solve/optimizer imports."""
from pathlib import Path
import csv,hashlib,json,math,sys
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;P=ROOT.parent;RUN=ROOT/'completed_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def checked(x):
    if pin(x['path'])!=x:raise ValueError('Changed pin '+x['path'])
    return Path(x['path'])
def need(x,label):
    if not x:raise AssertionError(label)
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))
def cross(a,b):return np.array([math.fsum([a[1]*b[2],-a[2]*b[1]]),math.fsum([a[2]*b[0],-a[0]*b[2]]),math.fsum([a[0]*b[1],-a[1]*b[0]])])
def close(a,b,tol=1e-10):return bool(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)
def Rquat(q):
    q=np.array(q,dtype=float);q/=np.linalg.norm(q);w=q[0];u=q[1:]
    return np.array([e+2*w*cross(u,e)+2*cross(u,cross(u,e)) for e in np.eye(3)]).T


def main():
    need(not (OUT/'report.json').exists(),'fresh assessment')
    rf=RUN/'report.json';need(pin(rf)['sha256']=='bf30f9372ead185d02a864a8d895a07392b1bd6ca62ecd9866710cca783176ff','expected immutable report')
    r=read(rf);c=read(checked(r['protocol']))
    for x in [*c['inputs'].values(),*c['source_pins'].values()]:checked(x)
    need(read(RUN/'protocol.json')==c,'protocol semantic copy')
    need((RUN/'execution_source.py').read_bytes()==checked(c['source_pins']['runner']).read_bytes(),'exact executed source')
    terminal=read(ROOT/'run1_process_terminal.json');need(terminal['returncode']==0,'terminal exit0');checked(terminal['stdout']);checked(terminal['stderr']);need(terminal['stderr']['bytes']==0,'empty process stderr')
    rows=[json.loads(x) for x in (RUN/'all_evaluations.jsonl').read_text().splitlines()]
    callbacks=[json.loads(x) for x in (RUN/'actual_iterations.jsonl').read_text().splitlines()]
    need(sum(x['purpose']=='base' for x in rows)==r['base_evaluations'] and sum(x['purpose']=='finite_difference' for x in rows)==r['FD_evaluations'],'base/FD global count')
    parent=P/'hand2_holding_v1/support_preparation_v1/actual_v1/report.json';parent_report=read(parent)
    with np.load(checked(parent_report['outputs']['support_operands_and_controls.npz']),allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    interface=P/'hand2_stage1/completed_v2/joint_actuator_interface.json';names=read(interface)['joint_order']
    checks={'all_protocol_source_model_byte_pins':True,'executed_copy_and_terminal_evidence':True,'base_FD_counts':True}
    cases=[];PD=[];fullq=[];totalLP=0;finals=0
    for S in r['seed_results']:
        seed=S['seed'];seedrows=[x for x in rows if x['seed']==seed];cb=[x for x in callbacks if x['seed']==seed]
        need(len(seedrows)==S['complete_evaluations'] and len(cb)==S['iterations'],'seed count')
        need([x['iteration'] for x in cb]==list(range(1,len(cb)+1)),'callback numbering')
        start=checked(S['source_start'])
        with np.load(start,allow_pickle=False) as z:qstart=z['qpos27'].copy()
        need(S['source_start']==c['inputs'][c['start_input_aliases'][seed]],'start alias identity')
        p0=qstart[20:23];R0=Rquat(qstart[23:27])
        for x in seedrows:
            need(len(x['q20'])==20 and np.isfinite(x['q20']).all(),'finite q-only rows')
            need(not any(k in x for k in ['phone_translation_palm_m','phone_rotvec_rad','x26']),'q-only recorded parameter space')
            q=np.r_[x['q20'],qstart[20:27]];fullq.append(q)
            need(np.array_equal(q[20:27],qstart[20:27]),'reconstructed row fixed phone')
        selected=S['selected_evaluation'];need(selected in seedrows and selected['purpose']=='base','selected actual base')
        state=S['verification_state'];finals+=1
        need(close(state['phone_p_world_m'],p0,1e-15) and close(state['phone_R_world'],R0,1e-15),'final fixed phone position/orientation')
        need(close(np.array(state['qpos27'])[20:23],p0,1e-15) and close(Rquat(np.array(state['qpos27'])[23:27]),R0,1e-15),'final qpos fixed phone')
        with np.load(checked(S['selected_state']),allow_pickle=False) as z:
            need(np.array_equal(z['qpos27'],np.array(state['qpos27'])) and np.array_equal(z['qpos27'][:20],np.array(selected['q20'])),'final raw qpos source relation')
            need(not np.any(z['ctrl20']),'saved state zero ctrl is placeholder, not enrolled controller')
        phone=state['phone_geometry_rows'];selfrows=state['self_geometry_rows'];need(len(phone)==22 and len(selfrows)==189,'complete22/189')
        physical=not state['unknown'];gaps=[]
        for g in phone:
            need(g['status']=='EXACT_NUMERICAL_IDEAL_CONVEX' and not g.get('not_exact_gap',False),'all phone exact final')
            lb=g['exact_gap_m']-g['guard_m'];ub=g['exact_gap_m']+g['guard_m']
            need(close(lb,g['lower_bound_m']) and close(ub,g['upper_bound_m']),'gap bounds arithmetic')
            physical=physical and lb>=0
            if g['is_pad']:physical=physical and ub<=.0002;gaps.append(g['exact_gap_m'])
        for g in selfrows:
            need(g['status'] in ['DISJOINT_AABB_CERTIFICATE','EXACT_NUMERICAL_IDEAL_CONVEX'],'self query classification')
            physical=physical and g['lower_bound_m']>=0
        need(bool(physical)==S['physical_geometry_pass'],'geometry qualification arithmetic')
        need(physical,'both known geometry passes')
        LP=S['support_LP'];totalLP+=LP['LP_calls'];need(not LP['feasible_force_balance'] and not LP['proposal_gate'] and LP['lambda_normal_N'] is None,'primary failure remains separate')
        need(LP['status']=='PRIMARY_FORCE_HARD_LP_INFEASIBLE' and LP['solver_status']==2,'reported force-hard infeasibility')
        V=LP['force_recovery'];need(V['status']=='FINITE_FORCE_RECOVERY_GUIDANCE_ONLY' and not V['qualified'] and not V['feasible_force_balance'],'recovery unqualified')
        op=state['support_operands'];n=np.array(op['normals_force_on_object']);n/=np.linalg.norm(n,axis=1)[:,None]
        pts=np.array(op['contact_points_world']);COM=np.array(op['object_COM']);J=np.array(op['J_robot_contact_3x20']);bias=np.array(op['robot_gravity_bias_tau']);gravity=np.array(op['gravity_world'])
        t1=np.array([np.array([0.,1,0])-nn*nn[1] for nn in n]);t1/=np.linalg.norm(t1,axis=1)[:,None];t2=np.array([cross(nn,a) for nn,a in zip(n,t1)])
        rays=np.array([[nn+.7/math.sqrt(2)*(math.cos(k*math.pi/4)*a+math.sin(k*math.pi/4)*b) for k in range(8)] for nn,a,b in zip(n,t1,t2)])
        need(close(rays,LP['rays_force_world_per_normal_N']) and LP['normal_cap_N']==[3.,3.] and close(LP['mu_actual'],[.7,.7]),'fixed cones and cap')
        lam=np.array(V['lambda_normal_N']);forces=np.array([[math.fsum(lam[i,k]*rays[i,k,j] for k in range(8)) for j in range(3)] for i in range(2)])
        fn=np.array([math.fsum(x) for x in lam]);fr=np.array([math.fsum(forces[:,j])+op['mass']*gravity[j] for j in range(3)])
        moments=np.array([cross(point-COM,f) for point,f in zip(pts,forces)]);moment=np.array([math.fsum(moments[:,j]) for j in range(3)])
        tau=np.array([bias[j]+math.fsum(J[i,k,j]*forces[i,k] for i in range(2) for k in range(3)) for j in range(20)])
        q=np.array(state['q20_rad']);gain=old['gain20'];offset=old['biasprm'][:,0]+old['biasprm'][:,1]*q
        PDlo=gain*old['ctrlrange'][:,0]+offset;PDhi=gain*old['ctrlrange'][:,1]+offset
        lo=np.maximum.reduce([PDlo,old['actuator_forcerange'][:,0],old['joint_actfrcrange'][:,0]])
        hi=np.minimum.reduce([PDhi,old['actuator_forcerange'][:,1],old['joint_actfrcrange'][:,1]])
        for a,b,label in [(forces,V['force_world_N'],'force'),(fn,V['normal_force_N'],'normal'),(fr,V['force_balance_residual_world_N'],'force defect'),(moment,V['moment_residual_world_Nm'],'moment'),(tau,V['tau_act_required_Nm'],'tau'),(lo,op['torque_lower'],'lower'),(hi,op['torque_upper'],'upper')]:need(close(a,b),label)
        need(lam.min()>=-1e-9 and fn.max()<=3+1e-9 and min(tau-lo)>=-1e-9 and min(hi-tau)>=-1e-9,'recovery hard bounds respected')
        force_inf=float(max(abs(fr)));need(close(force_inf,V['force_infinity_N']) and force_inf<=V['solver_force_slack_t_N']+1e-9,'force infinity recovery slack')
        need(force_inf>1e-8,'nonzero recovery not force pass')
        for i,ct in enumerate(state['contacts']):
            pair=ct['pairing']['closest_pair'];feat=ct['features']
            pp=np.array([dot(pair['object_weights'],np.array(feat['object_support_points_world_m'])[:,j]) for j in range(3)])
            hh=np.array([dot(pair['hand_weights'],np.array(feat['hand_support_points_world_m'])[:,j]) for j in range(3)])
            need(pair['rank']==pair['unknown_count'] and ct['pairing']['status']=='UNIQUE_CLOSEST_PAIR','unique actual witnesses')
            need(close(pp,pts[i]) and close(hh-pp,np.array(feat['normal_world'])*feat['signed_gap_m']) and close(n[i],-np.array(feat['normal_world'])),'paired witness/normal')
            need(close(J[i],np.array(ct['actual_hand_point_J3x26'])[:,:20]),'J at paired hand point')
        u=pts[1]-pts[0];u/=np.linalg.norm(u);arm=COM-pts[0];below=dot(arm-u*dot(u,arm),gravity-u*dot(u,gravity))
        # Primary force-hard infeasibility forbids both enrollment flags even
        # if the recovery's own actuator values and positive-gap geometry fit.
        need(not S['strict_support_proposal_pass'] and not S['eligible_for_bounded_dynamic_diagnostic'],'strict vs diagnostic flags')
        need('proposed_ctrl20' not in S,'recovery controls not enrolled')
        ctrl=(tau-offset)/gain
        diag=[]
        for j in range(20):
            diag.append({'joint_ordinal_1based':j+1,'joint_name':names[j],'q_rad':float(q[j]),'ctrl_upper_rad':float(old['ctrlrange'][j,1]),'q_to_ctrl_upper_rad':float(old['ctrlrange'][j,1]-q[j]),
              'PD_tau_lower_Nm':float(PDlo[j]),'PD_tau_upper_Nm':float(PDhi[j]),'intersection_lower_Nm':float(lo[j]),'intersection_upper_Nm':float(hi[j]),
              'recovery_tau_Nm':float(tau[j]),'slack_lower_Nm':float(tau[j]-lo[j]),'slack_upper_Nm':float(hi[j]-tau[j]),
              'recovery_ctrl_rad':float(ctrl[j]),'upper_source_PD':bool(abs(hi[j]-PDhi[j])<=1e-10)})
        PD.append({'seed':seed,'all20':diag,'closest_to_zero_upper':sorted(diag,key=lambda x:abs(x['intersection_upper_Nm']))[:5],
             'scope':'Co-occurrence/range algebra only. Not a counterfactual or unique-cause explanation; no bounds were changed and no LP rerun.'})
        cases.append({'seed':seed,'status':S['status'],'error':S['error'],'iterations':S['iterations'],'complete_rows':len(seedrows),'selection':S['selection'],
             'fixed_phone_position_max_difference_m':float(max(abs(np.array(state['phone_p_world_m'])-p0))),
             'fixed_phone_rotation_matrix_max_difference':float(np.max(abs(np.array(state['phone_R_world'])-R0))),
             'geometry_pass':bool(physical),'pad_gaps_m':gaps,'physical_unknown_count':len(state['unknown'])+len(S['unknown']),
             'primary_force_balance_feasible':False,'recovery_force_residual_world_N':fr.tolist(),'recovery_force_infinity_N':force_inf,
             'recovery_normal_force_N':fn.tolist(),'recovery_moment_world_Nm':moment.tolist(),'recovery_minimum_torque_slack_Nm':float(min(min(tau-lo),min(hi-tau))),
             'actual_COM_below_dot_m2_s2':below,'strict_support_proposal_pass':False,'eligible_for_bounded_dynamic_diagnostic':False,
             'recovery_optimality_or_primary_infeasibility_independently_certified':False})
    need(totalLP==r['LP_calls']==4,'two primary plus two recovery calls')
    D=r['counts'];N=D['evaluations'];need(all(D[k]==N for k in ['mj_kinematics','mj_comPos','mj_comVel','mj_rne','mj_passive','mj_transmission']),'native pipeline counters')
    need(D['anchor_mj_jac']==2*N and D['actual_pair_mj_jac']==D['closest_pair_unique'],'Jacobian counters')
    need(D['heavy_queries_attempted']==D['heavy_queries_returned'] and D['unknown_queries']==0,'query counts')
    need(N-len(rows)-finals==sum(bool(x['error']) for x in r['seed_results']),'partial deadline states')
    need(r['geometry_pass_count']==2 and r['strict_support_proposal_count']==0 and r['bounded_dynamic_diagnostic_eligible_count']==0,'global gates')
    need(not r['holding_or_ego_policy_qualified'] and r['integration_training_GPU']==0,'no dynamics qualification')
    checks.update({'all_q_only_rows_and_final_fixed_phone_pose':True,'all22_189_saved_geometry_gates':True,
      'both_recovery_lambda_force_moment_torque_bounds_recomputed':True,'strict_vs_dynamic_enrollment_flags_preserved':True,'stage_native_LP_counters':True,
      'no_model_FK_LP_optimizer_or_neural_imports':not any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp','scipy.optimize'])})
    save(OUT/'PD_reachability_diagnostics.json',PD)
    np.savez_compressed(OUT/'source_reconstructed_q_only_rows.npz',qpos27_reconstructed=np.array(fullq),seed_ids=np.array([x['seed'] for x in rows]),
       reconstruction_is_raw_native_capture=np.array(False))
    report={'schema':'fixed_phone_closure_v4_saved_independent_review:1','status':'PASS_SAVED_ARITHMETIC_GEOMETRY2_FORCE_HARD0',
      'source':pin(__file__),'producer_report':pin(rf),'protocol':r['protocol'],'terminal':pin(ROOT/'run1_process_terminal.json'),
      'prior_actual_PD_operands':parent_report['outputs']['support_operands_and_controls.npz'],'robot_joint_metadata':pin(interface),
      'checks':checks,'passed':sum(checks.values()),'total':len(checks),'complete_rows':len(rows),'base':r['base_evaluations'],'FD':r['FD_evaluations'],'callbacks':len(callbacks),
      'cases':cases,'static_counts':D,'LP_calls':totalLP,'geometry_count':2,'strict_support_proposal_count':0,'dynamic_diagnostic_eligible_count':0,'holding_qualified':False,
      'scope':'No new model/FK/LP/optimizer/physics. All saved q20 rows reconstruct the fixed phone pose by source design; per-row native phone matrices were not captured, so they are not claimed independently observed. Both final native pose captures match. Saved convex distances/Jacobians/actual metadata are source/prior-contract evidence; this review recomputes their inequality/linear algebra. LP primal recovery arithmetic is checked; absent duals, recovery optimality and true primary infeasibility are not independently certified. A nonzero force-recovery defect never authorizes dynamics.',
      'outputs':{f.name:pin(f) for f in OUT.iterdir() if f.is_file() and f.name!='report.json'}}
    save(OUT/'report.json',report);print(json.dumps({'report':pin(OUT/'report.json'),'checks':f"{report['passed']}/{report['total']}",'geometry_count':2,'force_hard_count':0,'dynamic_eligible':0}))


if __name__=='__main__':main()
