"""All six saved candidates; arithmetic only, no model/FK/LP/optimization."""
from pathlib import Path
import hashlib,json,math
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
P=ROOT.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(v):
    p=Path(v['path']);assert pin(p)==v,'Changed pin '+str(p);return p
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def close(a,b,tol=1e-10):return bool(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)
def bits(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);return a.shape==b.shape and a.tobytes()==b.tobytes()
def cross(a,b):return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def summed(v):return np.array([math.fsum(float(t) for t in v[:,k]) for k in range(v.shape[1])])

def lp_values(lp,kind):
    conditional=kind=='conditional8'
    if conditional:
        points=np.asarray(lp['contact_points_world_m']);normals=np.asarray(lp['normals_force_on_object']);t1=np.asarray(lp['tangent1_world']);t2=np.asarray(lp['tangent2_world']);mu=lp['mu_design']
        assert close(lp['mu_actual'],.7) and close(mu,.7/math.sqrt(2))
        rays=np.array([[n+mu*(math.cos(2*math.pi*k/8)*a+math.sin(2*math.pi*k/8)*b) for k in range(8)] for n,a,b in zip(normals,t1,t2)])
        assert close(rays,lp['rays_force_world_per_normal_N'],1e-14)
        lam=np.asarray(lp['lambda_normal_N']);J=np.asarray(lp['J_robot_contact_3x20']);com=np.asarray(lp['object_COM_world_m']);g=np.asarray(lp['gravity_world_m_s2']);bias=np.asarray(lp['actual_bias_Nm']);lo=np.asarray(lp['torque_lower_Nm']);hi=np.asarray(lp['torque_upper_Nm']);labels=lp['finger_group_ids']
    else:
        points=np.asarray(lp['points_world']);C=np.asarray(lp['frames_rows']);mu=np.asarray(lp['friction5']);sign=np.asarray(lp['force_on_object_sign'])
        rays=np.array([[sg*(fr[0]+multiple*m[axis]*fr[axis+1]) for axis,multiple in [(0,-1),(0,1),(1,-1),(1,1)]] for sg,fr,m in zip(sign,C,mu)])
        assert close(rays,lp['rays_world_per_normal_N'],1e-14)
        lam=np.asarray(lp['lambda4_per_contact_N']);J=np.asarray(lp['J_hand_3x20']);com=np.asarray(lp['COM_world']);g=np.asarray(lp['gravity_world']);bias=np.asarray(lp['bias_minus_passive20']);lo=np.asarray(lp['PD_torque_lower20']);hi=np.asarray(lp['PD_torque_upper20']);labels=lp['finger_groups']
    assert lam.shape==rays.shape[:2] and np.isfinite(lam).all()
    forces=np.array([[math.fsum(float(lam[c,k]*rays[c,k,axis]) for k in range(lam.shape[1])) for axis in range(3)] for c in range(len(points))])
    force=summed(forces)+lp['mass_kg']*g;moment=summed(np.array([cross(p-com,f) for p,f in zip(points,forces)]))
    tau=np.array([bias[j]+math.fsum(float(J[c,axis,j]*forces[c,axis]) for c in range(len(points)) for axis in range(3)) for j in range(20)])
    scale=np.r_[np.full(3,lp['mass_kg']*np.linalg.norm(g)),np.full(3,lp['mass_kg']*np.linalg.norm(g)*.077)];residual=np.r_[force,moment];scaled=residual/scale;loss=float(max(abs(scaled)))
    groups=list(dict.fromkeys(labels));groupforce=np.array([math.fsum(float(lam[c,k]) for c in range(len(points)) if labels[c]==label for k in range(lam.shape[1])) for label in groups]);caps=np.asarray(lp['group_cap_N'])
    assert bits(caps,[3.,3.])
    mappings=[('force_residual_world_N',force),('moment_residual_world_Nm',moment),('scaled_residual6',scaled),('scaled_loss',loss)]
    if conditional:mappings += [('force_world_N',forces),('normal_force_per_group_N',groupforce),('tau_act_required_Nm',tau),('torque_slack_lower_Nm',tau-lo),('torque_slack_upper_Nm',hi-tau)]
    else:mappings += [('forces_on_object_world_N',forces),('group_normal_N',groupforce),('tau_required20',tau),('torque_slack_lower20',tau-lo),('torque_slack_upper20',hi-tau)]
    for key,value in mappings:assert close(value,lp[key],1e-11),'Saved LP arithmetic '+kind+' '+key
    hard=bool(lam.min()>=-1e-9 and np.min(tau-lo)>=-1e-9 and np.min(hi-tau)>=-1e-9 and np.min(caps-groupforce)>=-1e-9)
    assert hard and (lp['hard_tau_group_cone_feasible'] if conditional else lp['hard_bounds_pass']) is True
    assert max(abs(scaled))<=lp['solver_t']+1e-8
    fg=bool(max(abs(force))<=1e-8);mg=bool(max(abs(moment))<=1e-4)
    assert not fg and not mg
    return dict(scaled_loss=loss,force_infinity_N=float(max(abs(force))),moment_infinity_Nm=float(max(abs(moment))),force_residual_N=force.tolist(),moment_residual_Nm=moment.tolist(),hard_bounds_pass=hard,force_gate=False,moment_gate=False,min_torque_slack_Nm=float(min(np.min(tau-lo),np.min(hi-tau))),normal_per_finger_N=groupforce.tolist()),tau

def main():
    reportpath=ROOT/'completed_v1/report.json';r=read(reportpath);protocol=read(checked(r['protocol']));cfg=protocol['config']
    for p in [*protocol['source_pins'].values(),*protocol['inputs'].values()]:checked(p)
    assert read(reportpath.parent/'protocol.json')==protocol
    terminalpath=ROOT/'run1_terminal.json';terminal=read(terminalpath)
    for key in ['source','protocol','stdout','stderr']:checked(terminal[key])
    assert terminal['returncode']==0 and terminal['stderr']['bytes']==0 and terminal['protocol']==r['protocol']
    anchors=read(checked(protocol['inputs']['surface_anchors']));alg=read(reportpath.parent/'algorithm_surface_configuration.json')
    assert bits(alg['hand_geom_local_points'],[s['surfaces']['hand']['projected_point_geom_local_m'] for s in anchors['anchors']])
    assert bits(alg['phone_body_material_points'],[s['surfaces']['phone']['projected_point_phone_body_m'] for s in anchors['anchors']])
    assert bits(alg['phone_body_outward_normals'],[s['surfaces']['phone']['outward_normal_phone_body'] for s in anchors['anchors']])
    original=read(checked(protocol['inputs']['native_search_report']));base=next(c for seed in original['seed_results'] if seed['seed']==0 for c in seed['shortlist'] if c['rank']==0)
    xbase=np.r_[base['x46'][:8],base['x46'][20:26]];qbase=np.asarray(base['x46'][:20]);fixedB=np.asarray(alg['phone_body_material_points']);nObody=np.asarray(alg['phone_body_outward_normals'])
    finalrows=[];filepins={}
    for seed in r['seed_results']:
        assert len(seed['shortlist'])==3
        for short in seed['shortlist']:
            path=checked(short['result']);j=read(path);v=j['optimizer_candidate'];s=v['static'];x=np.asarray(j['x14']);native=j['native_probe'];pd=native['contact_design']['PD_raw'];current=j['current_native_LP'];cond=v['LP']
            assert j['seed']==seed['seed'] and j['rank']==short['rank'] and bits(x,v['x14']) and bits(j['q20'],v['q20'])
            assert bits(j['q20'][:8],x[:8]) and bits(j['q20'][8:],qbase[8:]) and bits(native['raw']['qpos27'][:20],j['q20'])
            assert native['unknown']==[] and s['unknown']==[] and not np.any(native['raw']['qvel26'])
            assert bits(s['phone_translation_palm_m'],x[8:11]) and bits(s['phone_rotvec_rad'],x[11:14])
            A=np.asarray(s['anchors_world_m']);B=np.asarray(s['anchor_targets_world_m']);R=np.asarray(s['phone_R_world']);p=np.asarray(s['phone_p_world_m']);nH=np.asarray(cond['normals_force_on_object']);nO=nObody@R.T
            assert close(B,fixedB@R.T+p,1e-12) and bits(cond['contact_points_world_m'],B)
            assert close(cond['J_robot_contact_3x20'],np.asarray(s['anchor_J2x3x26'])[:,:,:20],1e-14)
            closure=B-A+25e-6*nH;angles=np.arccos(np.clip(np.sum(nH*(-nO),axis=1),-1,1));gap=np.sum((B-A)*nH,axis=1);tangent=np.linalg.norm(B-A-gap[:,None]*nH,axis=1)
            assert close(closure,v['closure_world_m'],1e-12) and close(angles,v['normal_angles_rad'],1e-11)
            anchor_gate=bool(np.linalg.norm(closure,axis=1).max()<=.0002 and np.degrees(angles).max()<=1.)
            assert anchor_gate==j['surface_anchor_gate_pass']==short['anchor_gate'] and not anchor_gate
            geo=j['final_geometry'];gs=geo['state'];assert len(gs['phone_geometry_rows'])==22 and len(gs['self_geometry_rows'])==189 and not gs['unknown']
            values=[];labels=[]
            for row in gs['phone_geometry_rows']:
                assert row['status']!='DISJOINT_AABB_CERTIFICATE'
                if row['is_pad']:values.extend([row['lower_bound_m']+.0001,.0002-row['upper_bound_m']]);labels.extend([row['name']+'_lower',row['name']+'_upper'])
                else:values.append(row['lower_bound_m']);labels.append(row['name']+'_nonpenetration')
            for row in gs['self_geometry_rows']:values.append(row['lower_bound_m']);labels.append('/'.join(row['names'])+'_self_nonpenetration')
            assert labels==geo['constraints']['labels'] and close(values,geo['constraints']['values_m'],1e-14)
            assert geo['complete22_189'] and not geo['geometry_gate_pass'] and not geo['constraints']['all_satisfied'] and min(values)<0
            failures=[dict(label=l,violation_m=float(-value)) for l,value in zip(labels,values) if value<0]
            assert all('tip_sensor_col_lower' in v['label'] for v in failures)
            conditional,tau=lp_values(cond,'conditional8');actual,ntau=lp_values(current,'native4')
            gain=np.asarray(s['PD_gain20']);offset=np.asarray(s['PD_offset20']);ctrl=(np.asarray(cond['tau_act_required_Nm'])-offset)/gain;rawctrl=np.asarray(native['raw']['ctrl20']);cr=np.asarray(pd['actuator_ctrlrange20x2'])
            assert bits(ctrl,rawctrl) and bits(rawctrl,native['PD']['requested_ctrl20']) and bits(rawctrl,native['PD']['effective_ctrl20'])
            assert np.all(rawctrl>=cr[:,0]) and np.all(rawctrl<=cr[:,1])
            assert close((tau-offset)/gain,ctrl,1e-12)
            contacts=native['contact_design']['phone_hand_contacts'];assert len(contacts)==native['native_contact_count']==current['contact_count']==2
            mismatch=[];actualpointdistance=[]
            for ci,c in enumerate(contacts):
                finger=c['finger_index']-1;assert finger in [0,1]
                assert bits(current['points_world'][ci],c['position_world_m']) and bits(current['frames_rows'][ci],c['frame_rows_normal_t1_t2_world']) and bits(current['J_hand_3x20'][ci],c['hand_Jp3x20_world'])
                cn=np.asarray(c['normal_force_on_object_world']);mismatch.append(float(np.degrees(np.arccos(np.clip(nH[finger]@cn,-1,1)))))
                actualpointdistance.append(float(np.linalg.norm(np.asarray(c['position_world_m'])-B[finger])))
            residual=np.r_[5*closure.ravel()/1e-4,10*(nH+nO).ravel(),100*np.asarray(cond['scaled_residual6']),50*np.minimum(v['constraints']['values_m'],0)/1e-4,.02*(x[:8]-qbase[:8])/.3,.005*(x[8:11]-xbase[8:11])/.02,.005*(x[11:14]-xbase[11:14])/.3]
            assert close(residual,v['residual'],1e-8) and close(math.fsum(float(z*z) for z in residual),v['cost'],1e-7)
            assert j['strict_native_support_LP'] is None and not j['fresh_command_probes'] and not j['native_entry'] and j['candidate_state'] is None
            assert not j['holding_diagnostic_eligible'] and not j['ideal_surface_native_support_candidate'] and not j['faithful_ego_or_holding_qualified']
            filepins[path.name]=pin(path)
            finalrows.append(dict(seed=seed['seed'],rank=j['rank'],closure_norm_m=np.linalg.norm(closure,axis=1).tolist(),anchor_signed_gap_along_hand_normal_m=gap.tolist(),anchor_tangential_displacement_m=tangent.tolist(),normal_opposition_error_deg=np.degrees(angles).tolist(),source_material_points_phone_body_m=fixedB.tolist(),source_material_deviation_from_ego_target_m=[float(np.linalg.norm(fixedB[k]-np.asarray(anchors['anchors'][k]['original_ego_material_target_phone_body_m']))) for k in range(2)],failed_geometry_constraints=failures,all_other_geometry_constraints_nonnegative=True,conditional8=conditional,native4=actual,native_vs_conditional_normal_angle_deg=mismatch,native_position_to_fixed_phone_material_point_m=actualpointdistance,native_pad_normal_N=[native['native_contact_aggregates'][key]['normal_force_signed_N'] for key in ['thumb_pad','index_pad']],initial_ctrl_legal_unclamped=True,strict_LP_status='NOT_RUN_GEOMETRY_FAILED',anchor_gate=False,geometry_gate=False,holding_eligible=False))
    logs=read(reportpath.parent/'evaluation_log.json');counts=r['counts'];complete={i:sum(x['seed']==i for x in logs) for i in [0,1]};searchcounts={s['seed']:s['search_evaluations'] for s in r['seed_results']}
    assert complete=={0:235,1:636} and searchcounts=={0:235,1:637}
    assert r['seed_results'][1]['stop']=="StaticBudgetExceeded('Deadline')"
    assert len(logs)+len(finalrows)==r['LP_calls']==877
    for k in ['native_evaluations_attempted','native_evaluations_returned','native_MjData','mj_forward_attempted','mj_forward_returned','final_geometry_evaluations']:assert counts[k]==6
    assert counts['contact_force_reads']==counts['phone_hand_contact_jac_returned']==12 and counts['integration_steps']==counts['training_GPU']==r['new_steps_training_GPU']==0
    assert r['holding_diagnostic_candidates']==0 and r['model_audit']['passed'] is True and r['model_audit']['counts']==counts
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True);xx=np.arange(6);labels=[f"s{x['seed']}r{x['rank']}" for x in finalrows]
    ax[0].plot(xx,[x['conditional8']['scaled_loss'] for x in finalrows],'o-',label='conditional 8-ray');ax[0].plot(xx,[x['native4']['scaled_loss'] for x in finalrows],'s-',label='native 4-ray');ax[0].set(title='Different points/normals: no cone-only attribution',ylabel='scaled residual');ax[0].legend(fontsize=8)
    for k,finger in [(0,'thumb'),(1,'index')]:ax[1].plot(xx,[x['normal_opposition_error_deg'][k] for x in finalrows],'o-',label=finger)
    ax[1].axhline(1,color='black',ls=':',label='1 degree gate');ax[1].set(title='Fixed surface-normal opposition',ylabel='degrees');ax[1].legend(fontsize=8)
    for k,finger in [(0,'thumb'),(1,'index')]:ax[2].plot(xx,[x['failed_geometry_constraints'][k]['violation_m']*1e6 for x in finalrows],'o-',label=finger)
    ax[2].axhline(0,color='black',ls=':');ax[2].set(title='Excess beyond 100 micrometre pad compression',ylabel='excess (micrometre)');ax[2].legend(fontsize=8)
    for a in ax:a.set_xticks(xx,labels,rotation=25);a.grid(alpha=.25)
    fig.suptitle('All six saved candidates: geometry/normal/support gates remain failed',fontsize=11);fig.savefig(OUT/'six_candidate_diagnostics.png',dpi=180);plt.close(fig)
    checks={k:True for k in ['frozen_sources_model_protocol_and_all6_output_pins','terminal_exit0_no_new_backend','same_fixed_material_points_rest12q_and_pose_ordering','all6_anchor_closure_gap_normal_and_residual_cost_arithmetic','all6_complete22_189_saved_constraints_failure_labels','all12_LP_lambda_force_moment_tau_budget_arithmetic','all6_fresh_native_initial_controls_legal_unclamped','six_all_rejections_preserved_no_strict_LP_or_holding','recorded871_completed_search_LP_plus6_native_LP_no_steps']}
    out=dict(schema='surface_anchor_pose_saved_arithmetic_review:8',status='PASS_SAVED_ARITHMETIC_ALL_SIX_CANDIDATES_FAILED',source=pin(__file__),producer_report=pin(reportpath),protocol=r['protocol'],terminal=pin(terminalpath),candidate_pins=filepins,checks=checks,passed=sum(checks.values()),total=len(checks),results=finalrows,completed_search_records=complete,search_attempts=searchcounts,deadline_prefix_scope='Seed1 search637 incremented before a registered static deadline interruption;636 complete logged conditional LP results. No retry. Static total878 includes872search attempts+6final geometry, while LP877=871complete search+6native minimax.',recorded_counts=counts,recorded_LP_calls=r['LP_calls'],new_model_FK_LP_solve_optimization_physics_calls=0,plot=pin(OUT/'six_candidate_diagnostics.png'),holding_or_ego_or_policy_qualified=False,scope='Pure saved arrays/JSON arithmetic. Fixed H material normals are taken from captured conditional LP inputs, not independently recomputed by FK. Conditional forces use B phone points while hand J is at A; finite A-B can create a small force couple. Native4 uses newly generated points/frame/J. These points differ1-1.6mm and normals differ, so losses cannot be attributed solely to8-vs4 rays. Geometry/J/model immutability and API counts are pinned original source/runtime evidence; saved constraint identities/numbers recomputed here, no query/replay. No independent LP optimality proof or global infeasibility claim about the pose family. All six strict/geometry/normal gates remain failed, with no holding or learned-method success.')
    save(OUT/'report.json',out);print(json.dumps({'report':pin(OUT/'report.json'),'passed':out['passed'],'total':out['total']}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'failure.json',dict(error=repr(e),source=pin(__file__),no_automatic_retry=True));raise
