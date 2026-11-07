"""Saved arrays/scalar arithmetic only: no native model, FK, LP solve or steps."""
from pathlib import Path
import hashlib,json,math
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
P=ROOT.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(x):
    p=Path(x['path']);assert pin(p)==x,'Changed pin '+str(p)
    return p
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
def close(a,b,tol=1e-10):return bool(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)
def bits(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return a.shape==b.shape and a.tobytes()==b.tobytes()
def cross(a,b):return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def summed(v):return np.asarray([math.fsum(float(x) for x in v[:,k]) for k in range(v.shape[1])])

def audit():
    reportpath=ROOT/'body_axis_scan_v1/report.json';r=read(reportpath);protocol=read(checked(r['protocol']))
    assert r['schema']=='fixed_wrist_bodyZ_contact_relocation_result:7'
    for x in [*protocol['inputs'].values(),*protocol['source_pins'].values()]:checked(x)
    assert read(reportpath.parent/'protocol.json')==protocol
    terminalpath=ROOT/'body_axis_scan_v1_terminal.json';terminal=read(terminalpath)
    for k in ['source','protocol','stdout','stderr']:checked(terminal[k])
    assert terminal['returncode']==0 and terminal['stderr']['bytes']==0 and terminal['protocol']==r['protocol']
    with np.load(checked(protocol['inputs']['initial_state']),allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    seed=read(checked(protocol['inputs']['native_search_report']))
    original=next(c for s in seed['seed_results'] if s['seed']==0 for c in s['shortlist'] if c['rank']==0)
    x=np.asarray(original['x46']);assert bits(x,state['x46'])
    raw0=original['native']['raw'];R0=np.asarray(raw0['phone_body_R_world']);p0=np.asarray(raw0['phone_root_world_m']);com0=np.asarray(raw0['phone_COM_world_m'])
    placement=P/'hand2_stage1/completed_v2/palm_keypoints_and_placement.json';Rp=np.asarray(read(placement)['T_world_from_palm'])[:3,:3]
    prep=read(checked(protocol['inputs']['fixed_human_material_targets']))['preparation'];targets={i+1:np.asarray(a['target_projected_phone_convex_object_m']) for i,a in enumerate(prep)}
    results=[];referencepins={};initialpins={};allstats=[]
    assert len(r['references'])==3 and protocol['bodyZ_offsets_m']==[0.,.0245,.049]
    for i,ref in enumerate(r['references']):
        file=checked(ref['result']);j=read(file);p=j['probe'];raw=p['raw'];lp=j['LP'];pd=p['contact_design']['PD_raw'];contacts=p['contact_design']['phone_hand_contacts']
        initialfile=reportpath.parent/f'reference_{i}_initial_probe.json';assert read(initialfile)==p
        referencepins[str(i)]=pin(file);initialpins[str(i)]=pin(initialfile)
        delta=protocol['bodyZ_offsets_m'][i];dw=R0@np.array([0.,0.,delta]);expected_t=x[20:23]+Rp.T@dw
        assert j['index']==ref['index']==i and j['bodyZ_offset_m']==ref['bodyZ_offset_m']==delta
        assert bits(raw['qpos27'][:20],state['qpos27'][:20]) and bits(raw['qpos27'][23:27],state['qpos27'][23:27])
        assert bits(raw['ctrl20'],state['ctrl20']) and not np.any(raw['qvel26']) and bits(raw['phone_body_R_world'],R0)
        assert close(j['world_translation_delta_m'],dw,1e-14) and close(j['translation_palm_m'],expected_t,1e-14)
        assert close(np.asarray(raw['phone_root_world_m'])-p0,dw,1e-14) and close(np.asarray(raw['phone_COM_world_m'])-com0,dw,1e-14)
        assert p['unknown']==[] and len(contacts)==p['native_contact_count']==3 and not p['contact_design']['positive_normal_force_filter_applied']
        material=[]
        for c,m in zip(contacts,j['source_material_contact_diagnostics']):
            assert c['native_contact_index']==m['native_contact_index'] and c['finger_index']==m['finger_index']
            point=np.asarray(c['position_world_m']);body=(point-np.asarray(raw['phone_root_world_m']))@np.asarray(raw['phone_body_R_world'])
            target=targets[c['finger_index']];distance=float(np.linalg.norm(body-target))
            assert close(body,m['point_in_phone_body_m'],1e-14) and bits(target,m['original_human_material_target_body_m']) and close(distance,m['material_distance_to_original_target_m'],1e-14)
            material.append(dict(contact_index=c['native_contact_index'],finger_index=c['finger_index'],point_body_m=body.tolist(),target_body_m=target.tolist(),distance_m=distance))
        geo=j['geometry'];gs=geo['state'];rows=gs['phone_geometry_rows'];selfrows=gs['self_geometry_rows'];assert len(rows)==22 and len(selfrows)==189 and gs['unknown']==[]
        values=[];labels=[]
        for row in rows:
            assert row['status']!='DISJOINT_AABB_CERTIFICATE'
            if row['is_pad']:
                values.extend([row['lower_bound_m']+.0001,.0002-row['upper_bound_m']]);labels.extend([row['name']+'_lower',row['name']+'_upper'])
            else:values.append(row['lower_bound_m']);labels.append(row['name']+'_nonpenetration')
        for row in selfrows:values.append(row['lower_bound_m']);labels.append('/'.join(row['names'])+'_self_nonpenetration')
        assert labels==geo['constraints']['labels'] and close(values,geo['constraints']['values_m'],1e-14)
        geometric_pass=bool(min(values)>=0)
        assert geometric_pass and geo['geometry_gate_pass'] and geo['complete22_189'] and geo['constraints']['all_satisfied'] and not geo['constraints']['unknown']
        # Rebuild all saved native-cone forces independently with scalar fsum.
        lam=np.asarray(lp['lambda4_per_contact_N']);assert lam.shape==(3,4) and np.isfinite(lam).all()
        frames=np.asarray([c['frame_rows_normal_t1_t2_world'] for c in contacts]);friction=np.asarray([c['native_friction5'] for c in contacts]);sign=np.asarray([c['force_on_object_sign'] for c in contacts]);points=np.asarray([c['position_world_m'] for c in contacts]);Js=np.asarray([c['hand_Jp3x20_world'] for c in contacts]);com=np.asarray(raw['phone_COM_world_m'])
        rays=np.asarray([[sg*(fr[0]+mult*mu[axis]*fr[axis+1]) for axis,mult in [(0,-1),(0,1),(1,-1),(1,1)]] for fr,mu,sg in zip(frames,friction,sign)])
        for key,operand in [('points_world',points),('frames_rows',frames),('force_on_object_sign',sign),('friction5',friction),('J_hand_3x20',Js),('COM_world',com)]:assert bits(lp[key],operand)
        assert close(rays,lp['rays_world_per_normal_N'],1e-15)
        forces=np.asarray([[math.fsum(lam[c,k]*rays[c,k,axis] for k in range(4)) for axis in range(3)] for c in range(3)])
        mass=p['model_configuration']['phone_mass_kg'];g=np.asarray(lp['gravity_world']);force=summed(forces)+mass*g
        moments=np.asarray([cross(point-com,f) for point,f in zip(points,forces)]);moment=summed(moments)
        bias=np.asarray(pd['qfrc_bias26'])[:20]-np.asarray(pd['qfrc_passive26'])[:20]
        gain=np.asarray(pd['PD_gain20']);offset=np.asarray(pd['PD_offset20']);cr=np.asarray(pd['actuator_ctrlrange20x2']);ar=np.asarray(pd['actuator_forcerange20x2']);jr=np.asarray(pd['joint_actfrcrange20x2'])
        assert mass==lp['mass_kg']==.2 and bits(g,[0.,0.,-9.81]) and np.all(gain>0)
        assert pd['native_disabled_actuator_group_bitmask']==0 and pd['direct_unit_joint_mapping_verified_by_frozen_factory'] is True
        for key in ['actuator_ctrllimited20','actuator_forcelimited20','joint_actfrclimited20']:assert all(pd[key])
        lo=np.maximum.reduce([gain*cr[:,0]+offset,ar[:,0],jr[:,0]]);hi=np.minimum.reduce([gain*cr[:,1]+offset,ar[:,1],jr[:,1]])
        tau=np.asarray([bias[joint]+math.fsum(float(Js[c,axis,joint]*forces[c,axis]) for c in range(3) for axis in range(3)) for joint in range(20)])
        ctrl=(tau-offset)/gain
        groups=list(dict.fromkeys(c['finger_group'] for c in contacts));normal=np.asarray([math.fsum(float(lam[c,k]) for c in range(3) if contacts[c]['finger_group']==group for k in range(4)) for group in groups])
        scale=np.r_[np.full(3,mass*np.linalg.norm(g)),np.full(3,mass*np.linalg.norm(g)*.077)];residual=np.r_[force,moment];scaled=residual/scale;loss=float(np.max(np.abs(scaled)));t=float(lp['solver_t'])
        violation=float(max(0.,np.max(np.abs(scaled)-t),np.max(tau-hi),np.max(lo-tau),np.max(normal-3),-np.min(lam)))
        hard=bool(np.isfinite(np.r_[lam.ravel(),t]).all() and violation<=1e-8 and np.min(ctrl-cr[:,0])>=-1e-9 and np.min(cr[:,1]-ctrl)>=-1e-9)
        fg=bool(hard and max(abs(force))<=1e-8);mg=bool(hard and max(abs(moment))<=1e-4)
        for key,val in [('forces_on_object_world_N',forces),('force_residual_world_N',force),('moment_residual_world_Nm',moment),('tau_required20',tau),('proposed_ctrl20',ctrl),('group_normal_N',normal),('PD_torque_lower20',lo),('PD_torque_upper20',hi),('torque_slack_lower20',tau-lo),('torque_slack_upper20',hi-tau),('scale6',scale),('scaled_residual6',scaled),('scaled_loss',loss),('max_inequality_violation',violation)]:assert close(val,lp[key],1e-11),'LP arithmetic '+key
        assert lp['group_names']==groups and bits(lp['group_cap_N'],[3.,3.]) and hard==lp['hard_bounds_pass'] and fg==lp['force_gate_pass'] and mg==lp['moment_gate_pass'] and (fg and mg)==lp['strict_force_moment_pass']
        eligible=bool(geometric_pass and hard and fg and mg)
        assert not eligible and j['ideal_geometry_PD_candidate']==ref['ideal_geometry_PD_candidate']==eligible
        assert j['candidate_state'] is ref['candidate_state'] is j['proposed_control_native_probe'] is None and not j['faithful_ego_reference_or_holding_qualified']
        assert not lp['holding_qualified'] and not lp['native_soft_equilibrium_qualified']
        results.append(dict(index=i,bodyZ_offset_m=delta,world_translation_delta_m=dw.tolist(),robot_q20_and_phone_R_bits_unchanged=True,geometry_22_189_gate=geometric_pass,min_geometry_constraint_m=float(min(values)),LP_scaled_loss=loss,force_residual_world_N=force.tolist(),moment_residual_world_Nm=moment.tolist(),force_infinity_N=float(max(abs(force))),moment_infinity_Nm=float(max(abs(moment))),group_normal_N=normal.tolist(),group_normal_cap_N=3.,min_torque_slack_Nm=float(min(np.min(tau-lo),np.min(hi-tau))),min_ctrl_slack_rad=float(min(np.min(ctrl-cr[:,0]),np.min(cr[:,1]-ctrl))),min_lambda_N=float(lam.min()),max_saved_solution_inequality_violation=violation,hard_bounds_pass=hard,force_gate_pass=fg,moment_gate_pass=mg,strict_ideal_candidate=eligible,material_contact_diagnostics=material))
    counts=r['counts'];assert r['LP_calls']==3 and r['candidate_count']==0
    for k in ['native_evaluations_attempted','native_evaluations_returned','native_MjData','mj_forward_attempted','mj_forward_returned','final_geometry_evaluations']:assert counts[k]==3
    assert counts['contact_force_reads']==counts['phone_hand_contact_jac_attempted']==counts['phone_hand_contact_jac_returned']==9 and counts['integration_steps']==counts['training_GPU']==r['new_training_physics_steps']==0
    assert r['model_audit']['passed'] is True and r['model_audit']['counts']==counts
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xx=[v['bodyZ_offset_m']*1000 for v in results];fig,ax=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    ax[0].plot(xx,[v['LP_scaled_loss'] for v in results],'o-');ax[0].set(xlabel='Phone +body Z relocation (mm)',ylabel='scaled 6D residual (dimensionless)',title='Saved ideal LP solution: all strict gates FAIL')
    for c,name in [(0,'thumb'),(1,'index contact 1'),(2,'index contact 2')]:ax[1].plot(xx,[v['material_contact_diagnostics'][c]['distance_m']*1000 for v in results],'o-',label=name)
    ax[1].set(xlabel='Phone +body Z relocation (mm)',ylabel='distance to original material target (mm)',title='Original ego-derived surface candidates');ax[1].legend(fontsize=8)
    for aa in ax:aa.grid(alpha=.25);aa.set_xticks(xx)
    fig.suptitle('Three fixed geometry hypotheses: residual tradeoff, no holding or ego qualification',fontsize=10);fig.savefig(OUT/'residual_vs_material_deviation.png',dpi=180);plt.close(fig)
    checks={k:True for k in ['source_model_protocol_and_output_pins','terminal_exit0_and_empty_stderr','all3_q_R_ctrl_unchanged_and_declared_body_axis_transform','all9_material_coordinates_and_target_deviations','all3_saved22_189_constraint_arithmetic_no_new_geometry','all_native_frames_J_PD_operands_bound_to_saved_LP','all3_saved_lambda_forces_moments_tau_ctrl_and_group_budget','hard_and_strict_gates_recomputed_all_candidates_false','recorded3forward3geometry3LP_counts_and_source_audit_scope']}
    return dict(schema='body_axis_scan_saved_arithmetic_review:7',status='PASS_SAVED_ARITHMETIC_ALL_STRICT_CANDIDATES_FAILED',producer_report=pin(reportpath),protocol=r['protocol'],terminal=pin(terminalpath),source=pin(__file__),placement_metadata=pin(placement),references=referencepins,initial_probe_pins=initialpins,checks=checks,passed=sum(checks.values()),total=len(checks),results=results,original_recorded_counts=counts,original_recorded_LP_calls=r['LP_calls'],new_model_forward_LP_solver_physics_calls=0,plot=pin(OUT/'residual_vs_material_deviation.png'),holding_or_ego_or_policy_qualified=False,scope='Only saved reference/JSON/NPZ arithmetic. Forces reconstructed from native-frame rays and saved lambda using scalar fsum; residual/PD inverse and all hard bounds rechecked, no re-solve or independent optimality/dual certificate. Geometry distance queries/Jacobians/model immutability and API counts remain pinned producer source/runtime evidence; only their saved constraints/identity/counter consistency are checked here. All three references retained. Sequential engineered material relocation, not faithful ego trajectory or holding proof.')

if __name__=='__main__':
    try:r=audit()
    except Exception as e:
        save(OUT/'failure.json',dict(error=repr(e),source=pin(__file__),no_automatic_retry=True));raise
    save(OUT/'report.json',r);print(json.dumps({'report':pin(OUT/'report.json'),'passed':r['passed'],'total':r['total']}))
