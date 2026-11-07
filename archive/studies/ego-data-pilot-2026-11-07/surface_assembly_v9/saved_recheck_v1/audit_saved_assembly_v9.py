"""All four saved assembly candidates. No model, FK, solver or physics."""
from pathlib import Path
import hashlib,importlib.util,json,math
import numpy as np
H=Path(__file__).resolve().parent
ROOT=H.parent
P=ROOT.parent
HELPER=P/'surface_anchor_pose_v8/saved_recheck_v2/audit_saved_surface_search_v8_v2.py'
HELPER_SHA='719a79a293c91233167c81b94ee36b846906ca103912e53f8dc38011489a602d'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
def checked(v):
    p=Path(v['path']);assert pin(p)==v,'Changed pin '+str(p);return p
def close(a,b,tol=1e-10):return bool(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)
def bits(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);return a.shape==b.shape and a.tobytes()==b.tobytes()
def unit(x):return x/np.linalg.norm(x)
def angle(a,b):return float(math.atan2(np.linalg.norm(np.cross(unit(a),unit(b))),np.clip(unit(a)@unit(b),-1,1)))

def main():
    assert pin(HELPER)['sha256']==HELPER_SHA
    spec=importlib.util.spec_from_file_location('frozen_scalar_wrench_arithmetic',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    rp=ROOT/'completed_v1/report.json';r=read(rp);protocol=read(checked(r['protocol']));cfg=protocol['config']
    for p in [*protocol['source_pins'].values(),*protocol['inputs'].values()]:checked(p)
    assert read(rp.parent/'protocol.json')==protocol
    terminalpath=ROOT/'run1_terminal.json';terminal=read(terminalpath)
    for key in ['source','protocol','stdout','stderr']:checked(terminal[key])
    assert terminal['returncode']==0 and terminal['stderr']['bytes']==0 and terminal['protocol']==r['protocol']
    anchors=read(checked(protocol['inputs']['surface_anchors']));alg=read(rp.parent/'algorithm_surface_configuration.json')
    fixedH=np.asarray([a['surfaces']['hand']['projected_point_geom_local_m'] for a in anchors['anchors']]);fixedB=np.asarray([a['surfaces']['phone']['projected_point_phone_body_m'] for a in anchors['anchors']]);nObody=np.asarray([a['surfaces']['phone']['outward_normal_phone_body'] for a in anchors['anchors']]);ego=np.asarray([a['original_ego_material_target_phone_body_m'] for a in anchors['anchors']])
    assert bits(alg['hand_geom_local_points'],fixedH) and bits(alg['phone_body_material_points'],fixedB) and bits(alg['original_ego_material_targets_body_m'],ego)
    tangent=[]
    for a,n in zip(anchors['anchors'],nObody):
        tri=np.asarray(a['surfaces']['phone']['source_triangle_vertices_phone_body_m']);nn=unit(n);edge=tri[1]-tri[0];t1=unit(edge-nn*(edge@nn));tangent.append([t1,unit(np.cross(nn,t1))])
    tangent=np.asarray(tangent);qbase=np.asarray(anchors['source_pose']['qpos27'])[:20]
    records=[];filepins={}
    for seed in r['seed_results']:
        assert len(seed['finals'])==2
        for final in seed['finals']:
            path=checked(final['result']);j=read(path);v=j['candidate'];k=v['kinematics'];s=v['static'];native=j['native_probe'];raw=native['raw'];pd=native['contact_design']['PD_raw'];cond=v['LP'];nlp=j['current_native_LP'];strict=j['strict_native_support_LP'];x=np.asarray(v['x14'])
            assert j['seed']==seed['seed'] and j['rank']==final['rank']
            assert bits(k['q20'][:8],x[:8]) and bits(k['q20'][8:],qbase[8:]) and bits(raw['qpos27'][:20],k['q20']) and not np.any(raw['qvel26'])
            assert k['anchor_pin']==protocol['inputs']['surface_anchors']
            A=np.asarray(k['A_hand_world_m']);B=np.asarray(k['B_phone_world_m']);R=np.asarray(k['phone_R_world']);p=np.asarray(k['phone_p_world_m']);nH=np.asarray(k['n_hand_world']);nO=np.asarray(k['n_phone_world']);T=tangent@R.T
            assert close(B,fixedB@R.T+p,1e-12) and close(nO,nObody@R.T,1e-12)
            assert close(A,s['anchors_world_m'],5e-12) and close(B,s['anchor_targets_world_m'],5e-12)
            assert close(T,k['phone_tangents_world2x2x3'],1e-12)
            closure=B-A+25e-6*nH;normal_eq=np.einsum('nki,ni->nk',T,nH);eq10=np.r_[closure.ravel(),normal_eq.ravel()];scaled_eq=np.r_[closure.ravel()/.001,normal_eq.ravel()];antidot=-np.sum(nH*nO,axis=1)
            assert close(closure,k['closure2x3'],1e-12) and close(normal_eq,k['normal_eq2x2'],1e-12) and close(eq10,k['eq10'],1e-12) and close(scaled_eq,v['eq10_scaled'],1e-10)
            maximum=float(np.max(abs(scaled_eq)));assert close(maximum,v['max_abs_eq10_scaled'],1e-10) and maximum<=1e-5
            degrees=np.degrees(np.arccos(np.clip(antidot,-1,1)));assert np.linalg.norm(closure,axis=1).max()<=.0002 and degrees.max()<=1 and v['assembly_gate'] and final['assembly_gate']
            geo=j['final_geometry'];gs=geo['state'];assert len(gs['phone_geometry_rows'])==22 and len(gs['self_geometry_rows'])==189 and not gs['unknown']
            values=[];labels=[];padgaps=[]
            for row in gs['phone_geometry_rows']:
                assert row['status']!='DISJOINT_AABB_CERTIFICATE'
                if row['is_pad']:
                    values.extend([row['lower_bound_m']+.0001,.0002-row['upper_bound_m']]);labels.extend([row['name']+'_lower',row['name']+'_upper']);padgaps.append(dict(geom=row['geom_id'],exact_gap_m=row['exact_gap_m'],lower=row['lower_bound_m'],upper=row['upper_bound_m']))
                else:values.append(row['lower_bound_m']);labels.append(row['name']+'_nonpenetration')
            for row in gs['self_geometry_rows']:values.append(row['lower_bound_m']);labels.append('/'.join(row['names'])+'_self_nonpenetration')
            assert labels==geo['constraints']['labels'] and close(values,geo['constraints']['values_m'],1e-14) and min(values)>=0
            assert geo['complete22_189'] and geo['geometry_gate_pass'] and final['geometry_pass'] and not geo['constraints']['unknown']
            # Common A in both object wrench and actual saved hand Jacobian.
            assert bits(cond['contact_points_world_m'],A) and close(cond['J_robot_contact_3x20'],np.asarray(s['anchor_J2x3x26'])[:,:,:20],1e-14) and close(cond['normals_force_on_object'],nH,1e-14)
            conditional,tau=helper.lp_values(cond,'conditional8');actual,ntau=helper.lp_values(nlp,'native4')
            gain=np.asarray(s['PD_gain20']);offset=np.asarray(s['PD_offset20']);expected_ctrl=(np.asarray(cond['tau_act_required_Nm'])-offset)/gain;cr=np.asarray(pd['actuator_ctrlrange20x2'])
            assert j['initial_native_probe_ctrl_source']=='legal_conditional_common_A_inverse_PD' and bits(expected_ctrl,j['conditional_inverse_ctrl20']) and bits(expected_ctrl,raw['ctrl20'])
            assert bits(raw['ctrl20'],native['PD']['requested_ctrl20']) and bits(raw['ctrl20'],native['PD']['effective_ctrl20']) and np.all(expected_ctrl>=cr[:,0]) and np.all(expected_ctrl<=cr[:,1])
            assert native['unknown']==[] and native['native_contact_count']==7
            contacts=native['contact_design']['phone_hand_contacts'];assert len(contacts)==7 and [sum(c['finger_index']==f for c in contacts) for f in [1,2]]==[3,4]
            assert len(j['actual_material_contact_diagnostics'])==7
            diagnostics=[]
            for ci,(c,d) in enumerate(zip(contacts,j['actual_material_contact_diagnostics'])):
                finger=c['finger_index']-1;point=np.asarray(c['position_world_m']);body=np.linalg.solve(np.asarray(raw['phone_body_R_world']),point-np.asarray(raw['phone_root_world_m']));normal=np.asarray(c['normal_force_on_object_world'])
                assert c['native_contact_index']==d['native_contact_index']==ci and finger+1==d['finger_index']
                assert close(body,d['native_point_phone_body_m'],1e-12) and bits(d['fixed_phone_material_point_body_m'],fixedB[finger]) and bits(d['original_ego_material_target_body_m'],ego[finger])
                fixed_distance=float(np.linalg.norm(body-fixedB[finger]));ego_distance=float(np.linalg.norm(body-ego[finger]));raw_angle=float(np.arccos(np.clip(normal@nH[finger],-1,1)))
                assert close(fixed_distance,d['distance_to_fixed_phone_material_point_m'],1e-12) and close(ego_distance,d['distance_to_original_ego_target_m'],1e-12) and close(raw_angle,d['native_vs_conditional_force_normal_angle_rad'],8*math.sqrt(np.finfo(float).eps))
                for key,operand in [('points_world',point),('frames_rows',c['frame_rows_normal_t1_t2_world']),('J_hand_3x20',c['hand_Jp3x20_world']),('friction5',c['native_friction5'])]:assert bits(nlp[key][ci],operand)
                da=point-A[finger];db=point-B[finger]
                diagnostics.append(dict(native_contact_index=ci,finger_index=finger+1,point_phone_body_m=body.tolist(),distance_to_fixed_phone_material_m=fixed_distance,distance_to_original_ego_target_m=ego_distance,distance_to_hand_A_world_m=float(np.linalg.norm(da)),point_minus_A_normal_m=float(da@nH[finger]),point_minus_A_tangent_m=float(np.linalg.norm(da-(da@nH[finger])*nH[finger])),point_minus_B_normal_m=float(db@nO[finger]),point_minus_B_tangent_m=float(np.linalg.norm(db-(db@nO[finger])*nO[finger])),angle_to_nH_rad=angle(normal,nH[finger]),angle_to_minus_nO_rad=angle(normal,-nO[finger])))
            assert strict['schema']=='strict_native_point_support_PD_witness_guarded:7' and strict['solver_status']==2 and strict['status']=='INFEASIBLE_TIGHTENED_IDEAL_POINT_WITNESS_NOT_ORIGINAL_BOX_PROOF'
            assert strict['proposed_ctrl20'] is None and strict['lambda4_per_contact_N'] is None and not strict['strict_force_moment_pass'] and strict['original_requested_box_infeasibility_established'] is False
            assert strict['force_tolerance_N']==1e-8 and strict['moment_tolerance_Nm']==1e-4 and strict['solver_moment_box_inset_Nm']==1e-10
            assert not j['ideal_native_support_candidate'] and not j['holding_diagnostic_eligible'] and not j['fresh_command_probes'] and j['candidate_state'] is None and not j['holding_or_faithful_ego_qualified']
            filepins[path.name]=pin(path)
            records.append(dict(seed=j['seed'],rank=j['rank'],max_abs_scaled_eq10=maximum,closure_norm_m=np.linalg.norm(closure,axis=1).tolist(),normal_opposition_error_deg=degrees.tolist(),independent_saved_geometry_gate=True,pad_gap_records=padgaps,min_geometry_constraint_m=float(min(values)),conditional_common_A8=conditional,native_contacts4ray=actual,native_contact_count=7,thumb_index_contact_counts=[3,4],native_material_diagnostics=diagnostics,initial_ctrl_legal_unclamped=True,strict_support_status=strict['status'],strict_support_source_solver_status=2,holding_eligible=False))
    attempts=[json.loads(x) for x in (rp.parent/'attempts.jsonl').read_text().splitlines()];counts=r['counts']
    for seed in r['seed_results']:
        aa=[a for a in attempts if a['seed']==seed['seed']];assert sum(a['event']=='attempt' for a in aa)==seed['full_attempts'] and sum(a['event']=='returned' for a in aa)==seed['full_returned']
    assert [s['full_attempts'] for s in r['seed_results']]==[362,347] and r['LP_calls']==717
    for k in ['native_evaluations_attempted','native_evaluations_returned','native_MjData','mj_forward_attempted','mj_forward_returned','final_geometry_evaluations']:assert counts[k]==4
    assert counts['contact_force_reads']==counts['phone_hand_contact_jac_returned']==28 and counts['integration_steps']==counts['training_GPU']==r['new_steps_training_GPU']==r['holding_diagnostic_candidates']==0
    assert r['model_audit']['passed'] is True and r['cheap_model_audit']['passed'] is True
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xx=np.arange(4);labels=[f"s{x['seed']}r{x['rank']}" for x in records];fig,ax=plt.subplots(1,3,figsize=(11,3.4),constrained_layout=True)
    ax[0].plot(xx,[x['conditional_common_A8']['scaled_loss'] for x in records],'o-',label='common A 8-ray');ax[0].plot(xx,[x['native_contacts4ray']['scaled_loss'] for x in records],'s-',label='native 7-point 4-ray');ax[0].set(title='Saved ideal residuals',ylabel='dimensionless');ax[0].legend(fontsize=8)
    ax[1].semilogy(xx,[x['max_abs_scaled_eq10'] for x in records],'o-');ax[1].axhline(1e-5,color='black',ls=':',label='hard equation gate');ax[1].set(title='Assembly equation max',ylabel='scaled eq10');ax[1].legend(fontsize=8)
    for finger,name in [(1,'thumb'),(2,'index')]:
        minimum=[];maximum=[]
        for x in records:
            v=[d['distance_to_fixed_phone_material_m']*1000 for d in x['native_material_diagnostics'] if d['finger_index']==finger];minimum.append(min(v));maximum.append(max(v))
        ax[2].plot(xx,maximum,'o-',label=name+' max');ax[2].plot(xx,minimum,':',label=name+' min')
    ax[2].set(title='Native point to fixed CAD material',ylabel='distance (mm)');ax[2].legend(fontsize=8)
    for a in ax:a.set_xticks(xx,labels,rotation=20);a.grid(alpha=.25)
    fig.suptitle('All four: assembly/geometry PASS; tightened strict LP has no witness; no holding qualification',fontsize=10);fig.savefig(H/'four_candidate_summary.png',dpi=180,bbox_inches='tight');plt.close(fig)
    checks={k:True for k in ['source_model_protocol_output_terminal_pins','four_fixed_material_and_rest12q_bindings','four_eq10_closure_normals_common_K_and_E_arithmetic','four_complete_saved22_189_geometry_constraints','eight_minimax_lambda_force_moment_tau_group_budget_checks','four_common_A_point_and_J_same_saved_context','28_native_contact_coordinates_material_distances_normal_angles','four_initial_controls_legal_and_unclamped','four_tightened_strict_infeasible_scope_and_no_candidates','recorded_attempts_counters_no_steps_or_new_solver']}
    result=dict(schema='assembly_support_saved_arithmetic_review:9',status='PASS_SAVED_ARITHMETIC_ASSEMBLED_GEOMETRY_WITHOUT_STRICT_SUPPORT_WITNESS',source=pin(__file__),arithmetic_helper=pin(HELPER),producer_report=pin(rp),protocol=r['protocol'],terminal=pin(terminalpath),candidate_pins=filepins,checks=checks,passed=sum(checks.values()),total=len(checks),results=records,recorded_counts=counts,recorded_LP_calls=r['LP_calls'],recorded_full_attempts_by_seed=[s['full_attempts'] for s in r['seed_results']],new_model_FK_geometry_LP_optimization_physics_calls=0,plot=pin(H/'four_candidate_summary.png'),holding_or_faithful_ego_qualified=False,scope='Saved scalar-fsum wrench and transform arithmetic only. Common A force point and J bundle are checked from saved K/E context, not freshly validated by FK. Native Jacobians, model immutability, geometry queries and API counts remain source/runtime-backed; saved constraints and counters are checked, not replayed. No LP optimality or solver infeasibility recertification. Reported infeasible solver problem is tightened by1e-10Nm and does not prove original moment box infeasible. No original pose-family impossibility, finite patch, physical calibration, holding, faithful ego or learned benefit claim.')
    save(H/'report.json',result);print(json.dumps({'report':pin(H/'report.json'),'passed':result['passed'],'total':result['total']}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(H/'failure.json',dict(error=repr(e),source=pin(__file__),no_automatic_retry=True));raise
