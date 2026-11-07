"""Independent saved-only R/F audit. No producer, native geometry or solver API."""
from pathlib import Path
import argparse,csv,hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
CANDIDATE_SHA='6364a6e262f7fbedac049d2d9e789b2c9077cbd6ef7656ef2e9b394161bab81a'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pin=lambda p:{'path':Path(p).resolve().as_posix(),'sha256':sha(p)}
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def csr(z,prefix,shape_key):return sparse.csr_matrix((z[prefix+'data'],z[prefix+'indices'],z[prefix+'indptr']),shape=z[shape_key])


def audit(inp,out):
    if sha(inp/'candidate.npz')!=CANDIDATE_SHA:raise ValueError('Only the frozen6364 pipeline result')
    out.mkdir(parents=True,exist_ok=False)
    c=npz(inp/'candidate.npz');p=npz(inp/'pipeline_return.npz');R=npz(inp/'R_optimizer_return.npz');q=npz(inp/'F_majors/01/fixed_QP_model.npz');o=npz(inp/'F_majors/01/QP_optimizer_return.npz')
    report=read(inp/'report.json');manifest=read(inp/'manifest.json');runtime=read(inp/'runtime.json');Rterm=read(inp/'R_terminal.json');Qterm=read(inp/'F_majors/01/QP_terminal.json')
    trace=[json.loads(s) for s in (inp/'trace.jsonl').read_text().splitlines()];events=[json.loads(s) for s in (inp/'collision_events.jsonl').read_text().splitlines()];predictions=[json.loads(s) for s in (inp/'QP_predictions.jsonl').read_text().splitlines()]
    seed_path=Path(manifest['seed_candidate']['path']);seed=npz(seed_path)
    linear_path=Path(manifest['linear_contract']['path']);l=npz(linear_path);A=csr(l,'A_CSR_','A_shape');metadata=read(linear_path.with_name('linear_contract_metadata.json'));groups=metadata['row_groups']
    depth=c['constraint_scaled_mm'];x=c['optimizer_x'];xseed=seed['optimizer_x'];linear_res=A@(x-l['x0'])-l['b_centered'];box=np.maximum(np.maximum(l['lower']-x,x-l['upper']),0.)
    expected_Rlo=np.maximum(l['lower'],xseed-.05);expected_Rhi=np.minimum(l['upper'],xseed+.05);trust=np.maximum(np.maximum(expected_Rlo-x,x-expected_Rhi),0.)
    delta=x-xseed;dq=c['q_goal_rad']-seed['q_goal_rad']
    Cq=csr(q,'C_','C_shape');step=o['x'];lv=A@step-q['linear_rhs'];dv=Cq@step-q['geometry_rhs'];bv=np.maximum(np.maximum(q['step_lower']-step,step-q['step_upper']),0.)
    J=csr(c,'jacobian_CSR_','jacobian_shape');expected_g=J.T@c['residual'];expected_H=(J.T@J).toarray()+1e-4*np.eye(340)
    checks={'candidate_hash':sha(inp/'candidate.npz')==report['candidate']['sha256']==CANDIDATE_SHA,
            'canonical_equals_pipeline_final_and_lastaccepted_and_Rterminal':np.array_equal(x,p['final_x']) and np.array_equal(x,p['last_accepted_x']) and np.array_equal(x,p['R_terminal_x']) and np.array_equal(x,R['x'][:340]),
            'pipeline_normaltermination_F0_metadata':bool(p['normal_termination']) and int(p['F_accepted_steps'])==int(p['accepted_seq'])==0 and str(p['F_stop'])==report['F_stop'],
            'R_terminal_selected_no_history_or_QP_replacement':manifest['canonical_candidate']['selected_phase']=='R_start_F_NO_ACCEPTED_STEP' and not np.array_equal(x,o['x']),
            'hardP0_q0_and_original_object_targets_exact':np.array_equal(c['q0_fixed_rad'],seed['q0_fixed_rad']) and np.array_equal(c['q_goal_rad'][0],seed['q_goal_rad'][0]) and np.array_equal(c['control_points_rad'][0],seed['control_points_rad'][0]) and np.array_equal(c['mapped_object_pose_w'],seed['mapped_object_pose_w']) and np.array_equal(c['planned_object_pose_w'],seed['planned_object_pose_w']) and np.array_equal(c['morphology_normalized_target_points_w'],seed['morphology_normalized_target_points_w']) and np.array_equal(c['desired_unit_bone_directions_w'],seed['desired_unit_bone_directions_w']),
            'Rtrust_recomputed_exact_and_inside':np.array_equal(c['R_trust_lower'],expected_Rlo) and np.array_equal(c['R_trust_upper'],expected_Rhi) and not np.any(trust>0),
            'original_coefficient_bounds_raw_exact':not np.any(box>0),
            'planning_depth_raw_exact_true':bool(np.all(depth<=2.999)),
            'original_qualification3mm_raw_exact_true':bool(np.all(depth<=3.)),
            'linear_arithmetic_pass_but_exact_raw_fail_recorded':float(linear_res.max())>0 and float(linear_res.max())<=1e-12 and not report['final_original_primal']['exact_raw']['linear'] and report['final_original_primal']['declared_arithmetic']['linear'],
            'linear_rawmax_recomputed_matches':abs(float(linear_res.max())-Rterm['original_primal_after_fresh61']['max_linear_Ax_minus_b_native'])<1e-20,
            'R_auxiliary_slack_not_used_in_original_depth_gate':float(R['x'][-1])==Rterm['slack_mm']==0. and Rterm['decision_ignores_optimizer_success_and_slack'],
            'QP_model_base_is_R_terminal':np.array_equal(q['x_base'],x) and np.array_equal(q['r'],c['residual']),
            'QP_gradient_and_GN_Hessian_match_saved_original_fit':np.array_equal(q['g'],expected_g) and np.array_equal(q['H'],expected_H),
            'QP_direction_raw_residuals_match_rejection':abs(float(lv.max())-Qterm['proposal']['max_original_linear_residual_native'])<1e-20 and abs(float(dv.max())-Qterm['proposal']['max_linearized_depth_residual_mm'])<1e-20 and lv.max()>1e-12 and dv.max()>1e-9,
            'QP_descent_nonzero_stepbox_pass':float(q['g']@step)<0 and np.abs(step).max()>1e-12 and not np.any(bv>0),
            'QP_rejected_before_any_actual_alpha':not Qterm['proposal']['eligible'] and runtime['F_alpha_trials']==runtime['F_actual_trials']==runtime['F_accepted_steps']==0 and not any(e['kind']=='F_ACTUAL_ALPHA_TRIAL' for e in trace),
            'normal_stop_is_declared_F0_not_fit_improvement':report['F_label']=='F_NO_ACCEPTED_STEP' and not report['fit_improvement_claim'] and not report['fit_sequence_had_strict_decrease'],
            'two_forced61_refreshes':runtime['counts']['forced_61_constraint_refreshes']==2,
            'actual_collision_events_unique_shared_total':len(events)==runtime['counts']['geometry']['actual_collision_calls']==424 and [e['query_index'] for e in events]==list(range(1,425)),
            'no_physics_GPU_forbidden_record':runtime['physics_steps']==0 and not runtime['GPU_used'] and not runtime['Torch_imported'] and not runtime['Warp_imported'] and all(v==0 for v in runtime['forbidden_calls'].values()),
            'QP_predictions_not_actual_geometry':all(e['scope']=='FIXED_LINEARIZATION_QP_PREDICTION_ONLY_NO_NATIVE_GEOMETRY' for e in predictions)}
    state_records=[];all_state_pins=True
    for folder in sorted((inp/'states').iterdir()):
        state=read(folder/'record.json');data=npz(folder/'state.npz')
        all_state_pins &= sha(folder/'state.npz')==state['state_npz']['sha256'] and sha(folder/'native_contacts.json')==state['native_contacts']['sha256']
        state_records.append({'generation':state['generation'],'folder':folder.name,'record':pin(folder/'record.json'),'x_sha256':state['x_sha256'],'original_fit_cost':state['metrics']['original_fit_cost'],'max_generated_depth_mm':state['metrics']['max_generated_depth_mm'],'max_violation_mm':state['metrics']['max_positive_planning_violation_mm'],'query_delta':state['creation_query_delta'],'numerical_primal':state['original_primal']['passed'],'exact_raw_primal':state['original_primal']['exact_raw_all_pass']})
    checks['all6_state_archives_link_raw_hashes']=bool(all_state_pins and len(state_records)==6)
    checks['fresh_R_and_final61_actual_deltas']=all(state_records[i]['query_delta']=={'fit16':0,'constraint61':61,'total_shared_g':61,'force61_requested':True} for i in (4,5))
    checks['lastaccepted_and_finalfresh_x_stable']=state_records[3]['x_sha256']==state_records[4]['x_sha256']==state_records[5]['x_sha256']==hashlib.sha256(x.tobytes()).hexdigest()
    all_source_pins=all(sha(v['path'])==v['sha256'] for v in manifest['source_pins'].values())
    checks['frozen_runtime_sources_still_match']=all_source_pins
    checks['actual_source_plan_hash_bound']=sha(manifest['source_plan']['path'])==manifest['source_plan']['sha256']==str(p['source_plan_sha256'])==runtime['plan_sha256']
    raw=read(inp/'native_contacts.json');raw_matches=True
    for i in range(61):
        for category,kind in enumerate(c['category_order']):
            values=[-contact['distance_m'] for contact in raw['planned61'] if contact['sample']==i and contact['category']==kind]
            raw_matches &= abs(max([0.]+values)-c['operational_generated_maxdepth_m'][i,category])<1e-17
    checks['all122_depths_independent_rawcontact_min']=bool(raw_matches)
    checks['unknown_gap_NaNs_preserved']=bool(np.isnan(c['planned_native_selected_contact_distance_m'][~c['planned_native_generated_pair_mask']]).all())
    def fit_metrics(z):
        r=z['residual'];a=r[:6016].reshape(16,376)
        chunks={'bone':a[:,:45],'overlap':a[:,45:318],'material_position':a[:,318:327],'normal':a[:,327:336],'q0':a[:,336:356],'nominal':a[:,356:376],'adjacent':r[6016:6316],'curvature':r[6316:]}
        return {'cost':float(.5*r@r),'cost_components':{k:float(.5*np.sum(v*v)) for k,v in chunks.items()},'bone_mean_deg':float(z['bone_angle_error_deg'].mean()),'tip_mean_m':float(np.linalg.norm(z['fk_hand_points_w'][:,[4,8,12,16,20]]-z['morphology_normalized_target_points_w'][:,[4,8,12,16,20]],axis=-1).mean()),'material_point_mean_m':z['anchor_position_error_m'].mean(0).tolist(),'normal_mean_deg':z['anchor_normal_error_deg'].mean(0).tolist()}
    fm_seed,fm_final=fit_metrics(seed),fit_metrics(c)
    checks['final_fidelity_cost_matches_report']=abs(fm_final['cost']-report['cost'])<1e-10
    lpviol=[]
    for row in np.flatnonzero(lv>1e-12):
        group=next(g for g in groups if g['start']<=row<g['stop']);offset=int(row-group['start']);shape=group['tensor_shape'];tensor=np.unravel_index(offset,shape)
        lpviol.append({'row':int(row),'group':group['name'],'units':group['units'],'group_tensor_index':[int(a) for a in tensor],'residual':float(lv[row]),'predeclared_tolerance':1e-12,'ratio_to_tolerance':float(lv[row]/1e-12)})
    depviol=[{'row':int(row),'sample':int(row//2),'time_s':float((row//2)*.005),'category':str(c['category_order'][row%2]),'residual_mm':float(dv[row]),'predeclared_tolerance_mm':1e-9,'ratio_to_tolerance':float(dv[row]/1e-9)} for row in np.flatnonzero(dv>1e-9)]
    R_majors=[e for e in trace if e['kind'] in ('R_INITIAL_SEED','R_MAJOR_ITERATE_NOT_ACCEPTED')]
    summary={'status':'SAVED_ONLY_R_F_AUDIT_COMPLETE_SAMPLED_GEOMETRY_RESTORED_F0','checks':{k:bool(v) for k,v in checks.items()},'passed':sum(bool(v) for v in checks.values()),'total':len(checks),'analyzer':pin(__file__),'inputs':{name:pin(inp/name) for name in ('candidate.npz','pipeline_return.npz','manifest.json','report.json','runtime.json','R_optimizer_return.npz','R_terminal.json','trace.jsonl','collision_events.jsonl','QP_predictions.jsonl','F_majors/01/fixed_QP_model.npz','F_majors/01/QP_optimizer_return.npz','F_majors/01/QP_terminal.json')},'seed':pin(seed_path),'linear':pin(linear_path),
             'selection':{'source':'Rterminal_after_actual_original_primal_gate','canonical_is_last_algorithmically_accepted':True,'F_accepted_steps':0,'F_alpha_trials':0,'normal_termination':True,'no_fit_improvement_claim':True,'no_historical_or_QP_replacement':True},
             'delta_seed':{'coefficient_Linf_rad':float(np.max(np.abs(delta))),'coefficient_L2_rad':float(np.linalg.norm(delta)),'largest_control_joint_index':[int(a) for a in np.unravel_index(np.abs(c['control_points_rad']-seed['control_points_rad']).argmax(),(18,20))],'q16_Linf_rad':float(np.max(np.abs(dq))),'R_trust_radius_rad':.05,'R_trust_excess_rad':float(trust.max()),'original_box_excess_rad':float(box.max())},
             'restoration_gate':{'R_nit':Rterm['optimizer']['nit'],'R_nfev':Rterm['optimizer']['nfev'],'R_slack_mm':Rterm['slack_mm'],'final_depth_max_mm':float(depth.max()),'planning_raw_depth_pass':bool(np.all(depth<=2.999)),'planning_margin_mm':float(2.999-depth.max()),'qualification3mm_margin_mm':float(3.-depth.max()),'linear_raw_max_native':float(linear_res.max()),'linear_exact_raw_pass':bool(np.all(linear_res<=0)),'linear_declared_arithmetic_pass':bool(np.all(linear_res<=1e-12)),'coefficient_box_raw_pass':bool(np.all(box==0)),'arithmetic_tolerance_not_collision_accuracy':True},
             'fidelity':{'seed':fm_seed,'restored':fm_final,'cost_delta':fm_final['cost']-fm_seed['cost'],'cost_delta_percent':float(100*(fm_final['cost']/fm_seed['cost']-1))},
             'F_stop':{'reason':report['F_stop'],'QP_status':Qterm['optimizer']['status'],'QP_nit':Qterm['optimizer']['nit'],'QP_fun_calls':Qterm['counts']['fun'],'g_dot_step':float(q['g']@step),'step_abs_max_rad':float(np.abs(step).max()),'step_box_excess_rad':float(bv.max()),'predicted_QP_objective':float(o['fun']),'actual_geometry_at_QP_step':'NOT_EVALUATED_NO_ALPHA_ALLOWED','violating_linear_rows':lpviol,'violating_linearized_depth_rows':depviol,'tolerance_was_not_changed':True},
             'R_major_sequence':[{'event':e['event'],'kind':e['kind'],'cost':e['original_fit_cost'],'max_depth_mm':e['max_depth_mm'],'max_violation_mm':e['max_positive_violation_mm'],'R_slack_mm':e.get('R_slack_mm'),'state':e['state']} for e in R_majors],
             'state_archives':state_records,'geometry_accounting':{'shared_total':len(events),'fit16_owned':sum(e['owner']=='fit16' for e in events),'constraint61_owned':sum(e['owner']=='constraint61' for e in events),'forced61_R_terminal':61,'forced61_final':61,'do_not_resum_parent_shared_g':True},
             'limits':['Restored61-point generated-overlap feasibility is not continuous clearance or physical tracking. Absent hand-self contacts remain unknown gaps.','R deliberately minimized slack, so its original fit cost may increase. F0 yields no demonstrated fit improvement.','The QP direction is an unexecuted fixed-linearization prediction. Its numerical violations exceeded frozen tolerances; no actual alpha or physical effect was measured.','No conclusion of global optimality, dynamic grasp, contact-force prediction, real ego-data validity or full151motion qualification. Root independently owns finalFP32 export and any later physics authorization.'],
             'geometry_queries':0,'LP_calls':0,'optimizer_calls':0,'physics_steps':0,'GPU_used':False,'controller_export_recomputed':False}
    write(out/'report.json',summary)
    rows=[{'event':e['event'],'kind':e['kind'],'wall_seconds':e['wall_seconds'],'generation':e['generation'],'cost':e['original_fit_cost'],'max_depth_mm':e['max_depth_mm'],'sum_violation_mm':e['sum_positive_violation_mm'],'R_slack_mm':e.get('R_slack_mm'),'state_record':e['state']['path']} for e in trace]
    with (out/'phase_trace.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'QP_rejection_exact.json',summary['F_stop'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    xx=np.arange(len(R_majors));ax[0,0].plot(xx,[e['max_depth_mm'] for e in R_majors],'.-',color='#2166ac',label='Max generated depth');ax[0,0].axhline(2.999,color='0.4',ls=':',label='2.999 mm planning')
    twin=ax[0,0].twinx();twin.plot(xx,[e['original_fit_cost'] for e in R_majors],'s--',color='#b2182b',label='Original fit cost');twin.set_ylabel('Original fit cost',color='#b2182b');ax[0,0].set(title='R major sequence: restore depth, retain fit tradeoff',xlabel='Seed / R major',ylabel='Max depth [mm]');h1,l1=ax[0,0].get_legend_handles_labels();h2,l2=twin.get_legend_handles_labels();ax[0,0].legend(h1+h2,l1+l2,fontsize=7,loc='center right')
    ax[0,1].plot(c['planned_time_s']*1000,seed['operational_generated_maxdepth_m'][:,0]*1000,label='Frozen seed64aa');ax[0,1].plot(c['planned_time_s']*1000,c['operational_generated_maxdepth_m'][:,0]*1000,label='Restored6364');ax[0,1].axhline(3.,color='0.4',ls=':',label='3 mm qualification');ax[0,1].set(title='Saved61-point generated hand-object geometry',xlabel='Reference time [ms]',ylabel='Depth [mm]');ax[0,1].legend(fontsize=8)
    by_control=np.max(np.abs(c['control_points_rad']-seed['control_points_rad']),axis=1);ax[1,0].bar(np.arange(18),by_control*1000,color='#2166ac');ax[1,0].axhline(50,color='0.4',ls=':',label='R trust box50mrad');ax[1,0].set(title='Coefficient change from fixed seed; P0 unchanged',xlabel='Control point',ylabel='Max joint change [mrad]');ax[1,0].legend(fontsize=8)
    ax[1,1].bar(['Linear native-unit\nresidual /1e-12','Linearized depth\nresidual /1e-9mm'],[lv.max()/1e-12,dv.max()/1e-9],color='#b2182b');ax[1,1].axhline(1,color='0.4',ls=':',label='Frozen eligibility threshold');ax[1,1].set_yscale('log');ax[1,1].set(title='Why no F alpha was allowed',ylabel='Violation / predeclared arithmetic tolerance');ax[1,1].legend(fontsize=8);ax[1,1].text(.57,.80,f'g·step={q["g"]@step:.3f}<0\nstep box passes',transform=ax[1,1].transAxes,va='top',fontsize=8)
    for a in ax.flat:a.grid(alpha=.2)
    fig.suptitle('R restores sampled geometry; F proposal rejected as prescribed\nCanonical is normal R terminal at F0; no fit-improvement or physics claim',fontsize=12)
    for suffix in ('png','svg'):fig.savefig(out/f'RF_saved_audit.{suffix}',dpi=160)
    plt.close(fig)
    print(json.dumps({'status':summary['status'],'passed':summary['passed'],'total':summary['total'],'delta_seed':summary['delta_seed'],'restoration_gate':summary['restoration_gate'],'F_stop':summary['F_stop'],'geometry':summary['geometry_accounting']},indent=2));return 0 if all(checks.values()) else 1

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,default=ROOT/'outputs/restoration_filter_prefix_v1');ap.add_argument('--output',type=Path,default=ROOT/'outputs/restoration_saved_review_v1');a=ap.parse_args();raise SystemExit(audit(a.input,a.output))
