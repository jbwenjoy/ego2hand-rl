"""Read saved terminal artifacts only; never import a geometry or solver API."""
from pathlib import Path
import argparse,csv,hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE_SHA='64aa2cfe44f20fee4546b07c4546916c95d57771404cb5fb4e39156206ad9b43'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pin(p):return {'path':Path(p).resolve().as_posix(),'sha256':sha(p)}


def analyze(inp,out):
    if sha(inp/'candidate.npz')!=CANDIDATE_SHA:raise ValueError('Only the frozen terminal64aa candidate')
    out.mkdir(parents=True,exist_ok=False)
    names=('candidate.npz','native_contacts.json','report.json','optimizer_iterations.jsonl','optimizer_return.npz','linear_contract.npz','linear_contract_metadata.json','runtime.json','phase0_LP.json','collision_events.jsonl','solver_initialization.npz')
    pins={name:pin(inp/name) for name in names}
    with np.load(inp/'candidate.npz',allow_pickle=False) as source:z={k:source[k].copy() for k in source.files}
    report=read(inp/'report.json');native=read(inp/'native_contacts.json');runtime=read(inp/'runtime.json');lp=read(inp/'phase0_LP.json')
    history=[json.loads(s) for s in (inp/'optimizer_iterations.jsonl').read_text().splitlines()]
    events=[json.loads(s) for s in (inp/'collision_events.jsonl').read_text().splitlines()]
    with np.load(inp/'optimizer_return.npz',allow_pickle=False) as src:opt={k:src[k].copy() for k in src.files}
    with np.load(inp/'linear_contract.npz',allow_pickle=False) as src:linear={k:src[k].copy() for k in src.files}
    with np.load(inp/'solver_initialization.npz',allow_pickle=False) as src:init={k:src[k].copy() for k in src.files}
    groups=read(inp/'linear_contract_metadata.json')['row_groups']
    r=z['residual'];local=r[:6016].reshape(16,376);Q=z['q_goal_rad'];q0=z['q0_fixed_rad'];P=z['control_points_rad'];depth=z['operational_generated_maxdepth_m'];t=z['planned_time_s']
    A=sparse.csr_matrix((linear['A_CSR_data'],linear['A_CSR_indices'],linear['A_CSR_indptr']),shape=linear['A_shape'])
    linear_residual=np.asarray(A@(z['optimizer_x']-linear['x0'])-linear['b_centered']).ravel()
    checks={'terminal_candidate_sha':pins['candidate.npz']['sha256']==report['candidate_sha256'],'terminal_optimizer_x_bitexact':np.array_equal(z['optimizer_x'],opt['x']),
            'source64_q0_hard_exact':np.array_equal(Q[0],q0) and np.array_equal(P[0],q0),
            'controls_reconstruct_q16_exact':np.array_equal(Q,q0+z['basis16'][:,1:]@(P[1:]-q0)),
            'controls_reconstruct_q61_exact':np.array_equal(z['planned_q_goal_rad'],q0+z['basis61'][:,1:]@(P[1:]-q0)),
            'cost_matches_residual_and_optimizer_return':abs(float(.5*r@r)-report['cost'])<1e-10 and abs(float(opt['fun'])-report['cost'])<1e-10,
            'linear_primal_residual_matches_report':abs(float(linear_residual.max())-report['linear_float64_audit']['max_Ax_minus_b_centered'])<1e-15,
            'coefficient_bounds_no_excess':bool(np.all(P>=z['joint_lower_rad']) and np.all(P<=z['joint_upper_rad'])),
            'unknown_distances_still_NaN':bool(np.isnan(z['planned_native_selected_contact_distance_m'][~z['planned_native_generated_pair_mask']]).all()),
            'raw_contact_phase_geometry_only':all(c['phase']=='native_CPU_geometry_not_solved_force' for records in native.values() for c in records),
            'only_one_LP_and_one_SLSQP':runtime['LP_calls']==runtime['SLSQP_calls']==1,
            '240_objective_trial_logs_ordered':[x['call'] for x in history]==list(range(1,241)),
            'terminal_cost_is_last_logged_trial':abs(history[-1]['cost']-report['cost'])<1e-10,
            'source_guard_record_zero':all(v==0 for v in runtime['forbidden_calls'].values()),
            'zero_physics_GPU_Torch_Warp_record':runtime['physics_steps']==0 and not runtime['GPU_used'] and not runtime['Torch_imported'] and not runtime['Warp_imported'],
            'first_SLSQP_fun_x_equals_raw_LP_x0':np.array_equal(init['phase0_LP_x0'],init['first_SLSQP_fun_x'])}
    raw_lookup={tuple(sorted(c['geom'])):c['names'] for records in native.values() for c in records}
    pairviol=[];aggregate=[];raw_ok=True;stored_ok=True
    for i,time_s in enumerate(t):
        sample=[c for c in native['planned61'] if c['sample']==i]
        for j,pair in enumerate(z['native_pair_indices']):
            raw=[c['distance_m'] for c in sample if tuple(sorted(c['geom']))==tuple(pair)]
            present=bool(z['planned_native_generated_pair_mask'][i,j]);d=z['planned_native_selected_contact_distance_m'][i,j]
            raw_ok &= (bool(raw)==present) and (abs(min(raw)-d)<1e-17 if raw else bool(np.isnan(d)))
            if present and -d>.002999:
                pairviol.append({'sample':i,'time_s':float(time_s),'pair_row':j,'geom':pair.tolist(),'names':raw_lookup[tuple(pair)],'depth_mm':float(-d*1000),'planning_excess_mm':float((-d-.002999)*1000),'qualification_excess_mm':float(max(-d-.003,0)*1000),'contacts_at_pair':int(z['planned_native_contact_count_by_pair'][i,j]),'ties_at_deepest_pair_contact':int(z['planned_native_contact_tie_count_by_pair'][i,j])})
        for category,kind in enumerate(z['category_order']):
            values=[-c['distance_m'] for c in sample if c['category']==kind]
            operational=max([0.]+values);stored_ok &= abs(operational-depth[i,category])<1e-17
            entry=report['planned61'][i]
            aggregate.append({'sample':i,'time_s':float(time_s),'category':str(kind),'generated':bool(z['category_generated_mask'][i,category]),'operational_depth_mm':float(depth[i,category]*1000),'planning_excess_mm':float(max(depth[i,category]-.002999,0)*1000),'qualification_excess_mm':float(max(depth[i,category]-.003,0)*1000),'selected_pair_row':int(z['category_selected_pair_row'][i,category]),'global_tie_contacts':len(entry['category_global_tied_contacts'][category])})
    checks.update(all_saved_pair_distances_match_raw_min=bool(raw_ok),all122_aggregate_depths_match_raw=bool(stored_ok))
    position=((z['anchor_material_points_w']-z['anchor_goal_points_w'])/.001).reshape(16,9)
    normal=(.25*(z['anchor_material_normals_w']-z['anchor_goal_hand_normals_w'])).reshape(16,9)
    checks['anchor_position_residual_exact']=np.array_equal(position,local[:,318:327])
    checks['anchor_normal_residual_exact']=np.array_equal(normal,local[:,327:336])
    overlap=np.zeros((16,273));mask=z['native_generated_pair_mask']&(z['native_selected_contact_distance_m']<-.0005);overlap[mask]=(-z['native_selected_contact_distance_m'][mask]-.0005)/.001
    checks['fit_soft_overlap_residual_exact']=np.array_equal(overlap,local[:,45:318])
    checks['sum_geometry_counter_is_not_double_counted']=len(events)==runtime['counts']['actual_collision_calls']==report['counts']['actual_collision_calls'] and [e['query_index'] for e in events]==list(range(1,len(events)+1))
    owners={owner:sum(e['owner']==owner for e in events) for owner in ('fit16','constraint61')}
    checks['geometry_owners_partition_one_shared_g']=sum(owners.values())==len(events) and owners['fit16']==report['counts']['fit_parent']['events_recorded']
    slices={'bone_direction':local[:,:45],'native_overlap':local[:,45:318],'material_position':local[:,318:327],'material_normal':local[:,327:336],'q0_prior':local[:,336:356],'nominal_prior':local[:,356:376],'adjacent_q':r[6016:6316],'second_q':r[6316:]}
    costs={k:{'cost':float(.5*np.sum(v*v)),'fraction_percent':float(100*.5*np.sum(v*v)/report['cost'])} for k,v in slices.items()}
    checks['cost_components_sum']=abs(sum(v['cost'] for v in costs.values())-report['cost'])<1e-10
    cost=np.array([h['cost'] for h in history]);wall=np.array([h['wall_s'] for h in history])
    convergence={'optimizer_status':report['optimizer_status'],'optimizer_success':report['optimizer_success'],'message':report['optimizer_message'],'nit':report['nit'],'objective_calls':runtime['objective_fun_calls'],'objective_jac_calls':runtime['objective_jac_calls'],'constraint_fun_calls':runtime['constraint_fun_calls'],'constraint_jac_calls':runtime['constraint_jac_calls'],'callbacks':runtime['callback_calls'],'unique_trial_x':runtime['unique_trial_x'],'LP_success':lp['success'],'LP_L1_sum_control_shift_rad':lp['fun'],
                 'first_cost':float(cost[0]),'terminal_cost':float(cost[-1]),'objective_reduction_fraction':float(1-cost[-1]/cost[0]),'max_trial_cost':float(cost.max()),'last20_trial_cost_min':float(cost[-20:].min()),'last20_trial_cost_max':float(cost[-20:].max()),'raw_objective_gradient_inf':float(np.max(np.abs(opt['jac']))),'history_scope':'240 objective line-search/trial calls, not 80 accepted-major iterates. Only latest full checkpoint was retained; no per-major aggregate depth/active-pair history exists. No historical feasibility, branch-churn or best-iterate claim reconstructed.'}
    linrows=[{'name':g['name'],'units':g['units'],'max_Ax_minus_b':float(linear_residual[g['start']:g['stop']].max()),'rows_within_1e_minus7_in_original_units':int(np.count_nonzero(linear_residual[g['start']:g['stop']]>-1e-7))} for g in groups]
    violations=[a for a in aggregate if a['planning_excess_mm']>0]
    rawworst=min(native['planned61'],key=lambda c:c['distance_m']);tie_slots=z['planned_native_contact_tie_count_by_pair']>1
    summary={'status':'SAVED_TERMINAL_REVIEW_COMPLETE_GEOMETRY_QUALIFICATION_FAIL','checks':{k:bool(v) for k,v in checks.items()},'passed':sum(bool(v) for v in checks.values()),'total':len(checks),'source':pin(__file__),'inputs':pins,'retarget_variant_id':str(z['retarget_variant_id']),
             'canonical_selection':'Terminal result.x64aa only. No historical substitute selected.','terminal_classification':report['terminal_classification'],'linear_primal':report['linear_float64_audit'],'linear_groups':linrows,'nonlinear_violations':violations,'all_violating_pair_times':pairviol,'worst_raw_contact':rawworst,
             'counts':{'aggregate_constraints_over_2_999mm':len(violations),'aggregate_constraints_over_3mm':sum(a['qualification_excess_mm']>0 for a in aggregate),'pair_time_rows_over_3mm':sum(a['qualification_excess_mm']>0 for a in pairviol),'planning_max_excess_mm':float(max(depth.max()-.002999,0)*1000),'planning_sum_excess_mm':float(np.maximum(depth-.002999,0).sum()*1000),'terminal_generated_pair_tie_slots':int(tie_slots.sum()),'terminal_samples_with_pair_ties':int(np.any(tie_slots,axis=1).sum()),'native_raw_planned61':len(native['planned61']),'native_raw_fit16':len(native['fit16']),'shared_g_total':len(events),'fit16_owned_events':owners['fit16'],'constraint61_owned_events':owners['constraint61'],'never_add_fit_parent_shared_counter':True},
             'fit_tradeoff':{'cost_components':costs,'bone_angle_mean_deg':float(z['bone_angle_error_deg'].mean()),'bone_angle_max_deg':float(z['bone_angle_error_deg'].max()),'tip_mean_m':float(np.linalg.norm(z['fk_hand_points_w'][:,[4,8,12,16,20]]-z['morphology_normalized_target_points_w'][:,[4,8,12,16,20]],axis=-1).mean()),'material_point_mean_m':z['anchor_position_error_m'].mean(axis=0).tolist(),'material_point_max_m':z['anchor_position_error_m'].max(axis=0).tolist(),'normal_mean_deg':z['anchor_normal_error_deg'].mean(axis=0).tolist(),'normal_max_deg':z['anchor_normal_error_deg'].max(axis=0).tolist()},'convergence':convergence,
             'limits':['Linear feasibility and coefficient bounds do not imply native nonlinear feasibility. Status9/maxiter is not a global infeasibility certificate.','Ties and native approximate contact derivatives remain a possible numerical difficulty, not an established cause. No historical constraint trace supports a causal attribution.','61 nominal samples do not certify continuous geometry or actual dynamics. Ungenerated hand-self remains UNKNOWN gap.','Unknown-confidence fixed material/sticking, fixed palm/object mapping, anatomy and control/Bernstein restrictions can constrain this hypothesis; failure cannot establish source-motion impossibility.','Final FP32/controller audit is a separate Root-owned output. No controller or physical benefit inferred from this geometry review.'],
             'prospective_review':['Freeze a distinct bounded feasibility-restoration experiment before any additional solve: retain original source/q0/linear constraints and minimize explicit nonlinear violation slack before fit; do not quietly extend80iterations or select a historical state.','Record every major iterate cost, maximum/summed violation, selected pair/contact and tie/branch signatures so merit-line-search progress can be assessed separately from trial costs.','Before treating a restored candidate as qualified, independently test active constraint derivatives with branch-aware local FD and guard against contact-discovery discontinuities; no positive-gap imputation for missing pairs.','Keep exact wholecurve linear/rate/force certificates and independent finalFP32 export. A prospective change to conservative Bernstein/held-control restrictions must be a separate method, not a threshold relaxation.','If feasibility is not established, report restricted local search/hypothesis failure; consider confidence-aware contact-mode/morphology hypotheses only under a new protocol. Full151motion still requires a continuous new trajectory, no old-tail splice.'],
             'geometry_queries':0,'optimization_calls':0,'LP_calls':0,'physics_steps':0,'GPU_used':False,'controller_qualification_recomputed':False,'full_motion_or_physical_qualification':False}
    write(out/'report.json',summary)
    for filename,rows in [('all122_constraints.csv',aggregate),('violating_pair_times.csv',pairviol),('objective_trials.csv',[{'call':h['call'],'wall_s':h['wall_s'],'cost':h['cost']} for h in history])]:
        with (out/filename).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax[0,0].plot(t*1000,depth[:,0]*1000,'.-',label='Generated HO maximum')
    ax[0,0].axhline(3.,color='#b2182b',ls='--',label='3 mm qualification')
    ax[0,0].axhline(2.999,color='0.35',ls=':',label='2.999 mm planning')
    idx=np.where(depth[:,0]>.003)[0];ax[0,0].scatter(t[idx]*1000,depth[idx,0]*1000,color='#b2182b',zorder=5)
    ax[0,0].set(title='Terminal saved native geometry',xlabel='Reference time [ms]',ylabel='Depth [mm]');ax[0,0].legend(fontsize=8)
    for j in range(3):ax[0,1].plot(z['time_s']*1000,z['anchor_position_error_m'][:,j]*1000,'.-',label=f'F{j+1} material point')
    ax[0,1].set(title='Terminal material fit',xlabel='Reference time [ms]',ylabel='Position error [mm]');ax[0,1].legend(fontsize=8)
    ax[1,0].semilogy(np.arange(1,len(cost)+1),cost,color='#2166ac',lw=1)
    ax[1,0].scatter([len(cost)],[cost[-1]],color='#b2182b',zorder=4)
    ax[1,0].set(title='Saved objective trial calls (not major iterates)',xlabel='Objective call',ylabel='Cost (log scale)')
    for j in range(3):ax[1,1].plot(z['time_s']*1000,z['anchor_normal_error_deg'][:,j],'.-',label=f'F{j+1} normal')
    ax[1,1].set(title='Terminal material normal fit',xlabel='Reference time [ms]',ylabel='Normal error [deg]');ax[1,1].legend(fontsize=8)
    for a in ax.flat:a.grid(alpha=.25)
    fig.suptitle('Constrained prefix: linear feasible; sampled nonlinear qualification failed\nTerminal SLSQP status9/maxiter80 retained; no historical replacement',fontsize=12)
    for suffix in ('png','svg'):fig.savefig(out/f'terminal_review.{suffix}',dpi=160)
    plt.close(fig)
    print(json.dumps({'status':summary['status'],'passed':summary['passed'],'total':summary['total'],'violations':violations,'cost_components':costs,'convergence':convergence},indent=2))
    return 0 if all(checks.values()) else 1

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,default=ROOT/'outputs/constrained_prefix_geometry_v1');ap.add_argument('--output',type=Path,default=ROOT/'outputs/constrained_terminal_review_v1');args=ap.parse_args();raise SystemExit(analyze(args.input,args.output))
