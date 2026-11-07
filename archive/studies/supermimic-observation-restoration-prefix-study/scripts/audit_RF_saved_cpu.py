"""Independent terminal R/F artifact audit, with no production/solver imports."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def csr(z,prefix,shape_key):return sparse.csr_matrix((z[prefix+'data'],z[prefix+'indices'],z[prefix+'indptr']),shape=z[shape_key])
def main():
 base=ROOT/'outputs/restoration_filter_prefix_v1';watch=base.with_name(base.name+'_watchdog');planfile=ROOT/'configs/restoration-filter-authorized-v1.json';p=read(planfile);report=read(base/'report.json');run=read(base/'runtime.json');manifest=read(base/'manifest.json');state=read(watch/'state.json')
 if state['status']!='completed':raise RuntimeError('This receipt requires normally terminal artifacts')
 z=arrays(base/'candidate.npz');pipeline=arrays(base/'pipeline_return.npz');R=arrays(base/'R_optimizer_return.npz');Rt=read(base/'R_terminal.json');trace=[json.loads(a) for a in (base/'trace.jsonl').read_text().splitlines() if a];events=[json.loads(a) for a in (base/'collision_events.jsonl').read_text().splitlines() if a];checks=[]
 def check(n,ok,d=None):checks.append({'name':n,'pass':bool(ok),'details':d})
 def pinok(d):return sha(d['path'])==d['sha256']
 pinrows=[{'name':n,'sha256':d['sha256'],'pass':pinok(d)} for n,d in {**p['inputs'],**p['source_pins'],**{k:p[k] for k in ('protocol','target_protocol','parent_protocol','seed_candidate','linear_contract','linear_contract_metadata')}}.items()]
 check('plan/protocol/seed/source pins current',sha(planfile)=='015ecf45d35abf823d2c20d0dee7de5927bd93d91076a43bcdd24c131f4e24d4' and all(q['pass'] for q in pinrows) and sha(p['protocol']['path'])=='3c4113a60567321fdfc57fa201ad5edc194475d72493f840d1ac8ce71acb387c' and p['seed_candidate']['sha256']=='64aa2cfe44f20fee4546b07c4546916c95d57771404cb5fb4e39156206ad9b43',len(pinrows))
 check('normal complete canonical authority',run['status']=='completed_normal_R_F_geometry_only' and state['exit_code']==0 and not state['timeout'] and manifest['status']=='NORMAL_ALGORITHMIC_LAST_ACCEPTED_ONLY' and pinok(manifest['candidate']) and pinok(manifest['report']) and manifest['source_plan']==report['source_plan'] and pinok(manifest['source_plan']) and not (base/'failure.json').exists())
 check('pipeline final equals lastaccepted equals Rterminal candidate',pipeline['normal_termination'].item() is True and np.array_equal(pipeline['final_x'],pipeline['last_accepted_x']) and np.array_equal(pipeline['final_x'],pipeline['R_terminal_x']) and np.array_equal(pipeline['final_x'],z['optimizer_x']) and np.array_equal(pipeline['final_x'],R['x'][:340]) and pinok(report['pipeline_return']) and report['pipeline_return']==manifest['pipeline_return'])
 check('canonical R no F fit improvement claimed',run['F_accepted_steps']==0 and report['F_label']=='F_NO_ACCEPTED_STEP' and not report['fit_improvement_claim'] and report['canonical_candidate']['selected_phase']=='R_start_F_NO_ACCEPTED_STEP' and report['canonical_candidate']['selection']=='last_algorithmically_accepted_not_best_history' and pipeline['accepted_seq'].item()==0)
 seed=arrays(p['seed_candidate']['path'])['optimizer_x'];linear=arrays(p['linear_contract']['path']);A=csr(linear,'A_CSR_','A_shape');x=z['optimizer_x'];lr=A@(x-linear['x0'])-linear['b_centered'];box=np.maximum(np.maximum(linear['lower']-x,x-linear['upper']),0);trust=np.maximum(abs(x-seed)-.05,0)
 check('original linear artifact/absolute centered preserved',A.shape==(5960,340) and np.max(abs(linear['b_ub']-A@linear['x0']-linear['b_centered']))<1e-12 and np.array_equal(seed,z['restoration_seed_optimizer_x']))
 check('raw linear exactfalse but declaredtoltrue retained',np.max(lr)>0 and np.max(lr)<=1e-12 and abs(float(np.max(lr))-report['final_original_primal']['max_linear_Ax_minus_b_native'])<1e-20 and report['final_original_primal']['exact_raw']['linear'] is False and report['final_original_primal']['declared_arithmetic']['linear'] is True,float(lr.max()))
 with np.load(p['inputs']['canonical_H123_q0']['path'],allow_pickle=False) as a:q0=a['joint_pos_rad'][0]
 check('340/P0/16/61 bounds and Rtrust preserved',x.shape==(340,) and np.array_equal(z['control_points_rad'][1:].ravel(),x) and np.array_equal(z['control_points_rad'][0],q0) and np.array_equal(z['q_goal_rad'][0],q0) and np.array_equal(z['planned_q_goal_rad'][0],q0) and not np.any(box) and not np.any(trust))
 init=arrays(base/'R_requested_initialization.npz');first=arrays(base/'R_solver_initialization.npz');expectedslack=max(0.,float(seed.size and arrays(p['seed_candidate']['path'])['constraint_scaled_mm'].max()-2.999))+1e-6
 check('R fixedseed z0/slack normal return',np.array_equal(init['seed_x'],seed) and np.array_equal(init['z0'][:340],seed) and abs(float(init['z0'][-1])-expectedslack)<1e-12 and np.array_equal(first['requested_z0'],init['z0']) and R['x'].shape==(341,) and R['x'][-1]==Rt['slack_mm']==0 and Rt['optimizer']['nit']==3 and Rt['optimizer']['nfev']==4 and pinok(Rt['optimizer']['raw_return']))
 counts=run['counts'];gcounts=counts['geometry'];owners=dict(Counter(e['owner'] for e in events));stages=dict(Counter(e['stage'] for e in events))
 check('424 single-global geometry count owner/stages not doubled',len(events)==gcounts['actual_collision_calls']==gcounts['fit_parent']['actual_collision_calls']==424 and gcounts['events_recorded']==424 and sum(stages.values())==424 and stages==gcounts['collision_calls_by_stage'] and owners=={'fit16':61,'constraint61':363} and [e['query_index'] for e in events]==list(range(1,425)),{'owners':owners,'stages':stages})
 check('two independent fresh61 decisions',counts['forced_61_constraint_refreshes']==2 and stages['R_terminal_independent_fresh61']==61 and stages['final_last_accepted_fresh61']==61 and Rt['actual_creation_query_delta']['constraint61']==61)
 check('R1 and QP1 finite budgets/no alpha/acceptance',run['R_SLSQP_calls']==run['F_QP_calls']==1 and run['R_objective_fun_calls']==4 and run['R_callbacks']==3 and run['F_major_count']==1 and run['QP_objective_fun_calls']==65 and run['QP_objective_jac_calls']==26 and run['QP_callback_calls']==25 and run['F_alpha_trials']==run['F_actual_trials']==run['F_accepted_steps']==0 and run['LP_calls']==0 and not run['GPU_used'] and run['physics_steps']==0 and all(v==0 for v in run['forbidden_calls'].values()))
 Q=arrays(base/'F_majors/01/fixed_QP_model.npz');qret=arrays(base/'F_majors/01/QP_optimizer_return.npz');qmeta=read(base/'F_majors/01/QP_terminal.json');C=csr(Q,'C_','C_shape');step=qret['x'];lv=A@step-Q['linear_rhs'];dv=C@step-Q['geometry_rhs'];bv=np.maximum(np.maximum(Q['step_lower']-step,step-Q['step_upper']),0);slope=float(Q['g']@step)
 check('QP25 return rejected on original and depth linearization',qmeta['optimizer']['nit']==25 and qmeta['optimizer']['status']==9 and qmeta['proposal']['eligible'] is False and lv.max()>1e-12 and dv.max()>1e-9 and not np.any(bv) and slope<0 and abs(float(lv.max())-qmeta['proposal']['max_original_linear_residual_native'])<1e-18 and abs(float(dv.max())-qmeta['proposal']['max_linearized_depth_residual_mm'])<1e-18 and pinok(qmeta['optimizer']['raw_return']),{'max_original_linear_excess_native':float(lv.max()),'max_linearized_depth_excess_mm':float(dv.max()),'g_dot_step':slope})
 check('zero actual trial events and only R algorithmicaccept',not any(e['kind']=='F_ACTUAL_ALPHA_TRIAL' for e in trace) and not any(e['kind']=='F_ALGORITHMIC_ACCEPT_FIRST_VALID_ALPHA' for e in trace) and sum(e['kind']=='R_TERMINAL_BECOMES_ALGORITHMIC_START' for e in trace)==1 and report['F_stop']=='QP_TERMINAL_NO_RELIABLE_DESCENT_PROPOSAL')
 archive_refs={e['state']['path']:e['state'] for e in trace};archiveissues=[]
 for ref in archive_refs.values():
  if not pinok(ref):archiveissues.append(ref['path'])
  record=read(ref['path']);az=arrays(record['state_npz']['path'])
  if not pinok(record['state_npz']) or not pinok(record['native_contacts']) or hashlib.sha256(az['optimizer_x'].tobytes()).hexdigest()!=record['x_sha256'] or az['archive_pid'].item()!=run['pid'] or az['archive_plan_sha256'].item()!=run['plan_sha256']:archiveissues.append(ref['path']+'/binding')
 check('six archived actual states source/PID/vector bindings',not archiveissues and len(archive_refs)==run['true_state_archives']==6,archiveissues)
 contacts=read(base/'native_contacts.json')['planned61'];depth=[];pairissues=[];lookup={tuple(v):i for i,v in enumerate(z['native_pair_indices'])}
 for i in range(61):
  rs=[c for c in contacts if c['sample']==i];buckets={}
  for c in rs:
   key=tuple(sorted(c['geom']))
   if key in lookup:buckets.setdefault(lookup[key],[]).append(c['distance_m'])
  if set(buckets)!=set(np.flatnonzero(z['planned_native_generated_pair_mask'][i])):pairissues.append([i,'mask'])
  for row,dist in buckets.items():
   if min(dist)!=z['planned_native_selected_contact_distance_m'][i,row] or len(dist)!=z['planned_native_contact_count_by_pair'][i,row]:pairissues.append([i,row])
  depth.append([max([max(-c['distance_m'],0.) for c in rs if c['category']==kind],default=0.)*1000 for kind in ('hand_object','hand_self')])
 depth=np.array(depth);maxdepth=float(depth.max());check('122 sampled depth regenerated saved contacts not queried',not pairissues and np.max(abs(depth-z['constraint_scaled_mm'].reshape(61,2)))<1e-12 and maxdepth<=2.999+1e-9 and abs(maxdepth-report['final_original_primal']['max_depth_mm'])<1e-12,pairissues)
 check('normal is still reference-only not physical/fullmotion',report['physical_reference_qualified'] is False and report['full_motion_qualified'] is False and report['controller_FP32_qualification']=='PENDING_INDEPENDENT_EXPORT' and not run['Torch_imported'] and not run['Warp_imported'])
 obs=state['actual_process_OS_observation'];check('watch/runtime actualPID and terminal owned identity',obs['status']=='exact_OS_identity_observed' and run['pid']==obs['pid']==state['actual_pid'] and run['parent_pid']==obs['parent_pid']==state['launcher_pid'] and read(watch/'post_terminal_CIM.json')['status']=='NO_OWNED_PID_REMAINING')
 out=ROOT/'outputs/RF_saved_independent_review_v1';out.mkdir(parents=True,exist_ok=False);result={'status':'PASS_RF_SAVED_INTEGRITY_R_CANONICAL_NO_F_IMPROVEMENT' if all(c['pass'] for c in checks) else 'FAIL_RF_SAVED_INTEGRITY','checks':checks,'source_input_pins':pinrows,'auditor_sha256':sha(__file__),'candidate_sha256':sha(base/'candidate.npz'),'plan_sha256':sha(planfile),'sampled_max_generated_depth_mm':maxdepth,'raw_linear_excess_native':float(lr.max()),'declared_arithmetic_primal_pass':True,'raw_exact_linear_pass':False,'canonical_phase':'R','F_accepted_steps':0,'R_vs_F_selection':'R terminal only, not best historical cost/violation','top_geometry_total':424,'owner_events':owners,'two_fresh61':True,'runtime_wall_seconds':run['wall_seconds'],'nested_watch_wall_seconds':state['elapsed_s'],'nested_walls_not_added':True,'new_optimizer_calls':0,'new_geometry_queries':0,'physics_steps':0,'GPU_used':False,'scope':'Saved-only source/linear/geometry/algorithmicselection audit; not controller-FP32/dynamics/physical/full151 qualification.'}
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[c['name'] for c in checks if not c['pass']],'candidate_sha256':result['candidate_sha256'],'maxdepth_mm':maxdepth,'rawlinear':float(lr.max())},indent=2));assert all(c['pass'] for c in checks)
if __name__=='__main__':main()
