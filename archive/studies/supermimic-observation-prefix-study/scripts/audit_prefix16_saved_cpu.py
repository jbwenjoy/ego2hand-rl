"""Artifact-only audit of the one formal prefix solve; no model/solver imports."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 base=ROOT/'outputs/prefix16_geometry_v1';watch=base.with_name(base.name+'_watchdog');planpath=ROOT/'configs/prefix16-geometry-authorized-v1.json'
 plan=read(planpath);runtime=read(base/'runtime.json');report=read(base/'report.json');state=read(watch/'state.json');contacts=read(base/'native_contacts.json')
 z=arrays(base/'candidate.npz');init=arrays(base/'solver_initialization.npz');cp=arrays(base/'latest_accepted_iterate.npz');head=read(base/'latest_accepted_iterate.json')
 events=[json.loads(x) for x in (base/'collision_events.jsonl').read_text().splitlines() if x.strip()];hist=[json.loads(x) for x in (base/'optimizer_iterations.jsonl').read_text().splitlines() if x.strip()]
 out=ROOT/'outputs/prefix16_raw_independent_review_v2';out.mkdir(parents=True,exist_ok=False);checks=[]
 def check(n,ok,details=None):checks.append({'name':n,'pass':bool(ok),'details':details})
 expectedplan='52336e48ea0a5e2e74215d930da4a4c15255cbdf9cb8b395cc221f02ee7c5b6f';candsha='1e67d99b25b6d991fa8e5209f521ea5c269b3212a45b054fe7353ab5b7277ae4'
 check('authorized plan/candidate SHA',sha(planpath)==expectedplan and sha(base/'candidate.npz')==candsha and report['candidate_sha256']==candsha)
 pinresults=[{'name':n,'path':p['path'],'pass':sha(p['path'])==p['sha256']} for n,p in {**plan['inputs'],**plan['source_pins'],'protocol':plan['protocol']}.items()]
 check('all actual plan source/input pins match',all(p['pass'] for p in pinresults),len(pinresults))
 check('plan snapshot exact',read(base/'plan_snapshot.json')==plan and sha(watch/'plan_snapshot.json')==expectedplan)
 check('single optimization/CPU/no physics',runtime['optimization_runs']==report['optimization_runs']==1 and runtime['status']=='completed_formal_CPU_geometry_only' and not runtime['GPU_used'] and runtime['physics_steps']==0 and not runtime['Torch_imported'] and not runtime['Warp_imported'] and all(v==0 for v in runtime['forbidden_calls'].values()))
 check('fun46 jac37 callback36 consistent',runtime['fun_calls']==report['nfev']==46 and runtime['jac_calls']==report['njev']==37 and runtime['callback_calls']==36 and len(hist)==46 and [r['call'] for r in hist]==list(range(1,47)))
 obs=state['actual_process_OS_observation']
 check('actual worker and redirector identity bound',runtime['pid']==state['actual_pid']==obs['pid']==37020 and runtime['parent_pid']==state['launcher_pid']==obs['parent_pid']==41752 and obs['status']=='exact_OS_identity_observed' and runtime['plan_sha256']==report['plan_sha256']==expectedplan)
 check('watchdog completed no timeout no remaining recordedPID',state['status']=='completed' and state['exit_code']==0 and not state['timeout'] and read(watch/'post_terminal_CIM.json')['status']=='NO_OWNED_PID_REMAINING')
 counts=runtime['counts'];stages=dict(Counter(x['stage'] for x in events))
 check('all706 collision events complete unique indexed',len(events)==counts['actual_collision_calls']==counts['events_recorded']==706 and [e['query_index'] for e in events]==list(range(1,707)))
 check('geometry stage16 plus690; no solver force calls',stages==counts['collision_calls_by_stage']=={'preflight_initial16':16,'formal_fun':690} and len(events[:16])==16 and sorted(e['frame'] for e in events[:16])==list(range(16)))
 check('evaluation/cache arithmetic honest',counts['evaluation_requests']==counts['full_evaluations']+counts['global_cache_hits'] and counts['full_evaluations']==47 and counts['frame_cache_hits']==46)
 with np.load(plan['inputs']['canonical_H123_q0']['path'],allow_pickle=False) as s:q0=s['joint_pos_rad'][0]
 check('16x20/q0 exact/initial16 repeat',z['q_goal_rad'].shape==(16,20) and np.array_equal(z['q_goal_rad'][0],q0) and np.array_equal(z['q0_fixed_rad'],q0) and np.array_equal(z['initial_guess_q_goal_rad'],np.repeat(q0[None],16,axis=0)))
 check('only free x0 shifted by TRF',np.array_equal(init['requested_x0'],np.tile(q0,15)) and np.array_equal(init['q0_source64'],q0) and np.array_equal(init['free_initialization_shift'],init['solver_first_fun_x']-init['requested_x0']) and np.array_equal(cp['solver_first_fun_x'],init['solver_first_fun_x']),float(np.max(abs(init['free_initialization_shift']))))
 check('finite q/residual/sparse values/hardq',np.isfinite(z['q_goal_rad']).all() and np.isfinite(z['residual']).all() and np.isfinite(z['jacobian_CSR_data']).all() and np.all(z['q_goal_rad']>=z['joint_lower_rad']) and np.all(z['q_goal_rad']<=z['joint_upper_rad']))
 check('native missing gaps remain unknown',np.all(np.isnan(z['native_selected_contact_distance_m'][~z['native_generated_pair_mask']])) and np.isfinite(z['native_selected_contact_distance_m'][z['native_generated_pair_mask']]).all())
 check('saved sparse CSR/frame0 block',z['jacobian_shape'].tolist()==[6596,300] and len(z['jacobian_CSR_indptr'])==6597 and np.all(z['jacobian_CSR_indptr'][:377]==0) and int(z['jacobian_CSR_indptr'][-1])==len(z['jacobian_CSR_data']) and np.all((z['jacobian_CSR_indices']>=0)&(z['jacobian_CSR_indices']<300)))
 common=set(z)&set(cp)
 check('accepted checkpoint payload equals candidate',all(np.array_equal(z[k],cp[k],equal_nan=z[k].dtype.kind in 'fc') for k in common))
 check('checkpoint NPZ SHA/embedded metadata/header bound',sha(base/'latest_accepted_iterate.npz')==head['checkpoint_npz_sha256'] and cp['checkpoint_plan_sha256'].item()==head['plan_sha256']==expectedplan and cp['checkpoint_protocol_sha256'].item()==head['protocol_sha256']==plan['protocol']['sha256'] and cp['checkpoint_pid'].item()==head['pid']==37020 and cp['checkpoint_parent_pid'].item()==head['parent_pid']==41752 and cp['checkpoint_source'].item()==head['source']=='final' and cp['checkpoint_accepted'].item() is True and cp['checkpoint_wall_seconds'].item()==head['wall_seconds'])
 check('saved objective norm exact',abs(float(.5*z['residual']@z['residual'])-report['cost'])<=1e-12)
 lookup={tuple(pair):i for i,pair in enumerate(z['native_pair_indices'])};frame_results=[];maxdepth={'hand_object':0.,'hand_self':0.};issues=[]
 for frame in range(16):
  records=[c for c in contacts if c['frame']==frame];pair_rows={}
  for c in records:
   pair=tuple(sorted(c['geom']))
   if pair in lookup:pair_rows.setdefault(lookup[pair],[]).append(c['distance_m'])
   if c['category'] in maxdepth:maxdepth[c['category']]=max(maxdepth[c['category']],max(-c['distance_m'],0))
  expected=z['native_generated_pair_mask'][frame]
  if set(np.flatnonzero(expected))!=set(pair_rows):issues.append([frame,'generated pair set'])
  for i,dist in pair_rows.items():
   if min(dist)!=z['native_selected_contact_distance_m'][frame,i] or len(dist)!=z['native_contact_count_by_pair'][frame,i]:issues.append([frame,int(i),'distance/count'])
  summary={}
  for kind in ('hand_object','hand_self'):
   records_kind=[c for c in records if c['category']==kind];pairs={tuple(sorted(c['geom'])) for c in records_kind};depth=max([max(-c['distance_m'],0) for c in records_kind],default=None)
   summary[kind]={'contacts':len(records_kind),'generated_pairs':len(pairs),'max_depth_m':depth}
   r=report['frames'][frame][kind]
   if r['generated_pairs']!=len(pairs) or r['max_depth_m']!=depth:issues.append([frame,kind,'summary'])
  if len(records)!=report['frames'][frame]['native_contact_count']:issues.append([frame,'contact_count'])
  frame_results.append({'frame':frame,**summary})
 check('raw contacts regenerate all16 saved pair/depth summaries',not issues,issues)
 check('geometry labels never treated as actual forces',all(c['phase']=='native_CPU_geometry_not_solved_force' and 'mount' not in ' '.join(c['names']) for c in contacts) and report['contacts_known'] is False and report['physical_reference_qualified'] is False)
 interpretation={'solver_termination':report['optimizer_message'],'optimizer_status':report['optimizer_status'],'optimality':report['optimality'],
  'ftol_is_not_gtol_or_global_solution':report['optimizer_status']==2 and report['optimality']>1e-8,
  'hand_object_depth_max_m':maxdepth['hand_object'],'hand_self_depth_max_m':maxdepth['hand_self'],
  'geometry_penetration3mm_gate_pass':max(maxdepth.values())<=.003,'physical_reference_qualified':False,
  'native_ABI_is_interface_only':'Independent CPU export/read does not qualify geometry, continuous curves, initialization, actual u or training'}
 final={'schema':'prefix16_saved_artifact_independent_review:1','status':'PASS_RAW_INTEGRITY_GEOMETRY_UNQUALIFIED' if all(x['pass'] for x in checks) else 'FAIL_RAW_INTEGRITY',
  'checks':checks,'source_input_pins':pinresults,'source_sha256':sha(__file__),'source_plan_sha256':expectedplan,'candidate_sha256':candsha,
  'measured_identity':{'worker_pid':37020,'launcher_pid':41752,'actual_OS_creation_UTC':obs['creation_date_utc'],'runtime_started_UTC':runtime['started_utc'],
   'scope':'OS creation and runtime initialization timestamps are distinct; no new OS query was performed'},
  'stage_counts':stages,'frames_from_saved_contacts':frame_results,'interpretation':interpretation,
  'worker_wall_seconds':runtime['wall_seconds'],'watchdog_nested_wall_seconds':state['elapsed_s'],'nested_walls_are_not_additive':True,
  'optimization_or_physics_or_collision_queries_by_audit':0,'GPU_used':False,'init_measured':None,'actual_u':None}
 (out/'report.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':final['status'],'checks':len(checks),'failed':[c['name'] for c in checks if not c['pass']],'interpretation':interpretation},indent=2))
 assert all(x['pass'] for x in checks)
if __name__=='__main__':main()
