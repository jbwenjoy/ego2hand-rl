"""Terminal-artifact-only constrained-solve audit; never reads live by default."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import sparse
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--audit',action='store_true');args=ap.parse_args()
 if not args.audit:print(json.dumps({'status':'PREPARED_ONLY_NO_ARTIFACT_READ','LP_calls':0,'optimizer_calls':0,'geometry_queries':0}));return
 base=args.input.resolve();watch=base.with_name(base.name+'_watchdog');state=read(watch/'state.json')
 if state['status'] not in ('completed','failed','timeout','timeout_cleanup_unverified','failed_cleanup_unverified'):raise RuntimeError('Refuse active/ambiguous watchdog state')
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);checks=[]
 def check(n,ok,d=None):checks.append({'name':n,'pass':bool(ok),'details':d})
 p=read(args.plan);run=read(base/'runtime.json');report=read(base/'report.json') if (base/'report.json').is_file() else {};events=[json.loads(s) for s in (base/'collision_events.jsonl').read_text().splitlines() if s.strip()]
 check('terminal snapshot and exact plan binding',sha(args.plan)==args.plan_sha256 and run['plan_sha256']==args.plan_sha256 and read(base/'plan_snapshot.json')==p)
 pinresults=[{'name':n,'pass':sha(v['path'])==v['sha256'],'sha256':v['sha256']} for n,v in {**p['inputs'],**p['source_pins'],'protocol':p['protocol'],'parent_protocol':p['parent_protocol']}.items()]
 check('current source/input pins match',all(x['pass'] for x in pinresults),len(pinresults))
 counts=run['counts'];globalcount=counts['actual_collision_calls'];stages=dict(Counter(e['stage'] for e in events));owners=dict(Counter(e['owner'] for e in events))
 check('top global/event/stage counts equal without doubleadd',globalcount==counts['events_recorded']==len(events)==sum(stages.values()) and stages==counts['collision_calls_by_stage'] and [e['query_index'] for e in events]==list(range(1,globalcount+1)))
 check('shared fit_parent g count equals global, not fit-owned',counts['fit_parent']['actual_collision_calls']==globalcount and owners.get('fit16',0)+owners.get('constraint61',0)==globalcount,{'global':globalcount,'fit_owned_events':owners.get('fit16',0),'constraint_owned_events':owners.get('constraint61',0),'fit_parent_shared_global':counts['fit_parent']['actual_collision_calls']})
 obs=state.get('actual_process_OS_observation',{})
 check('runtime actual/launcher/CIM identity',obs.get('status')=='exact_OS_identity_observed' and run['pid']==state['actual_pid']==obs['pid'] and run['parent_pid']==state['launcher_pid']==obs['parent_pid'])
 check('oneLP at most oneSLSQP/CPU/no physics',run['LP_calls']==1 and run['SLSQP_calls'] in (0,1) and run['objective_fun_calls']<=250 and run['GPU_used'] is False and run['physics_steps']==0 and not run['Torch_imported'] and not run['Warp_imported'] and all(x==0 for x in run['forbidden_calls'].values()))
 linear=arrays(base/'linear_contract.npz');A=sparse.csr_matrix((linear['A_CSR_data'],linear['A_CSR_indices'],linear['A_CSR_indptr']),shape=linear['A_shape'])
 check('5960x340 CSR centered/absolute identity',A.shape==(5960,340) and np.isfinite(A.data).all() and np.max(abs(linear['b_ub']-A@linear['x0']-linear['b_centered']))<1e-10)
 lpmeta=read(base/'phase0_LP.json');lpcheck=None
 if (base/'phase0_LP_solution.npz').is_file():
  lp=arrays(base/'phase0_LP_solution.npz');solution=lp['solution'];x,t=solution[:340],solution[340:];ineq=np.r_[linear['b_ub']-A@x,linear['x0']-x+t,-linear['x0']+x+t]
  lpcheck={'max_linear_excess':float(np.max(A@(x-linear['x0'])-linear['b_centered'])),'max_box_excess':float(np.maximum(np.maximum(linear['lower']-x,x-linear['upper']),0).max()),'sum_epigraph':float(t.sum())}
  check('raw LP solution/epigraph residuals preserved',solution.shape==(680,) and np.max(abs(ineq-lp['inequality_residual']))<1e-9 and np.array_equal(lp['requested_repeat_q0'],linear['x0']) and (lpmeta['fun'] is None or abs(float(t.sum())-lpmeta['fun'])<1e-8))
 canonical=(base/'candidate.npz').is_file();derived={}
 if canonical:
  z=arrays(base/'candidate.npz');contacts=read(base/'native_contacts.json');manifest=read(base/'manifest.json');returned=arrays(base/'optimizer_return.npz')
  x=z['optimizer_x'];points=z['control_points_rad'];q=z['q_goal_rad'];q61=z['planned_q_goal_rad'];depth=z['constraint_scaled_mm'];limit=float(z['planning_depth_bound_m'])/.001
  with np.load(p['inputs']['canonical_H123_q0']['path'],allow_pickle=False) as s:q0=s['joint_pos_rad'][0]
  check('canonical only terminal SLSQP x not historicalbest',np.array_equal(returned['x'],x) and np.array_equal(x,points[1:].ravel()) and report['canonical_selection'].startswith('Only normally returned terminal SLSQP result.x'))
  check('18points/16q/61q fixed q0 exact',points.shape==(18,20) and q.shape==(16,20) and q61.shape==(61,20) and np.array_equal(points[0],q0) and np.array_equal(q[0],q0) and np.array_equal(q61[0],q0))
  delta=points[1:]-q0
  check('basis grouping and original repeatq0 seed',np.array_equal(linear['x0'],np.tile(q0,17)) and np.array_equal(z['requested_solver_x0'],linear['x0']) and np.allclose(q,q0+z['basis16'][:,1:]@delta,atol=1e-14,rtol=0) and np.allclose(q61,q0+z['basis61'][:,1:]@delta,atol=1e-14,rtol=0))
  check('candidate/manifest/report SHA+newvariant',sha(base/'candidate.npz')==manifest['candidate']['sha256']==report['candidate_sha256'] and manifest['protocol']==p['protocol'] and manifest['parent_protocol']==p['parent_protocol'] and manifest.get('retarget_variant_id',p['retarget_variant_id'])==p['retarget_variant_id'])
  lres=np.asarray(A@(x-linear['x0'])-linear['b_centered']);bex=np.maximum(np.maximum(linear['lower']-x,x-linear['upper']),0)
  operational=bool(np.all(depth<=limit));primal=operational and bool(np.all(lres<=0)) and bool(np.all(bex==0))
  check('terminal primal recomputed independently',primal==report['primal_operational_plan_exact_float64_feasible'] and abs(float(lres.max())-report['linear_float64_audit']['max_Ax_minus_b_centered'])<1e-11 and operational==report['sampled_generated_overlap_constraints_pass'])
  maxraw=0.;pairissues=[];summary=[];catalog={tuple(pair):i for i,pair in enumerate(z['native_pair_indices'])}
  for sample in range(61):
   records=[c for c in contacts['planned61'] if c['sample']==sample];pairs={}
   for c in records:
    key=tuple(sorted(c['geom']))
    if key in catalog:pairs.setdefault(catalog[key],[]).append(c['distance_m'])
   mask=z['planned_native_generated_pair_mask'][sample]
   if set(pairs)!=set(np.flatnonzero(mask)):pairissues.append([sample,'generatedmask'])
   for row,dist in pairs.items():
    if min(dist)!=z['planned_native_selected_contact_distance_m'][sample,row] or len(dist)!=z['planned_native_contact_count_by_pair'][sample,row]:pairissues.append([sample,row,'distance/count'])
   per=[]
   for category in ('hand_object','hand_self'):
    ds=[max(-c['distance_m'],0.) for c in records if c['category']==category];per.append(max(ds,default=0.))
   if np.max(abs(np.array(per)-z['operational_generated_maxdepth_m'][sample]))>1e-15:pairissues.append([sample,'categorymax'])
   maxraw=max(maxraw,*per);summary.append({'sample':sample,'HO_m':per[0],'self_m':per[1]})
  check('61 saved native records regenerate depths/pairs',not pairissues,pairissues)
  check('absent gaps UNKNOWN/no forceclaim',np.isnan(z['planned_native_selected_contact_distance_m'][~z['planned_native_generated_pair_mask']]).all() and report['physical_reference_qualified'] is False and report['continuous_clearance_qualified'] is False)
  cp=arrays(base/'latest_iterate.npz');header=read(base/'latest_iterate.json')
  check('latest checkpoint actualPID/plan/SHA bound',sha(base/'latest_iterate.npz')==header['npz_sha256'] and cp['checkpoint_plan_sha256'].item()==args.plan_sha256 and cp['checkpoint_pid'].item()==run['pid'] and cp['checkpoint_parent_pid'].item()==run['parent_pid'] and cp['checkpoint_is_feasible_accepted'].item() is False)
  check('normal termination fun/jac budget metadata',report['nfev']==run['objective_fun_calls'] and report['njev']==run['objective_jac_calls'] and report['nit']<=80 and run['callback_calls']<=80)
  derived={'candidate_sha256':sha(base/'candidate.npz'),'primal_operational_plan_exact_float64_feasible':primal,'optimizer_success':report['optimizer_success'],'terminal_classification':report['terminal_classification'],'linear_max_excess':float(lres.max()),'box_max_excess':float(bex.max()),'sampled_max_generated_depth_m':maxraw,'sampled_planning_pass':operational,'sampled_original3mm_pass':maxraw<=.003,'depths61':summary,'physical_reference_qualified':False,'FP32_controller':'Requires independent finalexport; no physical/init qualification inferred'}
 else:check('partial/no canonical preserved honestly',report.get('canonical_candidate') is None and not (base/'manifest.json').exists())
 result={'schema':'constrained_saved_independent_audit:1','status':'PASS_SAVED_ARTIFACT_INTEGRITY_ONLY' if all(x['pass'] for x in checks) else 'FAIL_SAVED_ARTIFACT_INTEGRITY','checks':checks,'source_pin_results':pinresults,'plan_sha256':args.plan_sha256,'auditor_sha256':sha(__file__),'LP_saved_metadata':lpmeta,'LP_independent_residual':lpcheck,'canonical_candidate_exists':canonical,'derived':derived,'top_global_geometry_calls':globalcount,'owner_events':owners,'stage_events':stages,'fit_parent_counter_scope':'Shared g global, never add to topglobal or stageeventsum','actual_worker_OS_identity':obs,'runtime_wall_seconds':run['wall_seconds'],'watchdog_nested_elapsed_seconds':state.get('elapsed_s'),'nested_walls_not_additive':True,'new_LP_calls':0,'new_optimizer_calls':0,'new_geometry_queries':0,'physics_steps':0,'GPU_used':False}
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[x['name'] for x in checks if not x['pass']],'top_global_geometry_calls':globalcount,'canonical_candidate_exists':canonical,'derived':{k:v for k,v in derived.items() if k!='depths61'}},indent=2));assert all(x['pass'] for x in checks)
if __name__=='__main__':main()
