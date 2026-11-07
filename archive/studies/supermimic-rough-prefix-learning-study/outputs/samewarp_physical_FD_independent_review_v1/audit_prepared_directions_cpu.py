"""Independent NumPy/stdlib reconstruction of saved, fixed FD directions.

Does not import/call production direction code, generate new production data,
load a model, or run any simulator/framework. Existing synthetic24 is not rerun.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'outputs/current104_physical_directions_CPU_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':p.as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def loadnp(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


plan_path=ROOT/'configs/current104-physical-directions-CPU-v1.json'
plan=read(plan_path);report=read(DATA/'report.json')
arrays=loadnp(DATA/'directions_and_FP32_targets.npz');before=pin(DATA/'directions_and_FP32_targets.npz')
checks={};put=lambda k,v:checks.__setitem__(k,bool(v))
put('registered_CPU_no_physics_plan',plan['schema']=='rough_physical_probe_direction_CPU_preparation:1'
    and plan['GPU_authorized'] is False and plan['physical_branches_authorized'] is False)
put('frozen_scope_order_seed_eps_guard',plan['path_order']==['A0','A1L','P_ONLY','A2'] and plan['phase_order']==[0,4,8,12]
    and plan['seed']==2026110401 and plan['direction_count']==8 and plan['epsilons']==[.002,.001] and plan['guard_ulps']==8)
put('all_explicit_registered_inputs_source_pins_current',all(pin(p['path'])['sha256']==p['sha256'] for p in plan['pins'].values()))
put('existing24_boundary_proof_source_bound',plan['pins']['CPU_boundary_contract']['sha256']==pin(plan['pins']['CPU_boundary_contract']['path'])['sha256']
    and read(plan['pins']['CPU_boundary_contract']['path'])['check_count']==24 and read(plan['pins']['CPU_boundary_contract']['path'])['passed'] is True)
put('prepared_report_plan_dataset_pins',report['plan']['sha256']==pin(plan_path)['sha256'] and report['dataset']['sha256']==before['sha256'])
put('prepared_meta_no_WM_physics_or_signal_selection',report['status']=='PREPARED_NOT_PHYSICALLY_QUALIFIED' and report['direction_instances']==128
    and report['potential_FD_pairs']==256 and report['WM_forwards']==report['physics_steps']==0 and report['GPU_used'] is False
    and report['selection_uses_model_outputs'] is False and report['selection_uses_physical_perturbation_outputs'] is False
    and report['targets_clipped_or_resampled'] is False)
saved=loadnp(plan['pins']['saved_predictions']['path']);centers=[];keys=[]
for path in plan['path_order']:
    events=read(plan['pins'][path+'_control_events']['path'])
    for k in plan['phase_order']:
        centers.append(np.asarray(events[k]['actualu'][0],np.float32));keys.append((path,k))
        put(path+'_u_saved_events_match',np.array_equal(np.asarray([e['actualu'][0] for e in events],np.float32),saved[path+'__u']))
centers=np.asarray(centers,np.float32)
put('all16_centers_saved_bits_order_exact',np.array_equal(centers,arrays['centers']) and [(r['path'],r['phase']) for r in report['rows']]==keys)
descriptors=[]
for path in plan['path_order']:
    run=Path(plan['pins'][path+'_control_events']['path']).parent
    descriptors.append(read(run/'controller_descriptor.json'))
outerlo=np.asarray(descriptors[0]['target_lower_rad'],np.float32);outerhi=np.asarray(descriptors[0]['target_upper_rad'],np.float32)
lo=outerlo+np.float32(.020);hi=outerhi-np.float32(.020)
put('same_frozen_outer_bounds_all4_and_inner_FP32',all(np.array_equal(np.asarray(d['target_lower_rad'],np.float32),outerlo)
    and np.array_equal(np.asarray(d['target_upper_rad'],np.float32),outerhi) for d in descriptors)
    and np.array_equal(lo,arrays['lower']) and np.array_equal(hi,arrays['upper']))
cf=centers.astype(np.float64);lf=lo.astype(np.float64);hf=hi.astype(np.float64)
guard=8*np.finfo(np.float32).eps*np.maximum.reduce((np.ones_like(cf),np.abs(cf),np.broadcast_to(np.abs(lf),cf.shape),np.broadcast_to(np.abs(hf),cf.shape)))
slack=np.minimum(cf-lf,hf-cf);free=slack>=.002+guard
put('8eps32_coordinate_guard_exact',np.array_equal(guard,arrays['guard']))
put('freecoords_and_slack_exact_no_fresh_response_mask',np.array_equal(slack,arrays['slack']) and np.array_equal(free,arrays['free_mask']))
raw=np.random.default_rng(2026110401).standard_normal((16,8,20))
put('raw_PRNG_state_direction_coordinate_order_exact',np.array_equal(raw,arrays['standard_normals']))
masked=raw*free[:,None,:];norms=np.linalg.norm(masked,axis=-1)
unit=masked/norms[...,None]
put('all128_actual_vectors_nonzero',np.isfinite(norms).all() and bool((norms>0).all()))
put('masked_normalized_FP64_vectors_exact',np.array_equal(unit,arrays['vectors']))
put('inactive_coordinates_exact0_and_unitL2',np.all(unit*np.logical_not(free[:,None,:])==0)
    and np.max(np.abs(np.linalg.norm(unit,axis=-1)-1))<8*np.finfo(np.float64).eps)
eps=np.asarray([.002,.001],np.float64)
plus=(cf[:,None,None,:]+eps[None,None,:,None]*unit[:,:,None,:]).astype(np.float32)
minus=(cf[:,None,None,:]-eps[None,None,:,None]*unit[:,:,None,:]).astype(np.float32)
put('both_epsilon_FP32_plus_minus_exact_no_clamp',np.array_equal(plus,arrays['plus']) and np.array_equal(minus,arrays['minus']) and np.array_equal(eps,arrays['epsilons']))
cross=(plus<lo)|(plus>hi)|(minus<lo)|(minus>hi)
put('all256_domain_masks_recomputed_valid',np.array_equal(cross,arrays['crossing_mask']) and not cross.any()
    and arrays['valid_pair'].shape==(16,8,2) and arrays['valid_pair'].all() and report['valid_domain_pairs']==256)
effective=(plus.astype(float)-minus.astype(float))/(2*eps[None,None,:,None])
put('effective_FP32_secant_directions_exact',np.array_equal(effective,arrays['effective_vectors']))
put('all_FP32_pairs_changed_not_noop',np.all(np.linalg.norm(plus.astype(float)-minus.astype(float),axis=-1)>0))
rows=[]
for i,(path,k) in enumerate(keys):
    r=report['rows'][i];indices=np.flatnonzero(free[i]).tolist()
    put(f'{path}_{k}_row_free_count_and_all16pairs',r['free_coordinate_count']==len(indices) and r['free_coordinate_indices']==indices
        and r['potential_pair_count']==r['valid_pair_count']==16 and r['status']=='PREPARED_NOT_SIMULATED')
    rows.append({'path':path,'phase':k,'free_coordinate_count':len(indices),'direction_rank':int(np.linalg.matrix_rank(unit[i])),
        'max_effective_vs_nominal_component':float(np.max(np.abs(effective[i]-unit[i,:,None,:]))),
        'max_quantized_midpoint_drift_rad':float(np.max(np.abs((plus[i].astype(float)+minus[i].astype(float))*.5-cf[i]))),
        'min_masked_raw_norm':float(norms[i].min())})
put('review_no_original_mutation',pin(DATA/'directions_and_FP32_targets.npz')==before)
put('review_no_framework_imports',not any(k in sys.modules for k in ('torch','mujoco','warp','mujoco_warp')))
result={'schema':'physical_FD_prepared_directions_independent_CPU:1','status':'PASS_SAVED_PREPARATION_ONLY' if all(checks.values()) else 'FAIL_RETAINED',
    'checks':checks,'passed':sum(checks.values()),'total':len(checks),'rows':rows,'input_plan':pin(plan_path),'input_report':pin(DATA/'report.json'),'input_arrays':before,
    'reviewer_source':pin(__file__),'production_generator_reexecuted':False,'existing24_tests_reexecuted':False,'WM_or_physics_calls':0,'GPU_used':False,
    'scope':'Only fixed saved direction artifact math/source binding. No fresh donor u/state yet; physical experiment remains unauthorized/unqualified.',
    'general_policy_difference':'Proposal says numerically-zero masked vector is recorded with omitted branches; module raises when a free state yields zero norm. Actual fixed128 vectors all nonzero, so this does not alter this artifact. Resolve fail/omit wording before a generalized worker contract.',
    'direction_N':'Each snapshot supplies8 direction vectors in a conditional free-coordinate subspace. Rank/repetitions are not independent source or training N.'}
target=OUT/'directions_review.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(target)},indent=2))
raise SystemExit(0 if all(checks.values()) else 2)
