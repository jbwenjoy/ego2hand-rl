"""Future isolated component checkpoint saved readback, not native105 health.

Default emits a declaration without Torch/data reads. --run loads exactly the
own trusted initial component + two requested component checkpoints on CPU with
weights_only=False. No model is constructed or forwarded; no native Log or AD.
"""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='full151105_WM_common_pre_component_calibration:1'
ARMS=('pointwise','pointwise_plus_common_pre_response')
RECIPE_FILE=ROOT/'protocols/full151105_WM_common_pre_calibration_recipe_v1.json'
RECIPE_SHA='ea22ab07643c48e9bea24cb4be708a326c88922a57e2a06b97d73dd5d317b5d9'
PRODUCER_SHA='71efe05da17a37df4111158860443433304f1d57762e2377d4dbafc3b57ca27f'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}


def checked(p):
    if type(p) is not dict or set(p)!={'path','sha256'} or type(p['path']) is not str or not Path(p['path']).is_absolute():raise ValueError('Exact absolute trusted pin')
    q=Path(p['path']).resolve(strict=True)
    if sha(q)!=p['sha256']:raise ValueError('Exact trusted pin required')
    return q


def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def tensor_equal(a,b,torch):
    return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.device.type==b.device.type=='cpu' and a.dtype==b.dtype and a.shape==b.shape and bits(a.detach().contiguous().numpy(),b.detach().contiguous().numpy())


def tensor_digest(values,torch):
    h=hashlib.sha256()
    for name,tensor in sorted(values.items()):
        if not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu':raise ValueError('CPU saved tensors only')
        h.update(name.encode());h.update(str((str(tensor.dtype),tuple(tensor.shape))).encode());h.update(tensor.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def finite_tree(v,torch):
    if isinstance(v,torch.Tensor):return v.device.type=='cpu' and (not(v.is_floating_point() or v.is_complex()) or bool(torch.isfinite(v).all()))
    if isinstance(v,dict):return all(finite_tree(x,torch) for x in v.values())
    if isinstance(v,(tuple,list)):return all(finite_tree(x,torch) for x in v)
    return math.isfinite(v) if type(v) is float else v is None or type(v) in (int,bool,str)


def norm_state(state):return {k:v for k,v in state.items() if k.startswith(('norm.','delta_norm.')) or k=='error_floor'}


def rng_digest(v,torch):
    if v.get('CUDA_initialized') is not False or v['torch_CPU'].dtype!=torch.uint8 or v['torch_CPU'].device.type!='cpu':raise ValueError('CPU RNG scope')
    raw=dict(v);raw['torch_CPU']=v['torch_CPU'].contiguous().numpy().tobytes()
    return hashlib.sha256(pickle.dumps(raw,protocol=5)).hexdigest()


def optimizer_readback(state,iteration,torch,parameter_values=None):
    ids=[i for g in state['param_groups'] for i in g['params']]
    if not ids or any(type(x) is not int for x in ids) or len(ids)!=len(set(ids)) or set(ids)!=set(state['state']):raise ValueError('Every unique optimizer parameter must have saved state')
    steps=[]
    if parameter_values is not None and len(parameter_values)!=len(ids):raise ValueError('Source-native net parameter count differs from serialized optimizer')
    for offset,i in enumerate(ids):
        entry=state['state'][i];step=entry['step']
        if not isinstance(step,torch.Tensor) or step.device.type!='cpu' or step.numel()!=1 or not bool(torch.isfinite(step).all()) or float(step)!=iteration:raise ValueError('Actual serialized Adam step differs')
        a,b=entry['exp_avg'],entry['exp_avg_sq']
        if a.shape!=b.shape or a.dtype!=b.dtype or not finite_tree(entry,torch) or not bool((b>=0).all()):raise ValueError('Finite same-shape Adam moments')
        if parameter_values is not None and (a.shape!=parameter_values[offset].shape or a.dtype!=parameter_values[offset].dtype):raise ValueError('Serialized Adam moment shape/dtype does not match source-native parameter order')
        steps.append(float(step))
    for g in state['param_groups']:
        if g['lr']!=1e-4 or g['weight_decay']!=0 or tuple(g['betas'])!=(.9,.999) or g['eps']!=1e-8:raise ValueError('Fixed AdamW recipe')
    return {'serialized_parameter_states':len(ids),'actual_step_values':sorted(set(steps)),'all_steps_equal_iteration':True,'scope':'INDEPENDENT_SERIALIZED_CPU_READBACK'}


def postprocess_saved_losses(e,torch):
    """Use captured native FP32 deltas; no F64 shadow Log substitutes for them."""
    for k,shape in [('pred152_FP32',(152,58)),('point_delta52_FP32',(152,52)),('pred_common52_FP32',(152,52)),('target_common52_FP32',(152,52)),('scale52_FP32',(52,))]:
        if e[k].dtype!=np.float32 or e[k].shape!=shape or not np.isfinite(e[k]).all():raise ValueError('Native loss evidence '+k)
    mi,pl=e['minus_indices'],e['plus_indices'];du=e['actual_du_rad_F64']
    if mi.dtype!=np.int64 or pl.dtype!=np.int64 or mi.shape!=(64,) or pl.shape!=(64,) or (mi<0).any() or (pl<0).any() or (mi>=152).any() or (pl>=152).any():raise ValueError('All64 index vectors')
    if mi.shape!=(64,) or pl.shape!=(64,) or du.shape!=(64,) or du.dtype!=np.float64 or not np.isfinite(du).all() or (du==0).any():raise ValueError('All nonzero actual denominators')
    if (e['scale52_FP32']<=0).any():raise ValueError('Positive frozen scale52')
    point=torch.from_numpy(e['point_delta52_FP32']);s=torch.from_numpy(e['scale52_FP32'])
    pc=torch.from_numpy(e['pred_common52_FP32']);tc=torch.from_numpy(e['target_common52_FP32']);h=torch.from_numpy(du)
    m,p=torch.from_numpy(mi),torch.from_numpy(pl)
    point_loss=float((point/s).square().mean())
    response=(pc[p].double()-pc[m].double())-(tc[p].double()-tc[m].double())
    pair_loss=float((response/s.double()/h[:,None]).square().mean())
    return {'pointwise_loss':point_loss,'common_pre_pair_loss':pair_loss,
       'native_rotvec_norm_max':max(float(np.linalg.norm(e[k][...,43:46].astype('f8'),axis=-1).max()) for k in ('point_delta52_FP32','pred_common52_FP32','target_common52_FP32'))}


def load_component_cpu(path,torch):
    """Caller must supply only its explicitly trusted new component files."""
    value=torch.load(path,map_location='cpu',weights_only=False)
    if type(value) is not dict or value.get('schema')!=SCHEMA:raise ValueError('Not a new isolated component artifact; old105/104 forbidden')
    return value


def validate_recipe_protocol(proto,require_authorized=False):
    if sha(RECIPE_FILE)!=RECIPE_SHA:raise ValueError('Frozen pre-response recipe changed')
    frozen=read(RECIPE_FILE)
    if proto['schema']!='full151105_WM_common_pre_calibration_CPU_protocol:1' or proto['role']!='WM_only_common_pre_finite_response_calibration' or proto['GPU_run_authorized'] is not False:raise ValueError('Isolated CPU component protocol only')
    if json.dumps(proto['recipe'],sort_keys=True)!=json.dumps(frozen['recipe'],sort_keys=True):raise ValueError('Exact pre-registered numerical recipe')
    if set(proto['inputs'])!=set(frozen['inputs']) or not all(same(v,frozen['inputs'][k]) for k,v in proto['inputs'].items()):raise ValueError('Frozen fit inputs, not fresh-test data')
    if require_authorized and not(proto['execution_authorized'] is True and proto['CPU_run_authorized'] is True):raise ValueError('Explicit Root run authorization required')
    for p in proto['source_pins'].values():checked(p)
    if not any(p['sha256']==PRODUCER_SHA for p in proto['source_pins'].values()):raise ValueError('Frozen CPU-tested calibration source absent')
    return frozen


def nonrotation_delta_evidence(e,data):
    # Source-native FP32 target-minus-input subtraction, excluding Log3.
    for name,a,b in [('point_delta52_FP32',data['target'],e['pred152_FP32']),
        ('pred_common52_FP32',e['pred152_FP32'],data['common_pre']),('target_common52_FP32',data['target'],data['common_pre'])]:
        if not bits(e[name][:,:43],a[:,:43]-b[:,:43]) or not bits(e[name][:,46:],a[:,52:]-b[:,52:]):return False
    return True


def audit(protocol_path,run,iteration,out):
    if type(iteration) is not int or iteration<100 or iteration%100:raise ValueError('Fixed saved component checkpoint100-multiple')
    out.mkdir(exist_ok=False);auditor=pin(__file__);(out/'executed_auditor_source.py').write_bytes(Path(__file__).read_bytes())
    write(out/'source_scope_before_readback.json',{'auditor':auditor,'trusted_payloads':'Own initial component and exactly two fixed iteration checkpoints; CPU weights_only=False',
      'models_Env_nativeLog_FW_AD_GPU':0,'native105_health_or_policy_qualification':False})
    proto=read(protocol_path);recipe=proto['recipe'];I=proto['inputs'];health=read(run/('checkpoint_%04d_health.json'%iteration))
    validate_recipe_protocol(proto,require_authorized=True)
    if iteration>recipe['iterations']:raise ValueError('Requested iteration exceeds fixed component budget')
    for p in [*I.values(),*proto['source_pins'].values()]:checked(p)
    import torch
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():raise ValueError('No CUDA before own payload readback')
    calls=[];loadtrace=[]
    def own_load(name):
        path=(run/name).resolve(strict=True)
        if not path.parent.samefile(run):raise ValueError('Own component path only')
        before=pin(path);value=load_component_cpu(path,torch);after=pin(path)
        if before!=after:raise ValueError('Own immutable checkpoint changed during read')
        calls.append(before);loadtrace.append({'checkpoint':before,'map_location':'cpu','weights_only':False,'returned':True,'source_bytes_unchanged':True})
        return value
    initial=own_load('initial_component_state.pt');payload={a:own_load('%s_update_%04d.pt'%(a,iteration)) for a in ARMS}
    checks={};details={}
    def gate(k,v):checks[k]=bool(v)
    validation=read(run/'initial_validation.json');initial_sha=tensor_digest(initial['WM'],torch);norm_sha=tensor_digest(norm_state(initial['WM']),torch)
    gate('new_component_initial_parent_and_strict_samebatch_validation',same(initial['parent_checkpoint105'],I['fixed_A2_checkpoint']) and validation['strict_loader_calls']==1 and validation['actual_torch_load_calls']==2 and validation['strict_frozen_validation_forwards']==1 and initial_sha==validation['initial_WM_sha256']==validation['strict_loader']['WM_state_sha256'])
    gate('own_initial_raw_WM_normalizers_xi_scale_finite',finite_tree(initial['WM'],torch) and finite_tree(initial['frozen_s52'],torch) and initial['frozen_s52'].shape==(52,) and bool((initial['frozen_s52']>0).all()) and initial['physical_xi46'].shape==(46,) and finite_tree(initial['physical_xi46'],torch) and norm_sha==validation['frozen_norm_sha256'])
    context=arrays(checked(I['fixed_A2_fixed_physics_context']))['reader_context'];xi=initial['physical_xi46'].numpy()
    gate('initial_raw_physicalxi46_matches_all_fixed_context_worlds',context.dtype==xi.dtype==np.float32 and context.ndim==2 and context.shape[1]==46 and bits(context,np.broadcast_to(xi,context.shape).copy()) and tensor_digest({'xi':initial['physical_xi46']},torch)==validation['xi46_sha256'])
    scale=torch.maximum(torch.sqrt(initial['WM']['delta_norm.var']+float(initial['normalizer_host_configuration']['delta_norm']['eps'])**2),initial['WM']['error_floor'])
    gate('frozen_s52_recomputed_native_FP32_std_floor',tensor_equal(scale,initial['frozen_s52'],torch))
    adam={};rng0=rng_digest(initial['RNG'],torch)
    for arm,p in payload.items():
        gate(arm+'_schema_identity_count_recipe',p['arm']==arm and p['iteration']==iteration and p['recipe']==recipe and same(p['protocol'],pin(protocol_path)) and same(p['parent_checkpoint105'],I['fixed_A2_checkpoint']) and same(p['parent_health'],I['fixed_A2_final_health']) and p['initial_WM_sha256']==initial_sha and p['frozen_norm_sha256']==norm_sha and p['counts']=={'WM_forwards':iteration,'backward_calls':iteration,'optimizer_steps':iteration} and p['native105_checkpoint_or_policy_resume_compatible'] is p['fresh_test_evaluated'] is False)
        # This frozen architecture's only parameters are net.*; norm modules
        # contain buffers only. state_dict preserves the native net order.
        params=[v for k,v in p['WM_state'].items() if k.startswith('net.')]
        adam[arm]=optimizer_readback(p['optimizer_state'],iteration,torch,params)
        gate(arm+'_serialized_parameters_optimizer_finite_and_Adam_steps',set(p['WM_state'])==set(initial['WM']) and all(isinstance(v,torch.Tensor) and v.dtype==initial['WM'][k].dtype and v.shape==initial['WM'][k].shape for k,v in p['WM_state'].items()) and finite_tree(p['WM_state'],torch) and finite_tree(p['optimizer_state'],torch))
        ae=p['actual_Adam_step_evidence'];gate(arm+'_inmemory_Adam_evidence_matches_new_serialized_readback',ae['scope']=='IN_MEMORY_NOT_SERIALIZED_READBACK' and ae['parameter_states']==adam[arm]['serialized_parameter_states'] and ae['actual_step_values']==adam[arm]['actual_step_values'] and ae['expected_iteration']==iteration and ae['all_actual_steps_equal_index'] is True)
        gate(arm+'_all_normbuffer_count_hostconfig_xi_scale_unchanged',set(norm_state(p['WM_state']))==set(norm_state(initial['WM'])) and all(tensor_equal(v,p['WM_state'][k],torch) for k,v in norm_state(initial['WM']).items()) and p['normalizer_host_configuration']==initial['normalizer_host_configuration'] and tensor_equal(p['physical_xi46'],initial['physical_xi46'],torch) and tensor_equal(p['frozen_s52'],initial['frozen_s52'],torch))
        gate(arm+'_serialized_raw_RNG_equal_initial_no_repair',rng_digest(p['RNG'],torch)==rng0==validation['RNG_sha256'])
        gate(arm+'_source_data_provenance_current_exact',all(same(p['data_pins'][k],I[k]) for k in p['data_pins']) and all(checked(v) for v in p['source_pins'].values()))
    data=arrays(run/'matched_fit_data152.npz');raw=arrays(checked(I['response_raw152']));pairs=read(checked(I['response_pairs64']));mi=np.asarray([x['minus_branch_id'] for x in pairs],dtype='i8');pl=np.asarray([x['plus_branch_id'] for x in pairs],dtype='i8')
    actual_du=np.asarray([float(raw['actual_u152_FP32'].reshape(152,20)[p['plus_branch_id'],p['joint_index']].astype('f8')-raw['actual_u152_FP32'].reshape(152,20)[p['minus_branch_id'],p['joint_index']].astype('f8')) for p in pairs],dtype='f8')
    gate('all152_fit_data_64pairs_actualdu_original_source',bits(data['pre'],raw['pre_visible152_FP32'].reshape(152,58)) and bits(data['target'],raw['target_y152_FP32'].reshape(152,58)) and bits(data['u'],raw['actual_u152_FP32'].reshape(152,20)) and bits(data['common_pre'],np.broadcast_to(raw['original_snapshot_visible58_FP32'][:,None],(8,19,58)).reshape(152,58).copy()) and bits(data['minus'],mi) and bits(data['plus'],pl) and bits(data['actual_du_rad_F64'],actual_du) and (actual_du>0).all())
    gate('all152_commonpre_q40_bits_other18_numeric_original_mask',np.array_equal(data['pre'][:,:40].view('u4'),data['common_pre'][:,:40].view('u4')) and np.array_equal(data['pre'][:,40:],data['common_pre'][:,40:]) and bits(data['same_pre_bits18_diagnostic'],raw['pre_visible152_FP32'][:,:,40:].view('u4')==raw['original_snapshot_visible58_FP32'][:,None,40:].view('u4')))
    # The job may continue; never parse a partial or later appended line.
    logs=[];prefix=hashlib.sha256()
    with (run/'updates.jsonl').open('rb') as f:
        for _ in range(iteration):
            line=f.readline()
            if not line.endswith(b'\n'):raise ValueError('Requested immutable log prefix incomplete')
            prefix.update(line);logs.append(json.loads(line))
    gate('prefix_all_lockstep_log_iterations_and_counters_finite',len(logs)==iteration and all(r['iteration']==i and set(r['arms'])==set(ARMS) and all(v['counts']=={'WM_forwards':i,'backward_calls':i,'optimizer_steps':i} and all(math.isfinite(v[k]) for k in ('pointwise_loss','common_pre_pair_loss','optimized_loss','pre_clip_gradient_norm','update_seconds')) and v['gradient_clipped']==(v['pre_clip_gradient_norm']>1.) for v in r['arms'].values()) for i,r in enumerate(logs,1)))
    initial_pred=arrays(run/'strict_initial_samebatch152_validation.npz');evidence={};arithmetic={}
    for arm in ARMS:
        for it in (1,100):
            if it>iteration:continue
            e=arrays(run/('%s_iteration_%04d_preupdate_loss_evidence.npz'%(arm,it)));evidence[(arm,it)]=e;z=postprocess_saved_losses(e,torch);row=logs[it-1]['arms'][arm]
            arithmetic[arm+'_'+str(it)]=z
            gate(arm+'_iteration%d_PREoptimizer_loss_arithmetic'%it,z['pointwise_loss']==row['pointwise_loss'] and z['common_pre_pair_loss']==row['common_pre_pair_loss'] and row['optimized_loss']==(z['pointwise_loss'] if arm==ARMS[0] else z['pointwise_loss']+z['common_pre_pair_loss']) and z['native_rotvec_norm_max']<math.pi-.01 and bits(e['actual_du_rad_F64'],actual_du) and bits(e['minus_indices'],mi) and bits(e['plus_indices'],pl) and bits(e['scale52_FP32'],initial['frozen_s52'].numpy()) and nonrotation_delta_evidence(e,data))
    gate('all_preupdate_evidence_uses_identical_constant_native_targetcommon',all(bits(e['target_common52_FP32'],evidence[(ARMS[0],1)]['target_common52_FP32']) for e in evidence.values()))
    gate('first_preupdate152_raw_predictions_equal_strict_frozen_batch',all(bits(evidence[(a,1)]['pred152_FP32'],initial_pred['predicted152_FP32']) for a in ARMS) and initial_pred['self15_FP32'].shape==(152,15) and initial_pred['self15_FP32'].dtype==np.float32 and np.isfinite(initial_pred['self15_FP32']).all())
    eta=health['ETA'];elapsed=eta['elapsed_fit_seconds_including_prior_saves'];remaining=(1000-iteration)*elapsed/iteration
    gate('health_is_inmemory_not_serialized_readback_and_ETA_formula',health['health_verification_scope']=='IN_MEMORY_BEFORE_SAVE_NOT_SERIALIZED_READBACK' and health['serialized_checkpoint_readback_verified'] is False and health['accuracy_or_physical_gradient_qualified'] is False and math.isfinite(elapsed) and elapsed>=0 and eta['remaining_updates_per_arm']==1000-iteration and eta['remaining_seconds_linear_heuristic']==remaining and eta['planning_range_seconds_not_CI']==[remaining*.5,remaining*1.8] and eta['unmeasured_finalization_overhead_seconds']==20.)
    gate('health_fixed_checkpoint_pins_and_source_assertions',health['iteration']==iteration and health['passed']==health['total']==len(health['checks']) and all(v is True for v in health['checks'].values()) and all(same(health['checkpoints'][a],calls[i+1]) for i,a in enumerate(ARMS)))
    sources=read(run/'source_inventory.json')
    for p in sources.values():checked(p)
    with zipfile.ZipFile(run/'execution_sources.zip') as z:zipok=len(z.namelist())==len(sources) and all(hashlib.sha256(z.read(k)).hexdigest()==p['sha256'] for k,p in sources.items())
    gate('all_current_immutable_source_archive_bytes_and_protocol',zipok and pin(__file__)==auditor and same(pin(protocol_path),payload[ARMS[0]]['protocol']))
    gate('reviewer_only_three_own_CPU_loads_no_model_native_Env_GPU',len(calls)==3 and not torch.cuda.is_initialized() and not any(k in sys.modules for k in ('warp','mujoco','mujoco_warp','dex_supermimic','mjlab')))
    write(out/'own_checkpoint_CPU_read_trace.json',loadtrace)
    for p in calls:checked(p)
    details.update(serialized_Adam=adam,preupdate_evidence_arithmetic=arithmetic,after_update_iteration=iteration,first_checkpoint_eta=eta,log_prefix_rows=iteration,log_prefix_sha256=prefix.hexdigest(),
       evidence_phase='iteration100 loss evidence is after99/preoptimizer; serialized checkpoint100 is after100. No forward from checkpoint100.',
       gradients='Raw parameter gradients not saved; finite actual gradient claims remain source/log evidence, independently finite saved norm scalar only.',
       native_Log='Captured native FP32 deltas are source evidence; arithmetic replay does not reconstruct native Log or re-forward any model.')
    result={'schema':'full151105_WM_component_saved_checkpoint_CPU_review:1','status':'PASS_SERIALIZED_COMPONENT_HEALTH_NOT_ACCURACY' if all(checks.values()) else 'FAILED_COMPONENT_SAVED_REVIEW',
       'engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':auditor,'protocol':pin(protocol_path),'run_dir':str(run.resolve()),
       'iteration':iteration,'final_iteration_scope':iteration==recipe['iterations'],'initial_component':calls[0],'checkpoints':{a:calls[i+1] for i,a in enumerate(ARMS)},'producer_health':pin(run/('checkpoint_%04d_health.json'%iteration)),
       'details':details,'native105_health_qualified':False,'component_accuracy_confirmed':False,'fresh_test_evaluated':False,'physical_gradient_qualified':False,
       'own_trusted_Torch_CPU_loads':3,'new_model_Actor_WM_constructions_FW_AD_MJB_Env_physics_GPU':0}
    write(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total']},indent=2));return 0 if all(checks.values()) else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path);p.add_argument('--run-dir',type=Path);p.add_argument('--iteration',type=int,default=100);p.add_argument('--output',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:
        print(json.dumps({'status':'SOURCE_ONLY_COMPONENT_AUDITOR_NOT_ACTUAL_CHECKPOINT_HEALTH','auditor':pin(__file__),'actual_checkpoint_reads':0,'Torch_imported':'torch' in sys.modules,'model_native_Env_GPU':0},indent=2));return 0
    if None in (a.protocol,a.run_dir,a.output):raise ValueError('Explicit trusted protocol/run/output required')
    try:return audit(a.protocol,a.run_dir,a.iteration,a.output)
    except Exception as e:
        if a.output.exists() and not (a.output/'failure.json').exists():write(a.output/'failure.json',{'error':repr(e),'auditor':pin(__file__),'source_data_unchanged':True})
        raise


if __name__=='__main__':raise SystemExit(main())
