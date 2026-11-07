"""One-shot CPU serialized-array review; no NN construction or execution."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,math,sys,tomllib
import numpy as np
import torch
torch.set_num_threads(1)

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/phone_online_v1'
OUT=Path(__file__).with_name('checkpoint_review.json')
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning import online_contract as C

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(p):
    assert pin(p['path'])==p, 'File pin mismatch: '+p['path'];return Path(p['path'])
def raw(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()==b.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(raw(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(raw(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b
def finite(v):
    if isinstance(v,torch.Tensor):return v.device.type=='cpu' and bool(torch.isfinite(v).all())
    if isinstance(v,np.ndarray):return bool(np.isfinite(v).all())
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
    return math.isfinite(v) if isinstance(v,float) else True

report=dict(schema='supermimic_research.online_checkpoint_saved_review:1',auditor=pin(__file__),checks={},details={},counts=dict(own_CPU_checkpoint_loads=0,NN_constructors=0,NN_FW=0,backward=0,optimizer_constructors=0,optimizer_steps=0,native_calls=0,GPU_calls=0),scope='Independent inspection of saved arrays and metadata only. No forward/readback prediction reproduction, native label or policy/holding qualification. Named optimizer binding uses serialized state/gradient order and pinned module source, without constructing modules.')
def check(name,value):report['checks'][name]=bool(value)
try:
    h=read(RUN/'health_0010.json');result=read(RUN/'result.json');protocol=read(RUN/'protocol.json');terminal=read(ROOT/'runs/phone_online_process_v1/terminal.json')
    cp=checked(h['checkpoint']);config=tomllib.loads(checked(protocol['configuration_file']).read_text(encoding='utf8'))
    _,_,registry=C.configuration(protocol['configuration_file']['path']);root_meta=C.root_metadata(config,registry)
    report['inputs']=dict(checkpoint=pin(cp),root_checkpoint=root_meta['checkpoint'],health=pin(RUN/'health_0010.json'),result=pin(RUN/'result.json'),protocol=pin(RUN/'protocol.json'),terminal=pin(ROOT/'runs/phone_online_process_v1/terminal.json'))
    # Exactly these two trusted own files, each deserialized once onto CPU.
    ck=torch.load(cp,map_location='cpu',weights_only=False);report['counts']['own_CPU_checkpoint_loads']+=1
    root=torch.load(checked(root_meta['checkpoint']),map_location='cpu',weights_only=False);report['counts']['own_CPU_checkpoint_loads']+=1
    check('schema_root_cursor_and_stage_counts',ck['schema']==C.CHECKPOINT and root['schema']==C.ROOT_CHECKPOINT and (root['paired_round'],root['shared_WM_updates'])==(1000,1100) and type(ck['update']) is int and ck['update']==10 and ck['cursor']==dict(cycle=1,update_in_cycle=10,global_paired_updates=10,total_lineage_paired_updates=1010,shared_WM_lineage_updates=1110) and ck['counts']==dict(WM_updates=10,paired_updates=10))
    check('serialized_configuration_and_bindings',ck['configuration']==config==h['configuration']==protocol['configuration'] and ck['provenance']==h['provenance'] and ck['provenance']['sources']==protocol['sources']==terminal['sources'] and ck['provenance']['inputs']==protocol['inputs'] and ck['protocol']==pin(RUN/'protocol.json'))
    for p in ck['provenance']['sources'].values():checked(p)
    for p in ck['provenance']['inputs'].values():checked(p)
    checked(ck['provenance']['config']);checked(ck['provenance']['artifact_manifest'])
    check('all_current_source_and_input_filepins',C.fingerprint(C.source_pins())==C.fingerprint(ck['provenance']['sources']) and C.fingerprint(C.selected_inputs(config,registry,root_meta))==C.fingerprint(ck['provenance']['inputs']))
    check('root_parent_and_frozen_origin',ck['root_checkpoint']==ck['normalizer_origin_checkpoint']==ck['parent_checkpoint']==root_meta['checkpoint'] and ck['parent_health']==root_meta['root_health'])
    check('root_norm_scale_physical_raw',raw(ck['normalizers'],root['WM_normalizers']) and raw(ck['loss_scales'],root['WM_loss_scales']) and raw(ck['physical_context'],root['physical_context']))
    norms=True;networks={'WM':(ck['shared_WM'],root['shared_WM'])}
    for mode in C.R.MODES:
        norms &=raw(ck['arms'][mode]['norm'],root['arms'][mode]['norm'])
        for key in ['stateactor','statecritic','statecritic_target']:
            suffix={'stateactor':'Actor','statecritic':'Critic','statecritic_target':'Target_Critic'}[key]
            networks[mode+'_'+suffix]=(ck['arms'][mode][key],root['arms'][mode][key])
            saved_norm={n[len('norm.'):]:v for n,v in ck['arms'][mode][key].items() if n.startswith('norm.')}
            norm_key={'stateactor':'actor','statecritic':'critic','statecritic_target':'critic_target'}[key]
            norms &=raw(saved_norm,ck['arms'][mode]['norm'][norm_key])
    check('all_policy_and_target_norms_raw_root_and_state',norms)
    network_details={};all_changed=True;all_net_finite=True
    for name,(state,parent) in networks.items():
        same_layout=state.keys()==parent.keys() and all(v.dtype==parent[k].dtype and v.shape==parent[k].shape for k,v in state.items())
        changed=[k for k,v in state.items() if not raw(v,parent[k])];all_changed &=bool(changed);all_net_finite &=same_layout and finite(state)
        network_details[name]=dict(state_tensors=len(state),changed_tensors=len(changed),changed_names=changed,max_abs_change=max((float((v.double()-parent[k].double()).abs().max()) for k,v in state.items() if v.numel()),default=0.))
    report['details']['network_changes']=network_details
    check('all_serialized_weights_targets_and_norms_finite',all_net_finite)
    check('learning_and_target_weights_have_updates_not_root_identity',all_changed)
    optimizer_details={};adam_ok=True;grads_ok=True
    expected_sizes={'WM':6}|{m+'_'+k:n for m in C.R.MODES for k,n in [('Actor',8),('Critic',10)]}
    check('exact_five_optimizer_and_gradient_sets',set(ck['optimizers'])==set(ck['gradient_snapshots'])==set(expected_sizes))
    for name,N in expected_sizes.items():
        opt=ck['optimizers'][name];g=ck['gradient_snapshots'][name];state=networks[name][0];names=list(g['pre_clip']);groups=opt['param_groups'];ids=[i for group in groups for i in group['params']]
        order=[n for n in state if n in g['pre_clip']]
        valid=(len(names)==N and names==list(g['post_clip'])==order and len(groups)==1 and len(ids)==N and len(set(ids))==N and set(ids)==set(opt['state']))
        group=groups[0];lr=config['optimizer']['WM_lr' if name=='WM' else 'actor_lr' if name.endswith('_Actor') else 'critic_lr']
        valid &=group['lr']==lr and group['weight_decay']==0 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['foreach'] is False and group['capturable'] is False
        steps=[];gradient_valid=finite(g) and g['tag'].startswith('PRE') and '10' in g['tag']
        for pid,pname in zip(ids,names):
            p=state[pname];a=opt['state'][pid];step=a['step'];steps.append(float(step))
            valid &=step.numel()==1 and finite(step) and float(step)==10
            for key in ['exp_avg','exp_avg_sq']:
                valid &=a[key].shape==p.shape and a[key].dtype==p.dtype and finite(a[key])
            valid &=bool((a['exp_avg_sq']>=0).all())
            for phase in ['pre_clip','post_clip']:gradient_valid &=g[phase][pname].shape==p.shape and g[phase][pname].dtype==p.dtype and finite(g[phase][pname])
        adam_ok &=valid;grads_ok &=gradient_valid
        optimizer_details[name]=dict(parameter_tensors=N,named_order=names,step_values=sorted(set(steps)),Adam_valid=bool(valid),finite_named_PRE_POST_gradients=bool(gradient_valid),gradient_tag=g['tag'])
    report['details']['optimizers']=optimizer_details
    check('all_five_Adam_states_step10_finite_moments_named_order',adam_ok)
    check('all_named_PRE_POST_gradients_finite_shape_dtype',grads_ok)
    check('learning_call_budget',ck['learning_calls']==h['learning_calls']==dict(WM_supervised_FW=10,WM_backward=10,WM_optimizer_steps=10,Actor_FW=40,WM_imagination_FW=40,Critic_real_FW=40,Critic_imagination_FW=20,Actor_backward_microbatches=20,Actor_optimizer_steps=20,Critic_backward=20,Critic_optimizer_steps=20))
    lp=ck['log_prefix'];rawlog=Path(lp['path']).read_bytes();logs=[json.loads(v) for v in rawlog.splitlines()]
    check('exact_ten_log_rows_no11_and_prefix',len(logs)==10 and [r['update'] for r in logs]==list(range(1,11)) and lp==dict(path=str(RUN/'updates.jsonl'),bytes=len(rawlog),sha256=hashlib.sha256(rawlog).hexdigest(),records=10,range=[1,10]) and ck['parent_log_prefixes']==[])
    losses_ok=True;sample_ok=True
    selection=read(RUN/'pool_selection_cycle1.json');mapping=read(checked(ck['pool_identity_map']))
    wm_rows=selection['WM_selection']['included_global_row_ids'];actors=set(selection['selection']['Actor_H2_global_row_ids'])
    for row in logs:
        losses_ok &=math.isfinite(row['WM_loss']) and all(math.isfinite(v) for m in row['modes'].values() for v in m.values())
        sample_ok &=len(row['Actor_global_row_ids'])==8 and all(i in actors for i in row['Actor_global_row_ids'])
        for key,n,available,span in [('online',4,112,[0,4]),('old_original',2,684,[4,6]),('old_tripod',2,153,[6,8])]:
            s=row['WM_sampling'][key];sample_ok &=len(s['local_indices'])==n and s['available_rows']==available and s['output_range']==span and all(type(i) is int and 0<=i<available for i in s['local_indices'])
    check('finite_losses_and_fixed_4_2_2_Actor8_sampling',losses_ok and sample_ok)
    check('poolmap112_stable_identity_and_eligibility',len(mapping)==112 and [m['global_row_id'] for m in mapping]==list(range(112)) and len({(m['pool']['sha256'],m['original_row_id']) for m in mapping})==112 and wm_rows==list(range(112)) and actors==set(range(112)) and all(m['pool_index']==0 for m in mapping))
    C.verify_collections(ck['collections'],registry)
    check('collection_protocol_terminal_worker_pool_filepins',len(ck['collections'])==1 and ck['collections']==result['collections']==h['collections'] and ck['collections'][0]['dataset_report']==selection['selection']['pool_reports'][0])
    eta=float(np.mean([r['update_seconds'] for r in logs]))
    check('exit0_first10_result_health_and_ETA',terminal['returncode']==0 and terminal['configuration']==protocol['configuration_file'] and checked(terminal['stderr']).stat().st_size==0 and result['status']=='STOPPED_FIRST_HEALTHY_ONLINE_CHECKPOINT_WITH_ETA' and result['checkpoint']==h['checkpoint']==pin(cp) and result['health']==pin(RUN/'health_0010.json') and result['completed_updates']==10 and result['target_updates']==300 and h['first_healthy_pausepoint'] is True and h['stops_here'] is True and h['remaining_updates']==290 and abs(h['measured_update_seconds']-eta)<1e-12 and abs(h['ETA_learning_only_s']-290*eta)<1e-9)
    check('saved_RNG_sampler_serialized_and_no_GPU',finite(ck['RNG']) and ck['sample_generator'].dtype==torch.uint8 and ck['sample_generator'].ndim==1 and not torch.cuda.is_initialized() and 'mujoco' not in sys.modules)
    report['details']['ETA']=dict(measured_mean_update_s=eta,remaining_updates=290,learning_only_s=290*eta,remaining_collection_cache_health_IO_not_included=True)
except BaseException as error:
    report['error']=repr(error);check('review_completed_without_exception',False)
report.update(passed=sum(report['checks'].values()),total=len(report['checks']),engineering_qualified=all(report['checks'].values()),status='PASS_SAVED_ONLINE10_SERIALIZED_ENGINEERING_NOT_POLICY_QUALIFICATION' if all(report['checks'].values()) else 'FAIL_SAVED_REVIEW')
with OUT.open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(dict(report=pin(OUT),status=report['status'],passed=report['passed'],total=report['total'],counts=report['counts'],error=report.get('error'))))
raise SystemExit(0 if report['engineering_qualified'] else 1)
