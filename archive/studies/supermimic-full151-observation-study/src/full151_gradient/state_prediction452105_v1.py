"""Fixed A2 CPU WorldModel, all452 matched20ms state errors, no task scalar.

Metadata is NumPy/stdlib only. The explicit authorized run performs one strict
existing105 load and one batch452 prediction. No goal, objective, AD, state
conversion, Actor forward, new native geometry or physics path is called.
"""
from __future__ import annotations
import csv,hashlib,json,os,pickle,random,sys,time,zipfile
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from . import penalty_observability105_v3 as qualified

ROOT=qualified.ROOT
PREREG=ROOT/'protocols/full151105_state_prediction452_CPU_prereg_v2.json'
PREREG_SHA='37652718ed150421ed84d25bcafc14b27a1e7390a7dee7e729a860b3d91ef411'
SCHEMA='full151105_state_prediction452_CPU_run:1'
ROLE='fixed_A2_WM_all452_state_prediction_error'
KEYS={'schema','role','protocol','output','CPU_run_authorized','cpu_threads','source_pins','cpu_ready'}
OWN={'consumer':ROOT/'src/full151_gradient/state_prediction452105_v1.py',
     'CLI':ROOT/'scripts/diagnose_full151_state_prediction452_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_state_prediction452_cpu_v1.py'}
BLOCKS={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),'R':(43,52,'dimensionless'),'v':(52,55,'m/s'),'omega':(55,58,'rad/s')}
require=qualified.require;read=qualified.read;pin=qualified.pin;checked=qualified.checked;same_pin=qualified.same_pin
write=qualified.write;npwrite=qualified.npwrite;common=qualified.common


def canonical_sources(values):
    """Hash once per actual samefile+SHA; equal bytes in OTHER files stay distinct."""
    if type(values) is dict:values=values.values()
    selected=[];by_file={};by_spelling={}
    for value in values:
        require(type(value) is dict and set(value)=={'path','sha256'} and type(value['path']) is str and Path(value['path']).is_absolute(),'Exact absolute source pin required')
        sha=value['sha256'];require(type(sha) is str and len(sha)==64 and all(c in '0123456789abcdef' for c in sha),'Invalid source SHA')
        spelling=(value['path'].casefold(),sha)
        if spelling in by_spelling:continue
        f=Path(value['path']).resolve(strict=True);stat=f.stat()
        token=(stat.st_dev,stat.st_ino) if stat.st_ino else ('resolved_path',str(f).casefold())
        if token in by_file:
            previous=by_file[token]
            require(f.samefile(previous['path']) and sha==previous['sha256'],'One source identity has conflicting pins')
        else:
            checked(value);previous=dict(value);by_file[token]=previous;selected.append(previous)
        by_spelling[spelling]=previous
    return {'file_%04d__%s'%(i,Path(v['path']).name):v for i,v in enumerate(selected)}


def membership(values):
    result=set()
    for v in canonical_sources(values).values():
        f=Path(v['path']).resolve(strict=True);s=f.stat();result.add((s.st_dev,s.st_ino if s.st_ino else str(f).casefold(),v['sha256']))
    return result


def required_sources(p):
    values=list(p['source_pins'].values())
    for alias in ('physical_config','GPU_replay_config','fixed_A2_training_config'):
        values.extend(read(checked(p['inputs'][alias]))['source_pins'].values())
    values.extend(pin(v) for v in OWN.values())
    values.extend([pin(qualified.__file__),pin(qualified.legacy.__file__),pin(common.__file__),pin(qualified.replay.__file__)])
    return canonical_sources(values)


def validate_header(c):
    require(type(c) is dict and set(c)==KEYS and c['schema']==SCHEMA and c['role']==ROLE,'Exact state452 schema/role/keyset required')
    require(type(c['CPU_run_authorized']) is bool and type(c['cpu_threads']) is int and c['cpu_threads']==4,'Explicit CPU bool and fixed threads4')
    require(type(c['source_pins']) is dict and c['source_pins'],'Flat sources required')
    require(type(c['output']) is str and Path(c['output']).is_absolute(),'Absolute output required')
    out=Path(c['output']).resolve();require(out.parent==ROOT/'outputs' and out.name.startswith('full151105_state_prediction452_CPU_') and not out.exists(),'Fresh fixed study output required')


def declaration():
    require(all(m not in sys.modules for m in ('torch','warp','mujoco')),'Fresh metadata path only')
    require(pin(PREREG)['sha256']==PREREG_SHA,'Prereg drift')
    return {'schema':'full151105_state_prediction452_CPU_declaration:1','status':'DECLARED_NOT_LOADED_NOT_PREDICTED','runplan_schema':SCHEMA,'role':ROLE,
        'protocol':pin(PREREG),'own_sources':{k:pin(v) for k,v in OWN.items()},'keys':sorted(KEYS),'Torch_imported':False,'NPZ_opened':0,'model_loads':0,'WM_forwards':0,'actual_inputs_assembled':0}


def model_metadata(inputs,p):
    health=read(checked(inputs['fixed_A2_final_health']));common.all_checks(health)
    require(health['schema']=='full151105_checkpoint_CPU_health:1' and health['status']=='PASS_SAVED_ENGINEERING_HEALTH' and health['method']=='A2' and health['role']=='primary_matched','Actual A2 primary health required')
    require(type(health['iteration']) is int and health['iteration']==2000 and type(health['seed']) is int and health['seed']==0 and health['final_iteration_scope'] is True,'Only fixed final2000/seed0')
    # These two dict aliases are literal requirements of the unchanged strict loader.
    require(health['auditor']==inputs['fixed_A2_health_auditor'] and health['checkpoint']==inputs['fixed_A2_checkpoint'],'Existing strict loader literal aliases must be preserved')
    require(same_pin(health['config'],inputs['fixed_A2_training_config']),'Health/config alias differs')
    train=read(checked(inputs['fixed_A2_training_config']));run=Path(train['output']).resolve(strict=True)
    require(train['method']=='A2' and train['role']=='primary_matched' and type(train['seed']) is int and train['seed']==0 and train['training']['max_iterations']==2000 and train['training']['num_envs']==128,'Wrong fixed training route')
    require(Path(health['run_dir']).resolve(strict=True).samefile(run) and checked(inputs['fixed_A2_checkpoint']).samefile(run/'train/model_final.pt'),'Checkpoint is not actual fixed final')
    require(Path(p['fixed_model']['training_run']).resolve(strict=True).samefile(run) and checked(inputs['fixed_A2_fixed_physics_context']).samefile(run/'fixed_physics_context.npz'),'Fixed source context/run alias differs')
    require(same_pin(inputs['input_bundle'],inputs['full151_bundle']) and same_pin(train['input_bundle'],inputs['input_bundle']),'Training/reference input bundle differs')
    return {'training_run':str(run),'health':inputs['fixed_A2_final_health'],'checkpoint':inputs['fixed_A2_checkpoint'],'health_auditor':inputs['fixed_A2_health_auditor']}


def metadata_preflight(config_path):
    require(all(m not in sys.modules for m in ('torch','warp','mujoco')),'Fresh no-backend default process')
    c=read(config_path);validate_header(c);require(same_pin(c['protocol'],{'path':str(PREREG),'sha256':PREREG_SHA}),'Wrong frozen state452 prereg')
    p=read(checked(c['protocol']));require(p['schema']=='full151105_state_prediction452_CPU_preregistration:1' and p['CPU_run_authorized'] is False and p['runplan_schema']==SCHEMA and p['role']==ROLE,'Wrong source-only protocol scope')
    for value in p['inputs'].values():checked(value)
    needed=required_sources(p);have=canonical_sources(c['source_pins'])
    require(len(have)==len(c['source_pins']) and membership(needed)<=membership(have),'Sources must be complete and one entry per actualfile')
    ready=read(checked(c['cpu_ready']));proof=read(checked(ready['cpu_report']));common.all_checks(proof)
    require(ready['schema']=='full151105_state_prediction452_CPU_READY:1' and proof['schema']=='full151105_state_prediction452_CPU_contract:1' and proof['status']=='PASS_SYNTHETIC_GROUP_ERROR_AND_SINGLE_FORWARD_CONTRACT','Wrong new narrow CPU proof')
    require(same_pin(ready['protocol'],c['protocol']),'Ready protocol differs')
    for k,v in OWN.items():require(same_pin(ready['source_pins'][k],pin(v)) and same_pin(proof['source_pins'][k],pin(v)),'CPU executed source differs: '+k)
    for key in ('actual_saved_input_assembly','actual_WM_loads','actual_WM_forwards','actual_model_or_MJB_or_Env_or_GPU'):
        require(type(proof['scope'][key]) is int and proof['scope'][key]==0,'Synthetic proof elevated')
    physical=read(checked(p['inputs']['physical_saved_independent1762']));common.all_checks(physical)
    require(physical['completed_branches']==152 and physical['episode_step_returned']==1808 and same_pin(physical['config'],p['inputs']['physical_config']),'Incomplete actual physical dataset')
    replay_join=qualified.qualified_replay_metadata(p['inputs']);model_join=model_metadata(p['inputs'],p)
    require(all(m not in sys.modules for m in ('torch','warp','mujoco')),'Metadata imported backend')
    return {'config':c,'config_pin':pin(config_path),'protocol':p,'unique_sources':have,'replay_join':replay_join,'model_join':model_join,
        'status':'PASS_STATE452_HASH_JSON_ONLY','Torch_imported':False,'NPZ_opened':0,'model_loads':0,'WM_forwards':0,'actual_inputs_assembled':0}


def expected_schedule():
    rows=[]
    for pi,path in enumerate(('A0','A2')):
        slots=[('donor',k,None) for k in range(150)]
        for si,k in enumerate((0,49,99,149)):
            slots.extend(('baseline' if j<3 else 'direction',k,(pi*4+si)*19+j) for j in range(19))
        for mode,k,bid in slots:
            i=len(rows);rows.append({'group_id':i,'path':path,'mode':mode,'k':k,'branch_id':bid,'source_frame':200+k,
                'terminal_observation':k+1,'valid_next_policy_decision':k+1<150,'no_u150':True,'first_physical_pre_index':4*i,'last_physical_index':4*i+3,'post20_GPU_row_index':1808+i})
    return rows


def assemble_groups(samples,rewards,packet,raw_gpu,endpoint_labels,forensic_raw):
    """Saved bytes only; every physical group gets one matched pre/u/post row."""
    groups,reward_map=qualified.group_rows(samples,rewards);identities=expected_schedule()
    require(len(groups)==len(endpoint_labels)==len(identities)==452,'All452 groups required')
    mapping=qualified.packet_abi(packet,forensic_raw,expected_rows=1808)
    physical_q=qualified.exact_fp32([s['pre_qpos'] for s in samples],(1808,27),'physical pre-qpos')
    physical_v=qualified.exact_fp32([s['pre_qvel'] for s in samples],(1808,26),'physical pre-qvel')
    require(physical_q.tobytes()==forensic_raw['pre_qpos'].tobytes() and physical_v.tobytes()==forensic_raw['pre_qvel'].tobytes(),'Physical/qualified source pre-state rows differ')
    require(set(raw_gpu)=={'native_GPU_state58_raw','completed_mask'},'Exact raw native replay array keys')
    x=raw_gpu['native_GPU_state58_raw'];mask=raw_gpu['completed_mask']
    require(x.shape==(2260,58) and x.dtype==np.float32 and np.isfinite(x).all() and mask.shape==(2260,) and mask.dtype==np.bool_ and mask.all(),'Every native GPU2260 row must be completed finite FP32')
    require(x[:1808].tobytes()==packet['native_GPU_prestate58'].tobytes(),'Qualified packet and raw native prestate bytes differ')
    pre=[];u=[];refs=[];all_u=[]
    for i,((key,indices),wanted,label) in enumerate(zip(groups,identities,endpoint_labels)):
        require(key==(wanted['path'],wanted['k'],wanted['branch_id']) and indices==list(range(4*i,4*i+4)),'Source group/order changed')
        require(all(s['mode']==wanted['mode'] for s in (samples[j] for j in indices)),'Group mode changed')
        expected_label={'path':key[0],'k':key[1],'branch_id':key[2],'last_sample_index':indices[-1]}
        require(type(label) is dict and set(label)==set(expected_label) and all(type(label[k]) is type(v) and label[k]==v for k,v in expected_label.items()),'Endpoint identity/order/type differs')
        controls=qualified.exact_fp32([samples[j]['ctrl'] for j in indices],(4,20),'all actual absolute controls')
        require(controls.tobytes()==np.broadcast_to(controls[:1],(4,20)).copy().tobytes(),'Control changes within physical four-substep interval')
        last_q=qualified.exact_fp32(samples[indices[-1]]['post_qpos'],(27,),'physical last-qpos')
        last_v=qualified.exact_fp32(samples[indices[-1]]['post_qvel'],(26,),'physical last-qvel')
        require(x[1808+i,:20].tobytes()==last_q[mapping['hand_qpos']].tobytes() and x[1808+i,20:40].tobytes()==last_v[mapping['hand_dof']].tobytes(),'Target/physical last-state q/qd differ')
        pre.append(packet['native_GPU_prestate58'][indices[0]]);u.append(controls[0]);all_u.append(controls)
        refs.append(qualified.exact_fp32(reward_map[key]['physics'],(58,),'saved flat58 post20 reference'))
    pre=np.asarray(pre,np.float32);u=np.asarray(u,np.float32);truth=x[1808:].copy();refs=np.asarray(refs,np.float32)
    target_rule=qualified.replay.old_flags(truth,refs,state58=True)
    require(target_rule['old_flags'].all(),'Matched endpoint target violates ORIGINAL full58 arithmetic/bit rule')
    return {'pre_x452_FP32':pre,'target_x452_FP32':truth,'actual_u452_FP32':u,'all_substep_u452x4x20_FP32':np.asarray(all_u,np.float32),
        'group_first_pre_indices':np.arange(0,1808,4,dtype=np.int64),'group_last_physical_indices':np.arange(3,1808,4,dtype=np.int64),
        'post20_GPU_row_indices':np.arange(1808,2260,dtype=np.int64)},identities,target_rule


def error_statistics(predicted,target,identities):
    require(predicted.shape==target.shape==(452,58) and predicted.dtype==target.dtype==np.float32 and np.isfinite(predicted).all() and np.isfinite(target).all(),'Full452 finite FP32 states required')
    require(identities==expected_schedule(),'Identity order cannot be selected or sorted')
    delta=predicted.astype(np.float64)-target.astype(np.float64);absolute=abs(delta)
    def statistics(indices):
        d=delta[indices];result={'rows':len(d),'per_feature':{'signed_mean':d.mean(0).tolist(),'mean_absolute':abs(d).mean(0).tolist(),
            'RMSE':np.sqrt((d*d).mean(0)).tolist(),'maximum_absolute':abs(d).max(0).tolist()},'blocks':{}}
        for name,(lo,hi,unit) in BLOCKS.items():
            b=d[:,lo:hi];squared=b*b
            value={'start':lo,'stop_exclusive':hi,'component_count':hi-lo,'units':unit,'component_RMSE':float(np.sqrt(squared.mean())),
                'mean_absolute':float(abs(b).mean()),'maximum_absolute':float(abs(b).max())}
            if name in ('p','v','omega'):value['vector_RMS']=float(np.sqrt(squared.sum(-1).mean()))
            if name=='R':value['Frobenius_RMS']=float(np.sqrt(squared.sum(-1).mean()))
            result['blocks'][name]=value
        return result
    stats={'all452':statistics(np.arange(452))}
    for path in ('A0','A2'):
        for mode in ('donor','baseline','direction'):
            ids=np.array([i for i,r in enumerate(identities) if r['path']==path and r['mode']==mode],np.int64);stats[path+'/'+mode]=statistics(ids)
    return delta,absolute,stats


def verify_fixed_xi46(xi,expected):
    a,b=np.asarray(xi),np.asarray(expected)
    require(a.shape==b.shape==(46,) and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'Only finite physical xi46 FP32, not privileged context368')
    require(a.tobytes()==b.tobytes(),'Fixed physical xi46 must remain bitexact')


def predict_once(model,x,u):
    """One full batch, no fallback or automatic smaller-batch re-execution."""
    import torch
    require(isinstance(x,torch.Tensor) and isinstance(u,torch.Tensor) and x.shape==(452,58) and u.shape==(452,20),'Fixed full452 input shape')
    require(x.dtype==u.dtype==torch.float32 and x.device.type==u.device.type=='cpu' and torch.isfinite(x).all() and torch.isfinite(u).all(),'Finite FP32 CPU inputs')
    require(not x.requires_grad and not u.requires_grad and not model.training and all(not p.requires_grad for p in model.parameters()),'No input/parameter AD or training mode')
    with torch.no_grad():predicted,aux=model(x,u)
    require(predicted.shape==(452,58) and aux.shape==(452,15) and predicted.dtype==aux.dtype==torch.float32 and predicted.device.type==aux.device.type=='cpu','Single full452 output ABI')
    require(torch.isfinite(predicted).all() and torch.isfinite(aux).all() and not predicted.requires_grad and not aux.requires_grad,'Finite non-AD outputs')
    return predicted,aux


@contextmanager
def counted_checkpoint_reads(torch,checkpoint_pin,counts,trace):
    original=torch.load
    def tracked(path,*args,**kwargs):
        counts['checkpoint_torch_loads']['attempted']+=1
        require(Path(path).resolve(strict=True).samefile(checked(checkpoint_pin)) and kwargs.get('map_location')=='cpu','Only the fixed checkpoint CPU read is permitted')
        require(counts['checkpoint_torch_loads']['attempted']<=2,'Internal checkpoint load budget exceeded')
        result=original(path,*args,**kwargs);counts['checkpoint_torch_loads']['returned']+=1
        trace.append({'path':str(Path(path).resolve(strict=True)),'map_location':'cpu','weights_only':kwargs.get('weights_only'),'returned':True});return result
    torch.load=tracked
    try:yield
    finally:torch.load=original


def _jsonl(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]


def run_analysis(config_path):
    start=time.perf_counter();m=metadata_preflight(config_path);c=m['config'];require(c['CPU_run_authorized'] is True and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Explicit fresh CPU authorization required')
    out=Path(c['output']);out.mkdir();phase='archive';counts={k:{'attempted':0,'returned':0} for k in ('strict_WM_loader','checkpoint_torch_loads','WM_forward')};trace=[]
    def call(k,fn,*args,**kwargs):counts[k]['attempted']+=1;v=fn(*args,**kwargs);counts[k]['returned']+=1;return v
    try:
        write(out,'config.json',c);write(out,'protocol.json',m['protocol']);write(out,'source_inventory.json',m['unique_sources'])
        write(out,'input_qualification_metadata.json',{'replay':m['replay_join'],'fixed_model':m['model_join']})
        with zipfile.ZipFile(out/'execution_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for i,v in enumerate(m['unique_sources'].values()):z.write(checked(v),'%04d_%s'%(i,Path(v['path']).name))
        phase='all452_saved_input_alignment';inputs=m['protocol']['inputs']
        samples=_jsonl(checked(inputs['physical_samples']));rewards=_jsonl(checked(inputs['reward_snapshots']))
        with np.load(checked(inputs['qualified_GPU_prestate1808']),allow_pickle=False) as z:packet={k:z[k].copy() for k in z.files}
        with np.load(checked(inputs['GPU_raw_all2260']),allow_pickle=False) as z:raw_gpu={k:z[k].copy() for k in z.files}
        raw_keys=('pre_qpos','pre_qvel','palm_pos_all2260','palm_rot_all2260','hand_qpos','hand_dof','object_qpos_start','object_dof_start')
        with np.load(checked(inputs['forensic_raw_all2260']),allow_pickle=False) as z:forensic={k:z[k].copy() for k in raw_keys}
        arrays,identities,target_rule=assemble_groups(samples,rewards,packet,raw_gpu,read(checked(inputs['GPU_endpoint_identities'])),forensic)
        npwrite(out,'matched_input_arrays.npz',arrays);npwrite(out,'target452_original_rule_alignment.npz',target_rule);write(out,'group_identities452.json',identities)
        write(out,'input_alignment.json',{'all452_groups_aligned':True,'all_four_substep_controls_bitexact':True,'all_targets_original_full58_rule_passed':True,
            'donor_rows':300,'baseline_rows':24,'direction_rows':128,'state_conversion_calls':0,'score_or_goal_calls':0,'WM_calls_before_input_alignment':0})
        phase='one_existing_strict105_WM_loader'
        import torch
        require(not torch.cuda.is_initialized() and 'warp' not in sys.modules,'Fresh CPU-only runtime required');torch.set_num_threads(4)
        def rng():return {'Python_pickle':np.frombuffer(pickle.dumps(random.getstate(),protocol=5),np.uint8).copy(),
            'NumPy_pickle':np.frombuffer(pickle.dumps(np.random.get_state(),protocol=5),np.uint8).copy(),'Torch_CPU':torch.get_rng_state().numpy().copy()}
        before_rng=rng();npwrite(out,'RNG_before.npz',before_rng)
        from full151_native.bundle import load_bundle
        bundle=load_bundle(checked(inputs['input_bundle']),expected_sha256=inputs['input_bundle']['sha256'])
        from full151_native.world_model_checkpoint import load_world_model_checkpoint
        with counted_checkpoint_reads(torch,inputs['fixed_A2_checkpoint'],counts,trace):
            model,receipt=call('strict_WM_loader',load_world_model_checkpoint,checked(inputs['fixed_A2_checkpoint']),bundle,'A2',m['model_join']['training_run'],
                checked(inputs['fixed_A2_final_health']),expected_health_auditor=inputs['fixed_A2_health_auditor'],expected_iteration=2000)
        require(counts['checkpoint_torch_loads']=={'attempted':2,'returned':2},'Existing loader exact two reads required')
        require(not model.training and all(not t.requires_grad for t in model.parameters()),'Frozen evaluation model required')
        xi=model.physical_xi.detach().cpu().numpy().copy();expected=np.asarray(bundle['context']['reader_context'])[0]
        verify_fixed_xi46(xi,expected)
        write(out,'loader_receipt.json',receipt);write(out,'checkpoint_read_trace.json',trace)
        before_weights=common._tensor_digest(model.state_dict());before_inputs=common.array_digest(arrays)
        x=torch.from_numpy(arrays['pre_x452_FP32'].copy());u=torch.from_numpy(arrays['actual_u452_FP32'].copy())
        phase='one_batch452_CPU_forward_no_AD'
        predicted,aux=call('WM_forward',predict_once,model,x,u)
        require(counts['WM_forward']=={'attempted':1,'returned':1} and predicted.shape==(452,58) and aux.shape==(452,15) and predicted.dtype==aux.dtype==torch.float32 and predicted.device.type==aux.device.type=='cpu','Single full452 CPU output ABI')
        prediction=predicted.detach().numpy().copy();self15=aux.detach().numpy().copy()
        require(np.isfinite(prediction).all() and np.isfinite(self15).all() and not predicted.requires_grad and not aux.requires_grad,'Finite non-AD outputs required')
        delta,absolute,stats=error_statistics(prediction,arrays['target_x452_FP32'],identities)
        npwrite(out,'prediction_arrays.npz',{**arrays,'predicted_x452_FP32':prediction,'physical_xi46_FP32':xi,'aux_self15_logits_FP32':self15,
            'signed_error452x58_F64':delta,'absolute_error452x58_F64':absolute})
        write(out,'statistics.json',stats)
        with (out/'all452_errors.csv').open('x',newline='',encoding='utf-8') as f:
            fields=list(identities[0])+[name+'_component_RMS' for name in BLOCKS]+[name+'_maxabs' for name in BLOCKS]
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for i,row in enumerate(identities):
                v=dict(row)
                for name,(lo,hi,_) in BLOCKS.items():v[name+'_component_RMS']=float(np.sqrt(np.mean(delta[i,lo:hi]**2)));v[name+'_maxabs']=float(abs(delta[i,lo:hi]).max())
                w.writerow(v)
        after_rng=rng();npwrite(out,'RNG_after.npz',after_rng)
        invariants={'weights_normalizers_buffers_unchanged':common._tensor_digest(model.state_dict())==before_weights,
            'raw_CPU_RNG_vectors_unchanged':all(before_rng[k].tobytes()==after_rng[k].tobytes() for k in before_rng),
            'assembled_numpy_inputs_unchanged':common.array_digest(arrays)==before_inputs,
            'forward_tensor_inputs_unchanged':x.numpy().tobytes()==arrays['pre_x452_FP32'].tobytes() and u.numpy().tobytes()==arrays['actual_u452_FP32'].tobytes(),
            'no_parameter_gradients':all(p.grad is None for p in model.parameters()),'CUDA_uninitialized':not torch.cuda.is_initialized(),'Warp_not_imported':'warp' not in sys.modules,
            'objective_binding_not_imported':'full151_intent.binding_v3' not in sys.modules}
        require(all(invariants.values()),'Read-only diagnostic invariant failed')
        for value in inputs.values():checked(value)
        for value in m['unique_sources'].values():checked(value)
        require(same_pin(pin(config_path),m['config_pin']),'Actual plan changed')
        report={'schema':'full151105_state_prediction452_CPU_analysis:1','status':'COMPLETED_FIXED_A2_ALL452_STATE_ERROR_NOT_MODEL_QUALIFICATION',
            'config':m['config_pin'],'protocol':c['protocol'],'source_pins':m['unique_sources'],'logical_source_pin_count':len(c['source_pins']),'unique_source_file_count':len(m['unique_sources']),
            'input_pins':inputs,'counts':counts,'prediction_rows':452,'WM_batch_size':452,'statistics':stats,'invariants':invariants,
            'passive_MJB_source_accounted':{'caller_bundle_load':1,'strict_loader_internal_bundle_load':1,'strict_loader_internal_Actor_explicit_load':1,'strict_loader_internal_total':2,'process_path_total':3,
                'scope':'Source-accounted call paths, NOT an independent native API interceptor; no bundle cache or model load is removed.'},
            'internal_loader_source_accounted':{'Actor_validation_constructors':1,'Actor_forwards':0,'scope':'Original frozen119f loader internals.'},
            'native_state_conversions':0,'objective_factories_or_forwards':0,'taskscore_or_goal_calls':0,'AD_calls':0,'new_training_Env_nativeFK_collision_Warp_GPU_physics':0,
            'precision_scope':'FP32 CPU ampFalse shadow of fixed WM; input/target are saved native CUDA FP32 states. Not historical GPU AMP policy-gradient replay.',
            'physical_gradient_qualified':False,'HO_gradient_qualified':False,'whole_hidden_state_qualified':False,'task_quality_qualified':False,
            'scope':'All300donor+24baseline+128direction groups, dependent single-source/seed. Raw58 errors retain units with no mixed-unit scalar, R projection/angle, taskscore, goal or AD. Aux15 is finite saved selfcollision output only, not HO or a new label metric.',
            'elapsed_seconds':time.perf_counter()-start}
        write(out,'report.json',report);write(out,'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        write(out,'failure.json',{'schema':'full151105_state_prediction452_CPU_failure:1','status':'FAILED_PARTIAL_PRESERVED_NO_RETRY','phase':phase,'error':repr(error),
            'config':m['config_pin'],'counts':counts,'prediction_complete':False,'new_training_Env_GPU_physics':0});raise
