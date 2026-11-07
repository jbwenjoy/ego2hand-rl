"""Explicit native CUDA FP32/B1 Reader algebra replay; no simulator or model.

Metadata imports only NumPy/stdlib. Actual run uses forensic23 inputs and the
original unbound SimReader.state, not the CPU converter wrapper moved to CUDA.
This module contains no objective, learned weight, physics or scoring path.
"""
from __future__ import annotations
import hashlib,inspect,io,json,os,pickle,random,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from full151_native.io import jsonable
from full151_native.metadata import write,module as atomic_io

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
PROTOCOL=ROOT/'protocols/full151105-matched-native-GPU-reader-replay-prereg-v1.json'
PROTOCOL_SHA='fddcf970d7a8e44732fe93e6c83d1918fe017d4d5df159ae5c851a77e15fca6c'
SCHEMA='full151105_matched_native_GPU_reader_replay_run:1'
ROLE='full151105_native_GPU_state_replay'
KEYS={'schema','role','protocol','output','source_pins','CPU_ready','CPU_contract_report','GPU_run_authorized',
      'training_authorized','Env_or_physics_authorized','automatic_retry','compile_fallback','soft_worker_seconds','hard_process_seconds','evaluation_seed','device','allow_tf32'}
OWN={'module':ROOT/'src/full151_gradient/state_replay105_v2.py','entry':ROOT/'scripts/execute_full151_state_replay_v2.py',
     'supervisor':ROOT/'scripts/run_full151_state_replay_v2.py','checker':ROOT/'scripts/check_full151_state_replay_cpu_v2.py'}
EXPECTED={'native_state_calls':2260,'native_state_returned':2260,'native_state_batch_size':1,'converted_rows':2260,
    'pre_rows':1808,'known_post20_rows':452,'endpoint_comparisons':452,'anchor_comparisons':160,'solver_pose_comparisons':1808,
    'native_CPU_state_calls':0,'new_MJB_loads':0,'factory_calls':0,'objective_WM_Actor_forward':0,'weights_loads':0,'AD_calls':0,
    'optimizer_steps':0,'Env_created':0,'nativeFK_collision_queries':0,'physics_steps':0}


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    p=Path(path).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return {'path':str(p),'sha256':h.hexdigest()}


def checked(v):
    need(type(v) is dict and set(v)=={'path','sha256'} and type(v['path']) is str and Path(v['path']).is_absolute(),'Exact absolute pin required')
    need(type(v['sha256']) is str and len(v['sha256'])==64 and pin(v['path'])['sha256']==v['sha256'],'Pinned bytes changed')
    return Path(v['path']).resolve(strict=True)


def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def allchecks(report):
    checks=report.get('checks');need(type(checks) is dict and checks and all(type(v) is bool and v for v in checks.values()),'Strict alltrue actual checks required')
    need(type(report.get('passed')) is int and type(report.get('total')) is int and report['passed']==report['total']==len(checks),'Check counts differ')


def validate_header(c):
    need(type(c) is dict and set(c)==KEYS and c['schema']==SCHEMA and c['role']==ROLE,'Exact GPU replay schema/role/keyset required')
    for key in ('GPU_run_authorized','training_authorized','Env_or_physics_authorized','automatic_retry','compile_fallback','allow_tf32'):need(type(c[key]) is bool,'Explicit bool required: '+key)
    need(all(c[k] is False for k in ('training_authorized','Env_or_physics_authorized','automatic_retry','compile_fallback')),'No training/physics/retry/fallback')
    need(type(c['soft_worker_seconds']) is int and c['soft_worker_seconds']==90 and type(c['hard_process_seconds']) is int and c['hard_process_seconds']==120,'Fixed90/120 lifecycle')
    need(type(c['evaluation_seed']) is int and c['evaluation_seed']==0 and c['device']=='cuda:0' and c['allow_tf32'] is True,'Single predeclared native CUDA0 mode only')
    need(type(c['output']) is str and Path(c['output']).is_absolute() and type(c['source_pins']) is dict and c['source_pins'],'Absolute output/source map required')
    need(Path(c['output']).resolve().parent==(ROOT/'outputs').resolve(),'Replay output must stay in this study outputs')


def validate_config(c,bundle=None):
    """Only JSON/hash/source: no NPZ, Torch, MJB, or device operation."""
    need('torch' not in __import__('sys').modules and 'warp' not in __import__('sys').modules,'Fresh noTorch/Warp metadata path')
    validate_header(c);need(same(c['protocol'],{'path':str(PROTOCOL),'sha256':PROTOCOL_SHA}),'Wrong frozen replay condition')
    p=read(checked(c['protocol']));need(p['expected_counts']==EXPECTED and p['fixed_mode']=='NATIVE_FP32_NO_AUTOCAST_CUDA_BATCH1','Fixed B1 schedule changed')
    for value in p['inputs'].values():checked(value)
    for value in p['source_pins'].values():checked(value)
    have={(str(checked(v)).casefold(),v['sha256']) for v in c['source_pins'].values()}
    required=list(p['source_pins'].values())+[pin(v) for v in OWN.values()]
    need(all((str(checked(v)).casefold(),v['sha256']) in have for v in required),'Core/native dependency outside source closure')
    independent=read(checked(p['inputs']['forensic_saved_independent23']));allchecks(independent)
    need(independent['status']=='PASS_SAVED_FORENSIC_REVIEW_NOT_ELIGIBILITY' and independent['dense_scoring_eligibility'] is False,'Wrong forensic provenance scope')
    proof=read(checked(c['CPU_contract_report']));ready=read(checked(c['CPU_ready']));allchecks(proof)
    need(ready['schema']=='full151105_state_replay_CPU_READY:1' and proof['schema']=='full151105_state_replay_CPU_contract:1' and proof['status']=='PASS_PURE_CPU_CONTRACT_NO_GPU','Wrong CPU contract proof')
    need(same(ready['cpu_report'],c['CPU_contract_report']) and same(ready['protocol'],c['protocol']),'CPU proof aliases changed')
    for k,path in OWN.items():need(same(proof['source_pins'][k],pin(path)) and same(ready['source_pins'][k],pin(path)),'CPU proof bytes differ: '+k)
    for key in ('actual_GPU_calls','native_state_calls','NPZ_input_reads','MJB_loads','model_loads'):need(type(proof[key]) is int and proof[key]==0,'CPU fixture scientific scope changed')
    return {'status':'PASS_METADATA_NATIVE_GPU_REPLAY_ONLY','Torch_imported':False,'NPZ_input_reads':0,'planned_counts':EXPECTED.copy(),'GPU_started':False}


def array_hash(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()


def atomic_npz(path,values):
    """One serialized payload/temp; only bounded rename may retry, never replay."""
    path=Path(path);buffer=io.BytesIO();np.savez_compressed(buffer,**values);payload=buffer.getvalue()
    with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        temp=Path(f.name);f.write(payload)
    atomic_io.replace_with_bounded_retry(temp,path,max_seconds=.75)


def mapping_from_raw(raw):
    result={k:raw[k].copy() for k in ('hand_qpos','hand_dof')}
    for k in ('object_qpos_start','object_dof_start'):
        value=raw[k];need(value.shape==() and value.dtype.kind in 'iu','Typed scalar mapping');result[k]=int(value)
    for k,limit in (('hand_qpos',27),('hand_dof',26)):
        a=result[k];need(a.shape==(20,) and a.dtype.kind in 'iu' and len(set(a.tolist()))==20 and ((0<=a)&(a<limit)).all(),'Unique typed mapping20')
    for start,width,ids,limit in ((result['object_qpos_start'],7,result['hand_qpos'],27),(result['object_dof_start'],6,result['hand_dof'],26)):
        need(0<=start<=limit-width and not set(ids.tolist()).intersection(range(start,start+width)),'Hand/freejoint overlap')
    return result


def validate_inputs(raw):
    shapes={'pre_qpos':(1808,27),'pre_qvel':(1808,26),'post20_qpos':(452,27),'post20_qvel':(452,26),
        'palm_pos_all2260':(2260,3),'palm_rot_all2260':(2260,3,3),'post20_reference':(452,58),'anchor_references':(160,58)}
    for k,s in shapes.items():need(k in raw and raw[k].shape==s and raw[k].dtype==np.float32 and np.isfinite(raw[k]).all(),'Strict forensic FP32 input '+k)
    ids=raw['anchor_pre_indices'];need(ids.shape==(160,) and ids.dtype.kind in 'iu' and ((0<=ids)&(ids<1808)).all(),'Anchor indices not prestate')
    mapping=mapping_from_raw(raw)
    q=np.concatenate((raw['pre_qpos'],raw['post20_qpos']));v=np.concatenate((raw['pre_qvel'],raw['post20_qvel']))
    oq=mapping['object_qpos_start'];need(np.all(abs(np.linalg.norm(q[:,oq+3:oq+7].astype(float),axis=-1)-1)<=1e-5),'Malformed raw quaternion')
    return q,v,raw['palm_pos_all2260'].copy(),raw['palm_rot_all2260'].copy(),mapping


def old_flags(actual,reference,state58=False):
    a,b=np.asarray(actual),np.asarray(reference)
    need(a.shape==b.shape and a.ndim==2 and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'Finite FP32 comparison required')
    delta=a.astype(float)-b.astype(float);bound=8*2**-23*np.maximum(1.,np.maximum(abs(a.astype(float)),abs(b.astype(float))));flags=abs(delta)<=bound
    if state58:
        need(a.shape[1]==58,'Full58 required');flags[:,:40]=a[:,:40].view(np.uint32)==b[:,:40].view(np.uint32)
    return {'actual':a.copy(),'reference':b.copy(),'difference_F64':delta,'old_bound_F64':bound,'old_flags':flags}


def tensor_views_B1(q,v,pp,R,indices,device_kind):
    """Shared structural validator; production always supplies literal cuda."""
    import torch
    need(device_kind in ('cpu','cuda'),'Known fixture/production device only')
    for value,shape in ((q,(1,27)),(v,(1,26)),(pp,(1,3)),(R,(1,3,3))):
        need(isinstance(value,torch.Tensor) and tuple(value.shape)==shape and value.dtype==torch.float32 and value.device.type==device_kind,'Exact B1 FP32 views/device required')
    for k,n in (('hand_qpos',20),('hand_dof',20),('object_qpos',7),('object_dof',6)):
        t=indices[k];need(isinstance(t,torch.Tensor) and t.shape==(n,) and t.dtype==torch.long and t.device.type==device_kind,'Native same-device long indices required')


def native_cuda_state_B1(q,v,pp,R,indices,reader_pin):
    """Explicit CUDA ABI. There is no CPU-wrapper-to-CUDA fallback."""
    tensor_views_B1(q,v,pp,R,indices,'cuda')
    from dex_supermimic.rl.dex_supermimic.sim import SimReader
    need(Path(inspect.getsourcefile(SimReader.state)).samefile(checked(reader_pin)),'Foreign cached native Reader source')
    fake=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=q,qvel=v))),
      robot=SimpleNamespace(indexing=SimpleNamespace(joint_q_adr=indices['hand_qpos'],joint_v_adr=indices['hand_dof'])),
      obj=SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=indices['object_qpos'],free_joint_v_adr=indices['object_dof'])),palm_pos=pp,palm_rot=R)
    return SimReader.state(fake)


def row_schedule(n,operation,checktime,returned_callback):
    """One operation per row; fixture can verify scheduling without a device."""
    need(type(n) is int and n>0,'Positive fixed row count')
    for i in range(n):checktime();value=operation(i);returned_callback(i,value)


def flags_runtime(torch):
    def get(obj,name):
        try:return getattr(obj,name)
        except (AttributeError,RuntimeError):return 'UNAVAILABLE'
    mat=torch.backends.cuda.matmul
    return {'torch_version':str(torch.__version__),'CUDA_version':torch.version.cuda,'threads':torch.get_num_threads(),
      'allow_tf32':mat.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),
      'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
      'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
      'bf16_reduced_precision':get(mat,'allow_bf16_reduced_precision_reduction'),'fp16_reduced_precision':get(mat,'allow_fp16_reduced_precision_reduction'),
      'fp16_accumulation':get(mat,'allow_fp16_accumulation'),'bf16_splitK':get(mat,'allow_bf16_reduced_precision_reduction_split_k'),
      'autocast_enabled_CUDA':torch.is_autocast_enabled('cuda'),'autocast_dtype_CUDA':str(torch.get_autocast_dtype('cuda')),
      'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','CUBLAS_WORKSPACE_CONFIG')}}


def validate_result(r):
    need(r.get('schema')=='full151105_native_GPU_state_replay_report:1','Wrong native replay result')
    need(r.get('status') in ('COMPLETED_NATIVE_GPU_STATE_REPLAY_QUALIFIED','COMPLETED_RAW_NOT_QUALIFIED'),'Not a complete replay result')
    for k,v in EXPECTED.items():need(type(r['counts'].get(k)) is int and r['counts'][k]==v,'Actual replay count differs: '+k)
    for k in ('all_original_gates_passed','future_scoring_inputs_qualified','state_replay_qualified'):need(type(r.get(k)) is bool,'Explicit qualification bool')
    passed=r['all_original_gates_passed']
    need(r['future_scoring_inputs_qualified']==r['state_replay_qualified']==passed,'Qualification aliases differ')
    need(r['dense_scoring_eligibility'] is False and r['physical_gradient_qualified'] is False and r['whole_hidden_state_qualified'] is False,'No scoring/physical/hidden certificate')
    need((r['status']=='COMPLETED_NATIVE_GPU_STATE_REPLAY_QUALIFIED')==passed,'Result status and original flags differ')
    need((type(r.get('qualified_prestate')) is dict)==passed,'Qualified packet pointer must exist only for allpass')
    if not passed:need(r.get('qualified_prestate') is None,'No false qualification packet')
    return {'execution_complete':True,'all_original_gates_passed':passed,'future_scoring_inputs_qualified':passed,'exit_code':0 if passed else 1}


def run_replay(config_path,output,emit,*,started_perf=None):
    c=read(config_path);validate_config(c);config_pin=pin(config_path)
    need(c['GPU_run_authorized'] is True and os.environ.get('CUDA_VISIBLE_DEVICES')!='-1','Explicit new GPU authorization/visibility required')
    out=Path(output).resolve();need(out==Path(c['output']).resolve() and out.is_dir(),'Root-owned exact output directory required')
    started=time.perf_counter() if started_perf is None else started_perf;need(type(started) in (int,float) and started<=time.perf_counter(),'Startup clock required')
    def checktime():
        if time.perf_counter()-started>=90:raise TimeoutError('Whole-worker soft90 deadline; no retry')
    p=read(checked(c['protocol']));inputs=p['inputs'];attempted=returned=completed=0;result_array=np.full((2260,58),np.nan,np.float32);mask=np.zeros(2260,np.bool_);evidence=[];phase='input_read';rng_before=None
    def partial():
        atomic_npz(out/'replay_partial.npz',{'native_GPU_state58_raw':result_array,'completed_mask':mask})
        write(out/'call_evidence_partial.json',evidence)
        write(out/'partial_scope.json',{'phase':phase,'native_attempted':attempted,'native_returned':returned,'completed_rows':completed,'state_replay_qualified':False,'scoring_authorized':False})
    try:
        checktime()
        with np.load(checked(inputs['raw_conversion_all2260']),allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
        q,v,pp,R,mapping=validate_inputs(raw);input_arrays={'qpos':q,'qvel':v,'palm_pos':pp,'palm_rot':R};before_input={k:array_hash(a) for k,a in input_arrays.items()}
        phase='CUDA_startup';checktime()
        import torch
        need('warp' not in __import__('sys').modules,'No Warp needed for native Torch algebra')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
        torch.cuda.set_device(0);torch.cuda.synchronize(0)
        device=torch.device('cuda:0');properties=torch.cuda.get_device_properties(0)
        flags_before=flags_runtime(torch)
        write(out/'backend_before.json',{'flags':flags_before,'device':str(device),'name':properties.name,'capability':list(torch.cuda.get_device_capability(0)),
          'total_memory_bytes':properties.total_memory,'startup_initialization_scope':'Before RNG snapshots; only source-prescribed threads4/TF32True settings applied, no manual seed.'})
        def rng_arrays():
            return {'Python_pickle':np.frombuffer(pickle.dumps(random.getstate(),protocol=5),np.uint8).copy(),
              'NumPy_pickle':np.frombuffer(pickle.dumps(np.random.get_state(),protocol=5),np.uint8).copy(),
              'Torch_CPU':torch.get_rng_state().numpy().copy(),'Torch_CUDA_device0':torch.cuda.get_rng_state(0).numpy().copy()}
        rng_before=rng_arrays();atomic_npz(out/'RNG_before.npz',rng_before)
        uploaded={k:torch.tensor(a,dtype=torch.float32,device=device) for k,a in input_arrays.items()}
        indices={'hand_qpos':torch.tensor(mapping['hand_qpos'],dtype=torch.long,device=device),'hand_dof':torch.tensor(mapping['hand_dof'],dtype=torch.long,device=device),
          'object_qpos':torch.arange(mapping['object_qpos_start'],mapping['object_qpos_start']+7,dtype=torch.long,device=device),
          'object_dof':torch.arange(mapping['object_dof_start'],mapping['object_dof_start']+6,dtype=torch.long,device=device)}
        readback={k:t.cpu().numpy().copy() for k,t in uploaded.items()}
        need(all(array_hash(readback[k])==before_input[k] for k in input_arrays),'Input upload changed FP32 bytes')
        atomic_npz(out/'verified_upload_readback.npz',{**readback,**{k:t.cpu().numpy().copy() for k,t in indices.items()}})
        write(out/'upload_receipt.json',{'before':before_input,'GPU_readback':{k:array_hash(a) for k,a in readback.items()},'mapping_namespaces':'qpos/dof addresses from forensic23; no MJB/body/mocap inference'})
        phase='native_CUDA_B1_replay'
        def operation(i):
            nonlocal attempted,returned
            attempted+=1
            value=native_cuda_state_B1(uploaded['qpos'][i:i+1],uploaded['qvel'][i:i+1],uploaded['palm_pos'][i:i+1],uploaded['palm_rot'][i:i+1],indices,p['source_pins']['native_reader'])
            returned+=1
            evidence.append({'row':i,'batch_size':1,'q_shape':[1,27],'qvel_shape':[1,26],'palm_pos_shape':[1,3],'palm_R_shape':[1,3,3],
                'input_dtype':str(uploaded['qpos'].dtype),'input_device':str(uploaded['qpos'].device),'index_dtype':str(indices['hand_qpos'].dtype),
                'q_stride':list(uploaded['qpos'][i:i+1].stride()),'qvel_stride':list(uploaded['qvel'][i:i+1].stride()),'palm_pos_stride':list(uploaded['palm_pos'][i:i+1].stride()),
                'palm_R_stride':list(uploaded['palm_rot'][i:i+1].stride()),'q_storage_offset':uploaded['qpos'][i:i+1].storage_offset(),
                'output_shape':list(value.shape),'output_stride':list(value.stride()),'output_dtype':str(value.dtype),'output_device':str(value.device),'requires_grad':bool(value.requires_grad),
                'autocast_enabled_CUDA_inside':torch.is_autocast_enabled('cuda'),'grad_enabled_inside':torch.is_grad_enabled()})
            return value
        def receive(i,value):
            nonlocal completed
            need(value.shape==(1,58) and value.dtype==torch.float32 and value.device.type=='cuda' and not value.requires_grad,'Native output must be CUDA FP32 B1x58 without AD')
            cpu=value.detach().cpu().numpy().copy();need(cpu.dtype==np.float32 and np.isfinite(cpu).all(),'Nonfinite native output; no continuation')
            result_array[i]=cpu[0];mask[i]=True;completed+=1
            if completed%128==0:
                partial();emit(status='running_native_GPU_state_replay',phase=phase,converted_rows=completed,native_state_attempted=attempted,native_state_returned=returned)
        with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):row_schedule(2260,operation,checktime,receive)
        torch.cuda.synchronize(0);partial();checktime()
        need(mask.all() and attempted==returned==completed==2260,'Incomplete fixed replay')
        atomic_npz(out/'native_GPU_all2260_raw.npz',{'native_GPU_state58_raw':result_array,'completed_mask':mask})
        write(out/'call_evidence_all2260.json',evidence)
        phase='unchanged_full58_gates'
        endpoint=old_flags(result_array[1808:],raw['post20_reference'],True)
        anchor=old_flags(result_array[raw['anchor_pre_indices']],raw['anchor_references'],True)
        poses=[]
        for key in ('solver_position1808_old_rule','solver_rotation1808_old_rule'):
            with np.load(checked(inputs[key]),allow_pickle=False) as z:previous={k:z[k].copy() for k in z.files}
            current=old_flags(previous['actual'],previous['reference']);need(all(np.array_equal(current[k],previous[k]) for k in current),'Inherited forensic23 pose rule bytes/masks differ')
            poses.append(current);atomic_npz(out/(key+'_rechecked.npz'),current)
        atomic_npz(out/'endpoint452_original_rule.npz',endpoint);atomic_npz(out/'anchor160_original_rule.npz',anchor)
        write(out/'endpoint_identities.json',read(checked(inputs['endpoint_identities'])));write(out/'anchor_identities.json',read(checked(inputs['anchor_identities'])))
        all_numeric=bool(endpoint['old_flags'].all() and anchor['old_flags'].all() and all(x['old_flags'].all() for x in poses))
        rng_after=rng_arrays();atomic_npz(out/'RNG_after.npz',rng_after);flags_after=flags_runtime(torch);write(out/'backend_after.json',flags_after)
        invariants={'RNG_unchanged':all(array_hash(rng_before[k])==array_hash(rng_after[k]) for k in rng_before),'backend_flags_unchanged':flags_before==flags_after,
          'CPU_input_arrays_unchanged':all(array_hash(input_arrays[k])==before_input[k] for k in input_arrays),
          'GPU_input_readback_unchanged':all(array_hash(t.cpu().numpy())==before_input[k] for k,t in uploaded.items()),'no_Warp_import':'warp' not in __import__('sys').modules}
        need(all(invariants.values()),'Replay input/RNG/backend/source isolation failure')
        for x in inputs.values():checked(x)
        for x in c['source_pins'].values():checked(x)
        need(pin(config_path)==config_pin,'Actual config changed');checktime()
        if all_numeric:
            atomic_npz(out/'prestate1808_qualified_for_future_scoring.npz',{'native_GPU_prestate58':result_array[:1808],'completed_mask':mask[:1808],
                'source_pre_row':np.arange(1808,dtype=np.int64),'palm_pos':pp[:1808],'palm_rot':R[:1808],
                **{k:np.asarray(a) for k,a in mapping.items()}})
        counts=EXPECTED.copy();counts.update(native_state_calls=attempted,native_state_returned=returned,converted_rows=completed)
        report={'schema':'full151105_native_GPU_state_replay_report:1','status':'COMPLETED_NATIVE_GPU_STATE_REPLAY_QUALIFIED' if all_numeric else 'COMPLETED_RAW_NOT_QUALIFIED',
          'config':config_pin,'protocol':c['protocol'],'input_pins':inputs,'source_pins':c['source_pins'],'counts':counts,
          'all_original_gates_passed':all_numeric,'future_scoring_inputs_qualified':all_numeric,'state_replay_qualified':all_numeric,
          'qualified_prestate':pin(out/'prestate1808_qualified_for_future_scoring.npz') if all_numeric else None,
          'dense_scoring_eligibility':False,'proxy_scoring_authorized':False,'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,
          'original_CPU_failure_unchanged':'FAIL','invariants':invariants,'endpoint_failed_elements':int((~endpoint['old_flags']).sum()),'anchor_failed_elements':int((~anchor['old_flags']).sum()),
          'solver_pose_scope':'Pinned forensic23 actual/reference/oldflags independently recomputed; no new GPUquat/CPUstate/MJB call and not a full58 substitute',
          'RNG_scope':'after explicit CUDA0 initialization, selecteddevice0 only; no randomdraw/manualseed/restoration',
          'scope':'Single predeclared native CUDA+B1 arithmetic condition. No physics/hiddenstate/gradient certificate, scoring authorization or unique GPU/TF32/batch cause inference.',
          'elapsed_whole_worker_seconds':time.perf_counter()-started}
        validate_result(report);write(out/'report.json',report);write(out/'output_pins_core.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        try:partial()
        except BaseException as saving:error.add_note('Partial preservation also failed: '+repr(saving))
        try:
            write(out/'core_failure.json',{'schema':'full151105_native_GPU_replay_failure:1','error':repr(error),'phase':phase,'native_attempted':attempted,'native_returned':returned,
              'completed_rows':completed,'state_replay_qualified':False,'dense_scoring_eligibility':False,'proxy_scoring_authorized':False,'original_CPU_failure_unchanged':'FAIL','no_retry':True})
        except BaseException as saving:error.add_note('Failure metadata also failed: '+repr(saving))
        raise
