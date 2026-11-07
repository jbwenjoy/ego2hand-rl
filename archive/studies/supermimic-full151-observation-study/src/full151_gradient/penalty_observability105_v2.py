"""CPU sampled-penalty analysis of independently qualified native GPU states.

This NEW input route never calls a state converter or GPU. Default is JSON/hash
only. Explicit passive MJB constants preserve the original native geom catalog;
the original P_ONLY factory has its own separately disclosed passive model read.
The original failed CPU converter/result remain immutable and unqualified.
"""
from __future__ import annotations
import csv,json,os,pickle,random,re,sys,time,zipfile
from pathlib import Path
import numpy as np
from . import trained_bonus_analysis105_v1 as common
from . import penalty_observability105_v1 as legacy
from . import state_replay105_v2 as replay

ROOT=common.ROOT
PREREG=ROOT/'protocols/full151105_solver_phase_penalty_qualified_GPU_CPU_prereg_v4.json'
PREREG_SHA='c9a746f929d5943a03067c67d5b28f23e09b093084c2cde07cfae05f5db60028'
SCHEMA='full151105_solver_phase_penalty_qualified_GPU_CPU_run:1'
ROLE='same_solver_phase_penalty_observability_GPU_qualified_states'
KEYS={'schema','role','protocol','output','CPU_run_authorized','cpu_threads','source_pins','cpu_ready'}
OWN={'consumer':ROOT/'src/full151_gradient/penalty_observability105_v2.py',
     'CLI':ROOT/'scripts/diagnose_solver_phase_penalty_cpu_v2.py',
     'checker':ROOT/'scripts/check_solver_phase_penalty_cpu_v2.py'}
require=common.require;pin=common.pin;checked=common.checked;read=common.read;same_pin=common.same_pin
write=common._write;npwrite=common._npwrite
# These are EXACT frozen scientific definitions, not reimplemented classifiers.
pair_coverage=legacy.pair_coverage;group_rows=legacy.group_rows
native_contact_abi=legacy.native_contact_abi;exact_fp32=legacy.exact_fp32


def validate_header(c):
    require(type(c) is dict and set(c)==KEYS,'Exact qualified-state dense plan keys required')
    require(c['schema']==SCHEMA and c['role']==ROLE,'Wrong qualified-state dense schema/role')
    require(type(c['CPU_run_authorized']) is bool,'Native bool authorization required')
    require(type(c['cpu_threads']) is int and c['cpu_threads']==1,'CPUthreads1 fixed')
    require(type(c['source_pins']) is dict and c['source_pins'],'Nonempty flat source table required')
    require(type(c['output']) is str and Path(c['output']).is_absolute(),'Absolute output required')
    out=Path(c['output']).resolve()
    require(out.parent==ROOT/'outputs' and out.name.startswith('full151105_solver_phase_penalty_GPU_qualified_CPU_') and not out.exists(),'Fresh scoped output, no overwrite')


def required_sources(p):
    result=legacy.required_sources(p)
    gpu=read(checked(p['inputs']['GPU_replay_config']))
    for k,v in gpu['source_pins'].items():result['GPU_replay__'+k]=v
    saved=read(checked(p['inputs']['GPU_replay_saved_independent']))
    for k in ('auditor','math'):result['GPU_replay_saved__'+k]=saved[k]
    result['legacy_dense_scientific_definitions']=pin(legacy.__file__)
    result['native_GPU_replay_contract']=pin(replay.__file__)
    result.update({k:pin(v) for k,v in OWN.items()})
    return result


def declaration():
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Default noTorch/Warp required')
    require(pin(PREREG)['sha256']==PREREG_SHA,'Prereg changed')
    return {'schema':'full151105_qualified_dense_penalty_CPU_declaration:1','status':'DECLARED_NOT_READ_NOT_SCORED',
       'runplan_schema':SCHEMA,'role':ROLE,'keys':sorted(KEYS),'protocol':pin(PREREG),'source_pins':{k:pin(v) for k,v in OWN.items()},
       'Torch_imported':False,'Warp_imported':False,'NPZ_opened':0,'MJB_loads':0,'state_conversions':0,'objective_constructs':0,'proxy_forwards':0,'WM_Actor_Env_GPU':0}


def verify_core_manifest(core,mutable,run,expected_immutable_count):
    """The sole historical runtime update is explicit; every other pin is strict."""
    run=Path(run)
    require(type(core) is dict and set(core).issuperset({'runtime.json','report.json','prestate1808_qualified_for_future_scoring.npz'}),'Incomplete core immutable manifest')
    require(type(mutable) is dict and set(mutable)=={'runtime.json'},'Only runtime.json terminal update exception permitted')
    exception=mutable['runtime.json'];oldpin=core['runtime.json']
    require(type(oldpin) is dict and set(oldpin)=={'path','sha256'} and type(oldpin['path']) is str and Path(oldpin['path']).is_absolute() and type(oldpin['sha256']) is str and len(oldpin['sha256'])==64 and all(ch in '0123456789abcdef' for ch in oldpin['sha256']),'Malformed preterminal runtime pin')
    require(common.canonical(oldpin)==common.canonical(exception['core_preterminal_pin']),'Wrong preterminal mutable exception')
    terminal=checked(exception['terminal_pin']);require(terminal.samefile(run/'runtime.json') and Path(oldpin['path']).resolve(strict=True).samefile(terminal),'Mutable runtime exception moved to a different file')
    require(type(exception['hash_equal']) is bool and exception['hash_equal']==(oldpin['sha256']==exception['terminal_pin']['sha256']),'Mutable hash annotation differs')
    immutable=0
    for name,value in core.items():
        if name=='runtime.json':continue
        require(checked(value).samefile(run/name),'Core output pin route differs: '+name);immutable+=1
    require(type(expected_immutable_count) is int and immutable==expected_immutable_count,'Independent immutable output count differs')
    return terminal,immutable


def qualified_replay_metadata(inputs):
    """JSON/hash joins only; no NPZ, GPU, native state, or mutable-pin bypass."""
    saved=read(checked(inputs['GPU_replay_saved_independent']));common.all_checks(saved)
    require(saved['schema']=='full151105_matched_native_GPU_reader_saved_independent:1' and saved['status']=='PASS_SAVED_NATIVE_REPLAY_REVIEW' and saved['engineering_qualified'] is True,'Wrong independent native replay scope')
    require(same_pin(saved['config'],inputs['GPU_replay_config']) and same_pin(saved['producer_report'],inputs['GPU_replay_report']) and same_pin(saved['qualified_prestate'],inputs['qualified_GPU_prestate1808']),'Independent replay provenance aliases differ')
    for k in ('state_replay_qualified','future_scoring_inputs_qualified'):require(saved[k] is True,'Replay states not independently qualified')
    for k in ('dense_scoring_eligibility','physical_gradient_qualified','whole_hidden_state_qualified'):require(saved[k] is False,'Independent state proof must not elevate science scope')
    for k in ('auditor','math'):checked(saved[k])
    config=read(checked(inputs['GPU_replay_config']));replay.validate_header(config)
    require(config['GPU_run_authorized'] is True and same_pin(config['protocol'],inputs['GPU_replay_protocol']),'Actual replay config/protocol differs')
    report=read(checked(inputs['GPU_replay_report']));decision=replay.validate_result(report)
    require(decision['all_original_gates_passed'] is True and report['status']=='COMPLETED_NATIVE_GPU_STATE_REPLAY_QUALIFIED','Original complete58 replay gates did not pass')
    require(same_pin(report['config'],inputs['GPU_replay_config']) and same_pin(report['protocol'],inputs['GPU_replay_protocol']) and same_pin(report['qualified_prestate'],inputs['qualified_GPU_prestate1808']),'Producer replay aliases differ')
    run=checked(inputs['GPU_replay_report']).parent
    require(Path(saved['run_dir']).resolve(strict=True).samefile(run) and Path(config['output']).resolve(strict=True).samefile(run),'Actual replay output route differs')
    require(common.canonical(report['source_pins'])==common.canonical(config['source_pins']),'Replay source maps differ')
    from . import analysis105_v1 as old
    watch=Path(str(run)+'_watchdog')
    require(checked(inputs['GPU_replay_identity']).samefile(watch/'engineering_identity.json'),'Replay identity path differs')
    require(checked(inputs['GPU_replay_core_output_manifest']).samefile(run/'output_pins_core.json'),'Core manifest input route differs')
    require(checked(inputs['GPU_replay_watchdog_state']).samefile(watch/'state.json') and checked(inputs['GPU_replay_post_terminal_CIM']).samefile(watch/'post_terminal_CIM.json'),'Replay terminal watchdog route differs')
    core=read(checked(inputs['GPU_replay_core_output_manifest']));mutable=saved['details']['mutable_core_pins']
    terminal,immutable=verify_core_manifest(core,mutable,run,saved['details']['core_immutable_output_pin_count'])
    exception=mutable['runtime.json'];oldpin=core['runtime.json']
    require(same_pin(exception['terminal_pin'],inputs['GPU_replay_terminal_runtime']),'Actual terminal runtime input differs')
    state=read(watch/'state.json');identity=read(checked(inputs['GPU_replay_identity']));post=read(watch/'post_terminal_CIM.json');runtime=read(terminal)
    old._validate_lifecycle({'physical_run':str(run),'artifact_pins':{'producer_config':inputs['GPU_replay_config']}},
        {'watch_state':state,'engineering_identity':identity,'post_terminal_CIM':post,'runtime':runtime})
    entry=replay.OWN['entry'];require(Path(state['command'][2]).resolve(strict=True).samefile(entry),'Unexpected actual replay entry')
    require(runtime['source_sha256']==pin(entry)['sha256'] and runtime['status']=='completed' and runtime['execution_complete'] is True and runtime['exit_code']==0,'Replay terminal source/status differs')
    require(runtime['all_original_gates_passed'] is True and runtime['future_scoring_inputs_qualified'] is True,'Terminal qualification differs')
    require(runtime['pid']==identity['actual_pid']==state['actual_pid'] and runtime['parent_pid']==identity['launcher_pid']==state['launcher_pid'],'Replay owned PIDs differ')
    require(set(post['queried_pids'])=={runtime['pid'],runtime['parent_pid']} and post['actual_CIM_rows']==[],'Replay exact owned process absence unknown')
    completion=read(checked(inputs['GPU_replay_terminal_completion']))
    require(checked(inputs['GPU_replay_terminal_completion']).samefile(run/'completion.json') and completion['schema']=='full151105_native_GPU_state_replay_worker_completion:1' and same_pin(completion['config'],inputs['GPU_replay_config']),'Replay completion route differs')
    require(completion['result_status']==report['status'] and common.canonical(completion['result_validation'])==common.canonical(decision),'Replay completion/result differs')
    require(completion['physical_gradient_qualified'] is False and completion['whole_hidden_state_qualified'] is False and type(completion['factory_or_scoring_calls']) is int and completion['factory_or_scoring_calls']==0,'Replay completion scope elevated')
    for here,there in (('forensic_solver_position1808','solver_position1808_old_rule'),('forensic_solver_rotation1808','solver_rotation1808_old_rule')):
        require(same_pin(inputs[here],report['input_pins'][there]),'Forensic solver reference provenance differs')
    return {'independent_saved_review':inputs['GPU_replay_saved_independent'],'producer_report':inputs['GPU_replay_report'],
        'actual_config':inputs['GPU_replay_config'],'qualified_prestate':inputs['qualified_GPU_prestate1808'],
        'actual_runtime_terminal':pin(terminal),'mutable_core_runtime_preterminal':oldpin,'immutable_core_pin_count':immutable,
        'watch_state':pin(watch/'state.json'),'engineering_identity':inputs['GPU_replay_identity'],'post_terminal_CIM':pin(watch/'post_terminal_CIM.json'),
        'auditor':saved['auditor'],'math':saved['math'],'completion':inputs['GPU_replay_terminal_completion'],'future_scoring_inputs_qualified':True,'dense_scoring_was_previously_authorized':False}


def metadata_preflight(config_path):
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Fresh default metadata process only')
    c=read(config_path);validate_header(c)
    require(same_pin(c['protocol'],{'path':str(PREREG),'sha256':PREREG_SHA}),'Wrong fixed prereg')
    p=read(checked(c['protocol']));require(p['schema']=='full151105_penalty_solver_phase_observability_CPU_preregistration:1' and p['revision']==4 and p['CPU_run_authorized'] is False,'Wrong source-only protocol')
    require(p['runplan_schema']==SCHEMA and p['role']==ROLE and set(p['runplan_keys'])==KEYS,'New input-route protocol ABI differs')
    for v in p['inputs'].values():checked(v)
    needed=required_sources(p);have={(str(checked(v)).casefold(),v['sha256']) for v in c['source_pins'].values()}
    require(all((str(checked(v)).casefold(),v['sha256']) in have for v in needed.values()),'Transitive source closure missing')
    ready=read(checked(c['cpu_ready']));require(ready.get('schema')=='full151105_qualified_dense_penalty_CPU_contract_READY:1' and ready.get('status')=='PASS_SYNTHETIC_QUALIFIED_PACKET_ALIGNMENT_AND_HOOK_ONLY','Wrong narrow CPU READY')
    proof=read(checked(ready['cpu_report']));common.all_checks(proof)
    require(proof['schema']=='full151105_qualified_dense_penalty_CPU_contract_proof:1' and same_pin(ready['protocol'],c['protocol']),'READY/prereg mismatch')
    for k,v in OWN.items():require(same_pin(ready['source_pins'][k],pin(v)) and same_pin(proof['source_pins'][k],pin(v)),'Executed source mismatch: '+k)
    for k in ('actual_physical_state_conversions','actual_MJB_loads','actual_objective_constructs','actual_full05_data_reads','actual_GPU_packet_array_reads'):require(type(proof['scope'][k]) is int and proof['scope'][k]==0,'Fixture scope elevated: '+k)
    saved=read(checked(p['inputs']['physical_saved_independent1762']));common.all_checks(saved)
    require(saved['schema']=='full151105_full_serial_saved_independent:1' and saved['completed_branches']==152 and saved['baseline_count']==24 and saved['episode_step_returned']==1808 and saved['physical_gradient_qualified'] is False,'Physical completion mismatch')
    require(same_pin(saved['config'],p['inputs']['physical_config']),'Actual physical config mismatch')
    spec=read(checked(p['inputs']['objective_spec']));require(spec['schema']=='full151_declared_H123_objective_spec:2' and spec['CPU_authorized'] is True,'Exact CPU-capable trained spec required')
    replay_join=qualified_replay_metadata(p['inputs'])
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Metadata imported backend')
    return {'config':c,'config_pin':pin(config_path),'protocol':p,'replay_join':replay_join,'status':'PASS_HASH_JSON_ONLY_GPU_QUALIFIED_INPUT_NOT_SCORED',
        'Torch_imported':False,'Warp_imported':False,'NPZ_opened':0,'state_conversions':0,'objective_constructs':0}


def packet_abi(packet,raw,expected_rows=1808):
    """Typed/value validation only. No native state call, cast-repair or frame transform."""
    keys={'native_GPU_prestate58','completed_mask','source_pre_row','palm_pos','palm_rot','hand_qpos','hand_dof','object_qpos_start','object_dof_start'}
    require(type(packet) is dict and set(packet)==keys,'Exact qualified packet keys required')
    n=expected_rows;require(type(n) is int and n>0,'Positive row count required')
    for k,shape in (('native_GPU_prestate58',(n,58)),('palm_pos',(n,3)),('palm_rot',(n,3,3))):
        a=packet[k];require(isinstance(a,np.ndarray) and a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Exact finite packet FP32 ABI: '+k)
    mask=packet['completed_mask'];ids=packet['source_pre_row']
    require(isinstance(mask,np.ndarray) and mask.dtype==np.bool_ and mask.shape==(n,) and mask.all(),'Every qualified row must be completed')
    require(isinstance(ids,np.ndarray) and ids.shape==(n,) and ids.dtype==np.int64 and np.array_equal(ids,np.arange(n,dtype=np.int64)),'Exact int64 source rows without missing/reordering')
    mapping=replay.mapping_from_raw(packet)
    for k in ('hand_qpos','hand_dof','object_qpos_start','object_dof_start'):
        a,b=np.asarray(packet[k]),np.asarray(raw[k]);require(a.dtype.kind in 'iu' and b.dtype.kind in 'iu' and a.shape==b.shape and np.array_equal(a,b),'Mapping typed values changed: '+k)
    x=packet['native_GPU_prestate58'];q=raw['pre_qpos'];v=raw['pre_qvel']
    require(q.dtype==v.dtype==np.float32 and q.shape==(n,27) and v.shape==(n,26) and np.isfinite(q).all() and np.isfinite(v).all(),'Original pre-q/v exact FP32 ABI')
    require(x[:,:20].tobytes()==q[:,mapping['hand_qpos']].tobytes() and x[:,20:40].tobytes()==v[:,mapping['hand_dof']].tobytes(),'Qualified GPU q/qd no longer bitexact raw input')
    for k,source in (('palm_pos','palm_pos_all2260'),('palm_rot','palm_rot_all2260')):
        a,b=packet[k],raw[source][:n];require(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),'Recorded Reader frame changed: '+k)
    return mapping


def pose_alignment(x,palm_pos,palm_rot,solver_world_p,solver_world_R):
    """NEW declared F64-derived reference, SAME original 8eps32 component rule.

    This is not a replacement native state converter, quaternion operation or
    original GPU FP32 reference. All rows survive into saved arithmetic arrays.
    """
    x,pp,R,sp,sR=map(np.asarray,(x,palm_pos,palm_rot,solver_world_p,solver_world_R));n=len(x)
    for a,shape in ((x,(n,58)),(pp,(n,3)),(R,(n,3,3)),(sp,(n,3)),(sR,(n,3,3))):
        require(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Finite FP32 alignment operands required')
    inv=R.astype(np.float64).swapaxes(-1,-2)
    expected_p=(inv*(sp.astype(np.float64)-pp.astype(np.float64))[:,None,:]).sum(-1,dtype=np.float64)
    expected_R=(inv[:,:,:,None]*sR.astype(np.float64)[:,None,:,:]).sum(-2,dtype=np.float64)
    actual=np.concatenate((x[:,40:43],x[:,43:52]),axis=-1).astype(np.float64)
    reference=np.concatenate((expected_p,expected_R.reshape(n,9)),axis=-1)
    delta=actual-reference;bound=8*2**-23*np.maximum(1.,np.maximum(abs(actual),abs(reference)));flags=abs(delta)<=bound
    return {'actual_GPU_p_R_F64':actual,'derived_reference_F64':reference,'difference_F64':delta,'original_8eps_bound_F64':bound,
        'original_8eps_flags':flags,'palm_pos_FP32':pp.copy(),'palm_rot_FP32':R.copy(),'solver_world_p_FP32':sp.copy(),'solver_world_R_FP32':sR.copy()}


def original_solver_rule(record,rows,components):
    """Recheck saved FP32 pairs in their ACTUAL flat3/flat9 storage ABI."""
    keys={'actual','reference','difference_F64','old_bound_F64','old_flags'}
    require(type(record) is dict and set(record)==keys,'Exact original saved solver rule keys required')
    require(type(rows) is int and rows>0 and components in (3,9),'Fixed flat solver component count')
    for k in ('actual','reference'):
        a=record[k];require(a.dtype==np.float32 and a.shape==(rows,components) and np.isfinite(a).all(),'Original solver FP32 flat component ABI')
    calculated=replay.old_flags(record['actual'],record['reference'])
    for k in keys:
        a,b=np.asarray(record[k]),calculated[k]
        require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'Original solver arithmetic record differs: '+k)
    require(calculated['old_flags'].all(),'Original solver rule did not pass')
    return record['actual']


def _jsonl(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]


def run_analysis(config_path):
    started=time.perf_counter();p=metadata_preflight(config_path);c=p['config']
    require(c['CPU_run_authorized'] is True and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Explicit RootCPU authorization required')
    out=Path(c['output']);out.mkdir();phase='archive';counts={k:{'attempted':0,'returned':0} for k in ('passive_mapping_MJB','objective_factory','objective_forward')};completed=0;invariants={}
    def call(k,fn,*args,**kwargs):counts[k]['attempted']+=1;v=fn(*args,**kwargs);counts[k]['returned']+=1;return v
    try:
        write(out,'config.json',c);write(out,'protocol.json',p['protocol']);write(out,'source_inventory.json',c['source_pins']);write(out,'qualified_replay_provenance.json',p['replay_join'])
        with zipfile.ZipFile(out/'execution_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for i,v in enumerate(c['source_pins'].values()):z.write(checked(v),'%04d_%s'%(i,Path(v['path']).name))
        phase='saved_metadata';inputs=p['protocol']['inputs']
        samples=_jsonl(checked(inputs['physical_samples']));rewards=_jsonl(checked(inputs['reward_snapshots']));groups,reward_map=group_rows(samples,rewards)
        from . import analysis105_v1 as old
        run=checked(inputs['physical_samples']).parent;saved=read(checked(inputs['physical_saved_independent1762']))
        evidence={str(checked(v)).casefold():v for v in saved['evidence'].values()}
        def ev(path):return checked(evidence[str(Path(path).resolve(strict=True)).casefold()])
        old._validate_capacity(read(ev(run/'capacity_ledgers.json')))
        watch=Path(str(run)+'_watchdog')
        old._validate_lifecycle({'physical_run':str(run),'artifact_pins':{'producer_config':inputs['physical_config']}},
          {'watch_state':read(ev(watch/'state.json')),'engineering_identity':read(ev(watch/'engineering_identity.json')),'post_terminal_CIM':read(ev(watch/'post_terminal_CIM.json')),'runtime':read(ev(run/'runtime.json'))})
        phase='qualified_GPU_packet_and_NEW_declared_pose_alignment'
        forensic=read(checked(inputs['GPU_replay_report']))['input_pins']['raw_conversion_all2260']
        with np.load(checked(forensic),allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
        with np.load(checked(inputs['qualified_GPU_prestate1808']),allow_pickle=False) as z:packet={k:z[k].copy() for k in z.files}
        mapping=packet_abi(packet,raw,expected_rows=1808)
        converted=packet['native_GPU_prestate58'];q=exact_fp32([r['pre_qpos'] for r in samples],(1808,27),'allpreqpos');v=exact_fp32([r['pre_qvel'] for r in samples],(1808,26),'allpreqvel')
        require(q.tobytes()==raw['pre_qpos'].tobytes() and v.tobytes()==raw['pre_qvel'].tobytes(),'Forensic and physical source rows differ')
        pp=packet['palm_pos'];R=packet['palm_rot']
        for key,indices in groups:
            rw=reward_map[key]
            pos=exact_fp32(rw['state20_reader_palm_pos'],(1,3),'recordedReaderpalm')[0];rot=exact_fp32(rw['state20_reader_palm_rot'],(1,3,3),'recordedReaderR')[0]
            require(np.array_equal(pp[indices],np.broadcast_to(pos,(4,3))) and np.array_equal(R[indices],np.broadcast_to(rot,(4,3,3))),'GPU packet recorded Reader frame/group differs')
        with np.load(checked(inputs['forensic_solver_position1808']),allow_pickle=False) as z:solver_p={k:z[k].copy() for k in z.files}
        with np.load(checked(inputs['forensic_solver_rotation1808']),allow_pickle=False) as z:solver_R={k:z[k].copy() for k in z.files}
        source_pose=exact_fp32([s['object_solver_pose'] for s in samples],(1808,7),'allsolverpose')
        world_p=original_solver_rule(solver_p,1808,3)
        world_R_flat=original_solver_rule(solver_R,1808,9)
        world_R=world_R_flat.reshape(1808,3,3,order='C')
        require(world_R.tobytes()==world_R_flat.tobytes(),'Logical R reshape must preserve original flat9 bytes')
        require(np.array_equal(world_p,source_pose[:,:3]),'Saved solver position provenance differs')
        require(np.array_equal(raw['object_solver_pose'],source_pose),'Forensic and physical solver poses differ')
        alignment=pose_alignment(converted,pp,R,world_p,world_R)
        npwrite(out,'GPU_prestate_solver_pose_alignment_all1808.npz',alignment)
        npwrite(out,'consumed_GPU_state58.npz',{'pre_x':converted,'pre_qpos':q,'pre_qvel':v,'palm_pos':pp,'palm_rot':R,'source_pre_row':packet['source_pre_row'],**{k:np.asarray(value) for k,value in mapping.items()}})
        gates={'inherited_GPU_full58_replay_original_gates_passed':True,'independent_saved_replay_passed':True,
          'NEW_all1808_GPU_p_R_derived_alignment_passed':bool(alignment['original_8eps_flags'].all()),'alignment_reference':'NEW NumPy F64 explicit mul+sum from saved solver-world p/R and recorded palm; not original GPUFP32 reference',
          'old_8eps_rule_unchanged':True,'CPU_state_conversions':0,'factory_calls_before_gates':0,'frame_scope':'452actualReader frames + fixed-palm source/topology; NOT1808 directmocap readings',
          'historical_CPU_converter_failure_unchanged':True}
        write(out,'state_input_qualification.json',gates)
        require(gates['NEW_all1808_GPU_p_R_derived_alignment_passed'],'NEW GPU prestate/solver p-R derived alignment exceeds original8eps rule; no factory')
        phase='passive_native_geom_constants'
        import torch
        require(not torch.cuda.is_initialized() and 'warp' not in sys.modules,'Fresh noCUDA/noWarp CPU process required')
        torch.set_num_threads(1)
        def rng():return common.hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)+torch.get_rng_state().numpy().tobytes()).hexdigest()
        rng_before=rng()
        import mujoco
        from full151_native.bundle import load_bundle,FIELDS
        bundle=load_bundle(checked(inputs['full151_bundle']),expected_sha256=inputs['full151_bundle']['sha256'])
        for path in ('A0','A2'):
            with np.load(checked(inputs[path+'__expected_context_fields.npz']),allow_pickle=False) as z:expected={k:z[k].copy() for k in z.files}
            with np.load(checked(inputs[path+'__actual_context_readbacks.npz']),allow_pickle=False) as z:actual={k:z[k].copy() for k in z.files}
            require(set(expected)==set(FIELDS) and set(actual)=={stage+'__'+f for stage in ('after_copy','after_set_const_0') for f in FIELDS},'Actual8 sourcefield set differs')
            for f in FIELDS:
                wanted=np.asarray(bundle['context'][f])[0]
                require(np.array_equal(expected[f],wanted),'Saved expected actual8 differs: '+f)
                for stage in ('after_copy','after_set_const_0'):
                    got=actual[stage+'__'+f]
                    require(got.ndim==wanted.ndim+1 and got.shape[1:]==wanted.shape and np.array_equal(got,np.broadcast_to(wanted.astype(got.dtype),got.shape)),'Actual8 double readback differs: '+path+'/'+stage+'/'+f)
        model=call('passive_mapping_MJB',mujoco.MjModel.from_binary_path,str(checked(inputs['nominal_model'])))
        require((model.nq,model.nv)==(27,26) and model.nplugin==0,'Fixed passive geometry model ABI required')
        # Native depth is a generated-pair metric, never positive-force filtering.
        hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])}
        obj=model.body('object/cube').id;objects=set(np.flatnonzero(model.geom_bodyid==obj).tolist())
        native_depth=[]
        for s in samples:
            geom,typ=native_contact_abi(s,model.ngeom)
            info=pair_coverage(s['contacts'],hand,objects,set())
            info['primary_deepest_tie_coverage']=None;info['secondary_all_violating_coverage']=None
            info['proxy_coverage_status']='NOT_EVALUATED_BEFORE_FACTORY'
            require(info['native_generated_HO_depth_m']==s['measured_hand_object']['penetration_by_category_m']['hand_object'],'HO depth/category saved mismatch')
            native_depth.append(info)
        write(out,'native_depth_before_proxy.json',native_depth)
        phase='one_actual_P_ONLY_factory'
        from full151_intent.binding_v3 import build_penalty_only_objective
        objective=call('objective_factory',build_penalty_only_objective,read(checked(inputs['objective_spec'])),device='cpu');base=objective.base
        require(not objective.training and all(not t.requires_grad for t in objective.parameters()),'Frozen CPU objective required')
        before_buffers=common._tensor_digest(objective.state_dict())
        oldmeta=read(checked(inputs['previous_bonus_factory_geometry_manifest']))['P_ONLY']['geometry']
        require(common.canonical(base.geometry_proxy.metadata)==common.canonical(oldmeta),'Existing factory/sample geometry manifest differs')
        sample_geom=base.geometry_proxy.sample_geom_ids.detach().numpy().copy();unique=sorted(set(sample_geom.tolist()))
        require(sample_geom.shape==(1344,) and len(unique)==21,'Exact existing1344/21 sampling required')
        native_body_ids=list(map(int,base.geometry_proxy.fk.body_ids))
        write(out,'geometry_ID_mapping.json',{'native_ngeom':int(model.ngeom),'native_object_body_id':int(obj),
            'native_geom_catalog':[{'id':g,'name':model.geom(g).name,'type':int(model.geom_type[g]),'native_body_id':int(model.geom_bodyid[g]),'native_body_name':model.body(int(model.geom_bodyid[g])).name,'contype':int(model.geom_contype[g]),'conaffinity':int(model.geom_conaffinity[g])} for g in range(model.ngeom)],
            'native_body_catalog':[{'id':b,'name':model.body(b).name,'parent_id':int(model.body_parentid[b]),'mocap_id':int(model.body_mocapid[b])} for b in range(model.nbody)],
            'native_hand_rule':'fullmatch robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col) AND (contype OR conaffinity)',
            'FK_local_body_table':[{'local_FK_index':i,'native_body_id':b,'native_body_name':model.body(b).name} for i,b in enumerate(native_body_ids)],
            'sample_body_ids_semantics':'local_FK_index, not nativebody/mocapID','sample_geom_ids_semantics':'nativeMJB geomID',
            'native_hand_geoms':[{'id':g,'name':model.geom(g).name,'native_body_id':int(model.geom_bodyid[g])} for g in sorted(hand)],
            'native_object_geoms':[{'id':g,'name':model.geom(g).name,'native_body_id':int(model.geom_bodyid[g])} for g in sorted(objects)],
            'actual_proxy_hand_geometry_records':base.geometry_proxy.metadata['hand_collision_geoms'],'actual_proxy_object_geometry_records':base.geometry_proxy.metadata['object_geoms']})
        npwrite(out,'proxy_sample_buffers.npz',{'sample_geom_ids':sample_geom,'sample_body_ids':base.geometry_proxy.sample_body_ids.detach().numpy().copy(),
            'sample_group_ids':base.geometry_proxy.sample_group_ids.detach().numpy().copy(),'sample_points_body':base.geometry_proxy.sample_points_body.detach().numpy().copy()})
        write(out,'factory_receipt.json',objective.metadata)
        phases=np.asarray([s['phase'] for s in samples],np.int64);parts=[];penalties=[];bonuses=[];mins=[];pergeom=[]
        inputs_before=common.array_digest({'q':q,'v':v,'palm_pos':pp,'palm_R':R,'converted':converted})
        phase='fifteen_streamed_P_ONLY_forwards'
        for batch,start in enumerate(range(0,1808,128)):
            stop=min(start+128,1808)
            with torch.no_grad():score,terms,capture=call('objective_forward',common.evaluate_objective,objective,torch.from_numpy(converted[start:stop].copy()),phases[start:stop])
            sdf=capture['sampled_signed_distance_m'];require(sdf.shape==(stop-start,1344),'Actual SDF shape differs')
            P=terms['geometry_intent_penetration_prior'].numpy().copy();BP=score.numpy().copy();per=np.stack([sdf[:,sample_geom==g].min(-1) for g in unique],axis=-1)
            chunk={'row_indices':np.arange(start,stop),'sampled_sdf_m':sdf,'P':P,'BP':BP,'minimum_sdf_m':capture['minimum_signed_distance_m'],'geom_ids':np.asarray(unique),'per_geom_min_sdf_m':per}
            npwrite(out,'chunk_%02d.npz'%batch,chunk);parts.append(sdf);penalties.append(P);bonuses.append(BP);mins.append(capture['minimum_signed_distance_m']);pergeom.append(per);completed=stop
        sdf=np.concatenate(parts);P=np.concatenate(penalties);BP=np.concatenate(bonuses);minimum=np.concatenate(mins);per=np.concatenate(pergeom)
        require(counts['objective_forward']['returned']==15 and completed==1808,'Exact15batches/all1808 required')
        results=[]
        for i,s in enumerate(samples):
            info=pair_coverage(s['contacts'],hand,objects,set(unique),{g:float(per[i,j]) for j,g in enumerate(unique)})
            observed=bool(P[i]>0);info.update(row_id=i,path=s['path'],mode=s['mode'],k=s['phase'],branch_id=s['branch_id'],substep=s['substep'],
               canonical_pre_tick=s['canonical_pre_tick'],source_frame=s['source_frame'],P=float(P[i]),BP=float(BP[i]),minimum_sampled_sdf_m=float(minimum[i]),
               proxy_sample_depth_m=max(0.,-float(minimum[i])),proxy_P_positive=observed,proxy_min_sdf_below_minus3mm=bool(minimum[i]<-.003),
               proxy_min_sdf_exactly_minus_FP32_allowance=bool(minimum[i]==np.float32(-.003)),
               native_depth_minus_nominal_3mm_m=info['native_generated_HO_depth_m']-.003,
               primary_miss=bool(info['native_HO_depth_gt3mm'] and not observed),proxy_alert_without_observed_native_HO3mm=bool(observed and not info['native_HO_depth_gt3mm']))
            results.append(info)
        def summary(selected):
            return {'rows':len(selected),'native_HO_depth_gt3mm':sum(r['native_HO_depth_gt3mm'] for r in selected),'proxy_P_positive':sum(r['proxy_P_positive'] for r in selected),
                    'both_positive':sum(r['native_HO_depth_gt3mm'] and r['proxy_P_positive'] for r in selected),'native_positive_proxy_zero':sum(r['primary_miss'] for r in selected),
                    'proxy_alert_without_nativeHO3mm':sum(r['proxy_alert_without_observed_native_HO3mm'] for r in selected),
                    'both_nonpositive':sum(not r['native_HO_depth_gt3mm'] and not r['proxy_P_positive'] for r in selected),
                    'primary_deepest_tie_geom_gap_rows':sum(r['primary_miss'] and bool(r['primary_deepest_tie_coverage']['geom_set_gap_ids']) for r in selected),
                    'primary_deepest_tie_covered_undersensitivity_rows':sum(r['primary_miss'] and bool(r['primary_deepest_tie_coverage']['covered_proxy_undersensitivity_ids']) for r in selected),
                    'overlapping_categories_not_independent_N':True}
        stats={'all1808':summary(results)}
        for path in ('A0','A2'):
            for mode in ('donor','baseline','direction'):stats[path+'/'+mode]=summary([r for r in results if r['path']==path and r['mode']==mode])
        npwrite(out,'all1808_proxy_scores.npz',{'sampled_sdf_m':sdf,'P':P,'BP':BP,'minimum_sdf_m':minimum,'geom_ids':np.asarray(unique),'per_geom_min_sdf_m':per,
            'native_HO_depth_m':np.asarray([r['native_generated_HO_depth_m'] for r in results]),'proxy_P_positive':P>0,'phases':phases})
        write(out,'rows_all1808.json',results);write(out,'statistics.json',stats)
        fields=['row_id','path','mode','k','branch_id','substep','canonical_pre_tick','source_frame','native_generated_HO_count','native_generated_HO_depth_m','native_HO_depth_gt3mm','P','BP','minimum_sampled_sdf_m','proxy_P_positive','primary_miss']
        with (out/'rows_all1808.csv').open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
        invariants={'objective_parameters_buffers_unchanged':common._tensor_digest(objective.state_dict())==before_buffers,'RNG_unchanged':rng()==rng_before,
           'qualified_inputs_unchanged':common.array_digest({'q':q,'v':v,'palm_pos':pp,'palm_R':R,'converted':converted})==inputs_before,
           'CUDA_uninitialized':not torch.cuda.is_initialized(),'Warp_not_imported':'warp' not in sys.modules,'no_parameter_gradients':all(t.grad is None for t in objective.parameters())}
        require(all(invariants.values()),'Read-only CPU invariance failed')
        for value in inputs.values():checked(value)
        for value in c['source_pins'].values():checked(value)
        require(pin(config_path)==p['config_pin'],'Actual CPU plan changed')
        report={'schema':'full151105_same_solver_phase_penalty_qualified_GPU_CPU_analysis:1','status':'COMPLETED_SAME_SOLVER_PHASE_PROXY_OBSERVABILITY_NOT_PHYSICS_CERTIFICATE',
          'config':p['config_pin'],'protocol':c['protocol'],'source_pins':c['source_pins'],'input_pins':inputs,'counts':counts,'new_state_conversions':0,'qualified_GPU_rows_consumed':1808,'scored_rows':1808,'qualified_replay_provenance':p['replay_join'],
          'actual_factory_passive_MJB_read':'1source-accounted inside factory, not separately instrumented; explicitmapping load1 counted above',
          'state_input_gates':gates,'invariants':invariants,'statistics':stats,'primary_deepest_pairs':'EXACT all-tied deepest set; no firstargmax choice',
          'threshold_arithmetic':{'native_nominal_gate_m':.003,'proxy_FP32_scalar_allowance_m':float(np.float32(.003)),'unchanged_primary_values':True,'exact_FP32_boundary_flag_saved':True},
          'secondary_all_violating_pairs':'separately named overlapping flags, not replacing primarydeepest rule',
          'scope':'CPU FP32 originalP_ONLY on independently qualified native GPU solverPREstate, with NEW saved-only F64 p/R alignment. Historical CPU conversion remains FAILED; no conversion rerun. Native label generatedHOcandidate depth>3mm, not positiveforce, selfcontact or global geometrytruth. Nsource1/Nseed1/Nego0, repeatedbranches dependent. No wholehidden/physicalgradient/forceclosure/taskquality upgrade.',
          'WM_Actor_weight_loads':0,'new_AD':0,'Env_physics_nativeFK_collision_GPU':0,'physical_gradient_qualified':False,'wholehiddenstate_qualified':False,'elapsed_seconds':time.perf_counter()-started}
        write(out,'report.json',report);write(out,'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as e:
        write(out,'failure.json',{'schema':'full151105_same_solver_phase_penalty_qualified_GPU_CPU_failure:1','status':'FAILED_PARTIAL_PRESERVED_NO_RETRY','error':repr(e),'phase':phase,'counts':counts,'completed_scored_rows':completed,'remaining_rows':'NOT_RUN','invariants_available':invariants,'config':p['config_pin'],'new_physics_GPU':0});raise
