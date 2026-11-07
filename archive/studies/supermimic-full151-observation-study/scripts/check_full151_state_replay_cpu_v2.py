"""Pure CPU contract fixtures only. No native Reader/GPU/NPZ/model execution."""
import argparse,ast,copy,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import state_replay105_v2 as s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output).resolve()
    s.need(not out.exists(),'Fresh CPU fixture output only');out.mkdir(parents=True);(out/'executed_sources').mkdir()
    pins={k:s.pin(v) for k,v in s.OWN.items()}
    for k,v in pins.items():(out/'executed_sources'/(k+'_'+Path(v['path']).name)).write_bytes(Path(v['path']).read_bytes())
    s.write(out/'source_before.json',pins);checks={}
    def check(n,v):
        checks[n]=bool(v)
        if not checks[n]:raise AssertionError(n)
    def reject(n,fn):
        try:fn()
        except (ValueError,KeyError):check(n,True)
        else:check(n,False)
    try:
        for v in pins.values():ast.parse(Path(v['path']).read_text(encoding='utf-8-sig'))
        check('four_execution_sources_frozen_parse',True)
        check('metadata_import_noTorch_Warp_MuJoCo','torch' not in sys.modules and 'warp' not in sys.modules and 'mujoco' not in sys.modules)
        proto=s.read(s.PROTOCOL);check('fddc_source_prereg_and_counts_exact',s.pin(s.PROTOCOL)['sha256']==s.PROTOCOL_SHA and proto['expected_counts']==s.EXPECTED)
        c={'schema':s.SCHEMA,'role':s.ROLE,'protocol':s.pin(s.PROTOCOL),'output':str(ROOT/'outputs/fixture_state_replay_NEVER_EXECUTED'),
           'source_pins':{'fixture':{}},'CPU_ready':{},'CPU_contract_report':{},'GPU_run_authorized':False,'training_authorized':False,
           'Env_or_physics_authorized':False,'automatic_retry':False,'compile_fallback':False,'soft_worker_seconds':90,'hard_process_seconds':120,
           'evaluation_seed':0,'device':'cuda:0','allow_tf32':True}
        s.validate_header(c);check('exact17key_header_positive',len(c)==17)
        # Actual front module import is metadata only, not main/preflight/run.
        sp=importlib.util.spec_from_file_location('_CPU_fixture_replay_entry',s.OWN['entry']);entry=importlib.util.module_from_spec(sp);sp.loader.exec_module(entry)
        entry.validate_front_metadata(c);check('actual_entry_namespace_matches_core',entry.KEYS==s.KEYS and entry.SCHEMA==s.SCHEMA and entry.ROLE==s.ROLE)
        for k,v in [('evaluation_seed',True),('device','cpu'),('allow_tf32',False),('soft_worker_seconds',91),('hard_process_seconds',121),('automatic_retry',True),('compile_fallback',True),('Env_or_physics_authorized',True),('GPU_run_authorized',1)]:
            reject('core_reject_'+k,lambda k=k,v=v:s.validate_header({**c,k:v}));reject('entry_reject_'+k,lambda k=k,v=v:entry.validate_front_metadata({**c,k:v}))
        reject('reject_added_mode_search',lambda:s.validate_header({**c,'fallback_device':'cpu'}))
        reject('reject_escaped_output',lambda:s.validate_header({**c,'output':str(ROOT/'src/a')}))
        # Old q/qd bits and output-scaled arithmetic remain unchanged.
        a=np.zeros((2,58),np.float32);b=a.copy();a[0,55]=np.float32(1e-4)
        diag=s.old_flags(a,b,True);check('original8eps_failure_preserved',not diag['old_flags'][0,55] and diag['old_bound_F64'][0,55]==8*2**-23)
        az=b.copy();az[0,0]=np.float32(-0.)
        check('q_negativezero_bit_difference_not_hidden',not s.old_flags(az,b,True)['old_flags'][0,0])
        check('all452_160_shapes_are_not_truncated',s.old_flags(np.zeros((452,58),np.float32),np.zeros((452,58),np.float32),True)['old_flags'].shape==(452,58))
        huge=s.old_flags(np.full((1,3),3e38,np.float32),np.full((1,3),-3e38,np.float32));check('difference_promoted_before_subtract',np.isfinite(huge['difference_F64']).all())
        reject('reject_double_precision_replay_output',lambda:s.old_flags(a.astype(float),b,True))
        reject('reject_nonfinite_outputs',lambda:s.old_flags(np.full((1,58),np.nan,np.float32),np.zeros((1,58),np.float32),True))
        raw={'hand_qpos':np.arange(7,27,dtype=np.int64),'hand_dof':np.arange(6,26,dtype=np.int64),'object_qpos_start':np.asarray(0,np.int64),'object_dof_start':np.asarray(0,np.int64)}
        m=s.mapping_from_raw(raw);check('mapping_reused_noMJB_namespaces',m['object_qpos_start']==0 and m['hand_qpos'][0]==7)
        reject('mapping_wrong_namespace_overlap',lambda:s.mapping_from_raw({**raw,'hand_qpos':np.arange(20)}))
        reject('mapping_duplicate_index',lambda:s.mapping_from_raw({**raw,'hand_dof':np.full(20,6)}))
        calls=[];receives=[];clock={'calls':0}
        def op(i):calls.append(i);return np.array([[i]],np.float32)
        def timer():clock['calls']+=1
        s.row_schedule(2260,op,timer,lambda i,v:receives.append((i,v.shape)))
        check('fixed2260_one_operation_perrow_no_warmup',calls==list(range(2260)) and len(receives)==2260 and clock['calls']==2260)
        check('fixture_operation_sees_B1',all(shape==(1,1) for i,shape in receives))
        attempted=[]
        def broken(i):
            attempted.append(i)
            if i==2:raise RuntimeError('synthetic error no retry')
            return i
        try:s.row_schedule(2260,broken,lambda:None,lambda i,v:None)
        except RuntimeError:pass
        check('operation_error_stops_no_retry_or_fallback',attempted==[0,1,2])
        finite_partial=np.full((2260,58),np.nan,np.float32);mask=np.zeros(2260,bool);finite_partial[0]=0;mask[0]=True
        check('partial_unknown_rows_are_NaN_with_falsemask',np.isnan(finite_partial[~mask]).all() and mask.sum()==1)
        good={'schema':'full151105_native_GPU_state_replay_report:1','status':'COMPLETED_NATIVE_GPU_STATE_REPLAY_QUALIFIED','counts':s.EXPECTED.copy(),
           'all_original_gates_passed':True,'future_scoring_inputs_qualified':True,'state_replay_qualified':True,'dense_scoring_eligibility':False,
           'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'qualified_prestate':{'path':'synthetic','sha256':'synthetic'}}
        check('validate_complete_qualified_input_not_scoring',s.validate_result(good)=={'execution_complete':True,'all_original_gates_passed':True,'future_scoring_inputs_qualified':True,'exit_code':0})
        fail={**good,'status':'COMPLETED_RAW_NOT_QUALIFIED','all_original_gates_passed':False,'future_scoring_inputs_qualified':False,'state_replay_qualified':False,'qualified_prestate':None}
        check('finite_arithmetic_fail_complete_exit1_not_exception',s.validate_result(fail)['exit_code']==1)
        reject('fail_cannot_export_qualified_packet',lambda:s.validate_result({**fail,'qualified_prestate':{}}))
        reject('pass_cannot_authorize_scoring',lambda:s.validate_result({**good,'dense_scoring_eligibility':True}))
        reject('wrong_B2260_result_rejected',lambda:s.validate_result({**good,'counts':{**s.EXPECTED,'native_state_batch_size':2260}}))
        reject('boolean_count_rejected',lambda:s.validate_result({**good,'counts':{**s.EXPECTED,'new_MJB_loads':False}}))
        reject('missing_full58_anchor_count_rejected',lambda:s.validate_result({**good,'counts':{**s.EXPECTED,'anchor_comparisons':0}}))
        check('still_noTorch_before_tensor_only_fixtures','torch' not in sys.modules)
        import torch
        torch.set_num_threads(1)
        q=torch.zeros((1,27),dtype=torch.float32);v=torch.zeros((1,26),dtype=torch.float32);pp=torch.zeros((1,3),dtype=torch.float32);R=torch.eye(3)[None]
        idx={'hand_qpos':torch.arange(7,27),'hand_dof':torch.arange(6,26),'object_qpos':torch.arange(7),'object_dof':torch.arange(6)}
        s.tensor_views_B1(q,v,pp,R,idx,'cpu');check('structural_positive_B1_CPU_fixture',True)
        reject('production_GPU_ABI_rejects_CPU_before_native_call',lambda:s.native_cuda_state_B1(q,v,pp,R,idx,{}))
        reject('reject_B2_native_views',lambda:s.tensor_views_B1(q.expand(2,-1),v,pp,R,idx,'cpu'))
        reject('reject_FP64_native_views',lambda:s.tensor_views_B1(q.double(),v,pp,R,idx,'cpu'))
        reject('reject_float_index_dtype',lambda:s.tensor_views_B1(q,v,pp,R,{**idx,'object_qpos':torch.arange(7,dtype=torch.float32)},'cpu'))
        check('CPU_fixture_did_not_initialize_CUDA',not torch.cuda.is_initialized())
        metadata=s.flags_runtime(torch)
        expected_splitK=getattr(torch.backends.cuda.matmul,'allow_bf16_reduced_precision_reduction_split_k')
        check('correct_installed_BF16_splitK_getter_observed',metadata['bf16_splitK']!='UNAVAILABLE' and metadata['bf16_splitK']==expected_splitK)
        check('optional_backend_metadata_read_did_not_initialize_CUDA',not torch.cuda.is_initialized())
        check('native_reader_not_imported_by_fixture','dex_supermimic.rl.dex_supermimic.sim' not in sys.modules)
        text=Path(s.__file__).read_text(encoding='utf-8');tree=ast.parse(text)
        native=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_cuda_state_B1')
        check('exact_unbound_SimReader_state_call_once',sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='SimReader' and n.func.attr=='state' for n in ast.walk(native))==1)
        check('no_CPUwrapper_or_model_toCUDA','native_state58(' not in text and '.to(device' not in text and 'from_binary_path' not in text and 'torch.load' not in text)
        check('no_factory_WM_Actor_or_physics_path',all(name not in text for name in ('build_penalty_only_objective','load_world_model_checkpoint','mj_step','mj_forward','mj_kinematics','ManagerBasedRlEnv(')))
        check('qualified_packet_only_guarded_by_allnumeric',"if all_numeric:\n            atomic_npz(out/'prestate1808_qualified_for_future_scoring.npz'" in text)
        check('whole_worker_started_perf_and_bounded_partial',"started_perf=None" in text and "completed%128==0" in text and 'replace_with_bounded_retry' in text)
        check('front_complete_nonqual_exit_after_completion_outside_catch','raise SystemExit(result_validation[\'exit_code\'])' in Path(s.OWN['entry']).read_text())
        check('no_new_data_model_or_native_module_imports',all(name not in sys.modules for name in ('mujoco','warp','full151_intent.binding_v3','full151_native.world_model_checkpoint')))
        previous=s.read(ROOT/'outputs/full151105_state_replay_preparation_CPU_v1/READY.json')
        check('v1_all_four_executed_sources_preserved',all(s.pin(v['path'])==v for v in previous['source_pins'].values()))
        for k,v in pins.items():check('source_unchanged_'+k,s.pin(v['path'])==v)
        proof={'schema':'full151105_state_replay_CPU_contract:1','status':'PASS_PURE_CPU_CONTRACT_NO_GPU','checks':checks,'passed':sum(checks.values()),'total':len(checks),
           'source_pins':pins,'protocol':s.pin(s.PROTOCOL),'actual_GPU_calls':0,'native_state_calls':0,'NPZ_input_reads':0,'MJB_loads':0,'model_loads':0,
           'scope':'Synthetic row scheduler, old-rule algebra and CPU Tensor shape/device fixtures only. Production native CUDA helper rejected CPU before any native state call. Not actual state replay or qualification.'}
        s.write(out/'report.json',proof)
        ready={'schema':'full151105_state_replay_CPU_READY:1','source_pins':pins,'cpu_report':s.pin(out/'report.json'),'cpu_checks':{'passed':proof['passed'],'total':proof['total']},
           'protocol':s.pin(s.PROTOCOL),'api':{'module':str(s.OWN['module']),'entry':str(s.OWN['entry']),'metadata':'validate_config(dict,bundle=None)','authorized':'run_replay(config_path,output,emit,started_perf=STARTED)','result':'validate_result(report)'},
           'GPU_run_authorized':False,'native_replay_qualified':False,'proxy_scoring_authorized':False}
        s.write(out/'READY.json',ready);print(json.dumps({'passed':proof['passed'],'total':proof['total'],'report':s.pin(out/'report.json'),'READY':s.pin(out/'READY.json'),'source_pins':pins},indent=2))
    except BaseException as error:
        s.write(out/'failure.json',{'error':repr(error),'checks_completed':checks,'source_pins':pins,'status':'FAILED_PURE_CPU_FIXTURE_PRESERVED','actual_GPU_or_native_state':0});raise


if __name__=='__main__':main()
