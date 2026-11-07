"""NEW qualified-packet/alignment/identity contracts, synthetic CPU data only.

No actual physical/GPU packet NPZ, MJB, objective, native state, or model is read.
The real default CLI subprocess is metadata-only and uses the frozen prereg.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os,pickle,random,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import penalty_observability105_v2 as a


def synthetic_packet():
    n=3
    q=np.arange(n*27,dtype=np.float32).reshape(n,27)/128
    v=np.arange(n*26,dtype=np.float32).reshape(n,26)/64
    pp=np.broadcast_to(np.array([1.,2.,3.],np.float32),(n,3)).copy()
    R=np.broadcast_to(np.array([[0,-1,0],[1,0,0],[0,0,1]],np.float32),(n,3,3)).copy()
    local_p=np.broadcast_to(np.array([.25,-.5,1.],np.float32),(n,3)).copy()
    local_R=np.broadcast_to(np.eye(3,dtype=np.float32),(n,3,3)).copy()
    world_p=pp+(R*local_p[:,None,:]).sum(-1)
    world_R=R.copy()
    hq=np.arange(7,27,dtype=np.int32);hv=np.arange(6,26,dtype=np.int32)
    x=np.zeros((n,58),np.float32);x[:,:20]=q[:,hq];x[:,20:40]=v[:,hv];x[:,40:43]=local_p;x[:,43:52]=local_R.reshape(n,9)
    mapping={'hand_qpos':hq,'hand_dof':hv,'object_qpos_start':np.array(0,np.int32),'object_dof_start':np.array(0,np.int32)}
    packet={'native_GPU_prestate58':x,'completed_mask':np.ones(n,np.bool_),'source_pre_row':np.arange(n,dtype=np.int64),'palm_pos':pp,'palm_rot':R,**mapping}
    raw={'pre_qpos':q,'pre_qvel':v,'palm_pos_all2260':pp.copy(),'palm_rot_all2260':R.copy(),**{k:val.copy() for k,val in mapping.items()}}
    return packet,raw,world_p,world_R


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output).resolve();a.require(not out.exists(),'Fresh narrow CPU proof directory required');out.mkdir(parents=True);(out/'executed_sources').mkdir()
    pins={k:a.pin(v) for k,v in a.OWN.items()}
    deps={'legacy_dense':a.pin(a.legacy.__file__),'shared_hook':a.pin(a.common.__file__),'qualified_replay_contract':a.pin(a.replay.__file__)}
    for k,v in {**pins,**deps}.items():(out/'executed_sources'/(k+'_'+Path(v['path']).name)).write_bytes(Path(v['path']).read_bytes())
    a.write(out,'MODULE_FREEZE.json',{'source_pins':pins,'dependency_pins':deps,'protocol':a.pin(a.PREREG),'scope':'Frozen before synthetic tests; no actual packet/conversion/model/objective/geometry data.'})
    checks={};npz_attempts={'value':0};original_np_load=np.load
    def check(name,value):
        a.require(name not in checks,'Duplicate test name');checks[name]=bool(value)
        if not checks[name]:raise AssertionError(name)
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):check(name,True)
        else:check(name,False)
    def deny_npz(*args,**kwargs):npz_attempts['value']+=1;raise AssertionError('No actual or synthetic NPZ loads in this pure fixture')
    np.load=deny_npz
    try:
        for value in pins.values():ast.parse(Path(value['path']).read_text(encoding='utf-8-sig'))
        check('formed_sources_AST_valid',True)
        check('fresh_metadata_has_noTorch_Warp_MuJoCo',all(k not in sys.modules for k in ('torch','warp','mujoco')))
        check('frozen_protocol_c9a746_revision4_matches',a.pin(a.PREREG)['sha256']==a.PREREG_SHA)
        d=a.declaration();check('declaration_does_not_open_packet_or_construct',d['NPZ_opened']==d['MJB_loads']==d['state_conversions']==d['objective_constructs']==0)
        h={'schema':a.SCHEMA,'role':a.ROLE,'protocol':{},'output':str(ROOT/'outputs/full151105_solver_phase_penalty_GPU_qualified_CPU_fixture_NEVER_RUN'),
            'CPU_run_authorized':False,'cpu_threads':1,'source_pins':{'fixture':{}},'cpu_ready':{}}
        a.validate_header(h);check('new_eight_key_header_positive',True)
        for k,v in [('schema',a.legacy.SCHEMA),('role',a.legacy.ROLE),('CPU_run_authorized',1),('cpu_threads',True),('output',str(out))]:
            rejects('header_reject_'+k,lambda k=k,v=v:a.validate_header({**h,k:v}))
        rejects('header_reject_unknown_GPU_flag',lambda:a.validate_header({**h,'GPU_run_authorized':True}))
        packet,raw,sp,sR=synthetic_packet();mapping=a.packet_abi(packet,raw,expected_rows=3)
        check('typed_mapping_int32_value_accepted_without_CUDA_dtype_confusion',mapping['hand_qpos'].dtype==np.int32 and mapping['object_qpos_start']==0)
        changed=copy.deepcopy(packet);changed['hand_qpos']=changed['hand_qpos'].astype(np.int64)
        a.packet_abi(changed,raw,expected_rows=3);check('integer_storage_width_change_is_not_value_change',True)
        for name,change in [
          ('unknown_field',lambda p:p.update(extra=np.zeros(1))),
          ('partial_mask',lambda p:p['completed_mask'].__setitem__(1,False)),
          ('integer_completion_mask',lambda p:p.update(completed_mask=np.ones(3,np.int32))),
          ('reordered_rows',lambda p:p.update(source_pre_row=np.array([1,0,2]))),
          ('int32_source_rows',lambda p:p.update(source_pre_row=np.arange(3,dtype=np.int32))),
          ('F64_state',lambda p:p.update(native_GPU_prestate58=p['native_GPU_prestate58'].astype(np.float64))),
          ('NaN_state',lambda p:p['native_GPU_prestate58'].__setitem__((0,55),np.nan)),
          ('q_changed',lambda p:p['native_GPU_prestate58'].__setitem__((0,0),99)),
          ('qd_changed',lambda p:p['native_GPU_prestate58'].__setitem__((0,20),99)),
          ('palm_position_changed',lambda p:p['palm_pos'].__setitem__((0,0),99)),
          ('palm_rotation_changed',lambda p:p['palm_rot'].__setitem__((0,0,0),99)),
          ('duplicate_mapping',lambda p:p.update(hand_qpos=np.full(20,7,np.int32))),
          ('floating_mapping',lambda p:p.update(hand_dof=p['hand_dof'].astype(float)))]:
            bad=copy.deepcopy(packet);change(bad)
            rejects('packet_reject_'+name,lambda bad=bad:a.packet_abi(bad,raw,expected_rows=3))
        alignment=a.pose_alignment(packet['native_GPU_prestate58'],packet['palm_pos'],packet['palm_rot'],sp,sR)
        check('rotated_frame_closedform_p_R_reference_exact',np.all(alignment['difference_F64']==0) and alignment['original_8eps_flags'].all())
        check('all12_derived_components_F64_and_full_masks_saved',alignment['derived_reference_F64'].shape==(3,12) and alignment['derived_reference_F64'].dtype==np.float64 and alignment['original_8eps_flags'].shape==(3,12))
        flat=sR.reshape(3,9).copy();record=a.replay.old_flags(flat,flat.copy())
        actual_flat=a.original_solver_rule(record,3,9);logical=actual_flat.reshape(3,3,3,order='C')
        check('actual_saved_flat9_R_ABI_reshapes_without_byte_change',logical.tobytes()==flat.tobytes() and logical.shape==(3,3,3))
        check('flat9_R_route_retains_same_alignment',a.pose_alignment(packet['native_GPU_prestate58'],packet['palm_pos'],packet['palm_rot'],sp,logical)['original_8eps_flags'].all())
        badrecord=copy.deepcopy(record);badrecord['old_flags'][0,0]=False
        rejects('old_rule_flag_record_must_match_original_pairs',lambda:a.original_solver_rule(badrecord,3,9))
        badrecord=copy.deepcopy(record);badrecord['difference_F64'][0,0]=1
        rejects('old_rule_delta_cannot_be_trusted_without_recompute',lambda:a.original_solver_rule(badrecord,3,9))
        badrecord=copy.deepcopy(record);badrecord['old_bound_F64'][0,0]*=2
        rejects('old_rule_bound_cannot_be_relaxed',lambda:a.original_solver_rule(badrecord,3,9))
        badx=packet['native_GPU_prestate58'].copy();badx[0,40]+=.0001
        check('wrong_position_frame_fails_original_bound',not a.pose_alignment(badx,packet['palm_pos'],packet['palm_rot'],sp,sR)['original_8eps_flags'][0,:3].all())
        badx=packet['native_GPU_prestate58'].copy();badx[0,43:52]=packet['palm_rot'][0].reshape(9)
        check('world_rotation_is_not_reader_rotation',not a.pose_alignment(badx,packet['palm_pos'],packet['palm_rot'],sp,sR)['original_8eps_flags'][0,3:].all())
        zero=np.zeros((1,58),np.float32);zero[:,43:52]=np.eye(3,dtype=np.float32).reshape(1,9)
        pp=np.zeros((1,3),np.float32);RR=np.eye(3,dtype=np.float32)[None]
        boundary=zero.copy();boundary[0,40]=np.float32(8*2**-23)
        check('unchanged_8eps_exact_boundary_passes',a.pose_alignment(boundary,pp,RR,pp,RR)['original_8eps_flags'].all())
        boundary[0,40]=np.nextafter(boundary[0,40],np.float32(np.inf))
        check('next_FP32_above_8eps_boundary_fails',not a.pose_alignment(boundary,pp,RR,pp,RR)['original_8eps_flags'][0,0])
        rejects('alignment_reject_F64_operand_not_silent_cast',lambda:a.pose_alignment(zero.astype(float),pp,RR,pp,RR))
        rejects('alignment_reject_bad_world_R_shape',lambda:a.pose_alignment(zero,pp,RR,pp,RR.reshape(1,9)))
        nan=zero.copy();nan[0,42]=np.nan
        rejects('alignment_reject_nonfinite',lambda:a.pose_alignment(nan,pp,RR,pp,RR))
        # Synthetic files, not NPZ decoding: exactly one stale runtime entry.
        fake=out/'synthetic_manifest';fake.mkdir()
        for name in ('runtime.json','report.json','prestate1808_qualified_for_future_scoring.npz'):(fake/name).write_bytes(b'fixture_before')
        core={name:a.pin(fake/name) for name in ('runtime.json','report.json','prestate1808_qualified_for_future_scoring.npz')}
        oldruntime=copy.deepcopy(core['runtime.json']);(fake/'runtime.json').write_bytes(b'fixture_terminal')
        exception={'runtime.json':{'core_preterminal_pin':oldruntime,'terminal_pin':a.pin(fake/'runtime.json'),'hash_equal':False}}
        terminal,number=a.verify_core_manifest(core,exception,fake,2)
        check('only_stale_runtime_allowed_with_actual_terminal_pin',number==2 and terminal.samefile(fake/'runtime.json'))
        bad=copy.deepcopy(exception);bad['report.json']=bad['runtime.json']
        rejects('mutable_exception_cannot_expand_to_report',lambda:a.verify_core_manifest(core,bad,fake,2))
        bad=copy.deepcopy(exception);bad['runtime.json']['hash_equal']=True
        rejects('mutable_mismatch_cannot_be_concealed',lambda:a.verify_core_manifest(core,bad,fake,2))
        bad=copy.deepcopy(core);bad['report.json']['sha256']='0'*64
        rejects('alltrue_review_does_not_bypass_immutable_SHA',lambda:a.verify_core_manifest(bad,exception,fake,2))
        bad=copy.deepcopy(exception);bad['runtime.json']['terminal_pin']=a.pin(fake/'report.json')
        rejects('runtime_exception_cannot_change_file_identity',lambda:a.verify_core_manifest(core,bad,fake,2))
        rejects('boolean_immutable_count_rejected',lambda:a.verify_core_manifest(core,exception,fake,True))
        proof_r={'schema':'full151105_native_GPU_state_replay_report:1','status':'COMPLETED_NATIVE_GPU_STATE_REPLAY_QUALIFIED','counts':a.replay.EXPECTED.copy(),
            'all_original_gates_passed':True,'future_scoring_inputs_qualified':True,'state_replay_qualified':True,'dense_scoring_eligibility':False,
            'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'qualified_prestate':{'path':'synthetic_pointer_not_read','sha256':'0'*64}}
        check('full58_replay_validation_retained',a.replay.validate_result(proof_r)['exit_code']==0)
        bad=copy.deepcopy(proof_r);bad['counts']['converted_rows']=1808
        rejects('qualified1808_does_not_replace_original2260_count',lambda:a.replay.validate_result(bad))
        bad=copy.deepcopy(proof_r);bad['whole_hidden_state_qualified']=True
        rejects('hidden_state_certificate_cannot_be_upgraded',lambda:a.replay.validate_result(bad))
        # Actual fresh-process default: no source-science/data loading.
        proc=subprocess.Popen([sys.executable,'-B',str(a.OWN['CLI'])],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
        try:stdout,stderr=proc.communicate(timeout=20)
        except BaseException:
            proc.kill();proc.communicate(timeout=2);raise
        (out/'default_stdout.json').write_text(stdout,encoding='utf-8');(out/'default_stderr.txt').write_text(stderr,encoding='utf-8')
        a.require(proc.returncode==0,stderr);decl=json.loads(stdout)
        check('actual_default_subprocess_noTorch_noNPZ_no_conversion',decl['Torch_imported'] is False and decl['Warp_imported'] is False and decl['NPZ_opened']==decl['state_conversions']==decl['objective_constructs']==0)
        # Closed-form Torch fixture exercises the existing nonduplicating hook.
        import torch
        torch.set_num_threads(1)
        def rng():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)+torch.get_rng_state().numpy().tobytes()).hexdigest()
        before=rng()
        class ClosedProxy(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,x):
                self.calls+=1;b=len(x);dist=(-.003-.001*x[:,0:1]).expand(b,3)
                return {'support':x[:,1:2].expand(b,64),'presence':torch.ones((b,6)),'minimum_signed_distance_m':dist.amin(-1),
                    'sampled_signed_distance_m':dist,'soft_contact_weight':torch.ones((b,3)),
                    'projected_contact_com_o_m':torch.zeros((b,3,3)),'inward_normal_o':torch.ones((b,3,3))}
        class ClosedPenalty(torch.nn.Module):
            def __init__(self):super().__init__();self.geometry_proxy=ClosedProxy()
            def forward(self,x,m):
                d=self.geometry_proxy(x);P=((-d['sampled_signed_distance_m']-.003).clamp_min(0).amax(-1)/.001).square();S=d['support'].mean(-1)
                return -.1*P,{'geometry_intent_penetration_prior':P,'geometry_intent_support_match':S,'geometry_intent_phase_weight':torch.ones_like(P)}
        module=ClosedPenalty()
        with torch.no_grad():score,terms,capture=a.common.evaluate_objective(module,torch.ones((2,58),dtype=torch.float32),np.array([0,149]))
        check('existing_proxy_capture_has_one_call_and_removes_hook',module.geometry_proxy.calls==1 and len(module.geometry_proxy._forward_hooks)==0 and capture['sampled_signed_distance_m'].shape==(2,3))
        check('P_ONLY_direct_formula_not_support_subtraction',np.allclose(terms['geometry_intent_penetration_prior'].numpy(),1,atol=4e-7) and np.allclose(score.numpy(),-.1,atol=1e-7))
        rejects('terminal150_cannot_receive_scoring_action_memory',lambda:a.common.memory_for_phases(np.array([150])))
        check('fixture_RNG_unchanged_no_CUDA_initialization',rng()==before and not torch.cuda.is_initialized())
        check('no_binding_MuJoCo_Warp_WM_Actor_import',all(k not in sys.modules for k in ('full151_intent.binding_v3','mujoco','warp','full151_native.world_model_checkpoint','full151_native.checkpoint')))
        check('actual_npz_inputs_never_opened',npz_attempts['value']==0)
        # Read/AST assertions establish the reachable-source delta, not native qualification.
        source=Path(a.__file__).read_text(encoding='utf-8');tree=ast.parse(source)
        check('scientific_definitions_are_exact_original_objects',a.pair_coverage is a.legacy.pair_coverage and a.group_rows is a.legacy.group_rows and a.native_contact_abi is a.legacy.native_contact_abi)
        check('source_has_no_native_state_or_quaternion_call',all(name not in source for name in ('SimReader.state','native_state58(','quat_to_mat(','native_cuda_state_B1(')))
        check('new_alignment_saved_before_gate_before_factory',source.index("npwrite(out,'GPU_prestate_solver_pose_alignment_all1808.npz'")<source.index("require(gates['NEW_all1808_GPU_p_R_derived_alignment_passed']")<source.index("call('objective_factory'"))
        check('complete_native_catalog_uses_original_regex_enabled',"re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)'" in source and 'model.geom_contype[i] or model.geom_conaffinity[i]' in source)
        check('source_preserves_one_P_ONLY_factory_and_fifteen_batches',source.count("call('objective_factory'")==1 and "range(0,1808,128)" in source and "counts['objective_forward']['returned']==15" in source)
        check('native_coverage_never_derived_from_sample_geom_ids',source.index('hand={i for i in range(model.ngeom)')<source.index('sample_geom=base.geometry_proxy.sample_geom_ids'))
        for k,v in {**pins,**deps}.items():check('source_unchanged_'+k,a.pin(v['path'])==v)
        proof={'schema':'full151105_qualified_dense_penalty_CPU_contract_proof:1','status':'PASS_SYNTHETIC_QUALIFIED_PACKET_ALIGNMENT_AND_HOOK_ONLY',
            'checks':checks,'passed':sum(checks.values()),'total':len(checks),'source_pins':pins,'dependency_pins':deps,'protocol':a.pin(a.PREREG),
            'scope':{'actual_physical_state_conversions':0,'actual_MJB_loads':0,'actual_objective_constructs':0,'actual_full05_data_reads':0,'actual_GPU_packet_array_reads':0,
                'synthetic_packet_rows':3,'synthetic_alignment_rows':3,'closed_form_hook_rows':2,'actual_WM_Actor_weights_Env_physics_GPU':0,'actual_dense_qualification':False}}
        a.write(out,'report.json',proof)
        ready={'schema':'full151105_qualified_dense_penalty_CPU_contract_READY:1','status':proof['status'],'source_pins':pins,'dependency_pins':deps,
            'cpu_report':a.pin(out/'report.json'),'cpu_checks':{'passed':proof['passed'],'total':proof['total']},'protocol':a.pin(a.PREREG),
            'CPU_scientific_run_authorized':False,'GPU_run_authorized':False,'api':{'entry':str(a.OWN['CLI']),'module':str(a.OWN['consumer']),
                'default':'no args declaration; --config JSON hash-only','actual':'Root separate authorization; CUDA_VISIBLE_DEVICES=-1 --config JSON --run',
                'schema':a.SCHEMA,'role':a.ROLE,'keys':sorted(a.KEYS),'required_sources':'required_sources(protocol_dict)','header':'validate_header(config_dict)'}}
        a.write(out,'READY.json',ready)
        print(json.dumps({'passed':proof['passed'],'total':proof['total'],'report':a.pin(out/'report.json'),'READY':a.pin(out/'READY.json'),'source_pins':pins},indent=2))
    except BaseException as error:
        a.write(out,'failure.json',{'status':'FAILED_CPU_FIXTURE_PRESERVED','error':repr(error),'checks_completed':checks,'source_pins':pins,'actual_scientific_work':False});raise
    finally:np.load=original_np_load


if __name__=='__main__':main()
