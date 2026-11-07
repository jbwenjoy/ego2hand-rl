"""Synthetic saved-evidence contracts only. No real105 checkpoint/run audit."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,sys,time,traceback
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/audit_full151_checkpoint_cpu_v1.py'


def main(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();checks={};notes={}
    sp=importlib.util.spec_from_file_location('_full151105_health_CPU_tested',SOURCE);a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
    sources={k:a.pin(p) for k,p in {'auditor':SOURCE,'checker':Path(__file__),'learning':ROOT/'src/full151_native/learning.py','training_capacity':ROOT/'src/full151_native/training_capacity.py','controller':ROOT/'src/full151_native/controller.py','runtime':ROOT/'src/full151_native/runtime.py','native':ROOT/'src/full151_native/native.py'}.items()}
    def check(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise AssertionError(name)
    def reject(name,call):
        try:value=call()
        except (ValueError,KeyError,TypeError,IndexError):checks[name]=True;return
        if isinstance(value,tuple):value=value[0]
        check(name,(value is False) or (isinstance(value,dict) and not all(value.values())))
    try:
        check('CPU_mask_no_CUDA_or_native_imports',not torch.cuda.is_initialized() and not torch.cuda.is_available() and not any(n.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for n in sys.modules))
        rng=torch.get_rng_state().clone()
        common={'num_envs':128,'warmup_steps':64,'wm_window':8,'utd':.4,'wm_batches':2,'critic_epochs':2,'critic_minibatches':2,'horizon':8,'pi_batch':128}
        smoke=common|{'max_iterations':40,'wm_only_iters':20};primary=common|{'max_iterations':2000,'wm_only_iters':200}
        for name,cfg,I,expected in [('smoke40',smoke,40,(20992,80,80,20,80)),('primary2000',primary,2000,(648192,4000,4000,1800,7200)),('primary100',primary,100,(40192,200,200,0,0))]:
            counts,total=a.budget_counts(cfg,I);check('native_budget_'+name,(total,counts['wm'],counts['fail_head'],counts['actor'],counts['critic'])==expected)
        reject('reject_bool_iteration',lambda:a.budget_counts(primary,True));reject('reject_outside_budget',lambda:a.budget_counts(primary,2001))
        expected={'wm':200,'fail_head':200,'actor':0,'critic':0}
        check('missing_expected_zero_optimizer_keys_allowed',a.optimizer_counts_match({'wm':200,'fail_head':200},expected))
        for name,observed in [('missing_positive',{'wm':200}),('unknown',{'wm':200,'fail_head':200,'other':0}),('bool',{'wm':True,'fail_head':200}),('negative',{'wm':-1,'fail_head':200}),('unexpected_actor',{'wm':200,'fail_head':200,'actor':1})]:check('reject_optimizer_'+name,not a.optimizer_counts_match(observed,expected))
        cfg={'schema':a.CONFIG_SCHEMA,'role':'engineering_smoke','method':'A1L','contact_binding':None}
        hdr={'format_version':105,**{k:{'SYNTHETIC':True} for k in ('full151_controller','full151_runtime_evidence','full151_capacity_evidence','full151_initial_parameter_fingerprints')}}
        check('positive_new105_header_fixture',a.new105_header_ok(hdr,cfg))
        for n in (3,103,104,True):check('reject_old_or_bool_format_'+str(n),not a.new105_header_ok(hdr|{'format_version':n},cfg))
        check('reject_integer_relabel_old_namespace',not a.new105_header_ok({'format_version':105,'rough_controller':{}},cfg))
        check('reject_old_namespace_contamination',not a.new105_header_ok(hdr|{'rough_controller':{}},cfg))
        for role in ('native_a0','native_policy_eval'):check('reject_nontraining_role_'+role,not a.new105_header_ok(hdr,cfg|{'role':role}))
        for method in ('A0','P_ONLY','A2'):check('reject_unimplemented_arm_'+method,not a.new105_header_ok(hdr,cfg|{'method':method}))
        check('reject_legacy_prior_attachment',not a.new105_header_ok(hdr,cfg|{'contact_binding':{'path':'old16'}}))
        check('JSON_tuple_list_wire_equivalent',a.effective_config_equal({'wm_hidden':(512,512,512)},{'wm_hidden':[512,512,512]}))
        check('JSON_boolean_integer_not_equivalent',not a.effective_config_equal({'amp':True},{'amp':1}))
        check('JSON_integer_float_not_equivalent',not a.effective_config_equal({'num_envs':128},{'num_envs':128.0}))
        # A max-index filename is still a prefix; only model_final with actual
        # final markers can set the public final_iteration_scope bit.
        final=out/'fake_run/train/model_final.pt';indexed=out/'fake_run/train/model_2000.pt'
        finalchecks={k:True for k in ('actual_final_completion_not_task_quality','worker_training_completion_marker_schema_config','training_wrapper_return_final_capacity','worker_terminal_completed','capacity_final_scope_run_final_once','capacity_final_raw_OK_no_overflow_or_unknown')}
        check('normal_final_scope_requires_completed_markers',a.qualified_final_scope(final,out/'fake_run',2000,2000,finalchecks))
        check('indexed2000_is_not_final_scope',not a.qualified_final_scope(indexed,out/'fake_run',2000,2000,finalchecks))
        check('checkpoint100_is_prefix_not_final',not a.qualified_final_scope(out/'fake_run/train/model_100.pt',out/'fake_run',100,2000,finalchecks))
        for key in finalchecks:check('reject_missing_final_marker_'+key,not a.qualified_final_scope(final,out/'fake_run',2000,2000,{k:v for k,v in finalchecks.items() if k!=key}))
        # Synthetic H8 near-terminal windows, preserving inactive and bootstrap
        # eligibility arithmetic without constructing a policy/critic.
        starts=np.resize(np.array([0,142,143,148,149],float),128);mask=starts[None]+np.arange(8)[:,None]<150;last=starts+8<150
        flat=mask.reshape(-1);mini=[int(flat[:512].sum()),int(flat[512:].sum())]*2
        ev={'policy':{'start_phases':starts[:,None].tolist(),'active_mask_H_B':mask.tolist(),'critic_norm_eligible':int(mask.sum()+last.sum()),'critic_loss_minibatch_active_counts':mini,'optimizer_actor_calls':1,'optimizer_critic_calls':4,'critic_total_candidate_rows':1024,'hard_mask_rule':'phase<150 only; physical soft termination retains native survival semantics'},'latest_window':{'valid_window_entries':9,'candidate_window_entries':16,'terminal_valid_entries':1,'reset_suffix_entries_masked':7}}
        check('H8_terminal_masks_counts_eligibility_positive',all(a.policy_checks(ev,common).values()))
        changed=copy.deepcopy(ev);changed['policy']['active_mask_H_B'][1][4]=True;reject('reject_H8_inactive_future_row',lambda:a.policy_checks(changed,common))
        changed=copy.deepcopy(ev);changed['policy']['critic_norm_eligible']+=1;reject('reject_H8_critic_norm_inactive',lambda:a.policy_checks(changed,common))
        changed=copy.deepcopy(ev);changed['policy']['critic_loss_minibatch_active_counts']=[1000,0,1000,0];reject('reject_impossible_critic_minibatch_counts',lambda:a.policy_checks(changed,common))
        changed=copy.deepcopy(ev);changed['policy']['start_phases'][0][0]=150;reject('reject_H8_real_start_terminal150',lambda:a.policy_checks(changed,common))
        changed=copy.deepcopy(ev);changed['latest_window']['reset_suffix_entries_masked']=0;reject('reject_replay_summary_inconsistent_suffix',lambda:a.policy_checks(changed,common))
        # Sticky capacity saved records only: different flag-array lengths are
        # valid global flags and never treated as a per-world vector.
        def record(seq,event):return {'sequence':seq,'event':event,'raw_flags_dtype':'int32','raw_flags_shape':[4],'raw_flags_values':[0,0,0,0],'read_status':'OK','raw_bitmask':0,'cumulative_OR_bitmask':0,'cumulative_unknown_reads':0,'overall_status':'OK','read_attempts_for_record':1,'backend_field':'env.sim.wp_data.overflow','flag_shape_is_world_count':False,'flags_cleared_by_observer':False}
        records=[record(1,'after_constructor_before_actual8'),record(2,'before_reset'),record(3,'checkpoint')]
        cap={'schema':'rough_prefix_reset_aware_capacity_observation:1','backend_field':'env.sim.wp_data.overflow','configured_fixed_capacities':{'nconmax':192,'njmax':4096},'flag_shape_is_world_count':False,'records':records,'status':'OK','cumulative_OR_bitmask':0,'unknown_read_count':0,'record_count':3,'before_reset_read_count':1,'final_read_count':0,'finalized':False,'engineering_capacity_qualified':False,'evidence_kind':'checkpoint_prefix','prefix_at_checkpoint_capacity_qualified':True,'entire_run_finalized':False}
        check('capacity_prefix_not_final_positive',all(a.capacity_checks(cap,'checkpoint_prefix')[0].values()))
        done=copy.deepcopy(cap);done['records'].append(record(4,'final'));done.update(record_count=4,final_read_count=1,finalized=True,engineering_capacity_qualified=True,entire_run_finalized=True)
        check('capacity_final_actual_four_records_positive',all(a.capacity_checks(done,'finalrun')[0].values()))
        for name,edit in [('overflow',lambda v:v['records'][1].update(raw_flags_values=[2,0,0,0])),('unknown',lambda v:v['records'][1].update(read_status='UNKNOWN',raw_flags_values=None)),('cleared',lambda v:v['records'][1].update(flags_cleared_by_observer=True)),('world_inference',lambda v:v.update(flag_shape_is_world_count=True)),('fake_final',lambda v:v.update(finalized=True))]:
            changed=copy.deepcopy(cap);edit(changed);reject('reject_capacity_'+name,lambda changed=changed:a.capacity_checks(changed,'checkpoint_prefix'))
        # Controller JSONL with three checkpoint-prefix calls plus a later
        # append and incomplete tail. No future rows may change prefix totals.
        run=out/'synthetic_control';run.mkdir();I=2;fcfg=primary|{'wm_only_iters':200};baseline=(np.arange(150)[:,None]*.001+np.arange(20)[None,:]*.002).astype(np.float32)
        lo=np.full(20,-1.,np.float32);hi=np.full(20,1.,np.float32);rows=[]
        for call,steps in enumerate(a.expected_collect_steps(fcfg,I),1):
            phase=(np.arange(steps)[:,None]+np.arange(128)[None,:]+call*37)%150
            nominal=baseline[phase];residual=np.full_like(nominal,.1);residual[0,0,0]=2.
            proposed=nominal+residual;cached=np.clip(np.clip(proposed,lo,hi),lo+np.float32(.02),hi-np.float32(.02))
            hist=np.bincount(phase.reshape(-1),minlength=150).tolist()
            rows.append({'collect_call':call,'iteration':max(0,call-2),'steps':steps,'worlds':128,'real_transitions':steps*128,'all_batch_exact':True,'all_finite':True,'max_abs_errors':{k:0. for k in a.ERROR_KEYS},'phase_min':int(phase.min()),'phase_max':int(phase.max()),'phase_histogram':hist,'clamped_components':int(np.count_nonzero(proposed!=cached)),'inactive_real_actions':0})
            ix=np.s_[-min(4,steps):,:4]
            np.savez_compressed(run/f'full151_actualu_trace_{call:05d}.npz',phase=phase[ix],baseline=nominal[ix],proposed=proposed[ix],residual_before_filter=residual[ix],cached=cached[ix],actual=cached[ix],residual=residual[ix],replay_u=cached[ix],replay_c=residual[ix])
        prefix=''.join(json.dumps(r)+'\n' for r in rows);path=run/'full151_actualu_evidence.jsonl';path.write_text(prefix,encoding='utf-8')
        saved={'config':fcfg,'iteration':I,'sim_steps':128*sum(a.expected_collect_steps(fcfg,I)),'full151_controller':{'latest_actualu_evidence':rows[-1],'target_lower_rad':lo.tolist(),'target_upper_rad':hi.tolist()}}
        ca,da,pb=a.actualu_checks(run,saved,baseline);check('actualu_prefix_traces_projection_replay_positive',all(ca.values()))
        extra=copy.deepcopy(rows[-1]);extra.update(collect_call=4,iteration=2)
        path.write_text(prefix+json.dumps(extra)+'\n'+ '{"collect_call":5',encoding='utf-8')
        cb,db,pb2=a.actualu_checks(run,saved,baseline);check('live_later_append_ignored_at_checkpoint_marker',all(cb.values()) and pb==pb2 and da['prefix_sha256']==db['prefix_sha256'] and db['later_complete_rows_ignored']==1)
        changed=copy.deepcopy(saved);changed['full151_controller']['latest_actualu_evidence']=dict(rows[-1],iteration=0);reject('reject_misaligned_checkpoint_marker',lambda:a.actualu_checks(run,changed,baseline))
        trace=run/'full151_actualu_trace_00002.npz';z=a.loadnp(trace);original={k:v.copy() for k,v in z.items()};z['actual'][0,0,0]+=np.float32(.001);np.savez_compressed(trace,**z)
        reject('reject_actual_replay_mismatch',lambda:a.actualu_checks(run,saved,baseline));np.savez_compressed(trace,**original)
        z={k:v.copy() for k,v in original.items()};z['phase'][0,0]=150;np.savez_compressed(trace,**z);reject('reject_terminal150_trace_real_action',lambda:a.actualu_checks(run,saved,baseline));np.savez_compressed(trace,**original)
        changed=copy.deepcopy(rows);changed[-1]['phase_histogram']=[0]*15;path.write_text(''.join(json.dumps(r)+'\n' for r in changed),encoding='utf-8');reject('reject_old15_phase_histogram',lambda:a.actualu_checks(run,saved,baseline));path.write_text(prefix,encoding='utf-8')
        # Raw from-scratch parameter hash and independent delta. These tiny
        # tensors deliberately are not the actual network architecture.
        irun=out/'synthetic_initial';irun.mkdir();rawmodules={};savedmodules={};fingerprints={};deltas={}
        for name in ('wm','fail_head','actor','critic'):
            vals=[torch.arange(4,dtype=torch.float32).reshape(2,2),torch.zeros(2)];names=['net.0.weight','net.0.bias']
            rawmodules[name]={'parameters':vals,'parameter_names':names,'shapes':[list(v.shape) for v in vals],'dtypes':[str(v.dtype) for v in vals]}
            savedmodules[name]={k:v+.001 for k,v in zip(names,vals)};h=hashlib.sha256()
            for v in vals:h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
            fingerprints[name]={'sha256':h.hexdigest(),'tensors':2,'parameters':6};deltas[name]=sum(float((savedmodules[name][k]-v).double().square().sum()) for k,v in zip(names,vals))**.5
        sourcepins={k:sources[k] for k in ('learning','training_capacity','controller','runtime','native')}
        initial={'schema':'full151_native_initial_parameters:1','modules':rawmodules,'capture_point':'after_native_DexRunner_init_before_first_collect_or_optimizer','source_pins':sourcepins}
        torch.save(initial,irun/'initial_parameters.pt');a.json_write(irun/'initial_parameters.json',fingerprints)
        sd={'modules':savedmodules,'config':smoke,'iteration':40,'full151_initial_parameter_fingerprints':fingerprints,'full151_capacity_evidence':{'source_pins':sourcepins}}
        result={'parameter_delta_l2':deltas}
        check('raw_initial_allfour_hash_names_shapes_delta_positive',all(a.initial_checks(sd,irun,result)[0].values()))
        damaged=copy.deepcopy(initial);damaged['modules']['actor']['parameters'][0][0,0]+=.1;torch.save(damaged,irun/'initial_parameters.pt');reject('reject_raw_initial_parameter_byte_mutation',lambda:a.initial_checks(sd,irun,result));torch.save(initial,irun/'initial_parameters.pt')
        damaged=copy.deepcopy(sd);damaged['full151_initial_parameter_fingerprints']['actor']['sha256']='0'*64;reject('reject_checkpoint_initial_fingerprint_drift',lambda:a.initial_checks(damaged,irun,result))
        wm_prefix=copy.deepcopy(sd);wm_prefix['iteration']=20
        for name in ('actor','critic'):wm_prefix['modules'][name]={k:v.clone() for k,v in zip(rawmodules[name]['parameter_names'],rawmodules[name]['parameters'])}
        check('WM_only_prefix_requires_actor_critic_unchanged',all(a.initial_checks(wm_prefix,irun,{},require_result_delta=False)[0].values()))
        wm_prefix['modules']['actor']['net.0.bias'][0]=.001;reject('reject_actor_change_before_policy_training',lambda:a.initial_checks(wm_prefix,irun,{},require_result_delta=False))
        check('state_architecture15head_and_native_context',a.expected_shapes()['wm']['net.9.weight']==[67,512] and a.expected_shapes()['actor']['net.0.weight']==[512,155] and a.expected_shapes()['fail_head']['net.0.weight']==[256,426] and a.expected_shapes()['critic']['net.0.weight']==[512,301])
        check('no_RNG_draws_or_native_imports',torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized() and not any(n.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for n in sys.modules))
        check('all_executed_sources_unchanged',all(a.pin(v['path'])==v for v in sources.values()))
        status='PASS_SYNTHETIC_CPU_CONTRACTS_ONLY'
    except Exception as e:status='CPU_CHECKER_FAILURE_RETAINED';notes['exception']=repr(e);notes['traceback']=traceback.format_exc()
    report={'schema':'full151105_health_auditor_CPU_contract:1','status':status,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'sources':sources,'notes':notes,'wall_s':time.perf_counter()-tick,
       'actual105_training_checkpoint_audit':'NOT_RUN_NONE_EXISTS_THIS_STAGE','full_actual_audit_function':'IMPLEMENTED_NOT_ACTUALLY_QUALIFIED_BY_THESE_HELPER_FIXTURES',
       'fixture_scope':'Synthetic JSONL/masks/flags/parameter tensors and original budget algebra. No optimizer, network, Env, physical rollout or old checkpoint migration. Replay generation is only source-validated summary; no raw generation window available.',
       'GPU_used':False,'native_environment_created':False,'physics_steps':0,'new_FK_collision_or_WM':False}
    a.json_write(out/'report.json',report);print({'status':status,'passed':report['passed'],'total':report['total'],'report':a.pin(out/'report.json')},flush=True)
    return 0 if status.startswith('PASS_') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);v=p.parse_args();raise SystemExit(main(v.output))
