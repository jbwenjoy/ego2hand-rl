"""Independent formal source/pin closure and attempted/returned counter delta.

Only file/JSON/AST operations. Production modules are not imported or executed.
"""
import argparse,ast,copy,difflib,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def checked(p):return set(p)=={'path','sha256'} and Path(p['path']).is_absolute() and sha(p['path'])==p['sha256']
def assignments(tree):
    result={}
    class Substitute(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id in result:return ast.parse(repr(result[node.id]),mode='eval').body
            return node
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:result[n.targets[0].id]=ast.literal_eval(Substitute().visit(copy.deepcopy(n.value)))
            except (ValueError,TypeError):pass
    return result


def main(config,output):
    tick=time.perf_counter();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);checks={};facts={};exceptions=[]
    def check(k,v):checks[k]=bool(v)
    p=read(config);proto=read(p['protocol']['path']);pins=p['source_pins']
    try:
        check('exact_Root_authorized_config',sha(config)=='303d61c55ab62d03d1b8b62fa0f0af8aedc1fa3ee4846d1e6e6b3972d00c30b4')
        check('protocol_pin_and_alias',checked(p['protocol']) and p['protocol']==pins['protocol'])
        check('136_source_and_input_pins',len(pins)==136 and all(checked(v) for v in pins.values()))
        facts['source_pin_count']=len(pins);facts['source_pin_results']={k:checked(v) for k,v in pins.items()}
        check('A0_only_one_seed_no_prior_no_training',p['schema']=='full151_knownrough_native_A0_run:1' and p['role']=='native_a0' and p['method']=='A0' and type(p['seed']) is int and p['seed']==0 and p['training']=={'num_envs':1} and type(p['training']['num_envs']) is int and p['contact_binding'] is None and p['training_authorized'] is False)
        check('explicit_authorization_once_no_retry_fallback',p['GPU_run_authorized'] is True and p['automatic_retry'] is False and p['compile_fallback'] is False and p['soft_worker_seconds']==180 and p['hard_process_seconds']==240)
        check('original_full151_protocol_horizon_and_rough_scope',proto['reference_states']==151 and proto['control_intervals']==150 and proto['source_frame_range_half_open']==[200,351] and proto['physical_reference_qualified'] is False and proto['geometric_candidate_qualified'] is False and proto['known_rough_reference'] is True and proto['training_authorized'] is False)
        check('no_u150_native_end',proto['episode_counts']=={'real_intervals':150,'solver_steps':600,'raw_states':601,'native_postinterval_forwards':150,'last_real_phase':149,'terminal_observation':150,'u150_permitted':False})
        check('protocol_keeps_quality_early_stop_false',proto['environment']['quality_early_stop'] is False and proto['environment']['substeps_per_interval']==4 and proto['environment']['num_envs']==1)
        texts={k:Path(pins['full151__'+k+'.py']['path']).read_text(encoding='utf-8-sig') for k in ('native','probe','runtime','controller')}
        n,q=texts['native'],texts['probe'];qn=ast.parse(q);native=ast.parse(n)
        check('final_native_source_snapshot',pins['full151__native.py']['sha256']=='142cd9b710786d1973c6c518165e1d08bfd3a38b7a65b425819b538e3d025a2f')
        check('final_probe_source_snapshot',pins['full151__probe.py']['sha256']=='38112d5d50eeb48ecc926815327f8d57c9ba0784289b8b42dd7eb02b71f13a56')
        constants=assignments(qn)
        check('grade_and_clock_exact_protocol_config_source',constants['GATES']==proto['evaluation_gates']==p['evaluation_gates'] and constants['CLOCK_CONTRACT']==proto['clock_contract'])
        check('counter_start_zero_attempt_and_return_separate',"api_counts = {'step_attempted': 0, 'step_returned': 0," in q and "'forward_attempted': 0, 'forward_returned': 0}" in q)
        steps=next(x for x in ast.walk(qn) if isinstance(x,ast.FunctionDef) and x.name=='step');forwards=next(x for x in ast.walk(qn) if isinstance(x,ast.FunctionDef) and x.name=='forward')
        s=ast.get_source_segment(q,steps);f=ast.get_source_segment(q,forwards)
        check('step_attempt_before_call_return_after_call_before_sample',s.index("api_counts['step_attempted'] += 1")<s.index('original_step()')<s.index("api_counts['step_returned'] += 1")<s.index('sample(len(states))'))
        check('forward_attempt_before_call_return_after_success',f.index("api_counts['forward_attempted'] += 1")<f.index('result = original_forward()')<f.index("api_counts['forward_returned'] += 1")<f.index('return result'))
        check('normal_completion_all_four_exact600600150150',"api_counts != {\n                'step_attempted': 600, 'step_returned': 600,\n                'forward_attempted': 150, 'forward_returned': 150}" in q)
        check('partial_returned_and_sampled_rows_not_conflated',"physical_steps=api_counts['step_returned']" in q and "sampled_state_rows=len(states), sampled_solver_rows=len(solver)" in q and 'final_read_after_returned_episode_step=' in q and 'final_read_after_sampled_solver_row=' in q)
        check('startup_counts_unknown_authoritative_native_fields','UNKNOWN_NOT_WHOLEPROCESS_OBSERVED' in n and "construction_reset_setup_API_counts='UNKNOWN_NOT_WHOLEPROCESS_OBSERVED'" in q)
        check('A0_no_actual_Runner_network_or_optimizer_construction',not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id in ('DexSuperMimicRunner','Actor','WorldModel','Adam','AdamW','Critic','FailureHead') for x in ast.walk(native)))
        oldn=n.replace('Probe episode only; constructor, actual8 setup and reset API work is UNKNOWN_NOT_WHOLEPROCESS_OBSERVED.','Probe episode only; constructor, actual8 setup and reset work are separate.')
        inverse=[
          ("    api_counts = {'step_attempted': 0, 'step_returned': 0,\n                  'forward_attempted': 0, 'forward_returned': 0}\n    error = None; capacity = {'status': 'UNKNOWN'}", "    forward_calls = 0; error = None; capacity = {'status': 'UNKNOWN'}"),
          ("        api_counts['step_attempted'] += 1\n        original_step()\n        api_counts['step_returned'] += 1\n        sample(len(states))",'        original_step(); sample(len(states))'),
          ("        api_counts['forward_attempted'] += 1\n        result = original_forward()\n        api_counts['forward_returned'] += 1\n        return result",'        nonlocal forward_calls\n        forward_calls += 1\n        return original_forward()'),
          ("            if k % 10 == 9: emit(status='running', phase='full151_native_A0', control_intervals=k+1,\n                physics_steps=api_counts['step_returned'], sampled_solver_rows=len(solver), episode_API_counts=dict(api_counts))", "            if k % 10 == 9: emit(status='running', phase='full151_native_A0', control_intervals=k+1, physics_steps=len(solver))"),
          ("        if len(states) != 601 or len(solver) != 600 or api_counts != {\n                'step_attempted': 600, 'step_returned': 600,\n                'forward_attempted': 150, 'forward_returned': 150}:",'        if len(states) != 601 or len(solver) != 600 or forward_calls != 150:'),
          ("        capacity.update(final_read_after_returned_episode_step=api_counts['step_returned'],\n            final_read_after_sampled_solver_row=len(solver), contact_capacity=cc, constraint_capacity=jc,",'        capacity.update(final_read_after_solver_step=len(solver), contact_capacity=cc, constraint_capacity=jc,'),
          ("            physical_steps=api_counts['step_returned'], native_postinterval_forward_calls=api_counts['forward_returned'],\n            episode_API_counts=dict(api_counts), sampled_state_rows=len(states), sampled_solver_rows=len(solver),\n            counter_scope='Only this probe episode after startup readback; counts describe API attempts/returns, not device-kernel calls.',\n            construction_reset_setup_API_counts='UNKNOWN_NOT_WHOLEPROCESS_OBSERVED', control_intervals=len(events),",'            physical_steps=len(solver), native_postinterval_forward_calls=forward_calls, control_intervals=len(events),')]
        oldq=q
        for current,previous in inverse:
            if oldq.count(current)!=1:raise ValueError('Counter-only inverse delta anchor not unique: '+current[:60])
            oldq=oldq.replace(current,previous,1)
        oldhashes={}
        for name,text,expected in [('native',oldn,'d9177adf578db78fb999c7cd4a67883e0cefb5d3967033c892ad7319070929ab'),('probe',oldq,'19aff6ca2afffd9a649eca23a4c2e50b789d2deaa15ab006a7f843407c06ee48')]:
            payload=text.encode();h=hashlib.sha256(payload).hexdigest()
            if h!=expected:
                other=text.replace('\n','\r\n').encode()
                if hashlib.sha256(other).hexdigest()==expected:payload=other;h=expected
            check('counter_only_inverse_recovers_old_'+name+'_SHA',h==expected);oldhashes[name]=h
            (out/(name+'_reconstructed_old_snapshot.py')).write_bytes(payload)
            (out/(name+'_counter_delta.diff')).write_text(''.join(difflib.unified_diff(text.splitlines(True),texts[name].splitlines(True),fromfile='old_'+name,tofile='current_'+name)),encoding='utf-8')
        facts['inverse_recovered_old_hashes']=oldhashes
        bundle=read(p['input_bundle']['path'])
        alias_matches={k:[name for name,w in pins.items() if Path(w['path']).resolve()==Path(v['path']).resolve() and w['sha256']==v['sha256']] for k,v in bundle['source_pins'].items()}
        facts['bundle_source_aliases_in_136pin_manifest']=alias_matches
        check('all_loaded_bundle_pins_explicitly_frozen',all(alias_matches.values()))
        check('same_validated_bundle_and_no_new_target_input',p['input_bundle']==proto['input_bundle']==pins['input_bundle'] and bundle['contact_intent_bundle'] is None and bundle['human_contact_observation_known'] is False)
        facts['output_exists_at_review']=Path(p['output']).exists()
        facts['closure_review_time_scope']='Completed after Root terminal notice; original pre-execution key gates were sent separately. No claim that output is still absent.'
        facts['input_signature']=read(Path(p['input_bundle']['path']).with_name('descriptor.json'))['input_signature_sha256']
        check('source_input_signature_exact',facts['input_signature']=='3599bee61d92152b0c4f7c3bf8b71e99d2c00af460c81d23cecbd76d7e3fdd04')
        facts['actual_measured_contacts_path']=pin('D:/workspace/supermimic-rough-prefix-learning-study/vendor/measured_contacts.py')
        check('actual_contact_helper_bytes_equal_pinned_local_copy',facts['actual_measured_contacts_path']['sha256']==pins['local_measured_contacts']['sha256']=='478c3117489bd4a02238f6299a91638e791a751352b772321ce2676b0ec4f96f')
        worker=Path(pins['worker']['path']).read_text(encoding='utf-8-sig');supervisor=Path(pins['supervisor']['path']).read_text(encoding='utf-8-sig')
        check('worker_preflight_and_hash_source_snapshot_before_native','p,bundle,proof=preflight(path)' in worker and 'freeze_sources(out,p)' in worker and worker.index('freeze_sources(out,p)')<worker.index('from full151_native.native import run_native'))
        check('worker_quality_not_engineering_stop',"if not evaluation.get('complete')" in worker and "'quality_is_not_engineering_gate':True" in worker)
        check('supervisor_shared_lock_exact_owned_watchdog_and_identity',all(s in supervisor for s in ('with life.gpu_lock():','reject_other_workers()','life.watch(command,output,logs,hard_seconds=p[\'hard_process_seconds\']','d.qualify_terminal_identity','terminal_processes','CREATE_NO_WINDOW')))
        check('old_review_retained_not_relabelled',read(ROOT/'outputs/full151_native_A0_static_independent_v1/report.json')['sources']['native']['sha256']=='d9177adf578db78fb999c7cd4a67883e0cefb5d3967033c892ad7319070929ab')
        check('all136pins_postreview_unchanged',all(checked(v) for v in pins.values()))
    except Exception as e:exceptions.append(repr(e))
    status='PASS_CURRENT_SOURCE_FORMAL_STATIC_CLOSURE' if checks and all(checks.values()) and not exceptions else 'STATIC_REVIEW_FAILURE_RETAINED'
    report={'schema':'full151_native_A0_formal_static_independent:1','status':status,'reviewer':pin(__file__),'config':pin(config),'protocol':p['protocol'],'checks':checks,'passed':sum(checks.values()),'total':len(checks),'facts':facts,'exceptions':exceptions,'wall_s':time.perf_counter()-tick,
      'scope':['Only JSON/files/AST read; no production import, default entry rerun, Env, GPU or physical query.','Counter-only inverse reconstruction is qualified only when both explicit SHA checks pass; see checks/facts. Prior38 source review remains separate, never relabelled as current.','New native/probe explicit startup UNKNOWN fields are authoritative. Entry physics_step_scope wording separately recorded does not establish or permit inferring constructor/setup counts.','Actual old-path measured-contact helper and pinned local copy executable-content alias is qualified only by its explicit SHA check in this report.','This static receipt does not authorize or qualify an actual physical outcome. Actual API/clock/reset/capacity/identity must be audited from later raw logs.'],
      'production_imported':False,'Env_created':False,'new_FK_collision_or_physics':False,'GPU_used':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print({'status':status,'passed':report['passed'],'total':report['total'],'report':pin(out/'report.json')},flush=True)
    return 0 if status.startswith('PASS_') else 1


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--output',required=True);p=a.parse_args();raise SystemExit(main(p.config,p.output))
