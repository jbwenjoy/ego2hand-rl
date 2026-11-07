"""Independent saved-artifact/static review; no producer/physics imports or child."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):
    p = Path(p)
    return {'path': str(p.resolve()), 'sha256': sha(p)}
def checks_report(checks):
    return {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
            'check_count': len(checks), 'passed': sum(checks.values())}
def save(name, report):
    p = ROOT / 'outputs' / name / 'report.json'
    p.parent.mkdir(parents=True, exist_ok=False)
    report.update(reviewer_source=pin(__file__), reviewed_utc=datetime.now(timezone.utc).isoformat(),
                  review_execution={'child_processes': 0, 'GPU': 0, 'physics_steps': 0,
                                    'solver_calls': 0, 'geometry_queries': 0,
                                    'production_functions_imported': False})
    p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'report': str(p), 'sha256': sha(p), 'status': report['status']}))
    if report['status'] != 'PASS':
        raise RuntimeError(name+' failed; artifacts retained')

ready_path = ROOT/'outputs/READY_dynamic_worker_v3.json'
ready = read(ready_path)
proof_path = Path(ready['v3_28_actual_and_alias_identity_proof']['path'])
proof = read(proof_path)
proof_root = proof_path.parent
watch = read(proof_root/'actual_CPU_watchdog/state.json')
runtime = read(proof_root/'actual_CPU_child/runtime.json')
post = read(proof_root/'actual_CPU_watchdog/post_terminal_CIM.json')
identity = read(proof_root/'actual_CPU_watchdog/engineering_identity.json')
observed = watch['actual_process_OS_observation']
py = proof['selected_python_identity']
dt = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00'))
identity_checks = {
    'READY_source_and_proof_pins_current': all(sha(v['path']) == v['sha256'] for v in ready['source_pins'].values()) and sha(proof_path) == ready['v3_28_actual_and_alias_identity_proof']['sha256'],
    'final_driver_and_archived_source_identical': sha(ready['source_pins']['supervisor_v3']['path']) == proof['source_sha256'] == sha(proof_root/'run_track_prefix_v3.py'),
    'proof_28_pass_no_error': proof['status'] == 'PASS' and proof['check_count'] == proof['passed'] == 28 and all(c['passed'] for c in proof['checks']) and proof['real_CPU_probe_error'] is None,
    'actual_PID_and_redirector_parent_bound': runtime['pid'] == watch['actual_pid'] == observed['pid'] == 43576 and runtime['parent_pid'] == watch['launcher_pid'] == observed['parent_pid'] == 21020,
    'runtime_and_fixture_digests_bound': runtime['source_sha256'] == sha(proof_root/'stdlib_identity_worker.py') == proof['fixture_sha256'] and runtime['plan_sha256'] == sha(proof_root/'CPU_fixture_plan.json') == proof['fixture_plan_sha256'],
    'watch_captured_identity_matches_terminal_runtime': all(watch['runtime_metadata'][k] == runtime[k] for k in ('pid','parent_pid','started_utc','source_sha256','plan_sha256')),
    'creation_order_actual_saved': dt(watch['Popen_prelaunch_utc']) <= dt(observed['creation_date_utc']) <= dt(runtime['started_utc']) <= dt(observed['observed_utc']) <= dt(runtime['finished_utc']),
    'observed_executable_samefile_base_raw_and_resolved': os.path.samefile(observed['executable_path'], py['base_executable_raw']) and os.path.samefile(observed['executable_path'], py['base_executable_resolved']) and sha(observed['executable_path']) == py['base_executable_sha256'],
    'actual_alias_exercised_distinct_venv_not_miscompared': py['base_executable_raw'] != py['base_executable_resolved'] and os.path.samefile(py['base_executable_raw'], py['base_executable_resolved']) and not os.path.samefile(py['chosen_python'], py['base_executable_raw']),
    'both_owned_PID_queries_empty_after_completion': post['status'] == 'NO_OWNED_PID_REMAINING' and set(post['queried_pids']) == {21020,43576} and post['actual_CIM_rows'] == [],
    'watch_and_strict_identity_completed': watch['status'] == 'completed' and watch['exit_code'] == 0 and watch['timeout'] is False and identity['engineering_identity_qualified'] and all(identity['checks'].values()),
    'one_stdlib_child_no_future_or_physics': proof['attempted_CPU_children'] == 1 and proof['automatic_retry'] is False and proof['future_bundle_read'] is False and proof['future_physics_outputs_created'] is False and proof['physics_steps'] == 0 and proof['GPU_started'] is False and runtime['CPU_fixture_only'] and not runtime['Torch_imported'] and not runtime['Warp_imported'],
}
save('dynamic_identity_independent_review_v1', {
    **checks_report(identity_checks), 'source_pins': {'READY': pin(ready_path), 'proof': pin(proof_path), **ready['source_pins']},
    'actual_CPU_identity': {'actual_pid':43576,'launcher_pid':21020,'OS_creation_utc':observed['creation_date_utc'],'watch_elapsed_s':watch['elapsed_s']},
    'static_observation': 'Final v3 parses Windows argv0 with CommandLineToArgvW and compares actual executable files with samefile; PID/PPID/time/source/plan/output/postterminal fail-closed gates remain. The actual CPU proof exercises the host venv redirector and base-file junction; alias-positive and rejection fixtures are additional metadata tests.',
    'scope': 'Saved actual stdlib child and static source only. This closes executable-identity route evidence, not actual Warp initialization/capacity/control/physical quality.'})

worker = Path(ready['source_pins']['worker']['path'])
save('dynamic_worker_independent_static_review_v1', {
    **checks_report({'worker_8834_unchanged':sha(worker)=='8834a386f001cc7a138fba3906774bc11d6a2a996c0695ddfc772eff11a90db7', 'reviewed_dependency_pins_current':all(sha(ready['source_pins'][k]['path'])==ready['source_pins'][k]['sha256'] for k in ('reviewed_lifecycle','runtime_IO','measured_contacts','protocol'))}),
    'source_pins': {k:ready['source_pins'][k] for k in ('worker','reviewed_lifecycle','runtime_IO','measured_contacts','protocol')},
    'static_facts': [
        {'lines':[97,165],'fact':'Default CPU path validates frozen bundles, exact legacy[200,216) identity, analytic nonzero qd and world/body object twist, original hard/control intersection and FP32 C2 targets; only kinematics, no Torch/Warp/step/forward.'},
        {'lines':[196,224],'fact':'Explicit authorization only: original nominal model upload, all8 actual context copies, set_const_0 restore=True, actual context readbacks after both stages, then one startup forward. Full initial qpos/qvel/ctrl/mocap and FK readback precede any step.'},
        {'lines':[227,260],'fact':'Raw state k is postintegration; contacts/torque retain solver prestate k-1 and separate clock. 60 steps with no state/object/force assistance; 14 control updates before steps5,9,...57, identical write cadence in hold/track. At state4 ctrl is u0; step5 uses u1 at prestate4.'},
        {'lines':[261,278],'fact':'Sticky overflow is read once after step60, no reset/clear; strict capacity counts. Full61states/60solver/14events required; quality failure does not truncate, exceptions yield partial engineering failure.'},
        {'lines':[179,193],'fact':'Original position/shaft/state/depth/load/drop/controller/clock/capacity gates remain. Primary16 samples and diagnostic61 are separate. No full3s, contact-label, learned-controller, continuous-safety or hardware claim.'},
    ],
    'boundary': 'Static route review does not establish actual API behavior; post-run initialization/context/ctrl/capacity/identity/raw gates remain necessary. qd0 may exceed15rad/s because15 limits executed-target updates, not designed state velocity.',
    'paired_policy': 'One predeclared H0 then one T0. H0 qualityfalse does not stop T0; H0 engineering incomplete, identityunqualified or capacityUNKNOWN prevents the second attempt. No quality/history selection, retry or automatic launch.'})

plans={k:read(ROOT/f'configs/dynamic-prefix-{k}-cpu_v1.json') for k in ('hold','track')}
preflights={k:read(ROOT/f'outputs/dynamic_actual_CPU_preflight_v1/{k}.json') for k in plans}
p=plans['hold']; manifests=[read(p[k]['path']) for k in ('reference_manifest','init_manifest','control_manifest')]
bundles=[]
for m in manifests:
    with np.load(m['artifact']['path'],allow_pickle=False) as z:bundles.append({k:z[k].copy() for k in z.files})
ref,init,ctrl=bundles
track=ctrl['control__fp32_old_export__tracking_u_exec_15_FP32_rad']; hold=ctrl['control__fp32_old_export__hold_u_exec_15_FP32_rad']
qd=init['qd_init_requested_FP32_rad_s']; pose=init['object_pose_init_requested_FP32_w']
q=pose[3:].astype(float);q/=np.linalg.norm(q);w,x,y,z=q
R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
policy=p['campaign_policy']
plan_checks={
    'conditions_only_scientific_difference': {k for k in plans['hold'].keys()|plans['track'].keys() if plans['hold'].get(k)!=plans['track'].get(k)} == {'control_condition','output'},
    '101_source_pins_each_current': all(len(v['source_pins'])==101 and all(sha(t['path'])==t['sha256'] for t in v['source_pins'].values()) for v in plans.values()),
    'CPU_preflights_bind_current_plans_worker_driver': all(preflights[k]['status']=='passed_CPU_preflight_no_GPU' and preflights[k]['plan_sha256']==sha(ROOT/f'configs/dynamic-prefix-{k}-cpu_v1.json') and preflights[k]['worker_sha256']==sha(worker) and preflights[k]['supervisor_sha256']==ready['source_pins']['supervisor_v3']['sha256'] for k in plans),
    'both_CPU_no_Torch_Warp':all(not preflights[k]['GPU_started'] and not preflights[k]['input_contract']['Torch_imported'] and not preflights[k]['input_contract']['Warp_imported'] for k in plans),
    'checkonly_no_GPU_training_retry':all(v['GPU_run_authorized'] is False and v['training_authorized'] is False and v['automatic_retry'] is False for v in plans.values()),
    'three_bundles_hash_bound_and_unmeasured':all(sha(m['artifact']['path'])==m['artifact']['sha256'] and m['physical_reference_qualified'] is False and m['init_measured'] is None and m['u_readback'] is None for m in manifests),
    'same_new_u0_and_only_track_varies':np.array_equal(track[0],hold[0]) and np.array_equal(hold,np.broadcast_to(track[0],hold.shape)) and np.any(track[1:]!=track[0]) and np.array_equal(track[0],init['u_planned0_FP32_rad']),
    'q0_qd0_pose0_cast_from_new_analytic_reference':np.array_equal(init['q_init_requested_FP32_rad'],ref['q_goal_rad'][0].astype(np.float32)) and np.array_equal(qd,ref['analytic_qd16_rad_s'][0].astype(np.float32)) and np.array_equal(pose,ref['mapped_object_pose_w'][0].astype(np.float32)),
    'object_twist_world_body_conversion_independently_recomputed':np.array_equal(init['object_bodyorigin_v_requested_FP32_world_m_s'],ref['object_v_world_m_s'][0].astype(np.float32)) and np.array_equal(init['object_angular_requested_FP32_body_rad_s'],(R.T@ref['object_omega_world_rad_s'][0]).astype(np.float32)),
    'original_budget_and_rate':all(v['steps']==60 and v['dt_s']==.005 and v['hard_process_seconds']==45 and v['soft_worker_seconds']==40 for v in plans.values()) and np.abs(np.diff(track.astype(float),axis=0)/.02).max()<=15,
    'predeclared_pair_qualifies_engineering_not_taskquality':policy['conditions_in_order']==['H0_MATCHED_DYNAMIC_INIT_HOLD','T0_MATCHED_DYNAMIC_INIT_TRACK'] and policy['attempts_per_condition']==1 and policy['first_quality_failure_does_not_change_second_plan'] is True and policy['second_condition_requires_first_engineering_identity_full_horizon_capacity_OK'] is True and policy['automatic_retry'] is False and policy['same_14_control_write_events'] is True,
}
save('dynamic_actual_preflight_independent_review_v1', {
    **checks_report(plan_checks), 'source_pins':{**{k+'_plan':pin(ROOT/f'configs/dynamic-prefix-{k}-cpu_v1.json') for k in plans}, **{k+'_preflight':pin(ROOT/f'outputs/dynamic_actual_CPU_preflight_v1/{k}.json') for k in plans}},
    'new_initial_design':{'max_abs_qd0_rad_s':float(np.abs(qd).max()),'object_v_world_m_s':init['object_bodyorigin_v_requested_FP32_world_m_s'].tolist(),'object_omega_body_rad_s':init['object_angular_requested_FP32_body_rad_s'].tolist(),'track14_update_max_rad_s':float(np.abs(np.diff(track.astype(float),axis=0)/.02).max()),'analytic_qd_distinct_from_target_update_rate':True},
    'campaign_policy':policy,'source_identity':read(p['protocol']['path'])['source_identity'],
    'scope':'Inputs and previously generated CPU preflights only, no new FK/query/forward/step. These CPU-only plans are not executable GPU authorization. Formal plans must separately bind exact evidence and distinct outputs; initial/context/readback/identity/capacity/full horizon remain post-run gates.'})
