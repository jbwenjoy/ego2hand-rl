"""Saved-only diagnosis of the155 feature layout, without executing native code."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/full151_native_A1L_policy_eval_smoke40_v1'
OUT=ROOT/'outputs/full151_policy_observation_layout_diagnostic_v2'
if OUT.exists():raise FileExistsError('Keep old diagnostic attempts')
OUT.mkdir(exist_ok=False);(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
def pin(p):
 p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
raw=load(RUN/'raw_states.npz');events=read(RUN/'control_events.json');native=load(ROOT/'outputs/full151_native_input_v1/native_reader_arrays.npz')
obs=np.asarray([e['policy_input155'][0] for e in events],np.float32);ix=4*np.arange(150)
comparisons={
 'reference_joint_q_0_20':(obs[:,:20],native['joint_pos'][:150]),
 'reference_BODY21_links_26_89':(obs[:,26:89],native['link_pos'][:150].reshape(150,63)),
 'current_joint_q_89_109':(obs[:,89:109],raw['joint_rad'][ix].astype(np.float32)),
 'quarter_current_joint_qd_109_129':(obs[:,109:129],np.float32(.25)*raw['joint_velocity_rad_s'][ix].astype(np.float32)),
 'previous_action_135_155':(obs[:,135:155],np.asarray([np.zeros(20)]+[e['action'][0] for e in events[:-1]],np.float32))}
metrics={k:{'exact_equal':bool(np.array_equal(a,b)),'max_abs_difference':float(np.max(np.abs(a-b))),'different_components':int(np.count_nonzero(a!=b))} for k,(a,b) in comparisons.items()}
wrong=obs[:,:20]-raw['joint_rad'][ix].astype(np.float32)
previous_error=float(np.max(np.abs(obs[1:,89:109]-raw['joint_rad'][ix[:-1]].astype(np.float32))))
src={
 'native_env_config':Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/pen_spin_env_cfg.py'),
 'native_command':Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/mdp/commands.py'),
 'native_pen_mdp':Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py'),
 'native_runner':Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py'),
 'installed_obs_manager':Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/managers/observation_manager.py'),
 'installed_manager_env':Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/envs/manager_based_rl_env.py'),
 'installed_joint_observation':Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/envs/mdp/observations.py'),
 'installed_entity_data':Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/entity/data.py')}
inventory=read(RUN/'source/inventory.json');sourcepins={k:pin(p) for k,p in src.items()}
for k,p in sourcepins.items():p['prior_execution_archive_keys']=[name for name,v in inventory.items() if Path(v['path']).resolve()==Path(p['path']).resolve() and v['sha256']==p['sha256']]
text={k:p.read_text(encoding='utf-8') for k,p in src.items()}
checks={
 '150policyrows_shape155_finite':obs.shape==(150,155) and bool(np.isfinite(obs).all()),
 'all_five_independently_identified_featuregroups_exact':all(v['exact_equal'] for v in metrics.values()),
 'wrong_old_assert_false_after_initialframe':bool(np.array_equal(obs[0,:20],raw['joint_rad'][0])) and int(np.count_nonzero(wrong))==2980,
 'first20_is_reference_not_current_or_previous_rawq':metrics['reference_joint_q_0_20']['exact_equal'] and float(np.abs(wrong).max())>0,
 'actual_q_current_exact_not_one_controltick_lag':metrics['current_joint_q_89_109']['exact_equal'] and previous_error>0,
 'featureorder_source_motion_first':text['native_env_config'].index('"motion": ObservationTermCfg')<text['native_env_config'].index('"joint_pos": ObservationTermCfg'),
 'command89_is_refq20_points6_BODY63': '(self.joint_pos, self.bounded_object_axis_points, self.link_pos.flatten(1))' in text['native_command'],
 'manager_clones_terms_and_concatenates_declared_order':'term_cfg.func(self._env, **term_cfg.params).clone()' in text['installed_obs_manager'] and 'list(group_obs.values())' in text['installed_obs_manager'],
 'nextobs_after_nativeforward_commandadvance':'self.sim.forward()' in text['installed_manager_env'] and text['installed_manager_env'].index('self.command_manager.compute(dt=self.step_dt)',text['installed_manager_env'].index('  def step('))<text['installed_manager_env'].index('self.obs_buf = self.observation_manager.compute(update_history=True)',text['installed_manager_env'].index('  def step(')),
 'real_collect_uses_env_returned_policy_obs':'self.obs = obs["policy"]' in text['native_runner'],
 'imagined_observation_same_command_first_layout':'(command, q - self.default, qd * 0.25, self.points(pos, rot, True).flatten(-2), memory["a1"])' in text['native_pen_mdp'],
 'original178_report_unchanged':pin(ROOT/'outputs/full151_native_A1L_smoke40_eval_saved_independent_v1/report.json')['sha256']=='deeb0a02932f25560bfda3be93568d5bccb7708abb15f7fd43b8a5dc0bd291c6',
 'no_framework_or_model_import':not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp'))}
report={'schema':'full151_policy155_saved_layout_diagnostic:1','status':'RESOLVED_OLD_AUDIT_FEATURE_INDEX_ASSUMPTION_NOT_PRODUCTION_CADENCE_BUG' if all(checks.values()) else 'DIAGNOSTIC_CHECK_FAILED_PRESERVED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':pin(__file__),'source_pins':sourcepins,'saved_inputs':{k:pin(p) for k,p in {'events':RUN/'control_events.json','raw_states':RUN/'raw_states.npz','native_reader_arrays':ROOT/'outputs/full151_native_input_v1/native_reader_arrays.npz','old178report':ROOT/'outputs/full151_native_A1L_smoke40_eval_saved_independent_v1/report.json'}.items()},'comparisons':metrics,'wrong_old_assert':{'max_abs_difference_rad':float(np.abs(wrong).max()),'different_components':int(np.count_nonzero(wrong)),'initialframe_equal_because_reference_and_init_q_match':True},'wrong_lag_comparison_max_rad':previous_error,
 'feature_map_half_open':{'0:20':'phase-k reference joint q, not measured/current q','20:26':'bounded reference object +/-X shaft points in fixed palm frame','26:89':'phase-k reference native BODY21 positions flattened','89:109':'current measured joint q + frozen encoder_bias0 - default_joint_q0','109:129':'current joint qd minus default0, scaled .25','129:135':'current bounded object +/-X points in palm frame; play camera/noise disabled','135:155':'previous clipped policy action, startup zero'},
 'timeline':'Action k consumes obs returned after interval k-1: poststate raw[4k], current command/reference k, previous action k-1. Reset supplies k0. Interval k then makes4x5ms steps, reward uses reference k before original forward/command advance, and returned obs selects reference k+1 for the next action. Terminal obs150 is validated but no action150 is executed.',
 'impact':['The old additional assertion confused Actor155 with state58: state58 starts with actual q, whereas Actor155 starts with reference command. Phase0 happened to match because initialization fixes q0 to the reference.','This is an audit assumption error. No production source, checkpoint, saved episode or physical criterion is changed. All previously reconstructed7 quality failures remain failures.','Actual real-collector and imagination source layouts agree. Saved smoke data prove the relevant current-q, current-qd, reference-q/BODY21 and previous-action slices; this does not replace the independent primary evaluation audit.','No evidence of one-controltick joint lag or raw-state tensor aliasing is needed to explain the mismatch. The manager clones each observation term and concatenates a new tensor; the saved current-q slice is exactly synchronous.'],
 'provenance_boundary':'Native application sources above are matched to execution inventory where available. Four installed mjlab files are source observations pinned now, not falsely represented as members of the old152 archive. This limitation does not change the independent exact saved-array slice comparisons.',
 'scope':'Source text and saved NumPy/JSON only. No production function, Env, Actor/WM load/forward, Reader, FK, collision, optimizer, physics or GPU.'}
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'report':pin(OUT/'report.json'),'passed':report['passed'],'total':report['total'],'status':report['status']}))
if not all(checks.values()):raise SystemExit(2)
