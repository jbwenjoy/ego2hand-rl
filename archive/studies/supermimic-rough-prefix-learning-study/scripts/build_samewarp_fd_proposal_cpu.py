"""Create a NOT_AUTHORIZED, saved-input-bound physical-FD proposal only."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'proposals/current104_samewarp_physical_fd_v1'
NATIVE=Path('D:/workspace/dex-supermimic-research/src/dex_supermimic')


def pin(path):
    p=Path(path).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest()}


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    authority=ROOT/'protocols/rough-wm-onpolicy-cpu-diagnostic-v1.json'
    cpu=json.loads(authority.read_text(encoding='utf-8'))
    p={
      'schema':'current104_same_Warp_physical_FD_proposal:1',
      'status':'DRAFT_NOT_AUTHORIZED_NOT_IMPLEMENTED_NOT_RUN',
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'GPU_run_authorized':False,'training_authorized':False,'automatic_retry':False,
      'basis':'Post CPU-diagnostic design; known CPU175/288 out-of-domain pairs motivate a declared symmetric interior-coordinate experiment. This is not a pre-CPU preregistration.',
      'purpose':'Local one-control-interval absolute-target causal FD versus all3 frozen104 WMs, on same nativeWarp backend/cadence and newly captured complete snapshots.',
      'implementation':{'worker':None,'supervisor':None,'checker':None,'status':'PENDING_source_implementation_and_independent_CPU_review_then_separate_Root_authorization'},
      'inputs':{'CPU_protocol':pin(authority),'CPU_runplan':pin(ROOT/'configs/WM-onpolicy-CPU-diagnostic-runplan-v2.json'),
                'CPU_result_read_as_provenance_only':pin(ROOT/'outputs/WM_onpolicy_CPU_diagnostic_v1/report.json'),
                'models':cpu['inputs']['models'],'saved_trajectories':cpu['inputs']['trajectories'],
                'input_bundle':cpu['inputs']['input_bundle'],'reference_arrays':cpu['inputs']['reference_arrays'],
                'init_arrays':cpu['inputs']['init_arrays'],'baseline_control_arrays':cpu['inputs']['control_arrays'],
                'actual_context':cpu['inputs']['authoritative_context']},
      'source_authorities':{
          'native_snapshot':pin(NATIVE/'rl/dex_supermimic/snapshot.py'),
          'native_state_reader_and_sensor':pin(NATIVE/'rl/dex_supermimic/pen.py'),
          'native_state':pin(NATIVE/'rl/dex_supermimic/state.py'),
          'native_SimReader':pin(NATIVE/'rl/dex_supermimic/sim.py'),
          'native_Manager_step':pin(Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/envs/manager_based_rl_env.py')),
          'rough_runtime':pin(ROOT/'src/rough_prefix/runtime_v2.py'),
          'rough_controller':pin(ROOT/'src/rough_prefix/controller.py'),
          'rough_native_environment':pin(ROOT/'src/rough_prefix/native_v2.py'),
          'rough_policy_orchestration':pin(ROOT/'src/rough_prefix/evaluation_v2.py'),
          'rough_policy_probe':pin(ROOT/'src/rough_prefix/policy_probe.py'),
          'rough_A0_probe':pin(ROOT/'src/rough_prefix/probe.py'),
          'WM104_loader':pin(ROOT/'src/rough_prefix/world_model_checkpoint_v2.py'),
          'state58_codec':pin(ROOT/'src/rough_prefix/saved_state_codec_v3.py')},
      'historical_experience':{
          'old_serial_plan':pin(Path('D:/workspace/supermimic-chord-study/outputs/wm_gradient_probe_serial/plan.json')),
          'old_serial_summary':pin(Path('D:/workspace/supermimic-chord-study/outputs/wm_gradient_probe_serial/summary.json')),
          'old_serial_source_current_bytes':pin(Path('D:/workspace/supermimic-chord-study/scripts/probe_world_model_gradients.py')),
          'old_Manager_microprobe_source':pin(Path('D:/workspace/supermimic-control-feasibility-draft/scripts/braking_microprobe.py')),
          'interpretation':'Old296 serial qualification belongs to oldformat3/151reference/controller. A later Manager microprobe failed baseline despite exact restore. Neither result qualifies current104; fullsnapshot plus fresh replay gates remain mandatory.'},
      'selection':{'trajectory_order':['A0','A1L','P_ONLY','A2'],'phase_k':[0,4,8,12],
                   'state_count':16,'one_source_lineage':True,'training_seed':0,
                   'quality_or_signal_based_state_selection':False,'best_attempt_selection':False,
                   'source_window_half_open':[200,216],'reference_qualified':False},
      'donor':{
          'worlds':1,'environment_allocation':'One native Manager world per path; finish donor/snapshots/branches in that same instance before close; no cross-world solve used as truth.',
          'action_source':{'A0':'15x20 exact declared zero raw actions; original A0 event file has no action field, do not pretend it was logged.',
                           'A1L_P_ONLY_A2':'exact control_events[k].action[0], FP32, all15; actor_raw_action equality checked, no live Actor inference or new noise'},
          'length':'15 transitions,60x5ms,15 normal Manager postinterval forwards,16 observations; no auto reset or quality truncation',
          'init':'same checked bundle analytic nonzero qd/object world-v/body-omega, q/object/mocap/u0; no original source robot q or cached wrist substitution',
          'model_context_order':'nominalMJB upload -> actual8 host/devicecopy -> set_const_0(restore=True) -> verify all8+xi46 -> first reset -> full initial readback',
          'instrumentation':'read/copy/synchronize only; do not add forward, sense, observation compute, controller process or physical step',
          'snapshot_phase':'phase k immediately before process_action; k>0 is after previous original Manager forward/command/sense/obs; k0 after checked reset',
          'archive_comparison':'compare all61 q/qvel and executedu timeline, report selected snapshot and next-state drifts in physical units; do not replace any original quality outcome or pick a donor that resembles archive',
          'archive_gate':'initialization/u/context identity is setup-required; subsequent q/qvel/object drift is a separately reported archival-reproduction flag, not a replacement for fresh baseline qualification',
          'fresh_WM_inputs':'derive native58 from actual newly captured donor and nominal executedu, not silently reuse archivedx at a drifted state; retain oldCPU matrix as separate inputs/cost'},
      'snapshot_contract':{
          'integration_signature':'mjSTATE_INTEGRATION via actual installed mj_stateSize/get_state/set_state; exact integer signature/version recorded',
          'required_integration_channels':['time','qpos','qvel','act','qacc_warmstart','ctrl','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','eq_active','userdata','plugin/integration channels if present'],
          'full_inprocess_snapshot':'same-instance WarpTaskSnapshot semantics: all reachable wp_model/wp_data arrays including warmstart/contact/constraint caches, whitelisted task/manager tensors/containers, object identities and Python/NumPy/TorchCPU/CUDA RNG',
          'walker_limit':'Native walker does not traverse function closures, clone NumPy arrays or arbitrary rough_prefix custom classes. Explicit copies/inventory/restore of every used external mutable controller/observer/cache sidecar are mandatory, not inferred from integration exact0.',
          'integration_limit':'INTEGRATION does not include derived/contact/sensordata/overflow. Its set_state does not itself clear overflow. Full WarpTaskSnapshot array restoration can rewind those flags; distinguish the two operations in logs.',
          'extra_explicit_sidecars':['full PEN memory including phase/start/end/age/counters/c1/c2/a1/a2/frozen target/limits','ActionTerm raw/processed/targets/validmask and controller pending state','observation history/delay/cache','termination/manual-reset pending state','physics/episode/command counters','compiled graph/model/data identities'],
          'restore':'outside branch only: full snapshot.restore(); exact readback; integration set/get exact; explicit sidecar equality; no forward to make a failed restore look plausible',
          'inside_branch_state_writes':0,'only_inside_branch_intervention':'executed absolute target after original action processing; original action-manager physical ctrl writes remain normal',
          'no_reduced_state_restart':'qpos/qvel-only, mj_resetData, cross-world clone, nativeCPU solve are prohibited truth substitutes'},
      'branch':{
          'baseline_repetitions_per_snapshot':3,'physics_steps':4,'physics_dt_s':0.005,'control_dt_s':0.02,
          'cadence':'original env.step(source_raw_action): original process/update residual memory, apply held cached target4 times, original scene updates/termination/reward then exactly one original forward/command/sense/obs',
          'intervention_hook':'after original process_actions, replace cached executed_targets with chosen lawful absolute target; preserve raw action, processed EMA c1/c2 and phase semantics; verify actual native ctrl bitexact at all4substeps',
          'original_policy_probe_limit':'Do not run the original probe baseline+EMA/C2 actualtarget assertion unchanged on perturbation branches: it correctly rejects intentional absoluteu interventions. New observer must independently validate originalsource EMA sidecars and separately validate declared actualu_override, without mutating the frozen original probe.',
          'controller_semantics':'do not reconstruct residual from a previously clamped target; no second C2 projection, no clipping/resampling perturbations, no added state/object/force assistance',
          'phase':'x at snapshot k; x1 rawpostintegration20ms; physical objective current ref k primary, k+1 tracking secondary; no u15',
          'execution_order':'per path: full donor; phase0 baseline0/1/2 then directions; phase4,8,12 likewise; fixed model-independent directions; close then nextpath',
          'failure':'exact restore/setup/capacity/nonfinite failure immediately stops; a numerical output replay failure is judged after the predeclared3baseline repetitions, then stops before perturbations. Preserve all partial/NOT_RUN cells; no retry/new donor or luckybaseline selection.',
          'baseline_quality_failure':'position/shaft/taskquality never selects snapshots or aborts a finite engineering-valid branch'},
      'directions':{
          'count_per_snapshot':8,'shared_by_all3_models':True,'seed':2026110401,'PRNG':'NumPy default_rng',
          'eps_target_L2_rad':[0.002,0.001],
          'coordinate_roundoff_guard':'8*eps_float32*max(1,abs(saved_actualu_j),abs(L_C2_j),abs(U_C2_j)); NEW_PROPOSED pre-physical FP32 arithmetic allowance, not a physical tolerance',
          'mask':'free_j = min(saved_actualu_j-L_C2_j,U_C2_j-saved_actualu_j) >= .002 + coordinate_roundoff_guard_j, using actual FP32 inner(hardjoint intersect ctrlrange) bounds of the frozen C2_020 controller; require new donor baselineu bitexact to saved_actualu, never recompute a favorable mask after replay',
          'construction':'path order A0/A1L/P_ONLY/A2 then phase0/4/8/12; each state draw standard_normal((8,20)), retain raw20 and full20-unit vectors; zero non-free raw coordinates then L2 normalize once; same resulting8directions used for every WM',
          'empty_or_numerically_zero':'record NO_FREE_COORDINATES/ZERO_MASKED_VECTOR and all required omitted branches; never resample or select by response',
          'domain':'plusminus=(savedu.astype(float64) +/- eps*direction_float64).astype(float32); save all effectiveFP32directions and crossing masks, validate before physics; violation is invalid branch, no clamp; no internalctrl clipping may erase intended target difference unnoticed',
          'scope':'symmetric interior-coordinate subspace, not complete tangent cone; bound-near face-normal/one-sided inward directions remain untested',
          'full_dimension_CPU_evidence':'prior180CPU/full20-random FD results and175skips retained separately; new masked directions never relabeled as their completion'},
      'counts_maximum':{
          'snapshots':16,'direction_cases':128,'central_FD_pairs':256,'perturbation_branches':512,
          'baseline_branches':48,'all_branches':560,'branch_substeps':2240,'branch_native_forwards':560,
          'donor_transitions':60,'donor_substeps':240,'donor_native_forwards':60,'startup_forwards':4,
          'total_physics_substeps':2480,'total_postinterval_forwards':620,'aggregate_simulated_time_s':12.4,
          'physical_branch_cost_not_multiplied_by_WM_count':True,'WM_directional_AD_values':384,
          'maximum_WM_vs_physical_slope_comparisons_including_both_eps':768,
          'if_mask_or_gate_fails':'actual counts lower; retain planned560 denominator and exact omitted-reason table'},
      'numerical_gates':{
          'source':'Legacy numbers are preexisting old296 rules, not inherited qualification. Extensions explicitly NEW_PROPOSED before any new physical result.',
          'integration_restore_abs_atol':{'value':0.0,'origin':'LEGACY_old296'},
          'full_arrays_memory_RNG_ctrl_readback':{'value':'bitexact','origin':'NEW_PROPOSED_stronger_complete-state gate'},
          'baseline_output_vs_fresh_donor':{
              'q_maxabs_rad':{'value':2e-5,'origin':'LEGACY_old296_position-channel rule'},
              'object_p_maxabs_m':{'value':2e-5,'origin':'LEGACY_old296_position-channel rule'},
              'object_R_entry_maxabs':{'value':2e-5,'origin':'LEGACY_old296_position-channel rule; also report SO3 angle without an extra hidden gate'},
              'qd_maxabs_rad_s':{'value':2e-4,'origin':'LEGACY_old296_velocity-channel rule'},
              'object_v_maxabs_m_s':{'value':2e-4,'origin':'LEGACY_old296_velocity-channel rule'},
              'object_omega_maxabs_rad_s':{'value':2e-4,'origin':'LEGACY_old296_velocity-channel rule'},
              'unscaled_object_reward_abs':{'value':2e-5,'origin':'LEGACY_old296; now refk primary rather than silently refk+1'},
              'all3_required':True},
          'clock_abs_s':{'value':2e-6,'origin':'CURRENT_codec/raw timeline contract; raw drift retained'},
          'archival_drift_flags':'same per-channel output tolerances above, separately named; no tuning after observations',
          'real_signal_numerator_floor':{'value':1e-5,'origin':'LEGACY_old296 unscaled reward numerator'},
          'repeat_noise_multiplier':{'value':10.0,'origin':'LEGACY_old296'},
          'noise':'max(baseline3 reward peak-to-peak, max|baseline - fresh donor nextstate reward|)',
          'stable_nonflat':'both epsilon |rplus-rminus| > max(1e-5,10*noise), signs agree, relative slope disagreement <=.5',
          'epsilon_relative_slope_disagreement':{'value':0.5,'origin':'LEGACY_old296'},
          'flat_or_unstable':'report separately; never count as sign-correct or delete from denominators',
          'threshold_retuning_after_result':False},
      'sensor_force_and_time':{
          'authoritative_self15':'native PenSimReader.attach_capture reward_snapshot.contact plus original finger_collision.data.found, copied at original reward compute after4steps and before original Managerforward; no extra sense/query or HO substitution',
          'state58_label':'reward_snapshot.physics and independent rawpostintegration20ms q/qvel transform; save both and check agreement',
          'force_label':'original reward_snapshot.torque and per5ms preintegration actuator/contact solver quantities; before/afterforward records kept separate',
          'per5ms':'full q/qvel/rawclock/control/externalforces/nacon/nefc + original wp contact type/geom/efc association + contact dist/point/frame/force6 with conventions + solverCOM/pose; explicit pre/post indices and times',
          'contact_phase':'native self15/torque at lastsolver~kdt+.015; x1 at kdt+.020; no forward20ms contact substituted for15ms label',
          'sensor_cache':'Keep every original scene.update(.005), which invalidates sensor caches. A naked4directsteps then reader.contact can return restored donor cache and is prohibited. Per5ms before scene.update cache is not authoritative; use reward_snapshot contact as native training label and record direct rawbuffer/cached found with exact readpoint.',
          'force_causality':'force-on-object normal and direct contact moment preserved; no force invented from proximity; no net-wrench causal attribution from asynchronous samples',
          'binary_contact_gradient':'save fresh labels for calibration; no continuous physical derivative of binary contact or force prediction qualification claimed'},
      'capacity':{
          'expected_oneworld_capacities':{'contacts':192,'constraints':4096},
          'no_silent_capacity_change':True,
          'external_host_ledger':'must live outside snapshot traversal and never be restored/cleared',
          'readpoints':['before each snapshot.restore OR all flags into host ledger','after each branch OR all flags into host ledger','donor terminal before any restore','final before close'],
          'reason':'snapshot.restore rewrites Warp overflow array to snapshot value, so a single finalsticky0 cannot qualify priorbranches',
          'required':'every rawflag array finite/readable int and OR0, no reset/truncation, perstep actual maxima within actual capacities; unknown/nonzero invalidates engineering and stops',
          'save':'all rawflag arrays, event/read IDs, branch/phase, shape,dtype,capacity,maxcounts and accumulatedOR with no guessed nworld'},
      'analysis':{
          'objective':'same unscaled native object2+8 terms/radius.1/scales.05/.04 as frozenCPU protocol, at fixed refk primary; refk+1 separate tracking diagnostics',
          'all_models':'evaluate all3WM gradients at the same freshdonor native58/xi46/zero174/actualu before comparing each to shared physical slopes',
          'outputs':['physical rplus/rminus/baseline3 for all slots','3WM directional derivatives and their masked-coordinate norms','finite/domain/effectiveinput/saturation/noise/flat/stable classifications','q/qd/p/R/v/omega response deltas','source archive drifts and fresh replay errors','native self15 label consistency and logits/calibration kept separate'],
          'force_saturation':'record effective PD force saturation and actual ctrl deltas; do not call saturation-induced flatness a contact-gradient failure',
          'statistics':'sign agreement/correlation only for stated stable nonflat slots with complete denominators/coverage and path/phase breakdown. Snapshot/direction rows are dependent; one seen source/seed0, no inferential method-superiority CI/p-value.',
          'no_accuracy_or_sign_success_threshold':True,
          'no_suggested_step_or_policy_improvement_branches':True,
          'qualification':'local one-step sampled interior-coordinate causal agreement only, conditional on complete replay gates; not fullstate/control-face/contact-force/global dynamics/CHORD cross-embodiment/ego/hardware qualification'},
      'cost_and_lifecycle':{
          'new_teacher_cost':'all new donor capture and perturbation/baseline physical transitions are new probe/teacher costs, never charged to original matched training or old quality scores',
          'ledger':['sourcecopy/CPUpreflight','model/context/graph setup','donor replay perpath','snapshot capture/restore','baseline physics','FD physics','WM CPU evaluation','host transfers/serialization/analysis','failed attempts/cleanup'],
          'nested_times_not_added':True,'record_external_process_wall':True,
          'soft_worker_seconds':{'value':220,'status':'NEW_PROPOSED_ENGINEERING_LIMIT'},
          'hard_owned_process_seconds':{'value':240,'status':'NEW_PROPOSED_ENGINEERING_LIMIT'},
          'future_execution':'reviewed sharedGPUlock/reject worker/lifecycle exactOSPID/creation and bounded ownedtree cleanup, no retry; source/code/actualplan must be separately frozen and Root-authorized',
          'quality_failure_not_execution_retry':True},
      'generator':pin(__file__)
    }
    write(OUT/'protocol.json',p)
    write(OUT/'READY.json',{'schema':'same_Warp_FD_proposal_preparation:1','status':p['status'],
                          'proposal':pin(OUT/'protocol.json'),'generator':pin(__file__),
                          'GPU_run_authorized':False,'GPU_used':False,'physics_steps':0,
                          'WM_forward_calls':0,'Torch_imported':'torch' in sys.modules,
                          'native_solver_or_geometry_calls':0,'source_unmodified':True})
    print(json.dumps({'proposal':pin(OUT/'protocol.json'),'READY':pin(OUT/'READY.json')},indent=2))


if __name__=='__main__':main()
