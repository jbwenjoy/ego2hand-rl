"""Source/JSON-only design receipt. Does not import any studied module."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
OUT = ROOT / 'outputs/full151_mode_static_design_v1'
ROUGH = Path('D:/workspace/supermimic-rough-prefix-learning-study')
DEX = Path('D:/workspace/dex-supermimic-research')
SHARED = Path('D:/workspace/supermimic-shared-reference-study')

FILES = {
    'gap_document': ROUGH / 'docs/full151_integration_gap_current104_v1.md',
    'gap_report': ROUGH / 'outputs/full151_integration_gap_readonly_v1/report.json',
    'bundle104': ROUGH / 'src/rough_prefix/bundle_v2.py',
    'controller104': ROUGH / 'src/rough_prefix/controller.py',
    'runtime104': ROUGH / 'src/rough_prefix/runtime_v2.py',
    'actor_loader104': ROUGH / 'src/rough_prefix/checkpoint_v2.py',
    'WM_loader104': ROUGH / 'src/rough_prefix/world_model_checkpoint_v2.py',
    'native104': ROUGH / 'src/rough_prefix/native_v2.py',
    'training104': ROUGH / 'src/rough_prefix/training.py',
    'capacity104': ROUGH / 'src/rough_prefix/capacity.py',
    'probe104': ROUGH / 'src/rough_prefix/probe.py',
    'policy_probe104': ROUGH / 'src/rough_prefix/policy_probe.py',
    'contact_binding104': ROUGH / 'src/rough_prefix/contact_binding.py',
    'health104': ROUGH / 'scripts/audit_rough_checkpoint_cpu_v3.py',
    'native_pen': DEX / 'src/dex_supermimic/rl/dex_supermimic/pen.py',
    'native_runner': DEX / 'src/dex_supermimic/rl/dex_supermimic/runner.py',
    'native_commands': DEX / 'src/dex_supermimic/tasks/pen_spin/mdp/commands.py',
    'native_actions': DEX / 'src/dex_supermimic/tasks/pen_spin/mdp/actions.py',
    'native_terminations': DEX / 'src/dex_supermimic/tasks/pen_spin/mdp/terminations.py',
    'native_names': DEX / 'src/dex_supermimic/tasks/pen_spin/config/wuji_hand2/hand2_constants.py',
    'Manager': DEX / '.venv/Lib/site-packages/mjlab/envs/manager_based_rl_env.py',
    'native_ABI_exporter': SHARED / 'src/native_reference_abi.py',
    'native_ABI_report': SHARED / 'outputs/native_ABI_cpu_v1/report.json',
    'shared_registry': SHARED / 'src/shared_reference/reference_registry.py',
    'shared_target_registry': SHARED / 'src/shared_reference/target_registry.py',
    'coordinate_review': ROOT / 'outputs/full151_coordinate_intent_source_review_v1/report.json',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def index(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    return {node.name: {'line': node.lineno, 'end_line': node.end_lineno}
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'report.json'
    if target.exists():
        raise FileExistsError('Keep executed design receipts immutable')
    before = {key: pin(path) for key, path in FILES.items()}
    text = {key: path.read_text(encoding='utf-8-sig') for key, path in FILES.items()}
    abi = json.loads(text['native_ABI_report'])
    coordinate = json.loads(text['coordinate_review'])
    native_pin = abi['native_reference']
    assertions = {
        'native_ABI_report_is_151_CPU_only_and_physically_unqualified':
            abi['status'] == 'PASS_NATIVE_READER_ABI_CPU_ONLY'
            and abi['checks']['trajectory_length151'] is True
            and abi['physical_reference_qualified'] is False,
        'native151_artifact_bytes_match_saved_ABI_report':
            sha(native_pin['path']) == native_pin['sha256'],
        'new_coordinate_review_declares_legacy_200_351_not_shared_prefix':
            coordinate['observation_contract']['source_window_half_open'] == [200, 351]
            and coordinate['observation_contract']['frames'] == 151,
        'old_controller_real_phase15_is_rejected':
            '(time < 15).all()' in text['controller104'],
        'old_controller_has_finite_inactive_u14_and_version104':
            'phase.clamp(max=14)' in text['controller104']
            and 'FORMAT_VERSION = 104' in text['controller104'],
        'old_runtime_end_is_pre_advance_phase14':
            'return motion.time_steps>=14' in text['runtime104'],
        'old_runtime_has_selected_history_backfill_and_generation':
            'compute(update_history=True,env_ids=ids)' in text['runtime104']
            and 'env._rough_runtime_episode_generation[ids]+=1' in text['runtime104'],
        'old_runtime_has_reference_end_plus2_and_dead_loss_norm_masks':
            'm["time"] + 2 >= m["end"] - m["start"]' in text['runtime104']
            and 'inputs[rough_norm_valid]' in text['runtime104']
            and '_rough_masked_critic_mse' in text['runtime104'],
        'old_loader_strict104_and_baseline15_remain_unchanged':
            "state['format_version'] == 104" in text['actor_loader104']
            and "'baseline_shape': [15, 20]" in text['actor_loader104'],
        'native_reader_produces_FD_goals_and_accepts_per_clip_frame_count':
            'frames = joint_pos.shape[0]' in text['native_commands']
            and '"joint_vel": _diff(joint_pos, frame_dt)' in text['native_commands'],
        'native_reference_uses_memory_bounds_not_fixed16':
            '(memory["start"] + memory["time"].clamp_min(0)).minimum(memory["end"] - 1)' in text['native_pen'],
        'native_replay_u_is_executed_absolute_target_and_c_residual':
            'u = self.reader.executed_target() if self.is_pen else c' in text['native_runner']
            and 'c = self.env.action_manager.get_term("joint_pos").processed_action.clone()' in text['native_runner'],
        'Manager_source_termination_precedes_original_forward_command_and_obs':
            text['Manager'].index('self.reset_buf = self.termination_manager.compute()')
            < text['Manager'].index('self.sim.forward()', text['Manager'].index('self.reset_buf = self.termination_manager.compute()'))
            < text['Manager'].index('self.command_manager.compute(dt=self.step_dt)')
            < text['Manager'].index('self.obs_buf = self.observation_manager.compute(update_history=True)'),
        'native_BODY21_exporter_does_not_alias_macro_points':
            "f'right_finger{finger}_link{link}' for finger in range(1, 6) for link in range(1, 5)" in text['native_ABI_exporter']
            and 'mujoco.mj_kinematics(model, data)' in text['native_ABI_exporter'],
    }
    # This is an explicit parameter proposal, not an implemented runtime consumer.
    horizon = {
        'reference_states': 151, 'control_intervals': 150,
        'local_reference_range_half_open': [0, 151],
        'source_reference_range_half_open': [200, 351],
        'active_real_action_phases_half_open': [0, 150],
        'last_action_phase': 149, 'terminal_observation_phase': 150,
        'source_preaction_frames_half_open': [200, 350],
        'source_terminal_observation_frame': 350,
        'dead_imagination_placeholder_baseline_index': 149,
        'baseline_shape': [150, 20], 'control_dt_s': 0.02,
        'physics_dt_s': 0.005, 'decimation': 4, 'duration_s': 3.0,
        'planned_full_eval_solver_steps': 600, 'planned_full_eval_raw_states': 601,
        'planned_full_eval_native_postinterval_forwards': 150,
        'constructor_and_reset_steps_forwards': 'separately instrument; not included above',
        'imagined_horizon': 8,
    }
    negative_cases = [
        ('identity_old_format', '3,103,104 checkpoints rejected by new mode; new105 rejected by old loaders'),
        ('identity_relabel', '104 payload with only format->105 rejected by kind/root/baseline/source/horizon signatures'),
        ('identity_window', 'shared007[0,151), legacy prefix[200,216), and legacy full[200,351) remain distinct'),
        ('identity_variant', 'sameN151 but different q/velocity/init/u/native BODY21/recipe/objective/context SHA rejected'),
        ('horizon_consistency', 'N!=151/M!=N-1/terminal!=M/baseline!=[M,20] rejected before construction'),
        ('bad_time', 'nonfinite/noninteger/negative phase rejected; real phase150 and>=151 rejected'),
        ('last_interval', '149 executes exactly four substeps then done+timeout with terminalobs150; no u150 access'),
        ('imaginary_boundary', 'H8 from142/148/149/150 has active counts8/2/1/0; inactiveu149 finite, zero survival'),
        ('dead_critic', 'perturb finite dead state/reward/critic by1000: active norm/loss/grad unchanged; NaN still rejected'),
        ('replay_boundary', '149 terminal transition retained; reset suffix masked; generation change without done rejected'),
        ('metadata_types', 'bool-as-int/float151/malformed generation/unknown schema and duplicate keys rejected'),
        ('selected_reset', 'selected rows only; q/qd/pose/native twist/u0/EMA/action/time/obs history readback; no duplicateids'),
        ('velocity_authority', 'nativeFD substituted for declared analytic/design table rejected; wrongworld/bodyomega rejected'),
        ('controller_memory', 'c1/c2 contain residual only; u0-q0 never inserted; baseline selected by current phase only'),
        ('single_C2', 'zero residual recovers baseline when all150 plannedtargets alreadyC2identity; saturation/interior FP32/AMP/grad'),
        ('real_imag_parity', 'samephase/action/residual -> sameabsoluteu; cache/rawctrl/replayu/PD/WM all exact; c remains residual'),
        ('target_lawful', 'outsidehardintersectctrl/C2, malformedu150, missingu149 rejected; no silent clip/extend/repair'),
        ('source_labels', 'originalsourceq/wrist/link/contact or macroFK-as-nativeBODY21 injection rejected'),
        ('contact_independence', 'old16/contactbundle/policy-derivedcontact target rejected; unknown != measuredabsence'),
        ('save_and_restore', 'native atomic save wrapper adds only newmetadata/version; all class/functions restore on exception'),
        ('fromscratch', 'any resume/modelonly/oldWM/oldActor warmstart rejected unless a separate future transfer protocol exists'),
        ('capacity', 'raw stickyoverflow nonzero/unknown before any reset or final blocks engineering; prefix peek not final'),
        ('eval_full_horizon', 'qualityfail latch continues to150; hardnumericalstop censored, missingsteps cannot bepass'),
        ('budget', 'exact realtransitions/optimizer counts/validmask denominators, initialization+reset cost separately recorded'),
    ]
    report = {
        'schema': 'full151_mode_static_design:1',
        'status': 'COMPLETE_STATIC_DRAFT_IMPLEMENTATION_AND_AUTHORIZATION_PENDING',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_pins': before,
        'document': pin(ROOT / 'docs/full151_mode_contract_draft_v1.md'),
        'source_AST_symbols': {key: index(path) for key, path in FILES.items() if path.suffix == '.py'},
        'source_text_checks': assertions,
        'all_static_checks_pass': all(assertions.values()),
        'existing_native151_artifact': native_pin,
        'existing_reader_qualification': 'Saved CPU ABI report only; not reexecuted, not physical/runtime/training qualification',
        'proposed_horizon_not_runtime_code': horizon,
        'checkpoint_version': {'candidate': 105, 'status': 'PROPOSED_NOT_RESERVED_OR_IMPLEMENTED',
            'reason': 'Distinct controller/reference/horizon contract; no old checkpoint migration',
            'search_scope': 'Local current rough-prefix/C2/full151 source only; no global uniqueness guarantee'},
        'required_common_descriptor_fields': [
            'mode', 'retarget_variant_id', 'source_window_half_open', 'observation_lineage', 'native_recipe_and_arrays_pins',
            'input_bundle_pin_and_signature', 'N_M_dt_decimation', 'default_fixed_palm_and_native_BODY_joint_orders',
            'qgoal_velocity_and_object_twist_authority', 'baseline_u150_FP32_rawbytes_sha256', 'requested_init_allfields',
            'hard_intersect_ctrl_bounds_and_C2_source_margin', 'residual_EMA_semantics',
            'reward_phase_k_eval_phase_kplus1', 'reset_terminal_replay_masks_and_generation',
            'actual8_context_MJB_and_model_resources', 'contact_target_objective_pins_or_explicit_methodNone',
            'physical_reference_qualified_false', 'training_authorization_separate', 'shared10_enrollment_false',
            'fromscratch_only_no_resume', 'source_inventory',
        ],
        'CPU_negative_checks_to_implement_not_executed': [{'id': k, 'required_behavior': v} for k, v in negative_cases],
        'core_math_must_remain': {
            'original_residual_EMA': 'c=.5*clamp(.5*a,-.6,.6)+.5*c1; c1/c2 residual only',
            'absolute_target': 'C2_020(clamp(baseline_u[phase]+c-bias0,hard_intersect_ctrl)) once',
            'memory_frozen_target': 'actual filtered absoluteu; not residual or unfiltered target',
            'no_imagination_live_state_or_RNG_additions': True,
            'native_state_dim': 58, 'actor_input_output': [155, 20], 'physical_xi_dim': 46,
            'disturbance_dim': 174, 'WM_input_dim': 298, 'WM_delta_dim': 52,
            'self_collision_head_dim': 15,
            'note': 'Dimension15 is self-collision ABI, not old15interval horizon. No global number replacement.'},
        'qualification': {
            'data_input_finalized': False, 'native_full151_new_variant_reader_executed': False,
            'runtime_implemented': False, 'controller_implemented': False, 'loader_implemented': False,
            'physical_reference_qualified': False, 'training_authorized': False,
            'GPU_run_authorized': False, 'shared10_enrollment': False,
            'independent_ego_N': 0, 'source_lineage_N': 1,
        },
        'execution_scope': {
            'source_JSON_read_and_hash_AST_only': True, 'NPZ_arrays_loaded': False,
            'native_reader_rerun': False, 'FK_collision_query': False, 'Env_physics_GPU_WM': False,
            'Torch_Warp_MuJoCo_imported': any(x in sys.modules for x in ('torch', 'warp', 'mujoco')),
            'old_files_written': False, 'new_consumers_encoded': False,
        },
        'receipt_source': pin(__file__),
    }
    after = {key: pin(path) for key, path in FILES.items()}
    report['all_read_sources_unchanged_pre_post'] = before == after
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(target), 'checks': len(assertions), 'all_pass': all(assertions.values()), 'sources': len(before), 'framework_imported': report['execution_scope']['Torch_Warp_MuJoCo_imported']}, ensure_ascii=False))
    if not all(assertions.values()) or before != after:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
