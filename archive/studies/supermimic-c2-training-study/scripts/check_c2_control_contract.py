"""Native C2 command/input contract checks; CPU by default, no training.

CPU fixtures call the actual ActionTerm/PenMDP methods and the shared adapter.
--warp requires separately scheduled GPU authorization. It checks direct target
processing in two real Warp worlds, not physical stepping/replay or WM accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
STUDY = Path('D:/workspace/supermimic-chord-study')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixture_module():
    path = STUDY / 'scripts/check_pen_target_limits.py'
    spec = importlib.util.spec_from_file_location('native_pen_limit_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def consumers(mdp, x, u, context, dynamics, force_limit):
    """Exercise native consumers. This does not execute a learned WM forward."""
    import torch
    from dex_supermimic.rl.dex_supermimic.networks import WorldModel
    wm_input = WorldModel.inputs(None, dynamics, u)
    tau = mdp.pd_torque(x, u, context, limits=force_limit)
    q, qd = x[:, :20], x[:, 20:40]
    manual = (mdp.kp_default * context[:, -40:-20] * (u - q)
              + mdp.kd_default * context[:, -20:] * qd).clamp(-force_limit, force_limit)
    return wm_input, tau, bool(torch.equal(wm_input[:, -20:], u) and torch.equal(tau, manual))


def cpu_checks(model_path, output):
    import numpy as np
    import torch
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.pen import PenMDP
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from dex_supermimic.rl.dex_supermimic.state import pack
    from c2_training_adapter import training_controller
    from target_inset import target_inset
    fixture_module, fixture_path = load_fixture_module()
    model = mujoco.MjModel.from_binary_path(str(model_path))
    ids = np.array([model.joint('robot/' + name).id for name in fixture_module.RIGHT_FINGER_JOINT_NAMES])
    lo, hi = torch.tensor(model.jnt_range[ids], dtype=torch.float32).T
    prepared = []
    # Instantiate the real PenMDP before applying its constructor wrapper: these
    # lightweight fixtures are not real environments and have no env_init hook.
    for group in range(16):
        reference = ((lo + hi) / 2).repeat(2, 1)
        if group % 4 == 0:
            reference[0, group % 20] = hi[group % 20] + .5
        if group % 4 == 1:
            reference[1, group % 20] = lo[group % 20] - .5
        bundle = fixture_module.fixture(model_path, reference, reverse_ctrl_order=group % 2 == 1)
        _, entity, term, mdp, memory = bundle
        term._processed_actions[:] = torch.tensor([.1, -.1])[:, None]
        memory['c1'][:] = term.processed_action
        memory['c2'] = memory['c1'].clone() * .5
        memory['a1'] = torch.zeros_like(memory['c1'])
        entity.data.encoder_bias[:] = torch.linspace(-.02, .02, 20)
        aids = term._position_ctrl_ids.cpu().numpy()
        if not np.all(model.actuator_gear[aids, 0] == 1):
            raise ValueError('Native joint-torque comparison requires unit gear')
        mdp.kp_default = torch.tensor(model.actuator_gainprm[aids, 0], dtype=torch.float32)
        mdp.kd_default = torch.tensor(model.actuator_biasprm[aids, 2], dtype=torch.float32)
        mdp.force_limit = torch.tensor(np.abs(model.actuator_forcerange[aids]).max(-1), dtype=torch.float32)
        action = torch.linspace(-.7, .7, 20).repeat(2, 1)
        action[1] *= -1
        if group % 4 < 2:
            action[:] = 8. if group % 2 else -8.
        prepared.append((bundle, action))
    # Separate interior and saturated components for a meaningful derivative.
    reference = ((lo + hi) / 2)[None]
    reference[0, 0] = hi[0] + .5
    gradient_fixture = fixture_module.fixture(model_path, reference)
    arrays, rows = {}, []
    rng_before = torch.get_rng_state().clone()
    with training_controller(ManagerBasedRlEnv, PenMDP, DexSuperMimicRunner,
                             {'base_method': 'CPU_NATIVE_METHOD_CONTRACT', 'record_actual_u': False}):
        for group, (bundle, action) in enumerate(prepared):
            _, entity, term, mdp, memory = bundle
            old_c1, old_c2 = memory['c1'].clone(), memory['c2'].clone()
            residual, imagined_u = mdp.control_target(action, memory, entity.data.encoder_bias)
            # Documented insertion after the real native process method. The
            # actual environment-hook execution is checked separately by Warp.
            term.process_actions(action)
            proposed = term._targets.clone()
            low, high = term.target_position_limits
            term._targets.copy_(target_inset(proposed, low, high, .020))
            for _ in range(4):
                term.apply_actions()
            actual_u = entity.last_target
            x = pack(imagined_u.detach(), torch.zeros_like(actual_u), torch.zeros(2, 3),
                     torch.eye(3)[None].repeat(2, 1, 1), torch.zeros(2, 3), torch.zeros(2, 3))
            context = torch.ones(2, 46)
            wm_input, tau, consumer_ok = consumers(mdp, x, imagined_u, context, torch.zeros(2, 278), mdp.force_limit)
            advanced = DexSuperMimicRunner.advance_memory(SimpleNamespace(mdp=mdp), x, memory,
                         action, residual, torch.zeros_like(actual_u), torch.eye(3)[None].repeat(2, 1, 1))
            checks = {
                'real_imagined_target_exact': torch.equal(actual_u, imagined_u),
                'native_residual_preserved': torch.equal(residual, term.processed_action),
                'prior_c1_c2_unchanged': torch.equal(memory['c1'], old_c1) and torch.equal(memory['c2'], old_c2),
                'frozen_target_filtered': torch.equal(memory['frozen_target'], imagined_u),
                'advance_memory_uses_residual': torch.equal(advanced['c1'], residual) and torch.equal(advanced['c2'], old_c1),
                'inside_inset': bool(((actual_u >= low + .020) & (actual_u <= high - .020)).all()),
                'WM_input_and_native_PD_use_same_u': consumer_ok,
                'FP32': actual_u.dtype == imagined_u.dtype == wm_input.dtype == tau.dtype == torch.float32,
            }
            rows.append({'group': group, 'envs': 2, 'checks': checks, 'passed': all(checks.values())})
            for name, value in {'action': action, 'proposed': proposed, 'actual_u': actual_u,
                                'imagined_u': imagined_u, 'residual': residual, 'wm_input': wm_input,
                                'native_PD_torque': tau, 'lower': low, 'upper': high}.items():
                arrays[f'group{group:02d}_{name}'] = value.detach().cpu().numpy()
        _, entity, _, mdp, memory = gradient_fixture
        action = torch.zeros(1, 20, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            _, u = mdp.control_target(action, memory, entity.data.encoder_bias)
        u.sum().backward()
        expected_gradient = torch.full_like(action, .5 * mdp.alpha)
        expected_gradient[:, 0] = 0
        gradient_ok = bool(torch.equal(action.grad, expected_gradient) and u.dtype == torch.float32)
        arrays['gradient_action'] = action.detach().numpy()
        arrays['gradient_actual'] = action.grad.numpy()
        arrays['gradient_expected'] = expected_gradient.numpy()
    np.savez_compressed(output / 'cpu_cases.npz', **arrays)
    return {'passed': all(row['passed'] for row in rows) and gradient_ok and torch.equal(rng_before, torch.get_rng_state()),
            'scope': 'Actual native ActionTerm/PenMDP/advance_memory/WorldModel.inputs/PD methods on MJB-backed CPU fixtures; not live env.step, replay, learned WM accuracy or physical qualification.',
            'groups': rows, 'gradient_interior_and_saturated_under_outerAMP': gradient_ok,
            'CPU_RNG_unchanged_by_checks': bool(torch.equal(rng_before, torch.get_rng_state())),
            'fixture_helper': str(fixture_path), 'fixture_helper_sha256': sha(fixture_path),
            'native_actor_control_module': inspect.getfile(PenMDP), 'native_actor_control_sha256': sha(inspect.getfile(PenMDP)),
            'cuda_initialized': torch.cuda.is_initialized(), 'cases_sha256': sha(output / 'cpu_cases.npz')}


def warp_checks(recipe, clips_root, output):
    import numpy as np
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.rl.dex_supermimic.pen import PenMDP, PenSimReader, PenObservations
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from c2_training_adapter import training_controller
    wp.set_device('cuda:0')
    cfg = make_env_cfg(num_envs=2, play=True, recipe_path=recipe, clips_root=clips_root)
    cfg.seed = 0; cfg.auto_reset = False
    cfg.commands['motion'].first_frame_hold_prob = 0.
    cfg.commands['motion'].debug_vis = False
    cfg.actions['joint_pos'].mask_prob = 0.
    rows, arrays = [], {}
    with training_controller(ManagerBasedRlEnv, PenMDP, DexSuperMimicRunner,
                             {'base_method': 'CUDA_NATIVE_COMMAND_CONTRACT', 'record_actual_u': False}):
        env = ManagerBasedRlEnv(cfg, device='cuda:0')
        try:
            env.reset(seed=0); reader = PenSimReader(env)
            motion = env.command_manager.get_term('motion'); term = env.action_manager.get_term('joint_pos')
            initial_q = env.sim.wp_data.qpos.numpy().copy()
            initial_v = env.sim.wp_data.qvel.numpy().copy()
            initial_time = env.sim.wp_data.time.numpy().copy()
            for group in range(16):
                if time.perf_counter() - STARTED > 110:
                    raise TimeoutError('110s soft command-probe budget; outer watchdog must enforce120s')
                motion.time_steps[:] = group
                action = torch.linspace(-.7, .7, 20, device=env.device).repeat(2, 1)
                action[1] *= -1
                if group % 4 < 2:
                    action[:] = 8. if group % 2 else -8.
                memory = reader.memory(); prior_residual = term.processed_action.clone()
                residual, expected = reader.mdp.control_target(action, memory, reader.robot.data.encoder_bias)
                # This is direct command processing, not an env.step. Explicitly
                # activate only the same adapter gate for this narrow test.
                env._c2_action_active = True
                try:
                    env.action_manager.process_action(action)
                finally:
                    env._c2_action_active = False
                for _ in range(4):
                    env.action_manager.apply_action(); env.scene.write_data_to_sim()
                cached = term._targets.clone()
                actual = wp.to_torch(env.sim.wp_data.ctrl)[:, term._position_ctrl_ids].clone()
                x = reader.state(); dynamics = PenObservations.dynamics(x, reader.xi, x.new_zeros(2, 174))
                wm_input, tau, consumer_ok = consumers(reader.mdp, x, expected, reader.xi, dynamics, reader.torque_limits())
                low, high = term.target_position_limits
                checks = {'cached_imagined_actual_exact': torch.equal(cached, expected) and torch.equal(actual, expected),
                          'residual_exact': torch.equal(residual, term.processed_action),
                          'frozen_target_exact': torch.equal(memory['frozen_target'], expected),
                          'inside_inset': bool(((actual >= low + .020) & (actual <= high - .020)).all()),
                          'WM_input_native_PD_same_u': consumer_ok}
                rows.append({'group': group, 'phase': group, 'checks': checks, 'passed': all(checks.values())})
                for name, value in {'action': action, 'prior_residual': prior_residual, 'residual': residual,
                                    'cached': cached, 'actual_u': actual, 'imagined_u': expected,
                                    'wm_input': wm_input, 'native_PD_torque': tau}.items():
                    arrays[f'group{group:02d}_{name}'] = value.detach().cpu().numpy()
            unchanged = all(np.array_equal(before, after) for before, after in [
                (initial_q, env.sim.wp_data.qpos.numpy()), (initial_v, env.sim.wp_data.qvel.numpy()),
                (initial_time, env.sim.wp_data.time.numpy())])
            np.savez_compressed(output / 'warp_command_cases.npz', **arrays)
            return {'passed': all(r['passed'] for r in rows) and unchanged,
                    'scope': 'Two genuine Warp worlds,16 predefined phase/action command groups. Direct process/apply only after reset: no env.step, no physics/replay rollout, no learned WM forward/H8/provider/accuracy qualification.',
                    'groups': rows, 'qpos_qvel_clock_unchanged_after_reset': unchanged,
                    'constructor_scope': 'Native CUDA graph setup is backend initialization, not a validated physical rollout.',
                    'cases_sha256': sha(output / 'warp_command_cases.npz')}
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=STUDY / 'outputs/rotating_reference_warp_a0/model.mjb')
    parser.add_argument('--recipe', type=Path, default=STUDY / 'outputs/rotating_reference_selection/selected_recipe.json')
    parser.add_argument('--clips-root', type=Path, default=STUDY / 'outputs/rotating_reference_selection/selected_catalog')
    parser.add_argument('--warp', action='store_true')
    args = parser.parse_args()
    if not args.warp:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path.insert(0, str(ROOT / 'src'))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'model_sha256': sha(args.model), 'script_sha256': sha(__file__),
              'adapter_sha256': sha(ROOT / 'src/c2_training_adapter.py'), 'pure_projection_sha256': sha(ROOT / 'src/target_inset.py'),
              'GPU_requested': args.warp, 'training_updates': 0, 'physical_qualification': False}
    try:
        report['cpu'] = cpu_checks(args.model, args.output)
        if not report['cpu']['passed']:
            raise AssertionError('CPU native C2 command/input contract failed')
        if args.warp:
            report['warp'] = warp_checks(args.recipe, args.clips_root, args.output)
            if not report['warp']['passed']:
                raise AssertionError('Warp command contract failed')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['wall_seconds_including_imports'] = time.perf_counter() - STARTED
        (args.output / 'checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({k: report[k] for k in ['status', 'GPU_requested', 'physical_qualification', 'wall_seconds_including_imports']}))


if __name__ == '__main__':
    main()
