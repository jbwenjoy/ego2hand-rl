"""Regression checks for actual pen action vs imagined absolute-target limits.

Default: CPU-only fixtures execute the real ActionTerm and PenMDP methods using
the frozen task's actual MJB ranges/transmissions. --warp additionally verifies
the same contract inside a genuine CUDA environment; schedule GPU separately.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from dex_supermimic.tasks.pen_spin.mdp.actions import SingleHandResidualEMAAction, SingleHandResidualEMAActionCfg
from dex_supermimic.tasks.pen_spin.config.wuji_hand2.hand2_constants import RIGHT_FINGER_JOINT_NAMES, RIGHT_HAND_LINK_NAMES
from dex_supermimic.rl.dex_supermimic.pen import PenMDP

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(model_path, reference, *, ctrl_ranges=None, ctrl_limited=None, reverse_ctrl_order=False, mask=False):
    model = mujoco.MjModel.from_binary_path(str(model_path))
    jids = np.asarray([model.joint("robot/" + name).id for name in RIGHT_FINGER_JOINT_NAMES])
    aids = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == jid).item() for jid in jids])
    if ctrl_ranges is not None:
        model.actuator_ctrlrange[aids] = ctrl_ranges
    if ctrl_limited is not None:
        model.actuator_ctrllimited[aids] = ctrl_limited
    reference = torch.as_tensor(reference, dtype=torch.float32)
    n = len(reference)
    hard = torch.tensor(model.jnt_range[jids], dtype=torch.float32)
    hard[~torch.tensor(model.jnt_limited[jids], dtype=torch.bool)] = torch.tensor([-torch.inf, torch.inf])

    class Entity:
        joint_names = list(RIGHT_FINGER_JOINT_NAMES)
        indexing = SimpleNamespace(joint_ids=torch.tensor(jids), ctrl_ids=torch.tensor(aids[::-1].copy() if reverse_ctrl_order else aids))
        data = SimpleNamespace(joint_pos_limits=hard[None].repeat(n, 1, 1), default_joint_pos=torch.zeros(n, 20), encoder_bias=torch.zeros(n, 20))
        last_target = None
        def find_joints_by_actuator_names(self, patterns):
            return list(range(20)), self.joint_names
        def set_joint_position_target(self, target, joint_ids):
            self.last_target = target.clone()

    entity = Entity()
    frames = reference.repeat_interleave(2, dim=0)
    motion = SimpleNamespace(
        joint_names=RIGHT_FINGER_JOINT_NAMES, joint_pos=reference.clone(),
        cfg=SimpleNamespace(link_names=RIGHT_HAND_LINK_NAMES, axis_point_radius_m=.1, object_position_tanh_scale_m=.15),
        _frames={"joint_pos": frames, "object_pose": torch.zeros(2*n, 7), "link_pos": torch.zeros(2*n, 21, 3)},
        group=SimpleNamespace(trajectory_end=torch.arange(2, 2*n+1, 2)),
    )
    cfg = SingleHandResidualEMAActionCfg(entity_name="robot", actuator_names=("right_finger.*_joint.*",), mask_prob=.5 if mask else 0.)
    env = SimpleNamespace(num_envs=n, device="cpu", scene={"robot": entity},
        sim=SimpleNamespace(mj_model=model, model=SimpleNamespace(actuator_ctrlrange=torch.tensor(model.actuator_ctrlrange, dtype=torch.float32)[None].repeat(n, 1, 1))),
        command_manager=SimpleNamespace(get_term=lambda name: motion),
        cfg=SimpleNamespace(actions={"joint_pos": cfg}, observations={"policy": SimpleNamespace(enable_corruption=False)}))
    action = SingleHandResidualEMAAction(cfg, env)
    env.action_manager = SimpleNamespace(get_term=lambda name: action)
    kin = SimpleNamespace(names=[model.body(i).name for i in range(model.nbody)])
    imagined = PenMDP(SimpleNamespace(env=env, robot=entity, kin=kin))
    memory = {"c1": torch.zeros(n, 20), "start": 2*torch.arange(n)[:, None].float(),
              "time": torch.zeros(n, 1), "end": 2*torch.arange(n)[:, None].float()+2,
              "age": torch.ones(n, 1), "mask": torch.zeros(n, 20), "mask_remaining": torch.zeros(n, 1),
              "mask_active": torch.zeros(n, 1), "frozen_target": torch.zeros(n, 20)}
    return model, entity, action, imagined, memory


def cpu_checks(model_path):
    class Checks(unittest.TestCase):
        def test_reported_pinky_upper_limit_and_ema_memory(self):
            refs = torch.zeros(3, 20); refs[:, 16] = 1.75234
            model, entity, actual, imagined, memory = fixture(model_path, refs)
            actions = torch.ones(3, 20)
            residual, target = imagined.control_target(actions, memory, entity.data.encoder_bias)
            actual.process_actions(actions); actual.apply_actions()
            self.assertTrue(torch.equal(target, entity.last_target))
            self.assertTrue(torch.equal(residual, actual.processed_action))
            self.assertTrue(torch.equal(residual, torch.full_like(residual, .25)))
            self.assertAlmostEqual(float(target[0, 16]), 1.57, places=6)
            self.assertTrue(torch.equal(memory["frozen_target"], target))
            self.assertGreater(float(refs[0, 16]+residual[0, 16]), 2.)

        def test_upper_lower_extremes_and_nonzero_encoder_bias(self):
            refs = torch.stack((torch.full((20,), 5.), torch.full((20,), -5.)))
            model, entity, actual, imagined, memory = fixture(model_path, refs)
            entity.data.encoder_bias[:] = torch.linspace(-.2, .2, 20)
            actions = torch.stack((torch.full((20,), 8.), torch.full((20,), -8.)))
            memory["c1"][:] = .1; actual._processed_actions[:] = .1
            residual, target = imagined.control_target(actions, memory, entity.data.encoder_bias)
            actual.process_actions(actions); actual.apply_actions()
            self.assertTrue(torch.equal(target, entity.last_target))
            self.assertTrue(torch.equal(residual, actual.processed_action))
            lo, hi = actual.target_position_limits
            self.assertTrue(bool(((target >= lo) & (target <= hi)).all()))

        def test_stricter_actuator_range_and_reversed_actuator_order(self):
            base = mujoco.MjModel.from_binary_path(str(model_path))
            ranges = base.actuator_ctrlrange.copy(); ranges[0] = [-.1, .2]
            _, entity, actual, imagined, memory = fixture(model_path, torch.ones(2, 20), ctrl_ranges=ranges, reverse_ctrl_order=True)
            _, target = imagined.control_target(torch.zeros(2, 20), memory, entity.data.encoder_bias)
            actual.process_actions(torch.zeros(2, 20)); actual.apply_actions()
            self.assertTrue(torch.equal(target, entity.last_target))
            self.assertAlmostEqual(float(target[0, 0]), .2, places=6)

        def test_disabled_ctrl_limit_does_not_create_zero_target_range(self):
            base = mujoco.MjModel.from_binary_path(str(model_path))
            ranges = base.actuator_ctrlrange.copy(); ranges[0] = [0., 0.]
            limited = base.actuator_ctrllimited.copy(); limited[0] = 0
            _, entity, actual, imagined, memory = fixture(model_path, torch.full((1,20), .4), ctrl_ranges=ranges, ctrl_limited=limited)
            _, target = imagined.control_target(torch.zeros(1,20), memory, entity.data.encoder_bias)
            actual.process_actions(torch.zeros(1,20)); actual.apply_actions()
            self.assertTrue(torch.equal(target, entity.last_target))
            self.assertAlmostEqual(float(target[0,0]), .4, places=6)

        def test_mask_freezes_clamped_target_and_legacy_target_is_repaired(self):
            _, entity, actual, imagined, memory = fixture(model_path, torch.zeros(1,20), mask=True)
            actual._targets_valid[:] = True; actual._targets[:] = 4.
            actual._mask_active[:] = True; actual._mask_remaining[:] = 3; actual._mask_dofs[:] = True
            memory.update(frozen_target=torch.full((1,20),4.), mask=torch.ones(1,20), mask_active=torch.ones(1,1), mask_remaining=torch.full((1,1),3.))
            torch.manual_seed(9); actual.process_actions(torch.zeros(1,20)); actual.apply_actions()
            torch.manual_seed(9); residual, target = imagined.control_target(torch.zeros(1,20), memory, entity.data.encoder_bias)
            self.assertTrue(torch.equal(target, entity.last_target))
            self.assertTrue(torch.equal(memory["frozen_target"], target))
            self.assertTrue(torch.equal(residual, actual.processed_action))
            self.assertLess(float(target.max()), 4.)

        def test_direct_initialization_apply_is_bounded(self):
            _, entity, actual, _, _ = fixture(model_path, torch.zeros(1,20))
            actual._targets[:] = 4.; actual.apply_actions()
            lo, hi = actual.target_position_limits
            self.assertTrue(bool(((entity.last_target >= lo) & (entity.last_target <= hi)).all()))

        def test_clamp_has_zero_gradient_outside_and_preserves_interior_gradient(self):
            refs = torch.zeros(1,20); refs[:,16] = 3.
            _, entity, _, imagined, memory = fixture(model_path, refs)
            actions = torch.zeros(1,20,requires_grad=True)
            _, target = imagined.control_target(actions, memory, entity.data.encoder_bias)
            target.sum().backward()
            self.assertEqual(float(actions.grad[0,16]), 0.)
            self.assertAlmostEqual(float(actions.grad[0,0]), .25, places=7)

        def test_empty_intersection_is_rejected(self):
            model = mujoco.MjModel.from_binary_path(str(model_path))
            ranges = model.actuator_ctrlrange.copy(); ranges[0] = [10.,11.]
            with self.assertRaisesRegex(ValueError,"Empty"):
                fixture(model_path, torch.zeros(1,20), ctrl_ranges=ranges)

        def test_native_mujoco_implicit_ctrl_clip_matches_explicit_limit_one_step(self):
            model, _, action, _, _ = fixture(model_path, torch.zeros(1,20))
            raw = torch.zeros(1,20); raw[:,16] = 2.00234
            bounded = action.limit_target(raw).numpy()[0]
            before = mujoco.MjData(model); after = mujoco.MjData(model)
            before.ctrl[:] = raw.numpy()[0]; after.ctrl[:] = bounded
            mujoco.mj_step(model, before); mujoco.mj_step(model, after)
            self.assertTrue(np.array_equal(before.qpos, after.qpos))
            self.assertTrue(np.array_equal(before.qvel, after.qvel))
            self.assertTrue(np.array_equal(before.actuator_force, after.actuator_force))

    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return {"scope":"CPU real ActionTerm/PenMDP methods with actual MJB transmission/range fixtures; not a live Warp environment",
            "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
            "passed":result.wasSuccessful(),"log":stream.getvalue()}


def warp_checks(recipe, clips_root):
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.rl.dex_supermimic.pen import PenSimReader
    wp.set_device("cuda:0")
    cfg=make_env_cfg(num_envs=2,play=True,recipe_path=recipe,clips_root=clips_root)
    cfg.seed=0; cfg.auto_reset=False
    cfg.commands["motion"].first_frame_hold_prob=0.; cfg.commands["motion"].debug_vis=False
    env=ManagerBasedRlEnv(cfg,device="cuda:0")
    try:
        env.reset(seed=0); reader=PenSimReader(env)
        motion=env.command_manager.get_term("motion"); term=env.action_manager.get_term("joint_pos")
        lo,hi=term.target_position_limits
        errors=[]; clipped=0
        for k in range(16):
            motion.time_steps[:]=k
            action=torch.full((2,20), 8. if k%2 else -8.,device=env.device)
            memory=reader.memory()
            raw=reader.mdp.reference(memory)[0]+reader.mdp.alpha*(.5*action).clamp(-.6,.6)+(1-reader.mdp.alpha)*memory["c1"]-reader.robot.data.encoder_bias
            _,expected=reader.mdp.control_target(action,memory,reader.robot.data.encoder_bias)
            env.action_manager.process_action(action); env.action_manager.apply_action(); env.scene.write_data_to_sim()
            actual=term._targets
            errors.append(float((expected-actual).abs().max()))
            clipped+=int(((raw<lo)|(raw>hi)).sum())
            assert torch.equal(expected,actual)
            assert bool(((actual>=lo)&(actual<=hi)).all())
            assert torch.equal(env.sim.data.ctrl[:,term._position_ctrl_ids],actual)
        return {"passed":True,"scope":"genuine Warp CUDA environment target processing/application only; no physics rollout",
                "samples":16,"envs":2,"max_target_difference_rad":max(errors),"out_of_range_input_components":clipped}
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=ROOT/'outputs/rotating_reference_warp_a0/model.mjb')
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--warp",action="store_true")
    parser.add_argument("--recipe",type=Path,default=ROOT/'outputs/rotating_reference_selection/selected_recipe.json')
    parser.add_argument("--clips-root",type=Path,default=ROOT/'outputs/rotating_reference_selection/selected_catalog')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    result={"model_sha256":sha(args.model),"script_sha256":sha(__file__),"cpu":cpu_checks(args.model)}
    if args.warp:
        result["warp"]=warp_checks(args.recipe,args.clips_root)
    (args.output/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['cpu']['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
