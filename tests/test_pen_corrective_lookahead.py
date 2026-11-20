import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_research.mechanics.pen_corrective_lookahead import STATE, score_candidates


def fixture(pad_name, com_x=0.):
    # Put all twenty hinge joints before the free joint to exercise address lookup.
    joints = ''.join(f'<body pos="{1+i} 0 2"><joint name="j{i}" range="-1 1"/>'
                     '<geom size=".01" contype="0" conaffinity="0"/></body>' for i in range(20))
    actuators = ''.join(f'<position joint="j{i}" kp=".5"/>' for i in range(20))
    return mujoco.MjModel.from_xml_string(f"""
      <mujoco><compiler angle="radian"/><option timestep=".0025" integrator="implicitfast"/>
      <worldbody><geom name="{pad_name}" type="plane" size="1 1 .1"/>
      {joints}<body name="pen" pos="0 0 .012"><freejoint/>
      <inertial pos="{com_x} 0 0" mass=".03" diaginertia="3e-6 3e-6 3e-6"/>
      <geom name="object/pen" size=".01"/></body></worldbody>
      <actuator>{actuators}</actuator></mujoco>""")


def snapshot(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, state, STATE)
    return state


def evaluate(model, data, targets, ref_pos, ref_quat=None, **kwargs):
    if ref_quat is None:
        ref_quat = np.tile([1., 0, 0, 0], (len(ref_pos), 1))
    return score_candidates(model, snapshot(model, data), targets, ref_pos, ref_quat,
                            body='pen', qa=20, da=20, **kwargs)


class CorrectiveLookaheadTests(unittest.TestCase):
    def test_weight_support_is_hand_only_and_candidate_source_is_restored(self):
        for pad_name, by_hand in [('robot/pad', True), ('table', False)]:
            with self.subTest(pad=pad_name):
                model = fixture(pad_name)
                data = mujoco.MjData(model)
                mujoco.mj_step(model, data, nstep=500)
                reference = np.repeat(data.qpos[None, 20:23], 2, axis=0)
                before = snapshot(model, data)
                rows = evaluate(model, data, np.zeros((2, 2, 20)), reference)
                self.assertEqual(rows[0], rows[1])
                np.testing.assert_array_equal(snapshot(model, data), before)
                row = rows[0]
                self.assertEqual(row['native_steps'], 16)
                self.assertEqual(row['valid_controls'], 2 if by_hand else 0)
                self.assertLess(row['pose_cost'], 1e-20)
                self.assertAlmostEqual(row['support_cost'], 0. if by_hand else .5)
                for item in row['intervals']:
                    self.assertTrue(item['finite'])
                    self.assertEqual(item['geometry_ok'], by_hand)
                    weight = .03 * 9.81
                    np.testing.assert_allclose(item['requested_wrench_body'], [0,0,weight,0,0,0], atol=1e-12)
                    np.testing.assert_allclose(item['measured_wrench_body'],
                                               [0,0,weight if by_hand else 0,0,0,0], atol=1e-12)
                    self.assertAlmostEqual(item['max_nonhand_normal_N'], 0 if by_hand else weight)

    def test_measured_com_torque_matches_generalized_constraint_force(self):
        model = fixture('robot/pad', com_x=.004)
        data = mujoco.MjData(model)
        data.qpos[22] = .009
        rot = Rotation.from_euler('z', 40, degrees=True)
        data.qpos[23:27] = np.roll(rot.as_quat(), 1)
        reference = [data.qpos[20:23] + rot.apply(model.body_ipos[-1])]
        result, = evaluate(model, data, np.zeros((1, 1, 20)), reference)
        wrench = np.zeros((2, 3))
        for _ in range(8):
            mujoco.mj_step(model, data)
            force = data.qfrc_constraint[20:23]
            torque_at_root = data.xmat[-1].reshape(3,3) @ data.qfrc_constraint[23:26]
            wrench[0] += force
            wrench[1] += torque_at_root - np.cross(data.xipos[-1] - data.xpos[-1], force)
        expected = rot.inv().apply(wrench / 8).ravel()
        self.assertGreater(np.linalg.norm(expected[3:]), 1e-4)
        np.testing.assert_allclose(result['intervals'][0]['measured_wrench_body'], expected, atol=1e-12)
        end_rot = Rotation.from_quat(np.roll(data.qpos[23:27], -1))
        np.testing.assert_allclose(result['endpoint_pos'], data.qpos[20:23] + end_rot.apply(model.body_ipos[-1]))
        np.testing.assert_allclose(result['endpoint_quat_wxyz'], data.qpos[23:27])

    def test_free_fall_prefix_and_external_force_rejection(self):
        model = fixture('table')
        data = mujoco.MjData(model)
        data.qpos[22] = .019
        reference = np.tile(data.qpos[20:23], (5,1))
        targets = np.zeros((1,5,20))
        row, = evaluate(model, data, targets, reference)
        valid = [item['geometry_ok'] for item in row['intervals']]
        self.assertTrue(valid[0])
        self.assertIn(False, valid)
        self.assertEqual(row['valid_controls'], valid.index(False))
        data.xfrc_applied[-1, 0] = 1.
        with self.assertRaisesRegex(ValueError, 'externally applied'):
            evaluate(model, data, targets, reference)

    def test_requested_velocity_and_so3_angle_use_world_axes(self):
        model = fixture('robot/pad')
        data = mujoco.MjData(model)
        data.qpos[22] = .5
        start = Rotation.from_euler('z', 90, degrees=True)
        data.qpos[23:27] = np.roll(start.as_quat(), 1)
        v = np.array([[.3,0,0]])
        omega = np.array([[1.,0,0]])
        end = Rotation.from_rotvec(.02 * omega[0]) * start
        ref_pos = data.qpos[None, 20:23] + .02 * v
        row, = evaluate(model, data, np.zeros((1,1,20)), ref_pos,
                        [np.roll(end.as_quat(), 1)], ref_v=v, ref_omega=omega)
        item = row['intervals'][0]
        desired_world = np.array([[.03*20*.3, 0, .03*9.81], [3e-6*20, 0, 0]])
        np.testing.assert_allclose(item['requested_wrench_body'], start.inv().apply(desired_world).ravel(), atol=1e-12)
        self.assertAlmostEqual(item['deg'], np.rad2deg(.02))
        self.assertAlmostEqual(row['pose_cost'], (item['mm']/10)**2 + (item['deg']/10)**2)

    def test_directed_shaft_distinguishes_axial_twist_from_reversal(self):
        model = fixture('robot/pad')
        data = mujoco.MjData(model)
        data.qpos[22] = .5
        for axis, shaft_error in [('x', 0.), ('y', 180.)]:
            with self.subTest(axis=axis):
                end = Rotation.from_euler(axis, 180, degrees=True)
                row, = evaluate(model, data, np.zeros((1,1,20)), [data.qpos[20:23]],
                                [np.roll(end.as_quat(), 1)])
                item, = row['intervals']
                self.assertAlmostEqual(item['deg'], 180.)
                self.assertAlmostEqual(item['shaft_deg'], shaft_error)


if __name__ == '__main__':
    unittest.main()
