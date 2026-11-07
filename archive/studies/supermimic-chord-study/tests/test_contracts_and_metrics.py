import copy
import unittest
import numpy as np
from chord_study.reference import Reference, assert_disjoint_split_groups
from chord_study.evaluation import evaluate
from chord_study.experiments import perturb_reference


def reference():
    n=4
    pose=np.zeros((n,7)); pose[:,3]=1
    return Reference({"time_s":np.arange(n)*.02,"joint_pos_rad":np.zeros((n,20)),
                      "object_pose_w":pose.copy(),"wrist_pose_w":pose.copy(),"phase":np.zeros(n)},
                     {"schema":"supermimic_chord_reference:1","source_id":"test","source_kind":"synthetic_test",
                      "split_group":"original_1","joint_names":[f"q{i}" for i in range(20)],"units":"SI",
                      "quaternion_order":"wxyz","world_frame":"test_w","contact_normal":"force_on_object_inward",
                      "object_radius_m":.1,"object_mass_kg":.1,"object_inertia_kg_m2":(np.eye(3)*.0001).tolist(),
                      "orientation_metric":"SO3","provenance":{"purpose":"test fixture only"}})


def rollout(ref):
    a=ref.arrays; n=len(a["time_s"])
    return {"time_s":a["time_s"].copy(),"joint_pos_rad":a["joint_pos_rad"].copy(),
            "joint_target_rad":a["joint_pos_rad"].copy(),"object_pose_w":a["object_pose_w"].copy(),
            "wrist_pose_w":a["wrist_pose_w"].copy(),
            "joint_lower_rad":np.full(20,-1.),"joint_upper_rad":np.full(20,1.),
            "object_assist_wrench_w":np.zeros((n,6)),"object_pose_overwritten":np.zeros(n),
            "terminated":np.zeros(n),"penetration_m":np.zeros(n)}


class Contracts(unittest.TestCase):
    def test_perturbation_keeps_object_target_and_source_group(self):
        ref=reference();p=perturb_reference(ref,joint_bias_rad=.03,seed=7)
        np.testing.assert_array_equal(p.arrays["object_pose_w"],ref.arrays["object_pose_w"])
        np.testing.assert_allclose(np.abs(p.arrays["joint_pos_rad"]),.03)
        np.testing.assert_array_equal(ref.arrays["joint_pos_rad"],np.zeros((4,20)))
        self.assertEqual(p.meta["split_group"],ref.meta["split_group"])
        self.assertNotEqual(p.meta["source_id"],ref.meta["source_id"])

    def test_unknown_contacts_are_not_negative_labels(self):
        ref=reference()
        self.assertEqual(ref.validate()["contact_label_coverage"],0)
        with self.assertRaisesRegex(ValueError,"independent contact"):
            ref.validate(contact_required=True)

    def test_wrist_motion_is_rejected_by_screen(self):
        ref=reference();ref.arrays["wrist_pose_w"][-1,0]=.1
        self.assertFalse(ref.validate()["fixed_wrist_admissible"])

    def test_parent_trajectory_leakage(self):
        with self.assertRaisesRegex(ValueError,"leakage"):
            assert_disjoint_split_groups([{"split_group":"a","split":"train"},{"split_group":"a","split":"test"}])

    def test_bad_quaternion_does_not_get_normalized_silently(self):
        ref=reference();ref.arrays["object_pose_w"][0,3]=2
        with self.assertRaisesRegex(ValueError,"unit"):
            ref.validate()


class Evaluation(unittest.TestCase):
    def setUp(self):
        self.ref=reference(); self.run=rollout(self.ref)
        self.meta={"physics_engine":"mujoco","object_free_dynamics":True,"reference_source_id":"test","world_frame":"test_w",
                   "units":"SI","quaternion_order":"wxyz","joint_names":self.ref.meta["joint_names"].copy()}

    def test_early_stop_is_failure(self):
        run=copy.deepcopy(self.run)
        for k in run:
            if k not in {"joint_lower_rad","joint_upper_rad"}: run[k]=run[k][:2]
        self.assertFalse(evaluate(self.ref,run,self.meta)["screening_success"])

    def test_reference_teleport_cannot_pass(self):
        self.run["object_pose_overwritten"][1]=1
        self.assertFalse(evaluate(self.ref,self.run,self.meta)["screening_success"])

    def test_auxiliary_force_cannot_pass(self):
        self.run["object_assist_wrench_w"][1,0]=.01
        self.assertFalse(evaluate(self.ref,self.run,self.meta)["screening_success"])

    def test_declared_axis_task_metric_does_not_penalize_axial_roll(self):
        self.run["object_pose_w"][:,3:]=[0,1,0,0]
        full=evaluate(self.ref,self.run,self.meta)
        self.assertFalse(full["screening_success"])
        self.ref.meta["orientation_metric"]="directed_axis_x"
        axial=evaluate(self.ref,self.run,self.meta)
        self.assertTrue(axial["screening_success"])
        self.assertAlmostEqual(axial["full_so3_error_deg_diagnostic"]["max"],180.)

    def test_invalid_clock_is_error(self):
        self.run["time_s"][1]+=.001
        with self.assertRaisesRegex(ValueError,"clocks"):
            evaluate(self.ref,self.run,self.meta)

    def test_target_limits_are_checked(self):
        self.run["joint_target_rad"][2,0]=2
        self.assertFalse(evaluate(self.ref,self.run,self.meta)["screening_success"])

    def test_joint_order_mismatch_rejected(self):
        self.meta["joint_names"].reverse()
        with self.assertRaisesRegex(ValueError,"joint_names"):
            evaluate(self.ref,self.run,self.meta)

    def test_moving_wrist_cannot_pass(self):
        self.run["wrist_pose_w"][-1,0]=.1
        self.assertFalse(evaluate(self.ref,self.run,self.meta)["screening_success"])

    def test_negative_penetration_is_not_valid_depth(self):
        self.run["penetration_m"][1]=-.1
        with self.assertRaisesRegex(ValueError,"non-negative"):
            evaluate(self.ref,self.run,self.meta)

    def test_nonphysical_principal_inertia_rejected(self):
        self.ref.meta["object_inertia_kg_m2"]=np.diag([1.,1.,3.]).tolist()
        with self.assertRaisesRegex(ValueError,"triangle"):
            self.ref.validate()


if __name__=="__main__": unittest.main()
