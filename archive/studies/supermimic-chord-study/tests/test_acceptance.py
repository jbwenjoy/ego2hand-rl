import copy
import unittest
import numpy as np
from chord_study.reference import Reference
from chord_study.acceptance import make_pilot_protocol,protocol_digest,progress_from_axes,projected_shaft_progress,fit_reference_projected_rotation
from chord_study.evaluation import evaluate


def fixture():
    n=101;t=np.arange(n)*.02;theta=np.linspace(0,2*np.pi,n)
    pose=np.zeros((n,7));pose[:,3]=np.cos(theta/2);pose[:,6]=np.sin(theta/2)
    wrist=np.zeros((n,7));wrist[:,3]=1
    q=np.zeros((n,20))
    meta={"schema":"supermimic_chord_reference:1","source_id":"test_turn","source_kind":"synthetic_test","split_group":"test",
          "joint_names":[f"q{i}" for i in range(20)],"units":"SI","quaternion_order":"wxyz","world_frame":"test_w",
          "contact_normal":"force_on_object_inward","object_radius_m":.1,"object_mass_kg":.1,
          "object_inertia_kg_m2":(np.eye(3)*.0001).tolist(),"orientation_metric":"directed_axis_x",
          "provenance":{"purpose":"synthetic metric fixture, not a physical result"}}
    ref=Reference({"time_s":t,"joint_pos_rad":q,"object_pose_w":pose,"wrist_pose_w":wrist,"phase":np.zeros(n)},meta)
    protocol=make_pilot_protocol(ref,"test_hash")
    ref.meta["task_acceptance"]=protocol;ref.meta["task_acceptance_sha256"]=protocol_digest(protocol)
    run={"time_s":t.copy(),"joint_pos_rad":q.copy(),"joint_target_rad":q.copy(),"object_pose_w":pose.copy(),"wrist_pose_w":wrist.copy(),
         "joint_lower_rad":np.full(20,-1.),"joint_upper_rad":np.full(20,1.),"object_assist_wrench_w":np.zeros((n,6)),
         "object_pose_overwritten":np.zeros(n),"terminated":np.zeros(n),"penetration_m":np.zeros(n),"actuator_torque_Nm":q.copy(),
         "actuator_force_limit_Nm":np.ones(20),"object_ang_vel_w_rad_s":np.tile([0.,0.,np.pi],(n,1)),"object_environment_contact_count":np.zeros(n)}
    rm={"physics_engine":"mujoco","object_free_dynamics":True,"reference_source_id":"test_turn","world_frame":"test_w",
        "units":"SI","quaternion_order":"wxyz","joint_names":meta["joint_names"].copy(),"reference_bundle_sha256":"test_hash",
        "resets_during_rollout":0,"object_state_writes_after_initialization":0,"rollout_censored":False}
    return ref,run,rm


class PilotAcceptance(unittest.TestCase):
    def setUp(self):self.ref,self.run,self.meta=fixture()

    def test_complete_matching_turn_passes_all_declared_gates(self):
        result=evaluate(self.ref,self.run,self.meta)
        self.assertTrue(result["task_success"])
        self.assertAlmostEqual(result["reference_progress_deg"],360)
        self.assertAlmostEqual(result["rotation_progress_error_deg"]["endpoint"],0)

    def test_same_endpoint_does_not_hide_missing_turn(self):
        self.run["object_pose_w"][:,3:]=[1.,0.,0.,0.]
        self.run["object_ang_vel_w_rad_s"][:]=0
        result=evaluate(self.ref,self.run,self.meta)
        self.assertFalse(result["task_success"])
        self.assertAlmostEqual(result["rotation_progress_error_deg"]["endpoint"],360)

    def test_sampled_angular_alias_is_rejected(self):
        self.run["object_ang_vel_w_rad_s"][20,2]=200
        self.assertFalse(evaluate(self.ref,self.run,self.meta)["progress_alias_bound_ok"])

    def test_load_integral_disqualifies_sustained_saturation(self):
        self.run["actuator_torque_Nm"][:]=.9
        result=evaluate(self.ref,self.run,self.meta)
        self.assertFalse(result["load_ok"])
        self.assertFalse(result["task_success"])

    def test_contact_with_environment_disqualifies_floor_supported_pen(self):
        self.run["object_environment_contact_count"][30]=1
        result=evaluate(self.ref,self.run,self.meta)
        self.assertFalse(result["drop_ok"])

    def test_reference_protocol_cannot_be_changed_without_hash(self):
        self.ref.meta["task_acceptance"]["position_max_m"]=1
        with self.assertRaisesRegex(ValueError,"protocol hash"):
            evaluate(self.ref,self.run,self.meta)

    def test_missing_force_limits_remains_unavailable(self):
        del self.run["actuator_force_limit_Nm"]
        self.assertIsNone(evaluate(self.ref,self.run,self.meta)["task_success"])

    def test_singular_projection_does_not_invent_progress(self):
        self.run["object_pose_w"][10,3:]=[np.sqrt(.5),0,np.sqrt(.5),0]
        result=evaluate(self.ref,self.run,self.meta)
        self.assertFalse(result["progress_ok"])
        self.assertLess(result["progress_observed_fraction"],1)

    def test_fast_target_changes_disqualify_control(self):
        self.run["joint_target_rad"][::2]=.9
        self.assertFalse(evaluate(self.ref,self.run,self.meta)["target_rate_ok"])

    def test_nonempty_observation_gap_stays_unknown(self):
        axes=np.array([[1,0,0],[0,0,1],[0,1,0]],float)
        p,valid=progress_from_axes(axes,[0,0,1],[1,0,0])
        self.assertFalse(valid[1]);self.assertTrue(np.isnan(p[1]))

    def test_projected_shaft_rotation_counts_a_planar_turn(self):
        theta=np.linspace(0,2*np.pi,101)
        axes=np.stack((np.cos(theta),np.sin(theta),np.zeros_like(theta)),-1)
        p,valid=projected_shaft_progress(axes,[0,0,1])
        self.assertTrue(valid.all());self.assertAlmostEqual(p[-1],360)

    def test_projected_motion_gate_rejects_quiet_hold(self):
        pose=self.ref.arrays["object_pose_w"].copy()
        theta=np.linspace(0,np.deg2rad(25),len(pose));pose[:,3]=np.cos(theta/2);pose[:,6]=np.sin(theta/2)
        with self.assertRaisesRegex(ValueError,"lacks"):
            fit_reference_projected_rotation(pose)


if __name__=="__main__":unittest.main()
