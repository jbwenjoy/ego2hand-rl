"""Right-only Wuji Hand 2 embodiment for NVIDIA Video-to-Data FlashCHORD v0.4.0.

Import this module in the isolated FlashCHORD runtime. The companion exporter
builds the MJCF and numerical manifest from the repository's calibrated model.
"""
import hashlib
import json
import math
from pathlib import Path

import newton
import numpy as np
import warp as wp

from flash_chord.data.reference import NamedRobotReference
from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout, HandLayout, HandLinkGeometry, register_embodiment
from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.utils.quat import wxyz_to_rotvec, wxyz_to_xyzw


def _leaf(label):
    return label.rsplit("/",1)[-1]


def _checked(path, pin):
    if path.stat().st_size != pin["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest()!=pin["sha256"]:
        raise ValueError(f"Calibrated Wuji asset has changed: {path}")
    return path


@register_embodiment("wuji_hand2")
class WujiHand2:
    name = "wuji_hand2"

    def __init__(self, asset_dir):
        self.folder = Path(asset_dir)
        self.meta = json.loads((self.folder/"adapter.json").read_bytes())

    def build(self, builder):
        meta = self.meta
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        xml = _checked(self.folder/meta["xml"], meta["model_pin"])
        for pin in meta["mesh_pins"]:
            # Pin paths were recorded on Windows. Resolve only the published basename under this bundle.
            _checked(self.folder/"meshes"/pin["path"].replace("\\","/").rsplit("/",1)[-1],pin)
        builder.add_mjcf(str(xml), parse_visuals=False, parse_sites=False, parse_mujoco_options=False,
                         ignore_inertial_definitions=False, enable_self_collisions=True,
                         collapse_fixed_joints=False, mesh_maxhullvert=65536)
        bodies = {_leaf(n):i for i,n in enumerate(builder.body_label)}
        # Preserve fixed sensor links as actual Newton bodies. Hand contact slots contain only
        # the 21 collision-enabled palm, link and finger pad geoms of the native Wuji model.
        shapes = {_leaf(n):i for i,n in enumerate(builder.shape_label)}
        if set(shapes)!=set(meta["geometry"]):
            raise ValueError("Newton collision shapes differ from the native calibrated model")
        geometry = []
        for geom, shape in shapes.items():
            body = int(builder.shape_body[shape])
            name = _leaf(builder.body_label[body])
            if name!=meta["geometry"][geom]["body"]:
                raise ValueError(f"Collision geometry owner mismatch for {geom}")
            geometry.append(HandLinkGeometry(link_name=name, shape_ids=(shape,), object_contact=True))
        for body, mass in meta["native_body_masses"].items():
            np.testing.assert_allclose(builder.body_mass[bodies[body]],mass,atol=5e-6,rtol=0)
        labels = {_leaf(n):i for i,n in enumerate(builder.joint_label)}
        pos_joints = [labels[f"right_wrist_{a}"] for a in "xyz"]
        rot = labels["right_wrist_rotvec"]
        assert builder.joint_type[rot] == newton.JointType.BALL
        passive = builder.custom_attributes["mujoco:dof_passive_damping"].values
        for joint in pos_joints:
            dof = builder.joint_qd_start[joint]
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
            builder.joint_target_ke[dof],builder.joint_target_kd[dof] = 2080.,65.
            builder.joint_effort_limit[dof] = 55./math.sqrt(3.)
            builder.joint_velocity_limit[dof] = 1.8/math.sqrt(3.)
        angular_ids = tuple(builder.joint_qd_start[rot]+i for i in range(3))
        for dof in angular_ids:
            builder.joint_target_mode[dof]=int(newton.JointTargetMode.EFFORT)
            builder.joint_target_ke[dof]=0.;builder.joint_target_kd[dof]=0.
            builder.joint_effort_limit[dof]=4./math.sqrt(3.)
            builder.joint_velocity_limit[dof]=8./math.sqrt(3.)
            passive[dof]=1.1
        finger_names=tuple(f"right_finger{f}_joint{j}" for f in range(1,6) for j in range(1,5))
        finger_joints=[labels[n] for n in finger_names]
        for name,joint in zip(finger_names,finger_joints,strict=True):
            spec=meta["joints"][name]
            np.testing.assert_array_equal(spec["range"],spec["ctrl_range"])
            dof=builder.joint_qd_start[joint]
            np.testing.assert_allclose([builder.joint_limit_lower[dof],builder.joint_limit_upper[dof]],spec["range"],atol=2e-6)
            np.testing.assert_allclose([builder.joint_armature[dof],builder.joint_friction[dof]],
                                       [spec["armature"],spec["friction"]],atol=2e-6)
            builder.joint_target_ke[dof],builder.joint_target_kd[dof]=spec["kp"],spec["kd"]
            builder.joint_target_mode[dof]=int(newton.JointTargetMode.POSITION_VELOCITY)
            builder.joint_effort_limit[dof]=spec["effort"]
            passive[dof]=spec["damping"]
        tip_frames=tuple(BodyFrame(name=t["name"],body_id=bodies[t["parent"]],
                                  body_to_frame_pos=tuple(t["pos"]),body_to_frame_quat_xyzw=tuple(t["quat_xyzw"]))
                         for t in meta["tips"])
        distal=tuple(BodyFrame(name=n,body_id=bodies[n]) for n in (f"right_finger{i}_link4" for i in range(1,6)))
        hand=HandLayout(side="right",palm_frame=BodyFrame(name="right_palm_link",body_id=bodies["right_palm_link"]),
                        dp_frames=distal,fingertip_frames=tip_frames,link_geometry=tuple(geometry),
                        wrist_pos_dof_ids=tuple(builder.joint_qd_start[j] for j in pos_joints),
                        wrist_pos_q_ids=tuple(builder.joint_q_start[j] for j in pos_joints),
                        wrist_orient_dof_ids=angular_ids,wrist_orient_q_id=builder.joint_q_start[rot],
                        finger_joint_names=finger_names,finger_q_ids=tuple(builder.joint_q_start[j] for j in finger_joints),
                        finger_dof_ids=tuple(builder.joint_qd_start[j] for j in finger_joints))
        return EmbodimentLayout(num_joint_q=len(builder.joint_q),num_joint_dof=builder.joint_dof_count,hands=(hand,))

    def bind_reference(self,builder,layout,reference):
        if not isinstance(reference,NamedRobotReference):
            raise TypeError("Wuji requires 20 named robot joints and an absolute root pose")
        hand=layout.hand("right")
        ref_names=tuple(reference.robot_joint_names())
        if set(ref_names)!=set(hand.finger_joint_names) or len(ref_names)!=20:
            raise ValueError(f"Expected exactly the twenty named Wuji joints, received {ref_names}")
        fingers=reference.robot_joint_pos()[:,[ref_names.index(n) for n in hand.finger_joint_names]]
        pos=reference.robot_root_pos_w()
        quat=reference.robot_root_quat_w()
        joint_q=np.tile(np.asarray(builder.joint_q,dtype=np.float32),(reference.num_frames,1))
        target=np.tile(np.asarray(builder.joint_target_pos,dtype=np.float32),(reference.num_frames,1))
        joint_q[:,hand.wrist_pos_q_ids]=pos
        joint_q[:,hand.wrist_orient_q_id:hand.wrist_orient_q_id+4]=wxyz_to_xyzw(quat)
        joint_q[:,hand.finger_q_ids]=fingers
        target[:,hand.wrist_pos_dof_ids]=pos
        target[:,hand.wrist_orient_dof_ids]=wxyz_to_rotvec(quat)
        target[:,hand.finger_dof_ids]=fingers
        return build_robot_reference_binding(builder,layout,joint_q,target,reference.fps)

from dataclasses import dataclass
from flash_chord.runtime.actions.hand_pose import ActionConfig, ActionBuffers, ResidualHandPoseAction
from flash_chord.runtime.control import WristController
from flash_chord.runtime.delay import TargetDelayBuffer


@wp.kernel
def _limit_finger_commands(target: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.int32),
                           lower: wp.array(dtype=wp.float32), upper: wp.array(dtype=wp.float32),
                           dofs_per_world: int):
    world, finger = wp.tid()
    i = world*dofs_per_world + indices[finger]
    target[i] = wp.clamp(target[i], lower[i], upper[i])


class WujiResidualHandPose(ResidualHandPoseAction):
    def __init__(self,buffers,wrist,delay,scene,device):
        super().__init__(buffers,wrist,delay)
        self.finger_indices = wp.array(scene.layout.hand("right").finger_dof_ids, dtype=wp.int32,device=device)
        self.model = scene.model
        self.world_count = scene.world_count

    def process(self,action,timestep,state=None):
        super().process(action,timestep,state)
        # Newton generates split position/velocity actuators with a total joint torque
        # cap. Restore also the original calibrated position actuator's ctrlrange.
        wp.launch(_limit_finger_commands,dim=(self.world_count,20),inputs=[
            self.joint_target,self.finger_indices,self.model.joint_limit_lower,self.model.joint_limit_upper,
            self.model.joint_dof_count//self.world_count],device=self.joint_target.device)


@dataclass(frozen=True)
class WujiActionConfig(ActionConfig):
    """Official residual hand action with the actual side list and bounded wrist."""
    wrist_kp: float = 38.5
    wrist_tau_max: float = 4./math.sqrt(3.)

    def build(self,scene,device=None):
        if scene.layout.sides!=("right",):
            raise ValueError(f"Wuji requires exactly the right hand, got {scene.layout.sides}")
        if not (np.isfinite(self.wrist_kp) and self.wrist_kp>0 and 0<self.wrist_tau_max<=4./math.sqrt(3.)+1e-7):
            raise ValueError("Invalid wrist controller gains or torque bound")
        dofs=scene.model.joint_dof_count//scene.world_count
        buffers=ActionBuffers.build(scene.layout,scene.robot_reference,scene.world_count,config=self,
                                    sides=scene.layout.sides,num_joint_dof=dofs,device=device)
        wrist=WristController.from_layout(scene.layout,scene.model,scene.world_count,kp=self.wrist_kp,
                                          tau_max=self.wrist_tau_max,device=device)
        delay=None
        if self.delay.max_steps>0:
            delay=TargetDelayBuffer.build(world_count=scene.world_count,num_joint_dof=dofs,
                                         delayed_dof_ids=tuple(range(scene.layout.num_joint_dof)),
                                         config=self.delay,device=device)
        return WujiResidualHandPose(buffers,wrist,delay,scene,device)
