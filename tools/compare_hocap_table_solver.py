"""Separate table collision representation and solver stopping scale on a real Wuji scene."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from omegaconf import OmegaConf
import warp as wp

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json


def build(cfg, parquet, plane):
    cfg = deepcopy(cfg)
    cfg["collision"]["ground"] = plane
    cfg["scene"]["support"] = not plane
    cfg = OmegaConf.create(cfg)
    ec = instantiate_typed(cfg.env, RLEnvConfig)
    setup = setup_scene(parquet=parquet, control_fps=30, motion_speed=1, source_frame_playback=True,
                        embodiment=instantiate_typed(cfg.embodiment, Embodiment),
                        collision=instantiate_typed(cfg.collision, CollisionPolicy),
                        world_count=1, include_support=not plane,
                        decompose_objects=cfg.scene.decompose_objects,
                        object_scale_min=1., object_scale_max=1., contact_friction=cfg.scene.contact_friction,
                        object_free_joint_damping=cfg.scene.object_free_joint_damping)
    env = RLEnv(setup.scene,setup.reference,ec)
    env.reset(frame_id=0)
    if env.config.voc.max_force or env.config.voc.max_torque or env.config.auto_reset:
        raise ValueError("Diagnostic must use the unaided explicit reset")
    return env


def measure(env, stats_override, actions, target):
    solver = env.solver
    model, device_data = solver.mj_model, solver.mjw_data
    host_data = mujoco.MjData(model)
    compiled_mean = float(solver.mjw_model.stat.meaninertia.numpy()[0])
    if stats_override is not None:
        solver.mjw_model.stat.meaninertia.fill_(stats_override)
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    joint, = [j for j in free if "G04_2" in str(model.body(int(model.jnt_bodyid[j])).name)]
    dof = int(model.jnt_dofadr[joint])
    body = int(model.jnt_bodyid[joint])
    mass = float(model.body_mass[body])
    obj_shapes = np.array(list(env.scene.objects[0].shapes.ids()))
    hand_shapes = np.array([i for link in env.scene.layout.hand("right").link_geometry for i in link.shape_ids])
    original = solver.step
    rows = []
    def step(s0,s1,control,contacts,dt):
        # Rebuild the configuration-dependent physical inertia independently in float64.
        host_data.qpos[:] = device_data.qpos.numpy()[0]
        mujoco.mj_fwdPosition(model,host_data)
        original(s0,s1,control,contacts,dt)
        solver.update_contacts(contacts,s1)
        acc = device_data.qacc.numpy()[0].astype(np.float64)
        ma = np.empty_like(acc)
        mujoco.mj_mulM(model,host_data,ma,acc)
        force = device_data.qfrc_constraint.numpy()[0,dof:dof+3]
        smooth = device_data.qfrc_smooth.numpy()[0,dof:dof+3]
        n = int(contacts.rigid_contact_count.numpy()[0])
        a,b = contacts.rigid_contact_shape0.numpy()[:n],contacts.rigid_contact_shape1.numpy()[:n]
        hand = ((np.isin(a,obj_shapes) & np.isin(b,hand_shapes)) | (np.isin(b,obj_shapes) & np.isin(a,hand_shapes)))
        rows.append([force[2],smooth[2],ma[dof+2],np.linalg.norm(ma[dof:dof+3]-force-smooth),
                     device_data.solver_niter.numpy()[0],hand.sum(),s1.body_q.numpy()[env.scene.objects[0].bodies[0].body_id,2]])
    solver.step = step
    start=perf_counter()
    try:
        for action in actions:
            env.step(wp.array(action,dtype=wp.float32,device=env.device))
    finally:
        solver.step=original
    a=np.asarray(rows)
    np.save(target,a)
    # The last five pre-grasp frames give the resting check time to settle.
    end = a[-5*env.config.sim.substeps:]
    bodyweight = mass * abs(float(model.opt.gravity[2]))
    return dict(compiled_meaninertia=compiled_mean, used_meaninertia=compiled_mean if stats_override is None else stats_override,
                body_mass_kg=mass, tolerance=float(solver.mjw_model.opt.tolerance.numpy()[0]), substeps=len(a),
                sample_columns=["vertical_constraint_N","vertical_smooth_N","vertical_M_qacc_N",
                                "vector_balance_residual_N","solver_iterations","hand_object_contacts","object_world_z_m"],
                elapsed_s=perf_counter()-start, median_solver_iterations=float(np.median(a[:,4])),
                mean_vector_balance_residual_N=float(a[:,3].mean()), p95_vector_balance_residual_N=float(np.quantile(a[:,3],.95)),
                late_five_frames=dict(hand_contacts=int(end[:,5].sum()), mean_vertical_load_N=float(end[:,0].mean()),
                                      bodyweight_N=bodyweight, mean_vertical_M_qacc_N=float(end[:,2].mean())),
                samples=file_pin(target))


def run(a):
    a.output.mkdir(exist_ok=False,parents=True)
    wp.config.kernel_cache_dir=str(a.cache)
    wp.init()
    wp.set_device(a.device)
    cfg=checkpoint_config(read_checkpoint_metadata(a.checkpoint))
    original_termination=deepcopy(cfg["env"]["termination"])
    apply_evaluation_composition(cfg,world_count=1,reset_mode="explicit")
    cfg["env"]["termination"]=original_termination
    with np.load(a.trace,allow_pickle=False) as saved:
        actions=saved["actor_action"][:a.frames]
    outcomes={}
    for tag, plane, scale_from in (("flat_plane",True,None),("kinematic_box",False,None),
                                   ("kinematic_box_plane_scale",False,"flat_plane"),
                                   ("flat_plane_box_scale",True,"kinematic_box")):
        env=build(cfg,a.parquet,plane)
        override=outcomes[scale_from]["compiled_meaninertia"] if scale_from else None
        outcomes[tag]=measure(env,override,actions,a.output/(tag+".npy"))
        print(tag,outcomes[tag]["mean_vector_balance_residual_N"],flush=True)
    write_json(a.output/"report.json",dict(outcomes=outcomes,checkpoint=file_pin(a.checkpoint),
                                          trace=file_pin(a.trace),parquet=file_pin(a.parquet),code=file_pin(__file__),
                                          device=a.device,scope="Same actor commands and source frames. Plane matches top height inside the finite table footprint."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("parquet","checkpoint","trace","cache","output"):
        p.add_argument(k,type=Path)
    p.add_argument("--device",default="cpu")
    p.add_argument("--frames",type=int,default=15)
    run(p.parse_args())
