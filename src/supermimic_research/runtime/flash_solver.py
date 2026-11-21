"""Versioned FlashCHORD solver configuration for moveable Wuji bodies on a kinematic table."""
from dataclasses import dataclass
import sys

import mujoco
import numpy as np
from flash_chord.runtime import sim as flash_sim


_upstream_build_solver = flash_sim.build_solver


@dataclass
class WujiSimConfig(flash_sim.SimConfig):
    """Keep the official solver but condition its convergence statistics on dynamic DOFs."""
    conditioning: str = "movable_dof_mean_v1"

    def __post_init__(self):
        if self.conditioning != "movable_dof_mean_v1":
            raise ValueError(f"Unknown convergence calibration {self.conditioning}")
        install_solver_dispatch()


def condition_solver(solver):
    model = solver.mj_model
    batch=solver.model
    if batch.world_count>1:
        for field in ("body_mass","body_com","body_inertia","joint_armature"):
            values=getattr(batch,field).numpy().reshape(batch.world_count,-1)
            if not np.all(values==values[:1]):
                raise ValueError(f"One mass normalization per shard requires identical {field} across its worlds")
    support = [b for b in range(1, model.nbody)
               if model.body(b).name == "support" or model.body(b).name.endswith("_support")]
    if len(support) != 1:
        raise ValueError(f"Expected one named kinematic support in the solver template, got {support}")
    # Exclude only the six coordinates belonging to the source's kinematic support.
    j = int(model.body_jntadr[support[0]])
    if model.body_jntnum[support[0]] != 1 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected one carried free joint for the kinematic table")
    start = int(model.jnt_dofadr[j])
    retained = np.ones(model.nv, dtype=bool)
    retained[start:start+6] = False
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    mass = np.empty((model.nv, model.nv),dtype=float)
    mujoco.mj_fullM(model,mass,data.qM)
    value = float(np.mean(mass.diagonal()[retained]))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid effective mobile inertia {value}")
    # The replicated single-object scenes have identical mass and length scale within each
    # shard. No model masses, solver tolerances, constraint weights or actuators are changed.
    orig=solver.mjw_model.stat.meaninertia.numpy().copy()
    solver.mjw_model.stat.meaninertia.fill_(value)
    solver.wuji_conditioning=dict(scheme="movable_dof_mean_v1",original=float(orig[0]),
                                  conditioned=value,solver_dofs=model.nv,movable_dofs=int(retained.sum()),
                                  support_free_dofs=list(range(start,start+6)),
                                  target_tolerance=solver.mjw_model.opt.tolerance.numpy().tolist())
    return solver


def build_solver(model,cfg):
    solver=_upstream_build_solver(model,cfg)
    return condition_solver(solver) if isinstance(cfg,WujiSimConfig) else solver


def install_solver_dispatch():
    # v0.4.0 imports a module-level factory into both environment entry points.
    # This process-local router leaves the upstream checkout and old saved configs intact.
    for name in ("flash_chord.runtime.sim","flash_chord.envs.base","flash_chord.runtime.replay"):
        mod=sys.modules.get(name)
        if mod is not None:
            current=getattr(mod,"build_solver",None)
            if current is _upstream_build_solver:
                mod.build_solver=build_solver
            elif current is not build_solver:
                raise RuntimeError(f"Unexpected FlashCHORD solver dispatcher in {name}")
