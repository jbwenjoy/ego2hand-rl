"""Choose the closest admissible pen command using an eight-substep geometry forecast."""
from pathlib import Path

import mujoco
import numpy as np
import torch

from .controlled_pen_records import native_state55
from .pen_substep_pose import load_substep_pose


LABELS = ('actor', 'mix_ring_50', 'hold_ring', 'mix_all_50', 'hold_all', 'zero_raw')
GATES_MM = np.array([2., .2])
TICKS = 8


def _vector(values, size, dtype=np.float64):
    values = np.asarray(values, dtype=dtype).reshape(-1)
    if values.size != size or not np.isfinite(values).all():
        raise ValueError(f'Expected {size} finite values')
    return values


class PenContactActionFilter:
    """Forecast on private model data; all contact forces in the live task remain native."""

    def __init__(self, model, method, pose_checkpoint=None):
        self.model, self.method = model, method
        self.data = mujoco.MjData(model)
        if method not in ('cpu_dynamics', 'paired'):
            raise ValueError('Unknown pen contact predictor')
        self.net = None
        if method == 'paired':
            weights = torch.load(Path(pose_checkpoint), weights_only=True, map_location='cpu')
            self.net = load_substep_pose(weights).requires_grad_(False)
        if not np.isclose(model.opt.timestep,.0025) or model.nu != 20 or model.na or model.neq:
            raise ValueError('Requires the 20-servo free-pen native model')
        free, = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self.body = model.jnt_bodyid[free]
        self.qa, self.da = model.jnt_qposadr[free], model.jnt_dofadr[free]
        joints = model.actuator_trnid[:,0]
        self.qids, self.dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        np.testing.assert_array_equal(self.qids, np.arange(7,27))
        np.testing.assert_array_equal(self.dofs, np.arange(6,26))
        np.testing.assert_array_equal(model.body_ipos[self.body], np.zeros(3))
        self.ring = [i for i, joint in enumerate(joints) if model.joint(int(joint)).name.startswith('robot/right_finger4_')]
        if self.ring != list(range(12,16)) or self.qa != 0:
            raise ValueError('Unexpected free pen or ring-finger layout')
        self.pen_geoms = model.geom_bodyid == self.body
        self.hand_geoms = np.array([model.geom(i).name.startswith('robot/') for i in range(model.ngeom)])

    def _depths_mm(self):
        pairs = self.data.contact.geom
        pen = self.pen_geoms[pairs[:,0]] != self.pen_geoms[pairs[:,1]]
        hand = self.hand_geoms[pairs].all(axis=1)
        depth = -1000*np.asarray(self.data.contact.dist)
        return np.array([depth[pen].max(initial=0.), depth[hand].max(initial=0.)])

    def _peaks(self, targets, qpos, qvel, warm, mount, state):
        model, data = self.model, self.data
        peaks = np.zeros((len(targets), 2))
        if self.net is None:
            for i, target in enumerate(targets):
                mujoco.mj_resetData(model, data)
                data.qpos[:], data.qvel[:], data.qacc_warmstart[:] = qpos, qvel, warm
                data.mocap_pos[:], data.mocap_quat[:] = mount
                data.ctrl[:] = target
                for _ in range(TICKS):
                    mujoco.mj_step(model, data)
                    np.maximum(peaks[i], self._depths_mm(), out=peaks[i])
                if data.warning.number.any() or not np.isfinite(data.qpos).all():
                    raise FloatingPointError('Invalid CPU forecast')
        else:
            with torch.inference_mode():
                n = len(targets)
                paths = self.net.predict_qpos(torch.as_tensor(state).expand(n,-1),
                    torch.as_tensor(warm).expand(n,-1),torch.as_tensor(targets)).numpy()
            if not np.isfinite(paths).all():
                raise FloatingPointError('Nonfinite substep pose forecast')
            data.mocap_pos[:], data.mocap_quat[:] = mount
            for i,path in enumerate(paths):
                for pose in path:
                    data.qpos[:] = pose
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_comPos(model, data)
                    mujoco.mj_collision(model, data)
                    np.maximum(peaks[i],self._depths_mm(),out=peaks[i])
        return peaks

    def select_action(self, *, qpos, qvel, warmstart, mocap_pos, mocap_quat, actor_raw,
                      previous_raw, previous_ema, refq, encoder_bias, lower, upper):
        model = self.model
        qpos, qvel, warm = (_vector(a,n) for a,n in [(qpos,model.nq),(qvel,model.nv),(warmstart,model.nv)])
        mount = (_vector(mocap_pos,3*model.nmocap).reshape(-1,3),_vector(mocap_quat,4*model.nmocap).reshape(-1,4))
        actor, last, ema, ref, bias, lo, hi = (_vector(a,20,np.float32) for a in
                                               (actor_raw,previous_raw,previous_ema,refq,encoder_bias,lower,upper))
        actor = actor.clip(-1,1)
        last = last.clip(-1,1)
        if (lo>hi).any():
            raise ValueError('Empty joint and actuator range intersection')
        raws = np.tile(actor,(len(LABELS),1))
        raws[1,self.ring] = .5*(actor[self.ring]+last[self.ring])
        raws[2,self.ring] = last[self.ring]
        raws[3] = .5*(actor+last)
        raws[4] = last
        raws[5] = 0
        targets = np.clip(ref + (.5*ema + .25*raws) - bias,lo,hi)
        state = native_state55(qpos,qvel,self.qids,self.dofs,self.qa,self.da) if self.net is not None else None
        def forecast(values):
            return self._peaks(values,qpos,qvel,warm,mount,state)
        peaks = forecast(targets[:1])
        triggered = bool(np.any(peaks[0]>GATES_MM))
        chosen = 0
        if triggered:
            peaks = np.vstack([peaks,forecast(targets[1:])])
            maximum = (peaks/GATES_MM).max(axis=1)
            distance = np.linalg.norm(targets-targets[0],axis=1)
            # Satisfy both geometry limits first. Rank feasible alternatives by
            # deviation from the actor PD target, infeasible ones by peak ratio.
            chosen = min(range(len(peaks)),key=lambda i:(max(1.,maximum[i]),distance[i],i))
        details = {name:dict(object_mm=float(val[0]),self_mm=float(val[1]))
                   for name,val in zip(LABELS,peaks)}
        return raws[chosen].copy(), dict(choice=LABELS[chosen], triggered=triggered,
            pd=targets[chosen].tolist(), evaluations=details,
            planning_ticks=len(peaks)*TICKS if self.net is None else 0,
            predicted_poses=len(peaks)*TICKS if self.net is not None else 0)
