"""Bounded native kinematic proposals; fingertip alignment alone is not contact."""
import numpy as np
from scipy.optimize import least_squares


class HandTipIK:
    def __init__(self, model, qpos, targets, weights=None, regularization=.003):
        import mujoco
        self.mj, self.model = mujoco, model
        self.data = mujoco.MjData(model)
        self.base = np.asarray(qpos, float).copy()
        self.targets = np.asarray(targets, float)
        if self.base.shape != (27,) or self.targets.shape != (5, 3) or not np.isfinite(np.r_[self.base, self.targets.ravel()]).all():
            raise ValueError('Explicit finite27q state and five xyz targets required')
        self.site_ids = [model.site(f'right_finger{i}_tip').id for i in range(1, 6)]
        self.weights = np.ones(5) if weights is None else np.asarray(weights, float)
        if self.weights.shape != (5,) or np.any(self.weights <= 0):
            raise ValueError('Positive five tip weights')
        self.regularization = regularization
        self.lo = np.maximum(model.jnt_range[:20, 0], model.actuator_ctrlrange[:, 0])+1e-6
        self.hi = np.minimum(model.jnt_range[:20, 1], model.actuator_ctrlrange[:, 1])-1e-6
        self.counts = dict(kinematics=0, comPos=0, jacSite=0, residual=0, integration_steps=0)
        self.last = None

    def state(self, q):
        q = np.asarray(q, float)
        if q.shape != (20,) or not np.isfinite(q).all() or np.any(q < self.lo-1e-12) or np.any(q > self.hi+1e-12):
            raise ValueError('Raw legal20q proposal; no silent clamp')
        if self.last is None or q.tobytes() != self.last.tobytes():
            self.data.qpos[:] = self.base; self.data.qpos[:20] = q
            self.mj.mj_kinematics(self.model, self.data); self.counts['kinematics'] += 1
            self.mj.mj_comPos(self.model, self.data); self.counts['comPos'] += 1
            self.last = q.copy()
        return self.data.site_xpos[self.site_ids].copy()

    def residual(self, q):
        positions = self.state(q); self.counts['residual'] += 1
        return np.r_[((positions-self.targets)*np.sqrt(self.weights[:, None])/.01).ravel(),
                     self.regularization*np.asarray(q)]

    def jacobian(self, q):
        self.state(q)
        rows = []
        for site, weight in zip(self.site_ids, self.weights):
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            self.mj.mj_jacSite(self.model, self.data, jp, jr, site); self.counts['jacSite'] += 1
            rows.append(jp[:, :20]*np.sqrt(weight)/.01)
        return np.r_[np.concatenate(rows, axis=0), np.eye(20)*self.regularization]

    def solve(self, start, *, max_nfev=300):
        self.state(start)
        result = least_squares(self.residual, start, jac=self.jacobian, bounds=(self.lo, self.hi),
            max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-9)
        positions = self.state(result.x)
        return dict(q20=result.x.copy(), tip_positions_world=positions,
            tip_errors_m=np.linalg.norm(positions-self.targets, axis=1), solver_success=bool(result.success),
            solver_message=str(result.message), nfev=int(result.nfev), cost=float(result.cost),
            maximum_hard_bound_violation_rad=float(max(0., np.max(self.lo-result.x), np.max(result.x-self.hi))),
            source_tip_alignment_is_contact_qualification=False)
