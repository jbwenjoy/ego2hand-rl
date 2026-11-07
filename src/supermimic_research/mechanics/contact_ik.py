"""Native closest-pair collision penalties for proposals, never final clearance.

The minimum of per-piece native distances is used only as an optimization
penalty. It is not a concave-union SDF or a reliable containment classifier.
The independent all-piece convex audit must qualify the resulting pose.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from .retargeting import HandTipIK


class ProposalLimit(RuntimeError):
    pass


def separation_lower_bound(a, b):
    return float(np.linalg.norm(np.maximum(np.maximum(a[0]-b[1], b[0]-a[1]), 0.)))


def native_pair(model, data, g, h, distance_limit):
    import mujoco
    points = np.zeros(6)
    distance = float(mujoco.mj_geomDistance(model, data, g, h, distance_limit, points))
    if not np.isfinite(distance) or not np.isfinite(points).all():
        raise ValueError('Nonfinite native distance proposal')
    return distance, points.reshape(2, 3)


class ContactAwareIK(HandTipIK):
    def __init__(self, model, qpos, targets, audit):
        super().__init__(model, qpos, targets, weights=[1, 1, 1, .25, .25], regularization=.003)
        self.audit = audit
        super().state(self.base[:20])
        self.fixed = {g:audit.shape(self.data, g) for g in audit.parts+[audit.table]}
        self.natives = audit.self+audit.hand_table
        self.allowed = {g for group in audit.permitted.values() for g in group}
        self.counts.update(native_distance=0, jacPoint=0, zero_distance_normal_heuristic=0)
        self.cached_q = None
        self.cached_residual = self.cached_jacobian = None
        self.best = None; self.elapsed_limit = float('inf'); self.tick = time.perf_counter()
        self.anchor = self.base[:20].copy(); self.log = None

    def distance_jacobian(self, g, h, distance, points):
        vector = points[1]-points[0]
        if abs(distance) > 1e-10 and np.linalg.norm(vector) > 1e-10:
            normal = vector/distance
            if abs(np.linalg.norm(normal)-1) > .01:
                raise ValueError('Native from-to endpoints inconsistent with signed distance')
            normal /= np.linalg.norm(normal)
        else:
            # Explicit proposal heuristic at a nonsmooth/zero-distance tie.
            # It is counted and cannot substitute for final geometric evidence.
            normal = self.data.geom_xpos[h]-self.data.geom_xpos[g]
            if np.linalg.norm(normal) < 1e-10:
                raise ValueError('No direction at coincident native witness tie')
            normal = normal/np.linalg.norm(normal)
            self.counts['zero_distance_normal_heuristic'] += 1
        jac = np.zeros(20)
        for sign, gid, point in [(-1., g, points[0]), (1., h, points[1])]:
            if gid not in self.audit.hand:
                continue
            jp = np.zeros((3, self.model.nv)); jr = np.zeros_like(jp)
            self.mj.mj_jac(self.model, self.data, jp, jr, point, int(self.model.geom_bodyid[gid]))
            self.counts['jacPoint'] += 1
            jac += sign*(normal @ jp[:, :20])
        return jac

    def pair(self, shapes, g, h, limit):
        if separation_lower_bound(shapes[g]['bounds'], shapes[h]['bounds']) > limit:
            return limit, np.zeros(20)
        d, p = native_pair(self.model, self.data, g, h, limit); self.counts['native_distance'] += 1
        if d >= limit:
            return d, np.zeros(20)
        return d, self.distance_jacobian(g, h, d, p)

    def object_pair(self, shapes, hand):
        distances = [(separation_lower_bound(shapes[g]['bounds'], shapes[hand]['bounds']), g) for g in self.audit.parts]
        best, gradient = .03, np.zeros(20)
        for lower, part in sorted(distances):
            # Positive AABB separation bounds positive exact separation. A zero
            # lower bound cannot prune a potentially deeper intersecting piece.
            if lower > max(best, 0.):
                break
            d, jac = self.pair(shapes, part, hand, .03)
            if d < best:
                best, gradient = d, jac
        return best, gradient

    def evaluate(self, q):
        if self.cached_q is not None and q.tobytes() == self.cached_q.tobytes():
            return self.cached_residual, self.cached_jacobian
        if time.perf_counter()-self.tick >= self.elapsed_limit:
            raise ProposalLimit('Declared proposal wall-time limit')
        positions = super().state(q)
        shapes = dict(self.fixed); shapes.update({g:self.audit.shape(self.data, g) for g in self.audit.hand})
        root_jac = super().jacobian(q)
        residual = list(((positions-self.targets)*np.sqrt(self.weights[:, None])/.02).ravel())
        rows = list(root_jac[:15]/2)
        residual.extend(self.regularization*(q-self.anchor)); rows.extend(np.eye(20)*self.regularization)
        object_gaps = {}
        for hand in self.audit.hand:
            gap, jac = self.object_pair(shapes, hand); object_gaps[hand] = (gap, jac)
            floor = .00005 if hand in self.allowed else .0002
            active = gap < floor
            residual.append(max(floor-gap, 0.)/.0005)
            rows.append(-jac/.0005 if active else np.zeros(20))
        native_gaps = []
        for g, h in self.natives:
            gap, jac = self.pair(shapes, g, h, .001)
            native_gaps.append(gap); active = gap < .0002
            residual.append(max(.0002-gap, 0.)/.0005)
            rows.append(-jac/.0005 if active else np.zeros(20))
        intent_gaps = []
        for key, geoms in self.audit.permitted.items():
            gid = min(geoms, key=lambda g:object_gaps[g][0]); gap, jac = object_gaps[gid]
            intent_gaps.append(gap)
            active = gap > .0001
            residual.append(max(gap-.0001, 0.)/.002)
            rows.append(jac/.002 if active else np.zeros(20))
        residual, rows = np.array(residual), np.array(rows)
        cost = float(residual @ residual)
        record = dict(q20=q.tolist(), squared_cost=cost,
            maximum_native_object_penetration_m=max(0., -min(x[0] for x in object_gaps.values())),
            maximum_native_self_or_table_penetration_m=max(0., -min(native_gaps)),
            canonical_min_native_component_gap_m=intent_gaps, elapsed_s=time.perf_counter()-self.tick)
        if self.log is not None:
            import json
            self.log.write(json.dumps(record)+'\n'); self.log.flush()
        if self.best is None or cost < self.best['squared_cost']:
            self.best = record
        self.counts['residual'] += 1
        self.cached_q = q.copy(); self.cached_residual, self.cached_jacobian = residual, rows
        return residual, rows

    def solve_contact(self, start, log, *, max_nfev=160, maximum_seconds=45.):
        self.anchor = np.asarray(start).copy(); self.tick = time.perf_counter(); self.elapsed_limit = maximum_seconds
        self.best = None; self.cached_q = None; self.log = log
        success = False; nfev = None
        try:
            result = least_squares(lambda q:self.evaluate(q)[0], start,
                jac=lambda q:self.evaluate(q)[1], bounds=(self.lo, self.hi), max_nfev=max_nfev,
                ftol=1e-9, xtol=1e-9, gtol=1e-8, x_scale='jac')
            success, message, nfev = bool(result.success), str(result.message), int(result.nfev)
        except ProposalLimit as error:
            message = str(error)
        if self.best is None:
            raise RuntimeError('No completed proposal')
        q = np.array(self.best['q20']); positions = super().state(q)
        return dict(best=self.best, solver_success=success, solver_message=message, nfev=nfev,
            q20=q, tip_errors_m=np.linalg.norm(positions-self.targets, axis=1),
            native_penalties_are_not_independent_clearance=True, counts=self.counts.copy())
