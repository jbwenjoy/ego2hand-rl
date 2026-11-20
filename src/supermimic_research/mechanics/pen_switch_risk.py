"""Source-pose screen for impending hand/pen zero-gap crossings in CPU MuJoCo."""
import mujoco
import numpy as np
from scipy.special import expit

from .compound_collision import eligible_pairs


class PenSwitchRisk:
    """The score measures change in soft contact occupancy over a linearized 20 ms.

    This implementation owns scratch MjData. It evaluates kinematics and distances
    only; applied torques, future controls and integrated endpoints are unnecessary.
    A moving mocap base would require the corresponding body velocity. This screen
    assumes a fixed mount, as in the pen task.
    """

    def __init__(self, model, object_body, *, horizon_s=.02, distance_limit_m=.03,
                 gap_scale_m=.00025):
        params = np.array([horizon_s, distance_limit_m, gap_scale_m], dtype=float)
        if not np.isfinite(params).all() or (params <= 0).any():
            raise ValueError("Positive finite horizon and metric length scales required")
        self.model = model
        body = model.body(object_body).id if isinstance(object_body, str) else int(object_body)
        if not 0 < body < model.nbody:
            raise ValueError("Invalid pen body")
        parts = np.flatnonzero(model.geom_bodyid == body).tolist()
        hands = [g for g in range(model.ngeom)
                 if (model.geom(g).name or '').startswith('robot/') and g not in parts]
        self.pairs = [(g, h) for g, h in eligible_pairs(model, hands + parts)
                      if g in hands and h in parts]
        if not self.pairs:
            raise ValueError("No collision-eligible hand/pen geom pairs")
        self.horizon_s, self.distance_limit_m, self.gap_scale_m = params.tolist()
        self.data = mujoco.MjData(model)
        self.offset_data = mujoco.MjData(model)
        self.witness = np.zeros(6)
        self.jac = np.zeros((3, model.nv))

    def _near_touch_speed(self, hand, part):
        # At coincident witnesses the normalized segment has no direction. Take
        # a centered derivative of geometry along qvel, without integrating dynamics.
        dt = 1e-6
        d = self.offset_data
        d.mocap_pos[:] = self.data.mocap_pos
        d.mocap_quat[:] = self.data.mocap_quat
        gaps = []
        for sign in (-1., 1.):
            d.qpos[:] = self.data.qpos
            mujoco.mj_integratePos(self.model, d.qpos, self.data.qvel, sign * dt)
            mujoco.mj_kinematics(self.model, d)
            gaps.append(mujoco.mj_geomDistance(self.model, d, hand, part,
                                                self.distance_limit_m, None))
        return (gaps[1] - gaps[0]) / (2 * dt)

    def evaluate(self, qpos, qvel, mocap_pos, mocap_quat):
        m, d = self.model, self.data
        for key, value in (('qpos', qpos), ('qvel', qvel), ('mocap_pos', mocap_pos),
                           ('mocap_quat', mocap_quat)):
            dest = getattr(d, key)
            a = np.asarray(value, dtype=np.float64)
            # Accept either single-world native arrays or their leading W=1 views.
            if a.shape == (1,) + dest.shape:
                a = a[0]
            if a.shape != dest.shape or not np.isfinite(a).all():
                raise ValueError(f'Invalid {key} layout or values')
            dest[:] = a
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        result = []
        for hand, part in self.pairs:
            gap = float(mujoco.mj_geomDistance(m, d, hand, part, self.distance_limit_m,
                                               self.witness))
            if not np.isfinite(gap):
                raise ValueError('Nonfinite geometry distance')
            if gap >= self.distance_limit_m:
                continue  # API returns the cutoff for a censored query.
            segment = self.witness[3:] - self.witness[:3]
            length = np.linalg.norm(segment)
            if length < 1e-10 or abs(gap) < 1e-10:
                speed = self._near_touch_speed(hand, part)
                method = 'directional_distance'
            else:
                # Inside an overlap the two witness points are in reversed order.
                normal = segment * (np.sign(gap) / length)
                mujoco.mj_jac(m, d, self.jac, None, self.witness[:3], m.geom_bodyid[hand])
                hand_speed = normal @ self.jac @ d.qvel
                mujoco.mj_jac(m, d, self.jac, None, self.witness[3:], m.geom_bodyid[part])
                speed = float(normal @ self.jac @ d.qvel - hand_speed)
                method = 'witness_jacobian'
            if not np.isfinite(speed):
                raise ValueError('Nonfinite relative normal velocity')
            end = gap + self.horizon_s * speed
            crossing = (gap > 0 and end <= 0) or (gap <= 0 and end > 0)
            event = max(0., -gap / speed) if crossing else None
            occ = expit(-np.array([gap, (gap + end) / 2, end]) / self.gap_scale_m)
            score = float(np.mean(abs(occ[1:] - occ[0])))
            result.append(dict(hand_geom=int(hand), object_geom=int(part), gap_m=gap,
                               normal_speed_m_s=float(speed), predicted_gap_m=float(end),
                               predicted_crossing=bool(crossing), crossing_time_s=event,
                               event=('onset' if gap > 0 else 'release') if crossing else None,
                               derivative_method=method, score=score))
        events = [r for r in result if r['predicted_crossing']]
        return dict(score=max((r['score'] for r in result), default=0.),
                    predicted_crossing=bool(events),
                    closest_event=min(events, key=lambda r: r['crossing_time_s'], default=None),
                    candidate_pairs=len(self.pairs), censored_pairs=len(self.pairs)-len(result),
                    pairs=result)
