"""Native compound contacts aggregated once about the object's moving COM."""
import numpy as np


class CompoundContacts:
    def __init__(self, model, parts, body, hand_groups, table):
        self.model, self.body = model, int(body)
        self.parts = set(map(int, parts)); self.table = int(table)
        self.groups = {int(g):int(k) for k, geoms in hand_groups.items() for g in geoms}
        if not self.parts or any(model.geom_bodyid[g] != self.body for g in self.parts):
            raise ValueError('All compound geoms must have the declared one rigid BODY')
        if len(self.groups) != sum(len(v) for v in hand_groups.values()):
            raise ValueError('A hand geom belongs to exactly one canonical group')

    def read(self, data, *, raw=False):
        import mujoco
        # Call while xpos/xipos/xmat still describe the solver-PRE state.
        center = data.xipos[self.body].copy(); rotation = data.xmat[self.body].reshape(3, 3).copy()
        force, torque = np.zeros(3), np.zeros(3)
        normal = np.zeros(5); nonintent = table = penetration = self_penetration = 0.
        rows = []; used = 0
        for ci in range(data.ncon):
            con = data.contact[ci]; a, b = map(int, con.geom)
            local = np.zeros(6); mujoco.mj_contactForce(self.model, data, ci, local)
            if not np.isfinite(np.r_[con.dist, con.pos, con.frame, local]).all():
                raise ValueError('Nonfinite native contact/wrench')
            if raw:
                rows.append(np.r_[a, b, con.dim, con.efc_address, con.dist, con.pos, con.frame, local])
            penetration = max(penetration, -float(con.dist))
            object_a, object_b = a in self.parts, b in self.parts
            if object_a == object_b:
                if not object_a and self.table not in (a, b):
                    self_penetration = max(self_penetration, -float(con.dist))
                continue
            used += 1; sign = 1. if object_b else -1.; other = a if object_b else b
            frame = con.frame.reshape(3, 3)
            F = sign*(frame.T @ local[:3]); T = sign*(frame.T @ local[3:])
            force += F; torque += T+np.cross(con.pos-center, F)
            if other == self.table:
                table += max(float(local[0]), 0.)
            elif other in self.groups:
                group = self.groups[other]
                if 1 <= group <= 5:
                    normal[group-1] += max(float(local[0]), 0.)
                else:
                    nonintent += max(float(local[0]), 0.)
            else:
                nonintent += max(float(local[0]), 0.)
        return dict(force_world_N=force, torque_about_COM_world_Nm=torque,
            wrench_BODY_about_COM=np.r_[rotation.T@force, rotation.T@torque],
            object_COM_world_m=center, normal_by_finger_N=normal,
            other_normal_N=float(nonintent), table_normal_N=float(table),
            max_native_penetration_m=float(penetration), max_self_penetration_m=float(self_penetration),
            object_contacts=used, native_contacts=data.ncon, raw23=np.array(rows).reshape(-1, 23))
