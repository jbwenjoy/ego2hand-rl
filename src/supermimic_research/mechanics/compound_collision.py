"""All-piece ideal-convex qualification; component margins are not a union SDF."""
import numpy as np
from .collision import CollisionAudit
from .convex import ConvexPolytope


def eligible_pairs(model, geoms):
    """Compiled mask/weld-parent/exclude/explicit rules on a declared geom set."""
    import mujoco
    explicit = {tuple(sorted(map(int, (a, b)))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    excluded = set(map(int, model.exclude_signature))
    disable = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    pairs = []
    for i, g in enumerate(geoms):
        for h in geoms[i+1:]:
            b1, b2 = int(model.geom_bodyid[g]), int(model.geom_bodyid[h])
            w1, w2 = int(model.body_weldid[b1]), int(model.body_weldid[b2])
            pw1, pw2 = int(model.body_weldid[model.body_parentid[w1]]), int(model.body_weldid[model.body_parentid[w2]])
            if tuple(sorted((g, h))) not in explicit:
                if b1 == b2 or w1 == w2:
                    continue
                if not disable and w1 and w2 and (w1 == pw2 or w2 == pw1):
                    continue
                if ((min(b1, b2) << 16) | max(b1, b2)) in excluded:
                    continue
                if not ((int(model.geom_contype[g]) & int(model.geom_conaffinity[h])) or
                        (int(model.geom_contype[h]) & int(model.geom_conaffinity[g]))):
                    continue
            pairs.append((g, h))
    return pairs


class CompoundCollisionAudit(CollisionAudit):
    def __init__(self, model, parts, hand, table, permitted):
        self.model = model
        self.parts, self.hand = list(parts), list(hand)
        self.table = int(table)
        self.permitted = {int(k):tuple(map(int, v)) for k, v in permitted.items()}
        if not self.parts or len(set(self.parts+self.hand+[self.table])) != len(self.parts+self.hand)+1:
            raise ValueError('Disjoint explicit component, hand and table sets')
        if len(set(int(model.geom_bodyid[g]) for g in self.parts)) != 1:
            raise ValueError('All compound pieces must share one rigid BODY')
        self.self = eligible_pairs(model, self.hand)
        self.hand_table = [(g, self.table) for g in self.hand if (g, self.table) in eligible_pairs(model, [g, self.table])]
        self.locals = {}
        for gid in self.parts+self.hand+[self.table]:
            kind = int(model.geom_type[gid])
            if kind == 7:
                mid = int(model.geom_dataid[gid]); start, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                self.locals[gid] = ConvexPolytope(model.mesh_vert[start:start+n]).vertices
            elif kind == 6:
                from itertools import product
                self.locals[gid] = np.array(list(product(*[(-s, s) for s in model.geom_size[gid]])))
            elif kind != 3:
                raise ValueError('Only compiled convex meshes, boxes and capsules supported')

    def evaluate(self, data, *, pad_penetration_m=.0001, pad_gap_m=.0002):
        shapes = {g:self.shape(data, g) for g in self.parts+self.hand+[self.table]}
        allowed = {g for group in self.permitted.values() for g in group}
        contacts = {k:[] for k in self.permitted}
        rows = []
        sets = [('object', [(part, g) for part in self.parts for g in self.hand+[self.table]]),
                ('self', self.self), ('hand_table', self.hand_table)]
        for kind, pairs in sets:
            for g, h in pairs:
                intent = kind == 'object' and h in allowed
                try:
                    result = self.query(shapes[g], shapes[h])
                    passed = result['lower_bound_m'] >= (-pad_penetration_m if intent else 0.)
                    near = (intent and result['upper_bound_m'] is not None and
                            result['lower_bound_m'] >= -pad_penetration_m and result['upper_bound_m'] <= pad_gap_m)
                    if near:
                        for key, group in self.permitted.items():
                            if h in group:
                                contacts[key].append([g, h])
                    rows.append(dict(kind=kind, geom_ids=[g, h], permitted_intent=intent, passed=bool(passed), **result))
                except (ArithmeticError, ValueError, RuntimeError) as error:
                    rows.append(dict(kind=kind, geom_ids=[g, h], permitted_intent=intent, passed=False,
                                     status='UNKNOWN', error=repr(error)))
        return dict(rows=rows, object_pairs=len(self.parts)*(len(self.hand)+1), self_pairs=len(self.self),
            hand_table_pairs=len(self.hand_table), all_pairs_clear_with_declared_pad_band=all(r['passed'] for r in rows),
            canonical_contact_candidates=contacts, all_three_intent_groups_near=all(contacts.values()),
            passed=bool(all(r['passed'] for r in rows) and all(contacts.values())), unknown=sum(r['status']=='UNKNOWN' for r in rows),
            pad_component_margin_band_m=[-pad_penetration_m, pad_gap_m],
            scope='All declared convex components checked separately with float Qhull/support witnesses or conservative AABBs. Pairwise pure-translation margins are not a signed concave-union distance or global union exit-depth bound. Unknown never passes. Static candidate only, not native force/holding qualification.')
