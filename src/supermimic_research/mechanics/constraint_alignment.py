"""Readback-only physical row identities; never reorders an engine buffer."""
import numpy as np


def canonical_constraint_rows(contacts, fields, world, *, pyramidal=True):
    nefc = int(contacts['nefc'][world])
    if nefc < 0 or any(len(value[world]) < nefc for value in fields.values()):
        raise ValueError('Complete source constraint prefix')
    keys = [(1, int(fields['type'][world, i]), int(fields['id'][world, i]), 0, 0) for i in range(nefc)]
    assigned = set()
    for k in np.flatnonzero(contacts['worldid'] == world):
        dim = int(contacts['dim'][k])
        if dim not in [1, 3, 4, 6]:
            raise ValueError('Rigid contact dimension')
        if contacts['efc_address'][k, 0] < 0:
            continue
        width = 1 if dim == 1 else 2 * (dim - 1) if pyramidal else dim
        a, b = map(int, contacts['geom'][k])
        collision = int(contacts['geomcollisionid'][k])
        for edge, index in enumerate(contacts['efc_address'][k, :width]):
            index = int(index)
            if not 0 <= index < nefc or index in assigned:
                raise ValueError('Unique valid original EFC addresses')
            assigned.add(index)
            keys[index] = (0, a, b, collision, edge)
    if len(set(keys)) != nefc:
        raise ValueError('Ambiguous physical constraint identity')
    order = np.array(sorted(range(nefc), key=lambda i: keys[i]), dtype=int)
    return dict(keys=np.array([keys[i] for i in order], dtype=np.int64).reshape(-1, 5),
                source_addresses=order,
                fields={key: np.asarray(value[world, order]).copy() for key, value in fields.items() if key not in ['type', 'id']})
