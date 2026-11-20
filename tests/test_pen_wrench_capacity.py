"""Mechanical conventions and independent LP reference for directional support."""
from itertools import product
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from supermimic_research.mechanics.pen_wrench_capacity import (
    balanced_directions, body_wrench_rays, directional_support,
)


def contact(pos, frame, dim, mu):
    return SimpleNamespace(pos=np.array(pos), frame=np.asarray(frame), dim=dim,
                           friction=np.asarray(mu))


def lp_reference(contacts, signs, center, rot, length, direction):
    """Use physical local loads, imposing each Coulomb pyramid by its facets."""
    widths = [c.dim for c in contacts]
    width = sum(widths)
    normal_budget = np.zeros(width)
    bounds, inequalities, objective = [], [], []
    offset = 0
    # Pull the BODY wrench objective back to virtual world force and moment.
    v = rot @ direction[:3]
    omega = rot @ direction[3:] / length
    for c, sign in zip(contacts, signs):
        dim = c.dim
        point_v = v + np.cross(omega, c.pos - center)
        objective.extend(sign * np.r_[c.frame @ point_v, c.frame @ omega][:dim])
        normal_budget[offset] = 1.
        bounds.append((0., None))
        bounds.extend((None, None) if f else (0., 0.) for f in c.friction[:dim-1])
        active = np.flatnonzero(c.friction[:dim-1] > 0)
        for pattern in product((-1., 1.), repeat=len(active)):
            row = np.zeros(width)
            row[offset] = -1
            row[offset + active + 1] = np.asarray(pattern) / c.friction[active]
            inequalities.append(row)
        offset += dim
    result = linprog(-np.asarray(objective), A_ub=[*inequalities,normal_budget],
                     b_ub=[*np.zeros(len(inequalities)),1.], bounds=bounds, method='highs')
    assert result.success, result.message
    return -result.fun


class WrenchCapacityTest(unittest.TestCase):
    def setUp(self):
        self.center = np.array([.13,-.27,.44])
        self.rot = Rotation.from_rotvec([.43,-.27,.06]).as_matrix()
        self.length = .096
        self.contacts = [contact(self.center+[.011,.04,-.005], np.eye(3), 1, np.zeros(5)),
                         contact(self.center+[.03,-.04,.02], self.rot, 3, [.22,.4,0,0,0]),
                         contact(self.center+[-.007,.003,.03], self.rot.T, 4, [.15,.2,.001,0,0]),
                         contact(self.center+[.001,.012,.023], np.eye(3), 6, [.5,0,.004,.001,.003])]
        self.signs = [1,-1,1,-1]

    def rays(self, contacts=None, signs=None):
        return body_wrench_rays(self.contacts if contacts is None else contacts,
                                self.signs if signs is None else signs,
                                com_world=self.center, world_from_body=self.rot,
                                length_scale_m=self.length)

    def test_independent_local_load_lp_and_symmetric_unit_probes(self):
        d = balanced_directions()
        self.assertEqual(d.shape,(72,6))
        np.testing.assert_allclose(np.linalg.norm(d,axis=1),1.)
        np.testing.assert_allclose(d.T @ d / len(d),np.eye(6)/6,atol=1e-15)
        self.assertEqual(set(map(tuple,d)),set(map(tuple,-d)))
        expect = [lp_reference(self.contacts,self.signs,self.center,self.rot,self.length,row) for row in d]
        np.testing.assert_allclose(directional_support(self.rays()),expect,atol=1e-11)

    def test_global_rigid_motion_covariance(self):
        Q = Rotation.from_rotvec([-.64,.14,.11]).as_matrix()
        t = np.array([4.2,-.7,.2])
        moved = [contact(Q@c.pos+t,c.frame@Q.T,c.dim,c.friction) for c in self.contacts]
        rays = body_wrench_rays(moved,self.signs,com_world=Q@self.center+t,
                                world_from_body=Q@self.rot,length_scale_m=self.length)
        np.testing.assert_allclose(rays,self.rays(),atol=1e-13)
        np.testing.assert_allclose(directional_support(rays),directional_support(self.rays()),atol=1e-13)

    def test_shared_normal_budget_duplication_empty_and_unilateral_force(self):
        np.testing.assert_array_equal(directional_support(self.rays(self.contacts*3,self.signs*3)),
                                      directional_support(self.rays()))
        empty=self.rays([],[])
        self.assertEqual(empty.shape,(0,6))
        np.testing.assert_array_equal(directional_support(empty),np.zeros(72))
        c = contact([0.,1.,0.],np.eye(3),1,[])
        rays=body_wrench_rays([c],[1],com_world=np.zeros(3),world_from_body=np.eye(3),length_scale_m=.5)
        np.testing.assert_array_equal(rays,[[1.,0.,0.,0.,0.,-2.]])
        np.testing.assert_array_equal(directional_support(rays,[[-1,0,0,0,0,0]]),[0.])

    def test_input_conventions(self):
        with self.assertRaises(ValueError):self.rays(self.contacts,[])
        with self.assertRaises(ValueError):self.rays(self.contacts,[0]*4)
        with self.assertRaises(ValueError):
            body_wrench_rays([],[],com_world=self.center,world_from_body=self.rot,length_scale_m=0.)
        with self.assertRaises(ValueError):
            body_wrench_rays([],[],com_world=self.center,world_from_body=-np.eye(3),length_scale_m=1.)


if __name__=='__main__':unittest.main()
