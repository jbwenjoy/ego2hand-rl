import unittest
import numpy as np
from supermimic_research.mechanics.pen_impulse_balance import partition


class ImpulseBalanceTests(unittest.TestCase):
    def test_exact_labels_do_not_imply_acceleration_balance(self):
        mass=.03;dt=.0025;g=np.array([0.,0.,-9.81]);force=np.array([[.2,-.1,.3]]);error=np.array([[.001,.002,-.003]]);acc=(force+mass*g+error)/mass;v=np.zeros((1,3));p=partition(mass,dt,v,v+dt*acc,acc,force,force,g,g)
        np.testing.assert_allclose(p['total'],dt*error,rtol=0,atol=1e-18);np.testing.assert_allclose(p['constraint_contact'],0,rtol=0,atol=0);self.assertTrue(p['recomposition_within_float64'])

    def test_known_velocity_label_and_gravity_errors_partition_with_signs(self):
        m=.03;dt=.0025;gn=np.array([0.,0.,-9.81]);gd=gn+np.array([0,0,1e-6]);f=np.array([[.1,.2,.4]]);label=f+np.array([[.0001,-.0002,.0003]]);acc=(f+m*gn)/m;v0=np.ones((1,3));ve=np.array([[1e-5,-2e-5,3e-5]]);p=partition(m,dt,v0,v0+dt*acc+ve,acc,f,label,gn,gd)
        np.testing.assert_allclose(p['velocity_update'],m*ve,rtol=0,atol=1e-17);np.testing.assert_allclose(p['constraint_contact'],dt*(f-label),rtol=0,atol=0);np.testing.assert_allclose(p['gravity_representation'],[dt*m*(gn-gd)],rtol=0,atol=0);self.assertTrue(p['recomposition_within_float64'])


if __name__=='__main__':unittest.main()
