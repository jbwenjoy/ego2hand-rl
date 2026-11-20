import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_research.mechanics.pen_capacity_trace import packet_capacity


class PacketCapacityTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><option cone="pyramidal"/>
        <worldbody><body name="obj"><freejoint/><geom name="pen" type="sphere" size=".1"/></body>
        <geom name="robot/right_finger2_fingertip" type="sphere" size=".03" priority="1"/>
        <geom name="table" type="plane" size="1 1 .01"/></worldbody></mujoco>''')
        m = self.model
        self.obj = m.geom('pen').id
        self.finger = m.geom('robot/right_finger2_fingertip').id
        self.layout = dict(object_body=m.body('obj').id, object_qposadr=0)
        self.friction = np.zeros((m.ngeom, 3)); self.friction[self.obj] = [1.9,.5,.7]
        self.friction[self.finger] = [.5,.01,.02]
        self.com = np.repeat([[.4,.1,.3]], 8, axis=0)
        self.q = np.zeros((8,m.nq));self.q[:,3] = 1.
        self.q[1,3:7] = Rotation.from_rotvec([0.,0.,np.pi/2]).as_quat()[[3,0,1,2]]

    def row(self,t,g,h,fn,*,gap=0.,tangent=0.,twist=0.):
        return np.r_[t,g,h,4,0,gap,self.com[t]+[0,.1,0],np.eye(3).ravel(),fn,tangent,0,twist,0,0]

    def run_packet(self,rows):
        z = dict(raw_contact_tick_plus23=np.array(rows).reshape(-1,24),pre_qpos=self.q,pre_object_COM=self.com)
        return packet_capacity(z,self.model,self.layout,self.friction[None])

    def test_sign_frame_com_and_gap_selection(self):
        rows = [self.row(0,self.finger,self.obj,2.,twist=.02),
                self.row(1,self.obj,self.finger,2.,twist=.02),
                self.row(1,self.obj,self.finger,3.,gap=.002),
                self.row(1,self.model.geom('table').id,self.obj,5.),
                self.row(3,self.finger,self.obj,0.),
                self.row(5,self.finger,self.obj,1.,tangent=2.)]
        d=self.run_packet(rows)
        np.testing.assert_array_equal(d['mask'],[2,2,0,2,0,2,0,0])
        np.testing.assert_array_equal(d['normal_load_N'],[2,2,0,0,0,1,0,0])
        np.testing.assert_allclose(d['normalized_wrench_body'][0],[1,0,0,.1,0,-1],atol=1e-14)
        np.testing.assert_allclose(d['normalized_wrench_body'][1],[0,1,0,0,.1,1],atol=1e-14)
        self.assertAlmostEqual(d['positive_gap_skipped_max_load_N'],3.)
        np.testing.assert_allclose(d['containment_excess'][:5],0,atol=1e-14)
        self.assertGreaterEqual(d['containment_excess'][5],1.5-1e-14)
        self.assertEqual(d['capacity'].shape,(8,72))
        np.testing.assert_array_equal(d['capacity'][0],d['capacity'][3])
        self.assertAlmostEqual(d['capacity'][0,1],.5)

    def test_empty_and_invalid_indices(self):
        d=self.run_packet([])
        np.testing.assert_array_equal(d['capacity'],np.zeros((8,72)))
        self.assertFalse(d['containment_tested'].any())
        with self.assertRaises(ValueError):self.run_packet([self.row(0,-1,self.obj,2.)])


if __name__=='__main__':unittest.main()
