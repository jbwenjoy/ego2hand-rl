import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.response_decode import decode_response,momentum_defect
from supermimic_research.mechanics.transition_records import state_rotation


class ResponseRolloutTests(unittest.TestCase):
    def test_left_rotation_and_COM_velocity_decode(self):
        s=np.zeros(55);R=Rotation.from_rotvec([.5,.2,0]).as_matrix();s[43:49]=np.r_[R[:,0],R[:,1]];p=np.zeros(87);p[43:46]=[0,0,.2];p[40:43]=[.01,.02,.03];p[46:52]=np.arange(6)*.1
        n=decode_response(s,p);np.testing.assert_allclose(state_rotation(n),Rotation.from_rotvec(p[43:46]).as_matrix()@R,atol=1e-14);np.testing.assert_allclose(n[40:43],p[40:43]);np.testing.assert_allclose(n[49:],p[46:52])
        for _ in range(100):n=decode_response(n,p)
        np.testing.assert_allclose(state_rotation(n).T@state_rotation(n),np.eye(3),atol=2e-14)

    def test_gravity_and_source_BODY_wrench_have_correct_momentum_sign(self):
        s=np.zeros(55);R=Rotation.from_rotvec([.2,.3,.4]).as_matrix();s[43:49]=np.r_[R[:,0],R[:,1]];p=np.zeros(87);g=np.array([0,0,-9.81]);mass=.25;I=np.diag([.001,.002,.003]);F=np.array([.1,.2,2.]);p[78:81]=R.T@F;p[46:49]=.02*(F/mass+g)
        n=decode_response(s,p);np.testing.assert_allclose(momentum_defect(s,n,p,mass,I,g),0,atol=1e-15)
        free=p.copy();free[78:81]=0;free[46:49]=.02*g;n=decode_response(s,free);np.testing.assert_allclose(momentum_defect(s,n,free,mass,I,g),0,atol=1e-15)


if __name__=='__main__':unittest.main()
