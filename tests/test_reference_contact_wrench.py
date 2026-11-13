from pathlib import Path
import sys,unittest
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.reference_contact_wrench import reference_contact_wrench


class ReferenceWrenchTests(unittest.TestCase):
    def fixture(self):
        dt=.02;t=np.arange(9)*dt;a=np.array([.4,-.3,.2]);p=np.array([.1,.2,.3])+t[:,None]*[.02,.03,-.01]+.5*t[:,None]**2*a
        omega=np.array([.3,.5,.7]);R=Rotation.from_rotvec(t[:,None]*omega)
        q=R.as_quat()[:,[3,0,1,2]];I=np.diag([.01,.02,.025]);return dt,a,p,q,I,omega

    def test_accelerating_translation_and_gyroscopic_couple(self):
        dt,a,p,q,I,w=self.fixture();r=reference_contact_wrench(p,q,.3,I,[0,0,-9.81],dt);n=len(p)-3
        np.testing.assert_allclose(r['force_world'],np.tile(.3*(a-[0,0,-9.81]),(n,1)),rtol=0,atol=2e-12)
        expected=np.cross(w,I@w)
        np.testing.assert_allclose(r['wrench_BODY_COM'][:,3:],np.tile(expected,(n,1)),rtol=0,atol=1e-11)
        np.testing.assert_array_equal(r['valid_source_mask'],[True]*n+[False]*3)

    def test_object_frame_targets_are_invariant_to_world_coordinate_rotation(self):
        dt,_,p,q,I,_=self.fixture();Q=Rotation.from_rotvec([.7,-.2,.4]);g=np.array([0,0,-9.81])
        a=reference_contact_wrench(p,q,.3,I,g,dt)
        transformed=(Q*Rotation.from_quat(q[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
        b=reference_contact_wrench(Q.apply(p),transformed,.3,I,Q.apply(g),dt)
        np.testing.assert_allclose(a['wrench_BODY_COM'],b['wrench_BODY_COM'],rtol=0,atol=5e-12)


if __name__=='__main__':unittest.main()
