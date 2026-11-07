"""Offset-COM acceleration agrees with native free-joint integration."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.acceleration_projection import zero_velocity_COM_acceleration


class AccelerationProjectionTests(unittest.TestCase):
    def test_offset_COM_response_matches_native_quaternion_update(self):
        import mujoco
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos=".1 .2 .3" quat=".9238795325 0 0 .3826834324"><freejoint/><inertial pos=".02 .01 0" mass="1" diaginertia=".01 .01 .01"/><geom type="sphere" size=".01" contype="0"/></body></worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);d.qacc[:]=[1,2,3,4,5,6]
        linear,angular=zero_velocity_COM_acceleration(m,d,1,0);old=d.xipos[1].copy();q=d.qpos.copy();eps=1e-4
        mujoco.mj_integratePos(m,q,d.qacc*eps,eps);other=mujoco.MjData(m);other.qpos[:]=q;mujoco.mj_kinematics(m,other);mujoco.mj_comPos(m,other)
        np.testing.assert_allclose((other.xipos[1]-old)/eps**2,linear,atol=2e-7)
        np.testing.assert_allclose(angular,d.xmat[1].reshape(3,3)@d.qacc[3:])


if __name__=='__main__':unittest.main()
