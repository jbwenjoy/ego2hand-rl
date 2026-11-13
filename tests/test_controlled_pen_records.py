from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic_research.mechanics.controlled_pen_records import native_state55,pack_case


class PenRecordTests(unittest.TestCase):
    def test_object_first_indices_and_world_angular_velocity(self):
        q=np.r_[[2.,3.,4.],np.sqrt(.5),0.,0.,np.sqrt(.5),np.arange(20)+10.]
        v=np.r_[[1.,2.,3.],[1.,0.,0.],np.arange(20)+30.]
        s=native_state55(q,v,np.arange(7,27),np.arange(6,26),0,0)
        np.testing.assert_array_equal(s[:20],q[7:]);np.testing.assert_array_equal(s[20:40],v[6:])
        np.testing.assert_allclose(s[52:55],[0.,1.,0.],atol=1e-14)

    def test_source_frame_wrench_and_ring_palm_channels_survive(self):
        N=4;q=np.zeros((N,27));q[:,3]=1;v=np.zeros((N,26));post=q.copy()
        for i in range(N):
            a=Rotation.from_euler('z',i*np.pi/6).as_quat();b=Rotation.from_euler('z',(i+1)*np.pi/6).as_quat()
            q[i,3:7]=a[[3,0,1,2]];post[i,3:7]=b[[3,0,1,2]]
        raw=[]
        for tick in range(N):
            for geom,normal in [(1,2.),(2,3.)]:
                row=np.zeros(24);row[:3]=[tick,0,geom];row[18]=normal;raw.append(row)
        z=dict(pre_qpos=q,pre_qvel=v,post_qpos=post,post_qvel=v.copy(),post_ctrl=np.zeros((N,20)),post_qacc=np.zeros((N,26)),actions20=np.zeros((1,20)),
            raw_contact_tick_plus23=np.array(raw),overflow=np.zeros(N),post_qfrc_applied=np.zeros((N,26)),post_xfrc_applied=np.zeros((N,4,6)),
            pre_penetration_m=np.zeros(N),pre_self_penetration_m=np.zeros(N),post_joint_limit_excursion_rad=np.zeros(N))
        names=['object/shaft','robot/right_finger4_tip_sensor_col','robot/right_palm_col']
        m=SimpleNamespace(body_parentid=np.zeros(4,int),geom_bodyid=np.array([1,2,3]),geom=lambda g:SimpleNamespace(name=names[g]))
        c=dict(dt=.005,model_layout=dict(object_body=1,object_qposadr=0,object_dofadr=0,COM_BODY=[0,0,0]))
        w=dict(world_COM_wrench=np.tile([1.,0,0,0,0,0],(N,1)),force_residual_N=np.zeros((N,3)),torque_residual_Nm=np.zeros((N,3)),linear_impulse_residual_N_s=np.zeros((N,3)))
        p=pack_case(m,c,z,w,np.arange(7,27),np.arange(6,26),np.arange(20),np.zeros(26),dict(mass=1.,inertia_BODY=np.eye(3),gravity=[0,0,0]))
        np.testing.assert_allclose(p['y'][0,78:84],[1.,0,0,0,0,0],atol=1e-15)
        np.testing.assert_array_equal(p['y'][0,84:],[0,0,0,2,0,3]);self.assertEqual(p['y'].shape,(1,90))
