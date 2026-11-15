"""Analytic contact signs/frames, inactive scratch, errors and window resets."""
from types import SimpleNamespace
import unittest
import numpy as np
import warp as wp
from supermimic_research.runtime.resident_contact_metrics import ResidentContactMetrics,vec5


def fixture(*,device='cpu',cone=0):
    wp.init();W=2;B=3;K=8
    model=SimpleNamespace(ngeom=4,nbody=B,njnt=2,jnt_qposadr=np.array([0,1]),geom_bodyid=np.array([2,1,1,0]),
        geom=lambda i:SimpleNamespace(name=['object/shaft','robot/right_finger1_tip_sensor_col','robot/palm','plane'][i]))
    geom=np.full((K,2),-99,dtype=np.int32);world=np.full(K,99,dtype=np.int32);dim=np.full(K,999,dtype=np.int32)
    dist=np.full(K,np.nan,dtype=np.float32);pos=np.full((K,3),np.nan,dtype=np.float32)
    frame=np.full((K,3,3),np.nan,dtype=np.float32);friction=np.full((K,5),np.nan,dtype=np.float32)
    addr=np.full((K,6),999999,dtype=np.int32)
    geom[:4]=[[0,1],[2,0],[1,2],[0,3]];world[:4]=[0,1,0,0];dim[:4]=[3,4,1,1]
    dist[:4]=[-.001,-.002,-.003,.01];pos[:4]=[[0,1,0],[0,0,2],[0,0,0],[0,0,0]]
    frame[:4]=np.eye(3);friction[:4]=[.5,.5,.005,0,0]
    addr[:4]=-1;addr[0,:4]=[0,1,2,3];addr[1]=np.arange(6);addr[2,0]=4;addr[3,0]=5
    forces=np.zeros((W,16),dtype=np.float32);forces[0,:6]=[2,1,.5,.5,0,5];forces[1,:6]=[1,1,1,1,1.25,.75]
    R=np.tile(np.eye(3),(W,B,1,1)).astype(np.float32);R[1,2]=[[0,-1,0],[1,0,0],[0,0,1]]
    com=np.zeros((W,B,3),dtype=np.float32);q=np.array([[1.2,0],[0,-1.5]],dtype=np.float32)
    def array(v,dtype):return wp.array(v,dtype=dtype,device=device)
    data=SimpleNamespace(nacon=array(np.array([4],np.int32),int),naconmax=K,njmax=16,
        contact=SimpleNamespace(worldid=array(world,int),geom=array(geom,wp.vec2i),dim=array(dim,int),dist=array(dist,float),pos=array(pos,wp.vec3),
            frame=array(frame,wp.mat33),friction=array(friction,vec5),efc_address=array(addr,int)),
        nefc=array(np.array([6,6],np.int32),int),efc=SimpleNamespace(force=array(forces,float)),xmat=array(R,wp.mat33),xipos=array(com,wp.vec3),
        qpos=array(q,float),time=array(np.zeros(W,np.float32),float),overflow=array(np.zeros(W,np.int32),int))
    dm=SimpleNamespace(opt=SimpleNamespace(cone=cone),jnt_range=array(np.array([[[-1,1],[-1,1]]],np.float32),wp.vec2))
    metrics=ResidentContactMetrics(model,dm,data,2,[0,1],capture_graph=False)
    return model,dm,data,metrics


class ResidentContactMetricTests(unittest.TestCase):
    def test_pyramid_signs_fingers_nonhand_and_real_self_penetration(self):
        _,_,_,m=fixture();m.sample();r=m.numpy()['instant']
        np.testing.assert_allclose(r['geometry'],[[.001,.003,.2],[.002,0,.5]],atol=5e-8,rtol=0)
        np.testing.assert_array_equal(r['Fn6'][0],[4,0,0,0,0,0]);np.testing.assert_array_equal(r['Fn6'][1],[0,0,0,0,0,6])
        np.testing.assert_array_equal(r['nonhand_N'],[5,0]);np.testing.assert_array_equal(r['errors'],[0,0]);np.testing.assert_array_equal(r['global_error'],[0])
        # Object is geom0 in both contacts: the ground force also points-X.
        np.testing.assert_allclose(r['wrench'][0,0],[-9,-.5,0,0,0,4],atol=1e-14)
        np.testing.assert_allclose(r['wrench'][0,1],[-4,-.5,0,0,0,4],atol=1e-14)

    def test_torsional_couple_and_WORLD_to_BODY_columns(self):
        _,_,d,m=fixture();m.sample();r=m.numpy()['instant'];mu=float(d.contact.friction.numpy()[1,2])
        torque=(float(d.efc.force.numpy()[1,4])-float(d.efc.force.numpy()[1,5]))*mu
        np.testing.assert_allclose(r['wrench'][1,0],[6,0,0,torque,12,0],atol=1e-14)
        np.testing.assert_allclose(r['wrench'][1,2],[0,-6,0,12,-torque,0],atol=1e-14)

    def test_unused_nan_garbage_is_not_read(self):
        _,_,_,m=fixture();m.sample();r=m.numpy()['instant']
        self.assertTrue(np.isfinite(r['wrench']).all());self.assertFalse(r['errors'].any());self.assertFalse(r['global_error'].any())

    def test_bad_active_world_and_geom_report_errors_without_aliasing_world0(self):
        _,_,d,m=fixture();w=d.contact.worldid.numpy();w[0]=99;d.contact.worldid.assign(w);m.sample();r=m.numpy()['instant']
        self.assertEqual(int(r['global_error'][0])&1,1);self.assertEqual(r['Fn6'][0].sum(),0.)
        w[0]=0;d.contact.worldid.assign(w);g=d.contact.geom.numpy();g[0,0]=-1;d.contact.geom.assign(g);m.sample();r=m.numpy()['instant']
        self.assertEqual(int(r['errors'][0])&2,2);self.assertFalse(r['errors'][1])

    def test_bad_EFC_and_nonfinite_active_fields_stay_failed(self):
        _,_,d,m=fixture();a=d.contact.efc_address.numpy();a[0,2]=16;d.contact.efc_address.assign(a);m.sample()
        self.assertEqual(int(m.numpy()['instant']['errors'][0])&8,8)
        a[0,2]=2;d.contact.efc_address.assign(a);v=d.contact.dist.numpy();v[0]=np.nan;d.contact.dist.assign(v);m.sample()
        self.assertEqual(int(m.numpy()['instant']['errors'][0])&16,16)

    def test_window_folds_and_partial_reset_leave_other_world_unchanged(self):
        _,_,_,m=fixture();m.sample();m.sample();before=m.numpy()['window'];m.reset_window([0]);after=m.numpy()['window']
        np.testing.assert_array_equal(after['counts'][0],[0,0]);np.testing.assert_array_equal(after['counts'][1],before['counts'][1])
        np.testing.assert_array_equal(after['wrench_sum'][1],before['wrench_sum'][1]);self.assertEqual(before['counts'][0,0],2)
        np.testing.assert_array_equal(before['Fn6_sum'][0],[8,0,0,0,0,0]);self.assertEqual(before['nonhand_max_sum'][0,0],5.)

    def test_source_COM_shift_is_distinct_from_moving_COM_moment(self):
        _,_,d,m=fixture();c=d.xipos.numpy();c[0,2]=[0,2,0];d.xipos.assign(c)
        m.sample();r=m.numpy()['window'];# source_COM was0, current_COM moved+2Y
        # Hand force[-4,-.5,0]; moment at current COM=-4Z, at source COM=+4Z.
        self.assertAlmostEqual(r['wrench_sum'][0,5,5],-4.)
        self.assertAlmostEqual(r['wrench_sum'][0,7,5],4.)

    def test_capacity_and_negative_normal_are_explicit(self):
        _,_,d,m=fixture();f=d.efc.force.numpy();f[0,5]=-1;d.efc.force.assign(f);m.sample();r=m.numpy()['instant']
        self.assertEqual(int(r['errors'][0])&64,64);self.assertEqual(r['nonhand_N'][0],0.)
        d.nacon.assign(np.array([9],np.int32));m.sample();r=m.numpy()['instant']
        self.assertEqual(int(r['global_error'][0])&32,32)

    def test_elliptic_six_components_include_direct_rolling_couple(self):
        _,_,d,m=fixture(cone=1)
        d.nacon.assign(np.array([1],np.int32));dim=d.contact.dim.numpy();dim[0]=6;d.contact.dim.assign(dim)
        a=d.contact.efc_address.numpy();a[0]=np.arange(6);d.contact.efc_address.assign(a)
        f=d.efc.force.numpy();f[0,:6]=[2,.5,-.25,.125,-.5,1];d.efc.force.assign(f)
        m.sample();r=m.numpy()['instant']
        # Object geom0: reverse the complete contact wrench, then arm x F.
        np.testing.assert_array_equal(r['wrench'][0,0],[-2,-.5,.25,.125,.5,1])
        np.testing.assert_array_equal(r['Fn6'][0],[2,0,0,0,0,0]);self.assertFalse(r['errors'].any())

    def test_device_reset_invalid_and_duplicate_ids_do_not_reset_any_world(self):
        _,_,_,m=fixture();m.sample();before=m.numpy()['window']
        for ids in ([0,0],[0,2],[2**32]):
            m.reset_window(wp.array(ids,dtype=wp.int64,device='cpu'));after=m.numpy()['window']
            np.testing.assert_array_equal(after['counts'],before['counts'])
            np.testing.assert_array_equal(after['wrench_sum'],before['wrench_sum'])
            self.assertEqual(int(after['global_error'][0])&1024,1024)
        m.reset_window(wp.array([0],dtype=wp.int64,device='cpu'));after=m.numpy()['window']
        self.assertEqual(int(after['global_error'][0])&1024,1024)
        np.testing.assert_array_equal(after['counts'][0],[0,0])
        np.testing.assert_array_equal(after['counts'][1],before['counts'][1])
        m.reset_window(wp.array([1,0],dtype=wp.int64,device='cpu'))
        self.assertFalse(m.numpy()['window']['global_error'].any())

    def test_empty_reset_and_bad_host_ids(self):
        _,_,_,m=fixture();m.sample();before=m.numpy()['window'];m.reset_window([])
        np.testing.assert_array_equal(m.numpy()['window']['counts'],before['counts'])
        for ids in ([.5],[-1],[0,0],[2**32]):
            with self.assertRaises(ValueError):m.reset_window(ids)

    def test_bad_rotation_joint_limits_and_overflow_are_not_repaired(self):
        _,dm,d,m=fixture();R=d.xmat.numpy();R[0,2,0,0]=2;d.xmat.assign(R)
        bounds=dm.jnt_range.numpy();bounds[0,0]=[1,-1];dm.jnt_range.assign(bounds)
        d.overflow.assign(np.array([0,1],np.int32));m.sample();r=m.numpy()['instant']
        self.assertEqual(int(r['errors'][0])&512,512);self.assertEqual(int(r['errors'][0])&256,256)
        self.assertEqual(int(r['errors'][1])&128,128);self.assertEqual(d.xmat.numpy()[0,2,0,0],2)

    def test_pyramidal_six_components_read_all_ten_EFC_rows(self):
        _,_,d,m=fixture();d.nacon.assign(np.array([1],np.int32))
        dim=d.contact.dim.numpy();dim[0]=6;d.contact.dim.assign(dim)
        a=np.full((8,10),-1,np.int32);a[0]=np.arange(10)
        d.contact.efc_address=wp.array(a,dtype=int,device='cpu')
        d.nefc.assign(np.array([10,6],np.int32));f=d.efc.force.numpy();f[0,:10]=np.arange(1,11);d.efc.force.assign(f)
        mu=d.contact.friction.numpy();mu[0]=[.5,.25,.1,.05,.025];d.contact.friction.assign(mu)
        m.sample();r=m.numpy()['instant'];actual_mu=d.contact.friction.numpy()[0].astype(np.float64)
        F=np.array([-55,actual_mu[0],actual_mu[1]])
        M=actual_mu[2:]+np.cross([0,1,0],F)
        np.testing.assert_allclose(r['wrench'][0,0],np.r_[F,M],atol=1e-14,rtol=0)
        self.assertEqual(r['Fn6'][0,0],55.);self.assertFalse(r['errors'].any())

    def test_source_axes_remain_fixed_when_current_BODY_rotates(self):
        _,_,d,m=fixture();R=d.xmat.numpy();R[1,2]=np.eye(3);d.xmat.assign(R)
        m.sample();r=m.numpy()['window']
        # Current BODY now equals WORLD; the window's source axes remain Rz90.
        np.testing.assert_allclose(r['wrench_sum'][1,2,:3],[6,0,0],atol=1e-14)
        np.testing.assert_allclose(r['wrench_sum'][1,4,:3],[0,-6,0],atol=1e-14)


if __name__=='__main__':unittest.main()
