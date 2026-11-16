import unittest
import numpy as np
from supermimic_research.mechanics.pen_solve_exit_epoch import review
from supermimic_research.runtime.pen_solve_exit_design import schedule


class SolveExitEpoch(unittest.TestCase):
    def example(self):
        z=np.zeros((8,3));q=np.zeros((8,7));t=np.arange(8)*.0025
        packet=dict(pre_qpos=q.copy(),post_qpos=q.copy(),pre_qvel=z.copy(),post_qvel=z.copy(),post_qacc=z.copy(),post_qfrc_constraint=z.copy(),post_qfrc_applied=z.copy(),post_xfrc_applied=np.zeros((8,1,6)),post_GPU_time=t+.0025)
        fields=dict(qpos=q,qvel=z,time=t,qacc=z,M=np.repeat(np.eye(3)[None],8,axis=0),**{'efc.Ma':z},qfrc_smooth=z,qfrc_constraint=z,qfrc_applied=z,xfrc_applied=np.zeros((8,1,6)),solver_niter=np.ones(8,dtype=int))
        context=dict(grad=z,grad_scale=np.ones(8),grad_dot=np.zeros(8),newton_decrement=np.zeros(8),improvement=np.zeros(8))
        obs={'data.'+k:v[:,None] for k,v in fields.items()};obs.update({'context.'+k:v[:,None] for k,v in context.items()})
        model={'M_mulm_rowadr':None,'M_mulm_col':None,'M_mulm_madr':None,'stat.meaninertia':[1.],'opt.tolerance':[1e-12]}
        case=dict(dt=.0025,model_layout=dict(object_dofadr=0),physics=dict(mass=1.,gravity=[0,0,0]));decoded=dict(world_COM_wrench=np.zeros((8,6)),linear_impulse_residual_N_s=z.copy())
        return packet,obs,model,case,decoded,[0,0,0],50

    def test_exact_epoch_and_complete_impulse_recomposition(self):
        r,a=review(*self.example());self.assertEqual(r['passed'],r['total']);self.assertTrue(r['checks']['original_complete_residual']);np.testing.assert_array_equal(a['actual_scaled_gradient'],0)

    def test_forward_cache_or_wrong_tick_rejects_before_force_inference(self):
        args=self.example();args[1]['data.qacc']=args[1]['data.qacc'].copy();args[1]['data.qacc'][2,0,0]=1
        r,a=review(*args);self.assertFalse(r['checks']['post_qacc']);self.assertEqual(a,{})

    def test_nonzero_scaled_gradient_is_separated_from_contact_labels(self):
        args=self.example();args[1]['data.qacc']=np.tile([[[1.,0,0]]],(8,1,1));args[1]['data.efc.Ma']=args[1]['data.qacc'].copy();args[1]['context.grad']=args[1]['data.qacc'].copy()*.5;args[1]['context.grad_scale']*=2
        args[0]['post_qacc']=args[1]['data.qacc'][:,0].copy();args[0]['post_qvel']=args[0]['post_qacc']*.0025;args[4]['linear_impulse_residual_N_s']=args[0]['post_qvel'].copy()
        r,a=review(*args);self.assertEqual(r['passed'],r['total']);np.testing.assert_allclose(a['actual_scaled_gradient'],args[4]['linear_impulse_residual_N_s'])

    def test_native_and_decoder_gravity_remain_distinct(self):
        args=list(self.example());args[5]=[.001,0,0];args[4]['linear_impulse_residual_N_s']=np.tile([-.0000025,0,0],(8,1))
        r,a=review(*args);self.assertEqual(r['passed'],r['total']);np.testing.assert_allclose(a['original_gravity_representation'],args[4]['linear_impulse_residual_N_s'])


class CompleteReplaySchedule(unittest.TestCase):
    def rows(self):
        return [dict(name=('SUPPORT_' if i==0 else 'R0_RESIDUAL_')+str(i)+role,pair=i,role=role) for i in range(5) for role in ['failure','control']]

    def test_all_conditions_scopes_and_repeats_preserved(self):
        for replica in ['A','B']:
            s=schedule(self.rows(),replica);self.assertEqual(len(s),60);self.assertEqual(len({r['name'] for r in s}),60)
            for i in range(10):
                self.assertEqual({(r['repeat'],r['scope']) for r in s if r['row_index']==i},{(k,v) for k in range(3) for v in ['PACKET_BASELINE','PACKET_PLUS_SOLVE_EXIT']})
        self.assertNotEqual(schedule(self.rows(),'A')[0]['scope'],schedule(self.rows(),'B')[0]['scope'])

    def test_missing_or_duplicate_conditions_reject(self):
        rows=self.rows();rows[0]=rows[1]
        with self.assertRaises(ValueError):schedule(rows,'A')


if __name__=='__main__':unittest.main()
